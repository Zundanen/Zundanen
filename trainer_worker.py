from __future__ import annotations

import argparse
import ctypes
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

from rvc_locale import ensure_rvc_language_override

PYTHON = Path(sys.executable)


def prevent_sleep(enable: bool) -> None:
    if not sys.platform.startswith("win"):
        return
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED if enable else ES_CONTINUOUS
    ctypes.windll.kernel32.SetThreadExecutionState(flags)


def find_ffmpeg_dir() -> str | None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return str(Path(ffmpeg).resolve().parent)

    localappdata = os.environ.get("LOCALAPPDATA")
    if not localappdata:
        return None

    local = Path(localappdata)
    link = local / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe"
    if link.exists():
        try:
            return str(link.resolve().parent)
        except OSError:
            return str(link.parent)

    packages = local / "Microsoft" / "WinGet" / "Packages"
    if packages.exists():
        for pkg in packages.glob("Gyan.FFmpeg*"):
            for candidate in pkg.rglob("ffmpeg.exe"):
                return str(candidate.parent)
    return None


def ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} was not found: {path}")


def audio_files(dataset_dir: Path) -> list[Path]:
    exts = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}
    return sorted(
        p for p in dataset_dir.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    )


def artifact_stems(directory: Path, suffix: str) -> set[str]:
    if not directory.exists():
        return set()
    return {
        p.name.split(".")[0]
        for p in directory.iterdir()
        if p.is_file() and p.name.lower().endswith(suffix.lower())
    }


def run(cmd: list[object], *, rvc_root: Path, env: dict[str, str] | None = None) -> None:
    printable = subprocess.list2cmdline([str(x) for x in cmd])
    print("\n" + "=" * 80, flush=True)
    print(printable, flush=True)
    print("=" * 80, flush=True)

    proc_env = os.environ.copy() if env is None else env.copy()
    old_pythonpath = proc_env.get("PYTHONPATH", "")
    proc_env["PYTHONPATH"] = (
        str(rvc_root)
        if not old_pythonpath
        else str(rvc_root) + os.pathsep + old_pythonpath
    )
    proc_env["PYTHONUNBUFFERED"] = "1"
    # Force every upstream RVC subprocess to use its bundled English locale.
    # RVC's I18nAuto() reads the process locale when no explicit language is passed.
    proc_env["LANG"] = "en_US.UTF-8"
    proc_env["LC_ALL"] = "en_US.UTF-8"
    proc_env["LANGUAGE"] = "en_US"
    proc_env["RVC_LANGUAGE"] = "en_US"

    ffmpeg_dir = find_ffmpeg_dir()
    if ffmpeg_dir:
        proc_env["PATH"] = ffmpeg_dir + os.pathsep + proc_env.get("PATH", "")

    subprocess.run(
        [str(x) for x in cmd],
        cwd=str(rvc_root),
        env=proc_env,
        check=True,
    )


def prepare_filelist(
    rvc_root: Path,
    exp_name: str,
    version: str,
    sample_rate: str,
    speaker_id: int = 0,
) -> None:
    exp_dir = rvc_root / "logs" / exp_name
    gt_dir = exp_dir / "0_gt_wavs"
    feature_dir = exp_dir / ("3_feature256" if version == "v1" else "3_feature768")
    f0_dir = exp_dir / "2a_f0"
    f0nsf_dir = exp_dir / "2b-f0nsf"

    names = (
        artifact_stems(gt_dir, ".wav")
        & artifact_stems(feature_dir, ".npy")
        & artifact_stems(f0_dir, ".npy")
        & artifact_stems(f0nsf_dir, ".npy")
    )
    if not names:
        raise RuntimeError(
            "No preprocessed training data was found. "
            "Check the preprocess / F0 / HuBERT logs."
        )

    lines: list[str] = []
    for name in sorted(names):
        lines.append(
            f"{gt_dir.as_posix()}/{name}.wav|"
            f"{feature_dir.as_posix()}/{name}.npy|"
            f"{f0_dir.as_posix()}/{name}.wav.npy|"
            f"{f0nsf_dir.as_posix()}/{name}.wav.npy|"
            f"{speaker_id}"
        )

    fea_dim = 256 if version == "v1" else 768
    mute_root = rvc_root / "logs" / "mute"
    for _ in range(2):
        lines.append(
            f"{(mute_root / '0_gt_wavs' / f'mute{sample_rate}.wav').as_posix()}|"
            f"{(mute_root / f'3_feature{fea_dim}' / 'mute.npy').as_posix()}|"
            f"{(mute_root / '2a_f0' / 'mute.wav.npy').as_posix()}|"
            f"{(mute_root / '2b-f0nsf' / 'mute.wav.npy').as_posix()}|"
            f"{speaker_id}"
        )

    random.shuffle(lines)
    (exp_dir / "filelist.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] filelist.txt: {len(names)} training clips + 2 mute", flush=True)


def prepare_config(rvc_root: Path, exp_name: str, version: str, sample_rate: str) -> None:
    exp_dir = rvc_root / "logs" / exp_name
    dst = exp_dir / "config.json"

    # Current RVC uses v1/40k config for both v1 and v2 at 40 kHz.
    if version == "v1" or sample_rate == "40k":
        src = rvc_root / "configs" / "v1" / f"{sample_rate}.json"
    else:
        src = rvc_root / "configs" / "v2" / f"{sample_rate}.json"

    ensure_exists(src, "RVC config")
    shutil.copy2(src, dst)
    print(f"[OK] config: {src.relative_to(rvc_root)} -> logs/{exp_name}/config.json", flush=True)



def export_intermediate_weights(
    rvc_root: Path,
    exports_root: Path,
    exp_name: str,
    *,
    quiet_if_none: bool = False,
) -> list[Path]:
    """Copy RVC's inference-ready per-save models into exports/<name>/checkpoints."""
    weights_dir = rvc_root / "assets" / "weights"
    checkpoint_dir = exports_root / exp_name / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    copied: list[Path] = []
    prefix = f"{exp_name}_e"
    for src in sorted(weights_dir.glob(f"{exp_name}_e*_s*.pth"), key=lambda p: p.stat().st_mtime):
        name = src.stem
        if not name.startswith(prefix):
            continue
        epoch_part = name[len(prefix):].split("_s", 1)[0]
        if not epoch_part.isdigit():
            continue

        dst = checkpoint_dir / f"{exp_name}_e{int(epoch_part)}.pth"
        # During training this function is polled. Avoid copying an unchanged
        # checkpoint over and over; copy only when it is new or updated.
        if dst.exists():
            try:
                if dst.stat().st_size == src.stat().st_size and dst.stat().st_mtime >= src.stat().st_mtime:
                    continue
            except OSError:
                pass

        shutil.copy2(src, dst)
        copied.append(dst)
        print(f"[CHECKPOINT] {dst}", flush=True)

    if not copied and not quiet_if_none:
        print(f"[WARN] No new intermediate inference checkpoints were found in: {weights_dir}", flush=True)
    return copied


def run_training(
    cmd: list[object],
    *,
    rvc_root: Path,
    exports_root: Path,
    exp_name: str,
    env: dict[str, str] | None = None,
) -> None:
    """Run RVC training and mirror Save-every small models while it is running."""
    printable = subprocess.list2cmdline([str(x) for x in cmd])
    print("\n" + "=" * 80, flush=True)
    print(printable, flush=True)
    print("=" * 80, flush=True)

    proc_env = os.environ.copy() if env is None else env.copy()
    old_pythonpath = proc_env.get("PYTHONPATH", "")
    proc_env["PYTHONPATH"] = (
        str(rvc_root)
        if not old_pythonpath
        else str(rvc_root) + os.pathsep + old_pythonpath
    )
    proc_env["PYTHONUNBUFFERED"] = "1"
    proc_env["LANG"] = "en_US.UTF-8"
    proc_env["LC_ALL"] = "en_US.UTF-8"
    proc_env["LANGUAGE"] = "en_US"
    proc_env["RVC_LANGUAGE"] = "en_US"

    ffmpeg_dir = find_ffmpeg_dir()
    if ffmpeg_dir:
        proc_env["PATH"] = ffmpeg_dir + os.pathsep + proc_env.get("PATH", "")

    proc = subprocess.Popen(
        [str(x) for x in cmd],
        cwd=str(rvc_root),
        env=proc_env,
    )
    try:
        while proc.poll() is None:
            export_intermediate_weights(
                rvc_root, exports_root, exp_name, quiet_if_none=True
            )
            time.sleep(2.0)
    finally:
        # Catch a checkpoint that may have been written just before process exit.
        export_intermediate_weights(
            rvc_root, exports_root, exp_name, quiet_if_none=True
        )

    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, [str(x) for x in cmd])

def export_results(rvc_root: Path, exports_root: Path, exp_name: str) -> None:
    export_dir = exports_root / exp_name
    export_dir.mkdir(parents=True, exist_ok=True)

    weight = rvc_root / "assets" / "weights" / f"{exp_name}.pth"
    ensure_exists(weight, "Final inference .pth")
    out_weight = export_dir / f"{exp_name}.pth"
    shutil.copy2(weight, out_weight)

    exp_log = rvc_root / "logs" / exp_name
    indexes = sorted(
        exp_log.glob(f"added_IVF*_Flat_nprobe_*_{exp_name}_v2.index"),
        key=lambda p: p.stat().st_mtime,
    )
    if not indexes:
        indexes = sorted(
            (rvc_root / "assets" / "indices").glob(f"{exp_name}_*added*.index"),
            key=lambda p: p.stat().st_mtime,
        )
    if not indexes:
        raise FileNotFoundError("The generated added .index file was not found.")

    out_index = export_dir / f"{exp_name}.index"
    shutil.copy2(indexes[-1], out_index)

    print("\n" + "#" * 80, flush=True)
    print("TRAINING COMPLETE", flush=True)
    print(f"PTH   : {out_weight}", flush=True)
    print(f"INDEX : {out_index}", flush=True)
    print("#" * 80, flush=True)


def clean_previous(rvc_root: Path, exports_root: Path, exp_name: str) -> None:
    log_dir = rvc_root / "logs" / exp_name
    export_dir = exports_root / exp_name

    if log_dir.exists():
        shutil.rmtree(log_dir)
    if export_dir.exists():
        shutil.rmtree(export_dir)

    weights = rvc_root / "assets" / "weights"
    if weights.exists():
        for p in weights.glob(f"{exp_name}*.pth"):
            try:
                p.unlink()
            except OSError:
                pass

    indices = rvc_root / "assets" / "indices"
    if indices.exists():
        for p in indices.glob(f"{exp_name}_*.index"):
            try:
                p.unlink()
            except OSError:
                pass


def validate_character_name(name: str) -> str:
    name = name.strip()
    if not name:
        raise ValueError("Character name is empty.")
    if name in {".", ".."} or any(c in name for c in '<>:"/\\|?*'):
        raise ValueError("Character name contains characters that are not allowed on Windows.")
    return name


def main() -> None:
    parser = argparse.ArgumentParser(description="RVC model trainer worker")
    parser.add_argument("character")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--rvc-repo", required=True)
    parser.add_argument("--exports-root", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    exp_name = validate_character_name(args.character)
    rvc_root = Path(args.rvc_repo).resolve()
    exports_root = Path(args.exports_root).resolve()
    dataset_dir = Path(args.dataset).resolve()

    if args.epochs < 1:
        raise ValueError("Epochs must be at least 1.")
    if args.batch_size < 1:
        raise ValueError("Batch size must be at least 1.")
    if args.save_every < 1:
        raise ValueError("Save every must be at least 1.")
    if args.workers < 1:
        raise ValueError("Workers must be at least 1.")

    ensure_exists(rvc_root, "RVC-WebUI")
    ensure_rvc_language_override(rvc_root)
    os.environ["RVC_LANGUAGE"] = "en_US"
    ensure_exists(rvc_root / "assets" / "hubert_base", "HuBERT")
    ensure_exists(rvc_root / "assets" / "rmvpe" / "rmvpe.pt", "RMVPE")
    ensure_exists(rvc_root / "assets" / "pretrained_v2" / "f0G40k.pth", "pretrained generator")
    ensure_exists(rvc_root / "assets" / "pretrained_v2" / "f0D40k.pth", "pretrained discriminator")
    ensure_exists(rvc_root / "logs" / "mute", "mute assets")
    ensure_exists(dataset_dir, "dataset")

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Model Trainer currently requires an NVIDIA CUDA GPU. "
            "Voice Generation may run on CPU, but training must use an NVIDIA GPU."
        )

    files = audio_files(dataset_dir)
    if not files:
        raise RuntimeError(f"No supported audio files were found: {dataset_dir}")

    ffmpeg_dir = find_ffmpeg_dir()
    if not ffmpeg_dir:
        raise FileNotFoundError(
            "ffmpeg.exe was not found. Install FFmpeg, for example with WinGet."
        )

    (rvc_root / "assets" / "weights").mkdir(parents=True, exist_ok=True)
    (rvc_root / "assets" / "indices").mkdir(parents=True, exist_ok=True)
    exports_root.mkdir(parents=True, exist_ok=True)

    print(f"FFmpeg   : {Path(ffmpeg_dir) / 'ffmpeg.exe'}", flush=True)
    print(f"Character : {exp_name}", flush=True)
    print(f"Dataset   : {dataset_dir}", flush=True)
    print(f"Files     : {len(files)}", flush=True)
    print(f"Epochs    : {args.epochs}", flush=True)
    print(f"Batch     : {args.batch_size}", flush=True)
    print(f"GPU       : {args.gpu}", flush=True)
    print("RVC       : v2 / 40k / F0 ON / RMVPE", flush=True)
    print("Sleep     : prevented while training", flush=True)

    log_dir = rvc_root / "logs" / exp_name
    if args.fresh:
        clean_previous(rvc_root, exports_root, exp_name)
        print("[ZUNDANEN] FRESH_CLEANED", flush=True)
    elif log_dir.exists() and any(log_dir.iterdir()):
        raise RuntimeError(
            f"Existing training logs were found: {log_dir}\n"
            "Enable Fresh training to start over."
        )

    prevent_sleep(True)
    try:
        log_dir.mkdir(parents=True, exist_ok=True)

        print("[PHASE] PREPROCESS", flush=True)
        run([
            PYTHON,
            "-m", "train.preprocess",
            dataset_dir,
            "40000",
            str(args.workers),
            log_dir,
            "False",
            "3.7",
        ], rvc_root=rvc_root)

        print("[PHASE] F0", flush=True)
        run([
            PYTHON,
            "-m", "train.dataset.extract_f0",
            "cuda",
            "1",
            "0",
            args.gpu,
            log_dir,
            "True",
        ], rvc_root=rvc_root)

        print("[PHASE] HUBERT", flush=True)
        run([
            PYTHON,
            "-m", "train.dataset.extract_hubert_feature",
            f"cuda:{args.gpu}",
            "1",
            "0",
            args.gpu,
            log_dir,
            "v2",
            "True",
        ], rvc_root=rvc_root)

        prepare_filelist(rvc_root, exp_name, "v2", "40k", speaker_id=0)
        prepare_config(rvc_root, exp_name, "v2", "40k")

        pretrained_g = rvc_root / "assets" / "pretrained_v2" / "f0G40k.pth"
        pretrained_d = rvc_root / "assets" / "pretrained_v2" / "f0D40k.pth"

        print("[PHASE] TRAIN", flush=True)
        train_env = os.environ.copy()
        train_env["RVC_CUDA_GRAPH"] = "0"
        run_training([
            PYTHON,
            "-m", "train.train",
            "-e", exp_name,
            "-sr", "40k",
            "-f0", "1",
            "-bs", str(args.batch_size),
            "-g", args.gpu,
            "-te", str(args.epochs),
            "-se", str(args.save_every),
            "-pg", pretrained_g,
            "-pd", pretrained_d,
            "-l", "1",
            "-c", "0",
            "-sw", "1",
            "-v", "v2",
        ], rvc_root=rvc_root, exports_root=exports_root, exp_name=exp_name, env=train_env)

        print("[PHASE] INDEX", flush=True)
        run([
            PYTHON,
            "-m", "train.train_index",
            exp_name,
            "v2",
            rvc_root / "assets" / "indices",
            str(os.cpu_count() or 1),
            "single",
        ], rvc_root=rvc_root)

        export_results(rvc_root, exports_root, exp_name)
    finally:
        prevent_sleep(False)


if __name__ == "__main__":
    main()
