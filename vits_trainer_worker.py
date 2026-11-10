from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

AUDIO_EXTS = {".wav"}


def log(msg: str) -> None:
    print(msg, flush=True)


def path_fingerprint(path: Path | None) -> dict:
    if not path:
        return {"path": "", "size": 0, "mtime_ns": 0}
    path = path.resolve()
    st = path.stat()
    return {"path": str(path), "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}


def paired_dataset_files(root: Path) -> list[tuple[Path, Path]]:
    """Return unique same-stem WAV/TXT training pairs below *root*.

    Only stems that exist on both sides are considered. This deliberately
    ignores old Finnish VITS preview/sample WAVs and diagnostic TXT files that do not
    form a training pair. A stem is rejected only when the *paired* stem is
    ambiguous (multiple WAVs or multiple TXTs).
    """
    wavs: dict[str, list[Path]] = {}
    txts: dict[str, list[Path]] = {}
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext == ".wav":
            wavs.setdefault(p.stem, []).append(p)
        elif ext == ".txt":
            txts.setdefault(p.stem, []).append(p)
    common = sorted(set(wavs) & set(txts))
    if not common:
        raise RuntimeError(f"No same-stem WAV+TXT training pairs found in dataset: {root}")
    ambiguous = [stem for stem in common if len(wavs[stem]) != 1 or len(txts[stem]) != 1]
    if ambiguous:
        preview = ", ".join(ambiguous[:10])
        more = " ..." if len(ambiguous) > 10 else ""
        raise RuntimeError(f"Ambiguous paired WAV/TXT stems in dataset ({len(ambiguous)}): {preview}{more}")
    return [(wavs[stem][0], txts[stem][0]) for stem in common]


def dataset_fingerprint(root: Path) -> tuple[str, int, int]:
    h = hashlib.sha256()
    pairs = paired_dataset_files(root)
    for wav, txt in pairs:
        for p in (wav, txt):
            rel = p.relative_to(root).as_posix()
            st = p.stat()
            h.update(rel.encode("utf-8", "surrogatepass"))
            h.update(b"\0")
            h.update(str(int(st.st_size)).encode())
            h.update(b":")
            h.update(str(int(st.st_mtime_ns)).encode())
            h.update(b"\n")
    return h.hexdigest(), len(pairs), len(pairs)


def config_hash(payload: dict) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def run_stream(cmd: list[str], *, cwd: Path, env: dict | None = None) -> None:
    log("[CMD] " + subprocess.list2cmdline([str(x) for x in cmd]))
    proc = subprocess.Popen(
        [str(x) for x in cmd],
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line.rstrip("\r\n"), flush=True)
    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"Command failed with exit code {code}: {cmd[0]}")


def rvc_supports_directory_batch(rvc_python: Path, rvc_repo: Path, env: dict) -> bool:
    """Return True when the installed infer.cli advertises directory recursion.

    Older RVC CLI builds used by Zundanen may only accept one input WAV at a
    time.  Prefer the newer directory mode so the model is loaded once, but
    retain a compatible per-file fallback rather than failing a long job.
    """
    try:
        proc = subprocess.run(
            [str(rvc_python), "-m", "infer.cli", "--help"],
            cwd=str(rvc_repo),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
        help_text = proc.stdout or ""
        return proc.returncode == 0 and "--recursive" in help_text and "--format" in help_text
    except Exception as exc:
        log(f"[RVC] CLI capability check failed; using compatible per-file mode: {exc}")
        return False


def rvc_single_file_command(
    *, rvc_python: Path, model: Path, index: Path | None, src: Path, dst: Path,
    pitch: int, index_rate: float, protect: float,
) -> list[str]:
    cmd = [str(rvc_python), "-m", "infer.cli", "--model", str(model)]
    if index is not None:
        cmd += ["--index", str(index)]
    cmd += [
        "--input", str(src), "--output", str(dst),
        "--f0-method", "rmvpe", "--pitch", str(int(pitch)),
        "--index-rate", str(float(index_rate)), "--protect", str(float(protect)),
    ]
    return cmd


def prepare_rvc_dataset(
    *, source: Path, cache_root: Path, rvc_python: Path, rvc_repo: Path,
    model: Path, index: Path | None, pitch: int, index_rate: float, protect: float,
) -> Path:
    source_digest, wav_count, txt_count = dataset_fingerprint(source)
    effective_index_rate = float(index_rate)
    if index is None and effective_index_rate > 0:
        effective_index_rate = 0.0
        log("[RVC] No .index supplied; index_rate forced to 0.0 (RVC index is optional).")

    signature = {
        "schema": "zundanen-rvc-preprocess-v1",
        "source": str(source.resolve()),
        "source_digest": source_digest,
        "source_wavs": wav_count,
        "source_txt": txt_count,
        "model": path_fingerprint(model),
        "index": path_fingerprint(index),
        "pitch": int(pitch),
        "index_rate": round(effective_index_rate, 6),
        "protect": round(float(protect), 6),
        "f0_method": "rmvpe",
    }
    key = config_hash(signature)
    job_root = cache_root / f"{source.name}_{key}"
    converted = job_root / "converted"
    converted.mkdir(parents=True, exist_ok=True)
    meta_path = job_root / "rvc_preprocess.json"
    meta_path.write_text(json.dumps(signature, ensure_ascii=False, indent=2), encoding="utf-8")

    pairs = paired_dataset_files(source)
    source_wavs = [wav for wav, _txt in pairs]
    source_txts = [txt for _wav, txt in pairs]
    missing: list[Path] = []
    for src in source_wavs:
        rel = src.relative_to(source)
        dst = (converted / rel).with_suffix(".wav")
        if not dst.is_file() or dst.stat().st_size < 64:
            missing.append(src)

    reused = len(source_wavs) - len(missing)
    log(f"[RVC] cache={job_root}")
    log(f"[RVC] source={len(source_wavs)} reuse={reused} convert={len(missing)}")

    # Copy labels even for a fully reused conversion cache. prepare_dataset.py
    # pairs WAV/TXT by stem and should always see the current original labels.
    for txt in source_txts:
        rel = txt.relative_to(source)
        dst = converted / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if (not dst.exists()) or txt.stat().st_mtime_ns != dst.stat().st_mtime_ns or txt.stat().st_size != dst.stat().st_size:
            shutil.copy2(txt, dst)

    if missing:
        staging = job_root / "staging_missing"
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True, exist_ok=True)
        for i, src in enumerate(missing, 1):
            rel = src.relative_to(source)
            dst = staging / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
            if i == 1 or i % 500 == 0 or i == len(missing):
                log(f"Progress: {i} / {len(missing)}")

        env = os.environ.copy()
        env.update({"LANG": "en_US.UTF-8", "LC_ALL": "en_US.UTF-8", "LANGUAGE": "en_US", "RVC_LANGUAGE": "en_US"})
        try:
            if rvc_supports_directory_batch(rvc_python, rvc_repo, env):
                log("[RVC] Using directory batch mode (model loads once).")
                cmd = [
                    str(rvc_python), "-m", "infer.cli", "--model", str(model),
                ]
                if index is not None:
                    cmd += ["--index", str(index)]
                cmd += [
                    "--input", str(staging), "--output", str(converted),
                    "--format", "wav", "--recursive", "--f0-method", "rmvpe",
                    "--pitch", str(int(pitch)), "--index-rate", str(effective_index_rate),
                    "--protect", str(float(protect)), "--overwrite",
                ]
                run_stream(cmd, cwd=rvc_repo, env=env)
            else:
                log("[RVC] Installed CLI has no directory batch mode; falling back to one-WAV-at-a-time conversion.")
                for i, staged_src in enumerate(sorted(staging.rglob("*.wav")), 1):
                    rel = staged_src.relative_to(staging)
                    dst = converted / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    cmd = rvc_single_file_command(
                        rvc_python=rvc_python, model=model, index=index, src=staged_src, dst=dst,
                        pitch=pitch, index_rate=effective_index_rate, protect=protect,
                    )
                    run_stream(cmd, cwd=rvc_repo, env=env)
                    log(f"Progress: {i} / {len(missing)}")
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    converted_count = sum(1 for _ in converted.rglob("*.wav"))
    if converted_count != wav_count:
        raise RuntimeError(f"RVC preprocessing incomplete: expected {wav_count} WAV, found {converted_count}")
    log(f"[RVC] COMPLETE wav={converted_count} cache={converted}")
    return converted


def is_prepared_dataset(root: Path) -> bool:
    return all((root / x).is_file() for x in ("train.csv", "eval.csv", "vocab.json"))


def prepare_step7(*, source: Path, out: Path, python: Path, research: Path, source_name: str) -> Path:
    if is_prepared_dataset(source):
        log(f"[PREPARE] Already prepared; using {source}")
        return source
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(python), "-u", str(research / "prepare_dataset.py"),
        "--wav-dir", str(source),
        "--label-dir", str(source),
        "--out", str(out),
        "--source-name", source_name,
        "--device", "cuda",
    ]
    run_stream(cmd, cwd=research, env=os.environ.copy())
    if not is_prepared_dataset(out):
        raise RuntimeError(f"Finnish VITS dataset preparation did not create train/eval/vocab in: {out}")
    return out


def latest_checkpoint(run_dir: Path, target_steps: int) -> Path | None:
    exact = run_dir / "checkpoints" / f"vits_{target_steps:06d}.pt"
    if exact.is_file():
        return exact
    cps = sorted((run_dir / "checkpoints").glob("vits_*.pt")) if (run_dir / "checkpoints").is_dir() else []
    return cps[-1] if cps else None


def main() -> int:
    ap = argparse.ArgumentParser(description="Zundanen VITS trainer orchestration worker")
    ap.add_argument("--root", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--mode", choices=("scratch", "finetune", "resume"), default="scratch")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--use-rvc", action="store_true")
    ap.add_argument("--rvc-python", default="")
    ap.add_argument("--rvc-repo", default="")
    ap.add_argument("--rvc-model", default="")
    ap.add_argument("--rvc-index", default="")
    ap.add_argument("--rvc-pitch", type=int, default=0)
    ap.add_argument("--rvc-index-rate", type=float, default=0.75)
    ap.add_argument("--rvc-protect", type=float, default=0.33)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    source = Path(args.dataset).resolve()
    research = root / "research" / "finnish_style_tts"
    python = root / "runtime" / "tts" / ".venv" / "Scripts" / "python.exe"
    cache_root = root / "outputs" / "trainer_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    if not source.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {source}")
    if not python.is_file():
        raise FileNotFoundError(f"TTS Python not found: {python}")
    for required in (research / "prepare_dataset.py", research / "train_vits.py"):
        if not required.is_file():
            raise FileNotFoundError(f"Required Finnish VITS file missing: {required}")

    train_source = source
    if args.use_rvc:
        log("[PHASE] RVC_PREPROCESS")
        rvc_python = Path(args.rvc_python).resolve()
        rvc_repo = Path(args.rvc_repo).resolve()
        model = Path(args.rvc_model).resolve()
        index = Path(args.rvc_index).resolve() if args.rvc_index.strip() else None
        if not rvc_python.is_file():
            raise FileNotFoundError(f"RVC Python not found: {rvc_python}")
        if not rvc_repo.is_dir():
            raise FileNotFoundError(f"RVC repo not found: {rvc_repo}")
        if not model.is_file() or model.suffix.lower() != ".pth":
            raise FileNotFoundError(f"RVC .pth not found: {model}")
        if index is not None and (not index.is_file() or index.suffix.lower() != ".index"):
            raise FileNotFoundError(f"RVC .index not found: {index}")
        train_source = prepare_rvc_dataset(
            source=source, cache_root=cache_root / "rvc", rvc_python=rvc_python,
            rvc_repo=rvc_repo, model=model, index=index, pitch=args.rvc_pitch,
            index_rate=args.rvc_index_rate, protect=args.rvc_protect,
        )

    log("[PHASE] PREPARE")
    if is_prepared_dataset(train_source):
        prepared = train_source
        log(f"[PREPARE] Already prepared; using {prepared}")
    else:
        source_digest, wav_count, _ = dataset_fingerprint(train_source)
        prep_key = config_hash({"source": str(train_source), "digest": source_digest, "frontend": "step7-fixed-vocab-finbert-large"})
        prepared_out = cache_root / "prepared" / f"{train_source.name}_{prep_key}"
        prepared = prepare_step7(
            source=train_source, out=prepared_out, python=python, research=research,
            source_name=(args.run_name or "dataset"),
        )

    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint.strip() else None
    if args.mode in {"finetune", "resume"} and (checkpoint is None or not checkpoint.is_file()):
        raise FileNotFoundError(f"{args.mode} requires a checkpoint file: {checkpoint}")

    log("[PHASE] TRAIN")
    cmd = [
        str(python), "-u", str(research / "train_vits.py"),
        "--dataset", str(prepared),
        "--run-name", args.run_name,
        "--steps", str(max(1, args.steps)),
        "--save-every", str(max(1, args.save_every)),
        "--batch-size", str(max(1, args.batch_size)),
        "--workers", str(max(0, args.workers)),
        "--lr", str(float(args.lr)),
    ]
    if args.mode == "finetune":
        cmd += ["--init-checkpoint", str(checkpoint)]
    elif args.mode == "resume":
        cmd += ["--resume", str(checkpoint)]
    run_stream(cmd, cwd=research, env=os.environ.copy())

    run_dir = prepared / "vits_runs" / args.run_name
    cp = latest_checkpoint(run_dir, args.steps)
    if cp is None:
        raise RuntimeError(f"Training finished but no checkpoint was found in {run_dir / 'checkpoints'}")
    log(f"CHECKPOINT : {cp}")
    log(f"DATASET    : {prepared}")
    if args.use_rvc:
        log(f"RVC DATA   : {train_source}")
    log("TRAINING COMPLETE")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("[STOPPED] Interrupted")
        raise SystemExit(130)
    except Exception as exc:
        import traceback
        log(f"[ERROR] {exc}")
        log(traceback.format_exc())
        raise SystemExit(1)
