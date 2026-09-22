from __future__ import annotations

import csv
import io
import json
import logging
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import traceback
import webbrowser
import uuid
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from rvc_locale import ensure_rvc_language_override
from aalto_alignment import AlignmentError, run_alignment

logging.getLogger("werkzeug").setLevel(logging.ERROR)

APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = APP_DIR / "config.json"
OUTPUT_DIR = APP_DIR / "outputs"
TEMP_DIR = APP_DIR / "temp"
BATCH_STATE_PATH = TEMP_DIR / "batch_queue_state.json"
PREVIEW_WAV_PATH = TEMP_DIR / "voice_preview.wav"
PREVIEW_TTS_WAV_PATH = TEMP_DIR / "tts_preview.wav"
PREVIEW_TTS_TEXT_PATH = TEMP_DIR / "tts_preview.txt"
PREVIEW_ALIGNMENT_PATH = TEMP_DIR / "alignment_preview.json"
PREVIEW_SOURCE_META_PATH = TEMP_DIR / "preview_source.json"
PITCH_EDIT_TTS_WAV_PATH = TEMP_DIR / "tts_pitch_preview.wav"
PITCH_EDIT_WAV_PATH = TEMP_DIR / "voice_pitch_preview.wav"
WORLD_ACCENT_WORKER_PATH = APP_DIR / "world_accent_worker.py"
ACCENT_SUPPORT_BAT_PATH = APP_DIR / "install_accent_support.bat"
ACCENT_SUPPORT_PS1_PATH = APP_DIR / "install_accent_support.ps1"
ACCENT_SUPPORT_STATUS_PATH = APP_DIR / "runtime" / "accent_support_status.json"
OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)
for preview_temp in (
    PREVIEW_WAV_PATH,
    PREVIEW_TTS_WAV_PATH,
    PREVIEW_TTS_TEXT_PATH,
    PREVIEW_ALIGNMENT_PATH,
    PREVIEW_SOURCE_META_PATH,
    PITCH_EDIT_TTS_WAV_PATH,
    PITCH_EDIT_WAV_PATH,
):
    try:
        preview_temp.unlink(missing_ok=True)
    except OSError:
        pass

app = Flask(__name__, static_folder="static", static_url_path="/static")
generation_lock = threading.Lock()
training_state_lock = threading.Lock()
training_process_lock = threading.Lock()
training_process = None
training_state = {
    "running": False,
    "done": False,
    "stopping": False,
    "character": "",
    "dataset": "",
    "phase": "idle",
    "phase_label": "Ready",
    "epoch": 0,
    "epochs": 100,
    "progress": 0.0,
    "audio_files": 0,
    "log": [],
    "error": "",
    "result": {},
}

batch_state_lock = threading.Lock()
batch_process_lock = threading.Lock()
batch_current_process = None
batch_stop_event = threading.Event()
batch_state = {
    "running": False,
    "done": False,
    "stopping": False,
    "resumable": False,
    "queue_id": "",
    "mode": "lines",
    "create_subtitle_txt": True,
    "save_source_audio": False,
    "fix_seed": False,
    "seed": 0,
    "total": 0,
    "current_index": 0,
    "completed": 0,
    "failed": 0,
    "pending": 0,
    "progress": 0.0,
    "current_text": "",
    "current_filename": "",
    "output_dir": "",
    "items": [],
    "log": [],
    "error": "",
}


class BatchStopped(RuntimeError):
    pass


def default_config() -> dict:
    runtime = APP_DIR / "runtime"
    cb = runtime / "tts" / "Chatterbox-Finnish"
    rvc = runtime / "rvc"
    return {
        "chatterbox_repo": str(cb),
        "chatterbox_weights": str(
            cb / "models" / "best_finnish_multilingual_cp986.safetensors"
        ),
        "reference_audio": str(cb / "samples" / "reference_finnish.wav"),
        "rvc_python": str(rvc / ".venv" / "Scripts" / "python.exe"),
        "rvc_repo": str(rvc / "RVC-WebUI"),
        "exports_root": str(rvc / "exports"),
        "tts": {
            "repetition_penalty": 1.2,
            "temperature": 0.8,
            "exaggeration": 0.6,
        },
        "voice_presets": {
            "Zundamon": {"pitch": 0, "index_rate": 0.55, "protect": 0.15},
            "Metan": {"pitch": 0, "index_rate": 0.55, "protect": 0.15},
        },
        "trainer": {
            "epochs": 100,
            "batch_size": 6,
            "save_every": 10,
            "workers": min(8, os.cpu_count() or 1),
            "gpu": "0",
            "fresh": True,
        },
    }


def load_config() -> dict:
    defaults = default_config()
    if not CONFIG_PATH.exists():
        save_config(defaults)
        return defaults
    try:
        user = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return defaults

    cfg = defaults | user
    cfg["tts"] = defaults["tts"] | user.get("tts", {})
    cfg["voice_presets"] = defaults["voice_presets"] | user.get("voice_presets", {})
    cfg["trainer"] = defaults["trainer"] | user.get("trainer", {})
    return cfg


def save_config(cfg: dict) -> None:
    CONFIG_PATH.write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )



def apply_world_accent_edits(
    source_wav: Path, output_wav: Path, accents: list[dict], cfg: dict
) -> dict:
    """Apply automatic word-accent edits with WORLD/PyWORLD."""
    rvc_python = Path(cfg.get("rvc_python", ""))
    if not rvc_python.exists():
        raise RuntimeError(f"RVC Python was not found: {rvc_python}")
    if not WORLD_ACCENT_WORKER_PATH.exists():
        raise RuntimeError(f"WORLD accent worker is missing: {WORLD_ACCENT_WORKER_PATH.name}")
    if not source_wav.exists():
        raise FileNotFoundError(f"Input WAV was not found: {source_wav}")

    spec_path = TEMP_DIR / f"world_accents_{uuid.uuid4().hex[:12]}.json"
    spec_path.write_text(json.dumps(accents, ensure_ascii=False), encoding="utf-8")
    try:
        proc = subprocess.run(
            [
                str(rvc_python), str(WORLD_ACCENT_WORKER_PATH),
                "--input", str(source_wav),
                "--output", str(output_wav),
                "--accents", str(spec_path),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=240,
            creationflags=(getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0),
        )
        stdout = (proc.stdout or "").strip()
        data = None
        for line in reversed(stdout.splitlines()):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                data = json.loads(line)
                break
            except Exception:
                continue
        if proc.returncode != 0 or not isinstance(data, dict) or not data.get("ok"):
            detail = ((data or {}).get("error") if isinstance(data, dict) else None) or proc.stderr or stdout or "Unknown WORLD accent editing error."
            raise RuntimeError("WORLD accent editing failed:\n" + str(detail)[-6000:])
        if not output_wav.exists():
            raise FileNotFoundError(f"WORLD accent output WAV was not created: {output_wav}")
        return data
    finally:
        try:
            spec_path.unlink(missing_ok=True)
        except OSError:
            pass


def run_rvc_on_audio(*, input_wav: Path, output_wav: Path, voice: str, pitch: int | float, index_rate: float, protect: float, cfg: dict, batch: bool = False) -> Path:
    """Run the existing RVC model on an already-generated TTS WAV."""
    validate_config(cfg)
    voices = scan_voices(cfg)
    voice = str(voice).strip()
    if voice not in voices:
        raise ValueError(f"RVC model not found: {voice}")
    if not input_wav.exists():
        raise FileNotFoundError(f"Input WAV was not found: {input_wav}")

    pitch = int(round(float(pitch)))
    index_rate = max(0.0, min(1.0, float(index_rate)))
    protect = max(0.0, min(0.5, float(protect)))
    output_wav.parent.mkdir(parents=True, exist_ok=True)

    rvc_repo = Path(cfg["rvc_repo"])
    rvc_cmd = [
        cfg["rvc_python"], "-m", "infer.cli",
        "--model", voices[voice]["pth"],
        "--index", voices[voice]["index"],
        "--input", str(input_wav),
        "--output", str(output_wav),
        "--f0-method", "rmvpe",
        "--pitch", str(pitch),
        "--index-rate", str(index_rate),
        "--protect", str(protect),
        "--overwrite",
    ]
    rvc = _run_generation_process(rvc_cmd, cwd=rvc_repo, env=_rvc_env(rvc_repo), batch=batch)
    if rvc.returncode != 0:
        raise RuntimeError("RVC conversion failed:\n" + (rvc.stderr or rvc.stdout or "Unknown error")[-6000:])
    if not output_wav.exists():
        raise FileNotFoundError(f"Output WAV was not created: {output_wav}")
    return output_wav

def find_ffmpeg_dir() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return str(Path(found).resolve().parent)

    local = os.environ.get("LOCALAPPDATA")
    if not local:
        return None

    local = Path(local)
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


def scan_voices(cfg: dict) -> dict[str, dict]:
    root = Path(cfg["exports_root"]).expanduser()
    voices = {}
    if not root.exists():
        return voices

    for folder in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if not folder.is_dir():
            continue

        pths = list(folder.glob("*.pth"))
        indexes = list(folder.glob("*.index"))
        if not pths or not indexes:
            continue

        preferred_pth = folder / f"{folder.name}.pth"
        preferred_index = folder / f"{folder.name}.index"

        pth = preferred_pth if preferred_pth.exists() else max(
            pths, key=lambda p: p.stat().st_mtime
        )
        index = preferred_index if preferred_index.exists() else max(
            indexes, key=lambda p: p.stat().st_mtime
        )

        preset = cfg.get("voice_presets", {}).get(folder.name, {})
        voices[folder.name] = {
            "pth": str(pth),
            "index": str(index),
            "pitch": preset.get("pitch", 0),
            "index_rate": preset.get("index_rate", 0.55),
            "protect": preset.get("protect", 0.15),
        }
    return voices


def validate_config(cfg: dict) -> None:
    required = {
        "Chatterbox repo": Path(cfg["chatterbox_repo"]),
        "Finnish weights": Path(cfg["chatterbox_weights"]),
        "Reference audio": Path(cfg["reference_audio"]),
        "RVC Python": Path(cfg["rvc_python"]),
        "RVC repo": Path(cfg["rvc_repo"]),
        "RVC exports": Path(cfg["exports_root"]),
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required files/folders were not found:\n" + "\n".join(missing)
        )



def prevent_sleep(enabled: bool) -> None:
    if not sys.platform.startswith("win"):
        return
    try:
        import ctypes
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED if enabled else ES_CONTINUOUS
        ctypes.windll.kernel32.SetThreadExecutionState(flags)
    except Exception:
        pass


def safe_output_stem(text: str, fallback: str = "voice", limit: int = 42) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', " ", text)
    cleaned = re.sub(r"\s+", "_", cleaned.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("._ ")
    if not cleaned:
        cleaned = fallback
    return cleaned[:limit].rstrip("._ ") or fallback


def _rvc_env(rvc_repo: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(rvc_repo) + os.pathsep + env.get("PYTHONPATH", "")
    env["LANG"] = "en_US.UTF-8"
    env["LC_ALL"] = "en_US.UTF-8"
    env["LANGUAGE"] = "en_US"
    env["RVC_LANGUAGE"] = "en_US"
    ffdir = find_ffmpeg_dir()
    if ffdir:
        env["PATH"] = ffdir + os.pathsep + env.get("PATH", "")
    return env


def _run_generation_process(cmd: list[str], *, cwd: Path, env: dict | None = None, batch: bool = False) -> subprocess.CompletedProcess:
    global batch_current_process
    if not batch:
        return subprocess.run(
            cmd, cwd=str(cwd), env=env, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
        )

    if batch_stop_event.is_set():
        raise BatchStopped("Batch stopped")

    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform.startswith("win") else 0
    proc = subprocess.Popen(
        cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace", creationflags=creationflags,
    )
    with batch_process_lock:
        batch_current_process = proc
    try:
        stdout, stderr = proc.communicate()
        if batch_stop_event.is_set():
            raise BatchStopped("Batch stopped")
        return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
    finally:
        with batch_process_lock:
            if batch_current_process is proc:
                batch_current_process = None


def generate_voice_file(
    *, text: str, voice: str, pitch: int | float, index_rate: float,
    protect: float, expression: float, output: Path, cfg: dict | None = None,
    batch: bool = False, persist_preset: bool = False, seed: int | None = None,
    tts_copy: Path | None = None,
) -> Path:
    text = str(text).strip()
    voice = str(voice).strip()
    if not text:
        raise ValueError("Finnish text is empty.")

    cfg = cfg or load_config()
    validate_config(cfg)
    voices = scan_voices(cfg)
    if voice not in voices:
        raise ValueError(f"RVC model not found: {voice}")

    pitch = int(round(float(pitch)))
    index_rate = max(0.0, min(1.0, float(index_rate)))
    protect = max(0.0, min(0.5, float(protect)))
    expression = max(0.2, min(1.2, float(expression)))

    if persist_preset:
        cfg.setdefault("voice_presets", {})[voice] = {
            "pitch": pitch,
            "index_rate": round(index_rate, 3),
            "protect": round(protect, 3),
        }
        cfg.setdefault("tts", {})["exaggeration"] = round(expression, 3)
        save_config(cfg)

    token = uuid.uuid4().hex[:12]
    text_file = TEMP_DIR / f"text_{token}.txt"
    temp_tts = TEMP_DIR / f"tts_{token}.wav"
    output.parent.mkdir(parents=True, exist_ok=True)
    text_file.write_text(text, encoding="utf-8")

    try:
        tts_cmd = [
            sys.executable, str(APP_DIR / "tts_worker.py"),
            "--repo", cfg["chatterbox_repo"],
            "--weights", cfg["chatterbox_weights"],
            "--reference", cfg["reference_audio"],
            "--text-file", str(text_file),
            "--output", str(temp_tts),
            "--repetition-penalty", str(float(cfg["tts"]["repetition_penalty"])),
            "--temperature", str(float(cfg["tts"]["temperature"])),
            "--exaggeration", str(expression),
        ]
        if seed is not None:
            tts_cmd.extend(["--seed", str(int(seed))])
        tts = _run_generation_process(tts_cmd, cwd=APP_DIR, batch=batch)
        if tts.returncode != 0:
            raise RuntimeError("Finnish TTS failed:\n" + (tts.stderr or tts.stdout or "Unknown error")[-6000:])

        # Preserve the raw Finnish TTS audio for optional forced alignment
        # before the RVC conversion changes the voice.
        if tts_copy is not None:
            tts_copy = Path(tts_copy)
            tts_copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(temp_tts, tts_copy)

        run_rvc_on_audio(
            input_wav=temp_tts, output_wav=output, voice=voice, pitch=pitch,
            index_rate=index_rate, protect=protect, cfg=cfg, batch=batch,
        )
        return output
    finally:
        for temp in (text_file, temp_tts):
            try:
                if temp.exists():
                    temp.unlink()
            except OSError:
                pass


def _batch_snapshot() -> dict:
    with batch_state_lock:
        return json.loads(json.dumps(batch_state, ensure_ascii=False))


def _save_batch_state_locked() -> None:
    BATCH_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = BATCH_STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(batch_state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(BATCH_STATE_PATH)


def save_batch_state() -> None:
    with batch_state_lock:
        _save_batch_state_locked()


def set_batch(**updates) -> None:
    with batch_state_lock:
        batch_state.update(updates)
        _save_batch_state_locked()


def append_batch_log(message: str) -> None:
    with batch_state_lock:
        batch_state["log"].append(str(message))
        if len(batch_state["log"]) > 500:
            batch_state["log"] = batch_state["log"][-500:]
        _save_batch_state_locked()


def load_saved_batch_state() -> None:
    if not BATCH_STATE_PATH.exists():
        return
    try:
        saved = json.loads(BATCH_STATE_PATH.read_text(encoding="utf-8"))
        if not isinstance(saved, dict):
            return
        with batch_state_lock:
            batch_state.update(saved)
            batch_state.setdefault("create_subtitle_txt", True)
            batch_state.setdefault("save_source_audio", False)
            batch_state.setdefault("fix_seed", False)
            batch_state.setdefault("seed", 0)
            # A server restart cannot leave a worker running. Requeue an interrupted item.
            for item in batch_state.get("items", []):
                if "seed" not in item or item.get("seed") is None:
                    if batch_state.get("fix_seed"):
                        item["seed"] = int(batch_state.get("seed", 0))
                    else:
                        item["seed"] = secrets.randbelow(2147483648)
                if item.get("status") == "processing":
                    item["status"] = "pending"
                    item["error"] = ""
            batch_state["running"] = False
            batch_state["stopping"] = False
            pending = sum(1 for item in batch_state.get("items", []) if item.get("status") == "pending")
            batch_state["pending"] = pending
            batch_state["resumable"] = pending > 0
            batch_state["done"] = pending == 0 and bool(batch_state.get("items"))
            _save_batch_state_locked()
    except Exception:
        pass


def _as_number(value, default, *, lo=None, hi=None, integer=False):
    if value is None or str(value).strip() == "":
        number = default
    else:
        number = float(value)
    if lo is not None:
        number = max(lo, number)
    if hi is not None:
        number = min(hi, number)
    return int(round(number)) if integer else float(number)


def _as_seed(value, *, allow_blank: bool = False) -> int | None:
    if value is None or str(value).strip() == "":
        if allow_blank:
            return None
        raise ValueError("Seed must be an integer between 0 and 2147483647.")
    try:
        seed = int(str(value).strip())
    except (TypeError, ValueError):
        raise ValueError("Seed must be an integer between 0 and 2147483647.")
    if seed < 0 or seed > 2147483647:
        raise ValueError("Seed must be between 0 and 2147483647.")
    return seed


def build_batch_items(data: dict, cfg: dict) -> tuple[list[dict], str]:
    raw = str(data.get("content", ""))
    mode = str(data.get("mode", "lines")).lower()
    defaults = {
        "voice": str(data.get("voice", "")).strip(),
        "pitch": data.get("pitch", 0),
        "index_rate": data.get("index_rate", 0.55),
        "protect": data.get("protect", 0.15),
        "expression": data.get("expression", cfg["tts"]["exaggeration"]),
        "seed": data.get("seed", 0),
        "fix_seed": bool(data.get("fix_seed", False)),
    }
    voices = scan_voices(cfg)
    if not voices:
        raise ValueError("No RVC models were found.")
    if defaults["voice"] not in voices:
        raise ValueError(f"RVC model not found: {defaults['voice']}")

    rows: list[dict] = []
    if mode == "csv":
        reader = csv.DictReader(io.StringIO(raw.lstrip("\ufeff")))
        if not reader.fieldnames or "text" not in {str(x).strip().lower() for x in reader.fieldnames if x}:
            raise ValueError("CSV must contain a text column.")
        for row in reader:
            normalized = {str(k).strip().lower(): v for k, v in row.items() if k is not None}
            text = str(normalized.get("text", "") or "").strip()
            if not text:
                continue
            rows.append({
                "text": text,
                "filename": str(normalized.get("filename", "") or "").strip(),
                "voice": str(normalized.get("character", normalized.get("voice", defaults["voice"])) or defaults["voice"]).strip(),
                "pitch": normalized.get("pitch", defaults["pitch"]),
                "index_rate": normalized.get("index_rate", defaults["index_rate"]),
                "protect": normalized.get("protect", defaults["protect"]),
                "expression": normalized.get("expression", defaults["expression"]),
                "seed": normalized.get("seed", ""),
            })
    else:
        mode = "lines"
        for line in raw.splitlines():
            text = line.strip()
            if text:
                rows.append({"text": text, "filename": "", **defaults, "seed": ""})

    if not rows:
        raise ValueError("No text items were found.")
    if len(rows) > 5000:
        raise ValueError("A batch can contain at most 5000 items.")

    width = max(3, len(str(len(rows))))
    items: list[dict] = []
    for idx, row in enumerate(rows, 1):
        voice = str(row.get("voice", defaults["voice"])).strip()
        if voice not in voices:
            raise ValueError(f"RVC model not found on row {idx}: {voice}")
        pitch = _as_number(row.get("pitch"), voices[voice]["pitch"], lo=-24, hi=24, integer=True)
        index_rate = _as_number(row.get("index_rate"), voices[voice]["index_rate"], lo=0, hi=1)
        protect = _as_number(row.get("protect"), voices[voice]["protect"], lo=0, hi=0.5)
        expression = _as_number(row.get("expression"), cfg["tts"]["exaggeration"], lo=0.2, hi=1.2)

        row_seed = _as_seed(row.get("seed"), allow_blank=True)
        if row_seed is not None:
            seed = row_seed
        elif defaults["fix_seed"]:
            seed = _as_seed(defaults["seed"])
        else:
            seed = secrets.randbelow(2147483648)

        number = f"{idx:0{width}d}"
        custom = str(row.get("filename", "") or "").strip()
        if mode == "csv" and custom:
            stem = safe_output_stem(Path(custom).stem, fallback="")
            if stem in {number, str(idx)}:
                filename = f"{number}.wav"
            elif stem.startswith(number + "_"):
                filename = f"{stem}.wav"
            elif stem:
                filename = f"{number}_{stem}.wav"
            else:
                filename = f"{number}.wav"
        else:
            filename = f"{number}.wav"

        # Keep the seed visible in the saved filename so a generated voice can
        # be reproduced later without opening metadata. Preserve any existing
        # number/custom stem and append the seed at the end.
        filename_path = Path(filename)
        filename = f"{filename_path.stem}_seed{seed}{filename_path.suffix or '.wav'}"

        items.append({
            "id": idx,
            "text": row["text"],
            "voice": voice,
            "pitch": pitch,
            "index_rate": round(index_rate, 3),
            "protect": round(protect, 3),
            "expression": round(expression, 3),
            "seed": seed,
            "filename": filename,
            "status": "pending",
            "output": "",
            "error": "",
        })
    return items, mode


def _write_failed_csv(snapshot: dict) -> None:
    failed = [item for item in snapshot.get("items", []) if item.get("status") == "failed"]
    if not failed or not snapshot.get("output_dir"):
        return
    path = Path(snapshot["output_dir"]) / "failed_items.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filename", "text", "character", "pitch", "index_rate", "protect", "expression", "seed", "error"])
        for item in failed:
            writer.writerow([
                item.get("filename", ""), item.get("text", ""), item.get("voice", ""),
                item.get("pitch", ""), item.get("index_rate", ""), item.get("protect", ""),
                item.get("expression", ""), item.get("seed", ""), item.get("error", ""),
            ])




def _batch_source_paths(final_wav: Path) -> tuple[Path, Path]:
    """Visible Batch source-audio folder used only when the user opts in."""
    source_dir = final_wav.parent / "source_audio"
    stem = final_wav.stem
    return source_dir / f"{stem}_source.wav", source_dir / f"{stem}_source.json"


def _voice_source_paths(final_wav: Path) -> tuple[Path, Path]:
    """Voice Generation source files live beside the saved final WAV."""
    stem = final_wav.stem
    return final_wav.parent / f"{stem}_source.wav", final_wav.parent / f"{stem}_source.json"


def _find_batch_source(final_wav: Path) -> tuple[Path | None, dict]:
    """Find a saved clean TTS source. New visible layouts first; legacy layouts remain supported."""
    candidates = []
    # Voice Generation: same directory as the final WAV.
    candidates.append(_voice_source_paths(final_wav))
    # Batch Generation: all clean sources collected in one visible subfolder.
    candidates.append(_batch_source_paths(final_wav))
    # Backward compatibility with earlier hidden-sidecar builds.
    for source_dir in (final_wav.parent / ".zundanen_sources", final_wav.parent / "_zundanen_sources"):
        candidates.append((source_dir / final_wav.name, source_dir / f"{final_wav.stem}.json"))

    for source_wav, source_meta in candidates:
        if not source_wav.is_file():
            continue
        meta = {}
        if source_meta.is_file():
            try:
                loaded = json.loads(source_meta.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    meta = loaded
            except Exception:
                pass
        return source_wav, meta
    return None, {}

def batch_runner() -> None:
    batch_stop_event.clear()
    prevent_sleep(True)
    acquired = generation_lock.acquire(blocking=False)
    if not acquired:
        set_batch(running=False, error="Another generation job is already running.", resumable=True)
        prevent_sleep(False)
        return
    try:
        cfg = load_config()
        validate_config(cfg)
        append_batch_log("[BATCH] STARTED")
        while True:
            with batch_state_lock:
                pending_indices = [i for i, item in enumerate(batch_state["items"]) if item.get("status") == "pending"]
                if not pending_indices:
                    break
                if batch_stop_event.is_set() or batch_state.get("stopping"):
                    break
                i = pending_indices[0]
                item = batch_state["items"][i]
                item["status"] = "processing"
                item["error"] = ""
                batch_state["current_index"] = i + 1
                batch_state["current_text"] = item["text"]
                batch_state["current_filename"] = item["filename"]
                _save_batch_state_locked()

            append_batch_log(f"[BATCH] ITEM_START {i+1}/{len(batch_state['items'])} {item['filename']} SEED={item.get('seed', '')}")
            output = Path(batch_state["output_dir"]) / item["filename"]
            source_wav, source_meta = _batch_source_paths(output)
            with batch_state_lock:
                save_source_audio = bool(batch_state.get("save_source_audio", False))
            try:
                generate_voice_file(
                    text=item["text"], voice=item["voice"], pitch=item["pitch"],
                    index_rate=item["index_rate"], protect=item["protect"],
                    expression=item["expression"], output=output, cfg=cfg, batch=True,
                    seed=item.get("seed"), tts_copy=(source_wav if save_source_audio else None),
                )
                if save_source_audio:
                    source_meta.parent.mkdir(parents=True, exist_ok=True)
                    source_meta.write_text(json.dumps({
                        "format": "zundanen_batch_source_v2",
                        "final_wav": output.name,
                        "text": item["text"],
                        "voice": item["voice"],
                        "pitch": item["pitch"],
                        "index_rate": item["index_rate"],
                        "protect": item["protect"],
                        "expression": item["expression"],
                        "seed": item.get("seed"),
                    }, ensure_ascii=False, indent=2), encoding="utf-8")
                # Optional YMM4-friendly sidecar subtitle: same basename as the WAV.
                with batch_state_lock:
                    create_subtitle_txt = bool(batch_state.get("create_subtitle_txt", True))
                if create_subtitle_txt:
                    output.with_suffix(".txt").write_text(item["text"], encoding="utf-8")
                with batch_state_lock:
                    item["status"] = "done"
                    item["output"] = str(output)
                    batch_state["completed"] = sum(1 for x in batch_state["items"] if x.get("status") == "done")
                    batch_state["failed"] = sum(1 for x in batch_state["items"] if x.get("status") == "failed")
                    batch_state["pending"] = sum(1 for x in batch_state["items"] if x.get("status") == "pending")
                    finished = batch_state["completed"] + batch_state["failed"]
                    batch_state["progress"] = 100.0 * finished / max(1, batch_state["total"])
                    _save_batch_state_locked()
                append_batch_log(f"[BATCH] ITEM_DONE {i+1}/{len(batch_state['items'])} {item['filename']}")
            except BatchStopped:
                for temp_source in (source_wav, source_meta):
                    try:
                        temp_source.unlink(missing_ok=True)
                    except OSError:
                        pass
                with batch_state_lock:
                    item["status"] = "pending"
                    item["error"] = ""
                    batch_state["pending"] = sum(1 for x in batch_state["items"] if x.get("status") == "pending")
                    _save_batch_state_locked()
                break
            except Exception as exc:
                if batch_stop_event.is_set():
                    for temp_source in (source_wav, source_meta):
                        try:
                            temp_source.unlink(missing_ok=True)
                        except OSError:
                            pass
                    with batch_state_lock:
                        item["status"] = "pending"
                        item["error"] = ""
                        _save_batch_state_locked()
                    break
                for temp_source in (source_wav, source_meta):
                    try:
                        temp_source.unlink(missing_ok=True)
                    except OSError:
                        pass
                short_error = str(exc)[-4000:]
                with batch_state_lock:
                    item["status"] = "failed"
                    item["error"] = short_error
                    batch_state["completed"] = sum(1 for x in batch_state["items"] if x.get("status") == "done")
                    batch_state["failed"] = sum(1 for x in batch_state["items"] if x.get("status") == "failed")
                    batch_state["pending"] = sum(1 for x in batch_state["items"] if x.get("status") == "pending")
                    finished = batch_state["completed"] + batch_state["failed"]
                    batch_state["progress"] = 100.0 * finished / max(1, batch_state["total"])
                    _save_batch_state_locked()
                append_batch_log(f"[BATCH] ITEM_FAILED {i+1}/{len(batch_state['items'])} {item['filename']} :: {short_error.splitlines()[-1] if short_error else 'Unknown error'}")

        with batch_state_lock:
            pending = sum(1 for x in batch_state["items"] if x.get("status") == "pending")
            batch_state["pending"] = pending
            batch_state["running"] = False
            batch_state["stopping"] = False
            batch_state["current_text"] = ""
            batch_state["current_filename"] = ""
            batch_state["resumable"] = pending > 0
            batch_state["done"] = pending == 0
            if batch_state["done"]:
                batch_state["progress"] = 100.0
            _save_batch_state_locked()
        snap = _batch_snapshot()
        _write_failed_csv(snap)
        append_batch_log("[BATCH] STOPPED" if snap["resumable"] else "[BATCH] COMPLETE")
    except Exception as exc:
        append_batch_log(f"[BATCH] ERROR {exc}")
        set_batch(running=False, stopping=False, resumable=True, error=str(exc))
    finally:
        generation_lock.release()
        prevent_sleep(False)


load_saved_batch_state()


def _run_quiet(command: list[str], timeout: int = 15) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            creationflags=(getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0),
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def _find_docker_desktop_exe() -> Path | None:
    if os.name != "nt":
        return None
    candidates: list[Path] = []
    program_files = os.environ.get("ProgramFiles")
    local_app_data = os.environ.get("LOCALAPPDATA")
    if program_files:
        pf = Path(program_files)
        candidates.extend([
            pf / "Docker" / "Docker" / "Docker Desktop.exe",
            pf / "Docker" / "Docker Desktop.exe",
        ])
    if local_app_data:
        la = Path(local_app_data)
        candidates.extend([
            la / "Programs" / "DockerDesktop" / "Docker Desktop.exe",
            la / "Programs" / "Docker" / "Docker" / "Docker Desktop.exe",
            la / "Docker" / "Docker Desktop.exe",
        ])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _find_docker_cli() -> str | None:
    found = shutil.which("docker")
    if found:
        return found
    desktop = _find_docker_desktop_exe()
    candidates: list[Path] = []
    if desktop:
        candidates.extend([
            desktop.parent / "resources" / "bin" / "docker.exe",
            desktop.parent / "resources" / "docker.exe",
        ])
    program_files = os.environ.get("ProgramFiles")
    local_app_data = os.environ.get("LOCALAPPDATA")
    if program_files:
        pf = Path(program_files)
        candidates.extend([
            pf / "Docker" / "Docker" / "resources" / "bin" / "docker.exe",
            pf / "Docker" / "Docker" / "resources" / "docker.exe",
        ])
    if local_app_data:
        la = Path(local_app_data)
        candidates.extend([
            la / "Programs" / "DockerDesktop" / "resources" / "bin" / "docker.exe",
            la / "Programs" / "Docker" / "Docker" / "resources" / "bin" / "docker.exe",
        ])
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def _read_accent_installer_status() -> dict:
    if not ACCENT_SUPPORT_STATUS_PATH.exists():
        return {}
    try:
        data = json.loads(ACCENT_SUPPORT_STATUS_PATH.read_text(encoding="utf-8-sig"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _accent_support_status() -> dict:
    cfg = load_config()
    rvc_python = Path(str(cfg.get("rvc_python") or ""))
    pyworld_ok = False
    if rvc_python.exists():
        proc = _run_quiet([str(rvc_python), "-c", "import pyworld"], timeout=20)
        pyworld_ok = bool(proc and proc.returncode == 0)

    wsl_ok = False
    if os.name == "nt":
        wsl = Path(os.environ.get("WINDIR", r"C:\Windows")) / "System32" / "wsl.exe"
        if wsl.exists():
            proc = _run_quiet([str(wsl), "--status"], timeout=20)
            wsl_ok = bool(proc and proc.returncode == 0)

    docker_desktop = _find_docker_desktop_exe()
    docker_cli = _find_docker_cli()
    docker_running = False
    image_ready = False
    if docker_cli:
        proc = _run_quiet([docker_cli, "info", "--format", "{{.ServerVersion}}"], timeout=10)
        docker_running = bool(proc and proc.returncode == 0)
        if docker_running:
            inspect = _run_quiet(
                [docker_cli, "image", "inspect", "lingsoft/aalto-kaldi-align:5.1.1-elg"],
                timeout=20,
            )
            image_ready = bool(inspect and inspect.returncode == 0)

    worker_ok = WORLD_ACCENT_WORKER_PATH.exists()
    installed = bool(os.name == "nt" and wsl_ok and docker_desktop and pyworld_ok and worker_ok)
    ready = bool(installed and docker_running)
    installer = _read_accent_installer_status()
    return {
        "ok": True,
        "installed": installed,
        "ready": ready,
        "components": {
            "wsl2": wsl_ok,
            "docker_desktop": bool(docker_desktop),
            "docker_running": docker_running,
            "aalto_image": image_ready,
            "pyworld": pyworld_ok,
            "world_worker": worker_ok,
        },
        "installer_state": str(installer.get("state") or ""),
        "installer_message": str(installer.get("message") or ""),
        "reboot_required": bool(installer.get("reboot_required", False)),
    }


@app.get("/")
def index():
    return send_from_directory(APP_DIR, "index.html")


@app.get("/api/state")
def state():
    cfg = load_config()
    return jsonify(
        {
            "voices": scan_voices(cfg),
            "tts": cfg["tts"],
            "config": {
                "chatterbox_repo": cfg["chatterbox_repo"],
                "chatterbox_weights": cfg["chatterbox_weights"],
                "reference_audio": cfg["reference_audio"],
                "rvc_python": cfg["rvc_python"],
                "rvc_repo": cfg["rvc_repo"],
                "exports_root": cfg["exports_root"],
            },
            "trainer": cfg["trainer"],
        }
    )


@app.get("/api/accent-support/status")
def accent_support_status():
    return jsonify(_accent_support_status())


@app.post("/api/accent-support/install")
def accent_support_install():
    if os.name != "nt":
        return jsonify({"ok": False, "error": "Accent support installer is available on Windows only."}), 400
    if not ACCENT_SUPPORT_BAT_PATH.exists() or not ACCENT_SUPPORT_PS1_PATH.exists():
        return jsonify({"ok": False, "error": "Accent support installer files are missing."}), 500
    try:
        # ShellExecute with the runas verb opens the installer in its own visible
        # elevated console. This keeps UAC/restart/first-run Docker prompts out of
        # the web server process and lets a novice follow the installation safely.
        import ctypes
        try:
            ACCENT_SUPPORT_STATUS_PATH.unlink(missing_ok=True)
        except OSError:
            pass
        cmd_exe = Path(os.environ.get("WINDIR", r"C:\Windows")) / "System32" / "cmd.exe"
        params = f'/c ""{ACCENT_SUPPORT_BAT_PATH}""'
        result = ctypes.windll.shell32.ShellExecuteW(
            None,
            "runas",
            str(cmd_exe),
            params,
            str(APP_DIR),
            1,
        )
        if int(result) <= 32:
            raise OSError(f"Windows could not start the installer (ShellExecute code {int(result)}).")
        return jsonify({"ok": True, "started": True})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.post("/api/config")
def update_config():
    data = request.get_json(force=True) or {}
    cfg = load_config()
    for key in [
        "chatterbox_repo",
        "chatterbox_weights",
        "reference_audio",
        "rvc_python",
        "rvc_repo",
        "exports_root",
    ]:
        if key in data:
            cfg[key] = str(data[key]).strip()
    save_config(cfg)
    return jsonify({"ok": True})


@app.post("/api/generate")
def generate():
    with training_state_lock:
        if training_state["running"]:
            return jsonify({"ok": False, "error": "Model training is running."}), 409
    with batch_state_lock:
        if batch_state["running"]:
            return jsonify({"ok": False, "error": "Batch generation is running."}), 409

    if not generation_lock.acquire(blocking=False):
        return jsonify({"ok": False, "error": "Another generation job is already running."}), 409

    try:
        data = request.get_json(force=True) or {}
        text = str(data.get("text", "")).strip()
        voice = str(data.get("voice", "")).strip()
        cfg = load_config()
        voices = scan_voices(cfg)
        if not text:
            return jsonify({"ok": False, "error": "Enter Finnish text."}), 400
        if voice not in voices:
            return jsonify({"ok": False, "error": f"RVC model not found: {voice}"}), 400

        pitch = data.get("pitch", voices[voice]["pitch"])
        index_rate = data.get("index_rate", voices[voice]["index_rate"])
        protect = data.get("protect", voices[voice]["protect"])
        expression = data.get("expression", cfg["tts"]["exaggeration"])

        fix_seed = bool(data.get("fix_seed", False))
        if fix_seed:
            raw_seed = data.get("seed", 0)
            try:
                seed = int(str(raw_seed).strip())
            except (TypeError, ValueError):
                return jsonify({"ok": False, "error": "Seed must be an integer."}), 400
            if seed < 0 or seed > 2147483647:
                return jsonify({"ok": False, "error": "Seed must be between 0 and 2147483647."}), 400
        else:
            seed = secrets.randbelow(2147483648)

        # A new Voice Generation preview invalidates the old alignment.
        for preview_temp in (
            PREVIEW_TTS_WAV_PATH,
            PREVIEW_TTS_TEXT_PATH,
            PREVIEW_ALIGNMENT_PATH,
            PREVIEW_SOURCE_META_PATH,
            PITCH_EDIT_TTS_WAV_PATH,
            PITCH_EDIT_WAV_PATH,
        ):
            try:
                preview_temp.unlink(missing_ok=True)
            except OSError:
                pass

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suggested_name = f"{voice}_{stamp}_seed{seed}.wav"
        generate_voice_file(
            text=text, voice=voice, pitch=pitch, index_rate=index_rate, protect=protect,
            expression=expression, output=PREVIEW_WAV_PATH, cfg=cfg, batch=False, persist_preset=True,
            seed=seed, tts_copy=PREVIEW_TTS_WAV_PATH,
        )
        PREVIEW_TTS_TEXT_PATH.write_text(text, encoding="utf-8")
        # Keep the exact generation context together with the clean pre-RVC
        # Finnish TTS preview. When the user later saves the Voice Generation
        # result, this metadata can be saved with an optional source WAV so Import WAV
        # can reconstruct the high-quality WORLD -> RVC editing path.
        PREVIEW_SOURCE_META_PATH.write_text(
            json.dumps({
                "format": "zundanen_preview_source_v1",
                "mode": "generated_tts",
                "text": text,
                "voice": voice,
                "pitch": int(round(float(pitch))),
                "index_rate": max(0.0, min(1.0, float(index_rate))),
                "protect": max(0.0, min(0.5, float(protect))),
                "expression": max(0.2, min(1.2, float(expression))),
                "seed": seed,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return jsonify({
            "ok": True,
            "filename": suggested_name,
            "audio_url": "/api/preview-audio",
            "seed": seed,
        })
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc), "trace": traceback.format_exc(limit=5)}), 500
    finally:
        generation_lock.release()


@app.post("/api/alignment/run")
def alignment_run():
    if not PREVIEW_TTS_WAV_PATH.exists():
        return jsonify({
            "ok": False,
            "code": "preview_missing",
            "error": "Generate or import a WAV before running accent analysis.",
        }), 400

    data = request.get_json(silent=True) or {}
    supplied_text = str(data.get("text", "")).strip()
    if supplied_text:
        PREVIEW_TTS_TEXT_PATH.write_text(supplied_text, encoding="utf-8")
    if not PREVIEW_TTS_TEXT_PATH.exists() or not PREVIEW_TTS_TEXT_PATH.read_text(encoding="utf-8").strip():
        return jsonify({
            "ok": False,
            "code": "transcript_missing",
            "error": "Enter the Finnish text corresponding to the imported WAV.",
        }), 400

    # Do not let a new Voice/Batch generation replace the source audio
    # while Docker is aligning it.
    if not generation_lock.acquire(blocking=False):
        return jsonify({
            "ok": False,
            "code": "generation_busy",
            "error": "Another generation job is currently running.",
        }), 409

    try:
        transcript = PREVIEW_TTS_TEXT_PATH.read_text(encoding="utf-8").strip()
        result = run_alignment(PREVIEW_TTS_WAV_PATH, transcript)

        # Accent selection only needs Aalto's word/syllable timing.  Do not run
        # the extra pYIN/F0 analysis here; WORLD estimates the F0 track when
        # an accent preview is actually requested.
        f0_track = None
        f0_warning = ""

        payload = {
            "transcript": transcript,
            "source": result["source"],
            "image": result["image"],
            "words": result["words"],
            "phones": result["phones"],
            "syllables": result["syllables"],
            "syllabified_transcript": result.get("syllabified_transcript", transcript),
            "f0_track": f0_track,
            "f0_warning": f0_warning,
        }
        PREVIEW_ALIGNMENT_PATH.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return jsonify({
            "ok": True,
            **payload,
            "image_downloaded": bool(result.get("image_downloaded")),
        })
    except AlignmentError as exc:
        return jsonify({
            "ok": False,
            "code": exc.code,
            "error": str(exc),
        }), 500
    except Exception as exc:
        return jsonify({
            "ok": False,
            "code": "alignment_failed",
            "error": str(exc),
            "trace": traceback.format_exc(limit=5),
        }), 500
    finally:
        generation_lock.release()


@app.post("/api/alignment/accent-preview")
def alignment_accent_preview():
    if not PREVIEW_TTS_WAV_PATH.exists() or not PREVIEW_ALIGNMENT_PATH.exists():
        return jsonify({
            "ok": False,
            "error": "Run Finnish TTS generation and alignment before editing accents.",
        }), 400

    if not generation_lock.acquire(blocking=False):
        return jsonify({"ok": False, "error": "Another generation job is currently running."}), 409

    try:
        data = request.get_json(force=True) or {}
        alignment = json.loads(PREVIEW_ALIGNMENT_PATH.read_text(encoding="utf-8"))
        syllables = alignment.get("syllables") or []
        words = alignment.get("words") or []
        requested = data.get("accents") or []
        if not isinstance(requested, list):
            raise ValueError("Accent data must be a list.")

        # Index syllables by aligned word so the client only needs to select the
        # word and syllable number. Timing always comes from the saved alignment.
        by_word: dict[int, list[dict]] = {}
        for syllable in syllables:
            try:
                word_index = int(syllable.get("word_index"))
                syllable_index = int(syllable.get("syllable_index"))
            except (TypeError, ValueError):
                continue
            by_word.setdefault(word_index, []).append(syllable)
        for group in by_word.values():
            group.sort(key=lambda x: int(x.get("syllable_index", 0)))

        accents: list[dict] = []
        normalized: list[dict] = []
        used_words: set[int] = set()
        for item in requested:
            if not isinstance(item, dict):
                continue
            try:
                word_index = int(item.get("word_index"))
                syllable_index = int(item.get("syllable_index"))
                strength = float(item.get("strength", 0.0))
            except (TypeError, ValueError):
                continue
            if word_index in used_words:
                continue
            strength = max(0.0, min(100.0, strength))
            if strength <= 0.0:
                continue
            candidates = by_word.get(word_index) or []
            selected = next((x for x in candidates if int(x.get("syllable_index", -1)) == syllable_index), None)
            if selected is None:
                continue
            start = float(selected.get("start", 0.0))
            end = float(selected.get("end", start))
            if end <= start:
                continue
            # Send the whole aligned word span as well as the selected syllable.
            # The WORLD worker uses the whole word span to create contrast: the
            # non-selected part is lowered while the selected syllable receives
            # the rise-fall prominence gesture.
            word_start = min(float(x.get("start", start)) for x in candidates) if candidates else start
            word_end = max(float(x.get("end", end)) for x in candidates) if candidates else end
            accent = {
                "word_index": word_index,
                "syllable_index": syllable_index,
                "start": start,
                "end": end,
                "word_start": word_start,
                "word_end": word_end,
                "strength": strength,
                "syllable": str(selected.get("syllable", "")),
                "word": str(selected.get("display_word") or selected.get("word") or (words[word_index].get("display_word") if 0 <= word_index < len(words) else "")),
            }
            accents.append(accent)
            normalized.append({"word_index": word_index, "syllable_index": syllable_index, "strength": round(strength, 2)})
            used_words.add(word_index)

        cfg = load_config()
        voice = str(data.get("voice", "")).strip()
        pitch = data.get("pitch", 0)
        index_rate = data.get("index_rate", 0.55)
        protect = data.get("protect", 0.15)

        for path in (PITCH_EDIT_TTS_WAV_PATH, PITCH_EDIT_WAV_PATH):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

        source_mode = "generated_tts"
        if PREVIEW_SOURCE_META_PATH.exists():
            try:
                source_mode = str(json.loads(PREVIEW_SOURCE_META_PATH.read_text(encoding="utf-8")).get("mode") or source_mode)
            except Exception:
                pass

        world_result = {}
        if source_mode == "imported_final":
            # Imported WAVs are assumed to be already-finished voice audio (for
            # example a Batch Generation result). Apply WORLD directly and do
            # not run RVC a second time.
            if accents:
                world_result = apply_world_accent_edits(
                    PREVIEW_TTS_WAV_PATH, PITCH_EDIT_WAV_PATH, accents, cfg
                )
            else:
                shutil.copy2(PREVIEW_TTS_WAV_PATH, PITCH_EDIT_WAV_PATH)
        else:
            rvc_input = PREVIEW_TTS_WAV_PATH
            if accents:
                world_result = apply_world_accent_edits(
                    PREVIEW_TTS_WAV_PATH, PITCH_EDIT_TTS_WAV_PATH, accents, cfg
                )
                rvc_input = PITCH_EDIT_TTS_WAV_PATH

            run_rvc_on_audio(
                input_wav=rvc_input,
                output_wav=PITCH_EDIT_WAV_PATH,
                voice=voice,
                pitch=pitch,
                index_rate=index_rate,
                protect=protect,
                cfg=cfg,
                batch=False,
            )
        # Keep PREVIEW_WAV_PATH untouched: it is the original preview shown
        # in the main player. Accent-edited audio has its own endpoint/player so
        # the user can A/B compare original vs edited without regenerating.
        return jsonify({
            "ok": True,
            "audio_url": "/api/alignment/accent-audio",
            "original_audio_url": "/api/preview-audio",
            "accents": normalized,
            "edited_words": len(accents),
            "mode": (
                "world_on_imported_wav" if source_mode == "imported_final" and accents
                else "imported_wav_unchanged" if source_mode == "imported_final"
                else "world_accent_then_rvc" if accents
                else "original_tts_then_rvc"
            ),
            "world_accent": world_result,
        })
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc), "trace": traceback.format_exc(limit=5)}), 500
    finally:
        generation_lock.release()


@app.get("/api/alignment/accent-audio")
def alignment_accent_audio():
    if not PITCH_EDIT_WAV_PATH.exists():
        return jsonify({"ok": False, "error": "No accent-edited preview is available."}), 404
    return send_from_directory(PITCH_EDIT_WAV_PATH.parent, PITCH_EDIT_WAV_PATH.name)

@app.post("/api/batch/start")
def batch_start():
    with training_state_lock:
        if training_state["running"]:
            return jsonify({"ok": False, "error": "Model training is running."}), 409
    with batch_state_lock:
        if batch_state["running"]:
            return jsonify({"ok": False, "error": "A batch queue is already running."}), 409
    if generation_lock.locked():
        return jsonify({"ok": False, "error": "Another generation job is already running."}), 409

    data = request.get_json(force=True) or {}
    create_subtitle_txt = bool(data.get("create_subtitle_txt", True))
    save_source_audio = bool(data.get("save_source_audio", False))
    fix_seed = bool(data.get("fix_seed", False))
    try:
        batch_seed = _as_seed(data.get("seed", 0))
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    cfg = load_config()
    try:
        validate_config(cfg)
        items, mode = build_batch_items(data, cfg)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    queue_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    output_dir = OUTPUT_DIR / "batches" / queue_id
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_stop_event.clear()
    with batch_state_lock:
        batch_state.clear()
        batch_state.update({
            "running": True,
            "done": False,
            "stopping": False,
            "resumable": False,
            "queue_id": queue_id,
            "mode": mode,
            "create_subtitle_txt": create_subtitle_txt,
            "save_source_audio": save_source_audio,
            "fix_seed": fix_seed,
            "seed": batch_seed,
            "total": len(items),
            "current_index": 0,
            "completed": 0,
            "failed": 0,
            "pending": len(items),
            "progress": 0.0,
            "current_text": "",
            "current_filename": "",
            "output_dir": str(output_dir),
            "items": items,
            "log": [],
            "error": "",
        })
        _save_batch_state_locked()
    threading.Thread(target=batch_runner, daemon=True).start()
    return jsonify({"ok": True, "queue_id": queue_id, "total": len(items)})


@app.get("/api/batch/status")
def batch_status():
    snap = _batch_snapshot()
    # Keep status payload manageable for long overnight runs.
    snap["failed_items"] = [
        {"id": x.get("id"), "filename": x.get("filename"), "text": x.get("text"), "error": x.get("error")}
        for x in snap.get("items", []) if x.get("status") == "failed"
    ][-100:]
    snap.pop("items", None)
    return jsonify(snap)


@app.post("/api/batch/stop")
def batch_stop():
    with batch_state_lock:
        if not batch_state.get("running"):
            return jsonify({"ok": False, "error": "No batch queue is running."}), 400
        batch_state["stopping"] = True
        _save_batch_state_locked()
    batch_stop_event.set()
    with batch_process_lock:
        proc = batch_current_process
    if proc and proc.poll() is None:
        try:
            if sys.platform.startswith("win"):
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, text=True)
            else:
                proc.terminate()
        except Exception:
            pass
    return jsonify({"ok": True})


@app.post("/api/batch/resume")
def batch_resume():
    with training_state_lock:
        if training_state["running"]:
            return jsonify({"ok": False, "error": "Model training is running."}), 409
    with batch_state_lock:
        if batch_state.get("running"):
            return jsonify({"ok": False, "error": "A batch queue is already running."}), 409
        pending = sum(1 for x in batch_state.get("items", []) if x.get("status") == "pending")
        if pending == 0:
            return jsonify({"ok": False, "error": "There is no pending batch queue to resume."}), 400
        batch_state["running"] = True
        batch_state["stopping"] = False
        batch_state["resumable"] = False
        batch_state["done"] = False
        batch_state["error"] = ""
        batch_state["pending"] = pending
        _save_batch_state_locked()
    batch_stop_event.clear()
    threading.Thread(target=batch_runner, daemon=True).start()
    return jsonify({"ok": True, "pending": pending})


@app.post("/api/batch/clear")
def batch_clear():
    with batch_state_lock:
        if batch_state.get("running"):
            return jsonify({"ok": False, "error": "Stop the running batch before clearing it."}), 409
        batch_state.clear()
        batch_state.update({
            "running": False, "done": False, "stopping": False, "resumable": False,
            "queue_id": "", "mode": "lines", "create_subtitle_txt": True, "save_source_audio": False, "fix_seed": False, "seed": 0, "total": 0, "current_index": 0,
            "completed": 0, "failed": 0, "pending": 0, "progress": 0.0,
            "current_text": "", "current_filename": "", "output_dir": "",
            "items": [], "log": [], "error": "",
        })
        _save_batch_state_locked()
    return jsonify({"ok": True})


@app.post("/api/batch/open-output")
def batch_open_output():
    snap = _batch_snapshot()
    folder = Path(snap.get("output_dir") or (OUTPUT_DIR / "batches"))
    folder.mkdir(parents=True, exist_ok=True)
    if sys.platform.startswith("win"):
        os.startfile(folder)
        return jsonify({"ok": True})
    return jsonify({"ok": False, "error": str(folder)}), 400


def validate_trainer_config(cfg: dict) -> None:
    required = {
        "RVC Python": Path(cfg["rvc_python"]),
        "RVC repo": Path(cfg["rvc_repo"]),
        "RVC exports": Path(cfg["exports_root"]),
        "trainer_worker.py": APP_DIR / "trainer_worker.py",
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required training files/folders were not found:\n" + "\n".join(missing)
        )


def dataset_audio_count(path: Path) -> int:
    exts = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}
    if not path.exists() or not path.is_dir():
        return 0
    return sum(1 for p in path.iterdir() if p.is_file() and p.suffix.lower() in exts)


def training_snapshot() -> dict:
    with training_state_lock:
        snap = dict(training_state)
        snap["log"] = list(training_state["log"])
        snap["result"] = dict(training_state["result"])
        return snap


def set_training(**updates) -> None:
    with training_state_lock:
        training_state.update(updates)


def append_training_log(line: str) -> None:
    line = line.rstrip("\r\n")
    if not line:
        return

    with training_state_lock:
        training_state["log"].append(line)
        if len(training_state["log"]) > 450:
            training_state["log"] = training_state["log"][-450:]

        phase_match = re.search(r"\[PHASE\]\s+(PREPROCESS|F0|HUBERT|TRAIN|INDEX)", line)
        if phase_match:
            phase = phase_match.group(1)
            info = {
                "PREPROCESS": ("preprocess", "Audio preprocessing", 2.0),
                "F0": ("f0", "RMVPE / F0 extraction", 14.0),
                "HUBERT": ("hubert", "HuBERT feature extraction", 24.0),
                "TRAIN": ("train", "RVC training", 34.0),
                "INDEX": ("index", "Building FAISS index", 94.0),
            }[phase]
            training_state["phase"] = info[0]
            training_state["phase_label"] = info[1]
            training_state["progress"] = max(training_state["progress"], info[2])

        progress_match = re.search(r"(?:Progress|進捗):\s*(\d+)\s*/\s*(\d+)", line, re.IGNORECASE)
        if progress_match:
            current = int(progress_match.group(1))
            total = max(1, int(progress_match.group(2)))
            ratio = min(1.0, current / total)
            phase = training_state["phase"]
            bands = {
                "preprocess": (2.0, 14.0),
                "f0": (14.0, 24.0),
                "hubert": (24.0, 34.0),
                "index": (94.0, 99.0),
            }
            if phase in bands:
                lo, hi = bands[phase]
                training_state["progress"] = lo + (hi - lo) * ratio

        epoch_match = re.search(r"(?:Training Epoch|Training epoch|学習エポック):\s*(\d+)", line, re.IGNORECASE)
        if not epoch_match:
            epoch_match = re.search(r"====>\s*(?:Epoch|エポック):\s*(\d+)", line, re.IGNORECASE)
        if epoch_match:
            epoch = int(epoch_match.group(1))
            training_state["epoch"] = max(training_state["epoch"], epoch)
            total_epochs = max(1, int(training_state["epochs"]))
            training_state["progress"] = 34.0 + 60.0 * min(1.0, epoch / total_epochs)

        if "TRAINING COMPLETE" in line:
            training_state["phase"] = "complete"
            training_state["phase_label"] = "Complete"
            training_state["progress"] = 100.0

        if line.startswith("PTH   :"):
            training_state["result"]["pth"] = line.split(":", 1)[1].strip()
        elif line.startswith("INDEX :"):
            training_state["result"]["index"] = line.split(":", 1)[1].strip()


def training_runner(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    global training_process
    try:
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform.startswith("win") else 0
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creationflags,
        )
        with training_process_lock:
            training_process = proc
        set_training(pid=proc.pid)

        assert proc.stdout is not None
        for line in proc.stdout:
            append_training_log(line)

        code = proc.wait()
        snap = training_snapshot()
        if code == 0 and snap["phase"] == "complete":
            set_training(running=False, done=True, stopping=False, progress=100.0)
        elif snap["stopping"]:
            append_training_log("[STOPPED] Training was stopped by the user.")
            set_training(
                running=False,
                done=False,
                stopping=False,
                phase="stopped",
                phase_label="Stopped",
                error="",
            )
        else:
            tail = "\n".join(snap["log"][-30:])
            set_training(
                running=False,
                done=False,
                stopping=False,
                phase="error",
                phase_label="Training failed",
                error=f"trainer_worker exited with code {code}\n\n{tail}",
            )
    except Exception as exc:
        append_training_log(f"[ERROR] {exc}")
        set_training(
            running=False,
            done=False,
            stopping=False,
            phase="error",
            phase_label="Training failed",
            error=f"{exc}\n\n{traceback.format_exc(limit=5)}",
        )
    finally:
        with training_process_lock:
            training_process = None


@app.post("/api/dataset-info")
def dataset_info():
    data = request.get_json(force=True) or {}
    path = Path(str(data.get("path", "")).strip()).expanduser()
    if not path.exists() or not path.is_dir():
        return jsonify({"ok": False, "error_code": "folder_not_found", "error": "Folder not found.", "count": 0}), 400
    return jsonify({"ok": True, "path": str(path.resolve()), "count": dataset_audio_count(path)})


@app.post("/api/pick-folder")
def pick_folder():
    if not sys.platform.startswith("win"):
        return jsonify({"ok": False, "error": "Folder picker is currently Windows-only."}), 400

    data = request.get_json(silent=True) or {}
    initial = str(data.get("initial", "")).strip().replace("'", "''")
    script = (
        "$ErrorActionPreference='Stop';"
        "Add-Type -AssemblyName System.Windows.Forms;"
        "$d=New-Object System.Windows.Forms.FolderBrowserDialog;"
        "$d.Description='Select the folder containing training audio';"
        "$d.ShowNewFolderButton=$true;"
    )
    if initial:
        # PowerShell single-quoted strings escape an apostrophe by doubling it.
        safe_initial = initial.replace("'", "''")
        script += f"$d.SelectedPath='{safe_initial}';"
    script += (
        "if($d.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK){"
        "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8;"
        "Write-Output $d.SelectedPath}"
    )

    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-STA", "-Command", script],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "Folder picker failed")
        selected = result.stdout.strip()
        return jsonify({"ok": True, "path": selected})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.post("/api/trainer/start")
def trainer_start():
    with batch_state_lock:
        if batch_state.get("running"):
            return jsonify({"ok": False, "error": "Batch generation is running."}), 409
    if generation_lock.locked():
        return jsonify({"ok": False, "error": "Audio generation is running."}), 409

    with training_state_lock:
        if training_state["running"]:
            return jsonify({"ok": False, "error": "Model training is already running."}), 409

    data = request.get_json(force=True) or {}
    character = str(data.get("character", "")).strip()
    dataset = Path(str(data.get("dataset", "")).strip()).expanduser()

    if not character:
        return jsonify({"ok": False, "error": "Enter a Character name."}), 400
    if character in {".", ".."} or any(c in character for c in '<>:"/\\|?*'):
        return jsonify({"ok": False, "error": "Character name contains characters that are not allowed on Windows."}), 400
    if not dataset.exists() or not dataset.is_dir():
        return jsonify({"ok": False, "error": "Dataset folder was not found."}), 400

    count = dataset_audio_count(dataset)
    if count == 0:
        return jsonify({"ok": False, "error": "No supported audio files were found in the Dataset folder."}), 400

    try:
        epochs = max(1, int(data.get("epochs", 100)))
        batch_size = max(1, int(data.get("batch_size", 6)))
        save_every = max(1, int(data.get("save_every", 10)))
        workers = max(1, int(data.get("workers", min(8, os.cpu_count() or 1))))
        gpu = str(data.get("gpu", "0")).strip() or "0"
        fresh = bool(data.get("fresh", True))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "One or more training settings contain an invalid number."}), 400

    cfg = load_config()
    try:
        validate_trainer_config(cfg)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    cfg["trainer"] = {
        "epochs": epochs,
        "batch_size": batch_size,
        "save_every": save_every,
        "workers": workers,
        "gpu": gpu,
        "fresh": fresh,
    }
    save_config(cfg)

    with training_state_lock:
        training_state.update({
            "running": True,
            "done": False,
            "stopping": False,
            "character": character,
            "dataset": str(dataset.resolve()),
            "phase": "starting",
            "phase_label": "Starting trainer",
            "epoch": 0,
            "epochs": epochs,
            "progress": 0.5,
            "audio_files": count,
            "log": [],
            "error": "",
            "result": {},
            "pid": None,
        })

    cmd = [
        cfg["rvc_python"],
        "-u",
        str(APP_DIR / "trainer_worker.py"),
        character,
        "--dataset", str(dataset.resolve()),
        "--rvc-repo", cfg["rvc_repo"],
        "--exports-root", cfg["exports_root"],
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--save-every", str(save_every),
        "--workers", str(workers),
        "--gpu", gpu,
    ]
    if fresh:
        cmd.append("--fresh")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # trainer_worker and all RVC subprocesses start from an English locale.
    env["LANG"] = "en_US.UTF-8"
    env["LC_ALL"] = "en_US.UTF-8"
    env["LANGUAGE"] = "en_US"
    env["RVC_LANGUAGE"] = "en_US"
    thread = threading.Thread(
        target=training_runner,
        args=(cmd, APP_DIR, env),
        daemon=True,
    )
    thread.start()
    return jsonify({"ok": True, "audio_files": count})


@app.get("/api/trainer/status")
def trainer_status():
    return jsonify(training_snapshot())


@app.post("/api/trainer/stop")
def trainer_stop():
    with training_state_lock:
        if not training_state["running"]:
            return jsonify({"ok": False, "error": "Model training is not running."}), 400
        training_state["stopping"] = True
        training_state["phase_label"] = "Stopping…"

    with training_process_lock:
        proc = training_process

    if proc and proc.poll() is None:
        try:
            if sys.platform.startswith("win"):
                subprocess.run(
                    ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                    capture_output=True,
                    text=True,
                )
            else:
                proc.terminate()
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500
    return jsonify({"ok": True})


@app.post("/api/open-exports")
def open_exports():
    cfg = load_config()
    folder = Path(cfg["exports_root"])
    folder.mkdir(parents=True, exist_ok=True)
    if sys.platform.startswith("win"):
        os.startfile(folder)
        return jsonify({"ok": True})
    return jsonify({"ok": False, "error": str(folder)}), 400


@app.get("/favicon.ico")
def favicon():
    return ("", 204)


def _choose_import_wav_path() -> Path | None:
    if not sys.platform.startswith("win"):
        raise RuntimeError("Import WAV is currently supported on Windows only.")

    result_file = TEMP_DIR / f"open_wav_dialog_{uuid.uuid4().hex}.txt"
    env = os.environ.copy()
    env["ZUNDANEN_OPEN_RESULT"] = str(result_file)
    ps = r'''
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.OpenFileDialog
$dialog.Title = 'Import WAV'
$dialog.Filter = 'WAV audio (*.wav)|*.wav'
$dialog.Multiselect = $false
if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
    [System.IO.File]::WriteAllText($env:ZUNDANEN_OPEN_RESULT, $dialog.FileName, (New-Object System.Text.UTF8Encoding($false)))
}
'''
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        proc = subprocess.run(
            ["powershell.exe", "-NoProfile", "-STA", "-Command", ps],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=flags,
        )
        if proc.returncode != 0:
            raise RuntimeError("Import WAV dialog failed: " + (proc.stderr or "Unknown error").strip())
        if not result_file.exists():
            return None
        chosen = result_file.read_text(encoding="utf-8").strip()
        return Path(chosen) if chosen else None
    finally:
        try:
            result_file.unlink(missing_ok=True)
        except OSError:
            pass


@app.post("/api/import-wav")
def import_wav():
    with training_state_lock:
        if training_state["running"]:
            return jsonify({"ok": False, "error": "Model training is running."}), 409
    with batch_state_lock:
        if batch_state["running"]:
            return jsonify({"ok": False, "error": "Batch generation is running."}), 409
    if not generation_lock.acquire(blocking=False):
        return jsonify({"ok": False, "error": "Another generation job is already running."}), 409

    try:
        chosen = _choose_import_wav_path()
        if chosen is None:
            return jsonify({"ok": True, "cancelled": True})
        if not chosen.exists() or not chosen.is_file() or chosen.suffix.lower() != ".wav":
            return jsonify({"ok": False, "error": "Select a valid WAV file."}), 400

        for preview_temp in (
            PREVIEW_WAV_PATH,
            PREVIEW_TTS_WAV_PATH,
            PREVIEW_TTS_TEXT_PATH,
            PREVIEW_ALIGNMENT_PATH,
            PREVIEW_SOURCE_META_PATH,
            PITCH_EDIT_TTS_WAV_PATH,
            PITCH_EDIT_WAV_PATH,
        ):
            try:
                preview_temp.unlink(missing_ok=True)
            except OSError:
                pass

        # Main player: always keep the imported final WAV for A/B comparison.
        shutil.copy2(chosen, PREVIEW_WAV_PATH)

        # If an optional saved source WAV exists (Voice Generation beside the
        # final WAV, Batch Generation in source_audio/), edit that clean source
        # with WORLD and run RVC once. Legacy hidden sidecars are also supported.
        source_wav, source_meta = _find_batch_source(chosen)
        if source_wav is not None:
            shutil.copy2(source_wav, PREVIEW_TTS_WAV_PATH)
            source_mode = "imported_batch_source"
        else:
            shutil.copy2(chosen, PREVIEW_TTS_WAV_PATH)
            source_mode = "imported_final"

        transcript = ""
        txt_path = chosen.with_suffix(".txt")
        if txt_path.exists() and txt_path.is_file():
            try:
                transcript = txt_path.read_text(encoding="utf-8-sig").strip()
            except UnicodeDecodeError:
                transcript = txt_path.read_text(encoding="utf-8", errors="replace").strip()
        if not transcript:
            transcript = str(source_meta.get("text") or "").strip()
        if transcript:
            PREVIEW_TTS_TEXT_PATH.write_text(transcript, encoding="utf-8")

        rvc_settings = {}
        if source_wav is not None:
            for key in ("voice", "pitch", "index_rate", "protect", "expression", "seed"):
                if key in source_meta and source_meta.get(key) is not None:
                    rvc_settings[key] = source_meta.get(key)

        PREVIEW_SOURCE_META_PATH.write_text(json.dumps({
            "mode": source_mode,
            "source": str(chosen),
            "tts_source": str(source_wav) if source_wav is not None else "",
            "rvc_settings": rvc_settings,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

        return jsonify({
            "ok": True,
            "cancelled": False,
            "filename": chosen.name,
            "audio_url": "/api/preview-audio",
            "text": transcript,
            "text_found": bool(transcript),
            "source_found": source_wav is not None,
            "source_mode": source_mode,
            "rvc_settings": rvc_settings,
        })
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc), "trace": traceback.format_exc(limit=5)}), 500
    finally:
        generation_lock.release()


@app.get("/api/preview-audio")
def preview_audio():
    if not PREVIEW_WAV_PATH.exists():
        return jsonify({"ok": False, "error": "No generated preview is available."}), 404
    return send_from_directory(
        PREVIEW_WAV_PATH.parent,
        PREVIEW_WAV_PATH.name,
        mimetype="audio/wav",
        as_attachment=False,
        conditional=True,
    )


def _safe_wav_name(value: str) -> str:
    name = Path(str(value or "Zundanen.wav")).name
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")
    if not name:
        name = "Zundanen.wav"
    if not name.lower().endswith(".wav"):
        name += ".wav"
    return name


def _choose_save_wav_path(default_name: str) -> Path | None:
    if not sys.platform.startswith("win"):
        raise RuntimeError("Save As is currently supported on Windows only.")

    result_file = TEMP_DIR / f"save_dialog_{uuid.uuid4().hex}.txt"
    env = os.environ.copy()
    env["ZUNDANEN_SAVE_NAME"] = _safe_wav_name(default_name)
    env["ZUNDANEN_SAVE_RESULT"] = str(result_file)
    ps = r'''
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.SaveFileDialog
$dialog.Title = 'Save generated WAV'
$dialog.Filter = 'WAV audio (*.wav)|*.wav|All files (*.*)|*.*'
$dialog.DefaultExt = 'wav'
$dialog.AddExtension = $true
$dialog.FileName = $env:ZUNDANEN_SAVE_NAME
if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
    [System.IO.File]::WriteAllText($env:ZUNDANEN_SAVE_RESULT, $dialog.FileName, (New-Object System.Text.UTF8Encoding($false)))
}
'''
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        proc = subprocess.run(
            ["powershell.exe", "-NoProfile", "-STA", "-Command", ps],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=flags,
        )
        if proc.returncode != 0:
            raise RuntimeError("Save As dialog failed: " + (proc.stderr or "Unknown error").strip())
        if not result_file.exists():
            return None
        chosen = result_file.read_text(encoding="utf-8").strip()
        return Path(chosen) if chosen else None
    finally:
        try:
            result_file.unlink(missing_ok=True)
        except OSError:
            pass


@app.post("/api/save-accent-preview")
def save_accent_preview():
    if not PITCH_EDIT_WAV_PATH.exists():
        return jsonify({"ok": False, "error": "Create an Accent preview first."}), 404
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()
    create_subtitle_txt = bool(data.get("create_subtitle_txt", True))
    save_source_audio = bool(data.get("save_source_audio", False))
    try:
        target = _choose_save_wav_path(str(data.get("filename", "Zundanen_accent.wav")))
        if target is None:
            return jsonify({"ok": True, "cancelled": True})
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PITCH_EDIT_WAV_PATH, target)

        txt_target = None
        if create_subtitle_txt:
            txt_target = target.with_suffix(".txt")
            txt_target.write_text(text, encoding="utf-8")

        # Keep the clean Finnish TTS beside an Accent-edited Voice Generation
        # result when the user explicitly opts in. This enables a later import
        # to redo WORLD Accent editing before a single RVC pass.
        source_saved = False
        source_path = ""
        source_meta_path = ""
        preview_meta: dict = {}
        if PREVIEW_SOURCE_META_PATH.exists():
            try:
                loaded = json.loads(PREVIEW_SOURCE_META_PATH.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    preview_meta = loaded
            except Exception:
                preview_meta = {}

        source_mode = str(preview_meta.get("mode") or "")
        if save_source_audio and PREVIEW_TTS_WAV_PATH.exists() and source_mode != "imported_final":
            saved_source_wav, saved_source_meta = _voice_source_paths(target)
            saved_source_wav.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(PREVIEW_TTS_WAV_PATH, saved_source_wav)

            rvc_settings = preview_meta.get("rvc_settings")
            if not isinstance(rvc_settings, dict):
                rvc_settings = {}
            meta = {
                "format": "zundanen_voice_source_v2",
                "final_wav": target.name,
                "text": text or str(preview_meta.get("text") or ""),
            }
            # The Accent preview may have been rendered with settings changed
            # after the original WAV was generated/imported. Prefer the current
            # UI settings sent with this save request, then fall back to source metadata.
            for key in ("voice", "pitch", "index_rate", "protect", "expression", "seed"):
                value = data.get(key)
                if value is None:
                    value = preview_meta.get(key, rvc_settings.get(key))
                if value is not None:
                    meta[key] = value

            saved_source_meta.write_text(
                json.dumps(meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            source_saved = True
            source_path = str(saved_source_wav)
            source_meta_path = str(saved_source_meta)

        return jsonify({
            "ok": True,
            "cancelled": False,
            "path": str(target),
            "txt_path": str(txt_target) if txt_target else "",
            "source_saved": source_saved,
            "source_path": source_path,
            "source_meta_path": source_meta_path,
        })
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.post("/api/save-preview")
def save_preview():
    if not PREVIEW_WAV_PATH.exists():
        return jsonify({"ok": False, "error": "Generate a voice preview first."}), 404
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()
    create_subtitle_txt = bool(data.get("create_subtitle_txt", True))
    save_source_audio = bool(data.get("save_source_audio", False))
    try:
        target = _choose_save_wav_path(str(data.get("filename", "Zundanen.wav")))
        if target is None:
            return jsonify({"ok": True, "cancelled": True})
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PREVIEW_WAV_PATH, target)
        txt_target = None
        if create_subtitle_txt:
            txt_target = target.with_suffix(".txt")
            txt_target.write_text(text, encoding="utf-8")

        # Optional clean pre-RVC Finnish TTS for later high-quality accent editing.
        # Voice Generation keeps it visible beside the final WAV as *_source.wav.
        source_saved = False
        source_path = ""
        source_meta_path = ""
        preview_meta: dict = {}
        if PREVIEW_SOURCE_META_PATH.exists():
            try:
                loaded = json.loads(PREVIEW_SOURCE_META_PATH.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    preview_meta = loaded
            except Exception:
                preview_meta = {}

        source_mode = str(preview_meta.get("mode") or "")
        # PREVIEW_TTS_WAV_PATH is a genuine pre-RVC source for generated audio
        # and for imports that already had a saved clean source sidecar.
        # Do not create a misleading source sidecar for arbitrary final WAVs.
        if save_source_audio and PREVIEW_TTS_WAV_PATH.exists() and source_mode != "imported_final":
            saved_source_wav, saved_source_meta = _voice_source_paths(target)
            saved_source_wav.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(PREVIEW_TTS_WAV_PATH, saved_source_wav)

            meta = {
                "format": "zundanen_voice_source_v2",
                "final_wav": target.name,
                "text": text or str(preview_meta.get("text") or ""),
            }
            # Keep the original RVC/TTS settings when available. Imported Batch
            # sources store them either at the top level or under rvc_settings.
            rvc_settings = preview_meta.get("rvc_settings")
            if not isinstance(rvc_settings, dict):
                rvc_settings = {}
            for key in ("voice", "pitch", "index_rate", "protect", "expression", "seed"):
                value = preview_meta.get(key, rvc_settings.get(key))
                if value is not None:
                    meta[key] = value

            saved_source_meta.write_text(
                json.dumps(meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            source_saved = True
            source_path = str(saved_source_wav)
            source_meta_path = str(saved_source_meta)

        return jsonify({
            "ok": True,
            "cancelled": False,
            "path": str(target),
            "txt_path": str(txt_target) if txt_target else "",
            "source_saved": source_saved,
            "source_path": source_path,
            "source_meta_path": source_meta_path,
        })
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


def open_browser():
    webbrowser.open("http://127.0.0.1:8765")


if __name__ == "__main__":
    try:
        _cfg = load_config()
        ensure_rvc_language_override(_cfg["rvc_repo"])
    except Exception as exc:
        print(f"[WARN] Could not apply RVC English locale override: {exc}")
    port = int(os.environ.get("ZUNDANEN_PORT", "8765"))
    if os.environ.get("ZUNDANEN_NO_BROWSER", "0") != "1":
        threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    print("Zundanen")
    print(f"http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
