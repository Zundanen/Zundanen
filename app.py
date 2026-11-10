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
import time
import wave
import traceback
import webbrowser
import uuid
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from rvc_locale import ensure_rvc_language_override
from aalto_alignment import AlignmentError, native_alignment_status, run_alignment
from sentence_slicer import slice_generated_item, split_text_sentences

logging.getLogger("werkzeug").setLevel(logging.ERROR)

APP_DIR = Path(__file__).resolve().parent
APP_VERSION = "2.0.0"
CONFIG_PATH = APP_DIR / "config.json"
OUTPUT_DIR = APP_DIR / "outputs"
TEMP_DIR = APP_DIR / "temp"
BATCH_STATE_PATH = TEMP_DIR / "batch_queue_state.json"
BATCH_JOURNAL_PATH = TEMP_DIR / "batch_queue_journal.jsonl"
BATCH_STATE_CHECKPOINT_ITEMS = max(1, int(os.environ.get("ZUNDANEN_BATCH_CHECKPOINT_ITEMS", "500")))
BATCH_STATE_CHECKPOINT_SECONDS = max(5.0, float(os.environ.get("ZUNDANEN_BATCH_CHECKPOINT_SECONDS", "600")))
# 0 = unlimited. Set env ZUNDANEN_BATCH_HARD_LIMIT if an installation wants a guard.
BATCH_HARD_LIMIT = max(0, int(os.environ.get("ZUNDANEN_BATCH_HARD_LIMIT", "0")))
BATCH_LARGE_QUEUE_WARN = max(1, int(os.environ.get("ZUNDANEN_BATCH_LARGE_QUEUE_WARN", "50000")))
PREVIEW_WAV_PATH = TEMP_DIR / "voice_preview.wav"
PREVIEW_TTS_WAV_PATH = TEMP_DIR / "tts_preview.wav"
PREVIEW_TTS_TEXT_PATH = TEMP_DIR / "tts_preview.txt"
PREVIEW_ALIGNMENT_PATH = TEMP_DIR / "alignment_preview.json"
PREVIEW_SOURCE_META_PATH = TEMP_DIR / "preview_source.json"
PITCH_EDIT_TTS_WAV_PATH = TEMP_DIR / "tts_pitch_preview.wav"
PITCH_EDIT_WAV_PATH = TEMP_DIR / "voice_pitch_preview.wav"
STYLE_NEUTRAL_WAV_PATH = TEMP_DIR / "style_neutral_preview.wav"
STYLE_STYLED_WAV_PATH = TEMP_DIR / "style_styled_preview.wav"
WORLD_ACCENT_WORKER_PATH = APP_DIR / "world_accent_worker.py"
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
    STYLE_NEUTRAL_WAV_PATH,
    STYLE_STYLED_WAV_PATH,
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
    "step": 0,
    "steps": 0,   # set by the job; 0 = no job yet
    "progress": 0.0,
    "audio_files": 0,
    "log": [],
    "error": "",
    "result": {},
}

style_state_lock = threading.Lock()
style_process_lock = threading.Lock()
style_process = None
style_state = {
    "running": False,
    "done": False,
    "stopping": False,
    "phase": "idle",
    "phase_label": "Ready",
    "progress": 0.0,
    "current": 0,
    "total": 0,
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
    "source_only": False,
    "sentence_slicer": False,
    "fix_seed": False,
    "seed": 0,
    "total": 0,
    "current_index": 0,
    "completed": 0,
    "failed": 0,
    "pending": 0,
    "next_index": 0,
    "progress": 0.0,
    "current_text": "",
    "current_filename": "",
    "output_dir": "",
    "items": [],
    "recent_failed": [],
    "log": [],
    "error": "",
}

# In-memory persistence bookkeeping. Protected by batch_state_lock.
batch_persist_runtime = {
    "dirty_items": 0,
    "last_checkpoint": time.monotonic(),
}


class BatchStopped(RuntimeError):
    pass


# Persistent local Chatterbox worker used by Source-only mode.
# It keeps the model and reference-audio conditioning in RAM/VRAM between
# generations. No network service or online API is used.
source_tts_process_lock = threading.Lock()
source_tts_io_lock = threading.Lock()
source_tts_process = None
source_tts_signature = None
source_tts_request_id = 0
SOURCE_TTS_READY_PREFIX = "ZUNDANEN_TTS_READY\t"
SOURCE_TTS_RESULT_PREFIX = "ZUNDANEN_TTS_RESULT\t"
SOURCE_TTS_BATCH_RESTART_EVERY = 50

# Direct Zundanen VITS worker. Voice Generation and Batch Generation use this
# path exclusively; RVC is reserved for optional Model Trainer preprocessing.
vits_process_lock = threading.Lock()
vits_io_lock = threading.Lock()
vits_process = None
vits_signature = None
vits_request_id = 0
VITS_READY_PREFIX = "ZUNDANEN_VITS_READY\t"
VITS_RESULT_PREFIX = "ZUNDANEN_VITS_RESULT\t"
VITS_RESEARCH_DIR = APP_DIR / "research" / "finnish_style_tts"
VITS_WORKER_PATH = APP_DIR / "vits_persistent_worker.py"
VITS_TRAINER_WORKER_PATH = APP_DIR / "vits_trainer_worker.py"
STYLE_PROFILE_EXTRACTOR_PATH = VITS_RESEARCH_DIR / "extract_paired_style_profile.py"


def _path_signature(value: str) -> tuple:
    path = Path(str(value)).resolve()
    try:
        st = path.stat()
        return (str(path), int(st.st_size), int(st.st_mtime_ns))
    except OSError:
        return (str(path), None, None)


def _source_tts_config_signature(cfg: dict) -> tuple:
    return (
        str(Path(cfg["chatterbox_repo"]).resolve()),
        _path_signature(cfg["chatterbox_weights"]),
        _path_signature(cfg["reference_audio"]),
    )


def _kill_process_tree(proc) -> None:
    if not proc or proc.poll() is not None:
        return
    try:
        if sys.platform.startswith("win"):
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True,   # bytes: taskkill prints in the console code page (cp932), not UTF-8
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        else:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _stop_persistent_source_tts_worker() -> None:
    global source_tts_process, source_tts_signature
    with source_tts_process_lock:
        proc = source_tts_process
        source_tts_process = None
        source_tts_signature = None
    _kill_process_tree(proc)


def _ensure_persistent_source_tts_worker(cfg: dict):
    global source_tts_process, source_tts_signature
    signature = _source_tts_config_signature(cfg)
    with source_tts_process_lock:
        proc = source_tts_process
        if proc is not None and proc.poll() is None and source_tts_signature == signature:
            return proc
        old_proc = proc
        source_tts_process = None
        source_tts_signature = None
    _kill_process_tree(old_proc)

    cmd = [
        sys.executable, str(APP_DIR / "tts_persistent_worker.py"),
        "--repo", cfg["chatterbox_repo"],
        "--weights", cfg["chatterbox_weights"],
        "--reference", cfg["reference_audio"],
    ]
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    proc = subprocess.Popen(
        cmd,
        cwd=str(APP_DIR),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=creationflags,
    )

    startup_log: list[str] = []
    while True:
        line = proc.stdout.readline() if proc.stdout else ""
        if not line:
            code = proc.poll()
            detail = "".join(startup_log[-30:]).strip()
            _kill_process_tree(proc)
            raise RuntimeError(
                "Persistent Finnish TTS worker failed to start"
                + (f" (exit {code})" if code is not None else "")
                + (f":\n{detail}" if detail else ".")
            )
        if line.startswith(SOURCE_TTS_READY_PREFIX):
            payload = json.loads(line[len(SOURCE_TTS_READY_PREFIX):])
            if not payload.get("ok"):
                _kill_process_tree(proc)
                raise RuntimeError(payload.get("error") or "Persistent Finnish TTS worker failed to start.")
            break
        startup_log.append(line)

    with source_tts_process_lock:
        source_tts_process = proc
        source_tts_signature = signature
    return proc


def _persistent_source_tts_generate(
    *, text: str, output: Path, cfg: dict, expression: float,
    repetition_penalty: float, temperature: float, seed: int | None,
    batch: bool = False,
) -> dict:
    global source_tts_request_id, batch_current_process
    with source_tts_io_lock:
        proc = _ensure_persistent_source_tts_worker(cfg)
        if proc.stdin is None or proc.stdout is None:
            raise RuntimeError("Persistent Finnish TTS worker pipes are unavailable.")
        source_tts_request_id += 1
        request_id = source_tts_request_id
        request_payload = {
            "id": request_id,
            "text": str(text),
            "output": str(Path(output).resolve()),
            "repetition_penalty": float(repetition_penalty),
            "temperature": float(temperature),
            "exaggeration": float(expression),
            "seed": None if seed is None else int(seed),
        }
        if batch:
            with batch_process_lock:
                batch_current_process = proc
        try:
            proc.stdin.write(json.dumps(request_payload, ensure_ascii=False) + "\n")
            proc.stdin.flush()
            worker_log: list[str] = []
            while True:
                line = proc.stdout.readline()
                if not line:
                    if batch and batch_stop_event.is_set():
                        raise BatchStopped("Batch stopped")
                    detail = "".join(worker_log[-30:]).strip()
                    _stop_persistent_source_tts_worker()
                    raise RuntimeError(
                        "Persistent Finnish TTS worker stopped unexpectedly."
                        + (f"\n{detail}" if detail else "")
                    )
                if not line.startswith(SOURCE_TTS_RESULT_PREFIX):
                    worker_log.append(line)
                    continue
                payload = json.loads(line[len(SOURCE_TTS_RESULT_PREFIX):])
                if payload.get("id") != request_id:
                    continue
                if not payload.get("ok"):
                    raise RuntimeError(payload.get("error") or "Finnish TTS failed.")
                return payload
        except (BrokenPipeError, OSError):
            if batch and batch_stop_event.is_set():
                raise BatchStopped("Batch stopped")
            _stop_persistent_source_tts_worker()
            raise RuntimeError("Persistent Finnish TTS worker connection was lost.")
        finally:
            if batch:
                with batch_process_lock:
                    if batch_current_process is proc:
                        batch_current_process = None





def _vits_checkpoint_signature(checkpoint: str) -> tuple:
    return _path_signature(checkpoint)


def _stop_old_vits_worker_only() -> None:
    global vits_process, vits_signature
    with vits_process_lock:
        proc = vits_process
        vits_process = None
        vits_signature = None
    _kill_process_tree(proc)


def _stop_persistent_vits_worker() -> None:
    # v66: also stop the resident v65 emotion worker so training/style jobs get the VRAM.
    _stop_old_vits_worker_only()
    try:
        import emotion_engine
        emotion_engine.stop()
    except Exception:
        pass


def _ensure_persistent_vits_worker(checkpoint: str):
    global vits_process, vits_signature
    cp = Path(str(checkpoint)).expanduser().resolve()
    signature = _vits_checkpoint_signature(str(cp))
    with vits_process_lock:
        proc = vits_process
        if proc is not None and proc.poll() is None and vits_signature == signature:
            return proc
        old_proc = proc
        vits_process = None
        vits_signature = None
    _kill_process_tree(old_proc)

    if not cp.is_file():
        raise FileNotFoundError(f"VITS checkpoint not found: {cp}")
    if not VITS_WORKER_PATH.is_file():
        raise FileNotFoundError(f"VITS worker is missing: {VITS_WORKER_PATH}")
    if not VITS_RESEARCH_DIR.is_dir():
        raise FileNotFoundError(f"VITS research directory is missing: {VITS_RESEARCH_DIR}")

    cmd = [
        sys.executable, str(VITS_WORKER_PATH),
        "--root", str(APP_DIR),
        "--research-dir", str(VITS_RESEARCH_DIR),
        "--checkpoint", str(cp),
        "--device", "cuda",
    ]
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    proc = subprocess.Popen(
        cmd,
        cwd=str(APP_DIR),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=creationflags,
    )
    startup_log = []
    while True:
        line = proc.stdout.readline() if proc.stdout else ""
        if not line:
            code = proc.poll()
            detail = "".join(startup_log[-40:]).strip()
            _kill_process_tree(proc)
            raise RuntimeError(
                "Zundanen VITS worker failed to start"
                + (f" (exit {code})" if code is not None else "")
                + (f":\n{detail}" if detail else ".")
            )
        if line.startswith(VITS_READY_PREFIX):
            payload = json.loads(line[len(VITS_READY_PREFIX):])
            if not payload.get("ok"):
                _kill_process_tree(proc)
                raise RuntimeError(payload.get("error") or "Zundanen VITS worker failed to start.")
            break
        startup_log.append(line)
    with vits_process_lock:
        vits_process = proc
        vits_signature = signature
    return proc


def _persistent_vits_generate(*, text: str, output: Path, checkpoint: str, noise_scale: float,
                              length_scale: float, seed: int, batch: bool = False,
                              style_profile: str = "", style_name: str = "neutral",
                              style_strength: float = 0.0, apply_style_gain: bool = True) -> dict:
    global vits_request_id, batch_current_process
    with vits_io_lock:
        proc = _ensure_persistent_vits_worker(checkpoint)
        if proc.stdin is None or proc.stdout is None:
            raise RuntimeError("VITS worker pipes are unavailable.")
        vits_request_id += 1
        request_id = vits_request_id
        payload = {
            "id": request_id,
            "text": str(text),
            "output": str(Path(output).resolve()),
            "noise_scale": float(noise_scale),
            "length_scale": float(length_scale),
            "seed": int(seed),
            "style_profile": str(style_profile or ""),
            "style": str(style_name or "neutral"),
            "style_strength": float(style_strength),
            "apply_style_gain": bool(apply_style_gain),
        }
        if batch:
            with batch_process_lock:
                batch_current_process = proc
        try:
            proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            proc.stdin.flush()
            worker_log = []
            while True:
                line = proc.stdout.readline()
                if not line:
                    if batch and batch_stop_event.is_set():
                        raise BatchStopped("Batch stopped")
                    detail = "".join(worker_log[-40:]).strip()
                    _stop_persistent_vits_worker()
                    raise RuntimeError("VITS worker stopped unexpectedly." + (f"\n{detail}" if detail else ""))
                if not line.startswith(VITS_RESULT_PREFIX):
                    worker_log.append(line)
                    continue
                result = json.loads(line[len(VITS_RESULT_PREFIX):])
                if result.get("id") != request_id:
                    continue
                if not result.get("ok"):
                    raise RuntimeError(result.get("error") or "VITS generation failed.")
                return result
        except (BrokenPipeError, OSError):
            if batch and batch_stop_event.is_set():
                raise BatchStopped("Batch stopped")
            _stop_persistent_vits_worker()
            raise RuntimeError("VITS worker connection was lost.")
        finally:
            if batch:
                with batch_process_lock:
                    if batch_current_process is proc:
                        batch_current_process = None


def validate_vits_checkpoint(checkpoint: str) -> Path:
    cp = Path(str(checkpoint or "")).expanduser()
    if not str(checkpoint or "").strip():
        raise ValueError("Select a Zundanen VITS checkpoint (.pt).")
    cp = cp.resolve()
    if not cp.is_file() or cp.suffix.lower() != ".pt":
        raise FileNotFoundError(f"VITS checkpoint not found: {cp}")
    required = [VITS_WORKER_PATH, VITS_RESEARCH_DIR / "finbert_features.py",
                VITS_RESEARCH_DIR / "mas_vits_model.py",
                VITS_RESEARCH_DIR / "transformer_flow.py"]
    missing = [str(x) for x in required if not x.exists()]
    if missing:
        raise FileNotFoundError("Required VITS files are missing:\n" + "\n".join(missing))
    return cp


def generate_vits_file(*, text: str, output: Path, checkpoint: str, noise_scale: float = 0.4,
                       length_scale: float = 1.0, seed: int = 0, batch: bool = False,
                       style_profile: str = "", style_name: str = "neutral",
                       style_strength: float = 0.0, apply_style_gain: bool = True) -> dict:
    text = str(text).strip()
    if not text:
        raise ValueError("Finnish text is empty.")
    cp = validate_vits_checkpoint(checkpoint)
    noise_scale = max(0.0, min(1.5, float(noise_scale)))
    length_scale = max(0.2, min(3.0, float(length_scale)))
    seed = int(seed)
    if seed < 0 or seed > 2147483647:
        raise ValueError("Seed must be between 0 and 2147483647.")
    output.parent.mkdir(parents=True, exist_ok=True)
    return _persistent_vits_generate(
        text=text, output=output, checkpoint=str(cp), noise_scale=noise_scale,
        length_scale=length_scale, seed=seed, batch=batch,
        style_profile=style_profile, style_name=style_name,
        style_strength=style_strength, apply_style_gain=apply_style_gain,
    )


def _wav_duration_seconds(path: Path) -> float:
    """Return WAV duration using the stdlib only; 0.0 on unreadable files."""
    try:
        with wave.open(str(path), "rb") as wf:
            rate = int(wf.getframerate())
            frames = int(wf.getnframes())
        return (frames / rate) if rate > 0 else 0.0
    except Exception:
        return 0.0


def _append_source_only_timing_row(
    *, output_dir: Path, index: int, total: int, item: dict,
    generation_seconds: float, audio_seconds: float, worker_restart_count: int,
) -> bool:
    """Append Source-only timing metrics without ever failing audio generation.

    On Windows, spreadsheet programs commonly open CSV files with an exclusive
    lock. Timing telemetry is diagnostic only, so a locked CSV must never turn a
    successfully generated WAV into a failed batch item. When the main CSV is
    locked, store the row in a separate recovery CSV instead.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rtf = (generation_seconds / audio_seconds) if audio_seconds > 0 else 0.0
    row = [
        index, total, item.get("filename", ""), item.get("seed", ""),
        len(str(item.get("text", ""))),
        f"{audio_seconds:.4f}", f"{generation_seconds:.4f}", f"{rtf:.4f}",
        worker_restart_count,
    ]
    header = [
        "index", "total", "filename", "seed", "text_chars",
        "audio_seconds", "generation_seconds", "rtf", "worker_restart_count",
    ]

    def append_row(path: Path) -> None:
        exists = path.exists() and path.stat().st_size > 0
        with path.open("a", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            if not exists:
                writer.writerow(header)
            writer.writerow(row)

    main_path = output_dir / "source_only_timing.csv"
    try:
        append_row(main_path)
        return True
    except OSError:
        # Excel and similar applications may lock source_only_timing.csv on
        # Windows. Keep a complete recovery log while the main CSV is locked.
        fallback_path = output_dir / "source_only_timing_locked.csv"
        try:
            append_row(fallback_path)
        except OSError:
            # Timing telemetry must never affect the generated WAV or queue.
            pass
        return False

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
        "vits_checkpoint": "",
        "vits": {
            "noise_scale": 0.4,
            "length_scale": 1.0,
        },
        "voice_presets": {
            "Zundamon": {"pitch": 0, "index_rate": 0.55, "protect": 0.15},
            "Metan": {"pitch": 0, "index_rate": 0.55, "protect": 0.15},
        },
        "style_lab": {
            "checkpoint": "",
            "profile": "",
            "neutral_dir": "",
            "styles": [],
            "strength": 1.0,
            "noise_scale": 0.4,
            "length_scale": 1.0,
        },
        "trainer": {
            "mode": "scratch",
            "steps": 20000,
            "batch_size": 12,
            "save_every": 5000,
            "workers": min(6, os.cpu_count() or 1),
            "lr": 2e-4,
            "use_rvc": False,
            "rvc_model": "",
            "rvc_index": "",
            "rvc_pitch": 0,
            "rvc_index_rate": 0.75,
            "rvc_protect": 0.33,
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
    cfg["vits"] = defaults["vits"] | user.get("vits", {})
    cfg["voice_presets"] = defaults["voice_presets"] | user.get("voice_presets", {})
    cfg["trainer"] = defaults["trainer"] | user.get("trainer", {})
    cfg["style_lab"] = defaults["style_lab"] | user.get("style_lab", {})
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


def validate_tts_config(cfg: dict) -> None:
    required = {
        "Chatterbox repo": Path(cfg["chatterbox_repo"]),
        "Finnish weights": Path(cfg["chatterbox_weights"]),
        "Reference audio": Path(cfg["reference_audio"]),
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required Finnish TTS files/folders were not found:\n" + "\n".join(missing)
        )


def validate_config(cfg: dict) -> None:
    validate_tts_config(cfg)
    required = {
        "RVC Python": Path(cfg["rvc_python"]),
        "RVC repo": Path(cfg["rvc_repo"]),
        "RVC exports": Path(cfg["exports_root"]),
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required RVC files/folders were not found:\n" + "\n".join(missing)
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
    tts_copy: Path | None = None, source_only: bool = False,
) -> Path:
    text = str(text).strip()
    voice = str(voice).strip()
    if not text:
        raise ValueError("Finnish text is empty.")

    cfg = cfg or load_config()
    source_only = bool(source_only)
    if source_only:
        validate_tts_config(cfg)
        voices = scan_voices(cfg)
    else:
        validate_config(cfg)
        voices = scan_voices(cfg)
        if voice not in voices:
            raise ValueError(f"RVC model not found: {voice}")

    pitch = int(round(float(pitch)))
    index_rate = max(0.0, min(1.0, float(index_rate)))
    protect = max(0.0, min(0.5, float(protect)))
    expression = max(0.2, min(1.2, float(expression)))

    if persist_preset:
        if not source_only and voice in voices:
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
        if source_only:
            # Keep Chatterbox-Finnish resident for fast repeated Source-only
            # generation. This is a local child process; no network is used.
            _persistent_source_tts_generate(
                text=text, output=temp_tts, cfg=cfg, expression=expression,
                repetition_penalty=float(cfg["tts"]["repetition_penalty"]),
                temperature=float(cfg["tts"]["temperature"]), seed=seed, batch=batch,
            )
        else:
            # Normal RVC mode keeps the historical one-shot TTS path. Stop the
            # resident Source-only model first so it cannot occupy VRAM while a
            # second Chatterbox instance is being loaded.
            _stop_persistent_source_tts_worker()
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

        if source_only:
            # Fast source-only mode: the clean Chatterbox-Finnish output is the
            # final WAV. Do not invoke RVC at all.
            shutil.copy2(temp_tts, output)
        else:
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


def _batch_snapshot(*, include_items: bool = False) -> dict:
    """Return a JSON-safe snapshot without copying tens of thousands of items by default."""
    with batch_state_lock:
        if include_items:
            payload = batch_state
        else:
            payload = {k: v for k, v in batch_state.items() if k != "items"}
        return json.loads(json.dumps(payload, ensure_ascii=False))


def _clear_batch_journal_locked() -> None:
    try:
        BATCH_JOURNAL_PATH.unlink(missing_ok=True)
    except OSError:
        pass


def _save_batch_state_locked(*, clear_journal: bool = True) -> None:
    """Atomically save a compact full queue checkpoint.

    Large queues can contain 20k-50k+ items, so this is intentionally NOT called
    for every log line or every item. Small per-item status changes are journaled.
    """
    BATCH_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = BATCH_STATE_PATH.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(batch_state, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    tmp.replace(BATCH_STATE_PATH)
    batch_persist_runtime["dirty_items"] = 0
    batch_persist_runtime["last_checkpoint"] = time.monotonic()
    if clear_journal:
        _clear_batch_journal_locked()


def _append_batch_item_journal_locked(item: dict) -> None:
    """Persist one item transition without rewriting the whole large queue."""
    record = {
        "queue_id": batch_state.get("queue_id", ""),
        "id": item.get("id"),
        "status": item.get("status", "pending"),
        "output": item.get("output", ""),
        "error": item.get("error", ""),
        "slices": item.get("slices", []),
        "slice_warning": item.get("slice_warning", ""),
        "generation_seconds": item.get("generation_seconds"),
        "audio_seconds": item.get("audio_seconds"),
        "rtf": item.get("rtf"),
    }
    BATCH_JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with BATCH_JOURNAL_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:
            pass
    batch_persist_runtime["dirty_items"] += 1


def _replay_batch_journal_locked() -> None:
    if not BATCH_JOURNAL_PATH.is_file():
        return
    items = batch_state.get("items", [])
    by_id = {int(x.get("id", i + 1)): x for i, x in enumerate(items)}
    queue_id = str(batch_state.get("queue_id", ""))
    try:
        lines = BATCH_JOURNAL_PATH.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        try:
            rec = json.loads(line)
            if str(rec.get("queue_id", "")) != queue_id:
                continue
            item = by_id.get(int(rec.get("id")))
            if item is None:
                continue
            for key in (
                "status", "output", "error", "slices", "slice_warning",
                "generation_seconds", "audio_seconds", "rtf",
            ):
                if key in rec and rec[key] is not None:
                    item[key] = rec[key]
        except Exception:
            # A partially written last journal line after a hard crash is safe to ignore.
            continue


def _recount_batch_locked() -> None:
    items = batch_state.get("items", [])
    completed = 0
    failed = 0
    pending = 0
    first_pending = len(items)
    recent_failed = []
    for i, item in enumerate(items):
        status = item.get("status")
        if status == "processing":
            # Server restart means this item must be tried again.
            item["status"] = "pending"
            item["error"] = ""
            status = "pending"
        if status == "done":
            completed += 1
        elif status == "failed":
            failed += 1
            recent_failed.append({
                "id": item.get("id"),
                "filename": item.get("filename"),
                "text": item.get("text"),
                "error": item.get("error"),
            })
        else:
            if status != "pending":
                item["status"] = "pending"
            pending += 1
            if first_pending == len(items):
                first_pending = i
    batch_state["completed"] = completed
    batch_state["failed"] = failed
    batch_state["pending"] = pending
    batch_state["next_index"] = first_pending if pending else len(items)
    batch_state["recent_failed"] = recent_failed[-100:]
    finished = completed + failed
    batch_state["progress"] = 100.0 * finished / max(1, int(batch_state.get("total") or len(items) or 1))


def _checkpoint_batch_state_if_due_locked(*, force: bool = False) -> None:
    elapsed = time.monotonic() - float(batch_persist_runtime.get("last_checkpoint", 0.0))
    if (
        force
        or int(batch_persist_runtime.get("dirty_items", 0)) >= BATCH_STATE_CHECKPOINT_ITEMS
        or elapsed >= BATCH_STATE_CHECKPOINT_SECONDS
    ):
        _save_batch_state_locked(clear_journal=True)


def save_batch_state() -> None:
    with batch_state_lock:
        _save_batch_state_locked(clear_journal=True)


def set_batch(**updates) -> None:
    with batch_state_lock:
        batch_state.update(updates)
        _save_batch_state_locked(clear_journal=True)


def append_batch_log(message: str) -> None:
    # Logs are UI telemetry, not recovery-critical state. Keeping them in RAM and
    # persisting them at periodic/full checkpoints avoids rewriting a multi-MB
    # queue file for every ITEM_START / ITEM_DONE message.
    with batch_state_lock:
        batch_state["log"].append(str(message))
        if len(batch_state["log"]) > 500:
            batch_state["log"] = batch_state["log"][-500:]


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
            batch_state.setdefault("source_only", False)
            batch_state.setdefault("sentence_slicer", False)
            batch_state.setdefault("fix_seed", False)
            batch_state.setdefault("seed", 0)
            batch_state.setdefault("next_index", 0)
            batch_state.setdefault("recent_failed", [])

            # Apply item completions/failures written after the last large full checkpoint.
            _replay_batch_journal_locked()

            # Seed recovery + compatibility repair from older persistent-resume builds.
            for item in batch_state.get("items", []):
                if "seed" not in item or item.get("seed") is None:
                    if batch_state.get("fix_seed"):
                        item["seed"] = int(batch_state.get("seed", 0))
                    else:
                        item["seed"] = secrets.randbelow(2147483648)
                if item.get("status") == "failed":
                    err = str(item.get("error", ""))
                    if "source_only_timing.csv" in err and (
                        "Permission denied" in err or "Errno 13" in err
                    ):
                        recovered = False
                        try:
                            output_dir = Path(str(batch_state.get("output_dir") or ""))
                            output = output_dir / str(item.get("filename") or "")
                            wav_ok = False
                            if output.is_file() and output.stat().st_size > 44:
                                with wave.open(str(output), "rb") as wf:
                                    wav_ok = wf.getnframes() > 0 and wf.getframerate() > 0
                            txt_ok = True
                            if bool(batch_state.get("create_subtitle_txt", True)):
                                txt_ok = output.with_suffix(".txt").is_file()
                            if wav_ok and txt_ok:
                                item["status"] = "done"
                                item["output"] = str(output)
                                item["error"] = ""
                                recovered = True
                        except Exception:
                            recovered = False
                        if not recovered:
                            item["status"] = "pending"
                            item["error"] = ""

            batch_state["running"] = False
            batch_state["stopping"] = False
            _recount_batch_locked()
            batch_state["resumable"] = batch_state["pending"] > 0
            batch_state["done"] = batch_state["pending"] == 0 and bool(batch_state.get("items"))
            _save_batch_state_locked(clear_journal=True)
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


def build_batch_items(data: dict, cfg: dict, *, source_only: bool = False) -> tuple[list[dict], str]:
    raw = str(data.get("content", ""))
    mode = str(data.get("mode", "lines")).lower()
    v65 = str(data.get("engine", "")).lower() == "v65"
    checkpoint = "" if v65 else str(data.get("checkpoint") or cfg.get("vits_checkpoint") or "").strip()
    if not v65:
        validate_vits_checkpoint(checkpoint)
    defaults = {
        "noise_scale": data.get("noise_scale", cfg.get("vits", {}).get("noise_scale", 0.4)),
        "length_scale": data.get("length_scale", cfg.get("vits", {}).get("length_scale", 1.0)),
        "seed": data.get("seed", 0),
        "fix_seed": bool(data.get("fix_seed", False)),
    }
    rows = []
    if mode == "csv":
        reader = csv.DictReader(io.StringIO(raw.lstrip("\ufeff")))
        if not reader.fieldnames or "text" not in {str(x).strip().lower() for x in reader.fieldnames if x}:
            raise ValueError("CSV must contain a text column.")
        for row in reader:
            n = {str(k).strip().lower(): v for k, v in row.items() if k is not None}
            text = str(n.get("text", "") or "").strip()
            if not text:
                continue
            rows.append({
                "text": text,
                "filename": str(n.get("filename", "") or "").strip(),
                "noise_scale": n.get("noise_scale", defaults["noise_scale"]),
                "length_scale": n.get("length_scale", defaults["length_scale"]),
                "seed": n.get("seed", ""),
                "speed": str(n.get("speed", "") or "").strip(),
                "style": str(n.get("style", "") or "").strip().lower(),
                "strength": str(n.get("strength", "") or "").strip(),
            })
    else:
        mode = "lines"
        for line in raw.splitlines():
            text = line.strip()
            if text:
                rows.append({"text": text, "filename": "", **defaults, "seed": ""})
    if not rows:
        raise ValueError("No text items were found.")
    if BATCH_HARD_LIMIT and len(rows) > BATCH_HARD_LIMIT:
        raise ValueError(f"This installation limits one batch to {BATCH_HARD_LIMIT} items via ZUNDANEN_BATCH_HARD_LIMIT.")
    width = max(3, len(str(len(rows))))
    items = []
    for idx, row in enumerate(rows, 1):
        noise_scale = _as_number(row.get("noise_scale"), defaults["noise_scale"], lo=0, hi=1.5)
        if str(row.get("speed", "") or "").strip():   # v2: speed = 1 / length_scale (larger = faster)
            try:
                length_scale = _as_number(1.0 / float(row["speed"]), defaults["length_scale"], lo=0.2, hi=3.0)
            except (ValueError, ZeroDivisionError):
                raise ValueError("speed must be a positive number (1.0 = normal).")
        else:
            length_scale = _as_number(row.get("length_scale"), defaults["length_scale"], lo=0.2, hi=3.0)
        row_seed = _as_seed(row.get("seed"), allow_blank=True)
        seed = row_seed if row_seed is not None else (_as_seed(defaults["seed"]) if defaults["fix_seed"] else secrets.randbelow(2147483648))
        number = f"{idx:0{width}d}"
        custom = str(row.get("filename", "") or "").strip()
        if mode == "csv" and custom:
            stem = safe_output_stem(Path(custom).stem, fallback="")
            filename = f"{number}.wav" if stem in {number, str(idx)} or not stem else (f"{stem}.wav" if stem.startswith(number + "_") else f"{number}_{stem}.wav")
        else:
            filename = f"{number}.wav"
        fp = Path(filename)
        filename = f"{fp.stem}_seed{seed}{fp.suffix or '.wav'}"
        item = {
            "id": idx, "text": row["text"], "checkpoint": checkpoint,
            "noise_scale": round(noise_scale, 3), "length_scale": round(length_scale, 3),
            "seed": seed, "filename": filename, "status": "pending", "output": "", "error": "",
        }
        if v65:
            item["noise_scale"] = 0.4  # v2.0.0: fixed
            if row.get("style"):
                item["style"] = row["style"]
            if row.get("strength"):
                item["strength"] = _as_number(row["strength"], 1.0, lo=0, hi=2)
        items.append(item)
    return items, mode


def _write_failed_csv(snapshot: dict) -> None:
    failed = [item for item in snapshot.get("items", []) if item.get("status") == "failed"]
    if not failed or not snapshot.get("output_dir"):
        return
    path = Path(snapshot["output_dir"]) / "failed_items.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filename", "text", "checkpoint", "noise_scale", "length_scale", "seed", "error"])
        for item in failed:
            writer.writerow([
                item.get("filename", ""), item.get("text", ""), item.get("checkpoint", ""),
                item.get("noise_scale", ""), item.get("length_scale", ""),
                item.get("seed", ""), item.get("error", ""),
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

# the built-in Chatterbox model
STEP7A_100K_CHECKPOINT = APP_DIR / "models/chatterbox.pt"


def _generate_chatterbox(*, text: str, output: Path, seed: int, noise_scale: float, length_scale: float,
                         batch: bool = False) -> dict:
    """v73: Character "Chatterbox" = the Finnish source VITS (Finnish VITS 100k), Normal only.
    With Use RVC the caller converts it (e.g. Zundamon RVC +8): v73 measured 5.4% CER vs 7.1% for the
    emotion model's Normal on 60 held-out sentences."""
    result = generate_vits_file(text=text, output=output, checkpoint=str(STEP7A_100K_CHECKPOINT),
                                noise_scale=noise_scale, length_scale=length_scale, seed=seed, batch=batch)
    return {"character": "Chatterbox", "style": "normal", "strength": 0.0, "engine": "vits",
            "checkpoint": str(STEP7A_100K_CHECKPOINT), "warning": result.get("warning", "")}


def _batch_generate_v65(item: dict, output: Path, v65_cfg: dict, cfg: dict) -> None:
    """v2.0.0 batch item: emotion TTS (character / style / pitch), then optional RVC."""
    import emotion_engine
    if batch_stop_event.is_set():
        raise BatchStopped("Batch stopped")
    chatterbox = emotion_engine.is_chatterbox(v65_cfg["character"])
    if not chatterbox:
        _stop_old_vits_worker_only()
    style = item.get("style") or v65_cfg["style"]
    strength = item.get("strength", v65_cfg["strength"] if style == v65_cfg["style"] else None)
    tts_out = TEMP_DIR / f"batch_tts_{uuid.uuid4().hex}.wav" if v65_cfg.get("use_rvc") else output
    try:
        if chatterbox:
            _generate_chatterbox(text=item["text"], output=tts_out, seed=int(item["seed"]), noise_scale=0.4,
                                 length_scale=float(item["length_scale"]), batch=True)
        else:
            emotion_engine.generate(text=item["text"], output=tts_out, style=style, strength=strength, seed=int(item["seed"]),
                                noise_scale=0.4, length_scale=float(item["length_scale"]), level=v65_cfg["level"],
                                character=v65_cfg["character"])
        if v65_cfg.get("use_rvc"):
            run_rvc_on_audio(input_wav=tts_out, output_wav=output, voice=v65_cfg["rvc_voice"], pitch=v65_cfg["rvc_pitch"],
                             index_rate=v65_cfg["rvc_index_rate"], protect=v65_cfg["rvc_protect"], cfg=cfg, batch=True)
    finally:
        if tts_out != output:
            try:
                tts_out.unlink(missing_ok=True)
            except OSError:
                pass


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
        append_batch_log("[BATCH] STARTED")
        while True:
            with batch_state_lock:
                items = batch_state["items"]
                total = len(items)
                if batch_stop_event.is_set() or batch_state.get("stopping"):
                    break
                cursor = max(0, int(batch_state.get("next_index", 0)))
                while cursor < total and items[cursor].get("status") != "pending":
                    cursor += 1
                if cursor >= total and int(batch_state.get("pending", 0)) > 0:
                    cursor = 0
                    while cursor < total and items[cursor].get("status") != "pending":
                        cursor += 1
                if cursor >= total:
                    break
                i = cursor
                item = items[i]
                item["status"] = "processing"
                item["error"] = ""
                batch_state["pending"] = max(0, int(batch_state.get("pending", 0)) - 1)
                batch_state["next_index"] = i + 1
                batch_state["current_index"] = i + 1
                batch_state["current_text"] = item["text"]
                batch_state["current_filename"] = item["filename"]
                output_dir = Path(batch_state["output_dir"])
                v65_cfg = batch_state.get("v65")
                sentence_slicer = bool(batch_state.get("sentence_slicer", False))
                create_subtitle_txt = bool(batch_state.get("create_subtitle_txt", True))

            append_batch_log(f"[BATCH] ITEM_START {i+1}/{total} {item['filename']} SEED={item.get('seed','')}")
            output = output_dir / item["filename"]
            try:
                if v65_cfg:
                    _batch_generate_v65(item, output, v65_cfg, cfg)
                else:
                    generate_vits_file(
                        text=item["text"], output=output, checkpoint=item["checkpoint"],
                        noise_scale=item["noise_scale"], length_scale=item["length_scale"],
                        seed=item["seed"], batch=True,
                    )
                slice_files = []
                slice_warning = ""
                if sentence_slicer and len(split_text_sentences(item["text"])) > 1:
                    try:
                        slices = slice_generated_item(
                            final_wav=output, transcript=item["text"], alignment_wav=output,
                            source_wav=None, create_subtitle_txt=create_subtitle_txt,
                        )
                        if slices:
                            slice_files = [str(x.wav) for x in slices]
                            append_batch_log(f"[BATCH] SLICED {i+1}/{total} {item['filename']} -> {len(slices)} sentence WAVs")
                    except Exception as slicer_exc:
                        slice_warning = str(slicer_exc)
                        append_batch_log(f"[BATCH] SLICER_WARN {i+1}/{total} {slice_warning.splitlines()[-1]}")
                if create_subtitle_txt and output.exists() and not slice_files:
                    output.with_suffix(".txt").write_text(item["text"], encoding="utf-8")
                with batch_state_lock:
                    item["status"] = "done"
                    item["output"] = str(output)
                    item["slice_files"] = slice_files
                    item["slice_warning"] = slice_warning
                    batch_state["completed"] = int(batch_state.get("completed", 0)) + 1
                    finished = int(batch_state.get("completed", 0)) + int(batch_state.get("failed", 0))
                    batch_state["progress"] = 100.0 * finished / max(1, int(batch_state.get("total", total)))
                    _append_batch_item_journal_locked(item)
                    _checkpoint_batch_state_if_due_locked()
                append_batch_log(f"[BATCH] ITEM_DONE {i+1}/{total} {item['filename']}")
            except BatchStopped:
                with batch_state_lock:
                    item["status"] = "pending"
                    batch_state["pending"] = int(batch_state.get("pending", 0)) + 1
                    batch_state["next_index"] = min(int(batch_state.get("next_index", i)), i)
                    _append_batch_item_journal_locked(item)
                    _checkpoint_batch_state_if_due_locked(force=True)
                break
            except Exception as exc:
                short_error = str(exc)[-4000:]
                with batch_state_lock:
                    item["status"] = "failed"
                    item["error"] = short_error
                    batch_state["failed"] = int(batch_state.get("failed", 0)) + 1
                    recent = batch_state.setdefault("recent_failed", [])
                    recent.append({"id": item.get("id"), "filename": item.get("filename"), "text": item.get("text"), "error": short_error})
                    if len(recent) > 100: del recent[:-100]
                    finished = int(batch_state.get("completed", 0)) + int(batch_state.get("failed", 0))
                    batch_state["progress"] = 100.0 * finished / max(1, int(batch_state.get("total", total)))
                    _append_batch_item_journal_locked(item)
                    _checkpoint_batch_state_if_due_locked()
                append_batch_log(f"[BATCH] ITEM_FAILED {i+1}/{total} {item['filename']} :: {short_error.splitlines()[-1] if short_error else 'Unknown error'}")
        with batch_state_lock:
            _recount_batch_locked()
            batch_state["running"] = False
            batch_state["stopping"] = False
            batch_state["current_text"] = ""
            batch_state["current_filename"] = ""
            batch_state["resumable"] = batch_state["pending"] > 0
            batch_state["done"] = batch_state["pending"] == 0
            if batch_state["done"]: batch_state["progress"] = 100.0
            batch_state["log"].append("[BATCH] STOPPED" if batch_state["resumable"] else "[BATCH] COMPLETE")
            _save_batch_state_locked(clear_journal=True)
        _write_failed_csv(_batch_snapshot(include_items=True))
    except Exception as exc:
        append_batch_log(f"[BATCH] ERROR {exc}")
        set_batch(running=False, stopping=False, resumable=True, error=str(exc))
    finally:
        _stop_old_vits_worker_only()  # keep the resident emotion model for Voice Generation
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


def _accent_support_status() -> dict:
    cfg = load_config()
    rvc_python = Path(str(cfg.get("rvc_python") or ""))
    pyworld_ok = False
    if rvc_python.exists():
        proc = _run_quiet([str(rvc_python), "-c", "import pyworld"], timeout=20)
        pyworld_ok = bool(proc and proc.returncode == 0)

    worker_ok = WORLD_ACCENT_WORKER_PATH.exists()
    native = native_alignment_status()
    native_ready = bool(native.get("ready"))
    ready = bool(os.name == "nt" and pyworld_ok and worker_ok and native_ready)
    return {
        "ok": True,
        "installed": ready,
        "ready": ready,
        "components": {
            "aalto_native": native_ready,
            "aalto_native_root": str(native.get("root") or ""),
            "aalto_native_missing": native.get("missing") or [],
            "pyworld": pyworld_ok,
            "world_worker": worker_ok,
        },
    }


@app.get("/")
def index():
    return send_from_directory(APP_DIR, "index.html")


@app.get("/api/state")
def state():
    cfg = load_config()
    return jsonify({
        "version": APP_VERSION,
        "voices": scan_voices(cfg),  # legacy/import compatibility only
        "vits": cfg.get("vits", {}),
        "config": {
            "vits_checkpoint": cfg.get("vits_checkpoint", ""),
            "rvc_python": cfg["rvc_python"],
            "rvc_repo": cfg["rvc_repo"],
            "exports_root": cfg["exports_root"],
        },
        "trainer": cfg["trainer"],
        "style_lab": cfg.get("style_lab", {}),
    })


@app.get("/api/accent-support/status")
def accent_support_status():
    return jsonify(_accent_support_status())


@app.post("/api/config")
def update_config():
    data = request.get_json(force=True) or {}
    cfg = load_config()
    for key in ["vits_checkpoint", "rvc_python", "rvc_repo", "exports_root"]:
        if key in data:
            cfg[key] = str(data[key]).strip()
    save_config(cfg)
    _stop_persistent_vits_worker()
    return jsonify({"ok": True})


@app.post("/api/generate")
def generate():
    with training_state_lock:
        if training_state["running"]:
            return jsonify({"ok": False, "error": "Model training is running."}), 409
    with style_state_lock:
        if style_state["running"]:
            return jsonify({"ok": False, "error": "Style profile extraction is running."}), 409
    with batch_state_lock:
        if batch_state["running"]:
            return jsonify({"ok": False, "error": "Batch generation is running."}), 409
    if not generation_lock.acquire(blocking=False):
        return jsonify({"ok": False, "error": "Another generation job is currently running."}), 409
    try:
        data = request.get_json(force=True) or {}
        text = str(data.get("text", "")).strip()
        if not text:
            return jsonify({"ok": False, "error": "Enter Finnish text."}), 400
        cfg = load_config()
        checkpoint = str(data.get("checkpoint") or cfg.get("vits_checkpoint") or "").strip()
        try:
            noise_scale = max(0.0, min(1.5, float(data.get("noise_scale", cfg.get("vits", {}).get("noise_scale", 0.4)))))
            length_scale = max(0.2, min(3.0, float(data.get("length_scale", cfg.get("vits", {}).get("length_scale", 1.0)))))
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "Noise scale / Length scale contains an invalid number."}), 400
        fix_seed = bool(data.get("fix_seed", False))
        try:
            seed = int(data.get("seed", 0)) if fix_seed else secrets.randbelow(2147483648)
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "Seed must be an integer."}), 400
        if seed < 0 or seed > 2147483647:
            return jsonify({"ok": False, "error": "Seed must be between 0 and 2147483647."}), 400
        if str(data.get("engine", "vits")).lower() == "v65":
            # v66: Finnish emotion TTS (v65 preset: Normal / Ama / Sexy / Tsun).
            import emotion_engine
            # the Chatterbox character (Finnish VITS) does not need the emotion models (none in the public release)
            ok, msg = (True, "") if emotion_engine.is_chatterbox(data.get("character")) else emotion_engine.available()
            if not ok:
                return jsonify({"ok": False, "error": msg}), 400
            def _opt_float(name):
                v = data.get(name)
                return None if v in (None, "") else float(v)
            try:
                v65_strength = _opt_float("v65_strength")
                v65_level = _opt_float("v65_level")
                v65_range = _opt_float("v65_range_k")
            except (TypeError, ValueError):
                return jsonify({"ok": False, "error": "v65 settings contain an invalid number."}), 400
            for preview_temp in (PREVIEW_TTS_WAV_PATH, PREVIEW_TTS_TEXT_PATH, PREVIEW_ALIGNMENT_PATH,
                                 PREVIEW_SOURCE_META_PATH, PITCH_EDIT_TTS_WAV_PATH, PITCH_EDIT_WAV_PATH):
                try:
                    preview_temp.unlink(missing_ok=True)
                except OSError:
                    pass
            chatterbox = emotion_engine.is_chatterbox(data.get("character"))
            if not chatterbox:
                _stop_old_vits_worker_only()  # do not keep two TTS models in VRAM
            use_rvc = bool(data.get("use_rvc", False))
            rvc_voice = str(data.get("rvc_voice", "")).strip()
            if use_rvc and rvc_voice not in scan_voices(cfg):
                return jsonify({"ok": False, "error": f"RVC model not found in exports: {rvc_voice or '(none)'}"}), 400
            if chatterbox:   # v73: Finnish VITS 100k (Chatterbox voice), Normal only
                v65_result = _generate_chatterbox(text=text, output=PREVIEW_WAV_PATH, seed=seed,
                                                  noise_scale=noise_scale, length_scale=length_scale)
            else:
                v65_result = emotion_engine.generate(
                    text=text, output=PREVIEW_WAV_PATH, style=str(data.get("emotion_style", "normal")),
                    strength=v65_strength, seed=seed, noise_scale=noise_scale, length_scale=length_scale,
                    level=v65_level, range_k=v65_range, character=str(data.get("character", "zundamon")),
                )
            shutil.copy2(PREVIEW_WAV_PATH, PREVIEW_TTS_WAV_PATH)
            rvc_info = None
            if use_rvc:
                # v2.0.0: optional RVC voice conversion of the generated speech.
                try:
                    rvc_info = {"voice": rvc_voice, "pitch": int(round(float(data.get("rvc_pitch", 0)))),
                                "index_rate": max(0.0, min(1.0, float(data.get("rvc_index_rate", 0.1)))),
                                "protect": max(0.0, min(0.5, float(data.get("rvc_protect", 0.15))))}
                except (TypeError, ValueError):
                    return jsonify({"ok": False, "error": "RVC settings contain an invalid number."}), 400
                tmp_out = TEMP_DIR / f"v2_rvc_{uuid.uuid4().hex}.wav"
                try:
                    run_rvc_on_audio(input_wav=PREVIEW_TTS_WAV_PATH, output_wav=tmp_out, voice=rvc_voice,
                                     pitch=rvc_info["pitch"], index_rate=rvc_info["index_rate"],
                                     protect=rvc_info["protect"], cfg=cfg)
                    shutil.move(str(tmp_out), str(PREVIEW_WAV_PATH))
                finally:
                    try:
                        tmp_out.unlink(missing_ok=True)
                    except OSError:
                        pass
            PREVIEW_TTS_TEXT_PATH.write_text(text, encoding="utf-8")
            PREVIEW_SOURCE_META_PATH.write_text(json.dumps({
                "format": "zundanen_vits_preview_v1",
                "mode": "generated_vits",
                "engine": "v65",
                "text": text,
                "noise_scale": noise_scale,
                "length_scale": length_scale,
                "seed": seed,
                "v65": v65_result,
                "rvc": rvc_info,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            who = safe_output_stem(rvc_info["voice"], fallback="rvc") if rvc_info else v65_result.get("character", "zundamon")
            return jsonify({
                "ok": True,
                "engine": "v65",
                "filename": f"Zundanen_{who}_{v65_result['style']}_{stamp}_seed{seed}.wav",
                "rvc": rvc_info,
                "audio_url": "/api/preview-audio",
                "seed": seed,
                "noise_scale": noise_scale,
                "length_scale": length_scale,
                "warning": v65_result.get("warning", ""),
                "v65": v65_result,
                "learned_style": {"style": v65_result["style"], "bypass": v65_result["style"] == "normal"},
            })
        try:
            import emotion_engine
            if emotion_engine.running():
                emotion_engine.stop()  # free VRAM before loading the classic VITS worker
        except Exception:
            pass
        import style_models
        style_model_id = str(data.get('style_model', ''))
        learned_style = str(data.get('learned_style', 'neutral')).lower()
        try:
            learned_strength = float(data.get('style_strength', 1.0))
            style_models.resolve(style_model_id, learned_style, learned_strength)
        except (ValueError, TypeError) as exc:
            return jsonify(ok=False, error=str(exc)), 400
        cp = validate_vits_checkpoint(checkpoint)
        cfg["vits_checkpoint"] = str(cp)
        cfg.setdefault("vits", {})["noise_scale"] = noise_scale
        cfg.setdefault("vits", {})["length_scale"] = length_scale
        save_config(cfg)

        for preview_temp in (PREVIEW_TTS_WAV_PATH, PREVIEW_TTS_TEXT_PATH, PREVIEW_ALIGNMENT_PATH,
                             PREVIEW_SOURCE_META_PATH, PITCH_EDIT_TTS_WAV_PATH, PITCH_EDIT_WAV_PATH):
            try:
                preview_temp.unlink(missing_ok=True)
            except OSError:
                pass

        result = generate_vits_file(
            text=text, output=PREVIEW_WAV_PATH, checkpoint=str(cp), noise_scale=noise_scale,
            length_scale=length_scale, seed=seed, batch=False,
        )
        style_result = dict(style='neutral', bypass=True)
        if learned_style != 'neutral' and learned_strength > 0:
            # Keep the untouched Finnish waveform beside the converted output.
            style_id = secrets.token_hex(8)
            base_audio = TEMP_DIR / ('style_base_' + style_id + '.wav')
            styled_audio = TEMP_DIR / ('styled_' + style_id + '.wav')
            shutil.copy2(PREVIEW_WAV_PATH, base_audio)
            style_result = style_models.apply(base_audio, styled_audio, style_model_id,
                                               learned_style, learned_strength)
            shutil.copy2(styled_audio, PREVIEW_WAV_PATH)
        # Accent analysis works directly on the final VITS waveform.
        shutil.copy2(PREVIEW_WAV_PATH, PREVIEW_TTS_WAV_PATH)
        PREVIEW_TTS_TEXT_PATH.write_text(text, encoding="utf-8")
        PREVIEW_SOURCE_META_PATH.write_text(json.dumps({
            "format": "zundanen_vits_preview_v1",
            "mode": "generated_vits",
            "text": text,
            "checkpoint": str(cp),
            "noise_scale": noise_scale,
            "length_scale": length_scale,
            "seed": seed,
            "learned_style": style_result,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return jsonify({
            "ok": True,
            "filename": f"{safe_output_stem(cp.stem, fallback='Zundanen')}_{stamp}_seed{seed}.wav",
            "audio_url": "/api/preview-audio",
            "seed": seed,
            "noise_scale": noise_scale,
            "length_scale": length_scale,
            "warning": result.get("warning", ""),
            "learned_style": style_result,
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
    # while forced alignment is reading it.
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
        if source_mode in {"imported_final", "generated_tts_source_only", "generated_vits"}:
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
    with style_state_lock:
        if style_state.get("running"):
            return jsonify({"ok": False, "error": "Style profile extraction is running."}), 409
    with batch_state_lock:
        if batch_state["running"]:
            return jsonify({"ok": False, "error": "A batch queue is already running."}), 409
    if generation_lock.locked():
        return jsonify({"ok": False, "error": "Another generation job is already running."}), 409
    data = request.get_json(force=True) or {}
    create_subtitle_txt = bool(data.get("create_subtitle_txt", True))
    sentence_slicer = bool(data.get("sentence_slicer", False))
    fix_seed = bool(data.get("fix_seed", False))
    try:
        batch_seed = _as_seed(data.get("seed", 0))
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    cfg = load_config()
    try:
        items, mode = build_batch_items(data, cfg)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    v65_cfg = None
    if str(data.get("engine", "")).lower() == "v65":
        import emotion_engine
        req_char = str(data.get("character", "")) or "Zundamon"
        key = emotion_engine.CHATTERBOX if emotion_engine.is_chatterbox(req_char) else emotion_engine.find_character(req_char)
        if key is None:
            return jsonify({"ok": False, "error": "Select a character."}), 400
        try:
            v65_cfg = {"character": key, "style": "normal" if emotion_engine.is_chatterbox(key) else str(data.get("style", "normal")).lower(),
                       "strength": float(data.get("strength", 0)),
                       "level": 0.0 if emotion_engine.is_chatterbox(key) else float(data.get("level", emotion_engine.characters()[key]["level_st"])),
                       "use_rvc": bool(data.get("use_rvc", False)), "rvc_voice": str(data.get("rvc_voice", "")).strip(),
                       "rvc_pitch": int(round(float(data.get("rvc_pitch", 0)))),
                       "rvc_index_rate": max(0.0, min(1.0, float(data.get("rvc_index_rate", 0.1)))),
                       "rvc_protect": max(0.0, min(0.5, float(data.get("rvc_protect", 0.15))))}
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "Batch voice settings contain an invalid number."}), 400
        if v65_cfg["use_rvc"] and v65_cfg["rvc_voice"] not in scan_voices(cfg):
            return jsonify({"ok": False, "error": f"RVC model not found in exports: {v65_cfg['rvc_voice'] or '(none)'}"}), 400
    checkpoint = str(data.get("checkpoint") or cfg.get("vits_checkpoint") or "").strip()
    if v65_cfg is None:
        cfg["vits_checkpoint"] = checkpoint
    cfg.setdefault("vits", {})["noise_scale"] = float(data.get("noise_scale", 0.4))
    cfg.setdefault("vits", {})["length_scale"] = float(data.get("length_scale", 1.0))
    save_config(cfg)
    queue_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    output_dir = OUTPUT_DIR / "batches" / queue_id
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_stop_event.clear()
    with batch_state_lock:
        batch_state.clear()
        batch_state.update({
            "running": True, "done": False, "stopping": False, "resumable": False,
            "queue_id": queue_id, "mode": mode, "create_subtitle_txt": create_subtitle_txt,
            "save_source_audio": False, "source_only": False, "sentence_slicer": sentence_slicer,
            "fix_seed": fix_seed, "seed": batch_seed, "total": len(items), "current_index": 0,
            "completed": 0, "failed": 0, "pending": len(items), "next_index": 0, "progress": 0.0,
            "current_text": "", "current_filename": "", "output_dir": str(output_dir),
            "items": items, "recent_failed": [], "log": [], "error": "", "v65": v65_cfg,
        })
        _save_batch_state_locked()
    threading.Thread(target=batch_runner, daemon=True).start()
    return jsonify({"ok": True, "queue_id": queue_id, "total": len(items)})


@app.get("/api/batch/status")
def batch_status():
    # No full item-list copy/scan here: the browser polls this endpoint often,
    # and a 20k-50k queue must stay O(1) per poll.
    snap = _batch_snapshot(include_items=False)
    snap["failed_items"] = list(snap.get("recent_failed", []))[-100:]
    snap.pop("recent_failed", None)
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
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
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
    with style_state_lock:
        if style_state.get("running"):
            return jsonify({"ok": False, "error": "Style profile extraction is running."}), 409
    with batch_state_lock:
        if batch_state.get("running"):
            return jsonify({"ok": False, "error": "A batch queue is already running."}), 409
        pending = int(batch_state.get("pending", 0))
        if pending <= 0:
            # Compatibility fallback for a state file from an older build.
            _recount_batch_locked()
            pending = int(batch_state.get("pending", 0))
        if pending == 0:
            return jsonify({"ok": False, "error": "There is no pending batch queue to resume."}), 400
        batch_state["running"] = True
        batch_state["stopping"] = False
        batch_state["resumable"] = False
        batch_state["done"] = False
        batch_state["error"] = ""
        batch_state["pending"] = pending
        _save_batch_state_locked(clear_journal=True)
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
            "queue_id": "", "mode": "lines", "create_subtitle_txt": True, "save_source_audio": False, "source_only": False, "sentence_slicer": False, "fix_seed": False, "seed": 0, "total": 0, "current_index": 0,
            "completed": 0, "failed": 0, "pending": 0, "next_index": 0, "progress": 0.0,
            "current_text": "", "current_filename": "", "output_dir": "",
            "items": [], "recent_failed": [], "log": [], "error": "",
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


def validate_trainer_config(cfg: dict, *, use_rvc: bool = False) -> None:
    required = {
        "TTS Python": APP_DIR / "runtime" / "tts" / ".venv" / "Scripts" / "python.exe",
        "VITS trainer worker": VITS_TRAINER_WORKER_PATH,
        "Finnish VITS prepare": VITS_RESEARCH_DIR / "prepare_dataset.py",
        "Finnish VITS trainer": VITS_RESEARCH_DIR / "train_vits.py",
    }
    if use_rvc:
        required.update({"RVC Python": Path(cfg["rvc_python"]), "RVC repo": Path(cfg["rvc_repo"])})
    missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Required training files/folders were not found:\n" + "\n".join(missing))


def dataset_audio_count(path: Path) -> int:
    """Count same-stem WAV+TXT training pairs, ignoring unrelated preview audio."""
    if not path.exists() or not path.is_dir():
        return 0
    wav_stems = set()
    txt_stems = set()
    for p in path.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext == ".wav":
            wav_stems.add(p.stem)
        elif ext == ".txt":
            txt_stems.add(p.stem)
    return len(wav_stems & txt_stems)


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
        if len(training_state["log"]) > 500:
            training_state["log"] = training_state["log"][-500:]
        engine = training_state.get("engine", "")
        if engine == "rvc":
            _parse_rvc_training_line(line)
            return
        if engine == "vits_v2":
            _parse_v2_training_line(line)
            return
        phase_match = re.search(r"\[PHASE\]\s+(RVC_PREPROCESS|PREPARE|TRAIN)", line)
        if phase_match:
            phase = phase_match.group(1)
            info = {
                "RVC_PREPROCESS": ("rvc_preprocess", "RVC preprocessing", 2.0),
                "PREPARE": ("prepare", "Preparing VITS dataset", 25.0),
                "TRAIN": ("train", "VITS training", 35.0),
            }[phase]
            training_state["phase"], training_state["phase_label"] = info[0], info[1]
            training_state["progress"] = max(float(training_state.get("progress", 0)), info[2])
        # Finnish VITS prints: step 40000/60000 total=...
        m = re.search(r"\bstep\s+(\d+)\s*/\s*(\d+)", line, re.IGNORECASE)
        if m:
            step, total = int(m.group(1)), max(1, int(m.group(2)))
            training_state["step"] = step
            training_state["steps"] = total
            training_state["progress"] = 35.0 + 64.0 * min(1.0, step / total)
        p = re.search(r"(?:Progress|進捗):\s*(\d+)\s*/\s*(\d+)", line, re.IGNORECASE)
        if p and training_state.get("phase") == "rvc_preprocess":
            cur, total = int(p.group(1)), max(1, int(p.group(2)))
            training_state["progress"] = 2.0 + 22.0 * min(1.0, cur / total)
        if "TRAINING COMPLETE" in line:
            training_state["phase"] = "complete"
            training_state["phase_label"] = "Complete"
            training_state["progress"] = 100.0
        if line.startswith("CHECKPOINT :"):
            training_state["result"]["checkpoint"] = line.split(":", 1)[1].strip()
        elif line.startswith("DATASET    :"):
            training_state["result"]["dataset"] = line.split(":", 1)[1].strip()
        elif line.startswith("RVC DATA   :"):
            training_state["result"]["rvc_dataset"] = line.split(":", 1)[1].strip()


def _parse_rvc_training_line(line: str) -> None:
    """v1 RVC trainer (trainer_worker.py) log -> progress.  Caller holds training_state_lock."""
    phase_match = re.search(r"\[PHASE\]\s+(PREPROCESS|F0|HUBERT|TRAIN|INDEX)", line)
    if phase_match:
        info = {
            "PREPROCESS": ("preprocess", "Audio preprocessing", 2.0),
            "F0": ("f0", "RMVPE / F0 extraction", 14.0),
            "HUBERT": ("hubert", "HuBERT feature extraction", 24.0),
            "TRAIN": ("train", "RVC training", 34.0),
            "INDEX": ("index", "Building FAISS index", 94.0),
        }[phase_match.group(1)]
        training_state["phase"], training_state["phase_label"] = info[0], info[1]
        training_state["progress"] = max(float(training_state.get("progress", 0)), info[2])
    progress_match = re.search(r"(?:Progress|進捗):\s*(\d+)\s*/\s*(\d+)", line, re.IGNORECASE)
    if progress_match:
        ratio = min(1.0, int(progress_match.group(1)) / max(1, int(progress_match.group(2))))
        bands = {"preprocess": (2.0, 14.0), "f0": (14.0, 24.0), "hubert": (24.0, 34.0), "index": (94.0, 99.0)}
        if training_state["phase"] in bands:
            lo, hi = bands[training_state["phase"]]
            training_state["progress"] = lo + (hi - lo) * ratio
    epoch_match = re.search(r"(?:Training Epoch|Training epoch|学習エポック):\s*(\d+)", line, re.IGNORECASE)
    if not epoch_match:
        epoch_match = re.search(r"====>\s*(?:Epoch|エポック):\s*(\d+)", line, re.IGNORECASE)
    if epoch_match:
        epoch = int(epoch_match.group(1))
        training_state["epoch"] = max(int(training_state.get("epoch", 0)), epoch)
        training_state["step"] = training_state["epoch"]
        total = max(1, int(training_state.get("epochs", 1)))
        training_state["progress"] = 34.0 + 60.0 * min(1.0, epoch / total)
    if "TRAINING COMPLETE" in line:
        training_state["phase"], training_state["phase_label"], training_state["progress"] = "complete", "Complete", 100.0
    if line.startswith("PTH   :"):
        training_state["result"]["pth"] = line.split(":", 1)[1].strip()
    elif line.startswith("INDEX :"):
        training_state["result"]["index"] = line.split(":", 1)[1].strip()


def _parse_v2_training_line(line: str) -> None:
    """v2 VITS trainer (emotion_trainer_worker.py) log -> progress.  Caller holds training_state_lock."""
    m = re.match(r"\[V2PHASE\]\s*(.+)", line)
    if m:
        training_state["phase"], training_state["phase_label"] = "train", m.group(1).strip()
    m = re.match(r"\[V2PROGRESS\]\s*([\d.]+)", line)
    if m:
        training_state["progress"] = max(0.5, min(100.0, float(m.group(1))))
    m = re.search(r"\bstep\s+(\d+)\s*/\s*(\d+)", line, re.IGNORECASE)
    if m:
        training_state["step"], training_state["steps"] = int(m.group(1)), int(m.group(2))
    if line.startswith("RESULT_JSON :"):
        try:
            training_state["result"].update(json.loads(line.split(":", 1)[1].strip()))
        except ValueError:
            pass
    if "TRAINING COMPLETE" in line:
        training_state["phase"], training_state["phase_label"], training_state["progress"] = "complete", "Complete", 100.0


def training_runner(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    global training_process
    prevent_sleep(True)
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
            cp = str(snap.get("result", {}).get("checkpoint") or "")
            if cp:
                try:
                    cfg = load_config(); cfg["vits_checkpoint"] = cp; save_config(cfg)
                except Exception:
                    pass
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
                error=f"Trainer exited with code {code}\n\n{tail}",
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
        prevent_sleep(False)



def style_snapshot() -> dict:
    with style_state_lock:
        snap = dict(style_state)
        snap["log"] = list(style_state["log"])
        snap["result"] = dict(style_state["result"])
        return snap


def set_style_state(**updates) -> None:
    with style_state_lock:
        style_state.update(updates)


def append_style_log(line: str) -> None:
    line = line.rstrip("\r\n")
    if not line:
        return
    with style_state_lock:
        style_state["log"].append(line)
        if len(style_state["log"]) > 500:
            style_state["log"] = style_state["log"][-500:]
        m = re.search(r"Progress:\s*(\d+)\s*/\s*(\d+)", line, re.IGNORECASE)
        if m:
            cur, total = int(m.group(1)), max(1, int(m.group(2)))
            style_state["current"] = cur
            style_state["total"] = total
            style_state["progress"] = min(99.0, 5.0 + 93.0 * (cur / total))
            style_state["phase"] = "extract"
            style_state["phase_label"] = "Extracting paired acoustic style"
        elif line.startswith("[MODEL]"):
            style_state["phase"] = "model"
            style_state["phase_label"] = "Loading VITS acoustic model"
            style_state["progress"] = max(float(style_state.get("progress", 0.0)), 3.0)
        elif line.startswith("[PAIR]"):
            style_state["phase"] = "pair"
            style_state["phase_label"] = "Checking paired WAVs"
            style_state["progress"] = max(float(style_state.get("progress", 0.0)), 1.0)
        elif line.startswith("PROFILE_PT :"):
            style_state["result"]["profile"] = line.split(":", 1)[1].strip()
        elif line.startswith("PROFILE_JSON :"):
            style_state["result"]["profile_json"] = line.split(":", 1)[1].strip()
        elif "STYLE PROFILE COMPLETE" in line:
            style_state["phase"] = "complete"
            style_state["phase_label"] = "Complete"
            style_state["progress"] = 100.0


def style_profile_runner(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    global style_process
    prevent_sleep(True)
    try:
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform.startswith("win") else 0
        proc = subprocess.Popen(
            cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1, creationflags=creationflags,
        )
        with style_process_lock:
            style_process = proc
        assert proc.stdout is not None
        for line in proc.stdout:
            append_style_log(line)
        code = proc.wait()
        snap = style_snapshot()
        if code == 0 and snap.get("phase") == "complete" and snap.get("result", {}).get("profile"):
            profile = snap["result"]["profile"]
            set_style_state(running=False, done=True, stopping=False, progress=100.0)
            try:
                cfg = load_config()
                cfg.setdefault("style_lab", {})["profile"] = profile
                save_config(cfg)
            except Exception:
                pass
        elif snap.get("stopping"):
            append_style_log("[STOPPED] Style extraction was stopped by the user.")
            set_style_state(running=False, done=False, stopping=False, phase="stopped", phase_label="Stopped", error="")
        else:
            tail = "\n".join(snap.get("log", [])[-35:])
            set_style_state(
                running=False, done=False, stopping=False, phase="error", phase_label="Style extraction failed",
                error=f"style extractor exited with code {code}\n\n{tail}",
            )
    except Exception as exc:
        append_style_log(f"[ERROR] {exc}")
        set_style_state(
            running=False, done=False, stopping=False, phase="error", phase_label="Style extraction failed",
            error=f"{exc}\n\n{traceback.format_exc(limit=5)}",
        )
    finally:
        with style_process_lock:
            style_process = None
        prevent_sleep(False)


def _style_pair_key(path: Path) -> str | None:
    """Pair Style-Lab WAVs by the trailing 3-digit sample ID plus its suffix.

    Digits that belong to the style/prefix are intentionally ignored, e.g.
      emoNormal002.wav       -> 002
      emoAma002.wav          -> 002
      emoSasa1002.wav        -> 002
      emoSasa2002.wav        -> 002
      emoNormal090 (1).wav   -> 090 (1)
      emoSasa1090 (1).wav    -> 090 (1)

    The sample ID is expected to be zero-padded to 3 digits. Everything after
    those 3 digits is preserved as part of the key, so 090 and 090 (1) remain
    distinct pairs. Files without such a trailing 3-digit ID are ignored.
    """
    # Choose the *last* possible 3-digit run whose following character is not
    # another digit (or which is at end of stem). This lets prefixes such as
    # "Sasa1"/"Sasa2" contain digits without becoming part of the pair key.
    matches = list(re.finditer(r"(?<!\d)(\d{3})(?=\D|$)|(?<=\d)(\d{3})(?=\D|$)", path.stem))
    if not matches:
        return None
    m = matches[-1]
    sample_id = m.group(1) or m.group(2)
    suffix = path.stem[m.end():]
    return (sample_id + suffix).strip()


def _unique_wavs_by_style_key(folder: Path) -> tuple[dict[str, Path], list[str], list[str]]:
    found: dict[str, Path] = {}
    duplicates: list[str] = []
    ignored: list[str] = []
    for p in sorted(folder.rglob("*.wav")):
        key = _style_pair_key(p)
        if key is None:
            ignored.append(p.name)
            continue
        if key in found:
            duplicates.append(key)
        else:
            found[key] = p
    return found, sorted(set(duplicates)), ignored


def _parse_style_sources(data: dict) -> list[tuple[str, Path]]:
    raw = data.get("styles") or []
    if not isinstance(raw, list):
        raise ValueError("styles must be a list")
    out: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        path_s = str(item.get("path") or "").strip()
        if not name and not path_s:
            continue
        if not name or name.lower() == "neutral":
            raise ValueError("Each added style needs a unique name other than neutral.")
        if name in seen:
            raise ValueError(f"Duplicate style name: {name}")
        folder = Path(path_s).expanduser()
        if not folder.is_dir():
            raise FileNotFoundError(f"Style folder not found ({name}): {folder}")
        seen.add(name)
        out.append((name, folder.resolve()))
    if not out:
        raise ValueError("Add at least one style WAV folder.")
    return out


def style_pair_summary(neutral_dir: Path, styles: list[tuple[str, Path]]) -> dict:
    neutral, neutral_dups, neutral_ignored = _unique_wavs_by_style_key(neutral_dir)
    if neutral_dups:
        raise ValueError(
            f"Neutral folder has duplicate Style pair keys, e.g. {neutral_dups[0]!r}. "
            "Pair keys are the trailing 3-digit sample ID plus everything after it."
        )
    if not neutral:
        raise ValueError("No pairable WAV files found in the Neutral folder (filenames need a number).")
    result = {
        "neutral_count": len(neutral), "styles": {}, "all_common": None,
        "pair_rule": "trailing 3-digit sample ID + suffix", "neutral_ignored": len(neutral_ignored),
    }
    all_common = set(neutral)
    for name, folder in styles:
        wavs, dups, ignored = _unique_wavs_by_style_key(folder)
        if dups:
            raise ValueError(
                f"Style {name!r} has duplicate Style pair keys, e.g. {dups[0]!r}. "
                "Rename one of the files so sample-ID+suffix is unique within that folder."
            )
        common = set(neutral) & set(wavs)
        all_common &= set(wavs)
        result["styles"][name] = {
            "count": len(wavs), "matched": len(common),
            "missing_from_style": len(set(neutral) - set(wavs)),
            "extra_in_style": len(set(wavs) - set(neutral)),
            "ignored_without_number": len(ignored),
        }
    result["all_common"] = len(all_common)
    return result


@app.post("/api/style/pairs")
def style_pairs():
    try:
        data = request.get_json(force=True) or {}
        neutral = Path(str(data.get("neutral_dir") or "").strip()).expanduser()
        if not neutral.is_dir():
            return jsonify({"ok": False, "error": "Neutral WAV folder was not found."}), 400
        styles = _parse_style_sources(data)
        return jsonify({"ok": True, "summary": style_pair_summary(neutral.resolve(), styles)})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.post("/api/style/build")
def style_build():
    with training_state_lock:
        if training_state.get("running"):
            return jsonify({"ok": False, "error": "Model training is running."}), 409
    with batch_state_lock:
        if batch_state.get("running"):
            return jsonify({"ok": False, "error": "Batch generation is running."}), 409
    if generation_lock.locked():
        return jsonify({"ok": False, "error": "Audio generation is running."}), 409
    with style_state_lock:
        if style_state.get("running"):
            return jsonify({"ok": False, "error": "Style extraction is already running."}), 409
    try:
        data = request.get_json(force=True) or {}
        checkpoint = validate_vits_checkpoint(str(data.get("checkpoint") or ""))
        neutral = Path(str(data.get("neutral_dir") or "").strip()).expanduser().resolve()
        if not neutral.is_dir():
            return jsonify({"ok": False, "error": "Neutral WAV folder was not found."}), 400
        styles = _parse_style_sources(data)
        summary = style_pair_summary(neutral, styles)
        if any(int(info.get("matched", 0)) < 3 for info in summary["styles"].values()):
            return jsonify({"ok": False, "error": "Every style needs at least 3 Neutral/Style WAV pairs matched by trailing 3-digit sample ID + suffix."}), 400
        profile_name = safe_output_stem(str(data.get("profile_name") or "paired_style").strip(), fallback="paired_style")
        profile_dir = OUTPUT_DIR / "style_profiles" / profile_name
        profile_dir.mkdir(parents=True, exist_ok=True)
        profile = profile_dir / "style_profile.pt"
        if not STYLE_PROFILE_EXTRACTOR_PATH.is_file():
            return jsonify({"ok": False, "error": f"Style extractor is missing: {STYLE_PROFILE_EXTRACTOR_PATH}"}), 500

        cfg = load_config()
        cfg["style_lab"] = {
            "checkpoint": str(checkpoint), "profile": str(profile), "neutral_dir": str(neutral),
            "styles": [{"name": n, "path": str(f)} for n, f in styles],
            "strength": float(cfg.get("style_lab", {}).get("strength", 1.0)),
            "noise_scale": float(cfg.get("style_lab", {}).get("noise_scale", 0.4)),
            "length_scale": float(cfg.get("style_lab", {}).get("length_scale", 1.0)),
        }
        save_config(cfg)
        _stop_persistent_vits_worker()
        with style_state_lock:
            style_state.update({
                "running": True, "done": False, "stopping": False, "phase": "starting",
                "phase_label": "Starting paired style extraction", "progress": 0.5,
                "current": 0, "total": 0, "log": [], "error": "", "result": {},
            })
        cmd = [
            sys.executable, "-u", str(STYLE_PROFILE_EXTRACTOR_PATH),
            "--root", str(APP_DIR), "--checkpoint", str(checkpoint),
            "--neutral-dir", str(neutral), "--out", str(profile),
        ]
        for name, folder in styles:
            cmd += ["--style", f"{name}={folder}"]
        env = os.environ.copy(); env["PYTHONUNBUFFERED"] = "1"
        threading.Thread(target=style_profile_runner, args=(cmd, APP_DIR, env), daemon=True).start()
        return jsonify({"ok": True, "profile": str(profile), "summary": summary})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.get("/api/style/status")
def style_status():
    return jsonify(style_snapshot())


@app.post("/api/style/stop")
def style_stop():
    with style_state_lock:
        if not style_state.get("running"):
            return jsonify({"ok": False, "error": "Style extraction is not running."}), 400
        style_state["stopping"] = True
        style_state["phase_label"] = "Stopping…"
    with style_process_lock:
        proc = style_process
    if proc and proc.poll() is None:
        _kill_process_tree(proc)
    return jsonify({"ok": True})


@app.post("/api/style/profile-info")
def style_profile_info():
    try:
        data = request.get_json(force=True) or {}
        profile = Path(str(data.get("profile") or "").strip()).expanduser().resolve()
        meta = profile.with_suffix(".json")
        if not profile.is_file():
            return jsonify({"ok": False, "error": "Style profile .pt was not found."}), 400
        if not meta.is_file():
            return jsonify({"ok": False, "error": "Style profile metadata .json was not found."}), 400
        info = json.loads(meta.read_text(encoding="utf-8"))
        styles = list((info.get("styles") or {}).keys())
        return jsonify({"ok": True, "profile": str(profile), "styles": styles, "info": info})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.post("/api/style/generate")
def style_generate():
    with training_state_lock:
        if training_state.get("running"):
            return jsonify({"ok": False, "error": "Model training is running."}), 409
    with style_state_lock:
        if style_state.get("running"):
            return jsonify({"ok": False, "error": "Style profile extraction is running."}), 409
    with batch_state_lock:
        if batch_state.get("running"):
            return jsonify({"ok": False, "error": "Batch generation is running."}), 409
    if not generation_lock.acquire(blocking=False):
        return jsonify({"ok": False, "error": "Another generation job is running."}), 409
    try:
        data = request.get_json(force=True) or {}
        text = str(data.get("text") or "").strip()
        if not text:
            return jsonify({"ok": False, "error": "Enter Finnish test text."}), 400
        checkpoint = validate_vits_checkpoint(str(data.get("checkpoint") or ""))
        profile = Path(str(data.get("profile") or "").strip()).expanduser().resolve()
        if not profile.is_file():
            return jsonify({"ok": False, "error": "Select a built style_profile.pt."}), 400
        style_name = str(data.get("style") or "").strip()
        kind = str(data.get("kind") or "styled").strip().lower()
        if kind not in {"neutral", "styled"}:
            return jsonify({"ok": False, "error": "kind must be neutral or styled"}), 400
        strength = 0.0 if kind == "neutral" else max(0.0, min(2.0, float(data.get("strength", 1.0))))
        noise_scale = max(0.0, min(1.5, float(data.get("noise_scale", 0.4))))
        length_scale = max(0.2, min(3.0, float(data.get("length_scale", 1.0))))
        seed = int(data.get("seed", 123456))
        if seed < 0 or seed > 2147483647:
            raise ValueError("Seed must be between 0 and 2147483647")
        out = STYLE_NEUTRAL_WAV_PATH if kind == "neutral" else STYLE_STYLED_WAV_PATH
        result = generate_vits_file(
            text=text, output=out, checkpoint=str(checkpoint), noise_scale=noise_scale,
            length_scale=length_scale, seed=seed, batch=False, style_profile=str(profile),
            style_name=style_name, style_strength=strength,
            apply_style_gain=bool(data.get("apply_gain", True)),
        )
        cfg = load_config()
        lab = cfg.setdefault("style_lab", {})
        lab.update({"checkpoint": str(checkpoint), "profile": str(profile), "strength": strength,
                    "noise_scale": noise_scale, "length_scale": length_scale})
        save_config(cfg)
        return jsonify({
            "ok": True, "audio_url": f"/api/style/audio/{kind}", "kind": kind,
            "style": result.get("style", style_name), "style_strength": strength,
            "effective_length_scale": result.get("effective_length_scale", length_scale),
            "warning": result.get("warning", ""), "seed": seed,
        })
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc), "trace": traceback.format_exc(limit=5)}), 500
    finally:
        generation_lock.release()


@app.get("/api/style/audio/<kind>")
def style_audio(kind: str):
    path = STYLE_NEUTRAL_WAV_PATH if kind == "neutral" else STYLE_STYLED_WAV_PATH if kind == "styled" else None
    if path is None or not path.is_file():
        return jsonify({"ok": False, "error": "Style preview is not available."}), 404
    return send_from_directory(path.parent, path.name, mimetype="audio/wav", as_attachment=False, conditional=True)


@app.post("/api/style/open-output")
def style_open_output():
    snap = style_snapshot()
    profile = str(snap.get("result", {}).get("profile") or "")
    if not profile:
        try:
            profile = str(load_config().get("style_lab", {}).get("profile") or "")
        except Exception:
            profile = ""
    folder = Path(profile).expanduser().parent if profile else (OUTPUT_DIR / "style_profiles")
    folder.mkdir(parents=True, exist_ok=True)
    if sys.platform.startswith("win"):
        os.startfile(folder)
        return jsonify({"ok": True, "folder": str(folder)})
    return jsonify({"ok": False, "error": str(folder)}), 400


@app.post("/api/dataset-info")
def dataset_info():
    data = request.get_json(force=True) or {}
    path = Path(str(data.get("path", "")).strip()).expanduser()
    if not path.exists() or not path.is_dir():
        return jsonify({"ok": False, "error_code": "folder_not_found", "error": "Folder not found.", "count": 0}), 400
    kind = str(data.get("kind", "pairs")).lower()
    count = rvc_audio_count(path) if kind == "rvc" else dataset_audio_count(path)
    return jsonify({"ok": True, "path": str(path.resolve()), "count": count})


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


@app.post("/api/pick-file")
def pick_file():
    if not sys.platform.startswith("win"):
        return jsonify({"ok": False, "error": "File picker is currently Windows-only."}), 400
    data = request.get_json(silent=True) or {}
    title = str(data.get("title") or "Select file").replace("'", "''")
    kind = str(data.get("kind") or "all").lower()
    filters = {
        "pt": "PyTorch checkpoint (*.pt)|*.pt|All files (*.*)|*.*",
        "pth": "RVC model (*.pth)|*.pth|All files (*.*)|*.*",
        "index": "RVC index (*.index)|*.index|All files (*.*)|*.*",
        "wav": "WAV audio (*.wav)|*.wav|All files (*.*)|*.*",
    }
    filt = filters.get(kind, "All files (*.*)|*.*").replace("'", "''")
    initial = str(data.get("initial") or "").strip()
    initial_dir = str(Path(initial).expanduser().parent) if initial else ""
    initial_dir = initial_dir.replace("'", "''")
    result_file = TEMP_DIR / f"pick_file_{uuid.uuid4().hex}.txt"
    env = os.environ.copy(); env["ZUNDANEN_PICK_RESULT"] = str(result_file)
    ps = "Add-Type -AssemblyName System.Windows.Forms; $d=New-Object System.Windows.Forms.OpenFileDialog;"
    ps += f"$d.Title='{title}';$d.Filter='{filt}';$d.Multiselect=$false;"
    if initial_dir:
        ps += f"if(Test-Path '{initial_dir}'){{$d.InitialDirectory='{initial_dir}'}};"
    ps += "if($d.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK){[System.IO.File]::WriteAllText($env:ZUNDANEN_PICK_RESULT,$d.FileName,(New-Object System.Text.UTF8Encoding($false)))}"
    try:
        proc = subprocess.run(["powershell.exe", "-NoProfile", "-STA", "-Command", ps], env=env,
                              stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                              encoding="utf-8", errors="replace",
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or "File picker failed")
        chosen = result_file.read_text(encoding="utf-8").strip() if result_file.exists() else ""
        return jsonify({"ok": True, "path": chosen})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500
    finally:
        try: result_file.unlink(missing_ok=True)
        except OSError: pass


def rvc_audio_count(path: Path) -> int:
    exts = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}
    if not path.exists() or not path.is_dir():
        return 0
    return sum(1 for p in path.iterdir() if p.is_file() and p.suffix.lower() in exts)


V2_TRAINER_WORKER_PATH = APP_DIR / "emotion_trainer_worker.py"
V2_BRIDGE_RECOMMENDED = 0.25
V2_NEW_MODEL = "__new__"   # Base model value of "New model (from scratch)" in Joint training


def _training_busy_response():
    with style_state_lock:
        if style_state.get("running"):
            return jsonify({"ok": False, "error": "Style profile extraction is running."}), 409
    with batch_state_lock:
        if batch_state.get("running"):
            return jsonify({"ok": False, "error": "Batch generation is running."}), 409
    if generation_lock.locked():
        return jsonify({"ok": False, "error": "Audio generation is running."}), 409
    with training_state_lock:
        if training_state["running"]:
            return jsonify({"ok": False, "error": "Model training is already running."}), 409
    return None


def _begin_training(engine: str, label: str, dataset: str, count: int, steps: int, cmd: list[str], env: dict) -> None:
    _stop_persistent_vits_worker()
    _stop_persistent_source_tts_worker()
    with training_state_lock:
        training_state.update({
            "running": True, "done": False, "stopping": False, "engine": engine, "character": label,
            "dataset": dataset, "phase": "starting", "phase_label": "Starting trainer",
            "step": 0, "steps": steps, "epoch": 0, "epochs": steps, "progress": 0.5, "audio_files": count,
            "log": [], "error": "", "result": {}, "pid": None,
        })
    threading.Thread(target=training_runner, args=(cmd, APP_DIR, env), daemon=True).start()


@app.post("/api/trainer/start")
def trainer_start():
    busy = _training_busy_response()
    if busy:
        return busy
    data = request.get_json(force=True) or {}
    engine = str(data.get("engine", "vits")).lower()
    cfg = load_config()

    if engine == "rvc":
        # Same as Zundanen v1: trainer_worker.py inside the RVC Python.
        character = str(data.get("character", "")).strip()
        dataset = Path(str(data.get("dataset", "")).strip()).expanduser()
        if not character:
            return jsonify({"ok": False, "error": "Enter a Character name."}), 400
        if character in {".", ".."} or any(c in character for c in '<>:"/\\|?*'):
            return jsonify({"ok": False, "error": "Character name contains characters that are not allowed on Windows."}), 400
        if not dataset.exists() or not dataset.is_dir():
            return jsonify({"ok": False, "error": "Dataset folder was not found."}), 400
        count = rvc_audio_count(dataset)
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
        missing = [f"{n}: {p}" for n, p in {"RVC Python": Path(cfg["rvc_python"]), "RVC repo": Path(cfg["rvc_repo"]),
                                            "RVC exports": Path(cfg["exports_root"]),
                                            "trainer_worker.py": APP_DIR / "trainer_worker.py"}.items() if not p.exists()]
        if missing:
            return jsonify({"ok": False, "error": "Required training files/folders were not found:\n" + "\n".join(missing)}), 400
        cfg["rvc_trainer"] = {"epochs": epochs, "batch_size": batch_size, "save_every": save_every,
                              "workers": workers, "gpu": gpu, "fresh": fresh}
        save_config(cfg)
        cmd = [cfg["rvc_python"], "-u", str(APP_DIR / "trainer_worker.py"), character,
               "--dataset", str(dataset.resolve()), "--rvc-repo", cfg["rvc_repo"], "--exports-root", cfg["exports_root"],
               "--epochs", str(epochs), "--batch-size", str(batch_size), "--save-every", str(save_every),
               "--workers", str(workers), "--gpu", gpu]
        if fresh:
            cmd.append("--fresh")
        env = os.environ.copy()
        env.update({"PYTHONUNBUFFERED": "1", "LANG": "en_US.UTF-8", "LC_ALL": "en_US.UTF-8",
                    "LANGUAGE": "en_US", "RVC_LANGUAGE": "en_US"})
        _begin_training("rvc", character, str(dataset.resolve()), count, epochs, cmd, env)
        return jsonify({"ok": True, "audio_files": count})

    # ---- VITS (v2 emotion model)
    mode = str(data.get("mode", "joint")).lower()
    if mode not in {"joint", "finnish", "style"}:
        return jsonify({"ok": False, "error": "VITS mode must be joint, finnish or style."}), 400
    run_name = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(data.get("run_name", "")).strip()).strip("_")[:40]
    if not run_name:
        return jsonify({"ok": False, "error": "Enter a Model name."}), 400
    if not V2_TRAINER_WORKER_PATH.exists():
        return jsonify({"ok": False, "error": f"Missing {V2_TRAINER_WORKER_PATH}"}), 400
    try:
        epochs = max(0.01, min(50.0, float(data.get("epochs", 4.5))))
        branch_epochs = max(1.0, min(2000.0, float(data.get("branch_epochs", 80))))
        lr = float(data.get("lr", 1e-4))
        batch_size = max(1, min(64, int(data.get("batch_size", 10))))
        workers = max(0, min(16, int(data.get("workers", min(4, os.cpu_count() or 1)))))
        bridge = max(0.0, min(0.6, float(data.get("bridge_ratio", V2_BRIDGE_RECOMMENDED))))
        data_ratio = max(0.05, min(0.6, float(data.get("data_ratio", 0.25))))
        if not (0 < lr <= 0.01):
            raise ValueError("lr")
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "One or more training settings contain an invalid number."}), 400
    import emotion_engine
    speakers = ("zundamon", "metan", "itako")
    character = str(data.get("character", "zundamon")).lower()
    known = emotion_engine.characters()
    scratch = mode == "joint" and str(data.get("base_model", "")) == V2_NEW_MODEL
    if mode in {"joint", "finnish"}:
        if run_name.lower() in {k.lower() for k in known}:
            return jsonify({"ok": False, "error": f"A character named '{run_name}' already exists. Choose another model name."}), 400
        base_model = None if scratch else emotion_engine.find_character(str(data.get("base_model", "")) or next(iter(known)))
        if base_model is None and not scratch:
            return jsonify({"ok": False, "error": "Select an existing base model."}), 400
    if scratch:   # v2: new model from scratch (Finnish + emotion folders + RVC bridge), lr as Finnish VITS
        import dataset_scan
        fdir = Path(str(data.get("finnish_dir", "")).strip()).expanduser()
        rows = [r for r in (data.get("emotion_rows") or []) if isinstance(r, dict)]
        pairs = dataset_scan.finnish_pairs(fdir)
        if pairs < 50:
            return jsonify({"ok": False, "error": f"Finnish folder: need at least 50 WAV + same-name TXT pairs (found {pairs})."}), 400
        scan = dataset_scan.scan_rows(rows)
        if scan["problems"] or not scan["speakers"]:
            return jsonify({"ok": False, "error": "Emotion data: " + ("; ".join(scan["problems"]) or "no usable WAV + LAB recordings.")}), 400
        rvc_pth = Path(str(data.get("rvc_pth", "")).strip()).expanduser()
        if rvc_pth.suffix.lower() != ".pth" or not rvc_pth.is_file():
            return jsonify({"ok": False, "error": "Choose the RVC model file (.pth) for the bridge."}), 400
        target = dataset_scan.speaker_id(str(data.get("target_speaker", "")))
        if target not in scan["speakers"]:   # the RVC voice must be one of the emotion speakers (styles and pitch come from them)
            return jsonify({"ok": False, "error": "Choose the Character (one of the emotion speakers) the RVC model belongs to."}), 400
        rvc_index = dataset_scan.find_index(rvc_pth)
        spec = TEMP_DIR / f"v2_emotion_{run_name}_{uuid.uuid4().hex[:8]}.json"
        spec.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        mode = "scratch"
    run_lr = 2e-4 if mode == "scratch" else lr   # a model from random weights uses Finnish VITS's learning rate
    cmd = [sys.executable, "-u", str(V2_TRAINER_WORKER_PATH), mode, "--run-name", run_name,
           "--epochs", f"{epochs:g}", "--branch-epochs", f"{branch_epochs:g}",
           "--batch-size", str(batch_size), "--workers", str(workers),
           "--lr", str(run_lr), "--bridge-ratio", f"{bridge:.3f}"]
    if mode in {"joint", "finnish"}:
        cmd += ["--base-model", base_model]
    dataset_label, count = "", 0
    if mode == "scratch":
        cmd += ["--finnish-dir", str(fdir.resolve()), "--emotion-spec", str(spec), "--rvc-pth", str(rvc_pth.resolve()),
                "--rvc-index", rvc_index, "--target-speaker", target]
        dataset_label, count = str(fdir.resolve()), pairs
    if mode == "finnish":
        dataset = Path(str(data.get("dataset", "")).strip()).expanduser()
        if not dataset.is_dir():
            return jsonify({"ok": False, "error": "Dataset folder was not found."}), 400
        count = dataset_audio_count(dataset)
        if count < 20:
            return jsonify({"ok": False, "error": f"Need at least 20 WAV + same-name TXT pairs (found {count})."}), 400
        if character not in speakers:
            return jsonify({"ok": False, "error": f"Speaker must be one of {list(speakers)}."}), 400
        cmd += ["--dataset", str(dataset.resolve()), "--character", character, "--data-ratio", f"{data_ratio:.3f}"]
        dataset_label = str(dataset.resolve())
    elif mode == "style":
        name = str(data.get("style_name", "")).strip().lower()
        if not re.fullmatch(r"[a-z][a-z0-9_]{1,23}", name):
            return jsonify({"ok": False, "error": "Style name: 2-24 characters a-z, 0-9, _ (start with a letter)."}), 400
        if name in {"normal", "ama", "sexy", "tsun"} or name in (emotion_engine.preset().get("extra_styles") or {}):
            return jsonify({"ok": False, "error": f"Style '{name}' already exists."}), 400
        target = emotion_engine.find_character(str(data.get("model", "")) or next(iter(known)))
        if target is None:
            return jsonify({"ok": False, "error": "Select the model the style is for."}), 400
        nd = Path(str(data.get("normal_dir", "")).strip()).expanduser()
        sd = Path(str(data.get("style_dir", "")).strip()).expanduser()
        if not nd.is_dir() or not sd.is_dir():
            return jsonify({"ok": False, "error": "Normal folder or style folder was not found."}), 400
        count = sum(1 for _ in sd.glob("*.wav"))
        cmd += ["--style-name", name, "--label", str(data.get("style_label", "") or name), "--normal-dir", str(nd.resolve()),
                "--style-dir", str(sd.resolve()), "--model", target]
        dataset_label = str(sd.resolve())
    cfg["vits_v2_trainer"] = {"mode": "joint" if mode == "scratch" else mode, "epochs": epochs, "branch_epochs": branch_epochs,
                              "batch_size": batch_size, "workers": workers, "lr": lr, "bridge_ratio": bridge, "data_ratio": data_ratio}
    save_config(cfg)
    env = os.environ.copy()
    env.update({"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"})
    _begin_training("vits_v2", f"{mode}:{run_name}", dataset_label, count, 0, cmd, env)
    return jsonify({"ok": True, "audio_files": count})


@app.post("/api/v2/scan-emotion")
def v2_scan_emotion():
    """Per-row counts, speakers and problems of the emotion rows (folder, LAB folder, speaker, style) of the New model panel."""
    import dataset_scan
    data = request.get_json(force=True) or {}
    rows = [r for r in (data.get("rows") or []) if isinstance(r, dict)]
    res = dataset_scan.scan_rows(rows)
    out = {"ok": True, "speakers": res["speakers"], "problems": res["problems"], "items": len(res["items"]), "rows": res["rows"]}
    pth = str(data.get("rvc_pth", "")).strip()
    if pth:   # which emotion speaker the bridge RVC model belongs to (from its file name)
        out["rvc_speaker"] = dataset_scan.match_speaker(pth, res["speakers"])
        out["rvc_index"] = Path(dataset_scan.find_index(pth)).name if dataset_scan.find_index(pth) else ""
    return jsonify(out)


@app.post("/api/v2/scan-finnish")
def v2_scan_finnish():
    import dataset_scan
    data = request.get_json(force=True) or {}
    folder = Path(str(data.get("path", "")).strip()).expanduser()
    return jsonify({"ok": folder.is_dir(), "pairs": dataset_scan.finnish_pairs(folder) if folder.is_dir() else 0})


@app.get("/api/v2/options")
def v2_options():
    """Voice Generation (Character / styles / RVC voices) and trainer defaults."""
    import emotion_engine
    cfg = load_config()
    try:
        opts = emotion_engine.options()
        ok, msg = emotion_engine.available() if emotion_engine.characters() else (True, "")   # only Chatterbox installed
    except Exception as exc:
        opts, ok, msg = {"characters": [], "styles": []}, False, str(exc)
    return jsonify({"ok": True, "engine_ok": ok, "engine_error": msg, **opts,
                    "rvc_voices": sorted(scan_voices(cfg).keys(), key=str.lower),
                    "trainer_characters": ["zundamon", "metan", "itako"],
                    "bridge_recommended": V2_BRIDGE_RECOMMENDED,
                    "cpu_count": os.cpu_count() or 1,
                    "rvc_trainer": cfg.get("rvc_trainer", {}), "vits_v2_trainer": cfg.get("vits_v2_trainer", {})})


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
                )
            else:
                proc.terminate()
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500
    return jsonify({"ok": True})


@app.post("/api/open-exports")
def open_exports():
    snap = training_snapshot()
    cp = str(snap.get("result", {}).get("checkpoint") or "")
    folder = Path(cp).parent if cp else (OUTPUT_DIR / "trainer_cache")
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
        if save_source_audio and PREVIEW_TTS_WAV_PATH.exists() and source_mode not in {"imported_final", "generated_tts_source_only", "generated_vits"}:
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
        if save_source_audio and PREVIEW_TTS_WAV_PATH.exists() and source_mode not in {"imported_final", "generated_tts_source_only", "generated_vits"}:
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


from voice_conversion_routes import create_voice_conversion_blueprint

app.register_blueprint(create_voice_conversion_blueprint(
    app_dir=APP_DIR,
    load_config=load_config,
    scan_voices=scan_voices,
    run_rvc_on_audio=run_rvc_on_audio,
    generation_lock=generation_lock,
    rvc_env=_rvc_env,
    kill_process_tree=_kill_process_tree,
    choose_save_path=_choose_save_wav_path,
))


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
