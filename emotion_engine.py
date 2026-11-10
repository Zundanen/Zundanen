from __future__ import annotations

"""Process manager for emotion_worker.py (used by app.py).

Keeps one resident v65 worker; generate() sends a request and waits for the
result; stop() frees its VRAM (called whenever the app stops the normal VITS
worker, e.g. before training).  Original Zundanen code (MIT).
"""

import json
import os
import subprocess
import sys
import threading
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKER = ROOT / "emotion_worker.py"
READY = "ZUNDANEN_V65_READY\t"
RESULT = "ZUNDANEN_V65_RESULT\t"
STYLES = ("normal", "ama", "tsun", "sexy")   # display order

_lock = threading.Lock()
_proc = None
_req_id = 0


PRESET = ROOT / "emotion_tts/emotion_preset.json"
CHAR_DIR = ROOT / "outputs/v65/characters"   # model packages: one <name>.json per character, unpacked into the app folder
MODEL_DIR = ROOT / "models/characters"       # character files: one <Name>.pt per character (model + style branch + deltas + settings)
BUNDLE_META = "zundanen_character.json"      # settings stored inside the character file, read without torch


CHATTERBOX = "Chatterbox"   # v73: Finnish VITS 100k VITS character (see options())


def is_chatterbox(name) -> bool:
    return str(name or "").strip().lower() == CHATTERBOX.lower()


def preset() -> dict:
    return json.loads(PRESET.read_text(encoding="utf-8"))


def model_files(c: dict) -> list[Path]:
    """Files one character needs: base checkpoint, style branch (val.csv names the best step) and z-delta,
    or the one character file."""
    if c.get("bundle"):
        return [Path(c["bundle"])]
    out = ROOT / "outputs/v65"
    base = out / c["base_name"] / "checkpoints"
    return [next(iter(sorted(base.glob("v65_*.pt"))), base), out / c["branch_name"] / "val.csv", out / f"zdelta_{c['base_name']}.pt"]


def package_characters() -> dict:
    """Characters installed from a model package (outputs/v65/characters/<name>.json + the files it names).
    Unreadable entries and entries whose model files are missing are skipped, so a broken package never stops the app."""
    found = {}
    for p in sorted(CHAR_DIR.glob("*.json")):
        try:
            v = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(v, dict) and v.get("base_name") and v.get("branch_name") and all(f.exists() for f in model_files(v)):
            found[p.stem] = v
    for p in sorted(MODEL_DIR.glob("*.pt")):          # character files (models/characters/<Name>.pt)
        v = bundle_settings(p)
        if v is not None:
            found[p.stem] = {**v, "base_name": p.stem, "branch_name": p.stem, "bundle": str(p)}
    return found


def bundle_settings(p: Path) -> dict | None:
    """Settings of a character file (the .pt is a zip archive; the settings JSON is one entry of it)."""
    try:
        with zipfile.ZipFile(p) as z:   # <top folder>/zundanen_character.json (the top folder is torch's archive name)
            v = json.loads(z.read(next(n for n in z.namelist() if n.split("/")[-1] == BUNDLE_META)).decode("utf-8"))
    except (OSError, KeyError, ValueError, StopIteration, zipfile.BadZipFile):
        return None
    return v if isinstance(v, dict) and v.get("speaker") else None


def characters(pre: dict | None = None) -> dict:
    """Character registry: model name -> {label, speaker, base_name, branch_name, level_st, range_st}.
    Model packages first, then the preset (a preset entry with the same name wins)."""
    pre = pre or preset()
    out = {}
    for key, v in {**package_characters(), **(pre.get("characters") or {})}.items():
        out[key] = {"label": v.get("label") or key, "speaker": v.get("speaker") or key.lower(),
                    "base_name": v.get("base_name") or pre["base_name"], "branch_name": v.get("branch_name") or pre["branch_name"],
                    "level_st": float(v.get("level_st", pre.get("pitch_level_st", -6.66))),
                    "range_st": float(v.get("range_st", pre.get("pitch_range_st", 12.0)))}
        if isinstance(v.get("style_tempo"), dict):          # v2 new models: measured from their own recordings
            out[key]["style_tempo"] = {k: float(x) for k, x in v["style_tempo"].items()}
        if v.get("bundle"):
            out[key]["bundle"] = v["bundle"]
    if not out and "characters" not in pre:   # old presets without a registry; an explicit empty registry (release) stays empty
        out["Zundamon"] = {"label": "Zundamon", "speaker": "zundamon", "base_name": pre["base_name"],
                           "branch_name": pre["branch_name"], "level_st": float(pre.get("pitch_level_st", -6.66)),
                           "range_st": float(pre.get("pitch_range_st", 12.0))}
    return out


def find_character(name: str, pre: dict | None = None) -> str | None:
    chars = characters(pre)
    if name in chars:
        return name
    low = {k.lower(): k for k in chars}
    return low.get(str(name).lower())


def styles_for(base_name: str, pre: dict | None = None) -> list[dict]:
    pre = pre or preset()
    styles = [{"id": s, "label": s.capitalize(), "strength": float(pre["style_strength"].get(s, 0.0)), "builtin": True}
              for s in STYLES]
    for name, e in (pre.get("extra_styles") or {}).items():
        if e.get("base_name") == base_name:
            styles.append({"id": name, "label": e.get("label") or name, "strength": float(e.get("strength", 1.0)),
                           "builtin": False})
    return styles


def options() -> dict:
    """Characters (each with its styles) the GUI may offer; read from the preset, no worker needed."""
    pre = preset()
    chars = [{"id": k, **v, "styles": styles_for(v["base_name"], pre)} for k, v in characters(pre).items()]
    # v73: the Finnish source VITS (Finnish VITS 100k, Chatterbox voice), Normal only; app.py renders it with
    # the classic VITS worker and the optional RVC step turns it into Zundamon.  Not a v65 model.
    chars.append({"id": CHATTERBOX, "label": "Chatterbox", "engine": "vits", "speaker": "fi_chatterbox",
                  "level_st": 0.0, "range_st": 0.0, "rvc_pitch_hint": 8,
                  "styles": [{"id": "normal", "label": "Normal", "strength": 0.0, "builtin": True}]})
    return {"characters": chars, "styles": chars[0]["styles"], "base_name": pre["base_name"], "branch_name": pre["branch_name"],
            "range_k": float(pre.get("range_k", 1.0)), "z_strength": float(pre.get("z_strength", 0.5)),
            "noise_scale": float(pre.get("noise_scale", 0.4))}


def available() -> tuple[bool, str]:
    try:
        pre = preset()
    except Exception as exc:
        return False, f"v65 preset unreadable: {exc}"
    need = [WORKER, PRESET]
    for c in characters(pre).values():
        need += model_files(c)
    missing = [str(p) for p in need if not p.exists()]
    return (not missing, "" if not missing else "v65 files are missing:\n" + "\n".join(missing))


def _kill(proc):
    if proc is None:
        return
    try:
        if proc.poll() is None:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
            else:
                proc.kill()
    except Exception:
        pass


def stop() -> None:
    global _proc
    with _lock:
        proc, _proc = _proc, None
    _kill(proc)


def running() -> bool:
    return _proc is not None and _proc.poll() is None


def _ensure():
    global _proc
    if _proc is not None and _proc.poll() is None:
        return _proc
    ok, msg = available()
    if not ok:
        raise FileNotFoundError(msg)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.Popen([sys.executable, str(WORKER)], cwd=str(ROOT), stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                            errors="replace", bufsize=1, env=env,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0)
    log = []
    while True:
        line = proc.stdout.readline()
        if not line:
            _kill(proc)
            raise RuntimeError("v65 worker failed to start:\n" + "".join(log[-40:]))
        if line.startswith(READY):
            payload = json.loads(line[len(READY):])
            if not payload.get("ok"):
                _kill(proc)
                raise RuntimeError(payload.get("error") or "v65 worker failed to start.")
            break
        log.append(line)
    _proc = proc
    return proc


def generate(*, text: str, output: Path, style: str = "normal", strength: float | None = None, seed: int = 1234,
             noise_scale: float = 0.4, length_scale: float = 1.0, level: float | None = None,
             range_k: float | None = None, z: float | None = None, character: str = "zundamon") -> dict:
    global _req_id
    style = str(style).lower()
    pre = preset()
    key = find_character(character, pre)
    if key is None:
        raise ValueError(f"Unknown character: {character}")
    if style not in {s["id"] for s in styles_for(characters(pre)[key]["base_name"], pre)}:
        raise ValueError(f"Style {style!r} is not available for {key}")
    character = key
    with _lock:
        proc = _ensure()
        _req_id += 1
        req = {"id": _req_id, "text": str(text), "output": str(Path(output).resolve()), "style": style,
               "character": character,
               "seed": int(seed), "noise_scale": float(noise_scale), "length_scale": float(length_scale)}
        for k, v in (("strength", strength), ("level", level), ("range_k", range_k), ("z", z)):
            if v is not None:
                req[k] = float(v)
        try:
            proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
            proc.stdin.flush()
            log = []
            while True:
                line = proc.stdout.readline()
                if not line:
                    raise RuntimeError("v65 worker stopped unexpectedly.\n" + "".join(log[-40:]))
                if not line.startswith(RESULT):
                    log.append(line)
                    continue
                res = json.loads(line[len(RESULT):])
                if res.get("id") != _req_id:
                    continue
                if not res.get("ok"):
                    raise RuntimeError(res.get("error") or "v65 generation failed.")
                return res
        except (BrokenPipeError, OSError, RuntimeError):
            globals()["_proc"] = None
            _kill(proc)
            raise
