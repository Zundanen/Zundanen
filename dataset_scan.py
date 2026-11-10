"""Zundanen v2: scan an emotion dataset folder (stdlib only; used by app.py and emotion_trainer_worker.py).

Accepted layout (the same as the project's own dataset/ folder):

  <root>/<style>/<style>_wav/<Speaker>/<prefix><NNN>.wav      (also <style>_wab, or <root>/<style>/<Speaker>/)
  <root>/<style>/<style>_lab/<Speaker>/<prefix><NNN>.lab      speaker's own phoneme labels (OpenJTalk LAB)
  <root>/<style>/<style>_lab/<prefix><NNN>.lab                labels shared by all speakers
  (a .lab next to the .wav is also used)

<style> is normal / ama / sexy / tsun; NNN is the sentence number that pairs Normal
and style recordings.  A recording without its own LAB uses the shared LAB or
another speaker's LAB with the SAME FILE NAME (same corpus sentence); a file
whose name matches no LAB is skipped (e.g. recitation001.wav is never paired
with emoNormal001.lab).
"""
from __future__ import annotations

import re
from pathlib import Path

STYLES = ("normal", "ama", "sexy", "tsun")
_NUM = re.compile(r"(\d{1,4})$")


def sentence_id(path: Path) -> int | None:
    m = _NUM.search(path.stem)
    return int(m.group(1)) if m and " " not in path.stem else None


def _wav_root(root: Path, style: str) -> Path | None:
    for name in (f"{style}_wav", f"{style}_wab", ""):
        d = root / style / name if name else root / style
        if d.is_dir() and any(p.is_dir() and not p.name.endswith("_lab") for p in d.iterdir()):
            return d
    return None


def speaker_id(folder_name: str) -> str:
    s = re.sub(r"[^a-z0-9_]+", "_", folder_name.lower()).strip("_")
    return s or "speaker"


def scan(root) -> dict:
    """{'speakers': {id: {'folder': name, 'counts': {style: n}}}, 'items': [...], 'problems': [...]}"""
    root = Path(root)
    items, problems, speakers = [], [], {}
    if not root.is_dir():
        return {"speakers": {}, "items": [], "problems": [f"folder not found: {root}"]}
    labs: dict[tuple[str, str], list[Path]] = {}   # (style, file stem) -> candidate LABs (any speaker / shared)
    for style in STYLES:
        for lab in (root / style).rglob("*.lab") if (root / style).is_dir() else []:
            labs.setdefault((style, lab.stem.lower()), []).append(lab)
    for style in STYLES:
        wroot = _wav_root(root, style)
        if wroot is None:
            problems.append(f"no '{style}' recordings found")
            continue
        for spk_dir in sorted(p for p in wroot.iterdir() if p.is_dir() and not p.name.endswith("_lab")):
            spk = speaker_id(spk_dir.name)
            info = speakers.setdefault(spk, {"folder": spk_dir.name, "counts": {s: 0 for s in STYLES}})
            for wav in sorted(spk_dir.glob("*.wav")):
                sid = sentence_id(wav)
                if sid is None:
                    continue
                cands = labs.get((style, wav.stem.lower()), [])
                own = [c for c in cands if c.parent.name.lower() == spk_dir.name.lower() or c.parent == wav.parent]
                shared = [c for c in cands if c.parent.name.lower().endswith("_lab")]
                lab = (own or shared or cands or [None])[0]
                if lab is None:
                    info.setdefault("no_lab", 0)
                    info["no_lab"] += 1
                    continue
                kind = "speaker" if own else ("shared" if shared else "other_speaker")
                items.append({"speaker": spk, "style": style, "sentence_id": sid, "wav": str(wav), "lab": str(lab),
                              "lab_kind": kind})
                info["counts"][style] += 1
    speakers = {k: v for k, v in speakers.items() if sum(v["counts"].values())}
    for spk, info in speakers.items():
        if info["counts"]["normal"] < 10:
            problems.append(f"speaker '{spk}': fewer than 10 Normal recordings")
    return {"speakers": speakers, "items": items, "problems": problems}


def finnish_pairs(folder) -> int:
    """Number of WAV files with a same-name TXT (any depth)."""
    folder = Path(folder)
    if not folder.is_dir():
        return 0
    txt = {p.stem for p in folder.rglob("*.txt")}
    return sum(1 for p in folder.rglob("*.wav") if p.stem in txt)


def match_speaker(pth, speakers) -> str | None:
    """Emotion speaker an RVC model belongs to, from the .pth name: exact id, then the id the name starts
    with (Zundamon424.pth -> zundamon), then the id contained in the name; a folder with a single
    speaker needs no match.  None = ambiguous / no match."""
    ids = sorted(speakers, key=len, reverse=True)
    name = speaker_id(Path(pth).stem)
    for test in (lambda s: name == s, lambda s: name.startswith(s), lambda s: s in name):
        hits = [s for s in ids if test(s)]
        if hits:
            return hits[0]
    return ids[0] if len(ids) == 1 else None


def find_index(pth) -> str:
    """The RVC .index next to a .pth (same name first, else the only .index in that folder); '' if none."""
    pth = Path(pth)
    same = pth.with_suffix(".index")
    if same.is_file():
        return str(same)
    others = sorted(pth.parent.glob("*.index"))
    return str(others[0]) if len(others) == 1 else ""


def scan_rows(rows) -> dict:
    """Emotion data entered as rows in the Model Trainer:
    [{"wav_dir": ..., "lab_dir": "" (= wav_dir), "speaker": "zundamon", "style": "normal"}, ...]
    LAB = same file name as the WAV, first in the row's LAB folder, then in the LAB/WAV folders of other rows
    with the same style (a sentence label shared by speakers).  Same result shape as scan()."""
    items, problems, speakers = [], [], {}
    per_row = [{"wavs": 0, "with_lab": 0} for _ in rows]   # same order as the rows on the page
    clean = []
    for i, r in enumerate(rows):
        wav_dir = Path(str(r.get("wav_dir", "")).strip())
        lab_dir = Path(str(r.get("lab_dir", "")).strip()) if str(r.get("lab_dir", "")).strip() else wav_dir
        spk, style = speaker_id(str(r.get("speaker", ""))), str(r.get("style", "")).lower()
        if style not in STYLES or not str(r.get("speaker", "")).strip() or not wav_dir.is_dir():
            problems.append(f"row {i + 1}: " + ("unknown style" if style not in STYLES else
                                                "speaker name missing" if not str(r.get("speaker", "")).strip() else "WAV folder not found"))
            clean.append(None)
            continue
        clean.append((wav_dir, lab_dir, spk, style))
    labs_by_dir = {}

    def lab_index(d: Path):
        if d not in labs_by_dir:
            labs_by_dir[d] = {p.stem.lower(): p for p in d.rglob("*.lab")} if d.is_dir() else {}
        return labs_by_dir[d]

    for i, c in enumerate(clean):
        if c is None:
            continue
        wav_dir, lab_dir, spk, style = c
        own = lab_index(lab_dir)
        others = [lab_index(o[1]) for j, o in enumerate(clean) if o is not None and j != i and o[3] == style]
        info = speakers.setdefault(spk, {"folder": spk, "counts": {s: 0 for s in STYLES}})
        n_wav = n_lab = 0
        for wav in sorted(wav_dir.rglob("*.wav")):
            n_wav += 1
            sid = sentence_id(wav)
            key = wav.stem.lower()
            lab = own.get(key) or next((o[key] for o in others if key in o), None)
            if sid is None or lab is None:
                continue
            n_lab += 1
            items.append({"speaker": spk, "style": style, "sentence_id": sid, "wav": str(wav), "lab": str(lab),
                          "lab_kind": "speaker" if key in own else "other_row"})
            info["counts"][style] += 1
        per_row[i] = {"wavs": n_wav, "with_lab": n_lab}
    speakers = {k: v for k, v in speakers.items() if sum(v["counts"].values())}
    for spk, info in speakers.items():
        if info["counts"]["normal"] < 10:
            problems.append(f"speaker '{spk}': fewer than 10 Normal recordings with LAB (Normal is needed for every speaker)")
    return {"speakers": speakers, "items": items, "problems": problems, "rows": per_row}
