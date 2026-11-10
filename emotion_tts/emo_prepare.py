from __future__ import annotations

"""v60 data preparation and audit (read-only on all original data).

Writes ONLY under outputs/v60/<run>/data:
  wav24k/<speaker>/<style>/*.wav  Japanese recordings, 24 kHz mono, edge-trimmed
  ja.csv          all accepted Japanese rows with sentence-level split
  fi_train.csv    Finnish Chatterbox rows (reuse the Finnish VITS FinBERT caches)
  fi_eval.csv
  bridge_src.csv  Finnish subset to convert with an RVC model (optional step)
  vocab.json      Finnish VITS vocab + š/ž (old IDs unchanged)
  audit.json / audit.md
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from emo_common import (APPROXIMATE, DATASET_SPEAKER_DIRS, DEFAULT_TEST_IDS, DEFAULT_VALID_IDS,  # noqa: E402
                        NEW_TOKENS, STYLE_LAYOUT, STYLES, lab_to_tokens, read_csv, read_lab_phones,
                        repair_lab_phones,
                        sentence_id_from_wav, sentence_split, write_csv)

SR = 24000


def find_lab(dataset: Path, speaker_dir: str, style: str, sid: int) -> tuple[Path | None, str]:
    _wav, lab_dir, prefix = STYLE_LAYOUT[style]
    name = f"{prefix}{sid:03d}.lab"
    chain = [
        (dataset / style / lab_dir / speaker_dir / name, "speaker"),
        (dataset / style / lab_dir / name, "shared"),
        (dataset / style / lab_dir / "Zundamon" / name, "zundamon_fallback"),
        (dataset / style / lab_dir / "Metan" / name, "metan_fallback"),
    ]
    for p, kind in chain:
        if p.is_file():
            return p, kind
    return None, "missing"


def load_resample(path: Path):
    import soundfile as sf
    import torch
    import torchaudio
    arr, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = torch.from_numpy(arr.mean(axis=1).copy())
    if sr != SR:
        wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, SR).squeeze(0)
    return np.clip(wav.numpy(), -1.0, 1.0), sr


def energy_trim(wav: np.ndarray, *, top_db: float = 40.0, pad_s: float = 0.15) -> np.ndarray:
    frame, hop = int(0.02 * SR), int(0.01 * SR)
    if wav.size < frame * 2:
        return wav
    n = 1 + (wav.size - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    rms = np.sqrt(np.mean(wav[idx] ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-12)
    keep = np.where(db > db.max() - top_db)[0]
    if keep.size == 0:
        return wav
    pad = int(pad_s * SR)
    a = max(0, keep[0] * hop - pad)
    b = min(wav.size, keep[-1] * hop + frame + pad)
    return wav[a:b]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HERE.parent))
    ap.add_argument("--run", default="v60_main")
    ap.add_argument("--step7-dir", default="outputs/finnish_dataset")
    ap.add_argument("--styles", default=",".join(STYLES))
    ap.add_argument("--bridge-count", type=int, default=1500)
    ap.add_argument("--max-seconds", type=float, default=11.0)
    ap.add_argument("--seed", type=int, default=20261018)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    dataset = root / "dataset"
    step7 = (root / args.step7_dir).resolve()
    out = root / "outputs" / "v60" / args.run / "data"
    out.mkdir(parents=True, exist_ok=True)
    styles = tuple(s.strip() for s in args.styles.split(",") if s.strip())
    for s in styles:
        if s not in STYLE_LAYOUT:
            raise SystemExit(f"[FAIL] unknown style {s!r}; known={list(STYLE_LAYOUT)}")

    import soundfile as sf

    old_vocab = json.loads((step7 / "vocab.json").read_text(encoding="utf-8"))
    vocab = dict(old_vocab)
    for t in NEW_TOKENS:
        if t not in vocab:
            vocab[t] = len(vocab)
    (out / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------------- Japanese ----------------
    rows, issues = [], []
    lab_kinds = Counter()
    token_use = Counter()
    for spk, spk_dir in DATASET_SPEAKER_DIRS.items():
        for style in styles:
            wav_dir_name, _lab, prefix = STYLE_LAYOUT[style]
            wdir = dataset / style / wav_dir_name / spk_dir
            if not wdir.is_dir():
                issues.append(f"missing folder {wdir}")
                continue
            for wav in sorted(wdir.glob("*.wav")):
                sid = sentence_id_from_wav(wav)
                if sid is None or not wav.name.startswith(prefix):
                    issues.append(f"skip non-canonical file {wav.relative_to(root)}")
                    continue
                lab, kind = find_lab(dataset, spk_dir, style, sid)
                if lab is None:
                    issues.append(f"no LAB for {wav.relative_to(root)}")
                    continue
                try:
                    phones, repairs = repair_lab_phones(read_lab_phones(lab))
                    tokens = lab_to_tokens(phones)
                    if repairs:
                        issues.append(f"WARN repaired LAB {lab.relative_to(root)}: {', '.join(repairs)}")
                except ValueError as exc:
                    issues.append(f"LAB rejected {lab.relative_to(root)}: {exc}")
                    continue
                bad = [t for t in tokens if t not in vocab]
                if bad:
                    issues.append(f"tokens outside vocab {lab.relative_to(root)}: {sorted(set(bad))}")
                    continue
                dst = out / "wav24k" / spk / style / f"{spk}_{style}_{sid:03d}.wav"
                if dst.is_file():
                    seconds = sf.info(str(dst)).duration
                else:
                    audio, _sr = load_resample(wav)
                    audio = energy_trim(audio)
                    seconds = audio.size / SR
                    if seconds > args.max_seconds or seconds < 0.6:
                        issues.append(f"duration {seconds:.2f}s rejected {wav.relative_to(root)}")
                        continue
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(str(dst), audio.astype(np.float32), SR, subtype="PCM_16")
                if seconds > args.max_seconds:
                    continue
                lab_kinds[kind] += 1
                token_use.update(tokens)
                rows.append({
                    "stem": f"ja_{spk}_{style}_{sid:03d}", "lang": "ja", "speaker": spk, "style": style,
                    "sentence_id": sid,
                    "split": sentence_split(sid, test_ids=DEFAULT_TEST_IDS, valid_ids=DEFAULT_VALID_IDS),
                    "tokens": " ".join(tokens), "wav_path": str(dst), "bert_cache": "",
                    "duration_sec": f"{seconds:.3f}", "source_wav": str(wav), "lab_path": str(lab),
                    "lab_kind": kind,
                })
            print(f"[JA] {spk:8s} {style:6s} rows={sum(1 for r in rows if r['speaker']==spk and r['style']==style)}", flush=True)
    if not rows:
        raise SystemExit("[FAIL] no Japanese rows accepted")
    write_csv(out / "ja.csv", rows)

    # ---------------- Finnish ----------------
    fi_cols = ["stem", "lang", "speaker", "style", "sentence_id", "split", "tokens", "wav_path",
               "bert_cache", "duration_sec", "text"]
    fi_train, fi_eval = [], []
    for src, dst, split in ((step7 / "train.csv", fi_train, "train"), (step7 / "eval.csv", fi_eval, "test")):
        for r in read_csv(src):
            dst.append({"stem": r["stem"], "lang": "fi", "speaker": "fi_chatterbox", "style": "normal",
                        "sentence_id": "", "split": split, "tokens": "", "wav_path": r["wav_path"],
                        "bert_cache": r["bert_cache"], "duration_sec": r.get("duration_sec", ""),
                        "text": r.get("original_text") or r.get("text", "")})
    write_csv(out / "fi_train.csv", fi_train, fi_cols)
    write_csv(out / "fi_eval.csv", fi_eval, fi_cols)

    rng = random.Random(args.seed)
    pool = [r for r in fi_train if r["duration_sec"] and 2.0 <= float(r["duration_sec"]) <= 7.5]
    rng.shuffle(pool)
    bridge = pool[: max(0, args.bridge_count)]
    if bridge:
        write_csv(out / "bridge_src.csv", bridge, fi_cols)

    # ---------------- audit ----------------
    per = defaultdict(lambda: {"n": 0, "seconds": 0.0, "train": 0, "valid": 0, "test": 0})
    for r in rows:
        k = f"{r['speaker']}/{r['style']}"
        per[k]["n"] += 1
        per[k]["seconds"] += float(r["duration_sec"])
        per[k][r["split"]] += 1
    style_dur = defaultdict(list)
    by_key = {(r["speaker"], r["style"], r["sentence_id"]): float(r["duration_sec"]) for r in rows}
    for (spk, style, sid), d in by_key.items():
        n = by_key.get((spk, "normal", sid))
        if style != "normal" and n:
            style_dur[style].append(d / n)
    audit = {
        "schema": "zundanen-v60-audit-v1",
        "japanese_rows": len(rows), "finnish_train": len(fi_train), "finnish_eval": len(fi_eval),
        "bridge_candidates": len(bridge),
        "per_speaker_style": {k: dict(v, seconds=round(v["seconds"], 1)) for k, v in sorted(per.items())},
        "duration_ratio_vs_normal_median": {k: round(float(np.median(v)), 3) for k, v in style_dur.items()},
        "lab_source": dict(lab_kinds),
        "token_counts": dict(token_use.most_common()),
        "new_tokens": list(NEW_TOKENS), "approximations": APPROXIMATE,
        "test_sentence_ids": sorted(DEFAULT_TEST_IDS), "valid_sentence_ids": sorted(DEFAULT_VALID_IDS),
        "issues": issues,
    }
    (out / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# v60 data audit", "", f"Japanese rows: {len(rows)} / Finnish train {len(fi_train)} / bridge candidates {len(bridge)}", "",
             "| speaker/style | n | sec | train | valid | test |", "|---|---:|---:|---:|---:|---:|"]
    for k, v in sorted(per.items()):
        lines.append(f"| {k} | {v['n']} | {v['seconds']:.0f} | {v['train']} | {v['valid']} | {v['test']} |")
    lines += ["", "Median duration ratio vs same speaker/sentence Normal: "
              + ", ".join(f"{k}={v}" for k, v in audit["duration_ratio_vs_normal_median"].items()),
              "", "Approximate phone mappings (not identical sounds):"]
    lines += [f"- `{k}`: {v}" for k, v in APPROXIMATE.items()]
    lines += ["", f"Issues ({len(issues)}):"] + [f"- {x}" for x in issues[:200]]
    (out / "audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[PASS] Japanese={len(rows)} issues={len(issues)} -> {out}")
    print(f"[AUDIT] {out / 'audit.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
