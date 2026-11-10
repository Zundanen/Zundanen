from __future__ import annotations

"""Prepare any Finnish WAV+label dataset for Zundanen Finnish VITS.

The output is self-contained for Finnish VITS training:
  <out>/train.csv
  <out>/eval.csv
  <out>/all.csv
  <out>/vocab.json
  <out>/finbert_large_1024_cache/*.npz
  <out>/prepare.json
  <out>/rejected.csv

Inputs can be either:
  1) --wav-dir + --label-dir, paired by filename stem, or
  2) --manifest CSV with wav_path and either text or txt_path.

FinBERT is frozen; only cached 1024-D features are written.
"""

import argparse
import csv
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf

from finbert_features import (
    extract_features,
    load_feature_cache,
    load_finbert_model,
    save_feature_cache,
)
from finnish_frontend import BOS, EOS, SPACE, FINNISH_LETTERS, PUNCTUATION

FINBERT_LARGE = "TurkuNLP/bert-large-finnish-cased-v1"
DEFAULT_SEED = 20261008


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def fixed_frontend_vocab() -> dict[str, int]:
    # Fixed ordering makes independently prepared CSS/Chatterbox datasets
    # checkpoint-compatible for fine-tuning.
    symbols = ["<PAD>", BOS, EOS, SPACE]
    symbols += sorted(FINNISH_LETTERS)
    symbols += sorted(PUNCTUATION)
    deduped: list[str] = []
    seen: set[str] = set()
    for s in symbols:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    return {s: i for i, s in enumerate(deduped)}


def _unique_by_stem(files: Iterable[Path], what: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    dup: dict[str, list[Path]] = {}
    for p in files:
        if p.stem in out:
            dup.setdefault(p.stem, [out[p.stem]]).append(p)
        else:
            out[p.stem] = p
    if dup:
        sample = next(iter(dup.items()))
        paths = "\n    ".join(str(x) for x in sample[1])
        raise SystemExit(
            f"[FAIL] Duplicate {what} stem {sample[0]!r}. Stems must be unique.\n    {paths}"
        )
    return out


def collect_from_dirs(wav_dir: Path, label_dir: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    wavs = _unique_by_stem(wav_dir.rglob("*.wav"), "WAV")
    labels = _unique_by_stem(label_dir.rglob("*.txt"), "label")
    rows: list[dict[str, str]] = []
    rejected: list[dict[str, str]] = []

    for stem in sorted(set(wavs) | set(labels)):
        wav = wavs.get(stem)
        txt = labels.get(stem)
        if wav is None:
            rejected.append({"stem": stem, "reason": "missing_wav", "detail": str(txt or "")})
            continue
        if txt is None:
            rejected.append({"stem": stem, "reason": "missing_label", "detail": str(wav)})
            continue
        try:
            text = txt.read_text(encoding="utf-8-sig").strip()
        except Exception as exc:
            rejected.append({"stem": stem, "reason": "label_read_error", "detail": str(exc)})
            continue
        rows.append({"stem": stem, "wav_path": str(wav.resolve()), "text": text, "txt_path": str(txt.resolve())})
    return rows, rejected


def collect_from_manifest(path: Path, base_dir: Path | None) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    src = read_csv(path)
    rows: list[dict[str, str]] = []
    rejected: list[dict[str, str]] = []
    base = base_dir.resolve() if base_dir else path.parent.resolve()

    for i, r in enumerate(src, 1):
        stem = (r.get("stem") or "").strip()
        wav_s = (r.get("wav_path") or "").strip()
        filename = (r.get("filename") or "").strip()

        if not wav_s and filename:
            wav_s = filename
        if not wav_s:
            rejected.append({"stem": stem or str(i), "reason": "missing_wav_path", "detail": ""})
            continue

        wav = Path(wav_s)
        if not wav.is_absolute():
            wav = base / wav
        if wav.suffix.lower() != ".wav":
            # Common batch manifests store filename without extension.
            candidate = wav.with_suffix(".wav")
            if candidate.exists():
                wav = candidate

        if not stem:
            stem = wav.stem

        text = (r.get("text") or "").strip()
        txt_s = (r.get("txt_path") or "").strip()
        if not text and txt_s:
            txt = Path(txt_s)
            if not txt.is_absolute():
                txt = base / txt
            try:
                text = txt.read_text(encoding="utf-8-sig").strip()
            except Exception as exc:
                rejected.append({"stem": stem, "reason": "label_read_error", "detail": str(exc)})
                continue

        rows.append({"stem": stem, "wav_path": str(wav.resolve()), "text": text, "txt_path": txt_s})

    stems: set[str] = set()
    for r in rows:
        if r["stem"] in stems:
            raise SystemExit(f"[FAIL] Duplicate manifest stem: {r['stem']!r}")
        stems.add(r["stem"])
    return rows, rejected


def cache_is_reusable(path: Path, text: str) -> bool:
    if not path.exists():
        return False
    try:
        c = load_feature_cache(path)
        meta = c["metadata"]
        return (
            meta.get("model_id") == FINBERT_LARGE
            and int(meta.get("hidden_size", -1)) == 1024
            and meta.get("original_sha256") == hashlib.sha256(text.encode("utf-8")).hexdigest()
            and c["features"].ndim == 2
            and c["features"].shape[1] == 1024
        )
    except Exception:
        return False


def safe_cache_name(stem: str) -> str:
    s = re.sub(r"[^0-9A-Za-z_.-]+", "_", stem).strip("._") or "sample"
    digest = hashlib.sha1(stem.encode("utf-8")).hexdigest()[:8]
    return f"{s}_{digest}.npz"


def main() -> int:
    ap = argparse.ArgumentParser(description="Prepare generic Finnish data for Zundanen Finnish VITS")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", help="CSV with wav_path and text or txt_path")
    src.add_argument("--wav-dir", help="WAV directory; pair with --label-dir")
    ap.add_argument("--label-dir", help="TXT label directory, paired by stem")
    ap.add_argument("--manifest-base", default="", help="base directory for relative manifest paths")
    ap.add_argument("--out", required=True, help="output prepared dataset directory")
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--eval-ratio", type=float, default=0.02)
    ap.add_argument("--eval-count", type=int, default=0, help="override eval-ratio when >0")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--dtype", default="float16", choices=("float16", "float32"))
    ap.add_argument("--allow-normalizer-warnings", action="store_true")
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--source-name", default="", help="optional dataset label, e.g. css10 or chatterbox")
    args = ap.parse_args()

    out = Path(args.out.strip().strip('"')).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = out / "finbert_large_1024_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    rejected: list[dict[str, str]] = []
    if args.manifest:
        manifest = Path(args.manifest.strip().strip('"')).resolve()
        base = Path(args.manifest_base.strip().strip('"')).resolve() if args.manifest_base else None
        rows, rej = collect_from_manifest(manifest, base)
    else:
        if not args.label_dir:
            raise SystemExit("[FAIL] --label-dir is required with --wav-dir")
        wav_dir = Path(args.wav_dir.strip().strip('"')).resolve()
        label_dir = Path(args.label_dir.strip().strip('"')).resolve()
        if not wav_dir.exists() or not label_dir.exists():
            raise SystemExit(f"[FAIL] Missing input directory: wav={wav_dir} labels={label_dir}")
        rows, rej = collect_from_dirs(wav_dir, label_dir)
    rejected.extend(rej)

    if len(rows) < 6:
        raise SystemExit(f"[FAIL] Need at least 6 paired samples; found {len(rows)}")

    tokenizer = model = resolved_device = None
    prepared: list[dict[str, str]] = []
    reused = 0
    generated = 0

    print(f"[SCAN] paired candidates={len(rows)} pre-rejected={len(rejected)}")
    for i, r in enumerate(rows, 1):
        stem = r["stem"]
        text = r["text"].strip()
        wav = Path(r["wav_path"])
        if not text:
            rejected.append({"stem": stem, "reason": "empty_text", "detail": ""})
            continue
        if not wav.exists():
            rejected.append({"stem": stem, "reason": "missing_wav", "detail": str(wav)})
            continue
        try:
            info = sf.info(str(wav))
            duration = float(info.frames) / float(info.samplerate)
        except Exception as exc:
            rejected.append({"stem": stem, "reason": "wav_read_error", "detail": str(exc)})
            continue
        if not np.isfinite(duration) or duration <= 0.0:
            rejected.append({"stem": stem, "reason": "invalid_duration", "detail": str(duration)})
            continue

        cache_path = cache_dir / safe_cache_name(stem)
        reuse = (not args.rebuild_cache) and cache_is_reusable(cache_path, text)
        try:
            if reuse:
                c = load_feature_cache(cache_path)
                canonical = c["canonical_text"]
                tokens = tuple(c["grapheme_tokens"])
                warnings = c["metadata"].get("warnings") or []
                reused += 1
            else:
                if model is None:
                    print(f"[BERT] loading {FINBERT_LARGE} device={args.device}")
                    tokenizer, model, resolved_device = load_finbert_model(FINBERT_LARGE, device=args.device)
                result = extract_features(
                    text,
                    tokenizer=tokenizer,
                    model=model,
                    device=resolved_device,
                    model_id=FINBERT_LARGE,
                )
                if result.hidden_size != 1024:
                    raise RuntimeError(f"expected 1024-D FinBERT Large, got {result.hidden_size}")
                save_feature_cache(cache_path, result, dtype=args.dtype)
                canonical = result.canonical_text
                tokens = tuple(result.frontend_tokens)
                warnings = list(result.warnings)
                generated += 1
        except Exception as exc:
            rejected.append({"stem": stem, "reason": "finbert_error", "detail": str(exc)})
            continue

        unknown = sorted({t for t in tokens if t.startswith("<UNK:")})
        if unknown:
            rejected.append({"stem": stem, "reason": "unsupported_frontend", "detail": " ".join(unknown)})
            continue
        if warnings and not args.allow_normalizer_warnings:
            rejected.append({"stem": stem, "reason": "normalizer_warning", "detail": json.dumps(warnings, ensure_ascii=False, default=str)})
            continue

        prepared.append(
            {
                "stem": stem,
                "text": canonical,
                "original_text": text,
                "wav_path": str(wav.resolve()),
                "bert_cache": str(cache_path.resolve()),
                "duration_sec": f"{duration:.6f}",
                "source_dataset": args.source_name,
            }
        )
        if i == 1 or i % 100 == 0 or i == len(rows):
            print(f"[CACHE] {i}/{len(rows)} accepted={len(prepared)} generated={generated} reused={reused} rejected={len(rejected)}")

    if len(prepared) < 6:
        raise SystemExit(f"[FAIL] Only {len(prepared)} usable samples after validation")

    # Fixed vocabulary covers every token the canonical Finnish frontend can
    # emit. Unsupported characters were already rejected above as <UNK:...>.
    vocab = fixed_frontend_vocab()

    # Split by canonical-text groups so exact duplicate sentences cannot leak
    # between train and eval.  This keeps the split deterministic across runs.
    rng = random.Random(args.seed)
    groups: dict[str, list[int]] = {}
    for i, r in enumerate(prepared):
        groups.setdefault(r["text"], []).append(i)
    group_items = list(groups.items())
    rng.shuffle(group_items)
    if args.eval_count > 0:
        target_eval = args.eval_count
    else:
        target_eval = max(5, int(round(len(prepared) * args.eval_ratio)))
    target_eval = min(max(1, target_eval), len(prepared) - 1)
    eval_idx: set[int] = set()
    for _text, idxs in group_items:
        if len(eval_idx) >= target_eval:
            break
        if len(eval_idx) + len(idxs) >= len(prepared):
            continue
        eval_idx.update(idxs)
    if not eval_idx:
        eval_idx.add(group_items[0][1][0])
    train_rows = [r for i, r in enumerate(prepared) if i not in eval_idx]
    eval_rows = [r for i, r in enumerate(prepared) if i in eval_idx]

    fields = ["stem", "text", "original_text", "wav_path", "bert_cache", "duration_sec", "source_dataset"]
    write_csv(out / "all.csv", prepared, fields)
    write_csv(out / "train.csv", train_rows, fields)
    write_csv(out / "eval.csv", eval_rows, fields)
    write_csv(out / "rejected.csv", rejected, ["stem", "reason", "detail"])
    (out / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")

    meta = {
        "schema": "zundanen-step7-generic-dataset-v1",
        "finbert_model": FINBERT_LARGE,
        "finbert_hidden_size": 1024,
        "vocab_mode": "fixed_finnish_frontend",
        "seed": args.seed,
        "train_count": len(train_rows),
        "eval_count": len(eval_rows),
        "rejected_count": len(rejected),
        "cache_generated": generated,
        "cache_reused": reused,
        "source_name": args.source_name,
    }
    (out / "prepare.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[PASS] generic Finnish VITS dataset prepared")
    print(f"[DATA] train={len(train_rows)} eval={len(eval_rows)} rejected={len(rejected)}")
    print(f"[OUT]  {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
