from __future__ import annotations

"""FinBERT contextual feature extraction/cache for Zundanen Style-TTS (Step 4).

The model is frozen and used only as a feature extractor.  We intentionally
keep two parallel text views:

- canonical_text: Step-1/2 lowercase text used by the Finnish grapheme frontend
- bert_text: the same normalized text with original casing preserved where
  possible, so the recommended cased FinBERT can actually use case information

Character positions remain 1:1 between the two views because case folding here
never changes string length for the supported Finnish/ASCII alphabet.

Cached feature layout is [num_grapheme_tokens, hidden_size].  A grapheme gets
its covering WordPiece hidden state; <BOS>/<EOS> get [CLS]/[SEP], and <SP> is a
zero vector in Step 4.  The space policy is explicit metadata so it can be
changed later without ambiguity.
"""

from dataclasses import dataclass
import argparse
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from typing import Any, Sequence

import numpy as np

from finnish_frontend import BOS, EOS, SPACE, tokenize
from finnish_normalizer import normalize_finnish
from finbert_alignment import (
    FINBERT_MODEL_ID,
    BertToken,
    GraphemeAlignment,
    _bert_tokens_from_encoding,
    _find_covering_bert_token,
)

CACHE_SCHEMA = "zundanen-finbert-features-v1"
DEFAULT_LAYER = -1
DEFAULT_CACHE_DTYPE = "float16"
SPACE_POLICY = "zero"
_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class FeatureResult:
    original: str
    canonical_text: str
    bert_text: str
    warnings: tuple[Any, ...]
    frontend_tokens: tuple[str, ...]
    bert_tokens: tuple[BertToken, ...]
    grapheme_alignment: tuple[GraphemeAlignment, ...]
    features: np.ndarray
    model_id: str
    layer: int
    hidden_size: int
    space_policy: str = SPACE_POLICY

    @property
    def ok(self) -> bool:
        return (
            not self.warnings
            and self.features.ndim == 2
            and self.features.shape[0] == len(self.frontend_tokens)
            and self.features.shape[1] == self.hidden_size
            and np.isfinite(self.features).all()
        )


def _base_normalize_preserve_case(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be str")
    text = unicodedata.normalize("NFC", text)
    return _WS_RE.sub(" ", text).strip()


def normalize_finnish_preserve_case(text: str) -> tuple[str, tuple[Any, ...]]:
    """Run Step-2 rules while retaining case where the source retained it.

    Step 2 currently lowercases only at the very end.  To avoid duplicating the
    number/version grammar, we use a case-preserving shadow string only when no
    Step-2 rewrite was needed.  When Step-2 rewrites something (numbers, units,
    versions), the canonical normalized result is used.  This conservative
    behavior guarantees character offsets remain identical.

    In practical training text, ordinary Finnish sentences therefore preserve
    proper-noun/sentence case for FinBERT; rewritten numeric expressions are
    intentionally lowercase.
    """
    canonical, warnings = normalize_finnish(text)
    base = _base_normalize_preserve_case(text)

    # If lowercasing is the only difference, casing can be restored exactly.
    # Python lower() can theoretically change length for exotic Unicode.  We
    # only accept the shadow view when both length and lowercase text agree.
    if len(base) == len(canonical) and base.lower() == canonical:
        return base, warnings
    return canonical, warnings


def build_feature_alignment(
    *,
    canonical_text: str,
    bert_text: str,
    tokenizer: Any,
    input_ids: Sequence[int],
    offsets: Sequence[Sequence[int]],
) -> tuple[tuple[BertToken, ...], tuple[GraphemeAlignment, ...]]:
    """Align canonical graphemes to WordPieces generated from `bert_text`."""
    if len(canonical_text) != len(bert_text):
        raise ValueError("canonical_text and bert_text must have identical length")

    frontend = tokenize(canonical_text)
    bert_tokens = _bert_tokens_from_encoding(tokenizer, input_ids, offsets)

    cls_id = getattr(tokenizer, "cls_token_id", None)
    sep_id = getattr(tokenizer, "sep_token_id", None)
    cls_indices = [t.index for t in bert_tokens if cls_id is not None and t.token_id == cls_id]
    sep_indices = [t.index for t in bert_tokens if sep_id is not None and t.token_id == sep_id]
    cls_index = cls_indices[0] if cls_indices else None
    sep_index = sep_indices[-1] if sep_indices else None

    items: list[GraphemeAlignment] = []
    text_pos = 0
    for i, token in enumerate(frontend.tokens):
        if token == BOS:
            items.append(GraphemeAlignment(i, token, None, None, cls_index, "bos"))
            continue
        if token == EOS:
            items.append(GraphemeAlignment(i, token, None, None, sep_index, "eos"))
            continue

        if text_pos >= len(canonical_text):
            raise ValueError("frontend tokens exceed canonical text length")
        ch = canonical_text[text_pos]
        start, end = text_pos, text_pos + 1

        if token == SPACE:
            if ch != " ":
                raise ValueError("frontend/text space mismatch")
            items.append(GraphemeAlignment(i, token, start, end, None, "space"))
        else:
            expected = ch if token == ch else f"<UNK:{ch}>"
            if token != expected:
                raise ValueError(
                    f"frontend/text mismatch at {text_pos}: token={token!r}, char={ch!r}"
                )
            bert_index = _find_covering_bert_token(bert_tokens, start, end)
            items.append(GraphemeAlignment(i, token, start, end, bert_index, "text"))
        text_pos += 1

    if text_pos != len(canonical_text):
        raise ValueError("canonical text has unconsumed characters")

    missing = [x for x in items if x.kind in {"text", "bos", "eos"} and x.bert_token_index is None]
    if missing:
        raise ValueError(f"unmapped non-space graphemes: {missing}")
    return bert_tokens, tuple(items)


def project_wordpieces_to_graphemes(
    hidden_states: np.ndarray,
    alignment: Sequence[GraphemeAlignment],
    *,
    space_policy: str = SPACE_POLICY,
) -> np.ndarray:
    """Expand WordPiece hidden states to canonical grapheme length.

    `hidden_states` must be [num_bert_tokens, hidden_size].  Multiple graphemes
    belonging to one WordPiece intentionally receive the same contextual vector.
    """
    hidden_states = np.asarray(hidden_states)
    if hidden_states.ndim != 2:
        raise ValueError("hidden_states must have shape [tokens, hidden]")
    if space_policy != "zero":
        raise ValueError(f"unsupported Step-4 space policy: {space_policy}")

    out = np.zeros((len(alignment), hidden_states.shape[1]), dtype=hidden_states.dtype)
    for g in alignment:
        if g.kind == "space":
            continue
        if g.bert_token_index is None:
            raise ValueError(f"missing BERT index for {g}")
        if not 0 <= g.bert_token_index < hidden_states.shape[0]:
            raise IndexError(f"BERT token index out of range: {g.bert_token_index}")
        out[g.index] = hidden_states[g.bert_token_index]
    return out


def load_finbert_model(model_id: str = FINBERT_MODEL_ID, *, device: str = "auto") -> tuple[Any, Any, str]:
    """Load tokenizer + frozen AutoModel and return resolved device string."""
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("Step 4 dependencies are missing. Run setup_step4.bat first.") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("FinBERT fast tokenizer is required for offset_mapping")

    if device == "auto":
        resolved = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved = device
    if resolved.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    model = AutoModel.from_pretrained(model_id)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    model.to(resolved)
    return tokenizer, model, resolved


def extract_features(
    text: str,
    *,
    tokenizer: Any,
    model: Any,
    device: str,
    model_id: str = FINBERT_MODEL_ID,
    layer: int = DEFAULT_LAYER,
) -> FeatureResult:
    """Extract one frozen FinBERT layer and expand it to grapheme positions."""
    import torch

    canonical, warnings = normalize_finnish(text)
    bert_text, warnings_case = normalize_finnish_preserve_case(text)
    if tuple(warnings_case) != tuple(warnings):
        raise RuntimeError("canonical and case-preserving normalizers disagree on warnings")

    frontend = tokenize(canonical)
    encoded = tokenizer(
        bert_text,
        add_special_tokens=True,
        return_offsets_mapping=True,
        truncation=False,
        return_tensors="pt",
    )
    offsets_tensor = encoded.pop("offset_mapping")
    offsets = offsets_tensor[0].tolist()
    input_ids_list = encoded["input_ids"][0].tolist()

    max_len = int(getattr(tokenizer, "model_max_length", 512) or 512)
    if max_len < 100_000 and len(input_ids_list) > max_len:
        raise ValueError(
            f"FinBERT sequence is too long: {len(input_ids_list)} tokens > {max_len}"
        )

    bert_tokens, alignment = build_feature_alignment(
        canonical_text=canonical,
        bert_text=bert_text,
        tokenizer=tokenizer,
        input_ids=input_ids_list,
        offsets=offsets,
    )

    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.inference_mode():
        outputs = model(**encoded, output_hidden_states=True, return_dict=True)
    hidden_stack = outputs.hidden_states
    if hidden_stack is None:
        raise RuntimeError("FinBERT did not return hidden states")
    try:
        selected = hidden_stack[layer][0]
    except IndexError as exc:
        raise ValueError(
            f"requested layer {layer} is outside available hidden-state range {len(hidden_stack)}"
        ) from exc

    wp = selected.detach().float().cpu().numpy()
    grapheme_features = project_wordpieces_to_graphemes(wp, alignment)

    hidden_size = int(grapheme_features.shape[1])
    return FeatureResult(
        original=text,
        canonical_text=canonical,
        bert_text=bert_text,
        warnings=tuple(warnings),
        frontend_tokens=tuple(frontend.tokens),
        bert_tokens=bert_tokens,
        grapheme_alignment=alignment,
        features=grapheme_features,
        model_id=model_id,
        layer=int(layer),
        hidden_size=hidden_size,
    )


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def save_feature_cache(
    path: str | Path,
    result: FeatureResult,
    *,
    dtype: str = DEFAULT_CACHE_DTYPE,
) -> Path:
    """Save an allow_pickle=False NPZ cache with explicit schema metadata."""
    if dtype not in {"float16", "float32"}:
        raise ValueError("dtype must be float16 or float32")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    features = result.features.astype(np.float16 if dtype == "float16" else np.float32)
    grapheme_to_bert = np.array(
        [(-1 if g.bert_token_index is None else g.bert_token_index) for g in result.grapheme_alignment],
        dtype=np.int32,
    )
    bert_ids = np.array([t.token_id for t in result.bert_tokens], dtype=np.int32)
    meta = {
        "schema": CACHE_SCHEMA,
        "model_id": result.model_id,
        "layer": result.layer,
        "hidden_size": result.hidden_size,
        "space_policy": result.space_policy,
        "feature_dtype": dtype,
        "original_sha256": _text_sha256(result.original),
        "canonical_sha256": _text_sha256(result.canonical_text),
        "warnings": [getattr(w, "as_dict", lambda: str(w))() for w in result.warnings],
    }

    np.savez_compressed(
        target,
        features=features,
        grapheme_to_bert=grapheme_to_bert,
        bert_token_ids=bert_ids,
        grapheme_tokens=np.asarray(result.frontend_tokens, dtype=np.str_),
        canonical_text=np.asarray(result.canonical_text, dtype=np.str_),
        bert_text=np.asarray(result.bert_text, dtype=np.str_),
        metadata_json=np.asarray(json.dumps(meta, ensure_ascii=False, sort_keys=True), dtype=np.str_),
    )
    return target


def load_feature_cache(path: str | Path) -> dict[str, Any]:
    with np.load(Path(path), allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        if metadata.get("schema") != CACHE_SCHEMA:
            raise ValueError(f"unsupported cache schema: {metadata.get('schema')!r}")
        return {
            "features": data["features"].copy(),
            "grapheme_to_bert": data["grapheme_to_bert"].copy(),
            "bert_token_ids": data["bert_token_ids"].copy(),
            "grapheme_tokens": tuple(str(x) for x in data["grapheme_tokens"].tolist()),
            "canonical_text": str(data["canonical_text"].item()),
            "bert_text": str(data["bert_text"].item()),
            "metadata": metadata,
        }


def format_feature_result(result: FeatureResult, *, device: str) -> str:
    lines = [
        f"INPUT       : {result.original}",
        f"CANONICAL   : {result.canonical_text}",
        f"BERT TEXT   : {result.bert_text}",
        f"MODEL       : {result.model_id}",
        f"LAYER       : {result.layer}",
        f"DEVICE      : {device}",
        f"FEATURES    : {result.features.shape} float32",
        f"SPACE POLICY: {result.space_policy}",
        f"STATUS      : {'OK' if result.ok else 'NOT OK'}",
    ]
    if result.warnings:
        lines.append("WARNINGS:")
        for w in result.warnings:
            lines.append(f"  [{w.code}] {w.text}")
    return "\n".join(lines)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Zundanen FinBERT feature cache (Step 4)")
    p.add_argument("text", nargs="?", help="Finnish text")
    p.add_argument("--out", help="write .npz feature cache")
    p.add_argument("--inspect", help="inspect an existing cache instead of extracting")
    p.add_argument("--model", default=FINBERT_MODEL_ID)
    p.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--dtype", default=DEFAULT_CACHE_DTYPE, choices=("float16", "float32"))
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    if args.inspect:
        cache = load_feature_cache(args.inspect)
        print("CACHE       :", args.inspect)
        print("CANONICAL   :", cache["canonical_text"])
        print("BERT TEXT   :", cache["bert_text"])
        print("FEATURES    :", cache["features"].shape, cache["features"].dtype)
        print("METADATA    :", json.dumps(cache["metadata"], ensure_ascii=False, indent=2))
        return 0

    text = args.text
    if text is None:
        try:
            text = input("Finnish text> ")
        except EOFError:
            return 2

    try:
        tokenizer, model, device = load_finbert_model(args.model, device=args.device)
        result = extract_features(
            text,
            tokenizer=tokenizer,
            model=model,
            device=device,
            model_id=args.model,
            layer=args.layer,
        )
        print(format_feature_result(result, device=device))
        if args.out:
            path = save_feature_cache(args.out, result, dtype=args.dtype)
            print("CACHE SAVED :", path)
        return 0 if result.ok else 1
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
