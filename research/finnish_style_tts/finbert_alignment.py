from __future__ import annotations

"""FinBERT WordPiece <-> Finnish grapheme alignment for Zundanen Style-TTS.

Step 3 deliberately stops before BERT hidden-state extraction.  It uses only
FinBERT's *tokenizer* and its character offset mapping to establish a stable,
inspectable correspondence between:

    Step-2 normalized Finnish text
        -> Step-1 canonical grapheme tokens
        -> FinBERT WordPiece tokens

No FinBERT model weights are loaded here.
"""

from dataclasses import dataclass
import argparse
import json
from typing import Any, Iterable, Sequence

from finnish_frontend import BOS, EOS, SPACE, FrontendResult, tokenize
from finnish_normalizer import NormalizationWarning, normalize_finnish

FINBERT_MODEL_ID = "TurkuNLP/bert-base-finnish-cased-v1"


@dataclass(frozen=True)
class BertToken:
    index: int
    token: str
    token_id: int
    start: int
    end: int
    special: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "token": self.token,
            "token_id": self.token_id,
            "start": self.start,
            "end": self.end,
            "special": self.special,
        }


@dataclass(frozen=True)
class GraphemeAlignment:
    index: int
    token: str
    start: int | None
    end: int | None
    bert_token_index: int | None
    kind: str

    def as_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "token": self.token,
            "start": self.start,
            "end": self.end,
            "bert_token_index": self.bert_token_index,
            "kind": self.kind,
        }


@dataclass(frozen=True)
class AlignmentResult:
    original: str
    normalized: str
    warnings: tuple[NormalizationWarning, ...]
    frontend: FrontendResult
    bert_tokens: tuple[BertToken, ...]
    grapheme_alignment: tuple[GraphemeAlignment, ...]
    model_id: str

    @property
    def ok(self) -> bool:
        if self.warnings or not self.frontend.ok:
            return False
        return all(
            item.bert_token_index is not None
            for item in self.grapheme_alignment
            if item.kind in {"text", "bos", "eos"}
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "original": self.original,
            "normalized": self.normalized,
            "warnings": [w.as_dict() for w in self.warnings],
            "frontend": self.frontend.as_dict(),
            "model_id": self.model_id,
            "bert_tokens": [t.as_dict() for t in self.bert_tokens],
            "grapheme_alignment": [g.as_dict() for g in self.grapheme_alignment],
            "ok": self.ok,
        }


def load_finbert_tokenizer(model_id: str = FINBERT_MODEL_ID) -> Any:
    """Load the *fast tokenizer only*; this does not download BERT weights."""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "transformers is not installed. Run setup_step3.bat first."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError(
            "FinBERT tokenizer is not a fast tokenizer; offset_mapping is required."
        )
    return tokenizer


def _bert_tokens_from_encoding(
    tokenizer: Any,
    input_ids: Sequence[int],
    offsets: Sequence[Sequence[int]],
) -> tuple[BertToken, ...]:
    raw_tokens = tokenizer.convert_ids_to_tokens(list(input_ids))
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    if len(raw_tokens) != len(offsets) or len(raw_tokens) != len(input_ids):
        raise ValueError("token/id/offset lengths do not match")

    result: list[BertToken] = []
    for i, (token, token_id, span) in enumerate(zip(raw_tokens, input_ids, offsets)):
        if len(span) != 2:
            raise ValueError(f"invalid offset at token {i}: {span!r}")
        start, end = int(span[0]), int(span[1])
        result.append(
            BertToken(
                index=i,
                token=str(token),
                token_id=int(token_id),
                start=start,
                end=end,
                special=int(token_id) in special_ids,
            )
        )
    return tuple(result)


def _find_covering_bert_token(
    bert_tokens: Sequence[BertToken],
    start: int,
    end: int,
) -> int | None:
    # A grapheme is one normalized code point in Step 1.  A WordPiece token can
    # cover many graphemes; every covered grapheme therefore points to the same
    # BERT token index.  Special tokens have zero-length offsets and are skipped.
    candidates = [
        t.index
        for t in bert_tokens
        if not t.special and t.start <= start and end <= t.end and t.end > t.start
    ]
    if len(candidates) > 1:
        raise ValueError(
            f"overlapping tokenizer offsets for character span ({start}, {end}): {candidates}"
        )
    return candidates[0] if candidates else None


def align_from_encoding(
    *,
    original: str,
    normalized: str,
    warnings: Sequence[NormalizationWarning],
    frontend: FrontendResult,
    tokenizer: Any,
    input_ids: Sequence[int],
    offsets: Sequence[Sequence[int]],
    model_id: str = FINBERT_MODEL_ID,
) -> AlignmentResult:
    """Core alignment routine separated so it can be unit-tested offline."""
    bert_tokens = _bert_tokens_from_encoding(tokenizer, input_ids, offsets)

    cls_id = getattr(tokenizer, "cls_token_id", None)
    sep_id = getattr(tokenizer, "sep_token_id", None)
    cls_indices = [t.index for t in bert_tokens if cls_id is not None and t.token_id == cls_id]
    sep_indices = [t.index for t in bert_tokens if sep_id is not None and t.token_id == sep_id]
    cls_index = cls_indices[0] if cls_indices else None
    sep_index = sep_indices[-1] if sep_indices else None

    # Step-1 canonical tokens correspond 1:1 with normalized code points,
    # apart from BOS/EOS.  Spaces intentionally have no WordPiece token; Step 4
    # will assign a zero/learned boundary feature rather than invent a semantic
    # BERT representation for whitespace.
    grapheme_alignment: list[GraphemeAlignment] = []
    text_pos = 0
    for i, token in enumerate(frontend.tokens):
        if token == BOS:
            grapheme_alignment.append(
                GraphemeAlignment(i, token, None, None, cls_index, "bos")
            )
            continue
        if token == EOS:
            grapheme_alignment.append(
                GraphemeAlignment(i, token, None, None, sep_index, "eos")
            )
            continue

        if text_pos >= len(normalized):
            raise ValueError("frontend tokens exceed normalized text length")

        ch = normalized[text_pos]
        start, end = text_pos, text_pos + 1

        if token == SPACE:
            if ch != " ":
                raise ValueError(
                    f"frontend/text mismatch at {text_pos}: token={token!r}, char={ch!r}"
                )
            grapheme_alignment.append(
                GraphemeAlignment(i, token, start, end, None, "space")
            )
        else:
            expected = ch if token == ch else f"<UNK:{ch}>"
            if token != expected:
                raise ValueError(
                    f"frontend/text mismatch at {text_pos}: token={token!r}, char={ch!r}"
                )
            bert_index = _find_covering_bert_token(bert_tokens, start, end)
            grapheme_alignment.append(
                GraphemeAlignment(i, token, start, end, bert_index, "text")
            )

        text_pos += 1

    if text_pos != len(normalized):
        raise ValueError(
            f"normalized text has unconsumed characters: consumed={text_pos}, length={len(normalized)}"
        )

    return AlignmentResult(
        original=original,
        normalized=normalized,
        warnings=tuple(warnings),
        frontend=frontend,
        bert_tokens=bert_tokens,
        grapheme_alignment=tuple(grapheme_alignment),
        model_id=model_id,
    )


def align_text(
    text: str,
    *,
    tokenizer: Any | None = None,
    model_id: str = FINBERT_MODEL_ID,
) -> AlignmentResult:
    """Normalize Finnish and align Step-1 graphemes to FinBERT WordPieces."""
    normalized, warnings = normalize_finnish(text)
    frontend = tokenize(normalized)

    if tokenizer is None:
        tokenizer = load_finbert_tokenizer(model_id)

    encoded = tokenizer(
        normalized,
        add_special_tokens=True,
        return_offsets_mapping=True,
        truncation=False,
    )
    input_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]

    max_len = int(getattr(tokenizer, "model_max_length", 512) or 512)
    if max_len < 100_000 and len(input_ids) > max_len:
        raise ValueError(
            f"FinBERT sequence is too long: {len(input_ids)} tokens > model_max_length {max_len}"
        )

    return align_from_encoding(
        original=text,
        normalized=normalized,
        warnings=warnings,
        frontend=frontend,
        tokenizer=tokenizer,
        input_ids=input_ids,
        offsets=offsets,
        model_id=model_id,
    )


def format_alignment(result: AlignmentResult) -> str:
    lines: list[str] = []
    lines.append(f"INPUT      : {result.original}")
    lines.append(f"NORMALIZED : {result.normalized}")
    lines.append(f"MODEL      : {result.model_id}")
    lines.append("")
    lines.append("BERT TOKENS:")
    for t in result.bert_tokens:
        flags = " special" if t.special else ""
        lines.append(
            f"  [{t.index:>3}] {t.token:<24} id={t.token_id:<6} span=({t.start},{t.end}){flags}"
        )
    lines.append("")
    lines.append("GRAPHEME -> BERT:")
    for g in result.grapheme_alignment:
        if g.bert_token_index is None:
            target = "(none)"
        else:
            target_token = result.bert_tokens[g.bert_token_index].token
            target = f"[{g.bert_token_index}] {target_token}"
        span = "(-,-)" if g.start is None else f"({g.start},{g.end})"
        lines.append(
            f"  [{g.index:>3}] {g.token:<10} span={span:<9} -> {target}  <{g.kind}>"
        )
    if result.warnings:
        lines.append("")
        lines.append("WARNINGS:")
        for warning in result.warnings:
            lines.append(f"  [{warning.code}] {warning.text}")
    lines.append("")
    lines.append(f"STATUS     : {'OK' if result.ok else 'NOT OK'}")
    return "\n".join(lines)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Zundanen Finnish Style-TTS FinBERT alignment (Step 3)"
    )
    p.add_argument("text", nargs="?", help="Finnish text to align")
    p.add_argument("--json", action="store_true", help="emit JSON")
    p.add_argument("--model", default=FINBERT_MODEL_ID, help="tokenizer model id")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    text = args.text
    if text is None:
        try:
            text = input("Finnish text> ")
        except EOFError:
            return 2

    try:
        result = align_text(text, model_id=args.model)
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}")
        return 2

    if args.json:
        print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
    else:
        print(format_alignment(result))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
