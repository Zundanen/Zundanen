from __future__ import annotations

"""Minimal Finnish grapheme frontend for the Zundanen Style-TTS research track.

Step 1 goals:
- deterministic Unicode/text normalization
- canonical Finnish grapheme tokenization
- explicit whitespace/punctuation handling
- explicit reporting of unsupported characters (never silently drop them)
- optional Aalto-compatible phone rendering for regression/debug comparison

This module intentionally uses only the Python standard library.  It does not
load PyTorch, FinBERT, Chatterbox, RVC, Kaldi, or the Aalto runtime.
"""

from dataclasses import dataclass
import argparse
import json
import re
import unicodedata
from typing import Iterable, Sequence

BOS = "<BOS>"
EOS = "<EOS>"
SPACE = "<SP>"

# Canonical symbols for the future TTS frontend.  Keep Finnish orthography
# intact instead of converting it to Kaldi/Aalto acoustic-model symbols.
FINNISH_LETTERS = frozenset("abcdefghijklmnopqrstuvwxyzåäö")

# Keep prosodically useful punctuation as explicit tokens.  The exact set can
# be revised later without changing the basic frontend architecture.
PUNCTUATION = frozenset(".,!?;:-–—'\"()")

# Aalto Finnish Forced Alignment's documented Finnish->Finnish compatibility
# mapping.  This is used ONLY for debug comparison; canonical TTS tokens above
# remain ordinary Finnish graphemes.
_AALTO_PHONE_MAP: dict[str, tuple[str, ...]] = {
    "a": ("A",),
    "b": ("b",),
    "c": ("s",),
    "d": ("d",),
    "e": ("e",),
    "f": ("f",),
    "g": ("g",),
    "h": ("h",),
    "i": ("I",),
    "j": ("j",),
    "k": ("k",),
    "l": ("l",),
    "m": ("m",),
    "n": ("n",),
    "o": ("o",),
    "p": ("p",),
    "q": ("k",),
    "r": ("r",),
    "s": ("s",),
    "t": ("t",),
    "u": ("U",),
    "v": ("v",),
    "w": ("v",),
    "x": ("k", "s"),
    "y": ("y",),
    "z": ("t", "s"),
    "å": ("o",),
    "ä": ("{",),
    "ö": ("2",),
}

_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class FrontendResult:
    original: str
    normalized: str
    tokens: tuple[str, ...]
    unsupported: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.unsupported

    def as_dict(self) -> dict[str, object]:
        return {
            "original": self.original,
            "normalized": self.normalized,
            "tokens": list(self.tokens),
            "unsupported": list(self.unsupported),
            "ok": self.ok,
        }


def normalize_text(text: str) -> str:
    """Return deterministic Step-1 normalization.

    We deliberately keep semantic text normalization (numbers, abbreviations,
    symbols such as euro signs) for Step 2.  Here we only make Unicode and
    whitespace/case deterministic.
    """
    if not isinstance(text, str):
        raise TypeError("text must be str")
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = _WS_RE.sub(" ", text).strip()
    return text


def tokenize(text: str, *, add_boundaries: bool = True) -> FrontendResult:
    """Normalize and tokenize text into canonical Finnish grapheme tokens.

    Unsupported characters are represented as <UNK:...> and also returned in
    `unsupported`, so no input character disappears silently.
    """
    normalized = normalize_text(text)
    out: list[str] = []
    unsupported: list[str] = []

    if add_boundaries:
        out.append(BOS)

    for ch in normalized:
        if ch in FINNISH_LETTERS:
            out.append(ch)
        elif ch == " ":
            out.append(SPACE)
        elif ch in PUNCTUATION:
            out.append(ch)
        else:
            out.append(f"<UNK:{ch}>")
            if ch not in unsupported:
                unsupported.append(ch)

    if add_boundaries:
        out.append(EOS)

    return FrontendResult(
        original=text,
        normalized=normalized,
        tokens=tuple(out),
        unsupported=tuple(unsupported),
    )


def aalto_compatible_phones(
    text_or_result: str | FrontendResult,
    *,
    punctuation_as_silence: bool = True,
) -> tuple[str, ...]:
    """Render an Aalto-compatible phone sequence for debug comparison.

    This is NOT the representation intended for the future Style-TTS model.
    Unknown Step-2 material (digits, currency symbols, emoji, etc.) becomes
    <UNK> rather than being dropped.
    """
    if isinstance(text_or_result, FrontendResult):
        normalized = text_or_result.normalized
    else:
        normalized = normalize_text(text_or_result)

    phones: list[str] = []
    for ch in normalized:
        if ch in _AALTO_PHONE_MAP:
            phones.extend(_AALTO_PHONE_MAP[ch])
        elif ch == " ":
            # Word boundaries are visually useful in our debug output even
            # though the Aalto lexicon itself stores pronunciations per word.
            phones.append(SPACE)
        elif ch in PUNCTUATION:
            if punctuation_as_silence:
                phones.append("SIL")
        else:
            phones.append("<UNK>")
    return tuple(phones)


def format_tokens(tokens: Sequence[str]) -> str:
    return " ".join(tokens)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Zundanen Finnish Style-TTS research frontend (Step 1)"
    )
    p.add_argument("text", nargs="?", help="Finnish text to inspect")
    p.add_argument(
        "--aalto",
        action="store_true",
        help="also print Aalto-compatible debug phones",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable JSON",
    )
    p.add_argument(
        "--no-boundaries",
        action="store_true",
        help="omit <BOS>/<EOS> from canonical tokens",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    text = args.text
    if text is None:
        try:
            text = input("Finnish text> ")
        except EOFError:
            return 2

    result = tokenize(text, add_boundaries=not args.no_boundaries)

    if args.json:
        payload = result.as_dict()
        if args.aalto:
            payload["aalto_compatible_phones"] = list(aalto_compatible_phones(result))
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"INPUT      : {result.original}")
        print(f"NORMALIZED : {result.normalized}")
        print(f"TOKENS     : {format_tokens(result.tokens)}")
        print(
            "UNSUPPORTED: "
            + ("(none)" if result.ok else " ".join(result.unsupported))
        )
        if args.aalto:
            print(
                "AALTO DBG  : "
                + format_tokens(aalto_compatible_phones(result))
            )

    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
