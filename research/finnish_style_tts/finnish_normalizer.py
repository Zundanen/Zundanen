from __future__ import annotations

"""Conservative Finnish text normalizer for Zundanen Style-TTS research.

Step 2 adds spoken-form normalization *before* the Step-1 grapheme frontend.
The guiding rule is conservative: convert patterns whose intended reading is
clear, and keep ambiguous Finnish inflection/ordinal forms visible with a
warning instead of silently guessing.

Supported in v0.2:
- cardinal integers from -999999 to 999999
- decimal comma (e.g. 3,5)
- euro, percent and a small set of common metric units
- Celsius temperatures
- explicit/bare multi-part software versions (v1.1.0, versio 1.1.0, 1.1.0)
- whitespace / Unicode normalization inherited from Step 1

Intentionally unresolved for now:
- inflected numerals written with a colon (e.g. 11:ssä, 25:een)
- likely ordinal numerals before a lowercase word (e.g. 3. sija)
- ambiguous one-dot numbers without an explicit version marker (e.g. 3.12)
- abbreviations and product-name pronunciation rules

Only the Python standard library is required.
"""

from dataclasses import dataclass
import argparse
import json
import re
import unicodedata
from typing import Sequence

from finnish_frontend import FrontendResult, tokenize

MAX_CARDINAL = 999_999

_ONES = (
    "nolla",
    "yksi",
    "kaksi",
    "kolme",
    "neljä",
    "viisi",
    "kuusi",
    "seitsemän",
    "kahdeksan",
    "yhdeksän",
)

# Longest unit symbols first so "mm" is not partially consumed as "m".
_UNIT_FORMS: dict[str, tuple[str, str]] = {
    "kg": ("kilogramma", "kilogrammaa"),
    "mg": ("milligramma", "milligrammaa"),
    "g": ("gramma", "grammaa"),
    "km": ("kilometri", "kilometriä"),
    "cm": ("senttimetri", "senttimetriä"),
    "mm": ("millimetri", "millimetriä"),
    "m": ("metri", "metriä"),
    "ml": ("millilitra", "millilitraa"),
    "l": ("litra", "litraa"),
}

_WS_RE = re.compile(r"\s+")
_COLON_INFLECTION_RE = re.compile(r"(?<!\w)([+-]?\d+(?:,\d+)?):([a-zåäö]+)\b", re.IGNORECASE)
# Conservative heuristic: a numeral + period directly before a lowercase word
# is probably an ordinal expression. Sentence-final numbers followed by an
# uppercase next sentence are therefore not protected by this rule.
_ORDINAL_RE = re.compile(r"(?<!\w)(\d+)\.\s+(?=[a-zåäö])")
_AMBIGUOUS_ONE_DOT_RE = re.compile(r"(?<![\w.])\d+\.\d+(?![\d.])")
_EXPLICIT_VERSION_RE = re.compile(
    r"\b(?:v|versio\s+)(\d+(?:\.\d+)+)\b", re.IGNORECASE
)
_MULTI_DOT_VERSION_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+){2,})(?![\d.])")
_NUMBER_UNIT_RE = re.compile(
    r"(?<!\w)([+-]?\d+(?:,\d+)?)\s*(°\s*c|€|%|kg|mg|km|cm|mm|ml|g|m|l)(?![a-zåäö])",
    re.IGNORECASE,
)
_DECIMAL_RE = re.compile(r"(?<![\w\d])([+-]?\d+),(\d+)(?![\w\d])")
_INTEGER_RE = re.compile(r"(?<![\w\d])([+-]?\d+)(?![\w\d])")


@dataclass(frozen=True)
class NormalizationWarning:
    code: str
    text: str

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "text": self.text}


@dataclass(frozen=True)
class NormalizationResult:
    original: str
    normalized: str
    warnings: tuple[NormalizationWarning, ...]
    frontend: FrontendResult

    @property
    def ok(self) -> bool:
        return not self.warnings and self.frontend.ok

    def as_dict(self) -> dict[str, object]:
        return {
            "original": self.original,
            "normalized": self.normalized,
            "warnings": [w.as_dict() for w in self.warnings],
            "frontend": self.frontend.as_dict(),
            "ok": self.ok,
        }


def _under_100(n: int) -> str:
    if n < 10:
        return _ONES[n]
    if n == 10:
        return "kymmenen"
    if n < 20:
        return _ONES[n - 10] + "toista"
    tens, ones = divmod(n, 10)
    out = _ONES[tens] + "kymmentä"
    if ones:
        out += _ONES[ones]
    return out


def _under_1000(n: int) -> str:
    if n < 100:
        return _under_100(n)
    hundreds, rest = divmod(n, 100)
    if hundreds == 1:
        out = "sata"
    else:
        out = _ONES[hundreds] + "sataa"
    if rest:
        out += _under_100(rest)
    return out


def _positive_cardinal(n: int) -> str:
    if n < 1000:
        return _under_1000(n)
    thousands, rest = divmod(n, 1000)
    if thousands == 1:
        out = "tuhat"
    else:
        out = _under_1000(thousands) + "tuhatta"
    if rest:
        out += _under_1000(rest)
    return out


def number_to_words(n: int) -> str:
    """Convert a cardinal integer to a Finnish spoken form.

    Step 2 intentionally limits the range. Larger values remain unresolved in
    the main normalizer instead of receiving a guessed or partial reading.
    """
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError("n must be int")
    if abs(n) > MAX_CARDINAL:
        raise ValueError(f"supported cardinal range is ±{MAX_CARDINAL}")
    if n == 0:
        return _ONES[0]
    if n < 0:
        return "miinus " + _positive_cardinal(-n)
    return _positive_cardinal(n)


def _fraction_digits_to_words(digits: str) -> str:
    # Read decimal digits individually so leading/trailing zeroes are never
    # lost: 5,50 -> "viisi pilkku viisi nolla".
    return " ".join(_ONES[int(ch)] for ch in digits)


def decimal_to_words(token: str) -> str:
    if "," not in token:
        return number_to_words(int(token))
    whole, frac = token.split(",", 1)
    if not whole or not frac or not frac.isdigit():
        raise ValueError(f"invalid decimal token: {token!r}")
    return number_to_words(int(whole)) + " pilkku " + _fraction_digits_to_words(frac)


def _numeric_is_exactly_one(token: str) -> bool:
    # Avoid float semantics. 1,0 and 1,00 count as exactly one.
    signless = token.lstrip("+")
    if signless.startswith("-"):
        return False
    if "," in signless:
        whole, frac = signless.split(",", 1)
        return whole == "1" and bool(frac) and set(frac) <= {"0"}
    return int(signless) == 1


def _version_group_to_words(group: str) -> str:
    # Preserve leading zeroes in version components (e.g. 1.01).
    if len(group) > 1 and group.startswith("0"):
        return " ".join(_ONES[int(ch)] for ch in group)
    value = int(group)
    if value <= MAX_CARDINAL:
        return number_to_words(value)
    return " ".join(_ONES[int(ch)] for ch in group)


def version_to_words(version: str) -> str:
    groups = version.split(".")
    if len(groups) < 2 or any(not g.isdigit() for g in groups):
        raise ValueError(f"invalid version: {version!r}")
    return "versio " + " piste ".join(_version_group_to_words(g) for g in groups)


class _Protector:
    """Temporarily protect ambiguous spans while safe patterns are expanded."""

    def __init__(self) -> None:
        self._items: list[tuple[str, str]] = []

    def put(self, text: str) -> str:
        idx = len(self._items)
        if idx >= 0x1900:
            raise RuntimeError("too many protected spans")
        # Use a single Private Use Area codepoint with no ASCII digits inside
        # it, so later numeric substitutions cannot corrupt the placeholder.
        marker = chr(0xE000 + idx)
        self._items.append((marker, text))
        return marker

    def restore(self, text: str) -> str:
        for marker, original in self._items:
            text = text.replace(marker, original)
        return text


def _base_normalize(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be str")
    text = unicodedata.normalize("NFC", text)
    # Keep casing during pattern recognition; Step 1 lowercases later.
    text = _WS_RE.sub(" ", text).strip()
    return text


def normalize_finnish(text: str) -> tuple[str, tuple[NormalizationWarning, ...]]:
    """Normalize clear Finnish TTS patterns and report unresolved ambiguity."""
    work = _base_normalize(text)
    warnings: list[NormalizationWarning] = []
    protector = _Protector()

    # 1) Protect forms where Finnish morphology carries information that a
    # naive cardinal replacement would destroy.
    def protect_colon(m: re.Match[str]) -> str:
        raw = m.group(0)
        warnings.append(
            NormalizationWarning(
                "inflected_numeral",
                f"Inflected numeral left unresolved: {raw}",
            )
        )
        return protector.put(raw)

    work = _COLON_INFLECTION_RE.sub(protect_colon, work)

    def protect_ordinal(m: re.Match[str]) -> str:
        raw = m.group(0)
        warnings.append(
            NormalizationWarning(
                "ordinal_numeral",
                f"Possible ordinal numeral left unresolved: {raw.strip()}",
            )
        )
        return protector.put(raw)

    work = _ORDINAL_RE.sub(protect_ordinal, work)

    # 2) Explicit and clearly multi-part software versions.
    def repl_explicit_version(m: re.Match[str]) -> str:
        return version_to_words(m.group(1))

    work = _EXPLICIT_VERSION_RE.sub(repl_explicit_version, work)

    def repl_multi_dot_version(m: re.Match[str]) -> str:
        return version_to_words(m.group(1))

    work = _MULTI_DOT_VERSION_RE.sub(repl_multi_dot_version, work)

    # Protect ambiguous one-dot numeric forms (3.12). A Finnish decimal should
    # normally use a comma; the dot could instead mean version/date/time.
    def protect_one_dot(m: re.Match[str]) -> str:
        raw = m.group(0)
        warnings.append(
            NormalizationWarning(
                "ambiguous_dotted_number",
                f"Ambiguous dotted number left unresolved: {raw} (use v{raw} for a version)",
            )
        )
        return protector.put(raw)

    work = _AMBIGUOUS_ONE_DOT_RE.sub(protect_one_dot, work)

    # 3) Number + common unit/currency/percent.
    def repl_number_unit(m: re.Match[str]) -> str:
        num = m.group(1)
        unit = re.sub(r"\s+", "", m.group(2).lower())
        try:
            spoken_num = decimal_to_words(num)
        except (ValueError, OverflowError):
            warnings.append(
                NormalizationWarning(
                    "number_out_of_range",
                    f"Number left unresolved: {num}",
                )
            )
            return protector.put(m.group(0))

        singular = _numeric_is_exactly_one(num)
        if unit == "€":
            noun = "euro" if singular else "euroa"
        elif unit == "%":
            noun = "prosentti" if singular else "prosenttia"
        elif unit in {"°c", "° c"}:
            noun = "celsiusaste" if singular else "celsiusastetta"
        else:
            forms = _UNIT_FORMS[unit]
            noun = forms[0] if singular else forms[1]
        return f"{spoken_num} {noun}"

    work = _NUMBER_UNIT_RE.sub(repl_number_unit, work)

    # 4) Remaining decimals and integers.
    def repl_decimal(m: re.Match[str]) -> str:
        token = m.group(1) + "," + m.group(2)
        try:
            return decimal_to_words(token)
        except ValueError:
            warnings.append(
                NormalizationWarning(
                    "number_out_of_range",
                    f"Number left unresolved: {token}",
                )
            )
            return protector.put(token)

    work = _DECIMAL_RE.sub(repl_decimal, work)

    def repl_integer(m: re.Match[str]) -> str:
        token = m.group(1)
        try:
            return number_to_words(int(token))
        except ValueError:
            warnings.append(
                NormalizationWarning(
                    "number_out_of_range",
                    f"Number left unresolved: {token}",
                )
            )
            return protector.put(token)

    work = _INTEGER_RE.sub(repl_integer, work)

    work = protector.restore(work)
    work = _WS_RE.sub(" ", work).strip()
    # The Step-1 frontend owns canonical lowercasing. Return the same canonical
    # text here so callers see exactly what will be tokenized.
    work = work.lower()
    return work, tuple(warnings)


def normalize_and_tokenize(text: str) -> NormalizationResult:
    normalized, warnings = normalize_finnish(text)
    frontend = tokenize(normalized)
    return NormalizationResult(
        original=text,
        normalized=normalized,
        warnings=warnings,
        frontend=frontend,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Zundanen Finnish Style-TTS text normalizer (Step 2)"
    )
    p.add_argument("text", nargs="?", help="Finnish text to normalize")
    p.add_argument("--json", action="store_true", help="emit JSON")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    text = args.text
    if text is None:
        try:
            text = input("Finnish text> ")
        except EOFError:
            return 2

    result = normalize_and_tokenize(text)
    if args.json:
        print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
    else:
        print(f"INPUT      : {result.original}")
        print(f"NORMALIZED : {result.normalized}")
        if result.warnings:
            print("WARNINGS   :")
            for warning in result.warnings:
                print(f"  [{warning.code}] {warning.text}")
        else:
            print("WARNINGS   : (none)")
        print("TOKENS     : " + " ".join(result.frontend.tokens))
        print(
            "UNSUPPORTED: "
            + (
                "(none)"
                if result.frontend.ok
                else " ".join(result.frontend.unsupported)
            )
        )

    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
