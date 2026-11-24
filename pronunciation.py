"""Pronunciation pre-processing for Zundanen's Finnish TTS input (automatic, no settings).

Runs on the text before the Finnish normalizer and the models (which lowercase everything and only know
the letters a-z, å, ä, ö):

1. letters the models do not know are folded to near Finnish spellings (é -> e, ü -> y, š -> sh, ß -> ss);
2. abbreviations in capitals (2-5 letters: USA, TV, EU) and single capital letters (C-vitamiini, J. K.)
   are read with the Finnish letter names (USA -> uuäsaa) unless the whole text is written in capitals.

A Finnish inflection ending written after a colon (USA:ssa, TV:n) is attached to the reading
(uuäsaassa, teeveen). If something is read wrongly, the user rewrites it in the text (the app shows the
text as it is read). Only the Python standard library is required.
"""
from __future__ import annotations

import re
import unicodedata

LETTER_NAMES = {
    "A": "aa", "B": "bee", "C": "see", "D": "dee", "E": "ee", "F": "äf", "G": "gee", "H": "hoo", "I": "ii",
    "J": "jii", "K": "koo", "L": "äl", "M": "äm", "N": "än", "O": "oo", "P": "pee", "Q": "kuu", "R": "är",
    "S": "äs", "T": "tee", "U": "uu", "V": "vee", "W": "vee", "X": "äks", "Y": "yy", "Z": "tset",
    "Å": "oo", "Ä": "ää", "Ö": "öö",
}
# capitals read as an ordinary word, not letter by letter: abbreviations said as a word, and short Finnish
# words written in capitals for emphasis ("Hän sanoi EI.")
WORD_ABBREVIATIONS = {"AIDS", "FIFA", "UEFA", "UFO", "YLE", "ELY", "HUS", "ABBA"}
SHORT_WORDS = {"AI", "EI", "EN", "ET", "HE", "JA", "JO", "ME", "NE", "NO", "OI", "ON", "SE", "TE", "OLE", "OLI",
               "NYT", "VOI", "SAA", "MUN", "SUN", "TUO", "KUN", "MUT", "ETTÄ", "MITÄ"}
VOWELS = set("AEIOUYÄÖ")
FOREIGN = set("BCDFGQWXZÅ")
FOLD = {"ß": "ss", "æ": "ä", "Æ": "Ä", "ø": "ö", "Ø": "Ö", "œ": "ö", "Œ": "Ö", "ü": "y", "Ü": "Y",
        "ű": "y", "Ű": "Y", "ő": "ö", "Ő": "Ö", "š": "sh", "Š": "Sh", "ž": "z", "Ž": "Z", "ł": "l", "Ł": "L",
        "đ": "d", "Đ": "D", "þ": "th", "Þ": "Th", "ð": "d", "Ð": "D"}
KNOWN = set("abcdefghijklmnopqrstuvwxyzåäöABCDEFGHIJKLMNOPQRSTUVWXYZÅÄÖ")

_ACRONYM_RE = re.compile(r"(?<![\w:])([A-ZÅÄÖ]{1,5})(?::([a-zåäö]+))?(?![\w])")
_CAPS_WORD_RE = re.compile(r"(?<!\w)[A-ZÅÄÖ]{2,}(?!\w)")
_WORD_RE = re.compile(r"[^\W\d_]+")


def fold_letters(text: str) -> str:
    """Letters outside a-z/åäö -> the nearest Finnish spelling; everything else unchanged."""
    out = []
    for ch in text:
        if ch in KNOWN or not ch.isalpha():
            out.append(ch)
        elif ch in FOLD:
            out.append(FOLD[ch])
        else:
            base = "".join(c for c in unicodedata.normalize("NFKD", ch) if not unicodedata.combining(c))
            out.append(base if base and all(c in KNOWN for c in base) else ch)
    return "".join(out)


def reads_as_word(token: str) -> bool:
    """Capitals that are read as a word: known word-abbreviations, short Finnish words, and 4-5 letter
    tokens that look like a Finnish word (vowels, no foreign letters, no consonant clusters at the edges)."""
    if token in WORD_ABBREVIATIONS or token in SHORT_WORDS:
        return True
    if len(token) <= 3:
        return False
    if not any(c in VOWELS for c in token) or any(c in FOREIGN for c in token):
        return False
    cons = "".join("c" if c not in VOWELS else "v" for c in token)
    return not (cons.startswith("cc") or cons.endswith("cc") or "ccc" in cons)


def _shouting(text: str) -> bool:
    words = _WORD_RE.findall(text)
    caps = _CAPS_WORD_RE.findall(text)
    return len(caps) >= 3 and len(caps) * 2 >= len(words)


def apply(text: str) -> str:
    """Return the text as the TTS should read it."""
    out = fold_letters(text)
    if _shouting(out):
        return out

    def spell(m: re.Match) -> str:
        token, suffix = m.group(1), m.group(2) or ""
        if len(token) > 1 and reads_as_word(token):
            return token.lower() + suffix
        return "".join(LETTER_NAMES[c] for c in token) + suffix

    return _ACRONYM_RE.sub(spell, out)


if __name__ == "__main__":  # quick check: python pronunciation.py "USA:ssa on TV:n C-vitamiini"
    import sys
    print(apply(" ".join(sys.argv[1:]) or "USA:ssa katsottiin TV:tä ja syötiin C-vitamiinia. J. K. Rowling. Café Müller."))
