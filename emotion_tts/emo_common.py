from __future__ import annotations

"""v60 shared definitions: Japanese LAB -> Finnish-orthography tokens.

Original Zundanen code (MIT).  No Style-Bert-VITS2 / AGPL source is used.

Idea
----
Finnish orthography is almost phonemic, and it marks length exactly like the
OpenJTalk LAB stream does: a long vowel is a doubled vowel ("oo") and a
geminate is a doubled consonant ("tt").  Japanese phones are therefore written
in the *existing* Finnish VITS grapheme vocabulary wherever an honest Finnish letter
exists.  Japanese "a" and Finnish "a" share one token embedding, so a style that
is learned on Japanese tokens acts on the same token rows that Finnish text uses.

Only two new rows are added, both real Finnish loan letters:
  š (sh, ch = t+š)   ž (j = d+ž)
Language-specific realisation (Japanese [ɯ] vs Finnish [u], flap vs trill r)
is left to the explicit language embedding.  Approximations are listed in
APPROXIMATE so the audit report can show them instead of hiding them.
"""

import csv
import re
import sys
from pathlib import Path

# Windows consoles default to cp932; š/ž must never crash a log line.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="backslashreplace")
    except (AttributeError, ValueError):
        pass

BOS, EOS, SPACE, PAD = "<BOS>", "<EOS>", "<SP>", "<PAD>"

NEW_TOKENS = {"š": "s", "ž": "z"}  # new token -> existing row used for init

SPEAKERS = ("fi_chatterbox", "zundamon", "metan", "itako")
# v71: extra speakers appended for a run (e.g. ZUNDANEN_EXTRA_SPEAKERS=sora).  Unset = unchanged.
SPEAKERS = SPEAKERS + tuple(s.strip() for s in __import__("os").environ.get("ZUNDANEN_EXTRA_SPEAKERS", "").split(",")
                           if s.strip() and s.strip() not in SPEAKERS)
STYLES = ("normal", "ama", "sexy", "tsun")
LANGS = ("fi", "ja")

DATASET_SPEAKER_DIRS = {"zundamon": "Zundamon", "metan": "Metan", "itako": "Itako"}
# style -> (wav dir name, lab dir name, file prefix)
STYLE_LAYOUT = {
    "normal": ("normal_wav", "normal_lab", "emoNormal"),
    "ama": ("ama_wav", "ama_lab", "emoAma"),
    "sexy": ("sexy_wab", "sexy_lab", "emoSexy"),
    "tsun": ("tsun_wav", "tsun_lab", "emoTsun"),
}

# One Japanese phone -> Finnish-orthography token(s).
_DIRECT = {p: (p,) for p in "aiueokgsztdnhbpmrf"}
_DIRECT.update({
    "y": ("j",),
    "w": ("v",),
    "v": ("v",),
    "N": ("n",),
    "sh": ("š",),
    "ch": ("t", "š"),
    "ts": ("t", "s"),
    "j": ("d", "ž"),
})
for _c in ("k", "g", "n", "h", "m", "r", "b", "p", "d", "t", "f"):
    _DIRECT[_c + "y"] = (_c, "j")
_DIRECT["sy"] = ("š",)
_DIRECT["zy"] = ("d", "ž")

APPROXIMATE = {
    "u": "JA [ɯ] unrounded vs FI [u]; separated by language embedding",
    "w": "JA [ɰ] written as FI v [ʋ]",
    "f": "JA [ɸ] written as FI f",
    "r": "JA flap vs FI trill; same letter",
    "h": "JA hi/hy [ç] shares FI h",
    "N": "moraic nasal written as n (FI n before k/g is already [ŋ])",
    "j": "JA [dʑ] written d+ž",
    "ch": "JA [tɕ] written t+š",
    "sh": "JA [ɕ] written š",
}

SILENCE = {"sil", "pau"}
LAB_TIME_UNIT = 1e-7  # HTK 100 ns


def read_lab_phones(path: Path) -> list[str]:
    phones = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) >= 3:
            phones.append(parts[2])
        elif len(parts) == 1:
            phones.append(parts[0])
    return phones


# Unambiguous LAB typos seen in the corpus.  Each repair is reported in audit.md.
LAB_REPAIRS = {"_k": ("k",), "_o": ("o",), "_e": ("e",), "__N": ("N",), "ye": ("y", "e"), "S": ()}


def repair_lab_phones(phones: list[str]) -> tuple[list[str], list[str]]:
    out, notes = [], []
    for p in phones:
        if p in LAB_REPAIRS:
            out.extend(LAB_REPAIRS[p])
            notes.append(f"{p!r}->{' '.join(LAB_REPAIRS[p]) or '(dropped)'}")
        else:
            out.append(p)
    return out, notes


def lab_to_tokens(phones: list[str]) -> list[str]:
    """Japanese OpenJTalk LAB phones -> Finnish VITS grapheme tokens (+ š/ž).

    sil at the edges becomes BOS/EOS, inner pau becomes ",", and cl doubles
    the first letter of the next consonant (Finnish geminate spelling).
    Raises ValueError for phones the mapping does not know.
    """
    core = list(phones)
    while core and core[0] in SILENCE:
        core.pop(0)
    while core and core[-1] in SILENCE:
        core.pop()
    if not core:
        raise ValueError("LAB has no speech phones")

    out = [BOS]
    pending_cl = False
    for p in core:
        if p in SILENCE:
            if out[-1] != ",":
                out.append(",")
            pending_cl = False
            continue
        if p == "cl":
            pending_cl = True
            continue
        mapped = _DIRECT.get(p)
        if mapped is None:
            raise ValueError(f"unknown Japanese LAB phone {p!r}")
        if pending_cl:
            first = mapped[0]
            # Geminate only consonants; a glottal stop before a vowel / at an
            # odd place is simply dropped.
            if first not in "aiueo":
                out.append(first)
            pending_cl = False
        out.extend(mapped)
    out.append(EOS)
    return out


def sentence_split(sentence_id: int, *, test_ids: set[int], valid_ids: set[int]) -> str:
    if sentence_id in test_ids:
        return "test"
    if sentence_id in valid_ids:
        return "valid"
    return "train"


# 007 and 060 are the project's historical holdouts and never train.
DEFAULT_TEST_IDS = {7, 15, 32, 48, 60, 73, 86, 99}
DEFAULT_VALID_IDS = {21, 54, 91}


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = fieldnames or list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=names, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


_ID_RE = re.compile(r"^emo[A-Za-z]+(\d{3})\.wav$")


def sentence_id_from_wav(path: Path) -> int | None:
    """emoAma012.wav -> 12. Duplicates like 'emoAma090 (1).wav' return None."""
    m = _ID_RE.match(Path(path).name)
    return int(m.group(1)) if m else None
