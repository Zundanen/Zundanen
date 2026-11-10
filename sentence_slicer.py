from __future__ import annotations

import re
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

from aalto_alignment import run_alignment

WORD_RE = re.compile(r"[^\W_]+(?:[-’'][^\W_]+)*", re.UNICODE)
COMMON_ABBREVIATIONS = {
    "esim.", "jne.", "tms.", "ym.", "yms.", "nro.", "klo.", "v.", "s.", "lk.", "mm.", "vrt.",
}
CLOSERS = '"\'”’»)]}'
OPENERS = '"\'“‘«([{'


@dataclass(frozen=True)
class SentenceSegment:
    text: str
    word_start: int
    word_end: int  # exclusive


@dataclass(frozen=True)
class SliceFile:
    index: int
    text: str
    wav: Path
    start: float
    end: float
    source_wav: Path | None = None


def source_word_tokens(text: str) -> list[str]:
    return WORD_RE.findall(str(text))


def _preceding_token(text: str, punctuation_start: int) -> str:
    prefix = text[:punctuation_start].rstrip()
    if not prefix:
        return ""
    m = re.search(r"([^\s]+)$", prefix)
    return (m.group(1) if m else "") + "."


def split_text_sentences(text: str) -> list[SentenceSegment]:
    """Split on . ! ? only when there is a word on both sides.

    Conservative rules avoid decimal points, common Finnish abbreviations and
    punctuation not followed by another sentence. Closing quote/bracket marks
    stay with the preceding sentence.
    """
    raw = str(text).strip()
    if not raw:
        return []

    boundaries: list[int] = []
    i = 0
    n = len(raw)
    while i < n:
        if raw[i] not in ".!?":
            i += 1
            continue

        p0 = i
        while i + 1 < n and raw[i + 1] in ".!?":
            i += 1
        p1 = i + 1

        # Include closing quote/bracket characters in the preceding sentence.
        j = p1
        while j < n and raw[j] in CLOSERS:
            j += 1

        # A sentence boundary must have whitespace then a real next token.
        k = j
        if k >= n or not raw[k].isspace():
            i += 1
            continue
        while k < n and raw[k].isspace():
            k += 1
        while k < n and raw[k] in OPENERS:
            k += 1
        if k >= n or not raw[k].isalnum():
            i += 1
            continue

        # Must also have a word-like character before the punctuation.
        left = raw[:p0].rstrip()
        if not left or not left[-1].isalnum():
            i += 1
            continue

        punct = raw[p0:p1]
        if punct == ".":
            # Avoid Finnish abbreviations such as "esim. tämä".
            token = _preceding_token(raw, p0).casefold()
            if token in COMMON_ABBREVIATIONS:
                i += 1
                continue
            # Single-letter initials are too ambiguous to cut safely.
            last_word = source_word_tokens(left[-20:])
            if last_word and len(last_word[-1]) == 1 and last_word[-1].isalpha():
                i += 1
                continue

        boundaries.append(j)
        i = j

    if not boundaries:
        words = source_word_tokens(raw)
        return [SentenceSegment(raw, 0, len(words))] if words else []

    parts: list[str] = []
    start = 0
    for end in boundaries:
        part = raw[start:end].strip()
        if part:
            parts.append(part)
        start = end
    tail = raw[start:].strip()
    if tail:
        parts.append(tail)

    if len(parts) <= 1:
        words = source_word_tokens(raw)
        return [SentenceSegment(raw, 0, len(words))] if words else []

    segments: list[SentenceSegment] = []
    pos = 0
    for part in parts:
        wc = len(source_word_tokens(part))
        if wc <= 0:
            continue
        segments.append(SentenceSegment(part, pos, pos + wc))
        pos += wc
    return segments


def _norm_word(word: str) -> str:
    return "".join(ch for ch in str(word).casefold() if ch.isalnum() or ch in "åäöšž")


def map_source_to_aligned(source_words: list[str], aligned_words: list[dict]) -> list[int]:
    """Map each transcript word index to an Aalto word index, conservatively."""
    displays = [str(w.get("display_word") or w.get("word") or "") for w in aligned_words]
    if len(source_words) == len(displays):
        return list(range(len(source_words)))

    src = [_norm_word(x) for x in source_words]
    ali = [_norm_word(x) for x in displays]
    mapping: list[int] = []
    ai = 0
    for sw in src:
        if not sw:
            raise ValueError("Transcript contains an unmappable empty word token.")
        found = None
        for j in range(ai, min(len(ali), ai + 5)):
            if ali[j] == sw:
                found = j
                break
        if found is None:
            raise ValueError(
                "Aalto word count/token mapping differs from the transcript; "
                "sentence slicing was skipped to avoid cutting a word."
            )
        mapping.append(found)
        ai = found + 1
    return mapping


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


def _write_wave_slice(src: Path, dst: Path, start_sec: float, end_sec: float) -> None:
    """Copy a PCM WAV interval without resampling/re-encoding."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    try:
        with wave.open(str(src), "rb") as rf:
            params = rf.getparams()
            rate = rf.getframerate()
            total = rf.getnframes()
            start_frame = max(0, min(total, int(round(start_sec * rate))))
            end_frame = max(start_frame, min(total, int(round(end_sec * rate))))
            rf.setpos(start_frame)
            data = rf.readframes(end_frame - start_frame)
            with wave.open(str(tmp), "wb") as wf:
                wf.setparams(params)
                wf.writeframes(data)
        tmp.replace(dst)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def _sentence_cut_times(segments: list[SentenceSegment], words: list[dict], mapping: list[int], duration: float) -> list[float]:
    cuts = [0.0]
    for prev_seg, next_seg in zip(segments, segments[1:]):
        prev_src = prev_seg.word_end - 1
        next_src = next_seg.word_start
        prev_word = words[mapping[prev_src]]
        next_word = words[mapping[next_src]]
        prev_end = float(prev_word.get("end", 0.0))
        next_start = float(next_word.get("start", prev_end))
        cut = (prev_end + next_start) / 2.0
        cut = max(cuts[-1] + 0.001, min(duration - 0.001, cut))
        cuts.append(cut)
    cuts.append(duration)
    return cuts


def slice_generated_item(
    *,
    final_wav: Path,
    transcript: str,
    alignment_wav: Path,
    source_wav: Path | None = None,
    create_subtitle_txt: bool = True,
) -> list[SliceFile]:
    """Align a multi-sentence utterance and replace the full WAV with sentence WAVs.

    Safety behavior:
    - If text has only one sentence, returns [] and does nothing.
    - Original files are deleted only after every slice has been written.
    - Any exception leaves the original full WAV intact.
    """
    final_wav = Path(final_wav)
    alignment_wav = Path(alignment_wav)
    source_wav = Path(source_wav) if source_wav else None

    segments = split_text_sentences(transcript)
    if len(segments) <= 1:
        return []
    if not final_wav.is_file():
        raise FileNotFoundError(f"Final WAV not found: {final_wav}")
    if not alignment_wav.is_file():
        raise FileNotFoundError(f"Alignment WAV not found: {alignment_wav}")

    alignment = run_alignment(alignment_wav, transcript)
    words = alignment.get("words") or []
    if not words:
        raise ValueError("Aalto returned no aligned words.")

    src_words = source_word_tokens(transcript)
    if not src_words:
        raise ValueError("Transcript has no word tokens.")
    mapping = map_source_to_aligned(src_words, words)

    align_duration = wav_duration(alignment_wav)
    final_duration = wav_duration(final_wav)
    if align_duration <= 0 or final_duration <= 0:
        raise ValueError("Generated WAV has zero duration.")

    align_cuts = _sentence_cut_times(segments, words, mapping, align_duration)
    scale = final_duration / align_duration
    final_cuts = [max(0.0, min(final_duration, x * scale)) for x in align_cuts]
    final_cuts[0] = 0.0
    final_cuts[-1] = final_duration

    final_paths: list[Path] = []
    source_paths: list[Path | None] = []
    try:
        for idx, seg in enumerate(segments, 1):
            out = final_wav.with_name(f"{final_wav.stem}_s{idx:02d}{final_wav.suffix}")
            _write_wave_slice(final_wav, out, final_cuts[idx - 1], final_cuts[idx])
            final_paths.append(out)

            source_out = None
            if source_wav is not None and source_wav.is_file() and source_wav.resolve() != final_wav.resolve():
                source_out = source_wav.with_name(
                    f"{final_wav.stem}_s{idx:02d}_source{source_wav.suffix}"
                )
                _write_wave_slice(source_wav, source_out, align_cuts[idx - 1], align_cuts[idx])
            source_paths.append(source_out)

            if create_subtitle_txt:
                out.with_suffix(".txt").write_text(seg.text, encoding="utf-8")

        # Commit only after all outputs exist and are non-empty.
        for p in final_paths:
            if not p.is_file() or p.stat().st_size <= 44:
                raise IOError(f"Sentence slice was not written correctly: {p}")

        final_wav.unlink(missing_ok=True)
        final_wav.with_suffix(".txt").unlink(missing_ok=True)
        if source_wav is not None and source_wav.is_file() and source_wav.resolve() != final_wav.resolve():
            source_wav.unlink(missing_ok=True)

        return [
            SliceFile(
                index=i,
                text=segments[i - 1].text,
                wav=final_paths[i - 1],
                start=round(final_cuts[i - 1], 4),
                end=round(final_cuts[i], 4),
                source_wav=source_paths[i - 1],
            )
            for i in range(1, len(segments) + 1)
        ]
    except Exception:
        # Roll back partial slices. Full original files were not deleted yet.
        for p in final_paths:
            try:
                p.unlink(missing_ok=True)
                p.with_suffix(".txt").unlink(missing_ok=True)
            except OSError:
                pass
        for p in source_paths:
            if p is not None:
                try:
                    p.unlink(missing_ok=True)
                except OSError:
                    pass
        raise
