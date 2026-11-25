from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf


FRAME_PERIOD_MS = 5.0
MIN_DURATION_SCALE = 0.50
MAX_DURATION_SCALE = 1.60
MIN_STRENGTH = -50.0
MAX_STRENGTH = 50.0

# Values at 100%. 0% = no change; the slider stops at +50% (half of these), which is the strongest accent offered.
MAX_PRESET = {
    "peak_semitones": 7.5,
    "word_deaccent_semitones": -3.0,
    "duration_percent": 25.0,
    "gain_db": 20.0,
}

# Negative strength weakens the selected syllable instead; the rest of the word is left alone. At -50% the
# syllable's pitch movement relative to the word's median F0 is reduced by `flatten`, and the syllable gets the
# duration and gain changes below. Values in between scale linearly.
WEAKEN_PRESET = {
    "flatten": 0.8,
    "duration_percent": -12.5,
    "gain_db": -6.0,
}

# The WORLD-resynthesised audio is used only inside the edited words (plus this margin); everything else keeps the
# original samples, because WORLD resynthesis makes the whole utterance sound noisier.
SPLICE_MARGIN_S = 0.03
SPLICE_FADE_S = 0.02

# Net pitch target inside the selected syllable. The whole word is first
# de-accented, then this selected-syllable contour replaces that offset so the
# accented syllable reaches the intended rise-fall contour relative to the
# original F0.
ACCENT_ENVELOPE = (
    (0.00, -0.16),
    (0.12, 0.05),
    (0.28, 0.62),
    (0.45, 1.00),
    (0.60, 0.72),
    (0.78, 0.10),
    (1.00, -0.30),
)


def _params(strength: float) -> dict[str, float]:
    """Positive: MAX_PRESET * strength/100 (same scale as before; capped at +50%).
    Negative: WEAKEN_PRESET scaled 0..1 over 0..-50%."""
    strength = max(MIN_STRENGTH, min(MAX_STRENGTH, float(strength)))
    if strength < 0.0:
        amount = strength / MIN_STRENGTH
        return {key: float(value) * amount for key, value in WEAKEN_PRESET.items()}
    amount = strength / 100.0
    return {key: float(value) * amount for key, value in MAX_PRESET.items()}


def _edge_shape(rel: np.ndarray, edge: float = 0.18) -> np.ndarray:
    """1 inside the syllable, raised-cosine ramps over the first/last `edge` of it."""
    shape = np.ones(rel.size, dtype=np.float64)
    left = rel < edge
    right = rel > (1.0 - edge)
    if np.any(left):
        shape[left] = 0.5 - 0.5 * np.cos(np.pi * rel[left] / edge)
    if np.any(right):
        shape[right] = 0.5 - 0.5 * np.cos(np.pi * np.clip((1.0 - rel[right]) / edge, 0.0, 1.0))
    return shape


def _load_accents(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("WORLD accent spec must be a JSON list.")

    accents: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            start = float(item.get("start", 0.0))
            end = float(item.get("end", start))
            word_start = float(item.get("word_start", start))
            word_end = float(item.get("word_end", end))
            strength = float(item.get("strength", 0.0))
        except (TypeError, ValueError):
            continue
        strength = max(MIN_STRENGTH, min(MAX_STRENGTH, strength))
        if abs(strength) < 1e-9:
            continue
        if not all(math.isfinite(v) for v in (start, end, word_start, word_end, strength)):
            continue
        if end <= start:
            continue
        if word_end <= word_start:
            word_start, word_end = start, end
        accents.append({
            "start": max(0.0, start),
            "end": max(0.0, end),
            "word_start": max(0.0, min(word_start, start)),
            "word_end": max(end, word_end),
            "strength": strength,
            "word_index": item.get("word_index"),
            "syllable_index": item.get("syllable_index"),
        })
    accents.sort(key=lambda x: (x["start"], x["end"]))
    return accents


def _smooth_curve(curve: np.ndarray, radius: int) -> np.ndarray:
    if curve.size == 0 or radius <= 0 or not np.any(np.abs(curve) > 1e-9):
        return curve.astype(np.float64, copy=False)
    up = np.arange(1, radius + 2, dtype=np.float64)
    kernel = np.concatenate((up, up[-2::-1]))
    kernel /= np.sum(kernel)
    return np.convolve(curve, kernel, mode="same")


def _build_source_pitch_curve(
    frame_count: int, frame_seconds: float, accents: list[dict], f0: np.ndarray | None = None
) -> np.ndarray:
    curve = np.zeros(frame_count, dtype=np.float64)
    if frame_count <= 0:
        return curve

    times = np.arange(frame_count, dtype=np.float64) * frame_seconds
    f0_st = None
    if f0 is not None and len(f0) == frame_count:
        f0 = np.asarray(f0, dtype=np.float64)
        f0_st = np.where(f0 > 0.0, 12.0 * np.log2(np.maximum(f0, 1e-9)), np.nan)
    for accent in accents:
        if accent["strength"] < 0.0:
            # Weaken: pull the selected syllable's F0 toward the median F0 of the word's voiced frames.
            if f0_st is None:
                continue
            ws, we = float(accent["word_start"]), float(accent["word_end"])
            ss, se = float(accent["start"]), float(accent["end"])
            word_st = f0_st[(times >= ws) & (times < we)]
            word_st = word_st[np.isfinite(word_st)]
            selected = np.where((times >= ss) & (times <= se))[0]
            if word_st.size == 0 or selected.size == 0 or se <= ss:
                continue
            deviation = np.nan_to_num(f0_st[selected] - float(np.median(word_st)), nan=0.0)
            rel = (times[selected] - ss) / max(1e-9, se - ss)
            amount = _params(accent["strength"])["flatten"]
            curve[selected] = -amount * deviation * _edge_shape(rel)
            continue

        preset = _params(accent["strength"])
        deaccent = float(preset["word_deaccent_semitones"])
        peak = float(preset["peak_semitones"])
        ws = float(accent["word_start"])
        we = float(accent["word_end"])
        ss = float(accent["start"])
        se = float(accent["end"])

        # Whole-word lowering provides contrast with the chosen syllable.
        word_mask = (times >= ws) & (times < we)
        curve[word_mask] = deaccent

        # Selected syllable gets a net rise-fall contour relative to original F0.
        if se <= ss:
            continue
        selected = np.where((times >= ss) & (times <= se))[0]
        if selected.size == 0:
            continue
        rel = (times[selected] - ss) / max(1e-9, se - ss)
        env_x = np.array([p[0] for p in ACCENT_ENVELOPE], dtype=np.float64)
        env_y = np.array([p[1] for p in ACCENT_ENVELOPE], dtype=np.float64)
        curve[selected] = peak * np.interp(rel, env_x, env_y)

    # A short transition prevents hard F0 jumps at word/syllable boundaries.
    radius = max(1, int(round(0.020 / frame_seconds)))
    return _smooth_curve(curve, radius)


def _build_time_map(frame_count: int, frame_seconds: float, accents: list[dict]) -> tuple[np.ndarray, dict]:
    """Map output WORLD frames back to source frames, stretching selected syllables."""
    if frame_count <= 0:
        return np.zeros(0, dtype=np.float64), {}

    maps: list[np.ndarray] = []
    cursor = 0
    used_segments = 0
    duration_changed = 0

    for accent in accents:
        start_i = int(round(float(accent["start"]) / frame_seconds))
        end_i = int(round(float(accent["end"]) / frame_seconds))
        start_i = max(cursor, min(frame_count, max(0, start_i)))
        end_i = max(0, min(frame_count, end_i))
        if end_i <= start_i:
            continue

        if start_i > cursor:
            maps.append(np.arange(cursor, start_i, dtype=np.float64))

        source_count = end_i - start_i
        scale = 1.0 + _params(accent["strength"])["duration_percent"] / 100.0
        scale = max(MIN_DURATION_SCALE, min(MAX_DURATION_SCALE, scale))
        target_count = max(1, int(round(source_count * scale)))
        if source_count == 1:
            local_map = np.full(target_count, float(start_i), dtype=np.float64)
        else:
            local_map = np.linspace(float(start_i), float(end_i - 1), target_count, dtype=np.float64)
        maps.append(local_map)
        cursor = end_i
        used_segments += 1
        if target_count != source_count:
            duration_changed += 1

    if cursor < frame_count:
        maps.append(np.arange(cursor, frame_count, dtype=np.float64))

    source_map = np.concatenate(maps) if maps else np.arange(frame_count, dtype=np.float64)
    return source_map, {
        "used_segments": used_segments,
        "duration_changed_segments": duration_changed,
        "source_frames": int(frame_count),
        "output_frames": int(source_map.size),
    }


def _linear_rows(matrix: np.ndarray, positions: np.ndarray) -> np.ndarray:
    if matrix.ndim != 2:
        raise ValueError("WORLD spectral parameters must be two-dimensional.")
    if positions.size == 0:
        return matrix[:0].copy()
    max_i = matrix.shape[0] - 1
    lo = np.clip(np.floor(positions).astype(np.int64), 0, max_i)
    hi = np.clip(lo + 1, 0, max_i)
    frac = (positions - lo).astype(np.float64)[:, None]
    return matrix[lo] * (1.0 - frac) + matrix[hi] * frac


def _nearest_1d(values: np.ndarray, positions: np.ndarray) -> np.ndarray:
    if positions.size == 0:
        return values[:0].copy()
    idx = np.clip(np.rint(positions).astype(np.int64), 0, len(values) - 1)
    return np.asarray(values[idx], dtype=np.float64).copy()


def _build_source_gain_curve(frame_count: int, frame_seconds: float, accents: list[dict]) -> np.ndarray:
    gain_db = np.zeros(frame_count, dtype=np.float64)
    if frame_count <= 0:
        return gain_db
    times = np.arange(frame_count, dtype=np.float64) * frame_seconds
    for accent in accents:
        start = float(accent["start"])
        end = float(accent["end"])
        if end <= start:
            continue
        target_db = float(_params(accent["strength"])["gain_db"])
        idx = np.where((times >= start) & (times <= end))[0]
        if idx.size == 0:
            continue
        rel = (times[idx] - start) / max(1e-9, end - start)
        # Smooth raised-cosine-ish entry/exit. Most of the syllable remains at
        # full requested gain so high strengths are clearly audible.
        shaped = target_db * _edge_shape(rel)
        if target_db >= 0.0:
            gain_db[idx] = np.maximum(gain_db[idx], shaped)
        else:
            gain_db[idx] = np.minimum(gain_db[idx], shaped)
    return gain_db


def _splice_original(
    original: np.ndarray, synthesized: np.ndarray, source_map: np.ndarray, frame_count: int,
    frame_seconds: float, sample_rate: int, accents: list[dict],
) -> tuple[np.ndarray, float]:
    """Keep the original samples outside the edited words; use WORLD audio only inside them (with crossfades).
    Outside the edited syllables the time map is the identity plus a whole-frame shift, so those samples are copied
    unchanged. Returns the spliced audio and the share of output samples that are WORLD audio."""
    hop = float(sample_rate) * frame_seconds
    out_len = max(1, int(round(original.size + (source_map.size - frame_count) * hop)))
    world = np.zeros(out_len, dtype=np.float64)
    n = min(out_len, synthesized.size)
    world[:n] = synthesized[:n]

    src_time = source_map * frame_seconds
    weight_frames = np.zeros(source_map.size, dtype=np.float64)
    for accent in accents:
        lo = float(accent["word_start"]) - SPLICE_MARGIN_S
        hi = float(accent["word_end"]) + SPLICE_MARGIN_S
        weight_frames[(src_time >= lo) & (src_time <= hi)] = 1.0
    fade = max(1, int(round(SPLICE_FADE_S / frame_seconds)))
    weight_frames = np.convolve(weight_frames, np.ones(fade) / fade, mode="same")

    out_frame = np.arange(out_len, dtype=np.float64) / hop
    frame_axis = np.arange(source_map.size, dtype=np.float64)
    weight = np.interp(out_frame, frame_axis, weight_frames)
    last = float(source_map.size - 1)
    # Source frame of each output sample; past the last frame, continue 1:1 instead of holding still.
    src_frame = np.interp(out_frame, frame_axis, source_map) + np.maximum(0.0, out_frame - last)
    original_aligned = np.interp(src_frame * hop, np.arange(original.size, dtype=np.float64), original)
    mixed = weight * world + (1.0 - weight) * original_aligned
    return mixed, float(np.mean(weight > 0.5))


def _apply_output_gain(samples: np.ndarray, sample_rate: int, frame_gain_db: np.ndarray) -> tuple[np.ndarray, float]:
    if samples.size == 0 or frame_gain_db.size == 0 or not np.any(np.abs(frame_gain_db) > 1e-8):
        return samples, 0.0
    frame_x = np.linspace(0.0, 1.0, frame_gain_db.size, endpoint=True)
    sample_x = np.linspace(0.0, 1.0, samples.size, endpoint=True)
    gain_db = np.interp(sample_x, frame_x, frame_gain_db)
    gain = np.power(10.0, gain_db / 20.0)
    out = samples * gain
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0.985:
        out *= 0.985 / peak
    return out, float(np.max(frame_gain_db))


def process_world(source: Path, output: Path, accents_path: Path) -> dict:
    try:
        import pyworld as pw
    except ImportError as exc:
        raise RuntimeError(
            "PyWORLD is not installed in the RVC environment. "
            "Run setup.bat again to repair the Accent components."
        ) from exc

    if not source.is_file():
        raise FileNotFoundError(f"Input WAV was not found: {source}")
    if not accents_path.is_file():
        raise FileNotFoundError(f"Accent spec was not found: {accents_path}")

    audio, sample_rate = sf.read(str(source), dtype="float64", always_2d=False)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    audio = np.ascontiguousarray(audio, dtype=np.float64)
    if audio.size < max(16, int(sample_rate * 0.05)):
        raise RuntimeError("Input audio is too short for WORLD analysis.")

    accents = _load_accents(accents_path)
    if not accents:
        output.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output), audio, int(sample_rate), subtype="PCM_16")
        return {
            "engine": "world_accent",
            "accented_words": 0,
            "sample_rate": int(sample_rate),
            "input_duration": round(float(audio.size) / float(sample_rate), 4),
            "output_duration": round(float(audio.size) / float(sample_rate), 4),
        }

    frame_seconds = FRAME_PERIOD_MS / 1000.0
    f0, sp, ap = pw.wav2world(audio, int(sample_rate), frame_period=FRAME_PERIOD_MS)
    if len(f0) == 0:
        raise RuntimeError("WORLD analysis returned no frames.")

    source_pitch = _build_source_pitch_curve(len(f0), frame_seconds, accents, f0)
    source_gain = _build_source_gain_curve(len(f0), frame_seconds, accents)
    source_map, stats = _build_time_map(len(f0), frame_seconds, accents)

    out_f0 = _nearest_1d(np.asarray(f0, dtype=np.float64), source_map)
    out_sp = _linear_rows(np.asarray(sp, dtype=np.float64), source_map)
    out_ap = _linear_rows(np.asarray(ap, dtype=np.float64), source_map)
    out_pitch_st = np.interp(
        source_map,
        np.arange(len(source_pitch), dtype=np.float64),
        source_pitch,
    )
    out_gain_db = np.interp(
        source_map,
        np.arange(len(source_gain), dtype=np.float64),
        source_gain,
    )

    voiced = out_f0 > 0.0
    active_pitch = voiced & (np.abs(out_pitch_st) > 1e-6)
    if np.any(active_pitch):
        out_f0[active_pitch] *= np.power(2.0, out_pitch_st[active_pitch] / 12.0)
        out_f0[active_pitch] = np.clip(out_f0[active_pitch], 20.0, 2000.0)

    synthesized = pw.synthesize(
        np.ascontiguousarray(out_f0, dtype=np.float64),
        np.ascontiguousarray(out_sp, dtype=np.float64),
        np.ascontiguousarray(out_ap, dtype=np.float64),
        int(sample_rate),
        frame_period=FRAME_PERIOD_MS,
    )
    synthesized = np.asarray(synthesized, dtype=np.float64)
    if synthesized.size == 0 or not np.all(np.isfinite(synthesized)):
        raise RuntimeError("WORLD synthesis produced invalid audio.")

    synthesized, world_share = _splice_original(
        audio, synthesized, source_map, len(f0), frame_seconds, int(sample_rate), accents
    )
    synthesized, max_gain_db = _apply_output_gain(synthesized, int(sample_rate), out_gain_db)
    peak = float(np.max(np.abs(synthesized))) if synthesized.size else 0.0
    if peak > 0.999:
        synthesized *= 0.999 / peak

    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output), synthesized, int(sample_rate), subtype="PCM_16")

    stats.update({
        "engine": "world_accent",
        "pyworld_version": getattr(pw, "__version__", "unknown"),
        "accented_words": len(accents),
        "sample_rate": int(sample_rate),
        "frame_period_ms": FRAME_PERIOD_MS,
        "pitch_edited_frames": int(np.count_nonzero(active_pitch)),
        "max_gain_db": round(max_gain_db, 3),
        "world_share": round(world_share, 4),
        "max_preset": MAX_PRESET,
        "input_duration": round(float(audio.size) / float(sample_rate), 4),
        "output_duration": round(float(synthesized.size) / float(sample_rate), 4),
    })
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="WORLD word-accent editor for Zundanen.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--accents", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = process_world(Path(args.input), Path(args.output), Path(args.accents))
        print(json.dumps({"ok": True, **result}, ensure_ascii=False, separators=(",", ":")))
        return 0
    except Exception as exc:
        import traceback
        print(json.dumps({
            "ok": False,
            "error": str(exc),
            "trace": traceback.format_exc(limit=8),
        }, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
