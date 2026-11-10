
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

DEFAULT_CENTER_HZ = 2500.0
DEFAULT_Q = 0.7
DEFAULT_GAIN_DB = -18.0


def _peaking_eq_sos(
    sr: int,
    center_hz: float,
    effective_gain_db: float,
    q: float,
) -> np.ndarray:
    """
    Zero-phase peaking EQ.

    Because sosfiltfilt applies the magnitude response twice, the single-pass
    biquad is designed for half of the requested dB so that the final effective
    center attenuation is approximately `effective_gain_db`.
    """
    single_pass_gain_db = effective_gain_db / 2.0
    A = 10.0 ** (single_pass_gain_db / 40.0)
    w0 = 2.0 * math.pi * center_hz / sr
    cw = math.cos(w0)
    sw = math.sin(w0)
    alpha = sw / (2.0 * q)

    b0 = 1.0 + alpha * A
    b1 = -2.0 * cw
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * cw
    a2 = 1.0 - alpha / A

    return np.array(
        [[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]],
        dtype=np.float64,
    )


def apply_debuzz(
    audio: np.ndarray,
    sr: int,
    center_hz: float = DEFAULT_CENTER_HZ,
    q: float = DEFAULT_Q,
    gain_db: float = DEFAULT_GAIN_DB,
) -> np.ndarray:
    """
    Apply the current Pilot de-buzz candidate:
      center = 2.5 kHz
      Q      = 0.7
      gain   = -18 dB effective
    """
    if audio.ndim == 2:
        # Process channels independently.
        channels = []
        for ch in range(audio.shape[1]):
            channels.append(
                signal.sosfiltfilt(
                    _peaking_eq_sos(sr, center_hz, gain_db, q),
                    audio[:, ch].astype(np.float64),
                )
            )
        y = np.stack(channels, axis=1)
    else:
        y = signal.sosfiltfilt(
            _peaking_eq_sos(sr, center_hz, gain_db, q),
            audio.astype(np.float64),
        )

    return y.astype(np.float32)


def process_wav(
    src: Path,
    dst: Path,
    center_hz: float = DEFAULT_CENTER_HZ,
    q: float = DEFAULT_Q,
    gain_db: float = DEFAULT_GAIN_DB,
) -> None:
    audio, sr = sf.read(src, dtype="float32", always_2d=False)
    y = apply_debuzz(audio, sr, center_hz=center_hz, q=q, gain_db=gain_db)

    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0.999:
        y = y * (0.999 / peak)

    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dst, y, sr, subtype="PCM_16")
