
from __future__ import annotations

import math
import numpy as np
from scipy import signal

CENTER_HZ = 2500.0
Q = 0.7
GAIN_DB = -18.0


def _sos(sr: int, center_hz: float = CENTER_HZ, q: float = Q, gain_db: float = GAIN_DB) -> np.ndarray:
    # Zero-phase filtering applies magnitude twice, so design half the requested dB.
    single_pass_gain_db = gain_db / 2.0
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
    return np.array([[b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]], dtype=np.float64)


def apply_debuzz(audio: np.ndarray, sr: int) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 1:
        y = signal.sosfiltfilt(_sos(sr), x)
    elif x.ndim == 2:
        y = np.stack([signal.sosfiltfilt(_sos(sr), x[:, i]) for i in range(x.shape[1])], axis=1)
    else:
        raise ValueError(f"unsupported audio shape: {x.shape}")
    return y.astype(np.float32)
