from __future__ import annotations

"""v65: frame-level F0 conditioning of the VITS prior + an F0 predictor.

Original Zundanen code (MIT).  Built on the v63 base (v62 architecture).

  F0 features per spectrogram frame (hop 256 @ 24 kHz):
      f[0] = semitones vs the speaker's real Normal median / 12  (unvoiced: interpolated)
      f[1] = voiced flag
  Prior:   m_frames += f0_proj(f)            (Conv1d, zero-initialised -> starts == v63)
  F0 predictor (frames): Conv stack on the duration-expanded text hidden state
      -> semitone value + voiced logit.  Trained on ground-truth F0 (L1 on voiced
      frames + BCE on voicing); the text encoder is not updated by this loss.
  Inference: predict the contour, then EXPLICITLY shift its level and scale its
      range around the median (k), and condition the prior on that contour.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT / "emotion_tts", ROOT / "emotion_tts", ROOT / "emotion_tts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from pitch_model import build_v62  # noqa: E402

SCHEMA = "zundanen-v65-f0-conditioned-vits-v1"
F0_DIR = ROOT / "outputs/v65/f0frames"
SR, HOP, NFFT = 24000, 256, 1024


def f0_key(path: str) -> str:
    return hashlib.sha1(path.encode("utf-8")).hexdigest()


def f0_features_for_frames(f0_10ms: np.ndarray, n_frames: int, ref_hz: float) -> np.ndarray:
    """RMVPE 10 ms F0 (Hz, 0 = unvoiced) -> [2, n_frames] features on spectrogram frames."""
    t_spec = (np.arange(n_frames) * HOP + NFFT / 2) / SR
    t_f0 = np.arange(len(f0_10ms)) * 0.01
    voiced = f0_10ms > 40
    feat = np.zeros((2, n_frames), dtype=np.float32)
    if voiced.sum() >= 2:
        st = 12 * np.log2(f0_10ms[voiced] / ref_hz)
        feat[0] = np.interp(t_spec, t_f0[voiced], st) / 12.0
        idx = np.clip(np.round(t_spec / 0.01).astype(int), 0, len(f0_10ms) - 1)
        feat[1] = voiced[idx].astype(np.float32)
    return feat


class F0Predictor(nn.Module):
    def __init__(self, hidden=192, channels=256, layers=4):
        super().__init__()
        self.pre = nn.Conv1d(hidden, channels, 5, padding=2)
        self.convs = nn.ModuleList([nn.Conv1d(channels, channels, 5, padding=2) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.GroupNorm(1, channels) for _ in range(layers)])
        self.out = nn.Conv1d(channels, 2, 1)

    def forward(self, h_frames, mask):
        x = self.pre(h_frames) * mask
        for c, n in zip(self.convs, self.norms):
            x = x + torch.relu(n(c(x))) * mask
        o = self.out(x) * mask
        return o[:, :1], o[:, 1:]          # semitones/12, voiced logit


class F0Modules(nn.Module):
    def __init__(self, latent=192, hidden=192, flow_hidden=256, n_flows=4, flow_cond=False):
        super().__init__()
        self.f0_proj = nn.Conv1d(2, latent, 5, padding=2)
        nn.init.zeros_(self.f0_proj.weight)
        nn.init.zeros_(self.f0_proj.bias)
        self.predictor = F0Predictor(hidden)
        # v65b: F0 also conditions every flow coupling (like speaker g in VITS),
        # so pitch in z can come from the F0 input instead of the prior.
        self.flow_cond = nn.ModuleList()
        if flow_cond:
            for _ in range(n_flows):
                c = nn.Conv1d(2, flow_hidden, 5, padding=2)
                nn.init.zeros_(c.weight)
                nn.init.zeros_(c.bias)
                self.flow_cond.append(c)


def _coupling(layer, cond, x, mask, f0feat, reverse):
    """TransformerAdditiveCouplingLayer.forward with an added frame-level F0 term."""
    from transformer_flow import _sinusoidal_position_encoding
    x0, x1 = torch.split(x, [layer.half_channels, layer.half_channels], dim=1)
    x0, x1 = x0 * mask, x1 * mask
    h = layer.pre(x0) * mask
    if cond is not None and f0feat is not None:
        h = h + cond(f0feat) * mask
    h = h.transpose(1, 2)
    h = h + _sinusoidal_position_encoding(h.size(1), h.size(2), h.device, h.dtype)
    h = layer.encoder(h, src_key_padding_mask=~(mask[:, 0, :] > 0.5))
    shift = layer.post(h.transpose(1, 2) * mask) * mask
    y1 = (x1 - shift if reverse else x1 + shift) * mask
    return torch.cat([x0, y1], dim=1) * mask


def flow_apply(model, x, mask, f0feat=None, reverse=False):
    """model.flow with optional F0 conditioning (identical to model.flow if no flow_cond)."""
    conds = list(getattr(model.f0, "flow_cond", [])) if hasattr(model, "f0") else []
    if not conds:
        return model.flow(x, mask, reverse=reverse)
    flows = model.flow.flows
    if not reverse:
        for layer, c in zip(flows, conds):
            x = _coupling(layer, c, x, mask, f0feat, False)
            x = torch.flip(x, dims=[1]) * mask
    else:
        for layer, c in zip(reversed(flows), reversed(conds)):
            x = torch.flip(x, dims=[1]) * mask
            x = _coupling(layer, c, x, mask, f0feat, True)
    return x * mask


def build_v65(root, n_vocab, n_speakers, n_styles, n_langs, device, flow_cond=False):
    model, models, losses, vendor = build_v62(root, n_vocab, n_speakers, n_styles, n_langs, "cpu")
    model.f0 = F0Modules(flow_cond=flow_cond)
    return model.to(device), models, losses, vendor


def load_v65(root, path, device):
    ck = path if isinstance(path, dict) else torch.load(Path(path), map_location="cpu", weights_only=False)   # dict: character file
    if ck.get("schema") != SCHEMA:
        raise RuntimeError(f"not a v65 checkpoint: {path}")
    flow_cond = any(k.startswith("f0.flow_cond.") for k in ck["model"])
    model, models, losses, _ = build_v65(root, len(ck["vocab"]), len(ck["speakers"]), len(ck["styles"]), len(ck["langs"]),
                                         device, flow_cond=flow_cond)
    model.load_state_dict(ck["model"], strict=True)
    return ck, model, models, losses


def shape_contour(st_pred: torch.Tensor, uv_logit: torch.Tensor, mask: torch.Tensor, *, level_st: float | None = None,
                  range_k: float = 1.0) -> torch.Tensor:
    """Predicted [1,1,T] semitones/12 -> features [1,2,T] with explicit level/range control.

    level_st: target median in semitones vs the speaker's real Normal (None = keep).
    range_k:  scale of the contour around its voiced median (1 = as predicted).
    """
    st = st_pred[0, 0] * 12.0
    voiced = (torch.sigmoid(uv_logit[0, 0]) > 0.5) & (mask[0, 0] > 0)
    ref = st[voiced] if voiced.any() else st[mask[0, 0] > 0]
    med = ref.median()
    target = med if level_st is None else torch.tensor(float(level_st), device=st.device)
    st2 = target + range_k * (st - med)
    return torch.stack([st2 / 12.0, voiced.float()]).unsqueeze(0) * mask
