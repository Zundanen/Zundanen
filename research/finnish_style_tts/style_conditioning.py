from __future__ import annotations

"""Step 8A style-conditioning extension for Zundanen.

Original Zundanen code.  No Style-Bert-VITS2 source code is used.

Design goals:
- preserve an existing Finnish VITS neutral checkpoint exactly at style_strength=0
- learn only a residual style delta on top of the existing Finnish text encoder
- keep neutral style ID 0 mathematically neutral for all time
- support continuous style strength at inference
"""

import math
from typing import Iterable

import torch
from torch import nn

from mas_vits_model import FinBertVitsTextEncoder


DEFAULT_STYLES = ("neutral", "happy", "sad", "angry", "fun", "whisper")


def parse_style_names(value: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        parts = [x.strip() for x in value.split(",") if x.strip()]
    else:
        parts = [str(x).strip() for x in value if str(x).strip()]
    if not parts:
        raise ValueError("at least one style is required")
    if parts[0] != "neutral":
        raise ValueError("style 0 must be named 'neutral'")
    if len(set(parts)) != len(parts):
        raise ValueError(f"duplicate style names: {parts}")
    return tuple(parts)


class Step8StyleTextEncoder(FinBertVitsTextEncoder):
    """FinBERT+VITS text encoder with a neutral-safe residual style path.

    The inherited Step6/7 encoder already contains a legacy one-entry style
    port.  That exact path is kept untouched for checkpoint compatibility.

    Step8 adds:
      style_delta_emb:  style ID -> 128-D residual vector
      style_delta_proj: 128-D -> hidden residual, bias-free and ZERO initialized

    Neutral style row 0 is fixed to all zeros after every optimizer step by the
    training script.  Therefore style_strength=0 (and neutral ID 0) produces
    the exact old Finnish VITS text path.
    """

    def __init__(
        self,
        attentions_module,
        n_vocab: int,
        *,
        n_styles: int,
        style_names: tuple[str, ...],
        bert_dim: int = 1024,
        hidden: int = 192,
        latent: int = 192,
        filter_channels: int = 768,
        n_heads: int = 2,
        n_layers: int = 6,
        kernel_size: int = 3,
        p_dropout: float = 0.1,
        style_dim: int = 128,
    ):
        super().__init__(
            attentions_module,
            n_vocab,
            bert_dim=bert_dim,
            hidden=hidden,
            latent=latent,
            filter_channels=filter_channels,
            n_heads=n_heads,
            n_layers=n_layers,
            kernel_size=kernel_size,
            p_dropout=p_dropout,
            style_dim=style_dim,
        )
        if n_styles < 1:
            raise ValueError("n_styles must be >= 1")
        if len(style_names) != n_styles:
            raise ValueError("style_names length must equal n_styles")
        self.n_styles = int(n_styles)
        self.style_names = tuple(style_names)
        self.style_dim = int(style_dim)

        self.style_delta_emb = nn.Embedding(self.n_styles, self.style_dim)
        nn.init.normal_(self.style_delta_emb.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.style_delta_emb.weight[0].zero_()

        self.style_delta_proj = nn.Linear(self.style_dim, hidden, bias=False)
        nn.init.zeros_(self.style_delta_proj.weight)

    @staticmethod
    def _strength_tensor(style_strength, *, batch: int, device, dtype):
        if torch.is_tensor(style_strength):
            s = style_strength.to(device=device, dtype=dtype)
            if s.ndim == 0:
                s = s.expand(batch)
            elif s.ndim == 2 and s.size(1) == 1:
                s = s[:, 0]
            if s.ndim != 1 or s.numel() != batch:
                raise ValueError(f"style_strength tensor must be scalar or [B], got {tuple(s.shape)}")
            return s
        return torch.full((batch,), float(style_strength), device=device, dtype=dtype)

    def forward(
        self,
        tokens,
        bert,
        lengths,
        style_id=None,
        speaker_id=None,
        *,
        style_strength=1.0,
        style_vector=None,
    ):
        batch = tokens.size(0)
        device = tokens.device
        if style_id is None:
            style_id = torch.zeros(batch, dtype=torch.long, device=device)
        else:
            style_id = style_id.to(device=device, dtype=torch.long)
        if speaker_id is None:
            speaker_id = torch.zeros(batch, dtype=torch.long, device=device)

        if torch.any(style_id < 0) or torch.any(style_id >= self.n_styles):
            raise ValueError(f"style_id out of range 0..{self.n_styles - 1}")

        mb = self._sequence_mask(lengths, tokens.size(1))
        x = self.token_emb(tokens) * math.sqrt(self.hidden)
        x = x + self.bert_proj(self.bert_norm(bert.float()))

        # Preserve the historical Finnish VITS neutral path exactly.  The legacy
        # embedding only has ID 0 and remains part of the old checkpoint.
        legacy_style_id = torch.zeros(batch, dtype=torch.long, device=device)
        x = x + self.style_proj(self.style_emb(legacy_style_id)).unsqueeze(1)
        x = x + self.speaker_emb(speaker_id).unsqueeze(1)

        if style_vector is None:
            sv = self.style_delta_emb(style_id)
        else:
            sv = style_vector.to(device=device, dtype=x.dtype)
            if sv.ndim == 1:
                sv = sv.unsqueeze(0).expand(batch, -1)
            if sv.shape != (batch, self.style_dim):
                raise ValueError(
                    f"style_vector must be [{batch},{self.style_dim}], got {tuple(sv.shape)}"
                )

        strength = self._strength_tensor(
            style_strength, batch=batch, device=device, dtype=sv.dtype
        )
        style_delta = self.style_delta_proj(sv) * strength.unsqueeze(1)
        x = x + style_delta.unsqueeze(1)

        x = x.transpose(1, 2)
        mask = mb.unsqueeze(1).to(x.dtype)
        x = self.encoder(x * mask, mask)
        stats = self.proj(x) * mask
        m, logs = stats.chunk(2, dim=1)
        return x, m, logs, mask

    @torch.no_grad()
    def enforce_neutral_zero(self) -> None:
        self.style_delta_emb.weight[0].zero_()


def upgrade_text_encoder(model, attentions_module, n_vocab: int, style_names: tuple[str, ...]):
    """Replace model.text while preserving every Finnish VITS parameter exactly."""
    old = model.text
    new = Step8StyleTextEncoder(
        attentions_module,
        n_vocab,
        n_styles=len(style_names),
        style_names=style_names,
        bert_dim=old.bert_norm.normalized_shape[0],
        hidden=old.hidden,
        latent=old.proj.out_channels // 2,
        filter_channels=768,
        n_heads=2,
        n_layers=6,
        kernel_size=3,
        p_dropout=0.1,
        style_dim=old.style_emb.embedding_dim,
    )
    result = new.load_state_dict(old.state_dict(), strict=False)
    allowed_missing = {"style_delta_emb.weight", "style_delta_proj.weight"}
    if set(result.missing_keys) != allowed_missing or result.unexpected_keys:
        raise RuntimeError(
            "Unexpected Step8 text upgrade mismatch: "
            f"missing={result.missing_keys} unexpected={result.unexpected_keys}"
        )
    model.text = new.to(next(old.parameters()).device)
    return model.text


def load_step7_weights_into_step8(model, state_dict: dict) -> None:
    """Load a Finnish VITS model state into an already-upgraded Step8 model."""
    result = model.load_state_dict(state_dict, strict=False)
    allowed_missing = {
        "text.style_delta_emb.weight",
        "text.style_delta_proj.weight",
    }
    if set(result.missing_keys) != allowed_missing or result.unexpected_keys:
        raise RuntimeError(
            "Unexpected Finnish VITS -> Step8 checkpoint mismatch: "
            f"missing={result.missing_keys} unexpected={result.unexpected_keys}"
        )
    model.text.enforce_neutral_zero()
