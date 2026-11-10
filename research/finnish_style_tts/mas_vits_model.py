from __future__ import annotations

"""Step 6A: MAS-aligned full-joint VITS components.

Architecture is intentionally kept equal to Step 5L:
Finnish graphemes + cached FinBERT Large 1024 -> text prior,
MIT jaywalnut310/VITS PosteriorEncoder, ResidualCouplingBlock,
DurationPredictor, Generator, and MultiPeriodDiscriminator.

The training script owns MAS.  No Style-Bert-VITS2 / AGPL source is used.
"""

import math
import sys
import types
from pathlib import Path

import torch
from torch import nn


def load_mit_vits_vendor(root: Path):
    vendor = root / "research" / "vendor" / "VITS"
    required = ["models.py", "modules.py", "attentions.py", "commons.py", "losses.py", "LICENSE"]
    missing = [x for x in required if not (vendor / x).exists()]
    if missing:
        raise RuntimeError(
            f"MIT VITS vendor is incomplete: {missing}. "
            "Run setup_step5e_mitvits_vendor.bat first."
        )
    license_text = (vendor / "LICENSE").read_text(encoding="utf-8", errors="ignore")
    if "MIT License" not in license_text:
        raise RuntimeError("Expected MIT license was not found in research/vendor/VITS/LICENSE")

    # models.py imports monotonic_align at import time.  Step 6A uses an
    # independent PyTorch MAS implementation in train_step6_mas.py, so no
    # Cython extension is required here.
    if "monotonic_align" not in sys.modules:
        stub = types.ModuleType("monotonic_align")

        def _external_call_disabled(*args, **kwargs):
            raise RuntimeError(
                "VITS monotonic_align extension is not used by Step 6A; "
                "train_step6_mas.py provides the MAS implementation."
            )

        stub.maximum_path = _external_call_disabled
        sys.modules["monotonic_align"] = stub

    v = str(vendor)
    if v not in sys.path:
        sys.path.insert(0, v)
    import models  # type: ignore
    import attentions  # type: ignore
    import losses  # type: ignore

    return models, attentions, losses, vendor


class FinBertVitsTextEncoder(nn.Module):
    """Same text-prior architecture as Step 5L."""

    def __init__(
        self,
        attentions_module,
        n_vocab: int,
        *,
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
        super().__init__()
        self.hidden = hidden
        self.token_emb = nn.Embedding(n_vocab, hidden)
        nn.init.normal_(self.token_emb.weight, 0.0, hidden ** -0.5)
        self.bert_norm = nn.LayerNorm(bert_dim)
        self.bert_proj = nn.Linear(bert_dim, hidden)
        self.style_emb = nn.Embedding(1, style_dim)
        self.style_proj = nn.Linear(style_dim, hidden)
        self.speaker_emb = nn.Embedding(1, hidden)
        nn.init.zeros_(self.style_emb.weight)
        nn.init.zeros_(self.speaker_emb.weight)

        self.encoder = attentions_module.Encoder(
            hidden, filter_channels, n_heads, n_layers, kernel_size, p_dropout
        )
        self.proj = nn.Conv1d(hidden, latent * 2, 1)

    @staticmethod
    def _sequence_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        ids = torch.arange(max_len, device=lengths.device)
        return ids.unsqueeze(0) < lengths.unsqueeze(1)

    def forward(self, tokens, bert, lengths, style_id=None, speaker_id=None):
        if style_id is None:
            style_id = torch.zeros(tokens.size(0), dtype=torch.long, device=tokens.device)
        if speaker_id is None:
            speaker_id = torch.zeros(tokens.size(0), dtype=torch.long, device=tokens.device)
        mb = self._sequence_mask(lengths, tokens.size(1))
        x = self.token_emb(tokens) * math.sqrt(self.hidden)
        x = x + self.bert_proj(self.bert_norm(bert.float()))
        x = x + self.style_proj(self.style_emb(style_id)).unsqueeze(1)
        x = x + self.speaker_emb(speaker_id).unsqueeze(1)
        x = x.transpose(1, 2)
        mask = mb.unsqueeze(1).to(x.dtype)
        x = self.encoder(x * mask, mask)
        stats = self.proj(x) * mask
        m, logs = stats.chunk(2, dim=1)
        return x, m, logs, mask


class MasFullJointVits(nn.Module):
    """Step 5L network geometry, unchanged; only alignment supervision differs."""

    def __init__(self, models_module, attentions_module, n_vocab: int):
        super().__init__()
        self.sample_rate = 24000
        self.n_fft = 1024
        self.hop_length = 256
        self.win_length = 1024
        self.spec_channels = self.n_fft // 2 + 1
        self.latent = 192
        self.hidden = 192
        self.segment_frames = 64
        self.segment_samples = self.segment_frames * self.hop_length

        self.text = FinBertVitsTextEncoder(
            attentions_module,
            n_vocab,
            hidden=self.hidden,
            latent=self.latent,
            filter_channels=768,
            n_heads=2,
            n_layers=6,
            kernel_size=3,
            p_dropout=0.1,
        )
        self.posterior = models_module.PosteriorEncoder(
            in_channels=self.spec_channels,
            out_channels=self.latent,
            hidden_channels=self.hidden,
            kernel_size=5,
            dilation_rate=1,
            n_layers=16,
            gin_channels=0,
        )
        self.flow = models_module.ResidualCouplingBlock(
            channels=self.latent,
            hidden_channels=self.hidden,
            kernel_size=5,
            dilation_rate=1,
            n_layers=4,
            n_flows=4,
            gin_channels=0,
        )
        self.duration = models_module.DurationPredictor(
            in_channels=self.hidden,
            filter_channels=256,
            kernel_size=3,
            p_dropout=0.5,
            gin_channels=0,
        )
        self.generator = models_module.Generator(
            initial_channel=self.latent,
            resblock="1",
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            upsample_rates=[8, 8, 2, 2],
            upsample_initial_channel=512,
            upsample_kernel_sizes=[16, 16, 4, 4],
            gin_channels=0,
        )

    def config_dict(self) -> dict:
        return {
            "sample_rate": self.sample_rate,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "win_length": self.win_length,
            "spec_channels": self.spec_channels,
            "latent": self.latent,
            "hidden": self.hidden,
            "segment_frames": self.segment_frames,
            "segment_samples": self.segment_samples,
            "text_filter_channels": 768,
            "text_heads": 2,
            "text_layers": 6,
            "posterior_layers": 16,
            "flow_n_layers": 4,
            "flow_n_flows": 4,
            "duration": "deterministic; MAS supervised; log(w+1e-6) target",
            "alignment": "VITS-style Gaussian log-likelihood + maximum monotonic alignment search",
            "style_port": "neutral ID 0, 128-D",
            "speaker_port": "neutral ID 0",
            "source": "original Zundanen glue + jaywalnut310/VITS MIT vendor modules",
        }
