from __future__ import annotations

"""Zundanen Step 5B/C: compact VITS-like Finnish TTS PoC.

This module is an original, compact implementation created for the Zundanen
research path.  It uses the core VITS ideas needed for an end-to-end smoke test:

- text prior conditioned on grapheme IDs + frozen FinBERT features
- posterior acoustic encoder
- invertible affine coupling flow
- monotonic alignment search (pure PyTorch, no Cython build)
- deterministic duration predictor
- HiFi-GAN-like waveform generator
- lightweight waveform discriminators

It intentionally omits VITS's stochastic duration predictor and some of the
large reference implementation's capacity.  The goal of Step 5C is not final
quality: it is to prove that the Finnish frontend/cache/training/inference path
can overfit a tiny dataset on an RTX 3060 before spending days on full training.

The model already has explicit neutral style and speaker slots.  Step 5C uses
only ID 0 for both, so later style work does not require changing the public
model interface.
"""

from dataclasses import dataclass, asdict
import math
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class VitsLiteConfig:
    sample_rate: int = 24000
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    fmin: float = 0.0
    fmax: float = 12000.0
    bert_dim: int = 768
    hidden: int = 128
    latent: int = 128
    filter_channels: int = 512
    text_heads: int = 4
    text_layers: int = 4
    text_dropout: float = 0.1
    posterior_layers: int = 6
    flow_layers: int = 4
    style_dim: int = 128
    speaker_dim: int = 64
    upsample_rates: tuple[int, ...] = (8, 8, 2, 2)
    upsample_kernels: tuple[int, ...] = (16, 16, 4, 4)
    upsample_initial_channel: int = 256
    resblock_kernels: tuple[int, ...] = (3, 7, 11)
    segment_frames: int = 32

    @property
    def spec_channels(self) -> int:
        return self.n_fft // 2 + 1

    @property
    def segment_samples(self) -> int:
        return self.segment_frames * self.hop_length

    def to_dict(self) -> dict:
        d = asdict(self)
        # JSON-friendly tuples.
        for k in ("upsample_rates", "upsample_kernels", "resblock_kernels"):
            d[k] = list(d[k])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "VitsLiteConfig":
        d = dict(d)
        for k in ("upsample_rates", "upsample_kernels", "resblock_kernels"):
            if k in d:
                d[k] = tuple(d[k])
        return cls(**d)


def sequence_mask(lengths: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
    if max_len is None:
        max_len = int(lengths.max().item())
    ids = torch.arange(max_len, device=lengths.device)
    return ids.unsqueeze(0) < lengths.unsqueeze(1)


def rand_slice_segments(x: torch.Tensor, lengths: torch.Tensor, segment_size: int):
    """Slice [B,C,T] to a fixed number of frames and return start indices."""
    b, _, t = x.shape
    starts = []
    chunks = []
    for i in range(b):
        max_start = max(0, int(lengths[i].item()) - segment_size)
        if max_start > 0:
            start = int(torch.randint(0, max_start + 1, (1,), device=x.device).item())
        else:
            start = 0
        end = start + segment_size
        chunk = x[i : i + 1, :, start:min(end, t)]
        if chunk.size(-1) < segment_size:
            chunk = F.pad(chunk, (0, segment_size - chunk.size(-1)))
        chunks.append(chunk)
        starts.append(start)
    return torch.cat(chunks, dim=0), torch.tensor(starts, device=x.device, dtype=torch.long)


def slice_wave_segments(y: torch.Tensor, starts_frames: torch.Tensor, segment_samples: int, hop_length: int):
    chunks = []
    for i, start_f in enumerate(starts_frames.tolist()):
        start = int(start_f) * hop_length
        chunk = y[i : i + 1, :, start : start + segment_samples]
        if chunk.size(-1) < segment_samples:
            chunk = F.pad(chunk, (0, segment_samples - chunk.size(-1)))
        chunks.append(chunk)
    return torch.cat(chunks, dim=0)


def maximum_path(value: torch.Tensor, x_lengths: torch.Tensor, y_lengths: torch.Tensor) -> torch.Tensor:
    """Pure PyTorch monotonic maximum path.

    Args:
      value: [B, Tx, Ty] log-probability score.
      x_lengths: text lengths.
      y_lengths: acoustic lengths.
    Returns:
      path: [B, Tx, Ty] in value dtype.

    Each acoustic frame belongs to exactly one text token and token index can
    stay or advance by one.  This is deliberately simple and build-free for
    Windows smoke testing.  It runs under no_grad in training.
    """
    bsz, tx_max, ty_max = value.shape
    paths = torch.zeros_like(value)
    neg_inf = torch.finfo(value.dtype).min
    for b in range(bsz):
        tx = int(x_lengths[b].item())
        ty = int(y_lengths[b].item())
        if tx <= 0 or ty <= 0:
            continue
        if ty < tx:
            # Degenerate sample: let early tokens share a frame rather than
            # crash.  Normal TTS data should practically never hit this.
            idx = torch.linspace(0, tx - 1, ty, device=value.device).round().long()
            paths[b, idx, torch.arange(ty, device=value.device)] = 1.0
            continue
        score = value[b, :tx, :ty]
        dp = torch.full((tx, ty), neg_inf, device=value.device, dtype=value.dtype)
        ptr = torch.zeros((tx, ty), device=value.device, dtype=torch.bool)
        dp[0, 0] = score[0, 0]
        for y in range(1, ty):
            x_lo = max(0, tx - (ty - y))
            x_hi = min(tx - 1, y)
            stay = dp[x_lo : x_hi + 1, y - 1]
            if x_lo == 0:
                adv = torch.cat([torch.tensor([neg_inf], device=value.device, dtype=value.dtype), dp[0:x_hi, y - 1]])
            else:
                adv = dp[x_lo - 1 : x_hi, y - 1]
            choose_adv = adv > stay
            best = torch.where(choose_adv, adv, stay)
            dp[x_lo : x_hi + 1, y] = score[x_lo : x_hi + 1, y] + best
            ptr[x_lo : x_hi + 1, y] = choose_adv
        x = tx - 1
        for y in range(ty - 1, -1, -1):
            paths[b, x, y] = 1.0
            if y > 0 and x > 0 and bool(ptr[x, y].item()):
                x -= 1
    return paths


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, kernel: int = 3, dilation: int = 1, dropout: float = 0.0):
        super().__init__()
        pad = (kernel * dilation - dilation) // 2
        self.norm1 = nn.LayerNorm(channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel, padding=pad, dilation=dilation)
        self.norm2 = nn.LayerNorm(channels)
        self.conv2 = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x.transpose(1, 2)).transpose(1, 2)
        h = F.gelu(self.conv1(h))
        h = self.dropout(h)
        h = self.norm2(h.transpose(1, 2)).transpose(1, 2)
        h = self.conv2(F.gelu(h))
        return (x + h) * mask


class TextPriorEncoder(nn.Module):
    def __init__(self, n_vocab: int, cfg: VitsLiteConfig):
        super().__init__()
        h = cfg.hidden
        self.hidden = h
        self.token_emb = nn.Embedding(n_vocab, h)
        self.bert_proj = nn.Sequential(nn.LayerNorm(cfg.bert_dim), nn.Linear(cfg.bert_dim, h))
        self.style_emb = nn.Embedding(1, cfg.style_dim)
        self.style_proj = nn.Linear(cfg.style_dim, h)
        self.speaker_text = nn.Embedding(1, h)
        layer = nn.TransformerEncoderLayer(
            d_model=h,
            nhead=cfg.text_heads,
            dim_feedforward=cfg.filter_channels,
            dropout=cfg.text_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.text_layers)
        self.proj = nn.Conv1d(h, cfg.latent * 2, 1)
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=h ** -0.5)
        nn.init.zeros_(self.style_emb.weight)
        nn.init.zeros_(self.speaker_text.weight)

    def forward(self, tokens, bert, lengths, style_id=None, speaker_id=None):
        if style_id is None:
            style_id = torch.zeros(tokens.size(0), dtype=torch.long, device=tokens.device)
        if speaker_id is None:
            speaker_id = torch.zeros(tokens.size(0), dtype=torch.long, device=tokens.device)
        mask_bool = sequence_mask(lengths, tokens.size(1))
        x = self.token_emb(tokens) * math.sqrt(self.hidden)
        x = x + self.bert_proj(bert.float())
        x = x + self.style_proj(self.style_emb(style_id)).unsqueeze(1)
        x = x + self.speaker_text(speaker_id).unsqueeze(1)
        x = self.encoder(x, src_key_padding_mask=~mask_bool)
        x = x * mask_bool.unsqueeze(-1)
        x_cf = x.transpose(1, 2)
        stats = self.proj(x_cf)
        m, logs = stats.chunk(2, dim=1)
        logs = torch.clamp(logs, -7.0, 3.0)
        return x_cf, m, logs, mask_bool.unsqueeze(1).to(x.dtype)


class PosteriorEncoder(nn.Module):
    def __init__(self, cfg: VitsLiteConfig):
        super().__init__()
        h = cfg.hidden
        self.pre = nn.Conv1d(cfg.spec_channels, h, 1)
        self.speaker = nn.Embedding(1, cfg.speaker_dim)
        self.spk_proj = nn.Linear(cfg.speaker_dim, h)
        self.blocks = nn.ModuleList([
            ResidualConvBlock(h, 5, dilation=2 ** (i % 4), dropout=0.0)
            for i in range(cfg.posterior_layers)
        ])
        self.proj = nn.Conv1d(h, cfg.latent * 2, 1)
        nn.init.zeros_(self.speaker.weight)

    def forward(self, spec, lengths, speaker_id=None, sample: bool = True):
        if speaker_id is None:
            speaker_id = torch.zeros(spec.size(0), dtype=torch.long, device=spec.device)
        mask = sequence_mask(lengths, spec.size(-1)).unsqueeze(1).to(spec.dtype)
        h = self.pre(spec) * mask
        g = self.spk_proj(self.speaker(speaker_id)).unsqueeze(-1)
        h = (h + g) * mask
        for block in self.blocks:
            h = block(h, mask)
        stats = self.proj(h) * mask
        m, logs = stats.chunk(2, dim=1)
        logs = torch.clamp(logs, -7.0, 2.0)
        if sample:
            eps = torch.randn_like(m)
            z = (m + eps * torch.exp(logs)) * mask
        else:
            z = m * mask
        return z, m, logs, mask


class CouplingNet(nn.Module):
    def __init__(self, channels: int, hidden: int):
        super().__init__()
        self.pre = nn.Conv1d(channels, hidden, 1)
        self.blocks = nn.ModuleList([
            ResidualConvBlock(hidden, 5, dilation=d, dropout=0.0)
            for d in (1, 2, 4, 8)
        ])
        self.post = nn.Conv1d(hidden, channels * 2, 1)
        nn.init.zeros_(self.post.weight)
        nn.init.zeros_(self.post.bias)

    def forward(self, x, mask):
        h = self.pre(x) * mask
        for block in self.blocks:
            h = block(h, mask)
        return self.post(h) * mask


class AffineCoupling(nn.Module):
    def __init__(self, channels: int, hidden: int, flip: bool):
        super().__init__()
        assert channels % 2 == 0
        self.half = channels // 2
        self.net = CouplingNet(self.half, hidden)
        self.flip = flip

    def forward(self, x, mask, reverse=False):
        if self.flip:
            x = torch.flip(x, [1])
        x0, x1 = x[:, : self.half], x[:, self.half :]
        # Step 5C uses a volume-preserving additive coupling.
        #
        # The previous PoC version used a scale+shift affine coupling here,
        # but the sampled VITS KL objective below assumes zero flow log-det
        # (the same simplification used by mean-only coupling flows).  A
        # learnable scale without its Jacobian term lets the flow drive the
        # reported KL strongly negative.  Keeping only the learned shift makes
        # |det J| == 1, so the KL expression is valid and the flow stays
        # exactly invertible.
        m, _unused_logs = self.net(x0, mask).chunk(2, 1)
        if not reverse:
            y1 = (x1 - m) * mask
            y = torch.cat([x0, y1], 1)
        else:
            y1 = (x1 + m) * mask
            y = torch.cat([x0, y1], 1)
        if self.flip:
            y = torch.flip(y, [1])
        return y * mask


class Flow(nn.Module):
    def __init__(self, cfg: VitsLiteConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            AffineCoupling(cfg.latent, cfg.hidden, flip=bool(i % 2))
            for i in range(cfg.flow_layers)
        ])

    def forward(self, x, mask, reverse=False):
        layers = reversed(self.layers) if reverse else self.layers
        for layer in layers:
            x = layer(x, mask, reverse=reverse)
        return x


class DurationPredictor(nn.Module):
    def __init__(self, cfg: VitsLiteConfig):
        super().__init__()
        h = cfg.hidden
        self.net = nn.Sequential(
            nn.Conv1d(h, h, 3, padding=1),
            nn.ReLU(),
            nn.LayerNorm([h, 1], elementwise_affine=False) if False else nn.Identity(),
            nn.Dropout(0.1),
            nn.Conv1d(h, h, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(h, 1, 1),
        )

    def forward(self, x, mask):
        return self.net(x.detach()) * mask


class ResBlock1D(nn.Module):
    def __init__(self, channels: int, kernel: int):
        super().__init__()
        self.convs = nn.ModuleList()
        for dilation in (1, 3, 5):
            pad = (kernel * dilation - dilation) // 2
            self.convs.append(nn.Conv1d(channels, channels, kernel, padding=pad, dilation=dilation))
            self.convs.append(nn.Conv1d(channels, channels, kernel, padding=(kernel - 1) // 2))

    def forward(self, x):
        for i in range(0, len(self.convs), 2):
            h = F.leaky_relu(x, 0.1)
            h = self.convs[i](h)
            h = F.leaky_relu(h, 0.1)
            h = self.convs[i + 1](h)
            x = x + h
        return x


class Generator(nn.Module):
    def __init__(self, cfg: VitsLiteConfig):
        super().__init__()
        self.pre = nn.Conv1d(cfg.latent, cfg.upsample_initial_channel, 7, padding=3)
        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        ch = cfg.upsample_initial_channel
        for rate, kernel in zip(cfg.upsample_rates, cfg.upsample_kernels):
            out_ch = ch // 2
            self.ups.append(nn.ConvTranspose1d(ch, out_ch, kernel, stride=rate, padding=(kernel - rate) // 2))
            for rk in cfg.resblock_kernels:
                self.resblocks.append(ResBlock1D(out_ch, rk))
            ch = out_ch
        self.post = nn.Conv1d(ch, 1, 7, padding=3)
        self.num_kernels = len(cfg.resblock_kernels)

    def forward(self, z):
        x = self.pre(z)
        rb = 0
        for up in self.ups:
            x = F.leaky_relu(x, 0.1)
            x = up(x)
            acc = 0.0
            for _ in range(self.num_kernels):
                acc = acc + self.resblocks[rb](x)
                rb += 1
            x = acc / self.num_kernels
        x = torch.tanh(self.post(F.leaky_relu(x, 0.1)))
        return x


class ScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        chans = [1, 32, 128, 256, 512, 512]
        kernels = [15, 41, 41, 41, 5]
        strides = [1, 4, 4, 4, 1]
        groups = [1, 4, 16, 16, 1]
        layers = []
        for a, b, k, s, g in zip(chans[:-1], chans[1:], kernels, strides, groups):
            layers.append(nn.Conv1d(a, b, k, stride=s, padding=k // 2, groups=g))
        self.layers = nn.ModuleList(layers)
        self.post = nn.Conv1d(chans[-1], 1, 3, padding=1)

    def forward(self, x):
        fmap = []
        for layer in self.layers:
            x = F.leaky_relu(layer(x), 0.1)
            fmap.append(x)
        x = self.post(x)
        fmap.append(x)
        return x.flatten(1), fmap


class PeriodDiscriminator(nn.Module):
    def __init__(self, period: int):
        super().__init__()
        self.period = period
        chans = [1, 32, 128, 256, 512]
        self.layers = nn.ModuleList([
            nn.Conv2d(chans[i], chans[i + 1], (5, 1), (3, 1), padding=(2, 0))
            for i in range(len(chans) - 1)
        ])
        self.post = nn.Conv2d(chans[-1], 1, (3, 1), padding=(1, 0))

    def forward(self, x):
        b, c, t = x.shape
        if t % self.period:
            pad = self.period - (t % self.period)
            x = F.pad(x, (0, pad), mode="reflect")
            t = t + pad
        x = x.view(b, c, t // self.period, self.period)
        fmap = []
        for layer in self.layers:
            x = F.leaky_relu(layer(x), 0.1)
            fmap.append(x)
        x = self.post(x)
        fmap.append(x)
        return x.flatten(1), fmap


class MultiDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discs = nn.ModuleList([ScaleDiscriminator(), PeriodDiscriminator(2), PeriodDiscriminator(3), PeriodDiscriminator(5)])

    def forward(self, y_real, y_fake):
        real, fake, fmap_r, fmap_f = [], [], [], []
        for d in self.discs:
            r, fr = d(y_real)
            f, ff = d(y_fake)
            real.append(r); fake.append(f); fmap_r.append(fr); fmap_f.append(ff)
        return real, fake, fmap_r, fmap_f


def discriminator_loss(real_outputs: Iterable[torch.Tensor], fake_outputs: Iterable[torch.Tensor]):
    loss = 0.0
    for r, f in zip(real_outputs, fake_outputs):
        loss = loss + torch.mean((1.0 - r) ** 2) + torch.mean(f ** 2)
    return loss


def generator_adv_loss(fake_outputs: Iterable[torch.Tensor]):
    return sum(torch.mean((1.0 - f) ** 2) for f in fake_outputs)


def feature_matching_loss(fmap_r, fmap_f):
    loss = 0.0
    for rs, fs in zip(fmap_r, fmap_f):
        for r, f in zip(rs, fs):
            loss = loss + torch.mean(torch.abs(r.detach() - f))
    return loss * 2.0


def kl_loss(z_p, logs_q, m_p, logs_p, mask):
    # KL(q(z|y) transformed through flow || p(z|x, alignment)).
    kl = logs_p - logs_q - 0.5 + 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    return torch.sum(kl * mask) / torch.clamp_min(torch.sum(mask), 1.0)


class FinnishVitsLite(nn.Module):
    def __init__(self, n_vocab: int, cfg: VitsLiteConfig):
        super().__init__()
        self.cfg = cfg
        self.n_vocab = n_vocab
        self.text = TextPriorEncoder(n_vocab, cfg)
        self.posterior = PosteriorEncoder(cfg)
        self.flow = Flow(cfg)
        self.duration = DurationPredictor(cfg)
        self.generator = Generator(cfg)

    def forward(self, tokens, bert, token_lengths, spec, spec_lengths, *, sample_posterior: bool = True):
        h, m_p, logs_p, x_mask = self.text(tokens, bert, token_lengths)
        z, m_q, logs_q, y_mask = self.posterior(spec, spec_lengths, sample=sample_posterior)
        z_p = self.flow(z, y_mask, reverse=False)

        # Gaussian prior score for monotonic alignment search.
        with torch.no_grad():
            s_p_sq_r = torch.exp(-2.0 * logs_p)
            neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, dim=1, keepdim=True).transpose(1, 2)
            neg_cent2 = torch.matmul((-0.5 * (z_p ** 2)).transpose(1, 2), s_p_sq_r)
            neg_cent3 = torch.matmul(z_p.transpose(1, 2), (m_p * s_p_sq_r))
            neg_cent4 = torch.sum(-0.5 * (m_p ** 2) * s_p_sq_r, dim=1, keepdim=True)
            neg_cent = neg_cent1 + neg_cent2.transpose(1, 2) + neg_cent3.transpose(1, 2) + neg_cent4.transpose(1, 2)
            # [B, Tx, Ty]
            attn = maximum_path(neg_cent, token_lengths, spec_lengths)
            attn = attn * x_mask.transpose(1, 2) * y_mask

        # Expand text prior to acoustic length.
        m_p_exp = torch.matmul(attn.transpose(1, 2), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p_exp = torch.matmul(attn.transpose(1, 2), logs_p.transpose(1, 2)).transpose(1, 2)
        w = torch.sum(attn, dim=2, keepdim=False).unsqueeze(1)
        logw_target = torch.log1p(w) * x_mask
        logw_pred = self.duration(h, x_mask)

        z_slice, starts = rand_slice_segments(z, spec_lengths, self.cfg.segment_frames)
        y_hat = self.generator(z_slice)
        return {
            "y_hat": y_hat,
            "starts": starts,
            "attn": attn,
            "w": w,
            "logw_target": logw_target,
            "logw_pred": logw_pred,
            "z_p": z_p,
            "z_p_mean": self.flow(m_q * y_mask, y_mask, reverse=False),
            "m_q": m_q,
            "logs_q": logs_q,
            "m_p_exp": m_p_exp,
            "logs_p_exp": logs_p_exp,
            "y_mask": y_mask,
            "x_mask": x_mask,
        }

    @torch.no_grad()
    def infer(self, tokens, bert, token_lengths, *, noise_scale=0.667, length_scale=1.0, max_frames=2500):
        h, m_p, logs_p, x_mask = self.text(tokens, bert, token_lengths)
        logw = self.duration(h, x_mask)
        w = (torch.exp(logw) - 1.0) * x_mask * float(length_scale)
        w = torch.clamp(torch.ceil(w), min=1.0) * x_mask
        y_lengths = torch.clamp_min(torch.sum(w, dim=[1, 2]).long(), 1)
        if int(y_lengths.max().item()) > max_frames:
            scale = max_frames / float(y_lengths.max().item())
            w = torch.clamp(torch.floor(w * scale), min=1.0) * x_mask
            y_lengths = torch.sum(w, dim=[1, 2]).long()
        ty = int(y_lengths.max().item())
        attn = torch.zeros(tokens.size(0), tokens.size(1), ty, device=tokens.device, dtype=m_p.dtype)
        for b in range(tokens.size(0)):
            pos = 0
            tx = int(token_lengths[b].item())
            for i in range(tx):
                dur = max(1, int(w[b, 0, i].item()))
                end = min(ty, pos + dur)
                attn[b, i, pos:end] = 1.0
                pos = end
            if pos < int(y_lengths[b].item()):
                attn[b, tx - 1, pos : int(y_lengths[b].item())] = 1.0
        m_exp = torch.matmul(attn.transpose(1, 2), m_p.transpose(1, 2)).transpose(1, 2)
        logs_exp = torch.matmul(attn.transpose(1, 2), logs_p.transpose(1, 2)).transpose(1, 2)
        y_mask = sequence_mask(y_lengths, ty).unsqueeze(1).to(m_p.dtype)
        z_p = (m_exp + torch.randn_like(m_exp) * torch.exp(logs_exp) * float(noise_scale)) * y_mask
        z = self.flow(z_p, y_mask, reverse=True)
        audio = self.generator(z) * sequence_mask(y_lengths * self.cfg.hop_length, z.size(-1) * self.cfg.hop_length).unsqueeze(1).to(z.dtype)
        return audio, attn, y_lengths, w
