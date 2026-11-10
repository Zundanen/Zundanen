from __future__ import annotations

"""v60 multilingual / multi-speaker / multi-style text prior on top of Finnish VITS.

Original Zundanen code (MIT).  Uses the MIT jaywalnut310/VITS vendor modules
through the existing Step6/7 glue.  No Style-Bert-VITS2 / AGPL source is used.

Only the text encoder changes.  Posterior encoder, Transformer flow, duration
predictor and HiFi-GAN generator keep the Finnish VITS geometry and are initialised
from the Finnish VITS checkpoint, so Finnish pronunciation starts from the finished
model instead of from scratch (the failure mode of Step13).

Conditioning (all added to every token, plus a zero-initialised post-encoder
shift so a condition can move the prior mean directly):
  language  fi / ja            new, zero-initialised
  speaker   fi_chatterbox ...  every row starts from the Finnish VITS speaker row
  style     normal ama ...     every row starts from the Finnish VITS style row
FinBERT is Finnish-only.  Japanese rows use a learned null vector instead.
"""

import math
import sys
from pathlib import Path

import torch
from torch import nn

SCHEMA = "zundanen-v60-multilingual-style-vits-v1"
STEP7_SCHEMA = "zundanen-step7a-transformer-flow-mas-finbert-large-v1"


def research_dir(root: Path) -> Path:
    return Path(root) / "research" / "finnish_style_tts"


def import_research(root: Path):
    r = str(research_dir(root))
    if r not in sys.path:
        sys.path.insert(0, r)


class V60TextEncoder(nn.Module):
    def __init__(self, attentions_module, n_vocab: int, n_speakers: int, n_styles: int, n_langs: int,
                 *, bert_dim: int = 1024, hidden: int = 192, latent: int = 192, style_dim: int = 128):
        super().__init__()
        self.hidden = hidden
        self.token_emb = nn.Embedding(n_vocab, hidden)
        self.bert_norm = nn.LayerNorm(bert_dim)
        self.bert_proj = nn.Linear(bert_dim, hidden)
        self.null_bert = nn.Parameter(torch.zeros(bert_dim))
        self.style_emb = nn.Embedding(n_styles, style_dim)
        self.style_proj = nn.Linear(style_dim, hidden)
        self.speaker_emb = nn.Embedding(n_speakers, hidden)
        self.lang_emb = nn.Embedding(n_langs, hidden)
        nn.init.zeros_(self.lang_emb.weight)
        self.cond_post = nn.Linear(hidden, hidden)
        nn.init.zeros_(self.cond_post.weight)
        nn.init.zeros_(self.cond_post.bias)
        self.encoder = attentions_module.Encoder(hidden, 768, 2, 6, 3, 0.1)
        self.proj = nn.Conv1d(hidden, latent * 2, 1)

    def style_vector(self, style_id: torch.Tensor, style_mix: torch.Tensor | None = None) -> torch.Tensor:
        """[B,style_dim].  style_mix=[B,n_styles] weights allow strength/blend."""
        if style_mix is not None:
            return style_mix.to(self.style_emb.weight.dtype) @ self.style_emb.weight
        return self.style_emb(style_id)

    def forward(self, tokens, bert, lengths, *, speaker_id, style_id, lang_id, has_bert,
                style_mix=None):
        ids = torch.arange(tokens.size(1), device=tokens.device)
        mb = ids.unsqueeze(0) < lengths.unsqueeze(1)
        x = self.token_emb(tokens) * math.sqrt(self.hidden)
        hb = has_bert.to(bert.dtype).view(-1, 1, 1)
        bert = bert.float() * hb + self.null_bert.view(1, 1, -1) * (1.0 - hb)
        x = x + self.bert_proj(self.bert_norm(bert.float()))
        cond = (self.style_proj(self.style_vector(style_id, style_mix))
                + self.speaker_emb(speaker_id) + self.lang_emb(lang_id))
        x = x + cond.unsqueeze(1)
        x = x.transpose(1, 2)
        mask = mb.unsqueeze(1).to(x.dtype)
        x = self.encoder(x * mask, mask)
        x = (x + self.cond_post(cond).unsqueeze(-1)) * mask
        stats = self.proj(x) * mask
        m, logs = stats.chunk(2, dim=1)
        return x, m, logs, mask


def build_model(root: Path, n_vocab: int, n_speakers: int, n_styles: int, n_langs: int, device):
    import_research(root)
    from mas_vits_model import MasFullJointVits, load_mit_vits_vendor
    from transformer_flow import TransformerCouplingBlock

    models, attentions, losses, vendor = load_mit_vits_vendor(Path(root))
    model = MasFullJointVits(models, attentions, n_vocab)
    model.flow = TransformerCouplingBlock(channels=192, hidden_channels=256, n_heads=4,
                                          n_layers=2, n_flows=4, ffn_channels=768)
    model.text = V60TextEncoder(attentions, n_vocab, n_speakers, n_styles, n_langs)
    return model.to(device), models, losses, vendor


def init_from_step7(model, step7_state: dict, old_vocab: dict[str, int], new_vocab: dict[str, int],
                    new_token_init: dict[str, str], *, seed: int = 20261018) -> dict:
    """Copy Finnish VITS weights; expand token/speaker/style tables.  Returns a report."""
    g = torch.Generator().manual_seed(seed)
    own = model.state_dict()
    report = {"copied": 0, "expanded": [], "new": []}
    for k, v in step7_state.items():
        if k not in own:
            continue
        if k in ("text.token_emb.weight", "text.style_emb.weight", "text.speaker_emb.weight"):
            continue
        if own[k].shape != v.shape:
            raise RuntimeError(f"unexpected shape for {k}: {tuple(v.shape)} vs {tuple(own[k].shape)}")
        own[k] = v.clone()
        report["copied"] += 1

    old_tok = step7_state["text.token_emb.weight"]
    tok = own["text.token_emb.weight"]
    for t, i in new_vocab.items():
        src = old_vocab.get(t)
        if src is None:
            src = old_vocab[new_token_init[t]]
            report["new"].append(f"token {t!r} <- {new_token_init[t]!r}")
        tok[i] = old_tok[src]
    own["text.token_emb.weight"] = tok

    for name in ("style_emb", "speaker_emb"):
        old = step7_state[f"text.{name}.weight"][0]
        new = own[f"text.{name}.weight"]
        for i in range(new.size(0)):
            # Row 0 is the exact Finnish VITS row; others start from it plus a small
            # seeded perturbation so they can separate.
            noise = 0.0 if i == 0 else torch.randn(old.shape, generator=g) * 0.02 * (old.std() + 1e-3)
            new[i] = old + noise
        own[f"text.{name}.weight"] = new
        report["expanded"].append(f"{name}: 1 -> {new.size(0)}")
    model.load_state_dict(own, strict=True)
    return report


def load_v60_checkpoint(root: Path, path: Path, device):
    ck = torch.load(Path(path), map_location="cpu", weights_only=False)
    if ck.get("schema") != SCHEMA:
        raise RuntimeError(f"not a v60 checkpoint: {path} schema={ck.get('schema')!r}")
    model, models, losses, vendor = build_model(
        root, len(ck["vocab"]), len(ck["speakers"]), len(ck["styles"]), len(ck["langs"]), device)
    model.load_state_dict(ck["model"], strict=True)
    return ck, model, models, losses
