from __future__ import annotations

"""v61 language-blind style branch on top of a finished v60 model.

Original Zundanen code (MIT).  No Style-Bert-VITS2 / AGPL source is used.

Why v60 did not transfer
------------------------
Every Finnish training row was Normal.  The v60 text encoder sees language,
FinBERT and style together, so it could learn "if Finnish, ignore style" at no
cost.  Style worked in Japanese and was silently gated off in Finnish.

v61 design
----------
prior = v60 base(text, FinBERT, language, speaker, style=NORMAL)
      + StyleBranch(phone symbols, speaker, style) - StyleBranch(..., normal)

The branch has no access to language ID, FinBERT, or the <SP> token, and only a
7-token receptive field (3 conv layers, kernel 3), so it cannot recognise
"this is Finnish" and switch itself off.  It sees Finnish letters through a
fixed proxy table onto symbols that occur in Japanese (ä->a, ö->o, y->u, l->r,
...).  What it learns on Japanese Normal/Ama pairs is therefore added to Finnish
in the same way.  It predicts per-token shifts of the prior mean, log-scale and
log-duration.  Normal is exactly zero by construction (f(style) - f(normal)),
so style=normal reproduces v60 bit-for-bit.
"""

import sys
from pathlib import Path

import torch
from torch import nn

SCHEMA = "zundanen-v61-language-blind-style-branch-v1"

# Finnish-only symbols -> nearest symbol that occurs in the Japanese data.
PROXY = {"ä": "a", "ö": "o", "y": "u", "å": "o", "l": "r", "c": "k", "q": "k", "w": "v", "x": "k",
         ".": ",", "!": ",", "?": ",", ";": ",", ":": ","}


def root_paths(root: Path):
    for p in (Path(root) / "emotion_tts", Path(root) / "emotion_tts"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def branch_token_table(vocab: dict[str, int], seen: set[str]) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """vocab id -> branch symbol id (0 = masked) and a 0/1 mask per vocab id."""
    symbols = sorted(seen)
    sym_id = {s: i + 1 for i, s in enumerate(symbols)}
    table = torch.zeros(len(vocab), dtype=torch.long)
    mask = torch.zeros(len(vocab))
    mapping = {}
    for t, i in vocab.items():
        s = t if t in seen else PROXY.get(t)
        if s is not None and s in sym_id:
            table[i] = sym_id[s]
            mask[i] = 1.0
            mapping[t] = s
    return table, mask, {"symbols": symbols, "mapping": mapping}


class StyleBranch(nn.Module):
    def __init__(self, n_symbols: int, n_styles: int, n_speakers: int, *, hidden: int = 192, latent: int = 192,
                 style_dim: int = 64, use_speaker: bool = True, max_dlogs: float = 1.0):
        super().__init__()
        self.sym_emb = nn.Embedding(n_symbols + 1, hidden, padding_idx=0)
        self.style_emb = nn.Embedding(n_styles, style_dim)
        nn.init.normal_(self.style_emb.weight, 0.0, 0.5)
        self.speaker_emb = nn.Embedding(n_speakers, style_dim) if use_speaker else None
        self.cond = nn.Linear(style_dim, hidden)
        self.convs = nn.ModuleList([nn.Conv1d(hidden, hidden, 3, padding=1) for _ in range(3)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(3)])
        self.drop = nn.Dropout(0.1)
        self.head_m = nn.Conv1d(hidden, latent, 1)
        self.head_logs = nn.Conv1d(hidden, latent, 1)
        self.head_logw = nn.Conv1d(hidden, 1, 1)
        for h in (self.head_m, self.head_logs, self.head_logw):
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)
        self.max_dlogs = float(max_dlogs)

    def _f(self, sym, mask, style_vec, speaker_id):
        c = style_vec
        if self.speaker_emb is not None:
            c = c + self.speaker_emb(speaker_id)
        x = self.sym_emb(sym) + self.cond(c).unsqueeze(1)          # [B,T,H]
        for conv, norm in zip(self.convs, self.norms):
            y = conv((x * mask.unsqueeze(-1)).transpose(1, 2)).transpose(1, 2)
            x = x + self.drop(torch.relu(norm(y)))
        x = (x * mask.unsqueeze(-1)).transpose(1, 2)              # [B,H,T]
        return self.head_m(x), torch.tanh(self.head_logs(x)) * self.max_dlogs, self.head_logw(x)

    def forward(self, sym, mask, style_mix, normal_index: int, speaker_id):
        """Returns (dm, dlogs, dlogw), each [B,C,T], exactly zero for pure Normal."""
        w = self.style_emb.weight
        normal = w[normal_index].unsqueeze(0).expand(sym.size(0), -1)
        a = self._f(sym, mask, style_mix.to(w.dtype) @ w, speaker_id)
        b = self._f(sym, mask, normal, speaker_id)
        m = mask.unsqueeze(1)
        return tuple((x - y) * m for x, y in zip(a, b))


class V61(nn.Module):
    """Frozen v60 base + trainable StyleBranch."""

    def __init__(self, base, branch: StyleBranch, table: torch.Tensor, mask: torch.Tensor, styles, speakers, langs):
        super().__init__()
        self.base = base
        self.branch = branch
        self.register_buffer("sym_table", table, persistent=False)
        self.register_buffer("sym_mask", mask, persistent=False)
        self.styles, self.speakers, self.langs = list(styles), list(speakers), list(langs)
        self.normal = self.styles.index("normal")

    def prior(self, tok, bert, tl, *, speaker_id, lang_id, has_bert, style_mix):
        b = tok.size(0)
        normal_ids = torch.full((b,), self.normal, dtype=torch.long, device=tok.device)
        h, mp, lp, xm = self.base.text(tok, bert, tl, speaker_id=speaker_id, style_id=normal_ids,
                                       lang_id=lang_id, has_bert=has_bert)
        sym = self.sym_table[tok]
        mask = self.sym_mask[tok] * xm[:, 0]
        dm, dlogs, dlogw = self.branch(sym, mask, style_mix, self.normal, speaker_id)
        return h, mp + dm, lp + dlogs, xm, dlogw, (dm, dlogs, dlogw)

    def log_duration(self, h, xm, dlogw):
        return self.base.duration(h, xm) + dlogw * xm


def style_mix_tensor(styles, style: str, strength: float, batch: int = 1, device="cpu"):
    mix = torch.zeros(batch, len(styles), device=device)
    mix[:, styles.index("normal")] += 1.0 - strength
    mix[:, styles.index(style)] += strength
    return mix


def load_v61(root: Path, path: Path, device):
    root_paths(root)
    from emo_vits import load_v60_checkpoint
    ck = torch.load(Path(path), map_location="cpu", weights_only=False)
    if ck.get("schema") != SCHEMA:
        raise RuntimeError(f"not a v61 checkpoint: {path}")
    base_ck, base, _models, _losses = load_v60_checkpoint(root, Path(ck["base_checkpoint"]), device)
    table, mask, _info = branch_token_table(base_ck["vocab"], set(ck["seen_symbols"]))
    branch = StyleBranch(len(ck["seen_symbols"]), len(base_ck["styles"]), len(base_ck["speakers"]),
                         use_speaker=bool(ck.get("use_speaker", True)))
    branch.load_state_dict(ck["branch"], strict=True)
    model = V61(base, branch, table, mask, base_ck["styles"], base_ck["speakers"], base_ck["langs"]).to(device)
    model.eval()
    return ck, base_ck, model
