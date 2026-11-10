from __future__ import annotations

"""v62: v60 text encoder + explicit pitch condition (level, range).

Original Zundanen code (MIT).  The pitch projection is zero-initialised, so a
v62 model built from a v60 checkpoint is exactly the v60 model until trained.

pitch vector = [level_st / 4, (range_st - range_center) / 4]
Both are measured from audio (RMVPE) at training time and taken from real
recording presets (or a user value) at inference time.  Pitch varies inside
the Finnish data too (RVC bridge at +8/+12/+16), so unlike style it cannot be
switched off by language.
"""

import sys
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT / "emotion_tts", ROOT / "emotion_tts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from emo_vits import V60TextEncoder, build_model, import_research  # noqa: E402
from style_branch import V61  # noqa: E402

SCHEMA = "zundanen-v62-pitch-conditioned-vits-v1"
BRANCH_SCHEMA = "zundanen-v62-style-branch-v1"


class V62TextEncoder(V60TextEncoder):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.pitch_proj = nn.Linear(2, self.hidden)
        nn.init.zeros_(self.pitch_proj.weight)
        nn.init.zeros_(self.pitch_proj.bias)

    def forward(self, tokens, bert, lengths, *, speaker_id, style_id, lang_id, has_bert, style_mix=None, pitch=None):
        if pitch is None:
            pitch = torch.zeros(tokens.size(0), 2, device=tokens.device)
        # Same body as V60TextEncoder.forward with the pitch term added to the condition.
        import math
        ids = torch.arange(tokens.size(1), device=tokens.device)
        mb = ids.unsqueeze(0) < lengths.unsqueeze(1)
        x = self.token_emb(tokens) * math.sqrt(self.hidden)
        hb = has_bert.to(bert.dtype).view(-1, 1, 1)
        bert = bert.float() * hb + self.null_bert.view(1, 1, -1) * (1.0 - hb)
        x = x + self.bert_proj(self.bert_norm(bert.float()))
        cond = (self.style_proj(self.style_vector(style_id, style_mix)) + self.speaker_emb(speaker_id)
                + self.lang_emb(lang_id) + self.pitch_proj(pitch.to(x.dtype)))
        x = x + cond.unsqueeze(1)
        x = x.transpose(1, 2)
        mask = mb.unsqueeze(1).to(x.dtype)
        x = self.encoder(x * mask, mask)
        x = (x + self.cond_post(cond).unsqueeze(-1)) * mask
        stats = self.proj(x) * mask
        m, logs = stats.chunk(2, dim=1)
        return x, m, logs, mask


def build_v62(root: Path, n_vocab, n_speakers, n_styles, n_langs, device):
    model, models, losses, vendor = build_model(root, n_vocab, n_speakers, n_styles, n_langs, "cpu")
    import_research(root)
    from mas_vits_model import load_mit_vits_vendor
    _m, attentions, _l, _v = load_mit_vits_vendor(Path(root))
    model.text = V62TextEncoder(attentions, n_vocab, n_speakers, n_styles, n_langs)
    return model.to(device), models, losses, vendor


def pitch_vector(level_st: float, range_st: float, range_center: float) -> list[float]:
    return [float(level_st) / 4.0, (float(range_st) - float(range_center)) / 4.0]


def finnish_pitch(stats: dict, speaker: str, style: str, strength: float, level_base: float | None = None) -> list[float]:
    """Pitch vector for FINNISH output.

    Level: level_base (default = the speaker's real Normal, 0 st) plus the real
    Japanese style offset of that speaker times strength (Zundamon Ama +1.4 st).
    Range: Japanese has a much wider intonation range than Finnish (Zundamon
    Normal 12.5 st vs Finnish data ~6.3 st), so the Japanese absolute range is
    NOT copied.  The Finnish range of that speaker is scaled by the Japanese
    style/Normal range ratio (Ama 0.77x) raised to strength.
    """
    presets, groups = stats["presets_from_real_recordings"], stats["groups"]
    ja = presets.get(speaker if speaker in presets and "ama" in presets[speaker] else "zundamon")
    n, s = ja["normal"], ja.get(style, ja["normal"])
    if speaker == "fi_chatterbox":
        fi_range = presets["fi_chatterbox"]["normal"]["range_st"]
    else:
        vals = [g["range_median"] for k, g in groups.items() if k.startswith("bridge_p") and k.endswith(":" + speaker)]
        fi_range = float(sorted(vals)[len(vals) // 2]) if vals else n["range_st"]
    base = presets[speaker]["normal"]["level_st"] if level_base is None else level_base
    level = base + strength * (s["level_st"] - n["level_st"])
    rng = fi_range * (s["range_st"] / n["range_st"]) ** strength
    return pitch_vector(level, rng, stats["range_center_st"])


def load_v62(root: Path, path: Path, device):
    ck = torch.load(Path(path), map_location="cpu", weights_only=False)
    if ck.get("schema") != SCHEMA:
        raise RuntimeError(f"not a v62 checkpoint: {path}")
    model, models, losses, _ = build_v62(root, len(ck["vocab"]), len(ck["speakers"]), len(ck["styles"]),
                                         len(ck["langs"]), device)
    model.load_state_dict(ck["model"], strict=True)
    return ck, model, models, losses


class V62Combo(V61):
    """Frozen v62 base + language-blind style branch; pitch passed to the base."""

    def prior(self, tok, bert, tl, *, speaker_id, lang_id, has_bert, style_mix, pitch=None):
        b = tok.size(0)
        normal_ids = torch.full((b,), self.normal, dtype=torch.long, device=tok.device)
        h, mp, lp, xm = self.base.text(tok, bert, tl, speaker_id=speaker_id, style_id=normal_ids,
                                       lang_id=lang_id, has_bert=has_bert, pitch=pitch)
        sym = self.sym_table[tok]
        mask = self.sym_mask[tok] * xm[:, 0]
        dm, dlogs, dlogw = self.branch(sym, mask, style_mix, self.normal, speaker_id)
        return h, mp + dm, lp + dlogs, xm, dlogw, (dm, dlogs, dlogw)
