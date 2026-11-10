from __future__ import annotations

"""v63 emotion strengthening: style difference in the ACOUSTIC latent z.

The v62 style branch shifts the text prior and stays tiny (mean squared output
~0.0008), so voice quality (Ama brightness +4 dB in real recordings) barely
reaches Finnish.  The posterior encoder never sees language, so its latent z
(the generator's input) is language-independent acoustics.  Here:

  build  : for Japanese Normal/style pairs of the same speaker and sentence with
           identical phone sequences, align each to its tokens (MAS against the
           frozen v62 prior), average the posterior mean z per token, and
           accumulate  z(style) - z(normal)  per phone symbol (Finnish letters
           reach these symbols through the v61 proxy table, e.g. ae->a).
  apply  : at synthesis, add  strength_z * delta[symbol of the frame's token]
           to z before the generator (5-frame smoothing across boundaries).

No training; statistics only.  Original Zundanen code (MIT).
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (HERE, ROOT / "emotion_tts", ROOT / "emotion_tts", ROOT / "emotion_tts"):
    sys.path.insert(0, str(p))
from emo_common import read_csv  # noqa: E402
from emo_vits import import_research  # noqa: E402
from style_branch import branch_token_table, style_mix_tensor  # noqa: E402
from style_branch_train import _path_from_w  # noqa: E402
from pitch_model import load_v62  # noqa: E402
from pitch_train import V62Dataset, collate62  # noqa: E402

import_research(ROOT)
from mas_alignment import mas_attention  # noqa: E402
from train_vits import expand_prior  # noqa: E402

STYLES = ("ama", "sexy", "tsun")


@torch.no_grad()
def token_means(base, b, normal_idx, dev):
    """Posterior-mean z averaged per token, via MAS against the frozen prior."""
    tok, bert, tl = b["tokens"].to(dev), b["bert"].to(dev), b["token_lengths"].to(dev)
    spec, sl = b["spec"].to(dev), b["spec_lengths"].to(dev)
    z, mq, lq, ym = base.posterior(spec, sl)
    zp = base.flow(z, ym, reverse=False)
    n = tok.size(0)
    h, mp, lp, xm = base.text(tok, bert, tl, speaker_id=b["speaker"].to(dev),
                              style_id=torch.full((n,), normal_idx, device=dev), lang_id=b["lang"].to(dev),
                              has_bert=b["has_bert"].to(dev), pitch=b["pitch"].to(dev))
    attn = mas_attention(zp, mp, lp, xm, ym, tl, sl)                 # [B,Tx,Ty]
    cnt = attn.sum(-1)                                              # [B,Tx]
    means = torch.einsum("bxy,bcy->bxc", attn, mq) / cnt.clamp_min(1).unsqueeze(-1)
    return means, cnt


def build(args):
    dev = torch.device("cuda")
    v62 = ROOT / "outputs/v62" / args.v62_run
    base_dir = Path(args.base_dir) if args.base_dir else v62 / "base"
    base_cp = sorted((base_dir / "checkpoints").glob("v62_*.pt"))[-1]
    ck, base, *_ = load_v62(ROOT, base_cp, dev)
    base.eval()
    stats, vocab = ck["pitch_stats"], ck["vocab"]
    ja = read_csv(v62 / "data/ja_pitch.csv")
    seen = {t for r in ja if r["split"] == "train" for t in r["tokens"].split(" ")}
    table, _mask, info = branch_token_table(vocab, seen)
    normal_idx = ck["styles"].index("normal")
    rows = [r for r in ja if r["split"] == "train"]
    key = {(r["speaker"], r["style"], r["sentence_id"]): r for r in rows}
    sums = defaultdict(lambda: torch.zeros(192, device=dev))
    cnts = defaultdict(int)
    pairs = 0
    for (spk, st, sid), r in sorted(key.items()):
        if st not in STYLES or (spk, "normal", sid) not in key:
            continue
        rn = key[(spk, "normal", sid)]
        if rn["tokens"] != r["tokens"]:
            continue                                 # reading variants: not aligned
        ds = V62Dataset([rn, r], vocab, stats["range_center_st"])
        b = collate62([ds[0], ds[1]])
        means, cnt = token_means(base, b, normal_idx, dev)
        L = int(b["token_lengths"][0])
        ids = b["tokens"][0, :L]
        for i in range(L):
            if cnt[0, i] < 2 or cnt[1, i] < 2:
                continue
            sym = int(table[ids[i]])
            if sym == 0:
                continue
            d = means[1, i] - means[0, i]
            for k in ((spk, st, sym), ("*", st, sym), (spk, st, -1), ("*", st, -1)):
                sums[k] += d
                cnts[k] += 1
        pairs += 1
    delta = {k: (sums[k] / cnts[k]).cpu() for k in sums}
    out = ROOT / "outputs/v63" / args.run / args.out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema": "zundanen-v63-zdelta-v1", "base_checkpoint": str(base_cp), "delta": delta,
                "counts": dict(cnts), "symbols": info["symbols"], "seen_symbols": sorted(seen), "pairs": pairs}, out)
    for st in STYLES:
        g = delta.get(("zundamon", st, -1))
        print(f"[ZDELTA] zundamon {st}: pairs-tokens={cnts[('zundamon', st, -1)]} |mean delta|={float(g.norm()):.3f}" if g is not None else f"[ZDELTA] {st}: none")
    print(f"[PASS] {pairs} aligned pairs -> {out}")


def frame_delta(zd, model, tok, attn, speaker, style, min_count=5, smooth=5):
    """[1,192,T] delta for every output frame from its token's phone symbol."""
    sym = model.sym_table[tok[0]]                               # [Tx]
    glob_d = zd["delta"].get((speaker, style, -1), zd["delta"].get(("*", style, -1)))
    rows = []
    for s in sym.tolist():
        if s == 0:
            rows.append(torch.zeros(192))
            continue
        k = (speaker, style, s)
        if zd["counts"].get(k, 0) >= min_count:
            rows.append(zd["delta"][k])
        elif zd["counts"].get(("*", style, s), 0) >= min_count:
            rows.append(zd["delta"][("*", style, s)])
        else:
            rows.append(glob_d)
    D = torch.stack(rows).to(attn.device)                       # [Tx,192]
    fd = torch.einsum("xy,xc->cy", attn[0], D).unsqueeze(0)     # [1,192,Ty]
    if smooth > 1:
        k = torch.ones(192, 1, smooth, device=fd.device) / smooth
        fd = torch.nn.functional.conv1d(fd, k, padding=smooth // 2, groups=192)[..., : attn.size(-1)]
    return fd


@torch.no_grad()
def synthesize_z(model, zd, tok, bert, has_bert, *, speaker, style, lang, pitch, strength=1.0, z_strength=0.0,
                 z_style=None, noise=0.4, length_scale=1.0, seed=1234, device="cuda"):
    """v62_branch.synthesize + optional acoustic-latent style delta."""
    tok = tok.unsqueeze(0).to(device)
    bert = bert.unsqueeze(0).to(device)
    tl = torch.tensor([tok.size(1)], device=device)
    h, mp, lp, xm, dlogw, _ = model.prior(
        tok, bert, tl, speaker_id=torch.tensor([model.speakers.index(speaker)], device=device),
        lang_id=torch.tensor([model.langs.index(lang)], device=device),
        has_bert=torch.tensor([float(has_bert)], device=device),
        style_mix=style_mix_tensor(model.styles, style, strength, device=device),
        pitch=torch.tensor([pitch], dtype=torch.float32, device=device))
    w = torch.clamp(torch.ceil(torch.exp(model.log_duration(h, xm, dlogw)) * xm * float(length_scale)), min=1.0) * xm
    attn, yl = _path_from_w(w, tl)
    ym = (torch.arange(attn.size(-1), device=device).unsqueeze(0) < yl.unsqueeze(1)).unsqueeze(1).to(mp.dtype)
    m, logs = expand_prior(attn, mp, lp)
    g = torch.Generator(device=device).manual_seed(int(seed))
    zprior = m + torch.randn(m.shape, generator=g, device=device, dtype=m.dtype) * torch.exp(logs) * noise
    z = model.base.flow(zprior * ym, ym, reverse=True) * ym
    if z_strength:
        z = z + z_strength * frame_delta(zd, model, tok, attn, speaker, z_style or style) * ym
    return model.base.generator(z)[0, 0].float().cpu().numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v62-run", default="v62_main")
    ap.add_argument("--run", default="v63_main")
    ap.add_argument("--base-dir", default="", help="folder with checkpoints/ of the base to use (default v62 base)")
    ap.add_argument("--out-name", default="zdelta.pt")
    args = ap.parse_args()
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
