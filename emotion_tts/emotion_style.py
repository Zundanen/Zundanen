from __future__ import annotations

"""v65 styles on the F0-conditioned base (Ama / Sexy / Tsun).

  branch : v61/v62 language-blind style branch; the frozen base is conditioned on
           each utterance's ground-truth F0 contour, so the branch learns only the
           non-pitch part of a style.  4k steps, held-out validation loss.
  zdelta : acoustic-latent style delta (as v63) for the v65 base.
  eval   : Finnish x Zundamon: Normal with range scale k, and styles with the
           v63 preset (z 0.5), all with an explicit F0 contour.

  runtime/tts/.venv/Scripts/python.exe -I emotion_tts/emotion_style.py branch|zdelta|eval
"""

import argparse
import csv
import html
import json
import random
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (HERE, ROOT / "emotion_tts", ROOT / "emotion_tts", ROOT / "emotion_tts", ROOT / "emotion_tts"):
    sys.path.insert(0, str(p))
from emo_common import read_csv  # noqa: E402
from emo_vits import import_research  # noqa: E402
from emo_train_base import write_wav  # noqa: E402
from style_branch import StyleBranch, branch_token_table, style_mix_tensor  # noqa: E402
from style_branch_train import _path_from_w  # noqa: E402
from pitch_model import V62Combo, pitch_vector  # noqa: E402
from style_zdelta import frame_delta, token_means  # noqa: E402
from emotion_model import flow_apply, load_v65, shape_contour  # noqa: E402
from emotion_train import DATA, V65Dataset, collate65, expand_frames  # noqa: E402

import_research(ROOT)
from mas_alignment import mas_attention  # noqa: E402
from train_vits import expand_prior, vits_kl  # noqa: E402

OUT = ROOT / "outputs/v65"
TEXTS = ["Hyvää huomenta! Tänään on kaunis päivä.",
         "Voisitko tulla tänne hetkeksi? Minulla on sinulle jotain.",
         "En minä sinua odottanut, älä luulekaan."]
STRENGTH = {"normal": 0.0, "ama": 1.5, "sexy": 1.5, "tsun": 1.0}


BASE_NAME = "base"
BRANCH_NAME = "branch"


def base_ckpt():
    return sorted((OUT / BASE_NAME / "checkpoints").glob("v65_*.pt"))[-1]


def load_all(dev, with_branch=True, bundle=None):
    """bundle: a loaded character file ({"base", "branch", "zdelta", ...}) instead of BASE_NAME / BRANCH_NAME."""
    ck, base, *_ = load_v65(ROOT, bundle["base"] if bundle else base_ckpt(), "cpu")
    stats, vocab = ck["pitch_stats"], ck["vocab"]
    bck = None
    if bundle:
        bck = bundle["branch"]
    elif with_branch:
        rows = list(csv.DictReader(open(OUT / BRANCH_NAME / "val.csv", encoding="utf-8")))
        best = min(rows, key=lambda r: float(r["val_total"]))
        bcp = OUT / BRANCH_NAME / "checkpoints" / f"v65b_{int(best['step']):06d}.pt"
        bck = torch.load(bcp, map_location="cpu", weights_only=False)
    if bundle is None and (DATA / "ja_pitch.csv").exists():
        ja = read_csv(DATA / "ja_pitch.csv")
        seen = {t for r in ja if r["split"] == "train" for t in r["tokens"].split(" ")}
    elif bck is not None and bck.get("seen_symbols"):
        # v2 release: the training data is not distributed; the branch checkpoint stores the
        # symbol set it was trained with (identical to the one computed above, checked for v73).
        ja, seen = [], set(bck["seen_symbols"])
    else:
        raise FileNotFoundError(f"{DATA / 'ja_pitch.csv'} not found and no seen_symbols in the branch checkpoint")
    table, mask, info = branch_token_table(vocab, seen)
    br = StyleBranch(len(info["symbols"]), len(ck["styles"]), len(ck["speakers"]))
    if bck is not None:
        br.load_state_dict(bck["branch"])
    model = V62Combo(base, br, table, mask, ck["styles"], ck["speakers"], ck["langs"]).to(dev)
    return ck, stats, vocab, ja, seen, model


def cmd_branch(args):
    random.seed(20261023); np.random.seed(20261023); torch.manual_seed(20261023)
    ck, stats, vocab, ja, seen, model = load_all(torch.device("cuda"), with_branch=False)
    for p in model.base.parameters():
        p.requires_grad_(False)
    model.base.eval()
    rc, ref = stats["range_center_st"], stats["reference_hz"]
    want = (lambda st: st != "normal") if not args.only_style else (lambda st: st == args.only_style)
    train = [r for r in ja if r["split"] == "train" and want(r["style"])]
    val = [r for r in ja if r["split"] == "valid" and want(r["style"])]
    if not val:   # v2 user data without the fixed validation sentences: hold out every 10th sentence number
        val = [r for r in train if int(r["sentence_id"]) % 10 == 0]
        train = [r for r in train if int(r["sentence_id"]) % 10 != 0]
    print(f"[DATA] style rows train={len(train)} valid={len(val)} only_style={args.only_style or 'all'}", flush=True)
    loader = torch.utils.data.DataLoader(V65Dataset(train, vocab, rc, ref), batch_size=args.batch_size, shuffle=True,
                                         num_workers=args.workers, collate_fn=collate65, drop_last=True,
                                         persistent_workers=args.workers > 0)
    vloader = torch.utils.data.DataLoader(V65Dataset(val, vocab, rc, ref), batch_size=8, collate_fn=collate65)
    opt = torch.optim.AdamW(model.branch.parameters(), lr=2e-4, betas=(0.8, 0.99), weight_decay=1e-4)
    work = OUT / BRANCH_NAME
    (work / "checkpoints").mkdir(parents=True, exist_ok=True)
    (work / "val.csv").write_text("step,val_kl,val_dur,val_total\n", encoding="utf-8")

    def losses(b):
        tok, bert, tl = b["tokens"].cuda(), b["bert"].cuda(), b["token_lengths"].cuda()
        spec, sl = b["spec"].cuda(), b["spec_lengths"].cuda()
        mix = torch.nn.functional.one_hot(b["style"], len(model.styles)).float().cuda()
        okm = None
        with torch.no_grad():
            z, _mq, lq, ym = model.base.posterior(spec, sl)
            okm = ym * b["f0_ok"].cuda().view(-1, 1, 1)
            zp = flow_apply(model.base, z, ym, b["f0feat"].cuda() * okm, reverse=False)
        h, mp, lp, xm, dlogw, _ = model.prior(tok, bert, tl, speaker_id=b["speaker"].cuda(), lang_id=b["lang"].cuda(),
                                              has_bert=b["has_bert"].cuda(), style_mix=mix, pitch=b["pitch"].cuda())
        attn = mas_attention(zp, mp.detach(), lp.detach(), xm, ym, tl, sl)
        mpe, lpe = expand_prior(attn, mp, lp)
        with torch.no_grad():
            f0term = model.base.f0.f0_proj(b["f0feat"].cuda() * okm) * ym
        kl = vits_kl(zp, lq, mpe + f0term, lpe, ym)
        logwt = torch.log(attn.sum(dim=2).unsqueeze(1) + 1e-6) * xm
        dur = torch.sum((model.log_duration(h.detach(), xm, dlogw) - logwt) ** 2 * xm) / torch.clamp_min(xm.sum(), 1.0)
        return kl + dur, kl, dur

    def batches():
        while True:
            yield from loader
    it = batches()
    t0 = time.time()
    model.train(); model.base.eval()
    for step in range(1, args.steps + 1):
        total, kl, dur = losses(next(it))
        opt.zero_grad(set_to_none=True); total.backward()
        torch.nn.utils.clip_grad_norm_(model.branch.parameters(), 10.0); opt.step()
        if step == 1 or step % 100 == 0:
            print(f"step {step:5d}/{args.steps} KL={float(kl):.4f} dur={float(dur):.4f} {(time.time()-t0)/60:.1f}min", flush=True)
        if step % 1000 == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                v = np.mean([[float(x) for x in losses(vb)] for vb in vloader], axis=0)
            model.train(); model.base.eval()
            with (work / "val.csv").open("a", encoding="utf-8") as f:
                f.write(f"{step},{v[1]:.6f},{v[2]:.6f},{v[0]:.6f}\n")
            torch.save({"schema": "zundanen-v65-style-branch-v1", "step": step, "branch": model.branch.state_dict(),
                        "base_checkpoint": str(base_ckpt()), "seen_symbols": sorted(seen)},
                       work / "checkpoints" / f"v65b_{step:06d}.pt")
            print(f"[VAL] step {step}: KL={v[1]:.4f} dur={v[2]:.4f}", flush=True)
    print(f"[PASS] v65 branch {(time.time()-t0)/60:.1f} min")


def cmd_zdelta(args):
    dev = torch.device("cuda")
    ck, stats, vocab, ja, seen, model = load_all(dev, with_branch=False)
    base = model.base.eval()
    rc = stats["range_center_st"]
    rows = [r for r in ja if r["split"] == "train"]
    key = {(r["speaker"], r["style"], r["sentence_id"]): r for r in rows}
    sums, cnts = defaultdict(lambda: torch.zeros(192, device=dev)), defaultdict(int)
    ref = stats["reference_hz"]

    @torch.no_grad()
    def token_means65(b):
        tok, bert, tl = b["tokens"].to(dev), b["bert"].to(dev), b["token_lengths"].to(dev)
        spec, sl = b["spec"].to(dev), b["spec_lengths"].to(dev)
        z, mq, _lq, ym = base.posterior(spec, sl)
        okm = ym * b["f0_ok"].to(dev).view(-1, 1, 1)
        zp = flow_apply(base, z, ym, b["f0feat"].to(dev) * okm, reverse=False)
        n = tok.size(0)
        h, mp, lp, xm = base.text(tok, bert, tl, speaker_id=b["speaker"].to(dev), style_id=torch.zeros(n, dtype=torch.long, device=dev),
                                  lang_id=b["lang"].to(dev), has_bert=b["has_bert"].to(dev), pitch=b["pitch"].to(dev))
        attn = mas_attention(zp, mp, lp, xm, ym, tl, sl)
        cnt = attn.sum(-1)
        return torch.einsum("bxy,bcy->bxc", attn, mq) / cnt.clamp_min(1).unsqueeze(-1), cnt
    pairs = 0
    for (spk, st, sid), r in sorted(key.items()):
        if st == "normal" or (spk, "normal", sid) not in key or key[(spk, "normal", sid)]["tokens"] != r["tokens"]:
            continue
        ds = V65Dataset([key[(spk, "normal", sid)], r], vocab, rc, ref)
        b = collate65([ds[0], ds[1]])
        means, cnt = token_means65(b)
        L = int(b["token_lengths"][0])
        for i in range(L):
            if cnt[0, i] < 2 or cnt[1, i] < 2:
                continue
            sym = int(model.sym_table[b["tokens"][0, i].to(dev)])
            if sym == 0:
                continue
            d = means[1, i] - means[0, i]
            for k in ((spk, st, sym), ("*", st, sym), (spk, st, -1), ("*", st, -1)):
                sums[k] += d; cnts[k] += 1
        pairs += 1
    torch.save({"schema": "zundanen-v63-zdelta-v1", "base_checkpoint": str(base_ckpt()),
                "delta": {k: (sums[k] / cnts[k]).cpu() for k in sums}, "counts": dict(cnts), "pairs": pairs},
               OUT / f"zdelta_{BASE_NAME}.pt")
    print(f"[PASS] z-delta {pairs} pairs")


@torch.no_grad()
def synth(model, zd, tok, bert, *, style, strength, z, level_st, range_k, pitch, speaker="zundamon", lang="fi",
          noise=0.4, seed=1234, device="cuda", length_scale=1.0, tempo=None, dur_detail=0.0, token_scale=None,
          unvoiced_tokens=None):
    """tempo=None: per-token style durations from the branch (v65 original).
    tempo=float: keep the Normal per-token rhythm and scale the whole sentence by
    `tempo` (v67 fix: the branch's per-token durations scattered +-30% and broke
    Finnish long/short contrast); dur_detail in [0,1] re-adds that fraction of
    the branch's per-token deviation from its own sentence mean."""
    tok = tok.unsqueeze(0).to(device)
    bert = bert.unsqueeze(0).to(device)
    tl = torch.tensor([tok.size(1)], device=device)
    h, mp, lp, xm, dlogw, _ = model.prior(
        tok, bert, tl, speaker_id=torch.tensor([model.speakers.index(speaker)], device=device),
        lang_id=torch.tensor([model.langs.index(lang)], device=device), has_bert=torch.ones(1, device=device),
        style_mix=style_mix_tensor(model.styles, style, strength, device=device),
        pitch=torch.tensor([pitch], dtype=torch.float32, device=device))
    if tempo is None:
        logw = model.log_duration(h, xm, dlogw)
    else:
        logw = model.base.duration(h, xm)                       # Normal rhythm under the style's pitch
        if dur_detail:
            dev_ = (dlogw - (dlogw * xm).sum() / xm.sum().clamp_min(1)) * xm
            logw = logw + float(dur_detail) * dev_
        logw = logw + float(np.log(tempo))
    if token_scale:   # v73: {token_id: duration factor}, e.g. lengthen /h/; None = unchanged
        f = torch.ones_like(tok, dtype=logw.dtype)
        for tid, s in token_scale.items():
            f = torch.where(tok == int(tid), torch.full_like(f, float(s)), f)
        logw = logw + torch.log(f).unsqueeze(1) * xm
    w = torch.clamp(torch.ceil(torch.exp(logw) * xm * float(length_scale)), min=1.0) * xm
    attn, yl = _path_from_w(w, tl)
    ym = (torch.arange(attn.size(-1), device=device).unsqueeze(0) < yl.unsqueeze(1)).unsqueeze(1).to(mp.dtype)
    m, logs = expand_prior(attn, mp, lp)
    st_pred, uv = model.base.f0.predictor(expand_frames(attn, h), ym)
    feat = shape_contour(st_pred, uv, ym, level_st=level_st, range_k=range_k)
    if unvoiced_tokens:   # v73: force these tokens' frames to unvoiced (e.g. /h/); None = unchanged
        sel = torch.zeros_like(tok, dtype=feat.dtype)
        for tid in unvoiced_tokens:
            sel = torch.where(tok == int(tid), torch.ones_like(sel), sel)
        frames = torch.einsum("bx,bxy->by", sel, attn).clamp(max=1.0)    # [1, Ty] 1 on those frames
        feat = torch.stack([feat[:, 0], feat[:, 1] * (1 - frames)], dim=1)
    m = m + model.base.f0.f0_proj(feat) * ym
    g = torch.Generator(device=device).manual_seed(int(seed))
    zz = flow_apply(model.base, (m + torch.randn(m.shape, generator=g, device=device) * torch.exp(logs) * noise) * ym,
                    ym, feat, reverse=True) * ym
    if z and style != "normal":
        zz = zz + z * frame_delta(zd, model, tok, attn, speaker, style) * ym
    return model.base.generator(zz)[0, 0].float().cpu().numpy()


def cmd_eval(args):
    dev = torch.device("cuda")
    ck, stats, vocab, ja, seen, model = load_all(dev)
    model.eval()
    zd = torch.load(OUT / f"zdelta_{BASE_NAME}.pt", map_location="cpu", weights_only=False)
    pre = json.loads((ROOT / "emotion_tts/v63_preset.json").read_text(encoding="utf-8"))
    rc = stats["range_center_st"]
    zp = stats["presets_from_real_recordings"]["zundamon"]
    L0 = pre["pitch_level_st"]
    out = OUT / f"eval_{BASE_NAME}_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True)
    import_research(ROOT)
    from finbert_features import extract_features, load_finbert_model
    tokz, fb, fdev = load_finbert_model("TurkuNLP/bert-large-finnish-cased-v1", device="cuda")
    labels = {}
    ks = (1.0, 1.5, 2.0, 2.5)
    for ti, text in enumerate(TEXTS, 1):
        feat = extract_features(text, tokenizer=tokz, model=fb, device=fdev, model_id="TurkuNLP/bert-large-finnish-cased-v1")
        tok = torch.tensor([vocab[g] for g in feat.frontend_tokens])
        bert = torch.from_numpy(np.asarray(feat.features, dtype=np.float32))
        for k in ks:
            y = synth(model, zd, tok, bert, style="normal", strength=0.0, z=0.0, level_st=L0, range_k=k,
                      pitch=pitch_vector(L0, pre["pitch_range_st"], rc))
            p = out / f"fi{ti:02d}_normal_k{k:g}.wav"; write_wav(p, y); labels[str(p)] = ("normal", k)
        for st in ("ama", "sexy", "tsun"):
            a = STRENGTH[st]
            s, n = zp[st], zp["normal"]
            lvl = L0 + a * (s["level_st"] - n["level_st"])
            kk = args.style_k * (s["range_st"] / n["range_st"]) ** a
            y = synth(model, zd, tok, bert, style=st, strength=a, z=pre["z_strength"], level_st=lvl, range_k=kk,
                      pitch=pitch_vector(lvl, pre["pitch_range_st"] * (s["range_st"] / n["range_st"]) ** a, rc))
            p = out / f"fi{ti:02d}_{st}.wav"; write_wav(p, y); labels[str(p)] = (st, kk)
        print(f"[FI] {ti}", flush=True)
    (out / "list.txt").write_text("\n".join(labels), encoding="utf-8")
    rvc = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))["rvc_python"]
    subprocess.run([rvc, "-I", str(ROOT / "emotion_tts/f0_rmvpe.py"), str(out / "list.txt"), str(out / "f0.csv")], check=True, stdout=subprocess.DEVNULL)
    f0 = {r["wav_path"]: r for r in csv.DictReader(open(out / "f0.csv", encoding="utf-8-sig"))}
    agg = defaultdict(list)
    for p, key in labels.items():
        r = f0[p]
        agg[key].append((float(r["f0_median_hz"]), float(r["st_p90"]) - float(r["st_p10"])))
    lines = []
    for key, v in agg.items():
        lines.append(f"{key[0]:6s} k={key[1]:.2f}: F0 median {np.mean([x[0] for x in v]):.0f} Hz  range {np.mean([x[1] for x in v]):.1f} st")
        print("[EVAL] " + lines[-1], flush=True)
    parts = ["<!doctype html><meta charset=utf-8><title>v65 F0 control</title><style>"
             "body{font-family:sans-serif;margin:16px;background:#fcfcfb;color:#0b0b0b}"
             "td,th{padding:5px 8px;border-bottom:1px solid #ddd;text-align:left}.m{color:#52514e;font-size:12px}audio{width:210px}</style>",
             "<h1>v65：ピッチ予測器（抑揚の幅 k を明示的に操作）</h1>",
             "<p class=m>ピッチ +8 相当。Normal は予測した F0 の線を中央値まわりに k 倍。本物の Normal の幅は 12.8 半音（v63 は約 5 半音）。"
             "スタイルは v63 プリセット（z 0.5）＋本物のスタイル差。</p>",
             "<pre class=m>" + html.escape("\n".join(lines)) + "</pre>"]
    for ti, text in enumerate(TEXTS, 1):
        parts.append(f"<h2>{html.escape(text)}</h2><table><tr>" + "".join(f"<th>Normal k={k:g}</th>" for k in ks)
                     + "<th>Ama</th><th>Sexy</th><th>Tsun</th></tr><tr>"
                     + "".join(f"<td><audio controls src='fi{ti:02d}_normal_k{k:g}.wav'></audio></td>" for k in ks)
                     + "".join(f"<td><audio controls src='fi{ti:02d}_{st}.wav'></audio></td>" for st in ("ama", "sexy", "tsun"))
                     + "</tr></table>")
    (out / "index.html").write_text("\n".join(parts), encoding="utf-8")
    print(f"[PASS] {out / 'index.html'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["branch", "zdelta", "eval"])
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=16, help="branch: training batch")
    ap.add_argument("--workers", type=int, default=4, help="branch: data loader workers")
    ap.add_argument("--style-k", type=float, default=1.5, help="Normal range scale the styles start from")
    ap.add_argument("--base-name", default="base")
    ap.add_argument("--branch-name", default="branch")
    ap.add_argument("--only-style", default="", help="train a single-style branch (ama / sexy / tsun)")
    args = ap.parse_args()
    global BASE_NAME, BRANCH_NAME
    BASE_NAME, BRANCH_NAME = args.base_name, args.branch_name
    {"branch": cmd_branch, "zdelta": cmd_zdelta, "eval": cmd_eval}[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
