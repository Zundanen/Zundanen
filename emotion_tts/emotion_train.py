from __future__ import annotations

"""v65 base: continue the v63 base with frame-level F0 conditioning + F0 predictor.

Original Zundanen code (MIT).  Losses as v62 (mel x45 + GAN + FM + KL + duration)
plus F0 predictor loss (L1 semitones on voiced frames x0.1 + voicing BCE x0.5).
During training the prior is conditioned on the ground-truth F0 (RMVPE frames).

Each evaluation renders Finnish x Zundamon with the predicted contour at the RVC
+8 level and range scale k = 1 / 2, and logs the measured F0 range, so the new
range control is verified during training ([RANGE-K] lines).
"""

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (HERE, ROOT / "emotion_tts", ROOT / "emotion_tts", ROOT / "emotion_tts"):
    sys.path.insert(0, str(p))
from emo_common import LANGS, SPEAKERS, STYLES, read_csv  # noqa: E402
from emo_vits import import_research  # noqa: E402
from emo_train_base import predicted_path, row_weights, write_wav  # noqa: E402
from pitch_model import SCHEMA as V62_SCHEMA, pitch_vector  # noqa: E402
from pitch_train import V62Dataset, collate62  # noqa: E402
from emotion_model import F0_DIR, SCHEMA, build_v65, f0_features_for_frames, f0_key, flow_apply, shape_contour  # noqa: E402

import_research(ROOT)
from vits_lite import rand_slice_segments, slice_wave_segments  # noqa: E402
from mas_alignment import mas_attention  # noqa: E402
from train_vits import expand_prior, mel_fn, vits_kl  # noqa: E402

SR = 24000
DATA = ROOT / "outputs/v63/v63_main/data"
if __import__("os").environ.get("ZUNDANEN_V65_DATA"):   # v71: another prepared data folder.  Unset = unchanged.
    DATA = Path(__import__("os").environ["ZUNDANEN_V65_DATA"])


class V65Dataset(V62Dataset):
    def __init__(self, rows, vocab, range_center, ref_hz, **kw):
        super().__init__(rows, vocab, range_center, **kw)
        self.ref_hz = ref_hz

    def __getitem__(self, i):
        d = super().__getitem__(i)
        r = self.rows[i]
        p = F0_DIR / (f0_key(r["wav_path"]) + ".npy")
        n = d["spec"].shape[-1]
        if p.exists():
            d["f0feat"] = torch.from_numpy(f0_features_for_frames(np.load(p), n, self.ref_hz[r["speaker"]]))
            d["f0_ok"] = 1.0
        else:
            d["f0feat"] = torch.zeros(2, n)
            d["f0_ok"] = 0.0
        return d


def collate65(batch):
    out = collate62(batch)
    ty = out["spec"].shape[-1]
    f = torch.zeros(len(batch), 2, ty)
    for i, s in enumerate(batch):
        f[i, :, : s["f0feat"].shape[-1]] = s["f0feat"]
    out["f0feat"] = f
    out["f0_ok"] = torch.tensor([s["f0_ok"] for s in batch])
    return out


def expand_frames(attn, h):
    return torch.matmul(attn.transpose(1, 2), h.transpose(1, 2)).transpose(1, 2)


@torch.no_grad()
def synth65(model, tok, bert, has_bert, *, speaker, style, lang, pitch, level_st=None, range_k=1.0,
            strength=1.0, noise=0.4, length_scale=1.0, seed=1234, device="cuda", return_contour=False):
    tok = tok.unsqueeze(0).to(device)
    br = bert.unsqueeze(0).to(device)
    tl = torch.tensor([tok.size(1)], device=device)
    mix = torch.zeros(1, len(STYLES), device=device)
    mix[0, STYLES.index("normal")] += 1.0 - strength
    mix[0, STYLES.index(style)] += strength
    h, mp, lp, xm = model.text(tok, br, tl, speaker_id=torch.tensor([SPEAKERS.index(speaker)], device=device),
                               style_id=torch.tensor([STYLES.index(style)], device=device),
                               lang_id=torch.tensor([LANGS.index(lang)], device=device),
                               has_bert=torch.tensor([float(has_bert)], device=device), style_mix=mix,
                               pitch=torch.tensor([pitch], dtype=torch.float32, device=device))
    attn, yl = predicted_path(model, h, xm, tl, length_scale)
    ym = (torch.arange(attn.size(-1), device=device).unsqueeze(0) < yl.unsqueeze(1)).unsqueeze(1).to(mp.dtype)
    m, logs = expand_prior(attn, mp, lp)
    st_pred, uv = model.f0.predictor(expand_frames(attn, h), ym)
    feat = shape_contour(st_pred, uv, ym, level_st=level_st, range_k=range_k)
    m = m + model.f0.f0_proj(feat) * ym
    g = torch.Generator(device=device).manual_seed(int(seed))
    z = m + torch.randn(m.shape, generator=g, device=device, dtype=m.dtype) * torch.exp(logs) * noise
    y = model.generator(flow_apply(model, z * ym, ym, feat, reverse=True) * ym)[0, 0].float().cpu().numpy()
    return (y, feat) if return_contour else y


def f0_spread(audio):
    import librosa
    y = librosa.resample(audio.astype(np.float32), orig_sr=SR, target_sr=16000)
    f, _v, _p = librosa.pyin(y, fmin=100, fmax=900, sr=16000, frame_length=1024)
    f = f[~np.isnan(f)]
    if f.size < 10:
        return float("nan"), float("nan")
    # pYIN octave jumps inflate the range; drop frames > 9 st from the median.
    f = f[np.abs(12 * np.log2(f / np.median(f))) <= 9]
    p10, p50, p90 = np.percentile(f, [10, 50, 90])
    return float(p50), float(12 * np.log2(p90 / p10))


@torch.no_grad()
def save_eval(model, ds_fi, out: Path, step: int, stats, level8, target="zundamon"):
    model.eval()
    d = out / f"step_{step:06d}"
    rc = stats["range_center_st"]
    zp = stats["presets_from_real_recordings"][target]["normal"]
    res = {}
    for i, r in enumerate(ds_fi.rows):
        tok, bert, _ = ds_fi.text_inputs(r)
        for k in (1.0, 2.0):
            y = synth65(model, tok, bert, 1.0, speaker=target, style="normal", lang="fi",
                        pitch=pitch_vector(level8, zp["range_st"] / 2, rc), level_st=level8, range_k=k)
            write_wav(d / f"{i+1:02d}_{target}_k{k:g}.wav", y)
            res.setdefault(k, []).append(f0_spread(y))
    model.train()
    return {k: (float(np.nanmean([x[0] for x in v])), float(np.nanmean([x[1] for x in v]))) for k, v in res.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-checkpoint", default="outputs/v63/v63_main/base/checkpoints/v62_010000.pt")
    ap.add_argument("--from-scratch", action="store_true", help="v2: random initial weights (no init checkpoint)")
    ap.add_argument("--target-speaker", default="zundamon", help="v2: character used for the sample renders")
    ap.add_argument("--resume", default="")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--mix", default="fi=0.35,bridge=0.25,ja=0.40")
    ap.add_argument("--flow-cond", action="store_true", help="v65b: F0 conditions the flow couplings")
    ap.add_argument("--out-name", default="base")
    ap.add_argument("--extra-csv", action="append", default=[], help="v2: extra manifest(s) (mixed as group 'extra')")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.steps, args.batch_size, args.save_every, args.workers = 20, 2, 20, 0
    work = ROOT / "outputs/v65" / ("base_smoke" if args.smoke else args.out_name)
    random.seed(20261022); np.random.seed(20261022); torch.manual_seed(20261022)
    stats = json.loads((DATA / "pitch_stats.json").read_text(encoding="utf-8"))
    rc, ref = stats["range_center_st"], stats["reference_hz"]
    # sample renders at the character's lowest bridge level (v63 data: bridge_p8:zundamon = the RVC +8 level)
    bridge_levels = [g["level_median"] for k, g in stats["groups"].items()
                     if k.startswith("bridge_") and k.endswith(":" + args.target_speaker)]
    level8 = min(bridge_levels) if bridge_levels else stats["presets_from_real_recordings"][args.target_speaker]["normal"]["level_st"]
    vocab = json.loads((DATA / "vocab.json").read_text(encoding="utf-8"))
    ja = read_csv(DATA / "ja_pitch.csv")
    bridge = [r for p in sorted(list(DATA.glob("bridge_p*_pitch.csv")) + list(DATA.glob("wide_*_pitch.csv"))) for r in read_csv(p)]
    rows = read_csv(DATA / "fi_train_pitch.csv") + bridge + [r for r in ja if r["split"] == "train"]
    for extra in args.extra_csv:
        ex = read_csv(Path(extra))
        for r in ex:
            r["group"] = "extra"
        rows += ex
        print(f"[DATA] extra rows from {extra}: {len(ex)}")
    have = sum(1 for r in rows if (F0_DIR / (f0_key(r["wav_path"]) + ".npy")).exists())
    print(f"[F0] frame F0 available for {have}/{len(rows)} training rows")
    if have < 0.95 * len(rows) and not args.smoke:
        raise SystemExit("[FAIL] frame F0 extraction not finished (f0_frames.py)")
    weights, mix_info = row_weights(rows, {k.strip(): float(v) for k, v in (x.split("=") for x in args.mix.split(","))})
    print(f"[DATA] {mix_info}")
    ds = V65Dataset(rows, vocab, rc, ref, bert_dropout=0.05)
    ds_fi = V65Dataset(read_csv(DATA / "fi_eval_pitch.csv")[:3], vocab, rc, ref)
    sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=10**8, replacement=True)
    kw = dict(batch_size=args.batch_size, sampler=sampler, num_workers=args.workers, collate_fn=collate65,
              pin_memory=True, drop_last=True)
    if args.workers:
        kw.update(persistent_workers=True, prefetch_factor=2)
    loader = iter(torch.utils.data.DataLoader(ds, **kw))

    model, models, losses, vendor = build_v65(ROOT, len(vocab), len(SPEAKERS), len(STYLES), len(LANGS), "cpu",
                                              flow_cond=args.flow_cond)
    disc = models.MultiPeriodDiscriminator(use_spectral_norm=False)
    start, rck = 0, None
    if args.resume:
        rck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(rck["model"], strict=True); disc.load_state_dict(rck["discriminator"])
        start = int(rck["step"])
    elif args.from_scratch:
        print("[INIT] random initial weights (from scratch)")
    else:
        ck = torch.load(ROOT / args.init_checkpoint, map_location="cpu", weights_only=False)
        if ck.get("schema") not in (V62_SCHEMA, SCHEMA):
            raise SystemExit("[FAIL] init must be a v62-family (v63) or v65 base")
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        if ck.get("schema") == SCHEMA:   # v2: continue an existing v65 base (all keys must match)
            missing = [k for k in missing if not (k.startswith("f0.flow_cond.") and not args.flow_cond)]
        if unexpected or any(not k.startswith("f0.") for k in missing):
            raise SystemExit(f"[FAIL] key mismatch missing={missing[:5]} unexpected={unexpected[:5]}")
        disc.load_state_dict(ck["discriminator"])
        print(f"[INIT] {args.init_checkpoint} (new: {len(missing)} F0 tensors)")
    model, disc = model.cuda(), disc.cuda()
    opt_g = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.8, 0.99), eps=1e-9)
    opt_d = torch.optim.AdamW(disc.parameters(), lr=args.lr, betas=(0.8, 0.99), eps=1e-9)
    if rck is not None:
        opt_g.load_state_dict(rck["opt_g"]); opt_d.load_state_dict(rck["opt_d"])
    sg, sd = torch.amp.GradScaler("cuda"), torch.amp.GradScaler("cuda")
    (work / "checkpoints").mkdir(parents=True, exist_ok=True)
    config = {"schema": SCHEMA, "args": vars(args), "mix_info": mix_info}
    (work / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    shutil.copy2(vendor / "LICENSE", work / "VITS-MIT-LICENSE.txt")
    log = work / "train.csv"
    if start == 0:
        log.write_text("step,total,mel,kl,dur,f0_l1_st,uv_bce,D\n", encoding="utf-8")
    mel = mel_fn(torch.device("cuda"))
    t0 = time.time()
    model.train(); disc.train()
    for step in range(start + 1, args.steps + 1):
        b = next(loader)
        tok, bert, tl = b["tokens"].cuda(), b["bert"].cuda(), b["token_lengths"].cuda()
        wav, spec, sl = b["wav"].cuda(), b["spec"].cuda(), b["spec_lengths"].cuda()
        f0f, f0ok = b["f0feat"].cuda(), b["f0_ok"].cuda()
        cond = dict(speaker_id=b["speaker"].cuda(), style_id=b["style"].cuda(), lang_id=b["lang"].cuda(),
                    has_bert=b["has_bert"].cuda(), pitch=b["pitch"].cuda())
        with torch.amp.autocast("cuda", dtype=torch.float16):
            h, mp, lp, xm = model.text(tok, bert, tl, **cond)
            z, mq, lq, ym = model.posterior(spec, sl)
        okm = ym * f0ok.view(-1, 1, 1)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            zp = flow_apply(model, z, ym, f0f * okm, reverse=False)
        attn = mas_attention(zp, mp, lp, xm, ym, tl, sl)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            mpe, lpe = expand_prior(attn, mp, lp)
            mpe = mpe + model.f0.f0_proj(f0f * okm) * ym
            logwt = torch.log(attn.sum(dim=2).unsqueeze(1) + 1e-6) * xm
            dur_loss = torch.sum((model.duration(h, xm) - logwt) ** 2 * xm) / torch.clamp_min(xm.sum(), 1.0)
            st_pred, uv_logit = model.f0.predictor(expand_frames(attn, h.detach()), ym)
        vm = okm * f0f[:, 1:2]
        f0_l1 = torch.sum(torch.abs(st_pred.float() - f0f[:, :1]) * 12.0 * vm) / torch.clamp_min(vm.sum(), 1.0)
        uv_bce = torch.sum(F.binary_cross_entropy_with_logits(uv_logit.float(), f0f[:, 1:2], reduction="none") * okm) / torch.clamp_min(okm.sum(), 1.0)
        loss_kl = vits_kl(zp, lq, mpe, lpe, ym)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            zseg, starts = rand_slice_segments(z * ym, sl, model.segment_frames)
            yhat = model.generator(zseg)
            yreal = slice_wave_segments(wav, starts, model.segment_samples, model.hop_length)
        opt_d.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            dr, dg, _, _ = disc(yreal, yhat.detach())
            loss_d, _, _ = losses.discriminator_loss(dr, dg)
        sd.scale(loss_d).backward(); sd.unscale_(opt_d)
        torch.nn.utils.clip_grad_norm_(disc.parameters(), 1000.0); sd.step(opt_d); sd.update()
        with torch.amp.autocast("cuda", dtype=torch.float16):
            dr, dg, fr, fg = disc(yreal, yhat)
            loss_fm = losses.feature_loss(fr, fg)
            loss_adv, _ = losses.generator_loss(dg)
            mr = torch.log(torch.clamp(mel(yreal.squeeze(1)), min=1e-5))
            mf = torch.log(torch.clamp(mel(yhat.squeeze(1)), min=1e-5))
            tm = min(mr.size(-1), mf.size(-1))
            loss_mel = F.l1_loss(mf[..., :tm], mr[..., :tm])
        total = (loss_adv.float() + loss_fm.float() + 45.0 * loss_mel.float() + loss_kl + dur_loss.float()
                 + 0.1 * f0_l1 + 0.5 * uv_bce)
        if not torch.isfinite(total):
            raise RuntimeError(f"non-finite loss at step {step}: {b['stems']}")
        opt_g.zero_grad(set_to_none=True)
        sg.scale(total).backward(); sg.unscale_(opt_g)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1000.0); sg.step(opt_g); sg.update()
        if step == start + 1 or step % 20 == 0:
            vals = [float(x.detach().float()) for x in (total, loss_mel, loss_kl, dur_loss, f0_l1, uv_bce, loss_d)]
            print(f"step {step:6d}/{args.steps} total={vals[0]:.3f} mel={vals[1]:.3f} KL={vals[2]:.3f} dur={vals[3]:.3f} "
                  f"f0L1={vals[4]:.2f}st uv={vals[5]:.3f} D={vals[6]:.3f} {(time.time()-t0)/60:.1f}min", flush=True)
            with log.open("a", encoding="utf-8") as f:
                f.write(f"{step}," + ",".join(f"{v:.6f}" for v in vals) + "\n")
        if step % args.save_every == 0 or step == args.steps:
            cp = work / "checkpoints" / f"v65_{step:06d}.pt"
            torch.save({"schema": SCHEMA, "step": step, "model": model.state_dict(), "discriminator": disc.state_dict(),
                        "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(), "vocab": vocab,
                        "speakers": list(SPEAKERS), "styles": list(STYLES), "langs": list(LANGS),
                        "pitch_stats": stats, "config": config}, cp)
            rk = save_eval(model, ds_fi, work / "samples", step, stats, level8, args.target_speaker)
            print(f"[SAVE] {cp}", flush=True)
            print(f"[RANGE-K] Finnish x {args.target_speaker} at its bridge level: " + "  ".join(
                f"k={k:g}: median {v[0]:.0f}Hz range {v[1]:.1f}st" for k, v in rk.items()), flush=True)
    print(f"[PASS] v65 base finished in {(time.time()-t0)/60:.1f} min -> {work}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
