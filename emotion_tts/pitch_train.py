from __future__ import annotations

"""v62 base: continue v60 joint training with the explicit pitch condition.

Original Zundanen code (MIT).  Starts from the newest v60 checkpoint
(read-only); losses / MAS / KL are the Finnish VITS functions, as in v60.
Data: Finnish Chatterbox + RVC bridge at +8/+12/+16 (index 0.1, protect 0.5)
+ Japanese train split, all with measured pitch level / range.

Each evaluation renders Finnish x Zundamon at pitch levels -2/0/+2 st and
logs the measured F0 shift, so the pitch knob is verified while training.
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
sys.path.insert(0, str(HERE))
for p in (ROOT / "emotion_tts", ROOT / "emotion_tts"):
    sys.path.insert(0, str(p))
from emo_common import LANGS, SPEAKERS, STYLES, read_csv  # noqa: E402
from emo_vits import import_research  # noqa: E402
from emo_train_base import V60Dataset, collate, predicted_path, row_weights, write_wav  # noqa: E402
from pitch_model import SCHEMA, build_v62, pitch_vector  # noqa: E402

import_research(ROOT)
from vits_lite import rand_slice_segments, slice_wave_segments  # noqa: E402
from mas_alignment import mas_attention  # noqa: E402
from train_vits import expand_prior, mel_fn, vits_kl  # noqa: E402

SR = 24000


class V62Dataset(V60Dataset):
    def __init__(self, rows, vocab, range_center, **kw):
        super().__init__(rows, vocab, **kw)
        self.range_center = range_center

    def pitch(self, r):
        if r.get("pitch_ok") != "1":
            return [0.0, 0.0]
        return pitch_vector(float(r["pitch_level_st"]), float(r["pitch_range_st"]), self.range_center)

    def __getitem__(self, i):
        d = super().__getitem__(i)
        d["pitch"] = torch.tensor(self.pitch(self.rows[i]), dtype=torch.float32)
        return d


def collate62(batch):
    out = collate(batch)
    out["pitch"] = torch.stack([s["pitch"] for s in batch])
    return out


@torch.no_grad()
def synthesize(model, tok, bert, has_bert, *, speaker, style, lang, pitch, strength=1.0, noise=0.4,
               length_scale=1.0, seed=1234, device="cuda"):
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
    g = torch.Generator(device=device).manual_seed(int(seed))
    z = m + torch.randn(m.shape, generator=g, device=device, dtype=m.dtype) * torch.exp(logs) * noise
    return model.generator(model.flow(z * ym, ym, reverse=True) * ym)[0, 0].float().cpu().numpy()


def f0_median(audio):
    import librosa
    y = librosa.resample(audio.astype(np.float32), orig_sr=SR, target_sr=16000)
    f, _v, _p = librosa.pyin(y, fmin=120, fmax=900, sr=16000, frame_length=1024)
    return float(np.nanmedian(f)) if np.any(~np.isnan(f)) else float("nan")


@torch.no_grad()
def save_eval(model, ds_fi, ds_ja, out: Path, step: int, presets, range_center):
    model.eval()
    d = out / f"step_{step:06d}"
    z = presets["zundamon"]["normal"]
    shifts = {}
    for i, r in enumerate(ds_fi.rows):
        tok, bert, _ = ds_fi.text_inputs(r)
        for dl in (-2.0, 0.0, 2.0):
            y = synthesize(model, tok, bert, 1.0, speaker="zundamon", style="normal", lang="fi",
                           pitch=pitch_vector(z["level_st"] + dl, z["range_st"], range_center))
            write_wav(d / "fi" / f"{i+1:02d}_zundamon_normal_level{dl:+g}.wav", y)
            shifts.setdefault(dl, []).append(f0_median(y))
        cp = presets["fi_chatterbox"]["normal"]
        write_wav(d / "fi" / f"{i+1:02d}_fi_chatterbox_normal.wav",
                  synthesize(model, tok, bert, 1.0, speaker="fi_chatterbox", style="normal", lang="fi",
                             pitch=pitch_vector(cp["level_st"], cp["range_st"], range_center)))
    for r in ds_ja.rows:
        tok, bert, _ = ds_ja.text_inputs(r)
        pr = presets[r["speaker"]][r["style"]]
        write_wav(d / "ja" / f"{r['stem']}_TEXT.wav",
                  synthesize(model, tok, bert, 0.0, speaker=r["speaker"], style=r["style"], lang="ja",
                             pitch=pitch_vector(pr["level_st"], pr["range_st"], range_center)))
    model.train()
    base = np.array(shifts[0.0])
    return {dl: float(np.nanmean(12 * np.log2(np.array(v) / base))) for dl, v in shifts.items()}, float(np.nanmedian(base))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v60-run", default="v60_main")
    ap.add_argument("--run", default="v62_main")
    ap.add_argument("--version-dir", default="v62", help="outputs/<version-dir>/<run>")
    ap.add_argument("--init-checkpoint", default="", help="default: newest v60 checkpoint")
    ap.add_argument("--resume", default="")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--warmup-steps", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--mix", default="fi=0.35,bridge=0.25,ja=0.40")
    ap.add_argument("--bert-dropout", type=float, default=0.05)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.steps, args.batch_size, args.save_every, args.workers, args.warmup_steps = 20, 2, 20, 0, 10

    run = ROOT / "outputs" / args.version_dir / args.run
    data, work = run / "data", run / ("base_smoke" if args.smoke else "base")
    random.seed(20261020); np.random.seed(20261020); torch.manual_seed(20261020)
    stats = json.loads((data / "pitch_stats.json").read_text(encoding="utf-8"))
    presets, rc = stats["presets_from_real_recordings"], stats["range_center_st"]
    vocab = json.loads((data / "vocab.json").read_text(encoding="utf-8"))
    ja = read_csv(data / "ja_pitch.csv")
    bridge = [r for p in sorted(list(data.glob("bridge_p*_pitch.csv")) + list(data.glob("wide_*_pitch.csv")))
              for r in read_csv(p)]
    if len(bridge) < 3000 and not args.smoke:
        raise SystemExit(f"[FAIL] bridge rows={len(bridge)}; finish v62_bridge + v62_prepare_pitch first")
    train_rows = read_csv(data / "fi_train_pitch.csv") + bridge + [r for r in ja if r["split"] == "train"]
    mix = {k.strip(): float(v) for k, v in (x.split("=") for x in args.mix.split(","))}
    weights, mix_info = row_weights(train_rows, mix)
    print(f"[DATA] {mix_info}")
    ds = V62Dataset(train_rows, vocab, rc, bert_dropout=args.bert_dropout)
    ds_fi = V62Dataset(read_csv(data / "fi_eval_pitch.csv")[:3], vocab, rc)
    ds_ja = V62Dataset([r for r in ja if r["split"] == "test" and r["speaker"] == "zundamon" and r["sentence_id"] in {"7", "60"}], vocab, rc)
    sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=10**8, replacement=True)
    kw = dict(batch_size=args.batch_size, sampler=sampler, num_workers=args.workers, collate_fn=collate62,
              pin_memory=True, drop_last=True)
    if args.workers > 0:
        kw.update(persistent_workers=True, prefetch_factor=2)
    loader = iter(torch.utils.data.DataLoader(ds, **kw))

    model, models, losses, vendor = build_v62(ROOT, len(vocab), len(SPEAKERS), len(STYLES), len(LANGS), "cpu")
    disc = models.MultiPeriodDiscriminator(use_spectral_norm=False)
    start, rck = 0, None
    if args.resume:
        rck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(rck["model"], strict=True)
        disc.load_state_dict(rck["discriminator"])
        start = int(rck["step"])
        init = args.resume
    else:
        init = args.init_checkpoint or str(sorted((ROOT / "outputs/v60" / args.v60_run / "train/checkpoints").glob("v60_*.pt"))[-1])
        ck = torch.load(init, map_location="cpu", weights_only=False)
        if ck.get("schema") == SCHEMA:          # continue from a v62-family checkpoint
            model.load_state_dict(ck["model"], strict=True)
            missing, unexpected = [], []
        else:
            missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        if unexpected or (ck.get("schema") != SCHEMA and set(missing) != {"text.pitch_proj.weight", "text.pitch_proj.bias"}):
            raise SystemExit(f"[FAIL] unexpected v60->v62 key mismatch: missing={missing} unexpected={unexpected}")
        disc.load_state_dict(ck["discriminator"])
        del ck
    print(f"[INIT] {init}")
    model, disc = model.cuda(), disc.cuda()
    opt_g = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.8, 0.99), eps=1e-9)
    opt_d = torch.optim.AdamW(disc.parameters(), lr=args.lr, betas=(0.8, 0.99), eps=1e-9)
    if rck is not None:
        opt_g.load_state_dict(rck["opt_g"]); opt_d.load_state_dict(rck["opt_d"])
        del rck
    scaler_g = torch.amp.GradScaler("cuda")
    scaler_d = torch.amp.GradScaler("cuda")
    work.mkdir(parents=True, exist_ok=True)
    (work / "checkpoints").mkdir(exist_ok=True)
    config = {"schema": SCHEMA, "args": vars(args), "init": init, "mix_info": mix_info, "pitch_stats": stats}
    (work / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(vendor / "LICENSE", work / "VITS-MIT-LICENSE.txt")
    log = work / "train.csv"
    if start == 0 or not log.exists():
        log.write_text("step,phase,total,mel,fm,adv,kl,dur,D,vram_gb\n", encoding="utf-8")
    mel = mel_fn(torch.device("cuda"))
    frozen = None
    t0 = time.time()
    model.train(); disc.train()
    for step in range(start + 1, args.steps + 1):
        warm = step <= args.warmup_steps
        if frozen != warm:
            for p in list(model.posterior.parameters()) + list(model.generator.parameters()):
                p.requires_grad_(not warm)
            frozen = warm
            print(f"[PHASE] {'warmup: text/duration/flow' if warm else 'full joint'}", flush=True)
        b = next(loader)
        tok, bert, tl = b["tokens"].cuda(), b["bert"].cuda(), b["token_lengths"].cuda()
        wav, spec, sl = b["wav"].cuda(), b["spec"].cuda(), b["spec_lengths"].cuda()
        cond = dict(speaker_id=b["speaker"].cuda(), style_id=b["style"].cuda(), lang_id=b["lang"].cuda(),
                    has_bert=b["has_bert"].cuda(), pitch=b["pitch"].cuda())
        with torch.amp.autocast("cuda", dtype=torch.float16):
            h, mp, lp, xm = model.text(tok, bert, tl, **cond)
            if warm:
                with torch.no_grad():
                    z, mq, lq, ym = model.posterior(spec, sl)
            else:
                z, mq, lq, ym = model.posterior(spec, sl)
            zp = model.flow(z, ym, reverse=False)
        attn = mas_attention(zp, mp, lp, xm, ym, tl, sl)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            mpe, lpe = expand_prior(attn, mp, lp)
            logwt = torch.log(attn.sum(dim=2).unsqueeze(1) + 1e-6) * xm
            dur_loss = torch.sum((model.duration(h, xm) - logwt) ** 2 * xm) / torch.clamp_min(xm.sum(), 1.0)
        loss_kl = vits_kl(zp, lq, mpe, lpe, ym)
        if warm:
            loss_mel = loss_fm = loss_adv = loss_d = torch.zeros((), device="cuda")
            total = loss_kl + dur_loss.float()
        else:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                zseg, starts = rand_slice_segments(z * ym, sl, model.segment_frames)
                yhat = model.generator(zseg)
                yreal = slice_wave_segments(wav, starts, model.segment_samples, model.hop_length)
            opt_d.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                dr, dg, _, _ = disc(yreal, yhat.detach())
                loss_d, _, _ = losses.discriminator_loss(dr, dg)
            scaler_d.scale(loss_d).backward()
            scaler_d.unscale_(opt_d)
            torch.nn.utils.clip_grad_norm_(disc.parameters(), 1000.0)
            scaler_d.step(opt_d); scaler_d.update()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                dr, dg, fr, fg = disc(yreal, yhat)
                loss_fm = losses.feature_loss(fr, fg)
                loss_adv, _ = losses.generator_loss(dg)
                mr = torch.log(torch.clamp(mel(yreal.squeeze(1)), min=1e-5))
                mf = torch.log(torch.clamp(mel(yhat.squeeze(1)), min=1e-5))
                tm = min(mr.size(-1), mf.size(-1))
                loss_mel = F.l1_loss(mf[..., :tm], mr[..., :tm])
            total = loss_adv.float() + loss_fm.float() + 45.0 * loss_mel.float() + loss_kl + dur_loss.float()
        if not torch.isfinite(total):
            raise RuntimeError(f"non-finite loss at step {step}: {b['stems']}")
        opt_g.zero_grad(set_to_none=True)
        scaler_g.scale(total).backward()
        scaler_g.unscale_(opt_g)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1000.0)
        scaler_g.step(opt_g); scaler_g.update()
        if step == start + 1 or step % 20 == 0:
            vals = [float(x.detach().float()) for x in (total, loss_mel, loss_fm, loss_adv, loss_kl, dur_loss, loss_d)]
            vram = torch.cuda.max_memory_reserved() / 1024**3
            print(f"step {step:6d}/{args.steps} {'W' if warm else 'J'} total={vals[0]:.3f} mel={vals[1]:.3f} "
                  f"KL={vals[4]:.3f} dur={vals[5]:.3f} D={vals[6]:.3f} VRAM={vram:.1f}GB {(time.time()-t0)/60:.1f}min", flush=True)
            with log.open("a", encoding="utf-8") as f:
                f.write(f"{step},{'warm' if warm else 'joint'}," + ",".join(f"{v:.6f}" for v in vals) + f",{vram:.3f}\n")
        if step % args.save_every == 0 or step == args.steps:
            cp = work / "checkpoints" / f"v62_{step:06d}.pt"
            torch.save({"schema": SCHEMA, "step": step, "model": model.state_dict(), "discriminator": disc.state_dict(),
                        "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(), "vocab": vocab,
                        "speakers": list(SPEAKERS), "styles": list(STYLES), "langs": list(LANGS),
                        "pitch_stats": stats, "config": config}, cp)
            knob, f0 = save_eval(model, ds_fi, ds_ja, work / "samples", step, presets, rc)
            print(f"[SAVE] {cp}", flush=True)
            print(f"[PITCH-KNOB] Finnish x zundamon normal: preset F0={f0:.0f}Hz "
                  f"(real Zundamon Normal ref {stats['reference_hz']['zundamon']:.0f}Hz); measured shift for "
                  + "  ".join(f"{k:+g}st->{v:+.2f}st" for k, v in sorted(knob.items())), flush=True)
    print(f"[PASS] v62 base finished in {(time.time()-t0)/60:.1f} min -> {work}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
