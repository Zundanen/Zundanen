from __future__ import annotations

"""v60 joint fine-tune: Finnish Normal + Japanese multi-style, one VITS.

Original Zundanen code (MIT).  Loss / MAS / KL are the Finnish VITS functions,
imported unchanged.  Starts from the finished Finnish VITS checkpoint (read-only)
and writes only to outputs/v60/<run>/train.

Factorisation that makes Japanese-only emotion usable for Finnish:
  * style is shared by 3 Japanese speakers -> style cannot be "a voice"
  * the bridge speaker (Finnish converted to Zundamon) speaks both languages
    -> language cannot be "a voice"
  * Japanese phones use the same Finnish letter rows -> style acts on the
    token rows that Finnish text uses
  * Finnish replay every batch -> Finnish pronunciation is not forgotten
"""

import argparse
import json
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from emo_common import LANGS, NEW_TOKENS, SPEAKERS, STYLES, read_csv  # noqa: E402
from emo_vits import SCHEMA, STEP7_SCHEMA, build_model, import_research, init_from_step7  # noqa: E402

import_research(ROOT)
from vits_lite import rand_slice_segments, slice_wave_segments  # noqa: E402
from mas_alignment import mas_attention  # noqa: E402
from train_vits import expand_prior, mel_fn, vits_kl  # noqa: E402

SR, N_FFT, HOP, WIN = 24000, 1024, 256, 1024


class V60Dataset(torch.utils.data.Dataset):
    def __init__(self, rows, vocab, *, bert_dropout: float = 0.0):
        self.rows = rows
        self.vocab = vocab
        self.bert_dropout = float(bert_dropout)
        self.window = torch.hann_window(WIN)

    def __len__(self):
        return len(self.rows)

    def _wav(self, path):
        import soundfile as sf
        import torchaudio
        arr, sr = sf.read(str(path), dtype="float32", always_2d=True)
        wav = torch.from_numpy(arr.mean(axis=1).copy()).unsqueeze(0)
        if sr != SR:
            wav = torchaudio.functional.resample(wav, sr, SR)
        return torch.clamp(wav, -1.0, 1.0)

    def text_inputs(self, r):
        if r["lang"] == "fi":
            with np.load(r["bert_cache"], allow_pickle=False) as d:
                gs = [str(x) for x in d["grapheme_tokens"].tolist()]
                bert = torch.from_numpy(np.asarray(d["features"], dtype=np.float32))
            has_bert = 1.0
            if self.bert_dropout > 0 and random.random() < self.bert_dropout:
                has_bert = 0.0
        else:
            gs = r["tokens"].split(" ")
            bert = torch.zeros(len(gs), 1024)
            has_bert = 0.0
        ids = [self.vocab[g] for g in gs]
        if bert.shape[0] != len(ids):
            raise ValueError(f"{r['stem']}: bert/token mismatch {bert.shape[0]} vs {len(ids)}")
        return torch.tensor(ids, dtype=torch.long), bert, has_bert

    def __getitem__(self, i):
        r = self.rows[i]
        tokens, bert, has_bert = self.text_inputs(r)
        wav = self._wav(r["wav_path"])
        spec = torch.abs(torch.stft(wav.squeeze(0), n_fft=N_FFT, hop_length=HOP, win_length=WIN,
                                    window=self.window, center=False, return_complex=True))
        return {
            "stem": r["stem"], "tokens": tokens, "bert": bert, "has_bert": has_bert, "wav": wav, "spec": spec,
            "speaker": SPEAKERS.index(r["speaker"]), "style": STYLES.index(r["style"]),
            "lang": LANGS.index(r["lang"]),
        }


def collate(batch):
    b = len(batch)
    tx = max(s["tokens"].numel() for s in batch)
    ty = max(s["spec"].shape[-1] for s in batch)
    tw = max(s["wav"].shape[-1] for s in batch)
    out = {
        "tokens": torch.zeros(b, tx, dtype=torch.long), "bert": torch.zeros(b, tx, 1024),
        "wav": torch.zeros(b, 1, tw), "spec": torch.zeros(b, N_FFT // 2 + 1, ty),
        "token_lengths": torch.zeros(b, dtype=torch.long), "spec_lengths": torch.zeros(b, dtype=torch.long),
        "has_bert": torch.tensor([s["has_bert"] for s in batch]),
        "speaker": torch.tensor([s["speaker"] for s in batch]),
        "style": torch.tensor([s["style"] for s in batch]),
        "lang": torch.tensor([s["lang"] for s in batch]),
        "stems": [s["stem"] for s in batch],
    }
    for i, s in enumerate(batch):
        n, m, w = s["tokens"].numel(), s["spec"].shape[-1], s["wav"].shape[-1]
        out["tokens"][i, :n] = s["tokens"]
        out["bert"][i, :n] = s["bert"]
        out["wav"][i, 0, :w] = s["wav"][0]
        out["spec"][i, :, :m] = s["spec"]
        out["token_lengths"][i] = n
        out["spec_lengths"][i] = m
    return out


def row_weights(rows, mix: dict[str, float]) -> tuple[list[float], dict]:
    """Group mix (fi / bridge / ja) with Japanese balanced per speaker x style."""
    def group(r):
        if r.get("group"):          # v2: explicitly grouped rows (e.g. a user Finnish dataset = "extra")
            return r["group"]
        if r["lang"] == "ja":
            return "ja"
        return "fi" if r["speaker"] == "fi_chatterbox" else "bridge"
    groups = Counter(group(r) for r in rows)
    cells = Counter((r["speaker"], r["style"]) for r in rows if r["lang"] == "ja")
    active = {g: mix.get(g, 0.0) for g in groups if mix.get(g, 0.0) > 0}
    total = sum(active.values())
    w = []
    for r in rows:
        g = group(r)
        if g not in active:
            w.append(0.0)
        elif g == "ja":
            w.append(active[g] / total / len(cells) / cells[(r["speaker"], r["style"])])
        else:
            w.append(active[g] / total / groups[g])
    return w, {"groups": dict(groups), "ja_cells": len(cells), "mix": {g: v / total for g, v in active.items()}}


def predicted_path(model, h, x_mask, token_lengths, length_scale=1.0, max_frames=2500):
    logw = model.duration(h, x_mask)
    w = torch.clamp(torch.ceil(torch.exp(logw) * x_mask * float(length_scale)), min=1.0) * x_mask
    y_lengths = torch.clamp_min(torch.sum(w, dim=[1, 2]).long(), 1)
    if int(y_lengths.max()) > max_frames:
        w = torch.clamp(torch.floor(w * max_frames / float(y_lengths.max())), min=1.0) * x_mask
        y_lengths = torch.clamp_min(torch.sum(w, dim=[1, 2]).long(), 1)
    d = w[:, 0].long()
    attn = torch.zeros(d.size(0), d.size(1), int(y_lengths.max()), device=d.device, dtype=h.dtype)
    for b in range(d.size(0)):
        pos, yl = 0, int(y_lengths[b])
        for i in range(int(token_lengths[b])):
            end = min(yl, pos + max(1, int(d[b, i])))
            if end > pos:
                attn[b, i, pos:end] = 1.0
            pos = end
        if pos < yl:
            attn[b, int(token_lengths[b]) - 1, pos:yl] = 1.0
    return attn, y_lengths


@torch.no_grad()
def synthesize(model, tokens, bert, has_bert, *, speaker, style, lang, strength=1.0, noise=0.4,
               length_scale=1.0, seed=1234, device="cuda"):
    """Text-only synthesis.  strength mixes Normal -> style in style-embedding space."""
    tok = tokens.unsqueeze(0).to(device)
    br = bert.unsqueeze(0).to(device)
    tl = torch.tensor([tok.size(1)], device=device)
    mix = torch.zeros(1, len(STYLES), device=device)
    mix[0, STYLES.index("normal")] += 1.0 - strength
    mix[0, STYLES.index(style)] += strength
    h, mp, lp, xm = model.text(tok, br, tl, speaker_id=torch.tensor([SPEAKERS.index(speaker)], device=device),
                               style_id=torch.tensor([STYLES.index(style)], device=device),
                               lang_id=torch.tensor([LANGS.index(lang)], device=device),
                               has_bert=torch.tensor([float(has_bert)], device=device), style_mix=mix)
    attn, yl = predicted_path(model, h, xm, tl, length_scale)
    ym = (torch.arange(attn.size(-1), device=device).unsqueeze(0) < yl.unsqueeze(1)).unsqueeze(1).to(mp.dtype)
    m, logs = expand_prior(attn, mp, lp)
    g = torch.Generator(device=device).manual_seed(int(seed))
    z = m + torch.randn(m.shape, generator=g, device=device, dtype=m.dtype) * torch.exp(logs) * noise
    y = model.generator(model.flow(z * ym, ym, reverse=True) * ym)
    return y[0, 0].float().cpu().numpy()


def write_wav(path: Path, audio: np.ndarray):
    import soundfile as sf
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.999:
        audio = audio * (0.999 / peak)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.astype(np.float32), SR, subtype="PCM_16")


@torch.no_grad()
def save_eval(model, ds_eval_fi, ds_eval_ja, out: Path, step: int, speakers_for_fi):
    model.eval()
    d = out / f"step_{step:06d}"
    for i, r in enumerate(ds_eval_fi.rows[:3]):
        tok, bert, _hb = ds_eval_fi.text_inputs(r)
        for spk in speakers_for_fi:
            for style in STYLES:
                write_wav(d / "fi" / f"{i+1:02d}_{spk}_{style}.wav",
                          synthesize(model, tok, bert, 1.0, speaker=spk, style=style, lang="fi"))
    for r in ds_eval_ja.rows:
        tok, bert, _hb = ds_eval_ja.text_inputs(r)
        write_wav(d / "ja" / f"{r['stem']}_TEXT.wav",
                  synthesize(model, tok, bert, 0.0, speaker=r["speaker"], style=r["style"], lang="ja"))
        dst = d / "ja" / f"{r['stem']}_REAL.wav"
        if not dst.exists():
            shutil.copy2(r["wav_path"], dst)
    model.train()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--run", default="v60_main")
    ap.add_argument("--init-checkpoint", default="models/chatterbox.pt")
    ap.add_argument("--resume", default="")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--warmup-steps", type=int, default=1500,
                    help="text prior/duration/flow only; posterior+generator frozen, no GAN")
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--mix", default="fi=0.40,bridge=0.20,ja=0.40")
    ap.add_argument("--bert-dropout", type=float, default=0.05)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--work-name", default="train", help="output subfolder of the run")
    ap.add_argument("--smoke", action="store_true", help="20 steps, batch 2, save once: wiring test only")
    ap.set_defaults(amp=True)
    args = ap.parse_args()
    if args.smoke:
        args.steps, args.batch_size, args.save_every, args.workers = 20, 2, 20, 0
        args.warmup_steps = 10

    root = Path(args.root).resolve()
    run = root / "outputs" / "v60" / args.run
    data, work = run / "data", run / ("train_smoke" if args.smoke else args.work_name)
    if not torch.cuda.is_available():
        raise SystemExit("[FAIL] CUDA is required")
    random.seed(20261018); np.random.seed(20261018); torch.manual_seed(20261018)

    vocab = json.loads((data / "vocab.json").read_text(encoding="utf-8"))
    ja = read_csv(data / "ja.csv")
    fi_train = read_csv(data / "fi_train.csv")
    fi_eval = read_csv(data / "fi_eval.csv")
    bridges = sorted(data.glob("bridge_*.csv"))
    bridge_rows = [r for p in bridges if p.name != "bridge_src.csv" for r in read_csv(p)]
    train_rows = fi_train + bridge_rows + [r for r in ja if r["split"] == "train"]
    for r in train_rows:
        if r["speaker"] not in SPEAKERS or r["style"] not in STYLES or r["lang"] not in LANGS:
            raise SystemExit(f"[FAIL] unknown label in {r['stem']}")
    mix = {k.strip(): float(v) for k, v in (x.split("=") for x in args.mix.split(","))}
    weights, mix_info = row_weights(train_rows, mix)
    print(f"[DATA] {mix_info}")
    if not bridge_rows:
        print("[WARN] no bridge rows: speaker and language are fully confounded. Run -Bridge for better transfer.")
    elif len(bridge_rows) < 300 and mix.get("bridge", 0) > 0 and not args.smoke:
        # A handful of files at 20% of every batch would be memorised, not learned from.
        raise SystemExit(f"[FAIL] only {len(bridge_rows)} bridge rows. Finish -Bridge first, "
                         "or pass --mix with bridge=0.")

    ds = V60Dataset(train_rows, vocab, bert_dropout=args.bert_dropout)
    ds_fi_eval = V60Dataset(fi_eval, vocab)
    ja_test = [r for r in ja if r["split"] == "test" and r["speaker"] == "zundamon"
               and r["sentence_id"] in {"7", "60"}]
    ds_ja_eval = V60Dataset(ja_test, vocab)
    sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=10**8, replacement=True)
    kw = dict(batch_size=args.batch_size, sampler=sampler, num_workers=args.workers, collate_fn=collate,
              pin_memory=True, drop_last=True)
    if args.workers > 0:
        kw.update(persistent_workers=True, prefetch_factor=2)
    loader = iter(torch.utils.data.DataLoader(ds, **kw))

    model, models, losses, vendor = build_model(root, len(vocab), len(SPEAKERS), len(STYLES), len(LANGS), "cpu")
    disc = models.MultiPeriodDiscriminator(use_spectral_norm=False)
    start = 0
    resume_ck = None
    if args.resume:
        resume_ck = torch.load(Path(args.resume), map_location="cpu", weights_only=False)
        if resume_ck.get("schema") != SCHEMA:
            raise SystemExit("[FAIL] --resume needs a v60 checkpoint")
        model.load_state_dict(resume_ck["model"], strict=True)
        disc.load_state_dict(resume_ck["discriminator"])
        start = int(resume_ck["step"])
        init_report = {"resume": str(args.resume)}
    else:
        init_path = (root / args.init_checkpoint).resolve()
        ck7 = torch.load(init_path, map_location="cpu", weights_only=False)
        if ck7.get("schema") != STEP7_SCHEMA:
            raise SystemExit(f"[FAIL] init checkpoint is not Finnish VITS: {init_path}")
        init_report = init_from_step7(model, ck7["model"], ck7["vocab"], vocab, NEW_TOKENS)
        disc.load_state_dict(ck7["discriminator"])
        init_report["init_checkpoint"] = str(init_path)
        del ck7
    model, disc = model.cuda(), disc.cuda()
    print(f"[INIT] {init_report}")

    opt_g = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.8, 0.99), eps=1e-9)
    opt_d = torch.optim.AdamW(disc.parameters(), lr=args.lr, betas=(0.8, 0.99), eps=1e-9)
    if resume_ck is not None:
        opt_g.load_state_dict(resume_ck["opt_g"])
        opt_d.load_state_dict(resume_ck["opt_d"])
        del resume_ck
    scaler_g = torch.amp.GradScaler("cuda", enabled=args.amp)
    scaler_d = torch.amp.GradScaler("cuda", enabled=args.amp)

    work.mkdir(parents=True, exist_ok=True)
    (work / "checkpoints").mkdir(exist_ok=True)
    config = {"schema": SCHEMA, "args": vars(args), "mix_info": mix_info, "init": init_report,
              "speakers": SPEAKERS, "styles": STYLES, "langs": LANGS, "train_rows": len(train_rows),
              "bridge_files": [p.name for p in bridges], "vendor": str(vendor)}
    (work / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    shutil.copy2(vendor / "LICENSE", work / "VITS-MIT-LICENSE.txt")
    log = work / "train.csv"
    if start == 0 or not log.exists():
        log.write_text("step,phase,total,mel,fm,adv,kl,dur,D,ja_frac,vram_gb\n", encoding="utf-8")
    mel = mel_fn(torch.device("cuda"))
    fi_eval_speakers = ["fi_chatterbox"] + (["zundamon"] if bridge_rows else [])

    frozen = None

    def set_phase(warm: bool):
        nonlocal frozen
        if frozen == warm:
            return
        for p in list(model.posterior.parameters()) + list(model.generator.parameters()):
            p.requires_grad_(not warm)
        frozen = warm
        print(f"[PHASE] {'warmup: text/duration/flow only' if warm else 'full joint VITS'}", flush=True)

    t0 = time.time()
    model.train(); disc.train()
    for step in range(start + 1, args.steps + 1):
        warm = step <= args.warmup_steps
        set_phase(warm)
        b = next(loader)
        tok, bert, tl = b["tokens"].cuda(), b["bert"].cuda(), b["token_lengths"].cuda()
        wav, spec, sl = b["wav"].cuda(), b["spec"].cuda(), b["spec_lengths"].cuda()
        cond = dict(speaker_id=b["speaker"].cuda(), style_id=b["style"].cuda(), lang_id=b["lang"].cuda(),
                    has_bert=b["has_bert"].cuda())
        with torch.amp.autocast("cuda", enabled=args.amp, dtype=torch.float16):
            h, mp, lp, xm = model.text(tok, bert, tl, **cond)
            if warm:
                with torch.no_grad():
                    z, mq, lq, ym = model.posterior(spec, sl)
            else:
                z, mq, lq, ym = model.posterior(spec, sl)
            zp = model.flow(z, ym, reverse=False)
        attn = mas_attention(zp, mp, lp, xm, ym, tl, sl)
        with torch.amp.autocast("cuda", enabled=args.amp, dtype=torch.float16):
            mpe, lpe = expand_prior(attn, mp, lp)
            logwt = torch.log(attn.sum(dim=2).unsqueeze(1) + 1e-6) * xm
            dur_loss = torch.sum((model.duration(h, xm) - logwt) ** 2 * xm) / torch.clamp_min(xm.sum(), 1.0)
        loss_kl = vits_kl(zp, lq, mpe, lpe, ym)

        if warm:
            loss_mel = loss_fm = loss_adv = loss_d = torch.zeros((), device="cuda")
            total = loss_kl + dur_loss.float()
        else:
            with torch.amp.autocast("cuda", enabled=args.amp, dtype=torch.float16):
                zseg, starts = rand_slice_segments(z * ym, sl, model.segment_frames)
                yhat = model.generator(zseg)
                yreal = slice_wave_segments(wav, starts, model.segment_samples, model.hop_length)
            opt_d.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp, dtype=torch.float16):
                dr, dg, _, _ = disc(yreal, yhat.detach())
                loss_d, _, _ = losses.discriminator_loss(dr, dg)
            scaler_d.scale(loss_d).backward()
            scaler_d.unscale_(opt_d)
            torch.nn.utils.clip_grad_norm_(disc.parameters(), 1000.0)
            scaler_d.step(opt_d); scaler_d.update()
            with torch.amp.autocast("cuda", enabled=args.amp, dtype=torch.float16):
                dr, dg, fr, fg = disc(yreal, yhat)
                loss_fm = losses.feature_loss(fr, fg)
                loss_adv, _ = losses.generator_loss(dg)
                mr = torch.log(torch.clamp(mel(yreal.squeeze(1)), min=1e-5))
                mf = torch.log(torch.clamp(mel(yhat.squeeze(1)), min=1e-5))
                tm = min(mr.size(-1), mf.size(-1))
                loss_mel = F.l1_loss(mf[..., :tm], mr[..., :tm])
            total = loss_adv.float() + loss_fm.float() + 45.0 * loss_mel.float() + loss_kl + dur_loss.float()
        if not torch.isfinite(total):
            raise RuntimeError(f"non-finite loss at step {step}: {float(total)} stems={b['stems']}")
        opt_g.zero_grad(set_to_none=True)
        scaler_g.scale(total).backward()
        scaler_g.unscale_(opt_g)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1000.0)
        scaler_g.step(opt_g); scaler_g.update()

        if step == start + 1 or step % 20 == 0:
            vals = [float(x.detach().float()) for x in (total, loss_mel, loss_fm, loss_adv, loss_kl, dur_loss, loss_d)]
            jf = float((b["lang"] == 1).float().mean())
            vram = torch.cuda.max_memory_reserved() / 1024**3
            el = (time.time() - t0) / 60
            print(f"step {step:6d}/{args.steps} {'W' if warm else 'J'} total={vals[0]:.3f} mel={vals[1]:.3f} "
                  f"fm={vals[2]:.3f} adv={vals[3]:.3f} KL={vals[4]:.3f} dur={vals[5]:.3f} D={vals[6]:.3f} "
                  f"ja={jf:.2f} VRAM={vram:.1f}GB {el:.1f}min", flush=True)
            with log.open("a", encoding="utf-8") as f:
                f.write(f"{step},{'warm' if warm else 'joint'}," + ",".join(f"{v:.6f}" for v in vals)
                        + f",{jf:.3f},{vram:.3f}\n")

        if step % args.save_every == 0 or step == args.steps:
            cp = work / "checkpoints" / f"v60_{step:06d}.pt"
            torch.save({"schema": SCHEMA, "step": step, "model": model.state_dict(),
                        "discriminator": disc.state_dict(), "opt_g": opt_g.state_dict(),
                        "opt_d": opt_d.state_dict(), "vocab": vocab, "speakers": list(SPEAKERS),
                        "styles": list(STYLES), "langs": list(LANGS), "config": config}, cp)
            print("[SAVE]", cp, flush=True)
            save_eval(model, ds_fi_eval, ds_ja_eval, work / "samples", step, fi_eval_speakers)
            print("[EVAL]", work / "samples" / f"step_{step:06d}", flush=True)
    print(f"[PASS] v60 training finished in {(time.time() - t0) / 60:.1f} min -> {work}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
