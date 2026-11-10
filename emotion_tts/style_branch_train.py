from __future__ import annotations

"""v61: train ONLY the language-blind style branch on Japanese style data.

Original Zundanen code (MIT).  The v60 checkpoint is read-only and frozen
(posterior, flow, base text prior, duration, generator).  The branch learns the
Japanese Normal -> style difference in prior space through the usual VITS
objective (MAS + KL + log-duration).  No generator/discriminator pass is needed,
so this is fast.  Normal rows are not used: their branch output is zero by
construction.

Every evaluation writes Finnish samples and an automatic metric file comparing
F0 / length of each style against Normal for Finnish x Zundamon.  This is the
gate v60 failed (F0 ratio ~1.00 at 3x strength).
"""

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "emotion_tts"))
from emo_common import read_csv  # noqa: E402
from emo_vits import import_research, load_v60_checkpoint  # noqa: E402
from emo_train_base import V60Dataset, collate, write_wav  # noqa: E402
from style_branch import SCHEMA, V61, StyleBranch, branch_token_table, style_mix_tensor  # noqa: E402

import_research(ROOT)
from mas_alignment import mas_attention  # noqa: E402
from train_vits import expand_prior, vits_kl  # noqa: E402

SR = 24000


@torch.no_grad()
def synthesize(model: V61, tok, bert, has_bert, *, speaker, style, lang, strength=1.0, noise=0.4,
               length_scale=1.0, seed=1234, device="cuda"):
    tok = tok.unsqueeze(0).to(device)
    bert = bert.unsqueeze(0).to(device)
    tl = torch.tensor([tok.size(1)], device=device)
    h, mp, lp, xm, dlogw, _ = model.prior(
        tok, bert, tl, speaker_id=torch.tensor([model.speakers.index(speaker)], device=device),
        lang_id=torch.tensor([model.langs.index(lang)], device=device),
        has_bert=torch.tensor([float(has_bert)], device=device),
        style_mix=style_mix_tensor(model.styles, style, strength, device=device))
    logw = model.log_duration(h, xm, dlogw)
    w = torch.clamp(torch.ceil(torch.exp(logw) * xm * float(length_scale)), min=1.0) * xm
    # Reuse the v60 path builder by handing it a duration module output.
    attn, yl = _path_from_w(w, tl)
    ym = (torch.arange(attn.size(-1), device=device).unsqueeze(0) < yl.unsqueeze(1)).unsqueeze(1).to(mp.dtype)
    m, logs = expand_prior(attn, mp, lp)
    g = torch.Generator(device=device).manual_seed(int(seed))
    z = m + torch.randn(m.shape, generator=g, device=device, dtype=m.dtype) * torch.exp(logs) * noise
    y = model.base.generator(model.base.flow(z * ym, ym, reverse=True) * ym)
    return y[0, 0].float().cpu().numpy()


def _path_from_w(w, token_lengths, max_frames=2500):
    y_lengths = torch.clamp_min(torch.sum(w, dim=[1, 2]).long(), 1)
    if int(y_lengths.max()) > max_frames:
        w = torch.clamp(torch.floor(w * max_frames / float(y_lengths.max())), min=1.0)
        y_lengths = torch.clamp_min(torch.sum(w, dim=[1, 2]).long(), 1)
    d = w[:, 0].long()
    attn = torch.zeros(d.size(0), d.size(1), int(y_lengths.max()), device=d.device)
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


def f0_median(audio: np.ndarray) -> float:
    import librosa
    y = librosa.resample(audio.astype(np.float32), orig_sr=SR, target_sr=16000)
    f, _v, _p = librosa.pyin(y, fmin=80, fmax=800, sr=16000, frame_length=1024)
    return float(np.nanmedian(f)) if np.any(~np.isnan(f)) else float("nan")


@torch.no_grad()
def evaluate(model, fi_rows, ja_rows, vocab, out: Path, step: int, strengths=(1.0, 2.0)):
    """Finnish x zundamon x style (+ Japanese unseen test).  Returns metric rows."""
    model.eval()
    d = out / f"step_{step:06d}"
    ds_fi, ds_ja = V60Dataset(fi_rows, vocab), V60Dataset(ja_rows, vocab)
    rows = []
    for i, r in enumerate(fi_rows):
        tok, bert, _ = ds_fi.text_inputs(r)
        base = synthesize(model, tok, bert, 1.0, speaker="zundamon", style="normal", lang="fi")
        write_wav(d / "fi" / f"{i+1:02d}_zundamon_normal.wav", base)
        f0n, dn = f0_median(base), base.size / SR
        for style in model.styles:
            if style == "normal":
                continue
            for a in strengths:
                y = synthesize(model, tok, bert, 1.0, speaker="zundamon", style=style, lang="fi", strength=a)
                write_wav(d / "fi" / f"{i+1:02d}_zundamon_{style}_s{a:g}.wav", y)
                rows.append({"step": step, "lang": "fi", "item": i + 1, "style": style, "strength": a,
                             "f0_ratio": f0_median(y) / f0n, "dur_ratio": (y.size / SR) / dn})
    for r in ja_rows:
        tok, bert, _ = ds_ja.text_inputs(r)
        y = synthesize(model, tok, bert, 0.0, speaker=r["speaker"], style=r["style"], lang="ja")
        write_wav(d / "ja" / f"{r['stem']}_TEXT.wav", y)
    model.train()
    model.base.eval()
    return rows


def summarize(rows):
    out = {}
    for style in sorted({r["style"] for r in rows}):
        for a in sorted({r["strength"] for r in rows}):
            sel = [r for r in rows if r["style"] == style and r["strength"] == a]
            if sel:
                out[f"{style}@{a:g}"] = (float(np.nanmean([r["f0_ratio"] for r in sel])),
                                         float(np.mean([r["dur_ratio"] for r in sel])))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--v60-run", default="v60_main")
    ap.add_argument("--base-checkpoint", default="", help="default: newest v60 checkpoint of --v60-run")
    ap.add_argument("--run", default="v61_main")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--delta-l2", type=float, default=1e-3, help="small penalty on branch output size")
    ap.add_argument("--no-speaker", dest="use_speaker", action="store_false",
                    help="branch also blind to speaker (style averaged over voices)")
    ap.add_argument("--smoke", action="store_true")
    ap.set_defaults(use_speaker=True)
    args = ap.parse_args()
    if args.smoke:
        args.steps, args.batch_size, args.save_every, args.workers = 20, 4, 20, 0

    root = Path(args.root).resolve()
    v60 = root / "outputs" / "v60" / args.v60_run
    data = v60 / "data"
    base_cp = Path(args.base_checkpoint).resolve() if args.base_checkpoint else \
        sorted((v60 / "train" / "checkpoints").glob("v60_*.pt"))[-1]
    work = root / "outputs" / "v61" / (args.run + ("_smoke" if args.smoke else ""))
    work.mkdir(parents=True, exist_ok=True)
    (work / "checkpoints").mkdir(exist_ok=True)
    if not torch.cuda.is_available():
        raise SystemExit("[FAIL] CUDA is required")
    random.seed(20261019); np.random.seed(20261019); torch.manual_seed(20261019)

    base_ck, base, _models, _losses = load_v60_checkpoint(root, base_cp, "cpu")
    vocab = base_ck["vocab"]
    ja = read_csv(data / "ja.csv")
    seen = {t for r in ja if r["split"] == "train" for t in r["tokens"].split(" ")}
    table, mask, info = branch_token_table(vocab, seen)
    branch = StyleBranch(len(info["symbols"]), len(base_ck["styles"]), len(base_ck["speakers"]),
                         use_speaker=args.use_speaker)
    model = V61(base, branch, table, mask, base_ck["styles"], base_ck["speakers"], base_ck["langs"]).cuda()
    for p in model.base.parameters():
        p.requires_grad_(False)
    model.base.eval()
    masked = sorted(t for t in vocab if t not in info["mapping"])
    print(f"[BASE] {base_cp} (step {base_ck['step']}) frozen")
    print(f"[BRANCH] symbols={len(info['symbols'])} params={sum(p.numel() for p in branch.parameters()):,} "
          f"speaker={'on' if args.use_speaker else 'off'} masked={masked}")
    print("[BRANCH] proxies: " + ", ".join(f"{k}->{v}" for k, v in info["mapping"].items() if k != v))

    train_rows = [r for r in ja if r["split"] == "train" and r["style"] != "normal"]
    ds = V60Dataset(train_rows, vocab)
    kw = dict(batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate,
              pin_memory=True, drop_last=True)
    if args.workers > 0:
        kw.update(persistent_workers=True, prefetch_factor=2)
    loader = torch.utils.data.DataLoader(ds, **kw)
    fi_eval = read_csv(data / "fi_eval.csv")[:3]
    ja_eval = [r for r in ja if r["split"] == "test" and r["speaker"] == "zundamon" and r["sentence_id"] in {"7", "60"}]
    print(f"[DATA] Japanese style rows={len(train_rows)} (Normal rows skipped: branch is zero for Normal)")

    opt = torch.optim.AdamW(branch.parameters(), lr=args.lr, betas=(0.8, 0.99), weight_decay=1e-4)
    config = {"schema": SCHEMA, "args": vars(args), "base_checkpoint": str(base_cp), "symbols": info["symbols"],
              "mapping": info["mapping"], "masked": masked, "train_rows": len(train_rows)}
    (work / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    log = work / "train.csv"
    log.write_text("step,total,kl,dur,l2,dm_abs,dlogw_mean\n", encoding="utf-8")
    metrics_path = work / "fi_style_metrics.csv"

    def batches():
        while True:
            for b in loader:
                yield b
    it = batches()
    t0 = time.time()
    model.train(); model.base.eval()
    for step in range(1, args.steps + 1):
        b = next(it)
        tok, bert, tl = b["tokens"].cuda(), b["bert"].cuda(), b["token_lengths"].cuda()
        spec, sl = b["spec"].cuda(), b["spec_lengths"].cuda()
        spk, lang, hb = b["speaker"].cuda(), b["lang"].cuda(), b["has_bert"].cuda()
        mix = torch.nn.functional.one_hot(b["style"], len(model.styles)).float().cuda()
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.float16):
                z, _mq, lq, ym = model.base.posterior(spec, sl)
                zp = model.base.flow(z, ym, reverse=False)
            zp, lq, ym = zp.float(), lq.float(), ym.float()
        h, mp, lp, xm, dlogw, (dm, dlogs, _) = model.prior(tok, bert, tl, speaker_id=spk, lang_id=lang,
                                                           has_bert=hb, style_mix=mix)
        attn = mas_attention(zp, mp.detach(), lp.detach(), xm, ym, tl, sl)
        mpe, lpe = expand_prior(attn, mp, lp)
        loss_kl = vits_kl(zp, lq, mpe, lpe, ym)
        logwt = torch.log(attn.sum(dim=2).unsqueeze(1) + 1e-6) * xm
        logwp = model.log_duration(h.detach(), xm, dlogw)
        loss_dur = torch.sum((logwp - logwt) ** 2 * xm) / torch.clamp_min(xm.sum(), 1.0)
        l2 = (dm.pow(2).sum() + dlogs.pow(2).sum()) / torch.clamp_min(xm.sum() * dm.size(1), 1.0)
        total = loss_kl + loss_dur + args.delta_l2 * l2
        if not torch.isfinite(total):
            raise RuntimeError(f"non-finite loss at step {step}: {b['stems']}")
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(branch.parameters(), 10.0)
        opt.step()

        if step == 1 or step % 50 == 0:
            dm_abs = float((dm.abs().sum() / torch.clamp_min(xm.sum() * dm.size(1), 1.0)).detach())
            dw = float(((dlogw * xm).sum() / torch.clamp_min(xm.sum(), 1.0)).detach())
            vals = [float(x.detach()) for x in (total, loss_kl, loss_dur, l2)]
            print(f"step {step:5d}/{args.steps} total={vals[0]:.4f} KL={vals[1]:.4f} dur={vals[2]:.4f} "
                  f"|dm|={dm_abs:.4f} dlogw={dw:+.4f} {(time.time()-t0)/60:.1f}min", flush=True)
            with log.open("a", encoding="utf-8") as f:
                f.write(f"{step}," + ",".join(f"{v:.6f}" for v in vals) + f",{dm_abs:.6f},{dw:.6f}\n")

        if step % args.save_every == 0 or step == args.steps:
            cp = work / "checkpoints" / f"v61_{step:06d}.pt"
            torch.save({"schema": SCHEMA, "step": step, "branch": branch.state_dict(), "opt": opt.state_dict(),
                        "base_checkpoint": str(base_cp), "seen_symbols": sorted(seen),
                        "use_speaker": args.use_speaker, "config": config}, cp)
            rows = evaluate(model, fi_eval, ja_eval, vocab, work / "samples", step)
            new = not metrics_path.exists()
            with metrics_path.open("a", encoding="utf-8-sig", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                if new:
                    w.writeheader()
                w.writerows(rows)
            summ = summarize(rows)
            print(f"[SAVE] {cp}", flush=True)
            print("[FI-METRIC] Finnish x zundamon, ratio vs Normal (F0, length): "
                  + "  ".join(f"{k}=({v[0]:.3f},{v[1]:.3f})" for k, v in summ.items()), flush=True)
    print(f"[PASS] v61 finished in {(time.time()-t0)/60:.1f} min -> {work}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
