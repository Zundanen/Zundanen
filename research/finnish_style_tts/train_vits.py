from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

from vits_debuzz_filter import apply_debuzz
from vits_lite import VitsLiteConfig, rand_slice_segments, slice_wave_segments
from vits_debuzz import CENTER_HZ, Q, GAIN_DB
from vits_data import FullDataset, collate_batch, read_csv
from mas_vits_model import MasFullJointVits, load_mit_vits_vendor
from mas_alignment import mas_attention
from transformer_flow import TransformerCouplingBlock, validate_flow_roundtrip

SCHEMA = "zundanen-step7a-transformer-flow-mas-finbert-large-v1"


def seed_all(seed: int = 20261005) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mel_fn(device):
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=24000,
        n_fft=1024,
        win_length=1024,
        hop_length=256,
        f_min=0.0,
        f_max=12000.0,
        n_mels=80,
        power=1.0,
        center=False,
        norm=None,
        mel_scale="slaney",
    ).to(device)


def expand_prior(attn: torch.Tensor, m_p: torch.Tensor, logs_p: torch.Tensor):
    """Expand [B,C,Tx] prior stats to [B,C,Ty] using [B,Tx,Ty] path."""
    m = torch.matmul(attn.transpose(1, 2), m_p.transpose(1, 2)).transpose(1, 2)
    logs = torch.matmul(attn.transpose(1, 2), logs_p.transpose(1, 2)).transpose(1, 2)
    return m, logs


def vits_kl(z_p, logs_q, m_p, logs_p, mask):
    # VITS normalization: sum latent channels and valid frames; divide by
    # valid frame count only.
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    mask = mask.float()
    kl = logs_p - logs_q - 0.5 + 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    return torch.sum(kl * mask) / torch.clamp_min(torch.sum(mask), 1.0)


def safe_write(path: Path, audio: np.ndarray, sr: int = 24000) -> None:
    audio = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.999:
        audio = audio * (0.999 / peak)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr, subtype="PCM_16")


def rms_db(audio: np.ndarray) -> float:
    x = np.asarray(audio, dtype=np.float64)
    return float(20.0 * np.log10(np.sqrt(np.mean(x * x) + 1e-12) + 1e-12))


def make_predicted_path(
    model,
    h,
    x_mask,
    token_lengths,
    *,
    length_scale: float = 1.0,
    max_frames: int = 2500,
):
    """Production duration path using standard deterministic VITS exp(logw)."""
    logw = model.duration(h, x_mask)
    w = torch.exp(logw) * x_mask * float(length_scale)
    w = torch.clamp(torch.ceil(w), min=1.0) * x_mask
    y_lengths = torch.clamp_min(torch.sum(w, dim=[1, 2]).long(), 1)

    if int(y_lengths.max().item()) > max_frames:
        scale = max_frames / float(y_lengths.max().item())
        w = torch.clamp(torch.floor(w * scale), min=1.0) * x_mask
        y_lengths = torch.clamp_min(torch.sum(w, dim=[1, 2]).long(), 1)

    ty = int(y_lengths.max().item())
    d = w[:, 0].long()
    attn = torch.zeros(d.size(0), d.size(1), ty, device=d.device, dtype=h.dtype)
    for b in range(d.size(0)):
        pos = 0
        tx = int(token_lengths[b].item())
        yl = int(y_lengths[b].item())
        for i in range(tx):
            dur = max(1, int(d[b, i].item()))
            end = min(yl, pos + dur)
            if end > pos:
                attn[b, i, pos:end] = 1.0
            pos = end
        if pos < yl:
            attn[b, tx - 1, pos:yl] = 1.0
    return attn, y_lengths, w


def _optional_aalto_durations(eval_ds: FullDataset, index: int):
    """Evaluation-only diagnostic.  Never used by training or MAS."""
    try:
        p = Path(eval_ds.rows[index].get("aalto_duration_cache", ""))
        if not p.exists():
            return None
        with np.load(p, allow_pickle=False) as data:
            return np.asarray(data["durations"], dtype=np.int64).copy()
    except Exception:
        return None


@torch.no_grad()
def save_eval(model, eval_ds: FullDataset, out: Path, step: int) -> None:
    model.eval()
    step_dir = out / f"step_{step:06d}"
    raw_dir = step_dir / "raw"
    clean_dir = step_dir / "debuzz_2p5k_Q0p7_minus18dB"
    raw_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for i in range(min(5, len(eval_ds))):
        s = eval_ds[i]
        tok = s.tokens.unsqueeze(0).cuda()
        bert = s.bert.unsqueeze(0).cuda()
        tl = torch.tensor([s.tokens.numel()], device="cuda")
        spec = s.spec.unsqueeze(0).cuda()
        sl = torch.tensor([s.spec.shape[-1]], device="cuda")

        h, mp, lp, xm = model.text(tok, bert, tl)
        z, mq, lq, ym = model.posterior(spec, sl)
        z = z * ym
        zmean = mq * ym
        zp = model.flow(z, ym, reverse=False) * ym

        # A: acoustic ceiling.
        # A_flow_roundtrip verifies exact information preservation
        # through the new Transformer coupling flow.
        z_roundtrip = model.flow(zp, ym, reverse=True) * ym
        outputs = {
            "A_sample": model.generator(z),
            "A_mean": model.generator(zmean),
            "A_flow_roundtrip": model.generator(z_roundtrip),
        }

        # M: audio supplies only the oracle MAS alignment.  Waveform content is
        # generated from the text prior through flow reverse.
        attn_m = mas_attention(zp, mp, lp, xm, ym, tl, sl)
        mpm, lpm = expand_prior(attn_m, mp, lp)
        for ns, label in [
            (0.0, "M_mean"),
            (0.2, "M_p20"),
            (0.4, "M_p40"),
            (0.6, "M_p60"),
        ]:
            zpm = mpm if ns == 0 else mpm + torch.randn_like(mpm) * torch.exp(lpm) * ns
            zm = model.flow(zpm * ym, ym, reverse=True) * ym
            outputs[label] = model.generator(zm)

        # E: production-style text-only duration prediction.
        attn_e, yl, _w = make_predicted_path(model, h, xm, tl)
        yem = (
            torch.arange(attn_e.size(-1), device="cuda").unsqueeze(0)
            < yl.unsqueeze(1)
        ).unsqueeze(1).to(mp.dtype)
        mpe, lpe = expand_prior(attn_e, mp, lp)
        for ns, label in [(0.0, "E_mean"), (0.4, "E_p40")]:
            zpe = mpe if ns == 0 else mpe + torch.randn_like(mpe) * torch.exp(lpe) * ns
            ze = model.flow(zpe * yem, yem, reverse=True) * yem
            outputs[label] = model.generator(ze)

        base = f"{i + 1:02d}_{s.stem}"
        safe_write(raw_dir / f"{base}_SOURCE.wav", s.wav[0].numpy())
        for label, y in outputs.items():
            arr = y[0, 0].float().cpu().numpy()
            safe_write(raw_dir / f"{base}_{label}.wav", arr)
            safe_write(clean_dir / f"{base}_{label}.wav", apply_debuzz(arr, 24000))
            rows.append(
                {
                    "stem": s.stem,
                    "step": step,
                    "kind": label,
                    "rms_db": f"{rms_db(arr):.4f}",
                }
            )

        mas_d = attn_m[0, : int(tl[0].item())].sum(dim=1).long().cpu().numpy()
        aalto_d = _optional_aalto_durations(eval_ds, i)
        aalto_line = "aalto_compare=unavailable\n"
        if aalto_d is not None and aalto_d.shape == mas_d.shape:
            mae = float(np.mean(np.abs(aalto_d.astype(np.float64) - mas_d.astype(np.float64))))
            maxerr = int(np.max(np.abs(aalto_d - mas_d))) if mas_d.size else 0
            aalto_line = f"mas_vs_aalto_mae_frames={mae:.4f}\nmas_vs_aalto_max_abs_frames={maxerr}\n"

        (step_dir / f"{base}.txt").write_text(
            s.text
            + "\n"
            + f"true_frames={int(sl[0])}\n"
            + f"pred_frames={int(yl[0])}\n"
            + f"mas_token_frames={','.join(map(str, mas_d.tolist()))}\n"
            + aalto_line,
            encoding="utf-8",
        )

    with (step_dir / "rms_metrics.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stem", "step", "kind", "rms_db"])
        w.writeheader()
        w.writerows(rows)
    model.train()


def save_ck(path, model, disc, opt_g, opt_d, step, vocab, config, vendor_commit):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": SCHEMA,
            "step": step,
            "model": model.state_dict(),
            "discriminator": disc.state_dict(),
            "opt_g": opt_g.state_dict(),
            "opt_d": opt_d.state_dict(),
            "vocab": vocab,
            "config": config,
            "vendor_commit": vendor_commit,
        },
        path,
    )


def _clean_path(value: str) -> Path:
    return Path(value.strip().strip('"')).resolve()


def _load_step7_checkpoint(path: Path) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("schema") != SCHEMA:
        raise SystemExit(f"[FAIL] Wrong Finnish VITS checkpoint schema: {path}")
    if "model" not in ck or "vocab" not in ck:
        raise SystemExit(f"[FAIL] Incomplete Finnish VITS checkpoint: {path}")
    return ck


def _resolve_dataset_paths(args):
    """Resolve generic or legacy Finnish VITS dataset layout.

    Priority:
      1. explicit --train-csv/--eval-csv/--vocab
      2. <dataset>/vits_dataset/{train.csv,eval.csv,vocab.json}
      3. <dataset>/{train.csv,eval.csv,vocab.json}
      4. legacy Step5L/Aalto layout under <dataset>/step5a_validation
    """
    dataset = _clean_path(args.dataset) if args.dataset else None

    explicit = any([args.train_csv, args.eval_csv, args.vocab])
    if explicit:
        if not (args.train_csv and args.eval_csv and args.vocab):
            raise SystemExit("[FAIL] --train-csv, --eval-csv and --vocab must be supplied together")
        train_csv = _clean_path(args.train_csv)
        eval_csv = _clean_path(args.eval_csv)
        vocab_path = _clean_path(args.vocab)
        base = dataset if dataset is not None else train_csv.parent
        layout = "explicit"
        legacy_valdir = None
    else:
        if dataset is None:
            raise SystemExit("[FAIL] Give --dataset or explicit --train-csv/--eval-csv/--vocab")

        candidates = [dataset / "vits_dataset", dataset]
        found = None
        for c in candidates:
            tc, ec, vp = c / "train.csv", c / "eval.csv", c / "vocab.json"
            if tc.exists() and ec.exists() and vp.exists():
                found = (tc, ec, vp, c)
                break

        if found is not None:
            train_csv, eval_csv, vocab_path, _prepared_dir = found
            base = dataset
            layout = "generic"
            legacy_valdir = None
        else:
            valdir = dataset / "step5a_validation"
            source_work = valdir / "step5f_neutral_text_v3_aalto"
            train_csv = source_work / "train_aalto_large.csv"
            eval_csv = source_work / "eval5_aalto_large.csv"
            vocab_path = valdir / "step5d_pilot" / "vocab.json"
            base = dataset
            layout = "legacy"
            legacy_valdir = valdir

    for p in (train_csv, eval_csv, vocab_path):
        if not p.exists():
            raise SystemExit(f"[FAIL] Missing dataset file: {p}")

    if args.work_dir:
        work = _clean_path(args.work_dir)
    elif layout == "legacy" and legacy_valdir is not None:
        # Preserve the historical output location for old commands.
        work = legacy_valdir / args.run_name
    else:
        work = base / "vits_runs" / args.run_name

    return dataset, train_csv, eval_csv, vocab_path, work, layout


def _preflight_manifest(manifest: Path, vocab: dict[str, int], *, expected_bert_dim: int = 1024) -> None:
    rows = read_csv(manifest)
    if not rows:
        raise SystemExit(f"[FAIL] Empty manifest: {manifest}")
    required = {"stem", "text", "wav_path", "bert_cache"}
    missing_cols = sorted(required - set(rows[0].keys()))
    if missing_cols:
        raise SystemExit(f"[FAIL] {manifest} missing columns: {missing_cols}")

    unknown_tokens: dict[str, list[str]] = {}
    missing_paths: list[str] = []
    wrong_dims: list[str] = []
    for r in rows:
        stem = r.get("stem", "?")
        wav = Path(r.get("wav_path", ""))
        cache = Path(r.get("bert_cache", ""))
        if not wav.exists():
            missing_paths.append(f"{stem}: wav={wav}")
            continue
        if not cache.exists():
            missing_paths.append(f"{stem}: bert_cache={cache}")
            continue
        try:
            with np.load(cache, allow_pickle=False) as data:
                gs = tuple(str(x) for x in data["grapheme_tokens"].tolist())
                feat = data["features"]
                if feat.ndim != 2 or int(feat.shape[1]) != expected_bert_dim:
                    wrong_dims.append(f"{stem}: {tuple(feat.shape)}")
                bad = sorted({g for g in gs if g not in vocab})
                if bad:
                    unknown_tokens[stem] = bad
        except Exception as exc:
            missing_paths.append(f"{stem}: cache read error: {exc}")

    if missing_paths:
        detail = "\n  ".join(missing_paths[:20])
        raise SystemExit(f"[FAIL] Missing/unreadable dataset paths in {manifest}:\n  {detail}\n  total={len(missing_paths)}")
    if wrong_dims:
        detail = "\n  ".join(wrong_dims[:20])
        raise SystemExit(f"[FAIL] Expected FinBERT Large {expected_bert_dim}-D caches:\n  {detail}\n  total={len(wrong_dims)}")
    if unknown_tokens:
        detail = "\n  ".join(f"{k}: {v}" for k, v in list(unknown_tokens.items())[:20])
        raise SystemExit(
            "[FAIL] Dataset contains grapheme tokens absent from the selected checkpoint/vocab.\n"
            f"  {detail}\n  affected={len(unknown_tokens)}\n"
            "Prepare both datasets with prepare_dataset.py for fixed-vocab fine-tuning."
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="", help="dataset root; generic prepared layout or legacy Step5 layout")
    ap.add_argument("--train-csv", default="", help="explicit generic train.csv")
    ap.add_argument("--eval-csv", default="", help="explicit generic eval.csv")
    ap.add_argument("--vocab", default="", help="explicit vocab.json")
    ap.add_argument("--work-dir", default="", help="override output run directory")
    ap.add_argument("--run-name", default="vits_transformer_flow_mas_finbert_large")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6, help="DataLoader workers; 0 disables worker processes")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--resume", default="", help="continue the SAME run: model+discriminator+optimizers+step")
    ap.add_argument("--init-checkpoint", default="", help="fine-tune on a NEW dataset: load model weights, reset optimizers and step")
    ap.add_argument("--init-discriminator", action="store_true", help="with --init-checkpoint, also load discriminator weights")
    ap.add_argument("--no-preflight", dest="preflight", action="store_false", help="skip full manifest/cache validation")
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.set_defaults(amp=True, preflight=True)
    args = ap.parse_args()

    if args.resume and args.init_checkpoint:
        raise SystemExit("[FAIL] Use only one of --resume or --init-checkpoint")
    if args.steps < 1:
        raise SystemExit("[FAIL] --steps must be >= 1")

    seed_all()
    dataset, train_csv, eval_csv, vocab_path, work, layout = _resolve_dataset_paths(args)
    here = Path(__file__).resolve()
    root = here.parents[2]

    if not torch.cuda.is_available():
        raise SystemExit("[FAIL] CUDA is required for Finnish VITS.")

    resume_ck = None
    init_ck = None
    resume_path = None
    init_path = None
    if args.resume:
        resume_path = _clean_path(args.resume)
        resume_ck = _load_step7_checkpoint(resume_path)
    if args.init_checkpoint:
        init_path = _clean_path(args.init_checkpoint)
        init_ck = _load_step7_checkpoint(init_path)

    file_vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    source_ck = resume_ck or init_ck
    if source_ck is not None:
        # A checkpoint's token IDs are part of the learned model.  Use them
        # exactly, then validate that the target dataset is representable.
        vocab = {str(k): int(v) for k, v in source_ck["vocab"].items()}
        if file_vocab != vocab:
            print("[VOCAB] dataset vocab differs from checkpoint; using checkpoint vocab and validating target tokens")
    else:
        vocab = {str(k): int(v) for k, v in file_vocab.items()}

    if args.preflight:
        print("[PREFLIGHT] validating train/eval WAVs, caches, 1024-D features and vocabulary...")
        _preflight_manifest(train_csv, vocab)
        _preflight_manifest(eval_csv, vocab)
        print("[PREFLIGHT] PASS")

    models, attentions, losses, vendor = load_mit_vits_vendor(root)
    commit_path = vendor / "ZUNDANEN_PINNED_COMMIT.txt"
    vendor_commit = commit_path.read_text(encoding="utf-8").strip() if commit_path.exists() else "unknown"

    # Keep Step 6A/7A geometry exactly.  Aalto durations are optional diagnostics
    # only; generic manifests need wav_path/text/bert_cache and no Aalto cache.
    data_cfg = VitsLiteConfig(
        sample_rate=24000,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mels=80,
        fmin=0.0,
        fmax=12000.0,
        bert_dim=1024,
        hidden=256,
        latent=192,
        filter_channels=768,
        text_heads=8,
        text_layers=6,
        posterior_layers=8,
        flow_layers=6,
        upsample_initial_channel=512,
        segment_frames=64,
    )
    train_ds = FullDataset(train_csv, vocab, data_cfg)
    eval_ds = FullDataset(eval_csv, vocab, data_cfg)

    probe = train_ds[0]
    bert_shape = tuple(probe.bert.shape)
    print(f"[VITS] FinBERT probe shape={bert_shape}")
    if 1024 not in bert_shape:
        raise RuntimeError(f"[FAIL] Expected FinBERT Large 1024-D cache, got shape={bert_shape}")

    loader_kwargs = dict(
        dataset=train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=max(0, int(args.workers)),
        pin_memory=True,
        collate_fn=collate_batch,
        drop_last=True,
    )
    if int(args.workers) > 0:
        loader_kwargs.update(
            persistent_workers=True,
            prefetch_factor=2,
        )
    loader = torch.utils.data.DataLoader(**loader_kwargs)

    model = MasFullJointVits(models, attentions, len(vocab)).cuda()

    # Finnish VITS ONLY architectural change:
    # VITS ResidualCouplingBlock -> deterministic Transformer
    # mean-only additive coupling block.
    model.flow = TransformerCouplingBlock(
        channels=192,
        hidden_channels=256,
        n_heads=4,
        n_layers=2,
        n_flows=4,
        ffn_channels=768,
    ).cuda()

    flow_rt_error = validate_flow_roundtrip(
        model.flow,
        device=torch.device("cuda"),
    )
    if flow_rt_error > 1e-4:
        raise RuntimeError(f"Finnish VITS Transformer flow roundtrip failed: {flow_rt_error}")
    print(f"[FLOW TEST] Transformer coupling roundtrip max_abs_error={flow_rt_error:.9g}")

    disc = models.MultiPeriodDiscriminator(use_spectral_norm=False).cuda()

    # Load weights BEFORE constructing/restoring optimizer state.
    start = 0
    mode = "scratch"
    if resume_ck is not None:
        model.load_state_dict(resume_ck["model"])
        disc.load_state_dict(resume_ck["discriminator"])
        start = int(resume_ck["step"])
        mode = "resume"
        print("[RESUME]", resume_path)
    elif init_ck is not None:
        model.load_state_dict(init_ck["model"])
        if args.init_discriminator:
            disc.load_state_dict(init_ck["discriminator"])
        start = 0
        mode = "finetune"
        print("[INIT CHECKPOINT]", init_path)
        print("[FINETUNE] model weights loaded; optimizer state and step reset")
        print(f"[FINETUNE] discriminator={'loaded' if args.init_discriminator else 'fresh'}")

    params_g = list(model.parameters())
    opt_g = torch.optim.AdamW(
        params_g,
        lr=args.lr,
        betas=(0.8, 0.99),
        eps=1e-9,
        weight_decay=0.0,
    )
    opt_d = torch.optim.AdamW(
        disc.parameters(),
        lr=args.lr,
        betas=(0.8, 0.99),
        eps=1e-9,
        weight_decay=0.0,
    )
    if resume_ck is not None:
        opt_g.load_state_dict(resume_ck["opt_g"])
        opt_d.load_state_dict(resume_ck["opt_d"])

    scaler_g = torch.amp.GradScaler("cuda", enabled=args.amp)
    scaler_d = torch.amp.GradScaler("cuda", enabled=args.amp)

    work.mkdir(parents=True, exist_ok=True)
    ckdir = work / "checkpoints"
    samples = work / "samples"
    ckdir.mkdir(parents=True, exist_ok=True)
    samples.mkdir(parents=True, exist_ok=True)

    config = model.config_dict() | {
        "schema": SCHEMA,
        "flow_type": "transformer_mean_only_additive",
        "flow_channels": 192,
        "flow_hidden_channels": 256,
        "flow_heads": 4,
        "flow_transformer_layers_per_coupling": 2,
        "flow_n_flows": 4,
        "flow_ffn_channels": 768,
        "flow_dropout": 0.0,
        "flow_affine_scale": False,
        "flow_logdet": 0.0,
        "dataset_layout": layout,
        "train_csv": str(train_csv),
        "eval_csv": str(eval_csv),
        "vocab_path": str(vocab_path),
        "work_dir": str(work),
        "train_mode": mode,
        "init_checkpoint": str(init_path) if init_path is not None else "",
        "resume_checkpoint": str(resume_path) if resume_path is not None else "",
        "init_discriminator": bool(args.init_discriminator),
        "batch_size": args.batch_size,
        "workers": int(args.workers),
        "lr": args.lr,
        "c_mel": 45.0,
        "c_kl": 1.0,
        "c_duration": 1.0,
        "precision": "amp-fp16 networks; MAS+KL fp32",
        "train_samples": len(train_ds),
        "eval_samples": len(eval_ds),
        "vendor_commit": vendor_commit,
        "mas_impl": "original PyTorch cumulative-max DP; no AGPL source",
        "aalto_training_use": "none; optional eval-only comparison when aalto_duration_cache exists",
    }
    (work / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    shutil.copy2(vendor / "LICENSE", work / "VITS-MIT-LICENSE.txt")

    if mode == "resume" and args.steps <= start:
        raise SystemExit(
            f"[FAIL] --steps={args.steps} must be greater than resumed checkpoint step={start}. "
            "For a new-dataset fine-tune use --init-checkpoint instead."
        )
    props = torch.cuda.get_device_properties(0)
    print(
        f"[CUDA] {torch.cuda.get_device_name(0)} "
        f"VRAM={props.total_memory / 1024**3:.1f}GB torch={torch.__version__}"
    )
    print(f"[DATA] layout={layout} train={len(train_ds)} eval={len(eval_ds)} batch={args.batch_size}")
    print(f"[DATA] train_csv={train_csv}")
    print(f"[DATA] eval_csv ={eval_csv}")
    print(f"[OUT]  {work}")
    print(f"[VITS] mode={mode} start_step={start} target_step={args.steps}")
    print("[ALIGNMENT] posterior z_p + text prior likelihood -> PyTorch MAS; Aalto duration NOT used for training")
    print("[TEXT] Finnish grapheme + cached FinBERT Large 1024 -> VITS prior m_p/logs_p")
    print("[FLOW] Finnish VITS original PyTorch Transformer mean-only additive coupling; 4 flows x 2 Transformer layers")
    print("[ACOUSTIC] same MIT VITS PosteriorEncoder + Generator + MPD as Step 5L")
    print("[LOSS] mel x45 + GAN + feature matching + MAS-duration + VITS KL x1")
    print("[DURATION] train target=log(w_mas+1e-6); inference=exp(logw)")
    print("[EVAL] A_sample / A_flow_roundtrip / M_p40 / E_p40 are the primary comparison")

    mel = mel_fn(torch.device("cuda"))
    log_path = work / "train.csv"
    if not log_path.exists() or start == 0:
        log_path.write_text(
            "step,total,mel,fm,adv,kl,dur,D,qlog,plog,mas_min_dur,mas_max_dur,vram_alloc_gb,vram_reserved_gb\n",
            encoding="utf-8",
        )

    it = iter(loader)
    t0 = time.time()
    model.train()
    disc.train()

    for step in range(start + 1, args.steps + 1):
        try:
            b = next(it)
        except StopIteration:
            it = iter(loader)
            b = next(it)

        tok = b["tokens"].cuda(non_blocking=True)
        bert = b["bert"].cuda(non_blocking=True)
        tl = b["token_lengths"].cuda(non_blocking=True)
        wav = b["wav"].cuda(non_blocking=True)
        spec = b["spec"].cuda(non_blocking=True)
        sl = b["spec_lengths"].cuda(non_blocking=True)

        with torch.amp.autocast("cuda", enabled=args.amp, dtype=torch.float16):
            h, mp, lp, xm = model.text(tok, bert, tl)
            z, mq, lq, ym = model.posterior(spec, sl)
            zp = model.flow(z, ym, reverse=False)

        # Alignment is discrete and detached, as in VITS-style MAS training.
        # It is deliberately computed from the current posterior/prior pair,
        # not from Aalto forced-alignment durations.
        attn = mas_attention(zp, mp, lp, xm, ym, tl, sl)

        with torch.amp.autocast("cuda", enabled=args.amp, dtype=torch.float16):
            mpe, lpe = expand_prior(attn, mp, lp)
            w_mas = attn.sum(dim=2).unsqueeze(1)
            logwt = torch.log(w_mas + 1e-6) * xm
            logwp = model.duration(h, xm)
            dur_loss = torch.sum((logwp - logwt) ** 2 * xm) / torch.clamp_min(xm.sum(), 1.0)

            zseg, starts = rand_slice_segments(z * ym, sl, model.segment_frames)
            yhat = model.generator(zseg)
            yreal = slice_wave_segments(wav, starts, model.segment_samples, model.hop_length)

        # Discriminator.
        opt_d.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=args.amp, dtype=torch.float16):
            dr, dg, _, _ = disc(yreal, yhat.detach())
            loss_d, _, _ = losses.discriminator_loss(dr, dg)
        scaler_d.scale(loss_d).backward()
        scaler_d.unscale_(opt_d)
        torch.nn.utils.clip_grad_norm_(disc.parameters(), 1000.0)
        scaler_d.step(opt_d)
        scaler_d.update()

        # Joint generator / posterior / flow / text / duration update.
        opt_g.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=args.amp, dtype=torch.float16):
            dr, dg, fr, fg = disc(yreal, yhat)
            loss_fm = losses.feature_loss(fr, fg)
            loss_adv, _ = losses.generator_loss(dg)
            mr = torch.log(torch.clamp(mel(yreal.squeeze(1)), min=1e-5))
            mf = torch.log(torch.clamp(mel(yhat.squeeze(1)), min=1e-5))
            tm = min(mr.size(-1), mf.size(-1))
            loss_mel_raw = F.l1_loss(mf[..., :tm], mr[..., :tm])

        loss_kl = vits_kl(zp, lq, mpe, lpe, ym)
        total = (
            loss_adv.float()
            + loss_fm.float()
            + 45.0 * loss_mel_raw.float()
            + loss_kl
            + dur_loss.float()
        )
        if not torch.isfinite(total):
            raise RuntimeError(f"non-finite Finnish VITS loss at step {step}: {float(total)}")

        scaler_g.scale(total).backward()
        scaler_g.unscale_(opt_g)
        torch.nn.utils.clip_grad_norm_(params_g, 1000.0)
        scaler_g.step(opt_g)
        scaler_g.update()

        if step == 1 or step % 10 == 0:
            alloc = torch.cuda.max_memory_allocated() / 1024**3
            reserved = torch.cuda.max_memory_reserved() / 1024**3
            qlog = float(
                (lq * ym).sum().detach().float().cpu()
                / torch.clamp_min(ym.sum().detach().float().cpu() * lq.size(1), 1.0)
            )
            plog = float(
                (lpe * ym).sum().detach().float().cpu()
                / torch.clamp_min(ym.sum().detach().float().cpu() * lpe.size(1), 1.0)
            )
            valid_w = w_mas[xm.bool()]
            mas_min = float(valid_w.min().detach().float().cpu()) if valid_w.numel() else 0.0
            mas_max = float(valid_w.max().detach().float().cpu()) if valid_w.numel() else 0.0
            vals = [
                float(x.detach().float().cpu())
                for x in [total, loss_mel_raw, loss_fm, loss_adv, loss_kl, dur_loss, loss_d]
            ]
            print(
                f"step {step:5d}/{args.steps} total={vals[0]:.3f} mel={vals[1]:.3f} "
                f"fm={vals[2]:.3f} adv={vals[3]:.3f} KL={vals[4]:.3f} dur={vals[5]:.3f} "
                f"D={vals[6]:.3f} qlog={qlog:.3f} plog={plog:.3f} "
                f"MASdur={mas_min:.0f}..{mas_max:.0f} VRAM={alloc:.2f}/{reserved:.2f}GB"
            )
            with log_path.open("a", encoding="utf-8") as f:
                f.write(
                    f"{step},{vals[0]:.6f},{vals[1]:.6f},{vals[2]:.6f},{vals[3]:.6f},"
                    f"{vals[4]:.6f},{vals[5]:.6f},{vals[6]:.6f},{qlog:.6f},{plog:.6f},"
                    f"{mas_min:.4f},{mas_max:.4f},{alloc:.4f},{reserved:.4f}\n"
                )

        if step % args.save_every == 0 or step == args.steps:
            cp = ckdir / f"vits_{step:06d}.pt"
            save_ck(cp, model, disc, opt_g, opt_d, step, vocab, config, vendor_commit)
            print("[SAVE]", cp)
            save_eval(model, eval_ds, samples, step)

    print(f"[PASS] Finnish VITS finished in {(time.time() - t0) / 60:.1f} min")
    print("Listen to:", samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
