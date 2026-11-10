from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import traceback
from pathlib import Path

READY = "ZUNDANEN_VITS_READY\t"
RESULT = "ZUNDANEN_VITS_RESULT\t"
STEP7_SCHEMA = "zundanen-step7a-transformer-flow-mas-finbert-large-v1"
STEP8_SCHEMA = "zundanen-step8a-style-transformer-flow-v1"
PROFILE_SCHEMA = "zundanen-paired-style-profile-v1"


def emit(prefix: str, payload: dict) -> None:
    print(prefix + json.dumps(payload, ensure_ascii=False), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--research-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    research = Path(args.research_dir).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    if not research.is_dir():
        raise FileNotFoundError(f"Research directory not found: {research}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"VITS checkpoint not found: {checkpoint}")
    sys.path.insert(0, str(research))

    import numpy as np
    import soundfile as sf
    import torch

    from finbert_features import extract_features, load_finbert_model
    from mas_vits_model import MasFullJointVits, load_mit_vits_vendor
    from transformer_flow import TransformerCouplingBlock

    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    schema = str(ck.get("schema") or "")
    if schema not in {STEP7_SCHEMA, STEP8_SCHEMA}:
        raise RuntimeError(f"Unsupported checkpoint schema: {schema!r}.")
    vocab = ck.get("vocab")
    if not isinstance(vocab, dict) or not vocab:
        raise RuntimeError("Checkpoint does not contain a usable vocab.")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    models, attentions, _losses, _vendor = load_mit_vits_vendor(root)
    model = MasFullJointVits(models, attentions, len(vocab)).to(device)
    cfg = ck.get("config") or {}
    model.flow = TransformerCouplingBlock(
        channels=int(cfg.get("flow_channels", 192)),
        hidden_channels=int(cfg.get("flow_hidden_channels", 256)),
        n_heads=int(cfg.get("flow_heads", 4)),
        n_layers=int(cfg.get("flow_transformer_layers_per_coupling", 2)),
        n_flows=int(cfg.get("flow_n_flows", 4)),
        ffn_channels=int(cfg.get("flow_ffn_channels", 768)),
    ).to(device)
    checkpoint_styles = tuple(str(x) for x in (ck.get("styles") or []))
    if schema == STEP8_SCHEMA:
        from style_conditioning import upgrade_text_encoder
        if not checkpoint_styles or checkpoint_styles[0] != "neutral":
            raise RuntimeError("Step8 checkpoint has invalid styles")
        upgrade_text_encoder(model, attentions, len(vocab), checkpoint_styles)
    else:
        checkpoint_styles = ("neutral",)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()

    checkpoint_sha256_cache = None
    profile_cache = {}

    def checkpoint_sha256():
        nonlocal checkpoint_sha256_cache
        if checkpoint_sha256_cache is None:
            import hashlib
            h = hashlib.sha256()
            with checkpoint.open("rb") as f:
                while True:
                    b = f.read(1024 * 1024)
                    if not b:
                        break
                    h.update(b)
            checkpoint_sha256_cache = h.hexdigest()
        return checkpoint_sha256_cache

    def load_style_profile(value: str):
        path = Path(str(value or "")).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Style profile not found: {path}")
        st = path.stat()
        key = (str(path), int(st.st_size), int(st.st_mtime_ns))
        if key in profile_cache:
            return profile_cache[key]
        profile = torch.load(path, map_location="cpu", weights_only=False)
        if profile.get("schema") != PROFILE_SCHEMA:
            raise RuntimeError(f"Unsupported style profile schema: {profile.get('schema')!r}")
        expected = str(profile.get("checkpoint_sha256") or "")
        if expected and expected != checkpoint_sha256():
            raise RuntimeError("Style profile was extracted from a different VITS checkpoint. Rebuild the profile for this checkpoint.")
        styles = profile.get("styles")
        if not isinstance(styles, dict) or not styles:
            raise RuntimeError("Style profile contains no styles")
        profile_cache.clear()
        profile_cache[key] = profile
        return profile

    tokenizer, finbert, finbert_device = load_finbert_model(
        "TurkuNLP/bert-large-finnish-cased-v1", device=str(device)
    )

    def expand_prior(attn, m_p, logs_p):
        m = torch.matmul(attn.transpose(1, 2), m_p.transpose(1, 2)).transpose(1, 2)
        logs = torch.matmul(attn.transpose(1, 2), logs_p.transpose(1, 2)).transpose(1, 2)
        return m, logs

    def predicted_path(h, x_mask, token_lengths, length_scale: float, max_frames: int = 2500):
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
        return attn, y_lengths

    @torch.inference_mode()
    def infer(text: str, output: Path, noise_scale: float, length_scale: float, seed: int, *, style_profile: str = "", style_name: str = "neutral", style_strength: float = 0.0, apply_style_gain: bool = True):
        feat = extract_features(
            text,
            tokenizer=tokenizer,
            model=finbert,
            device=finbert_device,
            model_id="TurkuNLP/bert-large-finnish-cased-v1",
        )
        if feat.warnings:
            # Warnings are returned to the UI but do not silently change the text path.
            warning_text = "; ".join(str(x) for x in feat.warnings)
        else:
            warning_text = ""
        ids = []
        missing = []
        for g in feat.frontend_tokens:
            if g not in vocab:
                missing.append(g)
            else:
                ids.append(int(vocab[g]))
        if missing:
            raise ValueError(f"Checkpoint vocab is missing grapheme token(s): {sorted(set(missing))}")
        features = np.asarray(feat.features, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != 1024 or features.shape[0] != len(ids):
            raise ValueError(f"Invalid FinBERT feature shape: {features.shape}; tokens={len(ids)}")

        tok = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        bert = torch.from_numpy(features).to(device=device, dtype=torch.float32).unsqueeze(0)
        tl = torch.tensor([len(ids)], dtype=torch.long, device=device)
        # Step8 checkpoints keep their internal categorical style path neutral here.
        # The paired-language-independent profile is injected later in z_p space.
        if schema == STEP8_SCHEMA:
            sid0 = torch.zeros(1, dtype=torch.long, device=device)
            h, mp, lp, xm = model.text(tok, bert, tl, style_id=sid0, style_strength=0.0)
        else:
            h, mp, lp, xm = model.text(tok, bert, tl)

        strength = max(0.0, min(2.0, float(style_strength)))
        latent_delta = None
        duration_ratio = 1.0
        gain_ratio = 1.0
        resolved_style = "neutral"
        if style_profile and strength > 0.0 and str(style_name or "neutral") != "neutral":
            profile = load_style_profile(style_profile)
            resolved_style = str(style_name).strip()
            entry = profile["styles"].get(resolved_style)
            if entry is None:
                raise ValueError(f"Style {resolved_style!r} is not in the selected profile")
            latent_delta = entry.get("latent_delta")
            if not torch.is_tensor(latent_delta) or latent_delta.numel() != 192:
                raise RuntimeError(f"Invalid latent_delta for style {resolved_style!r}")
            latent_delta = latent_delta.to(device=device, dtype=mp.dtype).reshape(1, -1, 1)
            duration_ratio = max(0.65, min(1.6, float(entry.get("duration_ratio", 1.0))))
            gain_ratio = max(0.55, min(1.8, float(entry.get("gain_ratio", 1.0))))

        effective_length_scale = max(0.2, min(3.0, float(length_scale)))
        if strength > 0 and duration_ratio > 0:
            effective_length_scale *= math.exp(math.log(duration_ratio) * strength)
            effective_length_scale = max(0.2, min(3.0, effective_length_scale))
        attn, yl = predicted_path(h, xm, tl, length_scale=effective_length_scale)
        ym = (
            torch.arange(attn.size(-1), device=device).unsqueeze(0) < yl.unsqueeze(1)
        ).unsqueeze(1).to(mp.dtype)
        mpe, lpe = expand_prior(attn, mp, lp)
        if latent_delta is not None:
            mpe = mpe + latent_delta * strength
        ns = max(0.0, min(1.5, float(noise_scale)))
        if ns <= 0:
            zpe = mpe
        else:
            gen = torch.Generator(device=device)
            gen.manual_seed(int(seed))
            eps = torch.randn(mpe.shape, generator=gen, device=device, dtype=mpe.dtype)
            zpe = mpe + eps * torch.exp(lpe) * ns
        ze = model.flow(zpe * ym, ym, reverse=True) * ym
        wav = model.generator(ze)[0, 0].float().cpu().numpy()
        if latent_delta is not None and apply_style_gain:
            wav = wav * float(gain_ratio ** strength)
        peak = float(np.max(np.abs(wav))) if wav.size else 0.0
        if peak > 0.999:
            wav = wav * (0.999 / peak)
        output.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output), wav.astype(np.float32), 24000, subtype="PCM_16")
        return {
            "frames": int(yl[0].item()),
            "seconds": float(len(wav) / 24000.0),
            "normalized_text": feat.canonical_text,
            "warning": warning_text,
            "style": resolved_style,
            "style_strength": strength,
            "effective_length_scale": effective_length_scale,
        }

    emit(READY, {"ok": True, "checkpoint": str(checkpoint), "device": str(device), "schema": schema, "styles": list(checkpoint_styles)})
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        req_id = None
        try:
            req = json.loads(raw)
            req_id = req.get("id")
            text = str(req.get("text", "")).strip()
            if not text:
                raise ValueError("Finnish text is empty.")
            output = Path(str(req.get("output", ""))).resolve()
            noise_scale = float(req.get("noise_scale", 0.4))
            length_scale = float(req.get("length_scale", 1.0))
            seed = int(req.get("seed", 0))
            style_profile = str(req.get("style_profile", "") or "")
            style_name = str(req.get("style", "neutral") or "neutral")
            style_strength = float(req.get("style_strength", 0.0) or 0.0)
            apply_style_gain = bool(req.get("apply_style_gain", True))
            if seed < 0 or seed > 2147483647:
                raise ValueError("Seed must be between 0 and 2147483647.")
            meta = infer(text, output, noise_scale, length_scale, seed, style_profile=style_profile, style_name=style_name, style_strength=style_strength, apply_style_gain=apply_style_gain)
            emit(RESULT, {"ok": True, "id": req_id, "output": str(output), **meta})
        except Exception as exc:
            emit(RESULT, {
                "ok": False,
                "id": req_id,
                "error": str(exc),
                "trace": traceback.format_exc(limit=5),
            })
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        emit(READY, {"ok": False, "error": str(exc), "trace": traceback.format_exc(limit=8)})
        raise
