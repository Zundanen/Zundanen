from __future__ import annotations

"""Resident worker for the v65 Finnish emotion TTS (Normal / Ama / Sexy / Tsun + user styles).

Started by emotion_engine.py.  Loads FinBERT once and the character models on
demand, then answers one JSON request per stdin line:

  {"id": 1, "text": "...", "output": "C:/.../x.wav", "character": "Zundamon", "style": "ama", "strength": 1.5,
   "seed": 123, "noise_scale": 0.4, "length_scale": 1.0, "level": -6.66, "range_k": 1.0, "z": 0.5}

Every field except id/text/output is optional (preset defaults).
v2.0.1: a character is one trained model (preset["characters"][name]: speaker,
base_name, branch_name, level_st, range_st).  Models are loaded when first used
and at most MAX_MODELS stay in VRAM.  User styles (preset["extra_styles"],
Style fine-tuning) are offered for the characters whose base they were trained on.
Original Zundanen code (MIT).
"""

import json
import sys
import traceback
from collections import OrderedDict
from pathlib import Path

READY = "ZUNDANEN_V65_READY\t"
RESULT = "ZUNDANEN_V65_RESULT\t"
ROOT = Path(__file__).resolve().parent
STYLES = ("normal", "ama", "tsun", "sexy")
MAX_MODELS = 2


def _timing(pre, style, a, req=None, ch=None):
    """v67: Normal rhythm x style tempo (the character's own style_tempo if it has one, else the preset's)."""
    if pre.get("duration_mode") != "global_tempo":
        return {}
    detail = float((req or {}).get("dur_detail", pre.get("dur_detail", 0.0)))
    tempo = (ch or {}).get("style_tempo") or pre["style_tempo"]
    return {"tempo": float(tempo.get(style, 1.0)) ** a, "dur_detail": detail}


def emit(prefix, payload):
    print(prefix + json.dumps(payload, ensure_ascii=False), flush=True)


def main() -> int:
    for s in (sys.stdin, sys.stdout):
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    sys.path.insert(0, str(ROOT))
    for p in ("emotion_tts", "emotion_tts", "emotion_tts", "emotion_tts", "emotion_tts"):
        sys.path.insert(0, str(ROOT / p))
    import numpy as np
    import torch
    import emotion_engine as E
    import emotion_style as S
    from emo_vits import import_research
    from emo_train_base import write_wav
    from style_branch import StyleBranch
    from pitch_model import V62Combo, pitch_vector

    pre = E.preset()
    chars = E.characters(pre)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache: OrderedDict = OrderedDict()

    def load_model(ch):
        base_name, branch_name = ch["base_name"], ch["branch_name"]
        key = (base_name, branch_name, ch.get("bundle"))
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        while len(cache) >= MAX_MODELS:
            cache.popitem(last=False)
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        if ch.get("bundle"):                # one character file: base + style branch + z-delta
            bundle = torch.load(ch["bundle"], map_location="cpu", weights_only=False)
            ck, stats, vocab, _ja, _seen, model = S.load_all(dev, bundle=bundle)
            zd = bundle["zdelta"]
        else:
            S.BASE_NAME, S.BRANCH_NAME = base_name, branch_name
            ck, stats, vocab, _ja, _seen, model = S.load_all(dev)
            zd = torch.load(S.OUT / f"zdelta_{base_name}.pt", map_location="cpu", weights_only=False)
        model.eval()
        extra = {}
        for name, e in (pre.get("extra_styles") or {}).items():
            if e.get("base_name") != base_name:
                continue                   # trained on another base: would not fit this one
            br = StyleBranch(model.branch.sym_emb.num_embeddings - 1, 2, len(ck["speakers"]))
            br.load_state_dict(torch.load(ROOT / e["branch"], map_location="cpu", weights_only=False)["branch"])
            m = V62Combo(model.base, br, model.sym_table.cpu(), model.sym_mask.cpu(), ["normal", name],
                         ck["speakers"], ck["langs"]).to(dev).eval()
            extra[name] = (m, torch.load(ROOT / e["zdelta"], map_location="cpu", weights_only=False), e)
        cache[key] = {"ck": ck, "stats": stats, "vocab": vocab, "model": model, "zd": zd, "extra": extra}
        return cache[key]

    default = next(iter(chars))
    load_model(chars[default])
    import_research(ROOT)
    from finbert_features import extract_features, load_finbert_model
    tokz, fb, fdev = load_finbert_model("TurkuNLP/bert-large-finnish-cased-v1", device=str(dev))
    emit(READY, {"ok": True, "device": str(dev), "characters": list(chars), "preset": pre})

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        rid = None
        try:
            req = json.loads(raw)
            rid = req.get("id")
            text = str(req.get("text", "")).strip()
            if not text:
                raise ValueError("Finnish text is empty.")
            character = E.find_character(str(req.get("character", default)), pre)
            if character is None:
                raise ValueError(f"Unknown character {req.get('character')!r}; use one of {list(chars)}.")
            ch = chars[character]
            M = load_model(ch)
            model, zd, extra, stats, vocab = M["model"], M["zd"], M["extra"], M["stats"], M["vocab"]
            if ch["speaker"] not in M["ck"]["speakers"]:
                raise ValueError(f"Speaker {ch['speaker']!r} is not in this model.")
            rc = stats["range_center_st"]
            style = str(req.get("style", "normal")).lower()
            if style not in STYLES and style not in extra:
                raise ValueError(f"Style {style!r} is not available for {character}; use one of {list(STYLES) + list(extra)}.")
            default_a = extra[style][2].get("strength", 1.0) if style in extra else pre["style_strength"][style]
            a = 0.0 if style == "normal" else float(req.get("strength", default_a))
            a = max(0.0, min(2.0, a))
            level0 = float(req.get("level", ch["level_st"]))
            k0 = max(0.3, min(3.0, float(req.get("range_k", pre["range_k"]))))
            zs = float(req.get("z", pre["z_strength"]))
            feat = extract_features(text, tokenizer=tokz, model=fb, device=fdev, model_id="TurkuNLP/bert-large-finnish-cased-v1")
            missing = sorted({g for g in feat.frontend_tokens if g not in vocab})
            if missing:
                raise ValueError(f"Unsupported characters: {missing}")
            tok = torch.tensor([vocab[g] for g in feat.frontend_tokens])
            bert = torch.from_numpy(np.asarray(feat.features, dtype=np.float32))
            if style in extra:
                m_use, zd_use, e = extra[style]
                level = level0 + a * float(e.get("level_offset_st", 0.0))
                ratio = float(e.get("range_ratio", 1.0)) ** a
                timing = {"tempo": float(e.get("tempo", 1.0)) ** a, "dur_detail": 0.0}
            else:
                presets = stats["presets_from_real_recordings"]
                zp = presets.get(ch["speaker"]) or presets["zundamon"]
                s, n = zp.get(style, zp["normal"]), zp["normal"]
                m_use, zd_use = model, zd
                level = level0 + a * (s["level_st"] - n["level_st"])
                ratio = (s["range_st"] / n["range_st"]) ** a
                timing = _timing(pre, style, a, req, ch)
            y = S.synth(m_use, zd_use, tok, bert, style=style, strength=a, z=(zs if style != "normal" else 0.0),
                        level_st=level, range_k=k0 * ratio,
                        pitch=pitch_vector(level, ch["range_st"] * ratio, rc), speaker=ch["speaker"],
                        noise=max(0.0, min(1.5, float(req.get("noise_scale", pre["noise_scale"])))),
                        length_scale=max(0.5, min(2.0, float(req.get("length_scale", 1.0)))),
                        seed=int(req.get("seed", 1234)), device=dev, **timing)
            out = Path(str(req["output"])).resolve()
            write_wav(out, y)
            emit(RESULT, {"ok": True, "id": rid, "output": str(out), "style": style, "character": character,
                          "strength": a, "level_st": round(level, 3), "range_k": round(k0 * ratio, 3),
                          "seconds": round(len(y) / 24000, 3), "normalized_text": feat.canonical_text,
                          "warning": "; ".join(str(w) for w in (feat.warnings or []))})
        except Exception as exc:
            emit(RESULT, {"ok": False, "id": rid, "error": str(exc), "trace": traceback.format_exc(limit=5)})
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        emit(READY, {"ok": False, "error": str(exc), "trace": traceback.format_exc(limit=8)})
        raise
