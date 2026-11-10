from __future__ import annotations

import argparse
import json
import random
import sys
import time
import traceback
from pathlib import Path

READY_PREFIX = "ZUNDANEN_TTS_READY\t"
RESULT_PREFIX = "ZUNDANEN_TTS_RESULT\t"


def emit(prefix: str, payload: dict) -> None:
    print(prefix + json.dumps(payload, ensure_ascii=False), flush=True)


def set_seed(seed, np, torch) -> None:
    if seed is None:
        return
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--reference", required=True)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))

    try:
        import numpy as np
        import torch
        import soundfile as sf
        from safetensors.torch import load_file
        from src.chatterbox_.tts import ChatterboxTTS

        device = "cuda" if torch.cuda.is_available() else "cpu"
        engine = ChatterboxTTS.from_local(str(repo / "pretrained_models"), device=device)

        checkpoint = load_file(args.weights)
        state = {key[3:] if key.startswith("t3.") else key: value for key, value in checkpoint.items()}
        engine.t3.load_state_dict(state, strict=False)

        # Chatterbox supports caching its prompt conditioning. Prepare it once
        # and let generate() only update the scalar exaggeration when needed.
        cached_conditioning = hasattr(engine, "prepare_conditionals")
        if cached_conditioning:
            engine.prepare_conditionals(args.reference, exaggeration=0.5)

        emit(READY_PREFIX, {
            "ok": True,
            "device": device,
            "sample_rate": int(engine.sr),
            "reference_conditioning_cached": bool(cached_conditioning),
        })
    except Exception as exc:
        emit(READY_PREFIX, {"ok": False, "error": str(exc), "trace": traceback.format_exc(limit=8)})
        return 1

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        request_id = None
        try:
            req = json.loads(raw)
            request_id = req.get("id")
            text = str(req.get("text", "")).strip()
            if not text:
                raise ValueError("Finnish text is empty.")
            output = Path(str(req["output"]))
            output.parent.mkdir(parents=True, exist_ok=True)
            set_seed(req.get("seed"), np, torch)

            t0 = time.perf_counter()
            kwargs = dict(
                text=text,
                repetition_penalty=float(req.get("repetition_penalty", 1.2)),
                temperature=float(req.get("temperature", 0.8)),
                exaggeration=float(req.get("exaggeration", 0.6)),
            )
            if cached_conditioning:
                # Local in-memory conditioning prepared once at startup.
                wav = engine.generate(audio_prompt_path=None, **kwargs)
            else:
                # Compatibility fallback for an older local Chatterbox fork.
                wav = engine.generate(audio_prompt_path=args.reference, **kwargs)

            audio = wav.squeeze().cpu().numpy()
            sf.write(output, audio, engine.sr)
            elapsed = time.perf_counter() - t0
            audio_seconds = (len(audio) / float(engine.sr)) if int(engine.sr) > 0 else 0.0
            emit(RESULT_PREFIX, {
                "id": request_id,
                "ok": True,
                "output": str(output),
                "elapsed_seconds": round(elapsed, 4),
                "audio_seconds": round(audio_seconds, 4),
                "rtf": round((elapsed / audio_seconds), 4) if audio_seconds > 0 else 0.0,
                "reference_conditioning_cached": bool(cached_conditioning),
            })
        except Exception as exc:
            emit(RESULT_PREFIX, {
                "id": request_id,
                "ok": False,
                "error": str(exc),
                "trace": traceback.format_exc(limit=8),
            })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
