from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--text-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--exaggeration", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))

    import random
    import numpy as np
    import torch
    import soundfile as sf
    from safetensors.torch import load_file
    from src.chatterbox_.tts import ChatterboxTTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = ChatterboxTTS.from_local(
        str(repo / "pretrained_models"),
        device=device,
    )

    checkpoint = load_file(args.weights)
    state = {
        key[3:] if key.startswith("t3.") else key: value
        for key, value in checkpoint.items()
    }
    engine.t3.load_state_dict(state, strict=False)

    text = Path(args.text_file).read_text(encoding="utf-8").strip()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed % (2**32))
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    wav = engine.generate(
        text=text,
        audio_prompt_path=args.reference,
        repetition_penalty=args.repetition_penalty,
        temperature=args.temperature,
        exaggeration=args.exaggeration,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        args.output,
        wav.squeeze().cpu().numpy(),
        engine.sr,
    )


if __name__ == "__main__":
    main()
