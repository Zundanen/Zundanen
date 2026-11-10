from __future__ import annotations

"""Frame-level F0 (RMVPE, 10 ms) for every training WAV, saved per file.

Run with the RVC Python:
  runtime/rvc/.venv/Scripts/python.exe -I emotion_tts/f0_frames.py <list.txt> <out_dir> [--shard i/n]

Writes <out_dir>/<sha1(path)>.npy (float32 Hz, 0 = unvoiced).  Resumable;
--shard lets two processes split the list.
"""

import hashlib
import os
import sys
from pathlib import Path

import numpy as np


def key(path: str) -> str:
    return hashlib.sha1(path.encode("utf-8")).hexdigest()


def main() -> int:
    lst, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    shard = (0, 1)
    if "--shard" in sys.argv:
        a, b = sys.argv[sys.argv.index("--shard") + 1].split("/")
        shard = (int(a), int(b))
    out.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1] / "runtime/rvc/RVC-WebUI"
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    import librosa
    import torch
    from infer.rmvpe import RMVPE
    model = RMVPE(str(repo / "assets/rmvpe/rmvpe.pt"), is_half=False, device="cuda" if torch.cuda.is_available() else "cpu")
    paths = [x.strip() for x in lst.read_text(encoding="utf-8").splitlines() if x.strip()]
    paths = paths[shard[0]::shard[1]]
    done = 0
    for i, p in enumerate(paths):
        dst = out / (key(p) + ".npy")
        if dst.exists():
            continue
        y, _ = librosa.load(p, sr=16000, mono=True)
        f0 = model.infer_from_audio(y.astype(np.float32), thred=0.03).astype(np.float32)
        np.save(dst, f0)
        done += 1
        if done % 500 == 0:
            print(f"[F0F] shard {shard[0]}/{shard[1]}: {i + 1}/{len(paths)}", flush=True)
    print(f"[PASS] shard {shard[0]}/{shard[1]} new={done}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
