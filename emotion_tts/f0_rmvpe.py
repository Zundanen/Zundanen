from __future__ import annotations

"""v62 F0 statistics with RMVPE (the estimator RVC itself uses; robust to the
octave errors that pYIN showed on high voices).

Run with the RVC Python, cwd = RVC repo:
  runtime/rvc/.venv/Scripts/python.exe -I emotion_tts/f0_rmvpe.py <list.txt> <out.csv>
list.txt: one WAV path per line.  Resumable: rows already in out.csv are skipped.
Output columns: wav_path, f0_median_hz, st_p10, st_p90, voiced_ratio
(st_* are semitones relative to the utterance's own median).
"""

import csv
import os
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    lst, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    repo = Path(os.environ.get("RVC_REPO", Path(__file__).resolve().parents[1] / "runtime/rvc/RVC-WebUI"))
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    import librosa
    import torch
    from infer.rmvpe import RMVPE
    model = RMVPE(str(repo / "assets/rmvpe/rmvpe.pt"), is_half=False, device="cuda" if torch.cuda.is_available() else "cpu")
    paths = [x.strip() for x in lst.read_text(encoding="utf-8").splitlines() if x.strip()]
    done = set()
    if out.exists():
        with out.open(encoding="utf-8-sig") as f:
            done = {r["wav_path"] for r in csv.DictReader(f)}
    new = not out.exists()
    with out.open("a", encoding="utf-8-sig" if new else "utf-8", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["wav_path", "f0_median_hz", "st_p10", "st_p90", "voiced_ratio"])
        for i, p in enumerate(paths):
            if p in done:
                continue
            y, _ = librosa.load(p, sr=16000, mono=True)
            f0 = model.infer_from_audio(y.astype(np.float32), thred=0.03)
            v = f0[f0 > 40]
            if v.size < 10:
                w.writerow([p, "", "", "", f"{v.size / max(1, f0.size):.3f}"])
                continue
            med = float(np.median(v))
            st = 12 * np.log2(v / med)
            w.writerow([p, f"{med:.2f}", f"{np.percentile(st, 10):.3f}", f"{np.percentile(st, 90):.3f}",
                        f"{v.size / f0.size:.3f}"])
            if i % 500 == 0:
                f.flush()
                print(f"[F0] {i + 1}/{len(paths)}", flush=True)
    print(f"[PASS] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
