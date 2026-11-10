from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import torchaudio

from vits_lite import VitsLiteConfig

SPECIALS = ["<PAD>"]

CACHE_SCHEMA = "zundanen-finbert-features-v1"

def load_feature_cache(path: Path) -> dict:
    with np.load(Path(path), allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        if metadata.get("schema") != CACHE_SCHEMA:
            raise ValueError(f"unsupported cache schema: {metadata.get('schema')!r}")
        return {
            "features": data["features"].copy(),
            "grapheme_tokens": tuple(str(x) for x in data["grapheme_tokens"].tolist()),
            "metadata": metadata,
        }



def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def build_vocab(rows: list[dict[str, str]]) -> dict[str, int]:
    chars: set[str] = set()
    for i, row in enumerate(rows, 1):
        cache = load_feature_cache(Path(row["bert_cache"]))
        chars.update(cache["grapheme_tokens"])
        if i % 500 == 0:
            print(f"  vocab scan {i}/{len(rows)}")
    return {t: i for i, t in enumerate(SPECIALS + sorted(chars))}


def choose_eval_rows(rows: list[dict[str, str]], count: int = 5) -> list[dict[str, str]]:
    clean = []
    for r in rows:
        try:
            d = float(r.get("duration_sec") or 0)
        except ValueError:
            continue
        if r.get("fast_status") == "PASS" and r.get("aalto_status") == "PASS" and 2.5 <= d <= 6.5:
            clean.append(r)
    if len(clean) < count:
        clean = rows[:]
    clean = sorted(clean, key=lambda r: float(r.get("duration_sec") or 0))
    if len(clean) <= count:
        return clean
    idxs = np.linspace(0, len(clean) - 1, count).round().astype(int).tolist()
    return [clean[i] for i in idxs]


@dataclass
class Sample:
    stem: str
    text: str
    tokens: torch.Tensor
    bert: torch.Tensor
    wav: torch.Tensor
    spec: torch.Tensor


class FullDataset(torch.utils.data.Dataset):
    def __init__(self, manifest: Path, vocab: dict[str, int], cfg: VitsLiteConfig):
        self.rows = read_csv(manifest)
        self.vocab = vocab
        self.cfg = cfg
        self.window = torch.hann_window(cfg.win_length)

    def __len__(self):
        return len(self.rows)

    def _wav(self, path: Path) -> torch.Tensor:
        arr, sr = sf.read(str(path), dtype="float32", always_2d=True)
        wav = torch.from_numpy(arr.T.copy())
        if wav.size(0) > 1:
            wav = wav.mean(0, keepdim=True)
        if sr != self.cfg.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.cfg.sample_rate)
        return torch.clamp(wav, -1.0, 1.0)

    def __getitem__(self, idx: int) -> Sample:
        r = self.rows[idx]
        cache = load_feature_cache(Path(r["bert_cache"]))
        gs = cache["grapheme_tokens"]
        ids = [self.vocab.get(g) for g in gs]
        if any(v is None for v in ids):
            raise ValueError(f"unknown token in {r['stem']}: {gs}")
        bert = torch.from_numpy(np.asarray(cache["features"], dtype=np.float32))
        if bert.shape[0] != len(ids):
            raise ValueError(f"BERT/frontend length mismatch for {r['stem']}: {bert.shape[0]} vs {len(ids)}")
        wav = self._wav(Path(r["wav_path"]))
        spec = torch.abs(torch.stft(
            wav.squeeze(0), n_fft=self.cfg.n_fft, hop_length=self.cfg.hop_length,
            win_length=self.cfg.win_length, window=self.window, center=False,
            return_complex=True,
        ))
        return Sample(
            stem=r["stem"], text=r["text"],
            tokens=torch.tensor(ids, dtype=torch.long), bert=bert,
            wav=wav, spec=spec,
        )


def collate_batch(batch: list[Sample]) -> dict:
    b = len(batch)
    tx = max(s.tokens.numel() for s in batch)
    ty = max(s.spec.shape[-1] for s in batch)
    tw = max(s.wav.shape[-1] for s in batch)
    bert_dim = batch[0].bert.shape[1]
    spec_ch = batch[0].spec.shape[0]
    tokens = torch.zeros(b, tx, dtype=torch.long)
    bert = torch.zeros(b, tx, bert_dim, dtype=torch.float32)
    wav = torch.zeros(b, 1, tw, dtype=torch.float32)
    spec = torch.zeros(b, spec_ch, ty, dtype=torch.float32)
    tl = torch.zeros(b, dtype=torch.long)
    sl = torch.zeros(b, dtype=torch.long)
    for i, s in enumerate(batch):
        n = s.tokens.numel(); m = s.spec.shape[-1]; w = s.wav.shape[-1]
        tokens[i, :n] = s.tokens
        bert[i, :n] = s.bert
        wav[i, 0, :w] = s.wav[0]
        spec[i, :, :m] = s.spec
        tl[i] = n; sl[i] = m
    return {
        "stems": [s.stem for s in batch],
        "texts": [s.text for s in batch],
        "tokens": tokens, "bert": bert, "token_lengths": tl,
        "wav": wav, "spec": spec, "spec_lengths": sl,
    }
