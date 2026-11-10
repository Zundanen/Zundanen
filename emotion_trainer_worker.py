from __future__ import annotations

"""Zundanen v2 VITS trainer (Model Trainer -> VITS).  Original Zundanen code (MIT).

  joint   : retrain the emotion model on the prepared joint data
            (Finnish Chatterbox + RVC bridge + Japanese emotion corpus),
            bridge share adjustable.  base -> style branch -> z-delta.
  finnish : add a user Finnish dataset (WAV + same-stem TXT) for one character
            and continue training (existing data is mixed in so Japanese/styles
            are not forgotten).  prepare (FinBERT) -> F0 -> base -> branch -> z-delta.
  style   : add a NEW style from paired recordings (Normal + style, WAV + LAB,
            matched by the number in the file name).  A separate style branch,
            z-delta, tempo and pitch offsets are learned and the style is
            registered in emotion_tts/emotion_preset.json (existing styles untouched).

Progress protocol for app.py:  "[V2PHASE] label", "[V2PROGRESS] pct",
"RESULT_JSON : {...}", "TRAINING COMPLETE".  Nothing outside outputs/ is written
except the style registration in emotion_preset.json (backed up first).
"""

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TTS_PY = Path(sys.executable)
PRESET = ROOT / "emotion_tts/emotion_preset.json"
V2 = ROOT / "outputs/v2"
for p in ("emotion_tts", "emotion_tts", "emotion_tts", "emotion_tts", "emotion_tts"):
    sys.path.insert(0, str(ROOT / p))


def say(msg):
    print(msg, flush=True)


def phase(label, pct):
    say(f"[V2PHASE] {label}")
    say(f"[V2PROGRESS] {pct:.1f}")


def rvc_python() -> str:
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    return cfg["rvc_python"]


def run_stage(cmd, lo, hi, label):
    """Run a child process, stream its log and map 'step a/b' to [lo, hi] % (lo=None: no progress)."""
    if lo is None:
        say(f"[V2PHASE] {label}")
    else:
        phase(label, lo)
    proc = subprocess.Popen([str(c) for c in cmd], cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", bufsize=1)
    for line in proc.stdout:
        line = line.rstrip()
        if not line or "WeightNorm" in line or "FutureWarning" in line:
            continue
        m = re.search(r"\bstep\s+(\d+)\s*/\s*(\d+)", line)
        if m and lo is not None:
            say(f"[V2PROGRESS] {lo + (hi - lo) * min(1.0, int(m.group(1)) / max(1, int(m.group(2)))):.1f}")
        say("  " + line)
    if proc.wait() != 0:
        raise SystemExit(f"[FAIL] {label} failed (exit {proc.returncode})")
    if hi is not None:
        say(f"[V2PROGRESS] {hi:.1f}")


def move_aside(path: Path, mode: str, run: str):
    """A folder left by an unfinished run (no result file) is renamed, never deleted."""
    if not path.exists():
        return
    if (V2 / "results" / f"{mode}_{run}.json").exists():
        raise SystemExit(f"[FAIL] {path} belongs to a finished run; choose another name (nothing is overwritten)")
    dst = path.with_name(f"{path.name}_unfinished_{time.strftime('%Y%m%d_%H%M%S')}")
    path.rename(dst)
    say(f"[V2] previous unfinished run moved to {dst}")


def preset():
    return json.loads(PRESET.read_text(encoding="utf-8"))


def chars():
    import emotion_engine
    return emotion_engine.characters(preset())


def model_entry(name):
    import emotion_engine
    key = emotion_engine.find_character(name, preset())
    if key is None:
        raise SystemExit(f"[FAIL] unknown model {name!r}; available: {list(chars())}")
    return key, chars()[key]


def register_character(name, entry):
    """Add the trained model to Voice Generation (preset backed up first)."""
    backup = V2 / "preset_backups" / f"v65_preset_{time.strftime('%Y%m%d_%H%M%S')}.json"
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PRESET, backup)
    pre = preset()
    pre.setdefault("characters", {})[name] = entry
    PRESET.write_text(json.dumps(pre, ensure_ascii=False, indent=2), encoding="utf-8")
    say(f"[V2] registered character '{name}' in Voice Generation (preset backup: {backup})")


def latest_base(base_name):
    cps = sorted((ROOT / "outputs/v65" / base_name / "checkpoints").glob("v65_*.pt"))
    if not cps:
        raise SystemExit(f"[FAIL] no checkpoint in outputs/v65/{base_name}")
    return cps[-1]


def mix_arg(bridge_ratio, extra_ratio=0.0):
    rest = max(0.0, 1.0 - bridge_ratio - extra_ratio)
    parts = {"fi": rest * 0.35 / 0.75, "bridge": bridge_ratio, "ja": rest * 0.40 / 0.75, "extra": extra_ratio}
    return ",".join(f"{k}={v:.4f}" for k, v in parts.items() if v > 0)


DATA = ROOT / "outputs/v63/v63_main/data"


def branch_batch(base_batch):
    """The style branch is lighter than the base model, so it takes a larger batch (10 -> 16)."""
    return max(2, min(32, int(round(1.6 * base_batch))))


def epoch_steps(epochs, rows, batch, floor=200):
    """Steps that make `epochs` passes over `rows` training sentences."""
    return max(floor, int(round(float(epochs) * rows / batch)))


def base_rows(extra_rows=0):
    """Training sentences of the joint data (Finnish + RVC bridge + Japanese emotion) plus new data."""
    from emo_common import read_csv
    n = len(read_csv(DATA / "fi_train_pitch.csv")) + extra_rows
    for f in sorted(list(DATA.glob("bridge_p*_pitch.csv")) + list(DATA.glob("wide_*_pitch.csv"))):
        n += len(read_csv(f))
    return n + sum(1 for r in read_csv(DATA / "ja_pitch.csv") if r["split"] == "train")


def branch_rows():
    """Japanese Ama / Sexy / Tsun training sentences the style branch learns from."""
    from emo_common import read_csv
    return sum(1 for r in read_csv(DATA / "ja_pitch.csv") if r["split"] == "train" and r["style"] != "normal")


def base_branch_zdelta(run, args, extra_csv=None, extra_ratio=0.0, lo=10.0, extra_rows=0):
    base_name, branch_name = f"v2_{run}_base", f"v2_{run}_branch"
    for d in (ROOT / "outputs/v65" / base_name, ROOT / "outputs/v65" / branch_name):
        move_aside(d, args.mode, run)
    init = Path(args.init_checkpoint) if args.init_checkpoint else latest_base(model_entry(args.base_model)[1]["base_name"])
    say(f"[V2] start from {init}")
    n_base, n_branch = base_rows(extra_rows), branch_rows()
    bb = branch_batch(args.batch_size)
    steps = args.steps or epoch_steps(args.epochs, n_base, args.batch_size)
    branch_steps = args.branch_steps or epoch_steps(args.branch_epochs, n_branch, bb)
    say(f"[V2] base: {args.epochs:g} epochs of {n_base} sentences = {steps} steps; "
        f"style branch: {args.branch_epochs:g} epochs of {n_branch} sentences = {branch_steps} steps")
    cmd = [TTS_PY, "-u", "-I", ROOT / "emotion_tts/emotion_train.py", "--flow-cond", "--out-name", base_name,
           "--init-checkpoint", init, "--steps", steps, "--save-every", max(500, steps // 4),
           "--batch-size", args.batch_size, "--workers", args.workers, "--lr", args.lr, "--mix", mix_arg(args.bridge_ratio, extra_ratio)]
    if extra_csv:
        cmd += ["--extra-csv", extra_csv]
    run_stage(cmd, lo, 80.0, "Training the base model")
    run_stage([TTS_PY, "-u", "-I", ROOT / "emotion_tts/emotion_style.py", "branch", "--base-name", base_name,
               "--branch-name", branch_name, "--steps", branch_steps, "--batch-size", bb, "--workers", args.workers], 80.0, 95.0, "Training the style branch")
    run_stage([TTS_PY, "-u", "-I", ROOT / "emotion_tts/emotion_style.py", "zdelta", "--base-name", base_name,
               "--branch-name", branch_name], 95.0, 99.0, "Building voice-quality deltas")
    return {"base_name": base_name, "branch_name": branch_name}


def f0_for(paths, tag):
    """Frame F0 (for training) + utterance stats (for pitch labels) via RMVPE."""
    work = V2 / "f0" / tag
    work.mkdir(parents=True, exist_ok=True)
    lst = work / "list.txt"
    lst.write_text("\n".join(paths), encoding="utf-8")
    import hashlib
    for wp in paths:   # the frame-F0 cache is keyed by path: drop entries older than the WAV (re-recorded data)
        npy = ROOT / "outputs/v65/f0frames" / (hashlib.sha1(wp.encode("utf-8")).hexdigest() + ".npy")
        if npy.exists() and npy.stat().st_mtime < Path(wp).stat().st_mtime:
            npy.unlink()
    run_stage([rvc_python(), "-I", ROOT / "emotion_tts/f0_frames.py", lst, ROOT / "outputs/v65/f0frames"], None, None, "Measuring F0 (frames)")
    stats_csv = work / "f0_stats.csv"
    run_stage([rvc_python(), "-I", ROOT / "emotion_tts/f0_rmvpe.py", lst, stats_csv], None, None, "Measuring F0 (utterances)")
    return {r["wav_path"]: r for r in csv.DictReader(open(stats_csv, encoding="utf-8-sig"))}


def pitch_fields(stat, ref_hz):
    if stat and stat.get("f0_median_hz") and float(stat.get("voiced_ratio") or 0) >= 0.15:
        return {"pitch_level_st": f"{12 * math.log2(float(stat['f0_median_hz']) / ref_hz):.3f}",
                "pitch_range_st": f"{float(stat['st_p90']) - float(stat['st_p10']):.3f}", "pitch_ok": "1"}
    return {"pitch_level_st": "0", "pitch_range_st": "", "pitch_ok": "0"}


def ref_hz(character):
    stats = json.loads((ROOT / "outputs/v63/v63_main/data/pitch_stats.json").read_text(encoding="utf-8"))
    return stats["reference_hz"][character]


# ----------------------------------------------------------------------------- modes
def cmd_joint(args):
    phase("Joint training", 2.0)
    say(f"[V2] bridge share {args.bridge_ratio:.2f} -> mix {mix_arg(args.bridge_ratio)}")
    src_key, src = model_entry(args.base_model)
    res = base_branch_zdelta(args.run_name, args, lo=5.0)
    register_character(args.run_name, {"label": args.run_name, "speaker": src["speaker"], **res,
                                       "level_st": src["level_st"], "range_st": src["range_st"]})
    return {"mode": "joint", "character": args.run_name, "base_model": src_key, "speaker": src["speaker"], **res}


def cmd_finnish(args):
    import numpy as np
    run = args.run_name
    work = V2 / "finnish" / run
    move_aside(work, "finnish", run)
    work.mkdir(parents=True)
    src = Path(args.dataset)
    phase("Preparing the Finnish dataset (FinBERT)", 2.0)
    run_stage([TTS_PY, "-u", ROOT / "research/finnish_style_tts/prepare_dataset.py", "--wav-dir", src, "--label-dir", src,
               "--out", work / "prepared", "--source-name", run, "--eval-count", "1"], 2.0, 6.0, "Preparing the Finnish dataset (FinBERT)")
    rows = []
    for name in ("train.csv", "eval.csv"):
        if (work / "prepared" / name).exists():
            rows += list(csv.DictReader(open(work / "prepared" / name, encoding="utf-8-sig")))
    if len(rows) < 20:
        raise SystemExit(f"[FAIL] only {len(rows)} usable WAV+TXT pairs (need 20+)")
    stats = f0_for([r["wav_path"] for r in rows], f"finnish_{run}")
    ref = ref_hz(args.character)
    out_rows, levels, ranges = [], [], []
    for r in rows:
        pf = pitch_fields(stats.get(r["wav_path"]), ref)
        if pf["pitch_ok"] == "1":
            levels.append(float(pf["pitch_level_st"])); ranges.append(float(pf["pitch_range_st"]))
        out_rows.append({"stem": f"x_{run}_{r['stem']}", "lang": "fi", "speaker": args.character, "style": "normal",
                         "sentence_id": "", "split": "train", "tokens": "", "wav_path": r["wav_path"],
                         "bert_cache": r["bert_cache"], "duration_sec": r.get("duration_sec", ""), "text": r.get("text", ""), **pf})
    extra_csv = work / "extra_pitch.csv"
    with open(extra_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader(); w.writerows(out_rows)
    level = float(np.median(levels)) if levels else -6.66
    rng = float(np.median(ranges)) if ranges else 12.0
    say(f"[V2] {len(out_rows)} rows, character {args.character}, median pitch level {level:+.2f} st")
    res = base_branch_zdelta(run, args, extra_csv=extra_csv, extra_ratio=args.data_ratio, lo=8.0,
                             extra_rows=len(out_rows))
    register_character(run, {"label": run, "speaker": args.character, **res,
                             "level_st": round(level, 2), "range_st": round(rng, 2)})
    return {"mode": "finnish", "character": run, "speaker": args.character, "character_level_st": round(level, 2),
            "character_range_st": round(rng, 2), "rows": len(out_rows), **res}


BASE_SPEAKERS = ("fi_chatterbox", "zundamon", "metan", "itako")   # emo_common.SPEAKERS before extras


def _rvc_convert(src_rows, pitch, out_dir: Path, pth: str, index: str = "", index_rate=0.1, protect=0.15, chunk=400,
                 lo=None, hi=None):
    """Finnish WAVs -> RVC voice at `pitch` semitones; output files keep the row stem.  Resumable."""
    import os
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    env = os.environ.copy()
    env["PYTHONPATH"] = cfg["rvc_repo"] + os.pathsep + env.get("PYTHONPATH", "")
    env.update(LANG="en_US.UTF-8", LC_ALL="en_US.UTF-8", LANGUAGE="en_US", RVC_LANGUAGE="en_US", PYTHONUTF8="1")
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [r for r in src_rows if not (out_dir / f"{r['stem']}.wav").is_file()]
    for c in range(0, len(todo), chunk):
        part = todo[c:c + chunk]
        stage, tmp = out_dir.parent / f"_in_{out_dir.name}", out_dir.parent / f"_out_{out_dir.name}"
        for d in (stage, tmp):
            shutil.rmtree(d, ignore_errors=True)
        stage.mkdir()
        for r in part:
            shutil.copy2(r["wav_path"], stage / f"{r['stem']}.wav")
        cmd = [cfg["rvc_python"], "-m", "infer.cli", "--model", str(pth)]
        if index and Path(index).is_file():
            cmd += ["--index", str(index)]
        cmd += ["--input", str(stage), "--output", str(tmp), "--f0-method", "rmvpe", "--pitch", str(pitch),
                "--index-rate", str(index_rate), "--protect", str(protect), "--format", "wav"]
        if subprocess.run(cmd, cwd=cfg["rvc_repo"], env=env, stdout=subprocess.DEVNULL).returncode != 0:
            raise SystemExit(f"[FAIL] RVC conversion failed (pitch {pitch:+d})")
        for f in tmp.glob("*.wav"):
            shutil.move(str(f), str(out_dir / f.name))
        for d in (stage, tmp):
            shutil.rmtree(d, ignore_errors=True)
        say(f"  RVC {pitch:+d}: {min(c + chunk, len(todo))}/{len(todo)}")
        if lo is not None:
            say(f"[V2PROGRESS] {lo + (hi - lo) * min(1.0, (c + chunk) / max(1, len(todo))):.1f}")


def _prepare_emotion(spec_rows, work: Path, vocab: dict, max_seconds=16.0):
    """Emotion rows (folder, LAB folder, speaker, style) -> Japanese-style training rows
    (24 kHz, energy trim, LAB phones -> tokens, sentence split)."""
    import numpy as np
    import soundfile as sf
    import dataset_scan
    from emo_common import DEFAULT_TEST_IDS, DEFAULT_VALID_IDS, lab_to_tokens, read_lab_phones, repair_lab_phones, sentence_split
    from emo_prepare import SR, energy_trim, load_resample
    scan = dataset_scan.scan_rows(spec_rows)
    rows, skipped = [], {}
    for it in scan["items"]:
        try:
            tokens = lab_to_tokens(repair_lab_phones(read_lab_phones(Path(it["lab"])))[0])
        except ValueError:
            skipped["bad_lab"] = skipped.get("bad_lab", 0) + 1
            continue
        if any(tok not in vocab for tok in tokens):
            skipped["vocab"] = skipped.get("vocab", 0) + 1
            continue
        spk, st, sid = it["speaker"], it["style"], it["sentence_id"]
        dst = work / "ja_wav24k" / spk / st / f"{spk}_{st}_{sid:03d}.wav"
        if dst.is_file():
            seconds = sf.info(str(dst)).duration
        else:
            audio, _sr = load_resample(Path(it["wav"]))
            audio = energy_trim(audio)
            seconds = audio.size / SR
            if seconds > max_seconds or seconds < 0.6:
                skipped["length"] = skipped.get("length", 0) + 1
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(dst), audio.astype(np.float32), SR, subtype="PCM_16")
        rows.append({"stem": f"ja_{spk}_{st}_{sid:03d}", "lang": "ja", "speaker": spk, "style": st, "sentence_id": sid,
                     "split": sentence_split(sid, test_ids=DEFAULT_TEST_IDS, valid_ids=DEFAULT_VALID_IDS),
                     "tokens": " ".join(tokens), "wav_path": str(dst), "bert_cache": "", "duration_sec": f"{seconds:.3f}",
                     "source_wav": it["wav"], "lab_path": it["lab"], "lab_kind": it["lab_kind"]})
    return rows, skipped, scan


def cmd_scratch(args):
    """New model from scratch: Finnish WAV+TXT + emotion WAV+LAB + RVC bridge -> base (random init) -> branch -> z-delta."""
    import math as _m
    import os
    import random
    import numpy as np
    from emo_common import NEW_TOKENS, write_csv
    run, target = args.run_name, args.target_speaker
    fdir = Path(args.finnish_dir)
    spec = json.loads(Path(args.emotion_spec).read_text(encoding="utf-8"))
    work = V2 / "scratch" / run
    move_aside(work, "scratch", run)
    data = work / "data"
    data.mkdir(parents=True)
    (work / "emotion_rows.json").write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")

    # 1. Finnish dataset (FinBERT features, held-out eval split)
    run_stage([TTS_PY, "-u", ROOT / "research/finnish_style_tts/prepare_dataset.py", "--wav-dir", fdir,
               "--label-dir", fdir, "--out", work / "fi_prepared", "--source-name", run], 1.0, 10.0,
              "Preparing the Finnish dataset (FinBERT)")
    fi = {}
    for name, split in (("train.csv", "train"), ("eval.csv", "test")):
        fi[split] = [{"stem": f"fi_{r['stem']}", "lang": "fi", "speaker": "fi_chatterbox", "style": "normal", "sentence_id": "",
                      "split": split, "tokens": "", "wav_path": r["wav_path"], "bert_cache": r["bert_cache"],
                      "duration_sec": r.get("duration_sec", ""), "text": r.get("text", "")}
                     for r in csv.DictReader(open(work / "fi_prepared" / name, encoding="utf-8-sig"))]
    if len(fi["train"]) < 50:
        raise SystemExit(f"[FAIL] only {len(fi['train'])} usable Finnish WAV+TXT pairs (need 50+)")
    vocab = json.loads((work / "fi_prepared/vocab.json").read_text(encoding="utf-8"))
    for tok in NEW_TOKENS:                        # š / ž for Japanese phones (as emo_prepare)
        vocab.setdefault(tok, len(vocab))
    (data / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")

    # 2. Emotion dataset
    phase("Preparing the emotion dataset", 10.0)
    ja, skipped, _scan = _prepare_emotion(spec, work, vocab)
    spk_ids = sorted({r["speaker"] for r in ja})
    if target not in spk_ids:
        raise SystemExit(f"[FAIL] character '{target}' has no usable recordings in the emotion folder (found {spk_ids})")
    say(f"[V2] emotion rows {len(ja)} (" + ", ".join(f"{s}: {sum(r['speaker'] == s for r in ja)}" for s in spk_ids)
        + f"); skipped {skipped}")
    extras = [s for s in spk_ids if s not in BASE_SPEAKERS]

    # 3. F0 of the Finnish and emotion recordings
    stats_fi = f0_for([r["wav_path"] for r in fi["train"] + fi["test"]], f"scratch_{run}_fi")
    stats_ja = f0_for([r["wav_path"] for r in ja], f"scratch_{run}_ja")

    def med(v):
        return float(np.median(v)) if v else float("nan")

    def hz(st, path):
        s = st.get(path) or {}
        return float(s["f0_median_hz"]) if s.get("f0_median_hz") and float(s.get("voiced_ratio") or 0) >= 0.15 else None

    ref = {"fi_chatterbox": med([h for h in (hz(stats_fi, r["wav_path"]) for r in fi["train"]) if h])}
    for s in spk_ids:
        ref[s] = med([h for h in (hz(stats_ja, r["wav_path"]) for r in ja if r["speaker"] == s and r["style"] == "normal") if h])
    if not all(_m.isfinite(v) for v in ref.values()):
        raise SystemExit(f"[FAIL] could not measure the pitch of every speaker: {ref}")

    # 4. RVC bridge: Finnish sentences in the character's voice at two pitch levels (as the v62+ bridge;
    #    Chatterbox 163 Hz -> Zundamon 379 Hz gives +8 / +12)
    gap = 12 * _m.log2(ref[target] / ref["fi_chatterbox"])
    pitches = sorted({max(-24, min(24, round(gap - 6.6))), max(-24, min(24, round(gap - 2.6)))})
    pool = sorted(fi["train"], key=lambda r: r["stem"])
    # every Finnish sentence of the folder is bridged (the RVC bridge share then sets how often it is used)
    sel = pool if not args.bridge_sentences else random.Random(70).sample(pool, min(args.bridge_sentences, len(pool)))
    say(f"[V2] bridge: {len(sel)} sentences x pitch {pitches} (Finnish {ref['fi_chatterbox']:.0f} Hz -> {target} {ref[target]:.0f} Hz)")
    bridge = {}
    for i, p_ in enumerate(pitches):
        lo, hi = 20.0 + 15.0 * i / len(pitches), 20.0 + 15.0 * (i + 1) / len(pitches)
        phase(f"RVC bridge ({Path(args.rvc_pth).stem}, pitch {p_:+d})", lo)
        _rvc_convert(sel, p_, work / f"bridge_p{p_}", args.rvc_pth, args.rvc_index, lo=lo, hi=hi)
        bridge[p_] = [dict(r, stem=f"br{p_}_{r['stem']}", speaker=target, split="train", rvc_pitch=p_,
                           wav_path=str(work / f"bridge_p{p_}" / f"{r['stem']}.wav"))
                      for r in sel if (work / f"bridge_p{p_}" / f"{r['stem']}.wav").is_file()]
    stats_br = f0_for([r["wav_path"] for rs in bridge.values() for r in rs], f"scratch_{run}_bridge")

    # 5. Pitch labels + statistics (as v62_prepare_pitch.py; reference = each speaker's own Normal median)
    phase("Pitch labels", 36.0)
    groups = {}

    def label(rows, st, name):
        for r in rows:
            r.update(pitch_fields(st.get(r["wav_path"]), ref[r["speaker"]]))
            if r["pitch_ok"] == "1":
                key = f"{name}:{r['speaker']}:{r['style']}" if name == "ja" else f"{name}:{r['speaker']}"
                g = groups.setdefault(key, ([], []))
                g[0].append(float(r["pitch_level_st"]))
                g[1].append(float(r["pitch_range_st"]))

    label(fi["train"], stats_fi, "fi_train")
    label(fi["test"], stats_fi, "fi_eval")
    label(ja, stats_ja, "ja")
    for p_, rows in bridge.items():
        label(rows, stats_br, f"bridge_p{p_}")
    presets = {}
    for k, (lv, rs) in groups.items():
        if k.startswith("ja:") or k.startswith("fi_train:"):
            _n, spk, *st = k.split(":")
            presets.setdefault(spk, {})[st[0] if st else "normal"] = {"level_st": round(med(lv), 3), "range_st": round(med(rs), 3)}
    stats = {"schema": "zundanen-v62-pitch-stats-v1", "reference_hz": ref,
             "range_center_st": med([x for _lv, rs in groups.values() for x in rs]), "presets_from_real_recordings": presets,
             "groups": {k: {"n": len(lv), "level_median": round(med(lv), 3), "range_median": round(med(rs), 3)}
                        for k, (lv, rs) in sorted(groups.items())},
             "note": f"v2 scratch model '{run}' (finnish {fdir}, emotion {len(spec)} rows, bridge {Path(args.rvc_pth).name} -> {target})"}
    (data / "pitch_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(data / "fi_train_pitch.csv", fi["train"])
    write_csv(data / "fi_eval_pitch.csv", fi["test"])
    write_csv(data / "ja_pitch.csv", ja)
    for p_, rows in bridge.items():
        write_csv(data / f"bridge_p{p_}_pitch.csv", rows)

    # 6. Train: base from random weights -> style branch -> z-delta (env: this data folder and speaker list)
    os.environ["ZUNDANEN_V65_DATA"] = str(data)
    os.environ["ZUNDANEN_EXTRA_SPEAKERS"] = ",".join(extras)
    base_name, branch_name = f"v2_{run}_base", f"v2_{run}_branch"
    for d in (ROOT / "outputs/v65" / base_name, ROOT / "outputs/v65" / branch_name):
        move_aside(d, "scratch", run)
    n_rows = len(fi["train"]) + sum(len(r) for r in bridge.values()) + sum(r["split"] == "train" for r in ja)
    n_style = sum(r["split"] == "train" and r["style"] != "normal" for r in ja)
    bb = branch_batch(args.batch_size)
    steps = args.steps or epoch_steps(args.epochs, n_rows, args.batch_size)
    branch_steps = args.branch_steps or epoch_steps(args.branch_epochs, n_style, bb, floor=500)
    say(f"[V2] base from scratch: {args.epochs:g} epochs of {n_rows} rows = {steps} steps; "
        f"style branch: {args.branch_epochs:g} epochs of {n_style} rows = {branch_steps} steps")
    run_stage([TTS_PY, "-u", "-I", ROOT / "emotion_tts/emotion_train.py", "--flow-cond", "--from-scratch", "--target-speaker", target,
               "--out-name", base_name, "--steps", steps, "--save-every", max(500, steps // 4), "--batch-size", args.batch_size,
               "--workers", args.workers, "--lr", args.lr, "--mix", mix_arg(args.bridge_ratio)], 38.0, 90.0,
              "Training the base model (from scratch)")
    run_stage([TTS_PY, "-u", "-I", ROOT / "emotion_tts/emotion_style.py", "branch", "--base-name", base_name, "--branch-name", branch_name,
               "--steps", branch_steps, "--batch-size", bb, "--workers", args.workers], 90.0, 97.0, "Training the style branch")
    run_stage([TTS_PY, "-u", "-I", ROOT / "emotion_tts/emotion_style.py", "zdelta", "--base-name", base_name,
               "--branch-name", branch_name], 97.0, 99.0, "Building voice-quality deltas")

    # 7. Register: pitch 0 = the lower bridge level; style tempo = the character's own style/Normal length ratio
    low = min(bridge)
    level = med([float(r["pitch_level_st"]) for r in bridge[low] if r["pitch_ok"] == "1"])
    dur = {(r["style"], r["sentence_id"]): float(r["duration_sec"]) for r in ja if r["speaker"] == target}
    tempo = {"normal": 1.0}
    for st in ("ama", "sexy", "tsun"):
        ratios = [d / dur[("normal", sid)] for (s, sid), d in dur.items() if s == st and ("normal", sid) in dur]
        tempo[st] = round(med(ratios), 4) if ratios else 1.0
    register_character(run, {"label": run, "speaker": target, "base_name": base_name, "branch_name": branch_name,
                             "level_st": round(level, 2), "range_st": 12.0, "style_tempo": tempo})
    return {"mode": "scratch", "character": run, "speaker": target, "base_name": base_name, "branch_name": branch_name,
            "finnish_rows": len(fi["train"]), "emotion_rows": len(ja), "bridge_pitches": pitches,
            "bridge_rows": sum(len(r) for r in bridge.values()), "level_st": round(level, 2), "style_tempo": tempo,
            "data": str(data)}


def _number(stem):
    m = re.findall(r"(\d+)", stem)
    return int(m[-1]) if m else None


def cmd_style(args):
    import numpy as np
    import soundfile as sf
    import torch
    from emo_common import lab_to_tokens, read_lab_phones, repair_lab_phones
    from emo_prepare import energy_trim, load_resample
    name = args.style_name.strip().lower()
    if not re.fullmatch(r"[a-z][a-z0-9_]{1,23}", name) or name in ("normal", "ama", "sexy", "tsun"):
        raise SystemExit("[FAIL] style name must be 2-24 chars a-z0-9_, start with a letter, and not an existing style")
    pre = preset()
    model_key, target = model_entry(args.model)
    args.character = target["speaker"]
    base_name = target["base_name"]
    say(f"[V2] style for model '{model_key}' (speaker {args.character}, base {base_name})")
    if name in pre.get("extra_styles", {}):
        raise SystemExit(f"[FAIL] style '{name}' already exists")
    work = V2 / "styles" / name
    move_aside(work, "style", args.run_name)
    phase("Pairing Normal and style recordings", 2.0)

    def collect(folder):
        out = {}
        for wav in sorted(Path(folder).glob("*.wav")):
            n = _number(wav.stem)
            lab = wav.with_suffix(".lab")
            if n is not None and lab.is_file() and "(" not in wav.name:
                out[n] = (wav, lab)
        return out
    normal, style = collect(args.normal_dir), collect(args.style_dir)
    ids = sorted(set(normal) & set(style))
    say(f"[V2] Normal {len(normal)}, style {len(style)}, paired {len(ids)} (by the number in the file name, LAB next to WAV)")
    if len(ids) < 20:
        raise SystemExit("[FAIL] need at least 20 paired sentences (WAV + LAB in both folders)")
    rows = []
    for k, i in enumerate(ids):
        split = "valid" if k % 10 == 9 else "train"
        for st, (wav, lab) in (("normal", normal[i]), (name, style[i])):
            try:
                phones, _rep = repair_lab_phones(read_lab_phones(lab))
                toks = lab_to_tokens(phones)
            except ValueError as exc:
                say(f"[WARN] skip {lab.name}: {exc}")
                continue
            dst = work / "wav24k" / st / f"{st}_{i:03d}.wav"
            audio, _ = load_resample(wav)
            audio = energy_trim(audio)
            if not 0.6 <= audio.size / 24000 <= 11.0:
                say(f"[WARN] skip {wav.name}: {audio.size / 24000:.1f}s")
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(dst), audio.astype(np.float32), 24000, subtype="PCM_16")
            rows.append({"stem": f"s_{name}_{st}_{i:03d}", "lang": "ja", "speaker": args.character, "style": st,
                         "sentence_id": str(i), "split": split, "tokens": " ".join(toks), "wav_path": str(dst),
                         "bert_cache": "", "duration_sec": f"{audio.size / 24000:.3f}"})
    stats = f0_for([r["wav_path"] for r in rows], f"style_{name}")
    ref = ref_hz(args.character)
    for r in rows:
        r.update(pitch_fields(stats.get(r["wav_path"]), ref))
    (work / "rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- style branch (separate; styles = [normal, name])
    import emotion_style as S
    from style_branch import StyleBranch
    from pitch_model import V62Combo
    from emotion_model import flow_apply
    from emotion_train import V65Dataset, collate65
    from emo_vits import import_research
    import_research(ROOT)
    from mas_alignment import mas_attention
    from train_vits import expand_prior, vits_kl
    S.BASE_NAME = base_name
    dev = torch.device("cuda")
    bundle = torch.load(target["bundle"], map_location="cpu", weights_only=False) if target.get("bundle") else None
    ck, st_stats, vocab, _ja, _seen, proto = S.load_all(dev, with_branch=False, bundle=bundle)
    base = proto.base.eval()
    for p in base.parameters():
        p.requires_grad_(False)
    styles = ["normal", name]
    branch = StyleBranch(proto.branch.sym_emb.num_embeddings - 1, len(styles), len(ck["speakers"]))
    model = V62Combo(base, branch, proto.sym_table.cpu(), proto.sym_mask.cpu(), styles, ck["speakers"], ck["langs"]).to(dev)
    rc, refs = st_stats["range_center_st"], st_stats["reference_hz"]

    class DS(V65Dataset):
        def __init__(self, rs):
            self.sidx = [styles.index(r["style"]) for r in rs]
            super().__init__([dict(r, style="normal") for r in rs], vocab, rc, refs)

        def __getitem__(self, i):
            d = super().__getitem__(i)
            d["style"] = self.sidx[i]
            return d
    tr = [r for r in rows if r["style"] == name and r["split"] == "train"]
    va = [r for r in rows if r["style"] == name and r["split"] == "valid"]
    loader = torch.utils.data.DataLoader(DS(tr), batch_size=min(16, len(tr)), shuffle=True, num_workers=0,
                                         collate_fn=collate65, drop_last=True)   # small set; in-process (Windows spawn)
    vloader = torch.utils.data.DataLoader(DS(va), batch_size=8, collate_fn=collate65) if va else None
    opt = torch.optim.AdamW(branch.parameters(), lr=2e-4, betas=(0.8, 0.99), weight_decay=1e-4)

    def losses(b):
        tok, bert, tl = b["tokens"].to(dev), b["bert"].to(dev), b["token_lengths"].to(dev)
        spec, sl = b["spec"].to(dev), b["spec_lengths"].to(dev)
        with torch.no_grad():
            z, _mq, lq, ym = base.posterior(spec, sl)
            okm = ym * b["f0_ok"].to(dev).view(-1, 1, 1)
            zp = flow_apply(base, z, ym, b["f0feat"].to(dev) * okm, reverse=False)
            f0term = base.f0.f0_proj(b["f0feat"].to(dev) * okm) * ym
        mix = torch.nn.functional.one_hot(b["style"], len(styles)).float().to(dev)
        h, mp, lp, xm, dlogw, _ = model.prior(tok, bert, tl, speaker_id=b["speaker"].to(dev), lang_id=b["lang"].to(dev),
                                              has_bert=b["has_bert"].to(dev), style_mix=mix, pitch=b["pitch"].to(dev))
        attn = mas_attention(zp, mp.detach(), lp.detach(), xm, ym, tl, sl)
        mpe, lpe = expand_prior(attn, mp, lp)
        return vits_kl(zp, lq, mpe + f0term, lpe, ym)

    batch = min(branch_batch(args.batch_size), len(tr))
    total_steps = args.branch_steps or epoch_steps(args.branch_epochs, len(tr), batch, floor=500)
    args.branch_steps = total_steps
    say(f"[V2] style branch: {args.branch_epochs:g} epochs of {len(tr)} sentences = {total_steps} steps")
    phase("Training the style branch", 30.0)
    (work / "branch/checkpoints").mkdir(parents=True, exist_ok=True)
    best, best_cp, it = float("inf"), None, iter(())
    model.train(); base.eval()
    for step in range(1, args.branch_steps + 1):
        try:
            b = next(it)
        except StopIteration:
            it = iter(loader); b = next(it)
        loss = losses(b)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(branch.parameters(), 10.0); opt.step()
        if step % 100 == 0:
            say(f"  step {step}/{args.branch_steps} KL={float(loss):.4f}")
            say(f"[V2PROGRESS] {30 + 55 * step / args.branch_steps:.1f}")
        if step % 500 == 0 or step == args.branch_steps:
            v = float(loss)
            if vloader is not None:
                model.eval()
                with torch.no_grad():
                    v = float(np.mean([float(losses(vb)) for vb in vloader]))
                model.train(); base.eval()
            cp = work / "branch/checkpoints" / f"style_{step:06d}.pt"
            torch.save({"schema": "zundanen-v2-extra-style-branch-v1", "step": step, "branch": branch.state_dict(),
                        "styles": styles, "base_name": base_name}, cp)
            say(f"  [VAL] step {step}: KL={v:.4f}")
            if v < best:
                best, best_cp = v, cp

    # ---- z-delta, tempo, pitch offsets from the Normal/style pairs
    phase("Building voice-quality delta, tempo and pitch", 86.0)
    from collections import defaultdict
    sums, cnts = defaultdict(lambda: torch.zeros(192, device=dev)), defaultdict(int)
    by = {(r["style"], r["sentence_id"]): r for r in rows}
    durs, lv, rg = [], {"normal": [], name: []}, {"normal": [], name: []}
    for r in rows:
        if r["pitch_ok"] == "1":
            lv[r["style"]].append(float(r["pitch_level_st"])); rg[r["style"]].append(float(r["pitch_range_st"]))
    model.eval()
    for (st, sid), r in by.items():
        if st != name or ("normal", sid) not in by:
            continue
        rn = by[("normal", sid)]
        durs.append(float(r["duration_sec"]) / float(rn["duration_sec"]))
        if rn["tokens"] != r["tokens"]:
            continue
        b = collate65([DS([rn])[0], DS([r])[0]])
        with torch.no_grad():
            tok, bert, tl = b["tokens"].to(dev), b["bert"].to(dev), b["token_lengths"].to(dev)
            spec, sl = b["spec"].to(dev), b["spec_lengths"].to(dev)
            z, mq, _lq, ym = base.posterior(spec, sl)
            okm = ym * b["f0_ok"].to(dev).view(-1, 1, 1)
            zp = flow_apply(base, z, ym, b["f0feat"].to(dev) * okm, reverse=False)
            h, mp, lp, xm = base.text(tok, bert, tl, speaker_id=b["speaker"].to(dev), style_id=torch.zeros(2, dtype=torch.long, device=dev),
                                      lang_id=b["lang"].to(dev), has_bert=b["has_bert"].to(dev), pitch=b["pitch"].to(dev))
            attn = mas_attention(zp, mp, lp, xm, ym, tl, sl)
            cnt = attn.sum(-1)
            means = torch.einsum("bxy,bcy->bxc", attn, mq) / cnt.clamp_min(1).unsqueeze(-1)
        for i in range(int(b["token_lengths"][0])):
            sym = int(proto.sym_table[b["tokens"][0, i]])
            if sym == 0 or cnt[0, i] < 2 or cnt[1, i] < 2:
                continue
            d = means[1, i] - means[0, i]
            for k in ((args.character, name, sym), ("*", name, sym), (args.character, name, -1), ("*", name, -1)):
                sums[k] += d; cnts[k] += 1
    zfile = work / "zdelta.pt"
    torch.save({"schema": "zundanen-v63-zdelta-v1", "delta": {k: (sums[k] / cnts[k]).cpu() for k in sums}, "counts": dict(cnts)}, zfile)
    entry = {"label": args.label or name, "speaker": args.character, "base_name": base_name, "model": model_key,
             "branch": str(best_cp.relative_to(ROOT)), "zdelta": str(zfile.relative_to(ROOT)),
             "tempo": round(float(np.median(durs)), 4) if durs else 1.0,
             "level_offset_st": round(float(np.median(lv[name]) - np.median(lv["normal"])), 3) if lv[name] and lv["normal"] else 0.0,
             "range_ratio": round(float(np.median(rg[name]) / np.median(rg["normal"])), 3) if rg[name] and rg["normal"] else 1.0,
             "strength": 1.0, "pairs": len(durs)}
    say(f"[V2] style '{name}': tempo {entry['tempo']}x, pitch {entry['level_offset_st']:+.2f} st, range x{entry['range_ratio']}, pairs {len(durs)}")
    backup = V2 / "preset_backups" / f"v65_preset_{time.strftime('%Y%m%d_%H%M%S')}.json"
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PRESET, backup)
    pre = preset()
    pre.setdefault("extra_styles", {})[name] = entry
    PRESET.write_text(json.dumps(pre, ensure_ascii=False, indent=2), encoding="utf-8")
    say(f"[V2] registered style '{name}' (preset backup: {backup})")
    return {"mode": "style", "style": name, **entry}


def main() -> int:
    for s in (sys.stdout,):
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["joint", "finnish", "style", "scratch"])
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--epochs", type=float, default=4.5, help="base model: passes over the joint training data")
    ap.add_argument("--branch-epochs", type=float, default=80.0, help="style branch: passes over its training sentences")
    ap.add_argument("--steps", type=int, default=0, help="override --epochs (testing)")
    ap.add_argument("--branch-steps", type=int, default=0, help="override --branch-epochs (testing)")
    ap.add_argument("--batch-size", type=int, default=10, help="base model batch (the style branch uses 1.6x)")
    ap.add_argument("--workers", type=int, default=4, help="data loader workers (CPU)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bridge-ratio", type=float, default=0.25)
    ap.add_argument("--init-checkpoint", default="")
    ap.add_argument("--dataset", default="")
    ap.add_argument("--character", default="zundamon", help="finnish: speaker id of the new data")
    ap.add_argument("--base-model", default="Zundamon", help="joint/finnish: model (character) to start from")
    ap.add_argument("--model", default="Zundamon", help="style: model (character) the style is trained for")
    ap.add_argument("--data-ratio", type=float, default=0.25)
    ap.add_argument("--style-name", default="")
    ap.add_argument("--label", default="")
    ap.add_argument("--normal-dir", default="")
    ap.add_argument("--style-dir", default="")
    ap.add_argument("--finnish-dir", default="", help="scratch: Finnish WAV + same-name TXT folder")
    ap.add_argument("--emotion-spec", default="", help="scratch: JSON list of {wav_dir, lab_dir, speaker, style}")
    ap.add_argument("--rvc-pth", default="", help="scratch: RVC model (.pth) for the bridge")
    ap.add_argument("--rvc-index", default="", help="scratch: its .index ('' = none)")
    ap.add_argument("--target-speaker", default="", help="scratch: speaker in the emotion folder the RVC voice is")
    ap.add_argument("--bridge-sentences", type=int, default=0, help="scratch: 0 = every Finnish sentence (testing: fewer)")
    args = ap.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_\-]{1,40}", args.run_name):
        raise SystemExit("[FAIL] model name: letters, digits, _ and - only")
    if args.mode in ("joint", "finnish", "scratch") and args.run_name.lower() in {k.lower() for k in chars()}:
        raise SystemExit(f"[FAIL] a character named '{args.run_name}' already exists; choose another model name")
    t0 = time.time()
    res = {"joint": cmd_joint, "finnish": cmd_finnish, "style": cmd_style, "scratch": cmd_scratch}[args.mode](args)
    res["minutes"] = round((time.time() - t0) / 60, 1)
    (V2 / "results").mkdir(parents=True, exist_ok=True)
    out = V2 / "results" / f"{args.mode}_{args.run_name}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    say("RESULT_JSON : " + json.dumps({**res, "result_file": str(out)}, ensure_ascii=False))
    say("[V2PROGRESS] 100")
    say("TRAINING COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
