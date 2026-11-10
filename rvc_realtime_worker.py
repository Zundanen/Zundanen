"""Zundanen in-app realtime RVC (no extra window).

Runs inside the RVC Python with the RVC-WebUI repo as working directory.
The audio processing is the official RVC realtime pipeline (infer/rtrvc.py and
the SOLA crossfade callback of realtime_gui.py, RVC-Project, MIT License),
without the FreeSimpleGUI window: settings come from the environment variable
ZUNDANEN_RT_JSON, the system default microphone / speaker are used, and the
process runs until Zundanen stops it.

Printed status lines: "ZUNDANEN_RT_READY {json}", "ZUNDANEN_RT_ERROR text".
"""
import json
import os
import sys
import time
import traceback

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ["OMP_NUM_THREADS"] = "4"


HOSTAPI_ORDER = ("Windows WASAPI", "MME", "Windows DirectSound", "Windows WDM-KS")


def device_name(sd, i):
    """PortAudio returns UTF-8 names; sounddevice can mis-decode Japanese ones."""
    import re
    try:
        name = sd._ffi.string(sd._lib.Pa_GetDeviceInfo(i).name).decode("utf-8")
    except Exception:
        name = sd.query_devices(i)["name"]
    name = " ".join(name.split())
    m = re.search(r"\(@System32\\drivers\\[^;]+;%1 ([^%]*)%0 ;\((.+)\)\)$", name)
    if m:                                  # Bluetooth hands-free: "Headset (@System32...;(WH-CH520))"
        name = name[: m.start()] + f"({m.group(2)} {m.group(1)})"
    return name


def list_devices(sd):
    apis = sd.query_hostapis()
    out = []
    for d in sd.query_devices():
        out.append({"index": d["index"], "name": device_name(sd, d["index"]),
                    "hostapi": apis[d["hostapi"]]["name"], "inputs": d["max_input_channels"],
                    "outputs": d["max_output_channels"]})
    return out


def pick_devices(sd):
    """Default input/output of the first audio system that has both (Windows may have no default mic)."""
    apis = sd.query_hostapis()
    devs = sd.query_devices()
    order = sorted(range(len(apis)), key=lambda i: HOSTAPI_ORDER.index(apis[i]["name"]) if apis[i]["name"] in HOSTAPI_ORDER else 9)
    for i in order:
        a = apis[i]
        ins = [d for d in a["devices"] if devs[d]["max_input_channels"] > 0]
        outs = [d for d in a["devices"] if devs[d]["max_output_channels"] > 0]
        if not ins or not outs:
            continue
        di = a["default_input_device"] if a["default_input_device"] in ins else ins[0]
        do = a["default_output_device"] if a["default_output_device"] in outs else outs[0]
        return {"input": di, "output": do}
    return {"input": None, "output": None}


def main() -> int:
    if "--list-devices" in sys.argv:
        import sounddevice as sd
        print(json.dumps({"devices": list_devices(sd), "auto": pick_devices(sd)}, ensure_ascii=False))
        return 0
    cfg = json.loads(os.environ["ZUNDANEN_RT_JSON"])
    repo = os.path.abspath(cfg["rvc_repo"])
    os.chdir(repo)
    sys.path.insert(0, repo)
    sys.argv = [sys.argv[0]]              # RVC's Config() parses argv

    import librosa
    import numpy as np
    import sounddevice as sd
    import torch
    import torch.nn.functional as F
    import torchaudio.transforms as tat
    from configs.config import Config
    from infer import rtrvc as rvc_for_realtime

    config = Config()
    pitch = int(cfg.get("pitch", 0))
    index_rate = float(cfg.get("index_rate", 0.1))
    block_time, crossfade_time, extra_time = 0.25, 0.05, 2.5
    threshold = -60
    f0method = cfg.get("f0method", "rmvpe")   # rmvpe (default) | fcpe | pm

    rvc = rvc_for_realtime.RVC(pitch, 0.0, cfg["pth"], cfg.get("index", ""), index_rate if cfg.get("index") else 0.0, config)
    device = config.device
    mute = bool(cfg.get("mute"))          # test mode: full pipeline, silent output

    def stream_formats(i, o):
        """(samplerate, dtype, (in_ch, out_ch)) combinations this device pair accepts."""
        di, do = sd.query_devices(device=i), sd.query_devices(device=o)
        if di["hostapi"] != do["hostapi"] or di["max_input_channels"] < 1 or do["max_output_channels"] < 1:
            return []
        rates = list(dict.fromkeys([int(di["default_samplerate"]), int(do["default_samplerate"]), 48000, 44100, 16000]))
        chans = list(dict.fromkeys([(min(di["max_input_channels"], 2), min(do["max_output_channels"], 2)),
                                    (1, min(do["max_output_channels"], 2)), (min(di["max_input_channels"], 2), 2)]))
        out = []
        for rate in rates:
            for dt in ("float32", "int16"):
                for ci, co in chans:
                    if co > do["max_output_channels"]:
                        continue
                    try:
                        sd.check_input_settings(device=i, channels=ci, dtype=dt, samplerate=rate)
                        sd.check_output_settings(device=o, channels=co, dtype=dt, samplerate=rate)
                        out.append((rate, dt, (ci, co)))
                    except Exception:
                        pass
        return out

    def candidate_pairs():
        dev_in, dev_out = cfg.get("input_device"), cfg.get("output_device")
        if dev_in is not None and dev_out is not None:
            return [(dev_in, dev_out)]
        apis, devs = sd.query_hostapis(), sd.query_devices()
        auto = pick_devices(sd)
        pairs = []
        order = sorted(range(len(apis)), key=lambda k: HOSTAPI_ORDER.index(apis[k]["name"]) if apis[k]["name"] in HOSTAPI_ORDER else 9)
        for k in order:
            ids = apis[k]["devices"]
            ins = [d for d in ids if devs[d]["max_input_channels"] > 0] if dev_in is None else [dev_in]
            outs = [d for d in ids if devs[d]["max_output_channels"] > 0] if dev_out is None else [dev_out]
            # a real microphone first, the system default next, Bluetooth hands-free / stereo mix last
            def in_rank(d):
                n = device_name(sd, d).lower()
                return (0 if ("mic" in n or "マイク" in n) else 2 if ("hands-free" in n or "ステレオ" in n or "stereo" in n) else 1,
                        0 if d == auto["input"] else 1)
            def out_rank(d):
                return (0 if d == apis[k]["default_output_device"] else 1, 0 if "hands-free" not in device_name(sd, d).lower() else 1)
            pairs += [(i, o) for i in sorted(ins, key=in_rank) for o in sorted(outs, key=out_rank)]
        return pairs

    stream = None
    tried = []
    for dev_in, dev_out in candidate_pairs():
        for sr, dtype, (ch_in, ch_out) in stream_formats(dev_in, dev_out):
            tried.append(f"{dev_in}->{dev_out} {sr} {dtype} {ch_in}/{ch_out}")
            try:
                # callback mode: WDM-KS does not support blocking streams
                probe = sd.Stream(samplerate=sr, dtype=dtype, channels=(ch_in, ch_out), device=(dev_in, dev_out),
                                  callback=lambda i, o, f, t, s: o.fill(0))
                probe.close()
                stream = (dev_in, dev_out, sr, dtype, ch_in, ch_out)
                break
            except Exception:
                continue
        if stream:
            break
    if not stream:
        raise RuntimeError("Could not open the microphone and speaker together. Choose other devices (same audio system). "
                           "Tried: " + "; ".join(tried[:6]))
    dev_in, dev_out, sr, dtype, ch_in, ch_out = stream

    resampler2 = (tat.Resample(orig_freq=rvc.tgt_sr, new_freq=sr, dtype=torch.float32).to(device)
                  if rvc.tgt_sr != sr else None)
    zc = sr // 100
    block_frame = int(np.round(block_time * sr / zc)) * zc
    block_frame_16k = 160 * block_frame // zc
    crossfade_frame = int(np.round(crossfade_time * sr / zc)) * zc
    sola_buffer_frame = min(crossfade_frame, 4 * zc)
    sola_search_frame = zc
    extra_frame = int(np.round(extra_time * sr / zc)) * zc
    input_wav = torch.zeros(extra_frame + crossfade_frame + sola_search_frame + block_frame, device=device, dtype=torch.float32)
    input_wav_res = torch.zeros(160 * input_wav.shape[0] // zc, device=device, dtype=torch.float32)
    sola_buffer = torch.zeros(sola_buffer_frame, device=device, dtype=torch.float32)
    sola_den_kernel = torch.ones(1, 1, sola_buffer_frame, device=device, dtype=torch.float32)
    skip_head = extra_frame // zc
    return_length = (block_frame + sola_buffer_frame + sola_search_frame) // zc
    fade_in = torch.sin(0.5 * np.pi * torch.linspace(0.0, 1.0, steps=sola_buffer_frame, device=device, dtype=torch.float32)) ** 2
    fade_out = 1 - fade_in
    resampler = tat.Resample(orig_freq=sr, new_freq=16000, dtype=torch.float32).to(device)
    state = {"input_wav": input_wav, "input_wav_res": input_wav_res, "sola_buffer": sola_buffer, "n": 0, "t": 0.0}
    scale = 32768.0 if dtype == "int16" else 1.0

    def callback(indata, outdata, frames, times, status):
        try:
            t0 = time.perf_counter()
            x = librosa.to_mono(indata.T.astype(np.float32) / scale)
            iw, iwr, sb = state["input_wav"], state["input_wav_res"], state["sola_buffer"]
            iw[:-block_frame] = iw[block_frame:].clone()
            iw[-x.shape[0]:] = torch.from_numpy(x).to(device)
            iwr[:-block_frame_16k] = iwr[block_frame_16k:].clone()
            iwr[-160 * (x.shape[0] // zc + 1):] = resampler(iw[-x.shape[0] - 2 * zc:])[160:]
            y = rvc.infer(iwr, block_frame_16k, skip_head, return_length, f0method)
            if resampler2 is not None:
                y = resampler2(y)
            # SOLA (from https://github.com/yxlllc/DDSP-SVC, as in realtime_gui.py)
            conv_input = y[None, None, : sola_buffer_frame + sola_search_frame]
            cor_nom = F.conv1d(conv_input, sb[None, None, :])
            cor_den = torch.sqrt(F.conv1d(conv_input ** 2, sola_den_kernel) + 1e-8)
            off = int(torch.argmax(cor_nom[0, 0] / cor_den[0, 0]))
            y = y[off:]
            y[:sola_buffer_frame] *= fade_in
            y[:sola_buffer_frame] += sb * fade_out
            sb[:] = y[block_frame: block_frame + sola_buffer_frame]
            if mute:
                outdata[:] = 0
            else:
                out = y[:block_frame].clamp(-1, 1).repeat(ch_out, 1).t().cpu().numpy()
                outdata[:] = (out * 32767).astype(np.int16) if dtype == "int16" else out
            state["n"] += 1
            state["t"] += time.perf_counter() - t0
            if state["n"] in (8, 40):
                print("ZUNDANEN_RT_STATS " + json.dumps({"blocks": state["n"], "infer_ms": round(1000 * state["t"] / state["n"], 1)}), flush=True)
        except Exception:
            outdata[:] = 0
            print("ZUNDANEN_RT_ERROR " + traceback.format_exc(limit=3).replace("\n", " | "), flush=True)

    # Warm up (the first inference loads RMVPE and is slow; a slow first callback makes some drivers stop the stream).
    for _ in range(2):
        rvc.infer(input_wav_res, block_frame_16k, skip_head, return_length, f0method)
    rvc.cache_pitch.zero_()
    rvc.cache_pitchf.zero_()

    def open_stream():
        st = sd.Stream(callback=callback, blocksize=block_frame, samplerate=sr, channels=(ch_in, ch_out), dtype=dtype,
                       device=(dev_in, dev_out))
        st.start()
        return st

    stream = open_stream()
    print("ZUNDANEN_RT_READY " + json.dumps({"input": device_name(sd, dev_in), "output": device_name(sd, dev_out),
                                             "samplerate": sr, "format": dtype, "block_ms": int(block_time * 1000)},
                                            ensure_ascii=False), flush=True)
    restarts = 0
    try:
        while True:                        # runs until Zundanen stops the process
            time.sleep(0.5)
            if not stream.active:
                restarts += 1
                if restarts > 20:
                    raise RuntimeError("The audio device keeps stopping the stream.")
                stream.close()
                stream = open_stream()
    finally:
        stream.abort()
        stream.close()


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        print("ZUNDANEN_RT_ERROR " + traceback.format_exc(limit=6).replace("\n", " | "), flush=True)
        raise SystemExit(1)
