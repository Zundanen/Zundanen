from __future__ import annotations

"""Word-accent editing inside the v65 VITS (2026-11-25).

Instead of resynthesising the finished audio with WORLD (which makes the edited word sound noisy), the sentence is
rendered again with the same request and seed, with the selected syllable's tokens made longer/shorter and an F0
offset (semitones) added to its frames before the prior, flow and decoder run - the same idea as VOICEVOX, which edits
per-mora pitch and length and lets the model render them.

The noise is drawn on the unedited frame grid (exactly as emotion_style.synth draws it) and remapped token by token, so
the unedited tokens keep their noise.  The flow still has sentence-wide context, so the unedited words come out
slightly different; the unedited render is therefore kept outside the edited words (same splice as the WORLD editor).
Accent shapes and strengths (-50..+50 %) are the WORLD editor's (world_accent_worker.py), so both paths sound alike.

_render() follows emotion_style.synth step by step; with no edits it returns the same samples (checked 2026-11-25).
Original Zundanen code (MIT).
"""

import numpy as np
import torch

import emotion_style as S
import world_accent_worker as WA

SR = 24000
HOP = 256
HOP_S = HOP / SR


def _plan(w0: np.ndarray, edits: list[dict]):
    """Token duration factors from the syllable/word times (seconds on the unedited render)."""
    cum0 = np.concatenate([[0], np.cumsum(w0)])
    mid = (cum0[:-1] + cum0[1:]) / 2 * HOP_S
    factor = np.ones(len(w0))
    marks = []
    for e in edits:
        syl = np.where((w0 > 0) & (mid >= e["start"]) & (mid < e["end"]))[0]
        word = np.where((w0 > 0) & (mid >= e["word_start"]) & (mid < e["word_end"]))[0]
        if syl.size == 0:
            continue
        p = WA._params(e["strength"])
        factor[syl] *= max(WA.MIN_DURATION_SCALE, min(WA.MAX_DURATION_SCALE, 1.0 + p["duration_percent"] / 100.0))
        marks.append((e, syl, word if word.size else syl, p))
    w1 = np.where(w0 > 0, np.maximum(1, np.round(w0 * factor)), 0).astype(np.int64)
    return w1, marks


def _source_frames(w0: np.ndarray, w1: np.ndarray) -> np.ndarray:
    """Unedited frame index for every edited frame (token k's new frames walk over token k's old frames)."""
    cum0 = np.concatenate([[0], np.cumsum(w0)])
    idx = [cum0[k] + np.minimum(w0[k] - 1, (np.arange(w1[k]) * w0[k]) // w1[k]) for k in range(len(w0)) if w1[k] > 0]
    return np.concatenate(idx) if idx else np.zeros(0, dtype=np.int64)


def _pitch_offsets(feat: torch.Tensor, w1: np.ndarray, marks) -> np.ndarray:
    """Semitones per edited frame: word de-accent + rise-fall on the syllable, or (negative) flatten the syllable."""
    st = feat[0, 0].detach().float().cpu().numpy() * 12.0
    voiced = feat[0, 1].detach().float().cpu().numpy() > 0.5
    cum1 = np.concatenate([[0], np.cumsum(w1)])
    curve = np.zeros(st.size)
    env_x = np.array([p[0] for p in WA.ACCENT_ENVELOPE])
    env_y = np.array([p[1] for p in WA.ACCENT_ENVELOPE])
    for e, syl, word, p in marks:
        ws, we = int(cum1[word.min()]), int(cum1[word.max() + 1])
        ss, se = int(cum1[syl.min()]), int(cum1[syl.max() + 1])
        rel = (np.arange(ss, se) - ss + 0.5) / max(1, se - ss)
        if e["strength"] < 0:
            ref = st[ws:we][voiced[ws:we]]
            if ref.size:
                curve[ss:se] = -p["flatten"] * (st[ss:se] - np.median(ref)) * WA._edge_shape(rel)
        else:
            curve[ws:we] = p["word_deaccent_semitones"]
            curve[ss:se] = p["peak_semitones"] * np.interp(rel, env_x, env_y)
    return WA._smooth_curve(curve, max(1, int(round(0.020 / HOP_S))))


def _render(model, zd, tok, bert, *, style, strength, z, level_st, range_k, pitch, speaker="zundamon", lang="fi",
            noise=0.4, seed=1234, device="cuda", length_scale=1.0, tempo=None, dur_detail=0.0, s_fix=None,
            edits=()):
    tok = tok.unsqueeze(0).to(device)
    bert = bert.unsqueeze(0).to(device)
    tl = torch.tensor([tok.size(1)], device=device)
    prior_kw = dict(speaker_id=torch.tensor([model.speakers.index(speaker)], device=device),
                    lang_id=torch.tensor([model.langs.index(lang)], device=device), has_bert=torch.ones(1, device=device),
                    style_mix=S.style_mix_tensor(model.styles, style, strength, device=device),
                    pitch=torch.tensor([pitch], dtype=torch.float32, device=device))
    h, mp, lp, xm, dlogw, _ = model.prior(tok, bert, tl, **prior_kw)
    s_pos = []
    if s_fix:
        s_id, y_id, v_id = (int(x) for x in s_fix)
        ids = tok[0].tolist()
        s_pos = [k for k in range(len(ids) - 1) if ids[k] == s_id and ids[k + 1] == y_id]
    if tempo is None:
        logw = model.log_duration(h, xm, dlogw)
    else:
        logw = model.base.duration(h, xm)
        if dur_detail:
            dev_ = (dlogw - (dlogw * xm).sum() / xm.sum().clamp_min(1)) * xm
            logw = logw + float(dur_detail) * dev_
        logw = logw + float(np.log(tempo))
    w = torch.clamp(torch.ceil(torch.exp(logw) * xm * float(length_scale)), min=1.0) * xm
    w0 = w.reshape(-1).long().cpu().numpy()
    g = torch.Generator(device=device).manual_seed(int(seed))
    eps = torch.randn((1, mp.size(1), int(w0.sum())), generator=g, device=device)   # == synth's m.shape
    w1, marks = _plan(w0, list(edits)) if edits else (w0, [])
    src = _source_frames(w0, w1) if marks else np.arange(int(w0.sum()))
    if marks:
        w = torch.as_tensor(w1, dtype=w.dtype, device=device).reshape(w.shape)
        eps = eps[..., torch.as_tensor(src, device=device)]
    attn, yl = S._path_from_w(w, tl)
    ym = (torch.arange(attn.size(-1), device=device).unsqueeze(0) < yl.unsqueeze(1)).unsqueeze(1).to(mp.dtype)
    m, logs = S.expand_prior(attn, mp, lp)
    st_pred, uv = model.base.f0.predictor(S.expand_frames(attn, h), ym)
    feat = S.shape_contour(st_pred, uv, ym, level_st=level_st, range_k=range_k)
    if marks:
        add = torch.as_tensor(_pitch_offsets(feat, w1, marks) / 12.0, dtype=feat.dtype, device=device)
        feat = torch.stack([feat[:, 0] + add * ym[:, 0], feat[:, 1]], dim=1)
    fp = model.base.f0.f0_proj(feat) * ym
    m = m + fp
    zz = S.flow_apply(model.base, (m + eps * torch.exp(logs) * noise) * ym, ym, feat, reverse=True) * ym
    dz = z * S.frame_delta(zd, model, tok, attn, speaker, style) * ym if z and style != "normal" else None
    if dz is not None:
        zz = zz + dz
    y = model.base.generator(zz)[0, 0].float().cpu().numpy()
    if s_pos:
        tok2 = tok.clone()
        tok2[0, [k + 1 for k in s_pos]] = v_id
        _h2, mp2, lp2, *_r = model.prior(tok2, bert, tl, **prior_kw)
        m2, logs2 = S.expand_prior(attn, mp2, lp2)
        zz2 = S.flow_apply(model.base, (m2 + fp + eps * torch.exp(logs2) * noise) * ym, ym, feat, reverse=True) * ym
        if dz is not None:
            zz2 = zz2 + dz
        cum = np.concatenate([[0], np.cumsum(w.reshape(-1).long().cpu().numpy())])
        voiced, hiss = (feat[0, 1] > 0.5).cpu().numpy(), S._frame_hiss(y)
        zz = zz.clone()
        for k in s_pos:
            a, b = S._s_hiss_window(voiced, hiss, int(cum[k]), int(cum[k + 1]))
            if b > a:
                zz[..., a:b] = zz2[..., a:b]
        y = model.base.generator(zz)[0, 0].float().cpu().numpy()
    return y, w0, w1, src, marks


@torch.no_grad()
def synth_accent(model, zd, tok, bert, *, edits: list[dict], **kw) -> tuple[np.ndarray, dict]:
    """Render with accent edits ({start, end, word_start, word_end, strength} in seconds / %) and keep the unedited
    render outside the edited words.  kw = the emotion_style.synth keyword arguments of the original request."""
    base, w0, _w, _s, _m = _render(model, zd, tok, bert, **kw)
    y, w0, w1, src, marks = _render(model, zd, tok, bert, edits=edits, **kw)
    if not marks:
        return base, {"engine": "vits_accent", "edited_syllables": 0}
    y, share = WA._splice_original(base, y, src.astype(np.float64), int(w0.sum()), HOP_S, SR,
                                   [e for e, *_r in marks])
    # Syllable loudness: plain sample gain on the spliced audio (nothing is resynthesised).
    cum1 = np.concatenate([[0], np.cumsum(w1)])
    gain_db = np.zeros(y.size)
    for _e, syl, _word, p in marks:
        a, b = int(cum1[syl.min()] * HOP), min(y.size, int(cum1[syl.max() + 1] * HOP))
        if b > a and abs(p["gain_db"]) > 1e-9:
            gain_db[a:b] = p["gain_db"] * WA._edge_shape((np.arange(a, b) - a + 0.5) / (b - a))
    y = y * np.power(10.0, gain_db / 20.0)
    return y, {"engine": "vits_accent", "edited_syllables": len(marks), "vits_share": round(share, 4),
               "frames_before": int(w0.sum()), "frames_after": int(w1.sum())}
