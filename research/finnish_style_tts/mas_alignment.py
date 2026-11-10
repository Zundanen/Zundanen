from __future__ import annotations

"""Original PyTorch implementation of VITS-style MAS for Zundanen Step 6A.

No Style-Bert-VITS2 / AGPL source is used.  The score matrix follows the
standard diagonal-Gaussian log-likelihood used by VITS, while the maximum
monotonic path is solved here with a cumulative-max dynamic program.
"""

import math

import torch


@torch.no_grad()
def gaussian_log_likelihood_matrix(
    z_p: torch.Tensor,
    m_p: torch.Tensor,
    logs_p: torch.Tensor,
) -> torch.Tensor:
    """Return log p(z_p frame | text-token prior), shape [B, Ty, Tx]."""
    z = z_p.float()
    m = m_p.float()
    logs = logs_p.float()
    inv_var = torch.exp(-2.0 * logs)

    c1 = torch.sum(-0.5 * math.log(2.0 * math.pi) - logs, dim=1, keepdim=True)
    c2 = torch.matmul((-0.5 * z.square()).transpose(1, 2), inv_var)
    c3 = torch.matmul(z.transpose(1, 2), m * inv_var)
    c4 = torch.sum(-0.5 * m.square() * inv_var, dim=1, keepdim=True)
    return c1 + c2 + c3 + c4


@torch.no_grad()
def maximum_monotonic_path(
    scores: torch.Tensor,
    token_lengths: torch.Tensor,
    frame_lengths: torch.Tensor,
) -> torch.Tensor:
    """Find the maximum-likelihood monotonic path.

    Args:
        scores: [B, Ty, Tx] frame/token score matrix.
        token_lengths: [B] valid token counts.
        frame_lengths: [B] valid acoustic-frame counts.

    Returns:
        [B, Tx, Ty] path.  Every valid frame belongs to exactly one token and
        every valid token receives at least one frame.

    Instead of a frame-by-frame Python Viterbi loop, this formulation uses
    token segment boundaries:

      dp_i[e] = prefix_i[e] + max_{s < e}(dp_{i-1}[s] - prefix_i[s])

    `torch.cummax` evaluates all previous boundaries in parallel.  The only
    Python loop is over text-token positions.
    """
    if scores.ndim != 3:
        raise ValueError(f"scores must be [B,Ty,Tx], got {tuple(scores.shape)}")

    bsz, ty, tx = scores.shape
    if token_lengths.shape != (bsz,) or frame_lengths.shape != (bsz,):
        raise ValueError("length tensor shape mismatch")

    if bool(torch.any(frame_lengths < token_lengths).item()):
        bad = torch.nonzero(frame_lengths < token_lengths, as_tuple=False).flatten()
        pairs = [
            (int(token_lengths[i].item()), int(frame_lengths[i].item()))
            for i in bad[:8]
        ]
        raise RuntimeError(
            "MAS requires frame_length >= token_length; "
            f"bad (tokens,frames) examples={pairs}"
        )

    device = scores.device
    dtype = scores.dtype
    neg_inf = -1.0e30

    # prefix[:, e, i] = sum_{frame < e} score(frame, token i)
    prefix = torch.cat(
        [
            torch.zeros(bsz, 1, tx, device=device, dtype=dtype),
            torch.cumsum(scores, dim=1),
        ],
        dim=1,
    )

    e_idx = torch.arange(ty + 1, device=device).unsqueeze(0)
    dp_prev = torch.full((bsz, ty + 1), neg_inf, device=device, dtype=dtype)
    valid0 = (
        (e_idx >= 1)
        & (e_idx <= frame_lengths.unsqueeze(1))
        & (token_lengths.unsqueeze(1) >= 1)
    )
    dp_prev = torch.where(valid0, prefix[:, :, 0], dp_prev)

    # back[b, i, e] = best previous boundary s when token i ends at e.
    back = torch.full((bsz, tx, ty + 1), -1, device=device, dtype=torch.long)

    for i in range(1, tx):
        candidate = dp_prev - prefix[:, :, i]
        cum_best, cum_arg = torch.cummax(candidate, dim=1)

        best = torch.full_like(cum_best, neg_inf)
        arg = torch.full_like(cum_arg, -1)
        best[:, 1:] = cum_best[:, :-1]
        arg[:, 1:] = cum_arg[:, :-1]

        dp_cur = prefix[:, :, i] + best
        valid = (
            (token_lengths.unsqueeze(1) > i)
            & (e_idx >= i + 1)
            & (e_idx <= frame_lengths.unsqueeze(1))
        )
        dp_prev = torch.where(valid, dp_cur, torch.full_like(dp_cur, neg_inf))
        back[:, i, :] = arg

    # Recover token durations in parallel across the batch.
    durations = torch.zeros(bsz, tx, device=device, dtype=torch.long)
    end = frame_lengths.long().clone()
    b_idx = torch.arange(bsz, device=device)

    for offset in range(tx - 1):
        cur = token_lengths.long() - 1 - offset
        active = cur >= 1
        cur_idx = cur.clamp(min=0, max=tx - 1)
        end_idx = end.clamp(min=0, max=ty)
        start = back[b_idx, cur_idx, end_idx]
        start = torch.where(active, start, end)
        dur = torch.where(active, end - start, torch.zeros_like(end))
        durations.scatter_(1, cur_idx.unsqueeze(1), dur.unsqueeze(1))
        end = start

    durations[:, 0] = end

    ends = torch.cumsum(durations, dim=1)
    starts = ends - durations
    frame_idx = torch.arange(ty, device=device).view(1, 1, ty)
    path = (frame_idx >= starts.unsqueeze(2)) & (frame_idx < ends.unsqueeze(2))

    token_mask = (
        torch.arange(tx, device=device).view(1, tx, 1)
        < token_lengths.view(-1, 1, 1)
    )
    frame_mask = (
        torch.arange(ty, device=device).view(1, 1, ty)
        < frame_lengths.view(-1, 1, 1)
    )
    return (path & token_mask & frame_mask).to(dtype)


@torch.no_grad()
def mas_attention(
    z_p: torch.Tensor,
    m_p: torch.Tensor,
    logs_p: torch.Tensor,
    x_mask: torch.Tensor,
    y_mask: torch.Tensor,
    token_lengths: torch.Tensor,
    frame_lengths: torch.Tensor,
) -> torch.Tensor:
    """Build [B,Tx,Ty] MAS attention from posterior prior-space latents."""
    scores = gaussian_log_likelihood_matrix(z_p, m_p, logs_p)
    attn = maximum_monotonic_path(scores, token_lengths, frame_lengths)
    attn = attn * x_mask.transpose(1, 2).float() * y_mask.float()
    return attn.to(m_p.dtype)
