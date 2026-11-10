from __future__ import annotations

# Original Zundanen implementation.
# No Style-Bert-VITS2 source code is used here.
#
# Design:
# - invertible additive coupling
# - unit Jacobian determinant
# - Transformer conditioner with global temporal context
# - deterministic: dropout MUST remain 0 for exact forward/reverse consistency

import math

import torch
from torch import nn


def _sinusoidal_position_encoding(
    length: int,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Returns [1, T, C].
    Computed in float32 and cast to requested dtype.
    """

    pos = torch.arange(
        length,
        device=device,
        dtype=torch.float32,
    ).unsqueeze(1)

    even = torch.arange(
        0,
        channels,
        2,
        device=device,
        dtype=torch.float32,
    )

    div = torch.exp(
        even
        * (
            -math.log(10000.0)
            / float(channels)
        )
    )

    pe = torch.zeros(
        length,
        channels,
        device=device,
        dtype=torch.float32,
    )

    pe[:, 0::2] = torch.sin(
        pos * div
    )

    odd_width = pe[:, 1::2].shape[1]

    if odd_width:
        pe[:, 1::2] = torch.cos(
            pos
            * div[:odd_width]
        )

    return (
        pe.unsqueeze(0)
        .to(dtype=dtype)
    )


class TransformerAdditiveCouplingLayer(nn.Module):
    """
    Mean-only / additive coupling:

        x0 unchanged
        x1' = x1 + f(x0)

    reverse:

        x1 = x1' - f(x0)

    The Jacobian determinant is exactly 1, so logdet=0.

    This deliberately preserves the mathematical assumption used by
    the current Step6 VITS KL implementation.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        ffn_channels: int = 768,
    ):
        super().__init__()

        if channels % 2 != 0:
            raise ValueError(
                "channels must be even"
            )

        if hidden_channels % n_heads != 0:
            raise ValueError(
                "hidden_channels must be divisible by n_heads"
            )

        self.channels = int(channels)
        self.half_channels = (
            self.channels // 2
        )

        self.hidden_channels = int(
            hidden_channels
        )

        self.n_heads = int(
            n_heads
        )

        self.n_layers = int(
            n_layers
        )

        self.ffn_channels = int(
            ffn_channels
        )

        self.pre = nn.Conv1d(
            self.half_channels,
            self.hidden_channels,
            kernel_size=1,
        )

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=self.hidden_channels,
                nhead=self.n_heads,
                dim_feedforward=self.ffn_channels,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
        )

        self.encoder = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=self.n_layers,
                norm=nn.LayerNorm(
                    self.hidden_channels
                ),
                enable_nested_tensor=False,
            )
        )

        self.post = nn.Conv1d(
            self.hidden_channels,
            self.half_channels,
            kernel_size=1,
        )

        # Start as exact identity flow.
        #
        # This is intentionally zero initialized.
        # At the first update post learns, then gradients begin propagating
        # more strongly into the Transformer conditioner.
        nn.init.zeros_(
            self.post.weight
        )

        nn.init.zeros_(
            self.post.bias
        )

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g=None,
        reverse: bool = False,
    ) -> torch.Tensor:

        if g is not None:
            raise NotImplementedError(
                "Finnish VITS is single-speaker; g conditioning is not implemented."
            )

        x0, x1 = torch.split(
            x,
            [
                self.half_channels,
                self.half_channels,
            ],
            dim=1,
        )

        x0 = x0 * x_mask
        x1 = x1 * x_mask

        h = (
            self.pre(x0)
            * x_mask
        )

        # B,C,T -> B,T,C
        h = h.transpose(
            1,
            2,
        )

        pe = (
            _sinusoidal_position_encoding(
                length=h.size(1),
                channels=h.size(2),
                device=h.device,
                dtype=h.dtype,
            )
        )

        h = h + pe

        valid = (
            x_mask[:, 0, :]
            > 0.5
        )

        padding_mask = ~valid

        h = self.encoder(
            h,
            src_key_padding_mask=(
                padding_mask
            ),
        )

        # B,T,C -> B,C,T
        h = (
            h.transpose(
                1,
                2,
            )
            * x_mask
        )

        shift = (
            self.post(h)
            * x_mask
        )

        if not reverse:
            y1 = (
                x1 + shift
            ) * x_mask
        else:
            y1 = (
                x1 - shift
            ) * x_mask

        y = torch.cat(
            [
                x0,
                y1,
            ],
            dim=1,
        )

        return y * x_mask


class TransformerCouplingBlock(nn.Module):
    """
    Replacement for VITS ResidualCouplingBlock.

    Forward:
        coupling -> channel flip
        coupling -> channel flip
        ...

    Reverse performs exact inverse order.

    Important:
    - no dropout
    - no stochastic operations
    - additive only
    - log determinant = 0
    """

    def __init__(
        self,
        channels: int = 192,
        hidden_channels: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        n_flows: int = 4,
        ffn_channels: int = 768,
    ):
        super().__init__()

        self.channels = int(
            channels
        )

        self.hidden_channels = int(
            hidden_channels
        )

        self.n_heads = int(
            n_heads
        )

        self.n_layers = int(
            n_layers
        )

        self.n_flows = int(
            n_flows
        )

        self.ffn_channels = int(
            ffn_channels
        )

        # Compatibility-ish metadata.
        self.kernel_size = 0
        self.dilation_rate = 0
        self.gin_channels = 0

        self.flows = nn.ModuleList(
            [
                TransformerAdditiveCouplingLayer(
                    channels=self.channels,
                    hidden_channels=(
                        self.hidden_channels
                    ),
                    n_heads=self.n_heads,
                    n_layers=self.n_layers,
                    ffn_channels=(
                        self.ffn_channels
                    ),
                )
                for _ in range(
                    self.n_flows
                )
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g=None,
        reverse: bool = False,
    ) -> torch.Tensor:

        if not reverse:

            for flow in self.flows:

                x = flow(
                    x,
                    x_mask,
                    g=g,
                    reverse=False,
                )

                # Same role as VITS Flip.
                x = torch.flip(
                    x,
                    dims=[1],
                )

                x = x * x_mask

        else:

            for flow in reversed(
                self.flows
            ):

                # Inverse of forward flip.
                x = torch.flip(
                    x,
                    dims=[1],
                )

                x = x * x_mask

                x = flow(
                    x,
                    x_mask,
                    g=g,
                    reverse=True,
                )

        return x * x_mask

    def config_dict(self):
        return {
            "flow_type":
                "transformer_mean_only_additive",

            "flow_channels":
                self.channels,

            "flow_hidden_channels":
                self.hidden_channels,

            "flow_heads":
                self.n_heads,

            "flow_transformer_layers_per_coupling":
                self.n_layers,

            "flow_n_flows":
                self.n_flows,

            "flow_ffn_channels":
                self.ffn_channels,

            "flow_dropout":
                0.0,

            "flow_affine_scale":
                False,

            "flow_logdet":
                0.0,
        }


@torch.no_grad()
def validate_flow_roundtrip(
    flow: nn.Module,
    device=None,
) -> float:
    """
    Exact invertibility smoke test on valid frames.
    """

    if device is None:
        try:
            device = next(
                flow.parameters()
            ).device
        except StopIteration:
            device = torch.device(
                "cpu"
            )

    was_training = (
        flow.training
    )

    flow.eval()

    channels = int(
        getattr(
            flow,
            "channels",
            192,
        )
    )

    x = torch.randn(
        2,
        channels,
        37,
        device=device,
        dtype=torch.float32,
    )

    lengths = torch.tensor(
        [37, 23],
        device=device,
    )

    arange = torch.arange(
        37,
        device=device,
    )

    mask = (
        arange.unsqueeze(0)
        < lengths.unsqueeze(1)
    )

    mask = (
        mask.unsqueeze(1)
        .to(x.dtype)
    )

    x = x * mask

    y = flow(
        x,
        mask,
        reverse=False,
    )

    xr = flow(
        y,
        mask,
        reverse=True,
    )

    err = (
        (
            (xr - x)
            * mask
        )
        .abs()
        .max()
        .item()
    )

    if was_training:
        flow.train()

    return float(err)


def _gradient_smoke_test(
    flow: nn.Module,
    device,
):
    flow.train()

    channels = int(
        flow.channels
    )

    x = torch.randn(
        2,
        channels,
        31,
        device=device,
        requires_grad=True,
    )

    mask = torch.ones(
        2,
        1,
        31,
        device=device,
    )

    y = flow(
        x,
        mask,
        reverse=False,
    )

    loss = (
        y.square()
        .mean()
    )

    loss.backward()

    if x.grad is None:
        raise RuntimeError(
            "No input gradient"
        )

    if not torch.isfinite(
        x.grad
    ).all():
        raise RuntimeError(
            "Non-finite input gradient"
        )


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"[DEVICE] {device}"
    )

    flow = (
        TransformerCouplingBlock(
            channels=192,
            hidden_channels=256,
            n_heads=4,
            n_layers=2,
            n_flows=4,
            ffn_channels=768,
        )
        .to(device)
    )

    err = (
        validate_flow_roundtrip(
            flow,
            device=device,
        )
    )

    print(
        "[ROUNDTRIP] "
        f"max_abs_error={err:.9g}"
    )

    if err > 1e-4:
        raise RuntimeError(
            "Transformer flow roundtrip failed"
        )

    _gradient_smoke_test(
        flow,
        device,
    )

    params = sum(
        p.numel()
        for p in flow.parameters()
    )

    print(
        f"[PARAMS] {params:,}"
    )

    print(
        "[PASS] Finnish VITS Transformer flow self-test"
    )


if __name__ == "__main__":
    main()