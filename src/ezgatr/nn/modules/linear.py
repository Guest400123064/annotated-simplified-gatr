from __future__ import annotations

import math
from typing import ClassVar

import torch
from torch import nn

from ezgatr.nn.functional.linear import equi_linear


class EquiLinear(nn.Module):
    r"""Pin(3, 0, 1)-equivariant linear map.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    normalize_basis : bool
        Whether to normalize the basis elements according to the number of
        "inflow paths" for each blade.
    """

    __constants__: ClassVar[list[str]] = [
        "in_channels",
        "out_channels",
        "normalize_basis",
    ]

    in_channels: int
    out_channels: int
    normalize_basis: bool
    weight: torch.Tensor
    bias: torch.Tensor | None

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        normalize_basis: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize_basis = normalize_basis
        self.weight = nn.Parameter(
            torch.empty((out_channels, in_channels, 9), device=device, dtype=dtype)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_channels, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return equi_linear(x, self.weight, self.bias, self.normalize_basis)

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"bias={self.bias is not None}, normalize_basis={self.normalize_basis}"
        )
