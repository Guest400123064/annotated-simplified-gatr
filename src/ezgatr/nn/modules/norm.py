from __future__ import annotations

from typing import ClassVar

import torch
from torch import nn

from ezgatr.nn.functional.norm import equi_rms_norm


class EquiRMSNorm(nn.Module):
    r"""Applies PGA inner-induced RMS norm to a batch of multi-vectors.

    Parameters
    ----------
    in_channels : int
        Number of input multi-vector channels.
    eps : float, optional
        Small value to prevent division by zero.
    channelwise_rescale : bool, default to True
        Apply learnable channel-wise rescaling weights to the normalized
        multi-vector inputs. Initialized to ones.
    """

    __constants__: ClassVar[list[str]] = ["eps", "channelwise_rescale"]

    in_channels: int
    eps: float | None
    channelwise_rescale: bool
    weight: torch.Tensor | None

    def __init__(
        self,
        in_channels: int,
        eps: float | None = None,
        channelwise_rescale: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.eps = eps
        self.channelwise_rescale = channelwise_rescale
        if channelwise_rescale:
            self.weight = nn.Parameter(
                torch.empty(in_channels, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("weight", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.channelwise_rescale and (self.weight is not None):
            nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return equi_rms_norm(x, self.weight, self.eps)

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, "
            f"eps={self.eps}, "
            f"channelwise_rescale={self.channelwise_rescale}"
        )
