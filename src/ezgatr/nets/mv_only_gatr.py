from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import reduce
from typing import Any, Literal

import torch
from einops import rearrange
from torch import nn

from ezgatr.nn import EquiLinear, EquiRMSNorm
from ezgatr.nn.functional import (
    equi_geometric_attention,
    equi_join,
    geometric_product,
    scaler_gated_gelu,
)


@dataclass
class MVOnlyGATrConfig:
    r"""Configuration class for the ``MVOnlyGATr`` model.

    Parameters
    ----------
    num_layers : int, default to 4
        Number of GATr blocks in the network.
    size_context : int, default to 2048
        Number of elements, e.g., number of points in a point cloud,
        in the input sequence. This argument is not actually used in
        the model, but is kept for compatibility.
    size_channels_in : int, default to 1
        Number of input channels.
    size_channels_out : int, default to 1
        Number of output channels.
    size_channels_hidden : int, default to 32
        Number of hidden representation channels throughout the network, i.e.,
        the input/output number of channels of the next layer, block, or module.
    size_channels_intermediate : int, default to 32
        Number of intermediate channels for the geometric bilinear operation.
        Must be even. This intermediate size should not be confused with the size
        of hidden representations throughout the network. It only refers to the
        hidden sizes used for the equivariant join and geometric product operations.
    norm_eps : float, optional
        Small value to prevent division by zero in the normalization layer.
    norm_channelwise_rescale : bool, default to True
        Apply learnable channel-wise rescaling weights to the normalized multi-vector
        inputs. Initialized to ones if set to ``True``.
    gelu_approximate : str, default to "tanh"
        Approximation method for the GeLU function. Default to "tanh".
    """

    num_layers: int = 4
    size_context: int = 2048

    size_channels_in: int = 1
    size_channels_out: int = 1
    size_channels_hidden: int = 32
    size_channels_intermediate: int = 32

    attn_num_heads: int = 4
    attn_kinds: dict[Literal["ipa", "daa"], dict[str, Any] | None] = field(
        default_factory=lambda: {"ipa": None, "daa": None}
    )
    attn_dropout_p: float = 0.0
    attn_is_causal: bool = True
    attn_scale: float | None = None

    norm_eps: float | None = None
    norm_channelwise_rescale: bool = True

    gelu_approximate: Literal["none", "tanh"] = "tanh"


class MVOnlyGATrEmbedding(nn.Module):
    r"""Embedding layer to project input number of channels to hidden channels.

    This layer corresponds to the very first equivariant linear layer of the
    original design mentioned in the GATr paper.

    Parameters
    ----------
    config : MVOnlyGATrConfig
        Configuration object for the model. See ``MVOnlyGATrConfig`` for more details.
    """

    config: MVOnlyGATrConfig
    embedding: EquiLinear

    def __init__(self, config: MVOnlyGATrConfig) -> None:
        super().__init__()

        self.config = config

        self.embedding = EquiLinear(
            config.size_channels_in, config.size_channels_hidden
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] != self.config.size_channels_in:
            raise ValueError(
                f"Input tensor has {x.shape[-2]} channels, "
                f"expected {self.config.size_channels_in}."
            )
        return self.embedding(x)


class MVOnlyGATrBilinear(nn.Module):
    r"""Implements the geometric bilinear sub-layer of the geometric MLP.

    Geometric bilinear operation consists of geometric product and equivariant
    join operations. The results of two operations are concatenated along the
    hidden channel axis and passed through a final equivariant linear projection
    before being passed to the next layer, block, or module.

    In both geometric product and equivariant join operations, the input
    multi-vectors are first projected to a hidden space with the same number of
    channels, i.e., left and right. Then, the results of each operation are
    derived from the interaction of left and right hidden representations, each
    with half number of ``size_channels_intermediate``.

    Parameters
    ----------
    config : MVOnlyGATrConfig
        Configuration object for the model. See ``MVOnlyGATrConfig`` for more details.
    """

    config: MVOnlyGATrConfig
    proj_bil: EquiLinear
    proj_out: EquiLinear

    def __init__(self, config: MVOnlyGATrConfig) -> None:
        super().__init__()

        self.config = config

        self.proj_bil = EquiLinear(
            config.size_channels_hidden, config.size_channels_intermediate * 4
        )
        self.proj_out = EquiLinear(
            config.size_channels_intermediate * 2, config.size_channels_hidden
        )

    def forward(
        self, x: torch.Tensor, reference: torch.Tensor | None = None
    ) -> torch.Tensor:
        r"""Forward pass of the geometric bilinear sub-layer.

        Parameters
        ----------
        x : torch.Tensor
            Batch of input hidden multi-vector representation tensor.
        reference : torch.Tensor, optional
            Reference tensor for the equivariant join operation.

        Returns
        -------
        torch.Tensor
            Batch of output hidden multi-vector representation tensor of the
            same number of hidden channels.
        """
        size_inter = self.config.size_channels_intermediate
        lg, rg, lj, rj = torch.split(self.proj_bil(x), size_inter, dim=-2)

        x = torch.cat([geometric_product(lg, rg), equi_join(lj, rj, reference)], dim=-2)
        return self.proj_out(x)


class MVOnlyGATrMLP(nn.Module):
    r"""Geometric MLP block without scaler channels.

    Here we fix the structure of the MLP block to be a single equivariant linear
    projection followed by a gated GELU activation function. In addition, the
    equivariant normalization layer can be configured to be learnable, so the
    normalization layer needs to be included in the block instead of being shared
    across the network.

    Parameters
    ----------
    config : MVOnlyGATrConfig
        Configuration object for the model. See ``MVOnlyGATrConfig`` for more details.
    """

    config: MVOnlyGATrConfig
    layer_norm: EquiRMSNorm
    equi_bil: MVOnlyGATrBilinear
    proj_out: EquiLinear

    def __init__(self, config: MVOnlyGATrConfig) -> None:
        super().__init__()

        self.config = config

        self.layer_norm = EquiRMSNorm(
            config.size_channels_hidden,
            eps=config.norm_eps,
            channelwise_rescale=config.norm_channelwise_rescale,
        )
        self.equi_bil = MVOnlyGATrBilinear(config)
        self.proj_out = EquiLinear(
            config.size_channels_hidden, config.size_channels_hidden
        )

    def forward(
        self, x: torch.Tensor, reference: torch.Tensor | None = None
    ) -> torch.Tensor:
        r"""Forward pass of the geometric MLP block.

        Parameters
        ----------
        x : torch.Tensor
            Batch of input hidden multi-vector representation tensor.
        reference : torch.Tensor, optional
            Reference tensor for the equivariant join operation.

        Returns
        -------
        torch.Tensor
            Batch of output hidden multi-vector representation tensor of the
            same number of hidden channels.
        """
        residual = x

        x = self.layer_norm(x)
        x = self.equi_bil(x, reference)
        x = self.proj_out(scaler_gated_gelu(x, self.config.gelu_approximate))

        return x + residual


class MVOnlyGATrAttention(nn.Module):
    r"""Geometric attention block without scaler channels.

    The GATr attention calculation is slightly different from the original
    transformers implementation in that each head has the sample number of
    channels as the input tensor, instead of dividing into smaller chunks.
    In this case, the final output linear transformation maps from
    ``size_channels_hidden * attn_num_heads`` to ``size_channels_hidden``.

    One additional note here is that the ``attn_mix`` parameter is a dictionary
    of learnable weighting parameter **LOGITS** for each attention kind.
    They will be exponentiated before being used in the attention calculation.

    Parameters
    ----------
    config : MVOnlyGATrConfig
        Configuration object for the model. See ``MVOnlyGATrConfig`` for more details.
    """

    config: MVOnlyGATrConfig
    layer_norm: EquiRMSNorm
    attn_mix: dict[str, torch.Tensor]
    proj_qkv: EquiLinear

    def __init__(self, config: MVOnlyGATrConfig) -> None:
        super().__init__()

        self.config = config

        self.layer_norm = EquiRMSNorm(
            config.size_channels_hidden,
            eps=config.norm_eps,
            channelwise_rescale=config.norm_channelwise_rescale,
        )

        # The two dummy dimensions are for the sequence length
        # and blade dimension, respectively.
        attn_mix_shape = (config.attn_num_heads, 1, config.size_channels_hidden, 1)
        self.attn_mix = {}
        for kind in config.attn_kinds:
            param = nn.Parameter(torch.zeros(attn_mix_shape, dtype=torch.float32))
            self.attn_mix[kind] = param
            self.register_parameter(f"attn_mix_{kind}", param)

        self.proj_qkv = EquiLinear(
            config.size_channels_hidden,
            config.size_channels_hidden * config.attn_num_heads * 3,
        )
        self.proj_out = EquiLinear(
            config.size_channels_hidden * config.attn_num_heads,
            config.size_channels_hidden,
        )

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        r"""Forward pass of the geometric attention block.

        Parameters
        ----------
        x : torch.Tensor
            Batch of input hidden multi-vector representation tensor.
        attn_mask : torch.Tensor, optional
            Attention mask tensor for the attention operation. Usually
            used if any specific attention constraints are needed within
            a single sequence, such as padding mask or for discriminating
            different subsequences.

        Returns
        -------
        torch.Tensor
            Batch of output hidden multi-vector representation tensor of the
            same number of hidden channels.
        """
        residual = x

        x = self.layer_norm(x)
        q, k, v = rearrange(
            self.proj_qkv(x),
            "b t (qkv h c) k -> qkv b h t c k",
            qkv=3,
            h=self.config.attn_num_heads,
            c=self.config.size_channels_hidden,
        )
        x, _ = equi_geometric_attention(
            q,
            k,
            v,
            kinds=self.config.attn_kinds,
            weight=[w.exp() for w in self.attn_mix.values()],
            attn_mask=attn_mask,
            is_causal=self.config.attn_is_causal,
            dropout_p=self.config.attn_dropout_p,
            scale=self.config.attn_scale,
        )
        x = rearrange(x, "b h t c k -> b t (h c) k", h=self.config.attn_num_heads)
        x = self.proj_out(x)

        return x + residual


class MVOnlyGATrBlock(nn.Module):
    r"""GATr block without scaler channels.

    Parameters
    ----------
    config : MVOnlyGATrConfig
        Configuration object for the model. See ``MVOnlyGATrConfig`` for more details.
    layer_id : int
        Index of the current block in the network.
    """

    config: MVOnlyGATrConfig
    layer_id: int
    mlp: MVOnlyGATrMLP
    attn: MVOnlyGATrAttention

    def __init__(self, config: MVOnlyGATrConfig, layer_id: int) -> None:
        super().__init__()

        self.config = config
        self.layer_id = layer_id

        self.mlp = MVOnlyGATrMLP(config)
        self.attn = MVOnlyGATrAttention(config)

    def forward(
        self,
        x: torch.Tensor,
        reference: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.mlp(self.attn(x, attn_mask), reference)


class MVOnlyGATrModel(nn.Module):
    r"""Multi-Vector only GATr model.

    Parameters
    ----------
    config : MVOnlyGATrConfig
        Configuration object for the model. See ``MVOnlyGATrConfig`` for more details.
    """

    config: MVOnlyGATrConfig
    embedding: MVOnlyGATrEmbedding
    blocks: nn.ModuleList
    head: EquiLinear

    def __init__(self, config: MVOnlyGATrConfig) -> None:
        super().__init__()

        self.config = config

        self.embedding = MVOnlyGATrEmbedding(config)
        self.blocks = nn.ModuleList(
            MVOnlyGATrBlock(config, i) for i in range(config.num_layers)
        )
        self.head = EquiLinear(config.size_channels_hidden, config.size_channels_out)
        self.apply(self._init_params)

    def _init_params(self, module: nn.Module):
        r"""Slight adjustment to Kaiming init by down-scaling the weights
        by the number of encoder layers, following the GPT-2 paper.

        Parameters
        ----------
        module : nn.Module
            Module to initialize.
        """
        if isinstance(module, EquiLinear):
            nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
            module.weight.data /= math.sqrt(self.config.num_layers)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        reference: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        reference = reference or torch.mean(
            x,
            dim=tuple(range(1, len(x.shape) - 1)),
            keepdim=True,
        )
        x = reduce(
            lambda x, block: block(x, reference, attn_mask),
            self.blocks,
            self.embedding(x),
        )
        return self.head(x)
