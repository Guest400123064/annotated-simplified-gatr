from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from ezgatr.nn import EquiLinear, EquiRMSNorm
from ezgatr.nn.functional.dual import equi_join
from ezgatr.nn.functional.linear import geometric_product
from ezgatr.nn.functional.activation import scaler_gated_gelu


@dataclass
class ModelConfig:
    """Configuration class for the ``MVOnlyGATr`` model.

    Parameters
    ----------
    size_context : int, default to 2048
        Number of elements, e.g., number of points in a point cloud,
        in the input sequence.
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
    norm_eps : Optional[float], default to None
        Small value to prevent division by zero in the normalization layer.
    norm_channelwise_rescale : bool, default to True
        Apply learnable channel-wise rescaling weights to the normalized multi-vector
        inputs. Initialized to ones if set to ``True``.
    """

    size_context: int = 2048

    size_channels_in: int = 1
    size_channels_out: int = 1
    size_channels_hidden: int = 32
    size_channels_intermediate: int = 32

    norm_eps: Optional[float] = None
    norm_channelwise_rescale: bool = True


class Embedding(nn.Module):
    """Embedding layer to project input number of channels to hidden channels.

    This layer corresponds to the very first equivariant linear layer of the
    original design mentioned in the GATr paper. 

    Parameters
    ----------
    config : ModelConfig
        Configuration object for the model. See ``ModelConfig`` for more details.
    """

    config: ModelConfig
    embedding: EquiLinear

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.config = config

        self.embedding = EquiLinear(
            config.size_channels_in, config.size_channels_hidden
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)


class Bilinear(nn.Module):
    """Implements the geometric bilinear sub-layer of the geometric MLP.

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
    config : ModelConfig
        Configuration object for the model. See ``ModelConfig`` for more details.
    """

    config: ModelConfig
    proj_bil: EquiLinear
    proj_out: EquiLinear

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.config = config
        if config.size_channels_intermediate % 2 != 0:
            raise ValueError("Number of hidden channels must be even.")

        self.proj_bil = EquiLinear(
            config.size_channels_hidden, config.size_channels_intermediate * 2
        )
        self.proj_out = EquiLinear(
            config.size_channels_intermediate, config.size_channels_hidden
        )

    def forward(
        self, x: torch.Tensor, reference: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass of the geometric bilinear sub-layer.

        Parameters
        ----------
        x : torch.Tensor
            Batch of input hidden multi-vector representation tensor.
        reference : Optional[torch.Tensor], default to None
            Reference tensor for the equivariant join operation.

        Returns
        -------
        torch.Tensor
            Batch of output hidden multi-vector representation tensor of the
            same number of hidden channels.
        """
        size_inter = self.config.size_channels_intermediate // 2
        lg, rg, lj, rj = torch.split(self.proj_bil(x), size_inter, dim=-2)

        x = torch.cat([geometric_product(lg, rg), equi_join(lj, rj, reference)], dim=-2)
        return self.proj_out(x)


class MLP(nn.Module):
    """Geometric MLP block without scaler channels.

    Here we fix the structure of the MLP block to be a single equivariant linear
    projection followed by a gated GELU activation function. In addition, the
    equivariant normalization layer can be configured to be learnable, so the
    normalization layer needs to be included in the block instead of being shared
    across the network.

    Parameters
    ----------
    config : ModelConfig
        Configuration object for the model. See ``ModelConfig`` for more details.
    """

    config: ModelConfig
    layer_norm: EquiRMSNorm
    equi_bil: Bilinear
    proj_out: EquiLinear

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.config = config

        self.layer_norm = EquiRMSNorm(
            config.size_channels_hidden,
            eps=config.norm_eps,
            channelwise_rescale=config.norm_channelwise_rescale,
        )
        self.equi_bil = Bilinear(config)
        self.proj_out = EquiLinear(
            config.size_channels_hidden, config.size_channels_hidden
        )

    def forward(
        self, x: torch.Tensor, reference: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass of the geometric MLP block.

        Parameters
        ----------
        x : torch.Tensor
            Batch of input hidden multi-vector representation tensor.
        reference : Optional[torch.Tensor], default to None
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
        x = self.proj_out(scaler_gated_gelu(x))

        return x + residual


class Attention(nn.Module):
    """Geometric attention block with scaler channels."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        return x + residual


class TransformerBlock(nn.Module):
    pass


class Transformer(nn.Module):
    pass
