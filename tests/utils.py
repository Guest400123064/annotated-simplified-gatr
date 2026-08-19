from __future__ import annotations

import torch
import numpy as np
import clifford
from clifford import pga
from hypothesis import strategies


def mv_to_tensor(
    mvs: list[clifford.MultiVector], batch_dims: tuple[int, ...] | None = None
) -> torch.Tensor:
    r"""Convert a list of multi-vectors to a batched tensor.

    Parameters
    ----------
    mvs : list[cf.MultiVector]
        List of clifford PGA multi-vectors.
    batch_dims: tuple[int] | None
        The dimensions of the batch. Note that this is a general 'batch',
        i.e., different objects within a sequence or different channels
        are also considered batch dimension. If None, a single multi-vector
        is generated.

    Returns
    -------
    torch.Tensor with shape (*batch_dims, 16)
        Batched tensor of multi-vectors.
    """
    ret = (
        torch.from_numpy(np.array([mv.value for mv in mvs]))
            .to(torch.float32)
            .reshape(*(batch_dims or (1,)), 16)
    )
    return ret


def make_random_pga_mvs(
    batch_dims: tuple[int, ...] | None = None, rng: int = 42
) -> list[clifford.MultiVector]:
    r"""Generate batches of random multi-vectors for testing.

    Parameters
    ----------
    batch_dims : tuple[int] | None
        The dimensions of the batch. Note that this is a general 'batch',
        i.e., different objects within a sequence or different channels
        are also considered batch dimension. If None, a single multi-vector
        is generated.
    rng : int, default to 42
        Random seed for reproducibility.

    Returns
    -------
    list[clifford.MultiVector]
        List of random multi-vectors.
    """
    batch_dims = batch_dims or (1,)
    ret = clifford.randomMV(
        layout=pga.layout, n=np.prod(batch_dims), rng=rng
    )
    if isinstance(ret, clifford.MultiVector):
        return [ret]
    return ret


def make_random_3d_vectors(batch_dims: tuple[int, ...]):
    r"""Generate batches of random 3D vectors for testing.

    This utility function can be used to generate random 3D vectors
    for testing point cloud, plane normal vectors, translations, etc.
    """
    return torch.randn(*batch_dims, 3)


def strategy_batch_dims(
    max_dims: int = 3, min_size: int = 1, max_size: int = 2048
):
    r"""A hypothesis strategy for generating tensor shapes.

    Parameters
    ----------
    max_dims : int, default to 3
        Maximum number of dimensions.
    min_size : int
        Minimum size of each dimension.
    max_size : int
        Maximum size of each dimension.

    Returns
    -------
    strategies.lists
        A strategy for generating tensor shapes.
    """
    return strategies.lists(
        strategies.integers(min_size, max_size),
        min_size=1,
        max_size=max_dims,
    )
