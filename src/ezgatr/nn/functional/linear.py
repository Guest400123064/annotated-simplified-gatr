from __future__ import annotations

import functools
import os
import pathlib
from typing import Literal

import torch
import torch.nn.functional as F

_BASIS_FNAME: dict[str, str] = {
    "gp": os.path.join("basis", "geometric_product.pt"),
    "op": os.path.join("basis", "outer_product.pt"),
}
_BASIS_CACHE: dict[str, dict[tuple[torch.device, torch.dtype], torch.Tensor]] = {
    "gp": {},
    "op": {},
}


def _load_bilinear_basis(
    kind: Literal["gp", "op"], device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    r"""Load the bilinear basis for geometric product or outer product.

    The bilinear basis is a 3D tensor with shape (16, 16, 16). One can
    understand the basis tensor by thinking of how the calculation of
    the bilinear maps between two multi-vectors expressed as coefficient
    vectors w.r.t. k-blades. When two multi-vectors are multiplied, each
    new coefficient corresponding to the k-blade comes from the "cartesian
    product" of the coefficients of the two multi-vectors. For example,
    the coefficient of the 1-blade ``e_1`` comes from multiple sources
    including ``gp(e_1, 1)`` and ``gp(e_0, e_01)``. So, the basis tensor
    actually defines a "computation graph" for the bilinear maps.

    The source basis are stored under the ``basis`` directory in `torch.float32`.
    The basis loader use the ``torch.float32`` tensor loaded to CPU as the prototype
    for all other devices and data types.

    Parameters
    ----------
    kind : Literal["gp", "op"]
        Kind of the bilinear basis, ``"gp"`` for geometric product and
        ``"op"`` for outer (wedge) product.
    device : torch.device
        Device for the basis.
    dtype : torch.dtype
        Data type for the basis.

    Returns
    -------
    torch.Tensor
        Bilinear basis with shape (16, 16, 16).
    """
    cache = _BASIS_CACHE[kind]
    basis = cache.get((device, dtype))
    if basis is not None:
        return basis

    proto_key = (torch.device("cpu"), torch.float32)
    try:
        cache[(device, dtype)] = cache[proto_key].detach().to(device, dtype)
    except KeyError:
        cache[proto_key] = torch.load(
            pathlib.Path(__file__).parent.resolve() / _BASIS_FNAME[kind],
            weights_only=True,
        )

    return _load_bilinear_basis(kind, device, dtype)


@functools.lru_cache(maxsize=None, typed=True)
def _compute_inner_product_selector(
    device: torch.device, keep_tri_vector: bool = True
) -> torch.Tensor:
    r"""Load the indices for PGA inner product to the device.

    PGA inner product operation exclude the coefficients corresponding to
    basis containing ``e_0``. The indices are hard-coded here. The reason
    to have this cached function is to avoid repeated copying from CPU to
    target multi-vector device using the ``torch.Tensor.to`` method.

    Parameters
    ----------
    device : torch.device
        Device for the indices.
    keep_tri_vector : bool
        Drop the tri-vector index (``e_123``) if set to ``False``. This is
        used in the inner-product component of the geometric attention.

    Returns
    -------
        torch.Tensor
    """
    idx = [0, 2, 3, 4, 8, 9, 10, 14]
    if not keep_tri_vector:
        idx.pop(-1)
    return torch.tensor(idx, device=device)


@functools.lru_cache(maxsize=None, typed=True)
def _compute_pin_equi_linear_basis(
    device: torch.device, dtype: torch.dtype, normalize: bool = True
) -> torch.Tensor:
    r"""Constructs basis elements for Pin(3, 0, 1)-equivariant linear maps
    between multi-vectors.

    Parameters
    ----------
    device : torch.device
        Device for the basis.
    dtype : torch.dtype
        Data type for the basis.
    normalize : bool
        Whether to normalize the basis elements according to
        the number of "inflow paths".

    Returns
    -------
    torch.Tensor
        Basis with shape (9, 16, 16) for equivariant linear maps.
    """
    basis_elements = [
        [0],
        [1, 2, 3, 4],
        [5, 6, 7, 8, 9, 10],
        [11, 12, 13, 14],
        [15],
        [(1, 0)],
        [(5, 2), (6, 3), (7, 4)],
        [(11, 8), (12, 9), (13, 10)],
        [(15, 14)],
    ]

    basis = []
    for elements in basis_elements:
        w = torch.zeros((16, 16))
        for element in elements:
            if isinstance(element, tuple):
                i, j = element
                w[i, j] = 1.0
            else:
                w[element, element] = 1.0

        if normalize:
            w /= torch.linalg.norm(w)

        w = w.unsqueeze(0)
        basis.append(w)

    return torch.cat(basis, dim=0).to(device, dtype)


def geometric_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    r"""Geometric product between two batches of multi-vectors.

    The input tensors ``x`` and ``y`` are multi-vectors with shape (..., 16).
    where ``...`` dimensions can denote batches or batches plus channels.
    When channel dimensions are present, the geometric product is calculated
    channel-wise (and batch-wise). For instance, the first channel of ``x[0]``
    is multiplied with the first channel of ``y[0]``, and so on. No channel-mixing
    here.

    Parameters
    ----------
    x : torch.Tensor
        Multi-vectors with shape (..., 16).
    y : torch.Tensor
        Multi-vectors with shape (..., 16).

    Returns
    -------
    torch.Tensor
        Multi-vectors with shape (..., 16).
    """
    basis = _load_bilinear_basis("gp", x.device, x.dtype)
    return torch.einsum("ijk, ...j, ...k -> ...i", basis, x, y)


def outer_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    r"""Outer product between two batches of multi-vectors.

    The input tensors ``x`` and ``y`` are multi-vectors with shape (..., 16).
    where ``...`` dimensions can denote batches or batches plus channels. When
    channel dimensions are present, the outer product is calculated channel-wise
    (and batch-wise). For instance, the first channel of ``x[0]`` is multiplied
    with the first channel of ``y[0]``, and so on. No channel-mixing here.

    Parameters
    ----------
    x : torch.Tensor
        Multi-vectors with shape (..., 16).
    y : torch.Tensor
        Multi-vectors with shape (..., 16).

    Returns
    -------
    torch.Tensor
        Multi-vectors with shape (..., 16).
    """
    basis = _load_bilinear_basis("op", x.device, x.dtype)
    return torch.einsum("ijk, ...j, ...k -> ...i", basis, x, y)


def inner_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    r"""Compute the PGA inner product between two multi-vectors.

    Similarly, the PGA inner product is calculated channel-wise (and batch-wise).
    No channel-mixing here.

    Parameters
    ----------
    x : torch.Tensor
        Multi-vectors with shape (..., 16).
    y : torch.Tensor
        Multi-vectors with shape (..., 16).

    Returns
    -------
    torch.Tensor
        Inner product results of the multi-vectors with shape (..., 1).
    """
    selector = _compute_inner_product_selector(x.device)
    return torch.einsum(
        "...i, ...i -> ...",
        torch.index_select(x, -1, selector),
        torch.index_select(y, -1, selector),
    ).unsqueeze(-1)


def equi_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    normalize_basis: bool = True,
) -> torch.Tensor:
    r"""Perform Pin-equivariant linear map defined by weight on input x.

    The equivariant linear map is a channel-mixing operation. Using a
    parallel analogy with a regular ``nn.Linear`` layer, each multi-vector
    channel of a input tensor corresponds to a "feature value" of a simple
    hidden representation, and the number of output channels is the number
    of neurons in that hidden layer.

    Within each channel, similar to the geometric product implementation,
    the linear operation starts with a matrix multiplication between the
    (weighted, if normalized) 16-by-16 computation graph, i.e., the basis,
    and the input multi-vector to take into account the effect that blades
    containing ``e_0`` will be "downgraded" to a lower grade after the map.
    Again, a map from "source-to-destination" style. Note that we may want
    to optimize this operation as the basis is pretty sparse.

    Parameters
    ----------
    x : torch.Tensor with shape (..., in_channels, 16)
        Input multivector. Batch dimensions must be broadcastable between
        x and weight.
    weight : torch.Tensor with shape (out_channels, in_channels, 9)
        Coefficients for the 9 basis elements. Batch dimensions must be
        broadcastable between x and weight.
    bias : torch.Tensor with shape (out_channels,)
        Bias for the linear map. The bias values are only added to the
        scalar blades (i.e., index position 0) of each output channel.
    normalize_basis : bool
        Whether to normalize the basis elements according to
        the number of "inflow paths" of each blade.

    Returns
    -------
    torch.Tensor
        Output with shape (..., out_channels, 16).
    """
    basis = _compute_pin_equi_linear_basis(x.device, x.dtype, normalize_basis)

    # The abbreviations are as follows:
    #   o: output channels
    #   i: input channels
    #   w: learnable parameters
    #   d: destination blades
    #   s: source blades
    x = torch.einsum("oiw, wds, ...is -> ...od", weight, basis, x)
    if bias is not None:
        x[..., [0]] += bias.view(-1, 1)
    return x


def dense_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Reduce multi-vectors with a linear map applied to scalar and ``e_0``.

    This operation is not equivariant. The output of this operation contains
    only scalars therefore **the blade dimension is squeezed**.

    Parameters
    ----------
    x : torch.Tensor with shape (..., in_channels, 16)
        Input multivector. Batch dimensions must be broadcastable between
        x and weight.
    weight : torch.Tensor with shape (out_channels, in_channels * 2)
        Coefficients for the scalar and ``e_0`` elements.
    bias : torch.Tensor with shape (out_channels,), optional
        Bias for the linear map.

    Returns
    -------
    torch.Tensor
        Output with shape (..., out_channels).
    """
    return F.linear(torch.flatten(x[..., :2], -2), weight, bias)
