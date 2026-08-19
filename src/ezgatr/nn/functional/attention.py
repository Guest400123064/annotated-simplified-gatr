from __future__ import annotations

import functools
from typing import Any, Literal

import torch
import torch.nn.functional as F
from einops import rearrange

from ezgatr.nn.functional.linear import _compute_inner_product_selector

GeometricQKVType = torch.Tensor | tuple[torch.Tensor, torch.Tensor]
GeometricAttnKindType = Literal["ipa", "daa"]


def _flatten_ck(mv: torch.Tensor) -> torch.Tensor:
    """A shortcut to flatten the channel and blade dimensions."""

    return rearrange(mv, "... c k -> ... (c k)")


def _inflate_ck(mv: torch.Tensor) -> torch.Tensor:
    """A shortcut to inflate the channel and blade dimensions."""

    return rearrange(mv, "... (c k) -> ... c k", k=16)


@functools.lru_cache(maxsize=None, typed=True)
def _compute_tri_vector_selector(device: torch.device) -> torch.Tensor:
    r"""Load the indices corresponds to tri-vectors to the device.

    Parameters
    ----------
    device : torch.device
        Device for the indices.

    Returns
    -------
        torch.Tensor
    """
    return torch.tensor([11, 12, 13, 14], device=device)


@functools.lru_cache(maxsize=None, typed=True)
def _compute_daa_qk_basis(
    device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Compute basis queries and keys in the distance-aware attention.

    Parameters
    ----------
    device: torch.device
        Device for the basis.
    dtype: torch.dtype
        Data type for the basis.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Basis tensor for the queries and keys in the distance-aware attention.
        Both with shape (4, 4, 5).
    """
    bq = torch.zeros((4, 4, 5), device=device, dtype=dtype)
    bk = torch.zeros((4, 4, 5), device=device, dtype=dtype)
    r3 = torch.arange(3, device=device)

    bq[r3, r3, 0] = 1.0
    bk[3, 3, 0] = -1.0

    bq[3, 3, 1] = 1.0
    bk[r3, r3, 1] = -1.0

    bq[r3, 3, r3 + 2] = 1.0
    bk[r3, 3, r3 + 2] = 2.0

    return bq, bk


def _linear_square_normalizer(e123: torch.Tensor, eps: float) -> torch.Tensor:
    r"""Apply linear square normalization to the input tensor.

    Parameters
    ----------
    e123 : torch.Tensor
        Coefficients corresponds to the ``e_123`` blade.
    eps : float
        Small value to avoid division by zero.

    Returns
    -------
    torch.Tensor
        Normalized multi-vector tensor.
    """
    return e123 / (e123.pow(2) + eps)


def compute_qk_for_daa(
    query: torch.Tensor,
    key: torch.Tensor,
    eps: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Compute the query and key tensors for the distance-aware attention.

    Parameters
    ----------
    query : torch.Tensor
        Query tensor with shape (B, H, T, qk_channels, 16).
    key : torch.Tensor
        Key tensor with shape (B, H, T, qk_channels, 16).
    eps : float, default to ``1e-3``
        Small value to avoid division by zero used in the linear square normalizer.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Query and key tensors for the equivariant distance-aware attention.
        Blade dimensions are **NOT** flattened.
    """

    def _build_dist_vec(q_or_k, basis):
        sel = _compute_tri_vector_selector(query.device)
        tri = torch.index_select(q_or_k, -1, sel)
        ret = tri * _linear_square_normalizer(tri[..., [3]], eps=eps)
        return torch.einsum("ijk, ...i, ...j -> ...k", basis, ret, ret)

    bq, bk = _compute_daa_qk_basis(query.device, query.dtype)
    return _build_dist_vec(query, bq), _build_dist_vec(key, bk)


def compute_qk_for_ipa(
    query: torch.Tensor, key: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Compute the query and key tensors for the inner product attention.

    Parameters
    ----------
    query : torch.Tensor
        Query tensor with shape (B, H, T, qk_channels, 16).
    key : torch.Tensor
        Key tensor with shape (B, H, T, qk_channels, 16).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Query and key tensors for the equivariant inner product attention.
        Blade dimensions are **NOT** flattened.
    """

    def _build_inner_vec(q_or_k):
        sel = _compute_inner_product_selector(
            q_or_k.device,
            keep_tri_vector=False,
        )
        return torch.index_select(q_or_k, -1, sel)

    return _build_inner_vec(query), _build_inner_vec(key)


_ATTENTION_KIND_DISPATCH = {
    "ipa": compute_qk_for_ipa,
    "daa": compute_qk_for_daa,
}


def equi_geometric_attention(
    query: GeometricQKVType,
    key: GeometricQKVType,
    value: GeometricQKVType,
    kinds: dict[GeometricAttnKindType, dict[str, Any] | None],
    weight: list[torch.Tensor | float] | None = None,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    r"""Equivariant geometric attention.

    Parameters
    ----------
    query : GeometricQKVType
        Multi-vector query tensor with shape (B, H, T, qk_channels, 16). If
        scalar channel tensors are supplied, they should be included in a tuple
        with the multi-vector tensors, i.e., ``(q_mv, q_s)``. The scalar channel
        tensors should have shape (B, H, T, qk_scalar_dim).
    key : GeometricQKVType
        Multi-vector key tensor with shape (B, H, T, qk_channels, 16). If
        scalar channel tensors are supplied, they should be included in a tuple
        with the multi-vector tensors. The scalar channel tensors should have
        shape (B, H, T, qk_scalar_dim).
    value : GeometricQKVType
        Multi-vector value tensor with shape (B, H, T, qk_channels, 16). If
        scalar channel tensors are supplied, they should be included in a tuple
        with the multi-vector tensors. The scalar channel tensors should have
        shape (B, H, T, v_scalar_dim).
    kinds : dict[GeometricAttnKindType, dict[str, Any] | None]
        Kinds of similarity measures to consider in the attention calculation
        along with additional configuration/parameters sent to the corresponding
        query-key generating function. One should supply a dictionary mapping
        from the kind to parameters in addition to query and key tensors. Use
        ``None`` to denote no additional parameters supplied. Available options:
        - "ipa": Inner product attention
        - "daa": Distance-aware attention
    weight : list[torch.Tensor | float], optional
        Weight tensor for the attention kinds. If not provided, the weights are
        set to 1.0 for all kinds to represent equal importance. **Note that the
        weight tensors are NOT applied to the scalar inputs (if provided).** If
        scalar channel tensors are supplied, weights are fixed to 1.0.
    attn_mask : torch.Tensor, optional
        Attention mask tensor.
    dropout_p : float, default to 0.0
        Dropout probability.
    is_causal : bool, default to False
        Whether to apply causal masking.
    scale : float, optional
        Scaling factor for the attention scores, overwriting the default scale
        determined by the hidden dimension.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor | None]
        Return a tuple with the first element being the multi-vector channel tensor
        of shape (B, H, T, v_channels, 16). If scalar channel tensors are supplied,
        the second element is the scalar channel tensor of shape
        (B, H, T, v_scalar_dim), otherwise, it is ``None``.
    """
    qs: list[torch.Tensor] = []
    ks: list[torch.Tensor] = []

    if isinstance(query, tuple):
        try:
            (query, key, value), (query_scl, key_scl, value_scl) = zip(
                query, key, value
            )
        except ValueError:
            raise ValueError(
                "Error unpacking the query, key, and value tensors. "
                "Please make sure the query, key, and value tensors are ALL supplied as tuples "
                "of the form (multi-vectors, scalars) if scalar channel presents."
            )

        # We can safely append the scalar channel tensors to the list before multi-vector
        # channel tensors, as all similarities will be condensed by dot-product, i.e., the
        # order of the channels does not matter as long as query-key correspondence is kept.
        qs.append(query_scl)
        ks.append(key_scl)

        # Save the index for the scalar channel tensors to separate them from the multi-vector
        # channel tensors after the attention calculation.
        value = torch.cat([_flatten_ck(value), value_scl], dim=-1)
        index_scl = -value_scl.shape[-1]
    else:
        assert not isinstance(value, tuple)
        value = _flatten_ck(value)
        index_scl = None

    # After the branches above, query, key, and value are all plain multi-vector
    # tensors: either unpacked from the tuples or passed in directly.
    assert isinstance(query, torch.Tensor) and isinstance(key, torch.Tensor)

    # Weights are only applied to the multi-vector channel tensors even if
    # scalar channel tensors are supplied. The scalar channel weights would
    # be redundant because other weights can adjust themselves accordingly.
    weight = weight or [1.0] * len(kinds)
    if len(kinds.keys()) != len(weight):
        raise ValueError(
            f"The length of the kinds and weight must be the same. Got {len(kinds)} "
            f"kinds and {len(weight)} weights."
        )
    for (kind, kwargs), w in zip(kinds.items(), weight):
        q, k = _ATTENTION_KIND_DISPATCH[kind](query, key, **(kwargs or {}))
        qs.append(_flatten_ck(q * w))
        ks.append(_flatten_ck(k))

    ret = F.scaled_dot_product_attention(
        torch.cat(qs, dim=-1),
        torch.cat(ks, dim=-1),
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )
    if index_scl is not None:
        return _inflate_ck(ret[..., :index_scl]), ret[..., index_scl:]
    return _inflate_ck(ret), None
