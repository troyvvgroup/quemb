"""
Perform the sparse DF ERI transformation using semi-sparse tensors.

This module provides functionality to transform ERIs using semi-sparse tensors.
"""

from __future__ import annotations

import typing

import numpy

__all__ = [
    "SemiSparse3DTensor",
    "SemiSparseSym3DTensor",
    "SemiSparseSym3DWindow",
    "apply_inv_cholesky_inplace",
    "get_AO_per_MO",
    "get_log_level",
    "set_log_level",
    "transform_integral_precontracted",
]

class SemiSparse3DTensor:
    def __getitem__(self, arg0: tuple[int, int]) -> numpy.ndarray: ...
    @typing.overload
    def __init__(
        self, shape: tuple[int, int, int], AO_reachable_by_MO: list[list[int]]
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        dense_data: numpy.ndarray,
        shape: tuple[int, int, int],
        AO_reachable_by_MO: list[list[int]],
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        dense_data: numpy.ndarray,
        shape: tuple[int, int, int],
        AO_reachable_by_MO: list[list[int]],
        AO_reachable_by_MO_with_offsets: list[list[tuple[int, int]]],
        offsets: dict[int, int],
    ) -> None: ...
    def __repr__(self) -> str: ...
    @property
    def AO_reachable_by_MO(self) -> list[list[int]]: ...
    @property
    def AO_reachable_by_MO_with_offsets(self) -> list[list[tuple[int, int]]]: ...
    @property
    def dense_data(self) -> numpy.ndarray: ...
    @property
    def mut_dense_data(self) -> numpy.ndarray: ...
    @property
    def nonzero_size(self) -> int: ...
    @property
    def offsets(self) -> dict[int, int]: ...
    @property
    def shape(self) -> tuple[int, int, int]: ...
    @property
    def size(self) -> int: ...

class SemiSparseSym3DTensor:
    """
    Immutable, semi-sparse, partially symmetric 3-index tensor

    Assumes:
      - T_{ijk} = T_{jik} symmetry
      - Sparsity over (i, j), dense over k
      - Example use: 3-center integrals (μν|P)
    """
    def __getitem__(self, arg0: tuple[int, int]) -> numpy.ndarray: ...
    @typing.overload
    def __init__(
        self, shape: tuple[int, int, int], exch_reachable: list[list[int]]
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        unique_dense_data: numpy.ndarray,
        shape: tuple[int, int, int],
        exch_reachable: list[list[int]],
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        unique_dense_data: numpy.ndarray,
        shape: tuple[int, int, int],
        exch_reachable: list[list[int]],
        exch_reachable_unique: list[list[int]],
        exch_reachable_with_offsets: list[list[tuple[int, int]]],
        exch_reachable_unique_with_offsets: list[list[tuple[int, int]]],
        offsets: dict[int, int],
    ) -> None: ...
    def __repr__(self) -> str: ...
    @property
    def exch_reachable(self) -> list[list[int]]: ...
    @property
    def exch_reachable_unique(self) -> list[list[int]]: ...
    @property
    def mut_unique_dense_data(self) -> numpy.ndarray: ...
    @property
    def nonzero_size(self) -> int: ...
    @property
    def offsets(self) -> dict[int, int]: ...
    @property
    def shape(self) -> tuple[int, int, int]: ...
    @property
    def size(self) -> int: ...
    @property
    def unique_dense_data(self) -> numpy.ndarray: ...

class SemiSparseSym3DWindow:
    """
    Rolling window of L⁻¹ (P | mu nu) columns for the on-the-fly sparse DF
    """
    def __init__(
        self, shape: tuple[int, int, int], capacity: int, L_PQ: numpy.ndarray
    ) -> None: ...
    def evict(self, keys: numpy.ndarray) -> None:
        """
        Free the columns of the given pairs.
        """
    def insert(self, keys: numpy.ndarray, raw: numpy.ndarray) -> None:
        """
        Store L⁻¹ (P | mu nu) for the pairs keys = ravel_symmetric(mu, nu),
        where raw (naux, len(keys)) holds their (P | mu nu) in Fortran order.
        raw is overwritten with L⁻¹ raw.
        """
    def set_active(self, keys: numpy.ndarray) -> None:
        """
        Set the sorted pairs that are visible to the transformation.
        """
    @property
    def capacity(self) -> int: ...
    @property
    def n_stored(self) -> int: ...
    @property
    def shape(self) -> tuple[int, int, int]: ...

def apply_inv_cholesky_inplace(
    int_P_mu_nu: SemiSparseSym3DTensor, L_PQ: numpy.ndarray
) -> None:
    """
    Overwrite (P | mu nu) in place with L⁻¹ (P | mu nu),
    where L_PQ is the lower Cholesky factor of (P | Q).
    """

def contract_with_TA_1st(
    TA: numpy.ndarray, int_P_mu_nu: SemiSparseSym3DTensor, AO_by_MO: list[list[int]]
) -> SemiSparse3DTensor:
    """
    Compute (P | mu i) = sum_nu (P | mu nu) TA(nu, i) for the AOs mu of every MO i,
    with one dense GEMM per AO mu.
    """

def contract_with_TA_2nd_to_sym_dense(
    int_mu_i_P: SemiSparse3DTensor, TA: numpy.ndarray
) -> numpy.ndarray:
    """
    Compute (P | i j) = sum_mu (P | mu i) TA(mu, j) for j >= i (the AOs of the smaller
    index i) in the symmetric pair index,
    with one dense GEMM per MO i.
    """

def get_AO_per_MO(
    TA: numpy.ndarray, S_abs: numpy.ndarray, epsilon: float
) -> list[list[int]]:
    """
    Get AOs per MO based on TA and S_abs matrices with a threshold epsilon
    """

def get_log_level() -> int: ...
def set_log_level(log_level: int) -> None: ...
def transform_integral_precontracted(
    int_P_mu_nu: SemiSparseSym3DTensor | SemiSparseSym3DWindow,
    TA: numpy.ndarray,
    S_abs: numpy.ndarray,
    MO_coeff_epsilon: float,
) -> numpy.ndarray:
    """
    Transform the integral using TA and int_P_mu_nu = L⁻¹ (P | mu nu),
    i.e. after apply_inv_cholesky_inplace or the active pairs of a window,
    returning the transformed matrix
    """
