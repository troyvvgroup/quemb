"""
Minimal pybind11 + Eigen example
"""

from __future__ import annotations

import typing

import numpy

__all__ = [
    "GPU_MatrixHandle",
    "SemiSparse3DTensor",
    "SemiSparseSym3DTensor",
    "contract_with_TA_1st",
    "contract_with_TA_2nd_to_sym_dense",
    "extract_unique",
    "get_AO_per_MO",
    "get_AO_reachable_by_MO_with_offset",
    "transform_integral",
    "transform_integral_cuda",
]

class GPU_MatrixHandle:
    def __init__(self, L_host: numpy.ndarray) -> None:
        """
        Create a GPU_MatrixHandle from a host matrix.

        This allocates memory on the GPU and copies the data from the host to the GPU.
        """
    def __repr__(self) -> str: ...

class SemiSparse3DTensor:
    def __getitem__(self, arg0: tuple[int, int]) -> numpy.ndarray: ...
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
    @property
    def AO_reachable_by_MO(self) -> list[list[int]]: ...
    @property
    def AO_reachable_by_MO_with_offsets(self) -> list[list[tuple[int, int]]]: ...
    @property
    def dense_data(self) -> numpy.ndarray: ...
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
    @property
    def exch_reachable(self) -> list[list[int]]: ...
    @property
    def exch_reachable_unique(self) -> list[list[int]]: ...
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

def contract_with_TA_1st(
    TA: numpy.ndarray, int_P_mu_nu: SemiSparseSym3DTensor, AO_by_MO: list[list[int]]
) -> SemiSparse3DTensor: ...
def contract_with_TA_2nd_to_sym_dense(
    int_mu_i_P: SemiSparse3DTensor, TA: numpy.ndarray
) -> numpy.ndarray:
    """
    Contract with TA to get a symmetric dense tensor (P | i, j)
    """

def extract_unique(exch_reachable: list[list[int]]) -> list[list[int]]:
    """
    Extract unique reachable AOs from the provided exch_reachable structure
    """

def get_AO_per_MO(
    TA: numpy.ndarray, S_abs: numpy.ndarray, epsilon: float
) -> list[list[int]]:
    """
    Get AOs per MO based on TA and S_abs matrices with a threshold epsilon
    """

def get_AO_reachable_by_MO_with_offset(
    AO_reachable_by_MO: list[list[int]],
) -> list[list[tuple[int, int]]]:
    """
    Get AO reachable by MO with offsets based on the
    provided AO_reachable_by_MO structure
    """

def transform_integral(
    int_P_mu_nu: SemiSparseSym3DTensor,
    TA: numpy.ndarray,
    S_abs: numpy.ndarray,
    L_PQ: numpy.ndarray,
    MO_coeff_epsilon: float,
) -> numpy.ndarray:
    """
    Transform the integral using TA, int_P_mu_nu, AO_by_MO, and L_PQ,
    returning the transformed matrix
    """

def transform_integral_cuda(
    int_P_mu_nu: SemiSparseSym3DTensor,
    TA: numpy.ndarray,
    S_abs: numpy.ndarray,
    L_PQ: GPU_MatrixHandle,
    MO_coeff_epsilon: float,
) -> numpy.ndarray:
    """
    Transform the integral using TA, int_P_mu_nu, AO_by_MO, and L_PQ,
    returning the transformed matrix.
    This uses CUDA for performance.
    """
