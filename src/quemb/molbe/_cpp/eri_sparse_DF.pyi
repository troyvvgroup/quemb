"""
Minimal pybind11 + Eigen example
"""

from __future__ import annotations

import typing

import np

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
    def __init__(self, arg0: np.ndarray[np.float64]) -> None: ...
    def __repr__(self) -> str: ...

class SemiSparse3DTensor:
    @typing.overload
    def __init__(
        self,
        dense_data: np.ndarray[np.float64],
        shape: tuple[int, int, int],
        AO_reachable_by_MO: list[list[int]],
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        dense_data: np.ndarray[np.float64],
        shape: tuple[int, int, int],
        AO_reachable_by_MO: list[list[int]],
        AO_reachable_by_MO_with_offsets: list[list[tuple[int, int]]],
        offsets: dict[int, int],
    ) -> None: ...
    def get_aux_vector(self, mu: int, i: int) -> np.ndarray[np.float64]:
        """
        Return auxiliary vector for given AO and MO index
        """
    @property
    def AO_reachable_by_MO(self) -> list[list[int]]: ...
    @property
    def AO_reachable_by_MO_with_offsets(self) -> list[list[tuple[int, int]]]: ...
    @property
    def dense_data(self) -> np.ndarray[np.float64]: ...
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
    def __getitem__(self, arg0: tuple[int, int]) -> np.ndarray[np.float64]: ...
    @typing.overload
    def __init__(
        self,
        arg0: np.ndarray[np.float64],
        arg1: tuple[int, int, int],
        arg2: list[list[int]],
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        arg0: np.ndarray[np.float64],
        arg1: tuple[int, int, int],
        arg2: list[list[int]],
        arg3: list[list[int]],
        arg4: list[list[tuple[int, int]]],
        arg5: list[list[tuple[int, int]]],
        arg6: dict[int, int],
    ) -> None: ...
    def get_aux_vector(self, arg0: int, arg1: int) -> np.ndarray[np.float64]: ...
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
    def unique_dense_data(self) -> np.ndarray[np.float64]: ...

def contract_with_TA_1st(
    TA: np.ndarray[np.float64],
    int_P_mu_nu: SemiSparseSym3DTensor,
    AO_by_MO: list[list[int]],
) -> SemiSparse3DTensor: ...
def contract_with_TA_2nd_to_sym_dense(
    int_mu_i_P: SemiSparse3DTensor, TA: np.ndarray[np.float64]
) -> np.ndarray[np.float64]:
    """
    Contract with TA to get a symmetric dense tensor (P | i, j)
    """

def extract_unique(exch_reachable: list[list[int]]) -> list[list[int]]:
    """
    Extract unique reachable AOs from the provided exch_reachable structure
    """

def get_AO_per_MO(
    TA: np.ndarray[np.float64],
    S_abs: np.ndarray[np.float64],
    epsilon: float,
) -> list[list[int]]:
    """
    Get AOs per MO based on TA and S_abs matrices with a threshold epsilon
    """

def get_AO_reachable_by_MO_with_offset(
    AO_reachable_by_MO: list[list[int]],
) -> list[list[tuple[int, int]]]:
    """
    Get AO reachable by MO with offsets based on the provided
    AO_reachable_by_MO structure
    """

def transform_integral(
    int_P_mu_nu: SemiSparseSym3DTensor,
    TA: np.ndarray[np.float64],
    S_abs: np.ndarray[np.float64],
    L_PQ: np.ndarray[np.float64],
    MO_coeff_epsilon: float,
) -> np.ndarray[np.float64]:
    """
    Transform the integral using TA, int_P_mu_nu, AO_by_MO, and L_PQ,
    returning the transformed matrix
    """

def transform_integral_cuda(
    int_P_mu_nu: SemiSparseSym3DTensor,
    TA: np.ndarray[np.float64],
    S_abs: np.ndarray[np.float64],
    L_PQ: GPU_MatrixHandle,
    MO_coeff_epsilon: float,
) -> np.ndarray[np.float64]:
    """
    Transform the integral using TA, int_P_mu_nu, AO_by_MO, and L_PQ,
    returning the transformed matrix. This uses CUDA for performance.
    """
