from collections.abc import Sequence
from typing import TypeVar

import numpy as np
from numba import int64, prange
from numba.typed import List
from numba.types import (  # type: ignore[attr-defined]
    UniTuple,
)
from pyscf.ao2mo.addons import restore

from quemb.shared.helper import njit, ravel_symmetric
from quemb.shared.typing import (
    Vector,
)

_UniTuple_int64_2 = UniTuple(int64, 2)
_T = TypeVar("_T", int, np.integer)


@njit
def n_symmetric(n):
    return ravel_symmetric(n - 1, n - 1) + 1


@njit
def _jit_identify_contiguous_blocks(X: Vector[np.int64]) -> list[tuple[int, int]]:
    result = List.empty_list(_UniTuple_int64_2)
    if X.size == 0:
        return result

    start = 0  # Start index of a contiguous block
    for i in range(1, len(X)):
        if X[i] - X[i - 1] > 1:  # Gap detected
            result.append((start, i))
            start = i  # New block starts here
    result.append((start, len(X)))  # Add the final block
    return result


def identify_contiguous_blocks(X: Sequence[_T]) -> list[tuple[int, int]]:
    """Identify the indices of contiguous blocks in the sequence X.

    A block is defined as a sequence of consecutive integers.
    Returns a list of tuples, where each tuple contains the
    start and one-past-the-end indices of a block.
    This means that the returned tuples can be used in slicing operations.

    Parameters
    ----------
    X :

    Example
    --------
    >>> X = [1, 2, 3, 5, 6, 7, 9, 10]
    >>> blocks = identify_contiguous_blocks(X)
    >>> assert blocks  == [(0, 3), (3, 6), (6, 8)]
    >>> assert X[blocks[1][0] : blocks[1][1]] == [5, 6, 7]
    """
    if not X:
        return []
    result = []
    start = 0  # Start index of a contiguous block
    for i in range(1, len(X)):
        if X[i] - X[i - 1] > 1:  # Gap detected
            result.append((start, i))
            start = i  # New block starts here
    result.append((start, len(X)))  # Add the final block
    return result


@njit(parallel=False)
def idx_extract_block(keys, block_start, block_end, jit_ij_pairs):
    """Extract g[block_start : block_end, block_start : block_end,
    block_start : block_end,  block_start : block_end]"""
    idx = np.empty(n_symmetric(block_end - block_start), dtype=np.int64)
    start_offset = np.searchsorted(keys, ravel_symmetric(block_start, block_start))
    out_offset = 0
    idx[0] = start_offset
    out_offset += 1

    current_offset = start_offset

    for i in range(1, block_end - block_start):
        cut_out = (
            ravel_symmetric(block_start + i, block_start - 1)
            - ravel_symmetric(block_start + i, jit_ij_pairs[block_start + i][0])
        ) + 1
        current_offset = current_offset + cut_out + 1

        idx[out_offset : out_offset + i + 1] = np.arange(
            current_offset, current_offset + i + 1
        )
        out_offset += i + 1
        current_offset += i
    return idx


@njit(parallel=False)
def _get_diag_out_idx(blocks_offset, i):
    block_start, block_end = blocks_offset[i]
    idx = np.empty(n_symmetric(block_end - block_start), dtype=np.int64)
    start_offset = ravel_symmetric(block_start, block_start)
    out_offset = 0
    idx[0] = start_offset
    out_offset += 1

    current_offset = start_offset

    for i in range(1, block_end - block_start):
        cut_out = (
            ravel_symmetric(block_start + i, block_start - 1)
            - ravel_symmetric(block_start + i, 0)
        ) + 1
        current_offset = current_offset + cut_out + 1

        idx[out_offset : out_offset + i + 1] = np.arange(
            current_offset, current_offset + i + 1
        )
        out_offset += i + 1
        current_offset += i
    return idx


@njit(parallel=False)
def idx_extract_offdiag(
    keys, block_start_1, block_end_1, block_start_2, block_end_2, ji_ij_pairs
):
    new_idx = np.empty(
        (block_end_1 - block_start_1) * (block_end_2 - block_start_2), dtype=np.int64
    )
    out_counter = 0
    assert block_end_1 <= block_start_2
    L = block_end_1 - block_start_1
    start_offset = np.searchsorted(keys, ravel_symmetric(block_start_1, block_start_2))
    current_offset = start_offset
    new_idx[out_counter : out_counter + L] = np.arange(
        current_offset, current_offset + L
    )
    out_counter += L

    for i, current_start_2 in enumerate(range(block_start_2 + 1, block_end_2)):
        current_offset = (
            current_offset
            + 1
            + (
                ravel_symmetric(current_start_2 - 1, current_start_2 - 1)
                - ravel_symmetric(current_start_2 - 1, block_start_1)
            )
            + (
                ravel_symmetric(current_start_2, block_start_1)
                - ravel_symmetric(current_start_2, ji_ij_pairs[current_start_2][0])
            )
        )

        new_idx[out_counter : out_counter + L] = np.arange(
            current_offset, current_offset + L
        )
        out_counter += L
    return new_idx


@njit(parallel=False)
def _get_offdiag_out_idx(blocks_offset, i, j):
    block_start_1, block_end_1, block_start_2, block_end_2 = (
        *blocks_offset[i],
        *blocks_offset[j],
    )
    assert block_end_1 <= block_start_2
    new_idx = np.empty(
        ((block_end_1 - block_start_1) * (block_end_2 - block_start_2)), dtype=np.int64
    )
    out_counter = 0
    assert block_end_1 <= block_start_2
    L = block_end_1 - block_start_1
    start_offset = ravel_symmetric(block_start_1, block_start_2)
    current_offset = start_offset
    new_idx[out_counter : out_counter + L] = np.arange(
        current_offset, current_offset + L
    )
    out_counter += L

    for i, current_start_2 in enumerate(range(block_start_2 + 1, block_end_2)):
        current_offset = (
            current_offset
            + 1
            + (
                ravel_symmetric(current_start_2 - 1, current_start_2 - 1)
                - ravel_symmetric(current_start_2 - 1, block_start_1)
            )
            + (
                ravel_symmetric(current_start_2, block_start_1)
                - ravel_symmetric(current_start_2, 0)
            )
        )

        new_idx[out_counter : out_counter + L] = np.arange(
            current_offset, current_offset + L
        )
        out_counter += L
    return new_idx


@njit(parallel=False)
def block_diag_assign_ix(g_lhs, rows_out, g_rhs, rows_rhs):
    for row_counter in prange(len(rows_rhs)):
        for col_counter in prange(row_counter + 1):
            val = g_rhs[rows_rhs[row_counter], rows_rhs[col_counter]]
            g_lhs[rows_out[row_counter], rows_out[col_counter]] = val
            g_lhs[rows_out[col_counter], rows_out[row_counter]] = val


@njit(parallel=False)
def off_diag_assign_ix(g_lhs, rows_out, cols_out, g_rhs, rows_rhs, cols_rhs):
    for row_counter in prange(len(rows_rhs)):
        for col_counter in prange(len(cols_rhs)):
            val = g_rhs[rows_rhs[row_counter], cols_rhs[col_counter]]
            g_lhs[rows_out[row_counter], cols_out[col_counter]] = val
            g_lhs[cols_out[col_counter], rows_out[row_counter]] = val


def extract_g(keys, jit_ij_pairs, new_integrals, extract_idx):
    argsort_result = np.argsort(extract_idx)
    sorted_idx = extract_idx[argsort_result]
    inv_sorting = np.argsort(argsort_result)
    n_MO = len(sorted_idx)

    n_pairs = n_symmetric(n_MO)

    g = np.zeros((n_pairs, n_pairs), dtype=np.float64)

    blocks_offset = _jit_identify_contiguous_blocks(sorted_idx)

    blocks = List(
        [
            (sorted_idx[block[0]], sorted_idx[block[1] - 1] + 1)
            for block in blocks_offset
        ]
    )

    diag_blocks = List(
        [
            (
                _get_diag_out_idx(blocks_offset, i),
                idx_extract_block(keys, start, end, jit_ij_pairs),
            )
            for i, (start, end) in enumerate(blocks)
        ]
    )
    off_diag_blocks = List(
        [
            (
                _get_offdiag_out_idx(blocks_offset, i, j),
                idx_extract_offdiag(keys, start_1, end_1, start_2, end_2, jit_ij_pairs),
            )
            for j, (start_2, end_2) in enumerate(blocks)
            for i, (start_1, end_1) in enumerate(blocks[:j])
        ]
    )

    for lhs_idx, rhs_idx in diag_blocks:
        block_diag_assign_ix(g, lhs_idx, new_integrals, rhs_idx)
    for lhs_idx, rhs_idx in off_diag_blocks:
        block_diag_assign_ix(g, lhs_idx, new_integrals, rhs_idx)

    for i in prange(len(diag_blocks)):
        lhs_idx_2, rhs_idx_2 = diag_blocks[i]
        for lhs_idx_1, rhs_idx_1 in diag_blocks[:i]:
            off_diag_assign_ix(
                g, lhs_idx_1, lhs_idx_2, new_integrals, rhs_idx_1, rhs_idx_2
            )
    for i, (lhs_idx_2, rhs_idx_2) in enumerate(off_diag_blocks):
        for lhs_idx_1, rhs_idx_1 in off_diag_blocks[:i]:
            off_diag_assign_ix(
                g, lhs_idx_1, lhs_idx_2, new_integrals, rhs_idx_1, rhs_idx_2
            )
    for lhs_idx_2, rhs_idx_2 in diag_blocks:
        for lhs_idx_1, rhs_idx_1 in off_diag_blocks:
            off_diag_assign_ix(
                g, lhs_idx_1, lhs_idx_2, new_integrals, rhs_idx_1, rhs_idx_2
            )
    return restore("1", g, n_MO)[
        np.ix_(inv_sorting, inv_sorting, inv_sorting, inv_sorting)
    ]
