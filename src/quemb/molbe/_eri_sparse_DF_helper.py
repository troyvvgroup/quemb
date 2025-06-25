from collections.abc import Sequence
from typing import TypeVar

import numpy as np
from numba import int64
from numba.types import (  # type: ignore[attr-defined]
    UniTuple,
)

_UniTuple_int64_2 = UniTuple(int64, 2)
_T = TypeVar("_T", int, np.integer)


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
