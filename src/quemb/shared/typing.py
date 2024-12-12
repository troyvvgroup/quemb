"""Define some types that do not fit into one particular module

In particular it enables barebone typechecking for the shape of numpy arrays

Inspired by
https://stackoverflow.com/questions/75495212/type-hinting-numpy-arrays-and-batches

Note that most numpy functions return `ndarray[Any, Any]`
i.e. the type is mostly useful to document intent to the developer.
"""
import os
from typing import Any, Dict, Tuple, TypeAlias, TypeVar

import numpy as np

# We want the dtype to behave covariant, i.e. if a
#  Vector[float] is allowed, then the more specific
#  Vector[float64] should also be allowed.
# Also see here:
# https://stackoverflow.com/questions/61568462/what-does-typevara-b-covariant-true-mean
T_dtype_co = TypeVar("T_dtype_co", bound=np.generic, covariant=True)

Vector = np.ndarray[Tuple[int], np.dtype[T_dtype_co]]
Matrix = np.ndarray[Tuple[int, int], np.dtype[T_dtype_co]]
Tensor3D = np.ndarray[Tuple[int, int, int], np.dtype[T_dtype_co]]
Tensor4D = np.ndarray[Tuple[int, int, int, int], np.dtype[T_dtype_co]]
Tensor5D = np.ndarray[Tuple[int, int, int, int, int], np.dtype[T_dtype_co]]
Tensor6D = np.ndarray[Tuple[int, int, int, int, int, int], np.dtype[T_dtype_co]]
Tensor7D = np.ndarray[Tuple[int, int, int, int, int, int, int], np.dtype[T_dtype_co]]
Tensor = np.ndarray[Tuple[int, ...], np.dtype[T_dtype_co]]


PathLike: TypeAlias = str | os.PathLike
KwargDict: TypeAlias = Dict[str, Any]