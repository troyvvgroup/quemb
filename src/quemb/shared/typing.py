"""Enable barebone typechecking for the shape of numpy arrays

Inspired by
https://stackoverflow.com/questions/75495212/type-hinting-numpy-arrays-and-batches

Note that most numpy functions return `ndarray[Any, Any]`
i.e. the type is mostly useful to document intent to the developer.
"""

from typing import Tuple, TypeVar

import numpy as np

T_co = TypeVar("T_co", bound=np.generic, covariant=True)

Vector = np.ndarray[Tuple[int], np.dtype[T_co]]
Matrix = np.ndarray[Tuple[int, int], np.dtype[T_co]]
Tensor3D = np.ndarray[Tuple[int, int, int], np.dtype[T_co]]
Tensor4D = np.ndarray[Tuple[int, int, int, int], np.dtype[T_co]]
Tensor5D = np.ndarray[Tuple[int, int, int, int, int], np.dtype[T_co]]
Tensor6D = np.ndarray[Tuple[int, int, int, int, int, int], np.dtype[T_co]]
Tensor = np.ndarray[Tuple[int, ...], np.dtype[T_co]]
