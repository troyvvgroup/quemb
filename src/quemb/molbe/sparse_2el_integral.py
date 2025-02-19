from collections.abc import Hashable, Mapping
from typing import TypeVar

from numba import njit, typeof, types  # type: ignore[attr-defined]
from numba.experimental import jitclass
from numba.typed import Dict
from numba.types import DictType, float64, int64

Key = TypeVar("Key", bound=Hashable)
Val = TypeVar("Val")


def to_numba_dict(py_dict: Mapping[Key, Val]) -> Dict[Key, Val]:
    # Just check the types of the first key and value
    # and assume uniformness
    key_type = typeof(next(iter(py_dict.keys())))
    value_type = typeof(next(iter(py_dict.values())))

    numba_dict = Dict.empty(
        key_type=key_type,
        value_type=value_type,
    )
    for key, value in py_dict.items():
        numba_dict[key] = value
    return numba_dict


@njit(cache=True)
def gauss_sum(n: int) -> int:
    return (n * (n + 1)) // 2


kv_ty = (int64, float64)


@jitclass([("_data", DictType(*kv_ty))])
class TwoElIntegral:
    """Sparsely stores the 2-electron integrals using chemist's notation.

    The 2-electron integrals are stored in a dictionary.
    The keys of the dictionary are tuples of the form (i, j, k, l)
    where i, j, k, l are the indices of the basis functions.
    The values of the dictionary are the 2-electron integrals.
    """

    def __init__(self) -> None:
        self._data = Dict.empty(*kv_ty)

    def __getitem__(self, key: tuple[int, int, int, int]) -> types.float64:
        idx = self.compound(*key)
        return self._data.get(idx, 0.0)

    def __setitem__(self, key: tuple[int, int, int, int], value: float) -> None:
        self._data[self.compound(*key)] = value

    @staticmethod
    def compound(a: int, b: int, c: int, d: int) -> int:
        """Return compound index given four indices using Yoshimine sort"""
        ab = gauss_sum(a) + b if a > b else gauss_sum(b) + a
        cd = gauss_sum(c) + d if c > d else gauss_sum(d) + c
        return gauss_sum(ab) + cd if ab > cd else gauss_sum(cd) + ab
