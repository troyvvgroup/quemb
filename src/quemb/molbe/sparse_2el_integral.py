from collections.abc import Hashable, Mapping
from typing import TypeVar

from numba import typeof, types
from numba.experimental import jitclass
from numba.typed import Dict

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


@jitclass
class TwoElIntegral:
    """This class is a wrapper around the 2-electron integrals.
    It is used to store the 2-electron integrals in a sparse format.
    The 2-electron integrals are stored in a dictionary.
    The keys of the dictionary are tuples of the form (i, j, k, l)
    where i, j, k, l are the indices of the basis functions.
    The values of the dictionary are the 2-electron integrals.
    """

    def __init__(self) -> None:
        self._data = Dict.empty(
            key_type=types.int64,
            value_type=types.float64,
        )

    def __setitem__(self, key: tuple[int, int, int, int], value: float) -> None:
        self._data[key] = value
