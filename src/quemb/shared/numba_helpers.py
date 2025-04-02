from collections.abc import Hashable, Mapping, Sequence
from typing import TypeVar

from numba import njit, typeof
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


@njit
def extend_with(D_1: Dict, D_2: Dict) -> None:
    """Extend dictionary with values from :python:`D_2`

    Works inplace on :python:`D_1`.

    Parameter
    ---------
    D_1 :
    D_2 :
    """
    for k, v in D_2.items():
        D_1[k] = v


@njit
def merge_dictionaries(dictionaries: Sequence[Dict]) -> Dict:
    """Merge a sequence of numba dictionaries

    Parameter
    ---------
    dictionaries :
    """
    result = dictionaries[0].copy()
    for D in dictionaries[1:]:
        extend_with(result, D)
    return result
