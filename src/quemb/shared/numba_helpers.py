from collections.abc import Hashable, Mapping, Sequence
from typing import TypeVar

from numba import boolean, int64, typeof
from numba.typed import Dict, List
from numba.types import (  # type: ignore[attr-defined]
    DictType,
    ListType,
)

from quemb.shared.helper import jitclass, njit

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


key_type = int64
val_type = boolean
dict_type = DictType(key_type, val_type)
list_type = ListType(key_type)


@jitclass(
    [
        ("_lookup", dict_type),
        ("items", list_type),
    ]
)
class SortedIntSet:
    """A sorted set implementation using Numba's jitclass.

    This class maintains a set of unique integers in sorted order.
    Internally, it uses a dictionary for fast membership checks
    and a list for ordered storage of elements. All operations
    are JIT-compiled for high performance in numerical code.

    Attributes
    ----------
    _lookup : numba.typed.Dict
        A dictionary mapping integers to booleans, used for fast
        membership testing and ensuring uniqueness.
    items : numba.typed.List
        A list of sorted, unique integers representing the elements
        of the set in ascending order.

    Examples
    --------
    >>> s = SortedIntSet()
    >>> s.add(3)
    >>> s.add(1)
    >>> s.add(2)
    >>> s.items
    [1, 2, 3]

    >>> 2 in s
    True

    >>> s.remove(2)
    >>> s.items
    [1, 3]
    """

    def __init__(self):
        self._lookup = Dict.empty(key_type=key_type, value_type=val_type)
        self.items = List.empty_list(key_type)

    def add(self, val):
        if val in self._lookup:
            return
        self._lookup[val] = True

        # Binary search insert to maintain sorted order
        idx = 0
        while idx < len(self.items) and self.items[idx] < val:
            idx += 1

        self.items.insert(idx, val)

    def remove(self, val):
        if val not in self._lookup:
            return
        del self._lookup[val]
        # Rebuild list without val
        new_items = List.empty_list(key_type)
        for item in self.items:
            if item != val:
                new_items.append(item)
        self.items = new_items

    def __contains__(self, val):
        return val in self._lookup

    def __len__(self):
        return len(self.items)


Type_SortedIntSet = typeof(SortedIntSet())
