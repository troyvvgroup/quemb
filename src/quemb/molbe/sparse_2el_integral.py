from collections import defaultdict
from collections.abc import Hashable, Mapping, Set
from itertools import chain
from typing import TypeVar

from numba import njit, typeof  # type: ignore[attr-defined]
from numba.experimental import jitclass
from numba.typed import Dict, List
from numba.types import DictType, float64, int64  # type: ignore[attr-defined]

from quemb.shared.typing import (
    AtomIdx,
    OrbitalIdx,
)

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

    This is a :python:`jitclass` which can be used with numba functions.

    8-fold permutational symmetry is assumed for the spatial integrals, i.e.

    .. math::

        g_{ijkl} = g_{klij} = g_{jikl} = g_{jilk} = g_{lkji} = g_{lkij} = g_{ilkj} = g_{ikjl}

    There is no boundary checking! It will not crash, but just return 0.0
    if you try to access an index that is not in the dictionary.

    The 2-electron integrals are stored in a dictionary.
    The keys of the dictionary are tuples of the form (i, j, k, l)
    where i, j, k, l are the indices of the basis functions.
    The values of the dictionary are the 2-electron integrals.

    Examples
    --------

    >>> g = TwoElIntegral()
    >>> g[1, 2, 3, 4] = 3

    We can test all possible permutations:

    >>> assert g[1, 2, 3, 4] == 3
    >>> assert g[1, 2, 4, 3] == 3
    >>> assert g[2, 1, 3, 4] == 3
    >>> assert g[2, 1, 4, 3] == 3
    >>> assert g[3, 4, 1, 2] == 3
    >>> assert g[4, 3, 1, 2] == 3
    >>> assert g[3, 4, 2, 1] == 3
    >>> assert g[4, 3, 2, 1] == 3

    A non-existing index returns 0.0:

    >>> assert g[1, 2, 3, 10] == 0
    """  # noqa: E501

    def __init__(self) -> None:
        self._data = Dict.empty(*kv_ty)

    def __getitem__(self, key: tuple[int, int, int, int]) -> float:
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


def get_orb_per_atom(
    atom_per_orb: Mapping[OrbitalIdx, Set[AtomIdx]],
) -> dict[AtomIdx, set[OrbitalIdx]]:
    orb_per_atom = defaultdict(set)
    for i_AO, atoms in atom_per_orb.items():
        for i_atom in atoms:
            orb_per_atom[i_atom].add(i_AO)
    return dict(orb_per_atom)


def get_AOs_reachable_by_atom(
    orb_per_atom: Mapping[AtomIdx, Set[OrbitalIdx]],
    screened: Mapping[AtomIdx, Set[AtomIdx]],
) -> dict[AtomIdx, dict[AtomIdx, Set[OrbitalIdx]]]:
    return {
        i_atom: {j_atom: orb_per_atom[j_atom] for j_atom in sorted(connected)}
        for i_atom, connected in screened.items()
    }


def _orb_reachable_by_atom_for_numba(
    orb_in_reach: Mapping[AtomIdx, Mapping[AtomIdx, Set[OrbitalIdx]]],
) -> List[List[OrbitalIdx]]:
    list(range(len(orb_in_reach))) == sorted(orb_in_reach.keys())
    return List(
        List(sorted(set(chain(*orb_in_reach[i_atom].values()))))
        for i_atom in sorted(orb_in_reach)
    )
