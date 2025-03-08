from collections import defaultdict
from collections.abc import Mapping, Sequence, Set
from itertools import chain
from typing import cast

import numpy as np
from chemcoord import Cartesian
from numba import njit, prange
from numba.experimental import jitclass
from numba.typed import Dict, List
from numba.types import DictType, float64, uint64  # type: ignore[attr-defined]
from pyscf.gto import Mole

from quemb.molbe.chemfrag import (
    _get_AOidx_per_atom,
)
from quemb.shared.helper import symmetric_index
from quemb.shared.typing import (
    AtomIdx,
    OrbitalIdx,
    Vector,
)

kv_ty = (uint64, float64)


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
        return symmetric_index(symmetric_index(a, b), symmetric_index(c, d))


def get_orbs_per_atom(
    atom_per_orb: Mapping[OrbitalIdx, Set[AtomIdx]],
) -> dict[AtomIdx, set[OrbitalIdx]]:
    orb_per_atom = defaultdict(set)
    for i_AO, atoms in atom_per_orb.items():
        for i_atom in atoms:
            orb_per_atom[i_atom].add(i_AO)
    return dict(orb_per_atom)


def get_orbs_reachable_by_atom(
    orb_per_atom: Mapping[AtomIdx, Set[OrbitalIdx]],
    screened: Mapping[AtomIdx, Set[AtomIdx]],
) -> dict[AtomIdx, dict[AtomIdx, Set[OrbitalIdx]]]:
    return {
        i_atom: {j_atom: orb_per_atom[j_atom] for j_atom in sorted(connected)}
        for i_atom, connected in screened.items()
    }


def get_orbs_reachable_by_orb(
    reachable_orb_per_atom: Mapping[AtomIdx, Mapping[AtomIdx, Set[OrbitalIdx]]],
    atom_per_orb: Mapping[OrbitalIdx, Set[AtomIdx]],
) -> dict[OrbitalIdx, dict[AtomIdx, Mapping[AtomIdx, Set[OrbitalIdx]]]]:
    return {
        i_AO: {atom: reachable_orb_per_atom[atom] for atom in atoms}
        for i_AO, atoms in atom_per_orb.items()
    }


def get_atom_per_AO(mol: Mole) -> dict[OrbitalIdx, set[AtomIdx]]:
    AOs_per_atom = _get_AOidx_per_atom(mol, frozen_core=False)
    n_AO = AOs_per_atom[-1][-1] + 1

    def get_atom(
        i_AO: OrbitalIdx, AO_per_atom: Sequence[Sequence[OrbitalIdx]]
    ) -> AtomIdx:
        for i_atom, AOs in enumerate(AO_per_atom):
            if i_AO in AOs:
                return cast(AtomIdx, i_atom)
        raise ValueError(f"{i_AO} not contained in AO_per_atom")

    return {
        i_AO: {get_atom(i_AO, AOs_per_atom)}
        for i_AO in cast(Sequence[OrbitalIdx], range(n_AO))
    }


def get_reachable(
    mol: Mole, atoms_per_orb: Mapping[OrbitalIdx, Set[AtomIdx]], scale_vdW: float
) -> dict[OrbitalIdx, set[OrbitalIdx]]:
    orbs_per_atom = get_orbs_per_atom(atoms_per_orb)
    m = Cartesian.from_pyscf(mol)

    screen_conn = m.get_bonds(
        modify_element_data=lambda r: r * scale_vdW, self_bonding_allowed=True
    )

    orb_reachable_by_orb = get_orbs_reachable_by_orb(
        get_orbs_reachable_by_atom(orbs_per_atom, screen_conn), atoms_per_orb
    )

    return {
        i_orb: set(
            chain(
                *(
                    start_atoms[start_atom][target_atom]
                    for start_atom, target_atoms in start_atoms.items()
                    for target_atom in target_atoms
                )
            )
        )
        for i_orb, start_atoms in orb_reachable_by_orb.items()
    }


def to_numba_input(
    exch_reachable: Mapping[OrbitalIdx, set[OrbitalIdx]],
) -> List[Vector[OrbitalIdx]]:
    assert list(exch_reachable.keys()) == list(range(len(exch_reachable)))
    return List(
        [
            np.sort(np.array(orbitals, dtype=np.uint64))
            for orbitals in exch_reachable.values()
        ]
    )


@njit
def get_pq_reachable(
    exch_reachable: Sequence[Vector[OrbitalIdx]],
    coul_reachable: Sequence[Vector[OrbitalIdx]],
) -> dict[tuple[OrbitalIdx, OrbitalIdx], Vector[OrbitalIdx]]:
    prepared_coul_reachable = [set(orbitals) for orbitals in coul_reachable]

    return {
        cast(tuple[OrbitalIdx, OrbitalIdx], (np.uint64(p), q)): np.sort(
            np.array(
                prepared_coul_reachable[p] & prepared_coul_reachable[q],
                dtype=np.uint64,
            )
        )
        for p in range(len(exch_reachable))
        for q in exch_reachable[p]
    }


@njit(parallel=True)
def count_non_zero_2el(
    exch_reachable: list[Vector[OrbitalIdx]],
    pq_coul_reachable: Mapping[tuple[OrbitalIdx, OrbitalIdx], Vector[OrbitalIdx]],
    n_AO: int | None = None,
) -> int:
    n_AO = len(exch_reachable) if n_AO is None else n_AO
    result = 0
    for p in prange(n_AO):  # type: ignore[attr-defined]
        for q in exch_reachable[p]:
            if q > p:
                break
            for r in pq_coul_reachable[p, q]:
                if r > p:
                    break
                for s in exch_reachable[r]:
                    if s > r:
                        break
                    result += 1
    return result


@njit(parallel=True)
def count_non_zero_2el_with_permutations(
    exch_reachable: list[Vector[OrbitalIdx]],
    pq_coul_reachable: Mapping[tuple[OrbitalIdx, OrbitalIdx], Vector[OrbitalIdx]],
    n_AO: int | None = None,
) -> int:
    n_AO = len(exch_reachable) if n_AO is None else n_AO
    result = 0
    for p in prange(n_AO):  # type: ignore[attr-defined]
        for q in exch_reachable[p]:
            for r in pq_coul_reachable[p, q]:
                for s in exch_reachable[r]:
                    result += 1
    return result
