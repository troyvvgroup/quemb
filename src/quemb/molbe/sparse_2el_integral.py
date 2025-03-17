from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence, Set
from itertools import chain, takewhile
from typing import TypeVar, cast

import numpy as np
from chemcoord import Cartesian
from numba import njit, prange
from numba.experimental import jitclass
from numba.typed import Dict, List
from numba.types import (  # type: ignore[attr-defined]
    DictType,
    ListType,
    float64,
    int64,
    uint64,
)
from pyscf import df
from pyscf.gto import Mole

from quemb.molbe.chemfrag import (
    _get_AOidx_per_atom,
)
from quemb.shared.helper import ravel_symmetric
from quemb.shared.typing import (
    AOIdx,
    AtomIdx,
    Matrix,
    OrbitalIdx,
    Real,
    ShellIdx,
    Vector,
)


@jitclass
class SparseInt2:
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

    _data: DictType(uint64, float64)  # type: ignore[valid-type]

    def __init__(self) -> None:
        self._data = Dict.empty(uint64, float64)

    def __getitem__(self, key: tuple[int, int, int, int]) -> float:
        idx = self.compound(*key)
        return self._data.get(idx, 0.0)

    def __setitem__(self, key: tuple[int, int, int, int], value: float) -> None:
        self._data[self.compound(*key)] = value

    @staticmethod
    def compound(a: int, b: int, c: int, d: int) -> int:
        """Return compound index given four indices using Yoshimine sort"""
        return ravel_symmetric(ravel_symmetric(a, b), ravel_symmetric(c, d))


@jitclass
class SemiSparseInt3c2e:
    r"""Sparsely store the 2-electron integrals with the auxiliary basis

    This class semi-sparsely stores the elements of the 3-indexed tensor
    :math:`(\mu \nu | P)`.
    Semi-sparsely, because it is assumed that there are many
    exchange pairs :math:`\mu, \nu` which are zero, while the integral along
    the auxiliary basis :math:`P` is stored densely as numpy array.

    2-fold permutational symmetry for the :math:`\mu, \nu` pairs is assumed, i.e.

    .. math::

        (\mu \nu | P) == (\nu, \mu | P)

    Examples
    --------

    >>> g = SemiSparseInt3c2e()
    >>> g[1, 2] = np.array([3., 4., 5.])

    We can test all possible permutations:

    >>> assert g[1, 2] == np.array([3., 4., 5.])
    >>> assert g[2, 1] == np.array([3., 4., 5.])

    A non-existing index throws a :python:`KeyError`

    The triple :math:`\mu, \nu, P` is accessed as

    >>> g[mu, nu][P]
    """

    _data: DictType(int64, float64[::1])  # type: ignore[valid-type]
    nao: int64
    naux: int64
    exch_reachable: ListType(int64[::1])  # type: ignore[valid-type]
    exch_reachable_unique: ListType(int64[::1])  # type: ignore[valid-type]

    def __init__(
        self, nao: int, naux: int, exch_reachable: list[Vector[int64]]
    ) -> None:
        self._data = Dict.empty(int64, float64[::1])
        self.nao = nao
        self.naux = naux
        self.exch_reachable = exch_reachable
        self.exch_reachable_unique = _jit_account_for_symmetry(exch_reachable)

    def __getitem__(self, key: tuple[OrbitalIdx, OrbitalIdx]) -> Vector[float64]:
        # We have to ignore the type here, because tuples are invariant, i.e.
        # (OrbitalIdx, OrbitalIdx) is not a subtype of (int, int).
        return self._data[self.compound(*key)]  # type: ignore[arg-type]

    def __setitem__(
        self, key: tuple[OrbitalIdx, OrbitalIdx], value: Vector[float64]
    ) -> None:
        # We have to ignore the type here, because tuples are invariant, i.e.
        # (OrbitalIdx, OrbitalIdx) is not a subtype of (int, int).
        self._data[self.compound(*key)] = value  # type: ignore[arg-type]

    def get_dense_data(self) -> Matrix[float64]:
        result = np.empty((self.naux, len(self._data)))
        i = 0
        for p in range(self.nao):
            for q in self.exch_reachable_unique[p]:
                result[:, i] = self[p, q]
                i += 1
        return result

    @staticmethod
    def compound(a: int, b: int) -> int:
        """Return compound index given four indices using Yoshimine sort"""
        return ravel_symmetric(a, b)


T_start_orb = TypeVar("T_start_orb", bound=OrbitalIdx)
T_target_orb = TypeVar("T_target_orb", bound=OrbitalIdx)


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
    atom_per_orb: Mapping[T_start_orb, Set[AtomIdx]],
    reachable_orb_per_atom: Mapping[AtomIdx, Mapping[AtomIdx, Set[T_target_orb]]],
) -> dict[T_start_orb, dict[AtomIdx, Mapping[AtomIdx, Set[T_target_orb]]]]:
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


def conversions_AO_shell(
    mol: Mole,
) -> tuple[dict[ShellIdx, list[AOIdx]], dict[AOIdx, ShellIdx]]:
    """Return dictionaries that for a shell index return the corresponding AO indices
    and for an AO index return the corresponding shell index.

    Parameters
    ----------
    mol :
        The molecule.
    """
    shell_id_to_AO = {
        ShellIdx(shell_id): cast(
            list[AOIdx], list(range(*mol.nao_nr_range(shell_id, shell_id + 1)))
        )
        for shell_id in range(mol.nbas)
    }
    AO_to_shell_id = {
        cast(AOIdx, AO): shell_id
        for (shell_id, AOs) in shell_id_to_AO.items()
        for AO in AOs
    }
    return shell_id_to_AO, AO_to_shell_id


def get_reachable(
    mol: Mole,
    atoms_per_orb: Mapping[OrbitalIdx, Set[AtomIdx]],
    screening_cutoff: Real | Callable[[Real], Real] | Mapping[str, Real] = 5,
) -> dict[OrbitalIdx, set[OrbitalIdx]]:
    """Return the orbitals that can by reached for each orbital after screening.

    Parameters
    ----------
    mol :
        The molecule.
    atoms_per_orb :
        The atoms per orbital. For AOs this is the atom the AO is centered on,
        i.e. a set containing only one element,
        but for delocalised MOs there can be more than one atom.
    screening_cutoff :
        The screening cutoff is given by the overlap of van der Waals radii.
        By default, all radii are set to 5 Å, i.e. the screening distance is 10 Å.
        Alternatively, a callable or a dictionary can be passed.
        The callable is called with the tabulated van der Waals radius
        of the atom as argument and can be used to scale it up.
        The dictionary can be used to define different van der Waals radii
        for different elements. Compare to the :python:`modify_element_data`
        argument of :meth:`~chemcoord.Cartesian.get_bonds`.
    """
    m = Cartesian.from_pyscf(mol)

    screen_conn = m.get_bonds(
        modify_element_data=screening_cutoff, self_bonding_allowed=True
    )

    return _flatten(
        get_orbs_reachable_by_orb(
            atoms_per_orb,
            get_orbs_reachable_by_atom(get_orbs_per_atom(atoms_per_orb), screen_conn),
        )
    )


_T_orb_idx = TypeVar("_T_orb_idx", bound=OrbitalIdx)


def to_numba_input(
    exch_reachable: Mapping[_T_orb_idx, Set[_T_orb_idx]],
) -> List[Vector[_T_orb_idx]]:
    """Convert the reachable orbitals to a list of numpy arrays.

    This contains the same information but is a far more efficient layout for numba.
    """
    assert list(exch_reachable.keys()) == list(range(len(exch_reachable)))
    return List(
        [
            np.array(sorted(orbitals), dtype=np.int64)  # type: ignore[type-var]
            for orbitals in exch_reachable.values()
        ]
    )


def account_for_symmetry(
    reachable: Mapping[int, Sequence[int]],
) -> dict[int, list[int]]:
    """Account for permutational symmetry and remove all q that are larger than p.

    Paramaters
    ----------
    reachable :

    Example
    -------
    >>> account_for_symmetry({0: [0, 1, 2], 1: [0, 1, 2], 2: [0, 1, 2]})
    >>> {0: [0], 1: [0, 1], 2: [0, 1, 2]}
    """
    return {p: list(takewhile(lambda q: p >= q, qs)) for (p, qs) in reachable.items()}


@njit(cache=True)
def _jit_account_for_symmetry(
    reachable: list[Vector[int]],
) -> list[Vector[int]]:
    """Account for permutational symmetry and remove all q that are larger than p.

    Paramaters
    ----------
    reachable :
    """
    return List(
        [np.array([q for q in qs if p >= q]) for (p, qs) in enumerate(reachable)]
    )


def identify_contiguous_blocks(X: Sequence[int]) -> list[tuple[int, int]]:
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


_T = TypeVar("_T", bound=int)


def get_blocks(reachable: Sequence[_T]) -> list[tuple[_T, _T]]:
    """Return the value of the border elements of contiguous blocks in the sequence X."

    A block is defined as a sequence of consecutive integers.
    Returns a list of tuples, where each tuple contains the
    value at the start and at the end of a block.

    Parameters
    ----------
    X :

    Example
    --------
    >>> X = [1, 2, 3, 5, 6, 7, 9, 10]
    >>> get_blocks(X) == [(1, 3), (5, 7), (9, 10)]
    """
    return [
        (reachable[start], reachable[stop - 1])
        for (start, stop) in identify_contiguous_blocks(reachable)
    ]


def get_sparse_ints_3c2e(
    mol: Mole,
    auxmol: Mole,
) -> SemiSparseInt3c2e:
    """Return the 3-center 2-electron integrals in a sparse format." """
    exch_reachable = cast(
        Mapping[AOIdx, Set[AOIdx]], get_reachable(mol, get_atom_per_AO(mol))
    )
    sparse_ints_3c2e = SemiSparseInt3c2e(
        mol.nao, auxmol.nao, to_numba_input(exch_reachable)
    )
    shell_id_to_AO, AO_to_shell_id = conversions_AO_shell(mol)
    shell_reachable_by_shell = {
        AO_to_shell_id[k]: sorted({AO_to_shell_id[orb] for orb in v})
        for k, v in exch_reachable.items()
    }

    for i_shell, reachable in shell_reachable_by_shell.items():
        for start_block, stop_block in get_blocks(reachable):
            integrals = np.asarray(
                df.incore.aux_e2(
                    mol,
                    auxmol,
                    intor="int3c2e",
                    shls_slice=(
                        i_shell,
                        i_shell + 1,
                        start_block,
                        stop_block + 1,
                        0,
                        auxmol.nbas,
                    ),
                ),
                order="C",
            )
            for i, p in enumerate(shell_id_to_AO[i_shell]):
                for j, q in enumerate(
                    range(
                        shell_id_to_AO[start_block][0],
                        shell_id_to_AO[stop_block][-1] + 1,
                    )
                ):
                    # We have to ignore the type here, because tuples are invariant,
                    # i.e. (OrbitalIdx, OrbitalIdx) is not a subtype of (int, int).
                    sparse_ints_3c2e[p, q] = integrals[i, j, ::1]  # type: ignore[index]
    return sparse_ints_3c2e


def _flatten(
    orb_reachable_by_orb: Mapping[
        T_start_orb, Mapping[AtomIdx, Mapping[AtomIdx, Set[T_target_orb]]]
    ],
) -> dict[T_start_orb, set[T_target_orb]]:
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


@njit
def _get_pq_reachable(
    exch_reachable: Sequence[Vector[OrbitalIdx]],
    coul_reachable: Sequence[Vector[OrbitalIdx]],
) -> dict[tuple[OrbitalIdx, OrbitalIdx], Vector[OrbitalIdx]]:
    prepared_coul_reachable = [set(orbitals) for orbitals in coul_reachable]

    return {
        (np.uint64(p), q): np.sort(  # type: ignore[misc]
            np.array(
                list(prepared_coul_reachable[p] & prepared_coul_reachable[q]),
                dtype=np.uint64,
            )
        )
        for p in range(len(exch_reachable))
        for q in exch_reachable[p]
    }


@njit(parallel=True)
def _count_non_zero_2el(
    exch_reachable: list[Vector[OrbitalIdx]],
    n_AO: int | None = None,
) -> int:
    n_AO = len(exch_reachable) if n_AO is None else n_AO
    result = 0
    for p in prange(n_AO):  # type: ignore[attr-defined]
        for q in exch_reachable[p]:
            for r in range(p + 1):
                # perhaps I should account for permutational symmetry here as well.
                # for l in range(k + 1 if i > k else j + 1):
                for s in exch_reachable[r]:
                    result += 1
    return result
