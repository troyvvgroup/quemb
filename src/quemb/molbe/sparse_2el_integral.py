from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence, Set
from itertools import chain, takewhile
from typing import Final, TypeVar, cast

import numpy as np
from chemcoord import Cartesian
from numba import float64, int64, njit, prange, uint64  # type: ignore[attr-defined]

# from numba.experimental import jitclass
from numba.typed import Dict, List
from numba.types import (  # type: ignore[attr-defined]
    DictType,
    ListType,
)
from pyscf import df
from pyscf.gto import Mole
from scipy.linalg import solve
from scipy.optimize import bisect

from quemb.molbe.chemfrag import (
    _get_AOidx_per_atom,
)
from quemb.shared.helper import jitclass, ravel_symmetric
from quemb.shared.typing import (
    AOIdx,
    AtomIdx,
    Matrix,
    OrbitalIdx,
    Real,
    ShellIdx,
    Tensor4D,
    Vector,
)

_T_orb_idx = TypeVar("_T_orb_idx", bound=OrbitalIdx)
_T_start_orb = TypeVar("_T_start_orb", bound=OrbitalIdx)
_T_target_orb = TypeVar("_T_target_orb", bound=OrbitalIdx)


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
        idx = self.idx(*key)
        return self._data.get(idx, 0.0)

    def __setitem__(self, key: tuple[int, int, int, int], value: float) -> None:
        self._data[self.idx(*key)] = value

    @staticmethod
    def idx(a: int, b: int, c: int, d: int) -> int:
        """Return compound index given four indices using Yoshimine sort"""
        return ravel_symmetric(ravel_symmetric(a, b), ravel_symmetric(c, d))


def get_DF_integrals(
    mol: Mole, auxmol: Mole
) -> tuple[SemiSparseSym3DTensor, SemiSparseSym3DTensor]:
    ints_3c2e = get_sparse_ints_3c2e(mol, auxmol)
    ints_2c2e = auxmol.intor("int2c2e")
    df_coeffs_data = solve(ints_2c2e, ints_3c2e.dense_data.T).T
    df_coef = SemiSparseSym3DTensor(
        df_coeffs_data, ints_3c2e.nao, ints_3c2e.naux, ints_3c2e.exch_reachable
    )
    return ints_3c2e, df_coef


@njit(parallel=True)
def get_dense_integrals(
    ints_3c2e: SemiSparseSym3DTensor, df_coef: SemiSparseSym3DTensor
) -> Tensor4D[float64]:
    g = np.zeros((ints_3c2e.nao, ints_3c2e.nao, ints_3c2e.nao, ints_3c2e.nao))

    for mu in range(ints_3c2e.nao):
        for nu in ints_3c2e.exch_reachable_unique[mu]:
            for rho in range(ints_3c2e.nao):
                for sigma in ints_3c2e.exch_reachable_unique[rho]:
                    value = ints_3c2e[mu, nu] @ df_coef[rho, sigma]  # type: ignore[index]
                    g[mu, nu, rho, sigma] = value
                    g[mu, nu, sigma, rho] = value
                    g[nu, mu, rho, sigma] = value
                    g[nu, mu, sigma, rho] = value
                    g[rho, sigma, mu, nu] = value
                    g[sigma, rho, mu, nu] = value
                    g[rho, sigma, nu, mu] = value
                    g[sigma, rho, nu, mu] = value
    return g


# We cannot use the normal abstract base class here, because we later want to jit.
class _ABC_MutableSemiSparse3DTensor:
    r"""Semi-Sparsely store the 2-electron integrals with the auxiliary basis

    This class semi-sparsely stores the elements of the 3-indexed tensor
    :math:`(\mu \nu | P)`.
    Semi-sparsely, because it is assumed that there are many
    exchange pairs :math:`\mu, \nu` which are zero, while the integral along
    the auxiliary basis :math:`P` is stored densely as numpy array.

    2-fold permutational symmetry for the :math:`\mu, \nu` pairs is assumed, i.e.

    .. math::

        (\mu \nu | P) == (\nu, \mu | P)
    """

    _data: dict[int64, Vector[float64]]
    nao: int64
    naux: int64
    exch_reachable: list[Vector[OrbitalIdx]]
    exch_reachable_unique: list[Vector[OrbitalIdx]]

    def __getitem__(self, key: tuple[OrbitalIdx, OrbitalIdx]) -> Vector[float64]:
        # We have to ignore the type here, because tuples are invariant, i.e.
        # (OrbitalIdx, OrbitalIdx) is not a subtype of (int, int).
        return self._data[self.idx(*key)]  # type: ignore[arg-type]

    def n_unique_nonzero(self) -> int:
        return len(self._data)

    def traverse_nonzero(
        self, unique: bool = True
    ) -> Iterator[tuple[OrbitalIdx, OrbitalIdx, Vector[float64]]]:
        reachable = self.exch_reachable_unique if unique else self.exch_reachable
        for p in range(self.nao):
            for q in reachable[p]:
                yield (p, q, self._data[self.idx(p, q)])  # type: ignore[misc]

    @staticmethod
    def idx(a: int, b: int) -> int:
        """Return compound index"""
        return ravel_symmetric(a, b)


@jitclass(
    [
        ("_data", DictType(int64, float64[::1])),
        ("dense_data", float64[:, ::1]),
        ("exch_reachable", ListType(int64[::1])),
        ("exch_reachable_unique", ListType(int64[::1])),
    ]
)
class SemiSparseSym3DTensor(_ABC_MutableSemiSparse3DTensor):
    r"""Special datastructure for semi-sparse and partially symmetric 3-indexed tensors.

    For a tensor, :math:`T_{ijk}`, to be stored in this datastructure we assume

    - 2-fold permutational symmetry for the :math:`i, j` indices,
      i.e. :math:`T_{ijk} = T_{jik}`
    - sparsity along the :math:`i, j` indices, i.e. :math:`T_{ijk} = 0`
      for many :math:`i, j`
    - dense storage along the :math:`k` index

    It can be used for example to store the 3-center, 2-electron integrals
    :math:`(\mu \nu | P)`, with AOs :math:`\mu, \nu` and auxiliary basis indices
    :math:`P`.
    Semi-sparsely, because it is assumed that there are many
    exchange pairs :math:`\mu, \nu` which are zero, while the integral along
    the auxiliary basis :math:`P` is stored densely as numpy array.

    2-fold permutational symmetry for the :math:`\mu, \nu` pairs is assumed, i.e.

    .. math::

        (\mu \nu | P) == (\nu, \mu | P)

    Note that this class is immutable which enables to store the unique, non-zero data
    in a dense manner, which has some performance benefits.
    If you need a mutable version, use :class:`MutableSemiSparse3DTensor`,
    which can be always converted to an immutable version
    via :meth:`~MutableSemiSparse3DTensor.make_immutable`.
    """

    dense_data: Matrix[float64]

    def __init__(
        self,
        dense_data: Matrix[float64],
        nao: int,
        naux: int,
        exch_reachable: list[Vector[int64]],
    ) -> None:
        self._data = Dict.empty(int64, float64[::1])
        self.nao = nao
        self.naux = naux
        self.exch_reachable = exch_reachable
        self.exch_reachable_unique = _jit_account_for_symmetry(exch_reachable)

        self.dense_data = dense_data
        for p in range(self.nao):
            for q in self.exch_reachable_unique[p]:
                # Note that this assigns only a view into the dense data array.
                # This is exactly where we use the non-mutability to have a contiguous
                # array for storing the data.
                self._data[self.idx(p, q)] = dense_data[self.idx(p, q), :]


@jitclass(
    [
        ("_data", DictType(int64, float64[::1])),
        ("exch_reachable", ListType(int64[::1])),
        ("exch_reachable_unique", ListType(int64[::1])),
    ]
)
class MutableSemiSparse3DTensor(_ABC_MutableSemiSparse3DTensor):
    r"""Semi-Sparsely store the 2-electron integrals with the auxiliary basis

    This class semi-sparsely stores the elements of the 3-indexed tensor
    :math:`(\mu \nu | P)`.
    Semi-sparsely, because it is assumed that there are many
    exchange pairs :math:`\mu, \nu` which are zero, while the integral along
    the auxiliary basis :math:`P` is stored densely as numpy array.

    2-fold permutational symmetry for the :math:`\mu, \nu` pairs is assumed, i.e.

    .. math::

        (\mu \nu | P) == (\nu, \mu | P)
    """

    _data: dict[int64, Vector[float64]]
    nao: int64
    naux: int64
    exch_reachable: list[Vector[OrbitalIdx]]
    exch_reachable_unique: list[Vector[OrbitalIdx]]

    def __init__(
        self, nao: int, naux: int, exch_reachable: list[Vector[int64]]
    ) -> None:
        self._data = Dict.empty(int64, float64[::1])
        self.nao = nao
        self.naux = naux
        self.exch_reachable = exch_reachable
        self.exch_reachable_unique = _jit_account_for_symmetry(exch_reachable)

    def __setitem__(
        self, key: tuple[OrbitalIdx, OrbitalIdx], value: Vector[float64]
    ) -> None:
        # We have to ignore the type here, because tuples are invariant, i.e.
        # (OrbitalIdx, OrbitalIdx) is not a subtype of (int, int).
        self._data[self.idx(*key)] = value  # type: ignore[arg-type]

    # There is a very strange bug in sphinx-autodoc-typehints,
    #  https://github.com/tox-dev/sphinx-autodoc-typehints/issues/532
    # to circumenvent this, we must not annotate the return type of
    # ``get_dense_data`` and ``make_immutable``.

    def get_dense_data(self):  # type: ignore[no-untyped-def]
        """Return dense data array"""
        result = np.empty((self.n_unique_nonzero(), self.naux), dtype=float64)
        i = 0
        for p in range(self.nao):
            for q in self.exch_reachable_unique[p]:
                result[i, :] = self[p, q]  # type: ignore[index]
                i += 1
        return result

    def make_immutable(self):  # type: ignore[no-untyped-def]
        return SemiSparseSym3DTensor(
            self.get_dense_data(), self.nao, self.naux, self.exch_reachable
        )


def get_orbs_per_atom(
    atom_per_orb: Mapping[_T_orb_idx, Set[AtomIdx]],
) -> dict[AtomIdx, set[_T_orb_idx]]:
    orb_per_atom = defaultdict(set)
    for i_AO, atoms in atom_per_orb.items():
        for i_atom in atoms:
            orb_per_atom[i_atom].add(i_AO)
    return dict(orb_per_atom)


def get_orbs_reachable_by_atom(
    orb_per_atom: Mapping[AtomIdx, Set[_T_orb_idx]],
    screened: Mapping[AtomIdx, Set[AtomIdx]],
) -> dict[AtomIdx, dict[AtomIdx, Set[_T_orb_idx]]]:
    return {
        i_atom: {j_atom: orb_per_atom[j_atom] for j_atom in sorted(connected)}
        for i_atom, connected in screened.items()
    }


def get_orbs_reachable_by_orb(
    atom_per_orb: Mapping[_T_start_orb, Set[AtomIdx]],
    reachable_orb_per_atom: Mapping[AtomIdx, Mapping[AtomIdx, Set[_T_target_orb]]],
) -> dict[_T_start_orb, dict[AtomIdx, Mapping[AtomIdx, Set[_T_target_orb]]]]:
    return {
        i_AO: {atom: reachable_orb_per_atom[atom] for atom in atoms}
        for i_AO, atoms in atom_per_orb.items()
    }


def get_atom_per_AO(mol: Mole) -> dict[AOIdx, set[AtomIdx]]:
    AOs_per_atom = _get_AOidx_per_atom(mol, frozen_core=False)
    n_AO = AOs_per_atom[-1][-1] + 1

    def get_atom(i_AO: AOIdx, AO_per_atom: Sequence[Sequence[AOIdx]]) -> AtomIdx:
        for i_atom, AOs in enumerate(AO_per_atom):
            if i_AO in AOs:
                return cast(AtomIdx, i_atom)
        raise ValueError(f"{i_AO} not contained in AO_per_atom")

    return {
        i_AO: {get_atom(i_AO, AOs_per_atom)}
        for i_AO in cast(Sequence[AOIdx], range(n_AO))
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
    atoms_per_orb: Mapping[_T_orb_idx, Set[AtomIdx]],
    screening_cutoff: Real | Callable[[Real], Real] | Mapping[str, Real] = 5,
) -> dict[_T_orb_idx, set[_T_orb_idx]]:
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


def get_complement(
    reachable: Mapping[_T_orb_idx, Set[_T_orb_idx]],
) -> dict[_T_orb_idx, set[_T_orb_idx]]:
    """Return the orbitals that cannot be reached by an orbital after screening."""
    total: Final = cast(set[_T_orb_idx], set(range(len(reachable))))
    return {i_AO: total - reachable[i_AO] for i_AO in reachable}


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
    reachable: Mapping[_T_start_orb, Collection[_T_target_orb]],
) -> dict[_T_start_orb, list[_T_target_orb]]:
    """Account for permutational symmetry and remove all q that are larger than p.

    Paramaters
    ----------
    reachable :

    Example
    -------
    >>> account_for_symmetry({0: [0, 1, 2], 1: [0, 1, 2], 2: [0, 1, 2]})
    >>> {0: [0], 1: [0, 1], 2: [0, 1, 2]}
    """
    return {
        p: list(takewhile(lambda q: p >= q, sorted(qs)))  # type: ignore[type-var]
        for (p, qs) in reachable.items()
    }


@njit
def _jit_account_for_symmetry(
    reachable: list[Vector[_T_orb_idx]],
) -> list[Vector[_T_orb_idx]]:
    """Account for permutational symmetry and remove all q that are larger than p.

    This is a jitted version of :func:`account_for_symmetry`.

    Paramaters
    ----------
    reachable :
    """
    # TODO: make an early return for performance reasons
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
    screening_cutoff: Real | Callable[[Real], Real] | Mapping[str, Real] | None = None,
) -> SemiSparseSym3DTensor:
    """Return the 3-center 2-electron integrals in a sparse format."""

    if screening_cutoff is None:
        screening_cutoff = find_screening_radius(mol, auxmol)
    exch_reachable = cast(
        Mapping[AOIdx, Set[AOIdx]],
        get_reachable(mol, get_atom_per_AO(mol), screening_cutoff),
    )
    sparse_ints_3c2e = MutableSemiSparse3DTensor(
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
    return sparse_ints_3c2e.make_immutable()


def _flatten(
    orb_reachable_by_orb: Mapping[
        _T_start_orb, Mapping[AtomIdx, Mapping[AtomIdx, Set[_T_target_orb]]]
    ],
) -> dict[_T_start_orb, set[_T_target_orb]]:
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


def _get_test_mol(atom1: str, atom2: str, r: float, basis: str) -> Mole:
    """Return a PySCF Mole object with two atoms at a distance r."""
    m = Cartesian.set_atom_coords([atom1, atom2], np.array([[0, 0, 0], [0, 0, r]]))
    return m.to_pyscf(
        basis=basis,
        charge=m.add_data("atomic_number").loc[:, "atomic_number"].sum() % 2,
    )


def _calc_residual(mol: Mole) -> dict[tuple[AOIdx, AOIdx], float]:
    r"""Return the residual of the 2-electron integrals that are sceened away.

    This is only the diagonal elements of the type :math:`(\mu \nu | \mu \nu)` which
    give upper bounds to the other 2-electron integrals, due to the
    Schwarz inequality.
    """
    screened_away = account_for_symmetry(
        get_complement(get_reachable(mol, get_atom_per_AO(mol), 0.0))
    )
    g = mol.intor("int2e")
    return {
        (p, q): g[p, q, p, q] for p in screened_away.keys() for q in screened_away[p]
    }


def _calc_aux_residual(
    mol: Mole, auxmol: Mole
) -> dict[tuple[AOIdx, AOIdx], Vector[float64]]:
    r"""Return the residual of the :math:`(\mu,\nu | P) integrals that are sceened away.

    Here :math:`\mu, \nu` are the AO indices and :math:`P` is the auxiliary basis.
    For a screened AO pair :math:`(\mu, \nu)`, the whole vector along :math:`P`
    is returned.
    """
    screened_away = account_for_symmetry(
        get_complement(get_reachable(mol, get_atom_per_AO(mol), 0.0))
    )
    ints_3c2e = df.incore.aux_e2(mol, auxmol, intor="int3c2e")
    return {
        (p, q): ints_3c2e[p, q, :]
        for p in screened_away.keys()
        for q in screened_away[p]
    }


def _find_screening_cutoff_distance_aux(
    atom1: str, atom2: str, basis: str, auxbasis: str, threshold: float = 1e-8
) -> float:
    def f(r: float) -> float:
        mol = _get_test_mol(atom1, atom2, r, basis=basis)
        auxmol = df.make_auxmol(mol, auxbasis)
        residual = _calc_aux_residual(mol, auxmol)
        return max(sum(abs(x)) for x in residual.values())

    return bisect(lambda x: (f(x) - threshold), 1, 50, xtol=1e-2)


def _find_screening_cutoff_distance(
    atom1: str, atom2: str, basis: str, threshold: float = 1e-8
) -> float:
    """Return the distance at which the exchange part of the 2-electron integrals
    are lower than the threshold.
    """

    def f(r: float) -> float:
        mol = _get_test_mol(atom1, atom2, r, basis=basis)
        residual = _calc_residual(mol)
        return max(abs(x) for x in residual.values())

    return bisect(lambda x: (f(x) - threshold), 1, 50, xtol=1e-2)


def find_screening_radius(
    mol: Mole,
    auxmol: Mole | None = None,
    threshold: float = 1e-8,
    scale_factor: float = 1.03,
) -> dict[str, float]:
    r"""Return a dictionary with radii for each element in ``mol``
    that can be used to screen the 2-electron integrals to be lower than threshold.

    For a threshhold :math:`T` and for all screened pairs of
    :math:`\mu, \nu` the screening radius is defined in the following way:
    If ``auxmol`` is not given, the screening radius is calculated such that
    :math:`(\mu \nu | \mu \nu) < T`.
    If ``auxmol`` is given, the screening radius is calculated such that
    :math:`\Sum_P |(\mu \nu | P)| < T`.

    Parameters
    ----------
    mol :
        The molecule for which the screening radii are calculated.
    auxmol :
        The molecule with the auxiliary basis.
    threshold :
        The threshold for the integral values.
    scale_factor :
        The scaling factor for the screening radius.
    """
    basis = mol.basis
    auxbasis = auxmol.basis if auxmol is not None else None
    atoms = set(mol.elements)
    if auxbasis is None:
        return {
            atom: _find_screening_cutoff_distance(
                atom, atom, basis, threshold=threshold
            )
            / 2
            * scale_factor
            for atom in atoms
        }
    else:
        return {
            atom: _find_screening_cutoff_distance_aux(
                atom, atom, basis, auxbasis, threshold=threshold
            )
            / 2
            * scale_factor
            for atom in atoms
        }
