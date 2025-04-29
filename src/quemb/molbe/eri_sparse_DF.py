from __future__ import annotations

from collections import defaultdict
from collections.abc import (
    Callable,
    Collection,
    Hashable,
    Iterator,
    Mapping,
    Sequence,
    Set,
)
from itertools import chain, takewhile
from typing import Final, TypeVar, cast

import h5py
import numpy as np
from chemcoord import Cartesian
from numba import float64, int64, prange, uint64  # type: ignore[attr-defined]
from numba.typed import Dict, List
from numba.types import (  # type: ignore[attr-defined]
    DictType,
    ListType,
    UniTuple,
)
from pyscf import df, gto, scf
from pyscf.ao2mo.addons import restore
from pyscf.df import addons
from pyscf.gto import Mole
from pyscf.gto.moleintor import getints
from pyscf.lib import einsum
from scipy.linalg import solve
from scipy.optimize import bisect

from quemb.molbe.chemfrag import (
    _get_AOidx_per_atom,
)
from quemb.molbe.pfrag import Frags
from quemb.shared.helper import (
    jitclass,
    njit,
    ravel,
    ravel_symmetric,
)
from quemb.shared.numba_helpers import SortedIntSet, Type_SortedIntSet
from quemb.shared.typing import (
    AOIdx,
    AtomIdx,
    Integral,
    Matrix,
    MOIdx,
    OrbitalIdx,
    Real,
    ShellIdx,
    Tensor3D,
    Tensor4D,
    Vector,
)

_T_orb_idx = TypeVar("_T_orb_idx", bound=OrbitalIdx)
_T_start_orb = TypeVar("_T_start_orb", bound=OrbitalIdx)
_T_target_orb = TypeVar("_T_target_orb", bound=OrbitalIdx)

_T_start = TypeVar("_T_start", bound=np.integer)
_T_target = TypeVar("_T_target", bound=np.integer)
_T = TypeVar("_T", int, np.integer)


def _aux_e2(  # type: ignore[no-untyped-def]
    mol: Mole,
    auxmol_or_auxbasis: Mole | str,
    intor: str = "int3c2e",
    aosym: str = "s1",
    comp: int | None = None,
    out: Tensor4D[np.float64] | None = None,
    cintopt=None,
    shls_slice: tuple[int, int, int, int, int, int] | list[int] | None = None,
) -> Tensor3D[np.float64]:
    """3-center AO integrals (ij|L), where L is the auxiliary basis.

    Fixes a bug in the original implementation :func:`pyscf.df.incore.aux_e2`
    that does not accept all valid slices.
    Replace with the original, as soon as https://github.com/pyscf/pyscf/pull/2734
    is in the stable release.
    """
    if isinstance(auxmol_or_auxbasis, gto.MoleBase):
        auxmol = auxmol_or_auxbasis
    else:
        auxbasis = auxmol_or_auxbasis
        auxmol = addons.make_auxmol(mol, auxbasis)
    if shls_slice is None:
        shls_slice = (0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas + auxmol.nbas)
    else:
        assert len(shls_slice) == 6
        # The following line is the difference to pyscf
        assert shls_slice[5] <= auxmol.nbas
        shls_slice = list(shls_slice)
        shls_slice[4] += mol.nbas
        shls_slice[5] += mol.nbas

    intor = mol._add_suffix(intor)
    hermi = 0
    ao_loc = None
    atm, bas, env = gto.mole.conc_env(
        mol._atm, mol._bas, mol._env, auxmol._atm, auxmol._bas, auxmol._env
    )
    return getints(
        intor, atm, bas, env, shls_slice, comp, hermi, aosym, ao_loc, cintopt, out
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
        idx = self.idx(*key)
        return self._data.get(idx, 0.0)

    def __setitem__(self, key: tuple[int, int, int, int], value: float) -> None:
        self._data[self.idx(*key)] = value

    @staticmethod
    def idx(a: int, b: int, c: int, d: int) -> int:
        """Return compound index given four indices using Yoshimine sort"""
        return ravel_symmetric(ravel_symmetric(a, b), ravel_symmetric(c, d))


def get_sparse_DF_integrals(
    mol: Mole,
    auxmol: Mole,
    screening_cutoff: Real | Callable[[Real], Real] | Mapping[str, Real] | None = None,
) -> tuple[SemiSparseSym3DTensor, SemiSparseSym3DTensor]:
    """Return the 3-center 2-electron integrals in a sparse format with density fitting

    One can obtain :python:`g[p, q, r, s] == ints_3c2e[p, q, :] @ df_coef[r, s, :]`.

    Parameters
    ----------
    mol :
        The molecule.
    auxmol :
        The molecule with auxiliary basis functions.
    screening_cutoff :
        The screening cutoff is given by the overlap of van der Waals radii.
        By default, the radii are determined by :func:`find_screening_radius`.
        Alternatively, a fixed radius, callable or a dictionary can be passed.
        The callable is called with the tabulated van der Waals radius
        of the atom as argument and can be used to scale it up.
        The dictionary can be used to define different van der Waals radii
        for different elements. Compare to the :python:`modify_element_data`
        argument of :meth:`~chemcoord.Cartesian.get_bonds`.
    """
    ints_3c2e = _get_sparse_ints_3c2e(mol, auxmol, screening_cutoff)
    ints_2c2e = auxmol.intor("int2c2e")
    df_coeffs_data = solve(ints_2c2e, ints_3c2e.unique_dense_data.T).T
    df_coef = SemiSparseSym3DTensor(
        df_coeffs_data,
        ints_3c2e.nao,
        ints_3c2e.naux,
        ints_3c2e.exch_reachable,  # type: ignore[arg-type]
    )
    return ints_3c2e, df_coef


@njit(parallel=True)
def get_dense_integrals(
    ints_3c2e: SemiSparseSym3DTensor, df_coef: SemiSparseSym3DTensor
) -> Tensor4D[np.float64]:
    r"""Compute dense ERIs from sparse 3-center integrals and sparse DF coefficients.

    We evaluate the integrals via

    .. math::

        (\mu, \nu | \rho \sigma) = \sum_{P} (\mu \nu | P) C^{P}_{\rho\sigma}


    Parameters
    ----------
    ints_3c2e :
        Sparse 3-center integrals in the form of a :class:`SemiSparseSym3DTensor`.
        :math:`(\mu \nu | P)` is given by :python:`ints_3c2e[mu, nu][P]`.
    df_coef :
        DF coefficients in the form of a :class:`SemiSparseSym3DTensor`.
        :math:`C^{P}_{\mu\nu}` is given by :python:`df_coef[mu, nu][P]`.
    """
    g = np.zeros((ints_3c2e.nao, ints_3c2e.nao, ints_3c2e.nao, ints_3c2e.nao))

    for mu in prange(ints_3c2e.nao):  # type: ignore[attr-defined]
        for nu in ints_3c2e.exch_reachable_unique[mu]:
            for rho in prange(ints_3c2e.nao):  # type: ignore[attr-defined]
                # Ensure (mu nu, rho sigma) ordering
                if mu > rho:
                    idx = len(ints_3c2e.exch_reachable_unique[rho])
                else:
                    idx = np.searchsorted(
                        ints_3c2e.exch_reachable_unique[rho], nu, side="right"
                    )
                for sigma in ints_3c2e.exch_reachable_unique[rho][:idx]:
                    g[mu, nu, rho, sigma] = ints_3c2e[mu, nu] @ df_coef[rho, sigma]
                    g[mu, nu, sigma, rho] = g[mu, nu, rho, sigma]
                    g[nu, mu, rho, sigma] = g[mu, nu, rho, sigma]
                    g[nu, mu, sigma, rho] = g[mu, nu, rho, sigma]
                    g[rho, sigma, mu, nu] = g[mu, nu, rho, sigma]
                    g[sigma, rho, mu, nu] = g[mu, nu, rho, sigma]
                    g[rho, sigma, nu, mu] = g[mu, nu, rho, sigma]
                    g[sigma, rho, nu, mu] = g[mu, nu, rho, sigma]
    return g


@jitclass(
    [
        ("_keys", int64[::1]),
        ("shape", UniTuple(int64, 3)),
        ("unique_dense_data", float64[:, ::1]),
        ("exch_reachable", ListType(int64[::1])),
        ("exch_reachable_unique", ListType(int64[::1])),
    ]
)
class SemiSparseSym3DTensor:
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
    """

    _keys: Vector[np.int64]
    unique_dense_data: Matrix[np.float64]
    nao: int
    naux: int
    shape: tuple[int, int, int]
    exch_reachable: list[Vector[OrbitalIdx]]
    exch_reachable_unique: list[Vector[OrbitalIdx]]

    def __init__(
        self,
        unique_dense_data: Matrix[np.float64],
        nao: Integral,
        naux: Integral,
        exch_reachable: list[Vector[np.int64]],
    ) -> None:
        self.nao = nao  # type: ignore[assignment]
        self.naux = naux  # type: ignore[assignment]
        self.shape = (self.nao, self.nao, self.naux)
        self.exch_reachable = exch_reachable  # type: ignore[assignment]
        self.exch_reachable_unique = _jit_account_for_symmetry(exch_reachable)  # type: ignore[arg-type]

        self.unique_dense_data = unique_dense_data

        self._keys = np.array(  # type: ignore[call-overload]
            [
                self.idx(p, q)  # type: ignore[arg-type]
                for p in range(self.nao)
                for q in self.exch_reachable_unique[p]
            ],
            dtype=int64,
        )

    def __getitem__(self, key: tuple[OrbitalIdx, OrbitalIdx]) -> Vector[np.float64]:
        look_up_idx = np.searchsorted(self._keys, self.idx(key[0], key[1]))
        return self.unique_dense_data[look_up_idx]

    # We cannot annotate the return type of this function, because of a strange bug in
    #  sphinx-autodoc-typehints.
    #  https://github.com/tox-dev/sphinx-autodoc-typehints/issues/532
    def to_dense(self):  # type: ignore[no-untyped-def]
        """Convert to dense 3D tensor"""
        g = np.zeros((self.nao, self.nao, self.naux))
        for p in range(self.nao):
            for q in self.exch_reachable_unique[p]:
                g[p, q] = self[p, q]  # type: ignore[index]
                g[q, p] = self[p, q]  # type: ignore[index]
        return g

    @staticmethod
    def idx(a: OrbitalIdx, b: OrbitalIdx) -> int:
        """Return compound index"""
        return ravel_symmetric(a, b)  # type: ignore[return-value]


@njit
def _get_list_of_sortedset(n: int) -> list[SortedIntSet]:
    L = List.empty_list(Type_SortedIntSet)
    for _ in range(n):
        L.append(SortedIntSet())
    return L


_UniTuple_int64_2 = UniTuple(int64, 2)


@jitclass(
    [
        ("shape", UniTuple(int64, 3)),
        ("naux", int64),
        ("_data", DictType(_UniTuple_int64_2, float64[:])),
        ("exch_reachable_mu", ListType(Type_SortedIntSet)),
        ("exch_reachable_i", ListType(Type_SortedIntSet)),
    ]
)
class MutableSemiSparse3DTensor:
    def __init__(
        self,
        shape: tuple[int, int, int],
    ) -> None:
        self.shape = shape
        self.naux = shape[-1]
        self._data = Dict.empty(_UniTuple_int64_2, float64[:])
        self.exch_reachable_mu = _get_list_of_sortedset(self.shape[0])
        self.exch_reachable_i = _get_list_of_sortedset(self.shape[1])

    def __getitem__(self, key: tuple[OrbitalIdx, OrbitalIdx]) -> Vector[np.float64]:
        assert key[0] < self.shape[0] and key[1] < self.shape[1]
        return self._data[key]

    def __setitem__(
        self, key: tuple[OrbitalIdx, OrbitalIdx], value: Vector[np.float64]
    ) -> None:
        assert (
            key[0] < self.shape[0]
            and key[1] < self.shape[1]
            and len(value) == self.naux
        )
        self._data[key] = value
        self.exch_reachable_mu[key[0]].add(key[1])
        self.exch_reachable_i[key[1]].add(key[0])

    # We cannot annotate the return type of this function, because of a strange bug in
    #  sphinx-autodoc-typehints.
    #  https://github.com/tox-dev/sphinx-autodoc-typehints/issues/532
    def to_dense(self):  # type: ignore[no-untyped-def]
        result = np.zeros((self.shape), dtype="f8")
        for mu in range(self.shape[0]):
            for i in self.exch_reachable_mu[mu].items:
                result[mu, i, :] = self[mu, i]  # type: ignore[index]
        return result


@jitclass(
    [
        ("_keys", int64[::1]),
        ("unique_dense_data", float64[:, ::1]),
        ("shape", UniTuple(int64, 3)),
        ("exch_reachable", ListType(int64[::1])),
    ]
)
class SemiSparse3DTensor:
    r"""Special datastructure for semi-sparse and partially symmetric 3-indexed tensors.

    For a tensor, :math:`T_{ijk}`, to be stored in this datastructure we assume

    - sparsity along the :math:`i, j` indices, i.e. :math:`T_{ijk} = 0`
      for many :math:`i, j`
    - dense storage along the :math:`k` index

    It can be used for example to store the 3-center, 2-electron integrals
    :math:`(\mu i | P)`, with AOs :math:`\mu`, MOs :math:`i`,
    and auxiliary basis indices :math:`P`.
    Semi-sparsely, because it is assumed that there are many
    exchange pairs :math:`\mu, \nu` which are zero, while the integral along
    the auxiliary basis :math:`P` is stored densely as numpy array.

    Note that this class is immutable which enables to store the unique, non-zero data
    in a dense manner, which has some performance benefits.
    """

    _keys: Vector[np.int64]
    unique_dense_data: Matrix[np.float64]
    shape: tuple[int, int, int]
    naux: int
    exch_reachable: list[Vector[OrbitalIdx]]

    def __init__(
        self,
        unique_dense_data: Matrix[np.float64],
        shape: tuple[Integral, Integral, Integral],
        exch_reachable: list[Vector[np.int64]],
    ) -> None:
        self.shape = shape  # type: ignore[assignment]
        self.naux = shape[-1]  # type: ignore[assignment]
        self.exch_reachable = exch_reachable  # type: ignore[assignment]

        self.unique_dense_data = unique_dense_data

        self._keys = np.array(  # type: ignore[call-overload]
            [
                self._idx(p, q)  # type: ignore[arg-type]
                for p in range(self.shape[0])
                for q in self.exch_reachable[p]
            ],
            dtype=int64,
        )

    def __getitem__(self, key: tuple[OrbitalIdx, OrbitalIdx]) -> Vector[np.float64]:
        look_up_idx = np.searchsorted(self._keys, self._idx(key[0], key[1]))
        return self.unique_dense_data[look_up_idx]

    # We cannot annotate the return type of this function, because of a strange bug in
    #  sphinx-autodoc-typehints.
    #  https://github.com/tox-dev/sphinx-autodoc-typehints/issues/532
    def to_dense(self):  # type: ignore[no-untyped-def]
        """Convert to dense 3D tensor"""
        g = np.zeros(self.shape)
        for p in range(self.shape[0]):
            for q in self.exch_reachable[p]:
                g[p, q] = self[p, q]  # type: ignore[index]
        return g

    def _idx(self, a: OrbitalIdx, b: OrbitalIdx) -> int:
        """Return compound index"""
        return ravel(a, b, n_cols=self.shape[1])  # type: ignore[return-value]


def _traverse_reachable(
    reachable: Mapping[_T_start_orb, Collection[_T_target_orb]],
) -> Iterator[tuple[_T_start_orb, _T_target_orb]]:
    """Traverse reachable p, q pairs"""
    for p in reachable:
        for q in reachable[p]:
            yield p, q


def traverse_nonzero(
    g: SemiSparseSym3DTensor,
    unique: bool = True,
) -> Iterator[tuple[OrbitalIdx, OrbitalIdx]]:
    """Traverse the non-zero elements of a semi-sparse 3-index tensor.

    Parameters
    ----------
    g :
    unique :
        Whether to account for 2-fold permutational symmetry
        and only return :python:`p >= q`.
    """
    # NOTE that this cannot be a jitted method, since generators sometimes
    # introduce hard to debug memory-leaks
    # https://github.com/numba/numba/issues/5427
    # https://github.com/numba/numba/issues/5350
    # https://github.com/numba/numba/issues/6993
    # https://github.com/numba/numba/issues/5350
    reachable = g.exch_reachable_unique if unique else g.exch_reachable
    for p in range(g.nao):
        for q in reachable[p]:
            yield cast(tuple[OrbitalIdx, OrbitalIdx], (p, q))


_T_old_key = TypeVar("_T_old_key", bound=Hashable)
_T_new_key = TypeVar("_T_new_key", bound=Hashable)


def _invert_dict(
    D: Mapping[_T_old_key, Collection[_T_new_key]],
) -> dict[_T_new_key, set[_T_old_key]]:
    inverted_D = defaultdict(set)
    for old_key, new_keys in D.items():
        for new_key in new_keys:
            inverted_D[new_key].add(old_key)
    return dict(inverted_D)


def get_orbs_per_atom(
    atom_per_orb: Mapping[_T_orb_idx, Set[AtomIdx]],
) -> dict[AtomIdx, set[_T_orb_idx]]:
    return _invert_dict(atom_per_orb)


def get_orbs_reachable_by_atom(
    orb_per_atom: Mapping[AtomIdx, Set[_T_orb_idx]],
    screened: Mapping[AtomIdx, Set[AtomIdx]],
) -> dict[AtomIdx, dict[AtomIdx, Set[_T_orb_idx]]]:
    return {
        i_atom: {
            j_atom: orb_per_atom.get(j_atom, set()) for j_atom in sorted(connected)
        }
        for i_atom, connected in screened.items()
    }


def get_orbs_reachable_by_orb(
    atom_per_orb: Mapping[_T_start_orb, Set[AtomIdx]],
    reachable_orb_per_atom: Mapping[AtomIdx, Mapping[AtomIdx, Set[_T_target_orb]]],
) -> dict[_T_start_orb, dict[AtomIdx, Mapping[AtomIdx, Set[_T_target_orb]]]]:
    """Concatenate the :python:`atom_per_orb` and :python:`reachable_orb_per_atom`
    Such that it becomes a mapping :python:`i_orb -> i_atom -> j_atom`
    """
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


def get_atom_per_MO(
    atom_per_AO: Mapping[AOIdx, Set[AtomIdx]],
    TA: Matrix[np.float64],
    epsilon: float = 1e-8,
) -> dict[MOIdx, set[AtomIdx]]:
    n_MO = TA.shape[-1]
    large_enough = {
        i_MO: (TA[:, i_MO] ** 2 > epsilon).nonzero()[0]
        for i_MO in cast(Sequence[MOIdx], range(n_MO))
    }
    return {
        i_MO: set().union(*(atom_per_AO[AO] for AO in AO_indices))
        for i_MO, AO_indices in large_enough.items()
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
        cast(ShellIdx, shell_id): cast(
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
    atoms_per_start_orb: Mapping[_T_start_orb, Set[AtomIdx]],
    atoms_per_target_orb: Mapping[_T_target_orb, Set[AtomIdx]],
    screening_cutoff: Real | Callable[[Real], Real] | Mapping[str, Real] = 5,
) -> dict[_T_start_orb, list[_T_target_orb]]:
    """Return the sorted orbitals that can by reached for each orbital after screening.

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
            atoms_per_start_orb,
            get_orbs_reachable_by_atom(
                get_orbs_per_atom(atoms_per_target_orb), screen_conn
            ),
        )
    )


def get_complement(
    reachable: Mapping[_T_start, Sequence[_T_target]],
) -> dict[_T_start, list[_T_target]]:
    """Return the orbitals that cannot be reached by an orbital after screening."""
    total: Final = cast(set[_T_target], set(range(len(reachable))))
    return {i_AO: sorted(total - set(reachable[i_AO])) for i_AO in reachable}  # type: ignore[type-var]


def to_numba_input(
    exch_reachable: Mapping[_T_orb_idx, Collection[_T_orb_idx]],
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
    reachable: Mapping[_T_start, Collection[_T_target]],
) -> dict[_T_start, list[_T_target]]:
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


def identify_contiguous_blocks(X: Sequence[_T]) -> list[tuple[int, int]]:
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


def _get_sparse_ints_3c2e(
    mol: Mole,
    auxmol: Mole,
    screening_cutoff: Real | Callable[[Real], Real] | Mapping[str, Real] | None = None,
) -> SemiSparseSym3DTensor:
    """Return the 3-center 2-electron integrals in a sparse format."""

    if screening_cutoff is None:
        screening_cutoff = find_screening_radius(mol, auxmol)

    atom_per_AO = get_atom_per_AO(mol)
    exch_reachable = cast(
        Mapping[AOIdx, Set[AOIdx]],
        get_reachable(mol, atom_per_AO, atom_per_AO, screening_cutoff),
    )
    exch_reachable_unique = account_for_symmetry(exch_reachable)

    screened_unique_integrals = np.empty(
        (sum(len(v) for v in exch_reachable_unique.values()), auxmol.nao), order="C"
    )

    shell_id_to_AO, AO_to_shell_id = conversions_AO_shell(mol)
    shell_reachable_by_shell = account_for_symmetry(
        {
            AO_to_shell_id[k]: sorted({AO_to_shell_id[orb] for orb in v})  # type: ignore[type-var]
            for k, v in exch_reachable_unique.items()
        }
    )
    keys = np.array(
        [
            ravel_symmetric(p, q)
            for (p, q) in _traverse_reachable(exch_reachable_unique)
        ],
        dtype=np.int64,
    )

    for i_shell, reachable in shell_reachable_by_shell.items():
        for start_block, stop_block in get_blocks(reachable):
            integrals = np.asarray(  # type: ignore[call-overload]
                _aux_e2(
                    mol,
                    auxmol,
                    intor="int3c2e",
                    shls_slice=(  # type: ignore[arg-type]
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
                        shell_id_to_AO[start_block][0],  # type: ignore[index]
                        # still ensure p <= q
                        min(shell_id_to_AO[stop_block][-1] + 1, p + 1),  # type: ignore[index]
                    )
                ):
                    screened_unique_integrals[
                        np.searchsorted(keys, ravel_symmetric(p, q)), :
                    ] = integrals[i, j, ::1]

    return SemiSparseSym3DTensor(
        screened_unique_integrals, mol.nao, auxmol.nao, to_numba_input(exch_reachable)
    )


def _flatten(
    orb_reachable_by_orb: Mapping[
        _T_start_orb, Mapping[AtomIdx, Mapping[AtomIdx, Set[_T_target_orb]]]
    ],
) -> dict[_T_start_orb, list[_T_target_orb]]:
    return {
        i_orb: sorted(  # type: ignore[type-var]
            set(
                chain(
                    *(
                        start_atoms[start_atom][target_atom]
                        for start_atom, target_atoms in start_atoms.items()
                        for target_atom in target_atoms
                    )
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
    atom_per_AO = get_atom_per_AO(mol)
    screened_away = account_for_symmetry(
        get_complement(get_reachable(mol, atom_per_AO, atom_per_AO, 0.0))
    )
    g = mol.intor("int2e")
    return {
        (p, q): g[p, q, p, q] for p in screened_away.keys() for q in screened_away[p]
    }


def _calc_aux_residual(
    mol: Mole, auxmol: Mole
) -> dict[tuple[AOIdx, AOIdx], Vector[np.float64]]:
    r"""Return the residual of :math:`(\mu,\nu | P)` integrals that are sceened away.

    Here :math:`\mu, \nu` are the AO indices and :math:`P` is the auxiliary basis.
    For a screened AO pair :math:`(\mu, \nu)`, the whole vector along :math:`P`
    is returned.
    """
    atom_per_AO = get_atom_per_AO(mol)
    screened_away = account_for_symmetry(
        get_complement(get_reachable(mol, atom_per_AO, atom_per_AO, 0.0))
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
        nao = mol.nao
        naux = auxmol.nao

        sparse_ints_3c2e, sparse_df_coef = get_sparse_DF_integrals(
            mol, auxmol, screening_cutoff=0.01
        )

        ints_3c2e = df.incore.aux_e2(mol, auxmol, intor="int3c2e")
        ints_2c2e = auxmol.intor("int2c2e")
        df_coef = solve(ints_2c2e, ints_3c2e.reshape(nao * nao, naux).T)
        df_coef = df_coef.reshape((naux, nao, nao), order="F")

        df_eri = einsum("ijP,Pkl->ijkl", ints_3c2e, df_coef)
        df_eri_from_sparse = get_dense_integrals(sparse_ints_3c2e, sparse_df_coef)

        return np.abs(df_eri_from_sparse - df_eri).max()

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
    *,
    threshold: float = 1e-7,
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


@njit
def _first_contract_with_TA(
    TA: Matrix[np.float64],
    int_mu_nu_P: SemiSparseSym3DTensor,
    AO_reachable_by_MO: list[Vector[OrbitalIdx]],
) -> MutableSemiSparse3DTensor:
    assert TA.shape[0] == int_mu_nu_P.nao
    g = MutableSemiSparse3DTensor((int_mu_nu_P.nao, TA.shape[1], int_mu_nu_P.naux))

    for i in range(g.shape[1]):
        for mu in AO_reachable_by_MO[i]:
            tmp = np.zeros(int_mu_nu_P.naux, dtype="f8")
            for nu in int_mu_nu_P.exch_reachable[mu]:
                tmp += TA[nu, i] * int_mu_nu_P[mu, nu]  # type: ignore[index]
            g[mu, i] = tmp  # type: ignore[index]
    return g


@njit(parallel=True)
def _second_contract_with_TA(
    TA: Matrix[np.float64], int_mu_i_P: MutableSemiSparse3DTensor
) -> Tensor3D[np.float64]:
    assert TA.shape[0] == int_mu_i_P.shape[0]
    assert TA.shape[1] == int_mu_i_P.shape[1]

    g = np.zeros((TA.shape[1], TA.shape[1], int_mu_i_P.naux), dtype=np.float64)

    for i in prange(g.shape[0]):  # type: ignore[attr-defined]
        for j in prange(min(g.shape[1], i + 1)):  # type: ignore[attr-defined]
            for nu in int_mu_i_P.exch_reachable_i[i].items:
                g[i, j, :] += TA[nu, j] * int_mu_i_P[nu, i]

            g[j, i, :] = g[i, j, :]
    return g


def transform_sparse_DF_integral(
    mf: scf.hf.SCF,
    Fobjs: Sequence[Frags],
    file_eri_handler: h5py.File,
    auxbasis: str | None = None,
    screen_radius: Mapping[str, float] | None = None,
) -> None:
    eris = _slow_transform_sparse_DF_integral(
        mf,
        Fobjs,
        auxbasis,
        screen_radius,
    )
    write_eris(Fobjs, eris, file_eri_handler)


def _slow_transform_sparse_DF_integral(
    mf: scf.hf.SCF,
    Fobjs: Sequence[Frags],
    auxbasis: str | None = None,
    screen_radius: Mapping[str, float] | None = None,
) -> list[Matrix[np.float64]]:
    mol = mf.mol
    auxmol = addons.make_auxmol(mf.mol, auxbasis=auxbasis)
    if screen_radius is None:
        screen_radius = find_screening_radius(mol, auxmol, threshold=1e-4)
    sparse_ints_3c2e, sparse_df_coef = get_sparse_DF_integrals(
        mol, auxmol, screen_radius
    )
    ints_mu_nu_P = sparse_ints_3c2e.to_dense()
    ints_2c2e = auxmol.intor("int2c2e")

    ints_i_j_P = []

    for fragidx, fragobj in enumerate(Fobjs):
        transf = fragobj.TA.T.copy()
        int_mu_i_P = transf @ ints_mu_nu_P
        ints_i_j_P.append(transf @ np.moveaxis(int_mu_i_P, 1, 0))

    Ds_i_j_P = [solve(ints_2c2e, int_i_j_P.T).T for int_i_j_P in ints_i_j_P]

    return [
        einsum("ijP,klP->ijkl", ints, df_coef)
        for ints, df_coef in zip(ints_i_j_P, Ds_i_j_P)
    ]


def _fast_transform_sparse_DF_integral(
    mf: scf.hf.SCF,
    Fobjs: Sequence[Frags],
    auxbasis: str | None = None,
    screen_radius: Mapping[str, float] | None = None,
) -> list[Matrix[np.float64]]:
    mol = mf.mol
    auxmol = addons.make_auxmol(mf.mol, auxbasis=auxbasis)
    if screen_radius is None:
        screen_radius = find_screening_radius(mol, auxmol, threshold=1e-4)
    sparse_ints_3c2e, sparse_df_coef = get_sparse_DF_integrals(
        mol, auxmol, screen_radius
    )
    ints_2c2e = auxmol.intor("int2c2e")

    atom_per_AO = get_atom_per_AO(mol)

    ints_i_j_P = []
    for fragidx, fragobj in enumerate(Fobjs):
        TA = fragobj.TA

        AO_reachable_per_SchmidtMO = get_reachable(
            mol,
            get_atom_per_MO(atom_per_AO, TA, epsilon=1e-8),
            atom_per_AO,
            screen_radius,
        )
        sparse_int_mu_i_P = _first_contract_with_TA(
            TA, sparse_ints_3c2e, to_numba_input(AO_reachable_per_SchmidtMO)
        )
        int_i_j_P = _second_contract_with_TA(TA, sparse_int_mu_i_P)
        ints_i_j_P.append(int_i_j_P)

    Ds_i_j_P = [solve(ints_2c2e, int_i_j_P.T).T for int_i_j_P in ints_i_j_P]

    return [
        einsum("ijP,klP->ijkl", ints, df_coef)
        for ints, df_coef in zip(ints_i_j_P, Ds_i_j_P)
    ]


def write_eris(
    Fobjs: Sequence[Frags],
    eris: Sequence[Matrix[np.float64]],
    file_eri_handler: h5py.File,
) -> None:
    for fragidx, eri in enumerate(eris):
        file_eri_handler.create_dataset(
            Fobjs[fragidx].dname, data=restore("4", eri, len(eri))
        )
