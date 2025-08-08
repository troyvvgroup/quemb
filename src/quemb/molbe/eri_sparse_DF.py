# Author(s): Oskar Weser

from __future__ import annotations

import logging
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
from concurrent.futures import ThreadPoolExecutor
from itertools import chain, takewhile
from typing import Final, TypeVar, cast

import h5py
import numpy as np
from chemcoord import Cartesian
from numba import float64, int64, prange  # type: ignore[attr-defined]
from numba.typed import Dict, List
from numba.types import (  # type: ignore[attr-defined]
    DictType,
    ListType,
    UniTuple,
)
from pyscf import df, dft, gto, scf
from pyscf.ao2mo.addons import restore
from pyscf.df.addons import make_auxmol
from pyscf.gto import Mole
from pyscf.gto.moleintor import getints
from pyscf.lib import einsum
from scipy.linalg import cholesky, solve, solve_triangular
from scipy.optimize import bisect
from scipy.special import roots_hermite

import quemb.molbe._cpp.eri_sparse_DF as cpp_transforms
from quemb.molbe.chemfrag import (
    _get_AOidx_per_atom,
)
from quemb.molbe.pfrag import Frags
from quemb.shared.helper import (
    Timer,
    gauss_sum,
    jitclass,
    n_symmetric,
    njit,
    ravel_Fortran,
    ravel_symmetric,
    timer,
    unravel_symmetric,
)
from quemb.shared.numba_helpers import (
    PreIncr,
    SortedIntSet,
    Type_SortedIntSet,
)
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

logger = logging.getLogger(__name__)


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
    is merged in the stable release.
    """
    if isinstance(auxmol_or_auxbasis, gto.MoleBase):
        auxmol = auxmol_or_auxbasis
    else:
        auxbasis = auxmol_or_auxbasis
        auxmol = make_auxmol(mol, auxbasis)
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


@njit(nogil=True, inline="always")
def _assign_with_symmetry(
    g: Tensor4D[np.float64], i: int, j: int, k: int, l: int, val: float
) -> None:
    """Assign to ERI using 8-fold symmetry"""
    g[i, j, k, l] = val
    g[i, j, l, k] = val
    g[j, i, k, l] = val
    g[j, i, l, k] = val
    g[k, l, i, j] = val
    g[l, k, i, j] = val
    g[k, l, j, i] = val
    g[l, k, j, i] = val


def get_sparse_D_ints_and_coeffs(
    mol: Mole,
    auxmol: Mole,
    screening_radius: Real | Callable[[Real], Real] | Mapping[str, Real] | None = None,
) -> tuple[SemiSparseSym3DTensor, SemiSparseSym3DTensor]:
    """Return the 3-center 2-electron integrals and fitting coefficients in a
    semi-sparse format for the AO basis

    One can obtain :python:`g[p, q, r, s] == ints_3c2e[p, q] @ df_coef[r, s]`.

    The datastructures use sparsity in the :python:`p, q` and in the :python:`r, s`
    pairs but the dimension along the auxiliary basis is densely stored.


    Parameters
    ----------
    mol :
        The molecule.
    auxmol :
        The molecule with auxiliary basis functions.
    screening_radius :
        The screening cutoff is given by the overlap of van der Waals radii.
        By default, the radii are determined by :func:`find_screening_radius`.
        Alternatively, a fixed radius, callable or a dictionary can be passed.
        The callable is called with the tabulated van der Waals radius
        of the atom as argument and can be used to scale it up.
        The dictionary can be used to define different van der Waals radii
        for different elements. Compare to the :python:`modify_element_data`
        argument of :meth:`~chemcoord.Cartesian.get_bonds`.
    """
    if screening_radius is None:
        screening_radius = find_screening_radius(mol, auxmol)
    atom_per_AO = get_atom_per_AO(mol)
    exch_reachable = get_reachable(
        atom_per_AO, atom_per_AO, get_screened(mol, screening_radius)
    )
    ints_3c2e = get_sparse_P_mu_nu(mol, auxmol, exch_reachable)
    ints_2c2e = auxmol.intor("int2c2e")
    df_coeffs_data = solve(ints_2c2e, ints_3c2e.unique_dense_data.T, assume_a="pos").T
    df_coef = SemiSparseSym3DTensor(
        df_coeffs_data,
        ints_3c2e.nao,
        ints_3c2e.naux,
        ints_3c2e.exch_reachable,  # type: ignore[arg-type]
    )
    return ints_3c2e, df_coef


@njit(nogil=True, parallel=True)
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
                    val = float(ints_3c2e[mu, nu] @ df_coef[rho, sigma])
                    _assign_with_symmetry(g, mu, nu, rho, sigma, val)
    return g


@jitclass(
    [
        ("_keys", int64[::1]),
        ("shape", UniTuple(int64, 3)),
        ("unique_dense_data", float64[:, ::1]),
        ("exch_reachable", ListType(int64[::1])),
        ("exch_reachable_with_offsets", ListType(ListType(UniTuple(int64, 2)))),
        ("exch_reachable_unique", ListType(int64[::1])),
        ("exch_reachable_unique_with_offsets", ListType(ListType(UniTuple(int64, 2)))),
        ("_data", DictType(int64, float64[::1])),
    ]
)
class SemiSparseSym3DTensor:
    r"""Specialised datastructure for immutable, semi-sparse, and partially symmetric
    3-indexed tensors.

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

    Note that this class is immutable which enables to store the unique (symmetry),
    non-zero data in a dense manner, which has some performance benefits.
    """

    _keys: Vector[np.int64]
    #: The non-zero data that also accounts for symmetry.
    unique_dense_data: Matrix[np.float64]
    #: number of AOs
    nao: int
    #: number of auxiliary functions
    naux: int
    #: number of AOs, number of AOs, number of auxiliary functions.
    shape: tuple[int, int, int]
    #: For a given :python:`p` the :python:`exch_reachable[p]`
    #: returns all :python:`q` that are assumed unrelevant
    #: for (p q | r s) after screening.
    #: Note that (p q | P ) might still be non-zero.
    exch_reachable: list[Vector[OrbitalIdx]]
    #: This is the same as :python:`exch_reachable` except it is accounting for symmetry
    #: :python:`p >= q`
    exch_reachable_unique: list[Vector[OrbitalIdx]]

    #: The following datastructures also return the offset to index
    #: the :python:`unique_dense_data` directly and enables very fast
    #: loops without having to compute the offset.
    #:
    #: .. code-block:: python
    #:
    #:    for offset, q in self.exch_reachable_with_offsets[p]:
    #:        self.unique_dense_data[offset] == self[p, q]  # True
    exch_reachable_with_offsets: list[list[tuple[int, OrbitalIdx]]]
    #: The following datastructures also return the offset to index
    #: the :python:`unique_dense_data` directly and enables very fast
    #: loops without having to compute the offset.
    #: It only returns :python:`q` with :python:`p >= q`
    #:
    #: .. code-block:: python
    #:
    #:    for offset, q in self.exch_reachable_unique_with_offsets[p]:
    #:        self.unique_dense_data[offset] == self[p, q]  # True
    exch_reachable_unique_with_offsets: list[list[tuple[int, OrbitalIdx]]]

    _data: Mapping[int, Vector[np.float64]]

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
        assert (self._keys[:-1] < self._keys[1:]).all()

        counter = PreIncr()
        self.exch_reachable_unique_with_offsets = List(
            [
                List([(counter.incr(), q) for q in self.exch_reachable_unique[p]])
                for p in range(self.nao)
            ]
        )
        self.exch_reachable_with_offsets = List(
            [
                List(
                    [
                        (np.searchsorted(self._keys, self.idx(p, q)), q)  # type: ignore[arg-type]
                        for q in self.exch_reachable[p]
                    ]
                )
                for p in range(self.nao)
            ]
        )

        self._data = Dict.empty(int64, float64[::1])
        for i, k in enumerate(self._keys):
            self._data[k] = self.unique_dense_data[i, :]  # type: ignore[index]

    def __getitem__(self, key: tuple[OrbitalIdx, OrbitalIdx]) -> Vector[np.float64]:
        return self._data[self.idx(key[0], key[1])]

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

    # We cannot annotate the return type of this function, because of a strange bug in
    #  sphinx-autodoc-typehints.
    #  https://github.com/tox-dev/sphinx-autodoc-typehints/issues/532
    def _extract_dense(self, idx):  # type: ignore[no-untyped-def]
        """Convert to dense 3D tensor, but only for the slice of ``idx``."""
        n_MO = len(idx)
        g = np.zeros((n_MO, n_MO, self.naux), dtype=np.float64)
        # TODO: ensure that we can actually access these indices
        for i, p in enumerate(idx):
            for j, q in enumerate(idx[: i + 1]):
                g[i, j] = self[p, q]  # type: ignore[index]
                g[j, i] = g[i, j]  # type: ignore[index]
        return g

    @staticmethod
    def idx(a: OrbitalIdx, b: OrbitalIdx) -> int:
        """Return compound index"""
        return ravel_symmetric(a, b)  # type: ignore[return-value]


@njit(nogil=True)
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
        ("MO_reachable_by_AO", ListType(Type_SortedIntSet)),
        ("AO_reachable_by_MO", ListType(Type_SortedIntSet)),
    ]
)
class MutableSemiSparse3DTensor:
    r"""Specialised datastructure for semi-sparse 3-indexed tensors.

    For a tensor, :math:`T_{ijk}`, to be stored in this datastructure we assume

    - sparsity along the :math:`i, j` indices, i.e. :math:`T_{ijk} = 0`
      for many :math:`i, j`
    - dense storage along the :math:`k` index

    It can be used for example to store the partially contracted
    3-center, 2-electron integrals
    :math:`(\mu i | P)`, with AO :math:`\mu`, localised MO :math:`i`,
    and auxiliary basis indices :math:`P`.
    Semi-sparsely, because it is assumed that there are many
    exchange pairs :math:`\mu, i` which are zero, while the integral along
    the auxiliary basis :math:`P` is stored densely as numpy array.

    2-fold permutational symmetry for the :math:`\mu, i` pairs is not assumed.

    Note that this class is mutable which makes it more flexible in practice, but also
    less performant for certain operations. If possible, it is recommended to use
    :class:`SemiSparse3DTensor`.
    """

    def __init__(
        self,
        shape: tuple[int, int, int],
    ) -> None:
        self.shape = shape
        self.naux = shape[-1]
        self._data = Dict.empty(_UniTuple_int64_2, float64[:])
        self.MO_reachable_by_AO = _get_list_of_sortedset(self.shape[0])
        self.AO_reachable_by_MO = _get_list_of_sortedset(self.shape[1])

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
        self.MO_reachable_by_AO[key[0]].add(key[1])
        self.AO_reachable_by_MO[key[1]].add(key[0])

    # We cannot annotate the return type of this function, because of a strange bug in
    #  sphinx-autodoc-typehints.
    #  https://github.com/tox-dev/sphinx-autodoc-typehints/issues/532
    def to_dense(self):  # type: ignore[no-untyped-def]
        result = np.zeros((self.shape), dtype="f8")
        for mu in range(self.shape[0]):
            for i in self.MO_reachable_by_AO[mu].items:
                result[mu, i, :] = self[mu, i]  # type: ignore[index]
        return result


@jitclass(
    [
        ("_keys", int64[::1]),
        ("dense_data", float64[:, ::1]),
        ("shape", UniTuple(int64, 3)),
        ("AO_reachable_by_MO_with_offsets", ListType(ListType(UniTuple(int64, 2)))),
        ("AO_reachable_by_MO", ListType(int64[::1])),
        ("_data", DictType(int64, float64[::1])),
    ]
)
class SemiSparse3DTensor:
    r"""Specialised datastructure for immutable and semi-sparse 3-indexed tensors.

    For a tensor, :math:`T_{ijk}`, to be stored in this datastructure we assume

    - sparsity along the :math:`i, j` indices, i.e. :math:`T_{ijk} = 0`
      for many :math:`i, j`
    - dense storage along the :math:`k` index

    It can be used for example to store the partially contracted
    3-center, 2-electron integrals
    :math:`(\mu i | P)`, with AO :math:`\mu`, localised MO :math:`i`,
    and auxiliary basis indices :math:`P`.
    Semi-sparsely, because it is assumed that there are many
    exchange pairs :math:`\mu, i` which are zero, while the integral along
    the auxiliary basis :math:`P` is stored densely as numpy array.

    2-fold permutational symmetry for the :math:`\mu, i` pairs is not assumed.

    Note that this class is immutable which enables to store the non-zero data
    in a dense manner, which has some performance benefits.
    """

    #: We know there are no collisions and roll our own "dictionary".
    #: These are the ascendingly sorted lookup keys to access the non-zero data.
    #: For a given :python:`offset, mu, nu` triple, we have.
    #:
    #: .. code-block:: python
    #:
    #:    self._keys[offset] == self._idx(mu, nu)
    #:    self.dense_data[offset] == self[mu, nu]
    _keys: Vector[np.int64]
    dense_data: Matrix[np.float64]
    #: number of AOs, number of MOs, number of auxiliary functions.
    shape: tuple[int, int, int]
    #: number of auxiliary functions
    naux: int
    #: For a given MO index :python:`i` the :python:`self.AO_reachable_by_MO[i]`
    #: returns all :python:`mu` that are assumed unrelevant
    #: for :math:`(\mu i | r s)` after screening.
    #: Note that :math:`(p i | P )` might still be non-zero.
    AO_reachable_by_MO: list[Vector[AOIdx]]
    #: The following datastructures also return the offset to index
    #: the :python:`dense_data` directly and enables very fast
    #: loops without having to compute the offset.
    #:
    #: .. code-block:: python
    #:
    #:    for offset, mu in self.AO_reachable_by_MO[i]:
    #:        self.dense_data[offset] == self[mu, i]  # True
    AO_reachable_by_MO_with_offsets: list[list[tuple[int, AOIdx]]]

    _data: Mapping[int, Vector[np.float64]]

    def __init__(
        self,
        unique_dense_data: Matrix[np.float64],
        keys: Vector[np.int64],
        shape: tuple[Integral, Integral, Integral],
        AO_reachable_by_MO_with_offsets: list[list[tuple[int, AOIdx]]],
        AO_reachable_by_MO: list[Vector[AOIdx]],
    ) -> None:
        self.shape = shape  # type: ignore[assignment]
        self.naux = shape[-1]  # type: ignore[assignment]
        self.AO_reachable_by_MO_with_offsets = AO_reachable_by_MO_with_offsets  # type: ignore[assignment]
        self.AO_reachable_by_MO = AO_reachable_by_MO  # type: ignore[assignment]

        self.dense_data = unique_dense_data

        self._keys = keys

        self._data = Dict.empty(int64, float64[::1])
        for i, k in enumerate(self._keys):
            self._data[k] = self.dense_data[i, :]  # type: ignore[index]

    def __getitem__(self, key: tuple[OrbitalIdx, OrbitalIdx]) -> Vector[np.float64]:
        return self._data[self._idx(key[0], key[1])]

    # We cannot annotate the return type of this function, because of a strange bug in
    #  sphinx-autodoc-typehints.
    #  https://github.com/tox-dev/sphinx-autodoc-typehints/issues/532
    def to_dense(self):  # type: ignore[no-untyped-def]
        """Convert to dense 3D tensor"""
        g = np.zeros(self.shape)
        for i in range(self.shape[1]):
            for mu in self.AO_reachable_by_MO[i]:
                g[mu, i] = self[mu, i]  # type: ignore[index]
        return g

    def _idx(self, a: OrbitalIdx, b: OrbitalIdx) -> int:
        """Return compound index"""
        return ravel_Fortran(a, b, n_rows=self.shape[0])  # type: ignore[return-value]


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
    return {key: inverted_D[key] for key in sorted(inverted_D.keys())}  # type: ignore[type-var]


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
            j_atom: orb_per_atom.get(j_atom, set())
            for j_atom in sorted(connected)  # type: ignore[type-var]
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


def _modify_overlap(
    AO_per_atom: Mapping[AOIdx, Set[AtomIdx]], S: Matrix[np.float64]
) -> Matrix[np.float64]:
    S_mod = S.copy()
    for AOs_on_one_atom in AO_per_atom.values():
        idx = list(AOs_on_one_atom)
        S_mod[np.ix_(idx, idx)] = 1
    return S_mod


def _get_AO_per_MO(
    TA: Matrix[np.float64],
    S_abs: Matrix[np.float64],
    epsilon: float,
) -> dict[MOIdx, Sequence[AOIdx]]:
    n_MO = TA.shape[-1]
    X = np.abs(S_abs @ TA)
    return {
        i_MO: cast(Sequence[AOIdx], (X[:, i_MO] >= epsilon).nonzero()[0])
        for i_MO in cast(Sequence[MOIdx], range(n_MO))
    }


@njit(nogil=True)
def _jit_get_AO_per_MO(
    TA: Matrix[np.float64],
    S_abs: Matrix[np.float64],
    epsilon: float,
) -> List[Vector[np.int64]]:
    n_MO = TA.shape[-1]
    X = np.abs(S_abs @ TA)
    return List([(X[:, i_MO] >= epsilon).nonzero()[0] for i_MO in range(n_MO)])


def _get_AO_per_AO(
    S_abs: Matrix[np.float64],
    epsilon: float,
) -> dict[AOIdx, Sequence[AOIdx]]:
    n_AO = len(S_abs)
    return {
        i_AO: cast(Sequence[AOIdx], (S_abs[:, i_AO] >= epsilon).nonzero()[0])
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


def get_screened(
    mol: Mole, screening_radius: Real | Callable[[Real], Real] | Mapping[str, Real]
) -> dict[AtomIdx, set[AtomIdx]]:
    m = Cartesian.from_pyscf(mol)
    return m.get_bonds(modify_element_data=screening_radius, self_bonding_allowed=True)


def get_reachable(
    atoms_per_start_orb: Mapping[_T_start_orb, Set[AtomIdx]],
    atoms_per_target_orb: Mapping[_T_target_orb, Set[AtomIdx]],
    screened_connection: Mapping[AtomIdx, Set[AtomIdx]],
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
    screening_radius :
        The screening cutoff is given by the overlap of van der Waals radii.
        By default, all radii are set to 5 Å, i.e. the screening distance is 10 Å.
        Alternatively, a callable or a dictionary can be passed.
        The callable is called with the tabulated van der Waals radius
        of the atom as argument and can be used to scale it up.
        The dictionary can be used to define different van der Waals radii
        for different elements. Compare to the :python:`modify_element_data`
        argument of :meth:`~chemcoord.Cartesian.get_bonds`.
    """
    return _flatten(
        get_orbs_reachable_by_orb(
            atoms_per_start_orb,
            get_orbs_reachable_by_atom(
                get_orbs_per_atom(atoms_per_target_orb), screened_connection
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
    exch_reachable: Mapping[_T_start_orb, Collection[_T_target_orb]],
) -> List[Vector[_T_target_orb]]:
    """Convert the reachable orbitals to a list of numpy arrays.

    This contains the same information but is a far more efficient layout for numba.
    Ensures that the start orbs are contiguos and sorted and the target orbs are sorted
    (but not necessarily contiguos).
    """
    sorted_exch_reachable = {
        k: exch_reachable[k]
        for k in sorted(exch_reachable.keys())  # type: ignore[type-var]
    }
    assert list(sorted_exch_reachable.keys()) == list(range(len(sorted_exch_reachable)))
    return List(
        [
            np.array(sorted(orbitals), dtype=np.int64)  # type: ignore[type-var]
            for orbitals in sorted_exch_reachable.values()
        ]
    )


def account_for_symmetry(
    reachable: Mapping[_T_start, Collection[_T_target]],
) -> dict[_T_start, list[_T_target]]:
    """Account for permutational symmetry and remove all q that are larger than p.

    Parameters
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


@njit(nogil=True)
def _jit_account_for_symmetry(
    reachable: list[Vector[_T_orb_idx]],
) -> list[Vector[_T_orb_idx]]:
    """Account for permutational symmetry and remove all q that are larger than p.

    This is a jitted version of :func:`account_for_symmetry`.

    Parameters
    ----------
    reachable :
    """
    return List(
        [np.array([q for q in qs if p >= q]) for (p, qs) in enumerate(reachable)]
    )


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


def get_sparse_P_mu_nu(
    mol: Mole,
    auxmol: Mole,
    exch_reachable: Mapping[AOIdx, Sequence[AOIdx]],
) -> SemiSparseSym3DTensor:
    """Return the 3-center 2-electron integrals in a sparse format."""

    def to_shell_reachable_by_shell(
        exch_reachable: Mapping[AOIdx, Sequence[AOIdx]],
        AO_to_shell_id: Mapping[AOIdx, ShellIdx],
    ) -> Mapping[ShellIdx, list[ShellIdx]]:
        """Also accepts `exch_reachable_unique` to return
        symmetry-aware reachable shell mappings"""
        shell_reachable_by_shell: dict[ShellIdx, set[ShellIdx]] = defaultdict(set)

        for k, v in exch_reachable.items():
            shell_reachable_by_shell[AO_to_shell_id[k]] |= {
                AO_to_shell_id[orb] for orb in v
            }
        return {k: sorted(v) for k, v in shell_reachable_by_shell.items()}  # type: ignore[type-var]

    AO_timer = Timer("Time to compute sparse (mu nu | P)")
    exch_reachable_unique = account_for_symmetry(exch_reachable)

    n_unique = sum(len(v) for v in exch_reachable_unique.values())

    logger.info(
        "Semi-Sparse Memory for (mu nu | P) integrals is: "
        f"{n_unique * auxmol.nao * 8 * 2**-30} Gb"
    )
    logger.info(
        "Dense Memory for (mu nu | P) would be: "
        f"{n_symmetric(mol.nao) * auxmol.nao * 8 * 2**-30} Gb"
    )
    logger.info(f"Sparsity factor is: {(1 - n_unique / n_symmetric(mol.nao)) * 100} %")
    screened_unique_integrals = np.full(
        (n_unique, auxmol.nao), fill_value=np.nan, dtype=np.float64, order="C"
    )

    shell_id_to_AO, AO_to_shell_id = conversions_AO_shell(mol)
    shell_reachable_by_shell = to_shell_reachable_by_shell(
        exch_reachable_unique, AO_to_shell_id
    )
    keys = np.array(
        [
            ravel_symmetric(p, q)
            for (p, q) in _traverse_reachable(exch_reachable_unique)
        ],
        dtype=np.int64,
    )
    key_to_offset = {k: i for i, k in enumerate(keys)}

    assert len(keys) == n_unique

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
                    if ravel_symmetric(p, q) in key_to_offset:
                        screened_unique_integrals[
                            key_to_offset[ravel_symmetric(p, q)], :  # type: ignore[arg-type]
                        ] = integrals[i, j, ::1]

    logger.info(AO_timer.str_elapsed())

    assert not np.isnan(screened_unique_integrals).any()

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
                        orb_reachable_by_orb[i_orb][start_atom][target_atom]
                        for start_atom, target_atoms in orb_reachable_by_orb[
                            i_orb
                        ].items()
                        for target_atom in target_atoms
                    )
                )
            )
        )
        for i_orb in sorted(orb_reachable_by_orb.keys())  # type: ignore[type-var]
    }


@njit(nogil=True, parallel=True)
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
        get_complement(get_reachable(atom_per_AO, atom_per_AO, get_screened(mol, 0.0)))
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
        get_complement(get_reachable(atom_per_AO, atom_per_AO, get_screened(mol, 0.0)))
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
        auxmol = make_auxmol(mol, auxbasis)
        nao = mol.nao
        naux = auxmol.nao

        sparse_ints_3c2e, sparse_df_coef = get_sparse_D_ints_and_coeffs(
            mol, auxmol, screening_radius=0.01
        )

        ints_3c2e = df.incore.aux_e2(mol, auxmol, intor="int3c2e")
        ints_2c2e = auxmol.intor("int2c2e")
        df_coef = solve(ints_2c2e, ints_3c2e.reshape(nao * nao, naux).T, assume_a="pos")
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


@njit(nogil=True)
def contract_with_TA_1st(
    TA: Matrix[np.float64],
    int_mu_nu_P: SemiSparseSym3DTensor,
    AO_reachable_per_SchmidtMO: list[Vector[AOIdx]],
) -> SemiSparse3DTensor:
    return _jit_contract_with_TA_1st(
        TA,
        int_mu_nu_P,
        _jit_get_AO_MO_pair_with_offset(AO_reachable_per_SchmidtMO),
        _jit_get_AO_reachable_by_MO_with_offset(AO_reachable_per_SchmidtMO),
        AO_reachable_per_SchmidtMO,
    )


@njit(nogil=True, parallel=True)
def _jit_contract_with_TA_1st(
    TA: Matrix[np.float64],
    int_mu_nu_P: SemiSparseSym3DTensor,
    AO_MO_pair_with_offset: list[tuple[int, AOIdx, MOIdx]],
    AO_reachable_by_MO_with_offset: list[list[tuple[int, AOIdx]]],
    AO_reachable_by_MO: list[Vector[AOIdx]],
) -> SemiSparse3DTensor:
    assert TA.shape[0] == int_mu_nu_P.nao and TA.shape[1] == len(AO_reachable_by_MO)
    n_unique = len(AO_MO_pair_with_offset)

    print("Memory for (mu i | P): Gb", n_unique * int_mu_nu_P.naux * 8 * 2**-30)
    g_unique = np.zeros((n_unique, int_mu_nu_P.naux), dtype=np.float64)
    keys = np.full(n_unique, fill_value=np.nan, dtype=np.int64)

    # We cannot directly loop as in
    # ``for offset, mu, i in AO_MO_pair_with_offset```
    # if we want to parallelise,
    # hence we need the explicit counter variables.
    for outer_counter in prange(len(AO_MO_pair_with_offset)):  # type: ignore[attr-defined]
        offset, mu, i = AO_MO_pair_with_offset[outer_counter]
        keys[offset] = ravel_Fortran(mu, i, TA.shape[0])
        for nu_counter in range(len(int_mu_nu_P.exch_reachable[mu])):  # type: ignore[attr-defined]
            inner_offset, nu = int_mu_nu_P.exch_reachable_with_offsets[mu][nu_counter]
            # In an un-optimized, but readable way it would be written as:
            # g_unique[offset] += TA[nu, i] * int_mu_nu_P[mu, nu]
            # but we know ahead of time the offset in the `unique_dense_data`
            # and don't want to compute the lookup.
            g_unique[offset] += TA[nu, i] * int_mu_nu_P.unique_dense_data[inner_offset]  # type: ignore[index]

    return SemiSparse3DTensor(
        g_unique,
        keys,
        (TA.shape[0], TA.shape[1], int_mu_nu_P.naux),
        AO_reachable_by_MO_with_offset,
        AO_reachable_by_MO,
    )


@njit(nogil=True, parallel=True)
def contract_with_TA_2nd_to_sym_dense(
    TA: Matrix[np.float64], int_mu_i_P: SemiSparse3DTensor
) -> Tensor3D[np.float64]:
    r"""Contract the first dimension of ``int_mu_i_P``
    with the first dimension of ``TA``.
    We assume the result to be symmetric in the first two dimensions.

    If the result is known to be non-symmetric use
    :func:`contract_with_TA_2nd_to_dense` instead.

    Can be used to e.g. compute contractions to purely fragment,
    or purely bath integrals.

    .. math::

        (i j | P) = \sum_{\mu} T_{\mu,i} (\mu j | P) \\
        (a b | P) = \sum_{\mu} T_{\mu,a} (\mu b | P)

    Returns
    -------
    Tensor3D :
        A dense 3D tensor :math:`(i j | P)`, symmetric in the first two (MO) dimensions.
        The last dimension is along the auxiliary basis.
    """
    assert TA.shape[0] == int_mu_i_P.shape[0]
    assert TA.shape[1] == int_mu_i_P.shape[1]

    print("Memory for (i j | P): Gb", TA.shape[1] ** 2 * int_mu_i_P.naux * 8 * 2**-30)

    g = np.zeros((TA.shape[1], TA.shape[1], int_mu_i_P.naux), dtype=np.float64)

    for counter in prange(g.shape[0]):  # type: ignore[attr-defined]
        i = g.shape[0] - counter - 1
        for j in prange(min(g.shape[1], i + 1)):  # type: ignore[attr-defined]
            for offset, nu in int_mu_i_P.AO_reachable_by_MO_with_offsets[i]:
                g[i, j, :] += TA[nu, j] * int_mu_i_P.dense_data[offset]

            g[j, i, :] = g[i, j, :]
    return g


@njit(nogil=True)
def contract_with_TA_2nd_to_sym_sparse(
    TA: Matrix[np.float64],
    int_mu_i_P: SemiSparse3DTensor,
    i_reachable_by_j: List[Vector[np.int64]],
) -> SemiSparseSym3DTensor:
    r"""Contract the first dimension of ``int_mu_i_P``
    with the first dimension of ``TA``.
    We assume the result to be symmetric in the first two dimensions.

    If the result is known to be non-symmetric use
    :func:`contract_with_TA_2nd_to_dense` instead.

    Can be used to e.g. compute contractions to purely fragment,
    or purely bath integrals.

    .. math::

        (i j | P) = \sum_{\mu} T_{\mu,i} (\mu j | P) \\
        (a b | P) = \sum_{\mu} T_{\mu,a} (\mu b | P)

    Returns
    -------
    Tensor3D :
        A dense 3D tensor :math:`(i j | P)`, symmetric in the first two (MO) dimensions.
        The last dimension is along the auxiliary basis.
    """
    assert TA.shape[0] == int_mu_i_P.shape[0]
    assert TA.shape[1] == int_mu_i_P.shape[1]

    i_reachable_by_j_unique = _jit_account_for_symmetry(i_reachable_by_j)
    n_unique = sum([len(x) for x in i_reachable_by_j_unique])
    n_MO = TA.shape[1]

    unique_dense_data = np.zeros((n_unique, int_mu_i_P.naux), dtype=np.float64)

    i_j_offset = 0
    for i in prange(n_MO):  # type: ignore[attr-defined]
        for j in i_reachable_by_j_unique[i]:
            for mu_i_offset, nu in int_mu_i_P.AO_reachable_by_MO_with_offsets[i]:
                unique_dense_data[i_j_offset, :] += (
                    TA[nu, j] * int_mu_i_P.dense_data[mu_i_offset]
                )
            i_j_offset += 1

    return SemiSparseSym3DTensor(
        unique_dense_data, n_MO, int_mu_i_P.naux, i_reachable_by_j
    )


@njit(nogil=True, parallel=True)
def contract_with_TA_2nd_to_dense(
    TA: Matrix[np.float64], int_mu_i_P: SemiSparse3DTensor
) -> Tensor3D[np.float64]:
    r"""Contract the first dimension of ``int_mu_i_P``
    with the first dimension of ``TA``.
    If the result is known to be symmetric use
    :func:`contract_with_TA_2nd_to_sym_dense` or
    :func:`contract_with_TA_2nd_to_sym_sparse` instead.

    Can be used to e.g. compute contractions of mixed fragment-bath
    integrals.

    .. math::

        (i a | P) = \sum_{\mu} T_{\mu,i} (\mu a | P) \\
        (a i | P) = \sum_{\mu} T_{\mu,a} (\mu i | P)

    Returns
    -------
    Tensor3D :
        A dense 3D tensor :math:`(i a | P)`, which can have
        different lengths along the first two dimensions,
        e.g. fragment and bath orbitals.
        The last dimension is along the auxiliary basis.
    """
    assert TA.shape[0] == int_mu_i_P.shape[0]

    g = np.zeros((TA.shape[1], int_mu_i_P.shape[1], int_mu_i_P.naux), dtype=np.float64)

    for a in prange(g.shape[0]):  # type: ignore[attr-defined]
        for i in prange(g.shape[1]):  # type: ignore[attr-defined]
            for offset, mu in int_mu_i_P.AO_reachable_by_MO_with_offsets[i]:
                g[a, i, :] += TA[mu, a] * int_mu_i_P.dense_data[offset]
    return g


@njit(nogil=True)
def _jit_get_AO_MO_pair_with_offset(
    AO_reachable_by_MO: List[Vector[np.int64]],
) -> list[tuple[int, AOIdx, MOIdx]]:
    return List(
        [
            (offset, mu, i)
            for offset, (mu, i) in enumerate(
                [
                    (i_AO, i_MO)
                    for i_MO, reachable_AOs in enumerate(AO_reachable_by_MO)
                    for i_AO in sorted(reachable_AOs)  # type: ignore[type-var]
                ]
            )
        ]
    )


@njit(nogil=True)
def _jit_get_AO_reachable_by_MO_with_offset(
    AO_reachable_by_MO: List[Vector[np.int64]],
) -> List[List[int, AOIdx]]:
    # assert list(AO_reachable_by_MO.keys()) == list(range(len(AO_reachable_by_MO)))
    counter = PreIncr()
    return List(
        [List([(counter.incr(), AO) for AO in AOs]) for AOs in AO_reachable_by_MO]
    )


def _transform_sparse_DF_slow(
    mf: scf.hf.SCF,
    Fobjs: Sequence[Frags],
    auxbasis: str | None = None,
    screen_radius: Mapping[str, float] | None = None,
) -> list[Matrix[np.float64]]:
    """Only exist for reference. Can be deleted soon."""
    mol = mf.mol
    auxmol = make_auxmol(mf.mol, auxbasis=auxbasis)
    if screen_radius is None:
        screen_radius = find_screening_radius(mol, auxmol, threshold=1e-4)
    atom_per_AO = get_atom_per_AO(mol)
    exch_reachable = get_reachable(
        atom_per_AO, atom_per_AO, get_screened(mol, screen_radius)
    )
    sparse_ints_3c2e = get_sparse_P_mu_nu(mol, auxmol, exch_reachable)
    ints_mu_nu_P = sparse_ints_3c2e.to_dense()
    ints_2c2e = auxmol.intor("int2c2e")

    ints_i_j_P = []

    for fragidx, fragobj in enumerate(Fobjs):
        transf = fragobj.TA.T.copy()
        int_mu_i_P = transf @ ints_mu_nu_P
        ints_i_j_P.append(transf @ np.moveaxis(int_mu_i_P, 1, 0))

    Ds_i_j_P = [
        solve(ints_2c2e, int_i_j_P.T, assume_a="pos").T for int_i_j_P in ints_i_j_P
    ]

    return [
        einsum("ijP,klP->ijkl", ints, df_coef)
        for ints, df_coef in zip(ints_i_j_P, Ds_i_j_P)
    ]


def transform_sparse_DF_integral_nb(
    mf: scf.hf.SCF,
    Fobjs: Sequence[Frags],
    auxbasis: str | None,
    file_eri_handler: h5py.File,
    AO_coeff_epsilon: float,
    MO_coeff_epsilon: float,
    n_threads: int,
) -> None:
    mol = mf.mol
    auxmol = make_auxmol(mf.mol, auxbasis=auxbasis)

    S_abs = approx_S_abs(mol)
    exch_reachable = _get_AO_per_AO(S_abs, AO_coeff_epsilon)

    sparse_P_mu_nu = get_sparse_P_mu_nu(mol, auxmol, exch_reachable)
    PQ = auxmol.intor("int2c2e")
    low_triang_PQ = cholesky(PQ, lower=True)

    def f(fragobj: Frags) -> None:
        eri = restore(
            "4",
            _transform_fragment_integral(
                sparse_P_mu_nu,
                fragobj.TA,
                S_abs,
                low_triang_PQ,
                MO_coeff_epsilon,
            ),
            fragobj.TA.shape[1],
        )
        file_eri_handler.create_dataset(fragobj.dname, data=eri)

    if n_threads > 1:
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(f, fragobj) for fragobj in Fobjs]
            # You must call future.result() if you want to catch exceptions
            # raised during execution. Otherwise, errors may silently fail.
            for future in futures:
                future.result()
    else:
        for fragobj in Fobjs:
            f(fragobj)


def transform_sparse_DF_integral_cpp(
    mf: scf.hf.SCF,
    Fobjs: Sequence[Frags],
    auxbasis: str | None,
    file_eri_handler: h5py.File,
    AO_coeff_epsilon: float,
    MO_coeff_epsilon: float,
    n_threads: int,
) -> None:
    cpp_transforms.set_log_level(logging.getLogger().getEffectiveLevel())
    mol = mf.mol
    auxmol = make_auxmol(mf.mol, auxbasis=auxbasis)

    S_abs = approx_S_abs(mol)
    exch_reachable = _get_AO_per_AO(S_abs, AO_coeff_epsilon)

    py_P_mu_nu = get_sparse_P_mu_nu(mol, auxmol, exch_reachable)
    mu_nu_P = cpp_transforms.SemiSparseSym3DTensor(
        np.asfortranarray(py_P_mu_nu.unique_dense_data.T),
        tuple(reversed(py_P_mu_nu.shape)),  # type: ignore[arg-type]
        py_P_mu_nu.exch_reachable,  # type: ignore[arg-type]
    )
    PQ = auxmol.intor("int2c2e")
    low_triang_PQ = cholesky(PQ, lower=True)

    def f(fragobj: Frags) -> None:
        eri = restore(
            "4",
            cpp_transforms.transform_integral(
                mu_nu_P,
                fragobj.TA,
                S_abs,
                low_triang_PQ,
                MO_coeff_epsilon,
            ),
            fragobj.TA.shape[1],
        )
        file_eri_handler.create_dataset(fragobj.dname, data=eri)

    if n_threads > 1:
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(f, fragobj) for fragobj in Fobjs]
            # You must call future.result() if you want to catch exceptions
            # raised during execution. Otherwise, errors may silently fail.
            for future in futures:
                future.result()
    else:
        for fragobj in Fobjs:
            f(fragobj)


def transform_sparse_DF_integral_cpp_gpu(
    mf: scf.hf.SCF,
    Fobjs: Sequence[Frags],
    auxbasis: str | None,
    file_eri_handler: h5py.File,
    AO_coeff_epsilon: float,
    MO_coeff_epsilon: float,
    n_threads: int,
) -> None:
    cpp_transforms.set_log_level(logging.getLogger().getEffectiveLevel())
    mol = mf.mol
    auxmol = make_auxmol(mf.mol, auxbasis=auxbasis)

    S_abs = approx_S_abs(mol)
    exch_reachable = _get_AO_per_AO(S_abs, AO_coeff_epsilon)

    py_P_mu_nu = get_sparse_P_mu_nu(mol, auxmol, exch_reachable)
    mu_nu_P = cpp_transforms.SemiSparseSym3DTensor(
        np.asfortranarray(py_P_mu_nu.unique_dense_data.T),
        tuple(reversed(py_P_mu_nu.shape)),  # type: ignore[arg-type]
        py_P_mu_nu.exch_reachable,  # type: ignore[arg-type]
    )
    PQ = auxmol.intor("int2c2e")
    low_triang_PQ = cpp_transforms.GPU_MatrixHandle(cholesky(PQ, lower=True))

    def f(fragobj: Frags) -> None:
        eri = restore(
            "4",
            cpp_transforms.transform_integral_cuda(
                mu_nu_P,
                fragobj.TA,
                S_abs,
                low_triang_PQ,
                MO_coeff_epsilon,
            ),
            fragobj.TA.shape[1],
        )
        file_eri_handler.create_dataset(fragobj.dname, data=eri)

    if n_threads > 1:
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(f, fragobj) for fragobj in Fobjs]
            # You must call future.result() if you want to catch exceptions
            # raised during execution. Otherwise, errors may silently fail.
            for future in futures:
                future.result()
    else:
        for fragobj in Fobjs:
            f(fragobj)


@njit(nogil=True)
def _get_P_ij(
    sparse_ints_3c2e: SemiSparseSym3DTensor,
    TA: Matrix[np.float64],
    S_abs: Matrix[np.float64],
    MO_coeff_epsilon: float,
) -> Tensor3D[np.float64]:
    AO_reachable_per_SchmidtMO = _jit_get_AO_per_MO(TA, S_abs, MO_coeff_epsilon)
    return contract_with_TA_2nd_to_sym_dense(
        TA, contract_with_TA_1st(TA, sparse_ints_3c2e, AO_reachable_per_SchmidtMO)
    )


def _transform_fragment_integral(
    sparse_ints_3c2e: SemiSparseSym3DTensor,
    TA: Matrix[np.float64],
    S_abs: Matrix[np.float64],
    low_triang_PQ: Matrix[np.float64],
    MO_coeff_epsilon: float,
) -> Matrix[np.float64]:
    return _eval_via_cholesky(
        _get_P_ij(sparse_ints_3c2e, TA, S_abs, MO_coeff_epsilon), low_triang_PQ
    )


@njit(nogil=True, parallel=True)
def _account_for_symmetry(pqP: Tensor3D[np.float64]) -> Matrix[np.float64]:
    """(n_orb, n_orb | n_aux) -> (n_aux | sym n_orb pairs)"""
    n_orb, naux = len(pqP), pqP.shape[-1]
    sym_bb = np.full(
        (naux, ravel_symmetric(n_orb - 1, n_orb - 1) + 1),
        fill_value=np.nan,
        dtype=np.float64,
    )

    for i in prange(n_orb):  # type: ignore[attr-defined]
        for j in range(i + 1):  # type: ignore[attr-defined]
            sym_bb[:, ravel_symmetric(i, j)] = pqP[i, j, :]
    return sym_bb


def _eval_via_cholesky(
    pqP: Tensor3D[np.float64], low_triang_PQ: Matrix[np.float64]
) -> Matrix[np.float64]:
    symmetrised = _account_for_symmetry(pqP)
    bb = solve_triangular(
        low_triang_PQ,
        symmetrised,
        lower=True,
        overwrite_b=False,
        check_finite=False,
    )
    result = bb.T @ bb
    return result


try:
    import cupy as cp  # type: ignore
    from cupyx.scipy.linalg import (
        solve_triangular as cupy_solve_triangular,  # type: ignore
    )

    def _transform_fragment_integral_on_gpu(
        sparse_ints_3c2e: SemiSparseSym3DTensor,
        TA: Matrix[np.float64],
        S_abs: Matrix[np.float64],
        low_triang_PQ_on_gpu: cp.ndarray[np.float64],
        MO_coeff_epsilon: float,
    ) -> Matrix[np.float64]:
        return _eval_via_cholesky_gpu(
            _get_P_ij(sparse_ints_3c2e, TA, S_abs, MO_coeff_epsilon),
            low_triang_PQ_on_gpu,
        )

    def transform_sparse_DF_integral_nb_gpu(
        mf: scf.hf.SCF,
        Fobjs: Sequence[Frags],
        auxbasis: str | None,
        file_eri_handler: h5py.File,
        AO_coeff_epsilon: float,
        MO_coeff_epsilon: float,
        n_threads: int,
    ) -> None:
        mol = mf.mol
        auxmol = make_auxmol(mf.mol, auxbasis=auxbasis)

        S_abs = approx_S_abs(mol)
        exch_reachable = _get_AO_per_AO(S_abs, AO_coeff_epsilon)

        sparse_P_mu_nu = get_sparse_P_mu_nu(mol, auxmol, exch_reachable)
        PQ = auxmol.intor("int2c2e")
        low_triang_PQ_on_gpu = cp.asarray(cholesky(PQ, lower=True))

        def f(fragobj: Frags) -> None:
            eri = restore(
                "4",
                _transform_fragment_integral_on_gpu(
                    sparse_P_mu_nu,
                    fragobj.TA,
                    S_abs,
                    low_triang_PQ_on_gpu,
                    MO_coeff_epsilon,
                ),
                fragobj.TA.shape[1],
            )
            file_eri_handler.create_dataset(fragobj.dname, data=eri)

        if n_threads > 1:
            with ThreadPoolExecutor(max_workers=n_threads) as executor:
                futures = [executor.submit(f, fragobj) for fragobj in Fobjs]
                # You must call future.result() if you want to catch exceptions
                # raised during execution. Otherwise, errors may silently fail.
                for future in futures:
                    future.result()
        else:
            for fragobj in Fobjs:
                f(fragobj)

    def _eval_via_cholesky_gpu(
        pqP: Tensor3D[np.float64],
        low_triang_PQ_on_gpu: cp.ndarray[np.float64],
    ) -> np.ndarray:
        symmetrised = _account_for_symmetry(pqP)  # should return np.ndarray
        symmetrised_on_gpu = cp.asarray(symmetrised, dtype=cp.float64)

        # Double-check types if still unsure
        assert isinstance(symmetrised_on_gpu, cp.ndarray)
        assert isinstance(low_triang_PQ_on_gpu, cp.ndarray)

        bb = cupy_solve_triangular(
            low_triang_PQ_on_gpu,
            symmetrised_on_gpu,
            lower=True,
            overwrite_b=False,
            check_finite=False,
        )

        result = bb.T @ bb
        return cp.asnumpy(result)
except ImportError:
    pass


_T_ = ListType(_UniTuple_int64_2)


@njit(fastmath=True, nogil=True)
def _primitive_overlap(
    li: int,
    lj: int,
    ai: float,
    aj: float,
    ci: float,
    cj: float,
    Ra: Vector[np.floating],
    Rb: Vector[np.floating],
    roots: Vector[np.floating],
    weights: Vector[np.floating],
) -> Matrix[np.float64]:
    """Evaluate the absolute overlap for a given
    pair of **uncontracted cartesian** basis functions

    .. warning::

        The result is undefined if the basis functions are not uncontracted cartesians.
        Use the triangle inequality to convert to different bases as done
        and described in :func:`~quemb.molbe.eri_sparse_DF.approx_S_abs`.

    Parameters
    ----------
    li, lj :
        Angular momenta of the two primitives.
    ai, aj :
        Gaussian exponents.
    ci, cj :
        Normalization prefactors.
    Ra, Rb :
        Centers of the two primitives.
    roots, weights :
        Gauß-Hermite quadrature roots and weights.
    """
    norm_fac = ci * cj
    # Unconventional normalization for Cartesian functions in PySCF
    if li <= 1:
        norm_fac *= ((2 * li + 1) / (4 * np.pi)) ** 0.5
    if lj <= 1:
        norm_fac *= ((2 * lj + 1) / (4 * np.pi)) ** 0.5

    aij = ai + aj
    Rab = Ra - Rb
    Rp = (ai * Ra + aj * Rb) / aij
    theta_ij = ai * aj / aij
    scale = 1.0 / np.sqrt(aij)
    norm_fac *= scale**3 * np.exp(-theta_ij * (Rab @ Rab))

    nroots = len(weights)
    x = roots * scale + Rp[:, None]
    xa = x - Ra[:, None]
    xb = x - Rb[:, None]

    mu = np.empty((li + 1, 3, nroots))
    nu = np.empty((lj + 1, 3, nroots))
    mu[0, :, :] = 1.0
    nu[0, :, :] = 1.0

    for d in range(3):
        for p in range(1, li + 1):
            mu[p, d, :] = mu[p - 1, d, :] * xa[d, :]
        for p in range(1, lj + 1):
            nu[p, d, :] = nu[p - 1, d, :] * xb[d, :]

    nfi = (li + 1) * (li + 2) // 2
    nfj = (lj + 1) * (lj + 2) // 2
    s = np.empty((nfi, nfj))

    i = 0
    for ix in range(li, -1, -1):
        for iy in range(li - ix, -1, -1):
            iz = li - ix - iy
            j = 0
            for jx in range(lj, -1, -1):
                for jy in range(lj - jx, -1, -1):
                    jz = lj - jx - jy

                    Ix = 0.0
                    Iy = 0.0
                    Iz = 0.0
                    for n in range(nroots):
                        w = weights[n]
                        Ix += w * abs(mu[ix, 0, n] * nu[jx, 0, n])
                        Iy += w * abs(mu[iy, 1, n] * nu[jy, 1, n])
                        Iz += w * abs(mu[iz, 2, n] * nu[jz, 2, n])

                    s[i, j] = Ix * Iy * Iz * norm_fac
                    j += 1
            i += 1
    return s


@njit(nogil=True, parallel=True)
def _primitive_overlap_matrix(
    ls: Vector[np.integer],
    exps: Vector[np.floating],
    norm_coef: Vector[np.floating],
    bas_coords: Matrix[np.floating],
    roots: Vector[np.floating],
    weights: Vector[np.floating],
) -> Matrix[np.float64]:
    """Compute the absolute overlap matrix for uncontracted cartesians.

    Combines the results of :func:`~quemb.molbe.eri_sparse_DF._primitive_overlap`.

    .. warning::

        The result is undefined if the basis functions are not uncontracted cartesians.
        Use the triangle inequality to convert to different bases as done
        and described in :func:`~quemb.molbe.eri_sparse_DF.approx_S_abs`.
    """
    nbas = len(ls)
    dims = [(l + 1) * (l + 2) // 2 for l in ls]
    nao = sum(dims)
    smat = np.zeros((nao, nao))

    npairs = gauss_sum(nbas)

    for idx in prange(npairs):  # type: ignore[attr-defined]
        i, j = unravel_symmetric(idx)

        i0 = sum(dims[:i])
        j0 = sum(dims[:j])
        ni = dims[i]
        nj = dims[j]

        s = _primitive_overlap(
            ls[i],
            ls[j],
            exps[i],
            exps[j],
            norm_coef[i],
            norm_coef[j],
            bas_coords[i],
            bas_coords[j],
            roots,
            weights,
        )
        smat[i0 : i0 + ni, j0 : j0 + nj] = s
        if i != j:
            smat[j0 : j0 + nj, i0 : i0 + ni] = s.T

    return smat


def _cart_mol_abs_ovlp_matrix(
    mol: Mole, nroots: int = 500
) -> tuple[Matrix[np.float64], Matrix[np.float64]]:
    r"""Compute the absolute overlap

    This is given by:

    .. math::

        S^{\mathrm{abs}}_{ij} = \int | \phi_i(r) | |\phi_j(r) | \, \mathrm{d} r

    and can be used for screening.
    Taken from `pyscf examples <https://github.com/pyscf/pyscf/blob/master/examples/1-advanced/40-mole_api_and_numba_jit.py>`_.

    .. note::

        This requires cartesian AOs, instead of spherical harmonics.
        Use :python:`cart=True` when constructing your :python:`pyscf.gto.Mole` object.

    Parameters
    ----------
    mol :
    nroots :
        Number of roots for the Gauß-Hermite quadrature.
    """
    if not mol.cart:
        raise ValueError(
            "Cartesian basis functions are required. "
            "Please construct the ``Mole`` object with ``cart=True``."
        )
    # Integrals are computed using primitive GTOs. ctr_mat transforms the
    # primitive GTOs to the contracted GTOs.
    pmol, ctr_mat = mol.decontract_basis(aggregate=True)
    # Angular momentum for each shell
    ls = cast(
        Vector[np.int64], np.array([pmol.bas_angular(i) for i in range(pmol.nbas)])
    )
    # need to access only one exponent for primitive gaussians
    exps = cast(
        Vector[np.float64], np.array([pmol.bas_exp(i)[0] for i in range(pmol.nbas)])
    )
    # Normalization coefficients
    norm_coef = cast(Vector[np.float64], gto.gto_norm(ls, exps))
    # Position for each shell
    bas_coords = cast(
        Matrix[np.float64], np.array([pmol.bas_coord(i) for i in range(pmol.nbas)])
    )
    r, w = cast(tuple[Vector[np.float64], Vector[np.float64]], roots_hermite(nroots))
    s = _primitive_overlap_matrix(ls, exps, norm_coef, bas_coords, r, w)
    assert (s >= 0).all()
    return s, ctr_mat


def _get_cart_mol(mol: Mole) -> Mole:
    return gto.M(
        atom=mol.atom, basis=mol.basis, charge=mol.charge, spin=mol.spin, cart=True
    )


@timer.timeit
def approx_S_abs(mol: Mole, nroots: int = 500) -> Matrix[np.float64]:
    r"""Compute the approximated absolute overlap matrix.

    The calculation is only exact for uncontracted, cartesian basis functions.
    Since the absolute value is not a linear function, the
    value after contraction and/or transformation to spherical-harmonics is approximated
    via the RHS of the triangle inequality:

    .. math::

        \int |\phi_i(\mathbf{r})| \, |\phi_j(\mathbf{r})| \, d\mathbf{r}
        \leq
        \sum_{\alpha,\beta} |c_{\alpha i}| \, |c_{\beta j}| \int |\chi_\alpha(\mathbf{r})| \, |\chi_\beta(\mathbf{r})| \, d\mathbf{r}

    Parameters
    ----------
    mol :
    nroots :
        Number of roots for the Gauß-Hermite quadrature.
    """  # noqa: E501
    if mol.cart:
        s, ctr_mat = _cart_mol_abs_ovlp_matrix(mol, nroots)
        return abs(ctr_mat.T) @ s @ abs(ctr_mat)
    else:
        cart_mol = _get_cart_mol(mol)
        s, ctr_mat = _cart_mol_abs_ovlp_matrix(cart_mol, nroots)
        # get the transformation matrix from cartesian basis functions to spherical.
        cart2spher = cart_mol.cart2sph_coeff(normalized="sp")
        return abs(cart2spher.T @ ctr_mat.T) @ s @ abs(ctr_mat @ cart2spher)


@timer.timeit
def grid_S_abs(mol: Mole, grid_level: int = 2) -> Matrix[np.float64]:
    r"""
    Calculates the overlap matrix :math:`S_ij = \int |phi_i(r)| |phi_j(r)| dr`
    using numerical integration on a DFT grid.

    Parameters
    -----------
        mol :
    """
    grids = dft.gen_grid.Grids(mol)
    grids.level = grid_level
    grids.build()
    AO_abs_val = np.abs(dft.numint.eval_ao(mol, grids.coords, deriv=0))
    result = (AO_abs_val * grids.weights[:, np.newaxis]).T @ AO_abs_val
    assert np.allclose(result, result.T)
    return result


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
