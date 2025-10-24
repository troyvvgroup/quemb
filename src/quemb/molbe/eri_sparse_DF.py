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
from numba import prange  # type: ignore[attr-defined]
from numba.typed import List
from pyscf import df, dft, gto, scf
from pyscf.ao2mo.addons import restore
from pyscf.df.addons import make_auxmol
from pyscf.gto import Mole
from pyscf.gto.moleintor import getints
from scipy.linalg import cholesky
from scipy.special import roots_hermite

import quemb.molbe._cpp.eri_sparse_DF as cpp_transforms
from quemb.molbe._cpp.eri_sparse_DF import (
    SemiSparseSym3DTensor,
    set_log_level,
    transform_integral,
)
from quemb.molbe.chemfrag import (
    _get_AOidx_per_atom,
)
from quemb.molbe.pfrag import Frags
from quemb.shared.helper import (
    Timer,
    gauss_sum,
    n_symmetric,
    njit,
    ravel_symmetric,
    timer,
    unravel_symmetric,
)
from quemb.shared.typing import (
    AOIdx,
    AtomIdx,
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
    S_abs: Matrix[np.floating],
    epsilon: float,
    TA: Matrix[np.floating] | None = None,
) -> dict[AOIdx, Sequence[AOIdx]]:
    if TA is None:
        n_AO = len(S_abs)
        sources = cast(Sequence[AOIdx], range(n_AO))
    else:
        sources = cast(
            Sequence[AOIdx], ((S_abs @ abs(TA)).max(axis=1) > epsilon).nonzero()[0]
        )
    return {
        i_AO: cast(Sequence[AOIdx], (S_abs[:, i_AO] >= epsilon).nonzero()[0])
        for i_AO in sources
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


def _traverse_reachable(
    reachable: Mapping[_T_start_orb, Collection[_T_target_orb]],
) -> Iterator[tuple[_T_start_orb, _T_target_orb]]:
    """Traverse reachable p, q pairs"""
    for p in reachable:
        for q in reachable[p]:
            yield p, q


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

    result = SemiSparseSym3DTensor(
        (auxmol.nao, mol.nao, mol.nao),
        [v for v in exch_reachable.values()],  # type: ignore[misc]
    )
    exch_reachable_unique = account_for_symmetry(exch_reachable)

    n_unique = result.unique_dense_data.shape[1]

    logger.info(
        "Semi-Sparse Memory for (mu nu | P) integrals is: "
        f"{n_unique * auxmol.nao * 8 * 2**-30} Gb"
    )
    logger.info(
        "Dense Memory for (mu nu | P) would be: "
        f"{n_symmetric(mol.nao) * auxmol.nao * 8 * 2**-30} Gb"
    )
    logger.info(f"Sparsity factor is: {(1 - n_unique / n_symmetric(mol.nao)) * 100} %")

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
                        result.mut_unique_dense_data[
                            :, key_to_offset[ravel_symmetric(p, q)]  # type: ignore[arg-type]
                        ] = integrals[i, j, ::1]

    logger.info(AO_timer.str_elapsed())

    assert not np.isnan(result.unique_dense_data).any()

    return result


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


def transform_sparse_DF_integral_cpp(
    mf: scf.hf.SCF,
    Fobjs: Sequence[Frags],
    auxbasis: str | None,
    file_eri_handler: h5py.File,
    AO_coeff_epsilon: float,
    MO_coeff_epsilon: float,
    n_threads: int,
) -> None:
    set_log_level(logging.getLogger().getEffectiveLevel())
    mol = mf.mol
    auxmol = make_auxmol(mf.mol, auxbasis=auxbasis)

    S_abs = approx_S_abs(mol)
    exch_reachable = _get_AO_per_AO(S_abs, AO_coeff_epsilon)

    P_mu_nu = get_sparse_P_mu_nu(mol, auxmol, exch_reachable)
    PQ = auxmol.intor("int2c2e")
    low_triang_PQ = cholesky(PQ, lower=True)

    def f(fragobj: Frags) -> None:
        eri = restore(
            "4",
            transform_integral(
                P_mu_nu,
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


#  def transform_sparse_int_direct_DF_integral_cpp(
#      mf: scf.hf.SCF,
#      Fobjs: Sequence[Frags],
#      auxbasis: str | None,
#      file_eri_handler: h5py.File,
#      AO_coeff_epsilon: float,
#      MO_coeff_epsilon: float,
#      n_threads: int,
#  ) -> None:
#      cpp_transforms.set_log_level(logging.getLogger().getEffectiveLevel())
#      mol = mf.mol
#      auxmol = make_auxmol(mf.mol, auxbasis=auxbasis)
#
#      S_abs = approx_S_abs(mol)
#      exch_reachable = _get_AO_per_AO(S_abs, AO_coeff_epsilon)
#
#      py_P_mu_nu = get_sparse_P_mu_nu(mol, auxmol, exch_reachable)
#      mu_nu_P = cpp_transforms.SemiSparseSym3DTensor(
#          np.asfortranarray(py_P_mu_nu.unique_dense_data.T),
#          tuple(reversed(py_P_mu_nu.shape)),  # type: ignore[arg-type]
#          py_P_mu_nu.exch_reachable,  # type: ignore[arg-type]
#      )
#
#      PQ = auxmol.intor("int2c2e")
#      low_triang_PQ = cholesky(PQ, lower=True)
#
#      def f(fragobj: Frags) -> None:
#          eri = restore(
#              "4",
#              cpp_transforms.transform_integral(
#                  mu_nu_P,
#                  fragobj.TA,
#                  S_abs,
#                  low_triang_PQ,
#                  MO_coeff_epsilon,
#              ),
#              fragobj.TA.shape[1],
#          )
#          file_eri_handler.create_dataset(fragobj.dname, data=eri)
#
#      if n_threads > 1:
#          with ThreadPoolExecutor(max_workers=n_threads) as executor:
#              futures = [executor.submit(f, fragobj) for fragobj in Fobjs]
#              # You must call future.result() if you want to catch exceptions
#              # raised during execution. Otherwise, errors may silently fail.
#              for future in futures:
#                  future.result()
#      else:
#          for fragobj in Fobjs:
#              f(fragobj)


def transform_sparse_DF_integral_cpp_gpu(
    mf: scf.hf.SCF,
    Fobjs: Sequence[Frags],
    auxbasis: str | None,
    file_eri_handler: h5py.File,
    AO_coeff_epsilon: float,
    MO_coeff_epsilon: float,
    n_threads: int,
) -> None:
    set_log_level(logging.getLogger().getEffectiveLevel())
    mol = mf.mol
    auxmol = make_auxmol(mf.mol, auxbasis=auxbasis)

    S_abs = approx_S_abs(mol)
    exch_reachable = _get_AO_per_AO(S_abs, AO_coeff_epsilon)

    P_mu_nu = get_sparse_P_mu_nu(mol, auxmol, exch_reachable)
    PQ = auxmol.intor("int2c2e")
    low_triang_PQ = cpp_transforms.GPU_MatrixHandle(cholesky(PQ, lower=True))

    def f(fragobj: Frags) -> None:
        eri = restore(
            "4",
            cpp_transforms.transform_integral_cuda(
                P_mu_nu,
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
        return _ensure_normalization(
            abs(cart2spher.T @ ctr_mat.T) @ s @ abs(ctr_mat @ cart2spher)
        )


def _ensure_normalization(S_abs: Matrix[np.floating]) -> Matrix[np.float64]:
    N: Final = np.sqrt(np.diag(S_abs))
    return S_abs / (N[:, None] * N[None, :])


@timer.timeit
def grid_S_abs(mol: Mole, grid_level: int = 2) -> Matrix[np.float64]:
    r"""
    Calculates the overlap matrix :math:`S_ij = \int |phi_i(r)| |phi_j(r)| dr`
    using numerical integration on a DFT grid.

    Parameters
    -----------
    mol :
    grid_level :
        Directly passed on to `pyscf grid generation <https://github.com/pyscf/pyscf/blob/master/examples/dft/11-grid_scheme.py>`_.
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
