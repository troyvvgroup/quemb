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
from itertools import chain, takewhile
from typing import Final, TypeVar, cast

import h5py
import numpy as np
from chemcoord import Cartesian
from numba import prange  # type: ignore[attr-defined]
from numba.typed import List
from pyscf import dft, gto, scf
from pyscf.ao2mo.addons import restore
from pyscf.df.addons import make_auxmol
from pyscf.gto import Mole
from pyscf.gto.moleintor import getints, make_cintopt, make_loc
from pyscf.pbc import dft as pbc_dft
from pyscf.pbc.gto import Cell
from scipy.linalg import cholesky
from scipy.special import roots_hermite
from typing_extensions import assert_never

from quemb.molbe._cpp.eri_sparse_DF import (
    SemiSparseSym3DTensor,
    SemiSparseSym3DWindow,
    apply_inv_cholesky_inplace,
    set_log_level,
    transform_integral_precontracted,
)
from quemb.molbe.chemfrag import (
    _get_AOidx_per_atom,
)
from quemb.molbe.pfrag import Frags
from quemb.shared.helper import (
    Timer,
    gauss_sum,
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
    Vector,
)

_T_chemsystem = TypeVar("_T_chemsystem", Mole, Cell)

_T_orb_idx = TypeVar("_T_orb_idx", bound=OrbitalIdx)
_T_start_orb = TypeVar("_T_start_orb", bound=OrbitalIdx)
_T_target_orb = TypeVar("_T_target_orb", bound=OrbitalIdx)

_T_start = TypeVar("_T_start", bound=np.integer)
_T_target = TypeVar("_T_target", bound=np.integer)
_T = TypeVar("_T", int, np.integer)

logger = logging.getLogger(__name__)


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
        i_MO: cast(Sequence[AOIdx], (TA[:, i_MO] ** 2 > epsilon).nonzero()[0])
        for i_MO in cast(Sequence[MOIdx], range(n_MO))
    }
    return {
        i_MO: set().union(*(atom_per_AO[AO] for AO in AO_indices))
        for i_MO, AO_indices in large_enough.items()
    }


def _get_AO_per_MO(
    TA: Matrix[np.float64],
    S_abs: Matrix[np.float64],
    epsilon: float,
) -> dict[MOIdx, Sequence[AOIdx]]:
    n_MO = TA.shape[-1]
    # Upper bound S_abs @ |TA|; |S_abs @ TA| would allow cancellation of
    # coefficients with opposite sign.
    X = S_abs @ np.abs(TA)
    return {
        i_MO: cast(Sequence[AOIdx], (X[:, i_MO] >= epsilon).nonzero()[0])
        for i_MO in cast(Sequence[MOIdx], range(n_MO))
    }


@timer.timeit
def _get_AO_per_AO(
    S_abs: Matrix[np.floating],
    epsilon: float,
) -> dict[AOIdx, Sequence[AOIdx]]:
    n_AO = len(S_abs)
    return {
        i_AO: cast(Sequence[AOIdx], (S_abs[:, i_AO] >= epsilon).nonzero()[0])
        for i_AO in cast(Sequence[AOIdx], range(n_AO))
    }


def _get_used_AOs(
    S_abs: Matrix[np.floating],
    TA: Matrix[np.floating],
    MO_coeff_epsilon: float,
) -> Vector[np.bool_]:
    """The AOs :math:`\\mu` that the MO screening keeps for at least one MO of
    :python:`TA` (cf. :func:`_get_AO_per_MO`)."""
    return (S_abs @ np.abs(TA) >= MO_coeff_epsilon).any(axis=1)


def _get_pair_keys(
    S_abs: Matrix[np.floating],
    AO_coeff_epsilon: float,
    used: Vector[np.bool_],
) -> Vector[np.int64]:
    """Sorted keys :python:`ravel_symmetric(mu, nu)` of the AO pairs that are needed
    to transform :math:`(P | \\mu \\nu)` with the MOs that use the AOs :python:`used`.

    The first contraction :math:`(P | \\mu i) = \\sum_\\nu (P | \\mu \\nu) T_{\\nu i}`
    is evaluated for the used AOs :math:`\\mu` (cf. :func:`_get_used_AOs`)
    and all :math:`\\nu` with
    :math:`S^{\\text{abs}}_{\\mu \\nu} \\geq \\epsilon^{\\text{AO}}`.
    Hence the pairs agree with the screening of the precomputed
    :math:`(P | \\mu \\nu)` from :func:`_get_AO_per_AO`.
    Every unordered pair appears once, as :math:`\\mu \\geq \\nu`.
    """
    pairs = (S_abs >= AO_coeff_epsilon) & (used[:, None] | used[None, :])
    # row-major order of the lower triangle, i.e. increasing keys
    mu, nu = np.tril(pairs).nonzero()
    return cast(Vector[np.int64], ravel_symmetric(mu, nu))


def _order_by_overlap(used: Matrix[np.bool_]) -> list[int]:
    """Order the fragments, such that consecutive fragments overlap strongly.

    Greedy nearest neighbour on the number of shared used AOs
    (one row of :python:`used` per fragment, cf. :func:`_get_used_AOs`),
    starting from the fragment with the smallest total overlap with all others,
    e.g. an end of a chain.
    """
    n_frag = len(used)
    U = used.astype(np.float32)
    overlap = U @ U.T
    np.fill_diagonal(overlap, 0.0)
    order = [int(overlap.sum(axis=1).argmin())]
    visited = np.zeros(n_frag, dtype=bool)
    visited[order[0]] = True
    for _ in range(n_frag - 1):
        candidates = np.where(visited, -np.inf, overlap[order[-1]])
        order.append(int(candidates.argmax()))
        visited[order[-1]] = True
    return order


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
    # ao_loc_nr changes behaviour based on mol.cart (which is what we want)
    offsets = mol.ao_loc_nr()
    shell_id_to_AO = {
        cast(ShellIdx, shell_id): cast(
            list[AOIdx], list(range(offsets[shell_id], offsets[shell_id + 1]))
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


def _to_shell_reachable_by_shell(
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


def _fill_P_mu_nu_columns(
    mol: Mole,
    auxmol: Mole,
    exch_reachable_unique: Mapping[AOIdx, Sequence[AOIdx]],
    key_to_offset: Mapping[int, int],
    out: Matrix[np.float64],
) -> None:
    """Write :math:`(P | \\mu \\nu)` of the pairs :python:`key_to_offset`
    into the columns :python:`out[:, key_to_offset[ravel_symmetric(mu, nu)]]`.

    :python:`exch_reachable_unique` contains these pairs as :math:`\\mu \\geq \\nu`.
    The integrals are computed per shell block, pairs of a block
    that are not in :python:`key_to_offset` are skipped.
    """
    shell_id_to_AO, AO_to_shell_id = conversions_AO_shell(mol)
    shell_reachable_by_shell = _to_shell_reachable_by_shell(
        exch_reachable_unique, AO_to_shell_id
    )

    # Same as pyscf.df.incore.aux_e2, but the concatenated environment, AO offsets,
    # and the libcint optimizer are created once here instead of for every shell block.
    intor = mol._add_suffix("int3c2e")
    atm, bas, env = gto.mole.conc_env(
        mol._atm, mol._bas, mol._env, auxmol._atm, auxmol._bas, auxmol._env
    )
    ao_loc = make_loc(bas, intor)
    cintopt = make_cintopt(atm, bas, env, intor)

    for i_shell, reachable in shell_reachable_by_shell.items():
        for start_block, stop_block in get_blocks(reachable):
            integrals = np.asarray(  # type: ignore[call-overload]
                getints(
                    intor,
                    atm,
                    bas,
                    env,
                    (
                        i_shell,
                        i_shell + 1,
                        start_block,
                        stop_block + 1,
                        mol.nbas,
                        mol.nbas + auxmol.nbas,
                    ),
                    None,
                    0,
                    "s1",
                    ao_loc,
                    cintopt,
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
                    key = int(ravel_symmetric(p, q))
                    if key in key_to_offset:
                        out[:, key_to_offset[key]] = integrals[i, j, ::1]


def get_sparse_P_mu_nu(
    mol: Mole,
    auxmol: Mole,
    exch_reachable: Mapping[AOIdx, Sequence[AOIdx]],
) -> SemiSparseSym3DTensor:
    """Return the 3-center 2-electron integrals in a sparse format."""
    AO_timer = Timer("Time to compute sparse (P | mu nu)")

    result = SemiSparseSym3DTensor(
        (auxmol.nao, mol.nao, mol.nao),
        [
            exch_reachable.get(mu, cast(Sequence[AOIdx], []))  # type:ignore[misc]
            for mu in cast(Sequence[AOIdx], range(mol.nao))
        ],
    )
    exch_reachable_unique = account_for_symmetry(exch_reachable)

    n_unique = result.unique_dense_data.shape[1]

    keys = np.array(
        [
            ravel_symmetric(p, q)
            for (p, q) in _traverse_reachable(exch_reachable_unique)
        ],
        dtype=np.int64,
    )
    key_to_offset = {k: i for i, k in enumerate(keys)}

    assert len(keys) == n_unique

    _fill_P_mu_nu_columns(
        mol, auxmol, exch_reachable_unique, key_to_offset, result.mut_unique_dense_data
    )

    logger.info(AO_timer.str_elapsed())

    assert not np.isnan(result.unique_dense_data).any()

    return result


def _compute_P_mu_nu_columns(
    mol: Mole, auxmol: Mole, keys: Vector[np.int64]
) -> Matrix[np.float64]:
    """Return :math:`(P | \\mu \\nu)` for the pairs
    :python:`keys = ravel_symmetric(mu, nu)` as columns of a Fortran-ordered
    :python:`(naux, len(keys))` array."""
    # invert keys = mu (mu + 1) / 2 + nu with mu >= nu
    mu = ((np.sqrt(8 * keys.astype(np.float64) + 1) - 1) // 2).astype(np.int64)
    nu = keys - gauss_sum(mu)
    assert ((0 <= nu) & (nu <= mu)).all()
    starts = np.flatnonzero(np.r_[True, mu[1:] != mu[:-1]])
    exch_reachable_unique = {
        cast(AOIdx, int(mu[start])): cast(Sequence[AOIdx], nus.tolist())
        for start, nus in zip(starts, np.split(nu, starts[1:]))
    }
    out = np.full((auxmol.nao, len(keys)), np.nan, order="F")
    _fill_P_mu_nu_columns(
        mol,
        auxmol,
        exch_reachable_unique,
        {int(k): i for i, k in enumerate(keys)},
        out,
    )
    assert not np.isnan(out).any()
    return out


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


def _cholesky_PQ(auxmol: Mole) -> Matrix[np.float64]:
    """Lower Cholesky factor of :math:`(P | Q)`, in Fortran order.

    :math:`(P | Q)` itself is not kept; both are :python:`naux x naux` matrices,
    i.e. a few GB for large systems.
    """
    return cholesky(auxmol.intor("int2c2e"), lower=True)


def _run_rolling_window(
    mol: Mole,
    auxmol: Mole,
    Fobjs: Sequence[Frags],
    S_abs: Matrix[np.floating],
    file_eri_handler: h5py.File,
    AO_coeff_epsilon: float,
    MO_coeff_epsilon: float,
    max_chunk_bytes=2**30,
) -> None:
    """On-the-fly sparse DF with a rolling window of :math:`L^{-1} (P | \\mu \\nu)`,
    see :func:`transform_sparse_DF_integral`.

    The new pairs of a fragment are computed and inserted in chunks of at most
    :python:`max_chunk_bytes`, to keep the memory on top of the window small.
    """
    used = np.array([_get_used_AOs(S_abs, f.TA, MO_coeff_epsilon) for f in Fobjs])
    capacity = max(len(_get_pair_keys(S_abs, AO_coeff_epsilon, u)) for u in used)
    # The window keeps its own copy of L_PQ, so no reference must be kept here.
    window = SemiSparseSym3DWindow(
        (auxmol.nao, mol.nao, mol.nao), capacity, _cholesky_PQ(auxmol)
    )
    chunk_size = max(1, max_chunk_bytes // (8 * auxmol.nao))

    stored = np.empty(0, dtype=np.int64)
    for i_frag in _order_by_overlap(used):
        fragobj = Fobjs[i_frag]
        window_timer = Timer(f"Rolling window, fragment {fragobj.dname}")
        needed = _get_pair_keys(S_abs, AO_coeff_epsilon, used[i_frag])
        evicted = np.setdiff1d(stored, needed, assume_unique=True)
        new = np.setdiff1d(needed, stored, assume_unique=True)

        window.evict(evicted)
        for start in range(0, len(new), chunk_size):
            chunk = cast(Vector[np.int64], new[start : start + chunk_size])
            window.insert(chunk, _compute_P_mu_nu_columns(mol, auxmol, chunk))
        stored = needed
        window.set_active(needed)
        logger.info(
            window_timer.str_elapsed(
                f"{window_timer.message}: {len(needed)} pairs, "
                f"{len(new)} new, {len(evicted)} evicted; time for (P | mu nu) and L^-1"
            )
        )

        transformed = transform_integral_precontracted(
            window, fragobj.TA, S_abs, MO_coeff_epsilon
        )
        eri = restore("4", transformed, fragobj.TA.shape[1])
        file_eri_handler.create_dataset(fragobj.dname, data=eri)


def transform_sparse_DF_integral(
    mf: scf.hf.SCF,
    Fobjs: Sequence[Frags],
    auxbasis: str | None,
    file_eri_handler: h5py.File,
    AO_coeff_epsilon: float,
    MO_coeff_epsilon: float,
    precompute_P_mu_nu: bool,
) -> None:
    r"""Run the semi-sparse DF ERI transformation.

    The fragments are transformed one after another;
    the transformation of each fragment is parallelized with OpenMP and BLAS.

    Parameters
    ----------
    mf :
        Mean-field object providing the molecular orbitals and integrals.
    Fobjs :
        Fragment objects controlling the block structure of the transformation.
    auxbasis :
        Auxiliary basis used for density fitting. If ``None``, the default
        auxiliary basis of ``mf`` is used.
    file_eri_handler :
        Open HDF5 file handle, the :math:`(ij | kl)` of every fragment go into it.
    AO_coeff_epsilon :
        Threshold for discarding negligible AO coefficients.
    MO_coeff_epsilon :
        Threshold for discarding negligible MO coefficients.
    precompute_P_mu_nu :
        Whether to precompute :math:`(P | \mu \nu)` for all pairs,
        or compute them on the fly (reduces peak memory usage).
        In both cases :math:`L^{-1}_{PQ}` is applied to every
        :math:`(Q | \mu \nu)` instead of to :math:`(Q | ij)` of every fragment.
        On the fly, the fragments are traversed such that consecutive fragments
        overlap strongly, and a rolling window holds :math:`L^{-1}_{PQ} (Q | \mu \nu)`
        of the current fragment: per fragment, only the pairs that are not yet
        in the window are computed, and the ones that are no longer needed are evicted.
    """
    set_log_level(logging.getLogger().getEffectiveLevel())

    mol: Final[Mole] = mf.mol
    auxmol: Final[Mole] = make_auxmol(mol, auxbasis=auxbasis)

    S_abs: Final[Matrix[np.floating]] = approx_S_abs(mol)

    if not precompute_P_mu_nu:
        _run_rolling_window(
            mol,
            auxmol,
            Fobjs,
            S_abs,
            file_eri_handler,
            AO_coeff_epsilon,
            MO_coeff_epsilon,
        )
        return

    # Before (P | mu nu), such that (P | Q) and (P | mu nu) never coexist.
    L_PQ = _cholesky_PQ(auxmol)
    P_mu_nu: Final[SemiSparseSym3DTensor] = get_sparse_P_mu_nu(
        mol, auxmol, _get_AO_per_AO(S_abs, AO_coeff_epsilon)
    )
    # The triangular solve is linear in mu nu, so it commutes with the
    # contraction with TA. Do it once here, instead of for every fragment.
    apply_inv_cholesky_inplace(P_mu_nu, L_PQ)
    del L_PQ

    for fragobj in Fobjs:
        transformed = transform_integral_precontracted(
            P_mu_nu, fragobj.TA, S_abs, MO_coeff_epsilon
        )
        eri = restore("4", transformed, fragobj.TA.shape[1])
        file_eri_handler.create_dataset(fragobj.dname, data=eri)


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
def grid_S_abs(
    mol: _T_chemsystem,
    grid_level: int = 2,
    kpts: Matrix[np.floating] = np.zeros((1, 3)),
) -> Matrix[np.float64]:
    r"""
    Calculates the overlap matrix :math:`S_ij = \int |phi_i(r)| |phi_j(r)| dr`
    using numerical integration on a DFT grid.

    Parameters
    -----------
    mol :
    grid_level :
        Directly passed on to `pyscf grid generation <https://github.com/pyscf/pyscf/blob/master/examples/dft/11-grid_scheme.py>`_.
    """
    if isinstance(mol, Mole):
        grids = dft.gen_grid.Grids(mol)
    elif isinstance(mol, Cell):
        grids = pbc_dft.gen_grid.BeckeGrids(mol)
    else:
        raise TypeError(f"Input should be a PySCF Mole or Cell, got {type(mol)}")
        assert_never(mol)
    grids.level = grid_level
    grids.build()
    if isinstance(mol, Mole):
        AO_abs_val = np.abs(dft.numint.eval_ao(mol, grids.coords, deriv=0))
        result = (AO_abs_val * grids.weights[:, np.newaxis]).T @ AO_abs_val
        assert np.allclose(result, result.T)
    elif isinstance(mol, Cell):  # (nkpt, nao, nao)
        AO_abs_val = np.abs(
            pbc_dft.numint.eval_ao_kpts(mol, grids.coords, deriv=0, kpts=kpts)
        )
        result = [
            (AO_abs_val_k * grids.weights[:, np.newaxis]).T @ AO_abs_val_k
            for AO_abs_val_k in AO_abs_val
        ]
        assert np.all([np.allclose(r, r.T) for r in result])

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
