# Author(s): Hong-Zhou Ye
#            Henry Tran
# NOTICE: The following code is entirely written by Hong-Zhou Ye.
#         The code has been slightly modified.
#

import numpy as np
from numpy import allclose, diag, eye, where
from numpy.linalg import eigh, matrix_power, norm

from quemb.shared.typing import Matrix


def remove_core_mo(Clo: Matrix, Ccore: Matrix, S: Matrix, thr: float = 0.5) -> Matrix:
    """Remove core molecular orbitals from localized Clo"""
    assert allclose(Clo.T @ S @ Clo, eye(Clo.shape[1]))
    assert allclose(Ccore.T @ S @ Ccore, eye(Ccore.shape[1]))

    n, nlo = Clo.shape
    ncore = Ccore.shape[1]
    Pcore = Ccore @ Ccore.T @ S
    Clo1 = (eye(n) - Pcore) @ Clo
    pop = diag(Clo1.T @ S @ Clo1)
    idx_keep = where(pop > thr)[0]
    assert len(idx_keep) == nlo - ncore
    return symm_orth(Clo1[:, idx_keep], ovlp=S)


def dot_gen(A: Matrix, B: Matrix, ovlp: Matrix | None = None) -> Matrix:
    """Return product A.T @ B or A.T @ ovlp @ B"""
    return A.T @ B if ovlp is None else A.T @ ovlp @ B


def get_cano_orth_mat(
    A: Matrix, thr: float = 1.0e-6, ovlp: Matrix | None = None
) -> Matrix:
    """Perform canonical orthogonalization of A"""
    S = dot_gen(A, A, ovlp)
    e, u = eigh(S)
    if thr > 0:
        idx_keep = e / e[-1] > thr
    else:
        idx_keep = slice(0, e.shape[0])
    return u[:, idx_keep] * e[idx_keep] ** -0.5


def cano_orth(A: Matrix, thr: float = 1.0e-6, ovlp: Matrix | None = None) -> Matrix:
    """Canonically orthogonalize columns of A"""
    return A @ get_cano_orth_mat(A, thr, ovlp)


def get_symm_orth_mat(
    A: Matrix, thr: float = 1.0e-6, ovlp: Matrix | None = None
) -> Matrix:
    """Perform symmetric orthogonalization of A"""
    S = dot_gen(A, A, ovlp)
    e, u = eigh(S)
    if (e < thr).any():
        raise ValueError(
            "Linear dependence is detected in the column space of A: "
            "smallest eigenvalue (%.3E) is less than thr (%.3E). "
            "Please use 'cano_orth' instead." % (np.min(e), thr)
        )
    return u @ diag(e**-0.5) @ u.T


def symm_orth(A: Matrix, thr: float = 1.0e-6, ovlp: Matrix | None = None) -> Matrix:
    """Symmetrically orthogonalize columns of A"""
    return A @ get_symm_orth_mat(A, thr, ovlp)


def get_symm_mat_pow(A, p, check_symm=True, thresh=1.0e-8):
    """A ** p where A is symmetric

    Note:
        For integer p, it calls numpy.linalg.matrix_power
    """
    if abs(int(p) - p) < thresh:
        return matrix_power(A, int(p))

    if check_symm:
        assert norm(A - A.conj().T) < thresh

    e, u = eigh(A)
    Ap = u @ diag(e**p) @ u.conj().T

    return Ap


def get_aoind_by_atom(mol, atomind_by_motif=None):
    """Return a list across all atoms (motifs). Each element contains a list of
    AO indices for that atom (or motif, if atomind_by_motif True)"""
    natom = mol.natm
    aoslice_by_atom = mol.aoslice_by_atom()
    aoshift_by_atom = [0] + [aoslice_by_atom[ia][-1] for ia in range(natom)]
    # if motif info is provided, group lo by motif
    if atomind_by_motif is None:
        aoind_by_atom = [
            list(range(*aoshift_by_atom[ia : ia + 2])) for ia in range(natom)
        ]
    else:
        nmotif = len(atomind_by_motif)
        assert set([ia for im in range(nmotif) for ia in atomind_by_motif[im]]) == set(
            range(natom)
        )
        aoind_by_atom = [[] for im in range(nmotif)]
        for im in range(nmotif):
            for ia in atomind_by_motif[im]:
                aoind_by_atom[im] += list(range(*aoshift_by_atom[ia : ia + 2]))

    return aoind_by_atom


def reorder_by_atom_(Clo, aoind_by_atom, S, thr=0.5):
    """Reorder the ~LOCALIZED~ Clo orbitals by atom"""
    natom = len(aoind_by_atom)
    nlo = Clo.shape[1]
    X = get_symm_mat_pow(S, 0.5)

    Clo_soao = X @ Clo

    loind_reorder = []
    loind_by_atom = [None] * natom
    loshift = 0
    for ia in range(natom):
        ra = aoind_by_atom[ia]
        poplo_by_atom = np.sum(Clo_soao[ra] ** 2.0, axis=0)
        loind_a = where(poplo_by_atom > thr)[0].tolist()
        loind_reorder += loind_a
        nlo_a = len(loind_a)
        loind_by_atom[ia] = list(range(loshift, loshift + nlo_a))
        loshift += nlo_a
    if loind_reorder != list(range(nlo)):
        Clo_new = Clo[:, loind_reorder]
    else:
        Clo_new = Clo
    return Clo_new, loind_by_atom
