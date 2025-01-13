# Author(s): Henry Tran
#            Oinam Meitei
#

import numpy as np
import scipy
from numpy import (
    allclose,
    array,
    asarray,
    complex128,
    diag,
    eye,
    where,
    zeros,
    zeros_like,
)
from numpy.linalg import eigh, inv, multi_dot, norm
from pyscf.pbc import gto as pgto

from quemb.shared.helper import unused


def dot_gen(A, B, ovlp):
    if ovlp is None:
        return A.conj().T @ B
    else:
        return A.conj().T @ ovlp @ B


def get_cano_orth_mat(A, thr=1.0e-7, ovlp=None):
    S = dot_gen(A, A, ovlp)
    e, u = eigh(S)
    if thr > 0:
        idx_keep = e / e[-1] > thr
    else:
        idx_keep = list(range(e.shape[0]))
    return u[:, idx_keep] * e[idx_keep] ** -0.5


def cano_orth(A, thr=1.0e-7, ovlp=None):
    """Canonically orthogonalize columns of A"""
    return A @ get_cano_orth_mat(A, thr, ovlp)


def get_symm_orth_mat_k(A, thr=1.0e-7, ovlp=None):
    S = dot_gen(A, A, ovlp)
    e, u = scipy.linalg.eigh(S)
    if (e < thr).any():
        raise ValueError(
            "Linear dependence is detected in the column space of A: "
            "smallest eigenvalue (%.3E) is less than thr (%.3E). "
            "Please use 'cano_orth' instead." % (np.min(e), thr)
        )
    return u @ diag(e**-0.5) @ u.conj().T


def symm_orth_k(A, thr=1.0e-7, ovlp=None):
    """Symmetrically orthogonalize columns of A"""
    return A @ get_symm_orth_mat_k(A, thr, ovlp)


def get_xovlp_k(cell, kpts, basis="sto-3g"):
    """Gets set of valence orbitals based on smaller (should be minimal) basis

    Parameters
    ----------
    cell:
        pyscf cell object, just need it for the working basis
    basis:
        the IAO basis, Knizia recommended 'minao'

    Returns
    -------
    tuple:
        S12 - Overlap of two basis sets,
        S22 - Overlap in new basis set
    """
    cell_alt = cell.copy()
    cell_alt.basis = basis
    cell_alt.build()

    S22 = array(cell_alt.pbc_intor("int1e_ovlp", hermi=1, kpts=kpts), dtype=complex128)
    S12 = array(
        pgto.cell.intor_cross("int1e_ovlp", cell, cell_alt, kpts=kpts),
        dtype=complex128,
    )

    return (S12, S22)


def remove_core_mo_k(Clo, Ccore, S, thr=0.5):
    assert allclose(Clo.conj().T @ S @ Clo, eye(Clo.shape[1]))
    assert allclose(Ccore.conj().T @ S @ Ccore, eye(Ccore.shape[1]))

    n, nlo = Clo.shape
    ncore = Ccore.shape[1]
    Pcore = Ccore @ Ccore.conj().T @ S
    Clo1 = (eye(n) - Pcore) @ Clo
    pop = diag(Clo1.conj().T @ S @ Clo1)
    idx_keep = where(pop > thr)[0]
    assert len(idx_keep) == nlo - ncore
    Clo2 = symm_orth_k(Clo1[:, idx_keep], ovlp=S)

    return Clo2


def get_iao_k(Co, S12, S1, S2=None, ortho=True):
    """

    Parameters
    ----------
    Co:
        occupied coefficient matrix with core
    p:
        valence AO matrix in AO
    no:
        number of occ orbitals
    S12:
        ovlp between working basis and valence basis
        can be thought of as working basis in valence basis
    S1:
        ao ovlp matrix
    S2:
        valence AO ovlp
    """

    nk, nao, nmo = S12.shape
    unused(nmo)
    P1 = zeros_like(S1, dtype=complex128)
    P2 = zeros_like(S2, dtype=complex128)

    for k in range(nk):
        P1[k] = scipy.linalg.inv(S1[k])
        P2[k] = scipy.linalg.inv(S2[k])

    Ciao = zeros((nk, nao, S12.shape[-1]), dtype=complex128)
    for k in range(nk):
        # Cotil = P1[k] @ S12[k] @ P2[k] @ S12[k].conj().T @ Co[k]
        Cotil = multi_dot((P1[k], S12[k], P2[k], S12[k].conj().T, Co[k]))
        ptil = P1[k] @ S12[k]
        Stil = multi_dot((Cotil.conj().T, S1[k], Cotil))

        Po = Co[k] @ Co[k].conj().T

        Stil_inv = inv(Stil)

        Potil = multi_dot((Cotil, Stil_inv, Cotil.conj().T))

        Ciao[k] = (
            eye(nao, dtype=complex128)
            - (Po + Potil - 2.0 * multi_dot([Po, S1[k], Potil])) @ S1[k]
        ) @ ptil
        if ortho:
            Ciao[k] = symm_orth_k(Ciao[k], ovlp=S1[k])

            rep_err = norm(multi_dot(Ciao[k], Ciao[k].conj().T, S1[k], Po) - Po)
            if rep_err > 1.0e-10:
                raise RuntimeError

    return Ciao


def get_pao_k(Ciao, S, S12):
    """

    Parameters
    ----------
    Ciao:
        output of :func:`quemb.kbe.lo_k.get_iao_k`
    S:
        ao ovlp matrix
    S12:
        valence orbitals projected into ao basis

    Returns
    -------
    Cpao: numpy.ndarray
        (orthogonalized)
    """
    nk, nao, niao = Ciao.shape
    unused(niao)
    Cpao = []
    for k in range(nk):
        s12 = scipy.linalg.inv(S[k]) @ S12[k]
        nonval = eye(nao) - s12 @ s12.conj().T

        Piao = Ciao[k] @ Ciao[k].conj().T @ S[k]
        cpao_ = (eye(nao) - Piao) @ nonval

        Cpao.append(cano_orth(cpao_, ovlp=S[k]))
    return asarray(Cpao)


def get_pao_native_k(Ciao, S, mol, valence_basis, ortho=True):
    """

    Parameters
    ----------
    Ciao :
        output of :code:`get_iao_native`
    S :
        ao ovlp matrix
    mol :
        mol object
    valence_basis:
        basis used for valence orbitals

    Returns
    --------
    Cpao : numpy.ndarray
        (symmetrically orthogonalized)
    """
    nk, nao, niao = Ciao.shape

    # Form a mol object with the valence basis for the ao_labels
    mol_alt = mol.copy()
    mol_alt.basis = valence_basis
    mol_alt.build()

    full_ao_labels = mol.ao_labels()
    valence_ao_labels = mol_alt.ao_labels()

    vir_idx = [
        idx
        for idx, label in enumerate(full_ao_labels)
        if (label not in valence_ao_labels)
    ]

    niao = len(vir_idx)
    Cpao = zeros((nk, nao, niao), dtype=complex128)
    for k in range(nk):
        Piao = multi_dot((Ciao[k], Ciao[k].conj().T, S[k]))
        cpao_ = (eye(nao) - Piao)[:, vir_idx]
        if ortho:
            try:
                Cpao[k] = symm_orth_k(cpao_, ovlp=S[k])
            except ValueError:
                print("Symm orth PAO failed. Switch to cano orth", flush=True)
                npao0 = cpao_.shape[1]
                Cpao[k] = cano_orth(cpao_, ovlp=S[k])
                npao1 = cpao_.shape[1]
                print("# of PAO: %d --> %d" % (npao0, npao1), flush=True)
                print("", flush=True)
        else:
            Cpao[k] = cpao_.copy()

    return Cpao
