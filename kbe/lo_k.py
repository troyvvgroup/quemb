# Author(s): Henry Tran
#            Oinam Meitei
#

import numpy, sys, scipy
from functools import reduce


def dot_gen(A, B, ovlp):
    if ovlp is None:
        Ad = numpy.dot(A.conj().T, B)
    else:
        Ad = reduce(numpy.dot, (A.conj().T, ovlp, B))
    return Ad


def get_cano_orth_mat(A, thr=1.0e-7, ovlp=None):
    S = dot_gen(A, A, ovlp)
    e, u = numpy.linalg.eigh(S)
    if thr > 0:
        idx_keep = e / e[-1] > thr
    else:
        idx_keep = list(range(e.shape[0]))
    U = u[:, idx_keep] * e[idx_keep] ** -0.5

    return U


def cano_orth(A, thr=1.0e-7, ovlp=None):
    """Canonically orthogonalize columns of A"""
    U = get_cano_orth_mat(A, thr, ovlp)

    return A @ U


def get_symm_orth_mat_k(A, thr=1.0e-7, ovlp=None):
    S = dot_gen(A, A, ovlp)
    e, u = scipy.linalg.eigh(S)
    if int(numpy.sum(e < thr)) > 0:
        raise ValueError(
            "Linear dependence is detected in the column space of A: smallest eigenvalue (%.3E) is less than thr (%.3E). Please use 'cano_orth' instead."
            % (numpy.min(e), thr)
        )
    U = reduce(numpy.dot, (u, numpy.diag(e**-0.5), u.conj().T))
    # U = reduce(numpy.dot, (u/numpy.sqrt(e), u.conj().T))
    return U


def symm_orth_k(A, thr=1.0e-7, ovlp=None):
    """Symmetrically orthogonalize columns of A"""
    U = get_symm_orth_mat_k(A, thr, ovlp)
    AU = numpy.dot(A, U)

    return AU


def get_xovlp_k(cell, kpts, basis="sto-3g"):
    """
    Gets set of valence orbitals based on smaller (should be minimal) basis
    inumpy.t:
        cell - pyscf cell object, just need it for the working basis
        basis - the IAO basis, Knizia recommended 'minao'
    returns:
        S12 - Overlap of two basis sets
        S22 - Overlap in new basis set
    """

    from pyscf.pbc import gto as pgto  # intor_cross

    cell_alt = cell.copy()
    cell_alt.basis = basis
    cell_alt.build()

    S22 = numpy.array(
        cell_alt.pbc_intor("int1e_ovlp", hermi=1, kpts=kpts), dtype=numpy.complex128
    )
    S12 = numpy.array(
        pgto.cell.intor_cross("int1e_ovlp", cell, cell_alt, kpts=kpts),
        dtype=numpy.complex128,
    )

    return (S12, S22)


def remove_core_mo_k(Clo, Ccore, S, thr=0.5):
    assert numpy.allclose(Clo.conj().T @ S @ Clo, numpy.eye(Clo.shape[1]))
    assert numpy.allclose(Ccore.conj().T @ S @ Ccore, numpy.eye(Ccore.shape[1]))

    n, nlo = Clo.shape
    ncore = Ccore.shape[1]
    Pcore = Ccore @ Ccore.conj().T @ S
    Clo1 = (numpy.eye(n) - Pcore) @ Clo
    pop = numpy.diag(Clo1.conj().T @ S @ Clo1)
    idx_keep = numpy.where(pop > thr)[0]
    assert len(idx_keep) == nlo - ncore
    Clo2 = symm_orth_k(Clo1[:, idx_keep], ovlp=S)

    return Clo2


def get_iao_k(Co, S12, S1, S2=None, ortho=True):
    """
    Args:
        Co: occupied coefficient matrix with core
        p: valence AO matrix in AO
        no: number of occ orbitals
        S12: ovlp between working basis and valence basis
             can be thought of as working basis in valence basis
        S1: ao ovlp matrix
        S2: valence AO ovlp
    """

    nk, nao, nmo = S12.shape
    P1 = numpy.zeros_like(S1, dtype=numpy.complex128)
    P2 = numpy.zeros_like(S2, dtype=numpy.complex128)

    for k in range(nk):
        P1[k] = scipy.linalg.inv(S1[k])
        P2[k] = scipy.linalg.inv(S2[k])

    Ciao = numpy.zeros((nk, nao, S12.shape[-1]), dtype=numpy.complex128)
    for k in range(nk):
        # Cotil = P1[k] @ S12[k] @ P2[k] @ S12[k].conj().T @ Co[k]
        Cotil = reduce(numpy.dot, (P1[k], S12[k], P2[k], S12[k].conj().T, Co[k]))
        ptil = numpy.dot(P1[k], S12[k])
        Stil = reduce(numpy.dot, (Cotil.conj().T, S1[k], Cotil))

        Po = numpy.dot(Co[k], Co[k].conj().T)

        Stil_inv = numpy.linalg.inv(Stil)

        Potil = reduce(numpy.dot, (Cotil, Stil_inv, Cotil.conj().T))

        Ciao[k] = (
            numpy.eye(nao, dtype=numpy.complex128)
            - numpy.dot(
                (Po + Potil - 2.0 * reduce(numpy.dot, (Po, S1[k], Potil))), S1[k]
            )
        ) @ ptil
        if ortho:
            Ciao[k] = symm_orth_k(Ciao[k], ovlp=S1[k])

            rep_err = numpy.linalg.norm(Ciao[k] @ Ciao[k].conj().T @ S1[k] @ Po - Po)
            if rep_err > 1.0e-10:
                raise RuntimeError

    return Ciao


def get_pao_k(Ciao, S, S12, S2):
    """
    Args:
        Ciao: output of :func:`get_iao`
        S: ao ovlp matrix
        S12: valence orbitals projected into ao basis
        S2: valence ovlp matrix
        mol: pyscf mol instance
    Return:
        Cpao (orthogonalized)
    """

    nk, nao, niao = Ciao.shape
    Cpao = []
    for k in range(nk):
        s12 = scipy.linalg.inv(S[k]) @ S12[k]
        nonval = numpy.eye(nao) - s12 @ s12.conj().T

        Piao = Ciao[k] @ Ciao[k].conj().T @ S[k]
        cpao_ = (numpy.eye(nao) - Piao) @ nonval

        numpy.o0 = cpao_.shape[-1]
        Cpao.append(cano_orth(cpao_, ovlp=S[k]))
        numpy.o1 = Cpao[k].shape[-1]
    Cpao = numpy.asarray(Cpao)

    return Cpao


def get_pao_native_k(Ciao, S, mol, valence_basis, kpts, ortho=True):
    """
    Args:
        Ciao: output of :func:`get_iao_native`
        S: ao ovlp matrix
        mol: mol object
        valence basis: basis used for valence orbitals
    Return:
        Cpao (symmetrically orthogonalized)
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
        if (not label in valence_ao_labels)
    ]

    niao = len(vir_idx)
    Cpao = numpy.zeros((nk, nao, niao), dtype=numpy.complex128)
    for k in range(nk):
        Piao = reduce(numpy.dot, (Ciao[k], Ciao[k].conj().T, S[k]))
        cpao_ = (numpy.eye(nao) - Piao)[:, vir_idx]
        if ortho:
            try:
                Cpao[k] = symm_orth_k(cpao_, ovlp=S[k])
            except:
                print("Symm orth PAO failed. Switch to cano orth", flush=True)
                npao0 = cpao_.shape[1]
                Cpao[k] = cano_orth(cpao_, ovlp=S[k])
                npao1 = cpao_.shape[1]
                print("# of PAO: %d --> %d" % (npao0, npao1), flush=True)
                print("", flush=True)
        else:
            Cpao[k] = cpao_.copy()

    return Cpao
