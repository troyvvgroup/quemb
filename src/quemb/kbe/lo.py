# Author(s): Oinam Meitei
#            Henry Tran
#
import os

import numpy as np
from libdmet.lo import pywannier90
from numpy import (
    allclose,
    array,
    asarray,
    complex128,
    diag,
    eye,
    hstack,
    sqrt,
    where,
    zeros,
    zeros_like,
)
from numpy.linalg import eigh, multi_dot, svd

from quemb.kbe.lo_k import (
    get_iao_k,
    get_pao_native_k,
    get_xovlp_k,
    remove_core_mo_k,
    symm_orth_k,
)
from quemb.shared.external.lo_helper import get_aoind_by_atom, reorder_by_atom_
from quemb.shared.helper import ncore_, unused


class KMF:
    def __init__(self, cell, kpts=None, mo_coeff=None, mo_energy=None):
        self.cell = cell
        self.kpts = kpts
        self.mo_coeff = mo_coeff.copy()
        self.mo_energy = mo_energy
        self.mo_energy_kpts = mo_energy
        self.mo_coeff_kpts = mo_coeff.copy()


class Mixin_k_Localize:
    def localize(
        self,
        lo_method,
        iao_valence_basis="sto-3g",
        core_basis="sto-3g",
        iao_wannier=True,
        iao_val_core=True,
    ):
        """Orbital localization

        Performs orbital localization computations for periodic systems. For large
        basis, IAO is recommended augmented with PAO orbitals.

        Parameters
        ----------
        lo_method : str
            Localization method in quantum chemistry. 'lowdin', 'boys','iao',
            and 'wannier' are supported.
        iao_valence_basis : str
            Name of valence basis set for IAO scheme. 'sto-3g' suffice for most cases.
        core_basis : str
            Name of core basis set for IAO scheme. 'sto-3g' suffice for most cases.
        iao_wannier : bool
            Whether to perform Wannier localization in the IAO space
        """
        if lo_method == "iao" and iao_val_core:
            raise NotImplementedError("iao_val_core and lo_method='iao' not supported.")

        if lo_method == "lowdin":
            # Lowdin orthogonalization with k-points
            W = zeros_like(self.S)
            nk, nao, nmo = self.C.shape
            if self.frozen_core:
                W_nocore = zeros_like(self.S[:, :, self.ncore :])
                lmo_coeff = zeros_like(self.C[:, self.ncore :, self.ncore :])
                cinv_ = zeros((nk, nmo - self.ncore, nao), dtype=complex128)
            else:
                lmo_coeff = zeros_like(self.C)
                cinv_ = zeros((nk, nmo, nao), dtype=complex128)

            for k in range(self.nkpt):
                es_, vs_ = eigh(self.S[k])
                edx = es_ > 1.0e-14

                W[k] = (vs_[:, edx] / sqrt(es_[edx])) @ vs_[:, edx].conj().T
                for i in range(W[k].shape[1]):
                    if W[k][i, i] < 0:
                        W[:, i] *= -1
                if self.frozen_core:
                    pcore = eye(W[k].shape[0]) - (self.P_core[k] @ self.S[k])
                    C_ = pcore @ W[k]

                    # PYSCF has basis in 1s2s3s2p2p2p3p3p3p format
                    # fix no_core_idx - use population for now
                    # C_ = C_[:,self.no_core_idx]
                    Cpop = multi_dot((C_.conj().T, self.S[k], C_))
                    Cpop = diag(Cpop.real)

                    no_core_idx = where(Cpop > 0.7)[0]
                    C_ = C_[:, no_core_idx]

                    S_ = multi_dot((C_.conj().T, self.S[k], C_))

                    es_, vs_ = eigh(S_)
                    edx = es_ > 1.0e-14
                    W_ = (vs_[:, edx] / sqrt(es_[edx])) @ vs_[:, edx].conj().T
                    W_nocore[k] = C_ @ W_

                    lmo_coeff[k] = multi_dot(
                        (W_nocore[k].conj().T, self.S[k], self.C[k][:, self.ncore :]),
                    )
                    cinv_[k] = W_nocore[k].conj().T @ self.S[k]

                else:
                    lmo_coeff[k] = multi_dot((W[k].conj().T, self.S[k], self.C[k]))
                    cinv_[k] = W[k].conj().T @ self.S[k]
            if self.frozen_core:
                self.W = W_nocore
            else:
                self.W = W
            self.lmo_coeff = lmo_coeff
            self.cinv = cinv_

        elif lo_method == "iao":
            if not iao_val_core or not self.frozen_core:
                Co = self.C[:, :, : self.Nocc].copy()
                S12, S2 = get_xovlp_k(self.cell, self.kpts, basis=iao_valence_basis)
                ciao_ = get_iao_k(Co, S12, self.S, S2=S2)

                # tmp - aos are not rearrange and so below is not necessary
                nk, nao, nlo = ciao_.shape
                Ciao_ = zeros((nk, nao, nlo), dtype=complex128)
                for k in range(self.nkpt):
                    aoind_by_atom = get_aoind_by_atom(self.cell)
                    ctmp, iaoind_by_atom = reorder_by_atom_(
                        ciao_[k], aoind_by_atom, self.S[k]
                    )
                    Ciao_[k] = ctmp

                # get_pao_k returns canonical orthogonalized orbitals
                # Cpao = get_pao_k(Ciao, self.S, S12, S2, self.cell)
                # get_pao_native_k returns symm orthogonalized orbitals
                cpao_ = get_pao_native_k(
                    Ciao_, self.S, self.cell, iao_valence_basis, self.kpts
                )

                nk, nao, nlo = cpao_.shape
                Cpao_ = zeros((nk, nao, nlo), dtype=complex128)
                for k in range(self.nkpt):
                    aoind_by_atom = get_aoind_by_atom(self.cell)
                    ctmp, paoind_by_atom = reorder_by_atom_(
                        cpao_[k], aoind_by_atom, self.S[k]
                    )
                    Cpao_[k] = ctmp

                nk, nao, nlo = Ciao_.shape
                if self.frozen_core:
                    nk, nao, nlo = Ciao_.shape
                    Ciao_nocore = zeros((nk, nao, nlo - self.ncore), dtype=complex128)
                    for k in range(nk):
                        Ccore = self.C[k][:, : self.ncore]
                        Ciao_nocore[k] = remove_core_mo_k(Ciao_[k], Ccore, self.S[k])
                    Ciao_ = Ciao_nocore

            else:
                # Construct seperate IAOs for the core and valence

                # Begin core
                s12_core_, s2_core = get_xovlp_k(self.cell, self.kpts, basis=core_basis)
                C_core_ = self.C[:, :, : self.ncore].copy()
                nk_, nao_, nmo_ = C_core_.shape
                s1_core = zeros((nk_, nmo_, nmo_), dtype=self.S.dtype)
                s12_core = zeros(
                    (nk_, nmo_, s12_core_.shape[-1]), dtype=s12_core_.dtype
                )
                C_core = zeros((nk_, self.ncore, self.ncore), dtype=C_core_.dtype)
                for k in range(nk_):
                    C_core[k] = C_core_[k].conj().T @ self.S[k] @ C_core_[k]
                    s1_core[k] = C_core_[k].conj().T @ self.S[k] @ C_core_[k]
                    s12_core[k] = C_core_[k].conj().T @ s12_core_[k]
                ciao_core_ = get_iao_k(C_core, s12_core, s1_core, s2_core, ortho=False)
                ciao_core = zeros(
                    (nk_, nao_, ciao_core_.shape[-1]), dtype=ciao_core_.dtype
                )
                for k in range(nk_):
                    ciao_core[k] = C_core_[k] @ ciao_core_[k]
                    ciao_core[k] = symm_orth_k(ciao_core[k], ovlp=self.S[k])

                # Begin valence
                s12_val_, s2_val = get_xovlp_k(
                    self.cell, self.kpts, basis=iao_valence_basis
                )
                C_nocore = self.C[:, :, self.ncore :].copy()
                C_nocore_occ_ = C_nocore[:, :, : self.Nocc].copy()
                nk_, nao_, nmo_ = C_nocore.shape
                s1_val = zeros((nk_, nmo_, nmo_), dtype=self.S.dtype)
                s12_val = zeros((nk_, nmo_, s12_val_.shape[-1]), dtype=s12_val_.dtype)
                C_nocore_occ = zeros(
                    (nk_, nao_ - self.ncore, C_nocore_occ_.shape[-1]),
                    dtype=C_nocore_occ_.dtype,
                )
                for k in range(nk_):
                    C_nocore_occ[k] = (
                        C_nocore[k].conj().T @ self.S[k] @ C_nocore_occ_[k]
                    )
                    s1_val[k] = C_nocore[k].conj().T @ self.S[k] @ C_nocore[k]
                    s12_val[k] = C_nocore[k].conj().T @ s12_val_[k]
                ciao_val_ = get_iao_k(
                    C_nocore_occ, s12_val, s1_val, s2_val, ortho=False
                )
                Ciao_ = zeros((nk_, nao_, ciao_val_.shape[-1]), dtype=ciao_val_.dtype)
                for k in range(nk_):
                    Ciao_[k] = C_nocore[k] @ ciao_val_[k]
                    Ciao_[k] = symm_orth_k(Ciao_[k], ovlp=self.S[k])

                # stack core|val
                nao = self.S.shape[-1]
                c_core_val = zeros(
                    (nk_, nao, Ciao_.shape[-1] + self.ncore), dtype=Ciao_.dtype
                )
                for k in range(nk_):
                    c_core_val[k] = hstack((ciao_core[k], Ciao_[k]))

                # tmp - aos are not rearrange and so below is not necessary
                #   (iaoind_by_atom is used to stack iao|pao later)
                nk, nao, nlo = c_core_val.shape
                for k in range(self.nkpt):
                    aoind_by_atom = get_aoind_by_atom(self.cell)
                    ctmp, iaoind_by_atom = reorder_by_atom_(
                        c_core_val[k], aoind_by_atom, self.S[k]
                    )

                cpao_ = get_pao_native_k(
                    c_core_val,
                    self.S,
                    self.cell,
                    iao_valence_basis,
                    self.kpts,
                    ortho=True,
                )
                nk, nao, nlo = cpao_.shape
                Cpao_ = zeros((nk, nao, nlo), dtype=complex128)
                for k in range(self.nkpt):
                    aoind_by_atom = get_aoind_by_atom(self.cell)
                    ctmp, paoind_by_atom = reorder_by_atom_(
                        cpao_[k], aoind_by_atom, self.S[k]
                    )
                    Cpao_[k] = ctmp

            Cpao = Cpao_.copy()
            Ciao = Ciao_.copy()

            if iao_wannier:
                mo_energy_ = []
                for k in range(nk):
                    fock_iao = multi_dot((Ciao_[k].conj().T, self.FOCK[k], Ciao_[k]))
                    S_iao = multi_dot((Ciao_[k].conj().T, self.S[k], Ciao_[k]))
                    e_iao, v_iao = eigh(fock_iao, S_iao)
                    unused(v_iao)
                    mo_energy_.append(e_iao)
                iaomf = KMF(
                    self.mol, kpts=self.kpts, mo_coeff=Ciao_, mo_energy=mo_energy_
                )

                num_wann = asarray(iaomf.mo_coeff).shape[2]
                keywords = """
                num_iter = 5000
                dis_num_iter = 0
                conv_noise_amp = -2.0
                conv_window = 100
                conv_tol = 1.0E-09
                iprint = 3
                kmesh_tol = 0.00001
                """
                # set conv window
                # dis_num_iter=0
                w90 = pywannier90.W90(
                    iaomf, self.kmesh, num_wann, other_keywords=keywords
                )

                A_matrix = zeros((self.nkpt, num_wann, num_wann), dtype=complex128)

                for k in range(self.nkpt):
                    A_matrix[k] = eye(num_wann, dtype=complex128)
                A_matrix = A_matrix.transpose(1, 2, 0)

                w90.kernel(A_matrix=A_matrix)

                u_mat = array(
                    w90.U_matrix.transpose(2, 0, 1), order="C", dtype=complex128
                )

                os.system("cp wannier90.wout wannier90_iao.wout")
                os.system("rm wannier90.*")

                nk, nao, nlo = Ciao_.shape
                Ciao = zeros((nk, nao, nlo), dtype=complex128)

                for k in range(self.nkpt):
                    Ciao[k] = Ciao_[k] @ u_mat[k]

            # Stack Ciao
            Wstack = zeros(
                (self.nkpt, Ciao.shape[1], Ciao.shape[2] + Cpao.shape[2]),
                dtype=complex128,
            )
            if self.frozen_core:
                for k in range(self.nkpt):
                    shift = 0
                    ncore = 0
                    for ix in range(self.cell.natm):
                        nc = ncore_(self.cell.atom_charge(ix))
                        ncore += nc
                        niao = len(iaoind_by_atom[ix])
                        iaoind_ix = [i_ - ncore for i_ in iaoind_by_atom[ix][nc:]]
                        Wstack[k][:, shift : shift + niao - nc] = Ciao[k][:, iaoind_ix]
                        shift += niao - nc
                        npao = len(paoind_by_atom[ix])

                        Wstack[k][:, shift : shift + npao] = Cpao[k][
                            :, paoind_by_atom[ix]
                        ]
                        shift += npao
            else:
                for k in range(self.nkpt):
                    shift = 0
                    for ix in range(self.cell.natm):
                        niao = len(iaoind_by_atom[ix])
                        Wstack[k][:, shift : shift + niao] = Ciao[k][
                            :, iaoind_by_atom[ix]
                        ]
                        shift += niao
                        npao = len(paoind_by_atom[ix])
                        Wstack[k][:, shift : shift + npao] = Cpao[k][
                            :, paoind_by_atom[ix]
                        ]
                        shift += npao
            self.W = Wstack

            nmo = self.C.shape[2] - self.ncore
            nlo = self.W.shape[2]
            nao = self.S.shape[2]

            lmo_coeff = zeros((self.nkpt, nlo, nmo), dtype=complex128)
            cinv_ = zeros((self.nkpt, nlo, nao), dtype=complex128)

            if nmo > nlo:
                Co_nocore = self.C[:, :, self.ncore : self.Nocc]
                Cv = self.C[:, :, self.Nocc :]
                # Ensure that the LOs span the occupied space
                for k in range(self.nkpt):
                    assert allclose(
                        np.sum((self.W[k].conj().T @ self.S[k] @ Co_nocore[k]) ** 2.0),
                        self.Nocc - self.ncore,
                    )
                    # Find virtual orbitals that lie in the span of LOs
                    u, l, vt = svd(
                        self.W[k].conj().T @ self.S[k] @ Cv[k], full_matrices=False
                    )
                    unused(u)
                    nvlo = nlo - self.Nocc - self.ncore
                    assert allclose(np.sum(l[:nvlo]), nvlo)
                    C_ = hstack([Co_nocore[k], Cv[k] @ vt[:nvlo].conj().T])
                    lmo_ = self.W[k].conj().T @ self.S[k] @ C_
                    assert allclose(lmo_.conj().T @ lmo_, eye(lmo_.shape[1]))
                    lmo_coeff.append(lmo_)
            else:
                for k in range(self.nkpt):
                    lmo_coeff[k] = multi_dot(
                        (self.W[k].conj().T, self.S[k], self.C[k][:, self.ncore :]),
                    )
                    cinv_[k] = self.W[k].conj().T @ self.S[k]

                    assert allclose(
                        lmo_coeff[k].conj().T @ lmo_coeff[k],
                        eye(lmo_coeff[k].shape[1]),
                    )

            self.lmo_coeff = lmo_coeff
            self.cinv = cinv_

        elif lo_method == "wannier":
            nk, nao, nmo = self.C.shape
            lorb = zeros((nk, nao, nmo), dtype=complex128)
            lorb_nocore = zeros((nk, nao, nmo - self.ncore), dtype=complex128)
            for k in range(nk):
                es_, vs_ = eigh(self.S[k])
                edx = es_ > 1.0e-14
                lorb[k] = vs_[:, edx] / sqrt(es_[edx]) @ vs_[:, edx].conj().T

                if self.frozen_core:
                    Ccore = self.C[k][:, : self.ncore]
                    lorb_nocore[k] = remove_core_mo_k(lorb[k], Ccore, self.S[k])

            if not self.frozen_core:
                lmf = KMF(
                    self.mol, kpts=self.kpts, mo_coeff=lorb, mo_energy=self.mo_energy
                )
            else:
                mo_energy_nc = []
                for k in range(nk):
                    fock_lnc = multi_dot(
                        (lorb_nocore[k].conj().T, self.FOCK[k], lorb_nocore[k]),
                    )
                    S_lnc = multi_dot(
                        (lorb_nocore[k].conj().T, self.S[k], lorb_nocore[k])
                    )
                    e__, v__ = eigh(fock_lnc, S_lnc)
                    unused(v__)
                    mo_energy_nc.append(e__)
                lmf = KMF(
                    self.mol,
                    kpts=self.kpts,
                    mo_coeff=lorb_nocore,
                    mo_energy=mo_energy_nc,
                )

            num_wann = lmf.mo_coeff.shape[2]
            keywords = """
            num_iter = 10000
            dis_num_iter = 0
            conv_window = 10
            conv_tol = 1.0E-09
            iprint = 3
            kmesh_tol = 0.00001
            """

            w90 = pywannier90.W90(lmf, self.kmesh, num_wann, other_keywords=keywords)
            A_matrix = zeros((self.nkpt, num_wann, num_wann), dtype=complex128)
            # Using A=I + lowdin orbital and A=<psi|lowdin> + |psi> is the same
            for k in range(self.nkpt):
                A_matrix[k] = eye(num_wann, dtype=complex128)

            A_matrix = A_matrix.transpose(1, 2, 0)

            w90.kernel(A_matrix=A_matrix)
            u_mat = array(w90.U_matrix.transpose(2, 0, 1), order="C", dtype=complex128)

            nk, nao, nlo = lmf.mo_coeff.shape
            W = zeros((nk, nao, nlo), dtype=complex128)
            for k in range(nk):
                W[k] = lmf.mo_coeff[k] @ u_mat[k]

            self.W = W
            lmo_coeff = zeros((self.nkpt, nlo, nmo - self.ncore), dtype=complex128)
            cinv_ = zeros((self.nkpt, nlo, nao), dtype=complex128)

            for k in range(nk):
                lmo_coeff[k] = multi_dot(
                    (self.W[k].conj().T, self.S[k], self.C[k][:, self.ncore :]),
                )
                cinv_[k] = self.W[k].conj().T @ self.S[k]
                assert allclose(
                    lmo_coeff[k].conj().T @ lmo_coeff[k],
                    eye(lmo_coeff[k].shape[1]),
                )
            self.lmo_coeff = lmo_coeff
            self.cinv = cinv_

        else:
            raise ValueError(f"lo_method = {lo_method} not implemented!")
