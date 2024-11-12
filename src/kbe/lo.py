# Author(s): Oinam Meitei
#            Henry Tran
#
import sys
from functools import reduce

import numpy

from molbe.external.lo_helper import get_aoind_by_atom, reorder_by_atom_

# iao tmp


class KMF:
    def __init__(self, cell, kpts=None, mo_coeff=None, mo_energy=None):
        self.cell = cell
        self.kpts = kpts
        self.mo_coeff = mo_coeff.copy()
        self.mo_energy = mo_energy
        self.mo_energy_kpts = mo_energy
        self.mo_coeff_kpts = mo_coeff.copy()


def localize(
    self,
    lo_method,
    mol=None,
    valence_basis="sto-3g",
    iao_wannier=True,
    valence_only=False,
    iao_val_core=True,
):
    """Orbital localization

    Performs orbital localization computations for periodic systems. For large basis, IAO is recommended
    augmented with PAO orbitals.

    Parameters
    ----------
    lo_method : str
       Localization method in quantum chemistry. 'lowdin', 'boys','iao', and 'wannier' are supported.
    mol : pyscf.gto.Molecule
       pyscf.gto.Molecule object.
    valence_basis: str
       Name of minimal basis set for IAO scheme. 'sto-3g' suffice for most cases.
    valence_only: bool
       If this option is set to True, all calculation will be performed in the valence basis in the IAO partitioning.
       This is an experimental feature.
    iao_wannier : bool
       Whether to perform Wannier localization in the IAO space
    """
    import functools

    import scipy.linalg

    from molbe.helper import ncore_

    if lo_method == "iao":
        if valence_basis == "sto-3g":
            from .basis_sto3g_core_val import core_basis, val_basis
        elif valence_basis == "minao":
            from .basis_minao_core_val import core_basis, val_basis
        elif iao_val_core:
            sys.exit(
                "valence_basis="
                + valence_basis
                + " not supported for iao_val_core=True"
            )

    if lo_method == "lowdin":
        # Lowdin orthogonalization with k-points
        W = numpy.zeros_like(self.S)
        nk, nao, nmo = self.C.shape
        if self.frozen_core:
            W_nocore = numpy.zeros_like(self.S[:, :, self.ncore :])
            lmo_coeff = numpy.zeros_like(self.C[:, self.ncore :, self.ncore :])
            cinv_ = numpy.zeros((nk, nmo - self.ncore, nao), dtype=numpy.complex128)
        else:
            lmo_coeff = numpy.zeros_like(self.C)
            cinv_ = numpy.zeros((nk, nmo, nao), dtype=numpy.complex128)

        for k in range(self.nkpt):
            es_, vs_ = scipy.linalg.eigh(self.S[k])
            edx = es_ > 1.0e-14

            W[k] = numpy.dot(vs_[:, edx] / numpy.sqrt(es_[edx]), vs_[:, edx].conj().T)
            for i in range(W[k].shape[1]):
                if W[k][i, i] < 0:
                    W[:, i] *= -1
            if self.frozen_core:
                pcore = numpy.eye(W[k].shape[0]) - numpy.dot(self.P_core[k], self.S[k])
                C_ = numpy.dot(pcore, W[k])

                # PYSCF has basis in 1s2s3s2p2p2p3p3p3p format
                # fix no_core_idx - use population for now
                # C_ = C_[:,self.no_core_idx]
                Cpop = functools.reduce(numpy.dot, (C_.conj().T, self.S[k], C_))
                Cpop = numpy.diag(Cpop.real)

                no_core_idx = numpy.where(Cpop > 0.7)[0]
                C_ = C_[:, no_core_idx]

                S_ = functools.reduce(numpy.dot, (C_.conj().T, self.S[k], C_))

                es_, vs_ = scipy.linalg.eigh(S_)
                edx = es_ > 1.0e-14
                W_ = numpy.dot(vs_[:, edx] / numpy.sqrt(es_[edx]), vs_[:, edx].conj().T)
                W_nocore[k] = numpy.dot(C_, W_)

                lmo_coeff[k] = functools.reduce(
                    numpy.dot,
                    (W_nocore[k].conj().T, self.S[k], self.C[k][:, self.ncore :]),
                )
                cinv_[k] = numpy.dot(W_nocore[k].conj().T, self.S[k])

            else:
                lmo_coeff[k] = functools.reduce(
                    numpy.dot, (W[k].conj().T, self.S[k], self.C[k])
                )
                cinv_[k] = numpy.dot(W[k].conj().T, self.S[k])
        if self.frozen_core:
            self.W = W_nocore
        else:
            self.W = W
        self.lmo_coeff = lmo_coeff
        self.cinv = cinv_

    elif lo_method == "iao":
        import os

        from libdmet.lo import pywannier90

        from .lo_k import (
            get_iao_k,
            get_pao_native_k,
            get_xovlp_k,
            remove_core_mo_k,
            symm_orth_k,
        )

        if not iao_val_core or not self.frozen_core:
            Co = self.C[:, :, : self.Nocc].copy()
            S12, S2 = get_xovlp_k(self.cell, self.kpts, basis=valence_basis)
            ciao_ = get_iao_k(Co, S12, self.S, S2=S2)

            arrange_by_atom = True
            # tmp - aos are not rearrange and so below is not necessary
            if arrange_by_atom:
                nk, nao, nlo = ciao_.shape
                Ciao_ = numpy.zeros((nk, nao, nlo), dtype=numpy.complex128)
                for k in range(self.nkpt):
                    aoind_by_atom = get_aoind_by_atom(self.cell)
                    ctmp, iaoind_by_atom = reorder_by_atom_(
                        ciao_[k], aoind_by_atom, self.S[k]
                    )
                    Ciao_[k] = ctmp
            else:
                Ciao_ = ciao_.copy()

            # get_pao_k returns canonical orthogonalized orbitals
            # Cpao = get_pao_k(Ciao, self.S, S12, S2, self.cell)
            # get_pao_native_k returns symm orthogonalized orbitals
            cpao_ = get_pao_native_k(Ciao_, self.S, self.cell, valence_basis, self.kpts)

            if arrange_by_atom:
                nk, nao, nlo = cpao_.shape
                Cpao_ = numpy.zeros((nk, nao, nlo), dtype=numpy.complex128)
                for k in range(self.nkpt):
                    aoind_by_atom = get_aoind_by_atom(self.cell)
                    ctmp, paoind_by_atom = reorder_by_atom_(
                        cpao_[k], aoind_by_atom, self.S[k]
                    )
                    Cpao_[k] = ctmp
            else:
                Cpao_ = cpao_.copy()

            nk, nao, nlo = Ciao_.shape
            if self.frozen_core:
                nk, nao, nlo = Ciao_.shape
                Ciao_nocore = numpy.zeros(
                    (nk, nao, nlo - self.ncore), dtype=numpy.complex128
                )
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
            s1_core = numpy.zeros((nk_, nmo_, nmo_), dtype=self.S.dtype)
            s12_core = numpy.zeros(
                (nk_, nmo_, s12_core_.shape[-1]), dtype=s12_core_.dtype
            )
            C_core = numpy.zeros((nk_, self.ncore, self.ncore), dtype=C_core_.dtype)
            for k in range(nk_):
                C_core[k] = C_core_[k].conj().T @ self.S[k] @ C_core_[k]
                s1_core[k] = C_core_[k].conj().T @ self.S[k] @ C_core_[k]
                s12_core[k] = C_core_[k].conj().T @ s12_core_[k]
            ciao_core_ = get_iao_k(C_core, s12_core, s1_core, s2_core, ortho=False)
            ciao_core = numpy.zeros(
                (nk_, nao_, ciao_core_.shape[-1]), dtype=ciao_core_.dtype
            )
            for k in range(nk_):
                ciao_core[k] = C_core_[k] @ ciao_core_[k]
                ciao_core[k] = symm_orth_k(ciao_core[k], ovlp=self.S[k])

            # Begin valence
            s12_val_, s2_val = get_xovlp_k(self.cell, self.kpts, basis=val_basis)
            C_nocore = self.C[:, :, self.ncore :].copy()
            C_nocore_occ_ = C_nocore[:, :, : self.Nocc].copy()
            nk_, nao_, nmo_ = C_nocore.shape
            s1_val = numpy.zeros((nk_, nmo_, nmo_), dtype=self.S.dtype)
            s12_val = numpy.zeros((nk_, nmo_, s12_val_.shape[-1]), dtype=s12_val_.dtype)
            C_nocore_occ = numpy.zeros(
                (nk_, nao_ - self.ncore, C_nocore_occ_.shape[-1]),
                dtype=C_nocore_occ_.dtype,
            )
            for k in range(nk_):
                C_nocore_occ[k] = C_nocore[k].conj().T @ self.S[k] @ C_nocore_occ_[k]
                s1_val[k] = C_nocore[k].conj().T @ self.S[k] @ C_nocore[k]
                s12_val[k] = C_nocore[k].conj().T @ s12_val_[k]
            ciao_val_ = get_iao_k(C_nocore_occ, s12_val, s1_val, s2_val, ortho=False)
            Ciao_ = numpy.zeros((nk_, nao_, ciao_val_.shape[-1]), dtype=ciao_val_.dtype)
            for k in range(nk_):
                Ciao_[k] = C_nocore[k] @ ciao_val_[k]
                Ciao_[k] = symm_orth_k(Ciao_[k], ovlp=self.S[k])

            # stack core|val
            nao = self.S.shape[-1]
            c_core_val = numpy.zeros(
                (nk_, nao, Ciao_.shape[-1] + self.ncore), dtype=Ciao_.dtype
            )
            for k in range(nk_):
                c_core_val[k] = numpy.hstack((ciao_core[k], Ciao_[k]))

            arrange_by_atom = True
            # tmp - aos are not rearrange and so below is not necessary (iaoind_by_atom is used to stack iao|pao later)
            if arrange_by_atom:
                nk, nao, nlo = c_core_val.shape
                for k in range(self.nkpt):
                    aoind_by_atom = get_aoind_by_atom(self.cell)
                    ctmp, iaoind_by_atom = reorder_by_atom_(
                        c_core_val[k], aoind_by_atom, self.S[k]
                    )

            cpao_ = get_pao_native_k(
                c_core_val, self.S, self.cell, valence_basis, self.kpts, ortho=True
            )
            if arrange_by_atom:
                nk, nao, nlo = cpao_.shape
                Cpao_ = numpy.zeros((nk, nao, nlo), dtype=numpy.complex128)
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
                fock_iao = reduce(
                    numpy.dot, (Ciao_[k].conj().T, self.FOCK[k], Ciao_[k])
                )
                S_iao = reduce(numpy.dot, (Ciao_[k].conj().T, self.S[k], Ciao_[k]))
                e_iao, v_iao = scipy.linalg.eigh(fock_iao, S_iao)
                mo_energy_.append(e_iao)
            iaomf = KMF(self.mol, kpts=self.kpts, mo_coeff=Ciao_, mo_energy=mo_energy_)

            num_wann = numpy.asarray(iaomf.mo_coeff).shape[2]
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
            w90 = pywannier90.W90(iaomf, self.kmesh, num_wann, other_keywords=keywords)

            A_matrix = numpy.zeros(
                (self.nkpt, num_wann, num_wann), dtype=numpy.complex128
            )

            i_init = True
            for k in range(self.nkpt):
                if i_init:
                    A_matrix[k] = numpy.eye(num_wann, dtype=numpy.complex128)
                else:
                    ovlp_ciao = uciao[k].conj().T @ self.S[k] @ Ciao[k]
                    A_matrix[k] = ovlp_ciao
            A_matrix = A_matrix.transpose(1, 2, 0)

            w90.kernel(A_matrix=A_matrix)

            u_mat = numpy.array(
                w90.U_matrix.transpose(2, 0, 1), order="C", dtype=numpy.complex128
            )

            os.system("cp wannier90.wout wannier90_iao.wout")
            os.system("rm wannier90.*")

            nk, nao, nlo = Ciao_.shape
            Ciao = numpy.zeros((nk, nao, nlo), dtype=numpy.complex128)

            for k in range(self.nkpt):
                Ciao[k] = numpy.dot(Ciao_[k], u_mat[k])

        # Stack Ciao|Cpao
        Wstack = numpy.zeros(
            (self.nkpt, Ciao.shape[1], Ciao.shape[2] + Cpao.shape[2]),
            dtype=numpy.complex128,
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

                    Wstack[k][:, shift : shift + npao] = Cpao[k][:, paoind_by_atom[ix]]
                    shift += npao
        else:
            for k in range(self.nkpt):
                shift = 0
                for ix in range(self.cell.natm):
                    niao = len(iaoind_by_atom[ix])
                    Wstack[k][:, shift : shift + niao] = Ciao[k][:, iaoind_by_atom[ix]]
                    shift += niao
                    npao = len(paoind_by_atom[ix])
                    Wstack[k][:, shift : shift + npao] = Cpao[k][:, paoind_by_atom[ix]]
                    shift += npao
        self.W = Wstack

        nmo = self.C.shape[2] - self.ncore
        nlo = self.W.shape[2]
        nao = self.S.shape[2]

        lmo_coeff = numpy.zeros((self.nkpt, nlo, nmo), dtype=numpy.complex128)
        cinv_ = numpy.zeros((self.nkpt, nlo, nao), dtype=numpy.complex128)

        if nmo > nlo:
            Co_nocore = self.C[:, :, self.ncore : self.Nocc]
            Cv = self.C[:, :, self.Nocc :]
            # Ensure that the LOs span the occupied space
            for k in range(self.nkpt):
                assert numpy.allclose(
                    numpy.sum((self.W[k].conj().T @ self.S[k] @ Co_nocore[k]) ** 2.0),
                    self.Nocc - self.ncore,
                )
                # Find virtual orbitals that lie in the span of LOs
                u, l, vt = numpy.linalg.svd(
                    self.W[k].conj().T @ self.S[k] @ Cv[k], full_matrices=False
                )
                nvlo = nlo - self.Nocc - self.ncore
                assert numpy.allclose(numpy.sum(l[:nvlo]), nvlo)
                C_ = numpy.hstack([Co_nocore[k], Cv[k] @ vt[:nvlo].conj().T])
                lmo_ = self.W[k].conj().T @ self.S[k] @ C_
                assert numpy.allclose(lmo_.conj().T @ lmo_, numpy.eye(lmo_.shape[1]))
                lmo_coeff.append(lmo_)
        else:
            for k in range(self.nkpt):
                lmo_coeff[k] = reduce(
                    numpy.dot,
                    (self.W[k].conj().T, self.S[k], self.C[k][:, self.ncore :]),
                )
                cinv_[k] = numpy.dot(self.W[k].conj().T, self.S[k])

                assert numpy.allclose(
                    lmo_coeff[k].conj().T @ lmo_coeff[k],
                    numpy.eye(lmo_coeff[k].shape[1]),
                )

        self.lmo_coeff = lmo_coeff
        self.cinv = cinv_

    elif lo_method == "wannier":
        # from pyscf.pbc.tools import pywannier90
        from libdmet.lo import pywannier90

        from .lo_k import remove_core_mo_k

        nk, nao, nmo = self.C.shape
        lorb = numpy.zeros((nk, nao, nmo), dtype=numpy.complex128)
        lorb_nocore = numpy.zeros((nk, nao, nmo - self.ncore), dtype=numpy.complex128)
        for k in range(nk):
            es_, vs_ = scipy.linalg.eigh(self.S[k])
            edx = es_ > 1.0e-14
            lorb[k] = numpy.dot(
                vs_[:, edx] / numpy.sqrt(es_[edx]), vs_[:, edx].conj().T
            )

            if self.frozen_core:
                Ccore = self.C[k][:, : self.ncore]
                lorb_nocore[k] = remove_core_mo_k(lorb[k], Ccore, self.S[k])

        if not self.frozen_core:
            lmf = KMF(self.mol, kpts=self.kpts, mo_coeff=lorb, mo_energy=self.mo_energy)
        else:
            mo_energy_nc = []
            for k in range(nk):
                fock_lnc = reduce(
                    numpy.dot, (lorb_nocore[k].conj().T, self.FOCK[k], lorb_nocore[k])
                )
                S_lnc = reduce(
                    numpy.dot, (lorb_nocore[k].conj().T, self.S[k], lorb_nocore[k])
                )
                e__, v__ = scipy.linalg.eigh(fock_lnc, S_lnc)
                mo_energy_nc.append(e__)
            lmf = KMF(
                self.mol, kpts=self.kpts, mo_coeff=lorb_nocore, mo_energy=mo_energy_nc
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
        A_matrix = numpy.zeros((self.nkpt, num_wann, num_wann), dtype=numpy.complex128)
        i_init = (
            True  # Using A=I + lowdin orbital and A=<psi|lowdin> + |psi> is the same
        )
        for k in range(self.nkpt):
            if i_init:
                A_matrix[k] = numpy.eye(num_wann, dtype=numpy.complex128)

        A_matrix = A_matrix.transpose(1, 2, 0)

        w90.kernel(A_matrix=A_matrix)
        u_mat = numpy.array(
            w90.U_matrix.transpose(2, 0, 1), order="C", dtype=numpy.complex128
        )

        nk, nao, nlo = lmf.mo_coeff.shape
        W = numpy.zeros((nk, nao, nlo), dtype=numpy.complex128)
        for k in range(nk):
            W[k] = numpy.dot(lmf.mo_coeff[k], u_mat[k])

        self.W = W
        lmo_coeff = numpy.zeros(
            (self.nkpt, nlo, nmo - self.ncore), dtype=numpy.complex128
        )
        cinv_ = numpy.zeros((self.nkpt, nlo, nao), dtype=numpy.complex128)

        for k in range(nk):
            lmo_coeff[k] = reduce(
                numpy.dot, (self.W[k].conj().T, self.S[k], self.C[k][:, self.ncore :])
            )
            cinv_[k] = numpy.dot(self.W[k].conj().T, self.S[k])
            assert numpy.allclose(
                lmo_coeff[k].conj().T @ lmo_coeff[k], numpy.eye(lmo_coeff[k].shape[1])
            )
        self.lmo_coeff = lmo_coeff
        self.cinv = cinv_

    else:
        print("lo_method = ", lo_method, " not implemented!", flush=True)
        print("exiting", flush=True)
        sys.exit()
