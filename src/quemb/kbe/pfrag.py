# Author(s): Oinam Romesh Meitei

from collections.abc import Sequence

import h5py
import numpy as np
from numpy import (
    abs,
    complex128,
    diag_indices,
    einsum,
    float64,
    outer,
    result_type,
    trace,
    tril_indices,
    zeros,
    zeros_like,
)
from numpy.linalg import multi_dot

from quemb.kbe.helper import get_veff
from quemb.kbe.misc import get_phase, get_phase1
from quemb.kbe.solver import schmidt_decomp_svd
from quemb.molbe.helper import get_eri, get_scfObj
from quemb.shared.helper import unused
from quemb.shared.typing import (
    FragmentIdx,
    GlobalAOIdx,
    Matrix,
    PathLike,
    RelAOIdx,
    RelAOIdxInRef,
    SeqOverEdge,
    Tensor3D,
)


class Frags:
    """
    Class for handling fragments in periodic bootstrap embedding.

    This class contains various functionalities required for managing and manipulating
    fragments for periodic BE calculations.
    """

    def __init__(
        self,
        *,
        AO_in_frag: Sequence[GlobalAOIdx],
        ifrag: int,
        AO_per_edge: SeqOverEdge[Sequence[GlobalAOIdx]],
        ref_frag_idx_per_edge: SeqOverEdge[FragmentIdx],
        relAO_per_edge: SeqOverEdge[Sequence[RelAOIdx]],
        relAO_in_ref_per_edge: SeqOverEdge[Sequence[RelAOIdxInRef]],
        centerweight_and_relAO_per_center: tuple[float, Sequence[RelAOIdx]],
        relAO_per_origin: Sequence[RelAOIdx],
        eri_file: PathLike,
        unitcell_nkpt: int,
        unitcell: int,
    ) -> None:
        """Constructor function for :python:`Frags` class.

        Parameters
        ----------
        AO_per_frag: list
            list of AOs in the fragment (i.e. pbe.AO_per_frag[i]
            or FragPart.AO_per_frag[i])
            Read more detailed description in :class:`quemb.molbe.autofrag.FragPart`.
        ifrag : int
            fragment index (∈ [0, pbe.n_frag - 1])
        AO_per_edge : list, optional
            list of lists of edge site AOs for each atom in the fragment,
            by default None
            Read more detailed description in :class:`quemb.molbe.autofrag.FragPart`.
        ref_frag_idx_per_edge: list, optional
            list of fragment indices where edge site AOs are center site,
            by default None.
            Read more detailed description in :class:`quemb.molbe.autofrag.FragPart`.
        rel_AO_per_edge_per_frag: list, optional
            list of lists of indices for edge site AOs within the fragment,
            by default None
            Read more detailed description in :class:`quemb.molbe.autofrag.FragPart`.
        relAO_in_ref_per_edge :
            list of lists of indices within the fragment specified
            in :python:`center` that points to the edge site AOs,
            by default :python:`None`
            Read more detailed description in :class:`quemb.molbe.autofrag.FragPart`.
        centerweight_and_relAO_per_center :
            weight used for energy contributions, by default None
        eri_file : str, optional
            two-electron integrals stored as h5py file, by default 'eri_file.h5'
        relAO_per_origin : list, optional
            indices of the origin in the fragment, by default None
        """

        self.AO_in_frag = AO_in_frag
        self.unitcell = unitcell
        self.unitcell_nkpt = unitcell_nkpt
        self.n_frag = len(AO_in_frag)
        self.dname = "f" + str(ifrag)
        self.AO_per_edge = AO_per_edge
        self.ref_frag_idx_per_edge = ref_frag_idx_per_edge
        self.relAO_per_edge = relAO_per_edge
        self.relAO_in_ref_per_edge = relAO_in_ref_per_edge
        self.relAO_per_origin = relAO_per_origin
        self.centerweight_and_relAO_per_center = centerweight_and_relAO_per_center
        self.ifrag = ifrag

        self.TA: Matrix[float64]
        self.TA_lo_eo: Matrix[float64]
        self.h1 = None
        self.nao: int
        self.mo_coeffs: Matrix[float64]
        self._mo_coeffs: Matrix[float64]
        self.nsocc: int
        self._mf = None
        self._mc = None

        # CCSD
        self.t1 = None
        self.t2 = None

        self.heff: Matrix[float64]
        self.udim: int

        self._rdm1 = None
        self.rdm1__ = None
        self.rdm2__ = None
        self._del_rdm1 = None
        self.rdm1 = None
        self.genvs = None
        self.ebe = 0.0
        self.ebe_hf = 0.0
        self.fock: Matrix[float64]
        self.veff = None
        self.veff0 = None
        self.dm_init = None
        self.dm0: Matrix[float64]
        self.eri_file = eri_file
        self.pot = None
        self.ebe_hf0 = 0.0
        self.rdm1_lo_k: Tensor3D[float64]

    def sd(
        self,
        lao,
        lmo,
        nocc,
        thr_bath,
        cell=None,
        kpts=None,
        kmesh=None,
        h1=None,
    ) -> None:
        """
        Perform Schmidt decomposition for the fragment.

        Parameters
        ----------
        lao : numpy.ndarray
            Orthogonalized AOs
        lmo : numpy.ndarray
            Local molecular orbital coefficients.
        nocc : int
            Number of occupied orbitals.
        cell : pyscf.pbc.gto.cell.Cell
            PySCF pbc.gto.cell.Cell object defining the unit cell and lattice vectors.
        kpts : list of list of float
            k-points in the reciprocal space for periodic computations
        kmesh : list of int
            Number of k-points in each lattice vector direction
        """
        nk, nao, nlo = lao.shape
        rdm1_lo_k = zeros((nk, nlo, nlo), dtype=result_type(lmo, lmo))
        for k in range(nk):
            rdm1_lo_k[k] += lmo[k][:, :nocc] @ lmo[k][:, :nocc].conj().T
        self.rdm1_lo_k = rdm1_lo_k
        phase = get_phase(cell, kpts, kmesh)
        supcell_rdm = einsum("Rk,kuv,Sk->RuSv", phase, rdm1_lo_k, phase.conj())
        supcell_rdm = supcell_rdm.reshape(nk * nlo, nk * nlo)

        if (max_val := np.abs(supcell_rdm.imag).max()) < 1.0e-6:
            supcell_rdm = supcell_rdm.real
        else:
            raise ValueError(f"Imaginary density in Full SD {max_val}")

        Sites = [i + (nlo * 0) for i in self.AO_in_frag]

        TA_R = schmidt_decomp_svd(supcell_rdm, Sites, thr_bath=thr_bath)
        teo = TA_R.shape[-1]
        TA_R = TA_R.reshape(nk, nlo, teo)

        phase1 = get_phase1(cell, kpts, kmesh)
        TA_k = einsum("Rim, Rk -> kim", TA_R, phase1)
        self.TA_lo_eo = TA_k

        TA_ao_eo_k = zeros((nk, nao, teo), dtype=result_type(lao.dtype, TA_k.dtype))
        for k in range(nk):
            TA_ao_eo_k[k] = lao[k] @ TA_k[k]

        self.TA = TA_ao_eo_k
        self.nao = TA_ao_eo_k.shape[-1]

        # useful for debugging --
        rdm1_eo = zeros((teo, teo), dtype=complex128)
        for k in range(nk):
            rdm1_eo += multi_dot((TA_k[k].conj().T, rdm1_lo_k[k], TA_k[k]))
        rdm1_eo /= float(nk)

        h1_eo = zeros((teo, teo), dtype=complex128)
        for k in range(nk):
            h1_eo += multi_dot((self.TA[k].conj().T, h1[k], self.TA[k]))
        h1_eo /= float(nk)
        e1 = 2.0 * einsum("ij,ij->i", h1_eo[: self.n_frag], rdm1_eo[: self.n_frag])
        e_h1 = 0.0
        for i in self.centerweight_and_relAO_per_center[1]:
            e_h1 += self.centerweight_and_relAO_per_center[0] * e1[i]

    def cons_h1(self, h1):
        """
        Construct the one-electron Hamiltonian for the fragment.

        Parameters
        ----------
        h1 : numpy.ndarray
            One-electron Hamiltonian matrix.
        """

        nk, nao, teo = self.TA.shape
        unused(nao)
        h1_eo = zeros((teo, teo), dtype=complex128)
        for k in range(nk):
            h1_eo += multi_dot((self.TA[k].conj().T, h1[k], self.TA[k]))
        h1_eo /= float(nk)

        if np.abs(h1_eo.imag).max() < 1.0e-7:
            self.h1 = h1_eo.real
        else:
            raise ValueError(f"Imaginary Hcore {np.abs(h1_eo.imag).max()}")

    def cons_fock(self, hf_veff, S, dm, eri_=None):
        """
        Construct the Fock matrix for the fragment.

        Parameters
        ----------
        hf_veff : numpy.ndarray
            Hartree-Fock effective potential.
        S : numpy.ndarray
            Overlap matrix.
        dm : numpy.ndarray
            Density matrix.
        eri_ : numpy.ndarray, optional
            Electron repulsion integrals, by default None.
        """

        if eri_ is None:
            eri_ = get_eri(
                self.dname, self.TA.shape[1], ignore_symm=True, eri_file=self.eri_file
            )

        veff0, veff_ = get_veff(eri_, dm, S, self.TA, hf_veff, return_veff0=True)
        if np.abs(veff_.imag).max() < 1.0e-6:
            self.veff = veff_.real
            self.veff0 = veff0.real
        else:
            raise ValueError(f"Imaginary Veff {abs(veff_.imag).max()}")

        self.fock = self.h1 + veff_.real

    def get_nsocc(self, S, C, nocc, ncore=0):
        """
        Get the number of occupied orbitals for the fragment.

        Parameters
        ----------
        S : numpy.ndarray
            Overlap matrix.
        C : numpy.ndarray
            Molecular orbital coefficients.
        nocc : int
            Number of occupied orbitals.
        ncore : int, optional
            Number of core orbitals, by default 0.

        Returns
        -------
        numpy.ndarray
            Projected density matrix.
        """

        nk, nao, neo = self.TA.shape
        dm_ = zeros((nk, nao, nao), dtype=result_type(C, C))
        for k in range(nk):
            dm_[k] = 2.0 * (
                C[k][:, ncore : ncore + nocc] @ C[k][:, ncore : ncore + nocc].conj().T
            )
        P_ = zeros((neo, neo), dtype=complex128)
        for k in range(nk):
            Cinv = self.TA[k].conj().T @ S[k]
            P_ += multi_dot((Cinv, dm_[k], Cinv.conj().T))

        P_ /= float(nk)
        if np.abs(P_.imag).max() < 1.0e-6:
            P_ = P_.real
        else:
            raise ValueError(f"Imaginary density in get_nsocc {abs(P_.imag).max()}")

        nsocc_ = trace(P_)
        nsocc = int(round(nsocc_.real) / 2)

        self.nsocc = nsocc
        return P_

    def scf(
        self,
        heff=None,
        fs=False,
        eri=None,
        dm0=None,
    ):
        """
        Perform self-consistent field (SCF) calculation for the fragment.

        Parameters
        ----------
        heff : numpy.ndarray, optional
            Effective Hamiltonian, by default None.
        fs : bool, optional
            Flag for full SCF, by default False.
        eri : numpy.ndarray, optional
            Electron repulsion integrals, by default None.
        dm0 : numpy.ndarray, optional
            Initial density matrix, by default None.
        """

        if self._mf is not None:
            self._mf = None
        if self._mc is not None:
            self._mc = None
        if heff is None:
            heff = self.heff

        if eri is None:
            eri = get_eri(self.dname, self.nao, eri_file=self.eri_file)

        if dm0 is None:
            dm0 = 2.0 * (
                self._mo_coeffs[:, : self.nsocc]
                @ self._mo_coeffs[:, : self.nsocc].conj().T
            )

        mf_ = get_scfObj(
            self.fock + heff,
            eri,
            self.nsocc,
            dm0=dm0,
        )

        if not fs:
            self._mf = mf_
            self.mo_coeffs = mf_.mo_coeff.copy()
        else:
            self._mo_coeffs = mf_.mo_coeff.copy()

            dm0 = mf_.make_rdm1()
        mf_ = None

    def update_heff(
        self,
        u,
        cout=None,
        do_chempot=True,
        only_chem=False,
    ):
        """Update the effective Hamiltonian for the fragment."""
        heff_ = zeros_like(self.h1)

        if cout is None:
            cout = self.udim

        if do_chempot:
            for i, fi in enumerate(self.AO_in_frag):
                if not any(i in sublist for sublist in self.relAO_per_edge):
                    heff_[i, i] -= u[-1]

        if only_chem:
            self.heff = heff_
        else:
            for idx, i in enumerate(self.relAO_per_edge):
                for j in range(len(i)):
                    for k in range(len(i)):
                        if j > k:
                            continue
                        heff_[i[j], i[k]] = u[cout]
                        heff_[i[k], i[j]] = u[cout]
                        cout += 1

            self.heff = heff_

    def set_udim(self, cout):
        for i in self.relAO_per_edge:
            for j in range(len(i)):
                for k in range(len(i)):
                    if j > k:
                        continue
                    cout += 1
        return cout

    def update_ebe_hf(
        self, rdm_hf=None, mo_coeffs=None, eri=None, return_e1=False, unrestricted=False
    ):
        if mo_coeffs is None:
            mo_coeffs = self._mo_coeffs

        if rdm_hf is None:
            rdm_hf = mo_coeffs[:, : self.nsocc] @ mo_coeffs[:, : self.nsocc].conj().T

        unrestricted = 1.0 if unrestricted else 2.0

        e1 = unrestricted * einsum(
            "ij,ij->i", self.h1[: self.n_frag], rdm_hf[: self.n_frag]
        )

        ec = (
            0.5
            * unrestricted
            * einsum("ij,ij->i", self.veff[: self.n_frag], rdm_hf[: self.n_frag])
        )

        if self.TA.ndim == 3:
            jmax = self.TA[0].shape[1]
        else:
            jmax = self.TA.shape[1]
        if eri is None:
            with h5py.File(self.eri_file, "r") as r:
                eri = r[self.dname][()]

        e2 = zeros_like(e1)
        for i in range(self.n_frag):
            for j in range(jmax):
                ij = i * (i + 1) // 2 + j if i > j else j * (j + 1) // 2 + i
                Gij = (2.0 * rdm_hf[i, j] * rdm_hf - outer(rdm_hf[i], rdm_hf[j]))[
                    :jmax, :jmax
                ]
                Gij[diag_indices(jmax)] *= 0.5
                Gij += Gij.T
                e2[i] += 0.5 * unrestricted * Gij[tril_indices(jmax)] @ eri[ij]

        e_ = e1 + e2 + ec
        etmp = 0.0
        e1_ = 0.0
        e2_ = 0.0
        ec_ = 0.0
        for i in self.centerweight_and_relAO_per_center[1]:
            etmp += self.centerweight_and_relAO_per_center[0] * e_[i]
            e1_ += self.centerweight_and_relAO_per_center[0] * e1[i]
            e2_ += self.centerweight_and_relAO_per_center[0] * e2[i]
            ec_ += self.centerweight_and_relAO_per_center[0] * ec[i]

        self.ebe_hf = etmp
        if return_e1:
            e_h1 = 0.0
            e_coul = 0.0
            for i in self.centerweight_and_relAO_per_center[1]:
                e_h1 += self.centerweight_and_relAO_per_center[0] * e1[i]
                e_coul += self.centerweight_and_relAO_per_center[0] * (e2[i] + ec[i])
            return (e_h1, e_coul, e1 + e2 + ec)

        return e1 + e2 + ec
