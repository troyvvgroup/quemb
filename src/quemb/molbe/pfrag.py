# Author(s): Oinam Romesh Meitei

import h5py
import numpy as np
import scipy.linalg
from numpy import (
    argsort,
    array,
    diag_indices,
    einsum,
    eye,
    float64,
    outer,
    trace,
    tril_indices,
    zeros,
    zeros_like,
)
from numpy.linalg import eigh, multi_dot

from quemb.molbe.helper import get_eri, get_scfObj, get_veff
from quemb.shared.typing import Matrix


class Frags:
    """
    Class for handling fragments in bootstrap embedding.

    This class contains various functionalities required for managing and manipulating
    fragments for BE calculations.
    """

    def __init__(
        self,
        AO_per_frag,
        ifrag: int,
        AO_per_edge=None,
        ref_frag_idx_per_edge=None,
        relAO_per_edge=None,
        relAO_in_ref_per_edge=None,
        centerweight_and_relAO_per_center=None,
        eri_file="eri_file.h5",
        centerf_idx=None,
        unrestricted=False,
    ):
        """Constructor function for :python:`Frags` class.

        Parameters
        ----------
        AO_per_frag : list
            list of AOs in the fragment (i.e. ``BE.AO_per_frag[i]``
            or ``FragPart.AO_per_frag[i]``)
        ifrag : int
            fragment index (∈ [0, BE.n_frag - 1])
        edge : list, optional
            list of lists of edge site AOs for each atom in the fragment,
            by default None
        ref_frag_idx_per_edge : list, optional
            list of fragment indices where edge site AOs are center site,
            by default None
        relAO_per_edge : list, optional
            list of lists of indices for edge site AOs within the fragment,
            by default None
        relAO_in_ref_per_edge: list, optional
            list of lists of indices within the fragment specified in :python:`center`
            that points to the edge site AOs , by default None
        centerweight_and_relAO_per_center : list, optional
            weight used for energy contributions and the indices, by default None
        eri_file : str, optional
            two-electron integrals stored as h5py file, by default 'eri_file.h5'
        centerf_idx : list, optional
            indices of the center site atoms in the fragment, by default None
        unrestricted : bool, optional
            unrestricted calculation, by default False
        """

        self.AO_per_frag = AO_per_frag
        self.n_frag = len(AO_per_frag)
        self.TA: Matrix[float64] | None = None
        self.TA_lo_eo: Matrix[float64] | None = None
        self.h1 = None
        self.ifrag = ifrag
        if unrestricted:
            self.dname: str | list[str] = [
                "f" + str(ifrag) + "/aa",
                "f" + str(ifrag) + "/bb",
                "f" + str(ifrag) + "/ab",
            ]
        else:
            self.dname = "f" + str(ifrag)
        self.nao = None
        self.mo_coeffs = None
        self._mo_coeffs = None
        self.nsocc = None
        self._mf = None
        self._mc = None

        # CCSD
        self.t1 = None
        self.t2 = None

        self.heff: Matrix[float64] | None = None
        self.AO_per_edge = AO_per_edge
        self.ref_frag_idx_per_edge = ref_frag_idx_per_edge
        self.relAO_per_edge = relAO_per_edge
        self.relAO_in_ref_per_edge = relAO_in_ref_per_edge
        self.centerf_idx = centerf_idx
        self.udim: int | None = None

        self._rdm1 = None
        self.rdm1__ = None
        self.rdm2__ = None
        self.rdm1 = None
        self.genvs = None
        self.ebe = 0.0
        self.ebe_hf = 0.0
        self.centerweight_and_relAO_per_center = centerweight_and_relAO_per_center
        self.fock = None
        self.veff = None
        self.veff0 = None
        self.dm_init = None
        self.dm0 = None
        self.eri_file = eri_file
        self.unitcell_nkpt = 1.0

    def sd(self, lao, lmo, nocc, thr_bath, norb=None, return_orb_count=False):
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
        thr_bath : float,
            Threshold for bath orbitals in Schmidt decomposition
        norb : int, optional
            Specify number of bath orbitals.
            Used for UBE, where different number of alpha and beta orbitals
            Default is None, allowing orbitals to be chosen by threshold
        return_orb_count : bool, optional
            Retrun the number of orbitals in each space, for UBE use/
            Default is False
        """

        if return_orb_count:
            TA, n_f, n_b = schmidt_decomposition(
                lmo,
                nocc,
                self.AO_per_frag,
                thr_bath=thr_bath,
                norb=norb,
                return_orb_count=return_orb_count,
            )
        else:
            TA = schmidt_decomposition(lmo, nocc, self.AO_per_frag, thr_bath=thr_bath)
        self.C_lo_eo = TA
        TA = lao @ TA
        self.nao = TA.shape[1]
        self.TA = TA
        if return_orb_count:
            return [n_f, n_b]

    def cons_h1(self, h1):
        """
        Construct the one-electron Hamiltonian for the fragment.

        Parameters
        ----------
        h1 : numpy.ndarray
            One-electron Hamiltonian matrix.
        """

        h1_tmp = multi_dot((self.TA.T, h1, self.TA))
        self.h1 = h1_tmp

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

        veff_, veff0 = get_veff(eri_, dm, S, self.TA, hf_veff)
        self.veff = veff_.real
        self.veff0 = veff0
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
        C_ = multi_dot((self.TA.T, S, C[:, ncore : ncore + nocc]))
        P_ = C_ @ C_.T
        nsocc_ = trace(P_)
        nsocc = int(round(nsocc_))
        try:
            mo_coeffs = scipy.linalg.svd(C_)[0]
        except scipy.linalg.LinAlgError:
            mo_coeffs = scipy.linalg.eigh(C_)[1][:, -nsocc:]

        self._mo_coeffs = mo_coeffs
        self.nsocc = nsocc
        return P_

    def scf(
        self, heff=None, fs=False, eri=None, dm0=None, unrestricted=False, spin_ind=None
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
        unrestricted : bool, optional
            Specify if unrestricted calculation, by default False
        spin_ind : int, optional
            Alpha (0) or beta (1) spin for unrestricted calculation, by default None
        """

        if self._mf is not None:
            self._mf = None
        if self._mc is not None:
            self._mc = None
        if heff is None:
            heff = self.heff

        if eri is None:
            if unrestricted:
                dname = self.dname[spin_ind]
            else:
                dname = self.dname
            eri = get_eri(dname, self.nao, eri_file=self.eri_file)

        if dm0 is None:
            dm0 = 2.0 * (
                self._mo_coeffs[:, : self.nsocc]
                @ self._mo_coeffs[:, : self.nsocc].conj().T
            )

        mf_ = get_scfObj(self.fock + heff, eri, self.nsocc, dm0=dm0)
        if not fs:
            self._mf = mf_
            self.mo_coeffs = mf_.mo_coeff.copy()
        else:
            self._mo_coeffs = mf_.mo_coeff.copy()
        mf_ = None

    def update_heff(self, u, cout=None, only_chem=False):
        """Update the effective Hamiltonian for the fragment."""
        heff_ = zeros_like(self.h1)

        if cout is None:
            cout = self.udim

        for i, fi in enumerate(self.AO_per_frag):
            if not any(i in sublist for sublist in self.relAO_per_edge):
                heff_[i, i] -= u[-1]

        if only_chem:
            self.heff = heff_
            return
        else:
            for i in self.relAO_per_edge:
                for j in range(len(i)):
                    for k in range(len(i)):
                        if j > k:  # or j==k:
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
        self,
        rdm_hf=None,
        mo_coeffs=None,
        eri=None,
        return_e=False,
        unrestricted=False,
        spin_ind=None,
    ):
        if mo_coeffs is None:
            mo_coeffs = self._mo_coeffs

        if rdm_hf is None:
            rdm_hf = mo_coeffs[:, : self.nsocc] @ mo_coeffs[:, : self.nsocc].conj().T

        unrestricted_fac = 1.0 if unrestricted else 2.0

        e1 = unrestricted_fac * einsum(
            "ij,ij->i", self.h1[: self.n_frag], rdm_hf[: self.n_frag]
        )

        ec = (
            0.5
            * unrestricted_fac
            * einsum("ij,ij->i", self.veff[: self.n_frag], rdm_hf[: self.n_frag])
        )

        if self.TA.ndim == 3:
            jmax = self.TA[0].shape[1]
        else:
            jmax = self.TA.shape[1]
        if eri is None:
            with h5py.File(self.eri_file, "r") as r:
                if isinstance(self.dname, list):
                    eri = [r[self.dname[0]][()], r[self.dname[1]][()]]
                else:
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
                if (
                    unrestricted
                ):  # unrestricted ERI file has 3 spin components: a, b, ab
                    e2[i] += (
                        0.5
                        * unrestricted_fac
                        * Gij[tril_indices(jmax)]
                        @ eri[spin_ind][ij]
                    )
                else:
                    e2[i] += 0.5 * unrestricted_fac * Gij[tril_indices(jmax)] @ eri[ij]

        e_ = e1 + e2 + ec
        etmp = 0.0
        for i in self.centerweight_and_relAO_per_center[1]:
            etmp += self.centerweight_and_relAO_per_center[0] * e_[i]

        self.ebe_hf = etmp

        if return_e:
            e_h1 = 0.0
            e_coul = 0.0
            for i in self.centerweight_and_relAO_per_center[1]:
                e_h1 += self.centerweight_and_relAO_per_center[0] * e1[i]
                e_coul += self.centerweight_and_relAO_per_center[0] * (e2[i] + ec[i])
            return (e_h1, e_coul, e1 + e2 + ec)
        else:
            return None


def schmidt_decomposition(
    mo_coeff,
    nocc,
    Frag_sites,
    thr_bath=1.0e-10,
    cinv=None,
    rdm=None,
    norb=None,
    return_orb_count=False,
):
    """
    Perform Schmidt decomposition on the molecular orbital coefficients.

    This function decomposes the molecular orbitals into fragment and environment parts
    using the Schmidt decomposition method. It computes the transformation matrix (TA)
    which includes both the fragment orbitals and the entangled bath.

    Parameters
    ----------
    mo_coeff : numpy.ndarray
        Molecular orbital coefficients.
    nocc : int
        Number of occupied orbitals.
    Frag_sites : list of int
        List of fragment sites (indices).
    thr_bath : float,
        Threshold for bath orbitals in Schmidt decomposition
    cinv : numpy.ndarray, optional
        Inverse of the transformation matrix. Defaults to None.
    rdm : numpy.ndarray, optional
        Reduced density matrix. If not provided, it will be computed from the molecular
        orbitals. Defaults to None.
    norb : int, optional
        Specifies number of bath orbitals. Used for UBE to make alpha and beta
        spaces the same size. Defaults to None
    return_orb_count : bool, optional
        Return more information about the number of orbitals. Used in UBE.
        Defaults to False

    Returns
    -------
    numpy.ndarray
        Transformation matrix (TA) including both fragment and entangled bath orbitals.
    if return_orb_count:
        numpy.ndarray, int, int
        returns TA (above), number of orbitals in the fragment space, and number of
        orbitals in bath space
    """

    # Compute the reduced density matrix (RDM) if not provided
    if mo_coeff is not None:
        C = mo_coeff[:, :nocc]
    if rdm is None:
        Dhf = C @ C.T
        if cinv is not None:
            Dhf = multi_dot((cinv, Dhf, cinv.conj().T))
    else:
        Dhf = rdm

    # Total number of sites
    Tot_sites = Dhf.shape[0]

    # Identify environment sites (indices not in Frag_sites)
    Env_sites1 = array([i for i in range(Tot_sites) if i not in Frag_sites])
    Env_sites = array([[i] for i in range(Tot_sites) if i not in Frag_sites])
    Frag_sites1 = array([[i] for i in Frag_sites])

    # Compute the environment part of the density matrix
    Denv = Dhf[Env_sites, Env_sites.T]

    # Perform eigenvalue decomposition on the environment density matrix
    Eval, Evec = eigh(Denv)

    # Identify significant environment orbitals based on eigenvalue threshold
    Bidx = []

    # Set the number of orbitals to be taken from the environment orbitals
    # Based on an eigenvalue threshold ordering
    if norb is not None:
        n_frag_ind = len(Frag_sites1)
        n_bath_ind = norb - n_frag_ind
        ind_sort = argsort(np.abs(Eval))
        first_el = [x for x in ind_sort if x < 1.0 - thr_bath][-1 * n_bath_ind]
        for i in range(len(Eval)):
            if np.abs(Eval[i]) >= first_el:
                Bidx.append(i)
    else:
        for i in range(len(Eval)):
            if thr_bath < np.abs(Eval[i]) < 1.0 - thr_bath:
                Bidx.append(i)

    # Initialize the transformation matrix (TA)
    TA = zeros([Tot_sites, len(Frag_sites) + len(Bidx)])
    TA[Frag_sites, : len(Frag_sites)] = eye(len(Frag_sites))  # Fragment part
    TA[Env_sites1, len(Frag_sites) :] = Evec[:, Bidx]  # Environment part

    if return_orb_count:
        # return TA, norbs_frag, norbs_bath
        return TA, Frag_sites1.shape[0], len(Bidx)
    else:
        return TA
