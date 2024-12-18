# Author(s): Oinam Romesh Meitei

import h5py
import numpy
import scipy.linalg
from numpy import float64
from numpy.linalg import multi_dot

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
        fsites,
        ifrag: int,
        edge=None,
        center=None,
        edge_idx=None,
        center_idx=None,
        efac=None,
        eri_file="eri_file.h5",
        centerf_idx=None,
    ):
        """Constructor function for :python:`Frags` class.

        Parameters
        ----------
        fsites : list
            list of AOs in the fragment (i.e. BE.fsites[i] or fragpart.fsites[i])
        ifrag :
            fragment index (∈ [0, BE.Nfrag])
        edge : list, optional
            list of lists of edge site AOs for each atom in the fragment,
            by default None
        center : list, optional
            list of fragment indices where edge site AOs are center site,
            by default None
        edge_idx : list, optional
            list of lists of indices for edge site AOs within the fragment,
            by default None
        center_idx : list, optional
            list of lists of indices within the fragment specified in :python:`center`
            that points to the edge site AOs , by default None
        efac : list, optional
            weight used for energy contributions, by default None
        eri_file : str, optional
            two-electron integrals stored as h5py file, by default 'eri_file.h5'
        centerf_idx : list, optional
            indices of the center site atoms in the fragment, by default None
        unrestricted : bool, optional
            unrestricted calculation, by default False
        """

        self.fsites = fsites
        self.nfsites = len(fsites)
        self.TA: Matrix[float64] | None = None
        self.TA_lo_eo = None
        self.h1 = None
        self.ifrag = ifrag
        self.dname = f"f{ifrag}"
        self.nao = None
        self.mo_coeffs = None
        self._mo_coeffs = None
        self.nsocc = None
        self._mf = None
        self._mc = None

        # CCSD
        self.t1 = None
        self.t2 = None

        self.heff = None
        self.edge = edge
        self.center = center
        self.edge_idx = edge_idx
        self.center_idx = center_idx
        self.centerf_idx = centerf_idx
        self.udim = None

        self._rdm1 = None
        self.rdm1__ = None
        self.rdm2__ = None
        self.rdm1 = None
        self.genvs = None
        self.ebe = 0.0
        self.ebe_hf = 0.0
        self.efac = efac
        self.fock = None
        self.veff = None
        self.veff0 = None
        self.dm_init = None
        self.dm0 = None
        self.eri_file = eri_file
        self.unitcell_nkpt = 1.0

    def sd(self, lao, lmo, nocc, norb=None, return_orb_count=False):
        """Perform Schmidt decomposition for the fragment.

        Parameters
        ----------
        lao : numpy.ndarray
            Orthogonalized AOs
        lmo : numpy.ndarray
            Local molecular orbital coefficients.
        nocc : int
            Number of occupied orbitals.
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
                lmo, nocc, self.fsites, norb=norb, return_orb_count=return_orb_count
            )
        else:
            TA = schmidt_decomposition(lmo, nocc, self.fsites)
        self.C_lo_eo = TA
        TA = numpy.dot(lao, TA)
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

        veff_ = get_veff(eri_, dm, S, self.TA, hf_veff)
        self.veff = veff_.real
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
        P_ = numpy.dot(C_, C_.T)
        nsocc_ = numpy.trace(P_)
        nsocc = int(numpy.round(nsocc_))
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

        self._mf = None
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
            dm0 = (
                numpy.dot(
                    self._mo_coeffs[:, : self.nsocc],
                    self._mo_coeffs[:, : self.nsocc].conj().T,
                )
                * 2.0
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
        heff_ = numpy.zeros_like(self.h1)

        if cout is None:
            cout = self.udim

        for i, fi in enumerate(self.fsites):
            if not any(i in sublist for sublist in self.edge_idx):
                heff_[i, i] -= u[-1]

        if only_chem:
            self.heff = heff_
            return
        else:
            for i in self.edge_idx:
                for j in range(len(i)):
                    for k in range(len(i)):
                        if j > k:  # or j==k:
                            continue

                        heff_[i[j], i[k]] = u[cout]
                        heff_[i[k], i[j]] = u[cout]

                        cout += 1

            self.heff = heff_

    def set_udim(self, cout):
        for i in self.edge_idx:
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
            rdm_hf = numpy.dot(
                mo_coeffs[:, : self.nsocc], mo_coeffs[:, : self.nsocc].conj().T
            )

        unrestricted_fac = 1.0 if unrestricted else 2.0

        e1 = unrestricted_fac * numpy.einsum(
            "ij,ij->i", self.h1[: self.nfsites], rdm_hf[: self.nfsites]
        )

        ec = (
            0.5
            * unrestricted_fac
            * numpy.einsum(
                "ij,ij->i", self.veff[: self.nfsites], rdm_hf[: self.nfsites]
            )
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

        e2 = numpy.zeros_like(e1)
        for i in range(self.nfsites):
            for j in range(jmax):
                ij = i * (i + 1) // 2 + j if i > j else j * (j + 1) // 2 + i
                Gij = (2.0 * rdm_hf[i, j] * rdm_hf - numpy.outer(rdm_hf[i], rdm_hf[j]))[
                    :jmax, :jmax
                ]
                Gij[numpy.diag_indices(jmax)] *= 0.5
                Gij += Gij.T
                if (
                    unrestricted
                ):  # unrestricted ERI file has 3 spin components: a, b, ab
                    e2[i] += (
                        0.5
                        * unrestricted_fac
                        * Gij[numpy.tril_indices(jmax)]
                        @ eri[spin_ind][ij]
                    )
                else:
                    e2[i] += (
                        0.5 * unrestricted_fac * Gij[numpy.tril_indices(jmax)] @ eri[ij]
                    )

        e_ = e1 + e2 + ec
        etmp = 0.0
        for i in self.efac[1]:
            etmp += self.efac[0] * e_[i]

        self.ebe_hf = etmp

        if return_e:
            e_h1 = 0.0
            e_coul = 0.0
            for i in self.efac[1]:
                e_h1 += self.efac[0] * e1[i]
                e_coul += self.efac[0] * (e2[i] + ec[i])
            return (e_h1, e_coul, e1 + e2 + ec)
        else:
            return None


def schmidt_decomposition(
    mo_coeff, nocc, Frag_sites, cinv=None, rdm=None, norb=None, return_orb_count=False
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
    # Threshold for eigenvalue significance
    thres = 1.0e-10

    # Compute the reduced density matrix (RDM) if not provided
    if mo_coeff is not None:
        C = mo_coeff[:, :nocc]
    if rdm is None:
        Dhf = numpy.dot(C, C.T)
        if cinv is not None:
            Dhf = multi_dot((cinv, Dhf, cinv.conj().T))
    else:
        Dhf = rdm

    # Total number of sites
    Tot_sites = Dhf.shape[0]

    # Identify environment sites (indices not in Frag_sites)
    Env_sites1 = numpy.array([i for i in range(Tot_sites) if i not in Frag_sites])
    Env_sites = numpy.array([[i] for i in range(Tot_sites) if i not in Frag_sites])
    Frag_sites1 = numpy.array([[i] for i in Frag_sites])

    # Compute the environment part of the density matrix
    Denv = Dhf[Env_sites, Env_sites.T]

    # Perform eigenvalue decomposition on the environment density matrix
    Eval, Evec = numpy.linalg.eigh(Denv)

    # Identify significant environment orbitals based on eigenvalue threshold
    Bidx = []

    # Set the number of orbitals to be taken from the environment orbitals
    # Based on an eigenvalue threshold ordering
    if norb is not None:
        n_frag_ind = len(Frag_sites1)
        n_bath_ind = norb - n_frag_ind
        ind_sort = numpy.argsort(numpy.abs(Eval))
        first_el = [x for x in ind_sort if x < 1.0 - thres][-1 * n_bath_ind]
        for i in range(len(Eval)):
            if numpy.abs(Eval[i]) >= first_el:
                Bidx.append(i)
    else:
        for i in range(len(Eval)):
            if thres < numpy.abs(Eval[i]) < 1.0 - thres:
                Bidx.append(i)

    # Initialize the transformation matrix (TA)
    TA = numpy.zeros([Tot_sites, len(Frag_sites) + len(Bidx)])
    TA[Frag_sites, : len(Frag_sites)] = numpy.eye(len(Frag_sites))  # Fragment part
    TA[Env_sites1, len(Frag_sites) :] = Evec[:, Bidx]  # Environment part

    if return_orb_count:
        # return TA, norbs_frag, norbs_bath
        return TA, Frag_sites1.shape[0], len(Bidx)
    else:
        return TA
