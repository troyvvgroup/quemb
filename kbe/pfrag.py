# Author(s): Oinam Romesh Meitei

from .solver import schmidt_decomp_svd
from .helper import *
from molbe.helper import get_eri, get_scfObj
from .misc import *
import numpy, h5py
import functools, sys, math
from pyscf import ao2mo


class Frags:
    """
    Class for handling fragments in periodic bootstrap embedding.

    This class contains various functionalities required for managing and manipulating
    fragments for periodic BE calculations.
    """

    def __init__(
        self,
        fsites,
        ifrag,
        edge=None,
        center=None,
        edge_idx=None,
        center_idx=None,
        efac=None,
        eri_file="eri_file.h5",
        unitcell_nkpt=1,
        ewald_ek=None,
        centerf_idx=None,
        unitcell=1,
    ):
        """Constructor function for `Frags` class.

        Parameters
        ----------
        fsites : list
            list of AOs in the fragment (i.e. pbe.fsites[i] or fragpart.fsites[i])
        ifrag : int
            fragment index (∈ [0, pbe.Nfrag])
        edge : list, optional
            list of lists of edge site AOs for each atom in the fragment, by default None
        center : list, optional
            list of fragment indices where edge site AOs are center site, by default None
        edge_idx : list, optional
            list of lists of indices for edge site AOs within the fragment, by default None
        center_idx : list, optional
            list of lists of indices within the fragment specified in `center` that points to the edge site AOs , by default None
        efac : list, optional
            weight used for energy contributions, by default None
        eri_file : str, optional
            two-electron integrals stored as h5py file, by default 'eri_file.h5'
        centerf_idx : list, optional
            indices of the center site atoms in the fragment, by default None
        """

        self.fsites = fsites
        self.unitcell = unitcell
        self.unitcell_nkpt = unitcell_nkpt
        self.nfsites = len(fsites)
        self.TA = None
        self.TA_lo_eo = None
        self.h1 = None
        self.ifrag = ifrag
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

        self.heff = None
        self.edge = edge
        self.center = center
        self.edge_idx = edge_idx
        self.center_idx = center_idx
        self.centerf_idx = centerf_idx
        self.udim = None

        self._rdm1 = None
        self.__rdm1 = None
        self.__rdm2 = None
        self._del_rdm1 = None
        self.rdm1 = None
        self.genvs = None
        self.ebe = 0.0
        self.ebe_hf = 0.0
        self.efac = efac
        self.ewald_ek = ewald_ek
        self.fock = None
        self.veff = None
        self.veff0 = None
        self.dm_init = None
        self.dm0 = None
        self.eri_file = eri_file
        self.pot = None
        self.ebe_hf0 = 0.0
        self.rdm1_lo_k = None

    def sd(
        self,
        lao,
        lmo,
        nocc,
        frag_type="autogen",
        cell=None,
        kpts=None,
        kmesh=None,
        h1=None,
    ):
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
        cell : pyscf.pbc.gto.Cell
            PySCF pbc.gto.Cell object defining the unit cell and lattice vectors.
        kpts : list of list of float
            k-points in the reciprocal space for periodic computations
        kmesh : list of int
            Number of k-points in each lattice vector direction
        """

        from .misc import get_phase

        nk, nao, nlo = lao.shape
        rdm1_lo_k = numpy.zeros((nk, nlo, nlo), dtype=numpy.result_type(lmo, lmo))
        for k in range(nk):
            rdm1_lo_k[k] += numpy.dot(lmo[k][:, :nocc], lmo[k][:, :nocc].conj().T)
        self.rdm1_lo_k = rdm1_lo_k
        phase = get_phase(cell, kpts, kmesh)
        supcell_rdm = numpy.einsum("Rk,kuv,Sk->RuSv", phase, rdm1_lo_k, phase.conj())
        supcell_rdm = supcell_rdm.reshape(nk * nlo, nk * nlo)

        if numpy.abs(supcell_rdm.imag).max() < 1.0e-6:
            supcell_rdm = supcell_rdm.real
        else:
            print("Imaginary density in Full SD", numpy.abs(supcell_rdm.imag).max())
            sys.exit()

        Sites = [i + (nlo * 0) for i in self.fsites]
        if not frag_type == "autogen":
            Sites.sort()

        TA_R = schmidt_decomp_svd(supcell_rdm, Sites)
        teo = TA_R.shape[-1]
        TA_R = TA_R.reshape(nk, nlo, teo)

        phase1 = get_phase1(cell, kpts, kmesh)
        TA_k = numpy.einsum("Rim, Rk -> kim", TA_R, phase1)
        self.TA_lo_eo = TA_k

        TA_ao_eo_k = numpy.zeros(
            (nk, nao, teo), dtype=numpy.result_type(lao.dtype, TA_k.dtype)
        )
        for k in range(nk):
            TA_ao_eo_k[k] = numpy.dot(lao[k], TA_k[k])

        self.TA = TA_ao_eo_k
        self.nao = TA_ao_eo_k.shape[-1]

        # useful for debugging --
        rdm1_eo = numpy.zeros((teo, teo), dtype=numpy.complex128)
        for k in range(nk):
            rdm1_eo += functools.reduce(
                numpy.dot, (TA_k[k].conj().T, rdm1_lo_k[k], TA_k[k])
            )
        rdm1_eo /= float(nk)

        h1_eo = numpy.zeros((teo, teo), dtype=numpy.complex128)
        for k in range(nk):
            h1_eo += functools.reduce(
                numpy.dot, (self.TA[k].conj().T, h1[k], self.TA[k])
            )
        h1_eo /= float(nk)
        e1 = 2.0 * numpy.einsum(
            "ij,ij->i", h1_eo[: self.nfsites], rdm1_eo[: self.nfsites]
        )
        e_h1 = 0.0
        for i in self.efac[1]:
            e_h1 += self.efac[0] * e1[i]

    def cons_h1(self, h1):
        """
        Construct the one-electron Hamiltonian for the fragment.

        Parameters
        ----------
        h1 : numpy.ndarray
            One-electron Hamiltonian matrix.
        """

        nk, nao, teo = self.TA.shape
        h1_eo = numpy.zeros((teo, teo), dtype=numpy.complex128)
        for k in range(nk):
            h1_eo += functools.reduce(
                numpy.dot, (self.TA[k].conj().T, h1[k], self.TA[k])
            )
        h1_eo /= float(nk)

        if numpy.abs(h1_eo.imag).max() < 1.0e-7:
            self.h1 = h1_eo.real
        else:
            print("Imaginary Hcore ", numpy.abs(h1_eo.imag).max())
            sys.exit()

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
        if numpy.abs(veff_.imag).max() < 1.0e-6:
            self.veff = veff_.real
            self.veff0 = veff0.real
        else:
            print("Imaginary Veff ", numpy.abs(veff_.imag).max())
            sys.exit()
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
        import scipy.linalg

        nk, nao, neo = self.TA.shape
        dm_ = numpy.zeros((nk, nao, nao), dtype=numpy.result_type(C, C))
        for k in range(nk):
            dm_[k] = 2.0 * numpy.dot(
                C[k][:, ncore : ncore + nocc], C[k][:, ncore : ncore + nocc].conj().T
            )
        P_ = numpy.zeros((neo, neo), dtype=numpy.complex128)
        for k in range(nk):
            Cinv = numpy.dot(self.TA[k].conj().T, S[k])
            P_ += functools.reduce(numpy.dot, (Cinv, dm_[k], Cinv.conj().T))

        P_ /= float(nk)
        if numpy.abs(P_.imag).max() < 1.0e-6:
            P_ = P_.real
        else:
            print("Imaginary density in get_nsocc ", numpy.abs(P_.imag).max())
            sys.exit()
        nsocc_ = numpy.trace(P_)
        nsocc = int(numpy.round(nsocc_.real) / 2)

        self.nsocc = nsocc
        return P_

    def scf(
        self,
        heff=None,
        fs=False,
        eri=None,
        pert_h=False,
        pert_list=None,
        save_chkfile=False,
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
        import copy

        if self._mf is not None:
            self._mf = None
        if self._mc is not None:
            self._mc = None
        if heff is None:
            heff = self.heff

        if eri is None:
            eri = get_eri(self.dname, self.nao, eri_file=self.eri_file)

        if dm0 is None:
            dm0 = (
                numpy.dot(
                    self._mo_coeffs[:, : self.nsocc],
                    self._mo_coeffs[:, : self.nsocc].conj().T,
                )
                * 2.0
            )

        mf_ = get_scfObj(
            self.fock + heff,
            eri,
            self.nsocc,
            dm0=dm0,
            fname=self.dname,
            pert_h=pert_h,
            pert_list=pert_list,
            save_chkfile=save_chkfile,
        )

        if pert_h:
            return mf_

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
        return_heff=False,
        be_iter=None,
        no_chempot=False,
        tmp_add=False,
        only_chem=False,
    ):
        """
        Update the effective Hamiltonian for the fragment.
        """
        import h5py

        heff_ = numpy.zeros_like(self.h1)

        if cout is None:
            cout = self.udim
        else:
            cout = cout
        if not no_chempot:
            for i, fi in enumerate(self.fsites):
                if not any(i in sublist for sublist in self.edge_idx):
                    heff_[i, i] -= u[-1]

        if only_chem:
            self.heff = heff_
            if return_heff:
                if cout is None:
                    return heff_
                else:
                    return (cout, heff_)
            return cout

        for idx, i in enumerate(self.edge_idx):
            for j in range(len(i)):
                for k in range(len(i)):
                    if j > k:
                        continue

                    heff_[i[j], i[k]] = u[cout]
                    heff_[i[k], i[j]] = u[cout]

                    cout += 1

        self.heff = heff_
        if return_heff:
            if cout is None:
                return heff_
            else:
                return (cout, heff_)
        return cout

    def set_udim(self, cout):
        for i in self.edge_idx:
            for j in range(len(i)):
                for k in range(len(i)):
                    if j > k:
                        continue
                    cout += 1
        return cout

    def energy_hf(
        self, rdm_hf=None, mo_coeffs=None, eri=None, return_e1=False, unrestricted=False
    ):
        if mo_coeffs is None:
            mo_coeffs = self._mo_coeffs

        if rdm_hf is None:
            rdm_hf = numpy.dot(
                mo_coeffs[:, : self.nsocc], mo_coeffs[:, : self.nsocc].conj().T
            )

        unrestricted = 1.0 if unrestricted else 2.0

        e1 = unrestricted * numpy.einsum(
            "ij,ij->i", self.h1[: self.nfsites], rdm_hf[: self.nfsites]
        )

        ec = (
            0.5
            * unrestricted
            * numpy.einsum(
                "ij,ij->i", self.veff[: self.nfsites], rdm_hf[: self.nfsites]
            )
        )

        if self.TA.ndim == 3:
            jmax = self.TA[0].shape[1]
        else:
            jmax = self.TA.shape[1]
        if eri is None:
            r = h5py.File(self.eri_file, "r")
            eri = r[self.dname][()]

            r.close()

        e2 = numpy.zeros_like(e1)
        for i in range(self.nfsites):
            for j in range(jmax):
                ij = i * (i + 1) // 2 + j if i > j else j * (j + 1) // 2 + i
                Gij = (2.0 * rdm_hf[i, j] * rdm_hf - numpy.outer(rdm_hf[i], rdm_hf[j]))[
                    :jmax, :jmax
                ]
                Gij[numpy.diag_indices(jmax)] *= 0.5
                Gij += Gij.T
                e2[i] += 0.5 * unrestricted * Gij[numpy.tril_indices(jmax)] @ eri[ij]

        e_ = e1 + e2 + ec
        etmp = 0.0
        e1_ = 0.0
        e2_ = 0.0
        ec_ = 0.0
        for i in self.efac[1]:
            etmp += self.efac[0] * e_[i]
            e1_ += self.efac[0] * e1[i]
            e2_ += self.efac[0] * e2[i]
            ec_ += self.efac[0] * ec[i]

        self.ebe_hf = etmp
        if return_e1:
            e_h1 = 0.0
            e_coul = 0.0
            for i in self.efac[1]:
                e_h1 += self.efac[0] * e1[i]
                e_coul += self.efac[0] * (e2[i] + ec[i])
            return (e_h1, e_coul, e1 + e2 + ec)

        return e1 + e2 + ec
