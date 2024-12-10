# Author(s): Oinam Romesh Meitei

import os
import pickle

import h5py
import numpy
from pyscf import ao2mo, scf

from quemb.molbe._opt import BEOPT
from quemb.molbe.be_parallel import be_func_parallel
from quemb.molbe.eri_onthefly import integral_direct_DF
from quemb.molbe.lo import MixinLocalize
from quemb.molbe.misc import print_energy
from quemb.molbe.pfrag import Frags
from quemb.molbe.solver import be_func
from quemb.shared import be_var
from quemb.shared.external.optqn import (
    get_be_error_jacobian as _ext_get_be_error_jacobian,
)
from quemb.shared.helper import copy_docstring


class storeBE:
    def __init__(
        self,
        Nocc,
        hf_veff,
        hcore,
        S,
        C,
        hf_dm,
        hf_etot,
        W,
        lmo_coeff,
        enuc,
        E_core,
        C_core,
        P_core,
        core_veff,
        mo_energy,
    ):
        self.Nocc = Nocc
        self.hf_veff = hf_veff
        self.hcore = hcore
        self.S = S
        self.C = C
        self.hf_dm = hf_dm
        self.hf_etot = hf_etot
        self.W = W
        self.lmo_coeff = lmo_coeff
        self.enuc = enuc
        self.E_core = E_core
        self.C_core = C_core
        self.P_core = P_core
        self.core_veff = core_veff
        self.mo_energy = mo_energy


class BE(MixinLocalize):
    """
    Class for handling bootstrap embedding (BE) calculations.

    This class encapsulates the functionalities required for performing
    bootstrap embedding calculations, including setting up the BE environment,
    initializing fragments, performing SCF calculations, and
    evaluating energies.

    Attributes
    ----------
    mf : pyscf.scf.hf.SCF
        PySCF mean-field object.
    fobj : quemb.molbe.fragment.fragpart
        Fragment object containing sites, centers, edges, and indices.
    eri_file : str
        Path to the file storing two-electron integrals.
    lo_method : str
        Method for orbital localization, default is 'lowdin'.
    """

    def __init__(
        self,
        mf,
        fobj,
        eri_file="eri_file.h5",
        lo_method="lowdin",
        pop_method=None,
        compute_hf=True,
        restart=False,
        save=False,
        restart_file="storebe.pk",
        mo_energy=None,
        save_file="storebe.pk",
        hci_pt=False,
        nproc=1,
        ompnum=4,
        scratch_dir=None,
        hci_cutoff=0.001,
        ci_coeff_cutoff=None,
        select_cutoff=None,
        integral_direct_DF=False,
        auxbasis=None,
    ):
        """
        Constructor for BE object.

        Parameters
        ----------
        mf : pyscf.scf.hf.SCF
            PySCF mean-field object.
        fobj : quemb.molbe.fragment.fragpart
            Fragment object containing sites, centers, edges, and indices.
        eri_file : str, optional
            Path to the file storing two-electron integrals, by default 'eri_file.h5'.
        lo_method : str, optional
            Method for orbital localization, by default 'lowdin'.
        compute_hf : bool, optional
            Whether to compute Hartree-Fock energy, by default True.
        restart : bool, optional
            Whether to restart from a previous calculation, by default False.
        save : bool, optional
            Whether to save intermediate objects for restart, by default False.
        restart_file : str, optional
            Path to the file storing restart information, by default 'storebe.pk'.
        mo_energy : numpy.ndarray, optional
            Molecular orbital energies, by default None.
        save_file : str, optional
            Path to the file storing save information, by default 'storebe.pk'.
        nproc : int, optional
            Number of processors for parallel calculations, by default 1. If set to >1,
            threaded parallel computation is invoked.
        ompnum : int, optional
            Number of OpenMP threads, by default 4.
        integral_direct_DF: bool, optional
            If mf._eri is None (i.e. ERIs are not saved in memory using incore_anyway),
            this flag is used to determine if the ERIs are computed integral-directly
            using density fitting; by default False.
        auxbasis : str, optional
            Auxiliary basis for density fitting, by default None
            (uses default auxiliary basis defined in PySCF).
        """

        if restart:
            # Load previous calculation data from restart file
            with open(restart_file, "rb") as rfile:
                store_ = pickle.load(rfile)
                rfile.close()
            self.Nocc = store_.Nocc
            self.hf_veff = store_.hf_veff
            self.hcore = store_.hcore
            self.S = store_.S
            self.C = store_.C
            self.hf_dm = store_.hf_dm
            self.hf_etot = store_.hf_etot
            self.W = store_.W
            self.lmo_coeff = store_.lmo_coeff
            self.enuc = store_.enuc
            self.E_core = store_.E_core
            self.C_core = store_.C_core
            self.P_core = store_.P_core
            self.core_veff = store_.core_veff
            self.mo_energy = store_.mo_energy

        self.unrestricted = False
        self.nproc = nproc
        self.ompnum = ompnum
        self.integral_direct_DF = integral_direct_DF
        self.auxbasis = auxbasis

        # Fragment information from fobj
        self.frag_type = fobj.frag_type
        self.Nfrag = fobj.Nfrag
        self.fsites = fobj.fsites
        self.edge = fobj.edge
        self.center = fobj.center
        self.edge_idx = fobj.edge_idx
        self.center_idx = fobj.center_idx
        self.centerf_idx = fobj.centerf_idx
        self.ebe_weight = fobj.ebe_weight
        self.be_type = fobj.be_type
        self.mol = fobj.mol

        self.ebe_hf = 0.0
        self.ebe_tot = 0.0

        # HCI parameters
        self.hci_cutoff = hci_cutoff
        self.ci_coeff_cutoff = ci_coeff_cutoff
        self.select_cutoff = select_cutoff
        self.hci_pt = hci_pt

        self.mf = mf
        if not restart:
            self.mo_energy = mf.mo_energy

            self.mf = mf
            self.Nocc = mf.mol.nelectron // 2
            self.enuc = mf.energy_nuc()

            self.hcore = mf.get_hcore()
            self.S = mf.get_ovlp()
            self.C = numpy.array(mf.mo_coeff)
            self.hf_dm = mf.make_rdm1()
            self.hf_veff = mf.get_veff()
            self.hf_etot = mf.e_tot
            self.W = None
            self.lmo_coeff = None
            self.cinv = None

        self.print_ini()
        self.Fobjs = []
        self.pot = initialize_pot(self.Nfrag, self.edge_idx)
        self.eri_file = eri_file
        self.scratch_dir = scratch_dir

        # Set scratch directory
        jobid = ""
        if be_var.CREATE_SCRATCH_DIR:
            jobid = os.environ.get("SLURM_JOB_ID", "")
        if be_var.SCRATCH:
            self.scratch_dir = be_var.SCRATCH + str(jobid)
            os.system("mkdir -p " + self.scratch_dir)
        else:
            self.scratch_dir = None
        if not jobid:
            self.eri_file = be_var.SCRATCH + eri_file
        else:
            self.eri_file = self.scratch_dir + "/" + eri_file

        self.frozen_core = False if not fobj.frozen_core else True
        self.ncore = 0
        if not restart:
            self.E_core = 0
            self.C_core = None
            self.P_core = None
            self.core_veff = None

        if self.frozen_core:
            # Handle frozen core orbitals
            self.ncore = fobj.ncore
            self.no_core_idx = fobj.no_core_idx
            self.core_list = fobj.core_list

            if not restart:
                self.Nocc -= self.ncore
                self.hf_dm = 2.0 * numpy.dot(
                    self.C[:, self.ncore : self.ncore + self.Nocc],
                    self.C[:, self.ncore : self.ncore + self.Nocc].T,
                )
                self.C_core = self.C[:, : self.ncore]
                self.P_core = numpy.dot(self.C_core, self.C_core.T)
                self.core_veff = mf.get_veff(dm=self.P_core * 2.0)
                self.E_core = numpy.einsum(
                    "ji,ji->", 2.0 * self.hcore + self.core_veff, self.P_core
                )
                self.hf_veff -= self.core_veff
                self.hcore += self.core_veff

        if not restart:
            # Localize orbitals
            self.localize(
                lo_method,
                pop_method=pop_method,
                mol=self.mol,
                valence_basis=fobj.valence_basis,
                valence_only=fobj.valence_only,
            )

            if fobj.valence_only and lo_method == "iao":
                self.Ciao_pao = self.localize(
                    lo_method,
                    pop_method=pop_method,
                    mol=self.mol,
                    valence_basis=fobj.valence_basis,
                    hstack=True,
                    valence_only=False,
                    nosave=True,
                )

        if save:
            # Save intermediate results for restart
            store_ = storeBE(
                self.Nocc,
                self.hf_veff,
                self.hcore,
                self.S,
                self.C,
                self.hf_dm,
                self.hf_etot,
                self.W,
                self.lmo_coeff,
                self.enuc,
                self.E_core,
                self.C_core,
                self.P_core,
                self.core_veff,
                self.mo_energy,
            )

            with open(save_file, "wb") as rfile:
                pickle.dump(store_, rfile, pickle.HIGHEST_PROTOCOL)
            rfile.close()

        if not restart:
            # Initialize fragments and perform initial calculations
            self.initialize(mf._eri, compute_hf)
        else:
            self.initialize(None, compute_hf, restart=True)

    def rdm1_fullbasis(
        self,
        return_ao=True,
        only_rdm1=False,
        only_rdm2=False,
        return_lo=False,
        return_RDM2=True,
        print_energy=False,
    ):
        """Compute the one- and two-particle reduced density matrices (RDM1 and RDM2).

        Parameters
        ----------
        return_ao : bool, optional
            Whether to return the RDMs in the AO basis. Default is True.
        only_rdm1 : bool, optional
            Whether to compute only the RDM1. Default is False.
        only_rdm2 : bool, optional
            Whether to compute only the RDM2. Default is False.
        return_lo : bool, optional
            Whether to return the RDMs in the localized orbital (LO) basis.
            Default is False.
        return_RDM2 : bool, optional
            Whether to return the two-particle RDM (RDM2). Default is True.
        print_energy : bool, optional
            Whether to print the energy contributions. Default is False.

        Returns
        -------
        rdm1AO : numpy.ndarray
            The one-particle RDM in the AO basis.
        rdm2AO : numpy.ndarray
            The two-particle RDM in the AO basis (if return_RDM2 is True).
        rdm1LO : numpy.ndarray
            The one-particle RDM in the LO basis (if return_lo is True).
        rdm2LO : numpy.ndarray
            The two-particle RDM in the LO basis
            (if return_lo and return_RDM2 are True).
        rdm1MO : numpy.ndarray
            The one-particle RDM in the molecular orbital (MO) basis
            (if return_ao is False).
        rdm2MO : numpy.ndarray
            The two-particle RDM in the MO basis
            (if return_ao is False and return_RDM2 is True).
        """
        # Copy the molecular orbital coefficients
        C_mo = self.C.copy()
        nao = C_mo.shape[0]

        # Initialize density matrices for atomic orbitals (AO)
        rdm1AO = numpy.zeros((nao, nao))
        rdm2AO = numpy.zeros((nao, nao, nao, nao))

        for fobjs in self.Fobjs:
            if return_RDM2:
                # Adjust the one-particle reduced density matrix (RDM1)
                drdm1 = fobjs.rdm1__.copy()
                drdm1[numpy.diag_indices(fobjs.nsocc)] -= 2.0

                # Compute the two-particle reduced density matrix (RDM2) and subtract
                #   non-connected component
                dm_nc = numpy.einsum(
                    "ij,kl->ijkl", drdm1, drdm1, dtype=numpy.float64, optimize=True
                ) - 0.5 * numpy.einsum(
                    "ij,kl->iklj", drdm1, drdm1, dtype=numpy.float64, optimize=True
                )
                fobjs.rdm2__ -= dm_nc

            # Generate the projection matrix
            cind = [fobjs.fsites[i] for i in fobjs.efac[1]]
            Pc_ = (
                fobjs.TA.T
                @ self.S
                @ self.W[:, cind]
                @ self.W[:, cind].T
                @ self.S
                @ fobjs.TA
            )

            if not only_rdm2:
                # Compute RDM1 in the localized orbital (LO) basis
                #   and transform to AO basis
                rdm1_eo = fobjs.mo_coeffs @ fobjs.rdm1__ @ fobjs.mo_coeffs.T
                rdm1_center = Pc_ @ rdm1_eo
                rdm1_ao = fobjs.TA @ rdm1_center @ fobjs.TA.T
                rdm1AO += rdm1_ao

            if not only_rdm1:
                # Transform RDM2 to AO basis
                rdm2s = numpy.einsum(
                    "ijkl,pi,qj,rk,sl->pqrs",
                    fobjs.rdm2__,
                    *([fobjs.mo_coeffs] * 4),
                    optimize=True,
                )
                rdm2_ao = numpy.einsum(
                    "xi,ijkl,px,qj,rk,sl->pqrs",
                    Pc_,
                    rdm2s,
                    fobjs.TA,
                    fobjs.TA,
                    fobjs.TA,
                    fobjs.TA,
                    optimize=True,
                )
                rdm2AO += rdm2_ao

        if not only_rdm1:
            # Symmetrize RDM2 and add the non-cumulant part if requested
            rdm2AO = (rdm2AO + rdm2AO.T) / 2.0
            if return_RDM2:
                nc_AO = (
                    numpy.einsum(
                        "ij,kl->ijkl",
                        rdm1AO,
                        rdm1AO,
                        dtype=numpy.float64,
                        optimize=True,
                    )
                    - numpy.einsum(
                        "ij,kl->iklj",
                        rdm1AO,
                        rdm1AO,
                        dtype=numpy.float64,
                        optimize=True,
                    )
                    * 0.5
                )
                rdm2AO = nc_AO + rdm2AO

            # Transform RDM2 to the molecular orbital (MO) basis if needed
            if not return_ao:
                CmoT_S = self.C.T @ self.S
                rdm2MO = numpy.einsum(
                    "ijkl,pi,qj,rk,sl->pqrs",
                    rdm2AO,
                    CmoT_S,
                    CmoT_S,
                    CmoT_S,
                    CmoT_S,
                    optimize=True,
                )

            # Transform RDM2 to the localized orbital (LO) basis if needed
            if return_lo:
                CloT_S = self.W.T @ self.S
                rdm2LO = numpy.einsum(
                    "ijkl,pi,qj,rk,sl->pqrs",
                    rdm2AO,
                    CloT_S,
                    CloT_S,
                    CloT_S,
                    CloT_S,
                    optimize=True,
                )

        if not only_rdm2:
            # Symmetrize RDM1
            rdm1AO = (rdm1AO + rdm1AO.T) / 2.0

            # Transform RDM1 to the MO basis if needed
            if not return_ao:
                rdm1MO = self.C.T @ self.S @ rdm1AO @ self.S @ self.C

            # Transform RDM1 to the LO basis if needed
            if return_lo:
                rdm1LO = self.W.T @ self.S @ rdm1AO @ self.S @ self.W

        if return_RDM2 and print_energy:
            # Compute and print energy contributions
            Eh1 = numpy.einsum("ij,ij", self.hcore, rdm1AO, optimize=True)
            eri = ao2mo.restore(1, self.mf._eri, self.mf.mo_coeff.shape[1])
            E2 = 0.5 * numpy.einsum("pqrs,pqrs", eri, rdm2AO, optimize=True)
            print(flush=True)
            print("-----------------------------------------------------", flush=True)
            print(" BE ENERGIES with cumulant-based expression", flush=True)

            print("-----------------------------------------------------", flush=True)

            print(" 1-elec E        : {:>15.8f} Ha".format(Eh1), flush=True)
            print(" 2-elec E        : {:>15.8f} Ha".format(E2), flush=True)
            E_tot = Eh1 + E2 + self.E_core + self.enuc
            print(" E_BE            : {:>15.8f} Ha".format(E_tot), flush=True)
            print(
                " Ecorr BE        : {:>15.8f} Ha".format((E_tot) - self.ebe_hf),
                flush=True,
            )
            print("-----------------------------------------------------", flush=True)
            print(flush=True)

        if only_rdm1:
            if return_ao:
                return rdm1AO
            else:
                return rdm1MO
        if only_rdm2:
            if return_ao:
                return rdm2AO
            else:
                return rdm2MO

        if return_lo and return_ao:
            return (rdm1AO, rdm2AO, rdm1LO, rdm2LO)
        if return_lo and not return_ao:
            return (rdm1MO, rdm2MO, rdm1LO, rdm2LO)

        if return_ao:
            return rdm1AO, rdm2AO
        if not return_ao:
            return rdm1MO, rdm2MO

    def compute_energy_full(
        self, approx_cumulant=False, use_full_rdm=False, return_rdm=True
    ):
        """Compute the total energy using rdms in the full basis.

        Parameters
        ----------
        approx_cumulant : bool, optional
            If True, use an approximate cumulant for the energy computation.
            Default is False.
        use_full_rdm : bool, optional
            If True, use the full two-particle RDM for energy computation.
            Default is False.
        return_rdm : bool, optional
            If True, return the computed reduced density matrices (RDMs).
            Default is True.

        Returns
        -------
        tuple of numpy.ndarray or None
            If `return_rdm` is True, returns a tuple containing the one-particle
            and two-particle reduced density matrices (RDM1 and RDM2).
            Otherwise, returns None.

        Notes
        -----
        This function computes the total energy in the full basis, with options to use
        approximate or true cumulants, and to return the
        reduced density matrices (RDMs).  The energy components are printed as part
        of the function's output.
        """
        # Compute the one-particle reduced density matrix (RDM1) and the cumulant
        # (Kumul) in the full basis
        rdm1f, Kumul, _, _ = self.rdm1_fullbasis(
            return_lo=True, return_RDM2=False
        )  # rdm1f, Kumul, rdm1_lo, rdm2_lo !!

        if not approx_cumulant:
            # Compute the true two-particle reduced density matrix (RDM2) if not using
            # approximate cumulant
            Kumul_T = self.rdm1_fullbasis(only_rdm2=True)

        if return_rdm:
            # Construct the full RDM2 from RDM1
            RDM2_full = (
                numpy.einsum(
                    "ij,kl->ijkl", rdm1f, rdm1f, dtype=numpy.float64, optimize=True
                )
                - numpy.einsum(
                    "ij,kl->iklj", rdm1f, rdm1f, dtype=numpy.float64, optimize=True
                )
                * 0.5
            )

            # Add the cumulant part to RDM2
            if not approx_cumulant:
                RDM2_full += Kumul_T
            else:
                RDM2_full += Kumul

        # Compute the change in the one-particle density matrix (delta_gamma)
        del_gamma = rdm1f - self.hf_dm

        # Compute the effective potential
        veff = scf.hf.get_veff(self.mol, rdm1f, hermi=0)

        # Compute the one-electron energy
        Eh1 = numpy.einsum("ij,ij", self.hcore, rdm1f, optimize=True)

        # Compute the energy due to the effective potential
        EVeff = numpy.einsum("ij,ij", veff, rdm1f, optimize=True)

        # Compute the change in the one-electron energy
        Eh1_dg = numpy.einsum("ij,ij", self.hcore, del_gamma, optimize=True)

        # Compute the change in the effective potential energy
        Eveff_dg = numpy.einsum("ij,ij", self.hf_veff, del_gamma, optimize=True)

        # Restore the electron repulsion integrals (ERI)
        eri = ao2mo.restore(1, self.mf._eri, self.mf.mo_coeff.shape[1])

        # Compute the cumulant part of the two-electron energy
        EKumul = numpy.einsum("pqrs,pqrs", eri, Kumul, optimize=True)

        if not approx_cumulant:
            # Compute the true two-electron energy if not using approximate cumulant
            EKumul_T = numpy.einsum("pqrs,pqrs", eri, Kumul_T, optimize=True)

        if use_full_rdm and return_rdm:
            # Compute the full two-electron energy using the full RDM2
            E2 = numpy.einsum("pqrs,pqrs", eri, RDM2_full, optimize=True)

        # Compute the approximate BE total energy
        EKapprox = self.ebe_hf + Eh1_dg + Eveff_dg + EKumul / 2.0
        self.ebe_tot = EKapprox

        if not approx_cumulant:
            # Compute the true BE total energy if not using approximate cumulant
            EKtrue = Eh1 + EVeff / 2.0 + EKumul_T / 2.0 + self.enuc + self.E_core
            self.ebe_tot = EKtrue

        # Print energy results
        print("-----------------------------------------------------", flush=True)
        print(" BE ENERGIES with cumulant-based expression", flush=True)

        print("-----------------------------------------------------", flush=True)
        print(" E_BE = E_HF + Tr(F del g) + Tr(V K_approx)", flush=True)
        print(" E_HF            : {:>14.8f} Ha".format(self.ebe_hf), flush=True)
        print(" Tr(F del g)     : {:>14.8f} Ha".format(Eh1_dg + Eveff_dg), flush=True)
        print(" Tr(V K_aprrox)  : {:>14.8f} Ha".format(EKumul / 2.0), flush=True)
        print(" E_BE            : {:>14.8f} Ha".format(EKapprox), flush=True)
        print(
            " Ecorr BE        : {:>14.8f} Ha".format(EKapprox - self.ebe_hf), flush=True
        )

        if not approx_cumulant:
            print(flush=True)
            print(" E_BE = Tr(F[g] g) + Tr(V K_true)", flush=True)
            print(" Tr(h1 g)        : {:>14.8f} Ha".format(Eh1), flush=True)
            print(" Tr(Veff[g] g)   : {:>14.8f} Ha".format(EVeff / 2.0), flush=True)
            print(" Tr(V K_true)    : {:>14.8f} Ha".format(EKumul_T / 2.0), flush=True)
            print(" E_BE            : {:>14.8f} Ha".format(EKtrue), flush=True)
            if use_full_rdm and return_rdm:
                print(
                    " E(g+G)          : {:>14.8f} Ha".format(
                        Eh1 + 0.5 * E2 + self.E_core + self.enuc
                    ),
                    flush=True,
                )
            print(
                " Ecorr BE        : {:>14.8f} Ha".format(EKtrue - self.ebe_hf),
                flush=True,
            )
            print(flush=True)
            print(" True - approx   : {:>14.4e} Ha".format(EKtrue - EKapprox))
        print("-----------------------------------------------------", flush=True)

        print(flush=True)

        # Return the RDMs if requested
        if return_rdm:
            return (rdm1f, RDM2_full)

    def optimize(
        self,
        solver="MP2",
        method="QN",
        only_chem=False,
        conv_tol=1.0e-6,
        relax_density=False,
        use_cumulant=True,
        J0=None,
        nproc=1,
        ompnum=4,
        max_iter=500,
        scratch_dir=None,
        trust_region=False,
        **solver_kwargs,
    ):
        """BE optimization function

        Interfaces BEOPT to perform bootstrap embedding optimization.

        Parameters
        ----------
        solver : str, optional
            High-level solver for the fragment, by default 'MP2'
        method : str, optional
            Optimization method, by default 'QN'
        only_chem : bool, optional
            If true, density matching is not performed -- only global chemical potential
            is optimized, by default False
        conv_tol : float, optional
            Convergence tolerance, by default 1.e-6
        relax_density : bool, optional
            Whether to use relaxed or unrelaxed densities, by default False
            This option is for using CCSD as solver. Relaxed density here uses
            Lambda amplitudes, whereas unrelaxed density only uses T amplitudes.
            c.f. See http://classic.chem.msu.su/cgi-bin/ceilidh.exe/gran/gamess/forum/?C34df668afbHW-7216-1405+00.htm
            for the distinction between the two
        use_cumulant : bool, optional
            Use cumulant-based energy expression, by default True
        max_iter : int, optional
            Maximum number of optimization steps, by default 500
        nproc : int
            Total number of processors assigned for the optimization. Defaults to 1.
            When nproc > 1, Python multithreading is invoked.
        ompnum : int
            If nproc > 1, ompnum sets the number of cores for OpenMP parallelization.
            Defaults to 4
        J0 : list of list of float
            Initial Jacobian.
        trust_region : bool, optional
            Use trust-region based QN optimization, by default False
        """
        # Check if only chemical potential optimization is required
        if not only_chem:
            pot = self.pot
            if self.be_type == "be1":
                raise ValueError(
                    "BE1 only works with chemical potential optimization. "
                    "Set only_chem=True"
                )
        else:
            pot = [0.0]

        # Initialize the BEOPT object
        be_ = BEOPT(
            pot,
            self.Fobjs,
            self.Nocc,
            self.enuc,
            hf_veff=self.hf_veff,
            nproc=nproc,
            ompnum=ompnum,
            scratch_dir=scratch_dir,
            max_space=max_iter,
            conv_tol=conv_tol,
            only_chem=only_chem,
            hci_cutoff=self.hci_cutoff,
            ci_coeff_cutoff=self.ci_coeff_cutoff,
            relax_density=relax_density,
            select_cutoff=self.select_cutoff,
            hci_pt=self.hci_pt,
            solver=solver,
            ebe_hf=self.ebe_hf,
            **solver_kwargs,
        )

        if method == "QN":
            # Prepare the initial Jacobian matrix
            if only_chem:
                J0 = [[0.0]]
                J0 = self.get_be_error_jacobian(jac_solver="HF")
                J0 = [[J0[-1, -1]]]
            else:
                J0 = self.get_be_error_jacobian(jac_solver="HF")

            # Perform the optimization
            be_.optimize(method, J0=J0, trust_region=trust_region)
            self.ebe_tot = self.ebe_hf + be_.Ebe[0]
            # Print the energy components
            print_energy(
                be_.Ebe[0], be_.Ebe[1][1], be_.Ebe[1][0] + be_.Ebe[1][2], self.ebe_hf
            )
        else:
            raise ValueError("This optimization method for BE is not supported")

    @copy_docstring(_ext_get_be_error_jacobian)
    def get_be_error_jacobian(self, jac_solver="HF"):
        return _ext_get_be_error_jacobian(self.Nfrag, self.Fobjs, jac_solver)

    def print_ini(self):
        """
        Print initialization banner for the MOLBE calculation.
        """
        print("-----------------------------------------------------------", flush=True)

        print("  MMM     MMM    OOOO    LL           BBBBBBB    EEEEEEE ", flush=True)
        print("  M MM   MM M   OO  OO   LL           BB     B   EE      ", flush=True)
        print("  M  MM MM  M  OO    OO  LL           BB     B   EE      ", flush=True)
        print("  M   MMM   M  OO    OO  LL     ===   BBBBBBB    EEEEEEE ", flush=True)
        print("  M         M  OO    OO  LL           BB     B   EE      ", flush=True)
        print("  M         M   OO  OO   LL           BB     B   EE      ", flush=True)
        print("  M         M    OOOO    LLLLLL       BBBBBBB    EEEEEEE ", flush=True)

        print(flush=True)
        print("            MOLECULAR BOOTSTRAP EMBEDDING", flush=True)
        print("            BEn = ", self.be_type, flush=True)
        print("-----------------------------------------------------------", flush=True)
        print(flush=True)

    def initialize(self, eri_, compute_hf, restart=False):
        """
        Initialize the Bootstrap Embedding calculation.

        Parameters
        ----------
        eri_ : numpy.ndarray
            Electron repulsion integrals.
        compute_hf : bool
            Whether to compute Hartree-Fock energy.
        restart : bool, optional
            Whether to restart from a previous calculation, by default False.
        """
        if compute_hf:
            E_hf = 0.0

        # Create a file to store ERIs
        if not restart:
            file_eri = h5py.File(self.eri_file, "w")
        lentmp = len(self.edge_idx)
        for I in range(self.Nfrag):
            if lentmp:
                fobjs_ = Frags(
                    self.fsites[I],
                    I,
                    edge=self.edge[I],
                    eri_file=self.eri_file,
                    center=self.center[I],
                    edge_idx=self.edge_idx[I],
                    center_idx=self.center_idx[I],
                    efac=self.ebe_weight[I],
                    centerf_idx=self.centerf_idx[I],
                )
            else:
                fobjs_ = Frags(
                    self.fsites[I],
                    I,
                    edge=[],
                    center=[],
                    eri_file=self.eri_file,
                    edge_idx=[],
                    center_idx=[],
                    centerf_idx=[],
                    efac=self.ebe_weight[I],
                )
            fobjs_.sd(self.W, self.lmo_coeff, self.Nocc)

            self.Fobjs.append(fobjs_)

        if not restart:
            # Transform ERIs for each fragment and store in the file
            # ERI Transform Decision Tree
            # Do we have full (ij|kl)?
            #   Yes -- ao2mo, incore version
            #   No  -- Do we have (ij|P) from density fitting?
            #       Yes -- ao2mo, outcore version, using saved (ij|P)
            #       No  -- if integral_direct_DF is requested, invoke on-the-fly routine
            assert (
                (eri_ is not None)
                or (hasattr(self.mf, "with_df"))
                or (self.integral_direct_DF)
            ), "Input mean-field object is missing ERI (mf._eri) or DF (mf.with_df) "
            "object AND integral direct DF routine was not requested. "
            "Please check your inputs."
            if (
                eri_ is not None
            ):  # incore ao2mo using saved eri from mean-field calculation
                for I in range(self.Nfrag):
                    eri = ao2mo.incore.full(eri_, self.Fobjs[I].TA, compact=True)
                    file_eri.create_dataset(self.Fobjs[I].dname, data=eri)
            elif hasattr(self.mf, "with_df") and self.mf.with_df is not None:
                # pyscf.ao2mo uses DF object in an outcore fashion using (ij|P)
                #   in pyscf temp directory
                for I in range(self.Nfrag):
                    eri = self.mf.with_df.ao2mo(self.Fobjs[I].TA, compact=True)
                    file_eri.create_dataset(self.Fobjs[I].dname, data=eri)
            else:
                # If ERIs are not saved on memory, compute fragment ERIs integral-direct
                if (
                    self.integral_direct_DF
                ):  # Use density fitting to generate fragment ERIs on-the-fly
                    integral_direct_DF(
                        self.mf, self.Fobjs, file_eri, auxbasis=self.auxbasis
                    )
                else:  # Calculate ERIs on-the-fly to generate fragment ERIs
                    # TODO: Future feature to be implemented
                    # NOTE: Ideally, we want AO shell pair screening for this.
                    return NotImplementedError
        else:
            eri = None

        for fobjs_ in self.Fobjs:
            # Process each fragment
            eri = numpy.array(file_eri.get(fobjs_.dname))
            _ = fobjs_.get_nsocc(self.S, self.C, self.Nocc, ncore=self.ncore)

            fobjs_.cons_h1(self.hcore)

            if not restart:
                eri = ao2mo.restore(8, eri, fobjs_.nao)

            fobjs_.cons_fock(self.hf_veff, self.S, self.hf_dm, eri_=eri)

            fobjs_.heff = numpy.zeros_like(fobjs_.h1)
            fobjs_.scf(fs=True, eri=eri)

            fobjs_.dm0 = (
                numpy.dot(
                    fobjs_._mo_coeffs[:, : fobjs_.nsocc],
                    fobjs_._mo_coeffs[:, : fobjs_.nsocc].conj().T,
                )
                * 2.0
            )

            if compute_hf:
                _, _, _ = fobjs_.energy_hf(return_e1=True)  # eh1, ecoul, ef
                E_hf += fobjs_.ebe_hf

        if not restart:
            file_eri.close()

        if compute_hf:
            self.ebe_hf = E_hf + self.enuc + self.E_core
            hf_err = self.hf_etot - self.ebe_hf
            print(
                "HF-in-HF error                 :  {:>.4e} Ha".format(hf_err),
                flush=True,
            )
            if abs(hf_err) > 1.0e-5:
                print("WARNING!!! Large HF-in-HF energy error")

            print(flush=True)

        couti = 0
        for fobj in self.Fobjs:
            fobj.udim = couti
            couti = fobj.set_udim(couti)

    def oneshot(
        self,
        solver="MP2",
        nproc=1,
        ompnum=4,
        calc_frag_energy=False,
        clean_eri=False,
        scratch_dir=None,
        **solver_kwargs,
    ):
        """
        Perform a one-shot bootstrap embedding calculation.

        Parameters
        ----------
        solver : str, optional
            High-level quantum chemistry method, by default 'MP2'. 'CCSD', 'FCI',
            and variants of selected CI are supported.
        nproc : int, optional
            Number of processors for parallel calculations, by default 1.
            If set to >1, multi-threaded parallel computation is invoked.
        ompnum : int, optional
            Number of OpenMP threads, by default 4.
        calc_frag_energy : bool, optional
            Whether to calculate fragment energies, by default False.
        clean_eri : bool, optional
            Whether to clean up ERI files after calculation, by default False.
        """
        self.scratch_dir = scratch_dir
        self.solver_kwargs = solver_kwargs

        print("Calculating Energy by Fragment? ", calc_frag_energy)
        if nproc == 1:
            rets = be_func(
                None,
                self.Fobjs,
                self.Nocc,
                solver,
                self.enuc,
                hf_veff=self.hf_veff,
                hci_cutoff=self.hci_cutoff,
                ci_coeff_cutoff=self.ci_coeff_cutoff,
                select_cutoff=self.select_cutoff,
                nproc=ompnum,
                frag_energy=calc_frag_energy,
                ereturn=True,
                eeval=True,
                scratch_dir=self.scratch_dir,
                **self.solver_kwargs,
            )
        else:
            rets = be_func_parallel(
                None,
                self.Fobjs,
                self.Nocc,
                solver,
                self.enuc,
                hf_veff=self.hf_veff,
                hci_cutoff=self.hci_cutoff,
                ci_coeff_cutoff=self.ci_coeff_cutoff,
                select_cutoff=self.select_cutoff,
                ereturn=True,
                eeval=True,
                frag_energy=calc_frag_energy,
                nproc=nproc,
                ompnum=ompnum,
                scratch_dir=self.scratch_dir,
                **self.solver_kwargs,
            )

        print("-----------------------------------------------------", flush=True)
        print("             One Shot BE ", flush=True)
        print("             Solver : ", solver, flush=True)
        print("-----------------------------------------------------", flush=True)
        print(flush=True)
        if calc_frag_energy:
            print(
                "Final Tr(F del g) is         : {:>12.8f} Ha".format(
                    rets[1][0] + rets[1][2]
                ),
                flush=True,
            )
            print(
                "Final Tr(V K_approx) is      : {:>12.8f} Ha".format(rets[1][1]),
                flush=True,
            )
            print(
                "Final e_corr is              : {:>12.8f} Ha".format(rets[0]),
                flush=True,
            )

            self.ebe_tot = rets[0]

        if not calc_frag_energy:
            self.compute_energy_full(approx_cumulant=True, return_rdm=False)

        if clean_eri:
            try:
                os.remove(self.eri_file)
                os.rmdir(self.scratch_dir)
            except (FileNotFoundError, TypeError):
                print("Scratch directory not removed")

    def update_fock(self, heff=None):
        """
        Update the Fock matrix for each fragment with the effective Hamiltonian.

        Parameters
        ----------
        heff : list of numpy.ndarray, optional
            List of effective Hamiltonian matrices for each fragment, by default None.
        """
        if heff is None:
            for fobj in self.Fobjs:
                fobj.fock += fobj.heff
        else:
            for idx, fobj in self.Fobjs:
                fobj.fock += heff[idx]

    def write_heff(self, heff_file="bepotfile.h5"):
        """
        Write the effective Hamiltonian to a file.

        Parameters
        ----------
        heff_file : str, optional
            Path to the file to store effective Hamiltonian, by default 'bepotfile.h5'.
        """
        filepot = h5py.File(heff_file, "w")
        for fobj in self.Fobjs:
            print(fobj.heff.shape, fobj.dname, flush=True)
            filepot.create_dataset(fobj.dname, data=fobj.heff)
        filepot.close()

    def read_heff(self, heff_file="bepotfile.h5"):
        """
        Read the effective Hamiltonian from a file.

        Parameters
        ----------
        heff_file : str, optional
            Path to the file storing effective Hamiltonian, by default 'bepotfile.h5'.
        """
        filepot = h5py.File(heff_file, "r")
        for fobj in self.Fobjs:
            fobj.heff = filepot.get(fobj.dname)
        filepot.close()


def initialize_pot(Nfrag, edge_idx):
    """
    Initialize the potential array for bootstrap embedding.

    This function initializes a potential array for a given number of fragments
    (`Nfrag`) and their corresponding edge indices (`edge_idx`).
    The potential array is initialized with zeros for each pair of edge site indices
    within each fragment, followed by an
    additional zero for the global chemical potential.

    Parameters
    ----------
    Nfrag : int
        Number of fragments.
    edge_idx : list of list of list of int
        List of edge indices for each fragment. Each element is a list of lists,
        where each sublist contains the indices of edge sites for a particular fragment.

    Returns
    -------
    list of float
        Initialized potential array with zeros.
    """
    pot_ = []

    if not len(edge_idx) == 0:
        for I in range(Nfrag):
            for i in edge_idx[I]:
                for j in range(len(i)):
                    for k in range(len(i)):
                        if j > k:
                            continue
                        pot_.append(0.0)

    pot_.append(0.0)
    return pot_
