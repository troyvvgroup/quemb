# Author(s): Oinam Romesh Meitei

import pickle
from typing import Literal, TypeAlias
from warnings import warn

import h5py
import numpy
from attrs import define
from numpy import array, diag_indices, einsum, float64, floating, zeros, zeros_like
from numpy.linalg import multi_dot
from pyscf import ao2mo, scf
from typing_extensions import assert_never

from quemb.molbe.be_parallel import be_func_parallel
from quemb.molbe.eri_onthefly import integral_direct_DF
from quemb.molbe.eri_sparse_DF import (
    _transform_sparse_DF_integral,
    _transform_sparse_DF_integral_S_screening_everything,
    _transform_sparse_DF_integral_S_screening_MO,
    _transform_sparse_DF_S_screening_shared_ijP,
    _transform_sparse_DF_S_screening_shared_ijP_and_g,
    _transform_sparse_DF_S_screening_shared_ijP_and_g_fast,
    _transform_sparse_DF_use_shared_ijP,
    _write_eris,
)
from quemb.molbe.fragment import FragPart
from quemb.molbe.lo import MixinLocalize
from quemb.molbe.misc import print_energy_cumulant, print_energy_noncumulant
from quemb.molbe.opt import BEOPT
from quemb.molbe.pfrag import Frags, union_of_frag_MOs_and_index
from quemb.molbe.solver import Solvers, UserSolverArgs, be_func
from quemb.shared.config import settings
from quemb.shared.external.optqn import (
    get_be_error_jacobian as _ext_get_be_error_jacobian,
)
from quemb.shared.helper import Timer, copy_docstring, ensure
from quemb.shared.manage_scratch import WorkDir
from quemb.shared.typing import Matrix, PathLike

IntTransforms: TypeAlias = Literal[
    "in-core",
    "out-core-DF",
    "int-direct-DF",
    "sparse-DF",
    "sparse-DF-S-screening-onlyMO",  # screen the MOs via S_abs
    "sparse-DF-S-screening",  # screen AOs and MOs via S_abs
    "sparse-DF-S-screening-shared-ijP",  # screen AOs and MOs via S_abs and share ijP
    # screen AOs and MOs via S_abs and share ijP and g_ijkl
    "sparse-DF-S-screening-shared-ijP-g",
    # screen AOs and MOs via S_abs and share ijP and g_ijkl, different (faster?)
    "sparse-DF-S-screening-shared-ijP-g-faster",
    "sparse-DF-shared-ijP",
]


@define
class storeBE:
    Nocc: int
    hf_veff: Matrix[floating]
    hcore: Matrix[floating]
    S: Matrix[floating]
    C: Matrix[floating]
    hf_dm: Matrix[floating]
    hf_etot: float
    W: Matrix[floating]
    lmo_coeff: Matrix[floating]
    enuc: float
    ek: float
    E_core: float
    C_core: Matrix[floating]
    P_core: Matrix[floating]
    core_veff: Matrix[floating]


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
    fobj : quemb.molbe.autofrag.FragPart
        Fragment object containing sites, centers, edges, and indices.
    eri_file : str
        Path to the file storing two-electron integrals.
    lo_method : str
        Method for orbital localization, default is 'lowdin'.
    """

    def __init__(
        self,
        mf: scf.hf.SCF,
        fobj: FragPart,
        eri_file: PathLike = "eri_file.h5",
        lo_method: str = "lowdin",
        iao_loc_method: str | None = "SO",
        pop_method: str | None = None,
        compute_hf: bool = True,
        restart: bool = False,
        restart_file: PathLike = "storebe.pk",
        nproc: int = 1,
        ompnum: int = 4,
        thr_bath: float = 1.0e-10,
        scratch_dir: WorkDir | None = None,
        int_transform: IntTransforms = "in-core",
        auxbasis: str | None = None,
        MO_coeff_epsilon: float = 1e-4,
        AO_coeff_epsilon: float = 1e-10,
    ) -> None:
        r"""
        Constructor for BE object.

        Parameters
        ----------
        mf :
            PySCF mean-field object.
        fobj :
            Fragment object containing sites, centers, edges, and indices.
        eri_file :
            Path to the file storing two-electron integrals.
        lo_method :
            Method for orbital localization, by default 'lowdin'.
        iao_loc_method :
            Method for IAO localization, by default "SO"
        pop_method :
            Method for calculating orbital population, by default 'meta-lowdin'
            See pyscf.lo for more details and options
        compute_hf :
            Whether to compute Hartree-Fock energy, by default True.
        restart :
            Whether to restart from a previous calculation, by default False.
        restart_file :
            Path to the file storing restart information, by default 'storebe.pk'.
        nproc :
            Number of processors for parallel calculations, by default 1. If set to >1,
            threaded parallel computation is invoked.
        ompnum :
            Number of OpenMP threads, by default 4.
        thr_bath : float,
            Threshold for bath orbitals in Schmidt decomposition
        scratch_dir :
            Scratch directory.
        int_transform :
            The possible integral transformations.

            - :python:`"in-core"` (default): Use a dense representation of integrals
              in memory without density fitting (DF) and transform in-memory.
            - :python:`"out-core-DF"`: Use a dense, DF representation of integrals,
              the DF integrals :math:`(\mu, \nu | P)` are stored on disc.
            - :python:`"int-direct-DF"`: Use a dense, DF representation of integrals,
              the required DF integrals :math:`(\mu, \nu | P)` are computed and fitted
              on-demand for each fragment.
            - :python:`"sparse-DF"`:  Work in progress.
              Use a sparse, DF representation of integrals,
              and avoid recomputation of elements that are shared across fragments.
        auxbasis :
            Auxiliary basis for density fitting, by default None
            (uses default auxiliary basis defined in PySCF).
            Only relevant for :python:`int_transform in {"int-direct-DF", "sparse-DF"}`.
        """
        init_timer = Timer("Time to initialize BE object")
        if restart:
            # Load previous calculation data from restart file
            with open(restart_file, "rb") as rfile:
                store_ = pickle.load(rfile)
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

        self.MO_coeff_epsilon = MO_coeff_epsilon
        self.AO_coeff_epsilon = AO_coeff_epsilon

        self.unrestricted = False
        self.nproc = nproc
        self.ompnum = ompnum
        self.integral_transform = int_transform
        self.auxbasis = auxbasis
        self.thr_bath = thr_bath

        # Fragment information from fobj
        self.fobj = fobj

        self.ebe_hf = 0.0
        self.ebe_tot = 0.0

        self.mf = mf

        if not restart:
            self.mo_energy = mf.mo_energy

            self.mf = mf
            self.Nocc = mf.mol.nelectron // 2
            self.enuc = mf.energy_nuc()

            self.hcore = mf.get_hcore()
            self.S = mf.get_ovlp()
            self.C = array(mf.mo_coeff)
            self.hf_dm = mf.make_rdm1()
            self.hf_veff = mf.get_veff()
            self.hf_etot = mf.e_tot
            self.W = None
            self.lmo_coeff = None
            self.cinv = None

        self.print_ini()
        self.Fobjs: list[Frags] = []
        self.pot = initialize_pot(self.fobj.n_frag, self.fobj.relAO_per_edge_per_frag)

        if scratch_dir is None:
            self.scratch_dir = WorkDir.from_environment()
        else:
            self.scratch_dir = scratch_dir
        print(f"Scratch dir is in: {self.scratch_dir.path}")
        self.eri_file = self.scratch_dir / eri_file

        self.frozen_core = fobj.frozen_core
        self.ncore = 0
        if not restart:
            self.E_core = 0
            self.C_core = None
            self.P_core = None
            self.core_veff = None

        if self.frozen_core:
            # Handle frozen core orbitals
            assert not (
                fobj.ncore is None or fobj.no_core_idx is None or fobj.core_list is None
            )
            self.ncore = fobj.ncore
            self.no_core_idx = fobj.no_core_idx
            self.core_list = fobj.core_list

            if not restart:
                self.Nocc -= self.ncore
                self.hf_dm = 2.0 * (
                    self.C[:, self.ncore : self.ncore + self.Nocc]
                    @ self.C[:, self.ncore : self.ncore + self.Nocc].T
                )
                self.C_core = self.C[:, : self.ncore]
                self.P_core = self.C_core @ self.C_core.T
                self.core_veff = mf.get_veff(dm=self.P_core * 2.0)
                self.E_core = einsum(
                    "ji,ji->", 2.0 * self.hcore + self.core_veff, self.P_core
                )
                self.hf_veff -= self.core_veff
                self.hcore += self.core_veff

        if not restart:
            # Localize orbitals
            self.localize(
                lo_method,
                iao_valence_basis=fobj.iao_valence_basis,
                iao_loc_method=iao_loc_method,
                iao_valence_only=fobj.iao_valence_only,
                pop_method=pop_method,
            )

            if fobj.iao_valence_only and lo_method.upper() == "IAO":
                self.Ciao_pao = self.localize(
                    lo_method,
                    iao_valence_basis=fobj.iao_valence_basis,
                    iao_loc_method=iao_loc_method,
                    iao_valence_only=False,
                    pop_method=pop_method,
                    hstack=True,
                    nosave=True,
                )

        if not restart:
            # Initialize fragments and perform initial calculations
            self.initialize(
                mf._eri, compute_hf, restart=False, int_transform=int_transform
            )
        else:
            self.initialize(None, compute_hf, restart=True, int_transform=int_transform)
        if settings.PRINT_LEVEL >= 10:
            print(init_timer.str_elapsed())

    def save(self, save_file: PathLike = "storebe.pk") -> None:
        """
        Save intermediate results for restart.

        Parameters
        ----------
        save_file :
            Path to the file storing restart information, by default 'storebe.pk'.
        """
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
        rdm1AO = zeros((nao, nao))
        rdm2AO = zeros((nao, nao, nao, nao))

        for fobjs in self.Fobjs:
            if return_RDM2:
                # Adjust the one-particle reduced density matrix (RDM1)
                drdm1 = fobjs.rdm1__.copy()
                drdm1[diag_indices(fobjs.nsocc)] -= 2.0

                # Compute the two-particle reduced density matrix (RDM2) and subtract
                #   non-connected component
                dm_nc = einsum(
                    "ij,kl->ijkl", drdm1, drdm1, dtype=numpy.float64, optimize=True
                ) - 0.5 * einsum(
                    "ij,kl->iklj", drdm1, drdm1, dtype=numpy.float64, optimize=True
                )
                fobjs.rdm2__ -= dm_nc

            # Generate the projection matrix
            cind = [fobjs.AO_in_frag[i] for i in fobjs.weight_and_relAO_per_center[1]]
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
                rdm2s = einsum(
                    "ijkl,pi,qj,rk,sl->pqrs",
                    fobjs.rdm2__,
                    *([fobjs.mo_coeffs] * 4),
                    optimize=True,
                )
                rdm2_ao = einsum(
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
                    einsum(
                        "ij,kl->ijkl",
                        rdm1AO,
                        rdm1AO,
                        dtype=numpy.float64,
                        optimize=True,
                    )
                    - einsum(
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
                rdm2MO = einsum(
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
                rdm2LO = einsum(
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
            Eh1 = einsum("ij,ij", self.hcore, rdm1AO, optimize=True)
            eri = ao2mo.restore(1, self.mf._eri, self.mf.mo_coeff.shape[1])
            E2 = 0.5 * einsum("pqrs,pqrs", eri, rdm2AO, optimize=True)
            print(flush=True)
            print("-----------------------------------------------------", flush=True)
            print(" BE ENERGIES with cumulant-based expression", flush=True)

            print("-----------------------------------------------------", flush=True)

            print(f" 1-elec E        : {Eh1:>15.8f} Ha", flush=True)
            print(f" 2-elec E        : {E2:>15.8f} Ha", flush=True)
            E_tot = Eh1 + E2 + self.E_core + self.enuc
            print(f" E_BE            : {E_tot:>15.8f} Ha", flush=True)
            print(
                f" Ecorr BE        : {(E_tot) - self.ebe_hf:>15.8f} Ha",
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
            If :python:`return_rdm` is True, returns a tuple containing the one-particle
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
                einsum("ij,kl->ijkl", rdm1f, rdm1f, dtype=float64, optimize=True)
                - einsum("ij,kl->iklj", rdm1f, rdm1f, dtype=float64, optimize=True)
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
        veff = scf.hf.get_veff(self.fobj.mol, rdm1f, hermi=0)

        # Compute the one-electron energy
        Eh1 = einsum("ij,ij", self.hcore, rdm1f, optimize=True)

        # Compute the energy due to the effective potential
        EVeff = einsum("ij,ij", veff, rdm1f, optimize=True)

        # Compute the change in the one-electron energy
        Eh1_dg = einsum("ij,ij", self.hcore, del_gamma, optimize=True)

        # Compute the change in the effective potential energy
        Eveff_dg = einsum("ij,ij", self.hf_veff, del_gamma, optimize=True)

        # Restore the electron repulsion integrals (ERI)
        eri = ao2mo.restore(1, self.mf._eri, self.mf.mo_coeff.shape[1])

        # Compute the cumulant part of the two-electron energy
        EKumul = einsum("pqrs,pqrs", eri, Kumul, optimize=True)

        if not approx_cumulant:
            # Compute the true two-electron energy if not using approximate cumulant
            EKumul_T = einsum("pqrs,pqrs", eri, Kumul_T, optimize=True)

        if use_full_rdm and return_rdm:
            # Compute the full two-electron energy using the full RDM2
            E2 = einsum("pqrs,pqrs", eri, RDM2_full, optimize=True)

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
        print(f" E_HF            : {self.ebe_hf:>14.8f} Ha", flush=True)
        print(f" Tr(F del g)     : {Eh1_dg + Eveff_dg:>14.8f} Ha", flush=True)
        print(f" Tr(V K_aprrox)  : {EKumul / 2.0:>14.8f} Ha", flush=True)
        print(f" E_BE            : {EKapprox:>14.8f} Ha", flush=True)
        print(f" Ecorr BE        : {EKapprox - self.ebe_hf:>14.8f} Ha", flush=True)

        if not approx_cumulant:
            print(flush=True)
            print(" E_BE = Tr(F[g] g) + Tr(V K_true)", flush=True)
            print(f" Tr(h1 g)        : {Eh1:>14.8f} Ha", flush=True)
            print(f" Tr(Veff[g] g)   : {EVeff / 2.0:>14.8f} Ha", flush=True)
            print(f" Tr(V K_true)    : {EKumul_T / 2.0:>14.8f} Ha", flush=True)
            print(f" E_BE            : {EKtrue:>14.8f} Ha", flush=True)
            if use_full_rdm and return_rdm:
                print(
                    " E(g+G)          : {:>14.8f} Ha".format(
                        Eh1 + 0.5 * E2 + self.E_core + self.enuc
                    ),
                    flush=True,
                )
            print(
                f" Ecorr BE        : {EKtrue - self.ebe_hf:>14.8f} Ha",
                flush=True,
            )
            print(flush=True)
            print(f" True - approx   : {EKtrue - EKapprox:>14.4e} Ha")
        print("-----------------------------------------------------", flush=True)

        print(flush=True)

        # Return the RDMs if requested
        if return_rdm:
            return (rdm1f, RDM2_full)

    def optimize(
        self,
        solver: Solvers = "MP2",
        method: str = "QN",
        only_chem: bool = False,
        use_cumulant: bool = True,
        conv_tol: float = 1.0e-6,
        relax_density: bool = False,
        jac_solver: Literal["HF", "MP2", "CCSD"] = "HF",
        nproc: int = 1,
        ompnum: int = 4,
        max_iter: int = 500,
        trust_region: bool = False,
        solver_args: UserSolverArgs | None = None,
    ) -> None:
        """BE optimization function

        Interfaces BEOPT to perform bootstrap embedding optimization.

        Parameters
        ----------
        solver :
            High-level solver for the fragment, by default 'MP2'
        method :
            Optimization method, by default 'QN'
        only_chem :
            If true, density matching is not performed -- only global chemical potential
            is optimized, by default False
        use_cumulant :
            Whether to use the cumulant energy expression, by default True.
        conv_tol :
            Convergence tolerance, by default 1.e-6
        relax_density :
            Whether to use relaxed or unrelaxed densities, by default False
            This option is for using CCSD as solver. Relaxed density here uses
            Lambda amplitudes, whereas unrelaxed density only uses T amplitudes.
            c.f. See http://classic.chem.msu.su/cgi-bin/ceilidh.exe/gran/gamess/forum/?C34df668afbHW-7216-1405+00.htm
            for the distinction between the two
        max_iter :
            Maximum number of optimization steps, by default 500
        nproc :
            Total number of processors assigned for the optimization. Defaults to 1.
            When nproc > 1, Python multithreading is invoked.
        ompnum :
            If nproc > 1, ompnum sets the number of cores for OpenMP parallelization.
            Defaults to 4
        jac_solver :
            Method to form Jacobian used in optimization routine, by default HF.
            Options include HF, MP2, CCSD
        trust_region :
            Use trust-region based QN optimization, by default False
        """
        # Check if only chemical potential optimization is required
        if not only_chem:
            pot = self.pot
            if self.fobj.n_BE == 1:
                raise ValueError(
                    "BE1 only works with chemical potential optimization. "
                    "Set only_chem=True"
                )
            elif (
                #  The `all_centers_are_origins` test is not defined for IAOs
                not self.fobj.iao_valence_basis
                and self.fobj.n_BE >= 3
                and not self.fobj.all_centers_are_origins()
            ):
                raise ValueError(
                    "BE3 currently does not work with matching conditions, if there "
                    "are centers that are not origins.\n"
                    "See this issue https://github.com/troyvvgroup/quemb/issues/150 "
                    "for reference. "
                    "As a stop gap measure you can use the `swallow_replace=True` "
                    "option when fragmentating with chemgen."
                )
        else:
            pot = [0.0]

        # Initialize the BEOPT object
        be_ = BEOPT(
            pot,
            self.Fobjs,
            self.Nocc,
            self.enuc,
            nproc=nproc,
            ompnum=ompnum,
            scratch_dir=self.scratch_dir,
            max_space=max_iter,
            conv_tol=conv_tol,
            only_chem=only_chem,
            use_cumulant=use_cumulant,
            relax_density=relax_density,
            solver=solver,
            ebe_hf=self.ebe_hf,
            solver_args=solver_args,
        )

        if method == "QN":
            # Prepare the initial Jacobian matrix
            if only_chem:
                J0 = array([[0.0]])
                J0 = self.get_be_error_jacobian(jac_solver=jac_solver)
                J0 = J0[-1:, -1:]
            else:
                J0 = self.get_be_error_jacobian(jac_solver=jac_solver)

            # Perform the optimization
            be_.optimize(method, J0=J0, trust_region=trust_region)

            # Print the energy components
            if use_cumulant:
                self.ebe_tot = be_.Ebe[0] + self.ebe_hf
                print_energy_cumulant(
                    be_.Ebe[0],
                    be_.Ebe[1][1],
                    be_.Ebe[1][0] + be_.Ebe[1][2],
                    self.ebe_hf,
                )
            else:
                self.ebe_tot = be_.Ebe[0] + self.enuc
                print_energy_noncumulant(
                    be_.Ebe[0],
                    be_.Ebe[1][0],
                    be_.Ebe[1][2],
                    be_.Ebe[1][1],
                    self.ebe_hf,
                    self.enuc,
                )
        else:
            raise ValueError("This optimization method for BE is not supported")

    @copy_docstring(_ext_get_be_error_jacobian)
    def get_be_error_jacobian(self, jac_solver: str = "HF") -> Matrix[floating]:
        return _ext_get_be_error_jacobian(self.fobj.n_frag, self.Fobjs, jac_solver)

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
        print("            BEn = ", self.fobj.n_BE, flush=True)
        print("-----------------------------------------------------------", flush=True)
        print(flush=True)

    def initialize(
        self,
        eri_,
        compute_hf: bool,
        *,
        restart: bool,
        int_transform: IntTransforms,
    ) -> None:
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
        for I in range(self.fobj.n_frag):
            fobjs_ = self.fobj.to_Frags(I, eri_file=self.eri_file)
            fobjs_.sd(self.W, self.lmo_coeff, self.Nocc, thr_bath=self.thr_bath)

            self.Fobjs.append(fobjs_)

        self.all_fragment_MO_TA, frag_TA_index_per_frag = union_of_frag_MOs_and_index(
            self.Fobjs, self.mf.mol.intor("int1e_ovlp"), epsilon=1e-10
        )
        for fobj, frag_TA_offset in zip(self.Fobjs, frag_TA_index_per_frag):
            fobj.frag_TA_offset = frag_TA_offset

        eritransform_timer = Timer("Time to transform ERIs")

        if not restart:
            # Transform ERIs for each fragment and store in the file
            # ERI Transform Decision Tree
            # Do we have full (ij|kl)?
            #   Yes -- ao2mo, incore version
            #   No  -- Do we have (ij|P) from density fitting?
            #       Yes -- ao2mo, outcore version, using saved (ij|P)
            #       No  -- if integral_direct_DF is requested, invoke on-the-fly routine
            if int_transform == "in-core":
                ensure(eri_ is not None, "ERIs have to be available in memory.")
                for I in range(self.fobj.n_frag):
                    eri = ao2mo.incore.full(eri_, self.Fobjs[I].TA, compact=True)
                    file_eri.create_dataset(self.Fobjs[I].dname, data=eri)
            elif int_transform == "out-core-DF":
                ensure(
                    hasattr(self.mf, "with_df") and self.mf.with_df is not None,
                    "Pyscf mean field object has to support `with_df`.",
                )
                # pyscf.ao2mo uses DF object in an outcore fashion using (ij|P)
                #   in pyscf temp directory
                for I in range(self.fobj.n_frag):
                    eri = self.mf.with_df.ao2mo(self.Fobjs[I].TA, compact=True)
                    file_eri.create_dataset(self.Fobjs[I].dname, data=eri)
            elif int_transform == "int-direct-DF":
                # If ERIs are not saved on memory, compute fragment ERIs integral-direct
                ensure(bool(self.auxbasis), "`auxbasis` has to be defined.")
                integral_direct_DF(
                    self.mf, self.Fobjs, file_eri, auxbasis=self.auxbasis
                )
                eri = None
            elif int_transform == "sparse-DF":
                ensure(bool(self.auxbasis), "`auxbasis` has to be defined.")
                eris = _transform_sparse_DF_integral(
                    self.mf,
                    self.Fobjs,
                    auxbasis=self.auxbasis,
                    MO_coeff_epsilon=self.MO_coeff_epsilon,
                )
                _write_eris(self.Fobjs, eris, file_eri)
                eri = None
            elif int_transform == "sparse-DF-S-screening-onlyMO":
                ensure(bool(self.auxbasis), "`auxbasis` has to be defined.")
                eris = _transform_sparse_DF_integral_S_screening_MO(
                    self.mf,
                    self.Fobjs,
                    auxbasis=self.auxbasis,
                    MO_coeff_epsilon=self.MO_coeff_epsilon,
                )
                _write_eris(self.Fobjs, eris, file_eri)
                eri = None
            elif int_transform == "sparse-DF-S-screening":
                ensure(bool(self.auxbasis), "`auxbasis` has to be defined.")
                eris = _transform_sparse_DF_integral_S_screening_everything(
                    self.mf,
                    self.Fobjs,
                    auxbasis=self.auxbasis,
                    MO_coeff_epsilon=self.MO_coeff_epsilon,
                    AO_coeff_epsilon=self.AO_coeff_epsilon,
                )
                _write_eris(self.Fobjs, eris, file_eri)
                eri = None
            elif int_transform == "sparse-DF-S-screening-shared-ijP":
                ensure(bool(self.auxbasis), "`auxbasis` has to be defined.")
                eris = _transform_sparse_DF_S_screening_shared_ijP(
                    self.mf,
                    self.Fobjs,
                    self.all_fragment_MO_TA,
                    auxbasis=self.auxbasis,
                    MO_coeff_epsilon=self.MO_coeff_epsilon,
                    AO_coeff_epsilon=self.AO_coeff_epsilon,
                )
                _write_eris(self.Fobjs, eris, file_eri)
                eri = None
            elif int_transform == "sparse-DF-S-screening-shared-ijP-g":
                ensure(bool(self.auxbasis), "`auxbasis` has to be defined.")
                eris = _transform_sparse_DF_S_screening_shared_ijP_and_g(
                    self.mf,
                    self.Fobjs,
                    self.all_fragment_MO_TA,
                    auxbasis=self.auxbasis,
                    MO_coeff_epsilon=self.MO_coeff_epsilon,
                    AO_coeff_epsilon=self.AO_coeff_epsilon,
                )
                _write_eris(self.Fobjs, eris, file_eri)
                eri = None

            elif int_transform == "sparse-DF-S-screening-shared-ijP-g-faster":
                ensure(bool(self.auxbasis), "`auxbasis` has to be defined.")
                eris = _transform_sparse_DF_S_screening_shared_ijP_and_g_fast(
                    self.mf,
                    self.Fobjs,
                    self.all_fragment_MO_TA,
                    auxbasis=self.auxbasis,
                    MO_coeff_epsilon=self.MO_coeff_epsilon,
                    AO_coeff_epsilon=self.AO_coeff_epsilon,
                )
                _write_eris(self.Fobjs, eris, file_eri)
                eri = None

            elif int_transform == "sparse-DF-shared-ijP":
                ensure(bool(self.auxbasis), "`auxbasis` has to be defined.")
                eris = _transform_sparse_DF_use_shared_ijP(
                    self.mf, self.Fobjs, self.all_fragment_MO_TA, auxbasis=self.auxbasis
                )
                _write_eris(self.Fobjs, eris, file_eri)
                eri = None
            else:
                assert_never(int_transform)
        else:
            eri = None
        if settings.PRINT_LEVEL >= 10:
            print(eritransform_timer.str_elapsed())

        for fobjs_ in self.Fobjs:
            # Process each fragment
            eri = array(file_eri.get(fobjs_.dname))
            _ = fobjs_.get_nsocc(self.S, self.C, self.Nocc, ncore=self.ncore)

            assert fobjs_.TA is not None
            fobjs_.h1 = multi_dot((fobjs_.TA.T, self.hcore, fobjs_.TA))

            if not restart:
                eri = ao2mo.restore(8, eri, fobjs_.nao)

            fobjs_.cons_fock(self.hf_veff, self.S, self.hf_dm, eri_=eri)

            fobjs_.heff = zeros_like(fobjs_.h1)
            fobjs_.scf(fs=True, eri=eri)

            assert fobjs_.h1 is not None and fobjs_.nsocc is not None
            fobjs_.dm0 = 2.0 * (
                fobjs_._mo_coeffs[:, : fobjs_.nsocc]
                @ fobjs_._mo_coeffs[:, : fobjs_.nsocc].conj().T
            )

            if compute_hf:
                fobjs_.update_ebe_hf()  # Updates fragment HF energy.
                E_hf += fobjs_.ebe_hf

        if not restart:
            file_eri.close()

        if compute_hf:
            self.ebe_hf = E_hf + self.enuc + self.E_core
            hf_err = self.hf_etot - self.ebe_hf
            print(f"HF-in-HF error                 :  {hf_err:>.4e} Ha")
            if abs(hf_err) > 1.0e-5:
                warn("Large HF-in-HF energy error")

        couti = 0
        for fobj in self.Fobjs:
            fobj.udim = couti
            couti = fobj.set_udim(couti)

    def oneshot(
        self,
        solver: Solvers = "MP2",
        use_cumulant: bool = True,
        nproc: int = 1,
        ompnum: int = 4,
        solver_args: UserSolverArgs | None = None,
    ) -> None:
        """
        Perform a one-shot bootstrap embedding calculation.

        Parameters
        ----------
        solver :
            High-level quantum chemistry method, by default 'MP2'. 'CCSD', 'FCI',
            and variants of selected CI are supported.
        use_cumulant :
            Whether to use the cumulant energy expression, by default True.
        nproc :
            Number of processors for parallel calculations, by default 1.
            If set to >1, multi-threaded parallel computation is invoked.
        ompnum :
            Number of OpenMP threads, by default 4.
        """
        oneshot_timer = Timer("Time to perform one-shot BE")
        if nproc == 1:
            rets = be_func(
                None,
                self.Fobjs,
                self.Nocc,
                solver,
                self.enuc,
                eeval=True,
                scratch_dir=self.scratch_dir,
                solver_args=solver_args,
                use_cumulant=use_cumulant,
                return_vec=False,
            )
        else:
            rets = be_func_parallel(
                None,
                self.Fobjs,
                self.Nocc,
                solver,
                self.enuc,
                eeval=True,
                nproc=nproc,
                ompnum=ompnum,
                scratch_dir=self.scratch_dir,
                solver_args=solver_args,
                use_cumulant=use_cumulant,
                return_vec=False,
            )

        print("-----------------------------------------------------", flush=True)
        print("             One Shot BE ", flush=True)
        print("             Solver : ", solver, flush=True)
        print("-----------------------------------------------------", flush=True)
        print(flush=True)
        if use_cumulant:
            print_energy_cumulant(
                rets[0], rets[1][1], rets[1][0] + rets[1][2], self.ebe_hf
            )
            self.ebe_tot = rets[0] + self.ebe_hf
        else:
            print_energy_noncumulant(
                rets[0], rets[1][0], rets[1][2], rets[1][1], self.ebe_hf, self.enuc
            )
            self.ebe_tot = rets[0] + self.enuc + self.ebe_hf
        if settings.PRINT_LEVEL >= 10:
            print(oneshot_timer.str_elapsed())

    def update_fock(self, heff: list[Matrix[floating]] | None = None) -> None:
        """
        Update the Fock matrix for each fragment with the effective Hamiltonian.

        Parameters
        ----------
        heff : list of numpy.ndarray, optional
            List of effective Hamiltonian matrices for each fragment, by default None.
        """
        if heff is None:
            for fobj in self.Fobjs:
                assert fobj.fock is not None and fobj.heff is not None
                fobj.fock += fobj.heff
        else:
            for idx, fobj in enumerate(self.Fobjs):
                assert fobj.fock is not None
                fobj.fock += heff[idx]

    def write_heff(self, heff_file: str = "bepotfile.h5") -> None:
        """
        Write the effective Hamiltonian to a file.

        Parameters
        ----------
        heff_file : str, optional
            Path to the file to store effective Hamiltonian, by default 'bepotfile.h5'.
        """
        with h5py.File(heff_file, "w") as filepot:
            for fobj in self.Fobjs:
                assert fobj.heff is not None
                print(fobj.heff.shape, fobj.dname, flush=True)
                filepot.create_dataset(fobj.dname, data=fobj.heff)

    def read_heff(self, heff_file="bepotfile.h5"):
        """
        Read the effective Hamiltonian from a file.

        Parameters
        ----------
        heff_file : str, optional
            Path to the file storing effective Hamiltonian, by default 'bepotfile.h5'.
        """
        with h5py.File(heff_file, "r") as filepot:
            for fobj in self.Fobjs:
                fobj.heff = filepot.get(fobj.dname)


def initialize_pot(n_frag, relAO_per_edge):
    """
    Initialize the potential array for bootstrap embedding.

    This function initializes a potential array for a given number of fragments
    (:python:`n_frag`) and their corresponding edge indices
    (:python:`relAO_per_edge`).
    The potential array is initialized with zeros for each pair of edge site indices
    within each fragment, followed by an
    additional zero for the global chemical potential.

    Parameters
    ----------
    n_frag: int
        Number of fragments.
    relAO_per_edge: list of list of list of int
        List of edge indices for each fragment. Each element is a list of lists,
        where each sublist contains the indices of edge sites for a particular fragment.

    Returns
    -------
    list of float
        Initialized potential array with zeros.
    """
    pot_ = []

    if relAO_per_edge:
        for I in range(n_frag):
            for i in relAO_per_edge[I]:
                for j in range(len(i)):
                    for k in range(len(i)):
                        if j > k:
                            continue
                        pot_.append(0.0)

    pot_.append(0.0)
    return pot_
