# Author(s): Oinam Romesh Meitei

import logging
import os
import pickle
from typing import Dict, Final, List, Literal, TypeAlias
from warnings import warn

import h5py
import numpy as np
from attrs import define
from numpy import (
    allclose,
    array,
    concatenate,
    diag,
    diag_indices,
    einsum,
    eye,
    float64,
    floating,
    hstack,
    isnan,
    load,
    ndarray,
    save,
    shape,
    sqrt,
    where,
    zeros,
    zeros_like,
)
from numpy.linalg import eigh, multi_dot, svd
from pyscf import ao2mo, scf
from pyscf.gto import Mole
from scipy.optimize import linear_sum_assignment
from typing_extensions import assert_never

from quemb.molbe.be_parallel import be_func_parallel
from quemb.molbe.eri_onthefly import integral_direct_DF
from quemb.molbe.eri_sparse_DF import (
    transform_sparse_DF_integral_cpu,
)
from quemb.molbe.fragment import FragPart
from quemb.molbe.lo import (
    IAO_LocMethods,
    LocMethods,
    get_iao,
    get_loc,
    get_pao,
    get_xovlp,
    remove_core_mo,
)
from quemb.molbe.misc import print_energy_cumulant, print_energy_noncumulant
from quemb.molbe.opt import BEOPT
from quemb.molbe.pfrag import Frags, union_of_frag_MOs_and_index
from quemb.molbe.solver import Solvers, UserSolverArgs, be_func
from quemb.shared.external.eom_qchem_parser import dyson_parser
from quemb.shared.external.lo_helper import (
    get_aoind_by_atom,
    reorder_by_atom_,
)
from quemb.shared.external.optqn import (
    get_be_error_jacobian as _ext_get_be_error_jacobian,
)
from quemb.shared.helper import copy_docstring, ensure, ncore_, timer, unused
from quemb.shared.manage_scratch import WorkDir
from quemb.shared.typing import Matrix, PathLike

IntTransforms: TypeAlias = Literal[
    "in-core",
    "out-core-DF",
    "int-direct-DF",
    "sparse-DF",
    "sparse-DF-gpu",
    "on-fly-sparse-DF",
    "on-fly-sparse-DF-gpu",
]
r"""Literal type describing allowed transformation strategies.

- :python:`"in-core"`: Use a dense representation of integrals
  in memory without density fitting (DF) and transform in-memory.

- :python:`"out-core-DF"`: Use a dense, DF representation of integrals.
  The DF integrals :math:`(\mu, \nu | P)` are stored on disc.

- :python:`"int-direct-DF"`: Use a dense, DF representation of integrals.
  The required DF integrals :math:`(\mu, \nu | P)` are computed and fitted
  on-demand for each fragment.

- :python:`"sparse-DF"`, :python:`"sparse-DF-gpu"`,
  :python:`"on-fly-sparse-DF"`, and :python:`"on-fly-sparse-DF-gpu"`:
  Use a sparse, DF representation of integrals.
  The :python:`"-gpu"` versions use GPU and require CUDABlas.
  The :python:`"on-fly-"` versions use less memory but perform more
  on-the-fly computations.
"""


logger = logging.getLogger(__name__)


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


class BE:
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
    lo_method :
        Method for orbital localization, default is "lowdin".
    """

    @timer.timeit
    def __init__(
        self,
        mf: scf.hf.SCF,
        fobj: FragPart[Mole],
        *,
        eri_file: PathLike = "eri_file.h5",
        lo_method: LocMethods = "lowdin",
        iao_loc_method: IAO_LocMethods = "lowdin",
        lo_bath_post_schmidt: Literal["cholesky", "ER", "PM", "boys"] | None = None,
        pop_method: str | None = None,
        restart: bool = False,
        restart_file: PathLike = "storebe.pk",
        nproc: int = 1,
        ompnum: int = 4,
        thr_bath: float = 1.0e-10,
        scratch_dir: WorkDir | None = None,
        int_transform: IntTransforms = "in-core",
        auxbasis: str | None = None,
        MO_coeff_epsilon: float = 1e-5,
        AO_coeff_epsilon: float = 1e-10,
        re_eval_HF: bool = False,
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
            Method for orbital localization, by default "lowdin".
        iao_loc_method :
            Method for IAO localization, by default "lowdin"
        lo_method_bath_post_schmidt :
            If not :python:`None`, then perform a localization of the bath orbitals
            **after** the Schmidt decomposition.
        pop_method :
            Method for calculating orbital population, by default 'meta-lowdin'
            See pyscf.lo for more details and options
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
            - :python:`"sparse-DF"`, :python:`"sparse-DF-gpu"`,
              :python:`"on-fly-sparse-DF"`, and :python:`"on-fly-sparse-DF-gpu"`.
              Use a sparse, DF representation of integrals.
              The :python:`"-gpu"` versions use GPU and require CUDABlas,
              the :python:`"on-fly-"` versions use less memory, but do more
              on the fly computations.

        auxbasis :
            Auxiliary basis for density fitting, by default None
            (uses default auxiliary basis defined in PySCF).
            Only relevant for :python:`int_transform in {"int-direct-DF", "sparse-DF"}`.
        MO_coeff_epsilon:
            The cutoff value of the absolute overlap
            :math:`\int |\phi_i| |\varphi_{\mu}|`
            when a MO coefficient :math:`i` and an AO coefficient
            :math:`\mu` are considered to be connected for sparsity screening.
            Smaller value means less screening.
        AO_coeff_epsilon:
            The cutoff value of the absolute overlap
            :math:`\int |\varphi_{\mu}| |\varphi_{\nu}|`
            when two AO coefficient :math:`\mu, \nu`
            are considered to be connected for sparsity screening.
            Here the absolute overlap matrix is used.
            Smaller value means less screening.
        re_eval_HF:
            Re-evaluate the Fock matrix from the given MO-coefficients to determine
            the HF-in-HF error.
            This is False by default and has no relevance for "normal", unscreened
            Hartree-Fock calculations.
            However, if the Hartree-Fock was obtained from methods that never build
            one global Fock matrix in the traditional sense, such as ORCA's RIJCOSX,
            then the energy obtained by building the Fock matrix from the
            MO coefficients can actually differ from the reported HF energy.
        """
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

        # We hardcode it here, because in practice it is far better
        # to use the parallelization in each fragment.
        self.n_threads_integral_transform: Final = 1

        self.unrestricted = False
        self.nproc = nproc
        self.ompnum = ompnum
        self.integral_transform = int_transform
        self.auxbasis = auxbasis
        self.thr_bath = thr_bath

        # Alexa: this is a temporary solution for the EOM-BE code
        # TO DO: set up better QuEmb - QChem - QuEmb workflow

        # save full system HF Fock matrix to file

        if not os.path.exists("files_EOM"):
            os.makedirs("files_EOM")

        with open("files_EOM/full_syst_fock.npy", "wb") as f:
            save(f, mf.get_fock())

        # end Alexa

        # Fragment information from fobj
        self.fobj = fobj
        if not ((lo_method == "IAO") == (self.fobj.iao_valence_basis is not None)):
            raise ValueError(
                "If lo_method = 'IAO', "
                "then the fobj has to have an iao_valence_basis defined.\n"
                "Likewise, if lo_method != 'IAO', then fobj must not have "
                "an iao_valence_basis defined."
            )

        if int_transform[:9] == "sparse-DF" and (lo_method != "IAO"):
            warn(
                r'Sparse integral screening (int_transform="sparse-DF*") '
                'works best if lo_method="IAO".\n'
                "If you anyway used a minimal basis, you can ignore this warning."
            )

        self.lo_bath_post_schmidt: Literal["cholesky", "ER", "PM", "boys"] | None = (
            lo_bath_post_schmidt
        )

        self.ebe_hf = 0.0
        self.ebe_tot = 0.0

        self.mf = mf
        self.re_eval_HF = re_eval_HF

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
                fobj=fobj,
                iao_loc_method=iao_loc_method,
                iao_valence_only=fobj.iao_valence_only,
                pop_method=pop_method,
            )

            if fobj.iao_valence_only and lo_method == "IAO":
                self.Ciao_pao = self.localize(
                    lo_method,
                    fobj=fobj,
                    iao_loc_method=iao_loc_method,
                    iao_valence_only=False,
                    pop_method=pop_method,
                    hstack=True,
                    save=False,
                )

        if not restart:
            # Initialize fragments and perform initial calculations
            self.initialize(mf._eri, restart=False, int_transform=int_transform)
        else:
            self.initialize(None, restart=True, int_transform=int_transform)

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
        if not only_rdm2:
            rdm1AO = zeros((nao, nao))
        if not only_rdm1:
            rdm2AO = zeros((nao, nao, nao, nao))

        for fobjs in self.Fobjs:
            if not only_rdm1:
                rdm2 = fobjs.rdm2__.copy()
            if return_RDM2:
                # Adjust the one-particle reduced density matrix (RDM1)
                drdm1 = fobjs.rdm1__.copy()
                drdm1[diag_indices(fobjs.nsocc)] -= 2.0

                # Compute the two-particle reduced density matrix (RDM2) and subtract
                #   non-connected component
                dm_nc = einsum(
                    "ij,kl->ijkl", drdm1, drdm1, dtype=float64, optimize=True
                ) - 0.5 * einsum(
                    "ij,kl->iklj", drdm1, drdm1, dtype=float64, optimize=True
                )
                rdm2 -= dm_nc

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
                    rdm2,
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
                        dtype=float64,
                        optimize=True,
                    )
                    - einsum(
                        "ij,kl->iklj",
                        rdm1AO,
                        rdm1AO,
                        dtype=float64,
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

    def compute_overlap_dyson(self, ref_dyson, frag_dyson, n_ex, extra=3):
        """
        Compute overlaps between two Dyson orbitals
        (currently in the AO basis), used for state targetting.

        Parameters
        ----------
        ref_dyson : numpy.ndarray
            Reference Dyson orbital (currently chosen as fragment 0).
        frag_dyson : numpy.ndarray
            Current fragment Dyson orbital.
        n_ex : int
            Number of excited states considered.
        extra : int
            Number of additional excited states being included.
            Helpful to solve intruder state issues. Default: 3

        Returns
        -------
        ovlp : numpy.ndarray
            Overlap matrix between reference (rows) and current fragments (columns).
        """

        ovlp = zeros((n_ex, n_ex + extra))

        for i in range(n_ex):
            for j in range(n_ex + extra):
                # no need to normalize these - Dyson orbs. don't have norm 1
                ovlp[i, j] = ref_dyson[i, :] @ frag_dyson[j, :]
        return ovlp

    def match_fragments(self, ref_dyson, frag_dyson, n_ex, extra=3, threshold=0.6):
        """
        Hungarian matching + phase alignment for
        left Dyson orbitals (in AO basis)

        Parameters
        ----------
        ref_dyson : numpy.ndarray
            Reference Dyson orbital (currently chosen as fragment 0).
        frag_dyson : numpy.ndarray
            Current fragment Dyson orbital.
        n_ex : int
            Number of excited states considered.
        extra : int
            Number of additional excited states being included.
            Helpful to solve intruder state issues. Default: 3
        threshold : float
            Overlap threshold above which two excited states across the
            reference and fragment are considered to "match". Default: 0.6.

        Returns
        -------
        mapping : Dict[int, int]
            Map of reference excitation (row index)
            to current fragment excitation (column index).
        ovlp : numpy.ndarray
            Overlap matrix between reference and current fragments.
        excluded : List[int]
            List of excitations excluded from the current fragment
            due to no match being found.
        """

        excluded = []

        # ovlp: -1 to 1
        ovlp = self.compute_overlap_dyson(ref_dyson, frag_dyson, n_ex, extra)

        cost = -abs(ovlp)

        # if NaN in overlap matrix - assign a very large positive number
        # too costly to match this pair
        cost[isnan(cost)] = 1e6

        # if mapping[i]=j => reference excitation i matches fragment excitation j
        row_ind, col_ind = linear_sum_assignment(cost)
        mapping = {row: col for row, col in zip(row_ind, col_ind)}
        print(mapping)

        for i, j in mapping.items():
            ov = ovlp[i, j]
            mag = abs(ov)
            if mag < threshold:
                print(f"WARNING weak match: ref {i} -> frag {j} |overlap|={mag}")
                print("SKIP MATCHING")
                # found no good match for this excitation pair => exclude
                excluded.append(j)
        return mapping, ovlp, excluded

    def hij_full(self, n_ex):
        """Compute hij from Dyson orbitals - for averaged excited BE purposes.

        ---Alexa: current strategy as of 10/30/2025
        - now with added environment contributions

        Parameters
        ----------
        n_ex: int
            Number of excited states.

        Returns
        -------
        hijAO : numpy.ndarray
            Effective Hamiltonian in AO basis, formed from Dyson orbitals.
        delta_hijAO : numpy.ndarray
            Environment correction term in AO basis.
        ip_full : numpy.ndarray
            Full-system IPs.
        dyson_ip_full : numpy.ndarray
            Full system Dyson orbitals corresponding to Koopman's-like excitations.
        hijMO : numpy.ndarray
            Effective Hamiltonian in MO basis, formed from Dyson orbitals.
        delta_hijMO : numpy.ndarray
            Environment correction term in MO basis.
        """
        # Notation: _mo: fragment MO basis
        # MO: full system MO basis

        ha_to_ev = 27.2114

        # extra = compute additional excitations in case of intruder states
        extra = 0
        extra_koop = 0

        # Copy the full system MO coefficients
        C_MO = self.C.copy()
        nao = C_MO.shape[0]

        # in AO basis
        hijAO = zeros((nao, nao))
        delta_hijAO = zeros((nao, nao))

        # full system environment correction
        # simple Koopman's theorem IP reference
        # dyson_ip_full: in AO basis
        # dyson_ip_full_MO: in MO basis

        dyson_ip_full = zeros((n_ex + extra_koop, nao))
        dyson_ip_full_MO = zeros((n_ex + extra_koop, nao))
        ip_full = zeros(n_ex + extra_koop)

        for i in range(n_ex + extra_koop):
            ip_full[i] = self.mo_energy[self.Nocc - 1 - i]
            dyson_ip_full_MO[i, self.Nocc - 1 - i] = 1
            dyson_ip_full[i, :] = C_MO[:, self.Nocc - 1 - i]

        ip_full = (-ip_full) * ha_to_ev

        # parse Dyson info from Q-Chem outputs

        for frag_number, fobjs in enumerate(self.Fobjs):
            output = "qchem_fragment_" + str(frag_number) + "/eom.out"
            dyson_parser(fobjs, output, n_ex + extra)

        # compute overlaps and reorder excitations
        print("EOM-IP FRAGMENT REORDERING")
        print("Compute overlaps between Dyson orbital fragments: ")

        # Reference Dyson left orbitals - fragment 0:
        ref = self.Fobjs[0]
        # ref_dyson=ref.dyson_left
        ref_dyson = ref.dyson_ao

        all_mappings: List[Dict[int, int]] = []
        all_overlaps: List[ndarray] = []

        for frag_idx, fobj in enumerate(self.Fobjs):
            # frag_dyson = fobj.dyson_left
            frag_dyson = fobj.dyson_ao

            mapping, ovlp, excluded = self.match_fragments(
                ref_dyson, frag_dyson, n_ex, extra
            )

            all_mappings.append(mapping)
            all_overlaps.append(ovlp)
            print(f"[hij_full] fragment {frag_idx} mapping to reference:")
            for ri, fj in mapping.items():
                print(f"  ref {ri} -> frag {fj} |ov|={abs(ovlp[ri, fj]):.6f}")

            # reorder excitations to match fragment 0 (reference)
            new_order = [mapping[i] for i in range(n_ex) if mapping[i] not in excluded]
            print("EXCLUDE: ")
            print(excluded)
            print(new_order)

            # fobj.dyson_left: in fragment SO basis
            fobj.dyson_left = fobj.dyson_left[new_order, :]
            fobj.dyson_right = fobj.dyson_right[new_order, :]
            fobj.ex_e = fobj.ex_e[new_order]

        # build fragment contributions

        for frag_number, fobjs in enumerate(self.Fobjs):
            n_mo_full = shape(fobjs.TA)[0]
            occ_tot = self.Nocc  # full system
            SO_tot = shape(fobjs.TA)[1]
            SO_occ = fobjs.nsocc  # Schmidt space

            # fragment environment correction - simple Koopman's theorem IP

            dyson_ip_frag = zeros((n_ex, nao))

            dyson_ip_frag_ao = zeros((n_ex, nao))

            ip_frag = zeros(n_ex)
            delta_ex = zeros(n_ex)

            for i in range(n_ex):
                ### TO DO: sometimes dyson orbital doesn't correspond to
                # an excitation from HOMO->infty!!
                # what can we do about double excitations?
                idx_guess = occ_tot - 1 - i

                ip_frag[i] = -fobjs._mf.mo_energy[SO_occ - 1 - i] * ha_to_ev
                dyson_ip_frag[i, idx_guess] = -1
                delta_ex[i] = ip_frag[i]

                orbs = fobjs.TA @ fobjs.mo_coeffs

                dyson_ip_frag_ao[i, :] = orbs[:, SO_occ - 1 - i]

            exc = diag(fobjs.ex_e[:n_ex])

            delta_ex = diag(delta_ex)

            env_occ = occ_tot - SO_occ
            env_virt = n_mo_full - SO_tot - env_occ

            hij_mo = (
                fobjs.dyson_left[:, env_occ : n_mo_full - env_virt].T
                @ exc
                @ fobjs.dyson_right[:, env_occ : n_mo_full - env_virt]
            )

            # environment correction!
            delta_hij_mo = (
                dyson_ip_frag[:, env_occ : n_mo_full - env_virt].T
                @ delta_ex
                @ dyson_ip_frag[:, env_occ : n_mo_full - env_virt]
            )

            # projection matrix
            cind = [fobjs.fsites[i] for i in fobjs.efac[1]]
            Pc_ = (
                fobjs.TA.T
                @ self.S
                @ self.W[:, cind]
                @ self.W[:, cind].T
                @ self.S
                @ fobjs.TA
            )

            # Compute hij in the SO basis
            # and transform to AO basis
            hij_so = fobjs.mo_coeffs @ hij_mo @ fobjs.mo_coeffs.T
            hij_center = Pc_ @ hij_so
            hij_ao = fobjs.TA @ hij_center @ fobjs.TA.T  ###!!!just center!
            hijAO += hij_ao
            fobjs.hij_mo = hij_mo

            delta_hij_so = fobjs.mo_coeffs @ delta_hij_mo @ fobjs.mo_coeffs.T
            delta_hij_center = Pc_ @ delta_hij_so
            delta_hij_ao = fobjs.TA @ delta_hij_center @ fobjs.TA.T
            delta_hijAO += delta_hij_ao

            # Symmetrize hij??
            # hijAO = (hijAO + hijAO.T) / 2.0

            # Transform to the MO basis if needed
            hijMO = self.C.T @ self.S @ hijAO @ self.S @ self.C
            delta_hijMO = self.C.T @ self.S @ delta_hijAO @ self.S @ self.C

            ###TO DO: also need to match full system ip_full
            # and eigvecs_mf (from delta_hijAO) to eigvecs from hijAO!!!

        return hijAO, delta_hijAO, ip_full, dyson_ip_full, hijMO, delta_hijMO

    @timer.timeit
    def optimize(
        self,
        solver: Solvers = "CCSD",
        method: str = "QN",
        only_chem: bool = False,
        use_cumulant: bool = True,
        conv_tol: float = 1.0e-6,
        relax_density: bool = False,
        jac_solver: Literal["HF", "MP2", "CCSD", "Numerical"] = "HF",
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
            High-level solver for the fragment, by default "CCSD"
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
            if jac_solver == "Numerical":
                if only_chem:
                    J0 = self.get_be_error_jacobian_numerical(
                        only_chem, solver, relax_density, solver_args, use_cumulant
                    )
                else:
                    raise NotImplementedError(
                        "Numerical Jacobian is only implemented for only_chem=True"
                    )
            else:
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
    def get_be_error_jacobian(self, jac_solver: str = "HF") -> Matrix[float64]:
        return _ext_get_be_error_jacobian(self.fobj.n_frag, self.Fobjs, jac_solver)

    def get_be_error_jacobian_numerical(
        self,
        only_chem: bool,
        solver: Solvers,
        relax_density: bool,
        solver_args: UserSolverArgs | None,
        use_cumulant: bool,
    ) -> Matrix[float64]:
        """
        Obtain the Jacobian matrix for BE Optimization using numerical differentiation.
        (First-order Central Finite Differences)
        Note that this function is only implemented for the case
        where :python:`only_chem=True`.
        """
        step_size = 1e-6  # from frankenstein

        def be_func_err(x: list[float] | None) -> float:
            if self.nproc == 1:
                return be_func(
                    x,
                    self.Fobjs,
                    self.Nocc,
                    solver,
                    self.enuc,
                    only_chem=only_chem,
                    relax_density=relax_density,
                    scratch_dir=self.scratch_dir,
                    solver_args=solver_args,
                    use_cumulant=use_cumulant,
                    eeval=False,
                    return_vec=True,
                )[1]
            else:  # parallel
                return be_func_parallel(
                    x,
                    self.Fobjs,
                    self.Nocc,
                    solver,
                    self.enuc,
                    only_chem=only_chem,
                    nproc=self.nproc,
                    ompnum=self.ompnum,
                    relax_density=relax_density,
                    scratch_dir=self.scratch_dir,
                    solver_args=solver_args,
                    use_cumulant=use_cumulant,
                    eeval=False,
                    return_vec=True,
                )[1]

        if only_chem:
            return array(
                [
                    (be_func_err([step_size]) - be_func_err([-step_size]))
                    / (2 * step_size)
                ]
            )
        else:
            raise NotImplementedError(
                "Numerical Jacobian is only implemented for only_chem=True"
            )

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

    @timer.timeit
    def _eri_transform(
        self,
        int_transform: IntTransforms,
        eri_: Matrix[np.floating] | None,
        file_eri: h5py.File,
    ):
        """
        Transforms electron repulsion integrals (ERIs) for each fragment
        and stores them in a file.

        Transformation strategy follows a decision tree:

        1. If full (ij|kl) ERIs are available:
           - Use in-core ``ao2mo`` transformation.
        2. Else, if (ij|P) intermediates from density fitting are available:
           - Use out-of-core ``ao2mo`` transformation with saved (ij|P).
        3. Else, if ``integral_direct_DF`` is requested:
           - Use on-the-fly density-fitting integral evaluation.
        4. Else, for a sparse, DF representation of integrals:
           - Use ``sparse-DF`` for ``C++`` implementation.
           - Use ``sparse-DF-gpu`` for ``C++`` + ``CUDDA`` implementation.


        Parameters
        ----------
        int_transform : The transformation strategy.
        eri_ : The ERIs for the molecule.
        file_eri : The output file where transformed ERIs are stored.
        """
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
            for I in range(self.fobj.n_frag):
                eri = self.mf.with_df.ao2mo(self.Fobjs[I].TA, compact=True)
                file_eri.create_dataset(self.Fobjs[I].dname, data=eri)
        elif int_transform == "int-direct-DF":
            ensure(bool(self.auxbasis), "`auxbasis` has to be defined.")
            integral_direct_DF(self.mf, self.Fobjs, file_eri, auxbasis=self.auxbasis)
        elif int_transform == "sparse-DF":
            ensure(bool(self.auxbasis), "`auxbasis` has to be defined.")
            transform_sparse_DF_integral_cpu(
                self.mf,
                self.Fobjs,
                auxbasis=self.auxbasis,
                file_eri_handler=file_eri,
                MO_coeff_epsilon=self.MO_coeff_epsilon,
                AO_coeff_epsilon=self.AO_coeff_epsilon,
                n_threads=self.n_threads_integral_transform,
                precompute_P_mu_nu=True,
            )
        elif int_transform == "on-fly-sparse-DF":
            ensure(bool(self.auxbasis), "`auxbasis` has to be defined.")
            transform_sparse_DF_integral_cpu(
                self.mf,
                self.Fobjs,
                auxbasis=self.auxbasis,
                file_eri_handler=file_eri,
                MO_coeff_epsilon=self.MO_coeff_epsilon,
                AO_coeff_epsilon=self.AO_coeff_epsilon,
                n_threads=self.n_threads_integral_transform,
                precompute_P_mu_nu=False,
            )
        elif int_transform == "sparse-DF-gpu":
            from quemb.molbe.eri_sparse_DF import (  # noqa: PLC0415
                transform_sparse_DF_integral_gpu,
            )

            ensure(bool(self.auxbasis), "`auxbasis` has to be defined.")
            transform_sparse_DF_integral_gpu(
                self.mf,
                self.Fobjs,
                auxbasis=self.auxbasis,
                file_eri_handler=file_eri,
                MO_coeff_epsilon=self.MO_coeff_epsilon,
                AO_coeff_epsilon=self.AO_coeff_epsilon,
                n_threads=self.n_threads_integral_transform,
                precompute_P_mu_nu=True,
            )
        elif int_transform == "on-fly-sparse-DF-gpu":
            from quemb.molbe.eri_sparse_DF import (  # noqa: PLC0415
                transform_sparse_DF_integral_gpu,
            )

            ensure(bool(self.auxbasis), "`auxbasis` has to be defined.")
            transform_sparse_DF_integral_gpu(
                self.mf,
                self.Fobjs,
                auxbasis=self.auxbasis,
                file_eri_handler=file_eri,
                MO_coeff_epsilon=self.MO_coeff_epsilon,
                AO_coeff_epsilon=self.AO_coeff_epsilon,
                n_threads=self.n_threads_integral_transform,
                precompute_P_mu_nu=False,
            )
        else:
            assert_never(int_transform)

    @timer.timeit
    def _initialize_fragments(self, file_eri: h5py.File, restart: bool):
        """
        Processes all molecular fragments by constructing their Fock matrices,
        performing SCF, and computing fragment Hartree–Fock (HF) energies.

        This includes:

        - Loading and changing the symmetry of the ERIs (in-core or restored format).
        - Constructing 1-electron Hamiltonians via basis transformations.
        - Running fragment-level SCF calculations.
        - Building initial density matrices for subsequent scf
        - Computing and accumulating fragment HF energies
        - Verifying HF-in-HF energy consistency.

        Parameters
        ----------
        file_eri : h5py.File
            HDF5 file containing fragment ERIs.
        restart : bool
            If True, skips ERI transformation and file closure.
        """

        E_hf = 0.0
        for fobjs_ in self.Fobjs:
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

            fobjs_.update_ebe_hf()  # Updates fragment HF energy.
            E_hf += fobjs_.ebe_hf
        self.ebe_hf = E_hf + self.enuc + self.E_core
        hf_err = self.hf_etot - self.ebe_hf
        print(f"HF-in-HF error                 :  {hf_err:>.4e} Ha")
        if abs(hf_err) > 1.0e-5:
            warn("Large HF-in-HF energy error")

        if self.re_eval_HF:
            hf_err = self.mf.energy_tot() - self.ebe_hf
            print(f"HF-in-HF error (re-eval global):  {hf_err:>.4e} Ha")
            if abs(hf_err) > 1.0e-5:
                warn("Large HF-in-HF energy error")

    @timer.timeit
    def initialize(
        self,
        eri_,
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
        restart : bool, optional
            Whether to restart from a previous calculation, by default False.
        int_transfrom :
            Which integral transformation to perform.
        """
        for I in range(self.fobj.n_frag):
            fobjs_ = self.fobj.to_Frags(I, eri_file=self.eri_file)
            fobjs_.sd(self.W, self.lmo_coeff, self.Nocc, thr_bath=self.thr_bath)

            self.Fobjs.append(fobjs_)

        self.all_fragment_MO_TA, frag_TA_index_per_frag = union_of_frag_MOs_and_index(
            self.Fobjs, self.mf.mol.intor("int1e_ovlp"), epsilon=1e-10
        )
        for fobj, frag_TA_offset in zip(self.Fobjs, frag_TA_index_per_frag):
            fobj.frag_TA_offset = frag_TA_offset

        if self.lo_bath_post_schmidt is not None:
            for frag in self.Fobjs:
                frag.TA[:, frag.n_f :] = get_loc(
                    self.mf.mol,
                    frag.TA[:, frag.n_f :],
                    method=self.lo_bath_post_schmidt,
                )

        if not restart:
            file_eri = h5py.File(self.eri_file, "w")
            self._eri_transform(int_transform, eri_, file_eri)

        self._initialize_fragments(file_eri, restart)

        if not restart:
            file_eri.close()

        couti = 0
        for fobj in self.Fobjs:
            fobj.udim = couti
            couti = fobj.set_udim(couti)

    @timer.timeit
    def oneshot(
        self,
        solver: Solvers = "CCSD",
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
            High-level quantum chemistry method, by default 'CCSD'. 'CCSD', 'FCI',
            and variants of selected CI are supported.
        use_cumulant :
            Whether to use the cumulant energy expression, by default True.
        nproc :
            Number of processors for parallel calculations, by default 1.
            If set to >1, multi-threaded parallel computation is invoked.
        ompnum :
            Number of OpenMP threads, by default 4.
        """
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

    def localize(
        self,
        lo_method: LocMethods,
        fobj: FragPart,
        iao_loc_method: IAO_LocMethods = "lowdin",
        iao_valence_only: bool = False,
        pop_method: str | None = None,
        init_guess: Matrix[np.floating] | None = None,
        hstack: bool = False,
        save: bool = True,
    ) -> None | Matrix[np.float64]:
        """Molecular orbital localization

        Performs molecular orbital localization computations. For large basis,
        IAO is recommended augmented with PAO orbitals.

        NOTE: For molecular systems, with frozen core, the core and valence are
        localized TOGETHER. This is not the case for periodic systems.

        Parameters
        ----------
        lo_method :
            Method for orbital localization. Supports
            "lowdin" (Löwdin or symmetric orthogonalization),
            "boys" (Foster-Boys),
            "PM" (Pipek-Mezey", and
            "ER" (Edmiston-Rudenberg).
            By default "lowdin"
        fobj :
        iao_loc_method:
            Name of localization method in quantum chemistry for the IAOs and PAOs.
            Options include "lowdin", "boys", 'PM', 'ER' (as documented in PySCF).
            Default is "lowdin".
            If not using lowdin, we suggest using 'PM', as it is more robust than 'boys'
            localization and less expensive than 'ER'
        iao_valence_only : bool
            If this option is set to True, all calculation will be performed in the
            valence basis in the IAO partitioning. Default is False.
            This is an experimental feature: the returned energy is not accurate
        """
        if lo_method == "lowdin":
            es_, vs_ = eigh(self.S)
            edx = es_ > 1.0e-15
            self.W = vs_[:, edx] / sqrt(es_[edx]) @ vs_[:, edx].T
            if self.frozen_core:
                if self.unrestricted:
                    P_core = [
                        eye(self.W.shape[0]) - (self.P_core[s] @ self.S) for s in [0, 1]
                    ]
                    C_ = P_core @ self.W
                    unr_Cpop = [multi_dot((C_[s].T, self.S, C_[s])) for s in [0, 1]]
                    unr_Cpop = [diag(unr_Cpop[s]) for s in [0, 1]]
                    unr_no_core_idx = [where(unr_Cpop[s] > 0.7)[0] for s in [0, 1]]
                    C_ = [C_[s][:, unr_no_core_idx[s]] for s in [0, 1]]
                    S_ = [multi_dot((C_[s].T, self.S, C_[s])) for s in [0, 1]]
                    unr_W_ = []
                    for s in [0, 1]:
                        es_, vs_ = eigh(S_[s])
                        s_ = sqrt(es_)
                        s_ = diag(1.0 / s_)
                        unr_W_.append(multi_dot((vs_, s_, vs_.T)))
                    self.W = [C_[s] @ unr_W_[s] for s in [0, 1]]
                else:
                    P_core = eye(self.W.shape[0]) - self.P_core @ self.S
                    C_ = P_core @ self.W
                    # NOTE: PYSCF has basis in 1s2s3s2p2p2p3p3p3p format
                    # fix no_core_idx - use population for now
                    Cpop = multi_dot((C_.T, self.S, C_))
                    no_core_idx = where(diag(Cpop) > 0.7)[0]
                    C_ = C_[:, no_core_idx]
                    S_ = multi_dot((C_.T, self.S, C_))
                    es_, vs_ = eigh(S_)
                    s_ = sqrt(es_)
                    s_ = diag(1.0 / s_)
                    W_ = multi_dot((vs_, s_, vs_.T))
                    self.W = C_ @ W_

            if self.unrestricted:
                if self.frozen_core:
                    self.lmo_coeff_a = multi_dot(
                        (self.W[0].T, self.S, self.C_a[:, self.ncore :])  # type: ignore[attr-defined]
                    )
                    self.lmo_coeff_b = multi_dot(
                        (self.W[1].T, self.S, self.C_b[:, self.ncore :])  # type: ignore[attr-defined]
                    )
                else:
                    self.lmo_coeff_a = multi_dot((self.W.T, self.S, self.C_a))  # type: ignore[attr-defined]
                    self.lmo_coeff_b = multi_dot((self.W.T, self.S, self.C_b))  # type: ignore[attr-defined]
            else:
                if self.frozen_core:
                    self.lmo_coeff = multi_dot(
                        (self.W.T, self.S, self.C[:, self.ncore :])
                    )
                else:
                    self.lmo_coeff = multi_dot((self.W.T, self.S, self.C))
            return None
        elif lo_method == "boys" or lo_method == "PM" or lo_method == "ER":
            es_, vs_ = eigh(self.S)
            edx = es_ > 1.0e-15
            W_ = vs_[:, edx] / sqrt(es_[edx]) @ vs_[:, edx].T
            if self.frozen_core:
                P_core = eye(W_.shape[0]) - self.P_core @ self.S
                C_ = P_core @ W_
                Cpop = multi_dot((C_.T, self.S, C_))
                Cpop = diag(Cpop)
                no_core_idx = where(Cpop > 0.55)[0]
                C_ = C_[:, no_core_idx]
                S_ = multi_dot((C_.T, self.S, C_))
                es_, vs_ = eigh(S_)
                s_ = sqrt(es_)
                s_ = diag(1.0 / s_)
                W_ = multi_dot((vs_, s_, vs_.T))
                W_ = C_ @ W_

            self.W = get_loc(
                self.mf.mol,
                W_,
                lo_method,  # type: ignore[arg-type]
                pop_method=pop_method,
                init_guess=init_guess,
            )

            if not self.frozen_core:
                self.lmo_coeff = self.W.T @ self.S @ self.C
            else:
                self.lmo_coeff = self.W.T @ self.S @ self.C[:, self.ncore :]
            return None

        elif lo_method == "IAO":
            # IAO working basis: (w): (large) basis set we use
            # IAO valence basis: (v): minimal-like basis we try to resemble
            assert fobj.iao_valence_basis is not None

            # Occupied mo_coeff (with core)
            Co = self.C[:, : self.Nocc]

            # Get necessary overlaps, second arg is IAO valence basis
            S_vw, S_vv = get_xovlp(self.fobj.mol, basis=fobj.iao_valence_basis)

            # How do we describe the rest of the space?
            # If iao_valence_only=False, we use PAOs:
            if not iao_valence_only:
                Ciao = get_iao(
                    Co,
                    S_vw,
                    self.S,
                    S_vv,
                    self.fobj.mol,
                    fobj.iao_valence_basis,
                    iao_loc_method,
                )

                Cpao = get_pao(
                    Ciao,
                    self.S,
                    S_vw,
                    self.fobj.mol,
                    fobj.iao_valence_basis,
                    iao_loc_method,
                )

                if iao_loc_method != "lowdin":
                    # Localize IAOs and PAOs
                    Ciao = get_loc(self.fobj.mol, Ciao, iao_loc_method)
                    Cpao = get_loc(self.fobj.mol, Cpao, iao_loc_method)
            else:
                Ciao = get_iao(
                    Co,
                    S_vw,
                    self.S,
                    S_vv,
                    self.fobj.mol,
                    fobj.iao_valence_basis,
                    iao_loc_method,
                )

                if iao_loc_method != "lowdin":
                    Ciao = get_loc(self.fobj.mol, Ciao, iao_loc_method)

            # Rearrange by atom
            aoind_by_atom = get_aoind_by_atom(self.fobj.mol)
            Ciao, iaoind_by_atom = reorder_by_atom_(Ciao, aoind_by_atom, self.S)

            if not iao_valence_only:
                Cpao, paoind_by_atom = reorder_by_atom_(Cpao, aoind_by_atom, self.S)

            if self.frozen_core:
                # Remove core MOs
                Cc = self.C[:, : self.ncore]  # Assumes core are first
                Ciao = remove_core_mo(Ciao, Cc, self.S)

            shift = 0
            ncore = 0
            if not iao_valence_only:
                Wstack = zeros((Ciao.shape[0], Ciao.shape[1] + Cpao.shape[1]))
            else:
                Wstack = zeros((Ciao.shape[0], Ciao.shape[1]))

            if self.frozen_core:
                for ix in range(self.fobj.mol.natm):
                    nc = ncore_(self.fobj.mol.atom_charge(ix))
                    ncore += nc
                    niao = len(iaoind_by_atom[ix])
                    iaoind_ix = [i_ - ncore for i_ in iaoind_by_atom[ix][nc:]]
                    Wstack[:, shift : shift + niao - nc] = Ciao[:, iaoind_ix]
                    shift += niao - nc
                    if not iao_valence_only:
                        npao = len(paoind_by_atom[ix])
                        Wstack[:, shift : shift + npao] = Cpao[:, paoind_by_atom[ix]]
                        shift += npao
            else:
                if not hstack:
                    for ix in range(self.fobj.mol.natm):
                        niao = len(iaoind_by_atom[ix])
                        Wstack[:, shift : shift + niao] = Ciao[:, iaoind_by_atom[ix]]
                        shift += niao
                        if not iao_valence_only:
                            npao = len(paoind_by_atom[ix])
                            Wstack[:, shift : shift + npao] = Cpao[
                                :, paoind_by_atom[ix]
                            ]
                            shift += npao
                else:
                    Wstack = np.hstack((Ciao, Cpao))
            if save:
                self.W = Wstack
                assert allclose(self.W.T @ self.S @ self.W, eye(self.W.shape[1]))
            else:
                assert allclose(Wstack.T @ self.S @ Wstack, eye(Wstack.shape[1]))
                return Wstack

            nmo = self.C.shape[1] - self.ncore
            nlo = self.W.shape[1]

            if not iao_valence_only:
                if nmo > nlo:
                    Co_nocore = self.C[:, self.ncore : self.Nocc]
                    Cv = self.C[:, self.Nocc :]
                    # Ensure that the LOs span the occupied space
                    assert allclose(
                        np.sum((self.W.T @ self.S @ Co_nocore) ** 2.0),
                        self.Nocc - self.ncore,
                    )
                    # Find virtual orbitals that lie in the span of LOs
                    u, l, vt = svd(self.W.T @ self.S @ Cv, full_matrices=False)
                    unused(u)
                    nvlo = nlo - self.Nocc - self.ncore
                    assert allclose(np.sum(l[:nvlo]), nvlo)
                    C_ = np.hstack([Co_nocore, Cv @ vt[:nvlo].T])
                    self.lmo_coeff = self.W.T @ self.S @ C_
                else:
                    self.lmo_coeff = self.W.T @ self.S @ self.C[:, self.ncore :]
            else:
                self.lmo_coeff = self.W.T @ self.S @ self.C[:, self.ncore :]
            return None
        else:
            raise assert_never(lo_method)

    def qchem_setup(self):
        """
        Extracts the information necessary to run a Q-Chem EOM-CCSD calculation
        for each fragment.
        Constructs the following scratch files:
        - 99.0 (Total energy)
        - 53.0 (MO coefficients)
        - 58.0 (Fock matrix)
        """

        print("QChem:")
        print("Exporting files 99.0, 58.0, 53.0 to Q-Chem for EOM-CCSD calculation")

        pot = self.pot
        Fobjs = self.Fobjs

        # Loop over each fragment and extract necessary information from Qchem
        for frag_number, fobj in enumerate(Fobjs):
            # Update the effective Hamiltonian
            if pot is not None:
                fobj.update_heff(pot)

            # Perform SCF calculation
            fobj.scf()

            # export necessary information to Q-Chem
            mf = fobj._mf

            print("Fragment number: ", frag_number)
            ###numbers of electrons in:
            n_mo_full_syst = shape(fobj.TA)[0]
            occ_tot = self.Nocc  ###full system
            SO_tot = shape(fobj.TA)[1]
            SO_occ = fobj.nsocc  ###Schmidt space

            """if fobj.nfsites <= SO_occ:  ###if fragment is fully occupied
                frag_occ = fobj.nfsites
            else:
                frag_occ = occ_tot"""

            # bath_occ = SO_occ - frag_occ

            # bath_virt = SO_tot - frag_occ - bath_occ
            ###TO DO: automate generation of qchem input files
            env_occ = occ_tot - SO_occ
            print("Qchem: set n_frozen_core")
            print("Number occupied environment: ", env_occ)

            env_virt = n_mo_full_syst - SO_tot - env_occ
            print("Qchem: set n_frozen_virtual")
            print("Number virtual environment: ", env_virt)

            ###File 99.0 - energy file in Qchem

            energy = mf.kernel()

            energy_array = zeros(12)
            # placeholder value - exact value doesn't matter
            energy_array[0] = 3.7617453591977221e02
            energy_array[1] = energy
            energy_array[11] = energy

            ###File 58.0 - Fock matrix file in Qchem
            ###Full system Fock matrix (AO basis)
            fock_full_syst = load("files_EOM/full_syst_fock.npy")

            flat_fock = array(fock_full_syst.flatten(), dtype=float64)
            full_fock = concatenate((flat_fock, flat_fock), axis=None)

            ###File 53.0 - MO coefficient matrix
            ###TA: AOxSO; mf.mo_coeff: SOxMO
            TA_after_HF = fobj.TA @ mf.mo_coeff

            ###Pad TA matrix with orthogonal vectors
            ###use it as MO coefficient matrix for Qchem

            # m=n_orb_total (frag+bath+env)
            # n=n_frag+n_bath
            m, n = TA_after_HF.shape

            # compute the orthonormal basis for the null space of TA.T
            # do SVD
            _, _, vh = svd(TA_after_HF.T, full_matrices=True)

            # take the (m-n) right singular vectors orthogonal to TA.T
            orthogonal_vectors = vh[n:m].T

            # pad the original matrix with the orthogonal vectors
            TA_full_pyscf = hstack(
                (
                    orthogonal_vectors[:, :env_occ],
                    TA_after_HF,
                    orthogonal_vectors[:, env_occ:],
                )
            )

            TA_full_qchem = TA_full_pyscf.T

            flat_mos = array(TA_full_qchem.flatten(), dtype=float64)

            ###MO energies needed at the end of file 53.0
            mo_energies = zeros(n_mo_full_syst)
            # set to arbitrary low number to avoid recanonicalization in Qchem
            mo_energies[:env_occ] = -1000
            mo_energies[env_occ : env_occ + SO_tot] = mf.mo_energy
            mo_energies[env_occ + SO_tot :] = 1000

            print("Fragment: ", frag_number)
            print(mf.mo_energy)
            print("SO occ: ", SO_occ)

            full_mo_array = concatenate(
                (flat_mos, flat_mos, mo_energies, mo_energies), axis=None
            )

            if not os.path.exists("files_EOM/scratch_fragment_" + str(frag_number)):
                os.makedirs("files_EOM/scratch_fragment_" + str(frag_number))

            energy_array.tofile(
                "files_EOM/scratch_fragment_" + str(frag_number) + "/99.0"
            )
            full_fock.tofile("files_EOM/scratch_fragment_" + str(frag_number) + "/58.0")
            full_mo_array.tofile(
                "files_EOM/scratch_fragment_" + str(frag_number) + "/53.0"
            )

        return


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
