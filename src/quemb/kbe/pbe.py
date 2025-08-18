# Author(s): Oinam Romesh Meitei

import contextlib
import io
import os
import pickle
from multiprocessing import Pool
from typing import Literal
from warnings import warn

import h5py
import numpy as np
from numpy import array, einsum, floating, result_type, zeros, zeros_like
from pyscf import ao2mo
from pyscf.pbc import df, gto, scf
from pyscf.pbc.df.df_jk import _ewald_exxdiv_for_G0

from quemb.kbe.fragment import FragPart
from quemb.kbe.lo import Mixin_k_Localize
from quemb.kbe.misc import print_energy, storePBE
from quemb.kbe.pfrag import Frags
from quemb.molbe.be_parallel import be_func_parallel
from quemb.molbe.helper import get_eri, get_scfObj, get_veff
from quemb.molbe.opt import BEOPT
from quemb.molbe.solver import Solvers, UserSolverArgs, be_func
from quemb.shared.external.optqn import (
    get_be_error_jacobian as _ext_get_be_error_jacobian,
)
from quemb.shared.helper import copy_docstring, timer
from quemb.shared.manage_scratch import WorkDir
from quemb.shared.typing import Matrix, PathLike

with contextlib.redirect_stdout(io.StringIO()):
    # Since they don't want to reduce the printing, let's temporarily disable STDOUT
    # https://github.com/gkclab/libdmet_preview/issues/22
    from libdmet.basis_transform.eri_transform import get_emb_eri_fast_gdf


class BE(Mixin_k_Localize):
    """
    Class for handling periodic bootstrap embedding (BE) calculations.

    This class encapsulates the functionalities required for performing
    periodic bootstrap embedding calculations, including setting up the BE environment,
    initializing fragments, performing SCF calculations, and evaluating energies.

    Attributes
    ----------
    mf :
        PySCF mean-field object.
    fobj :
        Fragment object containing sites, centers, edges, and indices.
    eri_file :
        Path to the file storing two-electron integrals.
    lo_method :
        Method for orbital localization, default is 'lowdin'.
    """

    def __init__(
        self,
        mf: scf.khf.KRHF,
        fobj: FragPart,
        eri_file: PathLike = "eri_file.h5",
        lo_method: str = "lowdin",
        compute_hf: bool = True,
        restart: bool = False,
        restart_file: PathLike = "storebe.pk",
        nproc: int = 1,
        ompnum: int = 4,
        iao_val_core: bool = True,
        exxdiv: str | None = "ewald",
        kpts: list[list[float]] | None = None,
        cderi: PathLike | None = None,
        iao_wannier: bool = False,
        thr_bath: float = 1.0e-10,
        scratch_dir: WorkDir | None = None,
    ) -> None:
        """
        Constructor for BE object.

        Parameters
        ----------
        mf :
            PySCF periodic mean-field object.
        fobj :
            Fragment object containing sites, centers, edges, and indices.
        kpts :
            k-points in the reciprocal space for periodic computation
        eri_file :
            Path to the file storing two-electron integrals, by default 'eri_file.h5'.
        lo_method :
            Method for orbital localization, by default 'lowdin'.
        iao_wannier :
            Whether to perform Wannier localization on the IAO space, by default False.
        compute_hf :
            Whether to compute Hartree-Fock energy, by default True.
        restart :
            Whether to restart from a previous calculation, by default False.
        restart_file :
            Path to the file storing restart information, by default 'storebe.pk'.
        nproc :
            Number of processors for parallel calculations, by default 1. If set to >1,
            multi-threaded parallel computation is invoked.
        ompnum :
            Number of OpenMP threads, by default 4.
        thr_bath : float,
            Threshold for bath orbitals in Schmidt decomposition
        scratch_dir :
            Scratch directory.
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
            self.ek = store_.ek
            self.E_core = store_.E_core
            self.C_core = store_.C_core
            self.P_core = store_.P_core
            self.core_veff = store_.core_veff

        self.nproc = nproc
        self.ompnum = ompnum
        self.thr_bath = thr_bath

        # Fragment information from fobj
        self.fobj = fobj

        unitcell_nkpt = 1
        for i in self.fobj.kpt:
            if i > 1:
                unitcell_nkpt *= self.fobj.unitcell
        self.unitcell_nkpt = unitcell_nkpt
        self.ebe_hf = 0.0

        nkpts_ = 1
        for i in self.fobj.kpt:
            if i > 1:
                nkpts_ *= i
        self.nkpt = nkpts_
        self.kpts = kpts

        if not restart:
            self.mo_energy = mf.mo_energy
            mf.exxdiv = None
            self.mf = mf
            self.Nocc = mf.cell.nelectron // 2
            self.enuc = mf.energy_nuc()
            self.hcore = mf.get_hcore()
            self.S = mf.get_ovlp()
            self.C = array(mf.mo_coeff)
            self.hf_dm = mf.make_rdm1()
            self.hf_veff = mf.get_veff(
                self.fobj.mol,
                dm_kpts=self.hf_dm,
                hermi=1,
                kpts=self.kpts,
                kpts_band=None,
            )
            self.hf_etot = mf.e_tot
            self.W = None
            self.lmo_coeff = None

        self.print_ini()
        self.Fobjs: list[Frags] = []
        self.pot = initialize_pot(self.fobj.n_frag, self.fobj.relAO_per_edge_per_frag)
        self.eri_file = eri_file
        self.cderi = cderi

        if scratch_dir is None:
            self.scratch_dir = WorkDir.from_environment()
        else:
            self.scratch_dir = scratch_dir
        self.eri_file = self.scratch_dir / eri_file
        self.cderi = self.scratch_dir / cderi if cderi else None

        if exxdiv == "ewald":
            if not restart:
                self.ek = self.ewald_sum()
            print(
                "Energy contribution from Ewald summation : {:>12.8f} Ha".format(
                    self.ek
                ),
                flush=True,
            )
            print("Total HF Energy will contain this contribution. ")
            print(flush=True)
        elif exxdiv is None:
            print("Setting exxdiv=None")
            self.ek = 0.0
        else:
            print("exxdiv = ", exxdiv, "not implemented!", flush=True)
            print("Energy may diverse.", flush=True)
            print(flush=True)

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

                nk, nao = self.hf_dm.shape[:2]

                dm_nocore = zeros((nk, nao, nao), dtype=result_type(self.C, self.C))
                C_core = zeros((nk, nao, self.ncore), dtype=self.C.dtype)
                P_core = zeros((nk, nao, nao), dtype=result_type(self.C, self.C))

                for k in range(nk):
                    dm_nocore[k] += 2.0 * (
                        self.C[k][:, self.ncore : self.ncore + self.Nocc]
                        @ self.C[k][:, self.ncore : self.ncore + self.Nocc].conj().T
                    )
                    C_core[k] += self.C[k][:, : self.ncore]
                    P_core[k] += C_core[k] @ C_core[k].conj().T

                self.C_core = C_core
                self.P_core = P_core
                self.hf_dm = dm_nocore
                self.core_veff = mf.get_veff(
                    self.fobj.mol,
                    dm_kpts=self.P_core * 2.0,
                    hermi=1,
                    kpts=self.kpts,
                    kpts_band=None,
                )

                ecore_h1 = 0.0
                ecore_veff = 0.0
                for k in range(nk):
                    ecore_h1 += einsum("ij,ji", self.hcore[k], 2.0 * self.P_core[k])
                    ecore_veff += (
                        einsum("ij,ji", 2.0 * self.P_core[k], self.core_veff[k]) * 0.5
                    )

                ecore_h1 /= float(nk)
                ecore_veff /= float(nk)

                E_core = ecore_h1 + ecore_veff
                if np.abs(E_core.imag).max() < 1.0e-10:
                    self.E_core = E_core.real
                else:
                    raise ValueError(
                        f"Imaginary density in E_core {np.abs(E_core.imag).max()}"
                    )

                for k in range(nk):
                    self.hf_veff[k] -= self.core_veff[k]
                    self.hcore[k] += self.core_veff[k]

        # Needed for Wannier localization
        if lo_method == "wannier" or iao_wannier:
            self.FOCK = self.mf.get_fock(self.hcore, self.S, self.hf_veff, self.hf_dm)

        if not restart:
            # Localize orbitals
            self.localize(
                lo_method,
                iao_valence_basis=fobj.iao_valence_basis,
                iao_wannier=iao_wannier,
                iao_val_core=iao_val_core,
            )

        if not restart:
            self.initialize(compute_hf)

    @timer.timeit
    def optimize(
        self,
        solver: Solvers = "CCSD",
        method: str = "QN",
        only_chem: bool = False,
        use_cumulant: bool = True,
        conv_tol: float = 1.0e-6,
        relax_density: bool = False,
        nproc: int = 1,
        ompnum: int = 4,
        max_iter: int = 500,
        jac_solver: Literal["HF", "MP2", "CCSD"] = "HF",
        trust_region: bool = False,
    ) -> None:
        """BE optimization function

        Interfaces BEOPT to perform bootstrap embedding optimization.

        Parameters
        ----------
        solver : str, optional
            High-level solver for the fragment, by default 'CCSD'
        method : str, optional
            Optimization method, by default 'QN'
        only_chem : bool, optional
            If true, density matching is not performed --
            only global chemical potential is optimized, by default False
        use_cumulant :
            Whether to use the cumulant energy expression, by default True.
        conv_tol : float, optional
            Convergence tolerance, by default 1.e-6
        relax_density : bool, optional
            Whether to use relaxed or unrelaxed densities, by default False
            This option is for using CCSD as solver. Relaxed density here uses
            Lambda amplitudes, whereas unrelaxed density only uses T amplitudes.
            c.f. See http://classic.chem.msu.su/cgi-bin/ceilidh.exe/gran/gamess/forum/?C34df668afbHW-7216-1405+00.htm
            for the distinction between the two
        max_iter : int, optional
            Maximum number of optimization steps, by default 500
        nproc : int
            Total number of processors assigned for the optimization. Defaults to 1.
            When nproc > 1, Python multithreading is invoked.
        ompnum : int
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
        else:
            pot = [0.0]

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
            self.ebe_tot = self.ebe_hf + be_.Ebe[0]
            # Print the energy components
            if use_cumulant:
                print_energy(
                    be_.Ebe[0],
                    be_.Ebe[1][1],
                    be_.Ebe[1][0] + be_.Ebe[1][2],
                    self.ebe_hf,
                    self.unitcell_nkpt,
                )
            else:
                raise NotImplementedError("""Non-cumulant energy not yet supported
                                               for periodic code""")
        else:
            raise ValueError("This optimization method for BE is not supported")

    def save(self, restart_file: PathLike = "storebe.pk") -> None:
        """
        Save the current state of the BE calculation to a file.

        Parameters
        ----------
        restart_file : str, optional
            Path to the file storing restart information, by default 'storebe.pk'.
        """
        store_ = storePBE(
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
            self.ek,
            self.E_core,
            self.C_core,
            self.P_core,
            self.core_veff,
        )
        with open(restart_file, "wb") as rfile:
            pickle.dump(store_, rfile, pickle.HIGHEST_PROTOCOL)

    @copy_docstring(_ext_get_be_error_jacobian)
    def get_be_error_jacobian(self, jac_solver: str = "HF") -> Matrix[floating]:
        return _ext_get_be_error_jacobian(self.fobj.n_frag, self.Fobjs, jac_solver)

    def print_ini(self) -> None:
        """
        Print initialization banner for the kBE calculation.
        """
        print(
            "-----------------------------------------------------------",
            flush=True,
        )

        print("             BBBBBBB    EEEEEEE ", flush=True)
        print("             BB     B   EE      ", flush=True)
        print("   PP   PP   BB     B   EE      ", flush=True)
        print("   PP  PP    BBBBBBB    EEEEEEE ", flush=True)
        print("   PPPP      BB     B   EE      ", flush=True)
        print("   PP  PP    BB     B   EE      ", flush=True)
        print("   PP   PP   BBBBBBB    EEEEEEE ", flush=True)
        print(flush=True)

        print("            PERIODIC BOOTSTRAP EMBEDDING", flush=True)
        print("           BEn = ", self.fobj.n_BE, flush=True)
        print(
            "-----------------------------------------------------------",
            flush=True,
        )
        print(flush=True)

    def ewald_sum(self):
        dm_ = self.mf.make_rdm1()
        nk, nao = dm_.shape[:2]

        vk_kpts = zeros(dm_.shape) * 1j
        _ewald_exxdiv_for_G0(
            self.mf.cell,
            self.kpts,
            dm_.reshape(-1, nk, nao, nao),
            vk_kpts.reshape(-1, nk, nao, nao),
            kpts_band=self.kpts,
        )
        e_ = einsum("kij,kji->", vk_kpts, dm_) * 0.25
        e_ /= float(nk)

        return e_.real

    def initialize(self, compute_hf: bool, restart: bool = False) -> None:
        """
        Initialize the Bootstrap Embedding calculation.

        Parameters
        ----------
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
        transform_parallel = False  # hard set for now
        for fidx in range(self.fobj.n_frag):
            fobjs_ = self.fobj.to_Frags(fidx, self.eri_file, self.unitcell_nkpt)
            fobjs_.sd(
                self.W,
                self.lmo_coeff,
                self.Nocc,
                kmesh=self.fobj.kpt,
                cell=self.fobj.mol,
                kpts=self.kpts,
                h1=self.hcore,
                thr_bath=self.thr_bath,
            )

            fobjs_.cons_h1(self.hcore)
            fobjs_.heff = zeros_like(fobjs_.h1)
            fobjs_.dm_init = fobjs_.get_nsocc(
                self.S, self.C, self.Nocc, ncore=self.ncore
            )

            if self.cderi is None:
                if not restart:
                    eri = get_emb_eri_fast_gdf(
                        self.mf.cell,
                        self.mf.with_df,
                        t_reversal_symm=True,
                        symmetry=4,
                        C_ao_eo=fobjs_.TA,
                    )[0]

                    file_eri.create_dataset(fobjs_.dname, data=eri)
                    eri = ao2mo.restore(8, eri, fobjs_.nao)
                    fobjs_.cons_fock(self.hf_veff, self.S, self.hf_dm, eri_=eri)
                else:
                    eri = None
            self.Fobjs.append(fobjs_)

        # ERI & Fock parallelization for periodic calculations
        if self.cderi:
            if self.nproc == 1:
                raise ValueError("If cderi is set, try again with nproc > 1")

            nprocs = self.nproc // self.ompnum
            os.system("export OMP_NUM_THREADS=" + str(self.ompnum))
            with Pool(nprocs) as pool_:
                results = []
                for frg in range(self.fobj.n_frag):
                    result = pool_.apply_async(
                        eritransform_parallel,
                        [
                            self.mf.cell.a,
                            self.mf.cell.atom,
                            self.mf.cell.basis,
                            self.kpts,
                            self.Fobjs[frg].TA,
                            self.cderi,
                        ],
                    )
                    results.append(result)
                eris = [result.get() for result in results]

            for frg in range(self.fobj.n_frag):
                file_eri.create_dataset(self.Fobjs[frg].dname, data=eris[frg])
            del eris
            file_eri.close()

            nprocs = self.nproc // self.ompnum
            with Pool(nprocs) as pool_:
                results = []
                for frg in range(self.fobj.n_frag):
                    result = pool_.apply_async(
                        parallel_fock_wrapper,
                        [
                            self.Fobjs[frg].dname,
                            self.Fobjs[frg].nao,
                            self.hf_dm,
                            self.S,
                            self.Fobjs[frg].TA,
                            self.hf_veff,
                            self.eri_file,
                        ],
                    )
                    results.append(result)
                veffs = [result.get() for result in results]

            for frg in range(self.fobj.n_frag):
                veff0, veff_ = veffs[frg]
                if np.abs(veff_.imag).max() < 1.0e-6:
                    self.Fobjs[frg].veff = veff_.real
                    self.Fobjs[frg].veff0 = veff0.real
                else:
                    raise ValueError(f"Imaginary Veff {np.abs(veff_.imag).max()}")

                self.Fobjs[frg].fock = self.Fobjs[frg].h1 + veff_.real
            del veffs

        # SCF parallelized
        if self.nproc == 1 and not transform_parallel:
            for frg in range(self.fobj.n_frag):
                # SCF
                self.Fobjs[frg].scf(fs=True, dm0=self.Fobjs[frg].dm_init)
        else:
            nprocs = int(self.nproc / self.ompnum)
            with Pool(nprocs) as pool_:
                os.system("export OMP_NUM_THREADS=" + str(self.ompnum))
                results = []
                for frg in range(self.fobj.n_frag):
                    nao = self.Fobjs[frg].nao
                    nocc = self.Fobjs[frg].nsocc
                    dname = self.Fobjs[frg].dname
                    h1 = self.Fobjs[frg].fock + self.Fobjs[frg].heff
                    result = pool_.apply_async(
                        parallel_scf_wrapper,
                        [dname, nao, nocc, h1, self.Fobjs[frg].dm_init, self.eri_file],
                    )
                    results.append(result)
                mo_coeffs = [result.get() for result in results]

            for frg in range(self.fobj.n_frag):
                self.Fobjs[frg]._mo_coeffs = mo_coeffs[frg]

        for frg in range(self.fobj.n_frag):
            self.Fobjs[frg].dm0 = 2.0 * (
                self.Fobjs[frg]._mo_coeffs[:, : self.Fobjs[frg].nsocc]
                @ self.Fobjs[frg]._mo_coeffs[:, : self.Fobjs[frg].nsocc].conj().T
            )

            if compute_hf:
                self.Fobjs[frg].update_ebe_hf()  # Updates fragment HF energy.
                E_hf += self.Fobjs[frg].ebe_hf

        print(flush=True)
        if not restart:
            file_eri.close()

        if compute_hf:
            E_hf /= self.unitcell_nkpt
            hf_err = self.hf_etot - (E_hf + self.enuc + self.E_core - self.ek)

            self.ebe_hf = E_hf + self.enuc + self.E_core - self.ek
            print(f"HF-in-HF error                 :  {hf_err:>.4e} Ha")
            if abs(hf_err) > 1.0e-5:
                warn("Large HF-in-HF energy error")

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
            If set to >1, threaded parallel computation is invoked.
        ompnum :
            Number of OpenMP threads, by default 4.
        clean_eri :
            Whether to clean up ERI files after calculation, by default False.
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

        print(
            "Final Tr(F del g) is         : {:>12.8f} Ha".format(
                rets[1][0] + rets[1][2]
            ),
            flush=True,
        )
        print(
            f"Final Tr(V K_approx) is      : {rets[1][1]:>12.8f} Ha",
            flush=True,
        )
        print(
            f"Final e_corr is              : {rets[0]:>12.8f} Ha",
            flush=True,
        )

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
            for fidx, fobj in self.Fobjs:
                fobj.fock += heff[fidx]

    def write_heff(self, heff_file="bepotfile.h5"):
        """
        Write the effective Hamiltonian to a file.

        Parameters
        ----------
        heff_file : str, optional
            Path to the file to store effective Hamiltonian, by default 'bepotfile.h5'.
        """
        with h5py.File(heff_file, "w") as filepot:
            for fobj in self.Fobjs:
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


def initialize_pot(n_frag, rel_AO_per_edge_per_frag):
    """
    Initialize the potential array for bootstrap embedding.

    This function initializes a potential array for a given number of
    fragments (:python:`n_frag`) and their corresponding edge indices
    (:python:`rel_AO_per_edge_per_frag`).
    The potential array is initialized with zeros for each pair of
    edge site indices within each fragment, followed by an
    additional zero for the global chemical potential.

    Parameters
    ----------
    n_frag: int
        Number of fragments.
    rel_AO_per_edge_per_frag: list of list of list of int
        List of edge indices for each fragment. Each element is a list of lists,
        where each sublist contains the indices of edge sites for a particular fragment.

    Returns
    -------
    list of float
        Initialized potential array with zeros.
    """
    pot_ = []

    if not len(rel_AO_per_edge_per_frag) == 0:
        for fidx in range(n_frag):
            for i in rel_AO_per_edge_per_frag[fidx]:
                for j in range(len(i)):
                    for k in range(len(i)):
                        if j > k:
                            continue
                        pot_.append(0.0)

    pot_.append(0.0)
    return pot_


def eritransform_parallel(a, atom, basis, kpts, C_ao_emb, cderi):
    """
    Wrapper for parallel eri transformation
    """
    cell = gto.Cell()
    cell.a = a
    cell.atom = atom
    cell.basis = basis
    cell.charge = 0
    cell.verbose = 0
    cell.build()

    mydf = df.GDF(cell, kpts)
    mydf._cderi = cderi
    eri = get_emb_eri_fast_gdf(
        cell, mydf, t_reversal_symm=True, symmetry=4, C_ao_eo=C_ao_emb
    )

    return eri


def parallel_fock_wrapper(dname, nao, dm, S, TA, hf_veff, eri_file):
    """
    Wrapper for parallel Fock transformation
    """

    eri_ = get_eri(dname, nao, eri_file=eri_file, ignore_symm=True)
    veff0, veff_ = get_veff(eri_, dm, S, TA, hf_veff, return_veff0=True)

    return veff0, veff_


def parallel_scf_wrapper(dname, nao, nocc, h1, dm_init, eri_file):
    """
    Wrapper for performing fragment scf calculation
    """

    eri = get_eri(dname, nao, eri_file=eri_file)
    mf_ = get_scfObj(h1, eri, nocc, dm_init)

    return mf_.mo_coeff
