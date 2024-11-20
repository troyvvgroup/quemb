# Author(s): Oinam Romesh Meitei

from .pfrag import Frags
from molbe.helper import get_core
import molbe.be_var as be_var
import numpy, functools, sys, pickle
from pyscf import lib
import h5py, os

from .misc import storePBE


class BE:
    """
    Class for handling periodic bootstrap embedding (BE) calculations.

    This class encapsulates the functionalities required for performing periodic bootstrap embedding calculations,
    including setting up the BE environment, initializing fragments, performing SCF calculations, and
    evaluating energies.

    Attributes
    ----------
    mf : pyscf.scf.SCF
        PySCF mean-field object.
    fobj : molbe.fragpart
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
        compute_hf=True,
        restart=False,
        save=False,
        restart_file="storebe.pk",
        mo_energy=None,
        save_file="storebe.pk",
        hci_pt=False,
        nproc=1,
        ompnum=4,
        hci_cutoff=0.001,
        ci_coeff_cutoff=None,
        select_cutoff=None,
        iao_val_core=True,
        exxdiv="ewald",
        kpts=None,
        cderi=None,
        iao_wannier=False,
    ):
        """
        Constructor for BE object.

        Parameters
        ----------
        mf : pyscf.pbc.scf.SCF
            PySCF periodic mean-field object.
        fobj : kbe.fragpart
            Fragment object containing sites, centers, edges, and indices.
        kpts : list of list of float
            k-points in the reciprocal space for periodic computation
        eri_file : str, optional
            Path to the file storing two-electron integrals, by default 'eri_file.h5'.
        lo_method : str, optional
            Method for orbital localization, by default 'lowdin'.
        iao_wannier : bool, optional
            Whether to perform Wannier localization on the IAO space, by default False.
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
            Number of processors for parallel calculations, by default 1. If set to >1, threaded parallel computation is invoked.
        ompnum : int, optional
            Number of OpenMP threads, by default 4.
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
            self.ek = store_.ek
            self.E_core = store_.E_core
            self.C_core = store_.C_core
            self.P_core = store_.P_core
            self.core_veff = store_.core_veff

        self.nproc = nproc
        self.ompnum = ompnum

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
        self.unitcell = fobj.unitcell
        self.mol = fobj.mol
        self.cell = fobj.mol
        self.kmesh = fobj.kpt

        unitcell_nkpt = 1
        for i in self.kmesh:
            if i > 1:
                unitcell_nkpt *= self.unitcell
        self.unitcell_nkpt = unitcell_nkpt
        self.ebe_hf = 0.0

        nkpts_ = 1
        for i in self.kmesh:
            if i > 1:
                nkpts_ *= i
        self.nkpt = nkpts_
        self.kpts = kpts

        # HCI parameters
        self.hci_cutoff = hci_cutoff
        self.ci_coeff_cutoff = ci_coeff_cutoff
        self.select_cutoff = select_cutoff
        self.hci_pt = hci_pt

        if not restart:
            self.mo_energy = mf.mo_energy
            mf.exxdiv = None
            self.mf = mf
            self.Nocc = mf.cell.nelectron // 2
            self.enuc = mf.energy_nuc()
            self.hcore = mf.get_hcore()
            self.S = mf.get_ovlp()
            self.C = numpy.array(mf.mo_coeff)
            self.hf_dm = mf.make_rdm1()
            self.hf_veff = mf.get_veff(
                self.cell, dm_kpts=self.hf_dm, hermi=1, kpts=self.kpts, kpts_band=None
            )
            self.hf_etot = mf.e_tot
            self.W = None
            self.lmo_coeff = None

        self.print_ini()
        self.Fobjs = []
        self.pot = initialize_pot(self.Nfrag, self.edge_idx)
        self.eri_file = eri_file
        self.cderi = cderi

        # Set scratch directory
        jobid = ""
        if be_var.CREATE_SCRATCH_DIR:
            try:
                jobid = str(os.environ["SLURM_JOB_ID"])
            except:
                jobid = ""
        if not be_var.SCRATCH == "":
            os.system("mkdir " + be_var.SCRATCH + str(jobid))
        if jobid == "":
            self.eri_file = be_var.SCRATCH + eri_file
            if cderi:
                self.cderi = be_var.SCRATCH + cderi
        else:
            self.eri_file = be_var.SCRATCH + str(jobid) + "/" + eri_file
            if cderi:
                self.cderi = be_var.SCRATCH + str(jobid) + "/" + cderi

        if exxdiv == "ewald":
            if not restart:
                self.ek = self.ewald_sum(kpts=self.kpts)
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

                nk, nao, nao = self.hf_dm.shape

                dm_nocore = numpy.zeros(
                    (nk, nao, nao), dtype=numpy.result_type(self.C, self.C)
                )
                C_core = numpy.zeros((nk, nao, self.ncore), dtype=self.C.dtype)
                P_core = numpy.zeros(
                    (nk, nao, nao), dtype=numpy.result_type(self.C, self.C)
                )

                for k in range(nk):
                    dm_nocore[k] += 2.0 * numpy.dot(
                        self.C[k][:, self.ncore : self.ncore + self.Nocc],
                        self.C[k][:, self.ncore : self.ncore + self.Nocc].conj().T,
                    )
                    C_core[k] += self.C[k][:, : self.ncore]
                    P_core[k] += numpy.dot(C_core[k], C_core[k].conj().T)

                self.C_core = C_core
                self.P_core = P_core
                self.hf_dm = dm_nocore
                self.core_veff = mf.get_veff(
                    self.cell,
                    dm_kpts=self.P_core * 2.0,
                    hermi=1,
                    kpts=self.kpts,
                    kpts_band=None,
                )

                ecore_h1 = 0.0
                ecore_veff = 0.0
                for k in range(nk):
                    ecore_h1 += numpy.einsum(
                        "ij,ji", self.hcore[k], 2.0 * self.P_core[k]
                    )
                    ecore_veff += (
                        numpy.einsum("ij,ji", 2.0 * self.P_core[k], self.core_veff[k])
                        * 0.5
                    )

                ecore_h1 /= float(nk)
                ecore_veff /= float(nk)

                E_core = ecore_h1 + ecore_veff
                if numpy.abs(E_core.imag).max() < 1.0e-10:
                    self.E_core = E_core.real
                else:
                    print("Imaginary density in E_core ", numpy.abs(E_core.imag).max())
                    sys.exit()

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
                mol=self.cell,
                valence_basis=fobj.valence_basis,
                iao_wannier=iao_wannier,
                iao_val_core=iao_val_core,
            )
        if save:
            # Save intermediate results for restart
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
            with open(save_file, "wb") as rfile:
                pickle.dump(store_, rfile, pickle.HIGHEST_PROTOCOL)
            rfile.close()

        if not restart:
            self.initialize(mf._eri, compute_hf)

    from ._opt import optimize

    # this is a molbe method not BEOPT
    from molbe.external.optqn import get_be_error_jacobian
    from .lo import localize

    def print_ini(self):
        """
        Print initialization banner for the kBE calculation.
        """
        print("-----------------------------------------------------------", flush=True)

        print("             BBBBBBB    EEEEEEE ", flush=True)
        print("             BB     B   EE      ", flush=True)
        print("   PP   PP   BB     B   EE      ", flush=True)
        print("   PP  PP    BBBBBBB    EEEEEEE ", flush=True)
        print("   PPPP      BB     B   EE      ", flush=True)
        print("   PP  PP    BB     B   EE      ", flush=True)
        print("   PP   PP   BBBBBBB    EEEEEEE ", flush=True)
        print(flush=True)

        print("            PERIODIC BOOTSTRAP EMBEDDING", flush=True)
        print("           BEn = ", self.be_type, flush=True)
        print("-----------------------------------------------------------", flush=True)
        print(flush=True)

    def ewald_sum(self, kpts=None):
        from pyscf.pbc.df.df_jk import _ewald_exxdiv_for_G0

        dm_ = self.mf.make_rdm1()
        nk, nao, nao = dm_.shape

        vk_kpts = numpy.zeros(dm_.shape) * 1j
        _ewald_exxdiv_for_G0(
            self.mf.cell,
            self.kpts,
            dm_.reshape(-1, nk, nao, nao),
            vk_kpts.reshape(-1, nk, nao, nao),
            self.kpts,
        )
        e_ = numpy.einsum("kij,kji->", vk_kpts, dm_) * 0.25
        e_ /= float(nk)

        return e_.real

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
        from molbe.helper import get_scfObj
        from multiprocessing import Pool

        import h5py, os, logging
        from pyscf import ao2mo
        from pyscf.pbc.df import fft_ao2mo
        from pyscf.pbc.df import df_ao2mo
        from pyscf.pbc import ao2mo as pao2mo
        from libdmet.basis_transform.eri_transform import get_emb_eri_fast_gdf

        if compute_hf:
            E_hf = 0.0
        EH1 = 0.0
        ECOUL = 0.0
        EF = 0.0

        # Create a file to store ERIs
        if not restart:
            file_eri = h5py.File(self.eri_file, "w")
        lentmp = len(self.edge_idx)
        transform_parallel = False  # hard set for now
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
                    unitcell=self.unitcell,
                    unitcell_nkpt=self.unitcell_nkpt,
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
                    unitcell=self.unitcell,
                    unitcell_nkpt=self.unitcell_nkpt,
                )

            fobjs_.sd(
                self.W,
                self.lmo_coeff,
                self.Nocc,
                kmesh=self.kmesh,
                cell=self.cell,
                frag_type=self.frag_type,
                kpts=self.kpts,
                h1=self.hcore,
            )

            fobjs_.cons_h1(self.hcore)
            fobjs_.heff = numpy.zeros_like(fobjs_.h1)
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
                print("If cderi is set, try again with nproc > 1")
                sys.exit()

            nprocs = int(self.nproc / self.ompnum)
            pool_ = Pool(nprocs)
            os.system("export OMP_NUM_THREADS=" + str(self.ompnum))
            results = []
            eris = []
            for frg in range(self.Nfrag):
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
            [eris.append(result.get()) for result in results]
            pool_.close()

            for frg in range(self.Nfrag):
                file_eri.create_dataset(self.Fobjs[frg].dname, data=eris[frg])
            eris = None
            file_eri.close()

            nprocs = int(self.nproc / self.ompnum)
            pool_ = Pool(nprocs)
            results = []
            veffs = []
            for frg in range(self.Nfrag):
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
            [veffs.append(result.get()) for result in results]
            pool_.close()

            for frg in range(self.Nfrag):
                veff0, veff_ = veffs[frg]
                if numpy.abs(veff_.imag).max() < 1.0e-6:
                    self.Fobjs[frg].veff = veff_.real
                    self.Fobjs[frg].veff0 = veff0.real
                else:
                    print("Imaginary Veff ", numpy.abs(veff_.imag).max())
                    sys.exit()
                self.Fobjs[frg].fock = self.Fobjs[frg].h1 + veff_.real
            veffs = None

        # SCF parallelized
        if self.nproc == 1 and not transform_parallel:
            for frg in range(self.Nfrag):
                # SCF
                self.Fobjs[frg].scf(fs=True, dm0=self.Fobjs[frg].dm_init)
        else:
            nprocs = int(self.nproc / self.ompnum)
            pool_ = Pool(nprocs)
            os.system("export OMP_NUM_THREADS=" + str(self.ompnum))
            results = []
            mo_coeffs = []
            for frg in range(self.Nfrag):
                nao = self.Fobjs[frg].nao
                nocc = self.Fobjs[frg].nsocc
                dname = self.Fobjs[frg].dname
                h1 = self.Fobjs[frg].fock + self.Fobjs[frg].heff
                result = pool_.apply_async(
                    parallel_scf_wrapper,
                    [dname, nao, nocc, h1, self.Fobjs[frg].dm_init, self.eri_file],
                )
                results.append(result)
            [mo_coeffs.append(result.get()) for result in results]
            pool_.close()
            for frg in range(self.Nfrag):
                self.Fobjs[frg]._mo_coeffs = mo_coeffs[frg]

        for frg in range(self.Nfrag):
            self.Fobjs[frg].dm0 = (
                numpy.dot(
                    self.Fobjs[frg]._mo_coeffs[:, : self.Fobjs[frg].nsocc],
                    self.Fobjs[frg]._mo_coeffs[:, : self.Fobjs[frg].nsocc].conj().T,
                )
                * 2.0
            )

            # energy
            if compute_hf:
                eh1, ecoul, ef = self.Fobjs[frg].energy_hf(return_e1=True)
                E_hf += self.Fobjs[frg].ebe_hf

        print(flush=True)
        if not restart:
            file_eri.close()

        if compute_hf:
            E_hf /= self.unitcell_nkpt
            hf_err = self.hf_etot - (E_hf + self.enuc + self.E_core)

            self.ebe_hf = E_hf + self.enuc + self.E_core - self.ek
            print(
                "HF-in-HF error                 :  {:>.4e} Ha".format(hf_err),
                flush=True,
            )

            if abs(hf_err) > 1.0e-5:
                print("WARNING!!! Large HF-in-HF energy error")

        couti = 0
        for fobj in self.Fobjs:
            fobj.udim = couti
            couti = fobj.set_udim(couti)

    def oneshot(
        self, solver="MP2", nproc=1, ompnum=4, calc_frag_energy=False, clean_eri=False
    ):
        """
        Perform a one-shot bootstrap embedding calculation.

        Parameters
        ----------
        solver : str, optional
            High-level quantum chemistry method, by default 'MP2'. 'CCSD', 'FCI', and variants of selected CI are supported.
        nproc : int, optional
            Number of processors for parallel calculations, by default 1. If set to >1, threaded parallel computation is invoked.
        ompnum : int, optional
            Number of OpenMP threads, by default 4.
        calc_frag_energy : bool, optional
            Whether to calculate fragment energies, by default False.
        clean_eri : bool, optional
            Whether to clean up ERI files after calculation, by default False.
        """
        from .solver import be_func
        from .be_parallel import be_func_parallel

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

        if clean_eri == True:
            try:
                os.remove(self.eri_file)
                os.rmdir(self.scratch_dir)
            except:
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

    This function initializes a potential array for a given number of fragments (`Nfrag`)
    and their corresponding edge indices (`edge_idx`). The potential array is initialized
    with zeros for each pair of edge site indices within each fragment, followed by an
    additional zero for the global chemical potential.

    Parameters
    ----------
    Nfrag : int
        Number of fragments.
    edge_idx : list of list of list of int
        List of edge indices for each fragment. Each element is a list of lists, where each
        sublist contains the indices of edge sites for a particular fragment.

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


def eritransform_parallel(a, atom, basis, kpts, C_ao_emb, cderi):
    """
    Wrapper for parallel eri transformation
    """
    from molbe.external.eri_transform import get_emb_eri_fast_gdf
    from pyscf.pbc import gto, df

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
    from .helper import get_veff, get_eri

    eri_ = get_eri(dname, nao, eri_file=eri_file, ignore_symm=True)
    veff0, veff_ = get_veff(eri_, dm, S, TA, hf_veff, return_veff0=True)

    return veff0, veff_


def parallel_scf_wrapper(dname, nao, nocc, h1, dm_init, eri_file):
    """
    Wrapper for performing fragment scf calculation
    """
    from .helper import get_eri, get_scfObj

    eri = get_eri(dname, nao, eri_file=eri_file)
    mf_ = get_scfObj(h1, eri, nocc, dm_init)

    return mf_.mo_coeff
