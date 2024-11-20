# Author(s): Oinam Romesh Meitei

from .pfrag import Frags
from .helper import get_core
import molbe.be_var as be_var
import numpy, functools, sys, pickle
from pyscf import lib
import h5py, os, time


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


class BE:
    """
    Class for handling bootstrap embedding (BE) calculations.

    This class encapsulates the functionalities required for performing bootstrap embedding calculations,
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
        mf : pyscf.scf.SCF
            PySCF mean-field object.
        fobj : molbe.fragpart
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
            Number of processors for parallel calculations, by default 1. If set to >1, threaded parallel computation is invoked.
        ompnum : int, optional
            Number of OpenMP threads, by default 4.
        integral_direct_DF: bool, optional
            If mf._eri is None (i.e. ERIs are not saved in memory using incore_anyway), this flag is used to determine if the ERIs are computed integral-directly using density fitting; by default False.
        auxbasis : str, optional
            Auxiliary basis for density fitting, by default None (uses default auxiliary basis defined in PySCF).
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
            try:
                jobid = str(os.environ["SLURM_JOB_ID"])
            except:
                jobid = ""
        if not be_var.SCRATCH == "":
            self.scratch_dir = be_var.SCRATCH + str(jobid)
            os.system("mkdir -p " + self.scratch_dir)
        else:
            self.scratch_dir = None
        if jobid == "":
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

    from ._opt import optimize
    from molbe.external.optqn import get_be_error_jacobian
    from .lo import localize
    from .rdm import rdm1_fullbasis, compute_energy_full

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
        from .helper import get_scfObj
        import h5py
        from pyscf import ao2mo
        from multiprocessing import Pool

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
            #            Yes -- ao2mo, outcore version, using saved (ij|P)
            #            No  -- if integral_direct_DF is requested, invoke on-the-fly routine
            assert (
                (not eri_ is None)
                or (hasattr(self.mf, "with_df"))
                or (self.integral_direct_DF)
            ), "Input mean-field object is missing ERI (mf._eri) or DF (mf.with_df) object AND integral direct DF routine was not requested. Please check your inputs."
            if (
                not eri_ is None
            ):  # incore ao2mo using saved eri from mean-field calculation
                for I in range(self.Nfrag):
                    eri = ao2mo.incore.full(eri_, self.Fobjs[I].TA, compact=True)
                    file_eri.create_dataset(self.Fobjs[I].dname, data=eri)
            elif hasattr(self.mf, "with_df") and not self.mf.with_df is None:
                # pyscf.ao2mo uses DF object in an outcore fashion using (ij|P) in pyscf temp directory
                for I in range(self.Nfrag):
                    eri = self.mf.with_df.ao2mo(self.Fobjs[I].TA, compact=True)
                    file_eri.create_dataset(self.Fobjs[I].dname, data=eri)
            else:
                # If ERIs are not saved on memory, compute fragment ERIs integral-direct
                if (
                    self.integral_direct_DF
                ):  # Use density fitting to generate fragment ERIs on-the-fly
                    from .eri_onthefly import integral_direct_DF

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
            dm_init = fobjs_.get_nsocc(self.S, self.C, self.Nocc, ncore=self.ncore)

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
                eh1, ecoul, ef = fobjs_.energy_hf(return_e1=True)
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
