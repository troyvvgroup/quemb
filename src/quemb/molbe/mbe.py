# Author(s): Oinam Romesh Meitei

import functools
import os
import pickle
import sys

import h5py
import numpy
from numpy.linalg import eigh, multi_dot, svd
from pyscf import ao2mo

from quemb.molbe.be_parallel import be_func_parallel
from quemb.molbe.eri_onthefly import integral_direct_DF
from quemb.molbe.external.lo_helper import (
    get_aoind_by_atom,
    reorder_by_atom_,
)
from quemb.molbe.helper import ncore_, unused
from quemb.molbe.lo import (
    get_iao,
    get_loc,
    get_pao,
    get_pao_native,
    get_xovlp,
    remove_core_mo,
)
from quemb.molbe.pfrag import Frags
from quemb.molbe.solver import be_func
from quemb.shared import be_var


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

    This class encapsulates the functionalities required for performing
    bootstrap embedding calculations, including setting up the BE environment,
    initializing fragments, performing SCF calculations, and
    evaluating energies.

    Attributes
    ----------
    mf : pyscf.scf.hf.SCF
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
        mf : pyscf.scf.hf.SCF
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

    # The following imports turn the imported functions into proper methods
    #  cannot be moved to head of file.
    from quemb.molbe._opt import optimize  # noqa: PLC0415
    from quemb.molbe.external.optqn import get_be_error_jacobian  # noqa: PLC0415
    from quemb.molbe.rdm import compute_energy_full, rdm1_fullbasis  # noqa: PLC0415

    def localize(
        self,
        lo_method,
        mol=None,
        valence_basis="sto-3g",
        hstack=False,
        pop_method=None,
        init_guess=None,
        valence_only=False,
        nosave=False,
    ):
        """Molecular orbital localization

        Performs molecular orbital localization computations. For large basis,
        IAO is recommended augmented with PAO orbitals.

        Parameters
        ----------
        lo_method : str
        Localization method in quantum chemistry. 'lowdin', 'boys', and 'iao'
        are supported.
        mol : pyscf.gto.Molecule
        pyscf.gto.Molecule object.
        valence_basis: str
        Name of minimal basis set for IAO scheme. 'sto-3g' suffice for most cases.
        valence_only: bool
        If this option is set to True, all calculation will be performed in the valence
        basis in the IAO partitioning.
        This is an experimental feature.
        """
        if lo_method == "lowdin":
            es_, vs_ = eigh(self.S)
            edx = es_ > 1.0e-15
            self.W = numpy.dot(vs_[:, edx] / numpy.sqrt(es_[edx]), vs_[:, edx].T)
            if self.frozen_core:
                if self.unrestricted:
                    P_core = [
                        numpy.eye(self.W.shape[0]) - numpy.dot(self.P_core[s], self.S)
                        for s in [0, 1]
                    ]
                    C_ = numpy.dot(P_core, self.W)
                    Cpop = [
                        functools.reduce(numpy.dot, (C_[s].T, self.S, C_[s]))
                        for s in [0, 1]
                    ]
                    Cpop = [numpy.diag(Cpop[s]) for s in [0, 1]]
                    no_core_idx = [numpy.where(Cpop[s] > 0.7)[0] for s in [0, 1]]
                    C_ = [C_[s][:, no_core_idx[s]] for s in [0, 1]]
                    S_ = [
                        functools.reduce(numpy.dot, (C_[s].T, self.S, C_[s]))
                        for s in [0, 1]
                    ]
                    W_ = []
                    for s in [0, 1]:
                        es_, vs_ = eigh(S_[s])
                        s_ = numpy.sqrt(es_)
                        s_ = numpy.diag(1.0 / s_)
                        W_.append(functools.reduce(numpy.dot, (vs_, s_, vs_.T)))
                    self.W = [numpy.dot(C_[s], W_[s]) for s in [0, 1]]
                else:
                    P_core = numpy.eye(self.W.shape[0]) - numpy.dot(self.P_core, self.S)
                    C_ = numpy.dot(P_core, self.W)
                    # NOTE: PYSCF has basis in 1s2s3s2p2p2p3p3p3p format
                    # fix no_core_idx - use population for now
                    Cpop = functools.reduce(numpy.dot, (C_.T, self.S, C_))
                    Cpop = numpy.diag(Cpop)
                    no_core_idx = numpy.where(Cpop > 0.7)[0]
                    C_ = C_[:, no_core_idx]
                    S_ = functools.reduce(numpy.dot, (C_.T, self.S, C_))
                    es_, vs_ = eigh(S_)
                    s_ = numpy.sqrt(es_)
                    s_ = numpy.diag(1.0 / s_)
                    W_ = functools.reduce(numpy.dot, (vs_, s_, vs_.T))
                    self.W = numpy.dot(C_, W_)

            if self.unrestricted:
                if self.frozen_core:
                    self.lmo_coeff_a = functools.reduce(
                        numpy.dot, (self.W[0].T, self.S, self.C_a[:, self.ncore :])
                    )
                    self.lmo_coeff_b = functools.reduce(
                        numpy.dot, (self.W[1].T, self.S, self.C_b[:, self.ncore :])
                    )
                else:
                    self.lmo_coeff_a = functools.reduce(
                        numpy.dot, (self.W.T, self.S, self.C_a)
                    )
                    self.lmo_coeff_b = functools.reduce(
                        numpy.dot, (self.W.T, self.S, self.C_b)
                    )
            else:
                if self.frozen_core:
                    self.lmo_coeff = functools.reduce(
                        numpy.dot, (self.W.T, self.S, self.C[:, self.ncore :])
                    )
                else:
                    self.lmo_coeff = multi_dot((self.W.T, self.S, self.C))

        elif lo_method in ["pipek-mezey", "pipek", "PM"]:
            es_, vs_ = eigh(self.S)
            edx = es_ > 1.0e-15
            self.W = numpy.dot(vs_[:, edx] / numpy.sqrt(es_[edx]), vs_[:, edx].T)

            es_, vs_ = eigh(self.S)
            edx = es_ > 1.0e-15
            W_ = numpy.dot(vs_[:, edx] / numpy.sqrt(es_[edx]), vs_[:, edx].T)
            if self.frozen_core:
                P_core = numpy.eye(W_.shape[0]) - numpy.dot(self.P_core, self.S)
                C_ = numpy.dot(P_core, W_)
                Cpop = functools.reduce(numpy.dot, (C_.T, self.S, C_))
                Cpop = numpy.diag(Cpop)
                no_core_idx = numpy.where(Cpop > 0.55)[0]
                C_ = C_[:, no_core_idx]
                S_ = functools.reduce(numpy.dot, (C_.T, self.S, C_))
                es_, vs_ = eigh(S_)
                s_ = numpy.sqrt(es_)
                s_ = numpy.diag(1.0 / s_)
                W_ = functools.reduce(numpy.dot, (vs_, s_, vs_.T))
                W_ = numpy.dot(C_, W_)

            self.W = get_loc(
                self.mol, W_, "PM", pop_method=pop_method, init_guess=init_guess
            )

            if not self.frozen_core:
                self.lmo_coeff = self.W.T @ self.S @ self.C
            else:
                self.lmo_coeff = self.W.T @ self.S @ self.C[:, self.ncore :]

        elif lo_method == "iao":
            loc_type = "SO"
            val_basis = "sto-3g"

            # Occupied mo_coeff (with core)
            Co = self.C[:, : self.Nocc]
            # Get necessary overlaps, second arg is IAO basis
            S12, S2 = get_xovlp(self.mol, basis=val_basis)
            # Use these to get IAOs
            Ciao = get_iao(Co, S12, self.S, S2=S2)

            if not valence_only:
                # Now get PAOs
                if loc_type.upper() != "SO":
                    Cpao = get_pao(Ciao, self.S, S12, S2, self.mol)
                elif loc_type.upper() == "SO":
                    Cpao = get_pao_native(
                        Ciao, self.S, self.mol, valence_basis=val_basis
                    )

            # rearrange by atom
            aoind_by_atom = get_aoind_by_atom(self.mol)
            Ciao, iaoind_by_atom = reorder_by_atom_(Ciao, aoind_by_atom, self.S)

            if not valence_only:
                Cpao, paoind_by_atom = reorder_by_atom_(Cpao, aoind_by_atom, self.S)

            if self.frozen_core:
                # Remove core MOs
                Cc = self.C[:, : self.ncore]  # Assumes core are first
                Ciao = remove_core_mo(Ciao, Cc, self.S)

            # Localize orbitals beyond symm orth
            if loc_type.upper() != "SO":
                Ciao = get_loc(self.mol, Ciao, loc_type)
                if not valence_only:
                    Cpao = get_loc(self.mol, Cpao, loc_type)

            shift = 0
            ncore = 0
            if not valence_only:
                Wstack = numpy.zeros(
                    (Ciao.shape[0], Ciao.shape[1] + Cpao.shape[1])
                )  # -self.ncore))
            else:
                Wstack = numpy.zeros((Ciao.shape[0], Ciao.shape[1]))

            if self.frozen_core:
                for ix in range(self.mol.natm):
                    nc = ncore_(self.mol.atom_charge(ix))
                    ncore += nc
                    niao = len(iaoind_by_atom[ix])
                    iaoind_ix = [i_ - ncore for i_ in iaoind_by_atom[ix][nc:]]
                    Wstack[:, shift : shift + niao - nc] = Ciao[:, iaoind_ix]
                    shift += niao - nc
                    if not valence_only:
                        npao = len(paoind_by_atom[ix])
                        Wstack[:, shift : shift + npao] = Cpao[:, paoind_by_atom[ix]]
                        shift += npao
            else:
                if not hstack:
                    for ix in range(self.mol.natm):
                        niao = len(iaoind_by_atom[ix])
                        Wstack[:, shift : shift + niao] = Ciao[:, iaoind_by_atom[ix]]
                        shift += niao
                        if not valence_only:
                            npao = len(paoind_by_atom[ix])
                            Wstack[:, shift : shift + npao] = Cpao[
                                :, paoind_by_atom[ix]
                            ]
                            shift += npao
                else:
                    Wstack = numpy.hstack((Ciao, Cpao))
            if not nosave:
                self.W = Wstack
                assert numpy.allclose(
                    self.W.T @ self.S @ self.W, numpy.eye(self.W.shape[1])
                )
            else:
                assert numpy.allclose(
                    Wstack.T @ self.S @ Wstack, numpy.eye(Wstack.shape[1])
                )
                return Wstack
            nmo = self.C.shape[1] - self.ncore
            nlo = self.W.shape[1]

            if not valence_only:
                if nmo > nlo:
                    Co_nocore = self.C[:, self.ncore : self.Nocc]
                    Cv = self.C[:, self.Nocc :]
                    # Ensure that the LOs span the occupied space
                    assert numpy.allclose(
                        numpy.sum((self.W.T @ self.S @ Co_nocore) ** 2.0),
                        self.Nocc - self.ncore,
                    )
                    # Find virtual orbitals that lie in the span of LOs
                    u, l, vt = svd(self.W.T @ self.S @ Cv, full_matrices=False)
                    unused(u)
                    nvlo = nlo - self.Nocc - self.ncore
                    assert numpy.allclose(numpy.sum(l[:nvlo]), nvlo)
                    C_ = numpy.hstack([Co_nocore, Cv @ vt[:nvlo].T])
                    self.lmo_coeff = self.W.T @ self.S @ C_
                else:
                    self.lmo_coeff = self.W.T @ self.S @ self.C[:, self.ncore :]
            else:
                self.lmo_coeff = self.W.T @ self.S @ self.C[:, self.ncore :]

        elif lo_method == "boys":
            es_, vs_ = eigh(self.S)
            edx = es_ > 1.0e-15
            W_ = numpy.dot(vs_[:, edx] / numpy.sqrt(es_[edx]), vs_[:, edx].T)
            if self.frozen_core:
                P_core = numpy.eye(W_.shape[0]) - numpy.dot(self.P_core, self.S)
                C_ = numpy.dot(P_core, W_)
                Cpop = functools.reduce(numpy.dot, (C_.T, self.S, C_))
                Cpop = numpy.diag(Cpop)
                no_core_idx = numpy.where(Cpop > 0.55)[0]
                C_ = C_[:, no_core_idx]
                S_ = functools.reduce(numpy.dot, (C_.T, self.S, C_))
                es_, vs_ = eigh(S_)
                s_ = numpy.sqrt(es_)
                s_ = numpy.diag(1.0 / s_)
                W_ = functools.reduce(numpy.dot, (vs_, s_, vs_.T))
                W_ = numpy.dot(C_, W_)

            self.W = get_loc(self.mol, W_, "BOYS")

            if not self.frozen_core:
                self.lmo_coeff = self.W.T @ self.S @ self.C
            else:
                self.lmo_coeff = self.W.T @ self.S @ self.C[:, self.ncore :]

        else:
            print("lo_method = ", lo_method, " not implemented!", flush=True)
            print("exiting", flush=True)
            sys.exit()

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
