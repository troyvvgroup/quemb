# Author(s): Minsik Cho, Leah Weisburn

"""🍠
Bootstrap Embedding Calculation with an
Unrestricted Hartree-Fock Bath

Reference
  Tran, H.; Ye, H.; Van Voorhis, T.
  J. Chem. Phys. 153, 214101 (2020)

TODO
  Iterative UBE (edge matching) works for common_bath; equal_bath still
  only supports chemical-potential optimization.
"""

from pathlib import Path
from warnings import warn

import h5py
import numpy as np
from numpy import array, einsum, zeros_like
from numpy.linalg import multi_dot
from pyscf import ao2mo
from pyscf.scf.uhf import UHF

from quemb.molbe.be_parallel import be_func_parallel_u
from quemb.molbe.fragment import FragPart
from quemb.molbe.mbe import BE
from quemb.molbe.pfrag import (
    Frags,
    schmidt_decomposition_common,
)
from quemb.molbe.solver import be_func_u
from quemb.shared.external.optqn import FrankQN, get_be_error_jacobian_u
from quemb.shared.helper import unused
from quemb.shared.manage_scratch import WorkDir
from quemb.shared.typing import PathLike


class UBE(BE):  # 🍠
    def __init__(
        self,
        mf: UHF,
        fobj: FragPart,
        scratch_dir: WorkDir | None = None,
        eri_file: PathLike = "eri_file.h5",
        lo_method: PathLike = "lowdin",
        pop_method: str | None = None,
        compute_hf: bool = True,
        thr_bath: float = 1.0e-10,
        equal_bath: bool = True,
        common_bath: bool = False,
        nelec_prescription_override: dict | None = None,
        use_df: bool = False,
    ) -> None:
        """Initialize Unrestricted BE Object (ube🍠)

        .. note::
            Currently only supports embedding Hamiltonian construction for molecular
            systems In conjunction with molbe.misc.ube2fcidump,
            embedding Hamiltonians can be written for external use.
            See :python:`unrestricted` branch for a work-in-progress full implmentation

        Parameters
        ----------
        mf :
            pyscf meanfield UHF object
        fobj :
            object that contains fragment information
        eri_file :
            h5py file with ERIs
        lo_method :
            Method for orbital localization, by default "lowdin"
        pop_method :
            Method for calculating orbital population, by default 'meta-lowdin'
            See pyscf.lo for more details and options
        thr_bath :
            Threshold for bath orbitals in Schmidt decomposition
        equal_bath :
            Whether to use a bath with the same number of alpha and beta orbitals.
            Using equal_bath = False will require custom compiled functions in
            PySCF to perform integral transformations. Default is True
        common_bath :
            Use SVD-based common alpha/beta bath. Produces a single TA
            shared by both spin channels. Supersedes equal_bath when
            True. Default False.
        nelec_prescription_override : dict, optional
            Override automatic integer electron assignment for specific
            fragments. Keys are fragment indices, values are (nalpha, nbeta)
            tuples. E.g. {0: (21, 16)} fixes fragment 0 to 21alpha, 16beta.
            Use when automatic assignment gives physically wrong spin.
        """

        self.unrestricted = True
        self.thr_bath = thr_bath
        self.equal_bath = equal_bath
        self.common_bath = common_bath
        self.nelec_prescription_override = nelec_prescription_override or {}
        self.use_df = use_df
        if use_df:
            assert hasattr(mf, "with_df") and mf.with_df is not None, (
                "use_df=True requires a density-fitted mf: construct as scf.UHF(mol).density_fit()"
            )

        self.fobj = fobj

        self.ebe_hf = 0.0
        self.ebe_tot = 0.0

        self.mo_energy = mf.mo_energy

        self.mf = mf
        assert mf.mo_coeff is not None
        self.Nocc = [mf.mol.nelec[0], mf.mol.nelec[1]]
        self.enuc = mf.energy_nuc()

        self.hcore = mf.get_hcore()
        self.S = mf.get_ovlp()
        self.C = [array(mf.mo_coeff[0]), array(mf.mo_coeff[1])]
        self.hf_dm = [mf.make_rdm1()[0], mf.make_rdm1()[1]]
        self.hf_veff = [mf.get_veff()[0], mf.get_veff()[1]]

        self.hf_etot = mf.e_tot
        self.W = None
        self.lmo_coeff = None
        self.cinv = None

        self.print_ini()

        self.Fobjs_a: list[Frags] = []
        self.Fobjs_b: list[Frags] = []

        # Note: this file's own initialize_pot() (not mbe.py's) already
        # appends two chemical-potential slots (alpha, beta).
        self.pot = initialize_pot(self.fobj.n_frag, self.fobj.relAO_per_edge_per_frag)

        self.eri_file = Path(eri_file)
        self.frozen_core = fobj.frozen_core
        self.ncore = 0
        self.E_core = 0
        self.C_core = None
        self.P_core = None
        self.core_veff = None

        if self.frozen_core:
            assert not (
                fobj.ncore is None or fobj.no_core_idx is None or fobj.core_list is None
            )
            self.ncore = fobj.ncore
            self.no_core_idx = fobj.no_core_idx
            self.core_list = fobj.core_list

            self.Nocc[0] -= self.ncore
            self.Nocc[1] -= self.ncore

            self.hf_dm = [
                self.C[s][:, self.ncore : self.ncore + self.Nocc[s]]
                @ self.C[s][:, self.ncore : self.ncore + self.Nocc[s]].T
                for s in [0, 1]
            ]
            self.C_core = [self.C[s][:, : self.ncore] for s in [0, 1]]
            self.P_core = [self.C_core[s] @ self.C_core[s].T for s in [0, 1]]
            self.core_veff = 1.0 * mf.get_veff(dm=self.P_core)

            self.E_core = (
                sum(
                    [
                        einsum(
                            "ji,ji->",
                            2 * self.hcore + self.core_veff[s],
                            self.P_core[s],
                        )
                        for s in [0, 1]
                    ]
                )
                * 0.5
            )

        # iao ignored for now
        self.C_a = array(mf.mo_coeff[0])
        self.C_b = array(mf.mo_coeff[1])
        del self.C

        self.localize(
            lo_method,
            iao_valence_basis=fobj.iao_valence_basis,
            iao_valence_only=fobj.iao_valence_only,
            pop_method=pop_method,
        )

        if scratch_dir is None:
            self.scratch_dir = WorkDir.from_environment()
        else:
            self.scratch_dir = scratch_dir
        self.eri_file = self.scratch_dir / eri_file

        self.initialize(None if self.use_df else mf._eri, compute_hf)

    def initialize(self, eri_, compute_hf):
        if compute_hf:
            E_hf = 0.0
        EH1 = 0.0
        ECOUL = 0.0

        file_eri = h5py.File(self.eri_file, "w")
        # alpha orbitals
        self.Fobjs_a = [
            self.fobj.to_Frags(I, eri_file=self.eri_file, unrestricted=True)
            for I in range(self.fobj.n_frag)
        ]
        # beta
        self.Fobjs_b = [
            self.fobj.to_Frags(I, eri_file=self.eri_file, unrestricted=True)
            for I in range(self.fobj.n_frag)
        ]

        all_noccs = [self.Nocc for _ in range(self.fobj.n_frag)]

        for I in range(self.fobj.n_frag):
            fobj_a = self.Fobjs_a[I]
            fobj_b = self.Fobjs_b[I]

            if self.frozen_core:
                fobj_a.core_veff = self.core_veff[0]
                fobj_b.core_veff = self.core_veff[1]
            else:
                fobj_a.core_veff = None
                fobj_b.core_veff = None

            W = self.W[0] if self.frozen_core else self.W
            W_b = self.W[1] if self.frozen_core else self.W

            if self.common_bath:
                TA_lo_eo, n_f, n_b = schmidt_decomposition_common(
                    self.lmo_coeff_a,
                    self.lmo_coeff_b,
                    self.Nocc[0],
                    self.Nocc[1],
                    fobj_a.AO_in_frag,
                    thr_bath=self.thr_bath,
                )
                TA = W @ TA_lo_eo

                for fobj in [fobj_a, fobj_b]:
                    fobj.TA_lo_eo = TA_lo_eo.copy()
                    fobj.TA = TA.copy()
                    fobj.n_f = n_f
                    fobj.n_b = n_b
                    fobj.nao = TA.shape[1]

            else:
                # Original separate alpha/beta Schmidt decompositions
                fobj_a.sd(
                    W,
                    self.lmo_coeff_a,
                    self.Nocc[0],
                    thr_bath=self.thr_bath,
                )
                fobj_b.sd(
                    W_b,
                    self.lmo_coeff_b,
                    self.Nocc[1],
                    thr_bath=self.thr_bath,
                )

                if self.equal_bath:
                    tot_alpha = fobj_a.n_f + fobj_a.n_b
                    tot_beta = fobj_b.n_f + fobj_b.n_b
                    if tot_alpha > tot_beta:
                        fobj_b.sd(
                            W_b,
                            self.lmo_coeff_b,
                            self.Nocc[1],
                            thr_bath=self.thr_bath,
                            norb=fobj_a.n_b,
                        )
                    elif tot_beta > tot_alpha:
                        fobj_a.sd(
                            W,
                            self.lmo_coeff_a,
                            self.Nocc[0],
                            thr_bath=self.thr_bath,
                            norb=fobj_b.n_b,
                        )

            assert fobj_a.TA is not None and fobj_b.TA is not None
            if self.use_df:
                eri_a = self.mf.with_df.ao2mo(fobj_a.TA, compact=True)
                eri_b = self.mf.with_df.ao2mo(fobj_b.TA, compact=True)
                eri_ab = self.mf.with_df.ao2mo(
                    (fobj_a.TA, fobj_a.TA, fobj_b.TA, fobj_b.TA), compact=True
                )
            else:
                assert eri_ is not None, "eri_ is None: set incore_anyway for UHF"
                eri_a = ao2mo.incore.full(eri_, fobj_a.TA, compact=True)
                eri_b = ao2mo.incore.full(eri_, fobj_b.TA, compact=True)
                # cross-spin ERI term
                eri_ab = ao2mo.incore.general(
                    eri_,
                    (fobj_a.TA, fobj_a.TA, fobj_b.TA, fobj_b.TA),
                    compact=True,
                )

            file_eri.create_dataset(fobj_a.dname[0], data=eri_a)
            file_eri.create_dataset(fobj_a.dname[1], data=eri_b)
            file_eri.create_dataset(fobj_a.dname[2], data=eri_ab)

            # sab = self.C_a @ self.S @ self.C_b
            _ = fobj_a.get_nsocc(self.S, self.C_a, self.Nocc[0], ncore=self.ncore)
            if I in self.nelec_prescription_override:
                na_ov, _nb_ov = self.nelec_prescription_override[I]
                print(
                    f"  Fragment {I} nsocc_a override: {fobj_a.nsocc} -> {na_ov}",
                    flush=True,
                )
                fobj_a.nsocc = na_ov

            fobj_a.h1 = multi_dot((fobj_a.TA.T, self.hcore, fobj_a.TA))
            eri_a = ao2mo.restore(8, eri_a, fobj_a.nao)
            # Both spin densities are passed undoubled; the doubled-density
            # restricted formula does not apply here (see cons_fock).
            fobj_a.cons_fock(
                self.hf_veff[0],
                self.S,
                self.hf_dm[0],
                eri_=eri_a,
                dm_other=self.hf_dm[1],
            )

            fobj_a.hf_veff = self.hf_veff[0]
            fobj_a.heff = zeros_like(fobj_a.h1)
            # Project the full-system OTHER spin density into fobj_a's
            # embedding basis (same S@TA pattern get_veff uses internally)
            # so the embedded SCF below uses the unrestricted get_scfObj path.
            ST_a = self.S @ fobj_a.TA
            dm_b_embedded = multi_dot((ST_a.T, self.hf_dm[1], ST_a))
            # Stash this for reuse by be_func_u/run_solver_u later, instead
            # of re-projecting fobj_b's own relaxed dm0 -- that lives in
            # fobj_b's own basis, which only coincides with fobj_a's under
            # common_bath and has no clean meaning once re-expressed in a
            # different fragment's bath space otherwise.
            fobj_a.dm_other_embedded = dm_b_embedded
            fobj_a.scf(
                fs=True,
                eri=eri_a,
                unrestricted=True,
                spin_ind=0,
                dm_other=dm_b_embedded,
            )
            fobj_a.dm0 = (
                fobj_a._mo_coeffs[:, : fobj_a.nsocc]
                @ fobj_a._mo_coeffs[:, : fobj_a.nsocc].conj().T
            )

            if compute_hf:
                eh1_a, ecoul_a, ef_a = fobj_a.update_ebe_hf(
                    return_e=True,
                    unrestricted=True,
                    spin_ind=0,
                    dm_hf_other=dm_b_embedded,
                )
                unused(ef_a)
                EH1 += eh1_a
                ECOUL += ecoul_a
                E_hf += fobj_a.ebe_hf

            _ = fobj_b.get_nsocc(self.S, self.C_b, self.Nocc[1], ncore=self.ncore)
            if I in self.nelec_prescription_override:
                _na_ov, nb_ov = self.nelec_prescription_override[I]
                print(
                    f"  Fragment {I} nsocc_b override: {fobj_b.nsocc} -> {nb_ov}",
                    flush=True,
                )
                fobj_b.nsocc = nb_ov

            fobj_b.h1 = multi_dot((fobj_b.TA.T, self.hcore, fobj_b.TA))
            eri_b = ao2mo.restore(8, eri_b, fobj_b.nao)
            fobj_b.cons_fock(
                self.hf_veff[1],
                self.S,
                self.hf_dm[1],
                eri_=eri_b,
                dm_other=self.hf_dm[0],
            )

            fobj_b.hf_veff = self.hf_veff[1]
            fobj_b.heff = zeros_like(fobj_b.h1)
            ST_b = self.S @ fobj_b.TA
            dm_a_embedded = multi_dot((ST_b.T, self.hf_dm[0], ST_b))
            fobj_b.dm_other_embedded = dm_a_embedded
            fobj_b.scf(
                fs=True,
                eri=eri_b,
                unrestricted=True,
                spin_ind=1,
                dm_other=dm_a_embedded,
            )

            fobj_b.dm0 = (
                fobj_b._mo_coeffs[:, : fobj_b.nsocc]
                @ fobj_b._mo_coeffs[:, : fobj_b.nsocc].conj().T
            )

            if compute_hf:
                eh1_b, ecoul_b, ef_b = fobj_b.update_ebe_hf(
                    return_e=True,
                    unrestricted=True,
                    spin_ind=1,
                    dm_hf_other=dm_a_embedded,
                )
                unused(ef_b)
                EH1 += eh1_b
                ECOUL += ecoul_b
                E_hf += fobj_b.ebe_hf

        # Fractional vs. rounded alpha/beta electron count per fragment,
        # from get_nsocc()'s projection. Large deviations flag fragments
        # where the integer assignment is ambiguous -- rerun with
        # nelec_prescription_override={frag_idx: (na, nb)} to try a
        # different combination for that fragment.
        print(f"\n{'=' * 70}", flush=True)
        print("Fragment electron-count diagnostic", flush=True)
        print(f"{'=' * 70}", flush=True)
        print(
            f"  {'Frag':>4}  {'na_frac':>9}  {'na':>4}  {'dev_a':>7}  "
            f"{'nb_frac':>9}  {'nb':>4}  {'dev_b':>7}",
            flush=True,
        )
        print(f"  {'-' * 60}", flush=True)
        thr = 0.1
        any_flag = False
        for I, (fobj_a, fobj_b) in enumerate(zip(self.Fobjs_a, self.Fobjs_b)):
            dev_a = abs(fobj_a.nsocc_frac - fobj_a.nsocc)
            dev_b = abs(fobj_b.nsocc_frac - fobj_b.nsocc)
            flag = "*" if (dev_a > thr or dev_b > thr) else " "
            any_flag = any_flag or flag == "*"
            print(
                f" {flag}{I:>4}  {fobj_a.nsocc_frac:>9.4f}  "
                f"{fobj_a.nsocc:>4}  {dev_a:>7.4f}  "
                f"{fobj_b.nsocc_frac:>9.4f}  {fobj_b.nsocc:>4}  {dev_b:>7.4f}",
                flush=True,
            )
        print(f"  {'-' * 60}", flush=True)
        if any_flag:
            print(
                f"  * deviation > {thr} -- consider "
                "nelec_prescription_override for that fragment.",
                flush=True,
            )
        print(f"{'=' * 70}\n", flush=True)

        # nsocc here comes from get_nsocc()'s native projection onto the
        # shared common_bath TA (fragment+bath, overlapping across
        # fragments) -- the same formula equal_bath already uses, and
        # it's already close to integer per fragment without further
        # correction.
        #
        # common_bath previously overwrote these with a fragment-only
        # (non-overlapping) democratic-partitioning count, rounded by
        # largest remainder so per-fragment counts summed exactly to the
        # system total. That left the fragments with zero slack, forcing
        # chemical-potential matching of the global electron-count
        # constraint (Eq. 16, Tran/Ye/Van Voorhis 2020) into pushing all
        # density onto center sites with none left for bath orbitals --
        # an extreme solution rather than a gentle correction.
        orb_count_a = [(frag.n_f, frag.n_b) for frag in self.Fobjs_a]
        orb_count_b = [(frag.n_f, frag.n_b) for frag in self.Fobjs_b]

        file_eri.close()

        print("Number of Orbitals per Fragment:", flush=True)
        print(
            "____________________________________________________________________",
            flush=True,
        )
        print(
            "| Fragment |    Nocc   | Fragment Orbs | Bath Orbs | Schmidt Space |",
            flush=True,
        )
        print(
            "____________________________________________________________________",
            flush=True,
        )
        for I in range(self.fobj.n_frag):
            print(
                "|    {:>2}    | ({:>3},{:>3}) |   ({:>3},{:>3})   | ({:>3},{:>3}) |   ({:>3},{:>3})   |".format(  # noqa: E501
                    I,
                    all_noccs[I][0],
                    all_noccs[I][1],
                    orb_count_a[I][0],
                    orb_count_b[I][0],
                    orb_count_a[I][1],
                    orb_count_b[I][1],
                    orb_count_a[I][0] + orb_count_a[I][1],
                    orb_count_b[I][0] + orb_count_b[I][1],
                ),
                flush=True,
            )
        print(
            "____________________________________________________________________",
            flush=True,
        )
        if compute_hf:
            hf_err = self.hf_etot - (E_hf + self.enuc + self.E_core)
            self.ebe_hf = E_hf + self.enuc + self.E_core
            print(f"HF-in-HF error                 :  {hf_err:>.4e} Ha")
            if abs(hf_err) > 1.0e-5:
                warn("Large HF-in-HF energy error")
                print("eh1 ", EH1)
                print("ecoul ", ECOUL)

        couti = 0
        for fobj in self.Fobjs_a:
            fobj.udim = couti
            couti = fobj.set_udim(couti)

        couti = 0
        for fobj in self.Fobjs_b:
            fobj.udim = couti
            couti = fobj.set_udim(couti)

    def oneshot(self, solver="UCCSD", nproc=1, ompnum=4, relax_density=False):
        if nproc == 1:
            E, E_comp = be_func_u(
                None,
                list(zip(self.Fobjs_a, self.Fobjs_b)),
                solver,
                self.enuc,
                hf_veff=self.hf_veff,
                eeval=True,
                relax_density=relax_density,
                frozen=self.frozen_core,
            )
        else:
            E, E_comp = be_func_parallel_u(
                pot=None,
                Fobjs=list(zip(self.Fobjs_a, self.Fobjs_b)),
                solver=solver,
                enuc=self.enuc,
                hf_veff=self.hf_veff,
                nproc=nproc,
                ompnum=ompnum,
                relax_density=relax_density,
                frozen=self.frozen_core,
            )
        unused(E_comp)

        print("-----------------------------------------------------", flush=True)
        print("             One Shot BE ", flush=True)
        print("             Solver : ", solver, flush=True)
        print("-----------------------------------------------------", flush=True)
        print(flush=True)

        self.ebe_tot = E + self.hf_etot
        print(
            "Total Energy : {:>12.8f} Ha".format(
                (self.ebe_tot),
            )
        )
        print(
            "Corr  Energy : {:>12.8f} Ha".format(
                (E),
            )
        )

    def optimize(
        self,
        solver="UCCSD",
        only_chem=True,
        conv_tol=1.0e-6,
        max_iter=500,
        relax_density=False,
        use_cumulant=True,
        trust_region=False,
    ):
        """BE0-level or fully iterative optimization for UBE.

        Solves for the global alpha/beta chemical potentials (mu_alpha,
        mu_beta) such that the fragment-summed, center-site-restricted
        embedded 1RDM correctly reproduces the true system's alpha/beta
        electron counts (Tran, Ye, Van Voorhis, J. Chem. Phys. 153, 214101
        (2020), Eq. 16) -- the "UBE0" level.

        In practice, get_nsocc()'s native fragment+bath electron counts
        already satisfy Eq. 16 closely at mu=0 for both common_bath and
        equal_bath, so only_chem=True mainly matters for systems where
        that isn't the case.

        only_chem=False additionally fits shared edge potentials so the
        spin-summed (P^alpha + P^beta) embedded 1RDM matches at fragment
        edges too (Eq. 15). Requires common_bath=True, since that's what
        gives alpha and beta a shared embedding basis to match in.

        Parameters
        ----------
        only_chem : bool
            If True (default), fit only the chemical potentials. If False,
            also fit shared edge potentials; requires common_bath=True and
            n_BE >= 2.
        trust_region : bool
            Use trust-region based QN optimization instead of the default
            line-search step, by default False. Matches restricted BE's
            optimize() option of the same name.
        """
        if not only_chem:
            if not self.common_bath:
                raise ValueError(
                    "only_chem=False requires common_bath=True: alpha and "
                    "beta need a shared embedding basis to do spin-summed "
                    "edge matching in."
                )
            if self.fobj.n_BE == 1:
                raise ValueError(
                    "BE1 has no fragment edges to match. Set only_chem=True."
                )

        Fobjs_ab = list(zip(self.Fobjs_a, self.Fobjs_b))
        Nocc_ab = (self.Nocc[0], self.Nocc[1])
        state: dict = {"err": None, "E": None, "E_comp": None}

        def objfunc(xk):
            ernorm, ervec, (E, E_comp) = be_func_u(
                list(xk),
                Fobjs_ab,
                solver,
                self.enuc,
                hf_veff=self.hf_veff,
                eeval=True,
                relax_density=relax_density,
                use_cumulant=use_cumulant,
                frozen=self.frozen_core,
                only_chem=only_chem,
                return_vec=True,
                Nocc_ab=Nocc_ab,
            )
            state["err"] = ernorm
            state["E"] = E
            state["E_comp"] = E_comp
            return ervec

        print("-----------------------------------------------------", flush=True)
        print(
            "     Starting UBE chemical-potential optimization    "
            if only_chem
            else "     Starting full iterative UBE optimization        ",
            flush=True,
        )
        print("-----------------------------------------------------", flush=True)

        x0 = array(self.pot[-2:] if only_chem else self.pot, dtype=float)
        f0 = objfunc(x0)
        print(f"Initial error: {f0}", flush=True)
        print(f"RMS error: {state['err']:.4e}", flush=True)

        xfinal = x0
        if state["err"] < conv_tol:
            print("CONVERGED w/o optimization steps", flush=True)
        else:
            if only_chem:
                J0 = np.eye(len(x0))
            else:
                # HF-level analytic Jacobian seed (Fobjs_ab's fragments already
                # carry the embedded-UHF-level ._mf orbitals set by the
                # f0 = objfunc(x0) call above). Identity seeding does not
                # converge well for this branch; see JACOBIAN_SEED_HANDOFF.md.
                J0 = get_be_error_jacobian_u(Fobjs_ab)
            optQN = FrankQN(objfunc, x0, f0, J0, max_space=max_iter)
            converged = False
            for it in range(max_iter):
                optQN.next_step(it, trust_region=trust_region)
                print(f"-- iter {it}: RMS error = {state['err']:.4e}", flush=True)
                if state["err"] < conv_tol:
                    print("CONVERGED", flush=True)
                    converged = True
                    break
            xfinal = optQN.xnew
            if not converged:
                warn(f"UBE optimization did not converge in {max_iter} steps")

        if only_chem:
            self.pot[-2] = float(xfinal[0])
            self.pot[-1] = float(xfinal[1])
        else:
            self.pot = [float(x) for x in xfinal]

        E, E_comp = state["E"], state["E_comp"]
        unused(E_comp)
        self.ebe_tot = E + self.hf_etot
        print(f"Total Energy : {self.ebe_tot:>12.8f} Ha", flush=True)
        print(f"Corr  Energy : {E:>12.8f} Ha", flush=True)

    def urdm1_fullbasis(self, return_ao=True):
        """Assemble full-system alpha and beta 1-RDMs via democratic partitioning.

        Returns
        -------
        rdm1a_AO, rdm1b_AO : numpy.ndarray
          Alpha and beta 1-RDMs in the AO basis.
          Spin density = rdm1a_AO - rdm1b_AO.
        """
        from numpy import zeros

        nao = self.S.shape[0]
        rdm1a_AO = zeros((nao, nao))
        rdm1b_AO = zeros((nao, nao))

        def get_mo(fobj):
            if hasattr(fobj, "mo_coeff_uccsd"):
                return fobj.mo_coeff_uccsd
            if fobj._mf is not None:
                return fobj._mf.mo_coeff
            return fobj._mo_coeffs

        for fobj_a, fobj_b in zip(self.Fobjs_a, self.Fobjs_b):
            cind = [fobj_a.AO_in_frag[i] for i in fobj_a.weight_and_relAO_per_center[1]]

            # Build the projector in the LOCAL embedding-space basis (matching
            # the restricted rdm1_fullbasis pattern), not in full AO space.
            Pc_a = (
                fobj_a.TA.T
                @ self.S
                @ self.W[:, cind]
                @ self.W[:, cind].T
                @ self.S
                @ fobj_a.TA
            )
            Pc_b = (
                fobj_b.TA.T
                @ self.S
                @ self.W[:, cind]
                @ self.W[:, cind].T
                @ self.S
                @ fobj_b.TA
            )

            mca = get_mo(fobj_a)
            mcb = get_mo(fobj_b)

            # Local density in the embedding-orbital AO-equivalent space
            rdm1a_eo = mca @ fobj_a.rdm1__ @ mca.T
            rdm1b_eo = mcb @ fobj_b.rdm1__ @ mcb.T

            # Project in the SMALL local space, THEN expand to full AO space
            rdm1a_center = Pc_a @ rdm1a_eo
            rdm1b_center = Pc_b @ rdm1b_eo

            rdm1a_AO += fobj_a.TA @ rdm1a_center @ fobj_a.TA.T
            rdm1b_AO += fobj_b.TA @ rdm1b_center @ fobj_b.TA.T

        rdm1a_AO = (rdm1a_AO + rdm1a_AO.T) / 2.0
        rdm1b_AO = (rdm1b_AO + rdm1b_AO.T) / 2.0

        return rdm1a_AO, rdm1b_AO


def initialize_pot(n_frag, relAO_per_edge):
    pot_ = []

    if relAO_per_edge:
        for I in range(n_frag):
            for i in relAO_per_edge[I]:
                for j in range(len(i)):
                    for k in range(len(i)):
                        if j > k:
                            continue
                        pot_.append(0.0)

    pot_.append(0.0)  # alpha
    pot_.append(0.0)  # beta
    return pot_
