# Author(s): Minsik Cho, Leah Weisburn

"""🍠
Bootstrap Embedding Calculation with an
Unrestricted Hartree-Fock Bath

Reference
  Tran, H.; Ye, H.; Van Voorhis, T.
  J. Chem. Phys. 153, 214101 (2020)

TODO
  Add iterative UBE
"""

from pathlib import Path
from warnings import warn

import h5py
from numpy import array, einsum, zeros_like
from pyscf import ao2mo
from pyscf.scf.uhf import UHF

from quemb.molbe.be_parallel import be_func_parallel_u
from quemb.molbe.fragment import FragPart
from quemb.molbe.mbe import BE
from quemb.molbe.pfrag import Frags
from quemb.molbe.solver import be_func_u
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
            Method for orbital localization. Supports 'lowdin', 'boys', and 'wannier',
            by default "lowdin"
        pop_method :
            Method for calculating orbital population, by default 'meta-lowdin'
            See pyscf.lo for more details and options
        """
        self.unrestricted = True

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

        self.pot = initialize_pot(self.fobj.n_frag, self.fobj.edge_idx)

        self.eri_file = Path(eri_file)
        self.ek = 0.0
        self.frozen_core = fobj.frozen_core
        self.ncore = 0
        self.E_core = 0
        self.C_core = None
        self.P_core = None
        self.core_veff = None

        self.uhf_full_e = mf.e_tot

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

        self.initialize(mf._eri, compute_hf)

    def initialize(self, eri_, compute_hf):
        if compute_hf:
            E_hf = 0.0
        EH1 = 0.0
        ECOUL = 0.0

        file_eri = h5py.File(self.eri_file, "w")
        lentmp = len(self.fobj.edge_idx)

        # alpha orbitals
        for I in range(self.fobj.n_frag):
            if lentmp:
                fobjs_a = Frags(
                    self.fobj.AO_per_frag[I],
                    I,
                    edge=self.fobj.AO_per_edge_per_frag[I],
                    eri_file=self.eri_file,
                    ref_frag_idx_per_edge=self.fobj.ref_frag_idx_per_edge[I],
                    edge_idx=self.fobj.edge_idx[I],
                    center_idx=self.fobj.center_idx[I],
                    efac=self.fobj.ebe_weight[I],
                    centerf_idx=self.fobj.centerf_idx[I],
                    unrestricted=True,
                )
            else:
                fobjs_a = Frags(
                    self.fobj.AO_per_frag[I],
                    I,
                    edge=[],
                    ref_frag_idx_per_edge=[],
                    eri_file=self.eri_file,
                    edge_idx=[],
                    center_idx=[],
                    centerf_idx=[],
                    efac=self.fobj.ebe_weight[I],
                    unrestricted=True,
                )
            self.Fobjs_a.append(fobjs_a)
        # beta
        for I in range(self.fobj.n_frag):
            if lentmp:
                fobjs_b = Frags(
                    self.fobj.AO_per_frag[I],
                    I,
                    edge=self.fobj.AO_per_edge_per_frag[I],
                    eri_file=self.eri_file,
                    ref_frag_idx_per_edge=self.fobj.ref_frag_idx_per_edge[I],
                    edge_idx=self.fobj.edge_idx[I],
                    center_idx=self.fobj.center_idx[I],
                    efac=self.fobj.ebe_weight[I],
                    centerf_idx=self.fobj.centerf_idx[I],
                    unrestricted=True,
                )
            else:
                fobjs_b = Frags(
                    self.fobj.AO_per_frag[I],
                    I,
                    edge=[],
                    ref_frag_idx_per_edge=[],
                    eri_file=self.eri_file,
                    edge_idx=[],
                    center_idx=[],
                    centerf_idx=[],
                    efac=self.fobj.ebe_weight[I],
                    unrestricted=True,
                )
            self.Fobjs_b.append(fobjs_b)

        orb_count_a = []
        orb_count_b = []

        all_noccs = []

        for I in range(self.fobj.n_frag):
            fobj_a = self.Fobjs_a[I]
            fobj_b = self.Fobjs_b[I]

            if self.frozen_core:
                fobj_a.core_veff = self.core_veff[0]
                fobj_b.core_veff = self.core_veff[1]
                orb_count_a.append(
                    fobj_a.sd(
                        self.W[0], self.lmo_coeff_a, self.Nocc[0], return_orb_count=True
                    )
                )
                orb_count_b.append(
                    fobj_b.sd(
                        self.W[1], self.lmo_coeff_b, self.Nocc[1], return_orb_count=True
                    )
                )
            else:
                fobj_a.core_veff = None
                fobj_b.core_veff = None
                orb_count_a.append(
                    fobj_a.sd(
                        self.W, self.lmo_coeff_a, self.Nocc[0], return_orb_count=True
                    )
                )
                orb_count_b.append(
                    fobj_b.sd(
                        self.W, self.lmo_coeff_b, self.Nocc[1], return_orb_count=True
                    )
                )

            all_noccs.append(self.Nocc)

            if eri_ is None and self.mf.with_df is not None:
                # NOT IMPLEMENTED: should not be called, as no unrestricted DF tested
                # for density-fitted integrals; if mf is provided, pyscf.ao2mo uses DF
                # object in an outcore fashion
                eri_a = ao2mo.kernel(self.mf.mol, fobj_a.TA, compact=True)
                eri_b = ao2mo.kernel(self.mf.mol, fobj_b.TA, compact=True)
            else:
                eri_a = ao2mo.incore.full(
                    eri_, fobj_a.TA, compact=True
                )  # otherwise, do an incore ao2mo
                eri_b = ao2mo.incore.full(eri_, fobj_b.TA, compact=True)

                Csd_A = fobj_a.TA  # may have to add in nibath here
                Csd_B = fobj_b.TA

                # cross-spin ERI term
                eri_ab = ao2mo.incore.general(
                    eri_, (Csd_A, Csd_A, Csd_B, Csd_B), compact=True
                )

            file_eri.create_dataset(fobj_a.dname[0], data=eri_a)
            file_eri.create_dataset(fobj_a.dname[1], data=eri_b)
            file_eri.create_dataset(fobj_a.dname[2], data=eri_ab)

            # sab = self.C_a @ self.S @ self.C_b
            _ = fobj_a.get_nsocc(self.S, self.C_a, self.Nocc[0], ncore=self.ncore)

            fobj_a.cons_h1(self.hcore)
            eri_a = ao2mo.restore(8, eri_a, fobj_a.nao)
            fobj_a.cons_fock(self.hf_veff[0], self.S, self.hf_dm[0] * 2.0, eri_=eri_a)

            fobj_a.hf_veff = self.hf_veff[0]
            fobj_a.heff = zeros_like(fobj_a.h1)
            fobj_a.scf(fs=True, eri=eri_a)
            fobj_a.dm0 = (
                fobj_a._mo_coeffs[:, : fobj_a.nsocc]
                @ fobj_a._mo_coeffs[:, : fobj_a.nsocc].conj().T
            )

            if compute_hf:
                eh1_a, ecoul_a, ef_a = fobj_a.update_ebe_hf(
                    return_e=True, unrestricted=True, spin_ind=0
                )
                unused(ef_a)
                EH1 += eh1_a
                ECOUL += ecoul_a
                E_hf += fobj_a.ebe_hf

            _ = fobj_b.get_nsocc(self.S, self.C_b, self.Nocc[1], ncore=self.ncore)

            fobj_b.cons_h1(self.hcore)
            eri_b = ao2mo.restore(8, eri_b, fobj_b.nao)
            fobj_b.cons_fock(self.hf_veff[1], self.S, self.hf_dm[1] * 2.0, eri_=eri_b)
            fobj_b.hf_veff = self.hf_veff[1]
            fobj_b.heff = zeros_like(fobj_b.h1)
            fobj_b.scf(fs=True, eri=eri_b)

            fobj_b.dm0 = (
                fobj_b._mo_coeffs[:, : fobj_b.nsocc]
                @ fobj_b._mo_coeffs[:, : fobj_b.nsocc].conj().T
            )

            if compute_hf:
                eh1_b, ecoul_b, ef_b = fobj_b.update_ebe_hf(
                    return_e=True, unrestricted=True, spin_ind=1
                )
                unused(ef_b)
                EH1 += eh1_b
                ECOUL += ecoul_b
                E_hf += fobj_b.ebe_hf
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

            self.ebe_hf = E_hf + self.enuc + self.E_core - self.ek
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

    def oneshot(self, solver="UCCSD", nproc=1, ompnum=4):
        if nproc == 1:
            E, E_comp = be_func_u(
                None,
                zip(self.Fobjs_a, self.Fobjs_b),
                solver,
                self.enuc,
                hf_veff=self.hf_veff,
                eeval=True,
                ereturn=True,
                relax_density=False,
                frozen=self.frozen_core,
            )
        else:
            E, E_comp = be_func_parallel_u(
                pot=None,
                Fobjs=zip(self.Fobjs_a, self.Fobjs_b),
                solver=solver,
                enuc=self.enuc,
                hf_veff=self.hf_veff,
                nproc=nproc,
                ompnum=ompnum,
                relax_density=False,
                frozen=self.frozen_core,
            )
        unused(E_comp)

        print("-----------------------------------------------------", flush=True)
        print("             One Shot BE ", flush=True)
        print("             Solver : ", solver, flush=True)
        print("-----------------------------------------------------", flush=True)
        print(flush=True)

        self.ebe_tot = E + self.uhf_full_e
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


def initialize_pot(n_frag, edge_idx):
    pot_ = []

    if not len(edge_idx) == 0:
        for I in range(n_frag):
            for i in edge_idx[I]:
                for j in range(len(i)):
                    for k in range(len(i)):
                        if j > k:
                            continue
                        pot_.append(0.0)

    pot_.append(0.0)  # alpha
    pot_.append(0.0)  # beta
    return pot_
