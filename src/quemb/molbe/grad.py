import matplotlib.pyplot as plt
from pyscf import mp
import copy
import numpy as np
from pyscf import cc, grad, gto, scf, mcscf, fci, ao2mo
from quemb.molbe.helper import get_eri, get_scfObj
from quemb.molbe.mbe import BE
from quemb.shared.manage_scratch import WorkDir
from multiprocessing import Pool
from pyscf import cc, scf
from quemb.molbe import fragmentate
from quemb.molbe.chemfrag import Fragmented
import sys
import random
import string


class BEGrad:
    """
    Gradient routine for Bootstrap Embedding (finite-difference force embedding)
    """

    def __init__(self, ref_be_obj: BE, delta=1e-4, gradient_orb_space="lo-basis", h_treatment="treat_H_diff"):
        self.ref_be_obj = ref_be_obj
        self.delta = delta
        self.grad_hf_ref = None
        self.grad_corr_ref = None
        self.gradient_orb_space = gradient_orb_space

        self.mol = ref_be_obj.mf.mol
        self.coords0 = self.mol.atom_coords().copy()

        # Fragment info
        fragmented = Fragmented.from_mole(self.mol, n_BE=ref_be_obj.fobj.n_BE, h_treatment=h_treatment)
        self.frag_per_atom = fragmented.get_frag_per_atom()

    # =========================
    # Displacements
    # =========================
    def _generate_displacements(self):
        natm = self.mol.natm
        disps = []

        for atom_idx in range(natm):
            for xyz in range(3):
                for sign in (+1, -1):
                    vec = np.zeros((natm, 3))
                    vec[atom_idx, xyz] = sign * self.delta
                    disps.append((atom_idx, xyz, sign, vec))

        return disps

    # =========================
    # Fragment solver
    # =========================
    @staticmethod
    def _compute_frag(fobj, solver):
        eri = get_eri(fobj.dname, fobj.nao, eri_file=fobj.eri_file)
        fobj._mf = get_scfObj(fobj.fock, eri, fobj.nsocc, dm0=fobj.dm0.copy())

        if solver == "CCSD":
            mc = cc.CCSD(fobj._mf)
            mc.verbose = 0
            mc.incore_complete = True

            eri_embmo = mc.ao2mo()
            eri_embmo.mo_energy = fobj._mf.mo_energy
            eri_embmo.fock = np.diag(fobj._mf.mo_energy)

            mc.kernel(eris=eri_embmo)
            return mc.e_tot - fobj._mf.e_tot  # same as mc.e_corr

        if solver == "FCI":
            mc = fci.FCI(fobj._mf, fobj._mf.mo_coeff)
            mc.verbose = 0
            e_fci, _ = mc.kernel()
            return e_fci - fobj._mf.e_tot  # corr = fci - hf
        
        if solver == "MP2":
            print(f"Computing MP2 fragment correlation energy")
            mymp2 = mp.MP2(fobj._mf)
            mymp2.verbose = 0
            e_mp2, _ = mymp2.kernel()
            return e_mp2

        if solver == "CCSD(T)":
            print(f"Computing CCSD(T) fragment correlation energy")
            mc = cc.CCSD(fobj._mf)
            mc.verbose = 0
            mc.incore_complete = True

            eri_embmo = mc.ao2mo()
            eri_embmo.mo_energy = fobj._mf.mo_energy
            eri_embmo.fock = np.diag(fobj._mf.mo_energy)

            mc.kernel(eris=eri_embmo)
            et = mc.ccsd_t(eris=eri_embmo)
            return mc.e_tot - fobj._mf.e_tot + et


        raise NotImplementedError

    # =========================
    # Worker
    # =========================
    def _worker(self, task):
        atom_idx, xyz, sign, disp = task

        coords = self.coords0 + disp
        displaced_mol = self.ref_be_obj.mf.mol.copy()
        displaced_mol.set_geom_(coords, unit="Bohr")
        displaced_mol.build()

        displaced_mf = scf.RHF(displaced_mol)
        displaced_mf.kernel() # using self.ref_be_obj.hf_dm as the initial guess changes the result
        displaced_e_hf = displaced_mf.e_tot

        S_cross = gto.intor_cross("int1e_ovlp", displaced_mol, self.mol)

        rand = "".join(random.choices(string.ascii_lowercase, k=8))
        with WorkDir.from_environment(prefix=rand + "_") as workdir:
            frag_idx = self.frag_per_atom[atom_idx]
            be = BE( 
                displaced_mf,
                self.ref_be_obj.fobj,
                lo_method=self.ref_be_obj.lo_method,
                eq_fobjs=self.ref_be_obj.Fobjs,
                S_cross=S_cross,
                gradient_orb_space=self.gradient_orb_space,
                scratch_dir=workdir,
                initialize_fragment_idx = [frag_idx]
            )

            fobj = be.Fobjs[frag_idx]
            e_corr = self._compute_frag(fobj, self.ref_be_obj.solver)

            return atom_idx, xyz, sign, displaced_e_hf, e_corr, be.Fobjs[frag_idx]._mf.e_tot

    # =========================
    # Gradient computation
    # =========================
    def compute_grad(self, nproc=16):
        displacements = self._generate_displacements()

        #results = list(map(self._worker, displacements))
        with Pool(nproc) as p:
            results = p.map(self._worker, displacements)

        natm = self.mol.natm
        grad_corr = np.zeros((natm, 3))
        grad_hf = np.zeros((natm, 3))

        # collect
        results_dict = {}
        fragment_Hamiltonian_HF_energies = []
        for atom_idx, xyz, sign, e_hf, e_corr, fragment_Hamiltonian_HF_energy in results:
            results_dict.setdefault((atom_idx, xyz), {})[sign] = (e_hf, e_corr)
            fragment_Hamiltonian_HF_energies.append(fragment_Hamiltonian_HF_energy)

        # finite difference
        for (atom_idx, xyz), vals in results_dict.items():
            if +1 not in vals or -1 not in vals:
                raise RuntimeError(f"Missing displacement for {atom_idx},{xyz}")

            e_plus_hf, e_plus_corr = vals[+1]
            e_minus_hf, e_minus_corr = vals[-1]

            grad_corr[atom_idx, xyz] = (
                (e_plus_hf + e_plus_corr) - (e_minus_hf + e_minus_corr)
            ) / (2 * self.delta)

            grad_hf[atom_idx, xyz] = (e_plus_hf - e_minus_hf) / (2 * self.delta)

        self.grad_corr = grad_corr
        self.grad_hf = grad_hf

        np.savetxt('fragment_Hamiltonian_HF_energies.txt', fragment_Hamiltonian_HF_energies, fmt='%.12e')
        return grad_corr, grad_hf

    def set_reference(self, mf, solver="CCSD"):
        grad_hf = mf.Gradients()
        grad_hf.verbose = 0
        grad_hf.kernel()

        self.grad_hf_ref = grad_hf.de

        if solver == "CCSD":
            print("Computing CCSD reference gradient")
            mycc = cc.CCSD(mf)
            mycc.kernel()
            grad_ccsd = mycc.nuc_grad_method()
            self.grad_corr_ref = grad_ccsd.kernel()

        if solver == "FCI":
            print("Computing FCI reference gradient")
            mc = mcscf.CASCI(mf, ncas=mf.mo_coeff.shape[1], nelecas=sum(mf.mol.nelec))
            mc.kernel()
            grad_fci = mc.nuc_grad_method()
            self.grad_corr_ref = grad_fci.kernel()

        if solver == "MP2":
            print("Computing MP2 reference gradient")
            mymp2 = mp.MP2(mf)
            mymp2.kernel()
            grad_mp2 = mymp2.nuc_grad_method()
            self.grad_corr_ref = grad_mp2.kernel()

        if solver == "CCSD(T)":
            print("Computing CCSD(T) reference gradient by finite differences")

            mol0 = mf.mol
            coords0 = mol0.atom_coords()
            natm = mol0.natm
            delta = self.delta

            grad_ccsdt = np.zeros((natm, 3))

            for atom_idx in range(natm):
                for xyz in range(3):
                    e_plus = None
                    e_minus = None

                    for sign in (+1, -1):
                        disp = np.zeros((natm, 3))
                        disp[atom_idx, xyz] = sign * delta

                        mol_disp = mol0.copy()
                        mol_disp.set_geom_(coords0 + disp, unit="Bohr")
                        mol_disp.build()

                        mf_disp = scf.RHF(mol_disp)
                        mf_disp.verbose = 0
                        mf_disp.kernel()

                        mycc = cc.CCSD(mf_disp)
                        mycc.verbose = 0
                        mycc.kernel()
                        et = mycc.ccsd_t()

                        e_tot = mycc.e_tot + et

                        if sign == +1:
                            e_plus = e_tot
                        else:
                            e_minus = e_tot
                    grad_ccsdt[atom_idx, xyz] = (e_plus - e_minus) / (2 * delta)

            self.grad_corr_ref = grad_ccsdt
        
    def compute_rmse(self, which="both"):
        """
        Compute RMSE between computed and reference gradients.

        Parameters
        ----------
        which : str
            "hf", "corr", or "both"
        """

        if self.grad_hf is None or self.grad_corr is None:
            raise RuntimeError("Run compute_grad() first")

        results = {}

        if which in ("hf", "both"):
            if self.grad_hf_ref is None:
                raise RuntimeError("HF reference not set")
            rms_hf = np.sqrt(np.mean((self.grad_hf_ref - self.grad_hf) ** 2))
            print(f"RMS gradient difference (HF): {rms_hf:.12e}")
            results["hf"] = rms_hf

        if which in ("corr", "both"):
            if self.grad_corr_ref is None:
                raise RuntimeError("Correlated reference not set")
            rms_corr = np.sqrt(np.mean((self.grad_corr_ref - self.grad_corr) ** 2))
            print(f"RMS gradient difference (correlated): {rms_corr:.12e}")
            results["corr"] = rms_corr

        return results
