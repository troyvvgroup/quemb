import os
import time
from pyscf.dft import numint
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
from pyscf.tools import cubegen
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
import pickle
from quemb.molbe.mf_interfaces.main import load_scf, dump_scf
from quemb.molbe.chemfrag import ChemGenArgs

def get_e_corr_displaced(ref_be_obj, displaced_coords, embedding_type, gradient_orb_space, atom_idx):
    displaced_mol = ref_be_obj["mol_ref"].copy()
    displaced_mol.set_geom_(displaced_coords, unit="Bohr")
    displaced_mol.incore_anyway = True
    
    displaced_mf = get_mf(displaced_mol, auxbasis=ref_be_obj["auxbasis"])
    displaced_e_hf = displaced_mf.e_tot

    if embedding_type=="force":
        S_cross = gto.intor_cross("int1e_ovlp", displaced_mol, ref_be_obj["mol_ref"])
        frag_idx = ref_be_obj["frag_per_atom"][atom_idx]
        be = BE(
            displaced_mf,
            ref_be_obj["fobj"],
            int_transform=ref_be_obj["int_transform"],
            auxbasis=ref_be_obj["auxbasis"],
            lo_method=ref_be_obj["lo_method"],
            eq_fobjs=ref_be_obj["Fobjs"],
            S_cross=S_cross,
            gradient_orb_space=gradient_orb_space,
            initialize_fragment_idx = [frag_idx], 
        )
        fobj = be.Fobjs[frag_idx]
        eri = get_eri(fobj.dname, fobj.nao, eri_file=fobj.eri_file)
        fobj._mf = get_scfObj(fobj.fock, eri, fobj.nsocc, dm0=fobj.dm0.copy())
    
        mc = cc.CCSD(fobj._mf)
        mc.verbose = 0
        mc.incore_complete = True
    
        eri_embmo = mc.ao2mo()
        eri_embmo.mo_energy = fobj._mf.mo_energy
        eri_embmo.fock = np.diag(fobj._mf.mo_energy)
    
        mc.kernel(eris=eri_embmo)
        print(f"Did CCSD converge? {mc.converged}", flush=True)
        e_corr = mc.e_tot - fobj._mf.e_tot
    
    elif embedding_type in ("energy_noncumulant", "energy_cumulant"):
        be = BE(
            displaced_mf,
            ref_be_obj["fobj"],
            int_transform=ref_be_obj["int_transform"],
            auxbasis=ref_be_obj["auxbasis"],
            lo_method=ref_be_obj["lo_method"],
        )
    
        if embedding_type=="energy_noncumulant":
            be.oneshot(solver="CCSD",use_cumulant=False)
        elif embedding_type=="energy_cumulant":
            be.oneshot(solver="CCSD", use_cumulant=True)
        e_corr = be.rets0

    return e_corr, displaced_e_hf


def read_xyz(fname):
    atoms = []
    labels = []

    with open(fname) as f:
        lines = f.readlines()[2:]

    for line in lines:
        parts = line.split()
        if len(parts) >= 4:
            label = parts[0]
            coords = tuple(map(float, parts[1:4]))
            atoms.append((label, coords))
            labels.append(label)

    return atoms, labels

def get_mf(mol, auxbasis=None):
    if auxbasis:
        mf = scf.RHF(mol).density_fit(auxbasis=auxbasis)
    else:
        mf = scf.RHF(mol)
    mf.kernel()
    return mf

def get_ref_be_obj(mol_ref, mf_ref, filename, n_BE, auxbasis=None):
    print("Getting ref BE obj now, note for future that frag_type and int_transform are hardcoded here.", flush=True)
    fobj = fragmentate(mol=mol_ref, frag_type="chemgen", n_BE=n_BE)
    if auxbasis:
        ref_be_obj = BE(mf_ref, fobj, gradient_orb_space="Unmodified", int_transform="int-direct-DF", auxbasis=auxbasis)
    else:
        ref_be_obj = BE(mf_ref, fobj, gradient_orb_space="Unmodified")
    
    fragmented = Fragmented.from_mole(mol_ref, n_BE=n_BE)
    frag_per_atom = fragmented.get_frag_per_atom()

    state = {
        "mol_ref": mol_ref,
        "fobj": fobj,
        "frag_per_atom": frag_per_atom,
        "int_transform": ref_be_obj.int_transform,
        "auxbasis": ref_be_obj.auxbasis,
        "lo_method": ref_be_obj.lo_method,
        "Fobjs": ref_be_obj.Fobjs,
    }

    with open(f"ref_be_obj_{filename}.pkl", "wb") as f:
        pickle.dump(state, f)

    return ref_be_obj




def get_displaced_coords(ref_be_obj, delta=1e-4, natm=None, atom_idx=None, xyz=None, sign=None):
    print("Preparing displaced objects now.", flush=True)

    ref_coords = ref_be_obj["mol_ref"].atom_coords().copy()

    def make_displaced_coords(atom_idx, xyz, sign):
        disp = np.zeros((natm, 3))
        disp[atom_idx, xyz] = sign * delta
        return ref_coords + disp

    return make_displaced_coords(atom_idx, xyz, sign)

                    
def get_reference(mf_ref, filename, auxbasis=None):
    print("Getting reference gradients now, note for future that CCSD is hardcoded here.", flush=True)
    grad_hf_ref = mf_ref.nuc_grad_method().kernel()
    np.savetxt(f"grad_hf_ref_{filename}.txt", grad_hf_ref, fmt="%.12e")

    if auxbasis:
        mycc = cc.CCSD(mf_ref).density_fit(auxbasis=auxbasis)
    else:
        mycc = cc.CCSD(mf_ref)
    
    mycc.kernel()
    grad_total_ref = mycc.nuc_grad_method().kernel()
    np.savetxt(f"grad_total_ref_{filename}.txt", grad_total_ref, fmt="%.12e")



#def get_reference(mf, solver="CCSD"):
#    grad_hf = mf.Gradients()
#    grad_hf.verbose = 0
#    grad_hf.kernel()

#    grad_hf_ref = grad_hf.de

#    if solver == "CCSD":
#        print("Computing CCSD reference gradient")
#        mycc = cc.CCSD(mf)
#        mycc.kernel()
#        grad_ccsd = mycc.nuc_grad_method()
#        grad_corr_ref = grad_ccsd.kernel()

#    if solver == "FCI":
#        print("Computing FCI reference gradient")
#        mc = mcscf.CASCI(mf, ncas=mf.mo_coeff.shape[1], nelecas=sum(mf.mol.nelec))
#        mc.kernel()
#        grad_fci = mc.nuc_grad_method()
#        grad_corr_ref = grad_fci.kernel()

#    if solver == "MP2":
#        print("Computing MP2 reference gradient")
#        mymp2 = mp.MP2(mf)
#        mymp2.kernel()
#        grad_mp2 = mymp2.nuc_grad_method()
#        grad_corr_ref = grad_mp2.kernel()
#    return grad_hf_ref, grad_corr_ref


class BEGrad:
    """
    Gradient routine for Bootstrap Embedding (finite-difference force embedding)
    """

    def __init__(self, ref_be_obj: BE, delta=1e-4, gradient_orb_space="block-diagonal", h_treatment="treat_H_diff"):
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
        return [(atom_idx, xyz) for atom_idx in range(natm) for xyz in range(3)]

    #def _generate_displacements(self):
    #    natm = self.mol.natm
    #    disps = []

    #    for atom_idx in range(natm):
    #        for xyz in range(3):
    #            for sign in (+1, -1):
    #                vec = np.zeros((natm, 3))
    #                vec[atom_idx, xyz] = sign * self.delta
    #                disps.append((atom_idx, xyz, sign, vec))

    #    return disps

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
        atom_idx, xyz = task
        out = {}
    
        for sign in (+1, -1):
            disp = np.zeros((self.mol.natm, 3))
            disp[atom_idx, xyz] = sign * self.delta
    
            coords = self.coords0 + disp
            displaced_mol = self.ref_be_obj.mf.mol.copy()
            displaced_mol.set_geom_(coords, unit="Bohr")
            displaced_mol.incore_anyway = True
    
            start = time.time()
            displaced_mf = scf.RHF(displaced_mol)
            displaced_mf.kernel() # using self.ref_be_obj.hf_dm changes the result
            print(f"time on displaced mf {sign:+d} {time.time()-start:.12e}")
    
            displaced_e_hf = displaced_mf.e_tot
            S_cross = gto.intor_cross("int1e_ovlp", displaced_mol, self.mol)
    
            rand = "".join(random.choices(string.ascii_lowercase, k=8))
            with WorkDir.from_environment(prefix=rand + "_") as workdir:
                frag_idx = self.frag_per_atom[atom_idx]
                start = time.time()
                be = BE(
                    displaced_mf,
                    self.ref_be_obj.fobj,
                    int_transform=self.ref_be_obj.int_transform,
                    auxbasis=self.ref_be_obj.auxbasis,
                    lo_method=self.ref_be_obj.lo_method,
                    eq_fobjs=self.ref_be_obj.Fobjs,
                    S_cross=S_cross,
                    gradient_orb_space=self.gradient_orb_space,
                    scratch_dir=workdir,
                    initialize_fragment_idx=[frag_idx]
                )
                print(f"time on perturbed be {sign:+d} {time.time()-start:.12e}", flush=True)
    
                fobj = be.Fobjs[frag_idx]
    
                start = time.time()
                e_corr = self._compute_frag(fobj, self.ref_be_obj.solver)
                print(f"time on perturbed ecorr {sign:+d} {time.time()-start:.12e}", flush=True)
    
                out[sign] = (displaced_e_hf, e_corr, be.Fobjs[frag_idx]._mf.e_tot)
    
        return atom_idx, xyz, out
    
    #def _worker(self, task):
    #    atom_idx, xyz, sign, disp = task

    #    coords = self.coords0 + disp
    #    displaced_mol = self.ref_be_obj.mf.mol.copy()
    #    displaced_mol.set_geom_(coords, unit="Bohr")
    #    displaced_mol.incore_anyway = True

    #    start = time.time()
    #    displaced_mf = scf.RHF(displaced_mol)
    #    displaced_mf.kernel(self.ref_be_obj.hf_dm) # using self.ref_be_obj.hf_dm as the initial guess changes the result
    #    print(f"time on displaced mf {time.time()-start:.12e}")
    #    displaced_e_hf = displaced_mf.e_tot

    #    S_cross = gto.intor_cross("int1e_ovlp", displaced_mol, self.mol)

    #    prefix = f"a{atom_idx}_x{xyz}_s{sign}_pid{os.getpid()}_"
    #    with WorkDir.from_environment(prefix=prefix) as workdir:
    #        frag_idx = self.frag_per_atom[atom_idx]
    #        start = time.time()
    #        be = BE(  
    #            displaced_mf,
    #            self.ref_be_obj.fobj,
    #            int_transform=self.ref_be_obj.int_transform,
    #            auxbasis = self.ref_be_obj.auxbasis,
    #            lo_method=self.ref_be_obj.lo_method,
    #            eq_fobjs=self.ref_be_obj.Fobjs,
    #            S_cross=S_cross,
    #            gradient_orb_space=self.gradient_orb_space,
    #            scratch_dir=workdir,
    #            initialize_fragment_idx = [frag_idx]
    #        )
    #        print(f"time on perturbed be {time.time()-start:.12e}", flush=True)

    #        fobj = be.Fobjs[frag_idx]

    #        start = time.time()
    #        e_corr = self._compute_frag(fobj, self.ref_be_obj.solver)
    #        print(f"time on perturbed ecorr {time.time()-start:.12e}", flush=True)

    #        return atom_idx, xyz, sign, displaced_e_hf, e_corr, be.Fobjs[frag_idx]._mf.e_tot

    # =========================
    # Gradient computation
    # =========================
    def compute_grad(self, nproc=16):
        displacements = self._generate_displacements()

        results = list(map(self._worker, displacements))
        # with Pool(nproc) as p:
        #     results = p.map(self._worker, displacements)

        natm = self.mol.natm
        grad_corr = np.zeros((natm, 3))
        grad_hf = np.zeros((natm, 3))

        fragment_Hamiltonian_HF_energies = []

        for atom_idx, xyz, vals in results:
            if +1 not in vals or -1 not in vals:
                raise RuntimeError(f"Missing displacement for {atom_idx},{xyz}")

            e_plus_hf, e_plus_corr, hf_frag_plus = vals[+1]
            e_minus_hf, e_minus_corr, hf_frag_minus = vals[-1]

            fragment_Hamiltonian_HF_energies.extend([hf_frag_plus, hf_frag_minus])

            grad_corr[atom_idx, xyz] = (
                (e_plus_hf + e_plus_corr) - (e_minus_hf + e_minus_corr)
            ) / (2 * self.delta)

            grad_hf[atom_idx, xyz] = (e_plus_hf - e_minus_hf) / (2 * self.delta)
    
        #results = list(map(self._worker, displacements))
        #with Pool(nproc) as p:
        #    results = p.map(self._worker, displacements)

        #natm = self.mol.natm
        #grad_corr = np.zeros((natm, 3))
        #grad_hf = np.zeros((natm, 3))

        # collect
        #results_dict = {}
        #fragment_Hamiltonian_HF_energies = []
        #for atom_idx, xyz, sign, e_hf, e_corr, fragment_Hamiltonian_HF_energy in results:
        #    results_dict.setdefault((atom_idx, xyz), {})[sign] = (e_hf, e_corr)
        #    fragment_Hamiltonian_HF_energies.append(fragment_Hamiltonian_HF_energy)

        # finite difference
        #for (atom_idx, xyz), vals in results_dict.items():
        #    if +1 not in vals or -1 not in vals:
        #        raise RuntimeError(f"Missing displacement for {atom_idx},{xyz}")

        #    e_plus_hf, e_plus_corr = vals[+1]
        #    e_minus_hf, e_minus_corr = vals[-1]

        #    grad_corr[atom_idx, xyz] = (
        #        (e_plus_hf + e_plus_corr) - (e_minus_hf + e_minus_corr)
        #    ) / (2 * self.delta)

        #    grad_hf[atom_idx, xyz] = (e_plus_hf - e_minus_hf) / (2 * self.delta)

        self.grad_corr = grad_corr
        self.grad_hf = grad_hf

        return grad_corr, grad_hf

    def set_reference(self, grad_hf_ref, grad_corr_ref):
        self.grad_hf_ref = grad_hf_ref
        self.grad_corr_ref = grad_corr_ref

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
