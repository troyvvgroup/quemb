import numpy as np
from pyscf import gto
from pyscf.cc import CCSD
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

    def __init__(self, ref_be_obj: BE, delta=1e-4):
        self.ref_be_obj = ref_be_obj
        self.delta = delta
        self.grad_hf_ref = None
        self.grad_ccsd_ref = None

        self.mol = ref_be_obj.mf.mol
        self.coords0 = self.mol.atom_coords().copy()

        # Fragment info
        fragmented = Fragmented.from_mole(self.mol, n_BE=2)
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
    def _compute_frag(fobj, solver="CCSD"):
        eri = get_eri(fobj.dname, fobj.nao, eri_file=fobj.eri_file)
        mf = get_scfObj(fobj.fock, eri, fobj.nsocc, dm0=fobj.dm0.copy())

        if solver == "CCSD":
            mc = CCSD(mf)
            mc.verbose = 0
            mc.incore_complete = True

            eri_embmo = mc.ao2mo()
            eri_embmo.mo_energy = mf.mo_energy
            eri_embmo.fock = np.diag(mf.mo_energy)

            mc.kernel(eris=eri_embmo)
            return mc.e_tot - mf.e_tot

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

        S_cross = gto.intor_cross("int1e_ovlp", displaced_mol, self.mol)

        rand = "".join(random.choices(string.ascii_lowercase, k=8))
        with WorkDir.from_environment(prefix=rand + "_") as workdir:
            be = BE( # note to self: try to build displaced be obj in the same way minsik does
                displaced_mf,
                self.ref_be_obj.fobj,
                eq_fobjs=self.ref_be_obj.Fobjs, 
                S_cross=S_cross,
                gradient_orb_space="lo-basis",
                scratch_dir=workdir,
            )

            frag_idx = self.frag_per_atom[atom_idx]
            fobj = be.Fobjs[frag_idx]

            e_corr = self._compute_frag(fobj)

            return atom_idx, xyz, sign, be.ebe_hf, e_corr

    # =========================
    # Gradient computation
    # =========================
    def compute_grad(self, nproc=16):
        displacements = self._generate_displacements()

        results = list(map(self._worker, displacements))
        # with Pool(nproc) as p:
        #    results = p.map(self._worker, displacements)

        natm = self.mol.natm
        grad_ccsd = np.zeros((natm, 3))
        grad_hf = np.zeros((natm, 3))

        # collect
        results_dict = {}
        for atom_idx, xyz, sign, e_hf, e_corr in results:
            results_dict.setdefault((atom_idx, xyz), {})[sign] = (e_hf, e_corr)

        # finite difference
        for (atom_idx, xyz), vals in results_dict.items():
            if +1 not in vals or -1 not in vals:
                raise RuntimeError(f"Missing displacement for {atom_idx},{xyz}")

            e_plus_hf, e_plus_corr = vals[+1]
            e_minus_hf, e_minus_corr = vals[-1]

            grad_ccsd[atom_idx, xyz] = (
                (e_plus_hf + e_plus_corr) - (e_minus_hf + e_minus_corr)
            ) / (2 * self.delta)

            grad_hf[atom_idx, xyz] = (e_plus_hf - e_minus_hf) / (2 * self.delta)

        self.grad_ccsd = grad_ccsd
        self.grad_hf = grad_hf

        return grad_ccsd, grad_hf

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
            self.grad_ccsd_ref = grad_ccsd.kernel()

    def compute_rmse(self, which="both"):
        """
        Compute RMSE between computed and reference gradients.

        Parameters
        ----------
        which : str
            "hf", "ccsd", or "both"
        """

        if self.grad_hf is None or self.grad_ccsd is None:
            raise RuntimeError("Run compute_grad() first")

        results = {}

        if which in ("hf", "both"):
            if self.grad_hf_ref is None:
                raise RuntimeError("HF reference not set")
            rms_hf = np.sqrt(np.mean((self.grad_hf_ref - self.grad_hf) ** 2))
            print(f"RMS gradient difference (HF): {rms_hf:.12e}")
            results["hf"] = rms_hf

        if which in ("ccsd", "both"):
            if self.grad_ccsd_ref is None:
                raise RuntimeError("CCSD reference not set")
            rms_ccsd = np.sqrt(np.mean((self.grad_ccsd_ref - self.grad_ccsd) ** 2))
            print(f"RMS gradient difference (CCSD): {rms_ccsd:.12e}")
            results["ccsd"] = rms_ccsd

        return results


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


xyz_file = sys.argv[1]
atoms, labels = read_xyz(xyz_file)
mol_ref = gto.M(atom=atoms, basis="sto-3g", charge=0, unit="Angstrom")

mf = scf.RHF(mol_ref)
mf.kernel()

fobj = fragmentate(mol=mol_ref, frag_type="chemgen", n_BE=2)
mybe = BE(mf, fobj)

be_grad = BEGrad(mybe, delta=1e-4)
be_grad.set_reference(mf, solver="CCSD")

gradient_ccsd, gradient_hf = be_grad.compute_grad(nproc=16)
rmse = be_grad.compute_rmse(which="both")
