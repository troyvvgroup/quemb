from pyscf import gto, scf, cc, fci, mcscf
import numpy as np
from quemb.molbe import BE, fragmentate
from quemb.molbe.chemfrag import Fragmented
from quemb.molbe.chemfrag import ChemGenArgs
from pyscf import grad

def printmat(m, fmt="%12.12f"):
    """Prints the matrix m using the format code fmt."""
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            print((" " + fmt) % (m[i, j]), end="")
        print("")

mol = gto.M(
    atom="""
H 1 0.000000000 0.000000000
H 2 0.000000000 0.000000000
H 3 0.000000000 0.000000000
H 4 0.000000000 0.000000000
H 5 0.000000000 0.000000000
H 6 0.000000000 0.000000000
H 7 0.000000000 0.000000000
H 8 0.000000000 0.000000000
H 9 0.000000000 0.000000000
H 10 0.000000000 0.000000000
H 11 0.000000000 0.000000000
H 12 0.000000000 0.000000000
""",
basis="sto-3g",
    charge=0,
    unit="Angstrom",
)
coords0 = mol.atom_coords()

mf = scf.RHF(mol)
mf.kernel()

# 3. Compute HF gradient
hf_ref_grad_obj = grad.RHF(mf)
gradient_hf_ref = hf_ref_grad_obj.kernel()
print("HF Gradient:", gradient_hf_ref)

# 4. Full CASCI (equivalent to FCI since all orbitals are active)
norb = mf.mo_coeff.shape[1]   # total orbitals
nelec = mol.nelectron         # total electrons
mycas = mcscf.CASCI(mf, norb, nelec)
mycas.kernel()

# 5. CASCI gradient (analytic)
casci_grad_obj = grad.casci.Gradients(mycas)
gradient_fci_ref = casci_grad_obj.kernel()
print("CASCI (FCI) Gradient:\n", gradient_fci_ref)

# Do equilibrium BE calculation
args = ChemGenArgs(treat_H_different=False)
fobj = fragmentate(mol=mol, frag_type="chemgen", n_BE=1, additional_args=args)

mybe = BE(mf, fobj, thr_bath=1e-10)
print(f"doing the unperturbed calculation")
mybe.oneshot(solver="FCI")

#########################################################################
fragmented = Fragmented.from_mole(mol, n_BE=1, treat_H_different=False)
cpf = fragmented.frag_structure.centers_per_frag
apf = fragmented.frag_structure.atoms_per_frag

cpf_h = []
fidx = 0
for frag in apf:
    cpf_h_f = []
    for atm in frag:
        if atm in cpf[fidx]:  # if desired center
            cpf_h_f.append(atm)
        elif any(atm in f for f in cpf):
            break
        else:
            cpf_h_f.append(atm)
    cpf_h.append(cpf_h_f)
    fidx += 1

# build list of fragment indices per atom (where atom is a center)
frag_per_atom = []
for atom_idx in range(mol.natm):
    frag_per_atom.append([cpf_h.index(f) for f in cpf_h if atom_idx in f][0])
print(f"frag per atom is {frag_per_atom}")


delta = 1e-4
natoms = mol.natm

gradient_fci = np.zeros((natoms, 3))
gradient_hf = np.zeros((natoms, 3))
for atom_idx in range(natoms):
    print(f"working on {atom_idx} out of {natoms}")
    frag_idx = frag_per_atom[atom_idx]
    
    for xyz in range(3):    
        print("doing plus perturbation")
        coords_plus = coords0.copy()
        coords_plus[atom_idx, xyz] += delta
        
        mol_plus = mol.copy()
        mol_plus.set_geom_(coords_plus, unit="Bohr")

        mf_plus = scf.RHF(mol_plus)
        mf_plus.kernel()

        mycc_plus = cc.CCSD(mf_plus)
        mycc_plus.kernel()

        mybe_plus = BE(mf_plus, fobj, thr_bath=1e-10)
        mybe_plus.oneshot(solver="FCI")

        print("doing minus perturbation")
        coords_minus = coords0.copy()
        coords_minus[atom_idx, xyz] -= delta

        mol_minus = mol.copy()
        mol_minus.set_geom_(coords_minus, unit="Bohr")

        mf_minus = scf.RHF(mol_minus)
        mf_minus.kernel()

        mycc_minus = cc.CCSD(mf_minus)
        mycc_minus.kernel()

        mybe_minus = BE(mf_minus, fobj, thr_bath=1e-10)
        mybe_minus.oneshot(solver="FCI")

        e_plus_fci = mybe_plus.Fobjs[frag_idx].E_env + mybe_plus.Fobjs[frag_idx]._mf.e_tot + mybe_plus.enuc + mybe_plus.Fobjs[frag_idx].ecorr + mybe_plus.E_core
        e_minus_fci = mybe_minus.Fobjs[frag_idx].E_env + mybe_minus.Fobjs[frag_idx]._mf.e_tot + mybe_minus.enuc + mybe_minus.Fobjs[frag_idx].ecorr + mybe_minus.E_core 
        
        e_plus_hf = mybe_plus.Fobjs[frag_idx].E_env + mybe_plus.Fobjs[frag_idx]._mf.e_tot + mybe_plus.enuc + mybe_plus.E_core
        e_minus_hf = mybe_minus.Fobjs[frag_idx].E_env + mybe_minus.Fobjs[frag_idx]._mf.e_tot + mybe_minus.enuc + mybe_minus.E_core
        gradient_fci[atom_idx,xyz] = ( e_plus_fci - e_minus_fci ) / (2*delta)
        gradient_hf[atom_idx, xyz] = ( e_plus_hf - e_minus_hf ) / (2*delta)

    #schmidt_orbitals = mybe_plus.Fobjs[frag_idx].n_f + mybe_plus.Fobjs[frag_idx].n_b
    #orbitals[atom_idx, 0] = schmidt_orbitals
    #env_occ_orbitals = mybe_plus.Nocc - mybe_plus.Fobjs[frag_idx].nsocc
    #orbitals[atom_idx, 1] = env_occ_orbitals
    #env_virt_orbitals = mybe_plus.Fobjs[frag_idx].TAenv_lo_eo.shape[1]-mybe_plus.Nocc + mybe_plus.Fobjs[frag_idx].nsocc
    #orbitals[atom_idx, 2] = env_virt_orbitals



diff_fci = np.abs(gradient_fci_ref - gradient_fci)
rms_diff_fci = np.sqrt(np.mean(diff_fci**2))
print("RMS gradient difference (FCI):", rms_diff_fci)

diff_hf = np.abs(gradient_hf_ref - gradient_hf)
rms_diff_hf = np.sqrt(np.mean(diff_hf**2))
print("RMS gradient difference (HF):", rms_diff_hf)

print("successful termination")
