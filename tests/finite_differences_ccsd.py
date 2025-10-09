import numpy as np
from pyscf import cc, grad, gto, scf

from quemb.molbe import BE, fragmentate
from quemb.molbe.chemfrag import ChemGenArgs, Fragmented


def kabsch_rotation(P, Q):
    """Calculate the optimal rotation ``R`` from ``P`` unto ``Q``

    The rotation is optimal in the sense that the Frobenius-metric,  i.e. | R P - Q |_2, is minimized.
    The algorithm is described here http://en.wikipedia.org/wiki/Kabsch_algorithm"""

    # covariance matrix
    H = P.T @ Q

    U, S, Vt = np.linalg.svd(H)

    # determinant is +-1 for orthogonal matrices
    # d_val = 1. if np.linalg.det(U @ Vt) > 0 else -1.

    # D = np.eye(len(U))
    # D[-1, -1] = d_val # gaurentees final result is rotation

    return U @ Vt


def offdiag_fraction(A):
    off = A - np.diag(np.diag(A))
    return np.linalg.norm(off, "fro") / np.linalg.norm(A, "fro")


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

mf = scf.RHF(mol)
mf.kernel()

hf_ref_grad_obj = grad.RHF(mf)
gradient_hf_ref = hf_ref_grad_obj.kernel()

mycc = cc.CCSD(mf)
mycc.kernel()

grad_ccsd = mycc.nuc_grad_method()
gradient_ccsd_ref = grad_ccsd.kernel()


args = ChemGenArgs(treat_H_different=False)
fobj = fragmentate(mol=mol, frag_type="chemgen", n_BE=3, additional_args=args)

mybe = BE(mf, fobj, thr_bath=1e-10)
print("doing the unperturbed calculation")
mybe.oneshot(solver="CCSD")

#########################################################################
fragmented = Fragmented.from_mole(mol, n_BE=3, treat_H_different=False)
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


delta = 1e-2
natoms = mol.natm
coords = mol.atom_coords()  # Bohr

gradient_ccsd = np.zeros((natoms, 3))
gradient_hf = np.zeros((natoms, 3))
for atom_idx in range(natoms):
    print(f"working on {atom_idx} out of {natoms}")
    frag_idx = frag_per_atom[atom_idx]

    for xyz in range(3):
        print("doing plus perturbation")
        coords_plus = coords.copy()
        coords_plus[atom_idx, xyz] += delta

        mol_plus = mol.copy()
        mol_plus.set_geom_(coords_plus, unit="Bohr")

        mf_plus = scf.RHF(mol_plus)
        mf_plus.kernel()

        mycc_plus = cc.CCSD(mf_plus)
        mycc_plus.kernel()

        mybe_plus = BE(mf_plus, fobj, thr_bath=1e-10, eq_fobjs=mybe.Fobjs)
        mybe_plus.oneshot(solver="CCSD")

        print("doing minus perturbation")
        coords_minus = coords.copy()
        coords_minus[atom_idx, xyz] -= delta

        mol_minus = mol.copy()
        mol_minus.set_geom_(coords_minus, unit="Bohr")

        mf_minus = scf.RHF(mol_minus)
        mf_minus.kernel()

        mycc_minus = cc.CCSD(mf_minus)
        mycc_minus.kernel()

        mybe_minus = BE(mf_minus, fobj, thr_bath=1e-10, eq_fobjs=mybe.Fobjs)
        mybe_minus.oneshot(solver="CCSD")

        # Check
        # diff_plus = mybe_plus.Fobjs[frag_idx].TA - mybe.Fobjs[frag_idx].TA
        # diff_minus = mybe_minus.Fobjs[frag_idx].TA - mybe.Fobjs[frag_idx].TA

        # max_plus = np.max(np.abs(diff_plus))
        # max_minus = np.max(np.abs(diff_minus))
        # mean_plus = np.mean(np.abs(diff_plus))
        # mean_minus = np.mean(np.abs(diff_minus))
        # print(f"max_plus = {max_plus}, mean_plus = {mean_plus}")
        # print(f"the off-diagonal fraction for plus is {offdiag_fraction(mybe_plus.Fobjs[frag_idx].R_fragbath)}")
        # print(f"max_minus = {max_minus}, mean_minus = {mean_minus}")
        # print(f"the off-diagonal fraction for minus is {offdiag_fraction(mybe_minus.Fobjs[frag_idx].R_fragbath)}")

        e_plus_ccsd = (
            mybe_plus.Fobjs[frag_idx]._mf.e_tot
            + mybe_plus.enuc
            + mybe_plus.Fobjs[frag_idx].mycc.e_corr
            + mybe_plus.E_core
            + mybe_plus.Fobjs[frag_idx].E_env
        )
        e_minus_ccsd = (
            mybe_minus.Fobjs[frag_idx]._mf.e_tot
            + mybe_minus.enuc
            + mybe_minus.Fobjs[frag_idx].mycc.e_corr
            + mybe_minus.E_core
            + mybe_minus.Fobjs[frag_idx].E_env
        )
        e_plus_hf = (
            mybe_plus.Fobjs[frag_idx]._mf.e_tot
            + mybe_plus.enuc
            + mybe_plus.E_core
            + mybe_plus.Fobjs[frag_idx].E_env
        )
        e_minus_hf = (
            mybe_minus.Fobjs[frag_idx]._mf.e_tot
            + mybe_minus.enuc
            + mybe_minus.E_core
            + mybe_minus.Fobjs[frag_idx].E_env
        )
        gradient_ccsd[atom_idx, xyz] = (e_plus_ccsd - e_minus_ccsd) / (2 * delta)
        gradient_hf[atom_idx, xyz] = (e_plus_hf - e_minus_hf) / (2 * delta)


diff_ccsd = np.abs(gradient_ccsd_ref - gradient_ccsd)
rms_diff_ccsd = np.sqrt(np.mean(diff_ccsd**2))
print("RMS gradient difference (CCSD):", rms_diff_ccsd)

diff_hf = np.abs(gradient_hf_ref - gradient_hf)
rms_diff_hf = np.sqrt(np.mean(diff_hf**2))
print("RMS gradient difference (HF):", rms_diff_hf)

print("The atom-wise gradient component differences (x, y, z):")
for atom_idx in range(natoms):
    dx, dy, dz = diff_ccsd[atom_idx]
    print(f"{dx:.12f} {dy:.12f} {dz:.12f}")

print("successful termination")
