# Illustrates energy and nuclear gradient computations and scans using BE as
# the custom energy function fed through pyscf's finite difference driver
# Author(s): Beck Hanscam

import numpy as np
from pyscf import cc, gto, scf
from pyscf.tools import finite_diff

from quemb.molbe.chemfrag import ChemGenArgs
from quemb.molbe.scanner import BEArgs, Energy, energy_be

# build several different geometries
rng = np.random.default_rng()
mol0 = gto.M(
    atom="""
        H  0.04 -0.02  0.
        H  0.02  0.04  1.
        H -0.01 -0.00  2.
        H -0.03  0.01  3.
        H -0.00  0.04  4.
        H  0.04  0.02  5.
        """,
    basis="sto-3g",
    charge=0,
)
mol1 = mol0.copy()
mol2 = mol0.copy()
coords0 = mol0.atom_coords()
coords1 = coords0 + rng.uniform(-0.05, 0.05, size=coords0.shape)
coords2 = coords0 + rng.uniform(-0.10, 0.10, size=coords0.shape)
mol1.set_geom_(coords1, unit="Bohr")
mol2.set_geom_(coords2, unit="Bohr")


# =========================
#  Gradients
# =========================
print("\n\nGRADIENTS")
print("-" * 40)

#
# analytic gradient of mol0
#
print("\nanalytic CCSD gradient for mol0:")
mf = scf.RHF(mol0)
mf.conv_tol = 1e-12
mf.kernel()
mc = cc.CCSD(mf)
mc.kernel()
mc_grad = mc.nuc_grad_method()
ref_grad_corr = mc_grad.kernel()

#
# numerical gradients of mol0
# BEn-CCSD oneshot, n=1,2,3
#
print("\nnumerical BE-CCSD gradient for mol0:")
rms = []
for n_BE in [1, 2, 3]:
    be_args = BEArgs(
        n_BE=n_BE,
        solver="CCSD",
        use_cumulant=True,
        optimize=False,
        additional_args=ChemGenArgs(h_treatment="treat_H_like_heavy_atom"),
    )
    energy_method = Energy(mol0, energy_be, be_args=be_args)
    grad_method = finite_diff.Gradients(energy_method)
    grad_method.displacement = 1e-4  # Gradients class default is 1e-2 Bohr
    fd_grad_fromObj = grad_method.kernel()
    rms.append(np.sqrt(np.mean((ref_grad_corr - fd_grad_fromObj) ** 2)))


# =========================
#  Scanners
# =========================
print("\n\nSCANNERS")
print("-" * 40)

#
# energy point calculations using as_scanner()
# BE2-CCSD with chemicial potential optimization
#
print("\nUsing energy_method.as_scanner():")
be_args = BEArgs(
    n_BE=1,
    solver="CCSD",
    use_cumulant=True,
    optimize=True,
    only_chem=True,
    additional_args=ChemGenArgs(h_treatment="treat_H_like_heavy_atom"),
)
energy_method = Energy(mol0, energy_be, be_args=be_args)
energy_scanner = energy_method.as_scanner()
es_E0 = energy_scanner(mol0)
es_E1 = energy_scanner(mol1)
es_E2 = energy_scanner(mol2)

#
# gradient point calculations using as_scanner()
# BE3-CCSD oneshot
#
print("\nUsing grad_method.as_scanner():")
be_args = BEArgs(
    n_BE=3,
    solver="CCSD",
    use_cumulant=True,
    optimize=False,
    additional_args=ChemGenArgs(h_treatment="treat_H_like_heavy_atom"),
)
energy_method = Energy(mol0, energy_be, be_args=be_args)
grad_method = finite_diff.Gradients(energy_method)
grad_method.displacement = 1e-4  # Gradients class default is 1e-2 Bohr
grad_scanner = grad_method.as_scanner()
gs_E1, gs_grad1 = grad_scanner(mol1)
gs_E2, gs_grad2 = grad_scanner(mol2)


# =========================
#  Print results
# =========================

print("\n\n___NUMERICAL GRADIENTS___")
for n_BE in [1, 2, 3]:
    print(f"BE{n_BE}-CCSD RMSE from analytic benchmark for mol0: {rms[n_BE - 1]:.12e}")

print("\n___ENERGY SCANNER___")
print(f"BE2-CCSD (with chempot matching) energy of mol0: {es_E0:.8f}")
print(f"BE2-CCSD (with chempot matching) energy of mol1: {es_E1:.8f}")
print(f"BE2-CCSD (with chempot matching) energy of mol2: {es_E2:.8f}")

print("\n___GRADIENT SCANNER___")
print(f"BE3-CCSD energy of mol1: {gs_E1:.8f}")
print("BE3-CCSD numerical grad of mol1: \n", gs_grad1)
print(f"\nBE3-CCSD energy of mol2: {gs_E2:.8f}")
print("BE3-CCSD numerical grad of mol2: \n", gs_grad2)
