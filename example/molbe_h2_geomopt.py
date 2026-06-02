# Illustrates nuclear gradient and hessian computations as well as geometry optimizaton
# using a custom energy function fed through pyscf's finite difference driver

import numpy as np
from pyscf import grad, gto, scf
from pyscf.tools import finite_diff

from quemb.molbe.geomopt import Energy, energy_hf

# =========================
#  Gradients
# =========================
print("\n\nGRADIENTS")
print("-" * 40)
#
# analytic gradient of mol0
#
print("\nanalytic gradient for mol0:")
mol0 = gto.M(atom="H 0.01 0.02 -0.03; H -0.02 0.01 1.00", basis="6-31g")
mf = scf.RHF(mol0).run()
mf_grad = grad.RHF(mf)
ref_grad_hf = mf_grad.kernel()

#
# numerical gradient of mol0
#
energy_method = Energy(mol0, energy_hf)
fd_grad = finite_diff.kernel(energy_method, displacement=1e-4)  # default is 1e-2 Bohr
rms = np.sqrt(np.mean((ref_grad_hf - fd_grad) ** 2))
print("\nnumerical gradient for mol0:\n", fd_grad)
print(f"RMS from benchmark: {rms:.12e}")

#
# numerical gradient of mol0 using
# pyscf built-in Gradients object
#
grad_method = finite_diff.Gradients(energy_method)
grad_method.displacement = 1e-4  # Gradients class default is 1e-2 Bohr
fd_grad_fromObj = grad_method.kernel()
rms = np.sqrt(np.mean((ref_grad_hf - fd_grad_fromObj) ** 2))
print("\nnumerical gradient for mol0 from Gradients object:\n", fd_grad_fromObj)
print(f"RMS from benchmark: {rms:.12e}")


# =========================
#  Hessians
# =========================
print("\n\nHESSIANS")
print("-" * 40)

#
# analytic hessian of mol0
#
print("\nanalytic hessian for mol0:")
mf = scf.RHF(mol0).run()
ref_hess_hf = mf.Hessian().kernel()
print(ref_hess_hf)

#
# numerical hessian of mol0 using pyscf Gradients
# object as input to the built-in Hessian object
#
hess_method = finite_diff.Hessian(grad_method)
hess_method.displacement = 1e-4  # Hessian class default is 1e-2 Bohr
fd_hess = hess_method.kernel()
rms = np.sqrt(np.mean((ref_hess_hf - fd_hess) ** 2))
print("\nnumerical hessian for mol0 from Gradients object:\n", fd_hess)
print(f"RMS from benchmark: {rms:.12e}")

# =========================
#  Scanners
# =========================
print("\n\nSCANNERS")
print("-" * 40)

mol1 = gto.M(atom="H 0.02 -0.01  0.00; H  -0.02 -0.03 0.80", basis="6-31g")
mol2 = gto.M(atom="H 0.00  0.03  0.01; H   0.02 -0.01 0.70", basis="6-31g")

#
# energy point calculations using as_scanner()
#
print("\nUsing energy_method.as_scanner():")
energy_scanner = energy_method.as_scanner()
E0 = energy_scanner(mol0)
E1 = energy_scanner(mol1)
E2 = energy_scanner(mol2)
print(f"energy of mol0: {E0:.8f}")
print(f"energy of mol1: {E1:.8f}")
print(f"energy of mol2: {E2:.8f}")

#
# gradient point calculations using as_scanner()
#
print("\nUsing grad_method.as_scanner():")
grad_scanner = grad_method.as_scanner()
E0, grad0 = grad_scanner(mol0)
E1, grad1 = grad_scanner(mol1)
E2, grad2 = grad_scanner(mol2)
print(f"energy of mol0: {E0:.8f}")
print("numerical grad of mol0: \n", grad0)
print(f"\nenergy of mol1: {E1:.8f}")
print("numerical grad of mol1: \n", grad1)
print(f"\nenergy of mol2: {E2:.8f}")
print("numerical grad of mol2: \n", grad2)


# =========================
#  Geometry Optimization
# =========================
print("\n\nGEOMETRY OPTIMIZATION")
print("-" * 40)

#
# Analytical geometry optimization using
# geomeTRIC, pyberny (needs to be installed)
#
print("\nGeometry optimization using analytic gradients...")
mol0_opt = (
    mf.Gradients().optimizer(solver="geomeTRIC").kernel()
)  # solver='berny' can't do H-chains
print("\nanalytically optimized geometry from mol0:\n", mol0_opt.tostring())
mf = scf.RHF(mol0_opt).run()

#
# Numerical geometry optimization using Gradients object
# with geomeTRIC, pyberny (needs to be installed)
#
print("\nRunning the geometric geometry optimizer using fd class...")
mol0_opt_fd = grad_method.optimizer(
    solver="geomeTRIC"
).kernel()  # solver='berny' can't do H-chains
print("\nnumerically optimized geometry from mol0:\n", mol0_opt_fd.tostring())
mf = scf.RHF(mol0_opt_fd).run()
rms = np.sqrt(np.mean((mol0_opt.atom_coords() - mol0_opt_fd.atom_coords()) ** 2))
print(f"RMS from benchmark: {rms:.12e}")
