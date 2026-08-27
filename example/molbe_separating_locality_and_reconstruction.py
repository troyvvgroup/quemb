import numpy as np
from pyscf import cc, gto, scf
from pyscf.tools import finite_diff
from quemb.molbe.chemfrag import ChemGenArgs
from quemb.molbe.scanner import (
    BEArgs,
    Energy,
    be_ref_data,
    be_frag_ref_data,
    energy_be,
    energy_be_frag,
)

mol = gto.M(
    atom="""
C         -9.34365       -1.15645       -0.59044 
H         -9.69311       -0.40929       -1.31014 
H         -9.68356       -0.85885        0.40667 
C         -9.85706       -2.54152       -0.95028 
H         -9.47312       -2.82186       -1.93806 
C        -11.38479       -2.58895       -0.95421 
H         -9.46363       -3.26894       -0.23071 
H        -11.77324       -1.85666       -1.67249 
H        -11.76375       -2.30372        0.03478 
C        -11.89717       -3.98384       -1.31663 
H        -11.51823       -4.26884       -2.30567 
C        -13.42604       -4.03140       -1.32061 
H        -11.50876       -4.71596       -0.59820 
H        -13.81445       -3.29928       -2.03904 
H        -13.80498       -3.74641       -0.33157 
C        -13.93843       -5.42628       -1.68303 
H        -13.55946       -5.71152       -2.67202 
C        -15.46616       -5.47368       -1.68698 
H        -13.55001       -6.15859       -0.96475 
H        -15.85957       -4.74625       -2.40655 
H        -15.85012       -5.19334       -0.69921 
C        -15.97962       -6.85873       -2.04684 
H        -15.63969       -7.15633       -3.04395 
H        -17.07405       -6.86988       -2.04370 
H        -15.63019       -7.60591       -1.32714 
H         -8.24922       -1.14526       -0.59360 
""",
    basis="sto-3g",
    charge=0,
)
print(f"the number of basis functions is {mol.nao}", flush=True)

##### Compute the Reference Gradient #####
mf = scf.RHF(mol)
mf.conv_tol = 1e-12
mf._eri = mol.intor("int2e", aosym="s8")
mf.kernel()
mc = cc.CCSD(mf)
mc.kernel()
mc_grad = mc.nuc_grad_method()
ref_grad_corr = mc_grad.kernel()
np.savetxt("ref.txt", ref_grad_corr, fmt="%.12e")

##### Energy Embedding Gradient #####
be_args = BEArgs(
    n_BE=2,
    solver="CCSD",
    optimize=False,
    use_cumulant=False,
    additional_args=ChemGenArgs(h_treatment="treat_H_diff"),
)

energy_method = Energy(mol, energy_be, energy_args=be_args, ref_data_func=be_ref_data)
grad_method = finite_diff.Gradients(energy_method)
grad_method.displacement = 1e-4
fd_grad_fromObj = grad_method.kernel()

rmse = np.sqrt(np.mean((fd_grad_fromObj - ref_grad_corr) ** 2))
print(f"The RMSE for energy embedding is {rmse:.12e}")

##### Force Embedding #####
be_args = BEArgs(
    n_BE=2,
    solver="CCSD",
    optimize=False,
    use_cumulant=False,
    additional_args=ChemGenArgs(h_treatment="treat_H_diff"),
    reconstruct_frag_energy=False,
)

energy_method = Energy(
    mol, energy_be_frag, energy_args=be_args, ref_data_func=be_frag_ref_data
)
grad_method = finite_diff.Gradients(energy_method)
grad_method.displacement = 1e-4
fd_grad_fromObj = grad_method.kernel()

rmse = np.sqrt(np.mean((fd_grad_fromObj - ref_grad_corr) ** 2))
print(f"The RMSE for force embedding is {rmse:.12e}")

##### Force Embedding With Democratically Partitioned Fragment Energies #####
be_args = BEArgs(
    n_BE=2,
    solver="CCSD",
    optimize=False,
    use_cumulant=False,
    additional_args=ChemGenArgs(h_treatment="treat_H_diff"),
    reconstruct_frag_energy=True,
)

energy_method = Energy(
    mol, energy_be_frag, energy_args=be_args, ref_data_func=be_frag_ref_data
)
grad_method = finite_diff.Gradients(energy_method)
grad_method.displacement = 1e-4
fd_grad_fromObj = grad_method.kernel()

rmse = np.sqrt(np.mean((fd_grad_fromObj - ref_grad_corr) ** 2))
print(
    f"The RMSE for force embedding with democratically partitioned fragment energies is {rmse:.12e}"
)
