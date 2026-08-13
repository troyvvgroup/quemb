import numpy as np
from pyscf import gto
from pyscf.tools import finite_diff

from quemb.molbe.chemfrag import ChemGenArgs
from quemb.molbe.scanner import (
    BEArgs,
    Energy,
    be_frag_ref_data,
    energy_be_frag,
)

# build the desired geometry
mol = gto.M(
    atom="""
C 2.561733 -0.241464 1.108816
H 3.650762 -0.067606 1.243390
H 1.935495 0.042039 1.981612
H 2.449494 -1.346036 1.068253
C 2.027058 0.529218 -0.124770
H 2.516226 0.141411 -1.043815
H 2.219589 1.618991 -0.026549
C 0.493180 0.625626 -0.090989
H 0.320999 1.295987 0.778086
H 0.044043 1.154326 -0.958787
C -0.387124 -0.594234 0.019892
H -0.020644 -1.186178 0.885696
H -0.216729 -1.240377 -0.867681
C -1.918676 -0.385894 0.070722
H -2.312289 -1.407313 0.260708
H -2.159344 0.169308 1.002468
C -2.623323 0.141394 -1.167133
H -2.131252 -0.188280 -2.107081
H -2.593556 1.249712 -1.095942
H -3.655895 -0.237211 -1.009729
""",
    basis="3-21g",
    charge=0,
)

print(f"the number of basis functions is {mol.nao}", flush=True)

be_args = BEArgs(
    n_BE=1,
    solver="CCSD",
    lo_method="IAO",
    iao_valence_basis="sto-3g",
    additional_args=ChemGenArgs(h_treatment="treat_H_diff"),
)

energy_method = Energy(
    mol, energy_be_frag, energy_args=be_args, ref_data_func=be_frag_ref_data
)
grad_method = finite_diff.Gradients(energy_method)
grad_method.displacement = 1e-4
fd_grad_fromObj = grad_method.kernel()
np.savetxt("polyethylene_BE1.txt", fd_grad_fromObj, fmt="%.12e")
print("done with mol", flush=True)
