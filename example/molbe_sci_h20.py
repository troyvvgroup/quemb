# Illustrates parallelized BE computation on octane

import pathlib

from pyscf import gto, scf

from quemb.molbe import BE, fragmentate
from quemb.molbe.chemfrag import ChemGenArgs
from quemb.molbe.solver import SHCI_ArgsUser
from quemb.shared.config import settings

# Set the path to the scratch directory
settings.SCRATCH_ROOT = pathlib.Path("/tmp/")

# Initialize molecule and run reference HF calculation
mol = gto.M(
    atom="""
        H 0 0 0
        H 0.8 0 0
        H 2 0 0
        H 2.8 0 0
        H 4 0 0
        H 4.8 0 0
        H 6 0 0
        H 6.8 0 0
        H 8 0 0
        H 8.8 0 0
        H 10 0 0
        H 10.8 0 0
        H 12 0 0
        H 12.8 0 0
        H 14 0 0
        H 14.8 0 0
        H 16 0 0
        H 16.8 0 0
        H 18 0 0
        H 18.8 0 0
    """,
    basis="sto-3g",
    charge=0,
)


mf = scf.RHF(mol)
mf.conv_tol = 1e-12
mf.kernel()

# initialize fragments
# Using chemgen, treat_H_different is False to treat the hydrogen chain correctly
add_args = ChemGenArgs(treat_H_different=False)
fobj = fragmentate(
    n_BE=2,
    mol=mol,
    frozen_core=False,
    frag_type="chemgen",
    additional_args=add_args,
)

# Initialize BE
mybe = BE(mf, fobj, thr_bath=1.0e-8)

# Perform BE density matching.
# return_frag_data will save the SCI data from every iteration
add_solver_args = SHCI_ArgsUser(hci_cutoff=0.01, hci_pt=False, return_frag_data=False)
mybe.optimize(
    solver="SCI",
    nproc=1,
    only_chem=True,
    use_cumulant=True,
    solver_args=add_solver_args,
)

print("Expected be_corr for H20, BE(2), HCI_cutoff=0.01 is ", 0.20727426482094202)
