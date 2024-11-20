# Illustrates a one-shot BE UCCSD calculation from UHF reference
# for hexene anion

import numpy

from pyscf import gto, scf
from molbe import fragpart, UBE, be_var
import sys

# Set up scratch directory settings
# be_var.SCRATCH='{scratch location}'
# be_var.CREATE_SCRATCH_DIR=True

# Give path to structure xyz file
structure = "data/hexene.xyz"

# Build PySCF molecule object
mol = gto.M()
mol.atom = structure
mol.basis = "sto-3g"
mol.charge = -1
mol.spin = 1
mol.build()

# Run UHF with PySCF
mf = scf.UHF(mol)
mf.kernel()

# Specify number of processors
nproc = 1

# Initialize fragments without frozen core approximation at BE2 level
fobj = fragpart(frag_type="autogen", be_type="be2", mol=mol, frozen_core=False)
# Initialize UBE
mybe = UBE(mf, fobj)

# Perform one round of BE, without density or chemical potential matching,
# and return the energy.
# clean_eri will delete all of the ERI files from scratch
mybe.oneshot(solver="UCCSD", nproc=nproc, calc_frag_energy=True, clean_eri=True)
