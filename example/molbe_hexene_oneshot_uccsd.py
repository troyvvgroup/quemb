# Illustrates a one-shot BE UCCSD calculation from UHF reference
# for hexene anion


from pyscf import gto, scf

from quemb.molbe import UBE, fragpart

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
fobj = fragpart(frag_type="autogen", n_BE=2, mol=mol, frozen_core=False)
# Initialize UBE
mybe = UBE(mf, fobj)

# Perform one round of BE, without density or chemical potential matching,
# and return the energy.
# clean_eri will delete all of the ERI files from scratch
mybe.oneshot(solver="UCCSD", nproc=nproc)
