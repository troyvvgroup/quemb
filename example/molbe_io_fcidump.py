# Illustrates how fcidump file containing fragment hamiltonian
# can be generated using be2fcidump

from pyscf import gto, scf
from molbe import BE
from molbe import fragpart
from molbe.misc import *
from molbe import be_var

be_var.PRINT_LEVEL = 3

# Read in molecular integrals expressed in libint basis ordering
# numpy.loadtxt takes care of the input under the hood
mol, mf = libint2pyscf(
    "data/octane.xyz", "data/hcore_libint_octane.dat", "STO-3G", hcore_skiprows=1
)
mf.kernel()

# Construct fragments for BE
fobj = fragpart(be_type="be2", mol=mol)
oct_be = BE(mf, fobj)

# Write out fcidump file for each fragment
be2fcidump(oct_be, "octane", "fragment_mo")
