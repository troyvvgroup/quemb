# Illustrates how fcidump file containing fragment hamiltonian
# can be generated using be2fcidump

from quemb.molbe import BE, fragmentate
from quemb.molbe.misc import be2fcidump, libint2pyscf

# Read in molecular integrals expressed in libint basis ordering
# numpy.loadtxt takes care of the input under the hood
mol, mf = libint2pyscf(
    "data/octane.xyz", "data/hcore_libint_octane.dat", "STO-3G", hcore_skiprows=1
)
mf.kernel()

# Construct fragments for BE
fobj = fragmentate(n_BE=2, mol=mol)
oct_be = BE(mf, fobj)

# Write out fcidump file for each fragment
be2fcidump(oct_be, "octane", "fragment_mo")
