# Perform BE calculation with 6-31g basis set

from pyscf import gto, scf
from molbe import fragpart, BE

# Perform pyscf HF calculation to get mol & mf objects
mol = gto.M(
    atom="""
C  3.74360      5.55710      7.14890
C  3.18510      4.41510      6.58860
C  3.18510      4.41510      5.17210
C  3.74360      5.55710      4.61180
H  2.79260      3.59960      4.57700
H  2.79260      3.59960      7.18370
S  3.39270      4.78350      9.80840
S  4.27710      6.66240      5.88040
C  3.92620      5.88880     11.07700
C  4.48470      7.03070     10.51670
C  4.48470      7.03070      9.10020
C  3.92620      5.88870      8.53990
H  4.87720      7.84630      8.50510
H  4.87720      7.84630     11.11180
""",
    basis="6-31g",
    charge=0,
)


mf = scf.RHF(mol)
mf.conv_tol = 1e-12
mf.kernel()

# Define fragments; use IAO scheme with 'sto-3g' as the minimal basis set
fobj = fragpart(be_type="be2", mol=mol, valence_basis="sto-3g", frozen_core=True)

# Initialize BE
mybe = BE(mg, fobj, lo_method="iao")

# Density matching with CCSD as local solver
mybe.optimize(solver="CCSD")
