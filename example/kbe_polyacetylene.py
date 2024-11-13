# Illustrated periodic BE calculation on polyacetylene with 3x1x1 kpoints
# A supercell with 4 carbon & 4 hydrogen atoms is defined as unit cell in
# pyscf's periodic HF calculation

import numpy
from pyscf.pbc import df, gto, scf

from kbe import BE, fragpart

kpt = [1, 1, 3]
cell = gto.Cell()

a = 8.0
b = 8.0
c = 2.455 * 2.0

lat = numpy.eye(3)
lat[0, 0] = a
lat[1, 1] = b
lat[2, 2] = c

cell.a = lat

cell.atom = """
H      1.4285621630072645    0.0    -0.586173422487319
C      0.3415633681566205    0.0    -0.5879921146011252
H     -1.4285621630072645    0.0     0.586173422487319
C     -0.3415633681566205    0.0     0.5879921146011252
H      1.4285621630072645    0.0     1.868826577512681
C      0.3415633681566205    0.0     1.867007885398875
H     -1.4285621630072645    0.0     3.041173422487319
C     -0.3415633681566205    0.0     3.0429921146011254
"""

cell.unit = "Angstrom"
cell.basis = "sto-3g"
cell.verbose = 0
cell.build()

kpts = cell.make_kpts(kpt, wrap_around=True)

mydf = df.GDF(cell, kpts)
mydf.build()
kmf = scf.KRHF(cell, kpts)
kmf.with_df = mydf
kmf.exxdiv = None
kmf.conv_tol = 1e-12
kpoint_energy = kmf.kernel()

# Define fragment in the supercell
kfrag = fragpart(be_type="be2", mol=cell, kpt=kpt, frozen_core=True)
# Initialize BE
mykbe = BE(kmf, kfrag, kpts=kpts)

# Perform BE density matching
mykbe.optimize(solver="CCSD")
