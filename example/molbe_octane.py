# Illustrates parallelized BE computation on octane

from pyscf import gto, scf, cc
from molbe import fragpart, BE

# Perform pyscf HF calculation to get mol & mf objects
mol = gto.M(
    atom="""
C   0.4419364699  -0.6201930287   0.0000000000
C  -0.4419364699   0.6201930287   0.0000000000
H  -1.0972005331   0.5963340874   0.8754771384
H   1.0972005331  -0.5963340874  -0.8754771384
H  -1.0972005331   0.5963340874  -0.8754771384
H   1.0972005331  -0.5963340874   0.8754771384
C   0.3500410560   1.9208613544   0.0000000000
C  -0.3500410560  -1.9208613544   0.0000000000
H   1.0055486349   1.9450494955   0.8754071298
H  -1.0055486349  -1.9450494955  -0.8754071298
H   1.0055486349   1.9450494955  -0.8754071298
H  -1.0055486349  -1.9450494955   0.8754071298
C  -0.5324834907   3.1620985364   0.0000000000
C   0.5324834907  -3.1620985364   0.0000000000
H  -1.1864143468   3.1360988730  -0.8746087226
H   1.1864143468  -3.1360988730   0.8746087226
H  -1.1864143468   3.1360988730   0.8746087226
H   1.1864143468  -3.1360988730  -0.8746087226
C   0.2759781663   4.4529279755   0.0000000000
C  -0.2759781663  -4.4529279755   0.0000000000
H   0.9171145792   4.5073104916   0.8797333088
H  -0.9171145792  -4.5073104916  -0.8797333088
H   0.9171145792   4.5073104916  -0.8797333088
H  -0.9171145792  -4.5073104916   0.8797333088
H   0.3671153250  -5.3316378285   0.0000000000
H  -0.3671153250   5.3316378285   0.0000000000
""",
    basis="sto-3g",
    charge=0,
)


mf = scf.RHF(mol)
mf.conv_tol = 1e-12
mf.kernel()

# Perform CCSD calculation to get reference energy for comparison
mc = cc.CCSD(mf, frozen=8)
mc.verbose = 0
ccsd_ecorr = mc.kernel()[0]
print(f"*** CCSD Correlation Energy: {ccsd_ecorr:>14.8f} Ha", flush=True)

# initialize fragments (use frozen core approximation)
fobj = fragpart(be_type="be2", mol=mol, frozen_core=True)
# Initialize BE
mybe = BE(mf, fobj)

# Perform BE density matching.
# Uses 20 procs, each fragment calculation assigned OMP_NUM_THREADS to 4
# effectively running 5 fragment calculations in parallel
mybe.optimize(solver="CCSD", nproc=20, ompnum=4)

# Compute error
be_ecorr = mybe.ebe_tot - mybe.ebe_hf
err_ = (ccsd_ecorr - be_ecorr) * 100.0 / ccsd_ecorr
print(f"*** BE2 Correlation Energy Error (%) : {err_:>8.4f} %")
