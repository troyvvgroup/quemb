# Illustrating one-shot restricted BE with QM/MM for octane in an MM field
# using the be2puffin functionality, starting from a checkfile.
# Returns BE CCSD energy for the system

import numpy
from pyscf import gto, scf, qmmm
from molbe.misc import be2puffin
from molbe import be_var

# variables for scratch handling
# pbe_var.SCRATCH = '{}'
# pbe_var.CREATE_SCRATCH_DIR = True

# Set MM charges and their positions to use PySCF's QM/MM
# functionality. Note that the units for the coordinates are
# in Bohr and the units for the structure are in Angstrom
# to match Q4Bio application. This can be changed in
# misc/be2puffin

charges = [-0.2, -0.1, 0.15, 0.2]
coords = [(-3, -8, -2), (-2, 6, 1), (2, -5, 2), (1, 8, 1.5)]

# Give structure XYZ, in Angstroms
structure = "data/octane.xyz"

# returns BE energy with CCSD solver from RHF reference,
# using checkfile from converged RHF
be_energy = be2puffin(
    structure,  # the QM region XYZ geometry
    "sto-3g",  # the chosen basis set
    pts_and_charges=[coords, charges],  # the loaded hamiltonian
    use_df=False,  # density fitting
    charge=0,  # charge of QM region
    spin=0,  # spin of QM region
    nproc=1,  # number of processors to parallize across
    ompnum=2,
    be_type="be2",  # BE type: this sets the fragment size.
    frozen_core=False,  # Frozen core
    unrestricted=False,  # specify restricted calculation
    from_chk=True,  # can save the RHF as PySCF checkpoint.
    # Set to true if running from converged UHF chk
    checkfile="data/oneshot_rbe_qmmm.chk",
)  # if not None, will save RHF calculation to a checkfile.
# if rerunning from chk (from_chk=True), name the checkfile here
#            ecp = ecp) # can add ECP for heavy atoms as: {'Ru': 'def2-SVP'}

"""
To not use or generate checkfile:
from_chk = False
checkfile = None

To generate checkfile in be2puffin:
from_chk = False
checkfile = {Name_of_checkfile}

To use checkfile:
from_chk = True
checkfile = {Name_of_checkfile}
"""
