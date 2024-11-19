# Illustrating one-shot UBE with QM/MM for octane in an MM field
# using the be2puffin functionality.
# Returns UBE UCCSD energy for the system

from molbe.misc import be2puffin

# variables for scratch handling
# pbe_var.SCRATCH = '{}'
# pbe_var.CREATE_SCRATCH_DIR = True

# Give structure XYZ, in Angstroms
structure = "data/octane.xyz"

# Set MM charges and their positions to use PySCF's QM/MM
# functionality. Note that the units for the coordinates are
# in Bohr and the units for the structure are in Angstrom
# to match Q4Bio application. This can be changed in
# misc/be2puffin

charges = [-0.2, -0.1, 0.15, 0.2]
coords = [(-3, -8, -2), (-2, 6, 1), (2, -5, 2), (1, 8, 1.5)]


# returns UBE energy with UCCSD solver from UHF reference
be_energy = be2puffin(
    structure,  # the QM region XYZ geometry
    "STO-3G",  # the chosen basis set
    pts_and_charges=[coords, charges],  # the point coordinates and coordinates
    use_df=False,  # keep density fitting False for PySCF UHF
    charge=-1,  # charge of QM region
    spin=1,  # spin of QM region
    nproc=1,  # number of processors to parallize across
    ompnum=2,  # number of nodes to parallelize across
    be_type="be2",  # BE type: this sets the fragment size.
    frozen_core=False,  # keep this to False for non-minimal basis: localization and
    # numerical problems for ruthenium systems in non-minimal basis
    unrestricted=True,  # specify unrestricted calculation
    from_chk=False,  # can save the UHF as PySCF checkpoint.
    # Set to true if running from converged UHF chk
    checkfile=None,
)  # if not None, will save UHF calculation to a checkfile.
# if rerunning from chk (from_chk=True), name the checkfile here
#            ecp = ecp) # can add ECP for heavy atoms as: {'Ru': 'def2-SVP'}
