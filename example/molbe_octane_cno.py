# Illustrates BE computation on octane using CNOs
# This is intended for calculations in large basis sets, especially to reduce
# required fragment size.
# As such, this calculation takes a while!
# For now, this supports only ChemGen and GraphGen

import pathlib

from pyscf import gto, scf

from quemb.molbe import BE, fragmentate
from quemb.molbe.cno_utils import CNOArgs
from quemb.shared.config import settings

# Set the scratch directory for this calculation (optional scrach setting)
settings.SCRATCH_ROOT = pathlib.Path("path/to/scratch")

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
    basis="cc-pVDZ",
    charge=0,
)


mf = scf.RHF(mol)
mf.conv_tol = 1e-12
mf.kernel()

# initialize fragments, using the IAOs
# For now, we need to use chemgen to make sure we have the necessary geometry
# information passed to the fragment
fobj = fragmentate(
    n_BE=1,
    mol=mol,
    frag_type="chemgen",
    iao_valence_basis="minao",
)

# Try multiple basic CNO schemes, using only one keyword
# For descriptions of each scheme, read about CNOArgs in molbe/cno_utils.py

cno_schemes = ["Proportional", "ProportionalQQ", "HalfFilled"]
for cno in cno_schemes:
    # Initialize BE
    mybe = BE(
        mf,
        fobj,
        add_cnos=True,
        cno_args=CNOArgs(cno_scheme=cno),
        lo_method="IAO",
        iao_loc_method="boys",
    )

    # Perform chemical potential matching for the system
    mybe.optimize(solver="CCSD", nproc=8, only_chem=True)

# Perform thresholding CNO scheme
# Must also use the cno_thresh keyword in CNOArgs to specify the threshold for
# OCNOs and VCNOs to be chosen
# Initialize BE
mybe = BE(
    mf,
    fobj,
    add_cnos=True,
    cno_args=CNOArgs(cno_scheme="Threshold", cno_thresh=1e-4),
    lo_method="IAO",
    iao_loc_method="boys",
)

# Perform chemical potential matching for the system
mybe.optimize(solver="CCSD", nproc=8, only_chem=True)

# Perform exact fragment size CNO-BE
# Must also use the keyword in CNOArgs to specify how many orbitals should be
# in each fragment, and cno_active_scheme to determine the which are chosen
# Initialize BE
mybe = BE(
    mf,
    fobj,
    add_cnos=True,
    cno_args=CNOArgs(
        cno_scheme="ExactFragmentSize",
        cno_active_fragsize_scheme="AddBoth",
        tot_active_orbs=38,
    ),
    lo_method="IAO",
    iao_loc_method="boys",
)

# Perform chemical potential matching for the system
mybe.optimize(solver="CCSD", nproc=8, only_chem=True)

########################################################################################
##                                  Expected Results                                  ##
########################################################################################
#                                   Correlation     Ave. # Schmidt   Ave. #     Ave. # #
#                                    enegy (mHa),     Space Orbs,    OCNOs,     VCNOs  #
# CNO-BE(1)-Proportional:           -1.23419551,         44.5,         0.0,      12.0  #
# CNO-BE(1)-ProprtionalQQ:          -1.23492060,         45.0,         0.0,      12.5  #
# CNO-BE(1)-HalfFilled:             -1.17364571,         50.5,        18.0,       0.0  #
# CNO-BE(1)-Threshold=1e-4:         -1.28464935,        72.25,        7.75,      32.0  #
# CNO-BE(1)-ExactFragmentSize=38:   -1.21041018,         35.0,         0.0        5.5  #
