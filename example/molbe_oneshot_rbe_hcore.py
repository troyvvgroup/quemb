# Illustrating one-shot restricted BE with QM/MM for octane in an MM field
# using the be2puffin functionality, starting from integrals in the libint format.
# Returns BE CCSD energy for the system
# Included is a demonstration of the conversion of integrals between libint and
# PySCF formats

import numpy
from pyscf import gto, scf, qmmm
from molbe.misc import be2puffin
from molbe import be_var

# variables for scratch handling
# pbe_var.SCRATCH = '{}'
# pbe_var.CREATE_SCRATCH_DIR = True


# Convert PySCF integrals to libint format
def pyscf2lint(mol, hcore_pyscf):
    pyscf2libint_ind = []
    for labelidx, label in enumerate(mol.ao_labels()):
        # pyscf: px py pz // 1 -1 0
        # libint: py pz px // -1 0 1
        if "p" not in label.split()[2]:
            pyscf2libint_ind.append(labelidx)
        else:
            if "x" in label.split()[2]:
                pyscf2libint_ind.append(labelidx + 1)
            elif "y" in label.split()[2]:
                pyscf2libint_ind.append(labelidx + 1)
            elif "z" in label.split()[2]:
                pyscf2libint_ind.append(labelidx - 2)
    hcore_libint = hcore_pyscf[numpy.ix_(pyscf2libint_ind, pyscf2libint_ind)]
    return hcore_libint


# Set MM charges and their positions to use PySCF's QM/MM
# functionality. Note that the units for the coordinates are
# in Bohr and the units for the structure are in Angstrom
# to match Q4Bio application. This can be changed in
# misc/be2puffin

charges = [-0.2, -0.1, 0.15, 0.2]
coords = [(-3, -8, -2), (-2, 6, 1), (2, -5, 2), (1, 8, 1.5)]

# Give structure XYZ, in Angstroms
structure = "data/octane.xyz"

"""
Build hcore in the libint form for QM/MM
"""
mol = gto.M()
mol.atom = structure
mol.build()

# Build QM/MM RHF Hamiltonian
mf1 = scf.RHF(mol)
mf = qmmm.mm_charge(mf1, coords, charges, unit="bohr")
hcore_pyscf = mf.get_hcore()

# Note: can save hcore_pyscf with: numpy.savetxt("hcore_pyscf.dat", hcore_pyscf)
#       and load with: hcore = numpy.loadtext("hcore_pyscf.dat", skiprows = 0)

# Illustrative conversion to libint format
# Note: can directly feed hcore_pyscf into be2puffin with libint_inp = False
hcore_libint = pyscf2lint(mol, hcore_pyscf)

# returns BE energy with CCSD solver from RHF reference,
# using checkfile from converged RHF
be_energy = be2puffin(
    structure,  # the QM region XYZ geometry
    "sto-3g",  # the chosen basis set
    hcore=hcore_libint,  # the loaded hamiltonian
    libint_inp=True,  # True if passing hcore in libint format, False for PySCF
    use_df=False,  # density fitting
    charge=0,  # charge of QM region
    spin=0,  # spin of QM region
    nproc=1,  # number of processors to parallize across
    ompnum=2,
    be_type="be2",  # BE type: this sets the fragment size.
    frozen_core=True,  # Frozen core
    unrestricted=False,  # specify restricted calculation
    from_chk=False,  # can save the RHF as PySCF checkpoint.
    # Set to true if running from converged UHF chk
    checkfile=None,
)  # if not None, will save RHF calculation to a checkfile.
# if rerunning from chk (from_chk=True), name the checkfile here
#            ecp = ecp) # can add ECP for heavy atoms as: {'Ru': 'def2-SVP'}
