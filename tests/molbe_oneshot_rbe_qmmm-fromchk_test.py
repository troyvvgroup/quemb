# Illustrating one-shot restricted BE with QM/MM for octane in an MM field
# using the be2puffin functionality, starting from a checkfile.
# Returns BE CCSD energy for the system

import numpy as np

from quemb.molbe.misc import be2puffin


def test_rbe_qmmm_fromchk():
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
        n_BE=2,  # BE type: this sets the fragment size.
        frozen_core=False,  # Frozen core
        unrestricted=False,  # specify restricted calculation
        from_chk=True,  # can save the RHF as PySCF checkpoint.
        checkfile="data/oneshot_rbe_qmmm.chk",
    )
    assert np.isclose(be_energy, -0.54879605)
