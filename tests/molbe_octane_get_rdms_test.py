# Illustrates BE computation on octane with RDMs


import numpy as np
from pyscf import gto, scf

from quemb.molbe import BE, fragpart

# TODO: actually add meaningful tests for RDM elements,
#   energies etc.
#   At the moment the test fails already for technical reasons.


def test_rdm():
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

    # initialize fragments (use frozen core approximation)
    fobj = fragpart(be_type="be2", mol=mol, frozen_core=True)
    # Initialize BE
    mybe = BE(mf, fobj)

    # Perform BE density matching.
    mybe.optimize(solver="CCSD", nproc=1, ompnum=1)

    rdm1_ao, rdm2_ao = mybe.rdm1_fullbasis(return_ao=True)  # noqa: F841

    assert np.isclose(mybe.ebe_tot, -310.3311676424482)

    rdm1, rdm2 = mybe.compute_energy_full(approx_cumulant=True, return_rdm=True)  # noqa: F841

    assert np.isclose(mybe.ebe_tot, -310.3311676424482)


if __name__ == "__main__":
    test_rdm()
