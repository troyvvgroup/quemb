# Illustrates parallelized BE computation on octane


import os
import tempfile
from typing import Tuple

import numpy as np
import pytest
from pyscf import gto, scf

from quemb.molbe import BE, fragpart
from quemb.shared.io import write_cube


def test_BE2_octane_molbe() -> None:
    # Prepare octane molecule
    mol, mf = prepare_octane()

    # initialize fragments (without using frozen core approximation)
    for frag_type in ["autogen", "chemgen"]:
        fobj = fragpart(be_type="be2", frag_type=frag_type, mol=mol, frozen_core=False)
        # Initialize BE
        mybe = BE(mf, fobj)

        # Perform BE density matching.
        # Uses 4 procs, each fragment calculation assigned OMP_NUM_THREADS to 2
        # effectively running 2 fragment calculations in parallel
        mybe.optimize(solver="CCSD", nproc=4, ompnum=2)

        assert np.isclose(mybe.ebe_tot, -310.3347211309688)
        assert np.isclose(mybe.ebe_hf, -309.7847696458918)
        # Note that the test for the correlation energy is stricter, because np.isclose
        # scales the difference threshold by the absolute values of the inputs.
        assert np.isclose(mybe.ebe_tot - mybe.ebe_hf, -0.5499514850769742)


@pytest.mark.skipif(
    os.getenv("QUEMB_SKIP_EXPENSIVE_TESTS") == "true",
    reason="Skipped expensive BE3 test for QuEmb.",
)
def test_BE3_octane_molbe() -> None:
    # Prepare octane molecule
    mol, mf = prepare_octane()

    # initialize fragments (without using frozen core approximation)
    for frag_type in ["autogen", "chemgen"]:
        fobj = fragpart(be_type="be3", frag_type=frag_type, mol=mol, frozen_core=False)
        # Initialize BE
        mybe = BE(mf, fobj)

        # Perform BE density matching.
        # Uses 4 procs, each fragment calculation assigned OMP_NUM_THREADS to 2
        # effectively running 2 fragment calculations in parallel
        mybe.optimize(solver="CCSD", nproc=4, ompnum=2)

        assert np.isclose(mybe.ebe_tot, -310.3344717358742)
        assert np.isclose(mybe.ebe_hf, -309.7847695501025)
        # Note that the test for the correlation energy is stricter, because np.isclose
        # scales the difference threshold by the absolute values of the inputs.
        assert np.isclose(mybe.ebe_tot - mybe.ebe_hf, -0.5497021857717073)


def test_cubegen() -> None:
    # Prepare octane molecule
    mol, mf = prepare_octane()
    # Build fragments
    fobj = fragpart(be_type="be2", mol=mol, frozen_core=True)
    # Run BE2
    mybe = BE(mf, fobj)
    mybe.optimize(solver="CCSD", nproc=1, ompnum=1)
    # Write cube file to a temporary location
    with tempfile.TemporaryDirectory() as tmpdir:
        write_cube(mybe, tmpdir, fragment_idx=[3], cubegen_kwargs=dict(resolution=5))
        with open(os.path.join(tmpdir, "frag_3_orb_2.cube"), "r") as f:
            cube_content = np.fromstring(
                "".join(f.read().split("\n")[2:]), sep=" ", dtype=float
            )
        with open("data/octane_frag_3_orb_2.cube", "r") as f:
            reference_content = np.fromstring(
                "".join(f.read().split("\n")[2:]), sep=" ", dtype=float
            )
        assert np.isclose(cube_content, reference_content).all()


def prepare_octane() -> Tuple[gto.Mole, scf.hf.RHF]:
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
    return mol, mf
