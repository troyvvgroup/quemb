from os import getenv

import pytest
from chemcoord import Cartesian
from numpy import isclose
from pyscf import scf

from quemb.molbe.fragment import fragmentate
from quemb.molbe.mbe import BE


@pytest.mark.skipif(
    not getenv("QUEMB_DO_KNOWN_TO_FAIL_TESTS") == "true",
    reason="This test is known to fail.",
)
def test_matching_order():
    def get_energy(path):
        m = Cartesian.read_xyz(path)
        mol = m.to_pyscf(basis="sto-3g")

        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf.kernel()

        fobj = fragmentate(mol=mol, frag_type="autogen", n_BE=3)

        # fobj.center_idx[0][0][0] = 200

        mybe = BE(mf, fobj)

        mybe.optimize(solver="CCSD")
        return mybe.ebe_tot - mybe.ebe_hf

    assert isclose(
        get_energy("data/octane.xyz"), get_energy("./data/suspected_bug_octane.xyz")
    )
