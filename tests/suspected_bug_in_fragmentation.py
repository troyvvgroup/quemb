from chemcoord import Cartesian
from numpy import isclose
from pyscf import scf

from quemb.molbe.fragment import fragpart
from quemb.molbe.mbe import BE


def test_matching_order():
    def get_energy(path):
        m = Cartesian.read_xyz(path)
        mol = m.to_pyscf()

        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf.kernel()

        fobj = fragpart(mol=mol, frag_type="autogen", be_type="be3")

        # fobj.center_idx[0][0][0] = 200

        mybe = BE(mf, fobj)

        mybe.optimize(solver="CCSD")
        return mybe.ebe_tot - mybe.ebe_hf

    assert isclose(
        get_energy("data/octane.xyz"), get_energy("./data/suspected_bug_octane.xyz")
    )
