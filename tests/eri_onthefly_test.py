"""
This script tests the on-the-fly DF-based ERI transformation routine for
molecular systems.

Author(s): Minsik Cho
"""

import logging
import os
import unittest

from pyscf import gto, scf

from quemb.molbe import BE, fragmentate

CHKFILE = os.path.join(os.path.dirname(__file__), "chk/octane_ccpvtz.h5")

class TestDF_ontheflyERI(unittest.TestCase):
    @unittest.skipUnless(
        (
            os.getenv("QUEMB_DO_EXPENSIVE_TESTS") == "true"
            and not os.getenv("GITHUB_ACTIONS") == "true"
        ),
        "Skipped expensive tests for QuEmb.",
    )
    def test_octane_BE2_large(self):
        # Octane, cc-pvtz
        mol = gto.M()
        mol.atom = os.path.join(os.path.dirname(__file__), "xyz/octane.xyz")
        mol.basis = "cc-pvtz"
        mol.charge = 0.0
        mol.spin = 0.0
        mol.build()
        mf = scf.RHF(mol)
        mf.direct_scf = True

        if os.path.exists(CHKFILE):
            mf = mf.from_hdf5(CHKFILE, "scf") 
        else:
            mf.kernel()
            mf.to_hdf5(CHKFILE, "scf")

        fobj = fragmentate(frag_type="autogen", n_BE=2, mol=mol)
        mybe = BE(mf, fobj, auxbasis="cc-pvtz-ri", int_transform="int-direct-DF")
        self.assertAlmostEqual(
            mybe.ebe_hf,
            mf.e_tot,
            msg="HF-in-HF energy for Octane (BE2) does not match the HF energy!",
            delta=1e-6,
        )

    def test_octane_BE2_small(self):
        # Octane, cc-pvtz
        mol = gto.M(
            os.path.join(os.path.dirname(__file__), "xyz/octane.xyz"), basis="sto-3g"
        )

        mf = scf.RHF(mol)
        mf.kernel()

        fobj = fragmentate(frag_type="chemgen", n_BE=2, mol=mol)

        incore_BE = BE(mf, fobj, int_transform="in-core")
        incore_BE.oneshot(solver="CCSD")

        int_direct_DF_BE = BE(
            mf, fobj, auxbasis="weigend", int_transform="int-direct-DF"
        )
        int_direct_DF_BE.oneshot(solver="CCSD")

        self.assertAlmostEqual(
            (incore_BE.ebe_tot - int_direct_DF_BE.ebe_tot),
            -6.078869063230741e-05,
            delta=1e-8,
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,  # Set desired level
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    unittest.main()
