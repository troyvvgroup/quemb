"""
This script tests the on-the-fly DF-based ERI transformation routine for molecular systems.

Author(s): Minsik Cho
"""

import os
import unittest

from pyscf import gto, scf

from molbe import BE, fragpart


class TestDF_ontheflyERI(unittest.TestCase):
    @unittest.skipIf(
        os.getenv("QUEMB_SKIP_EXPENSIVE_TESTS") == "true",
        "Skipped expensive tests for QuEmb.",
    )
    def test_octane_BE2(self):
        # Octane, cc-pvtz
        mol = gto.M()
        mol.atom = os.path.join(os.path.dirname(__file__), "xyz/octane.xyz")
        mol.basis = "cc-pvtz"
        mol.charge = 0.0
        mol.spin = 0.0
        mol.build()
        mf = scf.RHF(mol)
        mf.direct_scf = True
        mf.kernel()
        fobj = fragpart(frag_type="autogen", be_type="be2", mol=mol)
        mybe = BE(mf, fobj, integral_direct_DF=True)
        self.assertAlmostEqual(
            mybe.ebe_hf,
            mf.e_tot,
            msg="HF-in-HF energy for Octane (BE2) does not match the HF energy!",
            delta=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
