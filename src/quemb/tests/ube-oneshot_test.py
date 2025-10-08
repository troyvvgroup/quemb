"""
This script tests the one-shot UBE energies for a selection of molecules.
This tests for hexene anion and cation in minimal basis with and
without frozen core.

Author(s): Leah Weisburn
"""

import os
import unittest

from pyscf import gto, scf

from quemb.molbe import UBE, fragmentate


class TestOneShot_Unrestricted(unittest.TestCase):
    @unittest.skipUnless(
        os.getenv("QUEMB_DO_KNOWN_TO_FAIL_TESTS") == "true",
        "This test is known to fail.",
    )
    def test_hexene_anion_sto3g_frz_ben(self):
        # Linear Equidistant (r=1Å) H8 Chain, STO-3G
        mol = gto.M()
        mol.atom = os.path.join(os.path.dirname(__file__), "xyz/hexene.xyz")
        mol.basis = "sto-3g"
        mol.charge = -1
        mol.spin = 1
        mol.build()
        self.molecular_unrestricted_oneshot_test(
            mol, 1, "Hexene Anion Frz (BE1)", True, -0.35753374
        )
        self.molecular_unrestricted_oneshot_test(
            mol, 2, "Hexene Anion Frz (BE2)", True, -0.34725961
        )
        self.molecular_unrestricted_oneshot_test(
            mol, 3, "Hexene Anion Frz (BE3)", True, -0.34300834
        )

    @unittest.skipUnless(
        os.getenv("QUEMB_DO_KNOWN_TO_FAIL_TESTS") == "true",
        "This test is known to fail.",
    )
    def test_hexene_cation_sto3g_frz_ben(self):
        # Linear Equidistant (r=1Å) H8 Chain, cc-pVDZ
        mol = gto.M()
        mol.atom = os.path.join(os.path.dirname(__file__), "xyz/hexene.xyz")
        mol.basis = "sto-3g"
        mol.charge = 1
        mol.spin = 1
        mol.build()
        self.molecular_unrestricted_oneshot_test(
            mol, 1, "Hexene Cation Frz (BE1)", True, -0.40383508
        )
        self.molecular_unrestricted_oneshot_test(
            mol, 2, "Hexene Cation Frz (BE2)", True, -0.36496690
        )
        self.molecular_unrestricted_oneshot_test(
            mol, 3, "Hexene Cation Frz (BE3)", True, -0.36996484
        )

    @unittest.skipUnless(
        os.getenv("QUEMB_DO_KNOWN_TO_FAIL_TESTS") == "true",
        "This test is known to fail.",
    )
    def test_hexene_anion_sto3g_unfrz_ben(self):
        # Octane, STO-3G
        mol = gto.M()
        mol.atom = os.path.join(os.path.dirname(__file__), "xyz/hexene.xyz")
        mol.basis = "sto-3g"
        mol.charge = -1
        mol.spin = 1
        mol.build()
        self.molecular_unrestricted_oneshot_test(
            mol, 1, "Hexene Anion Unfrz (BE1)", False, -0.38478279
        )
        self.molecular_unrestricted_oneshot_test(
            mol, 2, "Hexene Anion Unfrz (BE2)", False, -0.39053689
        )
        self.molecular_unrestricted_oneshot_test(
            mol, 3, "Hexene Anion Unfrz (BE3)", False, -0.38960174
        )

    @unittest.skipUnless(
        os.getenv("QUEMB_DO_KNOWN_TO_FAIL_TESTS") == "true",
        "This test is known to fail.",
    )
    def test_hexene_cation_sto3g_unfrz_ben(self):
        mol = gto.M()
        mol.atom = os.path.join(os.path.dirname(__file__), "xyz/hexene.xyz")
        mol.basis = "sto-3g"
        mol.charge = 1
        mol.spin = 1
        mol.build()
        self.molecular_unrestricted_oneshot_test(
            mol, 1, "Hexene Cation Frz (BE1)", False, -0.39471433
        )
        self.molecular_unrestricted_oneshot_test(
            mol, 2, "Hexene Cation Frz (BE2)", False, -0.39846777
        )
        self.molecular_unrestricted_oneshot_test(
            mol, 3, "Hexene Cation Frz (BE3)", False, -0.39729184
        )

    def molecular_unrestricted_oneshot_test(
        self, mol, n_BE, test_name, frz, exp_result, delta=1e-4
    ):
        mf = scf.UHF(mol)
        mf.kernel()
        fobj = fragmentate(frag_type="autogen", n_BE=n_BE, mol=mol, frozen_core=frz)
        mybe = UBE(mf, fobj)
        mybe.oneshot(solver="UCCSD", nproc=1)
        self.assertAlmostEqual(
            mybe.ebe_tot - mybe.uhf_full_e,
            exp_result,
            msg="Unrestricted One-Shot Energy for "
            + test_name
            + " is incorrect by"
            + str(mybe.ebe_tot - mybe.uhf_full_e - exp_result),
            delta=delta,
        )


if __name__ == "__main__":
    unittest.main()
