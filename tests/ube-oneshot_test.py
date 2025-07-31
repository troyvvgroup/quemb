"""
This script tests the one-shot UBE energies for a selection of molecules.
This tests for hexene anion and cation in minimal basis with and
without frozen core.

Note: we can now run these without custom PySCF and avoiding Numpy errors
by adding extra bath orbitals. The current prescription can
generate slightly different baths and isn't totally deterministic,
so some tests have a larger delta for now. This may be modified in the
future.

Author(s): Leah Weisburn
"""

import os
import unittest

from pyscf import gto, scf

from quemb.molbe import UBE, fragmentate


class TestOneShot_Unrestricted(unittest.TestCase):
    def test_hexene_anion_sto3g_frz_ben(self):
        # Hexene anion with frozen core, STO-3G
        mol = gto.M()
        mol.atom = os.path.join(os.path.dirname(__file__), "xyz/hexene.xyz")
        mol.basis = "sto-3g"
        mol.charge = -1
        mol.spin = 1
        mol.build()
        self.molecular_unrestricted_oneshot_test(
            mol, 1, "Hexene Anion Frz (BE1)", True, -0.35753375
        )
        self.molecular_unrestricted_oneshot_test(
            mol, 2, "Hexene Anion Frz (BE2)", True, -0.34617685, delta=1e-4
        )
        """ Cut for expense
        self.molecular_unrestricted_oneshot_test(
            mol, 3, "Hexene Anion Frz (BE3)", True, -0.34300832
        )
        """

    def test_hexene_cation_sto3g_frz_ben(self):
        # Hexene cation with frozen core, STO-3G
        mol = gto.M()
        mol.atom = os.path.join(os.path.dirname(__file__), "xyz/hexene.xyz")
        mol.basis = "sto-3g"
        mol.charge = 1
        mol.spin = 1
        mol.build()
        self.molecular_unrestricted_oneshot_test(
            mol, 1, "Hexene Cation Frz (BE1)", True, -0.40383505
        )
        self.molecular_unrestricted_oneshot_test(
            mol, 2, "Hexene Cation Frz (BE2)", True, -0.36736494, delta=1e-4
        )
        """ Cut for expense
        self.molecular_unrestricted_oneshot_test(
            mol, 3, "Hexene Cation Frz (BE3)", True, -0.36996482
        )
        """

    def test_hexene_anion_sto3g_unfrz_ben(self):
        # Hexene anion without frozen core, STO-3G
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
            mol, 2, "Hexene Anion Unfrz (BE2)", False, -0.39052993, delta=1e-4
        )
        """ Cut for expense
        self.molecular_unrestricted_oneshot_test(
            mol, 3, "Hexene Anion Unfrz (BE3)", False, -0.3895924
        )
        """

    def test_hexene_cation_sto3g_unfrz_ben(self):
        # Hexene cation without frozen core, STO-3G
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
            mol, 2, "Hexene Cation Frz (BE2)", False, -0.39849056, delta=1e-4
        )
        """ Cut for expense
        self.molecular_unrestricted_oneshot_test(
            mol, 3, "Hexene Cation Frz (BE3)", False, -0.39729215
        )
        """

    def molecular_unrestricted_oneshot_test(
        self, mol, n_BE, test_name, frz, exp_result, delta=1e-5
    ):
        mf = scf.UHF(mol)
        mf.kernel()
        fobj = fragmentate(frag_type="chemgen", n_BE=n_BE, mol=mol, frozen_core=frz)
        mybe = UBE(mf, fobj, equal_bath=True)
        mybe.oneshot(solver="UCCSD", nproc=1)
        self.assertAlmostEqual(
            mybe.ebe_tot - mybe.hf_etot,
            exp_result,
            msg="Unrestricted One-Shot Energy for "
            + test_name
            + " is incorrect by"
            + str(mybe.ebe_tot - mybe.hf_etot - exp_result),
            delta=delta,
        )


if __name__ == "__main__":
    unittest.main()
