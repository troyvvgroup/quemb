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
from quemb.molbe.ube import _opposite_spin_eri_supported


class TestOneShot_Unrestricted(unittest.TestCase):
    # Test A

    @unittest.skipUnless(
        os.getenv("QUEMB_DO_KNOWN_TO_FAIL_TESTS") == "true",
        "This test is known to fail.",
    )
    def test_hexene_anion_sto3g_frz_ben_autogen(self):
        # Hexene anion with frozen core, STO-3G
        mol = gto.M()
        mol.atom = os.path.join(os.path.dirname(__file__), "xyz/hexene.xyz")
        mol.basis = "sto-3g"
        mol.charge = -1
        mol.spin = 1
        mol.build()
        self.molecular_unrestricted_oneshot_test(
            mol,
            1,
            "Hexene Anion Frz (BE1)",
            True,
            -0.35753374,
            delta=1e-4,
            frag_type="autogen",
            equal_bath=False,
        )

    @unittest.skipUnless(
        os.getenv("QUEMB_DO_KNOWN_TO_FAIL_TESTS") == "true"
        and _opposite_spin_eri_supported(),
        "This test is known to fail and/or requires custom-compiled PySCF.",
    )
    def test_hexene_anion_sto3g_frz_ben_autogen_be2(self):
        mol = gto.M()
        mol.atom = os.path.join(os.path.dirname(__file__), "xyz/hexene.xyz")
        mol.basis = "sto-3g"
        mol.charge = -1
        mol.spin = 1
        mol.build()
        self.molecular_unrestricted_oneshot_test(
            mol,
            2,
            "Hexene Anion Frz (BE2)",
            True,
            -0.34725961,
            delta=1e-4,
            frag_type="autogen",
            equal_bath=False,
        )

    @unittest.skipUnless(
        os.getenv("QUEMB_DO_KNOWN_TO_FAIL_TESTS") == "true"
        and os.getenv("QUEMB_DO_EXPENSIVE_TESTS") == "true"
        and _opposite_spin_eri_supported(),
        "This test is known to fail, expensive, and/or requires custom-compiled PySCF.",
    )
    def test_hexene_anion_sto3g_frz_ben_autogen_be3(self):
        mol = gto.M()
        mol.atom = os.path.join(os.path.dirname(__file__), "xyz/hexene.xyz")
        mol.basis = "sto-3g"
        mol.charge = -1
        mol.spin = 1
        mol.build()
        self.molecular_unrestricted_oneshot_test(
            mol,
            3,
            "Hexene Anion Frz (BE3)",
            True,
            -0.34300834,
            delta=1e-4,
            frag_type="autogen",
            equal_bath=False,
        )

    @unittest.skipUnless(
        os.getenv("QUEMB_DO_KNOWN_TO_FAIL_TESTS") == "true",
        "This test is known to fail.",
    )
    def test_hexene_cation_sto3g_frz_ben_autogen(self):
        # Hexene cation with frozen core, STO-3G
        mol = gto.M()
        mol.atom = os.path.join(os.path.dirname(__file__), "xyz/hexene.xyz")
        mol.basis = "sto-3g"
        mol.charge = 1
        mol.spin = 1
        mol.build()
        self.molecular_unrestricted_oneshot_test(
            mol,
            1,
            "Hexene Cation Frz (BE1)",
            True,
            -0.40383508,
            delta=1e-4,
            frag_type="autogen",
            equal_bath=False,
        )

    @unittest.skipUnless(
        os.getenv("QUEMB_DO_KNOWN_TO_FAIL_TESTS") == "true"
        and _opposite_spin_eri_supported(),
        "This test is known to fail and/or requires custom-compiled PySCF.",
    )
    def test_hexene_cation_sto3g_frz_ben_autogen_be2(self):
        mol = gto.M()
        mol.atom = os.path.join(os.path.dirname(__file__), "xyz/hexene.xyz")
        mol.basis = "sto-3g"
        mol.charge = 1
        mol.spin = 1
        mol.build()
        self.molecular_unrestricted_oneshot_test(
            mol,
            2,
            "Hexene Cation Frz (BE2)",
            True,
            -0.36496690,
            delta=1e-4,
            frag_type="autogen",
            equal_bath=False,
        )

    @unittest.skipUnless(
        os.getenv("QUEMB_DO_KNOWN_TO_FAIL_TESTS") == "true"
        and os.getenv("QUEMB_DO_EXPENSIVE_TESTS") == "true"
        and _opposite_spin_eri_supported(),
        "This test is known to fail, expensive, and/or requires custom-compiled PySCF.",
    )
    def test_hexene_cation_sto3g_frz_ben_autogen_be3(self):
        mol = gto.M()
        mol.atom = os.path.join(os.path.dirname(__file__), "xyz/hexene.xyz")
        mol.basis = "sto-3g"
        mol.charge = 1
        mol.spin = 1
        mol.build()
        self.molecular_unrestricted_oneshot_test(
            mol,
            3,
            "Hexene Cation Frz (BE3)",
            True,
            -0.36996484,
            delta=1e-4,
            frag_type="autogen",
            equal_bath=False,
        )

    @unittest.skipUnless(
        os.getenv("QUEMB_DO_KNOWN_TO_FAIL_TESTS") == "true",
        "This test is known to fail.",
    )
    def test_hexene_anion_sto3g_unfrz_ben_autogen(self):
        # Hexene anion without frozen core, STO-3G
        mol = gto.M()
        mol.atom = os.path.join(os.path.dirname(__file__), "xyz/hexene.xyz")
        mol.basis = "sto-3g"
        mol.charge = -1
        mol.spin = 1
        mol.build()
        self.molecular_unrestricted_oneshot_test(
            mol,
            1,
            "Hexene Anion Unfrz (BE1)",
            False,
            -0.38478279,
            delta=1e-4,
            frag_type="autogen",
            equal_bath=False,
        )

    @unittest.skipUnless(
        os.getenv("QUEMB_DO_KNOWN_TO_FAIL_TESTS") == "true"
        and _opposite_spin_eri_supported(),
        "This test is known to fail and/or requires custom-compiled PySCF.",
    )
    def test_hexene_anion_sto3g_unfrz_ben_autogen_be2(self):
        mol = gto.M()
        mol.atom = os.path.join(os.path.dirname(__file__), "xyz/hexene.xyz")
        mol.basis = "sto-3g"
        mol.charge = -1
        mol.spin = 1
        mol.build()
        self.molecular_unrestricted_oneshot_test(
            mol,
            2,
            "Hexene Anion Unfrz (BE2)",
            False,
            -0.39053689,
            delta=1e-4,
            frag_type="autogen",
            equal_bath=False,
        )

    @unittest.skipUnless(
        os.getenv("QUEMB_DO_KNOWN_TO_FAIL_TESTS") == "true"
        and os.getenv("QUEMB_DO_EXPENSIVE_TESTS") == "true"
        and _opposite_spin_eri_supported(),
        "This test is known to fail, expensive, and/or requires custom-compiled PySCF.",
    )
    def test_hexene_anion_sto3g_unfrz_ben_autogen_be3(self):
        mol = gto.M()
        mol.atom = os.path.join(os.path.dirname(__file__), "xyz/hexene.xyz")
        mol.basis = "sto-3g"
        mol.charge = -1
        mol.spin = 1
        mol.build()
        self.molecular_unrestricted_oneshot_test(
            mol,
            3,
            "Hexene Anion Unfrz (BE3)",
            False,
            -0.38960174,
            delta=1e-4,
            frag_type="autogen",
            equal_bath=False,
        )

    @unittest.skipUnless(
        os.getenv("QUEMB_DO_KNOWN_TO_FAIL_TESTS") == "true",
        "This test is known to fail.",
    )
    def test_hexene_cation_sto3g_unfrz_ben_autogen(self):
        mol = gto.M()
        mol.atom = os.path.join(os.path.dirname(__file__), "xyz/hexene.xyz")
        mol.basis = "sto-3g"
        mol.charge = 1
        mol.spin = 1
        mol.build()
        self.molecular_unrestricted_oneshot_test(
            mol,
            1,
            "Hexene Cation Frz (BE1)",
            False,
            -0.39471433,
            delta=1e-4,
            frag_type="autogen",
            equal_bath=False,
        )

    @unittest.skipUnless(
        os.getenv("QUEMB_DO_KNOWN_TO_FAIL_TESTS") == "true"
        and _opposite_spin_eri_supported(),
        "This test is known to fail and/or requires custom-compiled PySCF.",
    )
    def test_hexene_cation_sto3g_unfrz_ben_autogen_be2(self):
        mol = gto.M()
        mol.atom = os.path.join(os.path.dirname(__file__), "xyz/hexene.xyz")
        mol.basis = "sto-3g"
        mol.charge = 1
        mol.spin = 1
        mol.build()
        self.molecular_unrestricted_oneshot_test(
            mol,
            2,
            "Hexene Cation Frz (BE2)",
            False,
            -0.39846777,
            delta=1e-4,
            frag_type="autogen",
            equal_bath=False,
        )

    @unittest.skipUnless(
        os.getenv("QUEMB_DO_KNOWN_TO_FAIL_TESTS") == "true"
        and os.getenv("QUEMB_DO_EXPENSIVE_TESTS") == "true"
        and _opposite_spin_eri_supported(),
        "This test is known to fail, expensive, and/or requires custom-compiled PySCF.",
    )
    def test_hexene_cation_sto3g_unfrz_ben_autogen_be3(self):
        mol = gto.M()
        mol.atom = os.path.join(os.path.dirname(__file__), "xyz/hexene.xyz")
        mol.basis = "sto-3g"
        mol.charge = 1
        mol.spin = 1
        mol.build()
        self.molecular_unrestricted_oneshot_test(
            mol,
            3,
            "Hexene Cation Frz (BE3)",
            False,
            -0.39729184,
            delta=1e-4,
            frag_type="autogen",
            equal_bath=False,
        )

    # Test B

    @unittest.skipUnless(
        os.getenv("QUEMB_DO_KNOWN_TO_FAIL_TESTS") == "true",
        "BE2 bath selection is not deterministic across Python/numpy versions with frozen core",
    )
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

    @unittest.skipUnless(
        os.getenv("QUEMB_DO_KNOWN_TO_FAIL_TESTS") == "true"
        and os.getenv("QUEMB_DO_EXPENSIVE_TESTS") == "true",
        "This test is known to fail and/or expensive.",
    )
    def test_hexene_anion_sto3g_frz_ben_be3(self):
        # Hexene anion with frozen core, STO-3G
        mol = gto.M()
        mol.atom = os.path.join(os.path.dirname(__file__), "xyz/hexene.xyz")
        mol.basis = "sto-3g"
        mol.charge = -1
        mol.spin = 1
        mol.build()
        self.molecular_unrestricted_oneshot_test(
            mol, 3, "Hexene Anion Frz (BE3)", True, -0.34300832
        )

    @unittest.skipUnless(
        os.getenv("QUEMB_DO_KNOWN_TO_FAIL_TESTS") == "true",
        "This test is known to fail.",
    )
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

    @unittest.skipUnless(
        os.getenv("QUEMB_DO_KNOWN_TO_FAIL_TESTS") == "true"
        and os.getenv("QUEMB_DO_EXPENSIVE_TESTS") == "true",
        "This test is known to fail and/or expensive.",
    )
    def test_hexene_cation_sto3g_frz_ben_be3(self):
        # Hexene cation with frozen core, STO-3G
        mol = gto.M()
        mol.atom = os.path.join(os.path.dirname(__file__), "xyz/hexene.xyz")
        mol.basis = "sto-3g"
        mol.charge = 1
        mol.spin = 1
        mol.build()
        self.molecular_unrestricted_oneshot_test(
            mol, 3, "Hexene Cation Frz (BE3)", True, -0.36996482
        )

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

    @unittest.skipUnless(
        os.getenv("QUEMB_DO_EXPENSIVE_TESTS") == "true",
        "Skipped expensive tests for QuEmb.",
    )
    def test_hexene_anion_sto3g_unfrz_ben_be3(self):
        # Hexene anion without frozen core, STO-3G
        mol = gto.M()
        mol.atom = os.path.join(os.path.dirname(__file__), "xyz/hexene.xyz")
        mol.basis = "sto-3g"
        mol.charge = -1
        mol.spin = 1
        mol.build()
        self.molecular_unrestricted_oneshot_test(
            mol, 3, "Hexene Anion Unfrz (BE3)", False, -0.3895924
        )

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

    @unittest.skipUnless(
        os.getenv("QUEMB_DO_EXPENSIVE_TESTS") == "true",
        "Skipped expensive tests for QuEmb.",
    )
    def test_hexene_cation_sto3g_unfrz_ben_be3(self):
        # Hexene cation without frozen core, STO-3G
        mol = gto.M()
        mol.atom = os.path.join(os.path.dirname(__file__), "xyz/hexene.xyz")
        mol.basis = "sto-3g"
        mol.charge = 1
        mol.spin = 1
        mol.build()
        self.molecular_unrestricted_oneshot_test(
            mol, 3, "Hexene Cation Frz (BE3)", False, -0.39729215
        )

    def molecular_unrestricted_oneshot_test(
        self,
        mol,
        n_BE,
        test_name,
        frz,
        exp_result,
        delta=1e-5,
        frag_type="autogen",
        equal_bath=True,
    ):
        mf = scf.UHF(mol)
        mf.kernel()
        fobj = fragmentate(frag_type=frag_type, n_BE=n_BE, mol=mol, frozen_core=frz)
        mybe = UBE(mf, fobj, equal_bath=equal_bath)
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
