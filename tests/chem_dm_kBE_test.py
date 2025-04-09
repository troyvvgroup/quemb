"""
This script tests the period BE1 and BE2 workflows using chemical potential
and density matching, respectively.
Also tests the gaussian density fitting interface, which is typically used by default.

Author(s): Shaun Weatherly
"""

import os
import unittest

from numpy import eye
from pyscf.pbc import df, gto, scf

from quemb.kbe import BE, fragmentate


class Test_kBE_Full(unittest.TestCase):
    def test_kc2_sto3g_be1_chempot(self) -> None:
        kpt = [1, 1, 1]
        cell = gto.Cell()

        a = 1.0
        b = 1.0
        c = 12.0

        lat = eye(3)
        lat[0, 0] = a
        lat[1, 1] = b
        lat[2, 2] = c

        cell.a = lat
        cell.atom = [["C", (0.0, 0.0, i * 3.0)] for i in range(2)]

        cell.unit = "Angstrom"
        cell.basis = "sto-3g"
        cell.verbose = 0
        cell.build()

        self.periodic_test(
            cell, kpt, "be1", "C2 (kBE1)", "autogen", -102.16547952, only_chem=True
        )
        self.periodic_test(
            cell, kpt, "be1", "C2 (kBE1)", "chemgen", -102.16547952, only_chem=True
        )

    @unittest.skipIf(
        os.getenv("QUEMB_SKIP_EXPENSIVE_TESTS") == "true",
        "Skipped expensive tests for QuEmb.",
    )
    def test_kc4_sto3g_be2_density(self) -> None:
        kpt = [1, 1, 1]
        cell = gto.Cell()

        a = 1.0
        b = 1.0
        c = 12.0

        lat = eye(3)
        lat[0, 0] = a
        lat[1, 1] = b
        lat[2, 2] = c

        cell.a = lat
        cell.atom = [["C", (0.0, 0.0, i * 3.0)] for i in range(4)]

        cell.unit = "Angstrom"
        cell.basis = "sto-3g"
        cell.verbose = 0
        cell.build()

        self.periodic_test(
            cell,
            kpt,
            "be2",
            "C4 (kBE2)",
            "autogen",
            -204.44557767,
            only_chem=False,
        )

    def test_kc4_sto3g_be2_mp2density(self) -> None:
        kpt = [1, 1, 2]
        cell = gto.Cell()

        a = 10.0
        b = 10.0
        c = 5.68

        lat = eye(3)
        lat[0, 0] = a
        lat[1, 1] = b
        lat[2, 2] = c

        cell.a = lat
        cell.atom = [["C", (0.0, 0.0, i * 1.42)] for i in range(4)]

        cell.unit = "Angstrom"
        cell.basis = "sto-3g"
        cell.verbose = 0
        cell.build()

        self.periodic_test(
            cell,
            kpt,
            "be2",
            "C4 (kBE2, MP2/frozen core)",
            "autogen",
            -120.87412293,
            solver="MP2",
            only_chem=False,
            frozen_core=True,
        )

    def periodic_test(
        self,
        cell,
        kpt,
        be_type,
        test_name,
        frag_type,
        target,
        delta=1e-4,
        solver="CCSD",
        only_chem=True,
        frozen_core=False,
    ) -> None:
        kpts = cell.make_kpts(kpt, wrap_around=True)
        mydf = df.GDF(cell, kpts)
        mydf.build()

        kmf = scf.KRHF(cell, kpts)
        kmf.with_df = mydf
        kmf.exxdiv = None
        kmf.conv_tol = 1e-12
        kmf.kernel()

        kfrag = fragmentate(
            be_type=be_type,
            mol=cell,
            frag_type=frag_type,
            kpt=kpt,
            frozen_core=frozen_core,
        )
        mykbe = BE(kmf, kfrag, kpts=kpts, exxdiv=None)
        mykbe.optimize(solver=solver, only_chem=only_chem)

        self.assertAlmostEqual(
            mykbe.ebe_tot,
            target,
            msg="kBE Correlation Energy for "
            + test_name
            + " does not match the expected correlation energy!",
            delta=delta,
        )


if __name__ == "__main__":
    unittest.main()
