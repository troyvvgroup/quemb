"""
This script tests the period BE1 and BE2 workflows using chemical potential
and density matching, respectively.
Also tests the gaussian density fitting interface, which is typically used by default.

Author(s): Shaun Weatherly
"""

import unittest

import numpy
from pyscf.pbc import df, gto, scf

from kbe import BE, fragpart


class Test_kBE_Full(unittest.TestCase):
    try:
        import libdmet
    except ImportError:
        libdmet = None

    @unittest.skipIf(libdmet is None, "Module `libdmet` not imported correctly.")
    def test_kc2_sto3g_be1_chempot(self):
        kpt = [1, 1, 1]
        cell = gto.Cell()

        a = 1.0
        b = 1.0
        c = 12.0

        lat = numpy.eye(3)
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
            cell, kpt, "be1", "C2 (kBE1)", "autogen", -74.64695833012868, only_chem=True
        )

    def test_kc4_sto3g_be2_density(self):
        kpt = [1, 1, 1]
        cell = gto.Cell()

        a = 1.0
        b = 1.0
        c = 12.0

        lat = numpy.eye(3)
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
            -149.4085332249809,
            only_chem=False,
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
        only_chem=True,
    ):
        kpts = cell.make_kpts(kpt, wrap_around=True)
        mydf = df.GDF(cell, kpts)
        mydf.build()

        kmf = scf.KRHF(cell, kpts)
        kmf.with_df = mydf
        kmf.exxdiv = None
        kmf.conv_tol = 1e-12
        kmf.kernel()

        kfrag = fragpart(
            be_type=be_type, mol=cell, frag_type=frag_type, kpt=kpt, frozen_core=True
        )
        mykbe = BE(kmf, kfrag, kpts=kpts)
        mykbe.optimize(solver="CCSD", only_chem=only_chem)

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
