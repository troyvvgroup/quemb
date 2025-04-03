"""
This script tests the correlation energies of sample restricted molecular
BE calculations from chemical potential matching

Author(s): Minsik Cho
"""

import os
import unittest

from pyscf import gto, scf

from quemb.molbe import BE, fragmentate
from quemb.molbe.fragment import ChemGenArgs


class TestBE_restricted(unittest.TestCase):
    def test_h8_sto3g_ben(self):
        # Linear Equidistant (r=1Å) H8 Chain, STO-3G
        # CCSD Total Energy: -4.306498896 Ha
        # Target BE Total energies from in-house code
        mol = gto.M()
        mol.atom = [["H", (0.0, 0.0, i)] for i in range(8)]
        mol.basis = "sto-3g"
        mol.charge = 0.0
        mol.spin = 0.0
        mol.build()
        mf = scf.RHF(mol)
        mf.kernel()
        self.molecular_restricted_test(
            mol,
            mf,
            "be2",
            "H8 (BE2)",
            "chemgen",
            -4.30628355,
            only_chem=True,
            additional_args=ChemGenArgs(treat_H_different=False),
        )
        self.molecular_restricted_test(
            mol,
            mf,
            "be3",
            "H8 (BE3)",
            "chemgen",
            -4.30649890,
            only_chem=True,
            additional_args=ChemGenArgs(treat_H_different=False),
        )

    def test_octane_sto3g_ben(self):
        # Octane, STO-3G
        # CCSD Total Energy: -310.3344616 Ha
        mol = gto.M()
        mol.atom = os.path.join(os.path.dirname(__file__), "xyz/octane.xyz")
        mol.basis = "sto-3g"
        mol.charge = 0.0
        mol.spin = 0.0
        mol.build()
        mf = scf.RHF(mol)
        mf.kernel()
        self.molecular_restricted_test(
            mol, mf, "be2", "Octane (BE2)", "autogen", -310.33471581, only_chem=True
        )
        self.molecular_restricted_test(
            mol, mf, "be3", "Octane (BE3)", "autogen", -310.33447096, only_chem=True
        )

    def molecular_restricted_test(
        self,
        mol,
        mf,
        be_type,
        test_name,
        frag_type,
        target,
        delta=1e-4,
        only_chem=True,
        additional_args=None,
    ):
        fobj = fragmentate(
            frag_type=frag_type,
            be_type=be_type,
            mol=mol,
            additional_args=additional_args,
        )
        mybe = BE(mf, fobj)
        mybe.optimize(solver="CCSD", method="QN", only_chem=only_chem)
        self.assertAlmostEqual(
            mybe.ebe_tot,
            target,
            msg="BE Correlation Energy (Chem. Pot. Optimization) for "
            + test_name
            + " does not match the expected correlation energy!",
            delta=delta,
        )


if __name__ == "__main__":
    unittest.main()
