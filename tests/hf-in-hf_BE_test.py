"""
This script tests the HF-in-HF energies of sample restricted molecular and periodic systems

Author(s): Minsik Cho
"""

import os, unittest
from pyscf import gto, scf
from molbe import fragpart, BE


class TestHFinHF_restricted(unittest.TestCase):
    def test_h8_sto3g_ben(self):
        # Linear Equidistant (r=1Å) H8 Chain, STO-3G
        mol = gto.M()
        mol.atom = [["H", (0.0, 0.0, i)] for i in range(8)]
        mol.basis = "sto-3g"
        mol.charge = 0.0
        mol.spin = 0.0
        mol.build()
        self.molecular_restricted_test(mol, "be1", "H8 (BE1)")
        self.molecular_restricted_test(mol, "be2", "H8 (BE2)")
        self.molecular_restricted_test(mol, "be3", "H8 (BE3)")

    def test_h8_ccpvdz_ben(self):
        # Linear Equidistant (r=1Å) H8 Chain, cc-pVDZ
        mol = gto.M()
        mol.atom = [["H", (0.0, 0.0, i)] for i in range(8)]
        mol.basis = "cc-pvdz"
        mol.charge = 0.0
        mol.spin = 0.0
        mol.build()
        self.molecular_restricted_test(mol, "be1", "H8 (BE1)")
        self.molecular_restricted_test(mol, "be2", "H8 (BE2)")
        self.molecular_restricted_test(mol, "be3", "H8 (BE3)")

    def test_octane_sto3g_ben(self):
        # Octane, STO-3G
        mol = gto.M()
        mol.atom = os.path.join(os.path.dirname(__file__), "xyz/octane.xyz")
        mol.basis = "sto-3g"
        mol.charge = 0.0
        mol.spin = 0.0
        mol.build()
        self.molecular_restricted_test(mol, "be1", "Octane (BE1)")
        self.molecular_restricted_test(mol, "be2", "Octane (BE2)")
        self.molecular_restricted_test(mol, "be3", "Octane (BE3)")

    def molecular_restricted_test(self, mol, be_type, test_name, delta=1e-5):
        mf = scf.RHF(mol)
        mf.kernel()
        fobj = fragpart(frag_type="autogen", be_type=be_type, mol=mol)
        mybe = BE(mf, fobj)
        self.assertAlmostEqual(
            mybe.ebe_hf,
            mf.e_tot,
            msg="HF-in-HF energy for " + test_name + " does not match the HF energy!",
            delta=delta,
        )


if __name__ == "__main__":
    unittest.main()
