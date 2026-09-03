"""
Tests for full iterative UBE (Eq. 15 local/edge density matching, on top
of Eq. 16 chemical-potential matching) restricted to common_bath.

H8 chain, STO-3G: cheap and converges reliably, so it's used for the
regression energy check. Reference value obtained by running this exact
calculation once with the implementation in place.

A closed-shell cross-check against restricted BE's own iterative
optimizer (tests/molbe_h8_test.py's test_BE_density_matching, BE2,
only_chem=False, FCI: -0.1343036698277933) confirms this is in the right
regime: UBE/UCCSD lands at -0.13221835 Ha, a ~2 mHa gap consistent with
CCSD vs. FCI correlation energy on this system -- not a discrepancy in
the embedding math itself.

Author(s): Leah Weisburn
"""

import os
import unittest

os.environ["OMP_NUM_THREADS"] = "1"

from pyscf import gto, scf

from quemb.molbe import UBE, fragmentate
from quemb.molbe.fragment import ChemGenArgs


def h8_mol(charge=0, spin=0):
    mol = gto.M()
    mol.atom = """
        H 0. 0. 0.
        H 0. 0. 1.
        H 0. 0. 2.
        H 0. 0. 3.
        H 0. 0. 4.
        H 0. 0. 5.
        H 0. 0. 6.
        H 0. 0. 7.
        """
    mol.basis = "sto-3g"
    mol.charge = charge
    mol.spin = spin
    mol.build()
    return mol


class TestIterative_Unrestricted(unittest.TestCase):
    def test_h8_common_bath_iterative(self):
        # Full iterative UBE (Eq. 15 + Eq. 16), common_bath, BE2, closed
        # shell H8. Converges to RMS error < 1e-6 well within max_iter.
        mol = h8_mol()
        mf = scf.UHF(mol)
        mf.conv_tol = 1e-12
        mf.kernel()
        fobj = fragmentate(
            n_BE=2,
            frag_type="chemgen",
            mol=mol,
            additional_args=ChemGenArgs(treat_H_different=False),
        )
        mybe = UBE(mf, fobj, common_bath=True)
        mybe.optimize(solver="UCCSD", only_chem=False, max_iter=100, conv_tol=1e-7)
        self.assertAlmostEqual(
            mybe.ebe_tot - mybe.hf_etot,
            -0.13221835,
            delta=1e-4,
        )

    def test_only_chem_false_requires_common_bath(self):
        mol = h8_mol()
        mf = scf.UHF(mol)
        mf.kernel()
        fobj = fragmentate(
            n_BE=2,
            frag_type="chemgen",
            mol=mol,
            additional_args=ChemGenArgs(treat_H_different=False),
        )
        mybe = UBE(mf, fobj, equal_bath=True)
        with self.assertRaises(ValueError):
            mybe.optimize(solver="UCCSD", only_chem=False, max_iter=1)

    def test_only_chem_false_requires_be2_plus(self):
        mol = h8_mol()
        mf = scf.UHF(mol)
        mf.kernel()
        fobj = fragmentate(
            n_BE=1,
            frag_type="chemgen",
            mol=mol,
            additional_args=ChemGenArgs(treat_H_different=False),
        )
        mybe = UBE(mf, fobj, common_bath=True)
        with self.assertRaises(ValueError):
            mybe.optimize(solver="UCCSD", only_chem=False, max_iter=1)


if __name__ == "__main__":
    unittest.main()
