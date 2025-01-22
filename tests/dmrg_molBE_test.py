"""
This script tests the QuEmb to block2 interface for performing ground state BE-DMRG.
Author(s): Shaun Weatherly
"""

import os
import unittest

from pathlib import Path
from pyscf import gto, scf

from quemb.molbe import BE, fragpart
from quemb.molbe.solver import DMRG_ArgsUser

try:
    from pyscf import dmrgscf
except ImportError:
    dmrgscf = None


class TestBE_DMRG(unittest.TestCase):
    @unittest.skipIf(
        dmrgscf is None,
        reason="Optional module 'dmrgscf' not imported correctly.",
    )
    def test_h8_pipek_interface(self):
        """ Test the QuEmb-to-block2 interface without calling block2."""
        mol = gto.M()
        mol.atom = [["H", (0.0, 0.0, i * 1.2)] for i in range(8)]
        mol.basis = "sto-3g"
        mol.charge = 0
        mol.spin = 0
        mol.build()
        self.molecular_DMRG_test(
            mol, True, "be1", "dmrg_h8_pipek_interface", "hchain_simple", -4.20236532,
        )

    @unittest.skipIf(
        os.getenv("GITHUB_ACTIONS") == "true",
        "This test cannot be run in github actions.",
    )
    def test_h8_pipek_full(self):
        """ Test the full QuEmb-to-block2 workflow; NOTE: requires block2."""
        mol = gto.M()
        mol.atom = [["H", (0.0, 0.0, i * 1.2)] for i in range(8)]
        mol.basis = "sto-3g"
        mol.charge = 0
        mol.spin = 0
        mol.build()
        self.molecular_DMRG_test(
            mol, False, "be1", "dmrg_h8_pipek_full", "hchain_simple", -4.20236532,
        )

    def molecular_DMRG_test(
        self, mol, force_earlystop, be_type, test_name, frag_type, target, delta=1e-4,
    ):
        scratch = Path.cwd() / "tests/data/molecular_DMRG_test/"
        mf = scf.RHF(mol)
        mf.kernel()
        fobj = fragpart(frag_type=frag_type, be_type=be_type, mol=mol)
        mybe = BE(
            mf,
            fobj,
            lo_method="pipek",
            pop_method="lowdin",
            scratch_dir=Path(scratch),
            cleanup_at_end=False,
        )
        mybe.oneshot(
            solver="block2",
            solver_args=DMRG_ArgsUser(
                maxM=100,
                max_iter=60,
                max_mem=3,
                force_earlystop=force_earlystop,
            ),
        )

        # First, verify the dmrg.conf files are generated correctly.
        for fdx, _ in enumerate(fobj.fsites):
            dname = "f" + str(fdx)
            frag_scratch = Path(scratch / dname)
            with open(frag_scratch / "dmrg.conf.target", 'r') as file:
                target_dmrg_conf = [f for f in file if not f.startswith("prefix")]
            with open(frag_scratch / "dmrg.conf", 'r') as file:
                current_dmrg_conf = [f for f in file if not f.startswith("prefix")]
            for (current_str, target_str) in zip(current_dmrg_conf, target_dmrg_conf):
                try:
                    assert current_str == target_str
                except AssertionError as e:
                    print(
                        f"ERROR: Inconsistent DMRG config files in frag '{dname}' - \n",
                        f"Expected: '{target_str}' \n",
                        f"Found: '{current_str}' \n",
                        flush=True,
                    )
                    raise e

        # Then verify the resulting energy is correct.
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
