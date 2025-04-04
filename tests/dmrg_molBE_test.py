"""
This script tests the QuEmb to block2 interface for performing ground state BE-DMRG.
Author(s): Shaun Weatherly
"""

import os
import tempfile
import unittest

from pyscf import gto, scf

from quemb.molbe import BE, fragmentate

try:
    from pyscf import dmrgscf  # type: ignore[attr-defined]
except ImportError:
    dmrgscf = None


class TestBE_DMRG(unittest.TestCase):
    @unittest.skipIf(
        not os.getenv("QUEMB_DO_KNOWN_TO_FAIL_TESTS") == "true",
        reason="This test is known to fail.",
    )
    def test_h8_sto3g_pipek(self):
        mol = gto.M()
        mol.atom = [["H", (0.0, 0.0, i * 1.2)] for i in range(8)]
        mol.basis = "sto-3g"
        mol.charge = 0
        mol.spin = 0
        mol.build()
        self.molecular_DMRG_test(mol, 1, 100, "H8 (BE1)", "hchain_simple", -4.20236532)

    def molecular_DMRG_test(
        self, mol, n_BE, maxM, test_name, frag_type, target, delta=1e-4
    ):
        with tempfile.TemporaryDirectory() as tmp:
            mf = scf.RHF(mol)
            mf.kernel()
            fobj = fragmentate(frag_type=frag_type, n_BE=n_BE, mol=mol)
            mybe = BE(mf, fobj, lo_method="pipek", pop_method="lowdin")
            mybe.oneshot(
                solver="block2",
                scratch_dir=str(tmp),
                maxM=int(maxM),
                maxIter=30,
                force_cleanup=True,
            )
            self.assertAlmostEqual(
                mybe.ebe_tot,
                target,
                msg="BE Total Energy (Chem. Pot. Optimization) for "
                + test_name
                + " does not match the expected total energy!",
                delta=delta,
            )
            tmpfiles = []
            for path, dir_n, file_n in os.walk(tmp, topdown=True):
                for n in file_n:
                    if (
                        n.startswith("F.")
                        or n.startswith("FCIDUMP")
                        or n.startswith("node")
                    ):
                        tmpfiles.append(n)
            try:
                assert len(tmpfiles) == 0
            except Exception as inst:
                print(
                    f"DMRG tempfiles not cleared correctly in {test_name}:\n",
                    inst,
                    tmpfiles,
                )


if __name__ == "__main__":
    unittest.main()
