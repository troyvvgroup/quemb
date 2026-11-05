import os
import unittest

import numpy
from pyscf.gto import Mole
from pyscf.scf.hf import RHF

from quemb.molbe import BE, fragmentate
from quemb.molbe.chemfrag import ChemGenArgs


class Test_Num_Jac(unittest.TestCase):
    def build_BE_object(self, mol, n_BE, h_treatment="treat_H_diff"):
        mf = RHF(mol)
        mf.kernel()
        fobj = fragmentate(
            mol,
            n_BE=n_BE,
            frag_type="chemgen",
            additional_args=ChemGenArgs(h_treatment=h_treatment),
        )
        return BE(mf, fobj, nproc=48, ompnum=8)

    @unittest.skipUnless(
        os.getenv("QUEMB_DO_EXPENSIVE_TESTS") == "true",
        "Skipped expensive tests for QuEmb.",
    )
    def test_numerical_jacobian_octane(self):
        mol = Mole()
        mol.atom = "./xyz/distorted_octane.xyz"
        mol.basis = "sto-3g"
        mol.build()
        beobj = self.build_BE_object(mol, n_BE=2)

        beobj.optimize(
            solver="MP2", only_chem=False, jac_solver="Numerical", nproc=48, ompnum=8
        )
        analytical_ebe = beobj.ebe_tot

        beobj.optimize(
            solver="MP2", only_chem=False, jac_solver="HF", nproc=48, ompnum=8
        )
        numerical_ebe = beobj.ebe_tot

        self.assertTrue(numpy.allclose(analytical_ebe, numerical_ebe, atol=1e-5))

    def test_numerical_jacobian_h8(self):
        mol = Mole()
        mol.atom = [["H", (0.0, 0.0, i)] for i in range(7)]
        mol.atom.append(["H", (0.0, 0.0, 4.2)])
        mol.basis = "sto-3g"
        mol.build()
        beobj = self.build_BE_object(mol, n_BE=2, h_treatment="treat_H_like_heavy_atom")

        beobj.optimize(
            solver="CCSD", only_chem=False, jac_solver="Numerical", nproc=48, ompnum=8
        )
        analytical_ebe = beobj.ebe_tot

        beobj.optimize(
            solver="CCSD", only_chem=False, jac_solver="HF", nproc=48, ompnum=8
        )
        numerical_ebe = beobj.ebe_tot

        self.assertTrue(numpy.allclose(analytical_ebe, numerical_ebe, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
