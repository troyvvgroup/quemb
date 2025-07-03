"""
Tests for the fragmentation modules.

Author(s): Shaun Weatherly
"""

import os
import unittest

from pyscf import gto, scf
import numpy as np

from quemb.molbe import BE, fragmentate
from ._expected_data_for_fragmentation_test import get_expected


class TestBE_Fragmentation(unittest.TestCase):
    def test_autogen_h_linear_be1(self):
        mol = gto.M()
        mol.atom = [["H", (0.0, 0.0, i)] for i in range(8)]
        mol.basis = "sto-3g"
        mol.charge = 0.0
        mol.spin = 0.0
        mol.build()

        mf = scf.RHF(mol)

        target = get_expected("test_autogen_h_linear_be1")

        self.run_indices_test(
            mf,
            1,
            "autogen_h_linear_be1",
            "autogen",
            target,
        )

    def test_autogen_h_linear_be2(self):
        mol = gto.M()
        mol.atom = [["H", (0.0, 0.0, i)] for i in range(8)]
        mol.basis = "sto-3g"
        mol.charge = 0.0
        mol.spin = 0.0
        mol.build()

        mf = scf.RHF(mol)

        target = get_expected("test_autogen_h_linear_be2")

        self.run_indices_test(
            mf,
            2,
            "autogen_h_linear_be2",
            "autogen",
            target,
        )

    def test_autogen_h_linear_be3(self):
        mol = gto.M()
        mol.atom = [["H", (0.0, 0.0, i)] for i in range(8)]
        mol.basis = "sto-3g"
        mol.charge = 0.0
        mol.spin = 0.0
        mol.build()

        mf = scf.RHF(mol)

        target = get_expected("test_autogen_h_linear_be3")

        self.run_indices_test(
            mf,
            3,
            "autogen_h_linear_be3",
            "autogen",
            target,
        )

    def test_autogen_octane_be1(self):
        mol = gto.M()
        mol.atom = os.path.join(os.path.dirname(__file__), "xyz/octane.xyz")
        mol.basis = "sto-3g"
        mol.charge = 0.0
        mol.spin = 0.0
        mol.build()

        mf = scf.RHF(mol)

        target = get_expected("test_autogen_octane_be1")

        self.run_indices_test(
            mf,
            1,
            "autogen_octane_be1",
            "autogen",
            target,
        )

    def test_autogen_octane_be2(self):
        mol = gto.M()
        mol.atom = os.path.join(os.path.dirname(__file__), "xyz/octane.xyz")
        mol.basis = "sto-3g"
        mol.charge = 0.0
        mol.spin = 0.0
        mol.build()

        mf = scf.RHF(mol)

        target = get_expected("test_autogen_octane_be2")

        self.run_indices_test(
            mf,
            2,
            "autogen_octane_be2",
            "autogen",
            target,
        )

    def test_autogen_octane_be3(self):
        mol = gto.M()
        mol.atom = os.path.join(os.path.dirname(__file__), "xyz/octane.xyz")
        mol.basis = "sto-3g"
        mol.charge = 0.0
        mol.spin = 0.0
        mol.build()

        mf = scf.RHF(mol)

        target = get_expected("test_autogen_octane_be3")

        self.run_indices_test(
            mf,
            3,
            "autogen_octane_be3",
            "autogen",
            target,
        )

    def test_graphgen_h_linear_be1(self):
        mol = gto.M()
        mol.atom = [["H", (0.0, 0.0, i)] for i in range(8)]
        mol.basis = "sto-3g"
        mol.charge = 0.0
        mol.spin = 0.0
        mol.build()

        mf = scf.RHF(mol)

        target = get_expected("test_graphgen_h_linear_be1")

        self.run_indices_test(
            mf,
            1,
            "graphgen_h_linear_be1",
            "graphgen",
            target,
        )

    def test_graphgen_h_linear_be2(self):
        mol = gto.M()
        mol.atom = [["H", (0.0, 0.0, i)] for i in range(8)]
        mol.basis = "sto-3g"
        mol.charge = 0.0
        mol.spin = 0.0
        mol.build()

        mf = scf.RHF(mol)

        target = get_expected("test_graphgen_h_linear_be2")

        self.run_indices_test(
            mf,
            2,
            "graphgen_h_linear_be2",
            "graphgen",
            target,
        )

    def test_graphgen_h_linear_be3(self):
        mol = gto.M()
        mol.atom = [["H", (0.0, 0.0, i)] for i in range(8)]
        mol.basis = "sto-3g"
        mol.charge = 0.0
        mol.spin = 0.0
        mol.build()

        mf = scf.RHF(mol)

        target = get_expected("test_graphgen_h_linear_be3")

        self.run_indices_test(
            mf,
            3,
            "graphgen_h_linear_be3",
            "graphgen",
            target,
        )

    def test_graphgen_octane_be1(self):
        mol = gto.M()
        mol.atom = os.path.join(os.path.dirname(__file__), "xyz/octane.xyz")
        mol.basis = "sto-3g"
        mol.charge = 0.0
        mol.spin = 0.0
        mol.build()

        mf = scf.RHF(mol)

        target = get_expected("test_graphgen_octane_be1")

        self.run_indices_test(
            mf,
            1,
            "graphgen_octane_be1",
            "graphgen",
            target,
        )

    def test_graphgen_octane_be2(self):
        mol = gto.M()
        mol.atom = os.path.join(os.path.dirname(__file__), "xyz/octane.xyz")
        mol.basis = "sto-3g"
        mol.charge = 0.0
        mol.spin = 0.0
        mol.build()

        mf = scf.RHF(mol)

        target = get_expected("test_graphgen_octane_be2")

        self.run_indices_test(
            mf,
            2,
            "graphgen_octane_be2",
            "graphgen",
            target,
        )

    def test_graphgen_octane_be3(self):
        mol = gto.M()
        mol.atom = os.path.join(os.path.dirname(__file__), "xyz/octane.xyz")
        mol.basis = "sto-3g"
        mol.charge = 0.0
        mol.spin = 0.0
        mol.build()

        mf = scf.RHF(mol)

        target = get_expected("test_graphgen_octane_be3")

        self.run_indices_test(
            mf,
            3,
            "graphgen_octane_be3",
            "graphgen",
            target,
        )

    def test_graphgen_autogen_h_linear_be2(self):
        mol = gto.M()
        mol.atom = [["H", (0.0, 0.0, i)] for i in range(8)]
        mol.basis = "sto-3g"
        mol.charge = 0.0
        mol.spin = 0.0
        mol.build()

        mf = scf.RHF(mol)
        mf.kernel()
        target = get_expected("test_graphgen_autogen_h_linear_be2")

        self.run_energies_test(
            mf,
            2,
            "energy_graphgen_autogen_h_linear_be2",
            target,
            delta=1e-2,
        )

    def test_graphgen_autogen_octane_be2(self):
        mol = gto.M()
        mol.atom = os.path.join(os.path.dirname(__file__), "xyz/octane.xyz")
        mol.basis = "sto-3g"
        mol.charge = 0.0
        mol.spin = 0.0
        mol.build()

        mf = scf.RHF(mol)
        mf.kernel()
        target = get_expected("test_graphgen_autogen_octane_be2")

        self.run_energies_test(
            mf,
            2,
            "energy_graphgen_autogen_octane_be2",
            target,
            delta=1e-2,
        )

    def test_shared_centers_autocratic_matching(self):
        mol = gto.M()
        mol.atom = os.path.join(
            os.path.dirname(__file__), "xyz/short_polypropylene.xyz"
        )
        mol.basis = "sto-3g"
        mol.build()
        mf = scf.RHF(mol)
        mf.kernel()

        fobj = fragmentate(
            mol, n_BE=2, frag_type="graphgen", print_frags=False, order_by_size=False
        )
        mybe = BE(mf, fobj)

        assert np.isclose(mf.e_tot, mybe.ebe_hf)

        fobj = fragmentate(
            mol, n_BE=3, frag_type="graphgen", print_frags=False, order_by_size=False
        )
        mybe = BE(mf, fobj)

        assert np.isclose(mf.e_tot, mybe.ebe_hf)

    def run_energies_test(
        self,
        mf,
        n_BE,
        test_name,
        target,
        delta,
    ):
        Es = {"target": target}
        for frag_type in ["autogen", "graphgen"]:
            fobj = fragmentate(
                frag_type=frag_type, n_BE=n_BE, mol=mf.mol, order_by_size=False
            )
            mbe = BE(mf, fobj)
            mbe.oneshot(solver="CCSD")
            Es.update({frag_type: mbe.ebe_tot - mbe.ebe_hf})

        for frag_type_A, E_A in Es.items():
            for frag_type_B, E_B in Es.items():
                self.assertAlmostEqual(
                    float(E_A),
                    float(E_B),
                    msg=f"{test_name}: BE Correlation Energy (oneshot) for "
                    + frag_type_A
                    + " does not match "
                    + frag_type_B
                    + f" ({E_A} != {E_B}) \n",
                    delta=delta,
                )

    def run_indices_test(
        self,
        mf,
        n_BE,
        test_name,
        frag_type,
        target,
    ):
        fobj = fragmentate(
            frag_type=frag_type, n_BE=n_BE, mol=mf.mol, order_by_size=False
        )
        print(fobj.frag_type)
        print("AO_per_frag", fobj.AO_per_frag)
        print("AO_per_edge_per_frag", fobj.AO_per_edge_per_frag)
        print("motifs_per_frag", fobj.motifs_per_frag)
        print("relAO_per_edge_per_frag", fobj.relAO_per_edge_per_frag)
        print("relAO_per_origin_per_frag", fobj.relAO_per_origin_per_frag)
        print("ref_frag_idx_per_edge_per_frag", fobj.ref_frag_idx_per_edge_per_frag)
        print("relAO_in_ref_per_edge_per_frag", fobj.relAO_in_ref_per_edge_per_frag)
        print("H_per_motif", fobj.H_per_motif)
        print("add_center_atom", fobj.add_center_atom)
        try:
            assert fobj.AO_per_frag == target["AO_per_frag"]
            assert fobj.AO_per_edge_per_frag == target["AO_per_edge_per_frag"]
            assert (
                fobj.ref_frag_idx_per_edge_per_frag
                == target["ref_frag_idx_per_edge_per_frag"]
            )
            assert fobj.relAO_per_origin_per_frag == target["relAO_per_origin_per_frag"]
            assert (
                fobj.weight_and_relAO_per_center_per_frag
                == target["weight_and_relAO_per_center_per_frag"]
            )
        except AssertionError as e:
            print(f"Fragmentation test failed at {test_name} \n")
            raise e


if __name__ == "__main__":
    unittest.main()
