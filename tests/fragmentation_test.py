"""
Tests for the fragmentation modules.

Author(s): Shaun Weatherly
"""

import os
import unittest

from pyscf import gto, scf
import numpy as np

from quemb.molbe import BE, fragmentate


class TestBE_Fragmentation(unittest.TestCase):
    def test_autogen_h_linear_be1(self):
        mol = gto.M()
        mol.atom = [["H", (0.0, 0.0, i)] for i in range(8)]
        mol.basis = "sto-3g"
        mol.charge = 0.0
        mol.spin = 0.0
        mol.build()

        mf = scf.RHF(mol)

        target = {
            "AO_per_frag": [[0], [1], [2], [3], [4], [5], [6], [7]],
            "edge": [],
            "center": [],
            "centerf_idx": [[0], [0], [0], [0], [0], [0], [0], [0]],
            "rel_AO_per_center_per_frag": [
                [1.0, [0]],
                [1.0, [0]],
                [1.0, [0]],
                [1.0, [0]],
                [1.0, [0]],
                [1.0, [0]],
                [1.0, [0]],
                [1.0, [0]],
            ],
        }

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

        target = {
            "AO_per_frag": [
                [1, 0, 2],
                [2, 1, 3],
                [3, 2, 4],
                [4, 3, 5],
                [5, 4, 6],
                [6, 7, 5],
            ],
            "edge": [[[2]], [[1], [3]], [[2], [4]], [[3], [5]], [[4], [6]], [[5]]],
            "center": [[1], [0, 2], [1, 3], [2, 4], [3, 5], [4]],
            "centerf_idx": [[0], [0], [0], [0], [0], [0]],
            "rel_AO_per_center_per_frag": [
                [1.0, [0, 1]],
                [1.0, [0]],
                [1.0, [0]],
                [1.0, [0]],
                [1.0, [0]],
                [1.0, [0, 1]],
            ],
        }

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

        target = {
            "AO_per_frag": [
                [2, 0, 1, 3, 4],
                [3, 2, 1, 4, 5],
                [4, 3, 2, 5, 6],
                [5, 6, 7, 4, 3],
            ],
            "edge": [
                [[3], [4]],
                [[2], [1], [4], [5]],
                [[3], [2], [5], [6]],
                [[4], [3]],
            ],
            "center": [[1, 2], [0, 0, 2, 3], [1, 0, 3, 3], [2, 1]],
            "centerf_idx": [[0], [0], [0], [0]],
            "rel_AO_per_center_per_frag": [
                [1.0, [0, 1, 2]],
                [1.0, [0]],
                [1.0, [0]],
                [1.0, [0, 1, 2]],
            ],
        }

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

        target = {
            "AO_per_frag": [
                [0, 1, 2, 3, 4, 11, 13],
                [5, 6, 7, 8, 9, 10, 12],
                [14, 15, 16, 17, 18, 24, 26],
                [19, 20, 21, 22, 23, 25, 27],
                [28, 29, 30, 31, 32, 38, 40],
                [33, 34, 35, 36, 37, 39, 41],
                [42, 43, 44, 45, 46, 52, 54, 57],
                [47, 48, 49, 50, 51, 53, 55, 56],
            ],
            "edge": [],
            "center": [],
            "centerf_idx": [
                [0, 1, 2, 3, 4, 5, 6],
                [0, 1, 2, 3, 4, 5, 6],
                [0, 1, 2, 3, 4, 5, 6],
                [0, 1, 2, 3, 4, 5, 6],
                [0, 1, 2, 3, 4, 5, 6],
                [0, 1, 2, 3, 4, 5, 6],
                [0, 1, 2, 3, 4, 5, 6, 7],
                [0, 1, 2, 3, 4, 5, 6, 7],
            ],
            "rel_AO_per_center_per_frag": [
                [1.0, [0, 1, 2, 3, 4, 5, 6]],
                [1.0, [0, 1, 2, 3, 4, 5, 6]],
                [1.0, [0, 1, 2, 3, 4, 5, 6]],
                [1.0, [0, 1, 2, 3, 4, 5, 6]],
                [1.0, [0, 1, 2, 3, 4, 5, 6]],
                [1.0, [0, 1, 2, 3, 4, 5, 6]],
                [1.0, [0, 1, 2, 3, 4, 5, 6, 7]],
                [1.0, [0, 1, 2, 3, 4, 5, 6, 7]],
            ],
        }

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

        target = {
            "AO_per_frag": [
                [
                    0,
                    1,
                    2,
                    3,
                    4,
                    11,
                    13,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    12,
                    19,
                    20,
                    21,
                    22,
                    23,
                    25,
                    27,
                ],
                [
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    12,
                    0,
                    1,
                    2,
                    3,
                    4,
                    11,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    24,
                    26,
                ],
                [
                    14,
                    15,
                    16,
                    17,
                    18,
                    24,
                    26,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    12,
                    28,
                    29,
                    30,
                    31,
                    32,
                    38,
                    40,
                ],
                [
                    19,
                    20,
                    21,
                    22,
                    23,
                    25,
                    27,
                    0,
                    1,
                    2,
                    3,
                    4,
                    11,
                    13,
                    33,
                    34,
                    35,
                    36,
                    37,
                    39,
                    41,
                ],
                [
                    28,
                    29,
                    30,
                    31,
                    32,
                    38,
                    40,
                    42,
                    43,
                    44,
                    45,
                    46,
                    52,
                    54,
                    57,
                    14,
                    15,
                    16,
                    17,
                    18,
                    24,
                    26,
                ],
                [
                    33,
                    34,
                    35,
                    36,
                    37,
                    39,
                    41,
                    47,
                    48,
                    49,
                    50,
                    51,
                    53,
                    55,
                    56,
                    19,
                    20,
                    21,
                    22,
                    23,
                    25,
                    27,
                ],
            ],
            "edge": [
                [[5, 6, 7, 8, 9, 10, 12], [19, 20, 21, 22, 23, 25, 27]],
                [[0, 1, 2, 3, 4, 11, 13], [14, 15, 16, 17, 18, 24, 26]],
                [[5, 6, 7, 8, 9, 10, 12], [28, 29, 30, 31, 32, 38, 40]],
                [[0, 1, 2, 3, 4, 11, 13], [33, 34, 35, 36, 37, 39, 41]],
                [[14, 15, 16, 17, 18, 24, 26]],
                [[19, 20, 21, 22, 23, 25, 27]],
            ],
            "center": [[1, 3], [0, 2], [1, 4], [0, 5], [2], [3]],
            "centerf_idx": [
                [0, 1, 2, 3, 4, 5, 6],
                [0, 1, 2, 3, 4, 5, 6],
                [0, 1, 2, 3, 4, 5, 6],
                [0, 1, 2, 3, 4, 5, 6],
                [0, 1, 2, 3, 4, 5, 6],
                [0, 1, 2, 3, 4, 5, 6],
            ],
            "rel_AO_per_center_per_frag": [
                [1.0, [0, 1, 2, 3, 4, 5, 6]],
                [1.0, [0, 1, 2, 3, 4, 5, 6]],
                [1.0, [0, 1, 2, 3, 4, 5, 6]],
                [1.0, [0, 1, 2, 3, 4, 5, 6]],
                [1.0, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]],
                [1.0, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]],
            ],
        }

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

        target = {
            "AO_per_frag": [
                [
                    0,
                    1,
                    2,
                    3,
                    4,
                    11,
                    13,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    12,
                    14,
                    15,
                    16,
                    17,
                    18,
                    24,
                    26,
                    19,
                    20,
                    21,
                    22,
                    23,
                    25,
                    27,
                    33,
                    34,
                    35,
                    36,
                    37,
                    39,
                    41,
                ],
                [
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    12,
                    0,
                    1,
                    2,
                    3,
                    4,
                    11,
                    13,
                    19,
                    20,
                    21,
                    22,
                    23,
                    25,
                    27,
                    14,
                    15,
                    16,
                    17,
                    18,
                    24,
                    26,
                    28,
                    29,
                    30,
                    31,
                    32,
                    38,
                    40,
                ],
                [
                    14,
                    15,
                    16,
                    17,
                    18,
                    24,
                    26,
                    28,
                    29,
                    30,
                    31,
                    32,
                    38,
                    40,
                    42,
                    43,
                    44,
                    45,
                    46,
                    52,
                    54,
                    57,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    12,
                    0,
                    1,
                    2,
                    3,
                    4,
                    11,
                    13,
                ],
                [
                    19,
                    20,
                    21,
                    22,
                    23,
                    25,
                    27,
                    33,
                    34,
                    35,
                    36,
                    37,
                    39,
                    41,
                    47,
                    48,
                    49,
                    50,
                    51,
                    53,
                    55,
                    56,
                    0,
                    1,
                    2,
                    3,
                    4,
                    11,
                    13,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    12,
                ],
            ],
            "edge": [
                [
                    [5, 6, 7, 8, 9, 10, 12],
                    [14, 15, 16, 17, 18, 24, 26],
                    [19, 20, 21, 22, 23, 25, 27],
                    [33, 34, 35, 36, 37, 39, 41],
                ],
                [
                    [0, 1, 2, 3, 4, 11, 13],
                    [19, 20, 21, 22, 23, 25, 27],
                    [14, 15, 16, 17, 18, 24, 26],
                    [28, 29, 30, 31, 32, 38, 40],
                ],
                [[5, 6, 7, 8, 9, 10, 12], [0, 1, 2, 3, 4, 11, 13]],
                [[0, 1, 2, 3, 4, 11, 13], [5, 6, 7, 8, 9, 10, 12]],
            ],
            "center": [[1, 2, 3, 3], [0, 3, 2, 2], [1, 0], [0, 1]],
            "centerf_idx": [
                [0, 1, 2, 3, 4, 5, 6],
                [0, 1, 2, 3, 4, 5, 6],
                [0, 1, 2, 3, 4, 5, 6],
                [0, 1, 2, 3, 4, 5, 6],
            ],
            "rel_AO_per_center_per_frag": [
                [1.0, [0, 1, 2, 3, 4, 5, 6]],
                [1.0, [0, 1, 2, 3, 4, 5, 6]],
                [
                    1.0,
                    [
                        0,
                        1,
                        2,
                        3,
                        4,
                        5,
                        6,
                        7,
                        8,
                        9,
                        10,
                        11,
                        12,
                        13,
                        14,
                        15,
                        16,
                        17,
                        18,
                        19,
                        20,
                        21,
                    ],
                ],
                [
                    1.0,
                    [
                        0,
                        1,
                        2,
                        3,
                        4,
                        5,
                        6,
                        7,
                        8,
                        9,
                        10,
                        11,
                        12,
                        13,
                        14,
                        15,
                        16,
                        17,
                        18,
                        19,
                        20,
                        21,
                    ],
                ],
            ],
        }

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

        target = {
            "AO_per_frag": [(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,)],
            "edge": [(), (), (), (), (), (), (), ()],
            "center": [(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,)],
            "centerf_idx": [(0,), (0,), (0,), (0,), (0,), (0,), (0,), (0,)],
            "rel_AO_per_center_per_frag": [
                (1.0, (0,)),
                (1.0, (0,)),
                (1.0, (0,)),
                (1.0, (0,)),
                (1.0, (0,)),
                (1.0, (0,)),
                (1.0, (0,)),
                (1.0, (0,)),
            ],
        }

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

        target = {
            "AO_per_frag": [
                (1, 0, 2),
                (2, 1, 3),
                (3, 2, 4),
                (4, 3, 5),
                (5, 4, 6),
                (6, 5, 7),
            ],
            "edge": [
                ((2,),),
                ((1,), (3,)),
                ((2,), (4,)),
                ((3,), (5,)),
                ((6,), (4,)),
                ((5,),),
            ],
            "center": [(0, 1), (2,), (3,), (4,), (5,), (6, 7)],
            "centerf_idx": [(1, 0), (0,), (0,), (0,), (0,), (0, 2)],
            "rel_AO_per_center_per_frag": [
                (1.0, (1, 0)),
                (1.0, (0,)),
                (1.0, (0,)),
                (1.0, (0,)),
                (1.0, (0,)),
                (1.0, (0, 2)),
            ],
        }

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

        target = {
            "AO_per_frag": [
                (2, 0, 1, 3, 4),
                (3, 1, 2, 4, 5),
                (4, 2, 3, 5, 6),
                (5, 3, 4, 6, 7),
            ],
            "edge": [
                ((3,), (4,)),
                ((1,), (2,), (4,), (5,)),
                ((6,), (2,), (3,), (5,)),
                ((3,), (4,)),
            ],
            "center": [(0, 1, 2), (3,), (4,), (5, 6, 7)],
            "centerf_idx": [(1, 2, 0), (0,), (0,), (0, 3, 4)],
            "rel_AO_per_center_per_frag": [
                (1.0, (1, 2, 0)),
                (1.0, (0,)),
                (1.0, (0,)),
                (1.0, (0, 3, 4)),
            ],
        }

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

        target = {
            "AO_per_frag": [
                (0, 1, 2, 3, 4),
                (5, 6, 7, 8, 9),
                (10,),
                (11,),
                (12,),
                (13,),
                (14, 15, 16, 17, 18),
                (19, 20, 21, 22, 23),
                (24,),
                (25,),
                (26,),
                (27,),
                (28, 29, 30, 31, 32),
                (33, 34, 35, 36, 37),
                (38,),
                (39,),
                (40,),
                (41,),
                (42, 43, 44, 45, 46),
                (47, 48, 49, 50, 51),
                (52,),
                (53,),
                (54,),
                (55,),
                (56,),
                (57,),
            ],
            "edge": [
                (),
                (),
                (),
                (),
                (),
                (),
                (),
                (),
                (),
                (),
                (),
                (),
                (),
                (),
                (),
                (),
                (),
                (),
                (),
                (),
                (),
                (),
                (),
                (),
                (),
                (),
            ],
            "center": [
                (0, 1, 2, 3, 4),
                (5, 6, 7, 8, 9),
                (10,),
                (11,),
                (12,),
                (13,),
                (14, 15, 16, 17, 18),
                (19, 20, 21, 22, 23),
                (24,),
                (25,),
                (26,),
                (27,),
                (28, 29, 30, 31, 32),
                (33, 34, 35, 36, 37),
                (38,),
                (39,),
                (40,),
                (41,),
                (42, 43, 44, 45, 46),
                (47, 48, 49, 50, 51),
                (52,),
                (53,),
                (54,),
                (55,),
                (56,),
                (57,),
            ],
            "centerf_idx": [
                (0, 1, 2, 3, 4),
                (0, 1, 2, 3, 4),
                (0,),
                (0,),
                (0,),
                (0,),
                (0, 1, 2, 3, 4),
                (0, 1, 2, 3, 4),
                (0,),
                (0,),
                (0,),
                (0,),
                (0, 1, 2, 3, 4),
                (0, 1, 2, 3, 4),
                (0,),
                (0,),
                (0,),
                (0,),
                (0, 1, 2, 3, 4),
                (0, 1, 2, 3, 4),
                (0,),
                (0,),
                (0,),
                (0,),
                (0,),
                (0,),
            ],
            "rel_AO_per_center_per_frag": [
                (1.0, (0, 1, 2, 3, 4)),
                (1.0, (0, 1, 2, 3, 4)),
                (1.0, (0,)),
                (1.0, (0,)),
                (1.0, (0,)),
                (1.0, (0,)),
                (1.0, (0, 1, 2, 3, 4)),
                (1.0, (0, 1, 2, 3, 4)),
                (1.0, (0,)),
                (1.0, (0,)),
                (1.0, (0,)),
                (1.0, (0,)),
                (1.0, (0, 1, 2, 3, 4)),
                (1.0, (0, 1, 2, 3, 4)),
                (1.0, (0,)),
                (1.0, (0,)),
                (1.0, (0,)),
                (1.0, (0,)),
                (1.0, (0, 1, 2, 3, 4)),
                (1.0, (0, 1, 2, 3, 4)),
                (1.0, (0,)),
                (1.0, (0,)),
                (1.0, (0,)),
                (1.0, (0,)),
                (1.0, (0,)),
                (1.0, (0,)),
            ],
        }

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

        target = {
            "AO_per_frag": [
                (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 19, 20, 21, 22, 23),
                (5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 10, 12, 14, 15, 16, 17, 18),
                (14, 15, 16, 17, 18, 5, 6, 7, 8, 9, 24, 26, 28, 29, 30, 31, 32),
                (19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 25, 27, 33, 34, 35, 36, 37),
                (28, 29, 30, 31, 32, 14, 15, 16, 17, 18, 38, 40, 42, 43, 44, 45, 46),
                (33, 34, 35, 36, 37, 19, 20, 21, 22, 23, 39, 41, 47, 48, 49, 50, 51),
                (42, 43, 44, 45, 46, 28, 29, 30, 31, 32, 52, 54, 57),
                (47, 48, 49, 50, 51, 33, 34, 35, 36, 37, 53, 55, 56),
            ],
            "edge": [
                ((5, 6, 7, 8, 9), (19, 20, 21, 22, 23)),
                ((0, 1, 2, 3, 4), (14, 15, 16, 17, 18)),
                ((5, 6, 7, 8, 9), (32, 28, 29, 30, 31)),
                ((0, 1, 2, 3, 4), (33, 34, 35, 36, 37)),
                ((42, 43, 44, 45, 46), (14, 15, 16, 17, 18)),
                ((19, 20, 21, 22, 23), (47, 48, 49, 50, 51)),
                ((32, 28, 29, 30, 31),),
                ((33, 34, 35, 36, 37),),
            ],
            "center": [
                (0, 1, 2, 3, 4, 11, 13),
                (5, 6, 7, 8, 9, 10, 12),
                (14, 15, 16, 17, 18, 24, 26),
                (19, 20, 21, 22, 23, 25, 27),
                (32, 38, 40, 28, 29, 30, 31),
                (33, 34, 35, 36, 37, 39, 41),
                (42, 43, 44, 45, 46, 52, 54, 57),
                (47, 48, 49, 50, 51, 53, 55, 56),
            ],
            "centerf_idx": [
                (0, 1, 2, 3, 4, 10, 11),
                (0, 1, 2, 3, 4, 10, 11),
                (0, 1, 2, 3, 4, 10, 11),
                (0, 1, 2, 3, 4, 10, 11),
                (4, 10, 11, 0, 1, 2, 3),
                (0, 1, 2, 3, 4, 10, 11),
                (0, 1, 2, 3, 4, 10, 11, 12),
                (0, 1, 2, 3, 4, 10, 11, 12),
            ],
            "rel_AO_per_center_per_frag": [
                (1.0, (0, 1, 2, 3, 4, 10, 11)),
                (1.0, (0, 1, 2, 3, 4, 10, 11)),
                (1.0, (0, 1, 2, 3, 4, 10, 11)),
                (1.0, (0, 1, 2, 3, 4, 10, 11)),
                (1.0, (4, 10, 11, 0, 1, 2, 3)),
                (1.0, (0, 1, 2, 3, 4, 10, 11)),
                (1.0, (0, 1, 2, 3, 4, 10, 11, 12)),
                (1.0, (0, 1, 2, 3, 4, 10, 11, 12)),
            ],
        }

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

        target = {
            "AO_per_frag": [
                (
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    19,
                    20,
                    21,
                    22,
                    23,
                    25,
                    27,
                ),
                (
                    5,
                    6,
                    7,
                    8,
                    9,
                    0,
                    1,
                    2,
                    3,
                    4,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    24,
                    26,
                ),
                (
                    14,
                    15,
                    16,
                    17,
                    18,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    12,
                    24,
                    26,
                    28,
                    29,
                    30,
                    31,
                    32,
                    38,
                    40,
                ),
                (
                    19,
                    20,
                    21,
                    22,
                    23,
                    0,
                    1,
                    2,
                    3,
                    4,
                    11,
                    13,
                    25,
                    27,
                    33,
                    34,
                    35,
                    36,
                    37,
                    39,
                    41,
                ),
                (
                    28,
                    29,
                    30,
                    31,
                    32,
                    14,
                    15,
                    16,
                    17,
                    18,
                    24,
                    26,
                    38,
                    40,
                    42,
                    43,
                    44,
                    45,
                    46,
                    52,
                    54,
                    57,
                ),
                (
                    33,
                    34,
                    35,
                    36,
                    37,
                    19,
                    20,
                    21,
                    22,
                    23,
                    25,
                    27,
                    39,
                    41,
                    47,
                    48,
                    49,
                    50,
                    51,
                    53,
                    55,
                    56,
                ),
            ],
            "edge": [
                ((12,), (27,), (5, 6, 7, 8, 9), (19, 20, 21, 22, 23), (10,), (25,)),
                ((14, 15, 16, 17, 18), (11,), (24,), (26,), (13,), (0, 1, 2, 3, 4)),
                ((12,), (32, 28, 29, 30, 31), (40,), (5, 6, 7, 8, 9), (10,), (38,)),
                ((41,), (11,), (33, 34, 35, 36, 37), (39,), (13,), (0, 1, 2, 3, 4)),
                ((24,), (26,), (14, 15, 16, 17, 18)),
                ((25,), (19, 20, 21, 22, 23), (27,)),
            ],
            "center": [
                (0, 1, 2, 3, 4, 11, 13),
                (5, 6, 7, 8, 9, 10, 12),
                (14, 15, 16, 17, 18, 24, 26),
                (19, 20, 21, 22, 23, 25, 27),
                (32, 38, 40, 42, 43, 44, 45, 46, 52, 54, 57, 28, 29, 30, 31),
                (33, 34, 35, 36, 37, 39, 41, 47, 48, 49, 50, 51, 53, 55, 56),
            ],
            "centerf_idx": [
                (0, 1, 2, 3, 4, 11, 13),
                (0, 1, 2, 3, 4, 10, 12),
                (0, 1, 2, 3, 4, 12, 13),
                (0, 1, 2, 3, 4, 12, 13),
                (4, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 0, 1, 2, 3),
                (0, 1, 2, 3, 4, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21),
            ],
            "rel_AO_per_center_per_frag": [
                (1.0, (0, 1, 2, 3, 4, 11, 13)),
                (1.0, (0, 1, 2, 3, 4, 10, 12)),
                (1.0, (0, 1, 2, 3, 4, 12, 13)),
                (1.0, (0, 1, 2, 3, 4, 12, 13)),
                (1.0, (4, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 0, 1, 2, 3)),
                (1.0, (0, 1, 2, 3, 4, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21)),
            ],
        }

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
        target = -0.13198886164212092

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
        target = -0.5499456086311243

        self.run_energies_test(
            mf,
            2,
            "energy_graphgen_autogen_octane_be2",
            target,
            delta=1e-2,
        )

    @unittest.skipIf(
        not os.getenv("QUEMB_DO_KNOWN_TO_FAIL_TESTS") == "true",
        "This test is known to fail.",
    )
    def test_shared_centers_autocratic_matching(self):
        mol = gto.M("xyz/short_polypropylene.xyz", basis="sto-3g")
        mf = scf.RHF(mol)
        mf.kernel()

        fobj = fragmentate(mol, n_BE=2, frag_type="graphgen", print_frags=False)
        mybe = BE(mf, fobj)

        assert np.isclose(mf.e_tot, mybe.ebe_hf)

        fobj = fragmentate(mol, n_BE=3, frag_type="graphgen", print_frags=False)
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
            fobj = fragmentate(frag_type=frag_type, n_BE=n_BE, mol=mf.mol)
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
        fobj = fragmentate(frag_type=frag_type, n_BE=n_BE, mol=mf.mol)
        try:
            assert fobj.AO_per_frag == target["AO_per_frag"]
            assert fobj.AO_per_edge_per_frag == target["edge"]
            assert fobj.ref_frag_idx_per_edge == target["center"]
            assert fobj.centerf_idx == target["centerf_idx"]
            assert (
                fobj.scale_rel_AO_per_center_per_frag
                == target["rel_AO_per_center_per_frag"]
            )
        except AssertionError as e:
            print(f"Fragmentation test failed at {test_name} \n")
            raise e


if __name__ == "__main__":
    unittest.main()
