import pytest
from chemcoord import Cartesian
from ordered_set import OrderedSet
from pyscf.gto import M

from quemb.molbe.chemfrag import (
    BondConnectivity,
    PurelyStructureFragmented,
    _cleanup_if_subset,
    _SubsetsCleaned,
)
from quemb.molbe.fragment import fragpart


def test_connectivity_data():
    m = Cartesian.read_xyz("data/octane.xyz")

    conn_data = BondConnectivity.from_cartesian(m)
    expected = BondConnectivity(
        bonds_atoms={
            0: OrderedSet([1, 3, 5, 7]),
            1: OrderedSet([0, 2, 4, 6]),
            2: OrderedSet([1]),
            3: OrderedSet([0]),
            4: OrderedSet([1]),
            5: OrderedSet([0]),
            6: OrderedSet([1, 8, 10, 12]),
            7: OrderedSet([0, 9, 11, 13]),
            8: OrderedSet([6]),
            9: OrderedSet([7]),
            10: OrderedSet([6]),
            11: OrderedSet([7]),
            12: OrderedSet([6, 14, 16, 18]),
            13: OrderedSet([7, 15, 17, 19]),
            14: OrderedSet([12]),
            15: OrderedSet([13]),
            16: OrderedSet([12]),
            17: OrderedSet([13]),
            18: OrderedSet([12, 20, 22, 25]),
            19: OrderedSet([13, 21, 23, 24]),
            20: OrderedSet([18]),
            21: OrderedSet([19]),
            22: OrderedSet([18]),
            23: OrderedSet([19]),
            24: OrderedSet([19]),
            25: OrderedSet([18]),
        },
        motifs=OrderedSet([0, 1, 6, 7, 12, 13, 18, 19]),
        bonds_motifs={
            0: OrderedSet([1, 7]),
            1: OrderedSet([0, 6]),
            6: OrderedSet([1, 12]),
            7: OrderedSet([0, 13]),
            12: OrderedSet([6, 18]),
            13: OrderedSet([7, 19]),
            18: OrderedSet([12]),
            19: OrderedSet([13]),
        },
        H_atoms=OrderedSet(
            [2, 3, 4, 5, 8, 9, 10, 11, 14, 15, 16, 17, 20, 21, 22, 23, 24, 25]
        ),
        H_per_motif={
            0: OrderedSet([3, 5]),
            1: OrderedSet([2, 4]),
            6: OrderedSet([8, 10]),
            7: OrderedSet([9, 11]),
            12: OrderedSet([14, 16]),
            13: OrderedSet([15, 17]),
            18: OrderedSet([20, 22, 25]),
            19: OrderedSet([21, 23, 24]),
        },
        atoms_per_motif={
            0: OrderedSet([0, 3, 5]),
            1: OrderedSet([1, 2, 4]),
            6: OrderedSet([6, 8, 10]),
            7: OrderedSet([7, 9, 11]),
            12: OrderedSet([12, 14, 16]),
            13: OrderedSet([13, 15, 17]),
            18: OrderedSet([18, 20, 22, 25]),
            19: OrderedSet([19, 21, 23, 24]),
        },
    )

    assert conn_data == expected

    # sort carbon atoms first and then by y coordinate,
    # i.e. the visual order of the atoms in the molecule
    resorted_conn_data = BondConnectivity.from_cartesian(
        m.sort_values(by=["atom", "y"]).reset_index()
    )

    resorted_expected = BondConnectivity(
        bonds_atoms={
            0: OrderedSet([1, 8, 9, 10]),
            1: OrderedSet([0, 2, 11, 12]),
            2: OrderedSet([1, 3, 13, 14]),
            3: OrderedSet([2, 4, 15, 16]),
            4: OrderedSet([3, 5, 17, 18]),
            5: OrderedSet([4, 6, 19, 20]),
            6: OrderedSet([5, 7, 21, 22]),
            7: OrderedSet([6, 23, 24, 25]),
            8: OrderedSet([0]),
            9: OrderedSet([0]),
            10: OrderedSet([0]),
            11: OrderedSet([1]),
            12: OrderedSet([1]),
            13: OrderedSet([2]),
            14: OrderedSet([2]),
            15: OrderedSet([3]),
            16: OrderedSet([3]),
            17: OrderedSet([4]),
            18: OrderedSet([4]),
            19: OrderedSet([5]),
            20: OrderedSet([5]),
            21: OrderedSet([6]),
            22: OrderedSet([6]),
            23: OrderedSet([7]),
            24: OrderedSet([7]),
            25: OrderedSet([7]),
        },
        motifs=OrderedSet([0, 1, 2, 3, 4, 5, 6, 7]),
        bonds_motifs={
            0: OrderedSet([1]),
            1: OrderedSet([0, 2]),
            2: OrderedSet([1, 3]),
            3: OrderedSet([2, 4]),
            4: OrderedSet([3, 5]),
            5: OrderedSet([4, 6]),
            6: OrderedSet([5, 7]),
            7: OrderedSet([6]),
        },
        H_atoms=OrderedSet(
            [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
        ),
        H_per_motif={
            0: OrderedSet([8, 9, 10]),
            1: OrderedSet([11, 12]),
            2: OrderedSet([13, 14]),
            3: OrderedSet([15, 16]),
            4: OrderedSet([17, 18]),
            5: OrderedSet([19, 20]),
            6: OrderedSet([21, 22]),
            7: OrderedSet([23, 24, 25]),
        },
        atoms_per_motif={
            0: OrderedSet([0, 8, 9, 10]),
            1: OrderedSet([1, 11, 12]),
            2: OrderedSet([2, 13, 14]),
            3: OrderedSet([3, 15, 16]),
            4: OrderedSet([4, 17, 18]),
            5: OrderedSet([5, 19, 20]),
            6: OrderedSet([6, 21, 22]),
            7: OrderedSet([7, 23, 24, 25]),
        },
    )

    assert resorted_conn_data == resorted_expected


def test_fragment_generation():
    """This does not yet test the cleanup of subsets, i.e. also fragments
    that are fully contained in others are returned."""

    m = Cartesian.read_xyz("data/octane.xyz")

    expected = {
        1: {
            0: OrderedSet([0]),
            1: OrderedSet([1]),
            6: OrderedSet([6]),
            7: OrderedSet([7]),
            12: OrderedSet([12]),
            13: OrderedSet([13]),
            18: OrderedSet([18]),
            19: OrderedSet([19]),
        },
        2: {
            0: OrderedSet([0, 1, 7]),
            1: OrderedSet([1, 0, 6]),
            6: OrderedSet([6, 1, 12]),
            7: OrderedSet([7, 0, 13]),
            12: OrderedSet([12, 6, 18]),
            13: OrderedSet([13, 7, 19]),
            18: OrderedSet([18, 12]),
            19: OrderedSet([19, 13]),
        },
        3: {
            0: OrderedSet([0, 1, 7, 6, 13]),
            1: OrderedSet([1, 0, 6, 7, 12]),
            6: OrderedSet([6, 1, 12, 0, 18]),
            7: OrderedSet([7, 0, 13, 1, 19]),
            12: OrderedSet([12, 6, 18, 1]),
            13: OrderedSet([13, 7, 19, 0]),
            18: OrderedSet([18, 12, 6]),
            19: OrderedSet([19, 13, 7]),
        },
        4: {
            0: OrderedSet([0, 1, 7, 6, 13, 12, 19]),
            1: OrderedSet([1, 0, 6, 7, 12, 13, 18]),
            6: OrderedSet([6, 1, 12, 0, 18, 7]),
            7: OrderedSet([7, 0, 13, 1, 19, 6]),
            12: OrderedSet([12, 6, 18, 1, 0]),
            13: OrderedSet([13, 7, 19, 0, 1]),
            18: OrderedSet([18, 12, 6, 1]),
            19: OrderedSet([19, 13, 7, 0]),
        },
        5: {
            0: OrderedSet([0, 1, 7, 6, 13, 12, 19, 18]),
            1: OrderedSet([1, 0, 6, 7, 12, 13, 18, 19]),
            6: OrderedSet([6, 1, 12, 0, 18, 7, 13]),
            7: OrderedSet([7, 0, 13, 1, 19, 6, 12]),
            12: OrderedSet([12, 6, 18, 1, 0, 7]),
            13: OrderedSet([13, 7, 19, 0, 1, 6]),
            18: OrderedSet([18, 12, 6, 1, 0]),
            19: OrderedSet([19, 13, 7, 0, 1]),
        },
        6: {
            0: OrderedSet([0, 1, 7, 6, 13, 12, 19, 18]),
            1: OrderedSet([1, 0, 6, 7, 12, 13, 18, 19]),
            6: OrderedSet([6, 1, 12, 0, 18, 7, 13, 19]),
            7: OrderedSet([7, 0, 13, 1, 19, 6, 12, 18]),
            12: OrderedSet([12, 6, 18, 1, 0, 7, 13]),
            13: OrderedSet([13, 7, 19, 0, 1, 6, 12]),
            18: OrderedSet([18, 12, 6, 1, 0, 7]),
            19: OrderedSet([19, 13, 7, 0, 1, 6]),
        },
        7: {
            0: OrderedSet([0, 1, 7, 6, 13, 12, 19, 18]),
            1: OrderedSet([1, 0, 6, 7, 12, 13, 18, 19]),
            6: OrderedSet([6, 1, 12, 0, 18, 7, 13, 19]),
            7: OrderedSet([7, 0, 13, 1, 19, 6, 12, 18]),
            12: OrderedSet([12, 6, 18, 1, 0, 7, 13, 19]),
            13: OrderedSet([13, 7, 19, 0, 1, 6, 12, 18]),
            18: OrderedSet([18, 12, 6, 1, 0, 7, 13]),
            19: OrderedSet([19, 13, 7, 0, 1, 6, 12]),
        },
        8: {
            0: OrderedSet([0, 1, 7, 6, 13, 12, 19, 18]),
            1: OrderedSet([1, 0, 6, 7, 12, 13, 18, 19]),
            6: OrderedSet([6, 1, 12, 0, 18, 7, 13, 19]),
            7: OrderedSet([7, 0, 13, 1, 19, 6, 12, 18]),
            12: OrderedSet([12, 6, 18, 1, 0, 7, 13, 19]),
            13: OrderedSet([13, 7, 19, 0, 1, 6, 12, 18]),
            18: OrderedSet([18, 12, 6, 1, 0, 7, 13, 19]),
            19: OrderedSet([19, 13, 7, 0, 1, 6, 12, 18]),
        },
    }

    fragments = {
        n_BE: BondConnectivity.from_cartesian(m).get_all_BE_fragments(n_BE)
        for n_BE in range(1, 9)
    }

    assert fragments == expected


def test_cleaned_fragments():
    """In this test fragments that are subsets of others are also removed.
    It is also tested if the assignment of an origin, i.e. the center with the
    lowest index whose fragment is a superset of other fragments, reliably works.
    """
    m = Cartesian.read_xyz("data/octane.xyz")

    expected = {
        1: _SubsetsCleaned(
            motif_per_frag={
                0: OrderedSet([0]),
                1: OrderedSet([1]),
                6: OrderedSet([6]),
                7: OrderedSet([7]),
                12: OrderedSet([12]),
                13: OrderedSet([13]),
                18: OrderedSet([18]),
                19: OrderedSet([19]),
            },
            swallowed_centers={},
        ),
        2: _SubsetsCleaned(
            motif_per_frag={
                0: OrderedSet([0, 1, 7]),
                1: OrderedSet([1, 0, 6]),
                6: OrderedSet([6, 1, 12]),
                7: OrderedSet([7, 0, 13]),
                12: OrderedSet([12, 6, 18]),
                13: OrderedSet([13, 7, 19]),
            },
            swallowed_centers={12: OrderedSet([18]), 13: OrderedSet([19])},
        ),
        3: _SubsetsCleaned(
            motif_per_frag={
                0: OrderedSet([0, 1, 6, 7, 13]),
                1: OrderedSet([1, 0, 6, 7, 12]),
                6: OrderedSet([6, 0, 1, 12, 18]),
                7: OrderedSet([7, 0, 1, 13, 19]),
            },
            swallowed_centers={6: OrderedSet([12, 18]), 7: OrderedSet([13, 19])},
        ),
        4: _SubsetsCleaned(
            motif_per_frag={
                0: OrderedSet([0, 1, 6, 7, 12, 13, 19]),
                1: OrderedSet([1, 0, 6, 7, 12, 13, 18]),
            },
            swallowed_centers={0: OrderedSet([7, 13, 19]), 1: OrderedSet([6, 12, 18])},
        ),
        5: _SubsetsCleaned(
            motif_per_frag={0: OrderedSet([0, 1, 6, 7, 12, 13, 18, 19])},
            swallowed_centers={0: OrderedSet([1, 7, 6, 13, 12, 19, 18])},
        ),
        6: _SubsetsCleaned(
            motif_per_frag={0: OrderedSet([0, 1, 6, 7, 12, 13, 18, 19])},
            swallowed_centers={0: OrderedSet([1, 7, 6, 13, 12, 19, 18])},
        ),
        7: _SubsetsCleaned(
            motif_per_frag={0: OrderedSet([0, 1, 6, 7, 12, 13, 18, 19])},
            swallowed_centers={0: OrderedSet([1, 7, 6, 13, 12, 19, 18])},
        ),
        8: _SubsetsCleaned(
            motif_per_frag={0: OrderedSet([0, 1, 6, 7, 12, 13, 18, 19])},
            swallowed_centers={0: OrderedSet([1, 7, 6, 13, 12, 19, 18])},
        ),
    }

    cleaned_fragments = {
        n_BE: _cleanup_if_subset(
            BondConnectivity.from_cartesian(m).get_all_BE_fragments(n_BE)
        )
        for n_BE in range(1, 9)
    }

    assert cleaned_fragments == expected


def test_fragmented_molecule():
    m = Cartesian.read_xyz("data/octane.xyz")
    mol = m.to_pyscf()

    fragmented = {
        n_BE: PurelyStructureFragmented.from_mole(mol, n_BE=n_BE)
        for n_BE in range(1, 8)
    }

    for fragment in fragmented.values():
        assert fragment.is_ordered()

    expected = {
        1: PurelyStructureFragmented(
            mol=mol,
            motifs_per_frag=[
                OrderedSet([0]),
                OrderedSet([1]),
                OrderedSet([6]),
                OrderedSet([7]),
                OrderedSet([12]),
                OrderedSet([13]),
                OrderedSet([18]),
                OrderedSet([19]),
            ],
            centers_per_frag=[
                OrderedSet([0]),
                OrderedSet([1]),
                OrderedSet([6]),
                OrderedSet([7]),
                OrderedSet([12]),
                OrderedSet([13]),
                OrderedSet([18]),
                OrderedSet([19]),
            ],
            edges_per_frag=[
                OrderedSet(),
                OrderedSet(),
                OrderedSet(),
                OrderedSet(),
                OrderedSet(),
                OrderedSet(),
                OrderedSet(),
                OrderedSet(),
            ],
            origin_per_frag=[
                OrderedSet([0]),
                OrderedSet([1]),
                OrderedSet([6]),
                OrderedSet([7]),
                OrderedSet([12]),
                OrderedSet([13]),
                OrderedSet([18]),
                OrderedSet([19]),
            ],
            atoms_per_frag=[
                OrderedSet([0, 3, 5]),
                OrderedSet([1, 2, 4]),
                OrderedSet([6, 8, 10]),
                OrderedSet([7, 9, 11]),
                OrderedSet([12, 14, 16]),
                OrderedSet([13, 15, 17]),
                OrderedSet([18, 20, 22, 25]),
                OrderedSet([19, 21, 23, 24]),
            ],
            frag_idx_per_edge=[{}, {}, {}, {}, {}, {}, {}, {}],
            conn_data=BondConnectivity(
                bonds_atoms={
                    0: OrderedSet([1, 3, 5, 7]),
                    1: OrderedSet([0, 2, 4, 6]),
                    2: OrderedSet([1]),
                    3: OrderedSet([0]),
                    4: OrderedSet([1]),
                    5: OrderedSet([0]),
                    6: OrderedSet([1, 8, 10, 12]),
                    7: OrderedSet([0, 9, 11, 13]),
                    8: OrderedSet([6]),
                    9: OrderedSet([7]),
                    10: OrderedSet([6]),
                    11: OrderedSet([7]),
                    12: OrderedSet([6, 14, 16, 18]),
                    13: OrderedSet([7, 15, 17, 19]),
                    14: OrderedSet([12]),
                    15: OrderedSet([13]),
                    16: OrderedSet([12]),
                    17: OrderedSet([13]),
                    18: OrderedSet([12, 20, 22, 25]),
                    19: OrderedSet([13, 21, 23, 24]),
                    20: OrderedSet([18]),
                    21: OrderedSet([19]),
                    22: OrderedSet([18]),
                    23: OrderedSet([19]),
                    24: OrderedSet([19]),
                    25: OrderedSet([18]),
                },
                motifs=OrderedSet([0, 1, 6, 7, 12, 13, 18, 19]),
                bonds_motifs={
                    0: OrderedSet([1, 7]),
                    1: OrderedSet([0, 6]),
                    6: OrderedSet([1, 12]),
                    7: OrderedSet([0, 13]),
                    12: OrderedSet([6, 18]),
                    13: OrderedSet([7, 19]),
                    18: OrderedSet([12]),
                    19: OrderedSet([13]),
                },
                H_atoms=OrderedSet(
                    [2, 3, 4, 5, 8, 9, 10, 11, 14, 15, 16, 17, 20, 21, 22, 23, 24, 25]
                ),
                H_per_motif={
                    0: OrderedSet([3, 5]),
                    1: OrderedSet([2, 4]),
                    6: OrderedSet([8, 10]),
                    7: OrderedSet([9, 11]),
                    12: OrderedSet([14, 16]),
                    13: OrderedSet([15, 17]),
                    18: OrderedSet([20, 22, 25]),
                    19: OrderedSet([21, 23, 24]),
                },
                atoms_per_motif={
                    0: OrderedSet([0, 3, 5]),
                    1: OrderedSet([1, 2, 4]),
                    6: OrderedSet([6, 8, 10]),
                    7: OrderedSet([7, 9, 11]),
                    12: OrderedSet([12, 14, 16]),
                    13: OrderedSet([13, 15, 17]),
                    18: OrderedSet([18, 20, 22, 25]),
                    19: OrderedSet([19, 21, 23, 24]),
                },
                treat_H_different=True,
            ),
            n_BE=1,
        ),
        2: PurelyStructureFragmented(
            mol=mol,
            motifs_per_frag=[
                OrderedSet([0, 1, 7]),
                OrderedSet([1, 0, 6]),
                OrderedSet([6, 1, 12]),
                OrderedSet([7, 0, 13]),
                OrderedSet([12, 18, 6]),
                OrderedSet([13, 19, 7]),
            ],
            centers_per_frag=[
                OrderedSet([0]),
                OrderedSet([1]),
                OrderedSet([6]),
                OrderedSet([7]),
                OrderedSet([12, 18]),
                OrderedSet([13, 19]),
            ],
            edges_per_frag=[
                OrderedSet([1, 7]),
                OrderedSet([0, 6]),
                OrderedSet([1, 12]),
                OrderedSet([0, 13]),
                OrderedSet([6]),
                OrderedSet([7]),
            ],
            origin_per_frag=[
                OrderedSet([0]),
                OrderedSet([1]),
                OrderedSet([6]),
                OrderedSet([7]),
                OrderedSet([12]),
                OrderedSet([13]),
            ],
            atoms_per_frag=[
                OrderedSet([0, 3, 5, 1, 2, 4, 7, 9, 11]),
                OrderedSet([1, 2, 4, 0, 3, 5, 6, 8, 10]),
                OrderedSet([6, 8, 10, 1, 2, 4, 12, 14, 16]),
                OrderedSet([7, 9, 11, 0, 3, 5, 13, 15, 17]),
                OrderedSet([12, 14, 16, 18, 20, 22, 25, 6, 8, 10]),
                OrderedSet([13, 15, 17, 19, 21, 23, 24, 7, 9, 11]),
            ],
            frag_idx_per_edge=[
                {1: 1, 7: 3},
                {0: 0, 6: 2},
                {1: 1, 12: 4},
                {0: 0, 13: 5},
                {6: 2},
                {7: 3},
            ],
            conn_data=BondConnectivity(
                bonds_atoms={
                    0: OrderedSet([1, 3, 5, 7]),
                    1: OrderedSet([0, 2, 4, 6]),
                    2: OrderedSet([1]),
                    3: OrderedSet([0]),
                    4: OrderedSet([1]),
                    5: OrderedSet([0]),
                    6: OrderedSet([1, 8, 10, 12]),
                    7: OrderedSet([0, 9, 11, 13]),
                    8: OrderedSet([6]),
                    9: OrderedSet([7]),
                    10: OrderedSet([6]),
                    11: OrderedSet([7]),
                    12: OrderedSet([6, 14, 16, 18]),
                    13: OrderedSet([7, 15, 17, 19]),
                    14: OrderedSet([12]),
                    15: OrderedSet([13]),
                    16: OrderedSet([12]),
                    17: OrderedSet([13]),
                    18: OrderedSet([12, 20, 22, 25]),
                    19: OrderedSet([13, 21, 23, 24]),
                    20: OrderedSet([18]),
                    21: OrderedSet([19]),
                    22: OrderedSet([18]),
                    23: OrderedSet([19]),
                    24: OrderedSet([19]),
                    25: OrderedSet([18]),
                },
                motifs=OrderedSet([0, 1, 6, 7, 12, 13, 18, 19]),
                bonds_motifs={
                    0: OrderedSet([1, 7]),
                    1: OrderedSet([0, 6]),
                    6: OrderedSet([1, 12]),
                    7: OrderedSet([0, 13]),
                    12: OrderedSet([6, 18]),
                    13: OrderedSet([7, 19]),
                    18: OrderedSet([12]),
                    19: OrderedSet([13]),
                },
                H_atoms=OrderedSet(
                    [2, 3, 4, 5, 8, 9, 10, 11, 14, 15, 16, 17, 20, 21, 22, 23, 24, 25]
                ),
                H_per_motif={
                    0: OrderedSet([3, 5]),
                    1: OrderedSet([2, 4]),
                    6: OrderedSet([8, 10]),
                    7: OrderedSet([9, 11]),
                    12: OrderedSet([14, 16]),
                    13: OrderedSet([15, 17]),
                    18: OrderedSet([20, 22, 25]),
                    19: OrderedSet([21, 23, 24]),
                },
                atoms_per_motif={
                    0: OrderedSet([0, 3, 5]),
                    1: OrderedSet([1, 2, 4]),
                    6: OrderedSet([6, 8, 10]),
                    7: OrderedSet([7, 9, 11]),
                    12: OrderedSet([12, 14, 16]),
                    13: OrderedSet([13, 15, 17]),
                    18: OrderedSet([18, 20, 22, 25]),
                    19: OrderedSet([19, 21, 23, 24]),
                },
                treat_H_different=True,
            ),
            n_BE=2,
        ),
        3: PurelyStructureFragmented(
            mol=mol,
            motifs_per_frag=[
                OrderedSet([0, 1, 6, 7, 13]),
                OrderedSet([1, 0, 6, 7, 12]),
                OrderedSet([6, 12, 18, 0, 1]),
                OrderedSet([7, 13, 19, 0, 1]),
            ],
            centers_per_frag=[
                OrderedSet([0]),
                OrderedSet([1]),
                OrderedSet([6, 12, 18]),
                OrderedSet([7, 13, 19]),
            ],
            edges_per_frag=[
                OrderedSet([1, 6, 7, 13]),
                OrderedSet([0, 6, 7, 12]),
                OrderedSet([0, 1]),
                OrderedSet([0, 1]),
            ],
            origin_per_frag=[
                OrderedSet([0]),
                OrderedSet([1]),
                OrderedSet([6]),
                OrderedSet([7]),
            ],
            atoms_per_frag=[
                OrderedSet([0, 3, 5, 1, 2, 4, 6, 8, 10, 7, 9, 11, 13, 15, 17]),
                OrderedSet([1, 2, 4, 0, 3, 5, 6, 8, 10, 7, 9, 11, 12, 14, 16]),
                OrderedSet([6, 8, 10, 12, 14, 16, 18, 20, 22, 25, 0, 3, 5, 1, 2, 4]),
                OrderedSet([7, 9, 11, 13, 15, 17, 19, 21, 23, 24, 0, 3, 5, 1, 2, 4]),
            ],
            frag_idx_per_edge=[
                {1: 1, 6: 2, 7: 3, 13: 3},
                {0: 0, 6: 2, 7: 3, 12: 2},
                {0: 0, 1: 1},
                {0: 0, 1: 1},
            ],
            conn_data=BondConnectivity(
                bonds_atoms={
                    0: OrderedSet([1, 3, 5, 7]),
                    1: OrderedSet([0, 2, 4, 6]),
                    2: OrderedSet([1]),
                    3: OrderedSet([0]),
                    4: OrderedSet([1]),
                    5: OrderedSet([0]),
                    6: OrderedSet([1, 8, 10, 12]),
                    7: OrderedSet([0, 9, 11, 13]),
                    8: OrderedSet([6]),
                    9: OrderedSet([7]),
                    10: OrderedSet([6]),
                    11: OrderedSet([7]),
                    12: OrderedSet([6, 14, 16, 18]),
                    13: OrderedSet([7, 15, 17, 19]),
                    14: OrderedSet([12]),
                    15: OrderedSet([13]),
                    16: OrderedSet([12]),
                    17: OrderedSet([13]),
                    18: OrderedSet([12, 20, 22, 25]),
                    19: OrderedSet([13, 21, 23, 24]),
                    20: OrderedSet([18]),
                    21: OrderedSet([19]),
                    22: OrderedSet([18]),
                    23: OrderedSet([19]),
                    24: OrderedSet([19]),
                    25: OrderedSet([18]),
                },
                motifs=OrderedSet([0, 1, 6, 7, 12, 13, 18, 19]),
                bonds_motifs={
                    0: OrderedSet([1, 7]),
                    1: OrderedSet([0, 6]),
                    6: OrderedSet([1, 12]),
                    7: OrderedSet([0, 13]),
                    12: OrderedSet([6, 18]),
                    13: OrderedSet([7, 19]),
                    18: OrderedSet([12]),
                    19: OrderedSet([13]),
                },
                H_atoms=OrderedSet(
                    [2, 3, 4, 5, 8, 9, 10, 11, 14, 15, 16, 17, 20, 21, 22, 23, 24, 25]
                ),
                H_per_motif={
                    0: OrderedSet([3, 5]),
                    1: OrderedSet([2, 4]),
                    6: OrderedSet([8, 10]),
                    7: OrderedSet([9, 11]),
                    12: OrderedSet([14, 16]),
                    13: OrderedSet([15, 17]),
                    18: OrderedSet([20, 22, 25]),
                    19: OrderedSet([21, 23, 24]),
                },
                atoms_per_motif={
                    0: OrderedSet([0, 3, 5]),
                    1: OrderedSet([1, 2, 4]),
                    6: OrderedSet([6, 8, 10]),
                    7: OrderedSet([7, 9, 11]),
                    12: OrderedSet([12, 14, 16]),
                    13: OrderedSet([13, 15, 17]),
                    18: OrderedSet([18, 20, 22, 25]),
                    19: OrderedSet([19, 21, 23, 24]),
                },
                treat_H_different=True,
            ),
            n_BE=3,
        ),
        4: PurelyStructureFragmented(
            mol=mol,
            motifs_per_frag=[
                OrderedSet([0, 7, 13, 19, 1, 6, 12]),
                OrderedSet([1, 6, 12, 18, 0, 7, 13]),
            ],
            centers_per_frag=[OrderedSet([0, 7, 13, 19]), OrderedSet([1, 6, 12, 18])],
            edges_per_frag=[OrderedSet([1, 6, 12]), OrderedSet([0, 7, 13])],
            origin_per_frag=[OrderedSet([0]), OrderedSet([1])],
            atoms_per_frag=[
                OrderedSet(
                    [
                        0,
                        3,
                        5,
                        7,
                        9,
                        11,
                        13,
                        15,
                        17,
                        19,
                        21,
                        23,
                        24,
                        1,
                        2,
                        4,
                        6,
                        8,
                        10,
                        12,
                        14,
                        16,
                    ]
                ),
                OrderedSet(
                    [
                        1,
                        2,
                        4,
                        6,
                        8,
                        10,
                        12,
                        14,
                        16,
                        18,
                        20,
                        22,
                        25,
                        0,
                        3,
                        5,
                        7,
                        9,
                        11,
                        13,
                        15,
                        17,
                    ]
                ),
            ],
            frag_idx_per_edge=[{1: 1, 6: 1, 12: 1}, {0: 0, 7: 0, 13: 0}],
            conn_data=BondConnectivity(
                bonds_atoms={
                    0: OrderedSet([1, 3, 5, 7]),
                    1: OrderedSet([0, 2, 4, 6]),
                    2: OrderedSet([1]),
                    3: OrderedSet([0]),
                    4: OrderedSet([1]),
                    5: OrderedSet([0]),
                    6: OrderedSet([1, 8, 10, 12]),
                    7: OrderedSet([0, 9, 11, 13]),
                    8: OrderedSet([6]),
                    9: OrderedSet([7]),
                    10: OrderedSet([6]),
                    11: OrderedSet([7]),
                    12: OrderedSet([6, 14, 16, 18]),
                    13: OrderedSet([7, 15, 17, 19]),
                    14: OrderedSet([12]),
                    15: OrderedSet([13]),
                    16: OrderedSet([12]),
                    17: OrderedSet([13]),
                    18: OrderedSet([12, 20, 22, 25]),
                    19: OrderedSet([13, 21, 23, 24]),
                    20: OrderedSet([18]),
                    21: OrderedSet([19]),
                    22: OrderedSet([18]),
                    23: OrderedSet([19]),
                    24: OrderedSet([19]),
                    25: OrderedSet([18]),
                },
                motifs=OrderedSet([0, 1, 6, 7, 12, 13, 18, 19]),
                bonds_motifs={
                    0: OrderedSet([1, 7]),
                    1: OrderedSet([0, 6]),
                    6: OrderedSet([1, 12]),
                    7: OrderedSet([0, 13]),
                    12: OrderedSet([6, 18]),
                    13: OrderedSet([7, 19]),
                    18: OrderedSet([12]),
                    19: OrderedSet([13]),
                },
                H_atoms=OrderedSet(
                    [2, 3, 4, 5, 8, 9, 10, 11, 14, 15, 16, 17, 20, 21, 22, 23, 24, 25]
                ),
                H_per_motif={
                    0: OrderedSet([3, 5]),
                    1: OrderedSet([2, 4]),
                    6: OrderedSet([8, 10]),
                    7: OrderedSet([9, 11]),
                    12: OrderedSet([14, 16]),
                    13: OrderedSet([15, 17]),
                    18: OrderedSet([20, 22, 25]),
                    19: OrderedSet([21, 23, 24]),
                },
                atoms_per_motif={
                    0: OrderedSet([0, 3, 5]),
                    1: OrderedSet([1, 2, 4]),
                    6: OrderedSet([6, 8, 10]),
                    7: OrderedSet([7, 9, 11]),
                    12: OrderedSet([12, 14, 16]),
                    13: OrderedSet([13, 15, 17]),
                    18: OrderedSet([18, 20, 22, 25]),
                    19: OrderedSet([19, 21, 23, 24]),
                },
                treat_H_different=True,
            ),
            n_BE=4,
        ),
        5: PurelyStructureFragmented(
            mol=mol,
            motifs_per_frag=[OrderedSet([0, 1, 6, 7, 12, 13, 18, 19])],
            centers_per_frag=[OrderedSet([0, 1, 6, 7, 12, 13, 18, 19])],
            edges_per_frag=[OrderedSet()],
            origin_per_frag=[OrderedSet([0])],
            atoms_per_frag=[
                OrderedSet(
                    [
                        0,
                        3,
                        5,
                        1,
                        2,
                        4,
                        6,
                        8,
                        10,
                        7,
                        9,
                        11,
                        12,
                        14,
                        16,
                        13,
                        15,
                        17,
                        18,
                        20,
                        22,
                        25,
                        19,
                        21,
                        23,
                        24,
                    ]
                )
            ],
            frag_idx_per_edge=[{}],
            conn_data=BondConnectivity(
                bonds_atoms={
                    0: OrderedSet([1, 3, 5, 7]),
                    1: OrderedSet([0, 2, 4, 6]),
                    2: OrderedSet([1]),
                    3: OrderedSet([0]),
                    4: OrderedSet([1]),
                    5: OrderedSet([0]),
                    6: OrderedSet([1, 8, 10, 12]),
                    7: OrderedSet([0, 9, 11, 13]),
                    8: OrderedSet([6]),
                    9: OrderedSet([7]),
                    10: OrderedSet([6]),
                    11: OrderedSet([7]),
                    12: OrderedSet([6, 14, 16, 18]),
                    13: OrderedSet([7, 15, 17, 19]),
                    14: OrderedSet([12]),
                    15: OrderedSet([13]),
                    16: OrderedSet([12]),
                    17: OrderedSet([13]),
                    18: OrderedSet([12, 20, 22, 25]),
                    19: OrderedSet([13, 21, 23, 24]),
                    20: OrderedSet([18]),
                    21: OrderedSet([19]),
                    22: OrderedSet([18]),
                    23: OrderedSet([19]),
                    24: OrderedSet([19]),
                    25: OrderedSet([18]),
                },
                motifs=OrderedSet([0, 1, 6, 7, 12, 13, 18, 19]),
                bonds_motifs={
                    0: OrderedSet([1, 7]),
                    1: OrderedSet([0, 6]),
                    6: OrderedSet([1, 12]),
                    7: OrderedSet([0, 13]),
                    12: OrderedSet([6, 18]),
                    13: OrderedSet([7, 19]),
                    18: OrderedSet([12]),
                    19: OrderedSet([13]),
                },
                H_atoms=OrderedSet(
                    [2, 3, 4, 5, 8, 9, 10, 11, 14, 15, 16, 17, 20, 21, 22, 23, 24, 25]
                ),
                H_per_motif={
                    0: OrderedSet([3, 5]),
                    1: OrderedSet([2, 4]),
                    6: OrderedSet([8, 10]),
                    7: OrderedSet([9, 11]),
                    12: OrderedSet([14, 16]),
                    13: OrderedSet([15, 17]),
                    18: OrderedSet([20, 22, 25]),
                    19: OrderedSet([21, 23, 24]),
                },
                atoms_per_motif={
                    0: OrderedSet([0, 3, 5]),
                    1: OrderedSet([1, 2, 4]),
                    6: OrderedSet([6, 8, 10]),
                    7: OrderedSet([7, 9, 11]),
                    12: OrderedSet([12, 14, 16]),
                    13: OrderedSet([13, 15, 17]),
                    18: OrderedSet([18, 20, 22, 25]),
                    19: OrderedSet([19, 21, 23, 24]),
                },
                treat_H_different=True,
            ),
            n_BE=5,
        ),
        6: PurelyStructureFragmented(
            mol=mol,
            motifs_per_frag=[OrderedSet([0, 1, 6, 7, 12, 13, 18, 19])],
            centers_per_frag=[OrderedSet([0, 1, 6, 7, 12, 13, 18, 19])],
            edges_per_frag=[OrderedSet()],
            origin_per_frag=[OrderedSet([0])],
            atoms_per_frag=[
                OrderedSet(
                    [
                        0,
                        3,
                        5,
                        1,
                        2,
                        4,
                        6,
                        8,
                        10,
                        7,
                        9,
                        11,
                        12,
                        14,
                        16,
                        13,
                        15,
                        17,
                        18,
                        20,
                        22,
                        25,
                        19,
                        21,
                        23,
                        24,
                    ]
                )
            ],
            frag_idx_per_edge=[{}],
            conn_data=BondConnectivity(
                bonds_atoms={
                    0: OrderedSet([1, 3, 5, 7]),
                    1: OrderedSet([0, 2, 4, 6]),
                    2: OrderedSet([1]),
                    3: OrderedSet([0]),
                    4: OrderedSet([1]),
                    5: OrderedSet([0]),
                    6: OrderedSet([1, 8, 10, 12]),
                    7: OrderedSet([0, 9, 11, 13]),
                    8: OrderedSet([6]),
                    9: OrderedSet([7]),
                    10: OrderedSet([6]),
                    11: OrderedSet([7]),
                    12: OrderedSet([6, 14, 16, 18]),
                    13: OrderedSet([7, 15, 17, 19]),
                    14: OrderedSet([12]),
                    15: OrderedSet([13]),
                    16: OrderedSet([12]),
                    17: OrderedSet([13]),
                    18: OrderedSet([12, 20, 22, 25]),
                    19: OrderedSet([13, 21, 23, 24]),
                    20: OrderedSet([18]),
                    21: OrderedSet([19]),
                    22: OrderedSet([18]),
                    23: OrderedSet([19]),
                    24: OrderedSet([19]),
                    25: OrderedSet([18]),
                },
                motifs=OrderedSet([0, 1, 6, 7, 12, 13, 18, 19]),
                bonds_motifs={
                    0: OrderedSet([1, 7]),
                    1: OrderedSet([0, 6]),
                    6: OrderedSet([1, 12]),
                    7: OrderedSet([0, 13]),
                    12: OrderedSet([6, 18]),
                    13: OrderedSet([7, 19]),
                    18: OrderedSet([12]),
                    19: OrderedSet([13]),
                },
                H_atoms=OrderedSet(
                    [2, 3, 4, 5, 8, 9, 10, 11, 14, 15, 16, 17, 20, 21, 22, 23, 24, 25]
                ),
                H_per_motif={
                    0: OrderedSet([3, 5]),
                    1: OrderedSet([2, 4]),
                    6: OrderedSet([8, 10]),
                    7: OrderedSet([9, 11]),
                    12: OrderedSet([14, 16]),
                    13: OrderedSet([15, 17]),
                    18: OrderedSet([20, 22, 25]),
                    19: OrderedSet([21, 23, 24]),
                },
                atoms_per_motif={
                    0: OrderedSet([0, 3, 5]),
                    1: OrderedSet([1, 2, 4]),
                    6: OrderedSet([6, 8, 10]),
                    7: OrderedSet([7, 9, 11]),
                    12: OrderedSet([12, 14, 16]),
                    13: OrderedSet([13, 15, 17]),
                    18: OrderedSet([18, 20, 22, 25]),
                    19: OrderedSet([19, 21, 23, 24]),
                },
                treat_H_different=True,
            ),
            n_BE=6,
        ),
        7: PurelyStructureFragmented(
            mol=mol,
            motifs_per_frag=[OrderedSet([0, 1, 6, 7, 12, 13, 18, 19])],
            centers_per_frag=[OrderedSet([0, 1, 6, 7, 12, 13, 18, 19])],
            edges_per_frag=[OrderedSet()],
            origin_per_frag=[OrderedSet([0])],
            atoms_per_frag=[
                OrderedSet(
                    [
                        0,
                        3,
                        5,
                        1,
                        2,
                        4,
                        6,
                        8,
                        10,
                        7,
                        9,
                        11,
                        12,
                        14,
                        16,
                        13,
                        15,
                        17,
                        18,
                        20,
                        22,
                        25,
                        19,
                        21,
                        23,
                        24,
                    ]
                )
            ],
            frag_idx_per_edge=[{}],
            conn_data=BondConnectivity(
                bonds_atoms={
                    0: OrderedSet([1, 3, 5, 7]),
                    1: OrderedSet([0, 2, 4, 6]),
                    2: OrderedSet([1]),
                    3: OrderedSet([0]),
                    4: OrderedSet([1]),
                    5: OrderedSet([0]),
                    6: OrderedSet([1, 8, 10, 12]),
                    7: OrderedSet([0, 9, 11, 13]),
                    8: OrderedSet([6]),
                    9: OrderedSet([7]),
                    10: OrderedSet([6]),
                    11: OrderedSet([7]),
                    12: OrderedSet([6, 14, 16, 18]),
                    13: OrderedSet([7, 15, 17, 19]),
                    14: OrderedSet([12]),
                    15: OrderedSet([13]),
                    16: OrderedSet([12]),
                    17: OrderedSet([13]),
                    18: OrderedSet([12, 20, 22, 25]),
                    19: OrderedSet([13, 21, 23, 24]),
                    20: OrderedSet([18]),
                    21: OrderedSet([19]),
                    22: OrderedSet([18]),
                    23: OrderedSet([19]),
                    24: OrderedSet([19]),
                    25: OrderedSet([18]),
                },
                motifs=OrderedSet([0, 1, 6, 7, 12, 13, 18, 19]),
                bonds_motifs={
                    0: OrderedSet([1, 7]),
                    1: OrderedSet([0, 6]),
                    6: OrderedSet([1, 12]),
                    7: OrderedSet([0, 13]),
                    12: OrderedSet([6, 18]),
                    13: OrderedSet([7, 19]),
                    18: OrderedSet([12]),
                    19: OrderedSet([13]),
                },
                H_atoms=OrderedSet(
                    [2, 3, 4, 5, 8, 9, 10, 11, 14, 15, 16, 17, 20, 21, 22, 23, 24, 25]
                ),
                H_per_motif={
                    0: OrderedSet([3, 5]),
                    1: OrderedSet([2, 4]),
                    6: OrderedSet([8, 10]),
                    7: OrderedSet([9, 11]),
                    12: OrderedSet([14, 16]),
                    13: OrderedSet([15, 17]),
                    18: OrderedSet([20, 22, 25]),
                    19: OrderedSet([21, 23, 24]),
                },
                atoms_per_motif={
                    0: OrderedSet([0, 3, 5]),
                    1: OrderedSet([1, 2, 4]),
                    6: OrderedSet([6, 8, 10]),
                    7: OrderedSet([7, 9, 11]),
                    12: OrderedSet([12, 14, 16]),
                    13: OrderedSet([13, 15, 17]),
                    18: OrderedSet([18, 20, 22, 25]),
                    19: OrderedSet([19, 21, 23, 24]),
                },
                treat_H_different=True,
            ),
            n_BE=7,
        ),
    }

    assert fragmented == expected
    assert (
        PurelyStructureFragmented.from_mole(mol, n_BE=6).motifs_per_frag
        == PurelyStructureFragmented.from_mole(mol, n_BE=20).motifs_per_frag
    )


def test_hydrogen_chain():
    mol = M(
        atom=[["H", (0.0, 0.0, 0.7 * i)] for i in range(8)],
        basis="sto-3g",
        charge=0.0,
        spin=0.0,
    )

    fragmented = {
        n_BE: PurelyStructureFragmented.from_mole(mol, n_BE, treat_H_different=False)
        for n_BE in range(1, 6)
    }

    expected = {
        1: PurelyStructureFragmented(
            mol=mol,
            motifs_per_frag=[
                OrderedSet([0]),
                OrderedSet([1]),
                OrderedSet([2]),
                OrderedSet([3]),
                OrderedSet([4]),
                OrderedSet([5]),
                OrderedSet([6]),
                OrderedSet([7]),
            ],
            centers_per_frag=[
                OrderedSet([0]),
                OrderedSet([1]),
                OrderedSet([2]),
                OrderedSet([3]),
                OrderedSet([4]),
                OrderedSet([5]),
                OrderedSet([6]),
                OrderedSet([7]),
            ],
            edges_per_frag=[
                OrderedSet(),
                OrderedSet(),
                OrderedSet(),
                OrderedSet(),
                OrderedSet(),
                OrderedSet(),
                OrderedSet(),
                OrderedSet(),
            ],
            origin_per_frag=[
                OrderedSet([0]),
                OrderedSet([1]),
                OrderedSet([2]),
                OrderedSet([3]),
                OrderedSet([4]),
                OrderedSet([5]),
                OrderedSet([6]),
                OrderedSet([7]),
            ],
            atoms_per_frag=[
                OrderedSet([0]),
                OrderedSet([1]),
                OrderedSet([2]),
                OrderedSet([3]),
                OrderedSet([4]),
                OrderedSet([5]),
                OrderedSet([6]),
                OrderedSet([7]),
            ],
            frag_idx_per_edge=[{}, {}, {}, {}, {}, {}, {}, {}],
            conn_data=BondConnectivity(
                bonds_atoms={
                    0: OrderedSet([1]),
                    1: OrderedSet([0, 2]),
                    2: OrderedSet([1, 3]),
                    3: OrderedSet([2, 4]),
                    4: OrderedSet([3, 5]),
                    5: OrderedSet([4, 6]),
                    6: OrderedSet([5, 7]),
                    7: OrderedSet([6]),
                },
                motifs=OrderedSet([0, 1, 2, 3, 4, 5, 6, 7]),
                bonds_motifs={
                    0: OrderedSet([1]),
                    1: OrderedSet([0, 2]),
                    2: OrderedSet([1, 3]),
                    3: OrderedSet([2, 4]),
                    4: OrderedSet([3, 5]),
                    5: OrderedSet([4, 6]),
                    6: OrderedSet([5, 7]),
                    7: OrderedSet([6]),
                },
                H_atoms=OrderedSet(),
                H_per_motif={
                    0: OrderedSet(),
                    1: OrderedSet(),
                    2: OrderedSet(),
                    3: OrderedSet(),
                    4: OrderedSet(),
                    5: OrderedSet(),
                    6: OrderedSet(),
                    7: OrderedSet(),
                },
                atoms_per_motif={
                    0: OrderedSet([0]),
                    1: OrderedSet([1]),
                    2: OrderedSet([2]),
                    3: OrderedSet([3]),
                    4: OrderedSet([4]),
                    5: OrderedSet([5]),
                    6: OrderedSet([6]),
                    7: OrderedSet([7]),
                },
                treat_H_different=False,
            ),
            n_BE=1,
        ),
        2: PurelyStructureFragmented(
            mol=mol,
            motifs_per_frag=[
                OrderedSet([1, 0, 2]),
                OrderedSet([2, 1, 3]),
                OrderedSet([3, 2, 4]),
                OrderedSet([4, 3, 5]),
                OrderedSet([5, 4, 6]),
                OrderedSet([6, 7, 5]),
            ],
            centers_per_frag=[
                OrderedSet([1, 0]),
                OrderedSet([2]),
                OrderedSet([3]),
                OrderedSet([4]),
                OrderedSet([5]),
                OrderedSet([6, 7]),
            ],
            edges_per_frag=[
                OrderedSet([2]),
                OrderedSet([1, 3]),
                OrderedSet([2, 4]),
                OrderedSet([3, 5]),
                OrderedSet([4, 6]),
                OrderedSet([5]),
            ],
            origin_per_frag=[
                OrderedSet([1]),
                OrderedSet([2]),
                OrderedSet([3]),
                OrderedSet([4]),
                OrderedSet([5]),
                OrderedSet([6]),
            ],
            atoms_per_frag=[
                OrderedSet([1, 0, 2]),
                OrderedSet([2, 1, 3]),
                OrderedSet([3, 2, 4]),
                OrderedSet([4, 3, 5]),
                OrderedSet([5, 4, 6]),
                OrderedSet([6, 7, 5]),
            ],
            frag_idx_per_edge=[
                {2: 1},
                {1: 0, 3: 2},
                {2: 1, 4: 3},
                {3: 2, 5: 4},
                {4: 3, 6: 5},
                {5: 4},
            ],
            conn_data=BondConnectivity(
                bonds_atoms={
                    0: OrderedSet([1]),
                    1: OrderedSet([0, 2]),
                    2: OrderedSet([1, 3]),
                    3: OrderedSet([2, 4]),
                    4: OrderedSet([3, 5]),
                    5: OrderedSet([4, 6]),
                    6: OrderedSet([5, 7]),
                    7: OrderedSet([6]),
                },
                motifs=OrderedSet([0, 1, 2, 3, 4, 5, 6, 7]),
                bonds_motifs={
                    0: OrderedSet([1]),
                    1: OrderedSet([0, 2]),
                    2: OrderedSet([1, 3]),
                    3: OrderedSet([2, 4]),
                    4: OrderedSet([3, 5]),
                    5: OrderedSet([4, 6]),
                    6: OrderedSet([5, 7]),
                    7: OrderedSet([6]),
                },
                H_atoms=OrderedSet(),
                H_per_motif={
                    0: OrderedSet(),
                    1: OrderedSet(),
                    2: OrderedSet(),
                    3: OrderedSet(),
                    4: OrderedSet(),
                    5: OrderedSet(),
                    6: OrderedSet(),
                    7: OrderedSet(),
                },
                atoms_per_motif={
                    0: OrderedSet([0]),
                    1: OrderedSet([1]),
                    2: OrderedSet([2]),
                    3: OrderedSet([3]),
                    4: OrderedSet([4]),
                    5: OrderedSet([5]),
                    6: OrderedSet([6]),
                    7: OrderedSet([7]),
                },
                treat_H_different=False,
            ),
            n_BE=2,
        ),
        3: PurelyStructureFragmented(
            mol=mol,
            motifs_per_frag=[
                OrderedSet([2, 0, 1, 3, 4]),
                OrderedSet([3, 1, 2, 4, 5]),
                OrderedSet([4, 2, 3, 5, 6]),
                OrderedSet([5, 6, 7, 3, 4]),
            ],
            centers_per_frag=[
                OrderedSet([2, 0, 1]),
                OrderedSet([3]),
                OrderedSet([4]),
                OrderedSet([5, 6, 7]),
            ],
            edges_per_frag=[
                OrderedSet([3, 4]),
                OrderedSet([1, 2, 4, 5]),
                OrderedSet([2, 3, 5, 6]),
                OrderedSet([3, 4]),
            ],
            origin_per_frag=[
                OrderedSet([2]),
                OrderedSet([3]),
                OrderedSet([4]),
                OrderedSet([5]),
            ],
            atoms_per_frag=[
                OrderedSet([2, 0, 1, 3, 4]),
                OrderedSet([3, 1, 2, 4, 5]),
                OrderedSet([4, 2, 3, 5, 6]),
                OrderedSet([5, 6, 7, 3, 4]),
            ],
            frag_idx_per_edge=[
                {3: 1, 4: 2},
                {1: 0, 2: 0, 4: 2, 5: 3},
                {2: 0, 3: 1, 5: 3, 6: 3},
                {3: 1, 4: 2},
            ],
            conn_data=BondConnectivity(
                bonds_atoms={
                    0: OrderedSet([1]),
                    1: OrderedSet([0, 2]),
                    2: OrderedSet([1, 3]),
                    3: OrderedSet([2, 4]),
                    4: OrderedSet([3, 5]),
                    5: OrderedSet([4, 6]),
                    6: OrderedSet([5, 7]),
                    7: OrderedSet([6]),
                },
                motifs=OrderedSet([0, 1, 2, 3, 4, 5, 6, 7]),
                bonds_motifs={
                    0: OrderedSet([1]),
                    1: OrderedSet([0, 2]),
                    2: OrderedSet([1, 3]),
                    3: OrderedSet([2, 4]),
                    4: OrderedSet([3, 5]),
                    5: OrderedSet([4, 6]),
                    6: OrderedSet([5, 7]),
                    7: OrderedSet([6]),
                },
                H_atoms=OrderedSet(),
                H_per_motif={
                    0: OrderedSet(),
                    1: OrderedSet(),
                    2: OrderedSet(),
                    3: OrderedSet(),
                    4: OrderedSet(),
                    5: OrderedSet(),
                    6: OrderedSet(),
                    7: OrderedSet(),
                },
                atoms_per_motif={
                    0: OrderedSet([0]),
                    1: OrderedSet([1]),
                    2: OrderedSet([2]),
                    3: OrderedSet([3]),
                    4: OrderedSet([4]),
                    5: OrderedSet([5]),
                    6: OrderedSet([6]),
                    7: OrderedSet([7]),
                },
                treat_H_different=False,
            ),
            n_BE=3,
        ),
        4: PurelyStructureFragmented(
            mol=mol,
            motifs_per_frag=[
                OrderedSet([3, 0, 1, 2, 4, 5, 6]),
                OrderedSet([4, 5, 6, 7, 1, 2, 3]),
            ],
            centers_per_frag=[OrderedSet([3, 0, 1, 2]), OrderedSet([4, 5, 6, 7])],
            edges_per_frag=[OrderedSet([4, 5, 6]), OrderedSet([1, 2, 3])],
            origin_per_frag=[OrderedSet([3]), OrderedSet([4])],
            atoms_per_frag=[
                OrderedSet([3, 0, 1, 2, 4, 5, 6]),
                OrderedSet([4, 5, 6, 7, 1, 2, 3]),
            ],
            frag_idx_per_edge=[{4: 1, 5: 1, 6: 1}, {1: 0, 2: 0, 3: 0}],
            conn_data=BondConnectivity(
                bonds_atoms={
                    0: OrderedSet([1]),
                    1: OrderedSet([0, 2]),
                    2: OrderedSet([1, 3]),
                    3: OrderedSet([2, 4]),
                    4: OrderedSet([3, 5]),
                    5: OrderedSet([4, 6]),
                    6: OrderedSet([5, 7]),
                    7: OrderedSet([6]),
                },
                motifs=OrderedSet([0, 1, 2, 3, 4, 5, 6, 7]),
                bonds_motifs={
                    0: OrderedSet([1]),
                    1: OrderedSet([0, 2]),
                    2: OrderedSet([1, 3]),
                    3: OrderedSet([2, 4]),
                    4: OrderedSet([3, 5]),
                    5: OrderedSet([4, 6]),
                    6: OrderedSet([5, 7]),
                    7: OrderedSet([6]),
                },
                H_atoms=OrderedSet(),
                H_per_motif={
                    0: OrderedSet(),
                    1: OrderedSet(),
                    2: OrderedSet(),
                    3: OrderedSet(),
                    4: OrderedSet(),
                    5: OrderedSet(),
                    6: OrderedSet(),
                    7: OrderedSet(),
                },
                atoms_per_motif={
                    0: OrderedSet([0]),
                    1: OrderedSet([1]),
                    2: OrderedSet([2]),
                    3: OrderedSet([3]),
                    4: OrderedSet([4]),
                    5: OrderedSet([5]),
                    6: OrderedSet([6]),
                    7: OrderedSet([7]),
                },
                treat_H_different=False,
            ),
            n_BE=4,
        ),
        5: PurelyStructureFragmented(
            mol=mol,
            motifs_per_frag=[OrderedSet([3, 0, 1, 2, 4, 5, 6, 7])],
            centers_per_frag=[OrderedSet([3, 0, 1, 2, 4, 5, 6, 7])],
            edges_per_frag=[OrderedSet()],
            origin_per_frag=[OrderedSet([3])],
            atoms_per_frag=[OrderedSet([3, 0, 1, 2, 4, 5, 6, 7])],
            frag_idx_per_edge=[{}],
            conn_data=BondConnectivity(
                bonds_atoms={
                    0: OrderedSet([1]),
                    1: OrderedSet([0, 2]),
                    2: OrderedSet([1, 3]),
                    3: OrderedSet([2, 4]),
                    4: OrderedSet([3, 5]),
                    5: OrderedSet([4, 6]),
                    6: OrderedSet([5, 7]),
                    7: OrderedSet([6]),
                },
                motifs=OrderedSet([0, 1, 2, 3, 4, 5, 6, 7]),
                bonds_motifs={
                    0: OrderedSet([1]),
                    1: OrderedSet([0, 2]),
                    2: OrderedSet([1, 3]),
                    3: OrderedSet([2, 4]),
                    4: OrderedSet([3, 5]),
                    5: OrderedSet([4, 6]),
                    6: OrderedSet([5, 7]),
                    7: OrderedSet([6]),
                },
                H_atoms=OrderedSet(),
                H_per_motif={
                    0: OrderedSet(),
                    1: OrderedSet(),
                    2: OrderedSet(),
                    3: OrderedSet(),
                    4: OrderedSet(),
                    5: OrderedSet(),
                    6: OrderedSet(),
                    7: OrderedSet(),
                },
                atoms_per_motif={
                    0: OrderedSet([0]),
                    1: OrderedSet([1]),
                    2: OrderedSet([2]),
                    3: OrderedSet([3]),
                    4: OrderedSet([4]),
                    5: OrderedSet([5]),
                    6: OrderedSet([6]),
                    7: OrderedSet([7]),
                },
                treat_H_different=False,
            ),
            n_BE=5,
        ),
    }

    assert fragmented == expected


def test_agreement_with_autogen():
    mol = M("data/octane.xyz")

    for n_BE in range(1, 4):
        chem_frags = PurelyStructureFragmented.from_mole(mol, n_BE)
        auto_frags = fragpart(mol=mol, frag_type="autogen", be_type=f"be{n_BE}")

        for chem_fragment, auto_fragment in zip(
            chem_frags.motifs_per_frag, auto_frags.Frag_atom
        ):
            # We assert that the first atom, i.e. the origin, is the same for both
            # chemfrag and autogen
            assert chem_fragment[0] == auto_fragment[0]
            # For the rest of the atoms the order can be different,
            # so we assert set equality
            assert set(chem_fragment) == set(auto_fragment)


def test_conn_data_manipulation_of_vdW():
    m = Cartesian.read_xyz("data/octane.xyz")

    # if hydrogens are shared among motifs we cannot treat H differently
    with pytest.raises(ValueError):
        conn_data = BondConnectivity.from_cartesian(m, vdW_radius=100)
        conn_data = BondConnectivity.from_cartesian(m, vdW_radius=lambda r: r * 100)
        conn_data = BondConnectivity.from_cartesian(m, vdW_radius={"C": 100})

    conn_data = BondConnectivity.from_cartesian(
        m, vdW_radius=100, treat_H_different=False
    )
    for atom, connected in conn_data.bonds_atoms.items():
        # check if everything is connected to everything
        assert {atom} | connected == set(m.index)

    conn_data = BondConnectivity.from_cartesian(
        m, vdW_radius=lambda r: r * 100, treat_H_different=False
    )
    for atom, connected in conn_data.bonds_atoms.items():
        # check if everything is connected to everything
        assert {atom} | connected == set(m.index)

    conn_data = BondConnectivity.from_cartesian(
        m, vdW_radius={"C": 100}, treat_H_different=False
    )
    for i_carbon in m.loc[m.atom == "C"].index:
        # check if carbons are connected to everything
        assert {i_carbon} | conn_data.bonds_atoms[i_carbon] == set(m.index)
