import inspect

import numpy as np
import pytest
from chemcoord import Cartesian
from ordered_set import OrderedSet
from pyscf import scf
from pyscf.gto import M

from quemb.molbe.chemfrag import (
    BondConnectivity,
    Fragmented,
    PurelyStructureFragmented,
    _cleanup_if_subset,
)
from quemb.molbe.fragment import fragmentate
from quemb.molbe.mbe import BE

from ._expected_data_for_chemfrag import get_expected


def get_calling_function_name() -> str:
    """Do stack inspection shenanigan to obtain the name
    of the calling function"""
    return inspect.stack()[1][3]


expected = get_expected()


def test_connectivity_data():
    m = Cartesian.read_xyz("data/octane.xyz")

    assert get_calling_function_name() == "test_connectivity_data"

    conn_data = BondConnectivity.from_cartesian(m)

    assert conn_data == expected[get_calling_function_name()]["octane.xyz"]

    # sort carbon atoms first and then by y coordinate,
    # i.e. the visual order of the atoms in the molecule
    resorted_conn_data = BondConnectivity.from_cartesian(
        m.sort_values(by=["atom", "y"]).reset_index()
    )

    assert (
        resorted_conn_data == expected[get_calling_function_name()]["resorted_octane"]
    )


def test_fragment_generation():
    """This does not yet test the cleanup of subsets, i.e. also fragments
    that are fully contained in others are returned."""

    m = Cartesian.read_xyz("data/octane.xyz")

    fragments = {
        n_BE: BondConnectivity.from_cartesian(m).get_all_BE_fragments(n_BE)
        for n_BE in range(1, 9)
    }

    assert fragments == expected[get_calling_function_name()]["octane.xyz"]


def test_cleaned_fragments():
    """In this test fragments that are subsets of others are also removed.
    It is also tested if the assignment of an origin, i.e. the center with the
    lowest index whose fragment is a superset of other fragments, reliably works.
    """
    m = Cartesian.read_xyz("data/octane.xyz")

    cleaned_fragments = {
        n_BE: _cleanup_if_subset(
            BondConnectivity.from_cartesian(m).get_all_BE_fragments(n_BE)
        )
        for n_BE in range(1, 9)
    }

    assert cleaned_fragments == expected[get_calling_function_name()]["octane.xyz"]


def test_pure_structure_fragmented():
    m = Cartesian.read_xyz("data/octane.xyz")
    mol = m.to_pyscf()

    fragmented = {
        n_BE: PurelyStructureFragmented.from_mole(mol, n_BE=n_BE)
        for n_BE in range(1, 8)
    }

    for fragment in fragmented.values():
        assert fragment.is_ordered()

    assert fragmented == expected[get_calling_function_name()]["octane.xyz"]
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

    assert fragmented == expected[get_calling_function_name()]["H8"]


def test_structure_agreement_with_autogen():
    mol = M("data/octane.xyz")

    for n_BE in range(1, 4):
        chem_frags = PurelyStructureFragmented.from_mole(mol, n_BE)
        auto_frags = fragmentate(mol=mol, frag_type="autogen", n_BE=n_BE)

        for chem_fragment, auto_fragment in zip(
            chem_frags.motifs_per_frag, auto_frags.Frag_atom
        ):
            # We assert that the first atom, i.e. the origin, is the same for both
            # chemfrag and autogen
            assert chem_fragment[0] == auto_fragment[0]
            # For the rest of the atoms the order can be different,
            # so we assert set equality
            assert set(chem_fragment) == set(auto_fragment)


def test_AO_indexing():
    octane_cart = Cartesian.read_xyz("data/octane.xyz")
    bases = [
        ("sto-3g", None),
        ("cc-pvdz", None),
        ("cc-pvdz", "sto-3g"),
    ]

    result = {
        (n_BE, basis, iao_valence_basis, frozen_core): Fragmented.from_mole(
            mol=octane_cart.to_pyscf(basis=basis),
            iao_valence_basis=iao_valence_basis,
            n_BE=n_BE,
            frozen_core=frozen_core,
        )
        for n_BE in range(1, 5)
        for basis, iao_valence_basis in bases
        for frozen_core in [True, False]
    }

    assert result == expected[get_calling_function_name()]["octane.xyz"]


def test_match_autogen_output():
    m = Cartesian.read_xyz("data/octane.xyz")
    bases = [
        ("sto-3g", None),
        ("cc-pvdz", None),
        ("cc-pvdz", "sto-3g"),
    ]

    calculated = {
        (
            n_BE,
            basis,
            iao_valence_basis,
            frozen_core,
            wrong_iao_indexing,
        ): Fragmented.from_mole(
            mol=m.to_pyscf(basis=basis),
            iao_valence_basis=iao_valence_basis,
            n_BE=n_BE,
            frozen_core=frozen_core,
        ).get_FragPart(wrong_iao_indexing=wrong_iao_indexing)
        for n_BE in range(1, 5)
        for basis, iao_valence_basis in bases
        for frozen_core in [True, False]
        for wrong_iao_indexing in [True, False]
    }
    for k, result in calculated.items():
        assert result == expected["test_match_autogen_output"]["octane.xyz"][k], k


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


def test_molecule_with_autocratic_matching():
    """This test was introduced because of https://github.com/troyvvgroup/quemb/issues/132
    and ensures that shared centers
    are autocratically assigned to one fragment correctly.
    """
    m = (
        Cartesian.read_xyz("xyz/short_polypropylene.xyz")
        .sort_values(by=["atom", "x", "y"])
        .reset_index()
    )
    mol = m.to_pyscf(basis="sto-3g")
    mf = scf.RHF(mol)
    mf.kernel()

    fobj = fragmentate(mol, n_BE=2, frag_type="chemgen", print_frags=False)
    mybe = BE(mf, fobj)

    assert np.isclose(mf.e_tot, mybe.ebe_hf)

    fobj = fragmentate(mol, n_BE=3, frag_type="chemgen", print_frags=False)
    mybe = BE(mf, fobj)

    assert np.isclose(mf.e_tot, mybe.ebe_hf)


def test_shared_centers():
    """Test the identification of shared centers and if errors are correctly raised if
    centers are unexpectedly shared."""

    m = Cartesian.read_xyz("xyz/short_polypropylene.xyz")
    mol = m.to_pyscf(basis="sto-3g")
    fragments = PurelyStructureFragmented.from_mole(mol, 3, autocratic_matching=False)

    assert fragments._get_shared_centers() == {
        2: OrderedSet([0, 1]),
        7: OrderedSet([3, 4, 5]),
    }
    assert fragments.shared_centers_exist()

    assert not fragments.get_autocratically_matched().shared_centers_exist()
    assert fragments.get_autocratically_matched()._get_shared_centers() == {}

    Fragmented.from_mole(mol, 3, autocratic_matching=True)
    with pytest.raises(ValueError):
        Fragmented.from_mole(mol, 3, autocratic_matching=False)
