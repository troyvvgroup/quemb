import numpy as np
import pytest
import scipy
from chemcoord import Cartesian
from pyscf import df, scf
from pyscf.df import make_auxmol
from pyscf.gto import M
from pyscf.lib import einsum

from quemb.molbe import BE, fragmentate
from quemb.molbe.eri_sparse_DF import (
    _invert_dict,
    find_screening_radius,
    get_atom_per_AO,
    get_atom_per_MO,
    get_dense_integrals,
    get_reachable,
    get_screened,
    get_sparse_D_ints_and_coeffs,
    get_sparse_P_mu_nu,
    traverse_nonzero,
)
from quemb.shared.helper import clean_overlap, get_calling_function_name

from ._expected_data_for_eri_sparse_DF import get_expected

expected = get_expected()


def test_semi_sparse_3d_tensor() -> None:
    m = Cartesian.read_xyz("data/octane.xyz")
    basis = "sto-3g"
    auxbasis = "weigend"
    mol = m.to_pyscf(basis=basis, charge=0)
    auxmol = df.addons.make_auxmol(mol, auxbasis)

    atom_per_AO = get_atom_per_AO(mol)
    exch_reachable = get_reachable(
        atom_per_AO, atom_per_AO, get_screened(mol, find_screening_radius(mol, auxmol))
    )
    sparse_ints_3c2e = get_sparse_P_mu_nu(mol, auxmol, exch_reachable)

    ints_3c2e = df.incore.aux_e2(mol, auxmol, intor="int3c2e")

    for p, q in traverse_nonzero(sparse_ints_3c2e):
        assert np.allclose(
            ints_3c2e[p, q, :], sparse_ints_3c2e[p, q], atol=1e-10, rtol=0
        ), (p, q)


def test_sparse_density_fitting() -> None:
    m = Cartesian.read_xyz("data/octane.xyz")
    basis = "sto-3g"
    auxbasis = "weigend"
    mol = m.to_pyscf(basis=basis, charge=0)
    auxmol = df.addons.make_auxmol(mol, auxbasis)
    nao = mol.nao
    naux = auxmol.nao

    sparse_ints_3c2e, sparse_df_coef = get_sparse_D_ints_and_coeffs(
        mol, auxmol, find_screening_radius(mol, auxmol, threshold=1e-11)
    )

    ints_3c2e = df.incore.aux_e2(mol, auxmol, intor="int3c2e")
    ints_2c2e = auxmol.intor("int2c2e")
    df_coef = scipy.linalg.solve(ints_2c2e, ints_3c2e.reshape(nao * nao, naux).T)
    df_coef = df_coef.reshape((naux, nao, nao), order="F")

    df_eri = einsum("ijP,Pkl->ijkl", ints_3c2e, df_coef)
    df_eri_from_sparse = get_dense_integrals(sparse_ints_3c2e, sparse_df_coef)

    assert np.abs(df_eri_from_sparse - df_eri).max() < 1e-10


def test_find_screening_radius() -> None:
    mol = M("./data/octane.xyz", basis="cc-pvdz", charge=0)
    auxmol = make_auxmol(mol, auxbasis="cc-pvdz-jkfit")

    assert find_screening_radius(mol, auxmol, threshold=1e-8) == {
        "H": 4.4548883056640625,
        "C": 4.430244750976563,
    }

    assert find_screening_radius(mol, threshold=1e-8) == {
        "C": 3.4198590087890626,
        "H": 3.2658367919921876,
    }


def test_sparse_DF_BE() -> None:
    mol = M("data/octane.xyz", basis="sto-3g", cart=True)

    mf = scf.RHF(mol)
    mf.kernel()

    fobj = fragmentate(frag_type="chemgen", n_BE=2, mol=mol, print_frags=False)
    sparse_DF_BE = BE(mf, fobj, auxbasis="weigend", int_transform="sparse-DF-cpp")
    sparse_DF_BE.oneshot(solver="CCSD")

    assert np.isclose(
        sparse_DF_BE.ebe_tot - sparse_DF_BE.ebe_hf,
        -0.5499817066329342,
        atol=1e-10,
        rtol=0,
    ), sparse_DF_BE.ebe_tot - sparse_DF_BE.ebe_hf

    mol = M("data/octane.xyz", basis="sto-3g", cart=False)

    mf = scf.RHF(mol)
    mf.kernel()

    fobj = fragmentate(frag_type="chemgen", n_BE=2, mol=mol, print_frags=False)
    sparse_DF_BE = BE(mf, fobj, auxbasis="weigend", int_transform="sparse-DF-cpp")
    sparse_DF_BE.oneshot(solver="CCSD")

    assert np.isclose(
        sparse_DF_BE.ebe_tot - sparse_DF_BE.ebe_hf,
        -0.5498957862386646,
        atol=1e-10,
        rtol=0,
    ), sparse_DF_BE.ebe_tot - sparse_DF_BE.ebe_hf


def test_invert_dict() -> None:
    X = {0: {1, 2}, 1: {2, 3, 4}}
    expected = {1: {0}, 2: {0, 1}, 3: {1}, 4: {1}}
    assert _invert_dict(X) == expected


def test_MO_screening(ikosan) -> None:
    mol, auxmol, mf, fobj, my_be = ikosan

    atom_per_AO = get_atom_per_AO(mol)

    screened = get_screened(mol, find_screening_radius(mol, auxmol))

    SchmidtMO_reachable_per_AO_per_frag = [
        get_reachable(
            atom_per_AO,
            get_atom_per_MO(atom_per_AO, TA, epsilon=1e-8),
            screened,
        )
        for TA in (fobj.TA for fobj in my_be.Fobjs)
    ]

    AO_reachable_per_SchmidtMO_per_frag = [
        get_reachable(
            get_atom_per_MO(atom_per_AO, TA, epsilon=1e-8),
            atom_per_AO,
            screened,
        )
        for TA in (fobj.TA for fobj in my_be.Fobjs)
    ]

    assert (
        SchmidtMO_reachable_per_AO_per_frag
        == expected[get_calling_function_name()]["SchmidtMO_reachable_per_AO"]
    )
    assert (
        AO_reachable_per_SchmidtMO_per_frag
        == expected[get_calling_function_name()]["AO_reachable_per_SchmidtMO"]
    )

    for MO_reachable_by_AO, AO_reachable_by_MO in zip(
        SchmidtMO_reachable_per_AO_per_frag, AO_reachable_per_SchmidtMO_per_frag
    ):
        for i_AO in MO_reachable_by_AO:
            for i_MO in MO_reachable_by_AO[i_AO]:
                assert i_AO in AO_reachable_by_MO[i_MO]

        for i_MO in AO_reachable_by_MO:
            for i_AO in AO_reachable_by_MO[i_MO]:
                assert i_MO in MO_reachable_by_AO[i_AO]


def test_reuse_schmidt_fragment_MOs(ikosan) -> None:
    mol, auxmol, mf, fobj, my_be = ikosan

    S = mol.intor("int1e_ovlp")
    for fobj in my_be.Fobjs:
        assert (
            clean_overlap(
                my_be.all_fragment_MO_TA[:, fobj.frag_TA_offset].T
                @ S
                @ fobj.TA[:, : fobj.n_f]
            )
            == np.eye(fobj.n_f)
        ).all()


@pytest.fixture(scope="session")
def ikosan():
    mol = M("xyz/E-polyacetylene/20.xyz", basis="sto-3g")
    auxbasis = "weigend"
    auxmol = make_auxmol(mol, auxbasis=auxbasis)

    mf = scf.RHF(mol)
    mf.kernel()

    fobj = fragmentate(frag_type="chemgen", n_BE=2, mol=mol, print_frags=False)
    my_be = BE(mf, fobj, auxbasis=auxbasis, int_transform="int-direct-DF")
    return mol, auxmol, mf, fobj, my_be
