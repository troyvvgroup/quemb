import numpy as np
import pytest
from pyscf import scf
from pyscf.df.addons import make_auxmol
from pyscf.gto import M

from quemb.molbe import BE, fragmentate
from quemb.molbe.eri_sparse_DF import (
    _get_AO_per_AO,
    _invert_dict,
    approx_S_abs,
    get_sparse_P_mu_nu,
)
from quemb.molbe.mbe import IntTransforms
from quemb.shared.helper import clean_overlap

from ._expected_data_for_eri_sparse_DF import get_expected

expected = get_expected()


@pytest.fixture(scope="module", params=[True, False], ids=["cart", "sph"])
def octane(request):
    """Octane mean field and fragments, in cartesian and spherical AOs."""
    mol = M("xyz/octane.xyz", basis="sto-3g", cart=request.param)

    mf = scf.RHF(mol)
    mf.kernel()

    fobj = fragmentate(frag_type="chemgen", n_BE=2, mol=mol, print_frags=False)
    return mf, fobj


@pytest.fixture(scope="module")
def octane_int_direct_DF(octane):
    """Octane mean field, fragments, and the int-direct-DF correlation energy.

    int-direct-DF uses the same auxiliary basis without any screening,
    hence it is the exact reference for the sparse DF transformations.
    """
    mf, fobj = octane
    ref_BE = BE(mf, fobj, auxbasis="weigend", int_transform="int-direct-DF")
    ref_BE.oneshot(solver="CCSD")
    return mf, fobj, ref_BE.ebe_tot - ref_BE.ebe_hf


@pytest.mark.parametrize("int_transform", ["sparse-DF", "on-fly-sparse-DF"])
def test_sparse_DF_BE(octane_int_direct_DF, int_transform: IntTransforms) -> None:
    mf, fobj, e_corr_ref = octane_int_direct_DF

    sparse_DF_BE = BE(mf, fobj, auxbasis="weigend", int_transform=int_transform)
    sparse_DF_BE.oneshot(solver="CCSD")
    e_corr = sparse_DF_BE.ebe_tot - sparse_DF_BE.ebe_hf

    assert np.isclose(e_corr, e_corr_ref, atol=1e-10, rtol=0), e_corr - e_corr_ref


@pytest.mark.parametrize("int_transform", ["sparse-DF", "on-fly-sparse-DF"])
def test_screened_sparse_DF_BE(octane, int_transform: IntTransforms) -> None:
    """Sparse DF with thresholds that truncate (P | mu nu) and the AOs per MO.

    The deviation from int-direct-DF is about 1e-6 Hartree, so a change of the
    screening is detected.
    Precomputed and on-the-fly (P | mu nu) have to agree, because for
    MO_coeff_epsilon >= AO_coeff_epsilon both keep the same AO pairs and AOs per MO.
    """
    mf, fobj = octane
    e_expected_by_cart = {True: -0.5499700118942314, False: -0.549884101392422}

    sparse_DF_BE = BE(
        mf,
        fobj,
        auxbasis="weigend",
        int_transform=int_transform,
        AO_coeff_epsilon=1e-4,
        MO_coeff_epsilon=1e-3,
    )
    sparse_DF_BE.oneshot(solver="CCSD")
    e_corr = sparse_DF_BE.ebe_tot - sparse_DF_BE.ebe_hf

    e_expected = e_expected_by_cart[mf.mol.cart]
    assert np.isclose(e_corr, e_expected, atol=1e-10, rtol=0), e_corr - e_expected


def test_invert_dict() -> None:
    X = {0: {1, 2}, 1: {2, 3, 4}}
    expected = {1: {0}, 2: {0, 1}, 3: {1}, 4: {1}}
    assert _invert_dict(X) == expected


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


def test_uncontracted_basis():
    mol = M("xyz/E-polyacetylene/8.xyz", basis="def2-svp", cart=True)
    auxmol = make_auxmol(mol, auxbasis="def2-svp-jkfit")
    S_abs = approx_S_abs(mol)
    exch_reachable = _get_AO_per_AO(S_abs, 1e-10)

    P_mu_nu = get_sparse_P_mu_nu(
        mol,
        auxmol,
        exch_reachable,
    )

    ref = np.loadtxt("data/P_mu_nu_ref.npy")
    assert np.allclose(P_mu_nu[10, 13], ref)
