import numpy as np
import pytest
from pyscf import scf
from pyscf.df import make_auxmol
from pyscf.gto import M

from quemb.molbe import BE, fragmentate
from quemb.molbe.eri_sparse_DF import _invert_dict
from quemb.shared.helper import clean_overlap

from ._expected_data_for_eri_sparse_DF import get_expected

expected = get_expected()


def test_sparse_DF_BE() -> None:
    mol = M("xyz/octane.xyz", basis="sto-3g", cart=True)

    mf = scf.RHF(mol)
    mf.kernel()

    fobj = fragmentate(frag_type="chemgen", n_BE=2, mol=mol, print_frags=False)
    sparse_DF_BE = BE(mf, fobj, auxbasis="weigend", int_transform="sparse-DF")
    sparse_DF_BE.oneshot(solver="CCSD")

    assert np.isclose(
        sparse_DF_BE.ebe_tot - sparse_DF_BE.ebe_hf,
        -0.5499707624383632,
        atol=1e-10,
        rtol=0,
    ), sparse_DF_BE.ebe_tot - sparse_DF_BE.ebe_hf

    mol = M("xyz/octane.xyz", basis="sto-3g", cart=False)

    mf = scf.RHF(mol)
    mf.kernel()

    fobj = fragmentate(frag_type="chemgen", n_BE=2, mol=mol, print_frags=False)
    sparse_DF_BE = BE(mf, fobj, auxbasis="weigend", int_transform="sparse-DF")
    sparse_DF_BE.oneshot(solver="CCSD")

    assert np.isclose(
        sparse_DF_BE.ebe_tot - sparse_DF_BE.ebe_hf,
        -0.5498858656383732,
        atol=1e-10,
        rtol=0,
    ), sparse_DF_BE.ebe_tot - sparse_DF_BE.ebe_hf


def test_on_the_fly_sparse_DF_BE() -> None:
    mol = M("xyz/octane.xyz", basis="sto-3g", cart=True)

    mf = scf.RHF(mol)
    mf.kernel()

    fobj = fragmentate(frag_type="chemgen", n_BE=2, mol=mol, print_frags=False)
    sparse_DF_BE = BE(mf, fobj, auxbasis="weigend", int_transform="on-fly-sparse-DF")
    sparse_DF_BE.oneshot(solver="CCSD")

    assert np.isclose(
        sparse_DF_BE.ebe_tot - sparse_DF_BE.ebe_hf,
        -0.5499707624383632,
        atol=1e-10,
        rtol=0,
    ), sparse_DF_BE.ebe_tot - sparse_DF_BE.ebe_hf

    mol = M("xyz/octane.xyz", basis="sto-3g", cart=False)

    mf = scf.RHF(mol)
    mf.kernel()

    fobj = fragmentate(frag_type="chemgen", n_BE=2, mol=mol, print_frags=False)
    sparse_DF_BE = BE(mf, fobj, auxbasis="weigend", int_transform="on-fly-sparse-DF")
    sparse_DF_BE.oneshot(solver="CCSD")

    assert np.isclose(
        sparse_DF_BE.ebe_tot - sparse_DF_BE.ebe_hf,
        -0.5498858656383732,
        atol=1e-10,
        rtol=0,
    ), sparse_DF_BE.ebe_tot - sparse_DF_BE.ebe_hf


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
