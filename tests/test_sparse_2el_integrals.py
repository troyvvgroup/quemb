import numpy as np
import scipy
from chemcoord import Cartesian
from pyscf import df
from pyscf.df import make_auxmol
from pyscf.gto import M
from pyscf.lib import einsum

from quemb.molbe.eri_sparse_DF import (
    SparseInt2,
    _get_sparse_ints_3c2e,
    find_screening_radius,
    get_dense_integrals,
    get_sparse_DF_integrals,
    traverse_nonzero,
)


def test_basic_indexing() -> None:
    g = SparseInt2()
    g[1, 2, 3, 4] = 3

    # test all possible permutations
    assert g[1, 2, 3, 4] == 3
    assert g[1, 2, 4, 3] == 3
    assert g[2, 1, 3, 4] == 3
    assert g[2, 1, 4, 3] == 3
    assert g[3, 4, 1, 2] == 3
    assert g[4, 3, 1, 2] == 3
    assert g[3, 4, 2, 1] == 3
    assert g[4, 3, 2, 1] == 3

    assert g[1, 2, 3, 10] == 0


def test_semi_sparse_3d_tensor() -> None:
    m = Cartesian.read_xyz("data/octane.xyz")
    basis = "sto-3g"
    auxbasis = "weigend"
    mol = m.to_pyscf(basis=basis, charge=0)
    auxmol = df.addons.make_auxmol(mol, auxbasis)
    sparse_ints_3c2e = _get_sparse_ints_3c2e(mol, auxmol)

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

    sparse_ints_3c2e, sparse_df_coef = get_sparse_DF_integrals(
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
