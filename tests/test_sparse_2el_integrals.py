import numpy as np
from chemcoord import Cartesian
from pyscf import df

from quemb.molbe.sparse_2el_integral import (
    SparseInt2,
    get_sparse_ints_3c2e,
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


def test_mutable_semi_sparse_3d_tensor() -> None:
    m = Cartesian.read_xyz("data/octane.xyz")
    basis = "cc-pvdz"
    auxbasis = "cc-pvdz-jkfit"
    mol = m.to_pyscf(basis=basis, charge=0)
    auxmol = df.addons.make_auxmol(mol, auxbasis)
    sparse_ints_3c2e = get_sparse_ints_3c2e(mol, auxmol)

    ints_3c2e = df.incore.aux_e2(mol, auxmol, intor="int3c2e")

    for p, q in sparse_ints_3c2e.traverse_nonzero():
        assert np.allclose(
            ints_3c2e[p, q, :], sparse_ints_3c2e[p, q], atol=1e-10, rtol=0
        ), (p, q)

    const_sparse_ints_3c2e = sparse_ints_3c2e.make_immutable()
    for p, q in sparse_ints_3c2e.traverse_nonzero():
        assert np.allclose(
            const_sparse_ints_3c2e[p, q], sparse_ints_3c2e[p, q], atol=1e-10, rtol=0
        ), (p, q)
