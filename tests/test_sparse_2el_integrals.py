from quemb.molbe.sparse_2el_integral import SparseInt2


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
