from quemb.shared.helper import ravel_eri_idx, unravel_eri_idx


def test_invert_idx():
    n_orbitals = 10

    for a in range(n_orbitals):
        for b in range(a + 1):
            for c in range(a + 1):
                for d in range(c + 1 if a > c else b + 1):
                    assert (a, b, c, d) == unravel_eri_idx(ravel_eri_idx(a, b, c, d))
