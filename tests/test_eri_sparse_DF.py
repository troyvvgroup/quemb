from typing import cast

import numpy as np
import pytest
from pyscf import scf
from pyscf.df.addons import make_auxmol
from pyscf.gto import M

import quemb.molbe._cpp.eri_sparse_DF as cpp
from quemb.molbe import BE, fragmentate
from quemb.molbe.eri_sparse_DF import (
    _cholesky_PQ,
    _compute_P_mu_nu_columns,
    _get_AO_per_AO,
    _get_pair_keys,
    _get_used_AOs,
    _invert_dict,
    _order_by_overlap,
    _plan_window,
    approx_S_abs,
    get_sparse_P_mu_nu,
)
from quemb.molbe.mbe import IntTransforms
from quemb.shared.helper import clean_overlap, ravel_symmetric
from quemb.shared.typing import AOIdx, Vector

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


@pytest.mark.parametrize("int_transform", ["sparse-DF", "on-fly-sparse-DF"])
def test_sparse_DF_BE(octane, int_transform: IntTransforms) -> None:
    """Sparse DF with the default screening thresholds.

    The deviation from int-direct-DF is about 2e-9 Hartree.
    """
    mf, fobj = octane
    e_expected_by_cart = {True: -0.5499707353683903, False: -0.5498848219573347}

    sparse_DF_BE = BE(mf, fobj, auxbasis="weigend", int_transform=int_transform)
    sparse_DF_BE.oneshot(solver="CCSD")
    e_corr = sparse_DF_BE.ebe_tot - sparse_DF_BE.ebe_hf

    e_expected = e_expected_by_cart[mf.mol.cart]
    assert np.isclose(e_corr, e_expected, atol=1e-10, rtol=0), e_corr - e_expected


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


@pytest.mark.parametrize("MO_coeff_epsilon", [1e-12, 1e-3])
def test_contract_with_TA_gemm(octane, MO_coeff_epsilon: float) -> None:
    """The GEMM variants of both contractions with TA agree with the original loops.

    MO_coeff_epsilon = 1e-3 truncates the AOs per MO, so that the per-mu and per-i
    blocks have different sizes.
    """
    mf, fobj = octane
    mol = mf.mol
    fragments = BE(mf, fobj, auxbasis="weigend", int_transform="int-direct-DF").Fobjs

    S_abs = approx_S_abs(mol)
    P_mu_nu = get_sparse_P_mu_nu(
        mol, make_auxmol(mol, "weigend"), _get_AO_per_AO(S_abs, 1e-8)
    )
    for fragobj in fragments:
        TA = fragobj.TA
        AO_by_MO = cpp.get_AO_per_MO(TA, S_abs, MO_coeff_epsilon)

        P_mu_i = cpp.contract_with_TA_1st(TA, P_mu_nu, AO_by_MO)
        P_mu_i_gemm = cpp.contract_with_TA_1st_gemm(TA, P_mu_nu, AO_by_MO)
        assert np.allclose(
            P_mu_i_gemm.dense_data, P_mu_i.dense_data, atol=1e-12, rtol=0
        )

        P_ij = cpp.contract_with_TA_2nd_to_sym_dense(P_mu_i, TA)
        P_ij_gemm = cpp.contract_with_TA_2nd_gemm(P_mu_i_gemm, TA)
        assert P_ij_gemm.shape == P_ij.shape
        assert np.allclose(P_ij_gemm, P_ij, atol=1e-12, rtol=0)


def test_pair_keys() -> None:
    """Pairs needed for TA: every pair of an AO used by an MO, once as mu >= nu."""
    mol = M("xyz/E-polyacetylene/20.xyz", basis="sto-3g")
    S_abs = approx_S_abs(mol)
    # an MO on the first two AOs only
    TA = np.zeros((mol.nao, 1))
    TA[:2, 0] = 1.0
    AO_eps, MO_eps = 1e-6, 1e-2

    used = _get_used_AOs(S_abs, TA, MO_eps)
    keys = _get_pair_keys(S_abs, AO_eps, used)
    full = _get_AO_per_AO(S_abs, AO_eps)

    used_AOs = cast(list[AOIdx], used.nonzero()[0].tolist())
    expected = sorted(
        {int(ravel_symmetric(mu, nu)) for mu in used_AOs for nu in full[mu]}
    )
    assert 0 < used.sum() < mol.nao
    assert keys.tolist() == expected

    # A non-symmetric selection must not be accepted silently.
    one_sided = {mu: list(full[mu]) for mu in cast(list[AOIdx], [0, 1])}
    with pytest.raises(IndexError, match="unordered_map"):
        get_sparse_P_mu_nu(mol, make_auxmol(mol, auxbasis="weigend"), one_sided)


def test_order_by_overlap() -> None:
    """Fragments along a chain, given in shuffled order, are traversed in order."""
    n_frag = 6
    used = np.zeros((n_frag, 3 * n_frag + 2), dtype=bool)
    for k in range(n_frag):
        used[k, 3 * k : 3 * k + 5] = True
    perm = [3, 0, 5, 1, 4, 2]

    order = [perm[i] for i in _order_by_overlap(used[perm])]

    assert order in (list(range(n_frag)), list(reversed(range(n_frag))))


def test_plan_window() -> None:
    """The window never overflows and holds the needed pairs of every fragment.
    Lazy eviction computes at most as many pairs as evicting every pair
    that the current fragment does not need, and every pair once if all fit.
    Batching the missing pairs of the following fragments needs fewer batches,
    without computing more pairs."""
    rng = np.random.default_rng(42)
    keys = [
        cast(Vector[np.int64], np.unique(rng.integers(0, 200, size=n)))
        for n in rng.integers(20, 80, size=30)
    ]
    n_eager = len(keys[0]) + sum(
        len(np.setdiff1d(b, a, assume_unique=True)) for a, b in zip(keys, keys[1:])
    )
    n_all = len(np.unique(np.concatenate(keys)))
    max_needed = max(map(len, keys))

    n_batches: dict[tuple[int, int], int] = {}
    for capacity in (max_needed, 100, 150, n_all):
        for batch_size in (1, 10, 50, n_all):
            stored: set[int] = set()
            n_computed = 0
            n_batches[capacity, batch_size] = 0
            for needed, (evicted, new) in zip(
                keys, _plan_window(keys, capacity, batch_size)
            ):
                assert stored.issuperset(evicted.tolist())
                assert stored.isdisjoint(new.tolist())
                assert set(needed.tolist()).isdisjoint(evicted.tolist())
                stored = stored.difference(evicted.tolist()).union(new.tolist())
                assert len(stored) <= capacity
                assert stored.issuperset(needed.tolist())
                n_computed += len(new)
                n_batches[capacity, batch_size] += len(new) > 0
            assert n_computed <= n_eager, (capacity, batch_size)
            if capacity == n_all:
                assert n_computed == n_all
    # With spare capacity, batching needs fewer batches; in the extreme only one.
    assert n_batches[150, 50] < n_batches[150, 1]
    assert n_batches[n_all, n_all] == 1
    for capacity in (max_needed, 100, 150, n_all):
        assert n_batches[capacity, 50] <= n_batches[capacity, 1]


def test_rolling_window() -> None:
    """The window of L⁻¹ (P | mu nu) gives the same (ij | kl) as the precomputed
    L⁻¹ (P | mu nu), also if pairs are evicted and inserted again in small chunks,
    and if the window holds more pairs than the current fragment needs."""
    # In octane every fragment uses all AOs, hence a longer chain.
    mol = M("xyz/E-polyacetylene/20.xyz", basis="sto-3g")
    mf = scf.RHF(mol)
    mf.kernel()
    fobj = fragmentate(frag_type="chemgen", n_BE=2, mol=mol, print_frags=False)
    auxmol = make_auxmol(mol, "weigend")
    fragments = BE(mf, fobj, auxbasis="weigend", int_transform="int-direct-DF").Fobjs
    # truncating thresholds, such that the fragments need different pairs
    AO_eps, MO_eps = 1e-3, 1e-1
    S_abs = approx_S_abs(mol)
    # Fortran order, as the window requires (scipy >= 1.18 returns copies in C order)
    L_PQ = _cholesky_PQ(auxmol)

    P_mu_nu = get_sparse_P_mu_nu(mol, auxmol, _get_AO_per_AO(S_abs, AO_eps))
    cpp.apply_inv_cholesky_inplace(P_mu_nu, L_PQ)

    keys = [
        _get_pair_keys(S_abs, AO_eps, _get_used_AOs(S_abs, f.TA, MO_eps))
        for f in fragments
    ]
    # some spare capacity, such that pairs of later fragments are batched
    batch_size = 20
    capacity = max(map(len, keys)) + batch_size
    window = cpp.SemiSparseSym3DWindow((auxmol.nao, mol.nao, mol.nao), capacity, L_PQ)
    n_evicted = 0
    n_more_stored = 0
    # every fragment twice in random order
    rng = np.random.default_rng(42)
    order = np.concatenate([rng.permutation(len(fragments)) for _ in range(2)])
    ordered_keys = [keys[i_frag] for i_frag in order]
    for i_frag, (evicted, new) in zip(
        order, _plan_window(ordered_keys, capacity, batch_size)
    ):
        window.evict(evicted)
        n_evicted += len(evicted)
        for start in range(0, len(new), 7):
            chunk = cast(Vector[np.int64], new[start : start + 7])
            window.insert(chunk, _compute_P_mu_nu_columns(mol, auxmol, chunk))
        window.set_active(keys[i_frag])
        assert len(keys[i_frag]) <= window.n_stored <= capacity
        n_more_stored += window.n_stored > len(keys[i_frag])

        TA = fragments[i_frag].TA
        assert np.allclose(
            cpp.transform_integral_precontracted(window, TA, S_abs, MO_eps),
            cpp.transform_integral_precontracted(P_mu_nu, TA, S_abs, MO_eps),
            atol=1e-12,
            rtol=0,
        )
    assert n_evicted > 0
    assert n_more_stored > 0


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
