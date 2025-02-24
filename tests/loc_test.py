# Tests the standard localization and IAO routines for hexene


import os
import unittest

import numpy as np
import pytest
from pyscf import gto, scf

from quemb.molbe import BE, fragpart


def test_hexene_loc_be1_froz_pm(hexene) -> None:
    be1_f_pm = ret_ecorr(
        hexene[0],
        hexene[1],
        be="be1",
        frozen=True,
        iao_valence_basis=None,
        lo_method="pipek-mezey",
        iao_loc_method=None,
    )
    assert np.isclose(be1_f_pm, -0.85564574)


@unittest.skipIf(
    os.getenv("QUEMB_SKIP_EXPENSIVE_TESTS") == "true",
    "Skipped expensive tests for QuEmb.",
)
def test_hexene_loc_be2_unfroz_lowdin(hexene) -> None:
    be2_nf_lo = ret_ecorr(
        hexene[0],
        hexene[1],
        be="be2",
        frozen=False,
        iao_valence_basis=None,
        lo_method="lowdin",
        iao_loc_method=None,
    )
    assert np.isclose(be2_nf_lo, -0.94588487)


def test_hexene_loc_be1_unfroz_iao_minao_so(hexene) -> None:
    be1_nf_iao_so = ret_ecorr(
        hexene[0],
        hexene[1],
        be="be1",
        frozen=False,
        iao_valence_basis="minao",
        lo_method="iao",
        iao_loc_method="SO",
    )
    assert np.isclose(be1_nf_iao_so, -0.83985647)


@unittest.skipIf(
    os.getenv("QUEMB_SKIP_EXPENSIVE_TESTS") == "true",
    "Skipped expensive tests for QuEmb.",
)
def test_hexene_loc_be2_froz_iao_sto3g_boys(hexene) -> None:
    be2_f_iao_fb = ret_ecorr(
        hexene[0],
        hexene[1],
        be="be2",
        frozen=True,
        iao_valence_basis="sto-3g",
        lo_method="iao",
        iao_loc_method="Boys",
    )
    assert np.isclose(be2_f_iao_fb, -0.92843714)


def ret_ecorr(
    mol: gto.Mole,
    mf: scf.hf.RHF,
    be: str,
    frozen: bool,
    iao_valence_basis: str | None,
    lo_method: str,
    iao_loc_method: str | None,
) -> float:
    # Fragment molecule
    fobj = fragpart(
        be_type=be,
        mol=mol,
        frozen_core=frozen,
        iao_valence_basis=iao_valence_basis,
        frag_type="chemgen",
    )

    # Run BE initialization and localization
    mybe = BE(
        mf,
        fobj,
        lo_method=lo_method,
        iao_loc_method=iao_loc_method,
    )

    # Solve and return the one-shot correlation energy
    mybe.oneshot(solver="CCSD", nproc=1, ompnum=1, use_cumulant=True)
    return mybe.ebe_tot - mybe.ebe_hf


def prepare_struct(structure):
    mol = gto.M(
        atom=structure,
        basis="cc-pVDZ",
        charge=0,
    )

    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    return mol, mf


@pytest.fixture(scope="session")
def hexene():
    mol, mf = prepare_struct(structure="data/hexene.xyz")
    return [mol, mf]


if __name__ == "__main__":
    test_hexene_loc_be1_froz_pm(hexene)
    test_hexene_loc_be2_unfroz_lowdin(hexene)
    test_hexene_loc_be1_unfroz_iao_minao_so(hexene)
    test_hexene_loc_be2_froz_iao_sto3g_boys(hexene)
