# Tests SCI-BE for 8H chain

import os
import unittest

import numpy as np
import pytest
from pyscf import gto, scf

from quemb.molbe import BE, fragmentate
from quemb.molbe.fragment import ChemGenArgs
from quemb.molbe.solver import SHCI_ArgsUser


# BE(1) Jobs
@unittest.skipIf(
    not os.getenv("QUEMB_DO_KNOWN_TO_FAIL_TESTS") == "true",
    "This test is known to fail.",
)
def test_h8_be1_thresh_p1(h8_be1) -> None:
    be1_p1 = ret_ecorr(
        h8_be1,
        use_cumulant=True,
        thresh=0.1,
    )
    assert np.isclose(be1_p1, -0.09546075053560976)


@unittest.skipIf(
    not os.getenv("QUEMB_DO_KNOWN_TO_FAIL_TESTS") == "true",
    "This test is known to fail.",
)
def test_h8_be1_thresh_p001(h8_be1) -> None:
    be1_p001 = ret_ecorr(
        h8_be1,
        use_cumulant=True,
        thresh=0.001,
    )
    assert np.isclose(be1_p001, -0.09325921453113128)


@unittest.skipIf(
    not os.getenv("QUEMB_DO_KNOWN_TO_FAIL_TESTS") == "true",
    "This test is known to fail.",
)
def test_h8_be1_thresh_p001_nocum(h8_be1) -> None:
    be1_p001_nocum = ret_ecorr(
        h8_be1,
        use_cumulant=False,
        thresh=0.001,
    )
    assert np.isclose(be1_p001_nocum, -0.09546075053560799)


# BE(2) Jobs
@unittest.skipIf(
    not os.getenv("QUEMB_DO_KNOWN_TO_FAIL_TESTS") == "true",
    "This test is known to fail.",
)
def test_h8_be2_thresh_p1(h8_be2) -> None:
    be2_p1 = ret_ecorr(
        h8_be2,
        use_cumulant=True,
        thresh=0.1,
    )
    assert np.isclose(be2_p1, -0.0428053424438577)


@unittest.skipIf(
    not os.getenv("QUEMB_DO_KNOWN_TO_FAIL_TESTS") == "true",
    "This test is known to fail.",
)
def test_h8_be2_thresh_p01(h8_be2) -> None:
    be2_p01 = ret_ecorr(
        h8_be2,
        use_cumulant=True,
        thresh=0.01,
    )
    assert np.isclose(be2_p01, -0.08477641560843807)


@unittest.skipIf(
    not os.getenv("QUEMB_DO_KNOWN_TO_FAIL_TESTS") == "true",
    "This test is known to fail.",
)
def test_h8_be2_thresh_p01_nocum(h8_be2) -> None:
    be2_p01_nocum = ret_ecorr(
        h8_be2,
        use_cumulant=False,
        thresh=0.01,
    )
    assert np.isclose(be2_p01_nocum, -0.08617119801475326)


def ret_ecorr(mybe, use_cumulant, thresh):
    # Solve and return the correlation energy
    add_solver_args = SHCI_ArgsUser(hci_cutoff=thresh, return_frag_data=False)
    mybe.optimize(
        solver="SCI",
        nproc=1,
        use_cumulant=use_cumulant,
        only_chem=True,
        solver_args=add_solver_args,
    )
    return mybe.ebe_tot - mybe.ebe_hf


def prepare_struct(structure):
    mol = gto.M(
        atom=structure,
        basis="STO-3G",
        charge=0,
    )

    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    return mol, mf


@pytest.fixture(scope="session")
def h8_be1():
    add_args = ChemGenArgs(treat_H_different=False)
    mol, mf = prepare_struct(structure="xyz/h8.xyz")
    fobj = fragmentate(
        n_BE=1,
        mol=mol,
        frag_type="chemgen",
        frozen_core=False,
        additional_args=add_args,
    )
    mybe = BE(mf, fobj)
    return mybe


@pytest.fixture(scope="session")
def h8_be2():
    add_args = ChemGenArgs(treat_H_different=False)
    mol, mf = prepare_struct(structure="xyz/h8.xyz")
    fobj = fragmentate(
        n_BE=2,
        mol=mol,
        frag_type="chemgen",
        frozen_core=False,
        additional_args=add_args,
    )
    mybe = BE(mf, fobj)
    return mybe


if __name__ == "__main__":
    test_h8_be1_thresh_p1(h8_be1)
    test_h8_be1_thresh_p001(h8_be1)
    test_h8_be1_thresh_p001_nocum(h8_be1)
    test_h8_be2_thresh_p1(h8_be2)
    test_h8_be2_thresh_p01(h8_be2)
    test_h8_be2_thresh_p01_nocum(h8_be2)
