# Tests the standard localization and IAO routines for octane

from typing import Tuple

import numpy as np
from pyscf import gto, scf

from quemb.molbe import BE, fragpart


class LocalizationOnly(BE):
    def initialize(self, *args):
        pass


def test_octane_loc() -> None:
    # Prepare octane molecule
    mol, mf = prepare_octane()

    be1_f_pm = ret_octane_lmo(
        mol,
        mf,
        be="be1",
        frozen=True,
        iao_valence_basis=None,
        lo_method="pipek-mezey",
        iao_loc_method=None,
    )
    assert np.isclose(be1_f_pm, np.load("data/lmo_coeff-be1_f_pm.npy")).all()

    be2_nf_lo = ret_octane_lmo(
        mol,
        mf,
        be="be2",
        frozen=False,
        iao_valence_basis=None,
        lo_method="lowdin",
        iao_loc_method=None,
    )
    assert np.isclose(be2_nf_lo, np.load("data/lmo_coeff-be2_nf_lo.npy")).all()

    be1_nf_iao_so = ret_octane_lmo(
        mol,
        mf,
        be="be1",
        frozen=False,
        iao_valence_basis="minao",
        lo_method="iao",
        iao_loc_method="SO",
    )
    assert np.isclose(be1_nf_iao_so, np.load("data/lmo_coeff-be1_nf_iao_so.npy")).all()

    be2_f_iao_fb = ret_octane_lmo(
        mol,
        mf,
        be="be2",
        frozen=True,
        iao_valence_basis="sto-3g",
        lo_method="iao",
        iao_loc_method="Boys",
    )
    assert np.isclose(be2_f_iao_fb, np.load("data/lmo_coeff-be2_f_iao_fb.npy")).all()


def ret_octane_lmo(
    mol,
    mf,
    be,
    frozen,
    iao_valence_basis,
    lo_method,
    iao_loc_method,
):
    # Fragment molecule
    fobj = fragpart(
        be_type=be,
        mol=mol,
        frozen_core=frozen,
        iao_valence_basis=iao_valence_basis,
        frag_type="autogen",
    )

    # Run BE initialization and localization, without integral transformation
    mybe = LocalizationOnly(
        mf,
        fobj,
        lo_method=lo_method,
        iao_loc_method=iao_loc_method,
    )

    return mybe.lmo_coeff


def prepare_octane() -> Tuple[gto.Mole, scf.hf.RHF]:
    mol = gto.M(
        atom="""
    C   0.4419364699  -0.6201930287   0.0000000000
    C  -0.4419364699   0.6201930287   0.0000000000
    H  -1.0972005331   0.5963340874   0.8754771384
    H   1.0972005331  -0.5963340874  -0.8754771384
    H  -1.0972005331   0.5963340874  -0.8754771384
    H   1.0972005331  -0.5963340874   0.8754771384
    C   0.3500410560   1.9208613544   0.0000000000
    C  -0.3500410560  -1.9208613544   0.0000000000
    H   1.0055486349   1.9450494955   0.8754071298
    H  -1.0055486349  -1.9450494955  -0.8754071298
    H   1.0055486349   1.9450494955  -0.8754071298
    H  -1.0055486349  -1.9450494955   0.8754071298
    C  -0.5324834907   3.1620985364   0.0000000000
    C   0.5324834907  -3.1620985364   0.0000000000
    H  -1.1864143468   3.1360988730  -0.8746087226
    H   1.1864143468  -3.1360988730   0.8746087226
    H  -1.1864143468   3.1360988730   0.8746087226
    H   1.1864143468  -3.1360988730  -0.8746087226
    C   0.2759781663   4.4529279755   0.0000000000
    C  -0.2759781663  -4.4529279755   0.0000000000
    H   0.9171145792   4.5073104916   0.8797333088
    H  -0.9171145792  -4.5073104916  -0.8797333088
    H   0.9171145792   4.5073104916  -0.8797333088
    H  -0.9171145792  -4.5073104916   0.8797333088
    H   0.3671153250  -5.3316378285   0.0000000000
    H  -0.3671153250   5.3316378285   0.0000000000
    """,
        basis="cc-pVDZ",
        charge=0,
    )

    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    return mol, mf
