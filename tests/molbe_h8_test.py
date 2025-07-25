# Illustrates a simple molecular BE calculation with BE
# density matching and pure chemical potential optimization
# between edge & centers of fragments.

import numpy as np
import pytest
from pyscf import gto, scf

from quemb.molbe import BE, fragmentate
from quemb.molbe.fragment import ChemGenArgs


def prepare_system():
    # PySCF HF generated mol & mf (molecular desciption & HF object)
    mol = gto.M(
        atom="""
            H 0. 0. 0.
            H 0. 0. 1.
            H 0. 0. 2.
            H 0. 0. 3.
            H 0. 0. 4.
            H 0. 0. 5.
            H 0. 0. 6.
            H 0. 0. 7.
            """,
        basis="sto-3g",
        charge=0,
    )

    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    return mol, mf


def do_BE(mol, mf, n_BE: int, only_chem: bool, swallow_replace: bool = False):
    fobj = fragmentate(
        n_BE=n_BE,
        frag_type="chemgen",
        mol=mol,
        additional_args=ChemGenArgs(
            treat_H_different=False, swallow_replace=swallow_replace
        ),
    )
    mybe = BE(mf, fobj)
    mybe.optimize(solver="FCI", only_chem=only_chem)
    return mybe


def test_BE_density_matching():
    mol, mf = prepare_system()

    BE2 = do_BE(mol, mf, 2, only_chem=False)
    assert np.isclose(BE2.ebe_tot - BE2.ebe_hf, -0.1343036698277933)

    with pytest.raises(ValueError):
        # should raise until https://github.com/troyvvgroup/quemb/issues/150
        # is resolved.
        BE3 = do_BE(mol, mf, 3, only_chem=False, swallow_replace=False)

    BE3 = do_BE(mol, mf, 3, only_chem=False, swallow_replace=True)
    assert np.isclose(BE3.ebe_tot - BE3.ebe_hf, -0.1332017928466369)


def test_BE_chemical_potential():
    mol, mf = prepare_system()

    BE1 = do_BE(mol, mf, 1, only_chem=True)
    print(BE1.ebe_tot - BE1.ebe_hf)
    assert np.isclose(BE1.ebe_tot - BE1.ebe_hf, -0.12831444938462155)

    BE2 = do_BE(mol, mf, 2, only_chem=True)
    print(BE2.ebe_tot - BE2.ebe_hf)
    assert np.isclose(BE2.ebe_tot - BE2.ebe_hf, -0.1343968038684169)

    BE3 = do_BE(mol, mf, 3, only_chem=True)
    print(BE3.ebe_tot - BE3.ebe_hf)
    assert np.isclose(BE3.ebe_tot - BE3.ebe_hf, -0.1332017928466369)


if __name__ == "__main__":
    test_BE_chemical_potential()
    test_BE_density_matching()
