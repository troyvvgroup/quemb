# Tests energy and nuclear gradient computations and scans using BE as
# the custom energy function fed through pyscf's finite difference driver
# Author(s): Beck Hanscam

import os
import unittest

import numpy as np
from pyscf import cc, gto, scf
from pyscf.tools import finite_diff

from quemb.molbe.chemfrag import ChemGenArgs
from quemb.molbe.mbe import BEArgs
from quemb.molbe.scanner import (
    Energy,
    be_ref_data,
    energy_be,
    energy_force_emb,
    force_emb_ref_data,
)


def ccsd_analytic_gradient(mol, conv_tol=1e-12):
    mf = scf.RHF(mol)
    mf.conv_tol = conv_tol
    mf.kernel()
    mc = cc.CCSD(mf)
    mc.kernel()
    mc_grad = mc.nuc_grad_method()
    return mc_grad.kernel()


def test_numerical_be_gradient():
    mol0 = gto.M(
        atom="""
            H  0.04 -0.02  0.
            H  0.02  0.04  1.
            H -0.01 -0.00  2.
            H -0.03  0.01  3.
            H -0.00  0.04  4.
            H  0.04  0.02  5.
            """,
        basis="sto-3g",
        charge=0,
    )

    mol1 = gto.M(
        atom="""
            H 0. 0. 0.
            H 0. 0. 1.
            H 0. 0. 2.
            H 0. 0. 3.
            H 0. 0. 4.
            H 0. 0. 5.
            """,
        basis="sto-3g",
        charge=0,
    )

    # CCSD analytic gradient of mol0
    ref_grad_corr = ccsd_analytic_gradient(mol0)

    # CCSD analytic gradient of mol1
    ref_grad_corr1 = ccsd_analytic_gradient(mol1)

    # build BE energy method
    be_args = BEArgs(
        n_BE=3,
        solver="CCSD",
        use_cumulant=True,
        optimize=False,
        additional_args=ChemGenArgs(h_treatment="treat_H_like_heavy_atom"),
    )
    energy_method = Energy(
        mol0, energy_be, energy_args=be_args, ref_data_func=be_ref_data
    )

    # BE3-CCSD oneshot numerical gradient
    fd_grad = finite_diff.kernel(energy_method, displacement=1e-4)
    assert np.isclose(
        np.sqrt(np.mean((ref_grad_corr - fd_grad) ** 2)), 5.898172966145e-08
    )

    # BE3-CCSD oneshot numerical gradient using pyscf built-in Gradients object
    grad_method = finite_diff.Gradients(energy_method)
    grad_method.displacement = 1e-4
    fd_grad_fromObj = grad_method.kernel()
    assert np.isclose(
        np.sqrt(np.mean((ref_grad_corr - fd_grad_fromObj) ** 2)), 5.898172966145e-08
    )

    # BE3-CCSD oneshot numerical gradient from scanner
    grad_scanner = grad_method.as_scanner()
    E1, fd_grad1 = grad_scanner(mol1)
    assert np.isclose(E1, -3.23567708251885)
    assert np.isclose(
        np.sqrt(np.mean((ref_grad_corr1 - fd_grad1) ** 2)), 5.8664061568865036e-08
    )


def test_numerical_force_emb_gradient():
    mol0 = gto.M(
        atom="""
            H  0.04 -0.02  0.
            H  0.02  0.04  1.
            H -0.01 -0.00  2.
            H -0.03  0.01  3.
            H -0.00  0.04  4.
            H  0.04  0.02  5.
            """,
        basis="sto-3g",
        charge=0,
    )

    # CCSD analytic gradient of mol0
    ref_grad_corr = ccsd_analytic_gradient(mol0)

    # build force embedding energy method
    force_emb_args = BEArgs(
        n_BE=3,
        solver="CCSD",
        use_cumulant=True,
        optimize=False,
        additional_args=ChemGenArgs(h_treatment="treat_H_like_heavy_atom"),
    )
    energy_method = Energy(
        mol0,
        energy_force_emb,
        energy_args=force_emb_args,
        ref_data_func=force_emb_ref_data,
    )

    # BE3-CCSD force embedding numerical gradient
    fd_grad = finite_diff.kernel(energy_method, displacement=1e-4)
    assert np.isclose(
        np.sqrt(np.mean((ref_grad_corr - fd_grad) ** 2)), 5.904036799315e-08
    )

    # BE3-CCSD force embedding numerical gradient using pyscf built-in Gradients object
    grad_method = finite_diff.Gradients(energy_method)
    grad_method.displacement = 1e-4
    fd_grad_fromObj = grad_method.kernel()
    assert np.isclose(
        np.sqrt(np.mean((ref_grad_corr - fd_grad_fromObj) ** 2)), 5.904036799315e-08
    )


@unittest.skipUnless(
    os.getenv("QUEMB_DO_EXPENSIVE_TESTS") == "true",
    "Skipped expensive tests for QuEmb.",
)
def test_numerical_be_gradient_withmatching():
    mol0 = gto.M(
        atom="""
            H  0.04 -0.02  0.
            H  0.02  0.04  1.
            H -0.01 -0.00  2.
            H -0.03  0.01  3.
            H -0.00  0.04  4.
            H  0.04  0.02  5.
            """,
        basis="sto-3g",
        charge=0,
    )

    mol1 = gto.M(
        atom="""
            H 0. 0. 0.
            H 0. 0. 1.
            H 0. 0. 2.
            H 0. 0. 3.
            H 0. 0. 4.
            H 0. 0. 5.
            """,
        basis="sto-3g",
        charge=0,
    )

    # CCSD analytic gradient of mol0
    ref_grad_corr = ccsd_analytic_gradient(mol0)

    # CCSD analytic gradient of mol1
    ref_grad_corr1 = ccsd_analytic_gradient(mol1)

    # build BE energy method
    be_args = BEArgs(
        n_BE=1,
        solver="CCSD",
        use_cumulant=True,
        optimize=True,
        only_chem=True,
        additional_args=ChemGenArgs(h_treatment="treat_H_like_heavy_atom"),
    )
    energy_method = Energy(
        mol0, energy_be, energy_args=be_args, ref_data_func=be_ref_data
    )

    # BE1-CCSD with chemical potential matching
    # numerical gradient using pyscf built-in Gradients object
    grad_method = finite_diff.Gradients(energy_method)
    grad_method.displacement = 1e-4
    fd_grad_fromObj = grad_method.kernel()
    assert np.isclose(
        np.sqrt(np.mean((ref_grad_corr - fd_grad_fromObj) ** 2)), 0.01592042531497887
    )

    # BE1-CCSD with chemical potential matching
    # numerical gradient from scanner
    grad_scanner = grad_method.as_scanner()
    E1, fd_grad1 = grad_scanner(mol1)
    assert np.isclose(E1, -3.2152636577260725)
    assert np.isclose(
        np.sqrt(np.mean((ref_grad_corr1 - fd_grad1) ** 2)), 0.01592294368834432
    )


if __name__ == "__main__":
    test_numerical_be_gradient()
    test_numerical_force_emb_gradient()
    test_numerical_be_gradient_withmatching()
