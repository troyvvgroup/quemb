# Tests energy and nuclear gradient computations and scans using BE as
# the custom energy function fed through pyscf's finite difference driver
# Author(s): Beck Hanscam

import numpy as np
from pyscf import cc, gto, scf
from pyscf.tools import finite_diff

from quemb.molbe.chemfrag import ChemGenArgs
from quemb.molbe.scanner import BEArgs, Energy, energy_be


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
    mf = scf.RHF(mol0)
    mf.conv_tol = 1e-12
    mf.kernel()
    mc = cc.CCSD(mf)
    mc.kernel()
    mc_grad = mc.nuc_grad_method()
    ref_grad_corr = mc_grad.kernel()

    # CCSD analytic gradient of mol1
    mf = scf.RHF(mol1)
    mf.conv_tol = 1e-12
    mf.kernel()
    mc = cc.CCSD(mf)
    mc.kernel()
    mc_grad = mc.nuc_grad_method()
    ref_grad_corr1 = mc_grad.kernel()

    # build BE energy method
    be_args = BEArgs(
        n_BE=3,
        solver="CCSD",
        use_cumulant=True,
        optimize=False,
        additional_args=ChemGenArgs(h_treatment="treat_H_like_heavy_atom"),
    )
    energy_method = Energy(mol0, energy_be, be_args=be_args)

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


if __name__ == "__main__":
    test_numerical_be_gradient()
