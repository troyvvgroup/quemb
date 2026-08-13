# Illustrates nuclear gradient and hessian computations and scans using a
# custom energy function fed through pyscf's finite difference driver
# Author(s): Beck Hanscam

import numpy as np
from pyscf import grad, gto, scf
from pyscf.tools import finite_diff

from quemb.molbe.scanner import Energy, energy_hf


def test_numerical_gradient():
    mol0 = gto.M(atom="H 0.01 0.02 -0.03; H -0.02 0.01 1.00", basis="6-31g")

    # analytic gradient
    mf = scf.RHF(mol0)
    mf.verbose = 0
    mf.kernel()
    mf_grad = grad.RHF(mf)
    ref_grad_hf = mf_grad.kernel()

    # numerical gradient of mol0
    energy_method = Energy(mol0, energy_hf)
    fd_grad = finite_diff.kernel(energy_method, displacement=1e-4)
    assert np.isclose(
        np.sqrt(np.mean((ref_grad_hf - fd_grad) ** 2)), 7.581109024762e-09
    )

    # numerical gradient of mol0 using pyscf built-in Gradients object
    grad_method = finite_diff.Gradients(energy_method)
    grad_method.displacement = 1e-4
    fd_grad_fromObj = grad_method.kernel()
    assert np.isclose(
        np.sqrt(np.mean((ref_grad_hf - fd_grad_fromObj) ** 2)), 7.581109024762e-09
    )


def test_numerical_hessian():
    mol0 = gto.M(atom="H 0.01 0.02 -0.03; H -0.02 0.01 1.00", basis="6-31g")

    # analytic hessian
    mf = scf.RHF(mol0)
    mf.verbose = 0
    mf.kernel()
    ref_hess_hf = mf.Hessian().kernel()

    # numerical hessian of mol0 using pyscf Gradients
    # object as input to the built-in Hessian object
    energy_method = Energy(mol0, energy_hf)
    grad_method = finite_diff.Gradients(energy_method)
    grad_method.displacement = 1e-4
    hess_method = finite_diff.Hessian(grad_method)
    hess_method.displacement = 1e-4
    fd_hess = hess_method.kernel()
    print("To illustrate the failure")
    assert np.isclose(
        np.sqrt(np.mean((ref_hess_hf - fd_hess) ** 2)), 2.596596383663e-08
    )


def test_scanners():

    mol0 = gto.M(atom="H 0.01  0.02 -0.03; H  -0.02  0.01 1.00", basis="6-31g")
    mol1 = gto.M(atom="H 0.02 -0.01  0.00; H  -0.02 -0.03 0.80", basis="6-31g")
    mol2 = gto.M(atom="H 0.00  0.03  0.01; H   0.02 -0.01 0.70", basis="6-31g")

    # energy point calculations using as_scanner()
    energy_method = Energy(mol0, energy_hf)
    energy_scanner = energy_method.as_scanner()
    assert np.isclose(energy_scanner(mol1), -1.123608970415078)
    assert np.isclose(energy_scanner(mol2), -1.1256408758743772)

    # gradient point calculations using as_scanner()
    grad_method = finite_diff.Gradients(energy_method)
    grad_method.displacement = 1e-4
    grad_scanner = grad_method.as_scanner()
    E1, grad1 = grad_scanner(mol1)
    E2, grad2 = grad_scanner(mol2)
    assert np.isclose(E1, -1.123608970415078)
    assert np.isclose(E2, -1.1256408758743772)

    # analytic gradient of mol1
    mf1 = scf.RHF(mol1)
    mf1.verbose = 0
    mf1.kernel()
    mf_grad1 = grad.RHF(mf1)
    ref_grad1_hf = mf_grad1.kernel()
    assert np.isclose(
        np.sqrt(np.mean((ref_grad1_hf - grad1) ** 2)), 7.344822125816994e-08
    )

    # analytic gradient of mol2
    mf2 = scf.RHF(mol2)
    mf2.verbose = 0
    mf2.kernel()
    mf_grad2 = grad.RHF(mf2)
    ref_grad2_hf = mf_grad2.kernel()
    assert np.isclose(
        np.sqrt(np.mean((ref_grad2_hf - grad2) ** 2)), 1.5532972165126092e-07
    )


if __name__ == "__main__":
    test_numerical_gradient()
    test_numerical_hessian()
    test_scanners()
