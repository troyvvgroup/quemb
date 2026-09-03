"""
Finite-difference cross-check for the analytic HF-level Jacobian seed used by
UBE.optimize()'s only_chem=False branch (get_be_error_jacobian_u in
quemb.shared.external.optqn, built on the UHF CPHF core in
quemb.shared.external.cphf_utils).

This is the primary correctness gate called for in JACOBIAN_SEED_HANDOFF.md:
the underlying UHF CPHF kernels this Jacobian is built from had zero test
coverage anywhere in the repo before this file.

The Jacobian is HF-level by construction (an analytic seed/preconditioner for
the quasi-Newton solver, not required to match the UCCSD-level solver -- see
the module docstring of get_atbe_Jblock_frag_u), so the correct
finite-difference target is the HF-level error vector: the same
solve_error_u formula, but evaluated on each fragment's embedded-HF-level
density (mo_coeff/occupied-projector from fobj._mf, the same orbitals the
Jacobian itself is built from) rather than the UCCSD density solve_error_u
normally reads off fobj.rdm1__/mo_coeff_uccsd. objfunc_hf below reproduces
be_func_u's potential-injection + fragment SCF steps directly (skipping the
UCCSD solve, which is irrelevant to an HF-level check and would only add
cost and an apples-to-oranges mismatch), then stages the HF density onto
fobj.rdm1__/mo_coeff_uccsd so the existing, tested solve_error_u can be
reused unmodified.

H8 chain, STO-3G, common_bath, BE2: same system as
tests/ube-iterative_test.py's regression test.

Author(s): Leah Weisburn
"""

import os
import unittest

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from pyscf import gto, scf

from quemb.molbe import UBE, fragmentate
from quemb.molbe.fragment import ChemGenArgs
from quemb.molbe.solver import solve_error_u
from quemb.shared.external.optqn import get_be_error_jacobian_u


def h8_mol():
    mol = gto.M()
    mol.atom = """
        H 0. 0. 0.
        H 0. 0. 1.
        H 0. 0. 2.
        H 0. 0. 3.
        H 0. 0. 4.
        H 0. 0. 5.
        H 0. 0. 6.
        H 0. 0. 7.
        """
    mol.basis = "sto-3g"
    mol.build()
    return mol


class TestJacobianSeed_Unrestricted(unittest.TestCase):
    def test_h8_common_bath_jacobian_finite_difference(self):
        mol = h8_mol()
        mf = scf.UHF(mol)
        mf.conv_tol = 1e-12
        mf.kernel()
        fobj = fragmentate(
            n_BE=2,
            frag_type="chemgen",
            mol=mol,
            additional_args=ChemGenArgs(treat_H_different=False),
        )
        mybe = UBE(mf, fobj, common_bath=True)

        Fobjs_ab = list(zip(mybe.Fobjs_a, mybe.Fobjs_b))
        Nocc_ab = (mybe.Nocc[0], mybe.Nocc[1])

        def objfunc_hf(xk):
            pot = list(xk)
            for fobj_a, fobj_b in Fobjs_ab:
                fobj_a.update_heff(pot[:-2] + [pot[-2]], only_chem=False)
                fobj_b.update_heff(pot[:-2] + [pot[-1]], only_chem=False)
                fobj_a.scf(
                    unrestricted=True, spin_ind=0, dm_other=fobj_a.dm_other_embedded
                )
                fobj_b.scf(
                    unrestricted=True, spin_ind=1, dm_other=fobj_b.dm_other_embedded
                )
                for fobj in (fobj_a, fobj_b):
                    nao = fobj.nao
                    fobj.mo_coeff_uccsd = fobj._mf.mo_coeff.copy()
                    fobj.rdm1__ = np.diag(
                        [1.0] * fobj.nsocc + [0.0] * (nao - fobj.nsocc)
                    )
            _, ervec = solve_error_u(
                Fobjs_ab, Nocc_ab[0], Nocc_ab[1], only_chem=False
            )
            return ervec

        x0 = np.array(mybe.pot, dtype=float)
        # Sets fobj_a/b._mf (embedded-HF-level orbitals) at x0, which the
        # analytic Jacobian below reuses -- must run before it, and before
        # any perturbed objfunc_hf calls below overwrite ._mf.
        objfunc_hf(x0)
        J_analytic = get_be_error_jacobian_u(Fobjs_ab)

        h = 1e-4
        n = len(x0)
        J_numeric = np.zeros((n, n))
        for i in range(n):
            xp = x0.copy()
            xp[i] += h
            xm = x0.copy()
            xm[i] -= h
            fp = objfunc_hf(xp)
            fm = objfunc_hf(xm)
            J_numeric[:, i] = (fp - fm) / (2.0 * h)

        np.testing.assert_allclose(J_analytic, J_numeric, atol=2e-3, rtol=2e-3)


if __name__ == "__main__":
    unittest.main()
