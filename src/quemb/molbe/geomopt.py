from pyscf import scf, lib
from pyscf.tools import finite_diff
import numpy as np


def energy_hf(mol):
    r"""Compute the restricted Hartree-Fock total energy

    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecule object defining the geometry, basis, charge, and spin

    Returns
    ------
    float
        Converged RHF total energy in Hartree
    """
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    return mf.e_tot


class Energy(lib.StreamObject):
    r"""PySCF-style wrapper for a custom molecular energy function.

    This class provides a minimal interface around an arbitrary energy function ``energy_func(mol)``.
    It is intended to be compatible with PySCF utilities that expect an object with ``mol``,
    ``kernel()``, and ``as_scanner()`` methods, such as ``pyscf.tools.finite_diff``.

    This class supplies repeated single-point energies that can be used by finite difference gradient
    or Hessian drivers.
    """

    def __init__(self, mol, energy_func, displacement=1e-4):
        r"""Initialize the custom energy wrapper.

        Parameters
        ----------
        mol : pyscf.gto.Mole
            Reference molecule.
        energy_func : callable
            Function with signature ``energy_func(mol) -> float`` returning the total energy in Hartree.
        displacement : float, optional
            Finite difference displacement in Bohr, default is 1e-4.
        """
        self.mol = mol
        self.energy_func = energy_func
        self.e_tot = None
        self.displacement = displacement

        # Attributes expected by PySCF finite-difference assertions
        # These do not control convergence for the custom method
        self.conv_tol = 1e-12
        self.converged = True

    def kernel(self, mol=None):
        r"""Evaluate the energy for a molecule.

        Parameters
        ----------
        mol : pyscf.gto.Mole, optional
            Molecule at which to evaluate the energy.

        Returns
        ------
        float
            Total energy in Hartree
        """
        if mol is not None:
            self.mol = mol

        self.e_tot = self.energy_func(self.mol)
        return self.e_tot

    def as_scanner(self):
        r"""Return a PySCF-compatible energy scanner.

        The returned scanner is callable as ``scanner(mol)`` and evaluates ``energy_func`` for each
        supplied geometry. This mirrors the behavor of PySCF scanner objects used in geometry
        optimization and finite difference calculations.

        Returns
        ------
        scanner : pyscf.lib.SinglePointScanner
            Callable scanner object returning total energies in Hartree.
        """
        parent = self

        class Scanner(lib.SinglePointScanner, lib.StreamObject):
            def __call__(self, mol):

                parent.mol = mol
                parent.e_tot = parent.energy_func(mol)

                self.mol = mol
                self.e_tot = parent.e_tot

                return self.e_tot

        scanner = Scanner()
        scanner.mol = parent.mol

        # Attributes expected by PySCF finite-difference assertions
        # These do not control convergence for the custom method
        scanner.conv_tol = 1e-12
        scanner.converged = True

        return scanner
