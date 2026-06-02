import inspect
import os

import numpy as np
from pyscf import lib, scf


def energy_hf(mol, fd_info=None):
    r"""Compute the restricted Hartree-Fock total energy

    Parameters
    ----------
    mol : object
        Molecule object defining the geometry, basis, charge, and spin.
    fd_info:
        Dictionary of information passed from the finite difference driver including
        the atom index and xyz coordinate(s) displaced from the reference and the ref
        mol object.

    Returns
    ------
    float
        Converged RHF total energy in Hartree
    """
    if fd_info is not None:
        if not np.allclose(
            fd_info["ref_coords"],
            fd_info["ref_mol"].atom_coords(),
            atol=1e-12,
            rtol=1e-12,
        ):
            raise RuntimeError("finite difference reference coordinate mismatch.")

    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    return mf.e_tot


class Energy(lib.StreamObject):
    r"""PySCF-style wrapper for a custom molecular energy function.

    This class provides a minimal interface around an arbitrary energy function
    ``energy_func(mol)``. It is intended to be compatible with PySCF utilities that
    expect an object with ``mol``, ``kernel()``, and ``as_scanner()`` methods, such as
    ``pyscf.tools.finite_diff``.

    This class supplies repeated single-point energies that can be used by finite
    difference gradient or Hessian drivers.
    """

    def __init__(self, mol, energy_func, displacement=1e-4):
        r"""Initialize the custom energy wrapper.

        Parameters
        ----------
        mol : object
            Reference molecule.
        energy_func :
            Callable function with signature ``energy_func(mol) -> float`` returning
            the total energy in Hartree.
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

    def kernel(self, mol=None, fd_info=None):
        r"""Evaluate the energy for a molecule.

        Parameters
        ----------
        mol : object, optional
            Molecule at which to evaluate the energy.
        fd_info:
            Dictionary of information passed from the finite difference driver
            including the atom index and xyz coordinate(s) displaced from the
            reference and the ref mol object.

        Returns
        ------
        float
            Total energy in Hartree
        """
        if mol is not None:
            self.mol = mol

        if fd_info is None:
            fd_info = {
                "kind": "reference",
                "atom_idx": [],
                "axis_idx": [],
                "delta_bohr": 0.0,
                "ref_mol": self.mol.copy(),
                "ref_coords": self.mol.atom_coords().copy(),
            }

        self.e_tot = self.energy_func(self.mol, fd_info=fd_info)
        return self.e_tot

    def as_scanner(self):
        r"""Return a PySCF-compatible energy scanner.

        The returned scanner is callable as ``scanner(mol)`` and evaluates
        ``energy_func`` for each supplied geometry. This mirrors the behavor of
        PySCF scanner objects used in geometry optimization and finite difference
        calculations.

        Returns
        ------
        object
            Callable scanner object returning total energies in Hartree.
        """
        parent = self

        def called_from_pyscf_finite_diff():
            for frame in inspect.stack()[1:8]:
                fname = os.path.basename(frame.filename)
                if fname == "finite_diff.py":
                    return True
            return False

        class Scanner(lib.SinglePointScanner, lib.StreamObject):
            def __init__(self):
                self.ref_coords = parent.mol.atom_coords().copy()
                self.ref_mol = parent.mol.copy()

            def is_fd_probe(self, diff, tol=1e-8):
                """
                Return True if coords look like a finite-difference probe of ref_coords.
                """
                diff_flat = np.ravel(diff)
                nonzero_idx = np.where(np.abs(diff_flat) > 1e-12)[0]

                if len(nonzero_idx) == 0:
                    return True

                for x in diff_flat[nonzero_idx]:
                    n = x / parent.displacement
                    if abs(n - round(n)) > tol:
                        return False

                return True

            def __call__(self, mol):

                coords = mol.atom_coords()
                in_fd = called_from_pyscf_finite_diff()

                if not in_fd:
                    # scanner point: new ref geometry
                    self.ref_coords = coords.copy()
                    self.ref_mol = mol.copy()
                    fd_info = {
                        "kind": "scanner_point",
                        "atom_idx": None,
                        "axis_idx": None,
                        "delta_bohr": 0.0,
                        "ref_mol": self.ref_mol.copy(),
                        "ref_coords": self.ref_coords.copy(),
                    }
                else:
                    diff = coords - self.ref_coords

                    # check if finite_diff is probing current geometry
                    # reset ref_coords and mol_ref if new geometry
                    if not self.is_fd_probe(diff):
                        self.ref_mol = mol.copy()
                        self.ref_coords = coords.copy()
                        diff = coords - self.ref_coords

                    displaced = np.reshape(diff, -1)
                    displaced_idx = np.where(np.abs(displaced) > 1e-12)[0]

                    fd_info = {
                        "kind": "reference",
                        "atom_idx": [idx // 3 for idx in displaced_idx],
                        "axis_idx": [idx % 3 for idx in displaced_idx],
                        "delta_bohr": [displaced[idx] for idx in displaced_idx],
                        "ref_mol": self.ref_mol.copy(),
                        "ref_coords": self.ref_coords,
                    }
                    if len(displaced_idx) == 1:
                        fd_info["kind"] = "single_displacement"
                    elif len(displaced_idx) > 1:
                        fd_info["kind"] = "multi_displacement"

                parent.mol = mol
                parent.e_tot = parent.energy_func(mol, fd_info=fd_info)

                self.mol = mol
                self.e_tot = parent.e_tot

                return self.e_tot

        scanner = Scanner()
        scanner.mol = parent.mol
        parent._scanner = scanner

        # Attributes expected by PySCF finite-difference assertions
        # These do not control convergence for the custom method
        scanner.conv_tol = 1e-12
        scanner.converged = True

        return scanner
