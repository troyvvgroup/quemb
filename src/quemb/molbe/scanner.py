# Author(s): Beck Hanscam

import inspect
import os
from dataclasses import dataclass, field

import h5py
import numpy as np
from pyscf import cc, gto, lib, scf

from quemb.molbe import BE, fragmentate
from quemb.molbe.chemfrag import Fragmented
from quemb.molbe.helper import get_eri, get_scfObj
from quemb.molbe.mbe import BEArgs
from quemb.shared.manage_scratch import WorkDir


def energy_hf(mol, energy_args=None, fd_info=None):
    r"""Compute the restricted Hartree-Fock total energy

    Parameters
    ----------
    mol : object
        Molecule object defining the geometry, basis, charge, and spin.
    energy_args: optional
        User defined arguments for energy calculation.
    fd_info: FDinfo, optional
        Finite difference metadata describing the displacement relative
        to the current reference geometry.

    Returns
    ------
    float
        Converged RHF total energy in Hartree
    """
    if energy_args is None:
        pass
    if fd_info is not None:
        if fd_info.ref_mol is None:
            raise RuntimeError("missing finite difference reference geometry.")

    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    return mf.e_tot


def be_ref_data(mol, energy_args=None):
    r"""Build reference-geometry data needed by BE energy functions.

    Parameters
    ----------
    mol : object
        Molecule object defining the geometry, basis, charge, and spin.
    energy_args: BEArgs, optional
        User defined arguments for BE calculation.

    Returns
    ------
    dict
        Dictionary containing reference-geometry data needed by ``energy_be``.
        The ``"ref_fobj"`` entry stores the fragmentate object built from ``mol``.
    """
    if energy_args is None:
        energy_args = BEArgs()

    ref_fobj = fragmentate(
        mol=mol,
        n_BE=energy_args.n_BE,
        frag_type=energy_args.frag_type,
        frozen_core=energy_args.frozen_core,
        additional_args=energy_args.additional_args,
    )

    return {"ref_fobj": ref_fobj}


def be_frag_ref_data(mol, energy_args=None):
    r"""Build reference-geometry data needed by force embedding energy functions.

    Parameters
    ----------
    mol : object
        Molecule object defining the geometry, basis, charge, and spin.
    energy_args: BEArgs, optional
        User defined arguments for BE calculation.

    Returns
    ------
    dict
        Dictionary containing reference-geometry data needed by ``energy_be_frag``.
        The ``"ref_fobj"`` entry stores the fragmentate object built from ``mol``.
        The ``"ref_mybe"`` entry stores the BE object.
        The ``"frag_per_atom"`` entry stores the fragment index for each atom.
    """
    if energy_args is None:
        energy_args = BEArgs()

    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()

    ref_fobj = fragmentate(
        mol=mol,
        n_BE=energy_args.n_BE,
        frag_type=energy_args.frag_type,
        frozen_core=energy_args.frozen_core,
        additional_args=energy_args.additional_args,
    )

    ref_mybe = BE(
        mf,
        ref_fobj,
        lo_method=energy_args.lo_method,
        int_transform=energy_args.int_transform,
        auxbasis=energy_args.auxbasis,
        nproc=energy_args.nproc,
        ompnum=energy_args.ompnum,
        initialize_fragment_idx=energy_args.initialize_fragment_idx,
    )

    fragmented = Fragmented.from_mole(
        mol, n_BE=energy_args.n_BE, h_treatment=energy_args.additional_args.h_treatment
    )
    frag_per_atom = fragmented.get_frag_per_atom()

    ref_dict = {
        "ref_mol": mol.copy(),
        "ref_fobj": ref_fobj,
        "ref_mybe": ref_mybe,
        "frag_per_atom": frag_per_atom,
    }

    return ref_dict


def energy_be(mol, energy_args=None, fd_info=None):
    r"""Compute the BEn total energy

    Parameters
    ----------
    mol : object
        Molecule object defining the geometry, basis, charge, and spin.
    energy_args: BEArgs, optional
        User defined arguments for BE calculation.
    fd_info: FDinfo, optional
        Finite difference metadata describing the displacement relative
        to the current reference geometry.

    Returns
    ------
    float
        Converged BE total energy in Hartree
    """
    if energy_args is None:
        energy_args = BEArgs()

    if fd_info is None:
        fobj = fragmentate(
            mol=mol,
            n_BE=energy_args.n_BE,
            frag_type=energy_args.frag_type,
            frozen_core=energy_args.frozen_core,
            additional_args=energy_args.additional_args,
        )
    else:
        if fd_info.ref_mol is None:
            raise RuntimeError("missing finite difference reference geometry.")

        try:
            fobj = fd_info.ref_data["ref_fobj"]
        except KeyError as exc:
            raise RuntimeError("missing reference BE fragmentate object.") from exc

    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()

    mybe = BE(
        mf,
        fobj,
        lo_method=energy_args.lo_method,
        int_transform=energy_args.int_transform,
        auxbasis=energy_args.auxbasis,
        nproc=energy_args.nproc,
        ompnum=energy_args.ompnum,
        initialize_fragment_idx=energy_args.initialize_fragment_idx,
    )

    if energy_args.optimize:
        mybe.optimize(
            solver=energy_args.solver,
            use_cumulant=energy_args.use_cumulant,
            nproc=energy_args.nproc,
            ompnum=energy_args.ompnum,
            only_chem=energy_args.only_chem,
            method=energy_args.method,
            conv_tol=energy_args.conv_tol,
            relax_density=energy_args.relax_density,
            jac_solver=energy_args.jac_solver,
            max_iter=energy_args.max_iter,
            trust_region=energy_args.trust_region,
            step_size=energy_args.step_size,
        )
    else:
        mybe.oneshot(
            solver=energy_args.solver,
            use_cumulant=energy_args.use_cumulant,
            nproc=energy_args.nproc,
            ompnum=energy_args.ompnum,
        )

    return mybe.ebe_tot


def energy_be_frag(mol, energy_args=None, fd_info=None):
    r"""Compute the BEn fragment energy

    Parameters
    ----------
    mol : object
        Molecule object defining the geometry, basis, charge, and spin.
    energy_args: BEArgs, optional
        User defined arguments for BE calculation.
    fd_info: FDinfo, optional
        Finite difference metadata describing the displacement relative
        to the current reference geometry.

    Returns
    ------
    float
        Converged BE fragment energy in Hartree
    """
    if energy_args is None:
        energy_args = BEArgs()

    if fd_info.kind == "multi_displacement":
        raise RuntimeError(
            "energy_be_frag currently supports only single finite-difference "
            f"displacements; got {fd_info.kind!r}"
        )
    if fd_info is None:
        raise RuntimeError("missing finite difference displacement info.")
    else:
        if fd_info.ref_mol is None:
            raise RuntimeError("missing finite difference reference geometry.")

        try:
            ref_fobj = fd_info.ref_data["ref_fobj"]
            ref_mybe = fd_info.ref_data["ref_mybe"]
            frag_per_atom = fd_info.ref_data["frag_per_atom"]
        except KeyError as exc:
            raise RuntimeError("missing reference BE info.") from exc

    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()

    if fd_info.kind in ("reference", "scanner_point"):
        # placeholder energy to allow Gradients.as_scanner()
        # to return (energy, gradient)
        return mf.e_tot

    assert len(fd_info.atom_idx) == 1, (
        "Expected fd_info.atom_idx to have length 1, "
        f"but got {len(fd_info.atom_idx)}: {fd_info.atom_idx}"
    )
    frag_idx = frag_per_atom[fd_info.atom_idx[0]]

    mybe = BE(
        mf,
        ref_fobj,
        lo_method=energy_args.lo_method,
        int_transform=energy_args.int_transform,
        auxbasis=energy_args.auxbasis,
        nproc=energy_args.nproc,
        ompnum=energy_args.ompnum,
        initialize_fragment_idx=[frag_idx],
    )

    S_cross = gto.intor_cross("int1e_ovlp", mol, fd_info.ref_mol)

    # Save the fragment's original ERI bookkeeping
    fobj = mybe.Fobjs[frag_idx]
    orig_dname = fobj.dname
    orig_eri_file = fobj.eri_file

    # Create a dedicated temporary scratch directory for the re-done ERIs
    redo_scratch = WorkDir(
        mybe.scratch_dir / "redo_eri",
        cleanup_at_end=True,
        ensure_empty=True,
    )
    tmp_eri_file = redo_scratch / "eri.h5"

    try:
        fobj.TA = np.linalg.inv(mybe.S) @ S_cross @ ref_mybe.Fobjs[frag_idx].TA
        fobj.dname = "redo" + str(frag_idx)
        fobj.eri_file = tmp_eri_file

        with h5py.File(tmp_eri_file, "w") as file_eri:
            mybe._eri_transform(
                energy_args.int_transform,
                mf._eri,
                file_eri,
                [frag_idx],
            )
            mybe._initialize_fragments(file_eri, False, [frag_idx])

        fobj = mybe.Fobjs[frag_idx]

        eri = get_eri(fobj.dname, fobj.nao, eri_file=tmp_eri_file)
        fobj._mf = get_scfObj(
            fobj.fock + fobj.heff,
            eri,
            fobj.nsocc,
            dm0=fobj.dm0.copy(),
        )

        mc = cc.CCSD(fobj._mf)
        mc.verbose = 0
        mc.incore_complete = True

        eri_embmo = mc.ao2mo()
        eri_embmo.mo_energy = fobj._mf.mo_energy
        eri_embmo.fock = np.diag(fobj._mf.mo_energy)

        mc.kernel(eris=eri_embmo)
        energy = mc.e_tot + mf.energy_nuc()

    finally:
        # Restore the fragment's original ERI bookkeeping
        fobj = mybe.Fobjs[frag_idx]
        fobj.dname = orig_dname
        fobj.eri_file = orig_eri_file

        redo_scratch.cleanup(ignore_error=True)

    return energy


@dataclass
class FDinfo:
    """Container for finite difference metadata."""

    kind: str = "reference"

    atom_idx: list[int] | None = field(default_factory=list)
    axis_idx: list[int] | None = field(default_factory=list)

    delta_bohr: list[float] | None = field(default_factory=list)

    ref_mol: gto.Mole | None = None
    ref_data: dict = field(default_factory=dict)


class Energy(lib.StreamObject):
    r"""PySCF-style wrapper for a custom molecular energy function.

    This class provides a minimal interface around an arbitrary energy function
    ``energy_func(mol)``. It is intended to be compatible with PySCF utilities that
    expect an object with ``mol``, ``kernel()``, and ``as_scanner()`` methods, such as
    ``pyscf.tools.finite_diff``.

    This class supplies repeated single-point energies that can be used by finite
    difference gradient or Hessian drivers.
    """

    def __init__(
        self,
        mol,
        energy_func,
        displacement=1e-4,
        energy_args=None,
        ref_data_func=None,
    ):
        r"""Initialize the custom energy wrapper.

        Parameters
        ----------
        mol : object
            Reference molecule.
        energy_func :
            Callable function with signature ``energy_func(mol) -> float`` returning
            the total energy in Hartree. Should optionally accept ``fd_info`` and
            ``energy_args`` for additional keyword arguments.
        displacement : float, optional
            Finite difference displacement in Bohr, default is 1e-4.
        energy_args : optional
            Additional keyword arguments passed to ``energy_func``.
        ref_data_func: optional
            Callable function with signature ``ref_data_func(mol) -> dict`` returning
            a dictionary containing the necessary reference geometry info for
            ``energy_func``. Should optionally accept additional keyword arguments.
        """
        self.mol = mol
        self.energy_func = energy_func
        self.energy_args = energy_args
        self.e_tot = None
        self.displacement = displacement
        self.ref_data_func = ref_data_func

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
        fd_info: FDinfo, optional
            Finite difference metadata describing the displacement relative
            to the current reference geometry.

        Returns
        ------
        float
            Total energy in Hartree
        """
        if mol is not None:
            self.mol = mol

        if fd_info is None:
            ref_data = (
                self.ref_data_func(self.mol, energy_args=self.energy_args)
                if self.ref_data_func is not None
                else {}
            )
            fd_info = FDinfo(
                kind="reference",
                atom_idx=[],
                axis_idx=[],
                delta_bohr=[0],
                ref_mol=self.mol.copy(),
                ref_data=ref_data,
            )

        self.e_tot = self.energy_func(
            self.mol, energy_args=self.energy_args, fd_info=fd_info
        )
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
                self.ref_data = (
                    parent.ref_data_func(self.ref_mol, energy_args=parent.energy_args)
                    if parent.ref_data_func is not None
                    else {}
                )

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
                    self.ref_data = (
                        parent.ref_data_func(
                            self.ref_mol, energy_args=parent.energy_args
                        )
                        if parent.ref_data_func is not None
                        else {}
                    )
                    fd_info = FDinfo(
                        kind="scanner_point",
                        atom_idx=[],
                        axis_idx=[],
                        delta_bohr=[0],
                        ref_mol=self.ref_mol.copy(),
                        ref_data=self.ref_data,
                    )
                else:
                    diff = coords - self.ref_coords

                    # check if finite_diff is probing current geometry
                    # reset ref_coords and mol_ref if new geometry
                    if not self.is_fd_probe(diff):
                        self.ref_mol = mol.copy()
                        self.ref_coords = coords.copy()
                        diff = coords - self.ref_coords
                        self.ref_data = (
                            parent.ref_data_func(
                                self.ref_mol, energy_args=parent.energy_args
                            )
                            if parent.ref_data_func is not None
                            else {}
                        )

                    displaced = np.reshape(diff, -1)
                    displaced_idx = np.where(np.abs(displaced) > 1e-12)[0]

                    fd_info = FDinfo(
                        kind="reference",
                        atom_idx=[idx // 3 for idx in displaced_idx],
                        axis_idx=[idx % 3 for idx in displaced_idx],
                        delta_bohr=[displaced[idx] for idx in displaced_idx],
                        ref_mol=self.ref_mol.copy(),
                        ref_data=self.ref_data,
                    )
                    if len(displaced_idx) == 1:
                        fd_info.kind = "single_displacement"
                    elif len(displaced_idx) > 1:
                        fd_info.kind = "multi_displacement"

                parent.mol = mol
                parent.e_tot = parent.energy_func(
                    mol, fd_info=fd_info, energy_args=parent.energy_args
                )

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
