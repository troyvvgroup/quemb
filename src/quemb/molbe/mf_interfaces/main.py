# ruff: noqa: PLC0415

from typing import Literal

import h5py
from pyscf.gto import Mole
from pyscf.lib.chkfile import load, load_mol, save, save_mol
from pyscf.scf.hf import RHF
from typing_extensions import assert_never

from quemb.molbe.mf_interfaces._orca_interface import get_mf_orca
from quemb.molbe.mf_interfaces._pyscf_interface import create_mf, get_mf_psycf
from quemb.shared.helper import timer
from quemb.shared.manage_scratch import WorkDir
from quemb.shared.typing import PathLike

SCF_Backends = Literal["pyscf", "orca", "orca-RIJCOSX", "orca-RIJONX"]


@timer.timeit
def get_mf(
    mol: Mole,
    *,
    n_procs: int = 1,
    work_dir: WorkDir | None = None,
    backend: SCF_Backends = "pyscf",
) -> RHF:
    """
    Compute the mean-field (SCF) object for a given molecule using the selected backend.

    Supports multiple SCF backends, including PySCF and ORCA, with optional RIJCOSX
    acceleration for large systems. The ORCA runs are isolated in a working directory,
    which can be provided or inferred from the environment.

    Parameters
    ----------
    mol :
        The molecule to perform the SCF calculation on.
    n_procs :
        Number of processor cores to use (only relevant for ORCA). Default is 1.
    work_dir :
        Working directory for external backend calculations (e.g., ORCA).
        If None, a directory is created based on the environment.
    backend :
        The SCF backend to use: "pyscf", "orca", or "orca-RIJCOSX".

        .. note::

            Using any of the ORCA options requires ``orca`` (version >= 6.1)
            in your path and the ORCA python interface
            (`OPI <https://www.faccts.de/docs/opi/nightly/docs/>`_)
            to be installed.

    Returns
    -------
        The resulting mean-field (RHF) object from the selected backend.
    """

    if work_dir is None:
        work_dir = WorkDir.from_environment(prefix="mf_calculation")

    if backend == "pyscf":
        return get_mf_psycf(mol)
    elif backend == "orca":
        return get_mf_orca(mol, work_dir, n_procs, simple_keywords=[])
    elif backend == "orca-RIJCOSX":
        from opi.input.simple_keywords import (  # type: ignore[import-not-found]
            Approximation,
        )

        return get_mf_orca(
            mol, work_dir, n_procs, simple_keywords=[Approximation.RIJCOSX]
        )
    elif backend == "orca-RIJONX":
        from opi.input.simple_keywords import (  # type: ignore[import-not-found]
            Approximation,
        )

        return get_mf_orca(
            mol, work_dir, n_procs, simple_keywords=[Approximation.RIJONX]
        )
    else:
        assert_never(backend)


def _force_eval_mol(mol: Mole) -> Mole:
    """
    Return a copy of `mol` with explicit atomic coordinates.

    Converts deferred geometries (e.g., from .xyz files) into an inline atom
    string by embedding the evaluated coordinates.

    Parameters
    ----------
    mol:
        A built PySCF Mole object.

    Returns
    -------
        Copy of `mol` with `atom` set to an explicit coordinate string.
    """

    if mol.unit != "Angstrom":
        raise ValueError("Has to be given in Angstrom.")

    new_mol = mol.copy()
    coords = mol.atom_coords(unit="angstrom")
    symbols = [mol.atom_symbol(i) for i in range(mol.natm)]

    new_mol.atom = "\n".join(
        f"{sym} {x:.10f} {y:.10f} {z:.10f}" for sym, (x, y, z) in zip(symbols, coords)
    )
    new_mol.unit = "angstrom"
    new_mol.build()
    return new_mol


def load_scf(chkfile: PathLike) -> tuple[Mole, RHF]:
    """Recreate a PySCF Mole and RHF object from an HDF5 file."""
    mf = create_mf(load_mol(chkfile), **load(chkfile, "scf"))
    return mf.mol.copy(), mf


def dump_scf(mf: RHF, chkfile: PathLike) -> None:
    """Store a PySCF RHF object to an HDF5 file."""
    save_mol(_force_eval_mol(mf.mol), chkfile)

    scf_data = {
        "e_tot": mf.e_tot,
        "mo_energy": mf.mo_energy,
        "mo_occ": mf.mo_occ,
        "mo_coeff": mf.mo_coeff,
    }
    save(chkfile, "scf", scf_data)


def store_to_hdf5(mf: RHF, path: PathLike) -> None:
    """Store a PySCF RHF object to an HDF5 file."""
    with h5py.File(path, "w") as f:
        mol_group = f.create_group("mol")
        mol = _force_eval_mol(mf.mol)

        mol_group.attrs["atom"] = mol.atom
        mol_group.attrs["basis"] = mol.basis
        mol_group.attrs["unit"] = mol.unit
        mol_group.attrs["charge"] = mol.charge
        mol_group.attrs["spin"] = mol.spin

        f.create_dataset("mo_coeff", data=mf.mo_coeff)
        f.create_dataset("mo_energy", data=mf.mo_energy)
        f.create_dataset("mo_occ", data=mf.mo_occ)
        f.attrs["e_tot"] = mf.e_tot


def read_hdf5(path: PathLike) -> RHF:
    """Recreate a PySCF RHF object from an HDF5 file."""
    with h5py.File(path, "r") as f:
        mol_data = f["mol"].attrs
        mol = Mole()
        mol.atom = mol_data["atom"]
        mol.basis = mol_data["basis"]
        mol.unit = mol_data["unit"]
        mol.charge = int(mol_data["charge"])
        mol.spin = int(mol_data["spin"])
        mol.build()

        mo_coeff = f["mo_coeff"][()]
        mo_energy = f["mo_energy"][()]
        mo_occ = f["mo_occ"][()]
        e_tot = float(f.attrs["e_tot"])
    return create_mf(mol, mo_coeff, mo_energy, mo_occ, e_tot)
