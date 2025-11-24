# ruff: noqa: PLC0415

from collections.abc import Mapping
from typing import Final, Literal

import numpy as np
from pyscf.pbc.gto import Cell
from pyscf.pbc.lib.chkfile import load, load_cell, save, save_cell
from pyscf.pbc.scf.khf import KRHF
from typing_extensions import assert_never

from quemb.kbe.mf_interfaces.pyscf_interface import (
    PYSCF_AVAILABLE,
    PySCFArgs,
    create_mf,
    get_mf_pyscf,
)
from quemb.shared.helper import timer
from quemb.shared.manage_scratch import WorkDir
from quemb.shared.typing import Matrix, PathLike

SCF_Backends = Literal["pyscf"]

AdditionalArgs = PySCFArgs

AVAILABLE_BACKENDS: Final[Mapping[SCF_Backends, bool]] = {
    "pyscf": PYSCF_AVAILABLE,
}


@timer.timeit
def get_mf(
    cell: Cell,
    kpts: Matrix[np.floating],
    *,
    work_dir: WorkDir | None = None,
    backend: SCF_Backends = "pyscf",
    additional_args: AdditionalArgs | None = None,
) -> KRHF:
    """
    Compute the mean-field (SCF) object for a given molecule using the selected backend.

    Supports PySCF as the SCF backend.

    Parameters
    ----------
    cell :
        The cell to perform the SCF calculation on.
    work_dir :
        Working directory for external backend calculations (e.g., ORCA).
        If None, a directory is created based on the environment.
    backend :
        The SCF backend to use: "pyscf"


    Returns
    -------
        The resulting mean-field (KRHF) object from the selected backend.
    """

    if work_dir is None:
        work_dir = WorkDir.from_environment(prefix="mf_calculation")

    if backend == "pyscf":
        return get_mf_pyscf(cell, kpts, additional_args=additional_args)
    else:
        assert_never(backend)


def _force_eval_cell(cell: Cell) -> Cell:
    """
    Return a copy of `cell` with explicit atomic coordinates.

    Converts deferred geometries (e.g., from .xyz files) into an inline atom
    string by embedding the evaluated coordinates.

    Parameters
    ----------
    cell:
        A built PySCF Cell object.

    Returns
    -------
        Copy of `cell` with `atom` set to an explicit coordinate string.
    """

    if cell.unit != "angstrom":
        raise ValueError("Has to be given in Angstrom.")

    new_cell = cell.copy()
    coords = cell.atom_coords(unit="angstrom")
    symbols = [cell.atom_symbol(i) for i in range(cell.natm)]

    new_cell.atom = "\n".join(
        f"{sym} {x:.10f} {y:.10f} {z:.10f}" for sym, (x, y, z) in zip(symbols, coords)
    )
    new_cell.unit = "angstrom"
    new_cell.build()
    return new_cell


def load_scf(chkfile: PathLike) -> tuple[Cell, KRHF]:
    """Recreate a PySCF Cell and KRHF object from an HDF5 file."""
    mf = create_mf(load_cell(chkfile), **load(chkfile, "scf"))
    return mf.cell.copy(), mf


def dump_scf(mf: KRHF, chkfile: PathLike) -> None:
    """Store a PySCF KRHF object to an HDF5 file."""
    save_cell(_force_eval_cell(mf.cell), chkfile)

    scf_data = {
        "e_tot": mf.e_tot,
        "mo_energy": mf.mo_energy,
        "mo_occ": mf.mo_occ,
        "mo_coeff": mf.mo_coeff,
        "kpts": mf.kpts,
    }
    save(chkfile, "scf", scf_data)
    save(chkfile, "kpts", mf.kpts)
