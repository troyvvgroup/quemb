# ruff: noqa: PLC0415

from collections.abc import Mapping
from typing import Final, Literal

from pyscf.gto import Mole
from pyscf.lib.chkfile import load, load_mol, save, save_mol
from pyscf.scf.hf import RHF
from typing_extensions import assert_never

from quemb.molbe.mf_interfaces.orca_interface import (
    ORCA_AVAILABLE,
    OrcaArgs,
    get_mf_orca,
    get_orca_basis,
)
from quemb.molbe.mf_interfaces.pyscf_interface import (
    PYSCF_AVAILABLE,
    create_mf,
    get_mf_psycf,
)
from quemb.shared.helper import timer
from quemb.shared.manage_scratch import WorkDir
from quemb.shared.typing import PathLike

SCF_Backends = Literal["pyscf", "orca"]

AdditionalArgs = OrcaArgs

AVAILABLE_BACKENDS: Final[Mapping[SCF_Backends, bool]] = {
    "pyscf": PYSCF_AVAILABLE,
    "orca": ORCA_AVAILABLE,
}


@timer.timeit
def get_mf(
    mol: Mole,
    *,
    work_dir: WorkDir | None = None,
    backend: SCF_Backends = "pyscf",
    additional_args: AdditionalArgs | None = None,
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

    additional_args :



    Returns
    -------
        The resulting mean-field (RHF) object from the selected backend.
    """

    if work_dir is None:
        work_dir = WorkDir.from_environment(prefix="mf_calculation")

    if backend == "pyscf":
        return get_mf_psycf(mol)
    elif backend == "orca":
        from opi.input.blocks.block_basis import BlockBasis

        if additional_args is None:
            additional_args = OrcaArgs(
                simple_keywords=[],
                blocks=[BlockBasis(basis=get_orca_basis(mol))],
            )
        else:
            assert isinstance(additional_args, OrcaArgs)
        return get_mf_orca(
            mol,
            work_dir,
            n_procs=additional_args.n_procs,
            simple_keywords=additional_args.simple_keywords,
            blocks=additional_args.blocks,
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

    if mol.unit != "angstrom":
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
