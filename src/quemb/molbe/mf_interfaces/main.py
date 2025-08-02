# ruff: noqa: PLC0415

from typing import Literal

from pyscf.gto import Mole
from pyscf.scf.hf import RHF
from typing_extensions import assert_never

from quemb.molbe.mf_interfaces._orca_interface import get_mf_orca
from quemb.molbe.mf_interfaces._pyscf_interface import get_mf_psycf
from quemb.shared.helper import timer
from quemb.shared.manage_scratch import WorkDir

SCF_Backends = Literal["pyscf", "orca", "orca-RIJCOSX"]


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
    else:
        assert_never(backend)
