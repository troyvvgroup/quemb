from typing import Literal, assert_never

from pyscf.gto import Mole
from pyscf.scf.hf import RHF

from quemb.molbe.mf_interfaces._orca_interface import get_mf_orca
from quemb.molbe.mf_interfaces._pyscf_interface import get_mf_psycf
from quemb.shared.helper import timer
from quemb.shared.manage_scratch import WorkDir

SCF_Backends = Literal["pyscf", "orca", "orca-RIJCOSX"]


@timer.timeit
def get_mf(
    mol: Mole, n_procs: int, work_dir: WorkDir, backend: SCF_Backends = "pyscf"
) -> RHF:
    if backend == "pyscf":
        return get_mf_psycf(mol)
    elif backend == "orca":
        return get_mf_orca(mol, work_dir, n_procs, simple_keywords=[])
    elif backend == "orca-RIJCOSX":
        from opi.input.simple_keywords import (  # noqa: PLC0415
            Approximation,  # type: ignore[import-not-found]
        )

        return get_mf_orca(
            mol, work_dir, n_procs, simple_keywords=[Approximation.RIJCOSX]
        )
    else:
        assert_never(backend)
