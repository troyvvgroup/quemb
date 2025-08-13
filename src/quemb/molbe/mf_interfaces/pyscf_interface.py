from typing import Final

import numpy as np
from pyscf.gto import Mole
from pyscf.scf.hf import RHF

from quemb.shared.helper import normalize_column_signs
from quemb.shared.typing import Matrix, Vector

PYSCF_AVAILABLE: Final = True


def create_mf(
    mol: Mole,
    mo_coeff: Matrix[np.floating],
    mo_energy: Vector[np.floating],
    mo_occ: Vector[np.floating],
    e_tot: float,
) -> RHF:
    """Create a pyscf mean-field object from data **without** running a calculation"""
    mf = RHF(mol)

    mf.mo_coeff = normalize_column_signs(mo_coeff)
    mf.mo_energy = mo_energy
    mf.mo_occ = mo_occ
    mf.e_tot = e_tot
    mf.converged = True
    return mf


def get_mf_pyscf(mol: Mole) -> RHF:
    "Run an RHF calculation in pyscf"
    mf = RHF(mol)
    mf.kernel()
    mf.mo_coeff = normalize_column_signs(mf.mo_coeff)
    return mf
