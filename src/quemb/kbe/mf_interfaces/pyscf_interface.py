from typing import Final

import numpy as np
from pyscf.pbc.gto import Cell
from pyscf.pbc.scf.khf import KRHF

from quemb.shared.typing import Matrix, Vector

PYSCF_AVAILABLE: Final = True


def create_mf(
    cell: Cell,
    mo_coeff: Matrix[np.floating],
    mo_energy: Vector[np.floating],
    mo_occ: Vector[np.floating],
    e_tot: float,
    kpts: Matrix[np.floating],
) -> KRHF:
    """Create a pyscf mean-field object from data **without** running a calculation"""
    mf = KRHF(cell)

    mf.mo_coeff = mo_coeff
    mf.mo_energy = mo_energy
    mf.mo_occ = mo_occ
    mf.e_tot = e_tot
    mf.kpts = kpts
    mf.converged = True
    return mf


def get_mf_pyscf(cell: Cell, kpts: Matrix[np.floating]) -> KRHF:
    "Run an KRHF (GDF) calculation in pyscf"
    mf = KRHF(cell, kpts=kpts).density_fit()
    mf.kernel()
    mf.mo_coeff = mf.mo_coeff
    return mf
