from typing import Final, Literal

import numpy as np
from attrs import define
from pyscf.pbc.gto import Cell
from pyscf.pbc.scf.khf import KRHF

from quemb.shared.typing import Matrix, Vector

PYSCF_AVAILABLE: Final = True


@define(frozen=True, kw_only=True)
class PySCFArgs:
    density_fit: Literal["GDF", "FFDF", "MDF"] | None = None


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


def get_mf_pyscf(
    cell: Cell, kpts: Matrix[np.floating], additional_args: PySCFArgs | None = None
) -> KRHF:
    "Run an KRHF calculation in pyscf"
    mf = KRHF(cell, kpts=kpts)
    if additional_args.density_fit == "GDF":
        mf = mf.density_fit()
    elif additional_args.density_fit == "MDF":
        mf = mf.mix_density_fit()
    else:
        pass  # PySCF does FFDF by default for PBC calculations

    mf.kernel()
    return mf
