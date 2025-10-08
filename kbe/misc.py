# Author(s): Oinam Romesh Meitei

from attrs import define
from numpy import arange, exp, floating, sqrt
from pyscf.lib import cartesian_prod
from pyscf.pbc import tools

from quemb.shared.typing import Matrix


def sgeom(cell, kmesh=None):
    """
    Get a supercell pyscf.pbc.gto.cell.Cell object

    Parameters
    ----------
    cell : pyscf.pbc.gto.cell.Cell
    kmesh : list of int
       Number of k-points in each lattice vector dimension
    """
    return tools.super_cell(cell, kmesh)


def get_phase(cell, kpts, kmesh):
    a_vec = cell.lattice_vectors()
    Ts = cartesian_prod((arange(kmesh[0]), arange(kmesh[1]), arange(kmesh[2])))
    NRs = Ts.shape[0]
    return 1 / sqrt(NRs) * exp(1j * (Ts @ a_vec @ kpts.T))


def get_phase1(cell, kpts, kmesh):
    a_vec = cell.lattice_vectors()
    Ts = cartesian_prod((arange(kmesh[0]), arange(kmesh[1]), arange(kmesh[2])))
    return exp(-1.0j * (Ts @ a_vec @ kpts.T))


@define
class storePBE:
    Nocc: int
    hf_veff: Matrix[floating]
    hcore: Matrix[floating]
    S: Matrix[floating]
    C: Matrix[floating]
    hf_dm: Matrix[floating]
    hf_etot: float
    W: Matrix[floating]
    lmo_coeff: Matrix[floating]
    enuc: float
    ek: float
    E_core: float
    C_core: Matrix[floating]
    P_core: Matrix[floating]
    core_veff: Matrix[floating]


def print_energy(ecorr, e_V_Kapprox, e_F_dg, e_hf, unitcell_nkpt):
    # Print energy results
    print("-----------------------------------------------------", flush=True)
    print(" BE ENERGIES with cumulant-based expression", flush=True)

    print("-----------------------------------------------------", flush=True)
    print(" E_BE = E_HF + Tr(F del g) + Tr(V K_approx)", flush=True)
    print(f" E_HF            : {e_hf / unitcell_nkpt:>14.8f} Ha", flush=True)
    print(f" Tr(F del g)     : {e_F_dg / unitcell_nkpt:>14.8f} Ha", flush=True)
    print(
        f" Tr(V K_aprrox)  : {e_V_Kapprox / unitcell_nkpt:>14.8f} Ha",
        flush=True,
    )
    print(
        f" E_BE            : {(ecorr + e_hf) / unitcell_nkpt:>14.8f} Ha",
        flush=True,
    )
    print(f" Ecorr BE        : {ecorr / unitcell_nkpt:>14.8f} Ha", flush=True)
    print("-----------------------------------------------------", flush=True)

    print(flush=True)
