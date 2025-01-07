# Author(s): Oinam Romesh Meitei

from attrs import define
from numpy import arange, exp, float64, sqrt
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
    hf_veff: Matrix[float64]
    hcore: Matrix[float64]
    S: Matrix[float64]
    C: Matrix[float64]
    hf_dm: Matrix[float64]
    hf_etot: float
    W: Matrix[float64]
    lmo_coeff: Matrix[float64]
    enuc: float
    ek: float
    E_core: float
    C_core: float
    P_core: float
    core_veff: float


def print_energy_cumulant(ecorr, e_V_Kapprox, e_F_dg, e_hf):
    # Print energy results
    print("-----------------------------------------------------", flush=True)
    print(" BE ENERGIES with cumulant-based expression", flush=True)

    print("-----------------------------------------------------", flush=True)
    print(" E_BE = E_HF + Tr(F del g) + Tr(V K_approx)", flush=True)
    print(" E_HF            : {:>14.8f} Ha".format(e_hf), flush=True)
    print(" Tr(F del g)     : {:>14.8f} Ha".format(e_F_dg), flush=True)
    print(" Tr(V K_aprrox)  : {:>14.8f} Ha".format(e_V_Kapprox), flush=True)
    print(" E_BE            : {:>14.8f} Ha".format(ecorr + e_hf), flush=True)
    print(" Ecorr BE        : {:>14.8f} Ha".format(ecorr), flush=True)
    print("-----------------------------------------------------", flush=True)

    print(flush=True)


def print_energy_noncumulant(be_tot, e1, ec, e2, e_hf, e_nuc):
    print("-----------------------------------------------------", flush=True)
    print(" BE ENERGIES with non-cumulant expression", flush=True)
    print("-----------------------------------------------------", flush=True)
    print(" E_BE = E_1 + E_C + E_2 + E_nuc", flush=True)
    print(" E_HF            : {:>14.8f} Ha".format(e_hf), flush=True)
    print(" E_Nuc           : {:>14.8f} Ha".format(e_nuc), flush=True)
    print(" E_BE total      : {:>14.8f} Ha".format(be_tot + e_nuc), flush=True)
    print(" E_1             : {:>14.8f} Ha".format(e1), flush=True)
    print(" E_C             : {:>14.8f} Ha".format(ec), flush=True)
    print(" E_2             : {:>14.8f} Ha".format(e2), flush=True)
    print(" Ecorr BE        : {:>14.8f} Ha".format(be_tot + e_nuc - e_hf), flush=True)
    print("-----------------------------------------------------", flush=True)

    print(flush=True)
