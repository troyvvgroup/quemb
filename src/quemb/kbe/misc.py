# Author(s): Oinam Romesh Meitei

from numpy import arange, exp, sqrt
from pyscf.lib import cartesian_prod
from pyscf.pbc import tools


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


class storePBE:
    def __init__(
        self,
        Nocc,
        hf_veff,
        hcore,
        S,
        C,
        hf_dm,
        hf_etot,
        W,
        lmo_coeff,
        enuc,
        ek,
        E_core,
        C_core,
        P_core,
        core_veff,
    ):
        self.Nocc = Nocc
        self.hf_veff = hf_veff
        self.hcore = hcore
        self.S = S
        self.C = C
        self.hf_dm = hf_dm
        self.hf_etot = hf_etot
        self.W = W
        self.lmo_coeff = lmo_coeff
        self.enuc = enuc
        self.ek = ek
        self.E_core = E_core
        self.C_core = C_core
        self.P_core = P_core
        self.core_veff = core_veff


def print_energy(ecorr, e_V_Kapprox, e_F_dg, e_hf, unitcell_nkpt):
    # Print energy results
    print("-----------------------------------------------------", flush=True)
    print(" BE ENERGIES with cumulant-based expression", flush=True)

    print("-----------------------------------------------------", flush=True)
    print(" E_BE = E_HF + Tr(F del g) + Tr(V K_approx)", flush=True)
    print(" E_HF            : {:>14.8f} Ha".format(e_hf / unitcell_nkpt), flush=True)
    print(" Tr(F del g)     : {:>14.8f} Ha".format(e_F_dg / unitcell_nkpt), flush=True)
    print(
        " Tr(V K_aprrox)  : {:>14.8f} Ha".format(e_V_Kapprox / unitcell_nkpt),
        flush=True,
    )
    print(
        " E_BE            : {:>14.8f} Ha".format((ecorr + e_hf) / unitcell_nkpt),
        flush=True,
    )
    print(" Ecorr BE        : {:>14.8f} Ha".format(ecorr / unitcell_nkpt), flush=True)
    print("-----------------------------------------------------", flush=True)

    print(flush=True)
