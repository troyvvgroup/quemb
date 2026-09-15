# Author: Minsik Cho
# Center-site Hamiltonian scheme
# Originally written in troyvvgroup/quemb_qpe repository

from typing import Literal, TypeAlias

from numpy import ndarray, zeros, zeros_like

from pyscf.ao2mo import restore
from pyscf.cc import RCCSDT, RCCSDTQ, UCCSDT

from quemb.kbe.pfrag import Frags as pFrags
from quemb.molbe.helper import get_scfObj
from quemb.molbe.pfrag import Frags
from quemb.shared.typing import Matrix

hcSolvers: TypeAlias = Literal["CCSDT", "CCSDTQ"]
uhcSolvers: TypeAlias = Literal["UCCSDT"]


def build_hc(fobj: Frags | pFrags):
    """Builds center-site Hamiltonian for a given fragment object.

    Parameters
    ----------
    fobj :
        Fragment object

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        Center-site Hamiltonian (Fock-like 1-body part and 2-body part)
    """
    # Number of embedding orbitals
    n_emb_orb = fobj.TA.shape[1] if isinstance(fobj, Frags) else fobj.TA.shape[2]
    # Allocate space for the center-site Hamiltonian
    h1 = zeros_like(fobj.h1)
    eri = restore(1, fobj._mf._eri, n_emb_orb)
    h2 = zeros_like(eri)

    # One electron part
    #     h_{pq} = h_{pq} if p and q in center_idx
    #     h_{pq} = 0.5 h_{pq} if p xor q in center_idx
    #     h_{pq} = 0 otherwise
    fock = fobj.h1 + fobj.heff + 0.5 * fobj.veff
    ind_mask = zeros(fock.shape[0], dtype=float)
    ind_mask[list(fobj.weight_and_relAO_per_center[1])] = 1.0
    h1_weight = (ind_mask[:, None] + ind_mask[None, :]) * 0.5
    h1[:] = fock * h1_weight
    # Two electron part
    #     eri_{pqrs} = {number of p,q,r,s in center_idx} * eri_{pqrs}
    h2_weight = (
        ind_mask[:, None, None, None]
        + ind_mask[None, :, None, None]
        + ind_mask[None, None, :, None]
        + ind_mask[None, None, None, :]
    ) * 0.25
    h2[:] = eri * h2_weight

    return h1, h2


def calc_energy(
    fobj: Frags | pFrags,
    dl: float,
    center_hamil: tuple[Matrix, Matrix],
    solver: hcSolvers,
):
    """Evaluates the energy of the center-site Hamiltonian displaced by a given delta."""
    # Mean-field object
    n_emb_orb = fobj.TA.shape[1] if isinstance(fobj, Frags) else fobj.TA.shape[2]
    eri = restore(1, fobj._mf._eri, n_emb_orb)
    mf = get_scfObj(
        fobj.fock + fobj.heff + dl * center_hamil[0],
        eri + dl * center_hamil[1],
        fobj.nsocc,
    )
    # Correlated solvers
    if solver == "CCSDT":
        mc = RCCSDT(mf)
    elif solver == "CCSDTQ":
        mc = RCCSDTQ(mf)
    else:
        raise ValueError(
            f"Solver not supported by center-site Hamiltonian scheme: {solver}"
        )
    mc.verbose = 0
    # TODO: re-eval these params (or expose)
    mc.conv_tol = 1e-8
    mc.max_cycle = 500
    mc.kernel()

    return mc.e_tot
