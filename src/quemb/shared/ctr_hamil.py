# Author: Minsik Cho
# Center-site Hamiltonian scheme
# Originally written in troyvvgroup/quemb_qpe repository

from typing import Literal, TypeAlias

from numpy import zeros, zeros_like

from pyscf.ao2mo import restore
from pyscf.cc import RCCSDT, RCCSDTQ, UCCSDT

from quemb.kbe.pfrag import Frags as pFrags
from quemb.molbe.helper import get_scfObj
from quemb.molbe.pfrag import Frags

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
