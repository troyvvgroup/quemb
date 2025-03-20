# Author(s): Oinam Romesh Meitei
from __future__ import annotations

from numpy import asarray, complex128, float64, zeros
from numpy.linalg import multi_dot
from pyscf import scf

from quemb.shared.helper import unused


def get_veff(eri_, dm, S, TA, hf_veff, return_veff0=False):
    """
    Calculate the effective HF potential (Veff) for a given density matrix
    and electron repulsion integrals.

    This function computes the effective potential by transforming the density matrix,
    computing the Coulomb (J) and exchange (K) integrals.

    Parameters
    ----------
    eri_ : numpy.ndarray
        Electron repulsion integrals.
    dm : numpy.ndarray
        Density matrix. 2D array.
    S : numpy.ndarray
        Overlap matrix.
    TA : numpy.ndarray
        Transformation matrix.
    hf_veff : numpy.ndarray
        Hartree-Fock effective potential for the full system.

    """

    # construct rdm
    nk, nao, neo = TA.shape
    unused(nao)
    P_ = zeros((neo, neo), dtype=complex128)
    for k in range(nk):
        Cinv = TA[k].conj().T @ S[k]
        P_ += multi_dot((Cinv, dm[k], Cinv.conj().T))
    P_ /= float(nk)

    P_ = asarray(P_.real, dtype=float64)

    eri_ = asarray(eri_, dtype=float64)
    vj, vk = scf.hf.dot_eri_dm(eri_, P_, hermi=1, with_j=True, with_k=True)
    Veff_ = vj - 0.5 * vk

    # remove core contribution from hf_veff

    Veff0 = zeros((neo, neo), dtype=complex128)
    for k in range(nk):
        Veff0 += multi_dot((TA[k].conj().T, hf_veff[k], TA[k]))
    Veff0 /= float(nk)

    Veff = Veff0 - Veff_

    if return_veff0:
        return (Veff0, Veff)

    return Veff
