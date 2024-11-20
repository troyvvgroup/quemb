# Author(s): Oinam Romesh Meitei

import numpy, sys, functools


def get_veff(eri_, dm, S, TA, hf_veff, return_veff0=False):
    import functools
    from pyscf import scf

    """
    Calculate the effective HF potential (Veff) for a given density matrix and electron repulsion integrals.

    This function computes the effective potential by transforming the density matrix, computing the Coulomb (J) and
    exchange (K) integrals.

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
    P_ = numpy.zeros((neo, neo), dtype=numpy.complex128)
    for k in range(nk):
        Cinv = numpy.dot(TA[k].conj().T, S[k])
        P_ += functools.reduce(numpy.dot, (Cinv, dm[k], Cinv.conj().T))
    P_ /= float(nk)

    P_ = numpy.asarray(P_.real, dtype=numpy.double)

    eri_ = numpy.asarray(eri_, dtype=numpy.double)
    vj, vk = scf.hf.dot_eri_dm(eri_, P_, hermi=1, with_j=True, with_k=True)
    Veff_ = vj - 0.5 * vk

    # remove core contribution from hf_veff

    Veff0 = numpy.zeros((neo, neo), dtype=numpy.complex128)
    for k in range(nk):
        Veff0 += functools.reduce(numpy.dot, (TA[k].conj().T, hf_veff[k], TA[k]))
    Veff0 /= float(nk)

    Veff = Veff0 - Veff_

    if return_veff0:
        return (Veff0, Veff)

    return Veff
