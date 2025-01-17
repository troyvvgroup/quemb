# Author(s): Oinam Romesh Meitei

import scipy.linalg
from numpy import array, complex128, eye, zeros

from quemb.shared.helper import unused


def schmidt_decomp_svd(rdm, Frag_sites):
    """
    Perform Schmidt decomposition on the orbital coefficients in the real space.

    This function decomposes the molecular orbitals into fragment and environment parts
    using the Schmidt decomposition method. It computes the transformation matrix (TA)
    which includes both the fragment orbitals and the entangled bath.

    Parameters
    ----------
    rdm : numpy.ndarray
        Density matrix (HF) in the real space.
    Frag_sites : list of int
        List of fragment sites (indices).

    Returns
    -------
    numpy.ndarray
        Transformation matrix (TA) including both fragment and entangled bath orbitals.
    """
    thres = 1.0e-10
    Tot_sites = rdm.shape[0]

    Fragsites = [i if i >= 0 else Tot_sites + i for i in Frag_sites]

    Env_sites1 = array([i for i in range(Tot_sites) if i not in Fragsites])
    nfs = len(Frag_sites)

    Denv = rdm[Env_sites1][:, Fragsites]
    U, sigma, V = scipy.linalg.svd(Denv, full_matrices=False, lapack_driver="gesvd")
    unused(V)
    nbath = (sigma >= thres).sum()
    TA = zeros((Tot_sites, nfs + nbath), dtype=complex128)
    TA[Fragsites, :nfs] = eye(nfs)
    TA[Env_sites1, nfs:] = U[:, :nbath]

    return TA
