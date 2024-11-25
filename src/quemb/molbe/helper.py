# Author(s): Oinam Romesh Meitei
#            Leah Weisburn

import functools

import h5py
import numpy


def get_frag_energy(
    mo_coeffs,
    nsocc,
    nfsites,
    efac,
    TA,
    h1,
    hf_veff,
    rdm1,
    rdm2s,
    dname,
    eri_file="eri_file.h5",
    veff0=None,
):
    """
    Compute the fragment energy.

    This function calculates the energy contribution of a fragment within a
    larger molecular system using the provided molecular orbital coefficients,
    density matrices, and effective potentials.

    Parameters
    ----------
    mo_coeffs : numpy.ndarray
        Molecular orbital coefficients.
    nsocc : int
        Number of occupied orbitals.
    nfsites : int
        Number of fragment sites.
    efac : list
        List containing energy scaling factors and indices.
    TA : numpy.ndarray
        Transformation matrix.
    h1 : numpy.ndarray
        One-electron Hamiltonian.
    hf_veff : numpy.ndarray
        Hartree-Fock effective potential.
    rdm1 : numpy.ndarray
        One-particle density matrix.
    rdm2s : numpy.ndarray
        Two-particle density matrix.
    dname : str
        Dataset name in the HDF5 file.
    eri_file : str, optional
        Filename of the HDF5 file containing the electron repulsion integrals.
        Defaults to 'eri_file.h5'.

    Returns
    -------
    list
        List containing the energy contributions: [e1_tmp, e2_tmp, ec_tmp].
    """

    # Rotate the RDM1 into the MO basis
    rdm1s_rot = mo_coeffs @ rdm1 @ mo_coeffs.T * 0.5

    # Construct the Hartree-Fock 1-RDM
    hf_1rdm = numpy.dot(mo_coeffs[:, :nsocc], mo_coeffs[:, :nsocc].conj().T)

    # Compute the difference between the rotated RDM1 and the Hartree-Fock 1-RDM
    delta_rdm1 = 2 * (rdm1s_rot - hf_1rdm)

    if veff0 is None:
        # Compute the effective potential in the transformed basis
        veff0 = functools.reduce(numpy.dot, (TA.T, hf_veff, TA))

    # Calculate the one-electron and effective potential energy contributions
    e1 = numpy.einsum("ij,ij->i", h1[:nfsites], delta_rdm1[:nfsites])
    ec = numpy.einsum("ij,ij->i", veff0[:nfsites], delta_rdm1[:nfsites])

    if TA.ndim == 3:
        jmax = TA[0].shape[1]
    else:
        jmax = TA.shape[1]

    # Load the electron repulsion integrals from the HDF5 file
    r = h5py.File(eri_file, "r")
    eri = r[dname][()]
    r.close()

    # Rotate the RDM2 into the MO basis
    rdm2s = numpy.einsum(
        "ijkl,pi,qj,rk,sl->pqrs", 0.5 * rdm2s, *([mo_coeffs] * 4), optimize=True
    )

    # Initialize the two-electron energy contribution
    e2 = numpy.zeros_like(e1)

    # Calculate the two-electron energy contribution
    for i in range(nfsites):
        for j in range(jmax):
            ij = i * (i + 1) // 2 + j if i > j else j * (j + 1) // 2 + i
            Gij = rdm2s[i, j, :jmax, :jmax].copy()
            Gij[numpy.diag_indices(jmax)] *= 0.5
            Gij += Gij.T
            e2[i] += Gij[numpy.tril_indices(jmax)] @ eri[ij]

    # Sum the energy contributions
    e_ = e1 + e2 + ec

    # Initialize temporary energy variables
    etmp = 0.0
    e1_tmp = 0.0
    e2_tmp = 0.0
    ec_tmp = 0.0

    # Calculate the total energy contribution for the specified fragment indices
    for i in efac[1]:
        etmp += efac[0] * e_[i]
        e1_tmp += efac[0] * e1[i]
        e2_tmp += efac[0] * e2[i]
        ec_tmp += efac[0] * ec[i]

    return [e1_tmp, e2_tmp, ec_tmp]


def get_frag_energy_u(
    mo_coeffs,
    nsocc,
    nfsites,
    efac,
    TA,
    h1,
    hf_veff,
    rdm1,
    rdm2s,
    dname,
    eri_file="eri_file.h5",
    gcores=None,
    frozen=False,
    veff0=None,
):
    """
    Compute the fragment energy for unrestricted calculations

    This function calculates the energy contribution of a fragment within
    a larger molecular system using the provided molecular orbital coefficients,
    density matrices, and effective potentials.

    Parameters
    ----------
    mo_coeffs : tuple of numpy.ndarray
        Molecular orbital coefficients.
    nsocc : tuple of int
        Number of occupied orbitals.
    nfsites : tuple of int
        Number of fragment sites.
    efac : tuple of list
        List containing energy scaling factors and indices.
    TA : tuple of numpy.ndarray
        Transformation matrix.
    h1 : tuple of numpy.ndarray
        One-electron Hamiltonian.
    hf_veff : tuple of numpy.ndarray
        Hartree-Fock effective potential.
    rdm1 : tuple of numpy.ndarray
        One-particle density matrix.
    rdm2s : tuple of numpy.ndarray
        Two-particle density matrix.
    dname : list
        Dataset name in the HDF5 file.
    eri_file : str, optional
        Filename of the HDF5 file containing the electron repulsion integrals.
        Defaults to 'eri_file.h5'.
    gcores :

    frozen : bool, optional
        Indicate frozen core. Default is False

    Returns
    -------
    list
        List containing the energy contributions: [e1_tmp, e2_tmp, ec_tmp].
    """

    # Rotate the RDM1 into the MO basis for both spins
    rdm1s_rot = [mo_coeffs[s] @ rdm1[s] @ mo_coeffs[s].T for s in [0, 1]]

    # Construct the Hartree-Fock RDM1 for both spin the the Schmidt space
    hf_1rdm = [
        numpy.dot(mo_coeffs[s][:, : nsocc[s]], mo_coeffs[s][:, : nsocc[s]].conj().T)
        for s in [0, 1]
    ]

    # Compute the difference between the rotated RDM1 and the HF RDM1
    delta_rdm1 = [2 * (rdm1s_rot[s] - hf_1rdm[s]) for s in [0, 1]]

    if veff0 is None:
        # Compute thte effective potential in the transformed basis
        veff0 = [
            functools.reduce(numpy.dot, (TA[s].T, hf_veff[s], TA[s])) for s in [0, 1]
        ]

    # For frozen care, remove core potential and Hamiltonian components
    if frozen:
        for s in [0, 1]:
            veff0[s] -= gcores[s]
            h1[s] -= gcores[s]

    # Calculate the one-electron and effective potential energy contributions
    e1 = [
        numpy.einsum("ij,ij->i", h1[s][: nfsites[s]], delta_rdm1[s][: nfsites[s]])
        for s in [0, 1]
    ]
    ec = [
        numpy.einsum("ij,ij->i", veff0[s][: nfsites[s]], delta_rdm1[s][: nfsites[s]])
        for s in [0, 1]
    ]

    jmax = [TA[0].shape[1], TA[1].shape[1]]

    # Load ERIs from the HDF5 file
    r = h5py.File(eri_file, "r")
    Vs = [r[dname[0]][()], r[dname[1]][()], r[dname[2]][()]]
    r.close()

    # Rotate the RDM2 into the MO basis
    rdm2s_k = [
        numpy.einsum(
            "ijkl,pi,qj,rk,sl->pqrs",
            rdm2s[s],
            *([mo_coeffs[s12[0]]] * 2 + [mo_coeffs[s12[1]]] * 2),
            optimize=True,
        )
        for s, s12 in zip([0, 1, 2], [[0, 0], [0, 1], [1, 1]])
    ]

    # Initialize the two-electron energy contribution
    e2 = [numpy.zeros(h1[0].shape[0]), numpy.zeros(h1[1].shape[0])]

    # Calculate the two-electron energy contribution for alpha and beta
    def contract_2e(jmaxs, rdm2_, V_, s, sym):
        e2_ = numpy.zeros(nfsites[s])
        jmax1, jmax2 = [jmaxs] * 2 if isinstance(jmaxs, int) else jmaxs
        for i in range(nfsites[s]):
            for j in range(jmax1):
                ij = i * (i + 1) // 2 + j if i > j else j * (j + 1) // 2 + i
                if sym in [4, 2]:
                    Gij = rdm2_[i, j, :jmax2, :jmax2].copy()
                    Vij = V_[ij]
                else:
                    Gij = rdm2_[:jmax2, :jmax2, i, j].copy()
                    Vij = V_[:, ij]
                Gij[numpy.diag_indices(jmax2)] *= 0.5
                Gij += Gij.T
                e2_[i] += Gij[numpy.tril_indices(jmax2)] @ Vij
        e2_ *= 0.5

        return e2_

    # the first nf are frag sites
    e2ss = [0.0, 0.0]
    e2os = [0.0, 0.0]
    for s in [0, 1]:
        e2ss[s] += contract_2e(jmax[s], rdm2s_k[2 * s], Vs[s], s, sym=4)

    # Calculate the cross-spin two-electron energy contributions
    V = Vs[2]

    # ab
    e2os[0] += contract_2e(jmax, rdm2s_k[1], V, 0, sym=2)
    # ba
    e2os[1] += contract_2e(jmax[::-1], rdm2s_k[1], V, 1, sym=-2)

    e2 = sum(e2ss) + sum(e2os)

    # Sum the energy contributions
    e_ = e1 + e2 + ec

    # Initialize temporary energy variables
    etmp = 0.0
    e1_tmp = 0.0
    e2_tmp = 0.0
    ec_tmp = 0.0

    # Calculate the total energy contribution for the specified fragment indices
    for i in efac[0][1]:
        e2_tmp += efac[0][0] * e2[i]
        for s in [0, 1]:
            etmp += efac[s][0] * e_[s][i]
            e1_tmp += efac[s][0] * e1[s][i]
            ec_tmp += efac[s][0] * ec[s][i]

    return [e1_tmp, e2_tmp, ec_tmp]
