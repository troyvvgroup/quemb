# Author(s): Oinam Romesh Meitei
#            Leah Weisburn


from typing import TypeVar

import h5py
from numpy import (
    array,
    asarray,
    diag_indices,
    einsum,
    eye,
    float64,
    tril_indices,
    zeros,
    zeros_like,
)
from numpy.linalg import multi_dot
from pyscf import ao2mo, gto, lib, scf
from pyscf.gto.mole import Mole
from pyscf.pbc.gto.cell import Cell

from quemb.shared.helper import ncore_


def get_veff(eri_, dm, S, TA, hf_veff):
    """
    Calculate the effective HF potential (Veff) for a given density matrix and
    electron repulsion integrals.

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

    Returns
    -------
    numpy.ndarray
        Effective HF potential in the embedding basis.
    """

    # Transform the density matrix
    ST = S @ TA
    P_ = multi_dot((ST.T, dm, ST))

    # Ensure the transformed density matrix and ERI are real and double-precision
    P_ = asarray(P_.real, dtype=float64)
    eri_ = asarray(eri_, dtype=float64)

    # Compute the Coulomb (J) and exchange (K) integrals
    vj, vk = scf.hf.dot_eri_dm(eri_, P_, hermi=1, with_j=True, with_k=True)
    Veff_ = vj - 0.5 * vk
    Veff0 = multi_dot((TA.T, hf_veff, TA))
    Veff = Veff0 - Veff_

    return Veff, Veff0


# create pyscf pbc scf object
def get_scfObj(
    h1,
    Eri,
    nocc,
    dm0=None,
):
    """Initialize and run a restricted Hartree-Fock (RHF) calculation.

    This function sets up an SCF (Self-Consistent Field) object using the provided
    one-electron Hamiltonian, electron repulsion integrals, and number
    of occupied orbitals. It then runs the SCF procedure, optionally using an initial
    density matrix.

    Parameters
    ----------
    h1 : numpy.ndarray
        One-electron Hamiltonian matrix.
    Eri : numpy.ndarray
        Electron repulsion integrals.
    nocc : int
        Number of occupied orbitals.
    dm0 : numpy.ndarray, optional
        Initial density matrix. If not provided, the SCF calculation will start
        from scratch. Defaults to None.

    Returns
    -------
    mf_ : pyscf.scf.hf.RHF
        The SCF object after running the Hartree-Fock calculation.
    """
    # from 40-customizing_hamiltonian.py in pyscf examples
    nao = h1.shape[0]

    # Initialize a dummy molecule with the required number of electrons
    S = eye(nao)
    mol = gto.M()
    mol.nelectron = nocc * 2
    mol.incore_anyway = True

    # Initialize an RHF object
    mf_ = scf.RHF(mol)
    mf_.get_hcore = lambda *args: h1  # noqa: ARG005
    mf_.get_ovlp = lambda *args: S  # noqa: ARG005
    mf_._eri = Eri
    mf_.incore_anyway = True
    mf_.max_cycle = 50
    mf_.verbose = 0

    # Run the SCF calculation
    if dm0 is None:
        mf_.kernel()
    else:
        mf_.kernel(dm0=dm0)

    # Check if the SCF calculation converged
    if not mf_.converged:
        print(flush=True)
        print(
            "WARNING!!! SCF not convereged - applying level_shift=0.2, diis_space=25 ",
            flush=True,
        )
        print(flush=True)
        mf_.verbose = 0
        mf_.level_shift = 0.2
        mf_.diis_space = 25
        if dm0 is None:
            mf_.kernel()
        else:
            mf_.kernel(dm0=dm0)
        if not mf_.converged:
            print(flush=True)
            print("WARNING!!! SCF still not convereged!", flush=True)
            print(flush=True)
        else:
            print(flush=True)
            print("SCF Converged!", flush=True)
            print(flush=True)

    return mf_


def get_eri(i_frag, Nao, symm=8, ignore_symm=False, eri_file="eri_file.h5"):
    """
    Retrieve and optionally restore electron repulsion integrals (ERI)
    from an HDF5 file.

    This function reads the ERI for a given fragment from an HDF5 file, and optionally
    restores the symmetry of the ERI.

    Parameters
    ----------
    i_frag : str
        Fragment identifier used to locate the ERI data in the HDF5 file.
    Nao : int
        Number of atomic orbitals.
    symm : int, optional
        Symmetry of the ERI. Defaults to 8.
    ignore_symm : bool, optional
        If True, the symmetry step is skipped. Defaults to False.
    eri_file : str, optional
        Filename of the HDF5 file containing the electron repulsion integrals.
        Defaults to 'eri_file.h5'.

    Returns
    -------
    numpy.ndarray
        Electron repulsion integrals, possibly restored with symmetry.
    """
    # Open the HDF5 file and read the ERI for the specified fragment
    with h5py.File(eri_file, "r") as r:
        eri__ = array(r.get(i_frag))

        # Optionally restore the symmetry of the ERI
        if not ignore_symm:
            # Set the number of threads for the library to 1
            lib.num_threads(1)
            eri__ = ao2mo.restore(symm, eri__, Nao)

    return eri__


def get_core(mol: Mole | Cell) -> tuple[int, list[int], list[int]]:
    """
    Calculate the number of cores for each atom in the molecule.

    Parameters
    ----------
    mol :
        Molecule or cell object from PySCF.

    Returns
    -------
    tuple
        (Ncore, idx, corelist)
    """
    Ncore = 0
    idx = []
    corelist = []
    for ix, bas in enumerate(mol.aoslice_by_atom()):
        ncore = ncore_(mol.atom_charge(ix))
        corelist.append(ncore)
        Ncore += ncore
        idx.extend([k for k in range(bas[2] + ncore, bas[3])])

    return (Ncore, idx, corelist)


def get_frag_energy(
    mo_coeffs,
    nsocc,
    n_frag,
    centerweight_and_relAO_per_center,
    TA,
    h1,
    rdm1,
    rdm2s,
    dname,
    veff0=None,
    veff=None,
    use_cumulant=True,
    eri_file="eri_file.h5",
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
    n_frag : int
        Number of fragment sites.
    centerweight_and_relAO_per_center :
        List containing energy scaling factors and indices.
    TA : numpy.ndarray
        Transformation matrix.
    h1 : numpy.ndarray
        One-electron Hamiltonian.
    rdm1 : numpy.ndarray
        One-particle density matrix.
    rdm2s : numpy.ndarray
        Two-particle density matrix.
    dname : str
        Dataset name in the HDF5 file.
    veff0 : numpy.ndarray
        veff0 matrix, the original hf_veff in the fragment Schmidt space
    veff : numpy.ndarray
        veff for non-cumulant energy expression
    use_cumulant: bool
        Whether to return cumulant energy, by default True
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
    hf_1rdm = mo_coeffs[:, :nsocc] @ mo_coeffs[:, :nsocc].conj().T

    if use_cumulant:
        # Compute the difference between the rotated RDM1 and the Hartree-Fock 1-RDM
        delta_rdm1 = 2 * (rdm1s_rot - hf_1rdm)

        # Calculate the one-electron contributions
        e1 = einsum("ij,ij->i", h1[:n_frag], delta_rdm1[:n_frag])
        ec = einsum("ij,ij->i", veff0[:n_frag], delta_rdm1[:n_frag])

    else:
        # Calculate the one-electron and effective potential energy contributions
        e1 = 2 * einsum("ij,ij->i", h1[:n_frag], rdm1s_rot[:n_frag])
        ec = einsum("ij,ij->i", veff[:n_frag], rdm1s_rot[:n_frag])

    if TA.ndim == 3:
        jmax = TA[0].shape[1]
    else:
        jmax = TA.shape[1]

    # Load the electron repulsion integrals from the HDF5 file
    with h5py.File(eri_file, "r") as r:
        eri = r[dname][()]

    # Rotate the RDM2 into the MO basis
    rdm2s = einsum(
        "ijkl,pi,qj,rk,sl->pqrs", 0.5 * rdm2s, *([mo_coeffs] * 4), optimize=True
    )

    # Initialize the two-electron energy contribution
    e2 = zeros_like(e1)

    # Calculate the two-electron energy contribution
    for i in range(n_frag):
        for j in range(jmax):
            ij = i * (i + 1) // 2 + j if i > j else j * (j + 1) // 2 + i
            Gij = rdm2s[i, j, :jmax, :jmax].copy()
            Gij[diag_indices(jmax)] *= 0.5
            Gij += Gij.T
            e2[i] += Gij[tril_indices(jmax)] @ eri[ij]

    # Sum the energy contributions
    e_ = e1 + e2 + ec

    # Initialize temporary energy variables
    etmp = 0.0
    e1_tmp = 0.0
    e2_tmp = 0.0
    ec_tmp = 0.0

    # Calculate the total energy contribution for the specified fragment indices
    for i in centerweight_and_relAO_per_center[1]:
        etmp += centerweight_and_relAO_per_center[0] * e_[i]
        e1_tmp += centerweight_and_relAO_per_center[0] * e1[i]
        e2_tmp += centerweight_and_relAO_per_center[0] * e2[i]
        ec_tmp += centerweight_and_relAO_per_center[0] * ec[i]

    return [e1_tmp, e2_tmp, ec_tmp]


def get_frag_energy_u(
    mo_coeffs,
    nsocc,
    n_frag,
    centerweight_and_relAO_per_center,
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
    n_frag : tuple of int
        Number of fragment sites.
    centerweight_and_relAO_per_center :
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
        mo_coeffs[s][:, : nsocc[s]] @ mo_coeffs[s][:, : nsocc[s]].conj().T
        for s in [0, 1]
    ]

    # Compute the difference between the rotated RDM1 and the HF RDM1
    delta_rdm1 = [2 * (rdm1s_rot[s] - hf_1rdm[s]) for s in [0, 1]]

    if veff0 is None:
        # Compute thte effective potential in the transformed basis
        veff0 = [multi_dot((TA[s].T, hf_veff[s], TA[s])) for s in [0, 1]]

    # For frozen care, remove core potential and Hamiltonian components
    if frozen:
        for s in [0, 1]:
            veff0[s] -= gcores[s]
            h1[s] -= gcores[s]

    # Calculate the one-electron and effective potential energy contributions
    e1 = [
        einsum("ij,ij->i", h1[s][: n_frag[s]], delta_rdm1[s][: n_frag[s]])
        for s in [0, 1]
    ]
    ec = [
        einsum("ij,ij->i", veff0[s][: n_frag[s]], delta_rdm1[s][: n_frag[s]])
        for s in [0, 1]
    ]

    jmax = [TA[0].shape[1], TA[1].shape[1]]

    # Load ERIs from the HDF5 file
    with h5py.File(eri_file, "r") as r:
        Vs = [r[dname[0]][()], r[dname[1]][()], r[dname[2]][()]]

    # Rotate the RDM2 into the MO basis
    rdm2s_k = [
        einsum(
            "ijkl,pi,qj,rk,sl->pqrs",
            rdm2s[s],
            *([mo_coeffs[s12[0]]] * 2 + [mo_coeffs[s12[1]]] * 2),
            optimize=True,
        )
        for s, s12 in zip([0, 1, 2], [[0, 0], [0, 1], [1, 1]])
    ]

    # Initialize the two-electron energy contribution
    e2 = [zeros(h1[0].shape[0]), zeros(h1[1].shape[0])]

    # Calculate the two-electron energy contribution for alpha and beta
    def contract_2e(jmaxs, rdm2_, V_, s, sym):
        e2_ = zeros(n_frag[s])
        jmax1, jmax2 = [jmaxs] * 2 if isinstance(jmaxs, int) else jmaxs
        for i in range(n_frag[s]):
            for j in range(jmax1):
                ij = i * (i + 1) // 2 + j if i > j else j * (j + 1) // 2 + i
                if sym in [4, 2]:
                    Gij = rdm2_[i, j, :jmax2, :jmax2].copy()
                    Vij = V_[ij]
                else:
                    Gij = rdm2_[:jmax2, :jmax2, i, j].copy()
                    Vij = V_[:, ij]
                Gij[diag_indices(jmax2)] *= 0.5
                Gij += Gij.T
                e2_[i] += Gij[tril_indices(jmax2)] @ Vij
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
    for i in centerweight_and_relAO_per_center[0][1]:
        e2_tmp += centerweight_and_relAO_per_center[0][0] * e2[i]
        for s in [0, 1]:
            etmp += centerweight_and_relAO_per_center[s][0] * e_[s][i]
            e1_tmp += centerweight_and_relAO_per_center[s][0] * e1[s][i]
            ec_tmp += centerweight_and_relAO_per_center[s][0] * ec[s][i]

    return [e1_tmp, e2_tmp, ec_tmp]


_T = TypeVar("_T", Mole, Cell)


def are_equal(m1: _T, m2: _T) -> bool:
    def compare(m1: _T, m2: _T) -> bool:
        return (
            m1.atom == m2.atom
            and m1.basis == m2.basis
            and m1.charge == m2.charge
            and m1.multiplicity == m2.multiplicity
        )

    if isinstance(m1, Cell) and isinstance(m2, Cell):
        return compare(m1, m2) and (m1.a == m2.a).all()
    elif isinstance(m1, Mole) and isinstance(m2, Mole):
        return compare(m1, m2)
    else:
        raise TypeError("Both objects must be of the same type (Mole or Cell).")
