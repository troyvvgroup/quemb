from typing import Literal, Tuple

import numpy as np
from attrs import define
from numpy import float64, floating
from pyscf import ao2mo, gto

from quemb.molbe.helper import get_scfObj
from quemb.shared.typing import Matrix

CNO_Schemes = Literal["Proportional", "ProportionalQQ", "ExactFragmentSize"]
CNO_FragSize_Schemes = Literal["AddVirtuals", "AddBoth"]


@define(frozen=True, kw_only=True)
class CNOArgs:
    """Additional arguments for CNOs.
    cno_scheme options (of type CNO_Schemes), for now, includes "Proportional",
    "ProportionalQQ", and "ExactFragSize"

    1. Proportional: Adding virtual CNOs ONLY until we reach our condition. We are
        making the fragments satisfy this condition:
        (All orbitals in fragment)/(Number of occupied Schmidt space orbitals)
        = (Number of fragment orbitals)/(Number of expected occupied orbitals,
        determined by atoms in fragment)
        In other words, we are replacing virtual orbitals so that the number of
        orbitals in the fragment corresponds to the number of electrons in the
        Schmidt space.
    2. ProportionalQQ: Similar to the Proportional scheme, but now we are also adding
        occupied orbitals. We first add occupied orbitals until (), then augment with
        virtual orbitals until we reach the proportion in (1).
    3. ExactFragmentSize: Enforcing that we add CNOs until each fragment is exactly the
        size given by `tot_active_orbs`. 
        Coupled with `cno_active_fragsize_scheme` with type `CNO_FragSize_Schemes`, we
        will add virtual CNOs until we reach the proportion given in (1). If we request
        more orbitals to be added, we rely on CNO_FragSize_Schemes to either add only
        virtuals (`AddVirtuals`) or add both occupied and virtual CNOs to maintain (as
        close as we can) that ratio (`AddBoth`).

    If you choose `ExactFragmentSize`, you also must specify `tot_active_orbs`, which
    gives the total number of orbitals for each fragment. You can also add specify how
    these orbitals are chosen with the `cno_active_fragsize_scheme`. Default is
    `AddVirtuals`.
    """
    cno_scheme: CNO_Schemes = "Proportional"
    tot_active_orbs: int | None = None
    cno_active_fragsize_scheme: CNO_FragSize_Schemes | None = "AddVirtuals"

def get_cnos(
    nfb: int,
    TA_x: Matrix[float64],
    hcore_full: Matrix[floating],
    eri_full: Matrix[floating],
    nsocc: int,
    nocc: int,
    occ: bool,
    )-> Matrix:
    """ Generates the occupied or virtual CNOs for a given fragment.

    Parameters
    ----------
    nfb : int
        Number of fragment and bath orbitals, directly from the original
        Schmidt decomposition
    TA_x : Matrix
        Augmented TA matrix with either the occupied or virtual environment orbitals
    hcore_full : Matrix
        hcore for the full system
    eri_full : Matrix
        ERIs for the full system
    nocc : int
        Number of occupied orbitals for the full system
    occ : bool
        Whether the CNOs being generated are occupied or virtual

    Returns
    -------
    cnos : Matrix
        Generated occupied or virtual orbitals, aligning with `occ`. This returns all
        orbitals in the Schmidt space. Coupled with the `choose_cnos`, a selection of
        these can be selected and directly concatenated to TA.

    Notes
    -----
    This first routine is the naive one. This is expensive. Multiple integral
    transformation steps can be reduced in cost, which is a TODO

    """
    # TA_x is either TA_occ or TA_vir, aligning with occ=True or False

    # Generate 1 and 2 electron orbitals in modified Schmidt space
    h_schmidt = np.einsum('mp,nq,mn->pq', TA_x, TA_x, hcore_full)
    eri_schmidt = ao2mo.incore.full(eri_full, TA_x, compact=True)

    # Get semicanonicalized C by solving HF with these 1 and 2 e integrals
    mf_SC = get_scfObj(h_schmidt, eri_schmidt, nocc)
    if occ:
        C_SC = mf_SC.mo_coeff[:, :nocc]
    else:
        C_SC = mf_SC.mo_coeff[:, nocc:]
 
    # Get 2 e integrals, transformed by semicanonicalized C
    # Then T amplitudes
    # Then pair densities
    # (all in one function)
    
    P = FormPairDensity(
        eri_schmidt,
        mf_SC.mo_occ,
        mf_SC.mo_coeff,
        mf_SC.mo_energy,
        occ
    )

    # Transform pair density in SO basis
    P_mat_SO = C_SC @ P @ C_SC.T

    # Project out FOs and BOs
    # You can do this all padded with zeros (as described in paper),
    # but reduced to non-zero blocks for cost
    P_mat_SO_env = P_mat_SO[nfb:, nfb:]

    # Find the pair natural orbitals by diagonalizing these orbitals
    P_mat_eigvals, P_mat_eigvecs = np.linalg.eig(P_mat_SO_env)

    # Pad pair natural orbitals
    PNO = np.zeros((TA_x.shape[1], TA_x.shape[1]-nfb))
    PNO[nfb:,:] = P_mat_eigvecs

    # Generate cluster natural orbitals, rotating into AO basis
    cnos = TA_x @ PNO

    return cnos

def choose_cnos(
    file: str,
    basis: str, 
    n_f: int,
    n_b: int,
    n_full_occ: int,
    n_full_vir: int,
    nsocc: int,
    args: CNOArgs | None,
)->Tuple[int,int]:
    """Chooses the number of Occupied and Virtual CNOs for a given fragment

    Parameters
    ----------
    file : str
        File path for the fragment geometry
    basis : str
        Basis set for the calculation
    n_f : int
        Number of fragment orbitals, from the original Schmidt decomposition
    n_b : int
        Number of bath orbitals, from the original Schmidt decomposition
    n_full_occ : int
        Total number of occupied environment orbitals for the system
    n_full_virt : int
        Total number of virtual environment orbitals for the system
    nsocc : int
        Number of occupied orbitals in the Schmidt space
    args : CNOArgs
        Options for CNO schemes, with keyword `cno_scheme`: `Proportional`,
        `ProportionalQQ`, and `ExactFragmentSize`.
        If using `ExactFragmentSize`, also include keyword `tot_active_orbs` to specify
        the desired size of the fragment. You also can include `cno_tot_frag_scheme`
        to choose how these are added (using `AddVirtuals` or `AddBoth`).
        Please look at `CNOArgs` to see further description of these options.

    Returns
    -------
    nocc_cno_add, nvir_cno_add : tuple(int, int)
        The desired number of occupied and virtual CNOs to augment TA, based
        on the chosen `cno_scheme`.
    
    """
    # Options for CNO schemes:
    ###
    assert(args is not None)
    assert((args.cno_scheme=="ExactFragmentSize")==(args.tot_active_orbs is not None))
    # Build mini fragment to figure out the number of electrons and orbitals

    mol = gto.M()
    mol.atom = file
    mol.basis = basis
    nelec = mol.nelectron

    if args.cno_scheme == "Proportional":
        # Ratio of the number of fragment orbitals to the number of expected 
        # occupied orbitals, based on the atoms in the fragment
        prop = n_f / (nelec / 2)
        nocc_cno_add = 0
        # Add virtual orbitals so that the proportion of all fragment orbitals 
        # (n_f + n_b + nvir_cno_add) to the number of occupied orbitals in the
        # Schmidt space (nsocc) is the same as the ratio `prop` above
        nvir_cno_add = np.round(prop * nsocc) - n_f - n_b

    elif args.cno_scheme == "ProportionalQQ":
        # Same ratio as above
        prop = n_f / (nelec / 2)
        # Add enough orbitals for the 
        total_orbs = int(1.5 * n_f) + n_b
        nocc_cno_add = max(int(np.round(total_orbs / prop - nsocc)), 0)
        nvir_cno_add = total_orbs - n_b - nocc_cno_add - n_f

    elif args.cno_scheme == "ExactFragSize":
        # Start by adding virtuals until `Proportional` is hit
        prop = n_f / (nelec / 2)
        max_vir_add_prop = np.round(prop * nsocc) - n_f - n_b

        # Schmidt state is already bigger than the max fragment size
        if args.tot_active_orbs < n_f + n_b:
            raise ValueError("Max fragment size larger than fragment + bath space")
        # We will add virtual CNOs until the ratio of ;the augmented 
        # fragment space to the number of occupieds in the Schmidt spaces reaches 
        # the proportion above: see `Proportional`
        if args.cno_active_fragsize_scheme == "AddVirtuals":
            nocc_cno_add = 0
            nvir_cno_add = args.tot_frag_orbs - n_f - n_b
        elif args.cno_tot_frag_scheme == "AddBoth":
            if 0 <= args.tot_active_orbs - n_f - n_b <= max_vir_add_prop:
                nocc_cno_add = 0
                nvir_cno_add = args.tot_active_orbs - n_f - n_b
            # We need to also add occupieds here. We will now try to satisfy the
            # proportional scheme by adding a certain number of occupieds and virtuals,
            # as closely as possible
            else:
                nocc_cno_add = np.round(args.tot_active_orbs / prop) - nsocc
                nvir_cno_add = args.tot_active_orbs - n_f - n_b - nocc_cno_add
    if nocc_cno_add +  n_f + n_b > n_full_occ:
        raise RuntimeError(
            "Request to add more occupied CNOs than exist. Choose different CNO scheme"
            )
    elif nvir_cno_add + n_f + n_b > n_full_vir:
        raise RuntimeError(
            "Request to add more virtual CNOs than exist. Choose different CNO scheme"
            )

    return int(nocc_cno_add), int(nvir_cno_add)

def FormPairDensity(
    Vs: Matrix[floating],
    mo_occs: Matrix[floating],
    mo_coeffs: Matrix[floating],
    mo_energys: Matrix[floating],
    occ: bool,
    )-> Matrix:
    """
    Adapted slightly from Frankenstein, as written by Henry Tran

    Parameters
    ----------
    Vs : Matrix
        ERIs for the given fragment, rotated into the Schmidt space in `get_cnos`
    mo_occs : Matrix
        MO occupancy matrix for the semi-canonicalized fragment
    mo_coeffs : Matrix
        MO coefficients for the semi-canonicalized fragment
    mo_energys : Matrix
        MO energies for the semi-canonicalized fragment
    occ : bool
        Whether the occupied or virtual pair density matrix is requested

    Returns
    -------
    P : Matrix
        Pair density matrix, to be used to generate CNOs
    """
    # Determine which MOs are occupied and virtual
    OccIdx = np.where(mo_occs > 2.0 - 1e-6)[0]
    VirIdx = np.where(mo_occs < 1e-6)[0]
    
    nOcc = OccIdx.shape[0]
    nVir = VirIdx.shape[0]

    COcc = mo_coeffs[:, OccIdx]
    CVir = mo_coeffs[:, VirIdx]
    print("mo_energys", mo_energys)
    print("mo_coeffs", mo_coeffs)
    print("Vs", Vs)
    # Transform 2 e integrals from the augmented Schmidt space
    V = ao2mo.kernel(Vs, [COcc, CVir, COcc, CVir], compact = False)
    V = V.reshape((nOcc, nVir, nOcc, nVir))

    # Generate the T and delta T term from the CNO paper
    mo_energy_occ = mo_energys[:nOcc]
    mo_energy_vir = mo_energys[nOcc:]
    eMat = np.add.outer(mo_energy_occ, mo_energy_occ)
    eMat = np.subtract.outer(eMat, mo_energy_vir)
    eMat = np.subtract.outer(eMat, mo_energy_vir)
    eMat = np.swapaxes(eMat, 1, 2)
    eMat = eMat**-1
    T = -1 * V * eMat
    T = np.swapaxes(T, 1, 2)

    delta_T_term = 2 * T - np.swapaxes(T, 2, 3)
    print("T", T)
    print("delta_T_term", delta_T_term)
    if occ:
        # Occupied pair density matrix
        P = 2 * np.einsum('kiab,kjab->ij', T, delta_T_term, optimize=True)
        P = np.eye(T.shape[0]) - P
    else:
        # Virtual pair density matrix
        P = 2.0 * np.einsum('ijac,jicb->ab', T, delta_T_term, optimize=True)
    print("P", P)
    return P

def augment_w_cnos(
    TA: Matrix[float64],
    nocc_cno: int,
    nvir_cno: int,
    occ_cno: Matrix[floating] | None,
    vir_cno: Matrix[floating] | None,
    )-> Matrix:
    """Augmenting TA with the chosen occupied and virtual CNOs

    Parameters
    ----------
    TA : Matrix
        Original Schmidt space TA matrix
    nocc_cno : int
        Number of occupied CNOs to augment
    nvir_cno : int
        Number of virtual CNOs to augment
    occ_cno : Matrix
        Full occupied CNO matrix
    vir_cno : Matrix
        Full virtual CNO matrix

    Returns
    -------
    TA : Matrix
        Augmented TA matrix with CNOs
    """
    if nocc_cno > 0:
        assert(occ_cno is not None)
        TA = np.hstack((TA, occ_cno[:, :nocc_cno]))
    if nvir_cno > 0:
        assert(vir_cno is not None)
        TA = np.hstack((TA, vir_cno[:, -nvir_cno:]))

    return TA