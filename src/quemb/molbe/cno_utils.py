from typing import Literal, Tuple

import numpy as np
from attrs import define
from numpy import float64, floating
from pyscf import ao2mo, gto, scf

from quemb.molbe.helper import get_core, get_scfObj
from quemb.shared.typing import Matrix

CNO_Schemes = Literal[
    "Proportional",
    "ProportionalQQ",
    "HalfFilled",
    "Threshold",
    "ExactFragmentSize",
]

CNO_FragSize_Schemes = Literal["AddOccupieds", "AddVirtuals", "AddBoth"]


@define(frozen=True, kw_only=True)
class CNOArgs:
    """Additional arguments for CNOs.
    cno_scheme options (of type CNO_Schemes), for now, includes "Proportional",
    "ProportionalQQ", "HalfFilled", "Threshold", and "ExactFragSize"

    1. Proportional: Adding virtual CNOs ONLY until we reach our condition. We are
        making the fragments satisfy this condition:
        (All orbitals in fragment)/(Number of occupied Schmidt space orbitals)
        = (Number of fragment orbitals)/(Number of expected occupied orbitals,
        determined by atoms in fragment)
        In other words, we are replacing virtual orbitals so that the number of
        orbitals in the fragment corresponds to the number of electrons in the
        Schmidt space.
    2. ProportionalQQ: Similar to the Proportional scheme, but now we are also adding
        occupied orbitals. We add a total of 1/2 * N_f orbitals. We add both OCNOs and
        VCNOs until we reach virtual orbitals until we reach the proportion in (1).
    3. HalfFilled: We add orbitals until each fragment is half-occupied. Note that this
        is in practice often a lot of occupied orbitals.
    4. Threshold: We add orbitals based on a given threshold. Coupled with `cno_thresh`.
    5. ExactFragmentSize: Enforcing that we add CNOs until each fragment is exactly the
        size given by `tot_active_orbs`.
        Coupled with `cno_active_fragsize_scheme` with type `CNO_FragSize_Schemes`, we
        will add virtual CNOs until we reach the proportion given in (1). If we request
        more orbitals to be added, we rely on CNO_FragSize_Schemes to either add only
        virtuals (`AddVirtuals`) or add both occupied and virtual CNOs to maintain (as
        close as we can) that ratio (`AddBoth`).

    If you choose `ExactFragmentSize`, you also must specify `tot_active_orbs`, which
    gives the total number of orbitals for each fragment. You can also add specify how
    these orbitals are chosen with the `cno_active_fragsize_scheme`. Default (and
    recommended option) is `AddBoth`.

    If you choose `Threshold`, you also must specify `cno_thresh`, which is the
    the threshold from which CNOs are chosen. OCNOs with values below 1-`cno_thresh` and
    VCNOs with values above `cno_thresh` are added. Default value is 1e-3.
    """

    cno_scheme: CNO_Schemes = "Proportional"
    tot_active_orbs: int | None = None
    cno_active_fragsize_scheme: CNO_FragSize_Schemes | None = "AddBoth"
    cno_thresh: float | None = 1e-3


def get_cnos(
    nfb: int,
    TA_x: Matrix[float64],
    hcore_full: Matrix[floating],
    eri_full: Matrix[floating],
    veff_full: Matrix[floating],
    C: Matrix[floating],
    S: Matrix[floating],
    nsocc: int,
    nocc: int,
    core_veff: Matrix[floating] | None,
    occ: bool,
) -> Tuple[Matrix, Matrix]:
    """Generates the occupied or virtual CNOs for a given fragment.

    Parameters
    ----------
    nfb :
        Number of fragment and bath orbitals, directly from the original
        Schmidt decomposition
    TA_x :
        Augmented TA matrix with either the occupied or virtual environment orbitals
    hcore_full :
        hcore for the full system
    eri_full :
        ERIs for the full system
    nocc :
        Number of occupied orbitals for the full system
    occ :
        Whether the CNOs being generated are occupied or virtual

    Returns
    -------
    cnos : numpy.ndarray
        Generated occupied or virtual orbitals, aligning with `occ`. This returns all
        orbitals in the Schmidt space. Coupled with the `choose_cnos`, a selection of
        these can be selected and directly concatenated to TA.

    Notes
    -----
    This first routine is the naive one. This is expensive. Multiple integral
    transformation steps can be reduced in cost, which is a TODO

    """
    # TA_x is either TA_occ or TA_vir, aligning with occ=True or False
    eri_schmidt = ao2mo.incore.full(eri_full, TA_x, compact=True)

    h_schmidt = preparing_h_cnos(
        nocc - nsocc,
        C[:, :nocc],
        S,
        hcore_full,
        TA_x,
        veff_full,
        core_veff,
        eri_schmidt,
    )

    # Get semicanonicalized C by solving HF with these 1 and 2 e integrals
    mf_SC = get_scfObj(h_schmidt, eri_schmidt, nsocc)

    if occ:
        C_SC = mf_SC.mo_coeff[:, :nsocc]
    else:
        C_SC = mf_SC.mo_coeff[:, nsocc:]

    # Get 2 e integrals, transformed by semicanonicalized C
    # Then T amplitudes
    # Then pair densities
    # (all in one function)
    P = FormPairDensity(eri_schmidt, mf_SC.mo_occ, mf_SC.mo_coeff, mf_SC.mo_energy, occ)

    # Transform pair density in SO basis
    P_mat_SO = C_SC @ P @ C_SC.T

    # Project out FOs and BOs
    # You can do this all padded with zeros (as described in paper),
    # but reduced to non-zero blocks for cost
    P_mat_SO_env = P_mat_SO[nfb:, nfb:]

    # Find the pair natural orbitals by diagonalizing these orbitals
    P_mat_eigvals, P_mat_eigvecs = np.linalg.eigh(P_mat_SO_env)

    # Pad pair natural orbitals
    PNO = np.zeros((TA_x.shape[1], TA_x.shape[1] - nfb))
    PNO[nfb:, :] = P_mat_eigvecs

    # Generate cluster natural orbitals, rotating into AO basis
    cnos = TA_x @ PNO

    return cnos, P_mat_eigvals


def preparing_h_cnos(
    nvir: int,
    hf_Cocc: Matrix,
    S: Matrix[floating],
    h: Matrix[floating],
    TA_x: Matrix[float64],
    hf_veff: Matrix[floating],
    core_veff: Matrix[floating] | None,
    eri_s: Matrix[floating],
) -> Matrix[float64]:
    """Building the correct 1-electron integrals to form the CNOs"""
    if nvir == 0:
        h_rot = np.einsum("mp,nq,mn->pq", TA_x, TA_x, h, optimize=True)
        G_envs = np.zeros_like(h_rot)

    else:
        ST = S @ TA_x
        G_s = TA_x.T @ hf_veff @ TA_x

        P_act = hf_Cocc @ hf_Cocc.T
        P_fbs = ST.T @ P_act @ ST

        vj, vk = scf.hf.dot_eri_dm(eri_s, P_fbs, hermi=1)
        G_fbs = 2.0 * vj - vk

        h_rot = TA_x.T @ h @ TA_x
        G_envs = G_s - G_fbs

    G_core = np.zeros_like(h_rot)
    if core_veff is not None:
        G_core = TA_x.T @ core_veff @ TA_x

    hs = h_rot + G_envs + G_core

    return hs


def choose_cnos(
    file: str,
    basis: str,
    n_f: int,
    n_b: int,
    n_full_occ: int,
    n_full_vir: int,
    nsocc: int,
    nocc: int,
    frz_core: bool,
    args: CNOArgs | None,
) -> Tuple[int, int, float | None]:
    """Chooses the number of Occupied and Virtual CNOs for a given fragment

    Parameters
    ----------
    file :
        File path for the fragment geometry
    basis :
        Basis set for the calculation
    n_f :
        Number of fragment orbitals, from the original Schmidt decomposition
    n_b :
        Number of bath orbitals, from the original Schmidt decomposition
    n_full_occ :
        Total number of occupied environment orbitals for the system
    n_full_virt :
        Total number of virtual environment orbitals for the system
    nsocc :
        Number of occupied orbitals in the Schmidt space
    nocc :
        Total number of occupied orbitals
    frz_core:
        Whether the core is frozen for the system
    args :
        Options for CNO schemes, with keyword `cno_scheme`: `Proportional`,
        `ProportionalQQ`, and `ExactFragmentSize`.
        If using `ExactFragmentSize`, also include keyword `tot_active_orbs` to specify
        the desired size of the fragment. You also can include `cno_tot_frag_scheme`
        to choose how these are added (using `AddVirtuals` or `AddBoth`).
        Please look at `CNOArgs` to see further description of these options.

    Returns
    -------
    nocc_cno_add:
        The number of OCNOs to add to augment TA, based on `cno_scheme`
    nvir_cno_add :
        The number of VCNOs to add to augment TA, based on `cno_scheme`
    cno_thresh :
        Threshold to select CNOs, if `cno_scheme`=`Thresholding`, else None
    """
    # Options for CNO schemes:
    ###
    assert args is not None

    # Build mini fragment to figure out the number of electrons and orbitals
    mol = gto.M()
    mol.atom = file
    mol.basis = basis
    # assigning nelec here rather than after build with charge
    nelec = mol.nelectron
    try:
        mol.charge = 0
        mol.build()
    except RuntimeError:
        mol.charge = -1
        mol.build()

    if frz_core:
        # Find number of core orbitals in the fragment
        frz_core_orbs = get_core(mol)[0]
        # Update the "expected" number of electrons for the fragment
        nelec -= 2 * frz_core_orbs

    cno_thresh = None
    if args.cno_scheme == "Proportional":
        # Ratio of the number of fragment orbitals to the number of expected
        # occupied orbitals, based on the atoms in the fragment
        prop = n_f / (nelec / 2)
        nocc_cno_add = 0

        # Add virtual orbitals so that the proportion of all fragment orbitals
        # (n_f + n_b + nvir_cno_add) to the number of occupied orbitals in the
        # Schmidt space (nsocc) is the same as the ratio `prop` above

        # Kinda ridiculous, but okay for now...
        if isinstance(prop, int):
            nvir_cno_add = prop - n_f - n_b
        else:
            nvir_cno_add = int(np.round(prop * nsocc) - n_f - n_b)

    elif args.cno_scheme == "ProportionalQQ":
        # Same ratio as above
        prop = n_f / (nelec / 2)

        # Add enough orbitals to fit the proportion, where the total number of orbitals
        # is some 1.5 * the size of the fragment and the bath
        total_orbs = int(1.5 * n_f) + n_b
        nocc_cno_add = int(max(int(np.round(total_orbs / prop - nsocc)), 0))
        nvir_cno_add = total_orbs - n_b - nocc_cno_add - n_f

    elif args.cno_scheme == "HalfFilled":
        # We add enough occupied and virtual orbitals so that fragment is half filled
        # Note that in practice, this tends to be a lot of occupied orbitals
        nocc_cno_add = int(np.round(min(n_f, nocc) - nsocc))
        nvir_cno_add = min(n_f, nocc) - n_b - nocc_cno_add

    elif args.cno_scheme == "Threshold":
        nocc_cno_add = nvir_cno_add = 0
        cno_thresh = args.cno_thresh

    elif args.cno_scheme == "ExactFragmentSize":
        assert args.tot_active_orbs is not None

        # Start by adding virtuals until `Proportional` is hit
        prop = n_f / (nelec / 2)
        max_vir_add_prop = np.round(prop * nsocc) - n_f - n_b

        # Schmidt state is already bigger than the max fragment size
        if args.tot_active_orbs < n_f + n_b:
            raise ValueError("Max fragment size larger than fragment + bath space")
        # We will add virtual CNOs until the ratio of ;the augmented
        # fragment space to the number of occupieds in the Schmidt spaces reaches
        # the proportion above: see `Proportional`
        if args.cno_active_fragsize_scheme == "AddOccupieds":
            nocc_cno_add = args.tot_active_orbs - n_f - n_b
            nvir_cno_add = 0
        if args.cno_active_fragsize_scheme == "AddVirtuals":
            nocc_cno_add = 0
            nvir_cno_add = args.tot_active_orbs - n_f - n_b
        elif args.cno_active_fragsize_scheme == "AddBoth":
            if 0 <= args.tot_active_orbs - n_f - n_b <= max_vir_add_prop:
                nocc_cno_add = 0
                nvir_cno_add = args.tot_active_orbs - n_f - n_b
            # We need to also add occupieds here. We will now try to satisfy the
            # proportional scheme by adding a certain number of occupieds and virtuals,
            # as closely as possible
            else:
                nocc_cno_add = np.round(args.tot_active_orbs / prop) - nsocc
                nvir_cno_add = args.tot_active_orbs - n_f - n_b - nocc_cno_add
    if args.cno_scheme != "Threshold":
        if nocc_cno_add + n_f + n_b > n_full_occ:
            raise RuntimeError(
                "Request to add more OCNOs than exist. Choose different CNO scheme"
            )
        if nvir_cno_add + n_f + n_b > n_full_vir:
            raise RuntimeError(
                "Request to add more VCNOs than exist. Choose different CNO scheme"
            )

    return nocc_cno_add, nvir_cno_add, cno_thresh


def FormPairDensity(
    Vs: Matrix[floating],
    mo_occs: Matrix[floating],
    mo_coeffs: Matrix[floating],
    mo_energys: Matrix[floating],
    occ: bool,
) -> Matrix:
    """
    Adapted slightly from Frankenstein, as written by Henry Tran

    Parameters
    ----------
    Vs :
        ERIs for the given fragment, rotated into the Schmidt space in `get_cnos`
    mo_occs :
        MO occupancy matrix for the semi-canonicalized fragment
    mo_coeffs :
        MO coefficients for the semi-canonicalized fragment
    mo_energys :
        MO energies for the semi-canonicalized fragment
    occ :
        Whether the occupied or virtual pair density matrix is requested

    Returns
    -------
    P : numpy.ndarray
        Pair density matrix, to be used to generate CNOs
    """
    # Determine which MOs are occupied and virtual
    OccIdx = np.where(mo_occs > 2.0 - 1e-6)[0]
    VirIdx = np.where(mo_occs < 1e-6)[0]

    nOcc = OccIdx.shape[0]
    nVir = VirIdx.shape[0]

    COcc = mo_coeffs[:, OccIdx]
    CVir = mo_coeffs[:, VirIdx]

    # Transform 2 e integrals from the augmented Schmidt space
    V = ao2mo.kernel(Vs, [COcc, CVir, COcc, CVir], compact=False)
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

    if occ:
        # Occupied pair density matrix
        P = 2 * np.einsum("kiab,kjab->ij", T, delta_T_term, optimize=True)
        P = np.eye(T.shape[0]) - P
    else:
        # Virtual pair density matrix
        P = 2.0 * np.einsum("ijac,jicb->ab", T, delta_T_term, optimize=True)

    return P


def augment_w_cnos(
    TA: Matrix[float64],
    nocc_cno: int,
    nvir_cno: int,
    occ_cno_eigvals: Matrix[floating] | None,
    vir_cno_eigvals: Matrix[floating] | None,
    occ_cno: Matrix[floating] | None,
    vir_cno: Matrix[floating] | None,
    thresh: float | None = None,
) -> Tuple[Matrix, int, int]:
    """Augmenting TA with the chosen occupied and virtual CNOs

    Parameters
    ----------
    TA :
        Original Schmidt space TA matrix
    nocc_cno : int
        Number of occupied CNOs to augment. If `thresh` is not None,
        we use the threshold to determine the CNOs instead
    nvir_cno : int
        Number of virtual CNOs to augment. If `thresh` is not None,
        we use the threshold to determine the CNOs instead
    occ_cno_eigvals :
        The eigenvalues from the occupied pair density, used for thesholding
        OCNOs if thresh is not None
    vir_cno_eigvals :
        The eigenvalues from the virtual pair density, used for thesholding
        VCNOs if thresh is not None
    occ_cno :
        Full occupied CNO matrix
    vir_cno :
        Full virtual CNO matrix
    thresh :
        If not None, the threshold we use to choose the CNOs

    Returns
    -------
    TA :
        Augmented TA matrix with CNOs
    nocc_cno:
        The number of OCNOs added to the fragment
    nvir_cno:
        The number of VCNOs added to the fragment
    """
    if thresh is None:
        if nocc_cno > 0:
            assert occ_cno is not None
            TA = np.hstack((TA, occ_cno[:, :nocc_cno]))
        if nvir_cno > 0:
            assert vir_cno is not None
            TA = np.hstack((TA, vir_cno[:, -nvir_cno:]))
    else:
        assert occ_cno_eigvals is not None
        assert vir_cno_eigvals is not None

        occ_ind = np.argwhere(occ_cno_eigvals < (1 - thresh))
        vir_ind = np.argwhere(vir_cno_eigvals > (thresh))

        nocc_cno = occ_ind.shape[0]
        nvir_cno = vir_ind.shape[0]
        if nocc_cno > 0:
            assert occ_cno is not None
            aug_occ = np.asarray([occ_cno[:, i] for [i] in occ_ind])
            TA = np.hstack((TA, aug_occ.T))
        if nvir_cno > 0:
            assert vir_cno is not None
            aug_vir = np.asarray([vir_cno[:, i] for [i] in vir_ind])
            TA = np.hstack((TA, aug_vir.T))

    return TA, nocc_cno, nvir_cno
