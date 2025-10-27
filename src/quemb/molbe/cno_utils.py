import time
from typing import Literal, Tuple

import numpy as np
from attrs import define
from chemcoord import Cartesian
from numpy import float64, floating
from ordered_set import OrderedSet
from pyscf import ao2mo, gto, scf
from pyscf.gto import Mole

# from pyscf.mp.dfmp2_native import DFMP2
from quemb.molbe.helper import get_core, get_scfObj, semicanonicalize_orbs
from quemb.shared.typing import AtomIdx, Matrix

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
    cnos :
        Generated occupied or virtual orbitals, aligning with `occ`. This returns all
        orbitals in the Schmidt space. Coupled with the `choose_cnos`, a selection of
        these can be selected and directly concatenated to TA.
    P_mat_eigvals:
        Eigenvalues from the pair density matrix, used for thresholding routine

    Notes
    -----
    This first routine is the naive one. This is expensive. Multiple integral
    transformation steps can be reduced in cost, which is a TODO

    """
    timea = time.time()
    semicanonical = True
    if not semicanonical:
        print("running canonical")
        # TA_x is either TA_occ or TA_vir, aligning with occ=True or False
        eri_schmidt = ao2mo.incore.full(eri_full, TA_x, compact=True)
        timeb = time.time()
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
        timec = time.time()
        # Get CANONICALIZED (note, not semicanonicalized!) C by solving HF with these 1
        # and 2 e integrals
        mf_SC = get_scfObj(
            h_schmidt,
            eri_schmidt,
            nsocc,
            max_cycles=50,
            skip_soscf=False,
        )

        timed = time.time()
        if occ:
            C_SC = mf_SC.mo_coeff[:, :nsocc]
        else:
            C_SC = mf_SC.mo_coeff[:, nsocc:]
        # Using RIMP2 Routine:
        # mf_SC.mol = frag_mol
        # a, b = DFMP2(mf_SC).run().make_natorbs()

        # Get 2 e integrals, transformed by semicanonicalized C
        # Then T amplitudes
        # Then pair densities
        # (all in one function)
        P = FormPairDensity(
            eri_schmidt,
            mf_SC.mo_occ,
            mf_SC.mo_coeff,
            mf_SC.mo_energy,
            occ,
        )
        timee = time.time()
        print("get_cnos: eri schmidt", timeb - timea, flush=True)
        print("get_cnos: preparing hcore", timec - timeb, flush=True)
        print("get_cnos: getting scfObj", timed - timec, flush=True)
        print("get_cnos: form pair density", timee - timed, flush=True)
    else:
        print("running semicanonical")
        # Rotate full-system Fock matrix into augmented Schmdit space
        F_rot = np.einsum(
            "mp,nq,mn->pq",
            TA_x,
            TA_x,
            hcore_full + veff_full,
            optimize=True,
        )

        timeb = time.time()
        # Semicanonicalize orbitals, after building the augmented fragment Fock matrix
        mo_coeff_sc, mo_energy_sc, mo_occ_sc = semicanonicalize_orbs(
            F_rot,
            nsocc,
        )
        timec = time.time()
        if occ:
            C_SC = mo_coeff_sc[:, :nsocc]
        else:
            C_SC = mo_coeff_sc[:, nsocc:]
        # Make semicanonical pair density (i.e., no integral rotation)
        P = FormPairDensity_SC(
            eri_full,
            TA_x,
            mo_occ_sc,
            mo_coeff_sc,
            mo_energy_sc,
            occ,
        )
        timed = time.time()
        print("get_cnos: initial h and v rot", timeb - timea, flush=True)
        print("get_cnos: semicanonicalization", timec - timeb, flush=True)
        print("get_cnos: form pair density", timed - timec, flush=True)

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
    timef = time.time()

    print("Total get_cnos routine", timef - timea, flush=True)

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
        # No environment component is necessary
        h_rot = np.einsum("mp,nq,mn->pq", TA_x, TA_x, h, optimize=True)
        G_envs = np.zeros_like(h_rot)

    else:
        # Build environment component
        ST = S @ TA_x
        G_s = TA_x.T @ hf_veff @ TA_x

        P_act = hf_Cocc @ hf_Cocc.T
        P_fbs = ST.T @ P_act @ ST

        vj, vk = scf.hf.dot_eri_dm(eri_s, P_fbs, hermi=1)
        G_fbs = 2.0 * vj - vk

        h_rot = TA_x.T @ h @ TA_x
        G_envs = G_s - G_fbs

    # Build core component, if frozen core
    G_core = np.zeros_like(h_rot)
    if core_veff is not None:
        G_core = TA_x.T @ core_veff @ TA_x

    hs = h_rot + G_envs + G_core

    return hs


def build_frag_mol(
    atom_per_frag: OrderedSet[AtomIdx],
    mole: Mole,
) -> Tuple[Mole, Mole, Mole, Mole]:
    """Build fragment Mole object, returning the object and the number of electrons in
    the neutral fragment

    Parameters
    ----------
    atom_per_frag :
        OrderedSet showing the atom indices in the given fragment. Used to generate the
        fragment geometry to build the fragment Mole object.
    mole :
        Mole object for the full system. We use this to get the basis set and build the
        fragment Mole object

    Returns
    -------
    mol :
        Fragment molecule object, built
    ghost_mol :
        Mole with fragment ghost atoms, built
    full_mol :
        Mole with all atoms, built
    no_f_mol :
        Mole with only non-fragment atoms, built
    """

    # Process structure information
    all_atoms = Cartesian.from_pyscf(mole.build())
    frag_atoms_full = all_atoms.loc[atom_per_frag, :].to_xyz()

    # Build full molecule
    full_mol = gto.M()
    full_mol.atom = "\n".join(all_atoms.to_xyz().split("\n")[2:])
    full_mol.basis = mole.basis
    full_mol.charge = mole.charge
    full_mol.build()

    # Non-fragment atoms in mole
    ind_not_in_frag = [i not in atom_per_frag for i in range(len(all_atoms))]
    non_frag_atoms_full = all_atoms.loc[ind_not_in_frag, :].to_xyz()

    # Replace fragment atoms with ghost atoms
    replace_ghost = Cartesian.from_pyscf(mole.build())
    for i in atom_per_frag:
        replace_ghost.loc[i, "atom"] = "X-" + str(replace_ghost.loc[i, "atom"])

    # Remove the first two lines so that pyscf doesn't hate the colon
    frag_atoms = "\n".join(frag_atoms_full.split("\n")[2:])
    non_frag_atoms = "\n".join(non_frag_atoms_full.split("\n")[2:])
    ghost_xyz = "\n".join(replace_ghost.to_xyz().split("\n")[2:])

    # Build mini fragment to figure out the number of electrons and orbitals
    frag_mol = gto.M()
    frag_mol.atom = frag_atoms
    frag_mol.basis = mole.basis

    try:
        frag_mol.charge = 0
        frag_mol.build()
    except RuntimeError:
        frag_mol.charge = -1
        frag_mol.build()

    no_f_mol = gto.M()
    no_f_mol.atom = non_frag_atoms
    no_f_mol.basis = mole.basis

    try:
        no_f_mol.charge = 0
        no_f_mol.build()
    except RuntimeError:
        no_f_mol.charge = -1
        no_f_mol.build()

    ghost_mol = gto.M()
    ghost_mol.atom = ghost_xyz
    ghost_mol.basis = mole.basis

    try:
        ghost_mol.charge = 0
        ghost_mol.build()
    except RuntimeError:
        ghost_mol.charge = -1
        ghost_mol.build()

    return frag_mol, ghost_mol, full_mol, no_f_mol


def choose_cnos(
    mol: Mole,
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
    mol :
        Mole object for the fragment
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

    # Establish fragment charge
    nelec = mol.nelectron + mol.charge

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
        print("Intended # OCNOs and VCNOs:", nocc_cno_add, nvir_cno_add)
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
    Forming the Pair Density with either the occupied or virtual space to generate
    OCNOs (VCNOs)

    Parameters
    ----------
    Vs :
        ERIs for the given fragment, rotated into the Schmidt space in `get_cnos`
    mo_occs :
        MO occupancy matrix for the canonicalized fragment
    mo_coeffs :
        MO coefficients for the canonicalized fragment
    mo_energys :
        MO energies for the canonicalized fragment
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

    time1 = time.time()
    # Transform 2 e integrals into the augmented Schmidt space
    V = ao2mo.kernel(Vs, [COcc, CVir, COcc, CVir], compact=False)
    V = V.reshape((nOcc, nVir, nOcc, nVir))

    time2 = time.time()
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
    time3 = time.time()
    if occ:
        # Occupied pair density matrix
        P = 2 * np.einsum("kiab,kjab->ij", T, delta_T_term, optimize=True)
        P = np.eye(T.shape[0]) - P
    else:
        # Virtual pair density matrix
        P = 2.0 * np.einsum("ijac,jicb->ab", T, delta_T_term, optimize=True)
    time4 = time.time()
    print("Pair density: integral transform", time2 - time1, flush=True)
    print("Pair density: forming T amplitudes", time3 - time2, flush=True)
    print("Pair density: form pair density", time4 - time3, flush=True)
    return P


def FormPairDensity_SC(
    V_full: Matrix[floating],
    TA_x: Matrix[floating],
    mo_occs: Matrix[floating],
    mo_coeffs: Matrix[floating],
    mo_energys: Matrix[floating],
    occ: bool,
) -> Matrix:
    """
    Adapted slightly from Frankenstein, as written by Henry Tran
    Forming the Pair Density with either the occupied or virtual space to generate
    OCNOs (VCNOs). This time, working off the semicanonical orbitals, not canonical
    orbitals.

    Parameters
    ----------
    V :
        ERIs for the full system
    TA_x :
        Transformation matrix from the full system to the augmented fragment
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

    # Orbitals to transform from full system to fragment
    COcc = TA_x @ mo_coeffs[:, OccIdx]
    CVir = TA_x @ mo_coeffs[:, VirIdx]

    time1 = time.time()
    # Transform 2 e integrals into the augmented Schmidt space
    V = ao2mo.general(V_full, [COcc, CVir, COcc, CVir], compact=False)
    V = V.reshape((nOcc, nVir, nOcc, nVir))

    time2 = time.time()
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
    time3 = time.time()
    if occ:
        # Occupied pair density matrix
        P = 2 * np.einsum("kiab,kjab->ij", T, delta_T_term, optimize=True)
        P = np.eye(T.shape[0]) - P
    else:
        # Virtual pair density matrix
        P = 2.0 * np.einsum("ijac,jicb->ab", T, delta_T_term, optimize=True)
    time4 = time.time()
    print("Pair density: integral transform", time2 - time1, flush=True)
    print("Pair density: forming T amplitudes", time3 - time2, flush=True)
    print("Pair density: form pair density", time4 - time3, flush=True)
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
