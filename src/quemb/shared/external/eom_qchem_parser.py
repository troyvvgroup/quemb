import numpy as np


def swap_block(spin, n_max):
    """
    This function swaps the alpha and beta blocks of the spin-orbital
    representation of 1- and 2-RDMs, from the Q-Chem convention to
    the PySCF convention.
    alpha alpha ... alpha beta beta ... beta ->
    alpha beta alpha beta ... alpha beta,
    in increasing energy order.

    Parameters
    ----------
    spin : int
        index of 2RDM element, in Q-Chem spin-orbital representation
    n_max : int
        maximum value of index along a specific dimension

    Returns
    -------
    int
        index of 2RDM element, in PySCF spin-orbital representation
    """

    # alpha block in QCHEM 2RDM
    if spin < n_max / 2:
        return int(spin * 2)  # alpha becomes even index
    # beta block in QCHEM 2RDM
    return int(2 * (spin - n_max / 2) + 1)  # beta becomes odd index


def get_idx(Q, R, S, idx):
    """
    This function transforms a 1D array coordinate of index idx,
    to a 4D array coordinate representing the indices of the
    reconstructed 2RDM.

    Parameters
    ----------
    Q : int
        max dimension of 2RDM, along 3rd dimension
    R : int
        max dimension of 2RDM, along 2nd dimension
    S : int
        max dimension of 2RDM, along 1st dimension
    idx : int
        index of 1D array element printed in Q-Chem output file.

    Returns
    -------
    p, q, r, s : int
        indices of 2RDM element
    """

    p = idx // (Q * R * S)
    q = (idx - p * Q * R * S) // (R * S)
    r = (idx - p * Q * R * S - q * R * S) // S
    s = idx - p * Q * R * S - q * R * S - r * S
    return p, q, r, s


def reorder_2rdm(
    imin, jmin, kmin, lmin, imax, jmax, kmax, lmax, rdm2, rdm2_pyscf_order
):
    """
    This function reorders the 2RDM in spin-orbital representation from the
    Q-Chem to PySCF convention.
    Swaps  the spin blocks as follows: [aa..abbb..b] -> [abab..ab]

    Parameters
    ----------
    imin : int
        min dimension of 2RDM, along 1st dimension
    jmin : int
        min dimension of 2RDM, along 2nd dimension
    kmin : int
        min dimension of 2RDM, along 3rd dimension
    lmin : int
        min dimension of 2RDM, along 4th dimension
    imax : int
        max dimension of 2RDM, along 1st dimension
    jmax : int
        max dimension of 2RDM, along 2nd dimension
    kmax : int
        max dimension of 2RDM, along 3rd dimension
    lmax : int
        max dimension of 2RDM, along 4th dimension
    rdm2 : numpy.ndarray
        2RDM, in Q-Chem ordering
    rdm2_pyscf_order : numpy.ndarray
        2RDM, in PySCF ordering
    """

    for i in range(imin, imax):
        for j in range(jmin, jmax):
            for k in range(kmin, kmax):
                for l in range(lmin, lmax):
                    p, q, r, s = (
                        swap_block(i - imin, imax - imin),
                        swap_block(j - jmin, jmax - jmin),
                        swap_block(k - kmin, kmax - kmin),
                        swap_block(l - lmin, lmax - lmin),
                    )
                    rdm2_pyscf_order[p + imin, q + jmin, r + kmin, s + lmin] = rdm2[
                        i, j, k, l
                    ]
    return


def parser_nocc(output):
    """
    This function parses the Q-Chem EOM output file for the number
    of active occupied and active virtual spin-orbitals.

    Parameters
    ----------
    output : string
        Q-Chem output file

    Returns
    -------
    n_occ : int
        number of active occupied spin-orbitals
    n_virt : int
        number of active virtual spin-orbitals
    """

    with open(output, "r") as file:
        lines = file.readlines()
        for l in lines:
            line = l.strip()
            if "Active occupied" in line:
                n_occ = 2 * int(line.split()[3])
            elif "Active virtual" in line:
                n_virt = 2 * int(line.split()[3])
    return n_occ, n_virt


def get_rdms(output, n_ex, n_occ, n_virt):
    """
    This function parses the Q-Chem EOM output file for and constructs
    the 1- and 2-RDMs of the EOM-EE-CCSD excited state n_ex.
    The RDMs are returned in the spin-orbital basis, in Q-Chem representation,
    without including the separable (HF) part.

    Parameters
    ----------
    output : string
        Q-Chem output file
    n_ex : int
        The excited state for which RDMs are computed
    n_occ : int
        number of active occupied spin-orbitals
    n_virt : int
        number of active virtual spin-orbitals

    Returns
    -------
    dm : numpy.ndarray
        1-RDM
    rdm2 : numpy.ndarray
        2-RDM
    """

    dm = np.zeros((n_occ + n_virt, n_occ + n_virt))
    rdm2 = np.zeros((n_occ + n_virt, n_occ + n_virt, n_occ + n_virt, n_occ + n_virt))
    rdm2_oooo = np.zeros((n_occ, n_occ, n_occ, n_occ))
    rdm2_ooov = np.zeros((n_occ, n_occ, n_occ, n_virt))
    rdm2_oovv = np.zeros((n_occ, n_occ, n_virt, n_virt))
    rdm2_ovoo = np.zeros((n_occ, n_virt, n_occ, n_occ))
    rdm2_ovov = np.zeros((n_occ, n_virt, n_occ, n_virt))
    rdm2_ovvv = np.zeros((n_occ, n_virt, n_virt, n_virt))
    rdm2_vvoo = np.zeros((n_virt, n_virt, n_occ, n_occ))
    rdm2_vvov = np.zeros((n_virt, n_virt, n_occ, n_virt))
    rdm2_vvvv = np.zeros((n_virt, n_virt, n_virt, n_virt))
    rdm2_oovo = np.zeros((n_occ, n_occ, n_virt, n_occ))
    rdm2_vooo = np.zeros((n_virt, n_occ, n_occ, n_occ))
    rdm2_voov = np.zeros((n_virt, n_occ, n_occ, n_virt))
    rdm2_vovo = np.zeros((n_virt, n_occ, n_virt, n_occ))
    rdm2_ovvo = np.zeros((n_occ, n_virt, n_virt, n_occ))
    rdm2_vovv = np.zeros((n_virt, n_occ, n_virt, n_virt))
    rdm2_vvvo = np.zeros((n_virt, n_virt, n_virt, n_occ))

    with open(output, "r") as file:
        lines = file.readlines()

    current_block = None
    buffer = []
    row_counter = 0
    n_ex_ok = 0

    for l in lines:
        line = l.strip()

        if "EOMEE-CCSD transition " + str(n_ex) + "/A" in line:
            print("Excited state " + str(n_ex) + " detected in output file.")
            n_ex_ok = 1

        if n_ex_ok == 1:
            # detect the start of a new block
            if "Printing DM_OO" in line:
                current_block = "OO"
                row_counter = 0
                buffer = []
            elif "Printing DM_OV" in line:
                current_block = "OV"
                row_counter = 0
                buffer = []
            elif "Printing DM_VO" in line:
                current_block = "VO"
                row_counter = n_occ
                buffer = []
            elif "Printing DM_VV" in line:
                current_block = "VV"
                row_counter = n_occ
                buffer = []

            elif "Printing 2RDM_OOOO" in line:
                current_block = "OOOO"
                buffer = []
            elif "Printing 2RDM_OOOV" in line:
                current_block = "OOOV"
                buffer = []
            elif "Printing 2RDM_OOVV" in line:
                current_block = "OOVV"
                buffer = []
            elif "Printing 2RDM_OVOO" in line:
                current_block = "OVOO"
                buffer = []
            elif "Printing 2RDM_OVOV" in line:
                current_block = "OVOV"
                buffer = []
            elif "Printing 2RDM_OVVV" in line:
                current_block = "OVVV"
                buffer = []
            elif "Printing 2RDM_VVOO" in line:
                current_block = "VVOO"
                buffer = []
            elif "Printing 2RDM_VVOV" in line:
                current_block = "VVOV"
                buffer = []
            elif "Printing 2RDM_VVVV" in line:
                current_block = "VVVV"
                buffer = []

            elif "EOMEE-CCSD transition " + str(n_ex + 1) + "/A" in line:
                rdm2 = np.block(
                    [
                        [
                            [[rdm2_oooo, rdm2_ooov], [rdm2_oovo, rdm2_oovv]],
                            [[rdm2_ovoo, rdm2_ovov], [rdm2_ovvo, rdm2_ovvv]],
                        ],
                        [
                            [[rdm2_vooo, rdm2_voov], [rdm2_vovo, rdm2_vovv]],
                            [[rdm2_vvoo, rdm2_vvov], [rdm2_vvvo, rdm2_vvvv]],
                        ],
                    ]
                )
                break

            elif current_block:
                # skip non-numeric lines
                if any(char == "e" for char in line) and ("S^2" not in line):
                    values = [float(x) for x in line.split()]
                    buffer.extend(values)

                    # once we accumulate enough values
                    # assign to 1RDM row and delete from buffer
                    if current_block == "OO" and len(buffer) >= n_occ:
                        dm[row_counter, 0:n_occ] = buffer[
                            0:n_occ
                        ]  # fill top-left block
                        buffer = buffer[n_occ:]
                        row_counter += 1
                    elif current_block == "OV" and len(buffer) >= n_virt:
                        dm[row_counter, n_occ:] = buffer[
                            0:n_virt
                        ]  # fill top-right block
                        buffer = buffer[n_virt:]
                        row_counter += 1
                    elif current_block == "VO" and len(buffer) >= n_occ:
                        dm[row_counter, 0:n_occ] = buffer[
                            0:n_occ
                        ]  # fill bottom-left block
                        buffer = buffer[n_occ:]
                        row_counter += 1
                    elif current_block == "VV" and len(buffer) >= n_virt:
                        dm[row_counter, n_occ:] = buffer[
                            0:n_virt
                        ]  # fill bottom-right block
                        buffer = buffer[n_virt:]
                        row_counter += 1
                        if row_counter >= n_occ + n_virt:
                            current_block = None

                    elif current_block == "OOOO" and len(buffer) == n_occ**4:
                        P = n_occ
                        Q = n_occ
                        R = n_occ
                        S = n_occ
                        for idx in range(len(buffer)):
                            p, q, r, s = get_idx(Q, R, S, idx)
                            rdm2_oooo[p, q, r, s] = buffer[idx]

                    elif current_block == "OOOV" and len(buffer) == n_occ**3 * n_virt:
                        P = n_occ
                        Q = n_occ
                        R = n_occ
                        S = n_virt
                        for idx in range(len(buffer)):
                            p, q, r, s = get_idx(Q, R, S, idx)
                            rdm2_ooov[p, q, r, s] = buffer[idx]

                        ### OOOV = -OOVO

                        for p in range(P):
                            for q in range(Q):
                                for r in range(R):
                                    for s in range(S):
                                        rdm2_oovo[p, q, s, r] = -rdm2_ooov[p, q, r, s]

                    elif (
                        current_block == "OOVV" and len(buffer) == n_occ**2 * n_virt**2
                    ):
                        P = n_occ
                        Q = n_occ
                        R = n_virt
                        S = n_virt
                        for idx in range(len(buffer)):
                            p, q, r, s = get_idx(Q, R, S, idx)
                            rdm2_oovv[p, q, r, s] = buffer[idx]

                    elif current_block == "OVOO" and len(buffer) == n_occ**3 * n_virt:
                        P = n_occ
                        Q = n_virt
                        R = n_occ
                        S = n_occ
                        for idx in range(len(buffer)):
                            p, q, r, s = get_idx(Q, R, S, idx)
                            rdm2_ovoo[p, q, r, s] = buffer[idx]

                        ### OVOO = -VOOO

                        for p in range(P):
                            for q in range(Q):
                                for r in range(R):
                                    for s in range(S):
                                        rdm2_vooo[q, p, r, s] = -rdm2_ovoo[p, q, r, s]

                    elif (
                        current_block == "OVOV" and len(buffer) == n_occ**2 * n_virt**2
                    ):
                        P = n_occ
                        Q = n_virt
                        R = n_occ
                        S = n_virt
                        for idx in range(len(buffer)):
                            p, q, r, s = get_idx(Q, R, S, idx)
                            rdm2_ovov[p, q, r, s] = buffer[idx]

                        ### OVOV = -VOOV

                        for p in range(P):
                            for q in range(Q):
                                for r in range(R):
                                    for s in range(S):
                                        rdm2_voov[q, p, r, s] = -rdm2_ovov[p, q, r, s]

                        ### OVOV = VOVO

                        for p in range(P):
                            for q in range(Q):
                                for r in range(R):
                                    for s in range(S):
                                        rdm2_vovo[q, p, s, r] = rdm2_ovov[p, q, r, s]

                        ### OVOV = -OVVO

                        for p in range(P):
                            for q in range(Q):
                                for r in range(R):
                                    for s in range(S):
                                        rdm2_ovvo[p, q, s, r] = -rdm2_ovov[p, q, r, s]

                    elif current_block == "OVVV" and len(buffer) == n_occ * n_virt**3:
                        P = n_occ
                        Q = n_virt
                        R = n_virt
                        S = n_virt
                        for idx in range(len(buffer)):
                            p, q, r, s = get_idx(Q, R, S, idx)
                            rdm2_ovvv[p, q, r, s] = buffer[idx]

                        ### OVVV = -VOVV

                        for p in range(P):
                            for q in range(Q):
                                for r in range(R):
                                    for s in range(S):
                                        rdm2_vovv[q, p, r, s] = -rdm2_ovvv[p, q, r, s]

                    elif (
                        current_block == "VVOO" and len(buffer) == n_occ**2 * n_virt**2
                    ):
                        P = n_virt
                        Q = n_virt
                        R = n_occ
                        S = n_occ
                        for idx in range(len(buffer)):
                            p, q, r, s = get_idx(Q, R, S, idx)
                            rdm2_vvoo[p, q, r, s] = buffer[idx]

                    elif current_block == "VVOV" and len(buffer) == n_occ * n_virt**3:
                        P = n_virt
                        Q = n_virt
                        R = n_occ
                        S = n_virt
                        for idx in range(len(buffer)):
                            p, q, r, s = get_idx(Q, R, S, idx)
                            rdm2_vvov[p, q, r, s] = buffer[idx]

                        ### VVOV = -VVVO

                        for p in range(P):
                            for q in range(Q):
                                for r in range(R):
                                    for s in range(S):
                                        rdm2_vvvo[p, q, s, r] = -rdm2_vvov[p, q, r, s]

                    elif current_block == "VVVV" and len(buffer) == n_virt**4:
                        P = n_virt
                        Q = n_virt
                        R = n_virt
                        S = n_virt
                        for idx in range(len(buffer)):
                            p, q, r, s = get_idx(Q, R, S, idx)
                            rdm2_vvvv[p, q, r, s] = buffer[idx]

    return dm, rdm2


def spin_traced(dm_matrix, n_occ, n_virt):
    """
    This function returns the spatial orbital representation of the EOM
    1-RDM, given the Q-Chem spin-orbital representation of the 1-RDM.

    Parameters
    ----------
    dm_matrix : numpy.ndarray
        1-RDM in spin-orbital Q-Chem representation
    n_occ : int
        number of active occupied spin-orbitals
    n_virt : int
        number of active virtual spin-orbitals

    Returns
    -------
    dm : numpy.ndarray
        1-RDM in spatial orbital representation
    """

    dm_upper = np.hstack(
        (
            dm_matrix[0 : (n_occ // 2), 0 : (n_occ // 2)],
            dm_matrix[0 : (n_occ // 2), n_occ : (n_occ + n_virt // 2)],
        )
    )

    dm_lower = np.hstack(
        (
            dm_matrix[n_occ : n_occ + n_virt // 2, 0 : n_occ // 2],
            dm_matrix[n_occ : n_occ + n_virt // 2, n_occ : n_occ + n_virt // 2],
        )
    )

    dm = np.vstack((dm_upper, dm_lower))
    return dm


def eom_parser(output="eom.out", n_ex=1, frag_number=0):
    """
    This function parses the Q-Chem EOM output file and constructs
    the 1- and 2-RDMs of the EOM-EE-CCSD excited state n_ex.
    The fragment RDMs are saved to .npy files, to be used in the EOM-CCSD
    BE solver.
    RDMs are expressed in the spatial orbital basis, and the 1-RDM also
    includes the separable (HF) part.

    Parameters
    ----------
    output : string
        Q-Chem output file
    n_ex : int
        The excited state for which RDMs are computed
    frag_number : int
        fragment index for which RDMs have been computed
    """

    n_occ, n_virt = parser_nocc(output)

    dm_matrix, rdm2 = get_rdms(output, n_ex, n_occ, n_virt)

    # 1RDM - swap to pyscf orbital ordering
    dm1_pyscf_order = np.zeros((n_occ + n_virt, n_occ + n_virt))

    ###OO block
    for i in range(n_occ):
        for j in range(n_occ):
            # swap spin block; [aa..abbb..b] -> [abab..ab]
            p, q = swap_block(i, n_occ), swap_block(j, n_occ)
            dm1_pyscf_order[p, q] = dm_matrix[i, j]

    ###OV block
    for i in range(n_occ):
        for j in range(n_occ, n_occ + n_virt):
            # swap spin block; [aa..abbb..b] -> [abab..ab]
            p, q = swap_block(i, n_occ), swap_block(j - n_occ, n_virt)
            dm1_pyscf_order[p, q + n_occ] = dm_matrix[i, j]

    ###VO block
    for i in range(n_occ, n_occ + n_virt):
        for j in range(n_occ):
            # swap spin block; [aa..abbb..b] -> [abab..ab]
            p, q = swap_block(i - n_occ, n_virt), swap_block(j, n_occ)
            dm1_pyscf_order[p + n_occ, q] = dm_matrix[i, j]

    ###VV block
    for i in range(n_occ, n_occ + n_virt):
        for j in range(n_occ, n_occ + n_virt):
            # swap spin block; [aa..abbb..b] -> [abab..ab]
            p, q = swap_block(i - n_occ, n_virt), swap_block(j - n_occ, n_virt)
            dm1_pyscf_order[p + n_occ, q + n_occ] = dm_matrix[i, j]

    ### multiply by two (both alpha and beta spins)
    dm_spin_tr = 2 * spin_traced(dm_matrix, n_occ, n_virt)

    ### add HF part (separable) to 1RDM
    for i in range(n_occ // 2):
        dm_spin_tr[i, i] += 2

    ###Reorder 2RDM from Q-Chem to PySCF ordering

    rdm2_pyscf_order = np.zeros(
        ((n_occ + n_virt), (n_occ + n_virt), (n_occ + n_virt), (n_occ + n_virt))
    )

    ###OOOO block
    reorder_2rdm(0, 0, 0, 0, n_occ, n_occ, n_occ, n_occ, rdm2, rdm2_pyscf_order)
    # OOOV
    reorder_2rdm(
        0, 0, 0, n_occ, n_occ, n_occ, n_occ, n_occ + n_virt, rdm2, rdm2_pyscf_order
    )
    # OOVO
    reorder_2rdm(
        0, 0, n_occ, 0, n_occ, n_occ, n_occ + n_virt, n_occ, rdm2, rdm2_pyscf_order
    )
    # OOVV
    reorder_2rdm(
        0,
        0,
        n_occ,
        n_occ,
        n_occ,
        n_occ,
        n_occ + n_virt,
        n_occ + n_virt,
        rdm2,
        rdm2_pyscf_order,
    )

    # OVOO
    reorder_2rdm(
        0, n_occ, 0, 0, n_occ, n_occ + n_virt, n_occ, n_occ, rdm2, rdm2_pyscf_order
    )
    # OVOV
    reorder_2rdm(
        0,
        n_occ,
        0,
        n_occ,
        n_occ,
        n_occ + n_virt,
        n_occ,
        n_occ + n_virt,
        rdm2,
        rdm2_pyscf_order,
    )
    # OVVO
    reorder_2rdm(
        0,
        n_occ,
        n_occ,
        0,
        n_occ,
        n_occ + n_virt,
        n_occ + n_virt,
        n_occ,
        rdm2,
        rdm2_pyscf_order,
    )
    # OVVV
    reorder_2rdm(
        0,
        n_occ,
        n_occ,
        n_occ,
        n_occ,
        n_occ + n_virt,
        n_occ + n_virt,
        n_occ + n_virt,
        rdm2,
        rdm2_pyscf_order,
    )

    # VOOO
    reorder_2rdm(
        n_occ, 0, 0, 0, n_occ + n_virt, n_occ, n_occ, n_occ, rdm2, rdm2_pyscf_order
    )
    # VOOV
    reorder_2rdm(
        n_occ,
        0,
        0,
        n_occ,
        n_occ + n_virt,
        n_occ,
        n_occ,
        n_occ + n_virt,
        rdm2,
        rdm2_pyscf_order,
    )
    # VOVO
    reorder_2rdm(
        n_occ,
        0,
        n_occ,
        0,
        n_occ + n_virt,
        n_occ,
        n_occ + n_virt,
        n_occ,
        rdm2,
        rdm2_pyscf_order,
    )
    # VOVV
    reorder_2rdm(
        n_occ,
        0,
        n_occ,
        n_occ,
        n_occ + n_virt,
        n_occ,
        n_occ + n_virt,
        n_occ + n_virt,
        rdm2,
        rdm2_pyscf_order,
    )

    # VVOO
    reorder_2rdm(
        n_occ,
        n_occ,
        0,
        0,
        n_occ + n_virt,
        n_occ + n_virt,
        n_occ,
        n_occ,
        rdm2,
        rdm2_pyscf_order,
    )
    # VVOV
    reorder_2rdm(
        n_occ,
        n_occ,
        0,
        n_occ,
        n_occ + n_virt,
        n_occ + n_virt,
        n_occ,
        n_occ + n_virt,
        rdm2,
        rdm2_pyscf_order,
    )
    # VVVO
    reorder_2rdm(
        n_occ,
        n_occ,
        n_occ,
        0,
        n_occ + n_virt,
        n_occ + n_virt,
        n_occ + n_virt,
        n_occ,
        rdm2,
        rdm2_pyscf_order,
    )
    # VVVV
    reorder_2rdm(
        n_occ,
        n_occ,
        n_occ,
        n_occ,
        n_occ + n_virt,
        n_occ + n_virt,
        n_occ + n_virt,
        n_occ + n_virt,
        rdm2,
        rdm2_pyscf_order,
    )

    n_mo = (n_occ + n_virt) // 2

    rdm2_spin_tr = np.zeros((n_mo, n_mo, n_mo, n_mo))

    # spin traced spatial orbital basis 2RDM

    for p in range(2 * n_mo):
        for q in range(2 * n_mo):
            for r in range(2 * n_mo):
                for s in range(2 * n_mo):
                    if r % 2 == p % 2 and s % 2 == q % 2:
                        rdm2_spin_tr[p // 2, r // 2, q // 2, s // 2] += (
                            rdm2_pyscf_order[p, q, r, s]
                        )

    # save to .npy files - temporary solution
    with open("ccsd-frag" + str(frag_number) + "-1rdm.npy", "wb") as f:
        np.save(f, dm_spin_tr)

    with open("ccsd-frag" + str(frag_number) + "-2rdm.npy", "wb") as f:
        np.save(f, rdm2_spin_tr)

    return
