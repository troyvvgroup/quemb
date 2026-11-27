# Author(s): Alexa Alexiu

from dataclasses import dataclass
from pathlib import Path

from numpy import argmax, delete, diag, load, ndarray, zeros
from numpy.linalg import norm

from .eom_qchem_parser import EOMDysonData, dyson_parser, dyson_parser_ea


@dataclass
class Fragment_Restart_Data:
    nsocc: int
    mo_energy: ndarray
    mo_coeff: ndarray
    TA: ndarray
    center_ao_indices: ndarray


@dataclass
class Full_Syst_Restart_Data:
    mo_energy: ndarray
    Nocc: int
    ncore: int
    C: ndarray
    S: ndarray
    W: ndarray
    fragments: list[Fragment_Restart_Data]


def load_restart_data(restart_dir: str | Path) -> Full_Syst_Restart_Data:
    restart_dir = Path(restart_dir)
    fragments = []

    for fragment_file in sorted(restart_dir.glob("fragment_*.npz")):
        with load(fragment_file) as fragment_data:
            fragments.append(
                Fragment_Restart_Data(
                    nsocc=int(fragment_data["nsocc"]),
                    mo_energy=fragment_data["mo_energy"],
                    mo_coeff=fragment_data["mo_coeff"],
                    TA=fragment_data["TA"],
                    center_ao_indices=fragment_data["center_ao_indices"],
                )
            )

    with load(restart_dir / "full-system.npz") as full_system_data:
        return Full_Syst_Restart_Data(
            mo_energy=full_system_data["mo_energy"],
            Nocc=int(full_system_data["Nocc"]),
            ncore=int(full_system_data["ncore"]),
            C=full_system_data["C"],
            S=full_system_data["S"],
            W=full_system_data["W"],
            fragments=fragments,
        )


def load_eom_data(
    qchem_dir: str | Path,
    n_fragments: int,
    n_ex: int,
    mode: str,
) -> list[EOMDysonData]:
    qchem_dir = Path(qchem_dir)

    if mode == "ip":
        parser = dyson_parser
    elif mode == "ea":
        parser = dyson_parser_ea
    else:
        raise ValueError("Only EOM-IP and EOM-EA are implemented for now.")

    return [
        parser(qchem_dir / f"qchem_fragment_{fragment_number}" / "eom.out", n_ex)
        for fragment_number in range(n_fragments)
    ]


def get_ip_koopman_full(
    restart_data: Full_Syst_Restart_Data,
) -> tuple[ndarray, ndarray, ndarray]:
    """
    Compute full system Koopman's IP reference.

    This constructs the full system IPs and the corresponding
    Dyson orbital reference vectors in both AO and MO bases. These provide the
    environment reference contribution used in DREAM-IP-BE.

    Parameters
    ----------
    restart_data : Full_Syst_Restart_Data
        Full-system BE information loaded from the
        saved excited state-specific restart files.

    Returns
    -------
    ip_full : ndarray
        Full system Koopman's IPs, in eV.
    dyson_ip_full : ndarray
        Corresponding full system Dyson orbitals in AO basis.
    dyson_ip_full_MO : ndarray
        Corresponding full system Dyson orbitals in MO basis.
    """

    ha_to_ev = 27.2114

    mo_energy = restart_data.mo_energy
    Nocc = restart_data.Nocc
    ncore = restart_data.ncore
    C_MO = restart_data.C

    nao = C_MO.shape[0]
    n_occ = Nocc + ncore

    dyson_ip_full = zeros((n_occ, nao))
    dyson_ip_full_MO = zeros((n_occ, nao))
    ip_full = zeros(n_occ)

    for i in range(n_occ):
        mo_index = n_occ - 1 - i
        ip_full[i] = mo_energy[mo_index]
        dyson_ip_full_MO[i, mo_index] = 1
        dyson_ip_full[i, :] = C_MO[:, mo_index]

    ip_full = (-ip_full) * ha_to_ev

    return ip_full, dyson_ip_full, dyson_ip_full_MO


def be_ip_fragment_koopman(
    restart_data: Full_Syst_Restart_Data,
    fragment_data: Fragment_Restart_Data,
    eom_data: EOMDysonData,
    frag_number: int,
    n_ex: int,
    extra: int = 0,
) -> tuple[ndarray, ndarray, ndarray]:
    """
    Compute the BE-Koopman's-theorem IP component.

    This identifies the dominant occupied MO for each EOM-IP fragment excitation,
    constructs the corresponding fragment Koopman's Dyson orbital, and
    excludes states with unreliable left-Dyson norms.

    Parameters
    ----------
    restart_data : Full_Syst_Restart_Data
        Full-system BE information loaded from the restart files.
    fragment_data : Fragment_Restart_Data
        Saved BE information for one fragment.
    eom_data : EOMDysonData
        EOM-IP energies and Dyson orbitals parsed from that fragment's
        Q-Chem output.
    frag_number : int
        Index of the current fragment
    n_ex : int
        Number of desired EOM-IP excited states
    extra : int
        Number of additional EOM-IP states (sometimes useful to handle intruder states)

    Returns
    -------
    dyson_ip_frag_left : ndarray
        Fragment Koopman left Dyson orbitals.
    dyson_ip_frag_right : ndarray
        Fragment Koopman right Dyson orbitals.
    delta_ex : ndarray
        Fragment Koopman's-theorem IPs, in eV.
    """
    ha_to_ev = 27.2114

    Nocc = restart_data.Nocc
    ncore = restart_data.ncore
    nao = restart_data.C.shape[0]

    mo_energy = fragment_data.mo_energy
    SO_occ = fragment_data.nsocc
    n_states = n_ex + extra

    env_occ = Nocc - SO_occ + ncore

    if eom_data.ex_e.shape[0] < n_states:
        raise ValueError(
            f"Fragment {frag_number} has only {eom_data.ex_e.shape[0]} "
            f"EOM-IP states computed by Qchem, but {n_states} were requested."
        )

    dyson_ip_frag_left = zeros((n_states, nao))
    dyson_ip_frag_right = zeros((n_states, nao))

    ip_frag = zeros(n_states)
    delta_ex = zeros(n_states)

    excluded_koop = []

    for i in range(n_states):
        idx_guess = argmax(abs(eom_data.dyson_right[i, :]))

        norm_left = norm(eom_data.dyson_left[i, :])

        norm_right = norm(eom_data.dyson_right[i, :])

        print(idx_guess, norm_left, norm_right)

        # safety check!
        # if idx_guess points to an environment occupied
        # instead of Schmidt space occupied orbital
        # test with frozen core!

        """if idx_guess - Nocc + SO_occ <= 0:
            idx_guess = Nocc - 1 - i"""

        if not env_occ <= idx_guess < env_occ + SO_occ:
            excluded_koop.append(i)
            print(
                f"WARNING: fragment {frag_number}, state {i} has dominant "
                f"Dyson coefficient outside the occupied Schmidt space: "
                f"MO {idx_guess}. Will be excluded."
            )
        else:
            ip_frag[i] = -mo_energy[idx_guess - env_occ] * ha_to_ev

            dyson_ip_frag_left[i, idx_guess] = 1
            dyson_ip_frag_right[i, idx_guess] = 1

            if norm_left < 0.3 or norm_left > 1.2:
                # not single excitation-like: exclude from Koopman's-BE description
                excluded_koop.append(i)

        if norm_left > 1.2:
            print(
                f"WARNING: fragment {frag_number}, state {i} has "
                f"unreasonable left Dyson norm: {norm_left}"
            )

        if norm_right > 1.2:
            print(
                f"WARNING: fragment {frag_number}, state {i} has "
                f"unreasonable right Dyson norm: {norm_right}"
            )

        delta_ex[i] = ip_frag[i]

        print(frag_number)
        print(
            "Fragment number",
            frag_number,
            "excited state",
            i,
            "excitation energy",
            eom_data.ex_e[i],
            "eV, norm",
            norm_right,
            "from MO:",
            argmax(abs(eom_data.dyson_right[i, :])),
        )

    print("EXCLUDE: ")
    print(frag_number)
    print(excluded_koop)

    dyson_ip_frag_left = delete(dyson_ip_frag_left, excluded_koop, axis=0)
    dyson_ip_frag_right = delete(dyson_ip_frag_right, excluded_koop, axis=0)
    delta_ex = delete(delta_ex, excluded_koop)

    return dyson_ip_frag_left, dyson_ip_frag_right, delta_ex


def be_ip_fragment(
    restart_data: Full_Syst_Restart_Data,
    fragment_data: Fragment_Restart_Data,
    eom_data: EOMDysonData,
    dyson_ip_frag_left: ndarray,
    dyson_ip_frag_right: ndarray,
    delta_ex: ndarray,
    n_ex: int,
    extra: int = 0,
) -> tuple[ndarray, ndarray, ndarray, ndarray]:
    """
    Compute one fragment's Dream-IP-BE matrix contributions (in the AO basis).

    Parameters
    ----------
    restart_data : Full_Syst_Restart_Data
        Full system BE information loaded from the restart files.
    fragment_data : Fragment_Restart_Data
        Saved BE information for one fragment.
    eom_data : EOMDysonData
        Parsed EOM-IP data for the fragment.
    dyson_ip_frag_left : ndarray
        Fragment Koopman left Dyson orbitals.
    dyson_ip_frag_right : ndarray
        Fragment Koopman right Dyson orbitals.
    delta_ex : ndarray
        Fragment Koopman's theorem IPs, in eV.
    n_ex : int
        Number of desired EOM-IP excited states.
    extra : int
        Number of additional EOM-IP states.

    Returns
    -------
    hij_ao : ndarray
        Fragment EOM-IP effective Hamiltonian contribution in the AO basis.
    delta_hij_ao : ndarray
        Fragment Koopman (environment-correction) contribution in the AO basis.
    m_0_ao : ndarray
        Fragment zeroth spectral moment contribution in the AO basis.
    delta_m_0_ao : ndarray
        Fragment Koopman zeroth spectral-moment contribution in the AO basis.
    """
    Nocc = restart_data.Nocc
    ncore = restart_data.ncore
    S = restart_data.S
    W = restart_data.W

    TA = fragment_data.TA
    mo_coeff = fragment_data.mo_coeff
    cind = fragment_data.center_ao_indices

    n_mo_full = TA.shape[0]
    SO_tot = TA.shape[1]
    SO_occ = fragment_data.nsocc
    n_states = n_ex + extra

    env_occ = Nocc - SO_occ + ncore
    env_virt = n_mo_full - SO_tot - env_occ

    exc = diag(eom_data.ex_e[:n_states])

    dyson_left = eom_data.dyson_left[:n_states, env_occ : n_mo_full - env_virt]

    dyson_right = eom_data.dyson_right[:n_states, env_occ : n_mo_full - env_virt]

    # EOM-BE terms
    hij_mo = dyson_left.T @ exc @ dyson_right
    m_0_mo = dyson_left.T @ dyson_right

    # Koopman-BE terms
    delta_ex = diag(delta_ex)

    dyson_ip_frag_left = dyson_ip_frag_left[:, env_occ : n_mo_full - env_virt]
    dyson_ip_frag_right = dyson_ip_frag_right[:, env_occ : n_mo_full - env_virt]

    delta_hij_mo = dyson_ip_frag_left.T @ delta_ex @ dyson_ip_frag_right
    delta_m_0_mo = dyson_ip_frag_left.T @ dyson_ip_frag_right

    # projection matrix
    Pc_ = TA.T @ S @ W[:, cind] @ W[:, cind].T @ S @ TA

    # rotate in SO basis
    hij_so = mo_coeff @ hij_mo @ mo_coeff.T
    # project out non-center contributions
    hij_center = Pc_ @ hij_so
    # transform to AO basis
    hij_ao = TA @ hij_center @ TA.T

    delta_hij_so = mo_coeff @ delta_hij_mo @ mo_coeff.T
    delta_hij_center = Pc_ @ delta_hij_so
    delta_hij_ao = TA @ delta_hij_center @ TA.T

    m_0_so = mo_coeff @ m_0_mo @ mo_coeff.T
    m_0_center = Pc_ @ m_0_so
    m_0_ao = TA @ m_0_center @ TA.T

    delta_m_0_so = mo_coeff @ delta_m_0_mo @ mo_coeff.T
    delta_m_0_center = Pc_ @ delta_m_0_so
    delta_m_0_ao = TA @ delta_m_0_center @ TA.T

    return hij_ao, delta_hij_ao, m_0_ao, delta_m_0_ao


def dream_ip_be(
    restart_data: Full_Syst_Restart_Data,
    eom_data_all: list[EOMDysonData],
    n_ex: int,
    extra: int = 0,
) -> tuple[
    ndarray,
    ndarray,
    ndarray,
    ndarray,
    ndarray,
    ndarray,
    ndarray,
    ndarray,
    ndarray,
]:
    """
    Construct the Dream-IP-BE effective Hamiltonian.

    Parameters
    ----------
    restart_data : Full_Syst_Restart_Data
        Full system and fragment BE information loaded from restart files.
    eom_data_all : list[EOMDysonData]
        Parsed EOM-IP data, ordered by fragment number.
    n_ex : int
        Number of desired EOM-IP excited states.
    extra : int
        Number of additional EOM-IP states.

    Returns
    -------
        hijAO : numpy.ndarray
            Effective Hamiltonian in AO basis, formed from Dyson orbitals.
        delta_hijAO : numpy.ndarray
            Environment correction term in AO basis.
        ip_full : numpy.ndarray
            Full-system IPs.
        dyson_ip_full : numpy.ndarray
            Full system Dyson orbitals corresponding to Koopman's-like excitations.
        hijMO : numpy.ndarray
            Effective Hamiltonian in MO basis, formed from Dyson orbitals.
        delta_hijMO : numpy.ndarray
            Environment correction term in MO basis.
        dyson_ip_full_MO : numpy.ndarray
            Full system Dyson orbitals corresponding to Koopman's-like excitations
            (in MO basis).
        M0_MO : numpy.ndarray
            "0th order spectral moment", or c_L @ c_R.
            Equal to identity matrix in the full system EOM-IP/EA case
            (due to biorthonormality).
        delta_M0_MO : numpy.ndarray
            "0th order spectral moment" for the environment correction term.
    """

    if len(restart_data.fragments) != len(eom_data_all):
        raise ValueError(
            "The number of fragment restart files does not match the "
            "number of available Q-Chem outputs."
        )

    C = restart_data.C
    S = restart_data.S
    nao = C.shape[0]

    hijAO = zeros((nao, nao))
    delta_hijAO = zeros((nao, nao))
    M_0_AO = zeros((nao, nao))
    delta_M_0_AO = zeros((nao, nao))

    # Full system Koopman theory: occupied part of Fock matrix
    ip_full, dyson_ip_full, dyson_ip_full_MO = get_ip_koopman_full(restart_data)

    for frag_number, (fragment_data, eom_data) in enumerate(
        zip(restart_data.fragments, eom_data_all)
    ):
        # Koop-BE
        dyson_ip_frag_left, dyson_ip_frag_right, delta_ex = be_ip_fragment_koopman(
            restart_data,
            fragment_data,
            eom_data,
            frag_number,
            n_ex,
            extra,
        )

        # EOM-BE
        hij_ao, delta_hij_ao, m_0_ao, delta_m_0_ao = be_ip_fragment(
            restart_data,
            fragment_data,
            eom_data,
            dyson_ip_frag_left,
            dyson_ip_frag_right,
            delta_ex,
            n_ex,
            extra,
        )

        hijAO += hij_ao
        delta_hijAO += delta_hij_ao
        M_0_AO += m_0_ao
        delta_M_0_AO += delta_m_0_ao

    hijMO = C.T @ S @ hijAO @ S @ C
    delta_hijMO = C.T @ S @ delta_hijAO @ S @ C

    M0_MO = C.T @ S @ M_0_AO @ S @ C
    delta_M0_MO = C.T @ S @ delta_M_0_AO @ S @ C

    return (
        hijAO,
        delta_hijAO,
        ip_full,
        dyson_ip_full,
        hijMO,
        delta_hijMO,
        dyson_ip_full_MO,
        M0_MO,
        delta_M0_MO,
    )


def run_dream_ip_be(
    restart_dir: str | Path,
    qchem_dir: str | Path,
    n_ex: int,
    extra: int = 0,
) -> tuple[
    ndarray,
    ndarray,
    ndarray,
    ndarray,
    ndarray,
    ndarray,
    ndarray,
    ndarray,
    ndarray,
]:
    """
    Load saved BE and Q-Chem EOM-IP data and construct Dream-IP-BE effective Hamiltonian

    Parameters
    ----------
    restart_dir : str | Path
        Directory containing ``full-system.npz`` and ``fragment_*.npz`` files.
    qchem_dir : str | Path
        Directory containing ``qchem_fragment_0``, ``qchem_fragment_1``, and
        the other fragment Q-Chem-output directories.
    n_ex : int
        Number of desired EOM-IP states.
    extra : int
        Number of additional EOM-IP states.

    Returns
    -------
        hijAO : numpy.ndarray
            Effective Hamiltonian in AO basis, formed from Dyson orbitals.
        delta_hijAO : numpy.ndarray
            Environment correction term in AO basis.
        ip_full : numpy.ndarray
            Full-system IPs.
        dyson_ip_full : numpy.ndarray
            Full system Dyson orbitals corresponding to Koopman's-like excitations.
        hijMO : numpy.ndarray
            Effective Hamiltonian in MO basis, formed from Dyson orbitals.
        delta_hijMO : numpy.ndarray
            Environment correction term in MO basis.
        dyson_ip_full_MO : numpy.ndarray
            Full system Dyson orbitals corresponding to Koopman's-like excitations
            (in MO basis).
        M0_MO : numpy.ndarray
            "0th order spectral moment", or c_L @ c_R.
            Equal to identity matrix in the full system EOM-IP/EA case
            (due to biorthonormality).
        delta_M0_MO : numpy.ndarray
            "0th order spectral moment" for the environment correction term.
    """

    restart_data = load_restart_data(restart_dir)

    eom_data_all = load_eom_data(
        qchem_dir,
        len(restart_data.fragments),
        n_ex=n_ex + extra,
        mode="ip",
    )

    return dream_ip_be(restart_data, eom_data_all, n_ex, extra)
