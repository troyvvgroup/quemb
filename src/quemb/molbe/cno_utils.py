import sys
from typing import Literal

import numpy as np
from attrs import define
from pyscf import ao2mo, gto

from quemb.molbe.helper import get_scfObj

CNO_Schemes = Literal["Proportional", "ProportionalQQ", "FragSize"]


@define(frozen=True, kw_only=True)
class CNOArgs:
    """Additional arguments for CNOs.
    cno_scheme options, for now, includes "Proportional", "ProportionalQQ",
    and "FragSize"
    If you specify "FragSize", you also must specify "tot_orbs", which gives
    the total number of orbitals for each fragment. CNOs are added (via some
    scheme) until the fragment size hits FragSize
    """
    cno_scheme: CNO_Schemes | None = "Proportional"
    tot_orbs: int | None = None

def get_cnos(TA, TA_x, hcore_full, eri_full, nocc, occ):
    # TA_x is either TA_occ or TA_vir, aligning with occ

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
    print("P", P)
    # Transform pair density in SO basis
    P_mat_SO = C_SC @ P @ C_SC.T
    print("P_mat_SO", P_mat_SO)
    # Project out FOs and BOs
    P_mat_SO_env_only = np.zeros_like(P_mat_SO)
    P_mat_SO_env_only[
        TA.shape[1]:, TA.shape[1]:] = P_mat_SO[TA.shape[1]:, TA.shape[1]:]
    print("P_mat_SO_env_only", P_mat_SO_env_only)
    # get PNOs by diagonalizing
    P_mat_eigvals, P_mat_eigvecs = np.linalg.eig(P_mat_SO_env_only)
    print("P_mat_eigvals and vecs", P_mat_eigvals, P_mat_eigvecs)

    # change back to AO basis
    cnos = TA_x @ P_mat_eigvecs
    #print("cnos", cnos)
    return cnos

def choose_cnos(file,
                basis: str, 
                n_f: float,
                n_b: float,
                nocc: float,
                nsocc: int,
                args: CNOArgs | None,
                ):
    """
    Options for CNO schemes:
    1. Proportional: Adding virtual until we reach some threshold
    2. ProportionalQQ: 
    3. FragSize: Maximum number of orbitals
    """
    # Options for CNO schemes:
    ###
    assert((args.cno_scheme=="FragSize")==(args.tot_orbs is not None))
    # Build mini fragment to figure out the number of electrons and orbitals

    mol = gto.M()
    mol.atom = file
    mol.basis = basis
    nelec = mol.nelectron

    print("n_f", n_f)
    print("n_b", n_b)
    print("nocc", nocc)
    print("args", args)
    print("nsocc", nsocc)
    print("nelec", nelec)

    print("args.cno_scheme", args.cno_scheme)
    if args.cno_scheme == "Proportional":
        prop = n_f / (nelec / 2)
        nocc_cno_add = 0
        nvir_cno_add = int(np.round(prop * nsocc)) - n_f - n_b
        print("nvir_cno_acc", nvir_cno_add)

    elif args.cno_scheme == "ProportionalQQ":
        prop = n_f / (nelec / 2)
        total_orbs = 1.5 * n_f + n_b
        nocc_cno_add = max(int(np.round(total_orbs / prop - nsocc)), 0)
        nvir_cno_add = total_orbs - n_b - nocc_cno_add - n_f
        print("nocc_cno_add", nocc_cno_add)
        print("nvir_cno_add", nvir_cno_add)

    return nocc_cno_add, nvir_cno_add

def FormPairDensity(Vs, mo_occs, mo_coeffs, mo_energys, occ):
    OccIdx = np.where(mo_occs > 2.0 - 1e-6)[0]
    VirIdx = np.where(mo_occs < 1e-6)[0]
    
    nOcc = OccIdx.shape[0]
    nVir = VirIdx.shape[0]

    COcc = mo_coeffs[:, OccIdx]
    CVir = mo_coeffs[:, VirIdx]

    # Transform 2 e integrals
    V = ao2mo.kernel(Vs, [COcc, CVir, COcc, CVir], compact = False)
    V = V.reshape((nOcc, nVir, nOcc, nVir))
    print("V shape", V.shape)
    print("mo_energys", mo_energys.shape)
    print("nOcc", nOcc)

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
    # Type = 0 or 1 for Occ or Vir
    if occ: # True is occ
        #P = -1 * np.einsum('ikab,kjab->ij', T, Tt)
        # Orig
        #P = 2.0 * np.einsum('kiab,kjab->ij', delta_T_term, T)# + np.einsum('ikab,jkab->ij', t, T)
        # Leah
        P = 2 * np.einsum('kiab,kjab->ij', T, delta_T_term)
        #P = np.eye(T.shape[0]) - P
        P = np.eye(T.shape[0]) - P
    else:
        #P = np.einsum('ijac,ijcb->ab', T, Tt)
        # Orig
        #P = 2.0 * np.einsum('ijca,ijcb->ab', delta_T_term, T)# + np.einsum('ij,ijac,ijbc->ab', N, t, T)
        # Leah
        P = 2.0 * np.einsum('ijac,jicb->ab', T, delta_T_term)

    return P
