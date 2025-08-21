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
    and "ExactFragSize"
    If you specify "ExactFragSize", you also must specify "tot_frag_orbs", which gives
    the total number of orbitals for each fragment. CNOs are added (via some
    scheme) until the fragment size hits ExactFragSize
    """
    cno_scheme: CNO_Schemes | None = "Proportional"
    tot_frag_orbs: int | None = None

def get_cnos(TA, TA_x, hcore_full, eri_full, nocc, occ):
    # TA_x is either TA_occ or TA_vir, aligning with occ=True or False
    ta0, nfb = TA.shape
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

def choose_cnos(file,
                basis: str, 
                n_f: float,
                n_b: float,
                n_full_occ: float,
                n_full_vir: float,
                nsocc: int,
                args: CNOArgs | None,
                ):
    """
    Options for CNO schemes:
    1. Proportional: Adding virtual until we reach some threshold
    2. ProportionalQQ: 
    3. ExactFragSize: Maximum number of orbitals
    """
    # Options for CNO schemes:
    ###
    assert((args.cno_scheme=="ExactFragSize")==(args.tot_frag_orbs is not None))
    # Build mini fragment to figure out the number of electrons and orbitals

    mol = gto.M()
    mol.atom = file
    mol.basis = basis
    nelec = mol.nelectron

    if args.cno_scheme == "Proportional":
        # Ratio of the number of fragment orbitals to the number of expected 
        # occupied, based on the atoms in the fragment
        prop = n_f / (nelec / 2)
        nocc_cno_add = 0
        # Add virtual orbitals so that the proportion of all fragment orbitals 
        # (n_f + n_b + nvir_cno_add) to the number of occupied orbitals in the
        # Schmidt space (nsocc) is the same as the ratio `prop` above
        nvir_cno_add = np.round(prop * nsocc) - n_f - n_b

    elif args.cno_scheme == "ProportionalQQ":
        prop = n_f / (nelec / 2)
        total_orbs = n_f + n_b + prop * n_f
        nocc_cno_add = max(int(np.round(total_orbs / 2 - nsocc)), 0)
        nvir_cno_add = total_orbs - n_b - nocc_cno_add - n_f

    elif args.cno_scheme == "ExactFragSize":
        # Start by adding virtuals until `Proportional` is hit
        prop = n_f / (nelec / 2)
        max_vir_add_prop = np.round(prop * nsocc) - n_f - n_b

        # Schmidt state is already bigger than the max fragment size
        if args.tot_frag_orbs < n_f + n_b:
            raise ValueError("Max fragment size larger than fragment + bath space")
        # We will add virtual CNOs until the ratio of the augmented 
        # fragment space to the number of occupieds in the Schmidt spaces reaches 
        # the proportion above: see `Proportional`
        elif 0 <= args.tot_frag_orbs - n_f - n_b <= max_vir_add_prop:
            nocc_cno_add = 0
            nvir_cno_add = args.tot_frag_orbs - n_f - n_b
        # We need to also add occupieds here. We will now try to satisfy the
        # proportional scheme by adding a certain number of occupieds and virtuals,
        # as closely as possible
        else:
            nocc_cno_add = np.round(args.tot_frag_orbs / prop) - nsocc
            nvir_cno_add = args.tot_frag_orbs - n_f - n_b - nocc_cno_add
    
    if nocc_cno_add +  n_f + n_b > n_full_occ:
        raise RuntimeError(
            "Request to add more occupied CNOs than exist. Choose different CNO scheme"
            )
    elif nvir_cno_add + n_f + n_b > n_full_vir:
        raise RuntimeError(
            "Request to add more virtual CNOs than exist. Choose different CNO scheme"
            )

    return int(nocc_cno_add), int(nvir_cno_add)

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
    
    if occ: # True is occ
        P = 2 * np.einsum('kiab,kjab->ij', T, delta_T_term)
        P = np.eye(T.shape[0]) - P
    else:
        P = 2.0 * np.einsum('ijac,jicb->ab', T, delta_T_term)

    return P

def augment_w_cnos(TA, nocc_cno, nvir_cno, occ_cno, vir_cno):
    if nocc_cno > 0:
        TA = np.hstack((TA, occ_cno[:, :nocc_cno]))
    if nvir_cno > 0:
        TA = np.hstack((TA, vir_cno[:, :nvir_cno]))
    return TA