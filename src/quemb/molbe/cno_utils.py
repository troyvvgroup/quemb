import sys
from typing import Literal

import numpy as np
from attrs import define
from pyscf import ao2mo, gto

from quemb.molbe.helper import get_scfObj
from quemb.shared.typing import Matrix

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

def get_cnos(TA, TA_x, S, hcore_full, eri_full, nocc, occ):
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
    P, V = FormPairDensity(
        eri_schmidt,
        mf_SC.mo_occ,
        mf_SC.mo_coeff,
        mf_SC.mo_energy,
        occ
    )

    # Transform pair density in SO basis
    P_mat_SO = C_SC @ P @ C_SC.T

    # Project out FOs and BOs
    P_mat_SO_env_only = np.zeros_like(P_mat_SO)
    P_mat_SO_env_only[
        TA.shape[1]:, TA.shape[1]:] = P_mat_SO[TA.shape[1]:, TA.shape[1]:]

    # get PNOs by diagonalizing
    P_mat_eigvals, P_mat_eigvecs = np.linalg.eig(P_mat_SO_env_only)
    print("P_mat_eigvals and vecs", P_mat_eigvals, P_mat_eigvecs)

    # change back to AO basis
    cnos = TA_x @ P_mat_eigvecs
    print("cnos", cnos)
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
    print("n_f", n_f)
    print("n_b", n_b)
    print("nocc", nocc)
    print("args", args)
    print("basis", basis)
    assert((args.cno_scheme=="FragSize")==(args.tot_orbs is not None))
    # Build mini fragment to figure out the number of electrons and orbitals

    mol = gto.M()
    mol.atom = file
    mol.basis = basis
    nelec = mol.nelectron
    print("nelec", nelec)
    print("2 * nsocc", 2 * nsocc)

    print("args.cno_scheme", args.cno_scheme)
    if args.cno_scheme == "Proportional":
        prop = n_f / (nelec / 2)
        # CORRECT!! MATCHES!!!
        print("prop", prop)
        nocc_cno_add = 0
        nvir_cno_add = int(np.round(prop * nsocc)) - n_f - n_b
        print("nvir_cno_acc", nvir_cno_add)

    elif args.cno_scheme == "ProportionalQQ":
        # nelec_f = norbs / (nelec / 2)
        # norbs_f = 0.5
        # cass_f = (nelec_f, norbs_f)
        prop = n_f / (nelec / 2)
        total_orbs = 1.5 * n_f + n_b
        nocc_cno_add = max(int(np.round(total_orbs / prop - nsocc)), 0)
        #orb_total = nfA + nbA + int(CASS[1] * nfA)
        #nocc_add = max(int(np.round(orb_total / CASS[0] - nelecA / 2)), 0)
        #CASS = list(CASS)
        #CASS[1] = orb_total # now an int!
        #nvir_add = int(np.round(CASS[1] * noA))
        #nvir_add = CASS[1] - nbA - nocc_add - nfA
        nvir_cno_add = total_orbs - n_b - nocc_cno_add - n_f
        print("propqq", prop)
        print("total_orbs", total_orbs)
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

    mo_energy_occ = mo_energys[:nOcc]
    mo_energy_vir = mo_energys[nOcc:]
    eMat = np.add.outer(mo_energy_occ, mo_energy_occ)
    eMat = np.subtract.outer(eMat, mo_energy_vir)
    eMat = np.subtract.outer(eMat, mo_energy_vir)
    eMat = np.swapaxes(eMat, 1, 2)
    eMat = eMat**-1
    T = -1 * V * eMat
    T = np.swapaxes(T, 1, 2)

    delta_T = 2 * T - np.swapaxes(T, 2, 3)
    # Type = 0 or 1 for Occ or Vir
    if occ: # True is occ
        #P = -1 * np.einsum('ikab,kjab->ij', T, Tt)
        # Orig
        #P = 2.0 * np.einsum('kiab,kjab->ij', delta_T, T)# + np.einsum('ikab,jkab->ij', t, T)
        # Leah
        P = np.einsum('kiab,kjab->ij', T, delta_T)
        P = np.eye(T.shape[0]) - P
    else:
        #P = np.einsum('ijac,ijcb->ab', T, Tt)
        # Orig
        #P = 2.0 * np.einsum('ijca,ijcb->ab', delta_T, T)# + np.einsum('ij,ijac,ijbc->ab', N, t, T)
        # Leah
        P = np.einsum('ijac,jicb->ab', T, delta_T)

    return P, V
"""

def get_cnos_1(TA, TA_vir, TA_occ, hcore_full, eri_full, nocc):
    ###################
    # Occupieds first #
    ###################

    # Generate 1 and 2 electron orbitals in modified Schmidt space
    h_schmidt_occ = np.einsum('mp,nq,mn->pq', TA_occ, TA_occ, hcore_full)
    eri_schmidt_occ = ao2mo.incore.full(eri_full, TA_occ, compact=True)

    # Get semicanonicalized C by solving HF with these 1 and 2 e integrals
    mf_occ_SC = get_scfObj(h_schmidt_occ, eri_schmidt_occ, nocc)
    C_occ_SC = mf_occ_SC.mo_coeff[:, :nocc]
    
    # Get 2 e integrals, transformed by semicanonicalized C
    # Then T amplitudes
    # Then pair densities
    # (all in one function)
    P_occ, V_occ = FormPairDensity(
        eri_schmidt_occ,
        mf_occ_SC.mo_occ,
        mf_occ_SC.mo_coeff,
        mf_occ_SC.mo_energy,
        0
    )

    # Transform pair density in SO basis
    P_mat_occ_SO = C_occ_SC @ P_occ @ C_occ_SC.T
    
    # Project out FOs and BOs
    P_mat_occ_SO_env_only = np.zeros_like(P_mat_occ_SO)
    P_mat_occ_SO_env_only[
        TA.shape[1]:, TA.shape[1]:] = P_mat_occ_SO[TA.shape[1]:, TA.shape[1]:]

    # get PNOs by diagonalizing
    P_mat_occ_eigvals, P_mat_occ_eigvecs = np.linalg.eig(P_mat_occ_SO_env_only)
    print("P_mat_eigvals and vecs", P_mat_occ_eigvals, P_mat_occ_eigvecs)

    # change back to AO basis
    conv_occ = TA_occ @ P_mat_occ_eigvecs
    print("conv_occ", conv_occ)

    #################
    # Virtuals Next #
    #################
    h_schmidt_vir = np.einsum('mp,nq,mn->pq', TA_vir, TA_vir, hcore_full)
    eri_schmidt_vir = ao2mo.incore.full(eri_full, TA_vir, compact=True)

    # Get semicanonicalized C by solving HF with these 1 and 2 e integrals
    mf_vir_SC = get_scfObj(h_schmidt_vir, eri_schmidt_vir, nocc)
    C_vir_SC = mf_vir_SC.mo_coeff[:, nocc:]

    # Get 2 e integrals, transformed by semicanonicalized C
    # Then make pair amplitudes
    # Make pair density matrices
    P_vir, V_vir = FormPairDensity(
        eri_schmidt_vir,
        mf_vir_SC.mo_occ,
        mf_vir_SC.mo_coeff,
        mf_vir_SC.mo_energy,
        1
    )

    # Transform pair densities in SO basis
    P_mat_vir_SO = C_vir_SC @ P_vir @ C_vir_SC.T
    
    # Project out FOs and BOs
    P_mat_vir_SO_env_only = np.zeros_like(P_mat_vir_SO)
    P_mat_vir_SO_env_only[
        TA.shape[1]:, TA.shape[1]:] = P_mat_vir_SO[TA.shape[1]:, TA.shape[1]:]

    # get PNOs by diagonalizing
    P_mat_vir_eigvals, P_mat_vir_eigvecs = np.linalg.eig(P_mat_vir_SO_env_only)

    # change back to AO basis
    conv_vir = TA_vir @ P_mat_vir_eigvecs
    print("P_mat_vir_eigvals and vecs", P_mat_vir_eigvals, P_mat_vir_eigvecs)
    print("conv_vir", conv_vir)

    return conv_occ, conv_vir

def get_pair_amplitude(V, e):
    # T_ij^ab = V_iajb
    # over      e_x + e_b - e_i -e_j
    (i_s, a_s, j_s, b_s) = V.shape
    print("V.shape", V.shape)
    print("i_s", i_s)
    print("a_s", a_s)
    print("j_s", j_s)
    print("b_s", b_s)
    T_amp = np.zeros((a_s, b_s, i_s, j_s))
    for a in range(a_s):
        for b in range(b_s):
            for i in range(i_s):
                for j in range(j_s):
                    T_amp[a,b,i,j] = V[i,a,j,b] / (e[a] + e[b] - e[i] - e[j])
    print("T_amp,", T_amp)
    return T_amp

def get_pair_density_mat(T, occ):
    if occ:
        # I - T_ik^ab (2 T_kj^ab - T_kj^ba)
        P_mat_prod = np.einsum('abik,abkj->ij',T,2*T) - np.einsum('abik,bakj->ij',T,T)
        P_mat = np.eyelike(P_mat_prod) - P_mat_prod
    else:
        # T_ij^ac(2 T_ij^cb - T_ij^bc)
        P_mat = np.einsum('acij,cbij->ab',2*T,T) - np.einsum('acij,bcij->ab',T,T)
    return P_mat

# From Henry:
def GramSchmidt(A, Start = 1):
    B = A.copy()
    for j in range(Start - 1, B.shape[1] - 1):
        CurrentColIdx = j + 1
        CurrentColVec = B[:, CurrentColIdx]
        Projections = CurrentColVec.T @ B[:, :CurrentColIdx]
        for i in range(j + 1):
            CurrentColVec -= Projections[i] * B[:, i]
        Norm = np.linalg.norm(CurrentColVec)
        if Norm < 1e-10:
            CurrentColVec = np.zeros(CurrentColVec.shape)
        else:
            CurrentColVec = CurrentColVec / Norm
        B[:, CurrentColIdx] = CurrentColVec
    return B

def Rotate4DTensor(A, R):
    Vmmaa = np.zeros((R.shape[1], R.shape[1], A.shape[2], A.shape[3]))
    V = np.zeros((R.shape[1], R.shape[1], R.shape[1], R.shape[1]))
    for i in range(A.shape[2]):
        for j in range(A.shape[3]):
            Y = reduce(np.dot, (R.T, A[:, :, i, j], R))
            Vmmaa[:, :, i, j] = Y
    for i in range(Vmmaa.shape[0]):
        for j in range(Vmmaa.shape[1]):
            Y = reduce(np.dot, (R.T, Vmmaa[i, j], R))
            V[i, j] = Y
    return V

def HalfRotate4DTensor(A, R):
    Vmmaa = np.zeros((R.shape[1], R.shape[1], A.shape[2], A.shape[3]))
    for i in range(A.shape[2]):
        for j in range(A.shape[3]):
            Y = reduce(np.dot, (R.T, A[:, :, i, j], R))
            Vmmaa[:, :, i, j] = Y
    return Vmmaa

def HalfRotate4DTensor2(A, R):
    Vaamm = np.zeros((A.shape[0], A.shape[1], R.shape[1], R.shape[1]))
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            Y = reduce(np.dot, (R.T, A[i, j, :, :], R))
            Vaamm[i, j, :, :] = Y
    return Vaamm


def MakePairDensity(T, t):
    PairP = np.zeros(T.shape) 
    for i in range(PairP.shape[0]):
        for j in range(PairP.shape[1]):
            Pij = np.dot(t[i, j].T, T[i, j]) + np.dot(t[i, j], T[i, j].T)
            if i == j:
                Pij = Pij / 2
            PairP[i, j] = Pij
    return PairP

def TracePairDensity(PairP, Idx = None):
    if Idx == None:
        Idx = list(range(PairP.shape[0]))
    P = np.zeros(PairP[0, 0].shape)
    for i in Idx:
        for j in Idx:
            P += PairP[i, j]
    return P

def T2LocalBasis(T2, Clao, mo_coeff):
    n = Clao.shape[0]
    T2LO = np.zeros((n, n, n, n))
    NumOcc = T2.shape[0]
    NumVir = T2.shape[2]
    OccIdx = list(range(NumOcc))
    VirIdx = list(range(NumOcc, n))
    T2LO[np.ix_(OccIdx, OccIdx, VirIdx, VirIdx)] = T2
    UMOtoAO = np.linalg.inv(mo_coeff)
    UMOtoLO = UMOtoAO @ Clao
    T2LO = Rotate4DTensor(T2LO, UMOtoLO)
    return T2LO

def FormPairDensityGeneral(V, mo_energys, Type):
    assert(len(mo_energys) == 4)
    eMat = np.add.outer(mo_energys[0], mo_energys[1])
    eMat = np.subtract.outer(eMat, mo_energys[2])
    eMat = np.subtract.outer(eMat, mo_energys[3])
    eMat = np.swapaxes(eMat, 1, 2)
    eMat = eMat**-1
    T = -1 * V * eMat
    T = np.swapaxes(T, 1, 2)

    Tt = 2 * T - np.swapaxes(T, 2, 3)
    # Type = 0 or 1 for Occ or Vir
    if Type == 0:
        P = -1 * np.einsum('ikab,kjab->ij', T, Tt)
    else:
        P = np.einsum('ijac,ijcb->ab', T, 2 * T) + np.einsum('ijac,ijbc->ab', T, T)
    return P


def GetCASSAdditions(CASS, nfA, nbA, nfA_, nelecA, nocc):
    # Add occupied orbitals until requested # of electrons is achieved and then add on 
    # virtual orbitals until orbitals are satisfied
    '''
    There are many difference cases here that I will describe.
        * For any integers number of electrons or orbitals, this reads the same CAS
          normally, include that many orbitals and that many electrons, taking into
          account the orbitals and electrons already in the fragment and bath.
        * For an integer, negative number of electrons or orbitals, this means add on
          that many electrons and orbitals, not taking into account the electrons and
          orbitals already in the CASS from the fragment and bath.
        * For a float of electrons, the number of orbitals represents the proportion of
          fragment orbitals to add to the space. CASS[0] is then the proportion of total
          orbitals to occupied orbitals desired in this expanded space. If fewer
          electrons are desired than already in the space, then no occupied orbitals are
          added. We prioritize fixing the total orbitals over the ratio of orbitals to
          occupied orbitals.
        * For a float of orbitals, this means add that fractional number of virtual
          orbitals to the CASS based on the number of occupied orbitals. Hence we
          multiply this number by the number of bath orbitals (usually the number of
          occupied orbitals) and then subtract the number of fragment orbitals (usually
          the number of virtual orbitals). The remaider is how many virtual orbitals are
          needed to get the ratio of occ:vir orbitals
        * For a None, this means add no electrons or virtual orbitals.
    '''
    noA = int(nelecA / 2)
    if isinstance(CASS[0], (int, np.integer)):
        if CASS[0] > 0:
            nocc_add = int(np.round((CASS[0] - nelecA) / 2))
        if CASS[0] <= 0:
            nocc_add = int(np.round(abs(CASS[0]) / 2))
    elif isinstance(CASS[0], float):
        orb_total = nfA + nbA + int(CASS[1] * nfA)
        nocc_add = max(int(np.round(orb_total / CASS[0] - nelecA / 2)), 0)
        CASS = list(CASS)
        CASS[1] = orb_total
    elif CASS[0] is None:
        nocc_add = 0
    else:
        raise RuntimeError("Unrecognized CASS input.")

    if isinstance(CASS[1], (int, np.integer)):
        if CASS[1] > 0:
            nvir_add = CASS[1] - nbA - nocc_add - nfA
        if CASS[1] <= 0:
            nvir_add = abs(CASS[1])
    elif isinstance(CASS[1], float):
        nvir_add = int(np.round(CASS[1] * noA))
        nvir_add = nvir_add - nfA - nbA
        if nvir_add < 0: # means we need to add occupied orbitals instead
            nvir_add = 0
            nocc_add = int(np.round((CASS[1] * noA - nbA - nfA) / (1. - CASS[1])))
    elif CASS[1] is None:
        nvir_add = 0
    else:
        raise RuntimeError("Unrecognized CASS input.")

    # Check that we don't have too few electrons or orbitals
    if nocc_add < 0:
        raise RuntimeError("Too few electrons in CASS")
    if nocc_add > nocc - int(np.round(nelecA / 2)):
        raise RuntimeError("Too many electrons in CASS")
    if nvir_add < 0:
        raise RuntimeError("Too few orbitals in CASS")
    if nvir_add > nfA + nfA_ - nocc:
        raise RuntimeError("Too many orbitals in CASS")

    return nocc_add, nvir_add

"""