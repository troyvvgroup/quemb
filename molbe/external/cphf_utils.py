# Author(s): Hong-Zhou Ye
# NOTICE: The following code is entirely written by Hong-Zhou Ye.
#         The code has been slightly modified.
#
import numpy as np

from pyscf import ao2mo


def get_cphf_A(C, moe, eri, no):
    nao = C.shape[0]
    nv = nao - no
    Co = C[:, :no]
    Cv = C[:, no:]
    nov = no * nv
    Vovov = ao2mo.incore.general(eri, (Co, Cv, Co, Cv), compact=False).reshape(
        no, nv, no, nv
    )
    Voovv = ao2mo.incore.general(eri, (Co, Co, Cv, Cv), compact=False).reshape(
        no, no, nv, nv
    )

    # 4*Viajb - Vibja - Vjiab
    A = (
        4.0 * Vovov - Vovov.transpose(0, 3, 2, 1) - Voovv.transpose(0, 2, 1, 3)
    ).reshape(nov, nov)
    denom = (moe[:no].reshape(-1, 1) - moe[no:]).ravel()
    A -= np.diag(denom)

    return A


def get_cphf_rhs(C, no, v):
    nao = C.shape[0]
    nv = nao - no
    Co = C[:, :no]
    Cv = C[:, no:]

    return (Co.T @ v @ Cv).ravel()


def cphf_kernel(C, moe, eri, no, v):
    nao = C.shape[0]
    nv = nao - no

    # build RHS vector, B0
    B0 = get_cphf_rhs(C, no, v)

    # build A matrix
    A = get_cphf_A(C, moe, eri, no)

    # solve for u
    u = np.linalg.solve(A, B0)

    return u


def cphf_kernel_batch(C, moe, eri, no, vs):
    nao = C.shape[0]
    nv = nao - no
    nov = no * nv
    npot = len(vs)

    # build RHS vectors
    B0s = np.zeros([nov, npot])
    for i, v in enumerate(vs):
        B0s[:, i] = get_cphf_rhs(C, no, v)

    # build A matrix
    A = get_cphf_A(C, moe, eri, no)

    # solve for u
    us = np.linalg.solve(A, B0s).T

    return us


def get_rhf_dP_from_u(C, no, u):
    n = C.shape[0]
    nv = n - no
    dP = -C[:, :no] @ u.reshape(no, nv) @ C[:, no:].T
    dP += dP.T

    return dP


def get_full_u(C, moe, eri, no, v, u, thresh=1e8):
    nao = C.shape[0]
    nv = nao - no
    Co = C[:, :no]
    Cv = C[:, no:]
    moeo = moe[:no]
    moev = moe[no:]

    Voovo = ao2mo.incore.general(eri, (Co, Co, Cv, Co), compact=False).reshape(
        no, no, nv, no
    )
    Vvvvo = ao2mo.incore.general(eri, (Cv, Cv, Cv, Co), compact=False).reshape(
        nv, nv, nv, no
    )
    vmo = C.T @ v @ C

    uvo = u.reshape(nv, no)
    U = np.zeros([nao, nao])
    U[no:, :no] = uvo
    U[:no, no:] = -uvo.T

    # occupied-occupied block
    # ( Qjk + sum_ai (4*Vjkai-Vjiak-Vikaj)*uai ) / (ek - ej)
    denom_oo = moeo.reshape(no, 1) - moeo
    denom_oo[np.diag_indices(no)] = 1.0
    if np.sum(np.abs(denom_oo**-1) > thresh) > 0:
        raise RuntimeError
    U[:no, :no] = (
        -(
            vmo[:no, :no]
            + np.einsum(
                "jkai,ai->jk",
                4.0 * Voovo - Voovo.transpose(0, 3, 2, 1) - Voovo.transpose(3, 1, 2, 0),
                uvo,
            )
        )
        / denom_oo
    )
    U[np.diag_indices(no)] = 0.0

    # virtual-virtual block
    # ( Qbc + sum_ai (4*Vbcai-Vacbi-Vbaci)*uai ) / (ec - eb)
    denom_vv = moev.reshape(nv, 1) - moev
    denom_vv[np.diag_indices(nv)] = 1.0
    if np.sum(np.abs(denom_vv**-1) > thresh) > 0:
        raise RuntimeError
    uvv = (
        -(
            vmo[no:, no:]
            + np.einsum(
                "bcai,ai->bc",
                4.0 * Vvvvo - Vvvvo.transpose(2, 1, 0, 3) - Vvvvo.transpose(0, 2, 1, 3),
                uvo,
            )
        )
        / denom_vv
    )
    uvv[np.diag_indices(nv)] = 0.0
    U[no:, no:] = uvv

    return U


def get_full_u_batch(C, moe, eri, no, vs, us, thresh=1e10, timing=False):
    nao = C.shape[0]
    nv = nao - no
    npot = len(vs)
    Co = C[:, :no]
    Cv = C[:, no:]
    moeo = moe[:no]
    moev = moe[no:]

    Voovo = ao2mo.incore.general(eri, (Co, Co, Cv, Co), compact=False).reshape(
        no, no, nv, no
    )
    Vvvvo = ao2mo.incore.general(eri, (Cv, Cv, Cv, Co), compact=False).reshape(
        nv, nv, nv, no
    )

    Us = [None for i in range(npot)]
    for i in range(npot):
        vmo = C.T @ vs[i] @ C

        uvo = us[:, i].reshape(nv, no)
        U = np.zeros([nao, nao])
        U[no:, :no] = uvo
        U[:no, no:] = -uvo.T

        # occupied-occupied block
        # ( Qjk + sum_ai (4*Vjkai-Vjiak-Vikaj)*uai ) / (ek - ej)
        denom_oo = moeo.reshape(no, 1) - moeo
        denom_oo[np.diag_indices(no)] = 1.0
        if np.sum(np.abs(denom_oo**-1) > thresh) > 0:
            raise RuntimeError
        U[:no, :no] = (
            -(
                vmo[:no, :no]
                + np.einsum(
                    "jkai,ai->jk",
                    4.0 * Voovo
                    - Voovo.transpose(0, 3, 2, 1)
                    - Voovo.transpose(3, 1, 2, 0),
                    uvo,
                )
            )
            / denom_oo
        )
        U[np.diag_indices(no)] = 0.0

        # virtual-virtual block
        # ( Qbc + sum_ai (4*Vbcai-Vacbi-Vbaci)*uai ) / (ec - eb)
        denom_vv = moev.reshape(nv, 1) - moev
        denom_vv[np.diag_indices(nv)] = 1.0
        if np.sum(np.abs(denom_vv**-1) > thresh) > 0:
            raise RuntimeError
        uvv = (
            -(
                vmo[no:, no:]
                + np.einsum(
                    "bcai,ai->bc",
                    4.0 * Vvvvo
                    - Vvvvo.transpose(2, 1, 0, 3)
                    - Vvvvo.transpose(0, 2, 1, 3),
                    uvo,
                )
            )
            / denom_vv
        )
        uvv[np.diag_indices(nv)] = 0.0
        U[no:, no:] = uvv

        Us[i] = U

    return Us


def uvo_as_full_u_batch(nao, no, us):
    ncon = us.shape[1]
    nv = nao - no
    Us = [None] * ncon
    for i in range(ncon):
        uvo = us[:, i].reshape(nv, no)
        U = np.zeros([nao, nao])
        U[no:, :no] = uvo
        U[:no, no:] = -uvo.T
        Us[i] = U

    return Us


def get_dP_lagrangian(C, no):
    nao = C.shape[0]
    nv = nao - no

    L = np.zeros([nv * no, nao * (nao + 1) // 2])
    mn = -1
    for mu in range(nao):
        for nu in range(mu, nao):
            mn += 1
            ai = -1
            for a in range(nv):
                for i in range(no):
                    ai += 1
                    L[ai, mn] = C[mu, a + no] * C[nu, i] + C[mu, i] * C[nu, a + no]

    return L


def get_zvec(C, moe, eri, no):
    nao = C.shape[0]
    nv = nao - no

    # build A matrix
    A = get_cphf_A(C, moe, eri, no)

    # build Lagrangian
    L = get_dP_lagrangian(C, no)

    # solve for z vector
    z = np.linalg.solve(np.eye(nv * no) - A.T, L)

    return z


""" For UHF
"""


def get_cpuhf_A_spinless_eri(C, moe, eri, no):
    n = C[0].shape[0]
    nv = [n - no[s] for s in [0, 1]]
    nov = [no[s] * nv[s] for s in [0, 1]]
    Co = [C[s][:, : no[s]] for s in [0, 1]]
    Cv = [C[s][:, no[s] :] for s in [0, 1]]
    moeo = [moe[s][: no[s]] for s in [0, 1]]
    moev = [moe[s][no[s] :] for s in [0, 1]]

    nA = sum(nov)
    A = np.empty([nA, nA])
    for s in [0, 1]:
        # same spin
        Vss_ovov = ao2mo.incore.general(
            eri, (Co[s], Cv[s], Co[s], Cv[s]), compact=False
        ).reshape(no[s], nv[s], no[s], nv[s])
        Vss_oovv = ao2mo.incore.general(
            eri, (Co[s], Co[s], Cv[s], Cv[s]), compact=False
        ).reshape(no[s], no[s], nv[s], nv[s])
        Ass = (Vss_ovov * 2 - Vss_ovov.transpose(0, 3, 2, 1)).reshape(
            nov[s], nov[s]
        ) - Vss_oovv.transpose(0, 2, 1, 3).reshape(nov[s], nov[s])
        Dss = (moeo[s].reshape(-1, 1) - moev[s]).ravel()
        Ass[np.diag_indices(nov[s])] -= Dss
        if s == 0:
            A[: nov[s], : nov[s]] = Ass
        else:
            A[nov[s] :, nov[s] :] = Ass
        # opposite spin
        Vos_ovov = (
            ao2mo.incore.general(
                eri, (Co[s], Cv[s], Co[1 - s], Cv[1 - s]), compact=False
            )
            * 2.0
        )
        if s == 0:
            A[: nov[s], nov[s] :] = Vos_ovov
        else:
            A[nov[s] :, : nov[s]] = Vos_ovov

    return A


def get_cpuhf_A_spin_eri(C, moe, eri, no):
    n = C[0].shape[0]
    nv = [n - no[s] for s in [0, 1]]
    nov = [no[s] * nv[s] for s in [0, 1]]
    Co = [C[s][:, : no[s]] for s in [0, 1]]
    Cv = [C[s][:, no[s] :] for s in [0, 1]]
    moeo = [moe[s][: no[s]] for s in [0, 1]]
    moev = [moe[s][no[s] :] for s in [0, 1]]

    nA = sum(nov)
    A = np.empty([nA, nA])
    for s in [0, 1]:
        # same spin
        Vss_ovov = ao2mo.incore.general(
            eri[s], (Co[s], Cv[s], Co[s], Cv[s]), compact=False
        )
        Vss_oovv = ao2mo.incore.general(
            eri[s], (Co[s], Co[s], Cv[s], Cv[s]), compact=False
        )
        Ass = (Vss_ovov * 2 - Vss_ovov.transpose(0, 3, 2, 1)).reshape(
            nov[s], nov[s]
        ) - Vss_oovv.transpose(0, 2, 1, 3).reshape(nov[s], nov[s])
        Dss = (moeo[s].reshape(-1, 1) - moev[s]).ravel()
        Ass[np.diag_indices(nov[s])] -= Dss
        if s == 0:
            A[: nov[s], : nov[s]] = Ass
        else:
            A[nov[s] :, nov[s] :] = Ass
        # opposite spin
        Vos_ovov = (
            ao2mo.incore.general(
                eri[2], (Co[0], Cv[0], Co[1], Cv[1]), compact=False
            ).reshape(nov[0], nov[1])
            * 2.0
        )
        if s == 1:
            Vos_ovov = Vos_ovov.T
        if s == 0:
            A[: nov[s], nov[s] :] = Vos_ovov
        else:
            A[nov[s] :, : nov[s]] = Vos_ovov

    return A


def get_cpuhf_A(C, moe, eri, no):
    """eri could be a single numpy array (in spinless basis) or a tuple/list of 3 numpy arrays in the order [aa,bb,ab]"""
    if isinstance(eri, np.ndarray):
        return get_cpuhf_A_spinless_eri(C, moe, eri, no)
    elif isinstance(eri, (list, tuple)):
        if len(eri) != 3:
            raise ValueError(
                "Input eri must be a list/tuple of 3 numpy arrays in the order [aa,bb,ab]"
            )
        return get_cpuhf_A_spin_eri(C, moe, eri, no)
    else:
        raise ValueError(
            "Input eri must be either a numpy array or a list/tuple of 3 numpy arrays in the order [aa,bb,ab]."
        )


def get_cpuhf_u(C, moe, eri, no, vpot):
    n = C[0].shape[0]
    nv = [n - no[s] for s in [0, 1]]
    nov = [no[s] * nv[s] for s in [0, 1]]
    Co = [C[s][:, : no[s]] for s in [0, 1]]
    Cv = [C[s][:, no[s] :] for s in [0, 1]]

    # build lhs
    A = get_cpuhf_A(C, moe, eri, no)

    # build rhs
    b = np.concatenate([(Co[s].T @ vpot[s] @ Cv[s]).ravel() for s in [0, 1]])

    # solve
    u = np.linalg.solve(A, b)

    return u


def get_cpuhf_u_batch(C, moe, eri, no, vpots):
    n = C[0].shape[0]
    nv = [n - no[s] for s in [0, 1]]
    nov = [no[s] * nv[s] for s in [0, 1]]
    Co = [C[s][:, : no[s]] for s in [0, 1]]
    Cv = [C[s][:, no[s] :] for s in [0, 1]]

    # build lhs
    A = get_cpuhf_A(C, moe, eri, no)

    # build rhs
    nA = A.shape[0]
    npot = len(vpots)
    bs = np.zeros([nA, npot])
    for i in range(npot):
        bs[:, i] = np.concatenate(
            [(Co[s].T @ vpots[i][s] @ Cv[s]).ravel() for s in [0, 1]]
        )

    # solve
    us = np.linalg.solve(A, bs)

    return us.T


def get_uhf_dP_from_u(C, no, u):
    n = C[0].shape[0]
    nv = [n - no[s] for s in [0, 1]]
    nov = [no[s] * nv[s] for s in [0, 1]]
    Co = [C[s][:, : no[s]] for s in [0, 1]]
    Cv = [C[s][:, no[s] :] for s in [0, 1]]
    dP = [None] * 2
    for s in [0, 1]:
        u_ = (
            u[: nov[0]].reshape([no[s], nv[s]])
            if s == 0
            else u[nov[0] :].reshape(no[s], nv[s])
        )
        dP[s] = -Co[s] @ u_ @ Cv[s].T
        dP[s] += dP[s].T

    return dP
