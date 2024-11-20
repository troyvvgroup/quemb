# Author(s): Hong-Zhou Ye
# NOTICE: The following code is entirely written by Hong-Zhou Ye.
#         The code has been slightly modified.
#

import numpy as np
import scipy.linalg as slg

from pyscf import ao2mo


""" RMP2 implementation
"""


def get_Diajb_r(moe, no):
    n = moe.size
    nv = n - no
    eo = moe[:no]
    ev = moe[no:]
    Dia = (eo.reshape(-1, 1) - ev).ravel()
    Diajb = (Dia.reshape(-1, 1) + Dia).reshape(no, nv, no, nv)

    return Diajb


def get_dF_r(no, V, C, Q, u):
    from pyscf import scf

    n = C.shape[0]
    nv = n - no
    Co = C[:, :no]
    Cv = C[:, no:]
    uov = u.reshape(no, nv)
    dP = -Co @ uov @ Cv.T
    dP += dP.T
    vj, vk = scf.hf.dot_eri_dm(V, dP * 2.0, hermi=1)
    dF = Q + vj - 0.5 * vk

    return dF


def get_dmoe_F_r(C, dF):
    de = np.einsum("pi,qi,pq->i", C, C, dF)

    return de


def get_full_u_F_r(no, C, moe, dF, u):
    n = C.shape[0]
    nv = n - no
    Co = C[:, :no]
    Cv = C[:, no:]
    eo = moe[:no]
    ev = moe[no:]
    uov = u.reshape(no, nv)
    Dij = -eo.reshape(-1, 1) + eo
    np.fill_diagonal(Dij, 1)
    dUoo = (Co.T @ dF @ Co) / Dij
    np.fill_diagonal(dUoo, 0.0)
    Dab = -ev.reshape(-1, 1) + ev
    np.fill_diagonal(Dab, 1)
    dUvv = (Cv.T @ dF @ Cv) / Dab
    np.fill_diagonal(dUvv, 0.0)

    return np.block([[dUoo, uov], [-uov.T, dUvv]])


def get_dVovov_r(no, V, C, u):
    n = C.shape[0]
    nv = n - no
    nov = no * nv
    Co = C[:, :no]
    Cv = C[:, no:]
    if u.size == no * nv:
        uov = u.reshape(no, nv)
        dCv = Co @ uov
        dCo = -Cv @ uov.T
    else:
        dC = C @ u
        dCo = dC[:, :no]
        dCv = dC[:, no:]
    Vovov_ = ao2mo.incore.general(V, (Co, Cv, Co, dCv)).reshape(nov, nov)
    Vovo_v = ao2mo.incore.general(V, (Co, Cv, dCo, Cv)).reshape(nov, nov)
    dVovov = (Vovov_ + Vovov_.T + Vovo_v + Vovo_v.T).reshape(no, nv, no, nv)

    return dVovov


def get_Pmp2_r(t2l, t2r):
    assert t2l.ndim == t2r.ndim == 4
    no = t2l.shape[0]
    Poo = -np.einsum(
        "iajb,majb->im", t2l, 2.0 * t2r - t2r.transpose(0, 3, 2, 1), optimize=True
    )
    Pvv = np.einsum(
        "iajb,icjb->ac", t2l, 2.0 * t2r - t2r.transpose(0, 3, 2, 1), optimize=True
    )

    return slg.block_diag(Poo, Pvv)


def get_dPmp2_batch_r(C, moe, V, no, Qs, aorep=True):
    """Derivative of oo and vv block of the MP2 density"""
    n = C.shape[0]
    nv = n - no
    Co = C[:, :no]
    Cv = C[:, no:]
    Vovov = ao2mo.incore.general(V, (Co, Cv, Co, Cv)).reshape(no, nv, no, nv)
    Diajb = get_Diajb_r(moe, no)
    t2 = Vovov / Diajb

    from .cphf_utils import cphf_kernel_batch as cphf_kernel

    us = cphf_kernel(C, moe, V, no, Qs)
    nQ = len(Qs)
    dPs = [None] * nQ
    Phf = np.diag([1 if i < no else 0 for i in range(n)])
    iQ = 0
    for u, Q in zip(us, Qs):
        dF = get_dF_r(no, V, C, Q, u)
        dmoe = get_dmoe_F_r(C, dF)
        dDiajb = get_Diajb_r(dmoe, no)
        U = get_full_u_F_r(no, C, moe, dF, u)
        dVovov = get_dVovov_r(no, V, C, U)
        dt2 = (dVovov - t2 * dDiajb) / Diajb

        P = get_Pmp2_r(t2, t2)
        P += Phf
        dP = U @ P - P @ U

        dP2 = get_Pmp2_r(dt2, t2)
        dP2 += dP2.T

        dPs[iQ] = (dP + dP2) * 2.0

        iQ += 1

    if aorep:
        for iQ in range(nQ):
            dPs[iQ] = C @ dPs[iQ] @ C.T

    return dPs


""" UMP2 implementation
"""


def get_Diajb_u(moe, no):
    n = [moe[s].size for s in [0, 1]]
    nv = [n[s] - no[s] for s in [0, 1]]
    eo = [moe[s][: no[s]] for s in [0, 1]]
    ev = [moe[s][no[s] :] for s in [0, 1]]
    Dia = [(eo[s].reshape(-1, 1) - ev[s]).ravel() for s in [0, 1]]
    Diajb = [
        (Dia[0].reshape(-1, 1) + Dia[0]).reshape(no[0], nv[0], no[0], nv[0]),
        (Dia[1].reshape(-1, 1) + Dia[1]).reshape(no[1], nv[1], no[1], nv[1]),
        (Dia[0].reshape(-1, 1) + Dia[1]).reshape(no[0], nv[0], no[1], nv[1]),
    ]

    return Diajb


def get_dF_u(no, V, C, Q, u):
    n = [C[s].shape[0] for s in [0, 1]]
    nv = [n[s] - no[s] for s in [0, 1]]
    Co = [C[s][:, : no[s]] for s in [0, 1]]
    Cv = [C[s][:, no[s] :] for s in [0, 1]]
    dP = [None] * 2
    for s in [0, 1]:
        uov = u[s].reshape(no[s], nv[s])
        dP[s] = -Co[s] @ uov @ Cv[s].T
        dP[s] += dP[s].T

    dF = [None] * 2
    for s in [0, 1]:
        vj_ss = np.einsum("pqrs,sr->pq", V[s], dP[s])
        vj_os = (
            np.einsum("pqrs,sr->pq", V[2], dP[1])
            if s == 0
            else np.einsum("pqrs,qp->rs", V[2], dP[0])
        )
        vk_ss = np.einsum("psrq,sr->pq", V[s], dP[s])
        dF[s] = Q[s] + vj_ss + vj_os - vk_ss

    return dF


def get_dmoe_F_u(C, dF):
    de = [np.einsum("pi,qi,pq->i", C[s], C[s], dF[s]) for s in [0, 1]]

    return de


def get_full_u_F_u(no, C, moe, dF, u):
    n = [C[s].shape[0] for s in [0, 1]]
    nv = [n[s] - no[s] for s in [0, 1]]
    Co = [C[s][:, : no[s]] for s in [0, 1]]
    Cv = [C[s][:, no[s] :] for s in [0, 1]]
    eo = [moe[s][: no[s]] for s in [0, 1]]
    ev = [moe[s][no[s] :] for s in [0, 1]]
    U = [None] * 2
    for s in [0, 1]:
        uov = u[s].reshape(no[s], nv[s])
        Dij = -eo[s].reshape(-1, 1) + eo[s]
        np.fill_diagonal(Dij, 1)
        dUoo = (Co[s].T @ dF[s] @ Co[s]) / Dij
        np.fill_diagonal(dUoo, 0.0)
        Dab = -ev[s].reshape(-1, 1) + ev[s]
        np.fill_diagonal(Dab, 1)
        dUvv = (Cv[s].T @ dF[s] @ Cv[s]) / Dab
        np.fill_diagonal(dUvv, 0.0)
        U[s] = np.block([[dUoo, uov], [-uov.T, dUvv]])

    return U


def get_dVovov_u(no, V, C, u):
    n = [C[s].shape[0] for s in [0, 1]]
    nv = [n[s] - no[s] for s in [0, 1]]
    nov = [no[s] * nv[s] for s in [0, 1]]
    Co = [C[s][:, : no[s]] for s in [0, 1]]
    Cv = [C[s][:, no[s] :] for s in [0, 1]]
    dCo = [None] * 2
    dCv = [None] * 2
    dVovov = [None] * 3
    for s in [0, 1]:
        dVovov[s] = get_dVovov_r(no[s], V[s], C[s], u[s])
        if u[s].size == no[s] * nv[s]:
            uov = u[s].reshape(no[s], nv[s])
            dCv[s] = Co[s] @ uov
            dCo[s] = -Cv[s] @ uov.T
        else:
            dC = C[s] @ u[s]
            dCo[s] = dC[:, : no[s]]
            dCv[s] = dC[:, no[s] :]

    dVovov[2] = (
        np.einsum(
            "pqrs,pi,qa,rj,sb->iajb", V[2], dCo[0], Cv[0], Co[1], Cv[1], optimize=True
        )
        + np.einsum(
            "pqrs,pi,qa,rj,sb->iajb", V[2], Co[0], dCv[0], Co[1], Cv[1], optimize=True
        )
        + np.einsum(
            "pqrs,pi,qa,rj,sb->iajb", V[2], Co[0], Cv[0], dCo[1], Cv[1], optimize=True
        )
        + np.einsum(
            "pqrs,pi,qa,rj,sb->iajb", V[2], Co[0], Cv[0], Co[1], dCv[1], optimize=True
        )
    )

    return dVovov


def get_Pmp2_u(t2l, t2r):
    assert len(t2l) == len(t2r) == 3  # aa,bb,ab
    assert t2l[0].ndim == t2r[0].ndim == 4  # ovov
    no = [t2l[s].shape[0] for s in [0, 1]]
    es_pattern = ["iajb,majb->im", "iajb,iamb->jm"]
    Poo = [
        -(
            np.einsum(
                es_pattern[0],
                t2l[s],
                t2r[s] - t2r[s].transpose(0, 3, 2, 1),
                optimize=True,
            )
            + np.einsum(es_pattern[s], t2l[2], t2r[2], optimize=True)
        )
        for s in [0, 1]
    ]
    es_pattern = ["iajb,icjb->ac", "iajb,iajc->bc"]
    Pvv = [
        (
            np.einsum(
                es_pattern[0],
                t2l[s],
                t2r[s] - t2r[s].transpose(0, 3, 2, 1),
                optimize=True,
            )
            + np.einsum(es_pattern[s], t2l[2], t2r[2], optimize=True)
        )
        for s in [0, 1]
    ]

    return [slg.block_diag(Poo[s], Pvv[s]) for s in [0, 1]]


def get_dPmp2_batch_u(C, moe, V, no, Qs, aorep=True):
    """no = [noa, nob]
    V = [Vaa, Vbb, Vab]
    C = [Ca, Cb]
    moe = [moea, moeb]
    """
    n = [C[s].shape[0] for s in [0, 1]]
    nv = [n[s] - no[s] for s in [0, 1]]
    nov = [no[s] * nv[s] for s in [0, 1]]
    Co = [C[s][:, : no[s]] for s in [0, 1]]
    Cv = [C[s][:, no[s] :] for s in [0, 1]]
    Vovov = [
        np.einsum(
            "pqrs,pi,qa,rj,sb->iajb", V[0], Co[0], Cv[0], Co[0], Cv[0], optimize=True
        ),
        np.einsum(
            "pqrs,pi,qa,rj,sb->iajb", V[1], Co[1], Cv[1], Co[1], Cv[1], optimize=True
        ),
        np.einsum(
            "pqrs,pi,qa,rj,sb->iajb", V[2], Co[0], Cv[0], Co[1], Cv[1], optimize=True
        ),
    ]
    Diajb = get_Diajb_u(moe, no)
    t2 = [Vovov[s] / Diajb[s] for s in range(3)]

    from .cphf_utils import get_cpuhf_u_batch as cphf_kernel

    us = cphf_kernel(C, moe, V, no, Qs)
    nQ = len(Qs)
    dPs = [None] * nQ
    Phf = [np.diag([1 if i < no[s] else 0 for i in range(n[s])]) for s in [0, 1]]
    iQ = 0
    for u_, Q in zip(us, Qs):
        u = [u_[: nov[0]], u_[nov[0] :]]
        dF = get_dF_u(no, V, C, Q, u)
        dmoe = get_dmoe_F_u(C, dF)
        dDiajb = get_Diajb_u(dmoe, no)
        U = get_full_u_F_u(no, C, moe, dF, u)
        dVovov = get_dVovov_u(no, V, C, U)
        dt2 = [(dVovov[s] - t2[s] * dDiajb[s]) / Diajb[s] for s in range(3)]

        P = get_Pmp2_u(t2, t2)
        for s in [0, 1]:
            P[s] += Phf[s]
        dP = [U[s] @ P[s] - P[s] @ U[s] for s in [0, 1]]

        dP2 = get_Pmp2_u(dt2, t2)
        for s in [0, 1]:
            dP2[s] += dP2[s].T

        dPs[iQ] = [dP[s] + dP2[s] for s in [0, 1]]

        iQ += 1

    if aorep:
        for iQ in range(nQ):
            for s in [0, 1]:
                dPs[iQ][s] = C[s] @ dPs[iQ][s] @ C[s].T

    return dPs
