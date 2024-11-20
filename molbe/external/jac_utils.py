# Author(s): Hong-Zhou Ye
# NOTICE: The following code is entirely written by Hong-Zhou Ye.
#         The code has been slightly modified.
#

import numpy as np

from pyscf import ao2mo


""" Derivative of approximate t1 amplitudes
t_ia = ((2*t2-t2)_ibjc g_cjba - g_ikbj (2*t2-t2)_jbka) / (e_i - e_a)
This approximate t1 is by substituting the MP2 t2 amplitudes into the CCSD equation and run it for one cycle.
"""


def get_t1(no, nv, moe, Vovov, Voovo, Vvovv):
    assert moe.size == no + nv
    assert Vovov.shape == (no, nv, no, nv)
    assert Voovo.shape == (no, no, nv, no)
    assert Vvovv.shape == (nv, no, nv, nv)
    eia = moe[:no].reshape(-1, 1) - moe[no:]
    eia_ = eia.ravel()
    eiajb = (eia_.reshape(-1, 1) + eia_).reshape(no, nv, no, nv)
    t2 = Vovov / eiajb
    t1approx = (
        2.0 * np.einsum("ibjc,cjba->ia", t2, Vvovv, optimize=True)
        - np.einsum("jbic,cjba->ia", t2, Vvovv, optimize=True)
        - 2.0 * np.einsum("ikbj,jbka->ia", Voovo, t2, optimize=True)
        + np.einsum("ikbj,kbja->ia", Voovo, t2, optimize=True)
    ) / eia

    return t1approx


def get_Vmogen_r(no, V, C, pattern):
    assert set(pattern) == set("ov")
    n = C.shape[0]
    nv = n - no
    nov = {"o": no, "v": nv}
    Co = C[:, :no]
    Cv = C[:, no:]
    Cov = {"o": Co, "v": Cv}

    Cs = []
    shape = []
    for p in pattern:
        Cs += [Cov[p]]
        shape += [nov[p]]

    return ao2mo.incore.general(V, Cs, compact=False).reshape(*shape)


def get_dVmogen_r(no, V, C, u, pattern):
    assert set(pattern) == set("ov")
    n = C.shape[0]
    nv = n - no
    nov = {"o": no, "v": nv}
    Co = C[:, :no]
    Cv = C[:, no:]
    Cov = {"o": Co, "v": Cv}
    if u.size == no * nv:
        uov = u.reshape(no, nv)
        dCv = Co @ uov
        dCo = -Cv @ uov.T
    else:
        dC = C @ u
        dCo = dC[:, :no]
        dCv = dC[:, no:]
    dCov = {"o": dCo, "v": dCv}
    # take care of symmetry
    p12 = pattern[:2]
    p34 = pattern[2:]

    def xform_1_index(ip0):
        Cs = []
        for ip, p in enumerate(pattern):
            Cs += [dCov[p]] if ip == ip0 else [Cov[p]]
        shape = tuple([nov[p] for p in pattern])
        return ao2mo.incore.general(V, Cs, compact=False).reshape(*shape)

    # xform ij
    dV = xform_1_index(0) + xform_1_index(1)
    # xform kl
    if p12 == p34:
        dV += dV.transpose(2, 3, 0, 1)
    else:
        dV += xform_1_index(2) + xform_1_index(3)

    return dV


def get_dt1ao_an(no, V, C, moe, Qs, us=None):
    n = C.shape[0]
    nv = n - no
    Co = C[:, :no]
    Cv = C[:, no:]

    def get_Dia_r(moe_, no_):
        return moe_[:no_].reshape(-1, 1) - moe_[no_:]

    # prepare integrals
    from .cpmp2_utils import get_Diajb_r

    Vovov = get_Vmogen_r(no, V, C, "ovov")
    Vvovv = get_Vmogen_r(no, V, C, "vovv")
    Voovo = get_Vmogen_r(no, V, C, "oovo")
    eov = get_Dia_r(moe, no)
    eovov = get_Diajb_r(moe, no)
    t2 = Vovov / eovov
    t1 = get_t1(no, nv, moe, Vovov, Voovo, Vvovv)

    # solve CPHF get u
    if us is None:
        from .cphf_utils import cphf_kernel_batch

        us = cphf_kernel_batch(C, moe, V, no, Qs)

    dt1s = [None] * len(Qs)
    iu = 0
    for u, Q in zip(us, Qs):
        # get A
        from .cpmp2_utils import get_dF_r

        A = -get_dF_r(no, V, C, Q, u)
        Aoo = Co.T @ A @ Co
        Avv = Cv.T @ A @ Cv

        # get tA
        tA = np.einsum("lajb,li->iajb", t2, Aoo, optimize=True) - np.einsum(
            "idjb,da->iajb", t2, Avv, optimize=True
        )
        tA += tA.transpose(2, 3, 0, 1)

        # get VU
        dVovov = get_dVmogen_r(no, V, C, u, "ovov")
        dVvovv = get_dVmogen_r(no, V, C, u, "vovv")
        dVoovo = get_dVmogen_r(no, V, C, u, "oovo")

        # get dmoe and deov
        from .cpmp2_utils import get_dmoe_F_r

        dmoe = get_dmoe_F_r(C, -A)
        deov = get_Dia_r(dmoe, no)

        # get dCov
        uov = u.reshape(no, nv)
        dCo = -Cv @ uov.T
        dCv = Co @ uov

        # show time
        dt1 = (
            Co
            @ (
                get_t1(no, nv, moe, tA, Voovo, Vvovv)
                + get_t1(no, nv, moe, dVovov, Voovo, Vvovv)
                + get_t1(no, nv, moe, Vovov, dVoovo, dVvovv)
                + (Aoo @ t1 - t1 @ Avv) / eov
            )
            @ Cv.T
        )
        dt1 += dCo @ t1 @ Cv.T + Co @ t1 @ dCv.T
        dt1 += dt1.T

        dt1s[iu] = dt1

        iu += 1

    return dt1s


def get_dPccsdurlx_batch_u(C, moe, eri, no, vpots):
    # cphf
    from .cphf_utils import cphf_kernel_batch

    us = cphf_kernel_batch(C, moe, eri, no, vpots)
    # get mp2 part
    dPt1aos = get_dt1ao_an(no, eri, C, moe, vpots, us=us)
    # HF contribution
    n = C.shape[0]
    nv = n - no
    for iu, u in enumerate(us):
        Co = C[:, :no]
        Cv = C[:, no:]
        dCo = -Cv @ u.reshape(no, nv).T
        dPhfao = 2.0 * dCo @ Co.T
        dPhfao += dPhfao.T
        dPt1aos[iu] += dPhfao
    # total
    return dPt1aos
