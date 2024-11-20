# Author(s): Hong-Zhou Ye
# NOTICE: The following code is entirely written by Hong-Zhou Ye.
#         The code has been slightly modified.
#

import numpy, functools, sys, time, os


def make_rdm1_ccsd_t1(t1):
    nocc, nvir = t1.shape
    nmo = nocc + nvir
    dm = numpy.zeros((nmo, nmo), dtype=t1.dtype)
    dm[:nocc, nocc:] = t1
    dm[nocc:, :nocc] = t1.T
    dm[numpy.diag_indices(nocc)] += 2.0

    return dm


def make_rdm2_urlx(t1, t2, with_dm1=True):
    nocc, nvir = t1.shape
    nmo = nocc + nvir

    goovv = (numpy.einsum("ia,jb->ijab", t1, t1) + t2) * 0.5
    dovov = goovv.transpose(0, 2, 1, 3) * 2 - goovv.transpose(1, 2, 0, 3)

    dm2 = numpy.zeros([nmo, nmo, nmo, nmo], dtype=t1.dtype)

    dovov = numpy.asarray(dovov)
    dm2[:nocc, nocc:, :nocc, nocc:] = dovov
    dm2[:nocc, nocc:, :nocc, nocc:] += dovov.transpose(2, 3, 0, 1)
    dm2[nocc:, :nocc, nocc:, :nocc] = (
        dm2[:nocc, nocc:, :nocc, nocc:].transpose(1, 0, 3, 2).conj()
    )
    dovov = None

    if with_dm1:
        dm1 = make_rdm1_ccsd_t1(t1)
        dm1[numpy.diag_indices(nocc)] -= 2

        for i in range(nocc):
            dm2[i, i, :, :] += dm1 * 2
            dm2[:, :, i, i] += dm1 * 2
            dm2[:, i, i, :] -= dm1
            dm2[i, :, :, i] -= dm1.T

        for i in range(nocc):
            for j in range(nocc):
                dm2[i, i, j, j] += 4
                dm2[i, j, j, i] -= 2

    return dm2


def make_rdm1_uccsd(ucc, relax=False):
    from pyscf.cc.uccsd_rdm import make_rdm1

    if relax == True:
        rdm1 = make_rdm1(ucc, ucc.t1, ucc.t2, ucc.l1, ucc.l2)
    else:
        l1 = [numpy.zeros_like(ucc.t1[s]) for s in [0, 1]]
        l2 = [numpy.zeros_like(ucc.t2[s]) for s in [0, 1, 2]]
        rdm1 = make_rdm1(ucc, ucc.t1, ucc.t2, l1, l2)
    return rdm1


def make_rdm2_uccsd(ucc, relax=False, with_dm1=True):
    from pyscf.cc.uccsd_rdm import make_rdm2

    if relax == True:
        rdm2 = make_rdm2(ucc, ucc.t1, ucc.t2, ucc.l1, ucc.l2, with_dm1=with_dm1)
    else:
        l1 = [numpy.zeros_like(ucc.t1[s]) for s in [0, 1]]
        l2 = [numpy.zeros_like(ucc.t2[s]) for s in [0, 1, 2]]
        rdm2 = make_rdm2(ucc, ucc.t1, ucc.t2, l1, l2, with_dm1=with_dm1)
    return rdm2
