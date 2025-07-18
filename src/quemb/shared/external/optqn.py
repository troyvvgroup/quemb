# Author(s): Hong-Zhou Ye
#            Oinam Romesh Meitei
#            Minsik Cho
# NOTICE: The following code is mostly written by Hong-Zhou Ye
#   (except for the trust region routine)
#         The code has been slightly modified.
#
import logging
from collections.abc import Sequence

from numpy import array, empty, float64, outer, zeros
from numpy.linalg import inv, norm, pinv

from quemb.kbe.pfrag import Frags as pFrags
from quemb.molbe.helper import get_eri, get_scfObj
from quemb.molbe.pfrag import Frags
from quemb.shared.external.cphf_utils import cphf_kernel_batch, get_rhf_dP_from_u
from quemb.shared.external.cpmp2_utils import get_dPmp2_batch_r
from quemb.shared.external.jac_utils import get_dPccsdurlx_batch_u
from quemb.shared.typing import GlobalAOIdx, Matrix, RelAOIdx, SeqOverEdge

logger = logging.getLogger(__name__)


def line_search_LF(func, xold, fold, dx, iter_):
    """Adapted from D.-H. Li and M. Fukushima, Optimization Metheods and Software,
    13, 181 (2000)"""
    beta = 0.1
    rho = 0.9
    sigma1 = 1e-3
    sigma2 = 1e-3
    eta = (iter_ + 1) ** -2.0

    xk = xold + dx
    lcout = 0

    fk = func(xk)
    lcout += 1

    norm_dx = norm(dx)
    norm_fk = norm(fk)
    norm_fold = norm(fold)
    alp = 1.0

    if norm_fk > rho * norm_fold - sigma2 * norm_dx**2.0:
        while norm_fk > (1.0 + eta) * norm_fold - sigma1 * alp**2.0 * norm_dx**2.0:
            alp *= beta
            xk = xold + alp * dx

            fk = func(xk)

            lcout += 1
            norm_fk = norm(fk)
            if lcout == 20:
                break

    print(" No. of line search steps in QN opt :", lcout, flush=True)
    print(flush=True)
    return alp, xk, fk


def trustRegion(func, xold, fold, Binv, c=0.5):
    """Perform Trust Region Optimization.

    See "A Broyden Trust Region Quasi-Newton Method
    for Nonlinear Equations" (https://www.iaeng.org/IJCS/issues_v46/issue_3/IJCS_46_3_09.pdf)"
    Algorithm 1 for more information

    Parameters
    ----------
    func : typing.Callable
        Cost function
    xold : list or numpy.ndarray
        Current x_p (potentials in BE optimization)
    fold : list or numpy.ndarray
        Current f(x_p) (error vector)
    Binv : numpy.ndarray
        Inverse of Jacobian approximate (B^{-1}); This is updated in Broyden's Method
        through Sherman-Morrison formula
    c : float, optional
        Initial value of trust radius ∈ (0, 1), by default 0.5

    Returns
    -------
    xnew, fnew: tuple
        x_{p+1} and f_{p+1}. These values are used to proceed with Broyden's Method.
    """
    # c := initial trust radius (trust_radius = c^p)
    microiter = 0  # p
    rho = 0.001  # Threshold for trust region subproblem
    ratio = 0  # Initial r
    B = inv(Binv)  # approx Jacobian
    # dx_gn = - Binv@fold
    dx_gn = -(Binv @ Binv.T) @ B.T @ fold
    dx_sd = -B.T @ fold  # Steepest Descent step
    t = norm(dx_sd) ** 2 / norm(B @ dx_sd) ** 2
    prevdx = None
    ared = 0.0  # Reduction in the objective function; Initialize value to 0
    while ratio < rho or ared < 0.0:
        # Trust Region subproblem
        # minimize (1/2) ||F_k + B_k d||^2 w.r.t. d, s.t. d w/i trust radius
        # to pick the optimal direction using dog leg method
        if norm(dx_gn) < max(1.0, norm(xold)) * (
            c**microiter
        ):  # Gauss-Newton step within the trust radius
            print(
                "  Trust Region Optimization Step ",
                microiter,
                ": Gauss-Newton",
                flush=True,
            )
            dx = dx_gn
        elif t * norm(dx_sd) > max(1.0, norm(xold)) * (
            c**microiter
        ):  # GN step outside, SD step also outside
            print(
                "  Trust Region Optimization Step ",
                microiter,
                ": Steepest Descent",
                flush=True,
            )
            dx = (c**microiter) / norm(dx_sd) * dx_sd
        else:  # GN step outside, SD step inside (dog leg step)
            # dx := t*dx_sd + s (dx_gn - t*dx_sd) s.t. ||dx|| = c^p
            print(
                "  Trust Region Optimization Step ", microiter, ": Dog Leg", flush=True
            )
            tdx_sd = t * dx_sd
            diff = dx_gn - tdx_sd
            # s = (-dx_sd.T@diff + sqrt((dx_sd.T@diff)**2 -
            #   norm(diff)**2*(norm(dx_sd)**2-(c ** microiter)**2)))
            #   / (norm(dx_sd))**2
            # s is largest value in [0, 1] s.t. ||dx|| \le trust radius
            s = 1
            dx = tdx_sd + s * diff
            while norm(dx) > c**microiter and s > 0:
                s -= 0.001
                dx = tdx_sd + s * diff
        if prevdx is None or not all(dx == prevdx):
            # Actual Reduction := f(x_k) - f(x_k + dx)
            fnew = func(xold + dx)
            ared = 0.5 * (norm(fold) ** 2 - norm(fnew) ** 2)
            # Predicted Reduction := q(0) - q(dx) where q = (1/2) ||F_k + B_k d||^2
            pred = 0.5 * (norm(fold) ** 2 - norm(fold + B @ dx) ** 2)
        # Trust Region convergence criteria
        # r = ared/pred \le rho
        ratio = ared / pred
        microiter += 1
        if prevdx is None or not all(dx == prevdx):
            logger.debug(f"    ||δx||: {norm(dx)}")
            logger.debug(
                f"    Reduction Ratio (Actual / Predicted): {ared} / {pred} = {ratio}"
            )
        prevdx = dx
    return xold + dx, fnew  # xnew


class FrankQN:
    """Quasi Newton Optimization

    Performs quasi newton optimization. Interfaces many functionalities of the
    frankestein code originally written by Hong-Zhou Ye


    """

    def __init__(self, func, x0, f0, J0, trust=0.5, max_space=500):
        self.x0 = x0
        self.n = x0.size
        self.f0 = f0
        self.func = func

        self.B0 = pinv(J0)

        self.tol_gmres = 1.0e-6
        self.xnew = None  # new errvec
        self.xold = None  # old errvec
        self.fnew = None  # new jacobian?
        self.fold = None  # old jacobian?
        self.max_subspace = max_space
        self.dxs = empty([self.max_subspace, self.n])
        self.fs = empty([self.max_subspace, self.n])
        self.us = empty([self.max_subspace, self.n])  # u_m = B_m @ f_m
        self.vs = empty([self.max_subspace, self.n])  # v_m = B_0 @ f_{m+1}
        self.B = None
        self.trust = trust

    def next_step(self, iter, trust_region=False):
        if iter == 0:
            self.xnew = self.x0
            self.fnew = self.func(self.xnew) if self.f0 is None else self.f0
            self.fs[0] = self.fnew.copy()
            self.us[0] = self.B0 @ self.fnew
            self.Binv = self.B0.copy()

        # Book keeping
        if not iter == 0:
            dx_i = self.xnew - self.xold
            df_i = self.fnew - self.fold

        self.xold = self.xnew.copy()
        self.fold = self.fnew.copy()

        if not iter == 0:
            tmp__ = outer(dx_i - self.Binv @ df_i, dx_i @ self.Binv) / (
                dx_i @ self.Binv @ df_i
            )
            self.Binv += tmp__

        if trust_region:
            self.xnew, self.fnew = trustRegion(
                self.func, self.xold, self.fold, self.Binv, c=self.trust
            )
        else:
            self.us[iter] = self.get_Bnfn(iter)

            _, self.xnew, self.fnew = line_search_LF(
                self.func, self.xold, self.fold, -self.us[iter], iter
            )

            # udpate vs, dxs, and fs
            self.vs[iter] = self.B0 @ self.fnew
        self.dxs[iter] = self.xnew - self.xold
        if iter + 1 < self.max_subspace:
            self.fs[iter + 1] = self.fnew.copy()
        else:
            print("Reached the maximum number of iterations:", self.max_subspace)

    def get_Bnfn(self, n):
        # self.us; self.dxs; self.vs
        if n == 0:
            return self.us[0]

        vs = [None] * n
        for i in range(n):
            vs[i] = self.vs[n - i - 1]
        for i in range(1, n + 1):
            un_ = self.us[i - 1]
            dxn_ = self.dxs[i - 1]
            vps = [None] * (n - i + 1)
            for j in range(n - i + 1):
                a = vs[j]
                b = vs[n - i] - un_

                vps[j] = a + (dxn_ @ a) / (dxn_ @ b) * (dxn_ - b)

            vs = vps

        return vs[0]


def get_be_error_jacobian(n_frag, Fobjs, jac_solver="HF"):
    Jes = [None] * n_frag
    Jcs = [None] * n_frag
    xes = [None] * n_frag
    xcs = [None] * n_frag
    ys = [None] * n_frag
    alphas = [None] * n_frag

    if jac_solver.upper() == "MP2":
        res_func = mp2res_func
    elif jac_solver.upper() == "CCSD":
        res_func = ccsdres_func
    elif jac_solver.upper() == "HF":
        res_func = hfres_func
    else:
        raise NotImplementedError("Jacobian solver input not implemented")

    Ncout = [None] * n_frag
    for A in range(n_frag):
        Jes[A], Jcs[A], xes[A], xcs[A], ys[A], alphas[A], Ncout[A] = (
            get_atbe_Jblock_frag(Fobjs[A], res_func)
        )

    alpha = sum(alphas)

    # build Jacobian
    """ ignore!
    F0-M1 F1-M2M2 F2-M3M3 F3-M4
       M1   M2  M2 M3  M3 M4
    M1 E0   C1-1
    M2 C0-0 E1  E1
    M2      E1  E1 C2-2
    M3      C1-1   E2  E2
    M3             E2  E2 C3-3
    M4             C2-2   E3
    """
    N_ = sum(Ncout)
    J = zeros((N_ + 1, N_ + 1))
    cout = 0

    for findx, fobj in enumerate(Fobjs):
        J[cout : Ncout[findx] + cout, cout : Ncout[findx] + cout] = Jes[findx]
        J[cout : Ncout[findx] + cout, N_:] = array(xes[findx]).reshape(-1, 1)
        J[N_:, cout : Ncout[findx] + cout] = ys[findx]

        coutc = 0
        coutc_ = 0
        for cindx, cens in enumerate(fobj.relAO_in_ref_per_edge):
            coutc += Jcs[fobj.ref_frag_idx_per_edge[cindx]].shape[0]
            start_ = sum(Ncout[: fobj.ref_frag_idx_per_edge[cindx]])
            end_ = start_ + Ncout[fobj.ref_frag_idx_per_edge[cindx]]
            J[cout + coutc_ : cout + coutc, start_:end_] += Jcs[
                fobj.ref_frag_idx_per_edge[cindx]
            ]
            J[cout + coutc_ : cout + coutc, N_:] += array(
                xcs[fobj.ref_frag_idx_per_edge[cindx]]
            ).reshape(-1, 1)
            coutc_ = coutc
        cout += Ncout[findx]
    J[N_:, N_:] = alpha

    return J


def get_atbe_Jblock_frag(
    fobj: Frags | pFrags, res_func
) -> tuple[Matrix[float64], Matrix[float64], list, list, list, float, int]:
    assert (
        fobj._mo_coeffs is not None
        and fobj.nsocc is not None
        and fobj.fock is not None
        and fobj.heff is not None
        and fobj.nao is not None
    )
    vpots = get_vpots_frag(fobj.nao, fobj.relAO_per_edge, fobj.AO_in_frag)
    eri_ = get_eri(fobj.dname, fobj.nao, eri_file=fobj.eri_file)
    dm0 = 2.0 * (fobj._mo_coeffs[:, : fobj.nsocc] @ fobj._mo_coeffs[:, : fobj.nsocc].T)
    mf_ = get_scfObj(fobj.fock + fobj.heff, eri_, fobj.nsocc, dm0=dm0)

    dPs, dP_mu = res_func(mf_, vpots, eri_, fobj.nsocc)

    Je = []
    Jc = []
    y = []
    xe = []
    xc = []
    cout = 0

    for edge in fobj.relAO_per_edge:
        for j_ in range(len(edge)):
            for k_ in range(len(edge)):
                if j_ > k_:
                    continue
                # response w.r.t matching pot
                # edges
                tmpje_ = []

                for edge_ in fobj.relAO_per_edge:
                    lene = len(edge_)

                    for j__ in range(lene):
                        for k__ in range(lene):
                            if j__ > k__:
                                continue

                            tmpje_.append(dPs[cout][edge_[j__], edge_[k__]])
                y_ = 0.0
                for fidx, fval in enumerate(fobj.AO_in_frag):
                    if not any(fidx in sublist for sublist in fobj.relAO_per_edge):
                        y_ += dPs[cout][fidx, fidx]

                y.append(y_)

                tmpjc_ = []
                # center on the same fragment
                # for cen in fobj.efac[1]:
                for j_relAO in fobj.relAO_per_origin:
                    for k_relAO in fobj.relAO_per_origin:
                        if j_relAO > k_relAO:
                            continue
                        tmpjc_.append(-dPs[cout][j_relAO, k_relAO])

                Je.append(tmpje_)

                Jc.append(tmpjc_)

                # response w.r.t. chem pot
                # edge
                xe.append(dP_mu[edge[j_], edge[k_]])
                cout += 1

    alpha = 0.0
    for fidx, _ in enumerate(fobj.AO_in_frag):
        if not any(fidx in sublist for sublist in fobj.relAO_per_edge):
            alpha += dP_mu[fidx, fidx]

    for j_relAO in fobj.relAO_per_origin:
        for k_relAO in fobj.relAO_per_origin:
            if j_relAO > k_relAO:
                continue
            xc.append(-dP_mu[j_relAO, k_relAO])

    return array(Je).T, array(Jc).T, xe, xc, y, alpha, cout


def get_be_error_jacobian_selffrag(self, jac_solver="HF"):
    Jes = [None] * self.Nfrag
    xes = [None] * self.Nfrag
    xcs = [None] * self.Nfrag
    ys = [None] * self.Nfrag
    alphas = [None] * self.Nfrag

    if jac_solver == "MP2":
        res_func = mp2res_func
    elif jac_solver == "CCSD":
        res_func = ccsdres_func
    elif jac_solver == "HF":
        res_func = hfres_func
    else:
        raise NotImplementedError("Jacobian solver option not implemented.")

    Jes, _, xes, xcs, ys, alphas, Ncout = get_atbe_Jblock_frag(self.Fobjs[0], res_func)

    N_ = Ncout
    J = zeros((N_ + 1, N_ + 1))

    J[:Ncout, :Ncout] = Jes
    J[:Ncout, N_:] = array(xes).reshape(-1, 1)
    J[N_:, :Ncout] = ys
    J[:Ncout, N_:] += array([*xcs, *xcs]).reshape(-1, 1)
    J[N_:, N_:] = alphas

    return J


def hfres_func(mf, vpots, eri, nsocc) -> tuple[list[Matrix[float64]], Matrix[float64]]:
    C = mf.mo_coeff
    moe = mf.mo_energy
    eri = mf._eri
    no = nsocc

    us = cphf_kernel_batch(C, moe, eri, no, vpots)
    dPs = [get_rhf_dP_from_u(C, no, us[I]) for I in range(len(vpots) - 1)]
    dP_mu = get_rhf_dP_from_u(C, no, us[-1])

    return dPs, dP_mu


def mp2res_func(mf, vpots, eri, nsocc):
    C = mf.mo_coeff
    moe = mf.mo_energy
    eri = mf._eri
    no = nsocc

    dPs_an = get_dPmp2_batch_r(C, moe, eri, no, vpots, aorep=True)
    dPs_an = array([dp_ * 0.5 for dp_ in dPs_an])
    dP_mu = dPs_an[-1]

    return dPs_an[:-1], dP_mu


def ccsdres_func(mf, vpots, eri, nsocc):
    C = mf.mo_coeff
    moe = mf.mo_energy
    eri = mf._eri
    no = nsocc

    dPs_an = get_dPccsdurlx_batch_u(C, moe, eri, no, vpots)

    dP_mu = dPs_an[-1]

    return dPs_an[:-1], dP_mu


def get_vpots_frag(
    nao: int,
    rel_AO_per_edge: SeqOverEdge[Sequence[RelAOIdx]],
    AO_in_frag: Sequence[GlobalAOIdx],
) -> list[Matrix[float64]]:
    vpots: list[Matrix[float64]] = []

    for edge_ in rel_AO_per_edge:
        lene = len(edge_)
        for j__ in range(lene):
            for k__ in range(lene):
                if j__ > k__:
                    continue

                tmppot = zeros((nao, nao))
                tmppot[edge_[j__], edge_[k__]] = tmppot[edge_[k__], edge_[j__]] = 1
                vpots.append(tmppot)

    # only the centers
    # outer edges not included
    tmppot = zeros((nao, nao))
    for fidx, fval in enumerate(AO_in_frag):
        if not any(fidx in sublist for sublist in rel_AO_per_edge):
            tmppot[fidx, fidx] = -1

    vpots.append(tmppot)
    return vpots
