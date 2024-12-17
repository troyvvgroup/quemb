# Author(s): Henry Tran, Oinam Meitei, Shaun Weatherly
#

import numpy
from numpy.linalg import eigh, inv, multi_dot, norm, svd
from pyscf.gto import intor_cross

from quemb.shared.external.lo_helper import (
    get_aoind_by_atom,
    reorder_by_atom_,
)
from quemb.shared.helper import ncore_, unused


def dot_gen(A, B, ovlp):
    return A.T @ B if ovlp is None else A.T @ ovlp @ B


def get_cano_orth_mat(A, thr=1.0e-6, ovlp=None):
    S = dot_gen(A, A, ovlp)
    e, u = eigh(S)
    if thr > 0:
        idx_keep = e / e[-1] > thr
    else:
        idx_keep = list(range(e.shape[0]))
    U = u[:, idx_keep] * e[idx_keep] ** -0.5
    return U


def cano_orth(A, thr=1.0e-6, ovlp=None):
    """Canonically orthogonalize columns of A"""
    return A @ get_cano_orth_mat(A, thr, ovlp)


def get_symm_orth_mat(A, thr=1.0e-6, ovlp=None):
    S = dot_gen(A, A, ovlp)
    e, u = eigh(S)
    if int(numpy.sum(e < thr)) > 0:
        raise ValueError(
            "Linear dependence is detected in the column space of A: "
            "smallest eigenvalue (%.3E) is less than thr (%.3E). "
            "Please use 'cano_orth' instead." % (numpy.min(e), thr)
        )
    return u @ numpy.diag(e**-0.5) @ u.T


def symm_orth(A, thr=1.0e-6, ovlp=None):
    """Symmetrically orthogonalize columns of A"""
    return A @ get_symm_orth_mat(A, thr, ovlp)


def remove_core_mo(Clo, Ccore, S, thr=0.5):
    assert numpy.allclose(Clo.T @ S @ Clo, numpy.eye(Clo.shape[1]))
    assert numpy.allclose(Ccore.T @ S @ Ccore, numpy.eye(Ccore.shape[1]))

    n, nlo = Clo.shape
    ncore = Ccore.shape[1]
    Pcore = Ccore @ Ccore.T @ S
    Clo1 = (numpy.eye(n) - Pcore) @ Clo
    pop = numpy.diag(Clo1.T @ S @ Clo1)
    idx_keep = numpy.where(pop > thr)[0]
    assert len(idx_keep) == nlo - ncore
    return symm_orth(Clo1[:, idx_keep], ovlp=S)


def get_xovlp(mol, basis="sto-3g"):
    """Gets set of valence orbitals based on smaller (should be minimal) basis

    Parameters
    ----------
    mol : pyscf.gto.mole.Mole
        just need it for the working basis
    basis : str
        the IAO basis, Knizia recommended 'minao'

    Returns
    ------
    S12 : numpy.ndarray
        Overlap of two basis sets
    S22 : numpy.ndarray
        Overlap in new basis set
    """
    mol_alt = mol.copy()
    mol_alt.basis = basis
    mol_alt.build()

    S12 = intor_cross("int1e_ovlp", mol, mol_alt)
    S22 = mol_alt.intor("int1e_ovlp")

    return S12, S22


def get_iao(Co, S12, S1, S2=None):
    """

    Parameters
    ----------
    Co:
        occupied coefficient matrix with core
    p:
        valence AO matrix in AO
    no:
        number of occ orbitals
    S12:
        ovlp between working basis and valence basis
        can be thought of as working basis in valence basis
    S1:
        ao ovlp matrix
    S2:
        valence AO ovlp
    """
    # define projection operators
    n = Co.shape[0]
    if S2 is None:
        S2 = S12.T @ inv(S1) @ S12
    P1 = inv(S1)
    P2 = inv(S2)

    # depolarized occ mo
    Cotil = P1 @ S12 @ P2 @ S12.T @ Co

    # repolarized valence AOs
    ptil = P1 @ S12
    Stil = Cotil.T @ S1 @ Cotil

    Po = Co @ Co.T
    Potil = Cotil @ inv(Stil) @ Cotil.T

    Ciao = (numpy.eye(n) - (Po + Potil - 2 * Po @ S1 @ Potil) @ S1) @ ptil
    Ciao = symm_orth(Ciao, ovlp=S1)

    # check span
    rep_err = norm(Ciao @ Ciao.T @ S1 @ Po - Po)
    if rep_err > 1.0e-10:
        raise RuntimeError
    return Ciao


def get_pao(Ciao, S, S12):
    """
    Parameters
    ----------
    Ciao: numpy.ndarray
        output of :func:`get_iao`
    S: numpy.ndarray
        ao ovlp matrix
    S12: numpy.ndarray
        valence orbitals projected into ao basis
    Returns
    -------
    Cpao: numpy.ndarray
        (orthogonalized)
    """
    n = Ciao.shape[0]
    s12 = inv(S) @ S12
    nonval = (
        numpy.eye(n) - s12 @ s12.T
    )  # set of orbitals minus valence (orth in working basis)

    Piao = Ciao @ Ciao.T @ S  # projector into IAOs
    Cpao = (numpy.eye(n) - Piao) @ nonval  # project out IAOs from non-valence basis

    # begin canonical orthogonalization to get rid of redundant orbitals
    Cpao = cano_orth(Cpao, ovlp=S)

    return Cpao


def get_pao_native(Ciao, S, mol, valence_basis):
    """

    Parameters
    ----------
    Ciao:
        output of :code:`get_iao_native`
    S:
        ao ovlp matrix
    mol:
        mol object
    valence_basis:
        basis used for valence orbitals
    Returns
    -------
    Cpao: numpy.ndarray
        (symmetrically orthogonalized)

    """
    n = Ciao.shape[0]

    # Form a mol object with the valence basis for the ao_labels
    mol_alt = mol.copy()
    mol_alt.basis = valence_basis
    mol_alt.build()

    full_ao_labels = mol.ao_labels()
    valence_ao_labels = mol_alt.ao_labels()

    vir_idx = [
        idx
        for idx, label in enumerate(full_ao_labels)
        if (label not in valence_ao_labels)
    ]

    Piao = Ciao @ Ciao.T @ S
    Cpao = (numpy.eye(n) - Piao)[:, vir_idx]

    try:
        Cpao = symm_orth(Cpao, ovlp=S)
    except ValueError:
        print("Symm orth PAO failed. Switch to cano orth", flush=True)
        npao0 = Cpao.shape[1]
        Cpao = cano_orth(Cpao, ovlp=S)
        npao1 = Cpao.shape[1]
        print("# of PAO: %d --> %d" % (npao0, npao1), flush=True)
        print("", flush=True)

    return Cpao


def get_loc(mol, C, method, pop_method=None, init_guess=None):
    if method.upper() == "ER":
        from pyscf.lo import ER as Localizer  # noqa: PLC0415
    elif method.upper() == "PM":
        from pyscf.lo import PM as Localizer  # noqa: PLC0415
    elif method.upper() == "FB" or method.upper() == "BOYS":
        from pyscf.lo import Boys as Localizer  # noqa: PLC0415
    else:
        raise NotImplementedError("Localization scheme not understood")

    mlo = Localizer(mol, C)
    if pop_method is not None:
        mlo.pop_method = str(pop_method)

    mlo.init_guess = init_guess
    C_ = mlo.kernel()

    return C_


class MixinLocalize:
    def localize(
        self,
        lo_method,
        valence_basis="sto-3g",
        hstack=False,
        pop_method=None,
        init_guess=None,
        valence_only=False,
        nosave=False,
    ):
        """Molecular orbital localization

        Performs molecular orbital localization computations. For large basis,
        IAO is recommended augmented with PAO orbitals.

        Parameters
        ----------
        lo_method : str
            Localization method in quantum chemistry. 'lowdin', 'boys', and 'iao'
            are supported.
        valence_basis : str
            Name of minimal basis set for IAO scheme. 'sto-3g' suffice for most cases.
        valence_only : bool
            If this option is set to True, all calculation will be performed in the
            valence basis in the IAO partitioning.
            This is an experimental feature.
        """
        if lo_method == "lowdin":
            es_, vs_ = eigh(self.S)
            edx = es_ > 1.0e-15
            self.W = numpy.dot(vs_[:, edx] / numpy.sqrt(es_[edx]), vs_[:, edx].T)
            if self.frozen_core:
                if self.unrestricted:
                    P_core = [
                        numpy.eye(self.W.shape[0]) - numpy.dot(self.P_core[s], self.S)
                        for s in [0, 1]
                    ]
                    C_ = numpy.dot(P_core, self.W)
                    Cpop = [multi_dot((C_[s].T, self.S, C_[s])) for s in [0, 1]]
                    Cpop = [numpy.diag(Cpop[s]) for s in [0, 1]]
                    no_core_idx = [numpy.where(Cpop[s] > 0.7)[0] for s in [0, 1]]
                    C_ = [C_[s][:, no_core_idx[s]] for s in [0, 1]]
                    S_ = [multi_dot((C_[s].T, self.S, C_[s])) for s in [0, 1]]
                    W_ = []
                    for s in [0, 1]:
                        es_, vs_ = eigh(S_[s])
                        s_ = numpy.sqrt(es_)
                        s_ = numpy.diag(1.0 / s_)
                        W_.append(multi_dot((vs_, s_, vs_.T)))
                    self.W = [numpy.dot(C_[s], W_[s]) for s in [0, 1]]
                else:
                    P_core = numpy.eye(self.W.shape[0]) - numpy.dot(self.P_core, self.S)
                    C_ = numpy.dot(P_core, self.W)
                    # NOTE: PYSCF has basis in 1s2s3s2p2p2p3p3p3p format
                    # fix no_core_idx - use population for now
                    Cpop = multi_dot((C_.T, self.S, C_))
                    Cpop = numpy.diag(Cpop)
                    no_core_idx = numpy.where(Cpop > 0.7)[0]
                    C_ = C_[:, no_core_idx]
                    S_ = multi_dot((C_.T, self.S, C_))
                    es_, vs_ = eigh(S_)
                    s_ = numpy.sqrt(es_)
                    s_ = numpy.diag(1.0 / s_)
                    W_ = multi_dot((vs_, s_, vs_.T))
                    self.W = numpy.dot(C_, W_)

            if self.unrestricted:
                if self.frozen_core:
                    self.lmo_coeff_a = multi_dot(
                        (self.W[0].T, self.S, self.C_a[:, self.ncore :])
                    )
                    self.lmo_coeff_b = multi_dot(
                        (self.W[1].T, self.S, self.C_b[:, self.ncore :])
                    )
                else:
                    self.lmo_coeff_a = multi_dot((self.W.T, self.S, self.C_a))
                    self.lmo_coeff_b = multi_dot((self.W.T, self.S, self.C_b))
            else:
                if self.frozen_core:
                    self.lmo_coeff = multi_dot(
                        (self.W.T, self.S, self.C[:, self.ncore :])
                    )
                else:
                    self.lmo_coeff = multi_dot((self.W.T, self.S, self.C))

        elif lo_method in ["pipek-mezey", "pipek", "PM"]:
            es_, vs_ = eigh(self.S)
            edx = es_ > 1.0e-15
            self.W = numpy.dot(vs_[:, edx] / numpy.sqrt(es_[edx]), vs_[:, edx].T)

            es_, vs_ = eigh(self.S)
            edx = es_ > 1.0e-15
            W_ = numpy.dot(vs_[:, edx] / numpy.sqrt(es_[edx]), vs_[:, edx].T)
            if self.frozen_core:
                P_core = numpy.eye(W_.shape[0]) - numpy.dot(self.P_core, self.S)
                C_ = numpy.dot(P_core, W_)
                Cpop = multi_dot((C_.T, self.S, C_))
                Cpop = numpy.diag(Cpop)
                no_core_idx = numpy.where(Cpop > 0.55)[0]
                C_ = C_[:, no_core_idx]
                S_ = multi_dot((C_.T, self.S, C_))
                es_, vs_ = eigh(S_)
                s_ = numpy.sqrt(es_)
                s_ = numpy.diag(1.0 / s_)
                W_ = multi_dot((vs_, s_, vs_.T))
                W_ = numpy.dot(C_, W_)

            self.W = get_loc(
                self.mol, W_, "PM", pop_method=pop_method, init_guess=init_guess
            )

            if not self.frozen_core:
                self.lmo_coeff = self.W.T @ self.S @ self.C
            else:
                self.lmo_coeff = self.W.T @ self.S @ self.C[:, self.ncore :]

        elif lo_method == "iao":
            loc_type = "SO"

            # Occupied mo_coeff (with core)
            Co = self.C[:, : self.Nocc]
            # Get necessary overlaps, second arg is IAO basis
            S12, S2 = get_xovlp(self.mol, basis=valence_basis)
            # Use these to get IAOs
            Ciao = get_iao(Co, S12, self.S, S2=S2)

            if not valence_only:
                # Now get PAOs
                if loc_type.upper() != "SO":
                    Cpao = get_pao(Ciao, self.S, S12)
                elif loc_type.upper() == "SO":
                    Cpao = get_pao_native(
                        Ciao, self.S, self.mol, valence_basis=valence_basis
                    )

            # rearrange by atom
            aoind_by_atom = get_aoind_by_atom(self.mol)
            Ciao, iaoind_by_atom = reorder_by_atom_(Ciao, aoind_by_atom, self.S)

            if not valence_only:
                Cpao, paoind_by_atom = reorder_by_atom_(Cpao, aoind_by_atom, self.S)

            if self.frozen_core:
                # Remove core MOs
                Cc = self.C[:, : self.ncore]  # Assumes core are first
                Ciao = remove_core_mo(Ciao, Cc, self.S)

            # Localize orbitals beyond symm orth
            if loc_type.upper() != "SO":
                Ciao = get_loc(self.mol, Ciao, loc_type)
                if not valence_only:
                    Cpao = get_loc(self.mol, Cpao, loc_type)

            shift = 0
            ncore = 0
            if not valence_only:
                Wstack = numpy.zeros(
                    (Ciao.shape[0], Ciao.shape[1] + Cpao.shape[1])
                )  # -self.ncore))
            else:
                Wstack = numpy.zeros((Ciao.shape[0], Ciao.shape[1]))

            if self.frozen_core:
                for ix in range(self.mol.natm):
                    nc = ncore_(self.mol.atom_charge(ix))
                    ncore += nc
                    niao = len(iaoind_by_atom[ix])
                    iaoind_ix = [i_ - ncore for i_ in iaoind_by_atom[ix][nc:]]
                    Wstack[:, shift : shift + niao - nc] = Ciao[:, iaoind_ix]
                    shift += niao - nc
                    if not valence_only:
                        npao = len(paoind_by_atom[ix])
                        Wstack[:, shift : shift + npao] = Cpao[:, paoind_by_atom[ix]]
                        shift += npao
            else:
                if not hstack:
                    for ix in range(self.mol.natm):
                        niao = len(iaoind_by_atom[ix])
                        Wstack[:, shift : shift + niao] = Ciao[:, iaoind_by_atom[ix]]
                        shift += niao
                        if not valence_only:
                            npao = len(paoind_by_atom[ix])
                            Wstack[:, shift : shift + npao] = Cpao[
                                :, paoind_by_atom[ix]
                            ]
                            shift += npao
                else:
                    Wstack = numpy.hstack((Ciao, Cpao))
            if not nosave:
                self.W = Wstack
                assert numpy.allclose(
                    self.W.T @ self.S @ self.W, numpy.eye(self.W.shape[1])
                )
            else:
                assert numpy.allclose(
                    Wstack.T @ self.S @ Wstack, numpy.eye(Wstack.shape[1])
                )
                return Wstack
            nmo = self.C.shape[1] - self.ncore
            nlo = self.W.shape[1]

            if not valence_only:
                if nmo > nlo:
                    Co_nocore = self.C[:, self.ncore : self.Nocc]
                    Cv = self.C[:, self.Nocc :]
                    # Ensure that the LOs span the occupied space
                    assert numpy.allclose(
                        numpy.sum((self.W.T @ self.S @ Co_nocore) ** 2.0),
                        self.Nocc - self.ncore,
                    )
                    # Find virtual orbitals that lie in the span of LOs
                    u, l, vt = svd(self.W.T @ self.S @ Cv, full_matrices=False)
                    unused(u)
                    nvlo = nlo - self.Nocc - self.ncore
                    assert numpy.allclose(numpy.sum(l[:nvlo]), nvlo)
                    C_ = numpy.hstack([Co_nocore, Cv @ vt[:nvlo].T])
                    self.lmo_coeff = self.W.T @ self.S @ C_
                else:
                    self.lmo_coeff = self.W.T @ self.S @ self.C[:, self.ncore :]
            else:
                self.lmo_coeff = self.W.T @ self.S @ self.C[:, self.ncore :]

        elif lo_method == "boys":
            es_, vs_ = eigh(self.S)
            edx = es_ > 1.0e-15
            W_ = numpy.dot(vs_[:, edx] / numpy.sqrt(es_[edx]), vs_[:, edx].T)
            if self.frozen_core:
                P_core = numpy.eye(W_.shape[0]) - numpy.dot(self.P_core, self.S)
                C_ = numpy.dot(P_core, W_)
                Cpop = multi_dot((C_.T, self.S, C_))
                Cpop = numpy.diag(Cpop)
                no_core_idx = numpy.where(Cpop > 0.55)[0]
                C_ = C_[:, no_core_idx]
                S_ = multi_dot((C_.T, self.S, C_))
                es_, vs_ = eigh(S_)
                s_ = numpy.sqrt(es_)
                s_ = numpy.diag(1.0 / s_)
                W_ = multi_dot((vs_, s_, vs_.T))
                W_ = numpy.dot(C_, W_)

            self.W = get_loc(self.mol, W_, "BOYS")

            if not self.frozen_core:
                self.lmo_coeff = self.W.T @ self.S @ self.C
            else:
                self.lmo_coeff = self.W.T @ self.S @ self.C[:, self.ncore :]

        else:
            raise ValueError(f"lo_method = {lo_method} not implemented!")
