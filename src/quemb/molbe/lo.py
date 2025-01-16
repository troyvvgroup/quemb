# Author(s): Henry Tran, Oinam Meitei, Shaun Weatherly
#

import numpy as np
from numpy import allclose, diag, eye, sqrt, where, zeros
from numpy.linalg import eigh, inv, multi_dot, norm, svd
from pyscf.gto import intor_cross
from pyscf.gto.mole import Mole

from quemb.shared.external.lo_helper import (
    get_aoind_by_atom,
    reorder_by_atom_,
)
from quemb.shared.helper import ncore_, unused
from quemb.shared.typing import Matrix, Tensor3D


def dot_gen(A: Matrix, B: Matrix, ovlp: Matrix | None = None) -> Matrix:
    return A.T @ B if ovlp is None else A.T @ ovlp @ B


def get_cano_orth_mat(
    A: Matrix, thr: float = 1.0e-6, ovlp: Matrix | None = None
) -> Matrix:
    S = dot_gen(A, A, ovlp)
    e, u = eigh(S)
    if thr > 0:
        idx_keep = e / e[-1] > thr
    else:
        idx_keep = slice(0, e.shape[0])
    return u[:, idx_keep] * e[idx_keep] ** -0.5


def cano_orth(A: Matrix, thr: float = 1.0e-6, ovlp: Matrix | None = None) -> Matrix:
    """Canonically orthogonalize columns of A"""
    return A @ get_cano_orth_mat(A, thr, ovlp)


def get_symm_orth_mat(
    A: Matrix, thr: float = 1.0e-6, ovlp: Matrix | None = None
) -> Matrix:
    S = dot_gen(A, A, ovlp)
    e, u = eigh(S)
    if (e < thr).any():
        raise ValueError(
            "Linear dependence is detected in the column space of A: "
            "smallest eigenvalue (%.3E) is less than thr (%.3E). "
            "Please use 'cano_orth' instead." % (np.min(e), thr)
        )
    return u @ diag(e**-0.5) @ u.T


def symm_orth(A: Matrix, thr: float = 1.0e-6, ovlp: Matrix | None = None) -> Matrix:
    """Symmetrically orthogonalize columns of A"""
    return A @ get_symm_orth_mat(A, thr, ovlp)


def remove_core_mo(Clo: Matrix, Ccore: Matrix, S: Matrix, thr: float = 0.5) -> Matrix:
    assert allclose(Clo.T @ S @ Clo, eye(Clo.shape[1]))
    assert allclose(Ccore.T @ S @ Ccore, eye(Ccore.shape[1]))

    n, nlo = Clo.shape
    ncore = Ccore.shape[1]
    Pcore = Ccore @ Ccore.T @ S
    Clo1 = (eye(n) - Pcore) @ Clo
    pop = diag(Clo1.T @ S @ Clo1)
    idx_keep = where(pop > thr)[0]
    assert len(idx_keep) == nlo - ncore
    return symm_orth(Clo1[:, idx_keep], ovlp=S)


def get_xovlp(
    mol: Mole, basis: str = "sto-3g"
) -> tuple[Matrix | Tensor3D, Matrix | Tensor3D]:
    """Gets set of valence orbitals based on smaller (should be minimal) basis

    Parameters
    ----------
    mol :
        just need it for the working basis
    basis :
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


def get_iao(
    Co: Matrix,
    S12: Matrix,
    S1: Matrix,
    S2: Matrix | None = None,
) -> Matrix:
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

    Ciao = (eye(n) - (Po + Potil - 2 * Po @ S1 @ Potil) @ S1) @ ptil
    Ciao = symm_orth(Ciao, ovlp=S1)

    # check span
    rep_err = norm(Ciao @ Ciao.T @ S1 @ Po - Po)
    if rep_err > 1.0e-10:
        raise RuntimeError
    return Ciao


def get_pao(Ciao: Matrix, S: Matrix, S12: Matrix) -> Matrix:
    """
    Parameters
    ----------
    Ciao:
        output of :func:`get_iao`
    S:
        ao ovlp matrix
    S12:
        valence orbitals projected into ao basis
    Returns
    -------
    Cpao: :class:`quemb.shared.typing.Matrix`
        (orthogonalized)
    """
    n = Ciao.shape[0]
    s12 = inv(S) @ S12
    nonval = (
        eye(n) - s12 @ s12.T
    )  # set of orbitals minus valence (orth in working basis)

    Piao = Ciao @ Ciao.T @ S  # projector into IAOs
    Cpao_redundant = (eye(n) - Piao) @ nonval  # project out IAOs from non-valence basis

    # begin canonical orthogonalization to get rid of redundant orbitals
    return cano_orth(Cpao_redundant, ovlp=S)


def get_pao_native(
    Ciao: Matrix,
    S: Matrix,
    mol: Mole,
    iao_valence_basis: str
) -> Matrix:
    """

    Parameters
    ----------
    Ciao:
        output of :code:`get_iao_native`
    S:
        ao ovlp matrix
    mol:
        mol object
    iao_valence_basis:
        basis used for valence orbitals
    Returns
    -------
    Cpao: :class:`quemb.shared.typing.Matrix`
        (symmetrically orthogonalized)

    """
    n = Ciao.shape[0]

    # Form a mol object with the valence basis for the ao_labels
    mol_alt = mol.copy()
    mol_alt.basis = iao_valence_basis
    mol_alt.build()

    full_ao_labels = mol.ao_labels()
    valence_ao_labels = mol_alt.ao_labels()

    vir_idx = [
        idx
        for idx, label in enumerate(full_ao_labels)
        if (label not in valence_ao_labels)
    ]

    Piao = Ciao @ Ciao.T @ S
    Cpao = (eye(n) - Piao)[:, vir_idx]

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


def get_loc(
    mol: Mole,
    C: Matrix,
    method: str,
    pop_method: str | None = None,
    init_guess: Matrix | None = None,
) -> Mole:
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
        mlo.pop_method = pop_method

    mlo.init_guess = init_guess
    return mlo.kernel()


class MixinLocalize:
    def localize(
        self,
        lo_method,
        iao_valence_basis="sto-3g",
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
        iao_valence_basis : str
            Name of minimal basis set for IAO scheme. 'sto-3g' suffice for most cases.
        valence_only : bool
            If this option is set to True, all calculation will be performed in the
            valence basis in the IAO partitioning.
            This is an experimental feature.
        """
        if lo_method == "lowdin":
            es_, vs_ = eigh(self.S)
            edx = es_ > 1.0e-15
            self.W = vs_[:, edx] / sqrt(es_[edx]) @ vs_[:, edx].T
            if self.frozen_core:
                if self.unrestricted:
                    P_core = [
                        eye(self.W.shape[0]) - (self.P_core[s] @ self.S) for s in [0, 1]
                    ]
                    C_ = P_core @ self.W
                    Cpop = [multi_dot((C_[s].T, self.S, C_[s])) for s in [0, 1]]
                    Cpop = [diag(Cpop[s]) for s in [0, 1]]
                    no_core_idx = [where(Cpop[s] > 0.7)[0] for s in [0, 1]]
                    C_ = [C_[s][:, no_core_idx[s]] for s in [0, 1]]
                    S_ = [multi_dot((C_[s].T, self.S, C_[s])) for s in [0, 1]]
                    W_ = []
                    for s in [0, 1]:
                        es_, vs_ = eigh(S_[s])
                        s_ = sqrt(es_)
                        s_ = diag(1.0 / s_)
                        W_.append(multi_dot((vs_, s_, vs_.T)))
                    self.W = [C_[s] @ W_[s] for s in [0, 1]]
                else:
                    P_core = eye(self.W.shape[0]) - self.P_core @ self.S
                    C_ = P_core @ self.W
                    # NOTE: PYSCF has basis in 1s2s3s2p2p2p3p3p3p format
                    # fix no_core_idx - use population for now
                    Cpop = multi_dot((C_.T, self.S, C_))
                    Cpop = diag(Cpop)
                    no_core_idx = where(Cpop > 0.7)[0]
                    C_ = C_[:, no_core_idx]
                    S_ = multi_dot((C_.T, self.S, C_))
                    es_, vs_ = eigh(S_)
                    s_ = sqrt(es_)
                    s_ = diag(1.0 / s_)
                    W_ = multi_dot((vs_, s_, vs_.T))
                    self.W = C_ @ W_

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
            self.W = vs_[:, edx] / sqrt(es_[edx]) @ vs_[:, edx].T

            es_, vs_ = eigh(self.S)
            edx = es_ > 1.0e-15
            W_ = vs_[:, edx] / sqrt(es_[edx]) @ vs_[:, edx].T
            if self.frozen_core:
                P_core = eye(W_.shape[0]) - self.P_core @ self.S
                C_ = P_core @ W_
                Cpop = multi_dot((C_.T, self.S, C_))
                Cpop = diag(Cpop)
                no_core_idx = where(Cpop > 0.55)[0]
                C_ = C_[:, no_core_idx]
                S_ = multi_dot((C_.T, self.S, C_))
                es_, vs_ = eigh(S_)
                s_ = sqrt(es_)
                s_ = diag(1.0 / s_)
                W_ = multi_dot((vs_, s_, vs_.T))
                W_ = C_ @ W_

            self.W = get_loc(
                self.mf.mol, W_, "PM", pop_method=pop_method, init_guess=init_guess
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
            S12, S2 = get_xovlp(self.mol, basis=iao_valence_basis)
            # Use these to get IAOs
            Ciao = get_iao(Co, S12, self.S, S2=S2)

            if not valence_only:
                # Now get PAOs
                if loc_type.upper() != "SO":
                    Cpao = get_pao(Ciao, self.S, S12)
                elif loc_type.upper() == "SO":
                    Cpao = get_pao_native(
                        Ciao,
                        self.S,
                        self.mol,
                        iao_valence_basis=iao_valence_basis,
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
                Wstack = zeros(
                    (Ciao.shape[0], Ciao.shape[1] + Cpao.shape[1])
                )  # -self.ncore))
            else:
                Wstack = zeros((Ciao.shape[0], Ciao.shape[1]))

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
                    Wstack = hstack((Ciao, Cpao))
            if not nosave:
                self.W = Wstack
                assert allclose(self.W.T @ self.S @ self.W, eye(self.W.shape[1]))
            else:
                assert allclose(Wstack.T @ self.S @ Wstack, eye(Wstack.shape[1]))
                return Wstack
            nmo = self.C.shape[1] - self.ncore
            nlo = self.W.shape[1]

            if not valence_only:
                if nmo > nlo:
                    Co_nocore = self.C[:, self.ncore : self.Nocc]
                    Cv = self.C[:, self.Nocc :]
                    # Ensure that the LOs span the occupied space
                    assert allclose(
                        np.sum((self.W.T @ self.S @ Co_nocore) ** 2.0),
                        self.Nocc - self.ncore,
                    )
                    # Find virtual orbitals that lie in the span of LOs
                    u, l, vt = svd(self.W.T @ self.S @ Cv, full_matrices=False)
                    unused(u)
                    nvlo = nlo - self.Nocc - self.ncore
                    assert allclose(np.sum(l[:nvlo]), nvlo)
                    C_ = hstack([Co_nocore, Cv @ vt[:nvlo].T])
                    self.lmo_coeff = self.W.T @ self.S @ C_
                else:
                    self.lmo_coeff = self.W.T @ self.S @ self.C[:, self.ncore :]
            else:
                self.lmo_coeff = self.W.T @ self.S @ self.C[:, self.ncore :]

        elif lo_method == "boys":
            es_, vs_ = eigh(self.S)
            edx = es_ > 1.0e-15
            W_ = vs_[:, edx] / sqrt(es_[edx]) @ vs_[:, edx].T
            if self.frozen_core:
                P_core = eye(W_.shape[0]) - self.P_core @ self.S
                C_ = P_core @ W_
                Cpop = multi_dot((C_.T, self.S, C_))
                Cpop = diag(Cpop)
                no_core_idx = where(Cpop > 0.55)[0]
                C_ = C_[:, no_core_idx]
                S_ = multi_dot((C_.T, self.S, C_))
                es_, vs_ = eigh(S_)
                s_ = sqrt(es_)
                s_ = diag(1.0 / s_)
                W_ = multi_dot((vs_, s_, vs_.T))
                W_ = C_ @ W_

            self.W = get_loc(self.mol, W_, "BOYS")

            if not self.frozen_core:
                self.lmo_coeff = self.W.T @ self.S @ self.C
            else:
                self.lmo_coeff = self.W.T @ self.S @ self.C[:, self.ncore :]

        else:
            raise ValueError(f"lo_method = {lo_method} not implemented!")
