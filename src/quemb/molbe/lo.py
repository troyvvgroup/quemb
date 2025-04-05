# Author(s): Henry Tran, Oinam Meitei, Shaun Weatherly
#

from typing import Literal, overload

import numpy as np
from numpy import allclose, diag, eye, sqrt, where, zeros
from numpy.linalg import eigh, inv, multi_dot, norm, solve, svd
from pyscf.gto import intor_cross
from pyscf.gto.mole import Mole
from pyscf.lo import Boys
from pyscf.lo.edmiston import EdmistonRuedenberg
from pyscf.lo.pipek import PipekMezey

from quemb.shared.external.lo_helper import (
    cano_orth,
    get_aoind_by_atom,
    reorder_by_atom_,
    symm_orth,
)
from quemb.shared.helper import ncore_, unused
from quemb.shared.typing import Matrix, Tensor3D


def remove_core_mo(Clo: Matrix, Ccore: Matrix, S: Matrix, thr: float = 0.5) -> Matrix:
    """Remove core molecular orbitals from localized Clo"""
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
    """Gets overlap matrix between the two bases and in secondary basis.
    Used for IAOs: returns the overlap between valence (minimal) and working
    (large) bases and overlap in the minimal basis

    Parameters
    ----------
    mol :
        mol object to get working (large) and valence (minimal) basis
    basis :
        the IAO valence (minimal-like) basis, Knizia recommended 'minao'

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
    S2: Matrix,
    mol: Mole,
    iao_valence_basis: str,
    iao_loc_method: str = "SO",
) -> Matrix:
    """Gets symmetrically orthogonalized IAO coefficient matrix from system MOs
    Derived from G. Knizia: J. Chem. Theory Comput. 2013, 9, 11, 4834–4843
    Note: same function as `get_iao_from_s12` in frankenstein

    Parameters
    ----------
    Co:
        occupied MO coefficient matrix with core
    S12:
        ovlp between working (large) basis and valence (minimal) basis
        can be thought of as working basis in valence basis
    S1:
        AO ovlp matrix, in working (large) basis
    S2:
        AO ovlp matrix, in valence (minimal) basis
    mol:
        mol object
    iao_valence_basis:
        (minimal-like) basis used for valence orbitals
    iao_loc_method:
        Localization method for the IAOs and PAOs.
        If symmetric orthogonalization is used, the overlap matrices between
        the valence (minimal) and working (large) basis are determined by
        separating S1 (working) by AO labels. If other localization methods
        are used, these matrices are calculated in full
        Default is SO
    Return
    ------
    Ciao :class:`quemb.shared.typing.Matrix`
        (symmetrically orthogonalized)
    """
    n = Co.shape[0]

    if iao_loc_method.upper() == "SO":
        # this is the "native" option in Frankenstein and older versions of Quemb.
        # Rather than form the full S matrices in all bases, the S1 in the
        # large, working basis is separated by labels

        # This should not be the default, and it tends to produce less accurate
        # results

        # Form a mol object with the valence basis for the ao_labels
        mol_alt = mol.copy()
        mol_alt.basis = iao_valence_basis
        mol_alt.build()

        full_ao_labels = mol.ao_labels()
        valence_ao_labels = mol_alt.ao_labels()

        # list of working basis indices which are in the valence basis
        nonvir_idx = [
            idx
            for idx, label in enumerate(full_ao_labels)
            if (label in valence_ao_labels)
        ]

        # Set up the overlap matrices
        S2 = S1[np.ix_(nonvir_idx, nonvir_idx)]
        S12 = S1[:, nonvir_idx]

    # Define Projection Matrices
    # Note: timing for this step is better than inverting, then multiplying
    P_12 = solve(S1, S12)
    P_21 = solve(S2, S12.T)

    # Generated polarized occupied states, in working basis, O in Knizia paper
    O_pol = Co @ Co.T

    # Generate depolarized occupied MOs
    C_depol = P_12 @ P_21 @ Co

    # Orthogonalize C_depol and get \tilde{O}, in Knizia paper
    S_til = C_depol.T @ S1 @ C_depol
    O_depol = C_depol @ inv(S_til) @ C_depol.T

    # Generate C_IAOs for the system
    Ciao_pol = (eye(n) - (O_depol + O_pol - 2 * O_pol @ S1 @ O_depol) @ S1) @ P_12

    # Orthoganize C_IAOs
    Ciao = symm_orth(Ciao_pol, ovlp=S1)

    # Check span
    rep_err = norm(Ciao @ Ciao.T @ S1 @ O_pol - O_pol)
    if rep_err > 1.0e-10:
        raise RuntimeError
    return Ciao


def get_pao(
    Ciao: Matrix,
    S1: Matrix,
    S12: Matrix,
    mol: Mole,
    iao_valence_basis: str,
    iao_loc_method: str = "SO",
) -> Matrix:
    """Get (symmetrically though often canonically) orthogonalized PAOs
    from given (localized) IAOs
    Defined in detail in J. Chem. Theory Comput. 2024, 20, 24, 10912–10921

    Parameters
    ----------
    Ciao:
        the orthogonalized IAO coefficient matrix
        (output of :func:get_iao)
    S1:
        ao ovlp matrix in working (large) basis
    S12:
        ovlp between working (large) basis and valence (minimal) basis
    mol:
        mol object
    iao_valence_basis:
        (minimal-like) basis used for valence orbitals
    iao_loc_method:
        Localization method for the IAOs and PAOs.
        If symmetric orthogonalization is used, the overlap matrices between
        the valence (minimal) and working (large) basis are determined by
        separating S1 (working) by AO labels. If other localization methods
        are used, these matrices are calculated in full
        Default is SO
    Returns
    -------
    Cpao: :class:`quemb.shared.typing.Matrix`
        (orthogonalized)
    """
    n = Ciao.shape[0]

    # projector into IAOs
    Piao = Ciao @ Ciao.T @ S1

    if iao_loc_method.upper() == "SO":
        # Read further info in `get_iao`
        # Form a mol object with the valence basis for the ao_labels
        mol_alt = mol.copy()
        mol_alt.basis = iao_valence_basis
        mol_alt.build()

        full_ao_labels = mol.ao_labels()
        valence_ao_labels = mol_alt.ao_labels()

        # list of working basis indices which are in the valence basis
        vir_idx = [
            idx
            for idx, label in enumerate(full_ao_labels)
            if (label not in valence_ao_labels)
        ]

        Cpao_redundant = (eye(n) - Piao)[:, vir_idx]
    else:
        P_12 = inv(S1) @ S12
        # set of orbitals minus valence (orth in working basis)
        nonval = eye(n) - P_12 @ P_12.T

        Cpao_redundant = (eye(n) - Piao) @ nonval

    # begin canonical orthogonalization to get rid of redundant orbitals
    try:
        Cpao = symm_orth(Cpao_redundant, ovlp=S1)
    except ValueError:
        Cpao = cano_orth(Cpao_redundant, ovlp=S1)
    return Cpao


@overload
def get_loc(
    mol: Mole,
    C: Matrix,
    method: Literal["PM"],
    pop_method: str | None = ...,
    init_guess: Matrix | None = ...,
) -> Mole: ...


@overload
def get_loc(
    mol: Mole,
    C: Matrix,
    method: Literal["ER", "FB"],
    pop_method: None = ...,
    init_guess: Matrix | None = ...,
) -> Mole: ...


def get_loc(
    mol: Mole,
    C: Matrix,
    method: Literal["ER", "PM", "FB"] = "ER",
    pop_method: str | None = None,
    init_guess: Matrix | str | None = "atomic",
) -> Mole:
    """Import, initialize, and call localization procedure `method` for C
    from `PySCF`

    Parameters
    ----------
    mol:
        mol object
    C:
        MO coefficients
    method:
        Localization method. Options include:
        EDMINSTON-RUEDENBERG, ER;
        PIPEK-MIZEY, PIPEK, PM;
        FOSTER-BOYS, BOYS, FB
    pop_method:
        Method for calculating orbital population, by default 'meta-lowdin'
        See pyscf.lo for more details and options. This is only used for
        Pipek-Mezey localization
    init_guess:
        Initial guess for localization optimization.
        Default is `atomic`, See pyscf.lo for more details and options
    Returns
    -------
    mlo: :class:`quemb.shared.typing.Matrix`
        Localized mol object
    """
    Localizer: type[EdmistonRuedenberg] | type[PipekMezey] | type[Boys]
    if method.upper() in ["EDMINSTON-RUEDENBERG", "ER"]:
        Localizer = EdmistonRuedenberg
    elif method.upper() in ["PIPEK-MEZEY", "PIPEK", "PM"]:
        Localizer = PipekMezey
    elif method.upper() in ["FOSTER-BOYS", "BOYS", "FB"]:
        Localizer = Boys
        # Note: Convergence issues for IAO-Boys with frozen core,
        # when not using an 'atomic' initial guess
    else:
        raise NotImplementedError("Localization scheme not understood")

    mlo = Localizer(mol, C)
    if pop_method is not None:
        assert isinstance(Localizer, PipekMezey)
        mlo.pop_method = pop_method

    mlo.init_guess = init_guess
    return mlo.kernel()


class MixinLocalize:
    def localize(
        self,
        lo_method,
        iao_valence_basis="sto-3g",
        iao_loc_method="SO",
        iao_valence_only=False,
        pop_method=None,
        init_guess=None,
        hstack=False,
        nosave=False,
    ):
        """Molecular orbital localization

        Performs molecular orbital localization computations. For large basis,
        IAO is recommended augmented with PAO orbitals.

        NOTE: For molecular systems, with frozen core, the core and valence are
        localized TOGETHER. This is not the case of periodic systems.

        Parameters
        ----------
        lo_method : str
            Localization method in quantum chemistry. 'lowdin', 'boys', 'er', 'pm', and
            'iao' are supported.
        iao_valence_basis : str
            Name of minimal basis set for IAO scheme. 'sto-3g' suffice for most cases.
        iao_loc_method: str
            Name of localization method in quantum chemistry for the IAOs and PAOs.
            Options include 'Boys', 'PM', 'ER' (as documented in PySCF). Default is
            'SO', or symmetric orthogonalization.
            If not using SO, we suggest using 'PM', as it is more robust than 'Boys'
            localization and less expensive than 'ER'
        iao_valence_only : bool
            If this option is set to True, all calculation will be performed in the
            valence basis in the IAO partitioning. Default is False.
            This is an experimental feature: the returned energy is not accurate
        """
        if lo_method.upper() == "LOWDIN":
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

        elif lo_method.upper() in [
            "PIPEK-MEZEY",
            "PIPEK",
            "PM",
            "FOSTER-BOYS",
            "BOYS",
            "FB",
            "EDMINSTON-RUEDENBERG",
            "ER",
        ]:
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
                self.mf.mol, W_, lo_method, pop_method=pop_method, init_guess=init_guess
            )

            if not self.frozen_core:
                self.lmo_coeff = self.W.T @ self.S @ self.C
            else:
                self.lmo_coeff = self.W.T @ self.S @ self.C[:, self.ncore :]

        elif lo_method.upper() == "IAO":
            # IAO working basis: (w): (large) basis set we use
            # IAO valence basis: (v): minimal-like basis we try to resemble

            # Occupied mo_coeff (with core)
            Co = self.C[:, : self.Nocc]

            # Get necessary overlaps, second arg is IAO valence basis
            S_vw, S_vv = get_xovlp(self.fobj.mol, basis=iao_valence_basis)

            # How do we describe the rest of the space?
            # If iao_valence_only=False, we use PAOs:
            if not iao_valence_only:
                Ciao = get_iao(
                    Co,
                    S_vw,
                    self.S,
                    S_vv,
                    self.fobj.mol,
                    iao_valence_basis,
                    iao_loc_method,
                )

                Cpao = get_pao(
                    Ciao, self.S, S_vw, self.fobj.mol, iao_valence_basis, iao_loc_method
                )

                if iao_loc_method.upper() != "SO":
                    # Localize IAOs and PAOs
                    Ciao = get_loc(self.fobj.mol, Ciao, iao_loc_method)
                    Cpao = get_loc(self.fobj.mol, Cpao, iao_loc_method)
            else:
                Ciao = get_iao(
                    Co,
                    S_vw,
                    self.S,
                    S_vv,
                    self.fobj.mol,
                    iao_valence_basis,
                    iao_loc_method,
                )

                if iao_loc_method.upper() != "SO":
                    Ciao = get_loc(self.fobj.mol, Ciao, iao_loc_method)

            # Rearrange by atom
            aoind_by_atom = get_aoind_by_atom(self.fobj.mol)
            Ciao, iaoind_by_atom = reorder_by_atom_(Ciao, aoind_by_atom, self.S)

            if not iao_valence_only:
                Cpao, paoind_by_atom = reorder_by_atom_(Cpao, aoind_by_atom, self.S)

            if self.frozen_core:
                # Remove core MOs
                Cc = self.C[:, : self.ncore]  # Assumes core are first
                Ciao = remove_core_mo(Ciao, Cc, self.S)

            shift = 0
            ncore = 0
            if not iao_valence_only:
                Wstack = zeros(
                    (Ciao.shape[0], Ciao.shape[1] + Cpao.shape[1])
                )  # -self.ncore))
            else:
                Wstack = zeros((Ciao.shape[0], Ciao.shape[1]))

            if self.frozen_core:
                for ix in range(self.fobj.mol.natm):
                    nc = ncore_(self.fobj.mol.atom_charge(ix))
                    ncore += nc
                    niao = len(iaoind_by_atom[ix])
                    iaoind_ix = [i_ - ncore for i_ in iaoind_by_atom[ix][nc:]]
                    Wstack[:, shift : shift + niao - nc] = Ciao[:, iaoind_ix]
                    shift += niao - nc
                    if not iao_valence_only:
                        npao = len(paoind_by_atom[ix])
                        Wstack[:, shift : shift + npao] = Cpao[:, paoind_by_atom[ix]]
                        shift += npao
            else:
                if not hstack:
                    for ix in range(self.fobj.mol.natm):
                        niao = len(iaoind_by_atom[ix])
                        Wstack[:, shift : shift + niao] = Ciao[:, iaoind_by_atom[ix]]
                        shift += niao
                        if not iao_valence_only:
                            npao = len(paoind_by_atom[ix])
                            Wstack[:, shift : shift + npao] = Cpao[
                                :, paoind_by_atom[ix]
                            ]
                            shift += npao
                else:
                    Wstack = np.hstack((Ciao, Cpao))
            if not nosave:
                self.W = Wstack
                assert allclose(self.W.T @ self.S @ self.W, eye(self.W.shape[1]))
            else:
                assert allclose(Wstack.T @ self.S @ Wstack, eye(Wstack.shape[1]))
                return Wstack
            nmo = self.C.shape[1] - self.ncore
            nlo = self.W.shape[1]

            if not iao_valence_only:
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
                    C_ = np.hstack([Co_nocore, Cv @ vt[:nvlo].T])
                    self.lmo_coeff = self.W.T @ self.S @ C_
                else:
                    self.lmo_coeff = self.W.T @ self.S @ self.C[:, self.ncore :]
            else:
                self.lmo_coeff = self.W.T @ self.S @ self.C[:, self.ncore :]

        else:
            raise ValueError(f"lo_method = {lo_method} not implemented!")
