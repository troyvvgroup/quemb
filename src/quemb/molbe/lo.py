# Author(s): Henry Tran, Oinam Meitei, Shaun Weatherly
#

from typing import Literal, overload

import numpy as np
from numpy import allclose, diag, eye, where
from numpy.linalg import inv, norm, solve
from pyscf.gto import intor_cross
from pyscf.gto.mole import Mole
from pyscf.lo import Boys
from pyscf.lo.edmiston import EdmistonRuedenberg
from pyscf.lo.pipek import PipekMezey
from typing_extensions import assert_never

from quemb.shared.external.lo_helper import (
    cano_orth,
    symm_orth,
)
from quemb.shared.typing import Matrix, Tensor3D

IAO_LocMethods = Literal["lowdin", "boys", "PM", "ER"]
LocMethods = Literal["lowdin", "boys", "ER", "PM", "IAO"]


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
    iao_loc_method: IAO_LocMethods = "lowdin",
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
        Default is lowdin
    Return
    ------
    Ciao :class:`quemb.shared.typing.Matrix`
        (symmetrically orthogonalized)
    """
    n = Co.shape[0]

    if iao_loc_method == "lowdin":
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
    iao_loc_method: IAO_LocMethods = "lowdin",
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
        Default is lowdin
    Returns
    -------
    Cpao: :class:`quemb.shared.typing.Matrix`
        (orthogonalized)
    """
    n = Ciao.shape[0]

    # projector into IAOs
    Piao = Ciao @ Ciao.T @ S1

    if iao_loc_method == "lowdin":
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
    elif iao_loc_method == "boys" or iao_loc_method == "PM" or iao_loc_method == "ER":
        P_12 = inv(S1) @ S12
        # set of orbitals minus valence (orth in working basis)
        nonval = eye(n) - P_12 @ P_12.T

        Cpao_redundant = (eye(n) - Piao) @ nonval
    else:
        assert_never(iao_loc_method)

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
) -> Matrix[np.float64]: ...


@overload
def get_loc(
    mol: Mole,
    C: Matrix,
    method: Literal["ER", "boys"],
    pop_method: None = ...,
    init_guess: Matrix | None = ...,
) -> Matrix[np.float64]: ...


def get_loc(
    mol: Mole,
    C: Matrix,
    method: Literal["ER", "PM", "boys"] = "ER",
    pop_method: str | None = None,
    init_guess: Matrix | str | None = "atomic",
) -> Matrix[np.float64]:
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
    if method == "ER":
        Localizer = EdmistonRuedenberg
    elif method == "PM":
        Localizer = PipekMezey
    elif method == "boys":
        Localizer = Boys
        # Note: Convergence issues for IAO-Boys with frozen core,
        # when not using an 'atomic' initial guess
    else:
        raise NotImplementedError(f"Localization scheme {method} not understood")
        assert_never(method)

    mlo = Localizer(mol, C)
    if pop_method is not None:
        assert isinstance(Localizer, PipekMezey)
        mlo.pop_method = pop_method

    mlo.init_guess = init_guess
    return mlo.kernel()
