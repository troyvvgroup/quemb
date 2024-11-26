# Author(s): Henry Tran, Oinam Meitei, Shaun Weatherly
#
import numpy
from pyscf.gto import intor_cross


def dot_gen(A, B, ovlp):
    return A.T @ B if ovlp is None else A.T @ ovlp @ B


def get_cano_orth_mat(A, thr=1.0e-6, ovlp=None):
    S = dot_gen(A, A, ovlp)
    e, u = numpy.linalg.eigh(S)
    if thr > 0:
        idx_keep = e / e[-1] > thr
    else:
        idx_keep = list(range(e.shape[0]))
    U = u[:, idx_keep] * e[idx_keep] ** -0.5
    return U


def cano_orth(A, thr=1.0e-6, ovlp=None):
    """Canonically orthogonalize columns of A"""
    U = get_cano_orth_mat(A, thr, ovlp)

    return A @ U


def get_symm_orth_mat(A, thr=1.0e-6, ovlp=None):
    S = dot_gen(A, A, ovlp)
    e, u = numpy.linalg.eigh(S)
    if int(numpy.sum(e < thr)) > 0:
        raise ValueError(
            "Linear dependence is detected in the column space of A: "
            "smallest eigenvalue (%.3E) is less than thr (%.3E). "
            "Please use 'cano_orth' instead." % (numpy.min(e), thr)
        )
    U = u @ numpy.diag(e**-0.5) @ u.T

    return U


def symm_orth(A, thr=1.0e-6, ovlp=None):
    """Symmetrically orthogonalize columns of A"""
    U = get_symm_orth_mat(A, thr, ovlp)
    return A @ U


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
    Clo2 = symm_orth(Clo1[:, idx_keep], ovlp=S)

    return Clo2


def get_xovlp(mol, basis="sto-3g"):
    """
    Gets set of valence orbitals based on smaller (should be minimal) basis
    inumpy.t:
        mol - pyscf mol object, just need it for the working basis
        basis - the IAO basis, Knizia recommended 'minao'
    returns:
        S12 - Overlap of two basis sets
        S22 - Overlap in new basis set
    """
    mol_alt = mol.copy()
    mol_alt.basis = basis
    mol_alt.build()

    S12 = intor_cross("int1e_ovlp", mol, mol_alt)
    S22 = mol_alt.intor("int1e_ovlp")

    return S12, S22


def get_iao(Co, S12, S1, S2=None):
    """
    Args:
        Co: occupied coefficient matrix with core
        p: valence AO matrix in AO
        no: number of occ orbitals
        S12: ovlp between working basis and valence basis
             can be thought of as working basis in valence basis
        S1: ao ovlp matrix
        S2: valence AO ovlp
    """
    # define projection operators
    n = Co.shape[0]
    if S2 is None:
        S2 = S12.T @ numpy.linalg.inv(S1) @ S12
    P1 = numpy.linalg.inv(S1)
    P2 = numpy.linalg.inv(S2)

    # depolarized occ mo
    Cotil = P1 @ S12 @ P2 @ S12.T @ Co

    # repolarized valence AOs
    ptil = P1 @ S12
    Stil = Cotil.T @ S1 @ Cotil

    Po = Co @ Co.T
    Potil = Cotil @ numpy.linalg.inv(Stil) @ Cotil.T

    Ciao = (numpy.eye(n) - (Po + Potil - 2 * Po @ S1 @ Potil) @ S1) @ ptil
    Ciao = symm_orth(Ciao, ovlp=S1)

    # check span
    rep_err = numpy.linalg.norm(Ciao @ Ciao.T @ S1 @ Po - Po)
    if rep_err > 1.0e-10:
        raise RuntimeError
    return Ciao


def get_pao(Ciao, S, S12, S2, mol):
    """
    Args:
        Ciao: output of :func:`get_iao`
        S: ao ovlp matrix
        S12: valence orbitals projected into ao basis
        S2: valence ovlp matrix
        mol: pyscf mol instance
    Return:
        Cpao (orthogonalized)
    """
    n = Ciao.shape[0]
    s12 = numpy.linalg.inv(S) @ S12
    nonval = (
        numpy.eye(n) - s12 @ s12.T
    )  # set of orbitals minus valence (orth in working basis)

    Piao = Ciao @ Ciao.T @ S  # projector into IAOs
    Cpao = (numpy.eye(n) - Piao) @ nonval  # project out IAOs from non-valence basis

    # begin canonical orthogonalization to get rid of redundant orbitals
    numpy.o0 = Cpao.shape[1]
    Cpao = cano_orth(Cpao, ovlp=S)
    numpy.o1 = Cpao.shape[1]

    return Cpao


def get_pao_native(Ciao, S, mol, valence_basis):
    """
    Args:
        Ciao: output of :func:`get_iao_native`
        S: ao ovlp matrix
        mol: mol object
        valence basis: basis used for valence orbitals
    Return:
        Cpao (symmetrically orthogonalized)
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
