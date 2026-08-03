from numpy import diag, einsum
from pyscf.cc.uccsd import _make_eris_incore


def make_eris_incore(mycc, Vss, Vos, mo_coeff=None, ao2mofn=None, frozen=False):
    vhf = frank_get_veff(mycc._scf.make_rdm1(mycc.mo_coeff, mycc.mo_occ), Vss, Vos)
    fockao = frank_get_fock(mycc, vhf, frozen)
    mycc._scf.get_veff = lambda *args, **kwargs: vhf  # noqa: ARG005
    mycc._scf.get_fock = lambda *args, **kwargs: fockao  # noqa: ARG005
    return _make_eris_incore(mycc, mo_coeff=mo_coeff, ao2mofn=ao2mofn)  # , frozen)


def frank_get_veff(dm, Vss, Vos):
    veffss = [
        einsum("pqrs,sr->pq", Vss[s], dm[s]) - einsum("psrq,sr->pq", Vss[s], dm[s])
        for s in [0, 1]
    ]
    veffos = [
        einsum("pqrs,sr->pq", Vos, dm[1]),
        einsum("pqrs,qp->rs", Vos, dm[0]),
    ]
    veff = [veffss[s] + veffos[s] for s in [0, 1]]

    return veff


def frank_get_fock(mycc, vhf, frozen):
    if not frozen:
        mycc._scf.full_gcore = None
        mycc._scf.full_hs = None
        # h1 + gcores_raw (the full-system Fock projected into the
        # embedding basis) is only a good stand-in for the embedded SCF's
        # own converged Fock when the embedded density stays close to
        # that initial snapshot -- true for small equal_bath-style
        # fragments, but not for common_bath's larger Schmidt spaces,
        # where the embedded SCF relaxes far from it. The mismatch breaks
        # CCSD's canonical-reference assumption and can cause amplitude
        # blowup and a singular DIIS matrix.
        #
        # mo_coeff/mo_energy are by definition the exact eigenvectors/
        # eigenvalues of whatever Fock the embedded SCF converged to, so
        # reconstructing fockao from them is exact rather than merely a
        # better approximation. mo_coeff is orthogonal here (get_scfObj
        # solves with S=I), so mo_coeff @ diag(mo_energy) @ mo_coeff.T
        # round-trips through _common_init_'s C.T @ fockao @ C transform
        # to give eris.mo_energy == mo_energy exactly and eris.focka/
        # fockb exactly diagonal.
        fock = [
            mycc._scf.mo_coeff[s]
            @ diag(mycc._scf.mo_energy[s])
            @ mycc._scf.mo_coeff[s].T
            for s in [0, 1]
        ]
    else:
        mycc._scf.full_gcore = [mycc._scf.gcores_raw[s] - vhf[s] for s in [0, 1]]
        mycc._scf.full_hs = [
            mycc._scf.h1[s] + mycc._scf.full_gcore[s] + mycc._scf.core_veffs[s]
            for s in [0, 1]
        ]
        fock = [mycc._scf.full_hs[s] + vhf[s] for s in [0, 1]]
    return fock
