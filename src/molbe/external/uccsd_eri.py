import numpy
from pyscf.cc.uccsd import _make_eris_incore


def make_eris_incore(mycc, Vss, Vos, mo_coeff=None, ao2mofn=None, frozen=False):
    vhf = frank_get_veff(
        mycc, mycc._scf.make_rdm1(mycc.mo_coeff, mycc.mo_occ), Vss, Vos
    )
    fockao = frank_get_fock(mycc, vhf, frozen)
    mycc._scf.get_veff = lambda *args, **kwargs: vhf
    mycc._scf.get_fock = lambda *args, **kwargs: fockao
    return _make_eris_incore(mycc, mo_coeff=mo_coeff, ao2mofn=ao2mofn)  # , frozen)


def frank_get_veff(mycc, dm, Vss, Vos):
    veffss = [
        numpy.einsum("pqrs,sr->pq", Vss[s], dm[s])
        - numpy.einsum("psrq,sr->pq", Vss[s], dm[s])
        for s in [0, 1]
    ]
    veffos = [
        numpy.einsum("pqrs,sr->pq", Vos, dm[1]),
        numpy.einsum("pqrs,qp->rs", Vos, dm[0]),
    ]
    veff = [veffss[s] + veffos[s] for s in [0, 1]]

    return veff


def frank_get_fock(mycc, vhf, frozen):
    if frozen == False:
        mycc._scf.full_gcore = None
        mycc._scf.full_hs = None
        fock = [mycc._scf.h1[s] + mycc._scf.gcores_raw[s] for s in [0, 1]]
    else:
        mycc._scf.full_gcore = [mycc._scf.gcores_raw[s] - vhf[s] for s in [0, 1]]
        mycc._scf.full_hs = [
            mycc._scf.h1[s] + mycc._scf.full_gcore[s] + mycc._scf.core_veffs[s]
            for s in [0, 1]
        ]
        fock = [mycc._scf.full_hs[s] + vhf[s] for s in [0, 1]]
    return fock
