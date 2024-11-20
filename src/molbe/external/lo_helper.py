# Author(s): Hong-Zhou Ye
#            Henry Tran
# NOTICE: The following code is entirely written by Hong-Zhou Ye.
#         The code has been slightly modified.
#

import numpy


def get_symm_mat_pow(A, p, check_symm=True, thresh=1.0e-8):
    """A ** p where A is symmetric

    Note:
        For integer p, it calls numpy.linalg.matrix_power
    """
    if abs(int(p) - p) < thresh:
        return numpy.linalg.matrix_power(A, int(p))

    if check_symm:
        assert numpy.linalg.norm(A - A.conj().T) < thresh

    e, u = numpy.linalg.eigh(A)
    Ap = u @ numpy.diag(e**p) @ u.conj().T

    return Ap


def get_aoind_by_atom(mol, atomind_by_motif=None):
    natom = mol.natm
    aoslice_by_atom = mol.aoslice_by_atom()
    aoshift_by_atom = [0] + [aoslice_by_atom[ia][-1] for ia in range(natom)]
    # if motif info is provided, group lo by motif
    if atomind_by_motif is None:
        aoind_by_atom = [
            list(range(*aoshift_by_atom[ia : ia + 2])) for ia in range(natom)
        ]
    else:
        nmotif = len(atomind_by_motif)
        assert set([ia for im in range(nmotif) for ia in atomind_by_motif[im]]) == set(
            range(natom)
        )
        aoind_by_atom = [[] for im in range(nmotif)]
        for im in range(nmotif):
            for ia in atomind_by_motif[im]:
                aoind_by_atom[im] += list(range(*aoshift_by_atom[ia : ia + 2]))

    return aoind_by_atom


def reorder_by_atom_(Clo, aoind_by_atom, S, thr=0.5):
    import numpy as np

    natom = len(aoind_by_atom)
    nlo = Clo.shape[1]
    X = get_symm_mat_pow(S, 0.5)

    Clo_soao = X @ Clo

    loind_reorder = []
    loind_by_atom = [None] * natom
    loshift = 0
    for ia in range(natom):
        ra = aoind_by_atom[ia]
        poplo_by_atom = np.sum(Clo_soao[ra] ** 2.0, axis=0)
        loind_a = np.where(poplo_by_atom > thr)[0].tolist()
        loind_reorder += loind_a
        nlo_a = len(loind_a)
        loind_by_atom[ia] = list(range(loshift, loshift + nlo_a))
        loshift += nlo_a
    if loind_reorder != list(range(nlo)):
        print("REORDERD")
        Clo_new = Clo[:, loind_reorder]
    else:
        Clo_new = Clo
    return Clo_new, loind_by_atom
