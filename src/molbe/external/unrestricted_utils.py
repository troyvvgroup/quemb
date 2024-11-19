# Authors: Leah Weisburn, Hongzhou Ye, Henry Tran

import functools

import h5py
import numpy


def make_uhf_obj(fobj_a, fobj_b, frozen=False):
    from pyscf import scf

    """
    Constructs UHF object from the alpha and beta components
    """

    # Create UHF object using the alpha, beta components
    uhf_a = scf.addons.convert_to_uhf(fobj_a._mf)
    uhf_b = scf.addons.convert_to_uhf(fobj_b._mf)
    full_uhf = scf.addons.convert_to_uhf(fobj_a._mf)

    new_mo_coeff = (uhf_a.mo_coeff[0], uhf_b.mo_coeff[1])

    # Give UHF object correct spin attributes
    full_uhf.mo_coeff = new_mo_coeff
    full_uhf.mo_occ = (uhf_a.mo_occ[0], uhf_b.mo_occ[1])
    full_uhf.mo_energy = (uhf_a.mo_energy[0], uhf_b.mo_energy[1])
    full_uhf.h1 = (fobj_a.h1, fobj_b.h1)

    # Define the HF veff in alpha and beta spin Schmidt spaces
    full_uhf.veff0_a = functools.reduce(
        numpy.dot, (fobj_a.TA.T, fobj_a.hf_veff, fobj_a.TA)
    )
    full_uhf.veff0_b = functools.reduce(
        numpy.dot, (fobj_b.TA.T, fobj_b.hf_veff, fobj_b.TA)
    )

    # Build correct eris
    Vs = uccsd_restore_eris((1, 1, 1), fobj_a, fobj_b)

    full_uhf.TA = [fobj_a.TA, fobj_b.TA]

    # Build core components
    if frozen:
        full_uhf.gcores_raw = [
            fobj_a.TA.T @ (fobj_a.hf_veff - fobj_a.core_veff) @ fobj_a.TA,
            fobj_b.TA.T @ (fobj_b.hf_veff - fobj_b.core_veff) @ fobj_b.TA,
        ]
        full_uhf.core_veffs = [
            fobj_a.TA.T @ fobj_a.core_veff @ fobj_a.TA,
            fobj_b.TA.T @ fobj_b.core_veff @ fobj_b.TA,
        ]
    else:
        full_uhf.gcores_raw = [full_uhf.veff0_a, full_uhf.veff0_b]
        #        full_uhf.gcores_raw = [fobj_a.TA.T @ fobj_a.hf_veff @ fobj_a.TA, fobj_b.TA.T @ fobj_b.hf_veff @ fobj_b.TA]
        full_uhf.core_veffs = None

    return full_uhf, Vs


def uccsd_restore_eris(symm, fobj_a, fobj_b, pad0=True, skip_Vab=False):
    from pyscf import ao2mo

    """
    restore ERIs in the correct spin spaces
    """
    Vsfile = fobj_a.eri_file
    Vsname = fobj_a.dname

    nf = (fobj_a._mf.mo_coeff.shape[1], fobj_b._mf.mo_coeff.shape[1])

    with h5py.File(Vsfile, "r") as fVs:
        Vs = [None] * 3
        Vs[0] = ao2mo.restore(symm[0], fVs[Vsname[0]], nf[0])
        Vs[1] = ao2mo.restore(symm[1], fVs[Vsname[1]], nf[1])
        Vs[2] = restore_eri_gen(symm[2], fVs[Vsname[2]][()], nf[0], nf[1])

    return Vs


def restore_eri_gen(targetsym, eri, norb1, norb2):
    """
    An extension of PySCF's ao2mo.restore to Vaabb.
    """
    assert targetsym in (1, 4)

    npair1 = norb1 * (norb1 + 1) // 2
    npair2 = norb2 * (norb2 + 1) // 2

    if eri.size == norb1**2 * norb2**2:  # s1
        if targetsym == 1:
            return eri.reshape(norb1, norb1, norb2, norb2)
        else:
            return _convert_eri_gen(1, 4, eri, norb1, norb2)
    elif eri.size == npair1 * npair2:  # s4
        if targetsym == 1:
            return _convert_eri_gen(4, 1, eri, norb1, norb2)
        else:
            return eri.reshape(npair1, npair2)


def _convert_eri_gen(origsym, targetsym, eri, norb1, norb2):
    import ctypes

    from pyscf import lib

    libao2mo = lib.load_library("libao2mo")
    """
    #NOTE: IF YOU GET AN ERROR ABOUT THIS ATTRIBUTE:
    This requires a custom PySCF compilation

    Add to your PySCF and recompile: /pyscf/lib/ao2mo/restore_eri.c

void AO2MOrestore_nr4to1_gen(double *eri4, double *eri1, int norb1, int norb2)
{
        size_t npair1 = norb1*(norb1+1)/2;
        size_t npair2 = norb2*(norb2+1)/2;
        size_t i, j, ij;
        size_t d2 = norb2 * norb2;
        size_t d3 = norb1 * d2;

        for (ij = 0, i = 0; i < norb1; i++) {
        for (j = 0; j <= i; j++, ij++) {
                NPdunpack_tril(norb2, eri4+ij*npair2, eri1+i*d3+j*d2, HERMITIAN);
                if (i > j) {
                        memcpy(eri1+j*d3+i*d2, eri1+i*d3+j*d2,
                               sizeof(double)*d2);
                }
        } }
}

void AO2MOrestore_nr1to4_gen(double *eri1, double *eri4, int norb1, int norb2)
{
        size_t npair1 = norb1*(norb1+1)/2;
        size_t npair2 = norb2*(norb2+1)/2;
        size_t i, j, k, l, ij, kl;
        size_t d1 = norb2;
        size_t d2 = norb2 * norb2;
        size_t d3 = norb1 * d2;

        for (ij = 0, i = 0; i < norb1; i++) {
        for (j = 0; j <= i; j++, ij++) {
                for (kl = 0, k = 0; k < norb2; k++) {
                for (l = 0; l <= k; l++, kl++) {
                        eri4[ij*npair2+kl] = eri1[i*d3+j*d2+k*d1+l];
                } }
        } }
}
    """
    fn = getattr(libao2mo, "AO2MOrestore_nr%sto%s_gen" % (origsym, targetsym))

    if targetsym == 1:
        eri_out = numpy.empty([norb1, norb1, norb2, norb2])
    elif targetsym == 4:
        npair1 = norb1 * (norb1 + 1) // 2
        npair2 = norb2 * (norb2 + 1) // 2
        eri_out = numpy.empty([npair1, npair2])

    fn(
        eri.ctypes.data_as(ctypes.c_void_p),
        eri_out.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(norb1),
        ctypes.c_int(norb2),
    )

    return eri_out
