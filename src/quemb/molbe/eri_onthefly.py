# Author(s): Minsik Cho, Hong-Zhou Ye

import logging

from numpy import moveaxis, transpose, zeros
from pyscf import lib
from pyscf.ao2mo.addons import restore
from pyscf.df.addons import make_auxmol
from pyscf.gto import mole
from pyscf.gto.moleintor import getints3c, make_cintopt, make_loc
from scipy.linalg import cholesky, solve_triangular

from quemb.shared.config import settings

logger = logging.getLogger(__name__)


def integral_direct_DF(mf, Fobjs, file_eri, auxbasis=None):
    """Calculate AO density-fitted 3-center integrals on-the-fly and transform to
    Schmidt space for given fragment objects

    Parameters
    ----------
    mf : pyscf.scf.hf.RHF
        Mean-field object for the chemical system (typically BE.mf)
    Fobjs : list of quemb.molbe.autofrag.FragPart
        List containing fragment objects (typically BE.Fobjs)
        The MO coefficients are taken from Frags.TA and the transformed ERIs are stored
        in Frags.dname as h5py datasets.
    file_eri : h5py.File
        HDF5 file object to store the transformed fragment ERIs
    auxbasis : str, optional
        Auxiliary basis used for density fitting. If not provided, use pyscf's default
        choice for the basis set used to construct mf object; by default None
    """

    def calculate_pqL(aux_range):
        """Internal function to calculate the 3-center integrals for a given range of
        auxiliary indices

        Parameters
        ----------
        aux_range : tuple of int
            (start index, end index) of the auxiliary basis functions
            to calculate the 3-center integrals, i.e.
            (:math:`(pq|L)`) with L :math:`\in [start, end)` is returned
        """
        logger.debug("Start calculating (μν|P) for range %s", aux_range)
        p0, p1 = aux_range
        shls_slice = (
            0,
            mf.mol.nbas,
            0,
            mf.mol.nbas,
            mf.mol.nbas + p0,
            mf.mol.nbas + p1,
        )
        ints = getints3c(
            mf.mol._add_suffix("int3c2e"),
            atm,
            bas,
            env,
            shls_slice,
            1,
            "s1",
            ao_loc,
            cintopt,
            out=None,
        )
        logger.debug("Finish calculating (μν|P) for range %s", aux_range)
        return ints

    def block_step_size(nfrag, naux, nao):
        """Internal function to calculate the block step size for the 3-center
        integrals calculation

        Parameters
        ----------
        nfrag : int
            Number of fragments
        naux : int
            Number of auxiliary basis functions
        nao : int
            Number of atomic orbitals
        """
        return max(
            1,
            int(
                settings.INTEGRAL_TRANSFORM_MAX_MEMORY
                * 1e9
                / 8
                / nao
                / nao
                / naux
                / nfrag
            ),
        )  # max(int(500*.24e6/8/nao),1)

    logger.info("Evaluating fragment ERIs on-the-fly using density fitting...")
    logger.info(
        "In this case, note that HF-in-HF error includes DF error on top of "
        "numerical error from embedding."
    )

    auxmol = make_auxmol(mf.mol, auxbasis=auxbasis)
    j2c = auxmol.intor(mf.mol._add_suffix("int2c2e"), hermi=1)  # (L|M)
    low = cholesky(j2c, lower=True)
    pqL_frag = [
        zeros((auxmol.nao, fragobj.nao, fragobj.nao)) for fragobj in Fobjs
    ]  # place to store fragment (pq|L)
    end = 0
    atm, bas, env = mole.conc_env(
        mf.mol._atm, mf.mol._bas, mf.mol._env, auxmol._atm, auxmol._bas, auxmol._env
    )
    ao_loc = make_loc(bas, mf.mol._add_suffix("int3c2e"))
    cintopt = make_cintopt(atm, bas, env, mf.mol._add_suffix("int3c2e"))
    blockranges = [
        (x, y)
        for x, y in lib.prange(
            0, auxmol.nbas, block_step_size(len(Fobjs), auxmol.nbas, mf.mol.nao)
        )
    ]
    logger.debug("Aux Basis Block Info: %s", blockranges)

    for idx, ints in enumerate(lib.map_with_prefetch(calculate_pqL, blockranges)):
        logger.debug("Calculating pq|L block #%d %s", idx, blockranges[idx])
        # Transform pq (AO) to fragment space (ij)
        start = end
        end += ints.shape[2]
        for fragidx in range(len(Fobjs)):
            logger.debug("(μν|P) -> (ij|P) for frag #%d", fragidx)
            Lqp = transpose(ints, axes=(2, 1, 0))
            Lqi = Lqp @ Fobjs[fragidx].TA
            Liq = moveaxis(Lqi, 2, 1)
            pqL_frag[fragidx][start:end, :, :] = Liq @ Fobjs[fragidx].TA
    # Fit to get B_{ij}^{L}
    for fragidx in range(len(Fobjs)):
        logger.debug("Fitting B_{ij}^{L} for frag #%d", fragidx)
        b = pqL_frag[fragidx].reshape(auxmol.nao, -1)
        bb = solve_triangular(low, b, lower=True, overwrite_b=True, check_finite=False)
        logger.debug("Finished obtaining B_{ij}^{L} for frag #%d", fragidx)
        eri_nosym = bb.T @ bb
        eri = restore("4", eri_nosym, Fobjs[fragidx].nao)
        file_eri.create_dataset(Fobjs[fragidx].dname, data=eri)
