import logging

from numpy import abs, complex128, sqrt, zeros
from pyscf import lib
from pyscf.ao2mo.addons import restore
from pyscf.pbc.df.df import make_auxcell, make_modrho_basis
from pyscf.pbc.df.ft_ao import ft_ao, ft_aopair
from pyscf.pbc.df.gdf_builder import _CCGDFBuilder
from pyscf.pbc.df.incore import aux_e2
from pyscf.pbc.tools import get_coulG
from scipy.linalg import LinAlgError, cholesky, eigh, solve_triangular

from quemb.molbe.eri_onthefly import block_step_size

logger = logging.getLogger(__name__)


def _j2c_cholesky_or_eig(j2c):
    r"""PBC DF Metric (P|Q) can be non-positive-definite.
    In this case, we want to fallback to eigenvalue decomposition.

    Parameters
    ----------
    j2c : ndarray
        (P|Q) metric matrix

    Returns
    -------
    j2c : ndarray
        If Cholesky was successful, lower triangle L such that L @ L.T = (P|Q)
        If fallback to eigenvalue decomposition, V d^{-1/2} V.T
    ischol : bool
        Whether Cholesky decomposition was successful or not.
    """
    try:
        low = cholesky(j2c, lower=True)
        logger.debug("(P|Q) positive definite. Cholesky decomposition successful.")
        return low, True
    except LinAlgError:
        logger.debug(
            "Cholesky decomposition of (P|Q) failed. Falling back to eigenvalue"
            "decomposition."
        )
        d, V = eigh(j2c)  # V d V.T = (P|Q)
        return V[:, d > 1e-14] / sqrt(d[d > 1e-14]) @ V[:, d > 1e-14].conj().T, False


def integral_direct_DF(mf, Fobjs, file_eri, auxbasis=None):
    r"""Calculate AO density-fitted 3-center integrals on-the-fly and transform to
    Schmidt space for given fragment objects

    Parameters
    ----------
    mf : pyscf.pbc.scf.khf.KRHF
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

    def calculate_Lpq_pbcRS(aux_range):
        r"""Internal function to calculate the 3-center integrals for a given range of
        auxiliary indices for periodic systems (with charge compensation in real space)

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
            mf.cell.nbas,
            0,
            mf.cell.nbas,
            p0,
            p1,
        )  # for pbc, we use aux_e2, which doesn't take in concatenated env

        ints = aux_e2(
            mf.cell,
            auxcell,
            mf.cell._add_suffix("int3c2e"),
            "s1",
            # kptij_list = , TODO: add kptij_list for k-point sampling
            shls_slice=shls_slice,
        ) - aux_e2(
            mf.cell,
            chgcell,
            mf.cell._add_suffix("int3c2e"),
            "s1",
            shls_slice=shls_slice,
        )  # Remove the part calculated in the reciprocal space to avoid double counting

        assert ints.shape[0] == mf.cell.nao**2

        logger.debug("Finish calculating (μν|P) for range %s", aux_range)

        # (nao*nao, naux) -> (naux, nao, nao)
        return ints.T.reshape(-1, mf.cell.nao, mf.cell.nao)

    def calculate_Gpq_pbcFS(pw_range):
        r"""Internal function to calculate the 3-center integrals for a given range of
        plane wave indices for periodic systems (in Fourier space)

        Parameters
        ----------
        pw_range : tuple of int
            (start index, end index) of the plane wave basis functions
            to calculate the 3-center integrals, i.e.
            (:math:`(pq|G)`) with G :math:`\in [start, end)` is returned
        """
        logger.debug("Start calculating (μν|G) for range %s", pw_range)
        g0, g1 = pw_range
        Gv_batch = Gv[g0:g1]
        coulG_batch = coulG[g0:g1]

        ft_aoao_batch = ft_aopair(mf.cell, Gv_batch)  # (G, nao, nao) # TODO kpts

        ints = ft_aoao_batch * coulG_batch.reshape(-1, 1, 1).conj()
        logger.debug("Finish calculating (μν|G) for range %s", pw_range)

        return ints  # (G, nao, nao)

    logger.info("Evaluating fragment ERIs on-the-fly using density fitting...")
    logger.info(
        "In this case, note that HF-in-HF error includes DF error on top of "
        "numerical error from embedding."
    )

    auxcell = make_auxcell(mf.cell, auxbasis)
    chgcell = make_modrho_basis(mf.cell, auxbasis)

    ccgdfbuilder = _CCGDFBuilder(mf.cell, auxcell, mf.kpts)
    ccgdfbuilder.build()

    # Prepare plane waves to evaluate in the reciprocal space for long-range contrib
    Gv, Gv_weights, kws = mf.cell.get_Gv_weights(ccgdfbuilder.mesh)
    coulG = get_coulG(mf.cell, mesh=ccgdfbuilder.mesh)  # kpt, exxdiv TODO
    coulG *= kws

    # Prepare storage for (pq|L) and (pq|G)
    pqL_frag = [
        zeros((auxcell.nao, fragobj.nao, fragobj.nao), dtype=complex128)
        for fragobj in Fobjs
    ]  # place to store fragment (pq|L)

    j2c = ccgdfbuilder.get_2c2e(
        zeros((1, 3))
    )  # (P|Q) accounting for periodic images and G=0 divergence
    # TODO: reexamine kpoint, as (P|Q) depends on momentum transfer
    j2c, ischol = _j2c_cholesky_or_eig(j2c[0])  # TODO: kpt
    # If ischol, j2c is triangular matrix from Cholesky decomposition
    # Otherwise, this is (P|Q)^(-1/2) from eigendecomposition
    # We intentionally reuse j2c name because this variable has the memory scaling
    # of (N_AO^2), which is how this ERI routine scales memory-wise.

    end = 0
    Granges = [
        (x, y)
        for x, y in lib.prange(
            0,
            len(Gv),
            block_step_size(len(Fobjs), len(Gv), mf.cell.nao, datatype=complex128),
        )
    ]

    for idx, ints in enumerate(lib.map_with_prefetch(calculate_Gpq_pbcFS, Granges)):
        logger.debug("Calculating pq|G block #%d %s", idx, Granges[idx])
        # Transform pq (AO) to fragment space (ij)
        start = end
        end += ints.shape[0]

        # Evaluate (L|G)
        ft_auxL_Gbatch = ft_ao(chgcell, Gv[start:end])
        ft_auxL_Gbatch = ft_auxL_Gbatch.conj().T

        for fragidx in range(len(Fobjs)):
            logger.debug("(μν|G) -> (ij|G) for frag #%d", fragidx)
            Gqi = ints @ Fobjs[fragidx].TA
            Giq = Gqi.transpose(0, 2, 1)
            Gij = Giq @ Fobjs[fragidx].TA.conj()

            # add in (L|G) (G|ij), the reciprocal space contribution
            pqL_frag[fragidx] += (
                ft_auxL_Gbatch @ Gij.reshape(ints.shape[0], -1)
            ).reshape(auxcell.nao, Fobjs[fragidx].nao, Fobjs[fragidx].nao)

    end = 0
    blockranges = [
        (x, y)
        for x, y in lib.prange(
            0, auxcell.nbas, block_step_size(len(Fobjs), auxcell.nbas, mf.cell.nao)
        )
    ]
    logger.debug("Aux Basis Block Info: %s", blockranges)

    for idx, ints in enumerate(lib.map_with_prefetch(calculate_Lpq_pbcRS, blockranges)):
        logger.debug("Calculating pq|L block #%d %s", idx, blockranges[idx])
        # Transform pq (AO) to fragment space (ij)
        start = end
        end += ints.shape[0]

        for fragidx in range(len(Fobjs)):
            logger.debug("(μν|P) -> (ij|P) for frag #%d", fragidx)
            Lqi = ints @ Fobjs[fragidx].TA
            Liq = Lqi.transpose(0, 2, 1)
            Lij = Liq @ Fobjs[fragidx].TA.conj()

            # Add in contributions from the reciprocal space
            pqL_frag[fragidx][start:end, :, :] += Lij

    # Fit to get B_{ij}^{L}
    for fragidx in range(len(Fobjs)):
        logger.debug("Fitting B_{ij}^{L} for frag #%d", fragidx)
        b = pqL_frag[fragidx].reshape(auxcell.nao, -1)
        if ischol:
            bb = solve_triangular(
                j2c, b, lower=True, overwrite_b=True, check_finite=False
            )
        else:
            bb = j2c @ b
        logger.debug("Finished obtaining B_{ij}^{L} for frag #%d", fragidx)
        eri_nosym = bb.T @ bb
        if (abs(eri_nosym.imag) > 1e-6).any():
            raise ValueError(
                f"Imaginary part of ERI is larger than 1e-6 for frag #{fragidx}."
            )
        else:
            eri_nosym = eri_nosym.real
        eri = restore("4", eri_nosym, Fobjs[fragidx].nao)  # TODO kpt
        file_eri.create_dataset(Fobjs[fragidx].dname, data=eri)
