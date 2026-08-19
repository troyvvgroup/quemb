# Author(s): Oinam Romesh Meitei, Oskar Weser

from collections.abc import Sequence

import h5py
import numpy as np
import scipy.linalg
from numpy import (
    array,
    diag_indices,
    einsum,
    eye,
    float64,
    int64,
    outer,
    trace,
    tril_indices,
    zeros,
    zeros_like,
)
from numpy.linalg import eigh, multi_dot, svd

from quemb.molbe.helper import get_eri, get_scfObj, get_veff
from quemb.shared.external.lo_helper import cano_orth
from quemb.shared.helper import clean_overlap
from quemb.shared.typing import (
    FragmentIdx,
    GlobalAOIdx,
    Matrix,
    PathLike,
    RelAOIdx,
    RelAOIdxInRef,
    SeqOverEdge,
    Vector,
)


class Frags:
    """
    Class for handling fragments in bootstrap embedding.

    This class contains various functionalities required for managing and manipulating
    fragments for BE calculations.
    """

    def __init__(
        self,
        AO_in_frag: Sequence[GlobalAOIdx],
        ifrag: int,
        AO_per_edge: SeqOverEdge[Sequence[GlobalAOIdx]],
        ref_frag_idx_per_edge: SeqOverEdge[FragmentIdx],
        relAO_per_edge: SeqOverEdge[Sequence[RelAOIdx]],
        relAO_in_ref_per_edge: SeqOverEdge[Sequence[RelAOIdxInRef]],
        weight_and_relAO_per_center: tuple[float, Sequence[RelAOIdx]],
        relAO_per_origin: Sequence[RelAOIdx],
        eri_file: PathLike = "eri_file.h5",
        unrestricted: bool = False,
    ) -> None:
        """Constructor function for :python:`Frags` class.

        Parameters
        ----------
        AO_in_frag :
            list of AOs in the fragment (i.e. ``BE.AO_per_frag[i]``
            or ``FragPart.AO_per_frag[i]``)
        ifrag :
            fragment index (∈ [0, BE.n_frag - 1])
        AO_per_edge :
            list of lists of edge site AOs for each atom in the fragment.
            Read more detailed description in :class:`quemb.molbe.autofrag.FragPart`.
        ref_frag_idx_per_edge :
            list of fragment indices where edge site AOs are center site.
            Read more detailed description in :class:`quemb.molbe.autofrag.FragPart`.
        relAO_per_edge :
            list of lists of indices for edge site AOs within the fragment.
            Read more detailed description in :class:`quemb.molbe.autofrag.FragPart`.
        relAO_in_ref_per_edge :
            list of lists of indices within the fragment specified in :python:`center`
            that points to the edge site AOs.
            Read more detailed description in :class:`quemb.molbe.autofrag.FragPart`.
        weight_and_relAO_per_center :
            weight used for energy contributions and the indices.
            Read more detailed description in :class:`quemb.molbe.autofrag.FragPart`.
        relAO_per_origin :
            indices of the origin site atoms in the fragment
            Read more detailed description in :class:`quemb.molbe.autofrag.FragPart`.
        eri_file :
            two-electron integrals stored as h5py file, by default 'eri_file.h5'
        unrestricted :
            unrestricted calculation, by default False
        """

        self.AO_in_frag = AO_in_frag
        self.n_frag = len(AO_in_frag)
        self.AO_per_edge = AO_per_edge
        self.ref_frag_idx_per_edge = ref_frag_idx_per_edge
        self.relAO_per_edge = relAO_per_edge
        self.relAO_in_ref_per_edge = relAO_in_ref_per_edge
        self.relAO_per_origin = relAO_per_origin
        self.weight_and_relAO_per_center = weight_and_relAO_per_center
        self.eri_file = eri_file

        self.ifrag = ifrag
        if unrestricted:
            self.dname: str | list[str] = [
                "f" + str(ifrag) + "/aa",
                "f" + str(ifrag) + "/bb",
                "f" + str(ifrag) + "/ab",
            ]
        else:
            self.dname = "f" + str(ifrag)

        self.TA: Matrix[float64]
        self.frag_TA_offset: Vector[int64]
        self.TA_lo_eo: Matrix[float64]

        self.h1: Matrix[float64]
        self.nao: int
        self.mo_coeffs: Matrix[float64]
        self._mo_coeffs: Matrix[float64]
        self.nsocc: int
        self._mf = None
        self._mc = None

        # CCSD
        self.t1 = None
        self.t2 = None

        self.heff: Matrix[float64]
        self.udim: int | None = None

        self._rdm1 = None
        self.rdm1__ = None
        self.rdm2__ = None
        self.rdm1 = None
        self.genvs = None
        self.ebe = 0.0
        self.ebe_hf = 0.0
        self.fock = None
        self.veff = None
        self.veff0 = None
        self.dm_init = None
        self.dm0: Matrix[float64]
        # OTHER spin's dm0, re-projected into THIS fragment's own
        # embedding basis (set in UBE.initialize; no-op under
        # common_bath, required for correctness under equal_bath).
        self.dm_other_embedded: Matrix[float64] | None = None
        self.unitcell_nkpt = 1.0

    def sd(
        self,
        lao: Matrix[float64],
        lmo: Matrix[float64],
        nocc: int,
        thr_bath: float,
        norb: int | None = None,
        dm_override: Matrix[float64] | None = None,
    ) -> None:
        """
        Perform Schmidt decomposition for the fragment.

        Parameters
        ----------
        lao : numpy.ndarray
            Orthogonalized AOs
        lmo : numpy.ndarray
            Local molecular orbital coefficients.
        nocc : int
            Number of occupied orbitals.
        thr_bath : float,
            Threshold for bath orbitals in Schmidt decomposition
        norb : int, optional
            Specify number of bath orbitals.
            Used for UBE, where different number of alpha and beta orbitals
            Default is None, allowing orbitals to be chosen by threshold
        dm_override : numpy.ndarray, optional
            If provided, use this density matrix for Schmidt decomposition
            instead of computing it from lmo and nocc. Used for common bath
            construction where the total (alpha+beta) density matrix is used
            to produce a single set of bath orbitals shared by both spin
            channels. Default is None.
        """
        self.TA_lo_eo, self.n_f, self.n_b = schmidt_decomposition(
            lmo,
            nocc,
            self.AO_in_frag,
            thr_bath=thr_bath,
            norb=norb,
            rdm=dm_override,
        )
        self.TA = lao @ self.TA_lo_eo
        self.nao = self.TA.shape[1]

    def cons_fock(self, hf_veff, S, dm, eri_=None, dm_other=None):
        """
        Construct the Fock matrix for the fragment.

        When dm_other is provided, both dm and dm_other must be the true
        (undoubled) single-spin density matrices -- see get_veff() for why
        the doubled-density restricted formula doesn't apply here.
        dm_other=None preserves the original restricted-compatible
        behavior (dm is the doubled total density).

        Parameters
        ----------
        hf_veff : numpy.ndarray
            Hartree-Fock effective potential (this spin's channel, for the
            unrestricted case).
        S : numpy.ndarray
            Overlap matrix.
        dm : numpy.ndarray
            Density matrix for THIS spin channel. Undoubled if dm_other is
            provided; doubled total density if dm_other is None.
        eri_ : numpy.ndarray, optional
            Electron repulsion integrals, by default None.
        dm_other : numpy.ndarray, optional
            Undoubled density matrix for the OTHER spin channel. Default None.
        """

        if eri_ is None:
            eri_ = get_eri(
                self.dname,
                self.TA.shape[1],
                ignore_symm=True,
                eri_file=self.eri_file,
            )

        veff_, veff0 = get_veff(
            eri_, dm, S, self.TA, hf_veff, dm_other=dm_other
        )
        self.veff = veff_.real
        self.veff0 = veff0
        self.fock = self.h1 + veff_.real

    def get_nsocc(self, S, C, nocc, ncore=0):
        """
        Get the number of occupied orbitals for the fragment.

        Parameters
        ----------
        S : numpy.ndarray
            Overlap matrix.
        C : numpy.ndarray
            Molecular orbital coefficients.
        nocc : int
            Number of occupied orbitals.
        ncore : int, optional
            Number of core orbitals, by default 0.

        Returns
        -------
        numpy.ndarray
            Projected density matrix.

        Also sets self.nsocc (rounded integer) and self.nsocc_frac (the
        raw fractional projection before rounding, for diagnostics).
        """
        C_ = multi_dot((self.TA.T, S, C[:, ncore : ncore + nocc]))
        P_ = C_ @ C_.T
        nsocc_ = trace(P_)
        nsocc = int(round(nsocc_))
        try:
            mo_coeffs = scipy.linalg.svd(C_)[0]
        except scipy.linalg.LinAlgError:
            mo_coeffs = scipy.linalg.eigh(C_)[1][:, -nsocc:]

        self._mo_coeffs = mo_coeffs
        self.nsocc = nsocc
        self.nsocc_frac = float(getattr(nsocc_, "real", nsocc_))
        return P_

    def scf(
        self,
        heff=None,
        fs=False,
        eri=None,
        dm0=None,
        unrestricted=False,
        spin_ind=None,
        dm_other=None,
    ):
        """
        Perform self-consistent field (SCF) calculation for the fragment.

        dm_other is threaded through to get_scfObj() so the embedded SCF
        loop rebuilds the unrestricted potential at every iteration, rather
        than relying only on cons_fock's one-time static correction to h1.
        When dm_other is provided, dm0 defaults to the undoubled
        occupied-orbital projector instead of the doubled restricted one.

        Parameters
        ----------
        heff : numpy.ndarray, optional
            Effective Hamiltonian, by default None.
        fs : bool, optional
            Flag for full SCF, by default False.
        eri : numpy.ndarray, optional
            Electron repulsion integrals, by default None.
        dm0 : numpy.ndarray, optional
            Initial density matrix, by default None. UNDOUBLED if dm_other is
            provided (see fix note above).
        unrestricted : bool, optional
            Specify if unrestricted calculation, by default False
        spin_ind : int, optional
            Alpha (0) or beta (1) spin for unrestricted calculation, by default None
        dm_other : numpy.ndarray, optional
            Frozen density (undoubled) for the OTHER spin channel, in the same
            embedding basis as this fragment's TA (true for common_bath, where
            TA is shared between the alpha/beta Frags objects). Default None
            (restricted-compatible behavior).
        """

        if self._mf is not None:
            self._mf = None
        if self._mc is not None:
            self._mc = None
        if heff is None:
            heff = self.heff

        if eri is None:
            if unrestricted:
                dname = self.dname[spin_ind]
            else:
                dname = self.dname
            eri = get_eri(dname, self.nao, eri_file=self.eri_file)

        if dm0 is None:
            if dm_other is not None:
                # UNRESTRICTED FIX: undoubled default, matching the true
                # single-spin-channel convention dm_other also requires.
                dm0 = (
                    self._mo_coeffs[:, : self.nsocc]
                    @ self._mo_coeffs[:, : self.nsocc].conj().T
                )
            else:
                dm0 = 2.0 * (
                    self._mo_coeffs[:, : self.nsocc]
                    @ self._mo_coeffs[:, : self.nsocc].conj().T
                )

        mf_ = get_scfObj(
            self.fock + heff, eri, self.nsocc, dm0=dm0, dm_other=dm_other
        )
        if not fs:
            self._mf = mf_
            self.mo_coeffs = mf_.mo_coeff.copy()
        else:
            self._mo_coeffs = mf_.mo_coeff.copy()

    def update_heff(self, u, cout=None, only_chem=False):
        """Update the effective Hamiltonian for the fragment."""
        heff_ = zeros_like(self.h1)

        if cout is None:
            cout = self.udim

        for i, fi in enumerate(self.AO_in_frag):
            if not any(i in sublist for sublist in self.relAO_per_edge):
                heff_[i, i] -= u[-1]

        if only_chem:
            self.heff = heff_
            return
        else:
            for i in self.relAO_per_edge:
                for j in range(len(i)):
                    for k in range(len(i)):
                        if j > k:  # or j==k:
                            continue

                        heff_[i[j], i[k]] = u[cout]
                        heff_[i[k], i[j]] = u[cout]

                        cout += 1

            self.heff = heff_

    def set_udim(self, cout):
        for i in self.relAO_per_edge:
            for j in range(len(i)):
                for k in range(len(i)):
                    if j > k:
                        continue
                    cout += 1
        return cout

    def update_ebe_hf(
        self,
        rdm_hf=None,
        mo_coeffs=None,
        eri=None,
        return_e=False,
        unrestricted=False,
        spin_ind=None,
        dm_hf_other=None,
    ):
        if mo_coeffs is None:
            mo_coeffs = self._mo_coeffs

        if rdm_hf is None:
            rdm_hf = (
                mo_coeffs[:, : self.nsocc] @ mo_coeffs[:, : self.nsocc].conj().T
            )

        unrestricted_fac = 1.0 if unrestricted else 2.0

        e1 = unrestricted_fac * einsum(
            "ij,ij->i", self.h1[: self.n_frag], rdm_hf[: self.n_frag]
        )

        ec = (
            0.5
            * unrestricted_fac
            * einsum(
                "ij,ij->i", self.veff[: self.n_frag], rdm_hf[: self.n_frag]
            )
        )

        if self.TA.ndim == 3:
            jmax = self.TA[0].shape[1]
        else:
            jmax = self.TA.shape[1]
        if eri is None:
            with h5py.File(self.eri_file, "r") as f:
                if isinstance(self.dname, list):
                    eri = [f[self.dname[0]][()], f[self.dname[1]][()]]
                else:
                    eri = f[self.dname][()]

        # A single spin channel's HF cumulant uses coefficient 1 (each
        # orbital holds one electron), not the doubled-occupancy
        # restricted coefficient 2.
        same_spin_coeff = 1.0 if unrestricted else 2.0

        e2 = zeros_like(e1)
        for i in range(self.n_frag):
            for j in range(jmax):
                ij = i * (i + 1) // 2 + j if i > j else j * (j + 1) // 2 + i
                Gij = (
                    same_spin_coeff * rdm_hf[i, j] * rdm_hf
                    - outer(rdm_hf[i], rdm_hf[j])
                )[:jmax, :jmax]
                Gij[diag_indices(jmax)] *= 0.5
                Gij += Gij.T
                # unrestricted ERI file has 3 spin components: a, b, ab
                if unrestricted:
                    e2[i] += (
                        0.5
                        * unrestricted_fac
                        * Gij[tril_indices(jmax)]
                        @ eri[spin_ind][ij]
                    )
                else:
                    e2[i] += (
                        0.5
                        * unrestricted_fac
                        * Gij[tril_indices(jmax)]
                        @ eri[ij]
                    )

        # Cross-spin Coulomb correction. self.veff (used in ec above) is a
        # static snapshot that includes -J[P_other_0] (P_other_0: the other
        # spin's full-system density in this fragment's own TA basis), but
        # the embedded SCF's converged Fock also contributes +J[P_other_0];
        # these don't cancel in the cumulant energy expression, so the
        # missing +0.5*Tr[J[P_other_0]*rdm_hf] term is added explicitly
        # here. dm_hf_other must be P_other_0 (matching cons_fock's
        # dm_other convention), contracted against the same-spin ERI block.
        if unrestricted and dm_hf_other is not None:
            G_other = dm_hf_other[:jmax, :jmax].copy()
            G_other[diag_indices(jmax)] *= 0.5
            G_other = G_other + G_other.T
            G_other_packed = G_other[tril_indices(jmax)]
            for i in range(self.n_frag):
                for j in range(jmax):
                    ij = (
                        i * (i + 1) // 2 + j
                        if i > j
                        else j * (j + 1) // 2 + i
                    )
                    e2[i] += (
                        0.5
                        * unrestricted_fac
                        * rdm_hf[i, j]
                        * (G_other_packed @ eri[spin_ind][ij])
                    )

        e_ = e1 + e2 + ec
        etmp = 0.0
        for i in self.weight_and_relAO_per_center[1]:
            etmp += self.weight_and_relAO_per_center[0] * e_[i]

        self.ebe_hf = etmp

        if return_e:
            e_h1 = 0.0
            e_coul = 0.0
            for i in self.weight_and_relAO_per_center[1]:
                e_h1 += self.weight_and_relAO_per_center[0] * e1[i]
                e_coul += self.weight_and_relAO_per_center[0] * (e2[i] + ec[i])
            return (e_h1, e_coul, e1 + e2 + ec)
        else:
            return None


def schmidt_decomposition(
    mo_coeff: Matrix[float64],
    nocc: int,
    AO_in_frag: Sequence[GlobalAOIdx],
    thr_bath: float = 1.0e-10,
    cinv: Matrix[float64] | None = None,
    rdm: Matrix[float64] | None = None,
    norb: int | None = None,
) -> tuple[Matrix[float64], int, int]:
    """
    Perform Schmidt decomposition on the molecular orbital coefficients.

    This function decomposes the molecular orbitals into fragment and environment parts
    using the Schmidt decomposition method. It computes the transformation matrix (TA)
    which includes both the fragment orbitals and the entangled bath.

    Parameters
    ----------
    mo_coeff :
        Molecular orbital coefficients.
    nocc :
        Number of occupied orbitals.
    Frag_sites : list of int
        List of fragment sites (indices).
    thr_bath :
        Threshold for bath orbitals in Schmidt decomposition
    cinv :
        Inverse of the transformation matrix. Defaults to None.
    rdm :
        Reduced density matrix. If not provided, it will be computed from the molecular
        orbitals. Defaults to None.
    norb :
        Specifies number of bath orbitals. Used for UBE to make alpha and beta
        spaces the same size. Defaults to None

    Returns
    -------
    tuple:
        TA, norbs_frag, norbs_bath

        Transformation matrix (TA) including both fragment and entangled bath orbitals.
    """

    # Compute the reduced density matrix (RDM) if not provided
    if mo_coeff is not None:
        C = mo_coeff[:, :nocc]
    if rdm is None:
        Dhf = C @ C.T
        if cinv is not None:
            Dhf = multi_dot((cinv, Dhf, cinv.conj().T))
    else:
        Dhf = rdm

    # Total number of sites
    Tot_sites = Dhf.shape[0]

    # Identify environment sites (indices not in Frag_sites)
    Env_sites1 = array([i for i in range(Tot_sites) if i not in AO_in_frag])
    Env_sites = array([[i] for i in range(Tot_sites) if i not in AO_in_frag])
    Frag_sites1 = array([[i] for i in AO_in_frag])

    if len(Env_sites1) == 0:
        raise ValueError(
            f"Fragment contains all {Tot_sites} sites in the system — "
            "no environment remains, so the bath is undefined. This makes "
            "the BE calculation equivalent to running the full system "
            "without embedding. If this is intentional (e.g. testing "
            "against exact UCCSD), consider restructuring the fragmentation "
            "to avoid this edge case, or contact the dev if you believe "
            "this should be supported as a degenerate single-fragment case."
        )

    # Compute the environment part of the density matrix
    Denv = Dhf[Env_sites, Env_sites.T]

    # Perform eigenvalue decomposition on the environment density matrix
    Eval, Evec = eigh(Denv)

    # Identify significant environment orbitals based on eigenvalue threshold
    Bidx = []
    for i in range(len(Eval)):
        if thr_bath < np.abs(Eval[i]) < 1.0 - thr_bath:
            Bidx.append(i)
    # Set the number of orbitals to be taken from the environment orbitals
    # Based on an eigenvalue threshold ordering
    if norb is not None:
        # add extra orbital from environment
        # this will likely have Eval = 1
        # note: there are normally very few orbitals with a Eval[i] <= thr_bath,
        # so adding Bidx from the "front of the list" doesn't work. Instead, we add
        # Bidx corresponding to a high eigenvalue from the environment
        # (this is analagous to tightening up the threshold of the bath for the alpha
        # or beta orbitals until they are the same size)

        # Get all excluded indices sorted by distance from threshold
        # Prefer orbitals just above 1-thr_bath (nearly occupied environment)
        excluded = [i for i in range(len(Eval)) if i not in set(Bidx)]
        # Sort by eigenvalue descending — closest to 1 first
        excluded_sorted = sorted(excluded, key=lambda i: Eval[i], reverse=True)
        # Bidx corresponds to sorted Eval and Evec, so this simply adds indices
        # corresponding to larger eigenvectors until the bath size reaches norb
        # When bath size is reached, it will stop
        for idx in excluded_sorted:
            if len(Bidx) >= norb:
                break
            Bidx.append(idx)

    # Initialize the transformation matrix (TA)
    TA = zeros([Tot_sites, len(AO_in_frag) + len(Bidx)])
    TA[AO_in_frag, : len(AO_in_frag)] = eye(len(AO_in_frag))  # Fragment part
    TA[Env_sites1, len(AO_in_frag) :] = Evec[:, Bidx]  # Environment part

    return TA, Frag_sites1.shape[0], len(Bidx)


def schmidt_decomposition_common(
    mo_coeff_a,
    mo_coeff_b,
    nocc_a,
    nocc_b,
    AO_in_frag,
    thr_bath=1.0e-3,
):
    """
    Common bath Schmidt decomposition for unrestricted calculations.

    Constructs a single set of bath orbitals spanning both alpha and beta
    environment spaces via a multi-power SVD of P = P^alpha + P^beta,
    adapted from finite-temperature DMET. This produces one TA shared by
    both spin channels, making the alpha and beta embedding spaces
    identical.

    This is a prerequisite for the spin-summed 1RDM matching condition
    in iterative UBE (Tran, Ye, Van Voorhis, J. Chem. Phys. 153, 214101,
    2020, Eq. 15), which matches (P^alpha + P^beta) rather than P^alpha
    and P^beta separately. Consistent matching requires both spin channels
    to be expressed in the same orbital basis.

    Standard Schmidt decomposition (a single eigh/SVD) is only rigorous
    for an idempotent density matrix: once P.rho is in the Schmidt space,
    so is P^2.rho, P^3.rho, ... automatically, because they're all equal.
    P^alpha and P^beta are each individually idempotent but generally
    *different* wherever the system is spin-polarized, so their sum is
    NOT idempotent -- it has a small number of fractionally-occupied
    natural orbitals, and P.rho, P^2.rho, P^3.rho, ... are all distinct
    and all need to be in the Schmidt space for integer per-fragment
    electron counts to come out right. This function pools singular
    vectors from the fragment x environment block of P^n across powers
    n = 1, 2, 3, ... (not just n=1) and canonically orthogonalizes the
    pooled set, rather than thresholding a single SVD.

    Parameters
    ----------
    mo_coeff_a : ndarray, shape (n_LO, n_MO_a)
        Alpha occupied+virtual MO coefficients in LO basis (lmo_coeff_a).
    mo_coeff_b : ndarray, shape (n_LO, n_MO_b)
        Beta occupied+virtual MO coefficients in LO basis (lmo_coeff_b).
    nocc_a : int
        Number of occupied alpha orbitals.
    nocc_b : int
        Number of occupied beta orbitals.
    AO_in_frag : sequence of int
        LO indices belonging to this fragment.
    thr_bath : float
        Relative eigenvalue-ratio threshold used by the final canonical
        orthogonalization (cano_orth) to discard near-linearly-dependent
        pooled candidate directions -- this is the real threshold
        controlling bath size. Note this is a different quantity than
        the old single-SVD implementation's absolute per-spin occupation
        threshold of the same name; a given numeric value does not mean
        the same thing under both implementations, and the safe-default
        *direction* flips: for the old algorithm, a tiny/permissive
        value was safe (a single SVD has no pooled redundancy to filter).
        Here, pooling candidates across powers means a too-permissive
        threshold (e.g. 1e-10) lets near-duplicate directions from later
        powers through as if they were independent, inflating the bath
        and degrading both HF-in-HF consistency and per-fragment electron
        counts -- confirmed empirically on the hexene test (HF-in-HF
        error and bath size both worsen outside roughly [1e-4, 5e-3];
        1e-3 was the best value found and is the default here).

    Returns
    -------
    TA_lo_eo : ndarray, shape (n_LO, n_frag + n_bath)
        Transformation matrix in LO basis. First n_frag columns are the
        fragment identity block; remaining n_bath columns are the common
        bath orbitals.
    n_frag : int
        Number of fragment orbitals (= len(AO_in_frag)).
    n_bath : int
        Number of common bath orbitals.
    """
    n_LO = mo_coeff_a.shape[0]

    # Fragment and environment site indices in LO basis
    frag_set = set(AO_in_frag)
    frag_sites = list(AO_in_frag)
    env_sites = [i for i in range(n_LO) if i not in frag_set]

    if len(env_sites) == 0:
        raise ValueError(
            f"Fragment contains all {n_LO} sites in the system — "
            "no environment remains for common bath construction. "
            "This typically happens when chemgen's motif-merging collapses "
            "a small/star-shaped molecule into a single fragment. "
            "Consider using autogen fragmentation instead for this system."
        )
    assert len(frag_sites) > 0, (
        "AO_in_frag is empty; this should never happen -- upstream "
        "fragmentation always assigns at least one AO per fragment."
    )

    frag_idx = np.array(frag_sites)
    env_idx = np.array(env_sites)
    n_frag = len(frag_sites)
    n_env = len(env_sites)

    # P = D_alpha + D_beta, explicit N_LO x N_LO site-basis density.
    C_occ_a = mo_coeff_a[:, :nocc_a]
    C_occ_b = mo_coeff_b[:, :nocc_b]
    D_a = C_occ_a @ C_occ_a.T
    D_b = C_occ_b @ C_occ_b.T

    # D_a, D_b are each idempotent (eigenvalues in {0,1} in this
    # orthonormal LO basis), so P = D_a + D_b has eigenvalues in [0,2].
    # Normalizing by the exact Loewner bound 2.0 (cheap, exact -- no
    # extra eigh needed to estimate a spectral radius) keeps every
    # power's frag-env block O(1)-scaled, so a single fixed noise floor
    # is meaningful at every n. Without this, ordinary doubly-occupied-
    # in-both-spins directions (eigenvalue near 2, common, not rare)
    # would grow as 2^n under repeated powering while the genuinely
    # fractional directions we want shrink, and the "no new information"
    # stopping criterion below would never be well-posed. SVD singular
    # *vectors* are unit-norm regardless of a uniform input scaling, so
    # this rescaling has zero effect on which directions get selected.
    P_hat = (D_a + D_b) / 2.0

    SV_FLOOR = 1.0e-10  # pure noise floor on raw per-power singular
    # values, to keep roundoff-zero columns out of the candidate pool.
    # NOT the physical threshold -- that's thr_bath, applied below via
    # cano_orth.
    HARD_CAP = 8  # safety net; the rank-stall criterion below should
    # trigger well before this for physically reasonable systems.

    pooled_vecs = []
    bath_orbs = zeros((n_env, 0))
    P_power = P_hat.copy()  # P_hat ** 1
    prev_n_bath = -1
    n_powers_used = 0

    for n in range(1, HARD_CAP + 1):
        n_powers_used = n
        M_n = P_power[np.ix_(frag_idx, env_idx)]  # (n_frag, n_env) block
        _, S_n, Vt_n = svd(M_n, full_matrices=False)
        keep = S_n > SV_FLOOR
        if np.any(keep):
            pooled_vecs.append(Vt_n[keep, :].T)  # env-side right sing vecs

        if pooled_vecs:
            candidate_pool = np.hstack(pooled_vecs)
            bath_orbs = cano_orth(candidate_pool, thr=thr_bath, ovlp=None)

        cur_n_bath = bath_orbs.shape[1]
        if n > 1 and cur_n_bath == prev_n_bath:
            break  # no new independent bath direction from this power
        prev_n_bath = cur_n_bath
        # NOTE: deliberately no "cur_n_bath >= min(n_frag, n_env)" early
        # stop here. A single power's frag-env block generically already
        # has rank n_frag (env is essentially always bigger than frag),
        # so that check would fire after n=1 almost every time -- which
        # would silently collapse this back to the single-power behavior
        # this function is replacing. Pooling across multiple powers is
        # expected to span MORE than one power's rank alone (bath size
        # bound is ~2*n_frag + M, not n_frag -- see docstring/task spec).

        P_power = P_power @ P_hat  # advance to P_hat ** (n + 1)

    n_bath = int(bath_orbs.shape[1])

    print(
        f"  [common bath] n_env={n_env}, nocc_a={nocc_a}, nocc_b={nocc_b}, "
        f"powers_used={n_powers_used}"
    )
    print(f"  [common bath] pooled candidates={sum(v.shape[1] for v in pooled_vecs)}")
    print(f"  [common bath] kept as bath after canonical orthogonalization: {n_bath}")

    # Build transformation matrix in LO basis:
    # [ I_frag |  0    ]   fragment block (identity)
    # [   0    | U_bath]   environment block (common bath orbitals)
    TA_lo_eo = zeros((n_LO, n_frag + n_bath))
    TA_lo_eo[frag_sites, :n_frag] = eye(n_frag)
    TA_lo_eo[env_sites, n_frag:] = bath_orbs

    return TA_lo_eo, n_frag, n_bath


def _get_contained(
    all_fragment_MOs_TA: Matrix[np.float64],
    TA: Matrix[np.float64],
    S: Matrix[np.float64],
    epsilon: float,
) -> Vector[np.bool]:
    r"""Get a boolean vector of the MOs in TA that are already contained in
    ``all_fragment_MOs_TA``

    Parameters
    ----------
    all_fragment_MOs_TA :
        A :math:`n_{\text{AO}} \times n_{\text{f,all}}` matrix that
        contains the fragment orbitals of all fragments.
    TA :
        A :math:`n_{\text{AO}} \times n_{\text{f}}` matrix that
        contains the fragment orbitals of a given fragment.
    S :
        The AO overlap matrix.
    epsilon :
        Cutoff to consider overlap values to be zero or one.
    """
    return (
        clean_overlap(all_fragment_MOs_TA.T @ S @ TA, epsilon=epsilon) == 1
    ).any(axis=0)


def _get_union_of_fragment_MOs(
    schmidt_TAs: Sequence[Matrix[np.float64]],
    S: Matrix[np.float64],
    epsilon: float,
) -> Matrix[np.float64]:
    all_fragment_MOs_TA = schmidt_TAs[0]
    for schmidt_TA in schmidt_TAs[1:]:
        all_fragment_MOs_TA = np.hstack(
            (
                all_fragment_MOs_TA,
                schmidt_TA[
                    :,
                    ~_get_contained(
                        all_fragment_MOs_TA, schmidt_TA, S, epsilon
                    ),
                ],
            )
        )
    return all_fragment_MOs_TA


def _get_index_offset(
    all_fragment_MOs_TA: Matrix[np.float64],
    TA: Matrix[np.float64],
    S: Matrix[np.float64],
    epsilon: float,
) -> Vector[np.int64]:
    idx_rows, idx_cols = (
        clean_overlap(all_fragment_MOs_TA.T @ S @ TA, epsilon) == 1
    ).nonzero()
    new_idx = np.argsort(idx_cols)
    idx_rows, idx_cols = idx_rows[new_idx], idx_cols[new_idx]
    assert (idx_cols == np.arange(TA.shape[1])).all()
    return idx_rows


def union_of_frag_MOs_and_index(
    Fobjs: Sequence[Frags], S: Matrix[np.float64], epsilon: float = 1e-10
) -> tuple[Matrix[np.float64], list[Vector[np.int64]]]:
    r"""Get the union of all fragment MOs as one Matrix and the respective
    indices for each fragment to refer to the global fragment MO matrix.

    This allows to reuse information such as integrals for the fragment MOs.

    Parameters
    ----------
    Fobjs:
        A sequence of Frags.
    S :
        The AO overlap matrix.
    epsilon :
        Cutoff to consider overlap values to be zero or one.
    """
    fragment_TAs = [fobj.TA[:, : fobj.n_f] for fobj in Fobjs]
    all_fragment_MOs_TA = _get_union_of_fragment_MOs(
        fragment_TAs, S, epsilon=epsilon
    )
    return all_fragment_MOs_TA, [
        _get_index_offset(all_fragment_MOs_TA, schmidt_TA, S, epsilon=epsilon)
        for schmidt_TA in fragment_TAs
    ]
