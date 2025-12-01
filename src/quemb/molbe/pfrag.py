# Author(s): Oinam Romesh Meitei, Oskar Weser
from __future__ import annotations
from collections.abc import Sequence
from typing import Literal, assert_never

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
from numpy.linalg import eigh, multi_dot

from quemb.molbe.helper import get_eri, get_scfObj, get_veff
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


def procrustes_right(
    P: Matrix[np.floating], Q: Matrix[np.floating]
) -> Matrix[np.float64]:
    """Solve min || P R - Q ||_F subject to R^T R = I.

    Parameters
    ----------
    P, Q : (m, n) arrays
        Corresponding point sets as row vectors.

    Returns
    -------
    R : (n, n) array
        Optimal orthogonal matrix.
    """
    H = P.T @ Q
    U, S, Vt = np.linalg.svd(H)
    return U @ Vt


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
        eq_fobj: Frags | None = None,
        TA_occ: np.ndarray | None = None,
        TA_virt: np.ndarray | None = None,
        eigvecs: np.ndarray | None = None,
        TA_lo_eo_frag: np.ndarray | None = None,
        TA_lo_eo_bath: np.ndarray | None = None,
    ) -> None:
        r"""Constructor function for :python:`Frags` class.

        Parameters
        ----------
        AO_in_frag :
            list of AOs in the fragment (i.e. ``BE.AO_per_frag[i]``
            or ``FragPart.AO_per_frag[i]``)
        ifrag :
            fragment index (:math:`\in [0, \text{BE.n\_frag} - 1]`)
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
        self.eq_fobj = eq_fobj
        self.TA_occ = TA_occ
        self.TA_virt = TA_virt
        self.eigvecs = eigvecs
        self.TA_lo_eo_frag = TA_lo_eo_frag
        self.TA_lo_eo_bath = TA_lo_eo_bath
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
        self.unitcell_nkpt = 1.0

    def sd(
        self,
        lao: Matrix[float64],
        lmo: Matrix[float64],
        nocc: int,
        thr_bath: float,
        gradient_orb_space: Literal[
            "RDM-invariant", "Schmidt-invariant", "Unmodified"
        ] = "RDM-invariant",
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
        """
        if self.eq_fobj is None:
            print("must be the equilibrium calculation")
            (
                self.Dhf,
                self.TA_lo_eo,
                self.TAenv_lo_eo,
                self.TAfull_lo_eo,
                self.n_f,
                self.n_b,
            ) = schmidt_decomposition(
                lmo,
                nocc,
                self.AO_in_frag,
                thr_bath=thr_bath,
            )
        else:
            if gradient_orb_space == "Unmodified":
                (
                    self.Dhf,
                    self.TA_lo_eo,
                    self.TAenv_lo_eo,
                    self.TAfull_lo_eo,
                    self.n_f,
                    self.n_b,
                ) = schmidt_decomposition(
                    lmo,
                    nocc,
                    self.AO_in_frag,
                    thr_bath=thr_bath,
                )
                pass
            elif gradient_orb_space == "RDM-invariant":
                assert self.eq_fobj is not None
                assert self.eq_fobj.TA_occ is not None
                assert self.eq_fobj.TA_virt is not None
                print("doing rdm invariant rotation")
                TA_occ = lmo[:, :nocc] @ procrustes_right(
                    lmo[:, :nocc], self.eq_fobj.TA_occ
                )
                TA_virt = lmo[:, nocc:] @ procrustes_right(
                    lmo[:, nocc:], self.eq_fobj.TA_virt
                )
                TA = np.hstack([TA_occ, TA_virt])
                assert self.eq_fobj.eigvecs is not None
                TAfull = TA @ self.eq_fobj.eigvecs.T
                self.TA_lo_eo = TAfull[:, : self.eq_fobj.n_f + self.eq_fobj.n_b]
                # self.TAenv_lo_eo = TAfull[:, self.eq_fobj.n_f + self.eq_fobj.n_b:]
                self.n_f = self.eq_fobj.n_f
                self.Dhf = lmo[:, :nocc] @ lmo[:, :nocc].T
            elif gradient_orb_space == "Schmidt-invariant":
                print("doing schmidt invariant rotation")
                assert self.eq_fobj.TA_lo_eo_frag is not None
                assert self.eq_fobj.TA_lo_eo_bath is not None
                TA_frag = self.TA_lo_eo[:, : self.n_f] @ procrustes_right(
                    self.TA_lo_eo[:, : self.n_f], self.eq_fobj.TA_lo_eo_frag
                )
                TA_bath = self.TA_lo_eo[:, self.n_f :] @ procrustes_right(
                    self.TA_lo_eo[:, self.n_f :], self.eq_fobj.TA_lo_eo_bath
                )
                self.TA_lo_eo = np.hstack([TA_frag, TA_bath])
            else:
                assert_never(gradient_orb_space)

        self.TA = lao @ self.TA_lo_eo  # (ao by lo) x (lo by so) = (ao by so)
        self.nao = self.TA.shape[1]

    def cons_fock(self, hf_veff, S, dm, eri_=None):
        """
        Construct the Fock matrix for the fragment.

        Parameters
        ----------
        hf_veff : numpy.ndarray
            Hartree-Fock effective potential.
        S : numpy.ndarray
            Overlap matrix.
        dm : numpy.ndarray
            Density matrix.
        eri_ : numpy.ndarray, optional
            Electron repulsion integrals, by default None.
        """

        if eri_ is None:
            eri_ = get_eri(
                self.dname, self.TA.shape[1], ignore_symm=True, eri_file=self.eri_file
            )

        veff_, veff0 = get_veff(eri_, dm, S, self.TA, hf_veff)
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
        """
        C_ = multi_dot((self.TA.T, S, C[:, ncore : ncore + nocc]))
        P_ = C_ @ C_.T
        nsocc_ = trace(P_)
        nsocc = int(round(nsocc_))
        try:
            mo_coeffs = scipy.linalg.svd(C_, lapack_driver="gesvd")[0]
        except scipy.linalg.LinAlgError:
            mo_coeffs = scipy.linalg.eigh(C_)[1][:, -nsocc:]

        self._mo_coeffs = mo_coeffs
        self.nsocc = nsocc
        return P_

    def scf(
        self, heff=None, fs=False, eri=None, dm0=None, unrestricted=False, spin_ind=None
    ):
        """
        Perform self-consistent field (SCF) calculation for the fragment.

        Parameters
        ----------
        heff : numpy.ndarray, optional
            Effective Hamiltonian, by default None.
        fs : bool, optional
            Flag for full SCF, by default False.
        eri : numpy.ndarray, optional
            Electron repulsion integrals, by default None.
        dm0 : numpy.ndarray, optional
            Initial density matrix, by default None.
        unrestricted : bool, optional
            Specify if unrestricted calculation, by default False
        spin_ind : int, optional
            Alpha (0) or beta (1) spin for unrestricted calculation, by default None
        """

        if self._mf is not None:  # does not execute
            self._mf = None
        if self._mc is not None:  # does not execute
            self._mc = None
        if heff is None:  # executes
            heff = self.heff

        if eri is None:  # executes
            if unrestricted:
                dname = self.dname[spin_ind]
            else:
                dname = self.dname
            eri = get_eri(dname, self.nao, eri_file=self.eri_file)

        if dm0 is None:  # executes
            dm0 = 2.0 * (
                self._mo_coeffs[:, : self.nsocc]
                @ self._mo_coeffs[:, : self.nsocc].conj().T
            )

        mf_ = get_scfObj(self.fock + heff, eri, self.nsocc, dm0=dm0)
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
    ):
        if mo_coeffs is None:
            mo_coeffs = self._mo_coeffs

        if rdm_hf is None:
            rdm_hf = mo_coeffs[:, : self.nsocc] @ mo_coeffs[:, : self.nsocc].conj().T

        unrestricted_fac = 1.0 if unrestricted else 2.0

        e1 = unrestricted_fac * einsum(
            "ij,ij->i", self.h1[: self.n_frag], rdm_hf[: self.n_frag]
        )

        ec = (
            0.5
            * unrestricted_fac
            * einsum("ij,ij->i", self.veff[: self.n_frag], rdm_hf[: self.n_frag])
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

        e2 = zeros_like(e1)
        for i in range(self.n_frag):
            for j in range(jmax):
                ij = i * (i + 1) // 2 + j if i > j else j * (j + 1) // 2 + i
                Gij = (2.0 * rdm_hf[i, j] * rdm_hf - outer(rdm_hf[i], rdm_hf[j]))[
                    :jmax, :jmax
                ]
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
                    e2[i] += 0.5 * unrestricted_fac * Gij[tril_indices(jmax)] @ eri[ij]

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
):
    """
    Perform Schmidt decomposition on the molecular orbital coefficients.

    This function decomposes the molecular orbitals into fragment and environment parts
    using the Schmidt decomposition method. It computes the transformation matrix (TA)
    which includes both the fragment orbitals and the entangled bath.

    Parameters
    ----------
    mo_coeff :
        Local molecular orbital coefficients, lo by mo
    nocc :
        Number of occupied orbitals in the full system
    Frag_sites : list of int
        List of fragment sites (indices).
    thr_bath :
        Threshold for bath orbitals in Schmidt decomposition
    cinv :
        Inverse of the transformation matrix. Defaults to None.
    rdm :
        Reduced density matrix. If not provided, it will be computed from the molecular
        orbitals. Defaults to None.

    Returns
    -------
    tuple:
        TA, norbs_frag, norbs_bath

        Transformation matrix (TA) including both fragment and entangled bath orbitals.
    """

    if mo_coeff is not None:
        C = mo_coeff[
            :, :nocc
        ]  # happens, just take the occupied part (which are the first nocc columns)
    if rdm is None:
        Dhf = C @ C.T  # happens, lo by lo
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

    # Compute the environment part of the density matrix
    Denv = Dhf[Env_sites, Env_sites.T]

    # Perform eigenvalue decomposition on the environment density matrix
    Eval, Evec = eigh(Denv)

    # Reverse order: largest → smallest
    Eval = Eval[::-1]
    Evec = Evec[:, ::-1]

    # Identify significant environment orbitals based on eigenvalue threshold
    Bidx = []
    Eidx = []

    # Set the number of orbitals to be taken from the environment orbitals
    # Based on an eigenvalue threshold ordering
    for i in range(len(Eval)):
        if thr_bath < np.abs(Eval[i]) < 1.0 - thr_bath:
            Bidx.append(i)
        else:
            Eidx.append(i)

    # Initialize the fragment + bath TA matrix
    TA = zeros([Tot_sites, len(AO_in_frag) + len(Bidx)])
    TA[AO_in_frag, : len(AO_in_frag)] = eye(len(AO_in_frag))  # Fragment part
    TA[Env_sites1, len(AO_in_frag) :] = Evec[:, Bidx]  # Bath part

    # Initialize the environment TA matrix
    if len(Eidx) == 0:
        print("no environment")
    TAenv = zeros([Tot_sites, len(Eidx)])
    TAenv[Env_sites1, :] = Evec[:, Eidx]

    TAfull = np.zeros((Tot_sites, len(AO_in_frag) + len(Bidx) + len(Eidx)))
    TAfull[AO_in_frag, : len(AO_in_frag)] = np.eye(len(AO_in_frag))
    TAfull[Env_sites1, len(AO_in_frag) : len(AO_in_frag) + len(Bidx)] = Evec[:, Bidx]
    TAfull[Env_sites1, len(AO_in_frag) + len(Bidx) :] = Evec[:, Eidx]

    return (
        Dhf,
        TA,
        TAenv,
        TAfull,
        Frag_sites1.shape[0],
        len(Bidx),
    )


def _get_contained(
    all_fragment_MOs_TA: Matrix[np.float64],
    TA: Matrix[np.float64],
    S: Matrix[np.float64],
    epsilon: float,
) -> Vector[bool]:
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
    return (clean_overlap(all_fragment_MOs_TA.T @ S @ TA, epsilon=epsilon) == 1).any(
        axis=0
    )


def _get_union_of_fragment_MOs(
    schmidt_TAs: Sequence[Matrix[np.float64]], S: Matrix[np.float64], epsilon: float
) -> Matrix[np.float64]:
    all_fragment_MOs_TA = schmidt_TAs[0]
    for schmidt_TA in schmidt_TAs[1:]:
        all_fragment_MOs_TA = np.hstack(
            (
                all_fragment_MOs_TA,
                schmidt_TA[
                    :, ~_get_contained(all_fragment_MOs_TA, schmidt_TA, S, epsilon)
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
    all_fragment_MOs_TA = _get_union_of_fragment_MOs(fragment_TAs, S, epsilon=epsilon)
    return all_fragment_MOs_TA, [
        _get_index_offset(all_fragment_MOs_TA, schmidt_TA, S, epsilon=epsilon)
        for schmidt_TA in fragment_TAs
    ]
