# These are all Alexa's functions, copied here for ease of use
# TODO: once Alexa's code is merged to main, delete these and point to Alexa's functions

import logging
import os
import pickle
from typing import Dict, Final, List, Literal, TypeAlias
from warnings import warn

import h5py
import numpy as np
from attrs import define
from numpy import (
    allclose,
    argmax,
    array,
    concatenate,
    delete,
    diag,
    diag_indices,
    einsum,
    eye,
    float64,
    floating,
    hstack,
    isnan,
    load,
    ndarray,
    save,
    shape,
    sqrt,
    where,
    zeros,
    zeros_like,
)
from numpy.linalg import eigh, multi_dot, norm, svd
from pyscf import ao2mo, scf
from pyscf.gto import Mole
from scipy.optimize import linear_sum_assignment
from typing_extensions import assert_never

from quemb.molbe.be_parallel import be_func_parallel
from quemb.molbe.eri_onthefly import integral_direct_DF
from quemb.molbe.eri_sparse_DF import (
    transform_sparse_DF_integral_cpu,
)
from quemb.molbe.fragment import FragPart
from quemb.molbe.lo import (
    IAO_LocMethods,
    LocMethods,
    get_iao,
    get_loc,
    get_pao,
    get_xovlp,
    remove_core_mo,
)
from quemb.molbe.misc import print_energy_cumulant, print_energy_noncumulant
from quemb.molbe.opt import BEOPT
from quemb.molbe.pfrag import Frags, union_of_frag_MOs_and_index
from quemb.molbe.solver import Solvers, UserSolverArgs, be_func
# from quemb.shared.external.eom_qchem_parser import dyson_parser
from quemb.shared.external.lo_helper import (
    get_aoind_by_atom,
    reorder_by_atom_,
)
from quemb.shared.external.optqn import (
    get_be_error_jacobian as _ext_get_be_error_jacobian,
)
from quemb.shared.helper import copy_docstring, ensure, ncore_, timer, unused
from quemb.shared.manage_scratch import WorkDir
from quemb.shared.typing import Matrix, PathLike

def qchem_setup(self):
        """
        Extracts the information necessary to run a Q-Chem EOM-CCSD calculation
        for each fragment.
        Constructs the following scratch files:
        - 99.0 (Total energy)
        - 53.0 (MO coefficients)
        - 58.0 (Fock matrix)
        """

        print("QChem:")
        print("Exporting files 99.0, 58.0, 53.0 to Q-Chem for EOM-CCSD calculation")

        pot = self.pot
        Fobjs = self.Fobjs

        # Loop over each fragment and extract necessary information from Qchem
        for frag_number, fobj in enumerate(Fobjs):
            # Update the effective Hamiltonian
            if pot is not None:
                fobj.update_heff(pot)

            # print("Fragment:", frag_number)
            # print("BEFORE")

            # print(fobj._mf.mo_coeff)
            # print(fobj.mo_coeffs)

            # Perform SCF calculation
            # fobj.scf() #-destroys the mo coeffs from the optimization...

            # export necessary information to Q-Chem
            mf = fobj._mf
            print(mf.mo_coeff)
            print(fobj.mo_coeffs)

            print("Fragment number: ", frag_number)
            ###numbers of electrons in:
            n_mo_full_syst = shape(fobj.TA)[0]
            occ_tot = self.Nocc  ###full system
            SO_tot = shape(fobj.TA)[1]
            SO_occ = fobj.nsocc  ###Schmidt space

            """if fobj.nfsites <= SO_occ:  ###if fragment is fully occupied
                frag_occ = fobj.nfsites
            else:
                frag_occ = occ_tot"""

            # bath_occ = SO_occ - frag_occ

            # bath_virt = SO_tot - frag_occ - bath_occ
            ###TO DO: automate generation of qchem input files
            env_occ = occ_tot - SO_occ
            print("Qchem: set n_frozen_core")
            print("Number occupied environment: ", env_occ)

            env_virt = n_mo_full_syst - SO_tot - env_occ
            print("Qchem: set n_frozen_virtual")
            print("Number virtual environment: ", env_virt)

            ###File 99.0 - energy file in Qchem

            energy = mf.kernel()

            energy_array = zeros(12)
            # placeholder value - exact value doesn't matter
            energy_array[0] = 3.7617453591977221e02
            energy_array[1] = energy
            energy_array[11] = energy

            ###File 58.0 - Fock matrix file in Qchem
            ###Full system Fock matrix (AO basis)
            fock_full_syst = load("files_EOM/full_syst_fock.npy")

            flat_fock = array(fock_full_syst.flatten(), dtype=float64)
            full_fock = concatenate((flat_fock, flat_fock), axis=None)

            ###File 53.0 - MO coefficient matrix
            ###TA: AOxSO; mf.mo_coeff: SOxMO
            TA_after_HF = fobj.TA @ mf.mo_coeff

            ###Pad TA matrix with orthogonal vectors
            ###use it as MO coefficient matrix for Qchem

            # m=n_orb_total (frag+bath+env)
            # n=n_frag+n_bath
            m, n = TA_after_HF.shape

            # compute the orthonormal basis for the null space of TA.T
            # do SVD
            _, _, vh = svd(TA_after_HF.T, full_matrices=True)

            # take the (m-n) right singular vectors orthogonal to TA.T
            orthogonal_vectors = vh[n:m].T

            # pad the original matrix with the orthogonal vectors
            TA_full_pyscf = hstack(
                (
                    orthogonal_vectors[:, :env_occ],
                    TA_after_HF,
                    orthogonal_vectors[:, env_occ:],
                )
            )

            TA_full_qchem = TA_full_pyscf.T

            flat_mos = array(TA_full_qchem.flatten(), dtype=float64)

            ###MO energies needed at the end of file 53.0
            mo_energies = zeros(n_mo_full_syst)
            # set to arbitrary low number to avoid recanonicalization in Qchem
            mo_energies[:env_occ] = -1000
            mo_energies[env_occ : env_occ + SO_tot] = mf.mo_energy
            mo_energies[env_occ + SO_tot :] = 1000

            print("Fragment: ", frag_number)
            print(mf.mo_energy)
            print("SO occ: ", SO_occ)

            full_mo_array = concatenate(
                (flat_mos, flat_mos, mo_energies, mo_energies), axis=None
            )

            print("MOs: ")
            print(flat_mos)

            if not os.path.exists("files_EOM/scratch_fragment_" + str(frag_number)):
                os.makedirs("files_EOM/scratch_fragment_" + str(frag_number))

            energy_array.tofile(
                "files_EOM/scratch_fragment_" + str(frag_number) + "/99.0"
            )
            full_fock.tofile("files_EOM/scratch_fragment_" + str(frag_number) + "/58.0")
            full_mo_array.tofile(
                "files_EOM/scratch_fragment_" + str(frag_number) + "/53.0"
            )

        return

def compute_overlap_dyson(self, ref_dyson, frag_dyson, n_ex, extra=3):
        """
        Compute overlaps between two Dyson orbitals
        (currently in the AO basis), used for state targetting.
        Parameters
        ----------
        ref_dyson : numpy.ndarray
            Reference Dyson orbital (currently chosen as fragment 0).
        frag_dyson : numpy.ndarray
            Current fragment Dyson orbital.
        n_ex : int
            Number of excited states considered.
        extra : int
            Number of additional excited states being included.
            Helpful to solve intruder state issues. Default: 3
        Returns
        -------
        ovlp : numpy.ndarray
            Overlap matrix between reference (rows) and current fragments (columns).
        """

        ovlp = zeros((n_ex, n_ex + extra))

        for i in range(n_ex):
            for j in range(n_ex + extra):
                # no need to normalize these - Dyson orbs. don't have norm 1
                # print(shape(ref_dyson[i, :]))
                # print(shape(self.S))
                # print(shape(frag_dyson[:,j]))
                ovlp[i, j] = (
                    ref_dyson[i, :] @ self.S @ frag_dyson[j, :]
                )  ###if change TO AOS - NEED self.S!!!
        return ovlp

def match_fragments(self, ref_dyson, frag_dyson, n_ex, extra=3, threshold=0.0):
    """
    Hungarian matching + phase alignment for
    left Dyson orbitals (in AO basis)
    Parameters
    ----------
    ref_dyson : numpy.ndarray
        Reference Dyson orbital (currently chosen as fragment 0).
    frag_dyson : numpy.ndarray
        Current fragment Dyson orbital.
    n_ex : int
        Number of excited states considered.
    extra : int
        Number of additional excited states being included.
        Helpful to solve intruder state issues. Default: 3
    threshold : float
        Overlap threshold above which two excited states across the
        reference and fragment are considered to "match". Default: 0.6.
    Returns
    -------
    mapping : Dict[int, int]
        Map of reference excitation (row index)
        to current fragment excitation (column index).
    ovlp : numpy.ndarray
        Overlap matrix between reference and current fragments.
    excluded : List[int]
        List of excitations excluded from the current fragment
        due to no match being found.
    """

    excluded = []

    # ovlp: -1 to 1
    ovlp = self.compute_overlap_dyson(ref_dyson, frag_dyson, n_ex, extra)

    cost = -abs(ovlp)

    # if NaN in overlap matrix - assign a very large positive number
    # too costly to match this pair
    cost[isnan(cost)] = 1e6

    # if mapping[i]=j => reference excitation i matches fragment excitation j
    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {int(row): int(col) for row, col in zip(row_ind, col_ind)}

    for i, j in mapping.items():
        ov = ovlp[i, j]
        mag = abs(ov)
        if mag < threshold:
            print(f"WARNING weak match: ref {i} -> frag {j} |overlap|={mag}")
            print("SKIP MATCHING")
            # found no good match for this excitation pair => exclude
            excluded.append(j)
    return mapping, ovlp, excluded

def hij_full(self, n_ex):
    """Compute hij from Dyson orbitals - for averaged excited BE purposes.
    EOM-in-Fock approach
    - now with added environment contributions
    Parameters
    ----------
    n_ex: int
        Number of excited states.
    Returns
    -------
    hijAO : numpy.ndarray
        Effective Hamiltonian in AO basis, formed from Dyson orbitals.
    delta_hijAO : numpy.ndarray
        Environment correction term in AO basis.
    ip_full : numpy.ndarray
        Full-system IPs.
    dyson_ip_full : numpy.ndarray
        Full system Dyson orbitals corresponding to Koopman's-like excitations.
    hijMO : numpy.ndarray
        Effective Hamiltonian in MO basis, formed from Dyson orbitals.
    delta_hijMO : numpy.ndarray
        Environment correction term in MO basis.
    dyson_ip_full_MO : numpy.ndarray
        Full system Dyson orbitals corresponding to Koopman's-like excitations
        (in MO basis).
    M0_MO : numpy.ndarray
        "0th order spectral moment", or c_L @ c_R.
        Equal to identity matrix in the full system EOM-IP/EA case
        (due to biorthonormality).
    delta_M0_MO : numpy.ndarray
        "0th order spectral moment" for the environment correction term.
    """

    ha_to_ev = 27.2114

    # extra = compute additional excitations in case of intruder states
    extra = 0
    extra_koop = self.Nocc - n_ex

    # Copy the full system MO coefficients
    C_MO = self.C.copy()
    nao = C_MO.shape[0]

    # in AO basis
    hijAO = zeros((nao, nao))
    delta_hijAO = zeros((nao, nao))
    M_0_AO = zeros((nao, nao))
    delta_M_0_AO = zeros((nao, nao))

    # full system environment correction
    # simple Koopman's theorem IP reference
    # dyson_ip_full: in AO basis
    # dyson_ip_full_MO: in MO basis

    dyson_ip_full = zeros((n_ex + extra_koop, nao))
    dyson_ip_full_MO = zeros((n_ex + extra_koop, nao))
    ip_full = zeros(n_ex + extra_koop)

    for i in range(n_ex + extra_koop):
        ip_full[i] = self.mo_energy[self.Nocc - 1 - i]
        dyson_ip_full_MO[i, self.Nocc - 1 - i] = 1
        dyson_ip_full[i, :] = C_MO[:, self.Nocc - 1 - i]

    ip_full = (-ip_full) * ha_to_ev

    # parse Dyson info from Q-Chem outputs

    for frag_number, fobjs in enumerate(self.Fobjs):
        output = "qchem_fragment_" + str(frag_number) + "/eom.out"
        dyson_parser(fobjs, output, n_ex + extra)

    # build fragment contributions

    for frag_number, fobjs in enumerate(self.Fobjs):
        n_mo_full = shape(fobjs.TA)[0]
        occ_tot = self.Nocc  # full system
        SO_tot = shape(fobjs.TA)[1]
        SO_occ = fobjs.nsocc  # Schmidt space

        # fragment environment correction - simple Koopman's theorem IP

        dyson_ip_frag_left = zeros((n_ex + extra, nao))
        dyson_ip_frag_right = zeros((n_ex + extra, nao))

        ip_frag = zeros(n_ex + extra)
        delta_ex = zeros(n_ex + extra)

        excluded_koop = []

        for i in range(n_ex + extra):
            ### TO DO: sometimes dyson orbital doesn't correspond to
            # an excitation from HOMO->infty!!
            # what can we do about double excitations?

            idx_guess = argmax(abs(fobjs.dyson_right[i, :]))

            norm_left = norm(fobjs.dyson_left[i, :])
            norm_right = norm(fobjs.dyson_right[i, :])

            print(idx_guess, norm_left, norm_right)

            if idx_guess - occ_tot + SO_occ <= 0:
                idx_guess = occ_tot - 1 - i

            # Perform SCF calculation
            # fobjs.scf()
            # running in parallel doesn't work without this - test

            ip_frag[i] = (
                -fobjs._mf.mo_energy[idx_guess - occ_tot + SO_occ] * ha_to_ev
            )

            dyson_ip_frag_left[i, idx_guess] = 1  # * norm_left
            dyson_ip_frag_right[i, idx_guess] = 1  # * norm_right

            if norm_left < 0.3:
                excluded_koop.append(i)

            delta_ex[i] = ip_frag[i]

            print("NEW FRAGMENT")
            print(frag_number)
            print(
                "Fragment number",
                frag_number,
                "excited state",
                i,
                "excitation energy",
                fobjs.ex_e[i],
                "eV, norm",
                norm(fobjs.dyson_right[i, :]),
                "from MO:",
                argmax(abs(fobjs.dyson_right[i, :])),
            )

        ###exclude - NEW
        print("EXCLUDE: ")
        print(frag_number)
        print(excluded_koop)

        dyson_ip_frag_left = delete(dyson_ip_frag_left, excluded_koop, axis=0)
        dyson_ip_frag_right = delete(dyson_ip_frag_right, excluded_koop, axis=0)
        delta_ex = delete(delta_ex, excluded_koop)
        print(delta_ex)

        exc = diag(fobjs.ex_e[: n_ex + extra])

        delta_ex = diag(delta_ex)

        env_occ = occ_tot - SO_occ
        env_virt = n_mo_full - SO_tot - env_occ

        hij_mo = (
            fobjs.dyson_left[:, env_occ : n_mo_full - env_virt].T
            @ exc
            @ fobjs.dyson_right[:, env_occ : n_mo_full - env_virt]
        )

        m_0_mo = (
            fobjs.dyson_left[:, env_occ : n_mo_full - env_virt].T
            @ fobjs.dyson_right[:, env_occ : n_mo_full - env_virt]
        )

        # environment correction!
        delta_hij_mo = (
            dyson_ip_frag_left[:, env_occ : n_mo_full - env_virt].T
            @ delta_ex
            @ dyson_ip_frag_right[:, env_occ : n_mo_full - env_virt]
        )

        delta_m_0_mo = (
            dyson_ip_frag_left[:, env_occ : n_mo_full - env_virt].T
            @ dyson_ip_frag_right[:, env_occ : n_mo_full - env_virt]
        )

        # projection matrix
        cind = [fobjs.AO_in_frag[i] for i in fobjs.weight_and_relAO_per_center[1]]
        Pc_ = (
            fobjs.TA.T
            @ self.S
            @ self.W[:, cind]
            @ self.W[:, cind].T
            @ self.S
            @ fobjs.TA
        )

        # Compute hij in the SO basis
        # and transform to AO basis

        hij_so = fobjs.mo_coeffs @ hij_mo @ fobjs.mo_coeffs.T

        hij_center = Pc_ @ hij_so
        # do we apply projection correctly?
        # hij_center = Pc_ @ hij_so @ Pc_

        # print("ARE NON-CENTER CONTRIBUTIONS IMPORTANT?")
        ###half of non-diag contributions important? - keep projection only on left?

        hij_ao = fobjs.TA @ hij_center @ fobjs.TA.T  ###!!!just center!
        hijAO += hij_ao
        fobjs.hij_mo = hij_mo

        delta_hij_so = fobjs.mo_coeffs @ delta_hij_mo @ fobjs.mo_coeffs.T
        delta_hij_center = Pc_ @ delta_hij_so
        delta_hij_ao = fobjs.TA @ delta_hij_center @ fobjs.TA.T

        delta_hijAO += delta_hij_ao

        m_0_so = fobjs.mo_coeffs @ m_0_mo @ fobjs.mo_coeffs.T
        m_0_center = Pc_ @ m_0_so
        m_0_ao = fobjs.TA @ m_0_center @ fobjs.TA.T
        M_0_AO += m_0_ao

        delta_m_0_so = fobjs.mo_coeffs @ delta_m_0_mo @ fobjs.mo_coeffs.T
        delta_m_0_center = Pc_ @ delta_m_0_so
        delta_m_0_ao = fobjs.TA @ delta_m_0_center @ fobjs.TA.T
        delta_M_0_AO += delta_m_0_ao

        # Transform to the MO basis if needed
        hijMO = self.C.T @ self.S @ hijAO @ self.S @ self.C
        delta_hijMO = self.C.T @ self.S @ delta_hijAO @ self.S @ self.C

        M0_MO = self.C.T @ self.S @ M_0_AO @ self.S @ self.C
        delta_M0_MO = self.C.T @ self.S @ delta_M_0_AO @ self.S @ self.C

    return (
        hijAO,
        delta_hijAO,
        ip_full,
        dyson_ip_full,
        hijMO,
        delta_hijMO,
        dyson_ip_full_MO,
        M0_MO,
        delta_M0_MO,
    )

def hij_full_koop(self, n_ex):
    """Compute hij from Dyson orbitals - for averaged excited BE purposes.
    ---Alexa: this is the original strategy, where we match Koop-BE
    with EOM-BE and full system IPs
    Parameters
    ----------
    n_ex: int
        Number of excited states.
    Returns
    -------
    hijAO : numpy.ndarray
        Effective Hamiltonian in AO basis, formed from Dyson orbitals.
    delta_hijAO : numpy.ndarray
        Environment correction term in AO basis.
    ip_full : numpy.ndarray
        Full-system IPs.
    dyson_ip_full : numpy.ndarray
        Full system Dyson orbitals corresponding to Koopman's-like excitations.
    hijMO : numpy.ndarray
        Effective Hamiltonian in MO basis, formed from Dyson orbitals.
    delta_hijMO : numpy.ndarray
        Environment correction term in MO basis.
    """
    # Notation: _mo: fragment MO basis
    # MO: full system MO basis

    ha_to_ev = 27.2114

    # extra = compute additional excitations in case of intruder states
    extra = 0
    # extra_koop = 0
    extra_koop = self.Nocc - n_ex

    # Copy the full system MO coefficients
    C_MO = self.C.copy()
    nao = C_MO.shape[0]

    # in AO basis
    hijAO = zeros((nao, nao))
    delta_hijAO = zeros((nao, nao))

    # full system environment correction
    # simple Koopman's theorem IP reference
    # dyson_ip_full: in AO basis
    # dyson_ip_full_MO: in MO basis

    dyson_ip_full = zeros((n_ex + extra_koop, nao))
    dyson_ip_full_MO = zeros((n_ex + extra_koop, nao))
    ip_full = zeros(n_ex + extra_koop)

    for i in range(n_ex + extra_koop):
        ip_full[i] = self.mo_energy[self.Nocc - 1 - i]
        dyson_ip_full_MO[i, self.Nocc - 1 - i] = 1
        dyson_ip_full[i, :] = C_MO[:, self.Nocc - 1 - i]

    ip_full = (-ip_full) * ha_to_ev

    # parse Dyson info from Q-Chem outputs

    for frag_number, fobjs in enumerate(self.Fobjs):
        output = "qchem_fragment_" + str(frag_number) + "/eom.out"
        dyson_parser(fobjs, output, n_ex + extra)

    ###ATTEMPT A DIFFERENT STYLE OF MATCHING - in AO basis
    """for frag_ref, fobj_ref in enumerate(self.Fobjs):
        ref_dyson=fobj_ref.dyson_ao
        for frag_idx, fobj in enumerate(self.Fobjs):
            frag_dyson=fobj.dyson_ao
            #print(frag_ref, frag_idx)
            ovlp = ref_dyson @ self.S @ frag_dyson.T"""
    # ovlp = self.compute_overlap_dyson(ref_dyson, frag_dyson, n_ex)
    # print(ovlp)

    # compute overlaps and reorder excitations
    print("EOM-IP FRAGMENT REORDERING")
    print("Compute overlaps between Dyson orbital fragments: ")

    # Reference Dyson left orbitals - fragment 0:
    ref = self.Fobjs[0]
    # ref_dyson = ref.dyson_left
    ref_dyson = ref.dyson_ao
    # ref_dyson = ref.dyson_right

    all_mappings: List[Dict[int, int]] = []
    all_overlaps: List[ndarray] = []

    for frag_idx, fobj in enumerate(self.Fobjs):
        # frag_dyson = fobj.dyson_left
        frag_dyson = fobj.dyson_ao
        # frag_dyson = fobj.dyson_right

        mapping, ovlp, excluded = self.match_fragments(
            ref_dyson, frag_dyson, n_ex, extra
        )

        all_mappings.append(mapping)
        all_overlaps.append(ovlp)
        print(f"[hij_full] fragment {frag_idx} mapping to reference:")
        for ri, fj in mapping.items():
            print(f"  ref {ri} -> frag {fj} |ov|={abs(ovlp[ri, fj]):.6f}")

        # reorder excitations to match fragment 0 (reference)
        new_order = [mapping[i] for i in range(n_ex) if mapping[i] not in excluded]
        print("EXCLUDE: ")
        print(excluded)
        print(new_order)
        fobj.new_order = new_order

        # fobj.dyson_left: in fragment SO basis
        # does state tracking hurt more than help?
        # March 17th: REINTRODUCE STATE TRACKING AND EXCLUDING!
        # fobj.dyson_left = fobj.dyson_left[new_order, :]
        # fobj.dyson_right = fobj.dyson_right[new_order, :]
        # fobj.ex_e = fobj.ex_e[new_order]

    # build fragment contributions

    for frag_number, fobjs in enumerate(self.Fobjs):
        n_mo_full = shape(fobjs.TA)[0]
        occ_tot = self.Nocc  # full system
        SO_tot = shape(fobjs.TA)[1]
        SO_occ = fobjs.nsocc  # Schmidt space

        ###NEW!
        # extra_koop=SO_occ-n_ex

        extra = SO_occ - n_ex

        # fragment environment correction - simple Koopman's theorem IP

        dyson_ip_frag = zeros((n_ex + extra, nao))

        dyson_ip_frag_ao = zeros((n_ex + extra, nao))

        ip_frag = zeros(n_ex + extra)
        delta_ex = zeros(n_ex + extra)

        for i in range(n_ex + extra):
            ### TO DO: sometimes dyson orbital doesn't correspond to
            # an excitation from HOMO->infty!!
            # what can we do about double excitations?
            idx_guess = occ_tot - 1 - i

            ip_frag[i] = -fobjs._mf.mo_energy[SO_occ - 1 - i] * ha_to_ev
            dyson_ip_frag[i, idx_guess] = -1
            delta_ex[i] = ip_frag[i]

            orbs = fobjs.TA @ fobjs.mo_coeffs

            dyson_ip_frag_ao[i, :] = orbs[:, SO_occ - 1 - i]

        ################################################################
        # compute overlaps and reorder excitations
        """print("Koop-IP-BE FRAGMENT REORDERING")
        print("Compute overlaps between Dyson orbital fragments: ")
        all_mappings: List[Dict[int, int]] = []
        all_overlaps: List[ndarray] = []
        # Reference Dyson left orbitals - fragment 0:
        print(frag_number)
        if frag_number==0:
            ref_dyson_koop = dyson_ip_frag_ao
        frag_dyson_koop = dyson_ip_frag_ao
        mapping, ovlp, excluded = self.match_fragments(
            ref_dyson_koop, frag_dyson_koop, n_ex, extra
        )
        all_mappings.append(mapping)
        all_overlaps.append(ovlp)
        print(f"[hij_full] fragment {frag_number} mapping to reference:")
        for ri, fj in mapping.items():
            print(f"  ref {ri} -> frag {fj} |ov|={abs(ovlp[ri, fj]):.6f}")
        # reorder excitations to match fragment 0 (reference)
        new_order = [mapping[i] for i in range(n_ex) if mapping[i] not in excluded]
        print("EXCLUDE: ")
        print(excluded)
        print(new_order)
        # fobj.dyson_left: in fragment SO basis
        # does state tracking hurt more than help?
        dyson_ip_frag = dyson_ip_frag[fobjs.new_order, :]
        delta_ex = delta_ex[fobjs.new_order]"""
        #################################################################

        exc = diag(fobjs.ex_e[:n_ex])

        delta_ex = diag(delta_ex)

        env_occ = occ_tot - SO_occ
        env_virt = n_mo_full - SO_tot - env_occ

        hij_mo = (
            fobjs.dyson_left[:, env_occ : n_mo_full - env_virt].T
            @ exc
            @ fobjs.dyson_right[:, env_occ : n_mo_full - env_virt]
        )

        # environment correction!
        delta_hij_mo = (
            dyson_ip_frag[:, env_occ : n_mo_full - env_virt].T
            @ delta_ex
            @ dyson_ip_frag[:, env_occ : n_mo_full - env_virt]
        )

        # projection matrix
        cind = [fobjs.AO_in_frag[i] for i in fobjs.weight_and_relAO_per_center[1]]
        Pc_ = (
            fobjs.TA.T
            @ self.S
            @ self.W[:, cind]
            @ self.W[:, cind].T
            @ self.S
            @ fobjs.TA
        )

        # Compute hij in the SO basis
        # and transform to AO basis
        hij_so = fobjs.mo_coeffs @ hij_mo @ fobjs.mo_coeffs.T
        hij_center = Pc_ @ hij_so
        # do we apply projection correctly?
        # hij_center = Pc_ @ hij_so @ Pc_

        # print("ARE NON-CENTER CONTRIBUTIONS IMPORTANT?")
        ###half of non-diag contributions important? - keep projection only on left?

        hij_ao = fobjs.TA @ hij_center @ fobjs.TA.T  ###!!!just center!
        hijAO += hij_ao
        fobjs.hij_mo = hij_mo

        delta_hij_so = fobjs.mo_coeffs @ delta_hij_mo @ fobjs.mo_coeffs.T
        delta_hij_center = Pc_ @ delta_hij_so
        # delta_hij_center = Pc_ @ delta_hij_so @ Pc_
        delta_hij_ao = fobjs.TA @ delta_hij_center @ fobjs.TA.T

        delta_hijAO += delta_hij_ao

        # Symmetrize hij??
        # hijAO = (hijAO + hijAO.T) / 2.0

        # Transform to the MO basis if needed
        hijMO = self.C.T @ self.S @ hijAO @ self.S @ self.C
        delta_hijMO = self.C.T @ self.S @ delta_hijAO @ self.S @ self.C

        ###TO DO: also need to match full system ip_full
        # and eigvecs_mf (from delta_hijAO) to eigvecs from hijAO!!!

    return (
        hijAO,
        delta_hijAO,
        ip_full,
        dyson_ip_full,
        hijMO,
        delta_hijMO,
        dyson_ip_full_MO,
    )

import re

from numpy import array, sqrt


def dyson_parser(fobj, output="eom.out", n_ex=15):
    """
    This function parses the Q-Chem EOM-IP output file and reads the
    excitation energies, as well as left- and right-Dyson orbitals.
    Parameters
    ----------
    output : str
        Q-Chem output file
    n_ex : int
        Number of excited states computed
    """

    with open(output, "r") as f:
        content = f.read()

    # extract excitation energies
    energy_pattern = re.compile(
        r"EOMIP transition\s+\d+/\w+\s+"
        r"Total energy = ([\-\d\.]+) a\.u\.\s+"
        r"Excitation energy = ([\d\.]+) eV\.",
        re.MULTILINE,
    )

    excitation_energies = [float(match[1]) for match in energy_pattern.findall(content)]
    excitation_energies = array(excitation_energies)

    # Dyson norms
    left_norms = [
        float(x)
        for x in re.findall(r"Left Dyson orbital norm is\s+([0-9Ee\+\-\.]+)", content)
    ]
    right_norms = [
        float(x)
        for x in re.findall(r"Right Dyson orbital norm is\s+([0-9Ee\+\-\.]+)", content)
    ]
    left_norms = array(left_norms)
    right_norms = array(right_norms)

    # fixes bug FOR MO BASIS DYSON ORBITALS (where NAO>100)

    # extract left Dyson orbitals
    # match everything until "Excitation energy" or end of file
    dyson_pattern = re.compile(
        r"Left alpha Dyson orbital in the MO basis "
        r"\(canonical Q-Chem's ordering\):\s*\n(.*?)(?:\n\n|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    dyson_matches_left = dyson_pattern.findall(content)

    # extract right Dyson orbitals
    dyson_pattern = re.compile(
        r"Right alpha Dyson orbital in the MO basis "
        r"\(canonical Q-Chem's ordering\):\s*\n(.*?)(?:\n\n|\Z)",
        re.MULTILINE | re.DOTALL,
    )

    dyson_matches_right = dyson_pattern.findall(content)

    coeff_matrix_left = []

    line_regex = re.compile(r"(\d+)\s*([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)")

    for block in dyson_matches_left[:n_ex]:
        block_coeffs = []
        for line in block.strip().splitlines():
            m = line_regex.match(line.strip())
            if m:
                index, coeff = m.groups()
                block_coeffs.append(float(coeff))
            else:
                print("Could not parse line:", line)
        coeff_matrix_left.append(block_coeffs)

    coeff_matrix_left = array(coeff_matrix_left)

    coeff_matrix_right = []

    line_regex = re.compile(r"(\d+)\s*([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)")

    for block in dyson_matches_right[:n_ex]:
        block_coeffs = []
        for line in block.strip().splitlines():
            m = line_regex.match(line.strip())
            if m:
                index, coeff = m.groups()
                block_coeffs.append(float(coeff))
            else:
                print("Could not parse line:", line)
        coeff_matrix_right.append(block_coeffs)

    coeff_matrix_right = array(coeff_matrix_right)

    # extract Dyson orbitals (AO basis) - right Dyson orbital only - used for matching!

    blocks = []
    lines = content.splitlines()

    i = 0
    while i < len(lines):
        if lines[i].startswith(
            "Decomposition over AOs for the right alpha Dyson orbital:"
        ):
            i += 1
            coeffs = []

            while i < len(lines) and lines[i].strip() != "*****":
                coeffs.append(float(lines[i].strip()))
                i += 1

            blocks.append(coeffs)

        i += 1

    coeff_matrix_ao = array(blocks[:n_ex])

    # save results to fobj
    # scale dyson orbitals by their sqrt(norms)!
    fobj.ex_e = excitation_energies[:n_ex]
    fobj.dyson_left = sqrt(left_norms[:n_ex][:, None]) * coeff_matrix_left
    fobj.dyson_right = sqrt(right_norms[:n_ex][:, None]) * coeff_matrix_right
    # fobj.dyson_left = coeff_matrix_left
    # fobj.dyson_right = coeff_matrix_right
    fobj.dyson_ao = coeff_matrix_ao

    fobj.norm_left = left_norms[:n_ex]
    fobj.norm_right = right_norms[:n_ex]

    return