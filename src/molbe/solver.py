# Author(s): Oinam Romesh Meitei, Leah Weisburn, Shaun Weatherly

import functools
import os
import sys

import numpy

from molbe import be_var
from molbe.external.ccsd_rdm import (
    make_rdm1_ccsd_t1,
    make_rdm1_uccsd,
    make_rdm2_uccsd,
    make_rdm2_urlx,
)


def be_func(
    pot,
    Fobjs,
    Nocc,
    solver,
    enuc,
    hf_veff=None,
    only_chem=False,
    nproc=4,
    hci_pt=False,
    hci_cutoff=0.001,
    ci_coeff_cutoff=None,
    select_cutoff=None,
    eeval=False,
    ereturn=False,
    frag_energy=False,
    relax_density=False,
    return_vec=False,
    ecore=0.0,
    ebe_hf=0.0,
    be_iter=None,
    use_cumulant=True,
    scratch_dir=None,
    **solver_kwargs,
):
    """
    Perform bootstrap embedding calculations for each fragment.

    This function computes the energy and/or error for each fragment in a molecular system using various quantum chemistry solvers.

    Parameters
    ----------
    pot : list
        List of potentials.
    Fobjs : list of MolBE.fragpart
        List of fragment objects.
    Nocc : int
        Number of occupied orbitals.
    solver : str
        Quantum chemistry solver to use ('MP2', 'CCSD', 'FCI', 'HCI', 'SHCI', 'SCI').
    enuc : float
        Nuclear energy.
    hf_veff : numpy.ndarray, optional
        Hartree-Fock effective potential. Defaults to None.
    only_chem : bool, optional
        Whether to only optimize the chemical potential. Defaults to False.
    nproc : int, optional
        Number of processors. Defaults to 4. This is only neccessary for 'SHCI' solver
    eeval : bool, optional
        Whether to evaluate the energy. Defaults to False.
    ereturn : bool, optional
        Whether to return the energy. Defaults to False.
    frag_energy : bool, optional
        Whether to calculate fragment energy. Defaults to False.
    relax_density : bool, optional
        Whether to relax the density. Defaults to False.
    return_vec : bool, optional
        Whether to return the error vector. Defaults to False.
    ecore : float, optional
        Core energy. Defaults to 0.
    ebe_hf : float, optional
        Hartree-Fock energy. Defaults to 0.
    be_iter : int or None, optional
        Iteration number. Defaults to None.
    use_cumulant : bool, optional
        Whether to use the cumulant-based energy expression. Defaults to True.

    Returns
    -------
    float or tuple
        Depending on the options, it returns the norm of the error vector, the energy, or a combination of these values.
    """
    import os

    from pyscf import ao2mo, fci

    from .helper import get_frag_energy

    rdm_return = False
    if relax_density:
        rdm_return = True
    E = 0.0
    if frag_energy or eeval:
        total_e = [0.0, 0.0, 0.0]

    # Loop over each fragment and solve using the specified solver
    for fobj in Fobjs:
        # Update the effective Hamiltonian
        if pot is not None:
            heff_ = fobj.update_heff(pot, return_heff=True, only_chem=only_chem)
        else:
            heff_ = None

        # Compute the one-electron Hamiltonian
        h1_ = fobj.fock + fobj.heff
        # Perform SCF calculation
        fobj.scf()

        # Solve using the specified solver
        if solver == "MP2":
            fobj._mc = solve_mp2(fobj._mf, mo_energy=fobj._mf.mo_energy)
        elif solver == "CCSD":
            if rdm_return:
                fobj.t1, fobj.t2, rdm1_tmp, rdm2s = solve_ccsd(
                    fobj._mf,
                    mo_energy=fobj._mf.mo_energy,
                    relax=True,
                    use_cumulant=use_cumulant,
                    rdm2_return=True,
                    rdm_return=True,
                )
            else:
                fobj.t1, fobj.t2 = solve_ccsd(
                    fobj._mf, mo_energy=fobj._mf.mo_energy, rdm_return=False
                )
                rdm1_tmp = make_rdm1_ccsd_t1(fobj.t1)

        elif solver == "FCI":

            mc = fci.FCI(fobj._mf, fobj._mf.mo_coeff)
            efci, civec = mc.kernel()
            rdm1_tmp = mc.make_rdm1(civec, mc.norb, mc.nelec)

        elif solver == "HCI":
            from pyscf import hci
            # pilot pyscf.hci only in old versions

            nao, nmo = fobj._mf.mo_coeff.shape

            eri = ao2mo.kernel(
                fobj._mf._eri, fobj._mf.mo_coeff, aosym="s4", compact=False
            ).reshape(4 * ((nmo),))

            ci_ = hci.SCI(fobj._mf.mol)
            if select_cutoff is None and ci_coeff_cutoff is None:
                select_cutoff = hci_cutoff
                ci_coeff_cutoff = hci_cutoff
            elif select_cutoff is None or ci_coeff_cutoff is None:
                sys.exit()

            ci_.select_cutoff = select_cutoff
            ci_.ci_coeff_cutoff = ci_coeff_cutoff

            nelec = (fobj.nsocc, fobj.nsocc)
            h1_ = fobj.fock + fobj.heff
            h1_ = functools.reduce(
                numpy.dot, (fobj._mf.mo_coeff.T, h1_, fobj._mf.mo_coeff)
            )
            eci, civec = ci_.kernel(h1_, eri, nmo, nelec)
            civec = numpy.asarray(civec)

            (rdm1a_, rdm1b_), (rdm2aa, rdm2ab, rdm2bb) = ci_.make_rdm12s(
                civec, nmo, nelec
            )
            rdm1_tmp = rdm1a_ + rdm1b_
            rdm2s = rdm2aa + rdm2ab + rdm2ab.transpose(2, 3, 0, 1) + rdm2bb

        elif solver == "SHCI":
            from pyscf.shciscf import shci

            if scratch_dir is None and be_var.CREATE_SCRATCH_DIR:
                tmp = os.path.join(be_var.SCRATCH, str(os.getpid()), str(fobj.dname))
            elif scratch_dir is None:
                tmp = be_var.SCRATCH
            else:
                tmp = os.path.join(scratch_dir, str(os.getpid()), str(fobj.dname))
            if not os.path.isdir(tmp):
                os.system("mkdir -p " + tmp)
            nao, nmo = fobj._mf.mo_coeff.shape

            nelec = (fobj.nsocc, fobj.nsocc)
            mch = shci.SHCISCF(fobj._mf, nmo, nelec, orbpath=fobj.dname)
            mch.fcisolver.mpiprefix = "mpirun -np " + str(nproc)
            if hci_pt:
                mch.fcisolver.stochastic = False
                mch.fcisolver.epsilon2 = hci_cutoff
            else:
                mch.fcisolver.stochastic = (
                    True  # this is for PT and doesnt add PT to rdm
                )
                mch.fcisolver.nPTiter = 0
            mch.fcisolver.sweep_iter = [0]
            mch.fcisolver.DoRDM = True
            mch.fcisolver.sweep_epsilon = [hci_cutoff]
            mch.fcisolver.scratchDirectory = scratch_dir
            mch.mc1step()
            rdm1_tmp, rdm2s = mch.fcisolver.make_rdm12(0, nmo, nelec)

        elif solver == "SCI":
            from pyscf import ao2mo, cornell_shci, mcscf

            nao, nmo = fobj._mf.mo_coeff.shape
            nelec = (fobj.nsocc, fobj.nsocc)
            cas = mcscf.CASCI(fobj._mf, nmo, nelec)
            h1, ecore = cas.get_h1eff(mo_coeff=fobj._mf.mo_coeff)
            eri = ao2mo.kernel(
                fobj._mf._eri, fobj._mf.mo_coeff, aosym="s4", compact=False
            ).reshape(4 * ((nmo),))

            ci = cornell_shci.SHCI()
            ci.runtimedir = fobj.dname
            ci.restart = True
            ci.config["var_only"] = True
            ci.config["eps_vars"] = [hci_cutoff]
            ci.config["get_1rdm_csv"] = True
            ci.config["get_2rdm_csv"] = True
            ci.kernel(h1, eri, nmo, nelec)
            rdm1_tmp, rdm2s = ci.make_rdm12(0, nmo, nelec)

        elif solver in ["block2", "DMRG", "DMRGCI", "DMRGSCF"]:
            solver_kwargs_ = solver_kwargs.copy()
            if scratch_dir is None and be_var.CREATE_SCRATCH_DIR:
                tmp = os.path.join(be_var.SCRATCH, str(os.getpid()), str(fobj.dname))
            else:
                tmp = os.path.join(scratch_dir, str(os.getpid()), str(fobj.dname))
            if not os.path.isdir(tmp):
                os.system("mkdir -p " + tmp)

            try:
                rdm1_tmp, rdm2s = solve_block2(
                    fobj._mf, fobj.nsocc, frag_scratch=tmp, **solver_kwargs_
                )
            except Exception as inst:
                raise inst
            finally:
                if solver_kwargs_.pop("force_cleanup", False):
                    os.system("rm -r " + os.path.join(tmp, "F.*"))
                    os.system("rm -r " + os.path.join(tmp, "FCIDUMP*"))
                    os.system("rm -r " + os.path.join(tmp, "node*"))

        else:
            print("Solver not implemented", flush=True)
            print("exiting", flush=True)
            sys.exit()

        if solver == "MP2":
            rdm1_tmp = fobj._mc.make_rdm1()
        fobj.__rdm1 = rdm1_tmp.copy()
        fobj._rdm1 = (
            functools.reduce(
                numpy.dot,
                (
                    fobj.mo_coeffs,
                    # fobj._mc.make_rdm1(),
                    rdm1_tmp,
                    fobj.mo_coeffs.T,
                ),
            )
            * 0.5
        )

        if eeval or ereturn:
            if solver == "CCSD" and not rdm_return:
                with_dm1 = True
                if use_cumulant:
                    with_dm1 = False
                rdm2s = make_rdm2_urlx(fobj.t1, fobj.t2, with_dm1=with_dm1)
            elif solver == "MP2":
                rdm2s = fobj._mc.make_rdm2()
            elif solver == "FCI":
                rdm2s = mc.make_rdm2(civec, mc.norb, mc.nelec)
                if use_cumulant:
                    hf_dm = numpy.zeros_like(rdm1_tmp)
                    hf_dm[numpy.diag_indices(fobj.nsocc)] += 2.0
                    del_rdm1 = rdm1_tmp.copy()
                    del_rdm1[numpy.diag_indices(fobj.nsocc)] -= 2.0
                    nc = (
                        numpy.einsum("ij,kl->ijkl", hf_dm, hf_dm)
                        + numpy.einsum("ij,kl->ijkl", hf_dm, del_rdm1)
                        + numpy.einsum("ij,kl->ijkl", del_rdm1, hf_dm)
                    )
                    nc -= (
                        numpy.einsum("ij,kl->iklj", hf_dm, hf_dm)
                        + numpy.einsum("ij,kl->iklj", hf_dm, del_rdm1)
                        + numpy.einsum("ij,kl->iklj", del_rdm1, hf_dm)
                    ) * 0.5
                    rdm2s -= nc
            fobj.__rdm2 = rdm2s.copy()
            if frag_energy or eeval:
                # Find the energy of a given fragment, with the cumulant definition.
                # Return [e1, e2, ec] as e_f and add to the running total_e.
                e_f = get_frag_energy(
                    fobj.mo_coeffs,
                    fobj.nsocc,
                    fobj.nfsites,
                    fobj.efac,
                    fobj.TA,
                    fobj.h1,
                    hf_veff,
                    rdm1_tmp,
                    rdm2s,
                    fobj.dname,
                    eri_file=fobj.eri_file,
                    veff0=fobj.veff0,
                )
                total_e = [sum(x) for x in zip(total_e, e_f)]
                fobj.energy_hf()

    if frag_energy or eeval:
        Ecorr = sum(total_e)

    if frag_energy:
        return (Ecorr, total_e)

    ernorm, ervec = solve_error(Fobjs, Nocc, only_chem=only_chem)

    if return_vec:
        return (ernorm, ervec, [Ecorr, total_e])

    if eeval:
        print("Error in density matching      :   {:>2.4e}".format(ernorm), flush=True)

    return ernorm


def be_func_u(
    pot,
    Fobjs,
    solver,
    enuc,
    hf_veff=None,
    eeval=False,
    ereturn=False,
    frag_energy=True,
    relax_density=False,
    ecore=0.0,
    ebe_hf=0.0,
    scratch_dir=None,
    use_cumulant=True,
    frozen=False,
):
    """
    Perform bootstrap embedding calculations for each fragment with UCCSD.

    This function computes the energy and/or error for each fragment in a molecular system using various quantum chemistry solvers.

    Parameters
    ----------
    pot : list
        List of potentials.
    Fobjs : zip list of MolBE.fragpart, alpha and beta
        List of fragment objects. Each element is a tuple with the alpha and beta components
    solver : str
        Quantum chemistry solver to use ('UCCSD').
    enuc : float
        Nuclear energy.
    hf_veff : tuple of numpy.ndarray, optional
        Hartree-Fock effective potential. Defaults to None.
    eeval : bool, optional
        Whether to evaluate the energy. Defaults to False.
    ereturn : bool, optional
        Whether to return the energy. Defaults to False.
    frag_energy : bool, optional
        Whether to calculate fragment energy. Defaults to True.
    relax_density : bool, optional
        Whether to relax the density. Defaults to False.
    return_vec : bool, optional
        Whether to return the error vector. Defaults to False.
    ecore : float, optional
        Core energy. Defaults to 0.
    ebe_hf : float, optional
        Hartree-Fock energy. Defaults to 0.
    use_cumulant : bool, optional
        Whether to use the cumulant-based energy expression. Defaults to True.
    frozen : bool, optional
        Frozen core. Defaults to False
    Returns
    -------
    float or tuple
        Depending on the options, it returns the norm of the error vector, the energy, or a combination of these values.
    """
    from molbe.external.unrestricted_utils import make_uhf_obj

    from .helper import get_frag_energy_u

    rdm_return = False
    if relax_density:
        rdm_return = True
    E = 0.0
    if frag_energy or eeval:
        total_e = [0.0, 0.0, 0.0]

    # Loop over each fragment and solve using the specified solver
    for fobj_a, fobj_b in Fobjs:
        heff_ = None  # No outside chemical potential implemented for unrestricted yet

        fobj_a.scf(unrestricted=True, spin_ind=0)
        fobj_b.scf(unrestricted=True, spin_ind=1)

        full_uhf, eris = make_uhf_obj(fobj_a, fobj_b, frozen=frozen)

        if solver == "UCCSD":
            if rdm_return:
                ucc, rdm1_tmp, rdm2s = solve_uccsd(
                    full_uhf,
                    eris,
                    relax=relax_density,
                    rdm_return=True,
                    rdm2_return=True,
                    frozen=frozen,
                )
            else:
                ucc = solve_uccsd(
                    full_uhf, eris, relax=relax_density, rdm_return=False, frozen=frozen
                )
                rdm1_tmp = make_rdm1_uccsd(ucc, relax=relax_density)
        else:
            print("Solver not implemented", flush=True)
            print("exiting", flush=True)
            sys.exit()

        fobj_a.__rdm1 = rdm1_tmp[0].copy()
        fobj_b._rdm1 = (
            functools.reduce(
                numpy.dot, (fobj_a._mf.mo_coeff, rdm1_tmp[0], fobj_a._mf.mo_coeff.T)
            )
            * 0.5
        )

        fobj_b.__rdm1 = rdm1_tmp[1].copy()
        fobj_b._rdm1 = (
            functools.reduce(
                numpy.dot, (fobj_b._mf.mo_coeff, rdm1_tmp[1], fobj_b._mf.mo_coeff.T)
            )
            * 0.5
        )

        if eeval or ereturn:
            if solver == "UCCSD" and not rdm_return:
                with_dm1 = True
                if use_cumulant:
                    with_dm1 = False
                rdm2s = make_rdm2_uccsd(ucc, with_dm1=with_dm1)
            fobj_a.__rdm2 = rdm2s[0].copy()
            fobj_b.__rdm2 = rdm2s[1].copy()
            if frag_energy:
                if frozen:
                    h1_ab = [
                        full_uhf.h1[0]
                        + full_uhf.full_gcore[0]
                        + full_uhf.core_veffs[0],
                        full_uhf.h1[1]
                        + full_uhf.full_gcore[1]
                        + full_uhf.core_veffs[1],
                    ]
                else:
                    h1_ab = [fobj_a.h1, fobj_b.h1]

                e_f = get_frag_energy_u(
                    (fobj_a._mo_coeffs, fobj_b._mo_coeffs),
                    (fobj_a.nsocc, fobj_b.nsocc),
                    (fobj_a.nfsites, fobj_b.nfsites),
                    (fobj_a.efac, fobj_b.efac),
                    (fobj_a.TA, fobj_b.TA),
                    h1_ab,
                    hf_veff,
                    rdm1_tmp,
                    rdm2s,
                    fobj_a.dname,
                    eri_file=fobj_a.eri_file,
                    gcores=full_uhf.full_gcore,
                    frozen=frozen,
                )
                total_e = [sum(x) for x in zip(total_e, e_f)]

    if frag_energy:
        E = sum(total_e)
        return (E, total_e)


def solve_error(Fobjs, Nocc, only_chem=False):
    """
    Compute the error for self-consistent fragment density matrix matching.

    This function calculates the error in the one-particle density matrix for a given fragment,
    matching the density matrix elements of the edges and centers. It returns the norm of the error
    vector and the error vector itself.

    Parameters
    ----------
    Fobjs : list of MolBE.fragpart
        List of fragment objects.
    Nocc : int
        Number of occupied orbitals.

    Returns
    -------
    float
        Norm of the error vector.
    numpy.ndarray
        Error vector.
    """

    err_edge = []
    err_chempot = 0.0

    if only_chem:
        for fobj in Fobjs:
            # Compute chemical potential error for each fragment
            for i in fobj.efac[1]:
                err_chempot += fobj._rdm1[i, i]
        err_chempot /= Fobjs[0].unitcell_nkpt
        err = err_chempot - Nocc

        return abs(err), numpy.asarray([err])

    # Compute edge and chemical potential errors
    for fobj in Fobjs:
        # match rdm-edge
        for edge in fobj.edge_idx:
            for j_ in range(len(edge)):
                for k_ in range(len(edge)):
                    if j_ > k_:
                        continue
                    err_edge.append(fobj._rdm1[edge[j_], edge[k_]])
        # chem potential
        for i in fobj.efac[1]:
            err_chempot += fobj._rdm1[i, i]

    err_chempot /= Fobjs[0].unitcell_nkpt
    err_edge.append(err_chempot)  # far-end edges are included as err_chempot

    # Compute center errors
    err_cen = []
    for findx, fobj in enumerate(Fobjs):
        # Match RDM for centers
        for cindx, cens in enumerate(fobj.center_idx):
            lenc = len(cens)
            for j_ in range(lenc):
                for k_ in range(lenc):
                    if j_ > k_:
                        continue
                    err_cen.append(Fobjs[fobj.center[cindx]]._rdm1[cens[j_], cens[k_]])

    err_cen.append(Nocc)
    err_edge = numpy.array(err_edge)
    err_cen = numpy.array(err_cen)

    # Compute the error vector
    err_vec = err_edge - err_cen

    # Compute the norm of the error vector
    norm_ = numpy.mean(err_vec * err_vec) ** 0.5

    return norm_, err_vec


def solve_mp2(mf, frozen=None, mo_coeff=None, mo_occ=None, mo_energy=None):
    """
    Perform an MP2 (2nd order Moller-Plesset perturbation theory) calculation.

    This function sets up and runs an MP2 calculation using the provided mean-field object.
    It returns the MP2 object after the calculation.

    Parameters
    ----------
    mf : pyscf.scf.hf.RHF
        Mean-field object from PySCF.
    frozen : list or int, optional
        List of frozen orbitals or number of frozen core orbitals. Defaults to None.
    mo_coeff : numpy.ndarray, optional
        Molecular orbital coefficients. Defaults to None.
    mo_occ : numpy.ndarray, optional
        Molecular orbital occupations. Defaults to None.
    mo_energy : numpy.ndarray, optional
        Molecular orbital energies. Defaults to None.

    Returns
    -------
    pyscf.mp.mp2.MP2
        The MP2 object after running the calculation.
    """
    from pyscf import mp

    # Set default values for optional parameters
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    if mo_energy is None:
        mo_energy = mf.mo_energy
    if mo_occ is None:
        mo_occ = mf.mo_occ

    # Initialize the MP2 object
    pt__ = mp.MP2(mf, frozen=frozen, mo_coeff=mo_coeff, mo_occ=mo_occ)
    mf = None
    pt__.verbose = 0

    # Run the MP2 calculation
    pt__.kernel(mo_energy=mo_energy)

    return pt__


def solve_ccsd(
    mf,
    frozen=None,
    mo_coeff=None,
    relax=False,
    use_cumulant=False,
    with_dm1=True,
    rdm2_return=False,
    mo_occ=None,
    mo_energy=None,
    rdm_return=False,
    verbose=0,
):
    """
    Solve the CCSD (Coupled Cluster with Single and Double excitations) equations.

    This function sets up and solves the CCSD equations using the provided mean-field object.
    It can return the CCSD amplitudes (t1, t2), the one- and two-particle density matrices, and the CCSD object.

    Parameters
    ----------
    mf : pyscf.scf.hf.RHF
        Mean-field object from PySCF.
    frozen : list or int, optional
        List of frozen orbitals or number of frozen core orbitals. Defaults to None.
    mo_coeff : numpy.ndarray, optional
        Molecular orbital coefficients. Defaults to None.
    relax : bool, optional
        Whether to use relaxed density matrices. Defaults to False.
    use_cumulant : bool, optional
        Whether to use cumulant-based energy expression. Defaults to False.
    with_dm1 : bool, optional
        Whether to include one-particle density matrix in the two-particle density matrix calculation. Defaults to True.
    rdm2_return : bool, optional
        Whether to return the two-particle density matrix. Defaults to False.
    mo_occ : numpy.ndarray, optional
        Molecular orbital occupations. Defaults to None.
    mo_energy : numpy.ndarray, optional
        Molecular orbital energies. Defaults to None.
    rdm_return : bool, optional
        Whether to return the one-particle density matrix. Defaults to False.
    verbose : int, optional
        Verbosity level. Defaults to 0.

    Returns
    -------
    tuple
        - t1 (numpy.ndarray): Single excitation amplitudes.
        - t2 (numpy.ndarray): Double excitation amplitudes.
        - rdm1a (numpy.ndarray, optional): One-particle density matrix (if rdm_return is True).
        - rdm2s (numpy.ndarray, optional): Two-particle density matrix (if rdm2_return is True and rdm_return is True).
        - cc__ (pyscf.cc.ccsd.CCSD, optional): CCSD object (if rdm_return is True and rdm2_return is False).
    """
    from pyscf import cc
    from pyscf.cc.ccsd_rdm import make_rdm2

    # Set default values for optional parameters
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    if mo_energy is None:
        mo_energy = mf.mo_energy
    if mo_occ is None:
        mo_occ = mf.mo_occ

    # Initialize the CCSD object
    cc__ = cc.CCSD(mf, frozen=frozen, mo_coeff=mo_coeff, mo_occ=mo_occ)
    cc__.verbose = 0
    mf = None
    cc__.incore_complete = True

    # Prepare the integrals and Fock matrix
    eris = cc__.ao2mo()
    eris.mo_energy = mo_energy
    eris.fock = numpy.diag(mo_energy)

    # Solve the CCSD equations
    try:
        cc__.verbose = verbose
        cc__.kernel(eris=eris)
    except:
        print(flush=True)
        print(
            "Exception in CCSD -> applying level_shift=0.2, diis_space=25", flush=True
        )
        print(flush=True)
        cc__.verbose = 4
        cc__.diis_space = 25
        cc__.level_shift = 0.2
        cc__.kernel(eris=eris)

    # Extract the CCSD amplitudes
    t1 = cc__.t1
    t2 = cc__.t2

    # Compute and return the density matrices if requested
    if rdm_return:
        if not relax:
            l1 = numpy.zeros_like(t1)
            l2 = numpy.zeros_like(t2)
            rdm1a = cc.ccsd_rdm.make_rdm1(cc__, t1, t2, l1, l2)
        else:
            rdm1a = cc__.make_rdm1(with_frozen=False)

        if rdm2_return:
            if use_cumulant:
                with_dm1 = False
            rdm2s = make_rdm2(
                cc__,
                cc__.t1,
                cc__.t2,
                cc__.l1,
                cc__.l2,
                with_frozen=False,
                ao_repr=False,
                with_dm1=with_dm1,
            )
            return (t1, t2, rdm1a, rdm2s)
        return (t1, t2, rdm1a, cc__)

    return (t1, t2)


def solve_block2(mf: object, nocc: int, frag_scratch: str = None, **solver_kwargs):
    """DMRG fragment solver using the pyscf.dmrgscf wrapper.

    Parameters
    ----------
        mf: pyscf.scf.hf.RHF
            Mean field object or similar following the data signature of the pyscf.RHF class.
        nocc: int
            Number of occupied MOs in the fragment, used for constructing the fragment 1- and 2-RDMs.
        frag_scratch: str|pathlike, optional
            Fragment-level DMRG scratch directory.
        max_mem: int, optional
            Maximum memory in GB.
        root: int, optional
            Number of roots to solve for.
        startM: int, optional
            Starting MPS bond dimension - where the sweep schedule begins.
        maxM: int, optional
            Maximum MPS bond dimension - where the sweep schedule terminates.
        max_iter: int, optional
            Maximum number of sweeps.
        twodot_to_onedot: int, optional
            Sweep index at which to transition to one-dot DMRG algorithm. All sweeps prior to this will use the two-dot algorithm.
        block_extra_keyword: list(str), optional
            Other keywords to be passed to block2. See: https://block2.readthedocs.io/en/latest/user/keywords.html

    Returns
    -------
        rdm1: numpy.ndarray
            1-Particle reduced density matrix for fragment.
        rdm2: numpy.ndarray
            2-Particle reduced density matrix for fragment.

    Other Parameters
    ----------------
        schedule_kwargs: dict, optional
            Dictionary containing DMRG scheduling parameters to be passed to block2.

            e.g. The default schedule used here would be equivalent to the following:
            schedule_kwargs = {
                'scheduleSweeps': [0, 10, 20, 30, 40, 50],
                'scheduleMaxMs': [25, 50, 100, 200, 500, 500],
                'scheduleTols': [1e-5,1e-5, 1e-6, 1e-6, 1e-8, 1e-8],
                'scheduleNoises': [0.01, 0.01, 0.001, 0.001, 1e-4, 0.0],
            }

    Raises
    ------


    """
    from pyscf import dmrgscf, mcscf

    use_cumulant = solver_kwargs.pop("use_cumulant", True)
    norb = solver_kwargs.pop("norb", mf.mo_coeff.shape[1])
    nelec = solver_kwargs.pop("nelec", mf.mo_coeff.shape[1])
    lo_method = solver_kwargs.pop("lo_method", None)
    startM = solver_kwargs.pop("startM", 25)
    maxM = solver_kwargs.pop("maxM", 500)
    max_iter = solver_kwargs.pop("max_iter", 60)
    max_mem = solver_kwargs.pop("max_mem", 100)
    max_noise = solver_kwargs.pop("max_noise", 1e-3)
    min_tol = solver_kwargs.pop("min_tol", 1e-8)
    twodot_to_onedot = solver_kwargs.pop("twodot_to_onedot", int((5 * max_iter) // 6))
    root = solver_kwargs.pop("root", 0)
    block_extra_keyword = solver_kwargs.pop("block_extra_keyword", ["fiedler"])
    schedule_kwargs = solver_kwargs.pop("schedule_kwargs", {})

    if norb <= 2:
        block_extra_keyword = [
            "noreorder"
        ]  # Other reordering algorithms explode if the network is too small.

    if lo_method is None:
        orbs = mf.mo_coeff
    elif isinstance(lo_method, str):
        raise NotImplementedError(
            "Localization within the fragment+bath subspace is currently not supported."
        )

    mc = mcscf.CASCI(mf, norb, nelec)
    mc.fcisolver = dmrgscf.DMRGCI(mf.mol)
    ###Sweep scheduling
    mc.fcisolver.scheduleSweeps = schedule_kwargs.pop(
        "scheduleSweeps",
        [
            (1 * max_iter) // 6,
            (2 * max_iter) // 6,
            (3 * max_iter) // 6,
            (4 * max_iter) // 6,
            (5 * max_iter) // 6,
            max_iter,
        ],
    )
    mc.fcisolver.scheduleMaxMs = schedule_kwargs.pop(
        "scheduleMaxMs",
        [
            startM if (startM < maxM) else maxM,
            startM * 2 if (startM * 2 < maxM) else maxM,
            startM * 4 if (startM * 4 < maxM) else maxM,
            startM * 8 if (startM * 8 < maxM) else maxM,
            maxM,
            maxM,
        ],
    )
    mc.fcisolver.scheduleTols = schedule_kwargs.pop(
        "scheduleTols",
        [min_tol * 1e3, min_tol * 1e3, min_tol * 1e2, min_tol * 1e1, min_tol, min_tol],
    )
    mc.fcisolver.scheduleNoises = schedule_kwargs.pop(
        "scheduleNoises",
        [max_noise, max_noise, max_noise / 10, max_noise / 100, max_noise / 100, 0.0],
    )
    ###Other DMRG parameters
    mc.fcisolver.threads = int(os.environ.get("OMP_NUM_THREADS", 8))
    mc.fcisolver.twodot_to_onedot = int(twodot_to_onedot)
    mc.fcisolver.maxIter = int(max_iter)
    mc.fcisolver.block_extra_keyword = list(block_extra_keyword)
    mc.fcisolver.scratchDirectory = str(frag_scratch)
    mc.fcisolver.runtimeDir = str(frag_scratch)
    mc.fcisolver.memory = int(max_mem)
    os.system("cd " + frag_scratch)

    mc.kernel(orbs)
    rdm1, rdm2 = dmrgscf.DMRGCI.make_rdm12(mc.fcisolver, root, norb, nelec)

    ###Subtract off non-cumulant contribution to correlated 2RDM.
    if use_cumulant:
        hf_dm = numpy.zeros_like(rdm1)
        hf_dm[numpy.diag_indices(nocc)] += 2.0

        del_rdm1 = rdm1.copy()
        del_rdm1[numpy.diag_indices(nocc)] -= 2.0
        nc = (
            numpy.einsum("ij,kl->ijkl", hf_dm, hf_dm)
            + numpy.einsum("ij,kl->ijkl", hf_dm, del_rdm1)
            + numpy.einsum("ij,kl->ijkl", del_rdm1, hf_dm)
        )
        nc -= (
            numpy.einsum("ij,kl->iklj", hf_dm, hf_dm)
            + numpy.einsum("ij,kl->iklj", hf_dm, del_rdm1)
            + numpy.einsum("ij,kl->iklj", del_rdm1, hf_dm)
        ) * 0.5

        rdm2 -= nc

    return rdm1, rdm2


def solve_uccsd(
    mf,
    eris_inp,
    frozen=None,
    mo_coeff=None,
    relax=False,
    use_cumulant=False,
    with_dm1=True,
    rdm2_return=False,
    mo_occ=None,
    mo_energy=None,
    rdm_return=False,
    verbose=0,
):
    """
    Solve the U-CCSD (Unrestricted Coupled Cluster with Single and Double excitations) equations.

    This function sets up and solves the UCCSD equations using the provided mean-field object.
    It can return the one- and two-particle density matrices and the UCCSD object.

    Parameters
    ----------
    mf : pyscf.scf.hf.UHF
        Mean-field object from PySCF. Constructed with make_uhf_obj
    eris_inp :
        Custom fragment ERIs object
    frozen : list or int, optional
        List of frozen orbitals or number of frozen core orbitals. Defaults to None.
    mo_coeff : numpy.ndarray, optional
        Molecular orbital coefficients. Defaults to None.
    relax : bool, optional
        Whether to use relaxed density matrices. Defaults to False.
    use_cumulant : bool, optional
        Whether to use cumulant-based energy expression. Defaults to False.
    with_dm1 : bool, optional
        Whether to include one-particle density matrix in the two-particle density matrix calculation. Defaults to True.
    rdm2_return : bool, optional
        Whether to return the two-particle density matrix. Defaults to False.
    mo_occ : numpy.ndarray, optional
        Molecular orbital occupations. Defaults to None.
    mo_energy : numpy.ndarray, optional
        Molecular orbital energies. Defaults to None.
    rdm_return : bool, optional
        Whether to return the one-particle density matrix. Defaults to False.
    verbose : int, optional
        Verbosity level. Defaults to 0.

    Returns
    -------
    tuple
        - ucc (pyscf.cc.ccsd.UCCSD): UCCSD object
        - rdm1 (tuple, numpy.ndarray, optional): One-particle density matrix (if rdm_return is True).
        - rdm2 (tuple, numpy.ndarray, optional): Two-particle density matrix (if rdm2_return is True and rdm_return is True).
    """
    from pyscf import ao2mo, cc

    from molbe.external.uccsd_eri import make_eris_incore

    C = mf.mo_coeff
    nao = [C[s].shape[0] for s in [0, 1]]

    Vss = eris_inp[:2]
    Vos = eris_inp[-1]

    def ao2mofn(moish):
        if isinstance(moish, numpy.ndarray):
            # Since inside '_make_eris_incore' it does not differentiate spin
            # for the two same-spin components, we here brute-forcely determine
            # what spin component we are dealing with by comparing the first
            # 2-by-2 block of the mo coeff matrix.
            # Note that this assumes we have at least two basis functions
            moish_feature = moish[:2, :2]
            s = -1
            for ss in [0, 1]:
                if numpy.allclose(moish_feature, C[ss][:2, :2]):
                    s = ss
                    break
            if s < 0:
                raise RuntimeError("Input mo coeff matrix matches neither moa nor mob.")
            return ao2mo.incore.full(Vss[s], moish, compact=False)
        elif isinstance(moish, list) or isinstance(moish, tuple):
            if len(moish) != 4:
                raise RuntimeError(
                    "Expect a list/tuple of 4 numpy arrays but get %d of them."
                    % len(moish)
                )
            moish_feature = [mo[:2, :2] for mo in moish]
            for s in [0, 1]:
                Cs_feature = C[s][:2, :2]
                if not (
                    numpy.allclose(moish_feature[2 * s], Cs_feature)
                    and numpy.allclose(moish_feature[2 * s + 1], Cs_feature)
                ):
                    raise RuntimeError(
                        "Expect a list/tuple of 4 numpy arrays in the order (moa,moa,mob,mob)."
                    )
            try:
                return ao2mo.incore.general(Vos, moish, compact=False)
            except:
                return numpy.einsum(
                    "ijkl,ip,jq,kr,ls->pqrs",
                    Vos,
                    moish[0],
                    moish[1],
                    moish[2],
                    moish[3],
                    optimize=True,
                )
        else:
            raise RuntimeError(
                "moish must be either a numpy array or a list/tuple of 4 numpy arrays."
            )

    # Initialize the UCCSD object
    ucc = cc.uccsd.UCCSD(mf, mo_coeff=mf.mo_coeff, mo_occ=mf.mo_occ)

    # Prepare the integrals
    eris = make_eris_incore(
        ucc, Vss, Vos, mo_coeff=mf.mo_coeff, ao2mofn=ao2mofn, frozen=frozen
    )

    # Solve UCCSD equations: Level shifting options to be tested for unrestricted code
    ucc.verbose = verbose
    ucc.kernel(eris=eris)

    # Compute and return the density matrices if requested
    if rdm_return:
        rdm1 = make_rdm1_uccsd(ucc, relax=relax)
        if rdm2_return:
            if use_cumulant:
                with_dm1 = False
            rdm2 = make_rdm2_uccsd(ucc, relax=relax, with_dm1=with_dm1)
            return (ucc, rdm1, rdm2)
        return (ucc, rdm1, None)
    return ucc


def schmidt_decomposition(
    mo_coeff, nocc, Frag_sites, cinv=None, rdm=None, norb=None, return_orb_count=False
):
    """
    Perform Schmidt decomposition on the molecular orbital coefficients.

    This function decomposes the molecular orbitals into fragment and environment parts
    using the Schmidt decomposition method. It computes the transformation matrix (TA)
    which includes both the fragment orbitals and the entangled bath.

    Parameters
    ----------
    mo_coeff : numpy.ndarray
        Molecular orbital coefficients.
    nocc : int
        Number of occupied orbitals.
    Frag_sites : list of int
        List of fragment sites (indices).
    cinv : numpy.ndarray, optional
        Inverse of the transformation matrix. Defaults to None.
    rdm : numpy.ndarray, optional
        Reduced density matrix. If not provided, it will be computed from the molecular orbitals. Defaults to None.
    norb : int, optional
        Specifies number of bath orbitals. Used for UBE to make alpha and beta
        spaces the same size. Defaults to None
    return_orb_count : bool, optional
        Return more information about the number of orbitals. Used in UBE.
        Defaults to False

    Returns
    -------
    numpy.ndarray
        Transformation matrix (TA) including both fragment and entangled bath orbitals.
    if return_orb_count:
        numpy.ndarray, int, int
        returns TA (above), number of orbitals in the fragment space, and number of orbitals in bath space
    """

    import functools

    # Threshold for eigenvalue significance
    thres = 1.0e-10

    # Compute the reduced density matrix (RDM) if not provided
    if mo_coeff is not None:
        C = mo_coeff[:, :nocc]
    if rdm is None:
        Dhf = numpy.dot(C, C.T)
        if cinv is not None:
            Dhf = functools.reduce(numpy.dot, (cinv, Dhf, cinv.conj().T))
    else:
        Dhf = rdm

    # Total number of sites
    Tot_sites = Dhf.shape[0]

    # Identify environment sites (indices not in Frag_sites)
    Env_sites1 = numpy.array([i for i in range(Tot_sites) if i not in Frag_sites])
    Env_sites = numpy.array([[i] for i in range(Tot_sites) if i not in Frag_sites])
    Frag_sites1 = numpy.array([[i] for i in Frag_sites])

    # Compute the environment part of the density matrix
    Denv = Dhf[Env_sites, Env_sites.T]

    # Perform eigenvalue decomposition on the environment density matrix
    Eval, Evec = numpy.linalg.eigh(Denv)

    # Identify significant environment orbitals based on eigenvalue threshold
    Bidx = []

    # Set the number of orbitals to be taken from the environment orbitals
    # Based on an eigenvalue threshold ordering
    if norb is not None:
        n_frag_ind = len(Frag_sites1)
        n_bath_ind = norb - n_frag_ind
        ind_sort = numpy.argsort(numpy.abs(Eval))
        first_el = [x for x in ind_sort if x < 1.0 - thres][-1 * n_bath_ind]
        for i in range(len(Eval)):
            if numpy.abs(Eval[i]) >= first_el:
                Bidx.append(i)
    else:
        for i in range(len(Eval)):
            if thres < numpy.abs(Eval[i]) < 1.0 - thres:
                Bidx.append(i)

    # Initialize the transformation matrix (TA)
    TA = numpy.zeros([Tot_sites, len(Frag_sites) + len(Bidx)])
    TA[Frag_sites, : len(Frag_sites)] = numpy.eye(len(Frag_sites))  # Fragment part
    TA[Env_sites1, len(Frag_sites) :] = Evec[:, Bidx]  # Environment part

    if return_orb_count:
        # return TA, norbs_frag, norbs_bath
        return TA, Frag_sites1.shape[0], len(Bidx)
    else:
        return TA
