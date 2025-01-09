# Author(s): Oinam Romesh Meitei, Leah Weisburn

import os
from multiprocessing.pool import ThreadPool as Pool

import numpy
from numpy import float64
from numpy.linalg import multi_dot
from pyscf import ao2mo, fci, mcscf

from quemb.kbe.pfrag import Frags as pFrags
from quemb.molbe.helper import (
    get_eri,
    get_frag_energy,
    get_frag_energy_u,
    get_scfObj,
)
from quemb.molbe.pfrag import Frags
from quemb.molbe.solver import (
    SHCI_Args,
    SHCI_ArgsUser,
    UserSolverArgs,
    make_rdm1_ccsd_t1,
    make_rdm2_urlx,
    solve_ccsd,
    solve_error,
    solve_mp2,
    solve_uccsd,
)
from quemb.shared.external.ccsd_rdm import make_rdm1_uccsd, make_rdm2_uccsd
from quemb.shared.external.unrestricted_utils import make_uhf_obj
from quemb.shared.helper import unused
from quemb.shared.manage_scratch import WorkDir
from quemb.shared.typing import Matrix


def run_solver(
    h1: Matrix[float64],
    dm0: Matrix[float64],
    scratch_dir: WorkDir,
    dname: str,
    nao: int,
    nocc: int,
    nfsites: int,
    efac: float,
    TA: Matrix[float64],
    h1_e: Matrix[float64],
    solver: str = "MP2",
    eri_file: str = "eri_file.h5",
    veff: Matrix[float64] | None = None,
    veff0: Matrix[float64] | None = None,
    ompnum: int = 4,
    writeh1: bool = False,
    eeval: bool = True,
    ret_vec: bool = False,
    use_cumulant: bool = True,
    relax_density: bool = False,
    solver_args: UserSolverArgs | None = None,
):
    """
    Run a quantum chemistry solver to compute the reduced density matrices.

    Parameters
    ----------
    h1 :
        One-electron Hamiltonian matrix.
    dm0 :
        Initial guess for the density matrix.
    scratch_dir :
        The scratch dir root.
    dname :
        Directory name for storing intermediate files.
        Fragment files will be stored in :code:`scratch_dir / dname`.
    scratch_dir :
        The scratch directory.
        Fragment files will be stored in :code:`scratch_dir / dname`.
    nao :
        Number of atomic orbitals.
    nocc :
        Number of occupied orbitals.
    nfsites :
        Number of fragment sites.
    efac :
        Scaling factor for the electronic energy.
    TA :
        Transformation matrix for embedding orbitals.
    h1_e :
        One-electron integral matrix.
    solver :
        Solver to use for the calculation ('MP2', 'CCSD', 'FCI', 'HCI', 'SHCI', 'SCI').
        Default is 'MP2'.
    eri_file :
        Filename for the electron repulsion integrals. Default is 'eri_file.h5'.
    veff :
        Veff matrix to be passed to energy, if non-cumulant energy.
    veff0 :
        Veff0 matrix, passed to energy, the hf_veff in the fragment Schmidt space
    ompnum :
        Number of OpenMP threads. Default is 4.
    writeh1 :
        If True, write the one-electron integrals to a file. Default is False.
    use_cumulant :
        If True, use the cumulant approximation for RDM2. Default is True.
    eeval :
        If True, evaluate the electronic energy. Default is True.
    ret_vec :
        If True, return vector with error and rdms. Default is True.
    relax_density :
        If True, use CCSD relaxed density. Default is False

    Returns
    -------
    tuple
        Depending on the input parameters, returns the molecular orbital coefficients,
        one-particle and two-particle reduced density matrices, and optionally the
        fragment energy.
    """

    # Get electron repulsion integrals (ERI)
    eri = get_eri(dname, nao, eri_file=eri_file)
    # Initialize SCF object
    mf_ = get_scfObj(h1, eri, nocc, dm0=dm0)

    # Select solver
    if solver == "MP2":
        mc_ = solve_mp2(mf_, mo_energy=mf_.mo_energy)
        rdm1_tmp = mc_.make_rdm1()

    elif solver == "CCSD":
        if not relax_density:
            t1, t2 = solve_ccsd(mf_, mo_energy=mf_.mo_energy, rdm_return=False)
            rdm1_tmp = make_rdm1_ccsd_t1(t1)
        else:
            t1, t2, rdm1_tmp, rdm2s = solve_ccsd(
                mf_,
                mo_energy=mf_.mo_energy,
                rdm_return=True,
                rdm2_return=True,
                use_cumulant=use_cumulant,
                relax=True,
            )
    elif solver == "FCI":
        mc_ = fci.FCI(mf_, mf_.mo_coeff)
        efci, civec = mc_.kernel()
        unused(efci)
        rdm1_tmp = mc_.make_rdm1(civec, mc_.norb, mc_.nelec)
    elif solver == "HCI":
        # pylint: disable-next=E0611
        from pyscf import hci  # noqa: PLC0415  # hci is an optional module

        assert isinstance(solver_args, SHCI_ArgsUser)
        SHCI_args = SHCI_Args.from_user_input(solver_args)

        nao, nmo = mf_.mo_coeff.shape
        eri = ao2mo.kernel(mf_._eri, mf_.mo_coeff, aosym="s4", compact=False).reshape(
            4 * ((nmo),)
        )
        ci_ = hci.SCI(mf_.mol)

        ci_.select_cutoff = SHCI_args.select_cutoff
        ci_.ci_coeff_cutoff = SHCI_args.ci_coeff_cutoff

        nelec = (nocc, nocc)
        h1_ = multi_dot((mf_.mo_coeff.T, h1, mf_.mo_coeff))
        eci, civec = ci_.kernel(h1_, eri, nmo, nelec)
        unused(eci)
        civec = numpy.asarray(civec)

        (rdm1a_, rdm1b_), (rdm2aa, rdm2ab, rdm2bb) = ci_.make_rdm12s(civec, nmo, nelec)
        rdm1_tmp = rdm1a_ + rdm1b_
        rdm2s = rdm2aa + rdm2ab + rdm2ab.transpose(2, 3, 0, 1) + rdm2bb

    elif solver == "SHCI":
        # pylint: disable-next=E0401,E0611
        from pyscf.shciscf import shci  # noqa: PLC0415    # shci is an optional module

        frag_scratch = WorkDir(scratch_dir / dname)

        assert isinstance(solver_args, SHCI_ArgsUser)
        SHCI_args = SHCI_Args.from_user_input(solver_args)

        nao, nmo = mf_.mo_coeff.shape
        nelec = (nocc, nocc)
        mch = shci.SHCISCF(mf_, nmo, nelec, orbpath=frag_scratch)
        mch.fcisolver.mpiprefix = "mpirun -np " + str(ompnum)
        mch.fcisolver.stochastic = True  # this is for PT and doesnt add PT to rdm
        mch.fcisolver.nPTiter = 0
        mch.fcisolver.sweep_iter = [0]
        mch.fcisolver.DoRDM = True
        mch.fcisolver.sweep_epsilon = [solver_args.hci_cutoff]
        mch.fcisolver.scratchDirectory = frag_scratch
        if not writeh1:
            mch.fcisolver.restart = True
        mch.mc1step()
        rdm1_tmp, rdm2s = mch.fcisolver.make_rdm12(0, nmo, nelec)

    elif solver == "SCI":
        # pylint: disable-next=E0611
        from pyscf import cornell_shci  # noqa: PLC0415  # optional module

        assert isinstance(solver_args, SHCI_ArgsUser)
        SHCI_args = SHCI_Args.from_user_input(solver_args)

        frag_scratch = WorkDir(scratch_dir / dname)

        nao, nmo = mf_.mo_coeff.shape
        nelec = (nocc, nocc)
        cas = mcscf.CASCI(mf_, nmo, nelec)
        h1, ecore = cas.get_h1eff(mo_coeff=mf_.mo_coeff)
        unused(ecore)
        eri = ao2mo.kernel(mf_._eri, mf_.mo_coeff, aosym="s4", compact=False).reshape(
            4 * ((nmo),)
        )

        ci = cornell_shci.SHCI()
        ci.runtimedir = frag_scratch
        ci.restart = True
        ci.config["var_only"] = True
        ci.config["eps_vars"] = [solver_args.hci_cutoff]
        ci.config["get_1rdm_csv"] = True
        ci.config["get_2rdm_csv"] = True
        ci.kernel(h1, eri, nmo, nelec)
        rdm1_tmp, rdm2s = ci.make_rdm12(0, nmo, nelec)

    else:
        raise ValueError("Solver not implemented")

    # Compute RDM1
    rdm1 = multi_dot((mf_.mo_coeff, rdm1_tmp, mf_.mo_coeff.T)) * 0.5

    if eeval:
        if solver == "CCSD" and not relax_density:
            rdm2s = make_rdm2_urlx(t1, t2, with_dm1=not use_cumulant)
        elif solver == "MP2":
            rdm2s = mc_.make_rdm2()
        elif solver == "FCI":
            rdm2s = mc_.make_rdm2(civec, mc_.norb, mc_.nelec)
            if use_cumulant:
                hf_dm = numpy.zeros_like(rdm1_tmp)
                hf_dm[numpy.diag_indices(nocc)] += 2.0
                del_rdm1 = rdm1_tmp.copy()
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
                rdm2s -= nc
        e_f = get_frag_energy(
            mf_.mo_coeff,
            nocc,
            nfsites,
            efac,
            TA,
            h1_e,
            rdm1_tmp,
            rdm2s,
            dname,
            veff0,
            veff,
            use_cumulant,
            eri_file,
        )
    if eeval and not ret_vec:
        return e_f

    return (e_f, mf_.mo_coeff, rdm1, rdm2s, rdm1_tmp)


def run_solver_u(
    fobj_a,
    fobj_b,
    solver,
    enuc,  # noqa: ARG001
    hf_veff,
    relax_density=False,
    frozen=False,
    use_cumulant=True,
    ereturn=True,
):
    """
    Run a quantum chemistry solver to compute the reduced density matrices.

    Parameters
    ----------
    fobj_a :
        Alpha spin molbe.pfrag.Frags object
    fobj_b :
        Beta spin molbe.pfrag.Frags object
    solver : str
        High-level solver in bootstrap embedding. Supported value is "UCCSD"
    enuc : float
        Nuclear component of the energy
    hf_veff : tuple of numpy.ndarray, optional
        Alpha and beta spin Hartree-Fock effective potentials.
    relax_density : bool, optional
        If True, uses  relaxed density matrix for UCCSD, defaults to False.
    frozen : bool, optional
        If True, uses frozen core, defaults to False
    use_cumulant : bool, optional
        If True, uses the cumulant approximation for RDM2. Default is True.
    ereturn : bool, optional
        If True, return the computed energy. Defaults to False.

    Returns
    -------
    float
        As implemented, only returns the UCCSD fragment energy
    """
    print("obj type", type(fobj_a))
    # Run SCF for alpha and beta spins
    fobj_a.scf(unrestricted=True, spin_ind=0)
    fobj_b.scf(unrestricted=True, spin_ind=1)

    # Construct UHF object
    full_uhf, eris = make_uhf_obj(fobj_a, fobj_b, frozen=frozen)

    if solver == "UCCSD":
        if relax_density:
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
        raise NotImplementedError("Only UCCSD Solver implemented")

    # Compute RDM1
    fobj_a.rdm1__ = rdm1_tmp[0].copy()
    fobj_a._rdm1 = (
        multi_dot((fobj_a._mf.mo_coeff, rdm1_tmp[0], fobj_a._mf.mo_coeff.T)) * 0.5
    )

    fobj_b.rdm1__ = rdm1_tmp[1].copy()
    fobj_b._rdm1 = (
        multi_dot((fobj_b._mf.mo_coeff, rdm1_tmp[1], fobj_b._mf.mo_coeff.T)) * 0.5
    )

    # Calculate Energies
    if ereturn:
        if solver == "UCCSD" and not relax_density:
            rdm2s = make_rdm2_uccsd(ucc, with_dm1=not use_cumulant)

        fobj_a.rdm2__ = rdm2s[0].copy()
        fobj_b.rdm2__ = rdm2s[1].copy()

        # Calculate energy on a per-fragment basis
        if frozen:
            h1_ab = [
                full_uhf.h1[0] + full_uhf.full_gcore[0] + full_uhf.core_veffs[0],
                full_uhf.h1[1] + full_uhf.full_gcore[1] + full_uhf.core_veffs[1],
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
        return e_f


def be_func_parallel(
    pot: list[float] | None,
    Fobjs: list[Frags] | list[pFrags],
    Nocc: int,
    solver: str,
    enuc: float,  # noqa: ARG001
    scratch_dir: WorkDir,
    solver_args: UserSolverArgs | None,
    nproc: int = 1,
    ompnum: int = 4,
    only_chem: bool = False,
    relax_density: bool = False,
    use_cumulant: bool = True,
    eeval: bool = False,
    return_vec: bool = False,
    writeh1: bool = False,
):
    """
    Embarrassingly Parallel High-Level Computation

    Performs high-level bootstrap embedding (BE) computation for each fragment.
    Computes 1-RDMs and 2-RDMs for each fragment. It also computes error vectors
    in BE density match. For selected CI solvers, this function exposes thresholds used
    in selected CI calculations (hci_cutoff, ci_coeff_cutoff, select_cutoff).

    Parameters
    ----------
    pot :
        Potentials (local & global) that are added to the 1-electron Hamiltonian
        component.  The last element in the list is the chemical potential.
    Fobjs :
        Fragment definitions.
    Nocc :
        Number of occupied orbitals for the full system.
    solver :
        High-level solver in bootstrap embedding. Supported values are 'MP2', 'CCSD',
        'FCI', 'HCI', 'SHCI', and 'SCI'.
    enuc :
        Nuclear component of the energy.
    nproc :
        Total number of processors assigned for the optimization. Defaults to 1.
        When nproc > 1, Python multithreading is invoked.
    ompnum :
        If nproc > 1, sets the number of cores for OpenMP parallelization.
        Defaults to 4.
    only_chem :
        Whether to perform chemical potential optimization only.
        Refer to bootstrap embedding literature. Defaults to False.
    eeval :
        Whether to evaluate energies. Defaults to False.
    scratch_dir :
        Scratch directory root
    use_cumulant :
        Use cumulant energy expression. Defaults to True
    return_vec :
        Whether to return the error vector. Defaults to False.
    writeh1 :
        Whether to write the one-electron integrals. Defaults to False.

    Returns
    -------
    float or tuple
        Depending on the parameters, returns the error norm or a tuple containing
        the error norm, error vector, and the computed energy.
    """
    # Set the number of OpenMP threads
    os.system("export OMP_NUM_THREADS=" + str(ompnum))
    nprocs = nproc // ompnum

    # Update the effective Hamiltonian with potentials
    if pot is not None:
        for fobj in Fobjs:
            fobj.update_heff(pot, only_chem=only_chem)

    with Pool(nprocs) as pool_:
        results = []
        # Run solver in parallel for each fragment
        for fobj in Fobjs:
            result = pool_.apply_async(
                run_solver,
                [
                    fobj.fock + fobj.heff,
                    fobj.dm0.copy(),
                    scratch_dir,
                    fobj.dname,
                    fobj.nao,
                    fobj.nsocc,
                    fobj.nfsites,
                    fobj.efac,
                    fobj.TA,
                    fobj.h1,
                    solver,
                    fobj.eri_file,
                    fobj.veff if not use_cumulant else None,
                    fobj.veff0,
                    ompnum,
                    writeh1,
                    eeval,
                    return_vec,
                    use_cumulant,
                    relax_density,
                    solver_args,
                ],
            )

            results.append(result)

        rdms = [result.get() for result in results]

    if not return_vec:
        # Compute and return fragment energy
        # rdms are the returned energies, not density matrices!
        e_1 = 0.0
        e_2 = 0.0
        e_c = 0.0
        for i in range(len(rdms)):
            e_1 += rdms[i][0]
            e_2 += rdms[i][1]
            e_c += rdms[i][2]
        return (e_1 + e_2 + e_c, (e_1, e_2, e_c))

    # Compute total energy
    e_1 = 0.0
    e_2 = 0.0
    e_c = 0.0

    # I have to type ignore here, because of stupid behaviour of
    # :code:`zip` and :code:`enumerate`
    # https://stackoverflow.com/questions/74374059/correctly-specify-the-types-of-unpacked-zip
    for fobj, rdm in zip(Fobjs, rdms):  # type: ignore[assignment]
        e_1 += rdm[0][0]
        e_2 += rdm[0][1]
        e_c += rdm[0][2]
        fobj.mo_coeffs = rdm[1]
        fobj._rdm1 = rdm[2]
        fobj.rdm2__ = rdm[3]

    del rdms
    ernorm, ervec = solve_error(Fobjs, Nocc, only_chem=only_chem)

    if return_vec:
        return (ernorm, ervec, [e_1 + e_2 + e_c, [e_1, e_2, e_c]])

    return ernorm


def be_func_parallel_u(
    pot,  # noqa: ARG001
    Fobjs,
    solver,
    enuc,
    hf_veff=None,
    nproc=1,
    ompnum=4,
    relax_density=False,
    use_cumulant=True,
    frozen=False,
):
    """
    Embarrassingly Parallel High-Level Computation

    Performs high-level unrestricted bootstrap embedding (UBE) computation for each
    fragment. Computes 1-RDMs and 2-RDMs for each fragment to return the energy.
    As such, this currently is equipped for one-shot U-CCSD BE.

    Parameters
    ----------
    pot : list of float
        Potentials (local & global) that are added to the 1-electron
        Hamiltonian component.  The last element in the list is the chemical potential.
        Should always be 0, as this is still a one-shot only implementation
    Fobjs : list of tuple of quemb.molbe.fragment.fragpart
        Fragment definitions, alpha and beta components.
    solver : str
        High-level solver in bootstrap embedding. Supported value is 'UCCSD'.
    enuc : float
        Nuclear component of the energy.
    hf_veff : tuple of numpy.ndarray, optional
        Alpha and beta Hartree-Fock effective potential.
    nproc : int, optional
        Total number of processors assigned for the optimization. Defaults to 1.
        When nproc > 1, Python multithreading is invoked.
    ompnum : int, optional
        If nproc > 1, sets the number of cores for OpenMP parallelization.
        Defaults to 4.
    use_cumulant :
        Whether to use the cumulant energy expression, by default True.
    frozen : bool, optional
        Frozen core. Defaults to False

    Returns
    -------
    float
        Returns the computed energy
    """
    # Set the number of OpenMP threads
    os.system("export OMP_NUM_THREADS=" + str(ompnum))
    nprocs = nproc // ompnum

    with Pool(nprocs) as pool_:
        results = []
        # Run solver in parallel for each fragment
        for fobj_a, fobj_b in Fobjs:
            result = pool_.apply_async(
                run_solver_u,
                [
                    fobj_a,
                    fobj_b,
                    solver,
                    enuc,
                    hf_veff,
                    relax_density,
                    frozen,
                    use_cumulant,
                    True,
                ],
            )
            results.append(result)

        energy_list = [result.get() for result in results]

    # Compute and return fragment energy
    e_1 = 0.0
    e_2 = 0.0
    e_c = 0.0
    for i in range(len(energy_list)):
        e_1 += energy_list[i][0]
        e_2 += energy_list[i][1]
        e_c += energy_list[i][2]
    return (e_1 + e_2 + e_c, (e_1, e_2, e_c))
