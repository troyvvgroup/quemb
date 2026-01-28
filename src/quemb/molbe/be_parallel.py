# Author(s): Oinam Romesh Meitei, Leah Weisburn

import os
from pathlib import Path
from warnings import warn

from numpy import diag_indices, einsum, float64, zeros_like
from numpy.linalg import multi_dot
from pathos.pools import ProcessPool
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
    SHCI_ArgsUser,
    Solvers,
    UserSolverArgs,
    _SHCI_Args,
    solve_ccsd,
    solve_error,
    solve_mp2,
    solve_uccsd,
)
from quemb.shared.external.unrestricted_utils import make_uhf_obj
from quemb.shared.helper import unused
from quemb.shared.manage_scratch import WorkDir
from quemb.shared.typing import (
    ListOverFrag,
    Matrix,
    RelAOIdx,
)


def run_solver(
    h1: Matrix[float64],
    dm0: Matrix[float64],
    scratch_dir: WorkDir,
    dname: str,
    nao: int,
    nocc: int,
    n_frag: int,
    weight_and_relAO_per_center: ListOverFrag[tuple[float, list[RelAOIdx]]],
    TA: Matrix[float64],
    h1_e: Matrix[float64],
    solver: Solvers = "CCSD",
    eri_file: str = "eri_file.h5",
    veff: Matrix[float64] | None = None,
    veff0: Matrix[float64] | None = None,
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
    n_frag :
        Number of fragment sites.
    weight_and_relAO_per_center :
        Scaling factor for the electronic energy **and**
        the relative AO indices per center per frag
    TA :
        Transformation matrix for embedding orbitals.
    h1_e :
        One-electron integral matrix.
    solver :
        Solver to use for the calculation.
        Default is 'CCSD'.
    eri_file :
        Filename for the electron repulsion integrals. Default is 'eri_file.h5'.
    veff :
        Veff matrix to be passed to energy, if non-cumulant energy.
    veff0 :
        Veff0 matrix, passed to energy, the hf_veff in the fragment Schmidt space
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
        mc_mp2 = solve_mp2(mf_, mo_energy=mf_.mo_energy)
        rdm1_tmp = mc_mp2.make_rdm1()
        if eeval:
            rdm2s = mc_mp2.make_rdm2()

    elif solver == "CCSD":
        if eeval:
            mycc, t1, t2, rdm1_tmp, rdm2s = solve_ccsd(
                mf_,
                mo_energy=mf_.mo_energy,
                relax=relax_density,
                use_cumulant=use_cumulant,
                rdm_return=True,
                rdm2_return=True,
            )
        else:
            mycc, t1, t2, rdm1_tmp, _ = solve_ccsd(
                mf_,
                mo_energy=mf_.mo_energy,
                relax=relax_density,
                use_cumulant=use_cumulant,
                rdm_return=True,
                rdm2_return=False,
            )

    elif solver == "FCI":
        mc_fci = fci.FCI(mf_, mf_.mo_coeff)
        efci, civec = mc_fci.kernel()
        unused(efci)
        rdm1_tmp = mc_fci.make_rdm1(civec, mc_fci.norb, mc_fci.nelec)

    elif solver == "HCI":  # TODO
        # pylint: disable-next=E0611
        raise NotImplementedError("HCI solver not implemented")
        """
        from pyscf import hci  # type: ignore[attr-defined]  # noqa: PLC0415

        assert isinstance(solver_args, SHCI_ArgsUser)
        SHCI_args = _SHCI_Args.from_user_input(solver_args)

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
        civec = asarray(civec) # import numpy.asarray

        (rdm1a_, rdm1b_), (rdm2aa, rdm2ab, rdm2bb) = ci_.make_rdm12s(civec, nmo, nelec)
        rdm1_tmp = rdm1a_ + rdm1b_
        rdm2s = rdm2aa + rdm2ab + rdm2ab.transpose(2, 3, 0, 1) + rdm2bb
        """

    elif solver == "SHCI":  # TODO
        # pylint: disable-next=E0401,E0611
        raise NotImplementedError("SHCI solver not implemented")
        """
        from pyscf.shciscf import shci  # type: ignore[attr-defined]  # noqa: PLC0415

        frag_scratch = scratch_dir.make_subdir(dname)

        assert isinstance(solver_args, SHCI_ArgsUser)
        SHCI_args = _SHCI_Args.from_user_input(solver_args)

        nao, nmo = mf_.mo_coeff.shape
        nelec = (nocc, nocc)
        mch = shci.SHCISCF(mf_, nmo, nelec, orbpath=frag_scratch)
        mch.fcisolver.mpiprefix = "mpirun -np " + str(ompnum) # need to pass in ompnum
        mch.fcisolver.stochastic = True  # this is for PT and doesnt add PT to rdm
        mch.fcisolver.nPTiter = 0
        mch.fcisolver.sweep_iter = [0]
        mch.fcisolver.DoRDM = True
        mch.fcisolver.sweep_epsilon = [solver_args.hci_cutoff]
        mch.fcisolver.scratchDirectory = frag_scratch
        if not writeh1: # writeh1 specifies whether to write the 1e integrals
            mch.fcisolver.restart = True
        mch.mc1step()
        rdm1_tmp, rdm2s = mch.fcisolver.make_rdm12(0, nmo, nelec)
        """

    elif solver == "SCI":
        # pylint: disable-next=E0611
        from pyscf import cornell_shci  # type: ignore[attr-defined]  # noqa: PLC0415

        assert isinstance(solver_args, SHCI_ArgsUser)
        SHCI_args = _SHCI_Args.from_user_input(solver_args)

        assert isinstance(dname, str)

        nmo = mf_.mo_coeff.shape[1]
        nelec = (nocc, nocc)
        cas = mcscf.CASCI(mf_, nmo, nelec)
        h1, ecore = cas.get_h1eff(mo_coeff=mf_.mo_coeff)
        unused(ecore)
        eri = ao2mo.kernel(mf_._eri, mf_.mo_coeff, aosym="s4", compact=False).reshape(
            4 * ((nmo),)
        )

        if SHCI_args.return_frag_data:
            warn(
                "If return_frag_data is True, RDMs and other data"
                "are written into a directory which is not"
                "cleaned: cleanup_at_end is False"
            )
            iter = 0
            frag_name = Path(f"{scratch_dir}-frag_data") / f"{dname}_iter{iter}"
            while frag_name.exists():
                iter += 1
                frag_name = Path(f"{scratch_dir}-frag_data") / f"{dname}_iter{iter}"
            frag_scratch = WorkDir(frag_name, cleanup_at_end=False)
            print("Fragment Scratch Directory:", frag_scratch)
        else:
            frag_scratch = WorkDir(scratch_dir / dname)
        ci = cornell_shci.SHCI()
        ci.runtimedir = frag_scratch
        ci.restart = True
        # var_only being True means no perturbation is added to the fragment
        # This is advised
        ci.config["var_only"] = not SHCI_args.hci_pt
        ci.config["eps_vars"] = [SHCI_args.hci_cutoff]
        # Returning the 1RDM and 2RDM as csv can be helpful,
        # but is false by default to save disc space
        ci.config["get_1rdm_csv"] = SHCI_args.return_frag_data
        ci.config["get_2rdm_csv"] = SHCI_args.return_frag_data
        ci.kernel(h1, eri, nmo, nelec)
        # We always return 1 and 2rdms, for now
        rdm1_tmp, rdm2s = ci.make_rdm12(0, nmo, nelec)

    else:
        raise ValueError("Solver not implemented")

    # Compute RDM1
    rdm1 = multi_dot((mf_.mo_coeff, rdm1_tmp, mf_.mo_coeff.T)) * 0.5

    if eeval:
        if solver == "FCI" or solver == "SCI":
            if solver == "FCI":
                rdm2s = mc_fci.make_rdm2(civec, mc_fci.norb, mc_fci.nelec)
            if use_cumulant:
                hf_dm = zeros_like(rdm1_tmp)
                hf_dm[diag_indices(nocc)] += 2.0
                del_rdm1 = rdm1_tmp.copy()
                del_rdm1[diag_indices(nocc)] -= 2.0
                nc = (
                    einsum("ij,kl->ijkl", hf_dm, hf_dm)
                    + einsum("ij,kl->ijkl", hf_dm, del_rdm1)
                    + einsum("ij,kl->ijkl", del_rdm1, hf_dm)
                )
                nc -= (
                    einsum("ij,kl->iklj", hf_dm, hf_dm)
                    + einsum("ij,kl->iklj", hf_dm, del_rdm1)
                    + einsum("ij,kl->iklj", del_rdm1, hf_dm)
                ) * 0.5
                rdm2s -= nc
        e_f = get_frag_energy(
            mf_.mo_coeff,
            nocc,
            n_frag,
            weight_and_relAO_per_center,
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
    fobj_a: Frags,
    fobj_b: Frags,
    solver,
    enuc,  # noqa: ARG001
    hf_veff,
    relax_density=False,
    frozen=False,
    use_cumulant=True,
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

    Returns
    -------
    float
        As implemented, only returns the UCCSD fragment energy
    """
    # Run SCF for alpha and beta spins
    fobj_a.scf(unrestricted=True, spin_ind=0)
    fobj_b.scf(unrestricted=True, spin_ind=1)

    # Construct UHF object
    full_uhf, eris = make_uhf_obj(fobj_a, fobj_b, frozen=frozen)

    if solver == "UCCSD":
        ucc, rdm1_tmp, rdm2s = solve_uccsd(
            full_uhf,
            eris,
            relax=relax_density,
            use_cumulant=use_cumulant,
            rdm_return=True,
            rdm2_return=True,
            frozen=frozen,
        )
    else:
        raise NotImplementedError("Only UCCSD Solver implemented")

    # Compute RDM1
    fobj_a.rdm1__ = rdm1_tmp[0].copy()
    assert fobj_a._mf is not None and fobj_b._mf is not None
    fobj_a._rdm1 = (
        multi_dot((fobj_a._mf.mo_coeff, rdm1_tmp[0], fobj_a._mf.mo_coeff.T)) * 0.5
    )

    fobj_b.rdm1__ = rdm1_tmp[1].copy()
    fobj_b._rdm1 = (
        multi_dot((fobj_b._mf.mo_coeff, rdm1_tmp[1], fobj_b._mf.mo_coeff.T)) * 0.5
    )

    # Calculate Energies
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
        (fobj_a.n_frag, fobj_b.n_frag),
        (
            fobj_a.weight_and_relAO_per_center,
            fobj_b.weight_and_relAO_per_center,
        ),
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

    with ProcessPool(nprocs) as pool_:
        results = []  # type: ignore[var-annotated]
        # Run solver in parallel for each fragment
        for fobj in Fobjs:
            assert (
                fobj.fock is not None and fobj.heff is not None and fobj.dm0 is not None
            )

            result = pool_.apipe(
                run_solver,
                fobj.fock + fobj.heff,
                fobj.dm0.copy(),
                scratch_dir,
                fobj.dname,
                fobj.nao,
                fobj.nsocc,
                fobj.n_frag,
                fobj.weight_and_relAO_per_center,
                fobj.TA,
                fobj.h1,
                solver,
                fobj.eri_file,
                fobj.veff if not use_cumulant else None,
                fobj.veff0,
                eeval,
                return_vec,
                use_cumulant,
                relax_density,
                solver_args,
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
    Fobjs : list of tuple of quemb.molbe.autofrag.FragPart
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

    with ProcessPool(nprocs) as pool_:
        results = []
        # Run solver in parallel for each fragment
        for fobj_a, fobj_b in Fobjs:
            result = pool_.apipe(
                run_solver_u,
                fobj_a,
                fobj_b,
                solver,
                enuc,
                hf_veff,
                relax_density,
                frozen,
                use_cumulant,
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
