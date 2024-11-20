# Author(s): Oinam Romesh Meitei, Leah Weisburn

from .solver import solve_error
from .solver import solve_mp2, solve_ccsd, make_rdm1_ccsd_t1, solve_uccsd
from .solver import make_rdm2_urlx
from .helper import get_frag_energy
from molbe.external.unrestricted_utils import make_uhf_obj
from molbe.external.ccsd_rdm import make_rdm1_uccsd, make_rdm2_uccsd
import functools, numpy, sys
from .helper import *


def run_solver(
    h1,
    dm0,
    dname,
    nao,
    nocc,
    nfsites,
    efac,
    TA,
    hf_veff,
    h1_e,
    solver="MP2",
    eri_file="eri_file.h5",
    veff0=None,
    hci_cutoff=0.001,
    ci_coeff_cutoff=None,
    select_cutoff=None,
    ompnum=4,
    writeh1=False,
    eeval=True,
    return_rdm_ao=True,
    use_cumulant=True,
    relax_density=False,
    frag_energy=False,
):
    """
    Run a quantum chemistry solver to compute the reduced density matrices.

    Parameters
    ----------
    h1 : numpy.ndarray
        One-electron Hamiltonian matrix.
    dm0 : numpy.ndarray
        Initial guess for the density matrix.
    dname : str
        Directory name for storing intermediate files.
    nao : int
        Number of atomic orbitals.
    nocc : int
        Number of occupied orbitals.
    nfsites : int
        Number of fragment sites.
    efac : float
        Scaling factor for the electronic energy.
    TA : numpy.ndarray
        Transformation matrix for embedding orbitals.
    hf_veff : numpy.ndarray
        Hartree-Fock effective potential matrix.
    h1_e : numpy.ndarray
        One-electron integral matrix.
    solver : str, optional
        Solver to use for the calculation ('MP2', 'CCSD', 'FCI', 'HCI', 'SHCI', 'SCI'). Default is 'MP2'.
    eri_file : str, optional
        Filename for the electron repulsion integrals. Default is 'eri_file.h5'.
    ompnum : int, optional
        Number of OpenMP threads. Default is 4.
    writeh1 : bool, optional
        If True, write the one-electron integrals to a file. Default is False.
    eeval : bool, optional
        If True, evaluate the electronic energy. Default is True.
    return_rdm_ao : bool, optional
        If True, return the reduced density matrices in the atomic orbital basis. Default is True.
    use_cumulant : bool, optional
        If True, use the cumulant approximation for RDM2. Default is True.
    frag_energy : bool, optional
        If True, compute the fragment energy. Default is False.
    relax_density : bool, optional
        If True, use CCSD relaxed density. Default is False

    Returns
    -------
    tuple
        Depending on the input parameters, returns the molecular orbital coefficients,
        one-particle and two-particle reduced density matrices, and optionally the fragment energy.
    """

    # Get electron repulsion integrals (ERI)
    eri = get_eri(dname, nao, eri_file=eri_file)
    # Initialize SCF object
    mf_ = get_scfObj(h1, eri, nocc, dm0=dm0)
    rdm_return = False

    if relax_density:
        rdm_return = True

    # Select solver
    if solver == "MP2":
        mc_ = solve_mp2(mf_, mo_energy=mf_.mo_energy)
        rdm1_tmp = mc_.make_rdm1()

    elif solver == "CCSD":
        if not rdm_return:
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
        from pyscf import fci

        mc_ = fci.FCI(mf_, mf_.mo_coeff)
        efci, civec = mc_.kernel()
        rdm1_tmp = mc_.make_rdm1(civec, mc_.norb, mc_.nelec)
    elif solver == "HCI":
        from pyscf import hci, ao2mo

        nao, nmo = mf_.mo_coeff.shape
        eri = ao2mo.kernel(mf_._eri, mf_.mo_coeff, aosym="s4", compact=False).reshape(
            4 * ((nmo),)
        )
        ci_ = hci.SCI(mf_.mol)
        if select_cutoff is None and ci_coeff_cutoff is None:
            select_cutoff = hci_cutoff
            ci_coeff_cutoff = hci_cutoff
        elif select_cutoff is None or ci_coeff_cutoff is None:
            sys.exit()

        ci_.select_cutoff = select_cutoff
        ci_.ci_coeff_cutoff = ci_coeff_cutoff

        nelec = (nocc, nocc)
        h1_ = functools.reduce(numpy.dot, (mf_.mo_coeff.T, h1, mf_.mo_coeff))
        eci, civec = ci_.kernel(h1_, eri, nmo, nelec)
        civec = numpy.asarray(civec)

        (rdm1a_, rdm1b_), (rdm2aa, rdm2ab, rdm2bb) = ci_.make_rdm12s(civec, nmo, nelec)
        rdm1_tmp = rdm1a_ + rdm1b_
        rdm2s = rdm2aa + rdm2ab + rdm2ab.transpose(2, 3, 0, 1) + rdm2bb

    elif solver == "SHCI":
        from pyscf.shciscf import shci

        nao, nmo = mf_.mo_coeff.shape
        nelec = (nocc, nocc)
        mch = shci.SHCISCF(mf_, nmo, nelec, orbpath=dname)
        mch.fcisolver.mpiprefix = "mpirun -np " + str(ompnum)
        mch.fcisolver.stochastic = True  # this is for PT and doesnt add PT to rdm
        mch.fcisolver.nPTiter = 0
        mch.fcisolver.sweep_iter = [0]
        mch.fcisolver.DoRDM = True
        mch.fcisolver.sweep_epsilon = [hci_cutoff]
        mch.fcisolver.scratchDirectory = "/scratch/oimeitei/" + dname
        if not writeh1:
            mch.fcisolver.restart = True
        mch.mc1step()
        rdm1_tmp, rdm2s = mch.fcisolver.make_rdm12(0, nmo, nelec)

    elif solver == "SCI":
        from pyscf import cornell_shci
        from pyscf import ao2mo, mcscf

        nao, nmo = mf_.mo_coeff.shape
        nelec = (nocc, nocc)
        cas = mcscf.CASCI(mf_, nmo, nelec)
        h1, ecore = cas.get_h1eff(mo_coeff=mf_.mo_coeff)
        eri = ao2mo.kernel(mf_._eri, mf_.mo_coeff, aosym="s4", compact=False).reshape(
            4 * ((nmo),)
        )

        ci = cornell_shci.SHCI()
        ci.runtimedir = dname
        ci.restart = True
        ci.config["var_only"] = True
        ci.config["eps_vars"] = [hci_cutoff]
        ci.config["get_1rdm_csv"] = True
        ci.config["get_2rdm_csv"] = True
        ci.kernel(h1, eri, nmo, nelec)
        rdm1_tmp, rdm2s = ci.make_rdm12(0, nmo, nelec)

    else:
        print("Solver not implemented", flush=True)
        print("exiting", flush=True)
        sys.exit()

    # Compute RDM1
    rdm1 = functools.reduce(numpy.dot, (mf_.mo_coeff, rdm1_tmp, mf_.mo_coeff.T)) * 0.5
    if eeval:
        if solver == "CCSD" and not rdm_return:
            with_dm1 = True
            if use_cumulant:
                with_dm1 = False
            rdm2s = make_rdm2_urlx(t1, t2, with_dm1=with_dm1)

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
            hf_veff,
            rdm1_tmp,
            rdm2s,
            dname,
            eri_file,
            veff0,
        )
        if frag_energy:
            return e_f

    if return_rdm_ao:
        return (e_f, mf_.mo_coeff, rdm1, rdm2s, rdm1_tmp)

    return (e_f, mf_.mo_coeff, rdm1, rdm2s)


def run_solver_u(
    fobj_a,
    fobj_b,
    solver,
    enuc,
    hf_veff,
    frag_energy=True,
    relax_density=False,
    frozen=False,
    eri_file="eri_file.h5",
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
    frag_energy : bool, optional
        If True, calculates and returns fragment energies, defaults to True.
    relax_density : bool, optional
        If True, uses  relaxed density matrix for UCCSD, defaults to False.
    frozen : bool, optional
        If True, uses frozen core, defaults to False
    eri_file : str, optional
       Filename for the electron repulsion integrals. Default is 'eri_file.h5'.
    use_cumulant : bool, optional
        If True, uses the cumulant approximation for RDM2. Default is True.
    ereturn : bool, optional
        If True, return the computed energy. Defaults to False.

    Returns
    -------

        As implemented, only returns the UCCSD fragment energy
    """
    print("obj type", type(fobj_a))
    # Run SCF for alpha and beta spins
    fobj_a.scf(unrestricted=True, spin_ind=0)
    fobj_b.scf(unrestricted=True, spin_ind=1)

    # Construct UHF object
    full_uhf, eris = make_uhf_obj(fobj_a, fobj_b, frozen=frozen)

    rdm_return = False
    if relax_density:
        rdm_return = True

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
        raise NotImplementedError("Only UCCSD Solver implemented")

    # Compute RDM1
    fobj_a.__rdm1 = rdm1_tmp[0].copy()
    fobj_a._rdm1 = (
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

    # Calculate Energies
    if ereturn:
        if solver == "UCCSD" and not rdm_return:
            with_dm1 = True
            if use_cumulant:
                with_dm1 = False
            rdm2s = make_rdm2_uccsd(ucc, with_dm1=with_dm1)
        else:
            raise NotImplementedError("RDM Return not Implemented")

        fobj_a.__rdm2 = rdm2s[0].copy()
        fobj_b.__rdm2 = rdm2s[1].copy()

        # Calculate energy on a per-fragment basis
        if frag_energy:
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
        else:
            return NotImplementedError("Energy only calculated on a per-fragment basis")


def be_func_parallel(
    pot,
    Fobjs,
    Nocc,
    solver,
    enuc,
    hf_veff=None,
    nproc=1,
    ompnum=4,
    only_chem=False,
    relax_density=False,
    use_cumulant=True,
    eeval=False,
    ereturn=False,
    frag_energy=False,
    hci_cutoff=0.001,
    ci_coeff_cutoff=None,
    select_cutoff=None,
    return_vec=False,
    ecore=0.0,
    ebe_hf=0.0,
    be_iter=None,
    writeh1=False,
):
    """
    Embarrassingly Parallel High-Level Computation

    Performs high-level bootstrap embedding (BE) computation for each fragment. Computes 1-RDMs
    and 2-RDMs for each fragment. It also computes error vectors in BE density match. For selected
    CI solvers, this function exposes thresholds used in selected CI calculations (hci_cutoff, ci_coeff_cutoff, select_cutoff).

    Parameters
    ----------
    pot : list of float
        Potentials (local & global) that are added to the 1-electron Hamiltonian component.
        The last element in the list is the chemical potential.
    Fobjs : list of MolBE.fragpart
        Fragment definitions.
    Nocc : int
        Number of occupied orbitals for the full system.
    solver : str
        High-level solver in bootstrap embedding. Supported values are 'MP2', 'CCSD', 'FCI', 'HCI', 'SHCI', and 'SCI'.
    enuc : float
        Nuclear component of the energy.
    hf_veff : numpy.ndarray, optional
        Hartree-Fock effective potential.
    nproc : int, optional
        Total number of processors assigned for the optimization. Defaults to 1. When nproc > 1, Python multithreading is invoked.
    ompnum : int, optional
        If nproc > 1, sets the number of cores for OpenMP parallelization. Defaults to 4.
    only_chem : bool, optional
        Whether to perform chemical potential optimization only. Refer to bootstrap embedding literature. Defaults to False.
    eeval : bool, optional
        Whether to evaluate energies. Defaults to False.
    ereturn : bool, optional
        Whether to return the computed energy. Defaults to False.
    frag_energy : bool, optional
        Whether to compute fragment energy. Defaults to False.
    return_vec : bool, optional
        Whether to return the error vector. Defaults to False.
    ecore : float, optional
        Core energy. Defaults to 0.
    ebe_hf : float, optional
        Hartree-Fock energy. Defaults to 0.
    be_iter : int or None, optional
        Iteration number for bootstrap embedding. Defaults to None.
    writeh1 : bool, optional
        Whether to write the one-electron integrals. Defaults to False.

    Returns
    -------
    float or tuple
        Depending on the parameters, returns the error norm or a tuple containing the error norm,
        error vector, and the computed energy.
    """
    from multiprocessing import Pool
    import os

    nfrag = len(Fobjs)
    # Create directories for fragments if required
    if writeh1 and solver == "SCI":
        for nf in range(nfrag):
            dname = Fobjs[nf].dname
            os.system("mkdir " + dname)

    # Set the number of OpenMP threads
    os.system("export OMP_NUM_THREADS=" + str(ompnum))
    nprocs = int(nproc / ompnum)

    # Update the effective Hamiltonian with potentials
    if not pot is None:
        for fobj in Fobjs:
            fobj.update_heff(pot, only_chem=only_chem)

    pool_ = Pool(nprocs)
    results = []
    rdms = []

    # Run solver in parallel for each fragment
    for nf in range(nfrag):
        h1 = Fobjs[nf].fock + Fobjs[nf].heff
        dm0 = Fobjs[nf].dm0.copy()
        dname = Fobjs[nf].dname
        nao = Fobjs[nf].nao
        nocc = Fobjs[nf].nsocc
        nfsites = Fobjs[nf].nfsites
        efac = Fobjs[nf].efac
        TA = Fobjs[nf].TA
        h1_e = Fobjs[nf].h1
        veff0 = Fobjs[nf].veff0

        result = pool_.apply_async(
            run_solver,
            [
                h1,
                dm0,
                dname,
                nao,
                nocc,
                nfsites,
                efac,
                TA,
                hf_veff,
                h1_e,
                solver,
                Fobjs[nf].eri_file,
                veff0,
                hci_cutoff,
                ci_coeff_cutoff,
                select_cutoff,
                ompnum,
                writeh1,
                True,
                True,
                use_cumulant,
                relax_density,
                frag_energy,
            ],
        )

        results.append(result)

    # Collect results
    [rdms.append(result.get()) for result in results]
    pool_.close()

    if frag_energy:
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
    for idx, fobj in enumerate(Fobjs):
        e_1 += rdms[idx][0][0]
        e_2 += rdms[idx][0][1]
        e_c += rdms[idx][0][2]
        fobj.mo_coeffs = rdms[idx][1]
        fobj._rdm1 = rdms[idx][2]
        fobj.__rdm2 = rdms[idx][3]
        fobj.__rdm1 = rdms[idx][4]

    del rdms
    ernorm, ervec = solve_error(Fobjs, Nocc, only_chem=only_chem)

    if return_vec:
        return (ernorm, ervec, [e_1 + e_2 + e_c, [e_1, e_2, e_c]])

    if eeval:
        print("Error in density matching      :   {:>2.4e}".format(ernorm), flush=True)

    return ernorm


def be_func_parallel_u(
    pot,
    Fobjs,
    solver,
    enuc,
    hf_veff=None,
    nproc=1,
    ompnum=4,
    relax_density=False,
    use_cumulant=True,
    eeval=False,
    ereturn=False,
    frag_energy=False,
    ecore=0.0,
    ebe_hf=0.0,
    frozen=False,
):
    """
    Embarrassingly Parallel High-Level Computation

    Performs high-level unrestricted bootstrap embedding (UBE) computation for each fragment. Computes 1-RDMs
    and 2-RDMs for each fragment to return the energy. As such, this currently is equipped for one-shot U-CCSD BE.

    Parameters
    ----------
    pot : list of float
        Potentials (local & global) that are added to the 1-electron Hamiltonian component.
        The last element in the list is the chemical potential.
        Should always be 0, as this is still a one-shot only implementation
    Fobjs : list of tuples of MolBE.fragpart
        Fragment definitions, alpha and beta components.
    solver : str
        High-level solver in bootstrap embedding. Supported value is 'UCCSD'.
    enuc : float
        Nuclear component of the energy.
    hf_veff : tuple of numpy.ndarray, optional
        Alpha and beta Hartree-Fock effective potential.
    nproc : int, optional
        Total number of processors assigned for the optimization. Defaults to 1. When nproc > 1, Python multithreading is invoked.
    ompnum : int, optional
        If nproc > 1, sets the number of cores for OpenMP parallelization. Defaults to 4.
    eeval : bool, optional
        Whether to evaluate energies. Defaults to False.
    ereturn : bool, optional
        Whether to return the computed energy. Defaults to False.
    frag_energy : bool, optional
        Whether to compute fragment energy. Defaults to False.
    ecore : float, optional
        Core energy. Defaults to 0.
    ebe_hf : float, optional
        Hartree-Fock energy. Defaults to 0.
    frozen : bool, optional
        Frozen core. Defaults to False

    Returns
    -------
    float
        Returns the computed energy
    """
    from multiprocessing import Pool
    import os

    # Set the number of OpenMP threads
    os.system("export OMP_NUM_THREADS=" + str(ompnum))
    nprocs = int(nproc / ompnum)

    pool_ = Pool(nprocs)
    results = []
    energy_list = []

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
                frag_energy,
                relax_density,
                frozen,
                use_cumulant,
                True,
            ],
        )
        results.append(result)

    # Collect results
    [energy_list.append(result.get()) for result in results]
    pool_.close()

    if frag_energy:
        # Compute and return fragment energy
        e_1 = 0.0
        e_2 = 0.0
        e_c = 0.0
        for i in range(len(energy_list)):
            e_1 += energy_list[i][0]
            e_2 += energy_list[i][1]
            e_c += energy_list[i][2]
        return (e_1 + e_2 + e_c, (e_1, e_2, e_c))
    else:
        return NotImplementedError(
            "Only fragment-wise energy return implemented, no RDM return"
        )
