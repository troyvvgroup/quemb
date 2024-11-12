# Author(s): Oinam Romesh Meitei

import sys

import numpy

from .be_parallel import be_func_parallel
from .solver import be_func


class BEOPT:
    """Perform BE optimization.

    Implements optimization algorithms for bootstrap optimizations, namely, chemical potential optimization
    and density matching. The main technique used in the optimization is a Quasi-Newton method. It interface to external
    (adapted version) module originally written by Hong-Zhou Ye.

    Parameters
    ----------
    pot : list
       List of initial BE potentials. The last element is for the global chemical potential.
    Fobjs : MolBE.fragpart
       Fragment object
    Nocc : int
       No. of occupied orbitals for the full system.
    enuc : float
       Nuclear component of the energy.
    solver : str
       High-level solver in bootstrap embedding. 'MP2', 'CCSD', 'FCI' are supported. Selected CI versions,
       'HCI', 'SHCI', & 'SCI' are also supported. Defaults to 'MP2'
    only_chem : bool
       Whether to perform chemical potential optimization only. Refer to bootstrap embedding literatures.
    nproc : int
       Total number of processors assigned for the optimization. Defaults to 1. When nproc > 1, Python multithreading
       is invoked.
    ompnum : int
       If nproc > 1, ompnum sets the number of cores for OpenMP parallelization. Defaults to 4
    max_space : int
       Maximum number of bootstrap optimizaiton steps, after which the optimization is called converged.
    conv_tol : float
       Convergence criteria for optimization. Defaults to 1e-6
    ebe_hf : float
       Hartree-Fock energy. Defaults to 0.0
    """

    def __init__(
        self,
        pot,
        Fobjs,
        Nocc,
        enuc,
        solver="MP2",
        ecore=0.0,
        nproc=1,
        ompnum=4,
        only_chem=False,
        hf_veff=None,
        hci_pt=False,
        hci_cutoff=0.001,
        ci_coeff_cutoff=None,
        select_cutoff=None,
        max_space=500,
        conv_tol=1.0e-6,
        relax_density=False,
        ebe_hf=0.0,
        scratch_dir=None,
        **solver_kwargs,
    ):
        # Initialize class attributes
        self.ebe_hf = ebe_hf
        self.hf_veff = hf_veff
        self.pot = pot
        self.Fobjs = Fobjs
        self.Nocc = Nocc
        self.enuc = enuc
        self.solver = solver
        self.ecore = ecore
        self.iter = 0
        self.err = 0.0
        self.Ebe = 0.0
        self.max_space = max_space
        self.nproc = nproc
        self.ompnum = ompnum
        self.only_chem = only_chem
        self.conv_tol = conv_tol
        self.relax_density = relax_density
        # HCI parameters
        self.hci_cutoff = hci_cutoff
        self.ci_coeff_cutoff = ci_coeff_cutoff
        self.select_cutoff = select_cutoff
        self.hci_pt = hci_pt
        self.solver_kwargs = solver_kwargs
        self.scratch_dir = scratch_dir

    def objfunc(self, xk):
        """
        Computes error vectors, RMS error, and BE energies.

        If nproc (set in initialization) > 1, a multithreaded function is called to perform high-level computations.

        Parameters
        ----------
        xk : list
            Current potentials in the BE optimization.

        Returns
        -------
        list
            Error vectors.
        """

        # Choose the appropriate function based on the number of processors
        if self.nproc == 1:
            err_, errvec_, ebe_ = be_func(
                xk,
                self.Fobjs,
                self.Nocc,
                self.solver,
                self.enuc,
                eeval=True,
                return_vec=True,
                hf_veff=self.hf_veff,
                only_chem=self.only_chem,
                hci_cutoff=self.hci_cutoff,
                nproc=self.ompnum,
                relax_density=self.relax_density,
                ci_coeff_cutoff=self.ci_coeff_cutoff,
                select_cutoff=self.select_cutoff,
                hci_pt=self.hci_pt,
                ecore=self.ecore,
                ebe_hf=self.ebe_hf,
                be_iter=self.iter,
                scratch_dir=self.scratch_dir,
                **self.solver_kwargs,
            )
        else:
            err_, errvec_, ebe_ = be_func_parallel(
                xk,
                self.Fobjs,
                self.Nocc,
                self.solver,
                self.enuc,
                eeval=True,
                return_vec=True,
                hf_veff=self.hf_veff,
                nproc=self.nproc,
                ompnum=self.ompnum,
                only_chem=self.only_chem,
                hci_cutoff=self.hci_cutoff,
                relax_density=self.relax_density,
                ci_coeff_cutoff=self.ci_coeff_cutoff,
                select_cutoff=self.select_cutoff,
                ecore=self.ecore,
                ebe_hf=self.ebe_hf,
                be_iter=self.iter,
                scratch_dir=self.scratch_dir,
                **self.solver_kwargs,
            )

        # Update error and BE energy
        self.err = err_
        self.Ebe = ebe_
        return errvec_

    def optimize(self, method, J0=None, trust_region=False):
        """Main kernel to perform BE optimization

        Parameters
        ----------
        method : str
           High-level quantum chemistry method.
        J0 : list of list of float, optional
           Initial Jacobian
        trust_region : bool, optional
           Use trust-region based QN optimization, by default False
        """
        import sys

        from molbe.external.optqn import FrankQN

        print("-----------------------------------------------------", flush=True)
        print("             Starting BE optimization ", flush=True)
        print("             Solver : ", self.solver, flush=True)
        if self.only_chem:
            print("             Chemical Potential Optimization", flush=True)
        print("-----------------------------------------------------", flush=True)
        print(flush=True)

        if method == "QN":
            print("-- In iter ", self.iter, flush=True)

            # Initial step
            f0 = self.objfunc(self.pot)

            print(
                "Error in density matching      :   {:>2.4e}".format(self.err),
                flush=True,
            )
            print(flush=True)

            # Initialize the Quasi-Newton optimizer
            optQN = FrankQN(
                self.objfunc, numpy.array(self.pot), f0, J0, max_space=self.max_space
            )

            if self.err < self.conv_tol:
                print(flush=True)
                print("CONVERGED w/o Optimization Steps", flush=True)
                print(flush=True)
            else:
                # Perform optimization steps
                for iter_ in range(self.max_space):
                    print("-- In iter ", self.iter, flush=True)
                    optQN.next_step(trust_region=trust_region)
                    self.iter += 1
                    print(
                        "Error in density matching      :   {:>2.4e}".format(self.err),
                        flush=True,
                    )
                    print(flush=True)
                    if self.err < self.conv_tol:
                        print(flush=True)
                        print("CONVERGED", flush=True)
                        print(flush=True)
                        break
        else:
            print("This optimization method for BE is not supported")
            sys.exit()


def optimize(
    self,
    solver="MP2",
    method="QN",
    only_chem=False,
    conv_tol=1.0e-6,
    relax_density=False,
    use_cumulant=True,
    J0=None,
    nproc=1,
    ompnum=4,
    max_iter=500,
    scratch_dir=None,
    trust_region=False,
    **solver_kwargs,
):
    """BE optimization function

    Interfaces BEOPT to perform bootstrap embedding optimization.

    Parameters
    ----------
    solver : str, optional
        High-level solver for the fragment, by default 'MP2'
    method : str, optional
        Optimization method, by default 'QN'
    only_chem : bool, optional
        If true, density matching is not performed -- only global chemical potential is optimized, by default False
    conv_tol : _type_, optional
        Convergence tolerance, by default 1.e-6
    relax_density : bool, optional
        Whether to use relaxed or unrelaxed densities, by default False
        This option is for using CCSD as solver. Relaxed density here uses Lambda amplitudes, whereas unrelaxed density only uses T amplitudes.
        c.f. See http://classic.chem.msu.su/cgi-bin/ceilidh.exe/gran/gamess/forum/?C34df668afbHW-7216-1405+00.htm for the distinction between the two
    use_cumulant : bool, optional
        Use cumulant-based energy expression, by default True
    max_iter : int, optional
        Maximum number of optimization steps, by default 500
    nproc : int
       Total number of processors assigned for the optimization. Defaults to 1. When nproc > 1, Python multithreading
       is invoked.
    ompnum : int
       If nproc > 1, ompnum sets the number of cores for OpenMP parallelization. Defaults to 4
    J0 : list of list of float
       Initial Jacobian.
    trust_region : bool, optional
       Use trust-region based QN optimization, by default False
    """
    from .misc import print_energy

    # Check if only chemical potential optimization is required
    if not only_chem:
        pot = self.pot
        if self.be_type == "be1":
            sys.exit(
                "BE1 only works with chemical potential optimization. Set only_chem=True"
            )
    else:
        pot = [0.0]

    # Initialize the BEOPT object
    be_ = BEOPT(
        pot,
        self.Fobjs,
        self.Nocc,
        self.enuc,
        hf_veff=self.hf_veff,
        nproc=nproc,
        ompnum=ompnum,
        scratch_dir=scratch_dir,
        max_space=max_iter,
        conv_tol=conv_tol,
        only_chem=only_chem,
        hci_cutoff=self.hci_cutoff,
        ci_coeff_cutoff=self.ci_coeff_cutoff,
        relax_density=relax_density,
        select_cutoff=self.select_cutoff,
        hci_pt=self.hci_pt,
        solver=solver,
        ecore=self.E_core,
        ebe_hf=self.ebe_hf,
        **solver_kwargs,
    )

    if method == "QN":
        # Prepare the initial Jacobian matrix
        if only_chem:
            J0 = [[0.0]]
            J0 = self.get_be_error_jacobian(jac_solver="HF")
            J0 = [[J0[-1, -1]]]
        else:
            J0 = self.get_be_error_jacobian(jac_solver="HF")

        # Perform the optimization
        be_.optimize(method, J0=J0, trust_region=trust_region)
        self.ebe_tot = self.ebe_hf + be_.Ebe[0]
        # Print the energy components
        print_energy(
            be_.Ebe[0], be_.Ebe[1][1], be_.Ebe[1][0] + be_.Ebe[1][2], self.ebe_hf
        )
    else:
        print("This optimization method for BE is not supported")
        sys.exit()
