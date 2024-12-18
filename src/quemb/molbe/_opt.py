# Author(s): Oinam Romesh Meitei


import numpy
from numpy import array, float64

from quemb.molbe.be_parallel import be_func_parallel
from quemb.molbe.solver import be_func
from quemb.shared.external.optqn import FrankQN
from quemb.shared.typing import KwargDict, Matrix


class BEOPT:
    """Perform BE optimization.

    Implements optimization algorithms for bootstrap optimizations, namely,
    chemical potential optimization and density matching. The main technique used in
    the optimization is a Quasi-Newton method. It interface to external
    (adapted version) module originally written by Hong-Zhou Ye.

    Parameters
    ----------
    pot : list
       List of initial BE potentials. The last element is for the global
       chemical potential.
    Fobjs : quemb.molbe.fragment.fragpart
       Fragment object
    Nocc : int
       No. of occupied orbitals for the full system.
    enuc : float
       Nuclear component of the energy.
    solver : str
       High-level solver in bootstrap embedding. 'MP2', 'CCSD', 'FCI' are supported.
       Selected CI versions,
       'HCI', 'SHCI', & 'SCI' are also supported. Defaults to 'MP2'
    only_chem : bool
       Whether to perform chemical potential optimization only.
       Refer to bootstrap embedding literatures.
    nproc : int
       Total number of processors assigned for the optimization. Defaults to 1.
       When nproc > 1, Python multithreading
       is invoked.
    ompnum : int
       If nproc > 1, ompnum sets the number of cores for OpenMP parallelization.
       Defaults to 4
    max_space : int
       Maximum number of bootstrap optimizaiton steps, after which the optimization
       is called converged.
    conv_tol : float
       Convergence criteria for optimization. Defaults to 1e-6
    """

    def __init__(
        self,
        pot,
        Fobjs,
        Nocc,
        enuc,
        solver="MP2",
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
        scratch_dir=None,
        solver_kwargs: KwargDict | None = None,
    ):
        # Initialize instance attributes
        self.hf_veff = hf_veff
        self.pot = pot
        self.Fobjs = Fobjs
        self.Nocc = Nocc
        self.enuc = enuc
        self.solver = solver
        self.iter = 0
        self.err = 0.0
        self.Ebe: Matrix[float64] = array([[0.0]])
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

        If nproc (set in initialization) > 1, a multithreaded function is called to
        perform high-level computations.

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
                scratch_dir=self.scratch_dir,
                solver_kwargs=self.solver_kwargs,
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
                scratch_dir=self.scratch_dir,
                solver_kwargs=self.solver_kwargs,
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
            raise ValueError("This optimization method for BE is not supported")
