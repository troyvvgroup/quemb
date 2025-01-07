# Author(s): Oinam Romesh Meitei


import numpy
from attrs import define
from numpy import array, float64

from quemb.molbe.be_parallel import be_func_parallel
from quemb.molbe.pfrag import Frags
from quemb.molbe.solver import be_func
from quemb.shared.external.optqn import FrankQN
from quemb.shared.manage_scratch import WorkDir
from quemb.shared.typing import KwargDict, Matrix, Vector


@define
class BEOPT:
    """Perform BE optimization.

    Implements optimization algorithms for bootstrap optimizations, namely,
    chemical potential optimization and density matching. The main technique used in
    the optimization is a Quasi-Newton method. It interface to external
    (adapted version) module originally written by Hong-Zhou Ye.

    Parameters
    ----------
    pot :
       List of initial BE potentials. The last element is for the global
       chemical potential.
    Fobjs :
       Fragment object
    Nocc :
       No. of occupied orbitals for the full system.
    enuc :
       Nuclear component of the energy.
    scratch_dir :
        Scratch directory
    solver :
       High-level solver in bootstrap embedding. 'MP2', 'CCSD', 'FCI' are supported.
       Selected CI versions,
       'HCI', 'SHCI', & 'SCI' are also supported. Defaults to 'MP2'
    nproc :
       Total number of processors assigned for the optimization. Defaults to 1.
       When nproc > 1, Python multithreading
       is invoked.
    ompnum :
       If nproc > 1, ompnum sets the number of cores for OpenMP parallelization.
       Defaults to 4
    only_chem :
       Whether to perform chemical potential optimization only.
       Refer to bootstrap embedding literatures.
    max_space :
       Maximum number of bootstrap optimizaiton steps, after which the optimization
       is called converged.
    conv_tol :
       Convergence criteria for optimization. Defaults to 1e-6
    ebe_hf :
       Hartree-Fock energy. Defaults to 0.0
    """

    pot: list[float]
    Fobjs: list[Frags]
    Nocc: int
    enuc: float
    scratch_dir: WorkDir
    hf_veff: Matrix[float64] | None = None
    nproc: int = 1
    ompnum: int = 4
    only_chem: bool = False
    solver: str = "MP2"
    use_cumulant: bool = True
    frag_energy: bool = True

    max_space: int = 500
    conv_tol: float = 1.0e-6
    relax_density: bool = False
    ebe_hf: float = 0.0

    iter: int = 0
    err: float = 0.0
    Ebe: Matrix[float64] = array([[0.0]])

    DMRG_solver_kwargs: KwargDict | None = None

    # HCI parameters
    hci_cutoff: float = 0.001
    ci_coeff_cutoff: float | None = None
    select_cutoff: float | None = None
    hci_pt: bool = False

    def objfunc(self, xk: list[float]) -> Vector[float64]:
        """
        Computes error vectors, RMS error, and BE energies.

        If nproc (set in initialization) > 1, a multithreaded function is called to
        perform high-level computations.

        Parameters
        ----------
        xk :
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
                hf_veff=self.hf_veff,
                only_chem=self.only_chem,
                nproc=self.ompnum,
                relax_density=self.relax_density,
                use_cumulant=self.use_cumulant,
                frag_energy=self.frag_energy,
                eeval=True,
                return_vec=True,
                hci_cutoff=self.hci_cutoff,
                ci_coeff_cutoff=self.ci_coeff_cutoff,
                select_cutoff=self.select_cutoff,
                hci_pt=self.hci_pt,
                scratch_dir=self.scratch_dir,
                DMRG_solver_kwargs=self.DMRG_solver_kwargs,
            )
        else:
            err_, errvec_, ebe_ = be_func_parallel(
                xk,
                self.Fobjs,
                self.Nocc,
                self.solver,
                self.enuc,
                hf_veff=self.hf_veff,
                only_chem=self.only_chem,
                nproc=self.nproc,
                ompnum=self.ompnum,
                relax_density=self.relax_density,
                use_cumulant=self.use_cumulant,
                frag_energy=self.frag_energy,
                eeval=True,
                return_vec=True,
                hci_cutoff=self.hci_cutoff,
                ci_coeff_cutoff=self.ci_coeff_cutoff,
                select_cutoff=self.select_cutoff,
                scratch_dir=self.scratch_dir,
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
