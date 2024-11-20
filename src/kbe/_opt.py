# Author(s): Oinam Romesh Meitei

from molbe._opt import BEOPT


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
        max_space=max_iter,
        conv_tol=conv_tol,
        only_chem=only_chem,
        hci_cutoff=self.hci_cutoff,
        ci_coeff_cutoff=self.ci_coeff_cutoff,
        relax_density=relax_density,
        select_cutoff=self.select_cutoff,
        solver=solver,
        ecore=self.E_core,
        ebe_hf=self.ebe_hf,
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
        be_.optimize(method, J0=J0)
        self.ebe_tot = self.ebe_hf + be_.Ebe[0]
        # Print the energy components
        print_energy(
            be_.Ebe[0],
            be_.Ebe[1][1],
            be_.Ebe[1][0] + be_.Ebe[1][2],
            self.ebe_hf,
            self.unitcell_nkpt,
        )
    else:
        print("This optimization method for BE is not supported")
        sys.exit()
