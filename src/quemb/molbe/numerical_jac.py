# Numerical Jacobian Routine
# To save compute, we only compute the fragments that contain the edges

from numpy import floating, zeros, zeros_like

from quemb.molbe.be_parallel import be_func_parallel, run_solver
from quemb.molbe.solver import Solvers, be_func, solve_error
from quemb.shared.typing import Matrix


def compute_numerical_jacobian(
    beobj, solver: Solvers, only_chem: bool, nproc: int, step_size: float = 1e-6
) -> Matrix[floating]:
    """Compute the numerical Jacobian for the BE optimization.

    This function computes the numerical Jacobian by perturbing the potentials
    of the fragments that contain the edges or edges connected to the perturbed
    fragment's origin. The Jacobian is computed using first-order central differences.

    Parameters
    ----------
    beobj
        The BE object.
    solver
        The solver to use for the computation.
    only_chem
        Whether to compute only the chemical potential part of the Jacobian.
    nproc
        Number of processors to use for parallel computation.

    Returns
    -------
    numpy.ndarray
        The computed numerical Jacobian matrix.
    """
    # Prepare space to store the Jacobian
    J0 = zeros((len(beobj.pot), len(beobj.pot)))

    # Chem pot
    # +x
    x = beobj.pot.copy()
    x[-1] += step_size
    J0[:, -1] = chem_pot_dispatch(x, beobj, solver, only_chem, nproc)
    # -x
    x[-1] -= 2 * step_size
    J0[:, -1] -= chem_pot_dispatch(x, beobj, solver, only_chem, nproc)
    J0[:, -1] /= 2 * step_size

    if only_chem:
        return J0

    # First perform oneshot calculation to get reference densities
    if nproc == 1:
        be_func(
            None,
            beobj.Fobjs,
            beobj.Nocc,
            solver,
            beobj.enuc,
            only_chem=only_chem,
            relax_density=False,
            scratch_dir=beobj.scratch_dir,
            solver_args=None,
            use_cumulant=True,
            eeval=False,
            return_vec=False,
        )
    else:
        be_func_parallel(
            None,
            beobj.Fobjs,
            beobj.Nocc,
            solver,
            beobj.enuc,
            only_chem=only_chem,
            nproc=beobj.nproc,
            ompnum=beobj.ompnum,
            relax_density=False,
            scratch_dir=beobj.scratch_dir,
            solver_args=None,
            use_cumulant=True,
            eeval=True,  # Fix after #264 resolves
            return_vec=True,
        )

    # Save zero-potential 1-RDMs
    rdm1_ref = [f._rdm1 for f in beobj.Fobjs]

    # Loop over each condition
    for idx in range(len(beobj.pot) - 1):
        # First find which fragment the condition belongs to
        frag_idx = sum([f.udim <= idx for f in beobj.Fobjs]) - 1
        # Plus x
        x = beobj.pot.copy()
        x[idx] += step_size

        # Calculate heff
        heff = calc_heff(beobj.Fobjs[frag_idx], x, only_chem)
        rdm1_plus = run_solver(
            beobj.Fobjs[frag_idx].fock + heff,
            beobj.Fobjs[frag_idx].dm0.copy(),
            beobj.scratch_dir,
            beobj.Fobjs[frag_idx].dname,
            beobj.Fobjs[frag_idx].nao,
            beobj.Fobjs[frag_idx].nsocc,
            beobj.Fobjs[frag_idx].n_frag,
            beobj.Fobjs[frag_idx].weight_and_relAO_per_center,
            beobj.Fobjs[frag_idx].TA,
            beobj.Fobjs[frag_idx].h1,
            solver,
            beobj.Fobjs[frag_idx].eri_file,
            None,
            beobj.Fobjs[frag_idx].veff0,
            False,
            False,
            True,
            False,
            None,  # TODO: Allow passing in SolverArgs
        )[2]

        # Compute P(+x) - P(0)
        rdm1_list = rdm1_ref.copy()
        rdm1_list[frag_idx] = rdm1_plus
        _, err_vec_plus = solve_error(
            beobj.Fobjs, beobj.Nocc, only_chem, rdm1_list=rdm1_list
        )

        # Minus x
        x[idx] -= 2 * step_size
        heff = calc_heff(beobj.Fobjs[frag_idx], x, only_chem)
        rdm1_minus = run_solver(
            beobj.Fobjs[frag_idx].fock + heff,
            beobj.Fobjs[frag_idx].dm0.copy(),
            beobj.scratch_dir,
            beobj.Fobjs[frag_idx].dname,
            beobj.Fobjs[frag_idx].nao,
            beobj.Fobjs[frag_idx].nsocc,
            beobj.Fobjs[frag_idx].n_frag,
            beobj.Fobjs[frag_idx].weight_and_relAO_per_center,
            beobj.Fobjs[frag_idx].TA,
            beobj.Fobjs[frag_idx].h1,
            solver,
            beobj.Fobjs[frag_idx].eri_file,
            None,
            beobj.Fobjs[frag_idx].veff0,
            False,
            False,
            True,
            False,
            None,  # TODO: Allow passing in SolverArgs
        )[2]

        # Compute P(-x) - P(0)
        rdm1_list[frag_idx] = rdm1_minus
        _, err_vec_minus = solve_error(
            beobj.Fobjs, beobj.Nocc, only_chem, rdm1_list=rdm1_list
        )

        # Compute central difference
        J0[:, idx] = (err_vec_plus - err_vec_minus) / (2 * step_size)
    return J0


def calc_heff(fobj, pot, only_chem):
    # This function is defined to evaluate heff without modifying the Frags object
    # If pfrag.py::pFrag.update_heff is fixed to be side effect free, this function
    # can be removed and the original update_heff can be used instead.
    heff = zeros_like(fobj.h1)
    cout = fobj.udim

    for i, fi in enumerate(fobj.AO_in_frag):
        if not any(i in sublist for sublist in fobj.relAO_per_edge):
            heff[i, i] -= pot[-1]

    if only_chem:
        return heff
    else:
        for i in fobj.relAO_per_edge:
            for j in range(len(i)):
                for k in range(len(i)):
                    if j > k:  # or j==k:
                        continue

                    heff[i[j], i[k]] = pot[cout]
                    heff[i[k], i[j]] = pot[cout]

                    cout += 1

        return heff


def chem_pot_dispatch(pot, beobj, solver, only_chem, nproc):
    if nproc == 1:
        return be_func(
            pot,
            beobj.Fobjs,
            beobj.Nocc,
            solver,
            beobj.enuc,
            only_chem=only_chem,
            relax_density=False,
            scratch_dir=beobj.scratch_dir,
            solver_args=None,
            use_cumulant=True,
            eeval=False,
            return_vec=True,
        )[1]
    else:
        return be_func_parallel(
            pot,
            beobj.Fobjs,
            beobj.Nocc,
            solver,
            beobj.enuc,
            only_chem=only_chem,
            nproc=beobj.nproc,
            ompnum=beobj.ompnum,
            relax_density=False,
            scratch_dir=beobj.scratch_dir,
            solver_args=None,
            use_cumulant=True,
            eeval=True,  # Fix after #264 resolves
            return_vec=True,
        )[1]
