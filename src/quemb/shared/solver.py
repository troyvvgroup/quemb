# Author(s): Oinam Romesh Meitei, Leah Weisburn, Shaun Weatherly

import os
from abc import ABC
from collections.abc import Sequence
from pathlib import Path
from typing import Final, Literal, TypeAlias, cast
from warnings import warn

import numpy as np
from attrs import Factory, define, field
from numpy import (
    allclose,
    array,
    asarray,
    diag,
    diag_indices,
    einsum,
    floating,
    mean,
    ndarray,
    zeros_like,
)
from numpy.linalg import multi_dot
from pyscf import ao2mo, cc, fci, mcscf, mp
from pyscf.cc.ccsd_rdm import make_rdm2
from pyscf.mp.mp2 import MP2
from pyscf.scf.hf import RHF

from quemb.kbe.pfrag import Frags as pFrags
from quemb.molbe.helper import get_frag_energy, get_frag_energy_u
from quemb.molbe.pfrag import Frags
from quemb.shared.external.ccsd_rdm import (
    make_rdm1_uccsd,
    make_rdm2_uccsd,
    make_rdm2_urlx,
)
from quemb.shared.external.uccsd_eri import make_eris_incore
from quemb.shared.external.unrestricted_utils import make_uhf_obj
from quemb.shared.helper import delete_multiple_files, unused
from quemb.shared.manage_scratch import WorkDir
from quemb.shared.typing import Matrix, Vector

Solvers: TypeAlias = Literal["MP2", "CCSD", "FCI", "HCI", "SHCI", "SCI", "DMRG"]
USolvers: TypeAlias = Literal["UCCSD"]


class UserSolverArgs(ABC):
    pass


@define(frozen=True)
class DMRG_ArgsUser(UserSolverArgs):
    """

    Parameters
    ----------
    max_mem:
        Maximum memory in GB.
    root:
        Number of roots to solve for.
    startM:
        Starting MPS bond dimension - where the sweep schedule begins.
    maxM:
        Maximum MPS bond dimension - where the sweep schedule terminates.
    max_iter:
        Maximum number of sweeps.
    twodot_to_onedot:
        Sweep index at which to transition to one-dot DMRG algorithm.
        All sweeps prior to this will use the two-dot algorithm.
    block_extra_keyword:
        Other keywords to be passed to block2.
        See: https://block2.readthedocs.io/en/latest/user/keywords.html
    schedule_kwargs:
        Dictionary containing DMRG scheduling parameters to be passed to block2.

        e.g. The default schedule used here would be equivalent to the following:

        .. code-block:: python

            schedule_kwargs = {
                'scheduleSweeps': [0, 10, 20, 30, 40, 50],
                'scheduleMaxMs': [25, 50, 100, 200, 500, 500],
                'scheduleTols': [1e-5,1e-5, 1e-6, 1e-6, 1e-8, 1e-8],
                'scheduleNoises': [0.01, 0.01, 0.001, 0.001, 1e-4, 0.0],
            }
    """

    #: Becomes mf.mo_coeff.shape[1] by default
    norb: Final[int | None] = None
    #: Becomes mf.mo_coeff.shape[1] by default
    nelec: Final[int | None] = None

    startM: Final[int] = 25
    maxM: Final[int] = 500
    max_iter: Final[int] = 60
    max_mem: Final[int] = 100
    max_noise: Final[float] = 1e-3
    min_tol: Final[float] = 1e-8
    twodot_to_onedot: Final[int] = (5 * max_iter) // 6
    root: Final[int] = 0
    block_extra_keyword: Final[list[str]] = Factory(lambda: ["fiedler"])
    schedule_kwargs: dict[str, list[int] | list[float]] = field()
    force_cleanup: Final[bool] = False

    @schedule_kwargs.default
    def _get_schedule_kwargs_default(self) -> dict[str, list[int] | list[float]]:
        return {
            "scheduleSweeps": [(i * self.max_iter) // 6 for i in range(1, 7)],
            "scheduleMaxMs": [
                self.startM if (self.startM < self.maxM) else self.maxM,
                self.startM * 2 if (self.startM * 2 < self.maxM) else self.maxM,
                self.startM * 4 if (self.startM * 4 < self.maxM) else self.maxM,
                self.startM * 8 if (self.startM * 8 < self.maxM) else self.maxM,
                self.maxM,
                self.maxM,
            ],
            "scheduleTols": [
                self.min_tol * 1e3,
                self.min_tol * 1e3,
                self.min_tol * 1e2,
                self.min_tol * 1e1,
                self.min_tol,
                self.min_tol,
            ],
            "scheduleNoises": [
                self.max_noise,
                self.max_noise,
                self.max_noise / 10,
                self.max_noise / 100,
                self.max_noise / 100,
                0.0,
            ],
        }


@define(frozen=True)
class _DMRG_Args:
    """Properly initialized DMRG arguments

    Some default values of :class:`DMRG_ArgsUser` can only be filled
    later in the calculation.
    Use :func:`from_user_input` to properly initialize.
    """

    norb: Final[int]
    nelec: Final[int]

    startM: Final[int]
    maxM: Final[int]
    max_iter: Final[int]
    max_mem: Final[int]
    max_noise: Final[float]
    min_tol: Final[float]
    twodot_to_onedot: Final[int]
    root: Final[int]
    block_extra_keyword: Final[list[str]]
    schedule_kwargs: Final[dict[str, list[int] | list[float]]]
    force_cleanup: Final[bool]

    @classmethod
    def from_user_input(cls, user_args: DMRG_ArgsUser, mf: RHF):
        if user_args.norb is None:
            assert mf.mo_coeff is not None
            norb = mf.mo_coeff.shape[1]
        else:
            norb = user_args.norb
        if user_args.nelec is None:
            assert mf.mo_coeff is not None
            nelec = mf.mo_coeff.shape[1]
        else:
            nelec = user_args.nelec
        if norb <= 2:
            block_extra_keyword = [
                "noreorder"
            ]  # Other reordering algorithms explode if the network is too small.
        else:
            block_extra_keyword = user_args.block_extra_keyword
        return cls(
            norb=norb,
            nelec=nelec,
            startM=user_args.startM,
            maxM=user_args.maxM,
            max_iter=user_args.max_iter,
            max_mem=user_args.max_mem,
            max_noise=user_args.max_noise,
            min_tol=user_args.min_tol,
            twodot_to_onedot=user_args.twodot_to_onedot,
            root=user_args.root,
            block_extra_keyword=block_extra_keyword,
            schedule_kwargs=user_args.schedule_kwargs,
            force_cleanup=user_args.force_cleanup,
        )


@define(frozen=True)
class SHCI_ArgsUser(UserSolverArgs):
    hci_cutoff: Final[float] = 0.001
    hci_pt: Final[bool] = False
    return_frag_data: Final[bool] = False
    # ci_coeff_cutoff: Final[float | None] = None  # TODO SOLVER
    # select_cutoff: Final[float | None] = None  # TODO SOLVER


@define(frozen=True)
class _SHCI_Args:
    """Properly initialized SCHI arguments

    Some default values of :class:`SHCI_ArgsUser` can only be filled
    later in the calculation.
    Use :func:`from_user_input` to properly initialize.
    """

    hci_cutoff: Final[float]
    hci_pt: Final[bool]
    return_frag_data: Final[bool]
    # ci_coeff_cutoff: Final[float]  # TODO SOLVER
    # select_cutoff: Final[float]  # TODO SOLVER

    @classmethod
    def from_user_input(cls, args: SHCI_ArgsUser):
        """
        if (args.select_cutoff is None) and (args.ci_coeff_cutoff is None):
            select_cutoff = args.hci_cutoff
            ci_coeff_cutoff = args.hci_cutoff
        elif (args.select_cutoff is not None) and (args.ci_coeff_cutoff is not None):
            ci_coeff_cutoff = args.ci_coeff_cutoff
            select_cutoff = args.select_cutoff
        else:
            raise ValueError(
                "Solver args `ci_coeff_cutoff` and `select_cutoff` must both "
                "be specified or both be `None`!"
            )
        """
        if args.hci_pt:
            warn("hci_pt is set True: note that the perturbed SCI solver is untested")
        return cls(
            hci_pt=args.hci_pt,
            hci_cutoff=args.hci_cutoff,
            return_frag_data=args.return_frag_data,
            # ci_coeff_cutoff=ci_coeff_cutoff,
            # select_cutoff=select_cutoff,
        )


def be_func(
    pot: list[float] | None,
    Fobjs: list[Frags] | list[pFrags],
    Nocc: int,
    solver: Solvers,
    enuc: float,  # noqa: ARG001
    solver_args: UserSolverArgs | None,
    scratch_dir: WorkDir,
    only_chem: bool = False,
    eeval: bool = False,
    relax_density: bool = False,
    return_vec: bool = False,
    use_cumulant: bool = True,
    Delta_n_el: float = 0.0,
):
    """
    Perform bootstrap embedding calculations for each fragment.

    This function computes the energy and/or error for each fragment in a
    molecular system using various quantum chemistry solvers.

    Parameters
    ----------
    pot :
        List of potentials.
    Fobjs : list of quemb.molbe.autofrag.FragPart
        List of fragment objects.
    Nocc :
        Number of occupied orbitals.
    solver :
        Quantum chemistry solver to use ('MP2', 'CCSD', 'FCI', 'SCI). TODO 'HCI', 'SHCI'
    enuc :
        Nuclear energy.
    only_chem :
        Whether to only optimize the chemical potential. Defaults to False.
    eeval :
        Whether to evaluate the energy. Defaults to False.
    relax_density :
        Whether to use the relaxed CCSD density matrix. Defaults to False.
    return_vec :
        Whether to return the error vector. Defaults to False.
    use_cumulant :
        Whether to use the cumulant-based energy expression. Defaults to True.
    eeval :
        Whether to evaluate the energy. Defaults to False.
    return_vec :
        Whether to return the error vector. Defaults to False.

    Returns
    -------
    float or tuple
        Depending on the options, it returns the norm of the error vector, the energy,
        or a combination of these values.
    """
    if eeval:
        total_e = [0.0, 0.0, 0.0]

    # Loop over each fragment and solve using the specified solver
    for fobj in Fobjs:
        # Update the effective Hamiltonian
        if pot is not None:
            fobj.update_heff(pot, only_chem=only_chem)

        assert fobj.fock is not None and fobj.heff is not None

        # Perform SCF calculation
        fobj.scf()

        # Solve using the specified solver
        assert fobj._mf is not None
        if solver == "MP2":
            fobj._mc = solve_mp2(fobj._mf, mo_energy=fobj._mf.mo_energy)
            rdm1_tmp = fobj._mc.make_rdm1()
            if eeval:
                rdm2s = fobj._mc.make_rdm2()
        elif solver == "CCSD":
            if eeval:
                fobj.t1, fobj.t2, rdm1_tmp, rdm2s = solve_ccsd(
                    fobj._mf,
                    mo_energy=fobj._mf.mo_energy,
                    relax=relax_density,
                    use_cumulant=use_cumulant,
                    rdm_return=True,
                    rdm2_return=True,
                )
            else:
                # currently passing mycc: likely unnecessary
                fobj.t1, fobj.t2, rdm1_tmp, _ = solve_ccsd(  # mycc
                    fobj._mf,
                    mo_energy=fobj._mf.mo_energy,
                    relax=relax_density,
                    use_cumulant=use_cumulant,
                    rdm_return=True,
                    rdm2_return=False,
                )

        elif solver == "FCI":
            mc = fci.FCI(fobj._mf, fobj._mf.mo_coeff)
            _, civec = mc.kernel()
            rdm1_tmp = mc.make_rdm1(civec, mc.norb, mc.nelec)

        elif solver == "HCI":  # TODO
            # pylint: disable-next=E0611
            raise NotImplementedError("HCI solver not implemented")
            """
            from pyscf import hci  # type: ignore[attr-defined]  # noqa: PLC0415

            assert isinstance(solver_args, SHCI_ArgsUser)
            SHCI_args = _SHCI_Args.from_user_input(solver_args)
            nmo = fobj._mf.mo_coeff.shape[1]

            eri = ao2mo.kernel(
                fobj._mf._eri, fobj._mf.mo_coeff, aosym="s4", compact=False
            ).reshape(4 * ((nmo),))

            ci_ = hci.SCI(fobj._mf.mol)

            ci_.select_cutoff = SHCI_args.select_cutoff
            ci_.ci_coeff_cutoff = SHCI_args.ci_coeff_cutoff

            nelec = (fobj.nsocc, fobj.nsocc)
            h1_ = fobj.fock + fobj.heff
            h1_ = multi_dot((fobj._mf.mo_coeff.T, h1_, fobj._mf.mo_coeff))
            eci, civec = ci_.kernel(h1_, eri, nmo, nelec)
            unused(eci)
            civec = asarray(civec)

            (rdm1a_, rdm1b_), (rdm2aa, rdm2ab, rdm2bb) = ci_.make_rdm12s(
                civec, nmo, nelec
            )
            rdm1_tmp = rdm1a_ + rdm1b_
            rdm2s = rdm2aa + rdm2ab + rdm2ab.transpose(2, 3, 0, 1) + rdm2bb
            """
        elif solver == "SHCI":  # TODO
            # pylint: disable-next=E0611,E0401
            raise NotImplementedError("SHCI solver not implemented")
            """
            from pyscf.shciscf import (  # type: ignore[attr-defined]  # noqa: PLC0415
                shci,
            )

            assert isinstance(solver_args, SHCI_ArgsUser)
            SHCI_args = _SHCI_Args.from_user_input(solver_args)

            assert isinstance(fobj.dname, str)
            frag_scratch = WorkDir(scratch_dir / fobj.dname)

            nmo = fobj._mf.mo_coeff.shape[1]

            nelec = (fobj.nsocc, fobj.nsocc)
            mch = shci.SHCISCF(fobj._mf, nmo, nelec, orbpath=fobj.dname)
            mch.fcisolver.mpiprefix = "mpirun -np " + str(nproc)
            # need to pass nproc through be_func
            if SHCI_args.hci_pt:
                mch.fcisolver.stochastic = False
                mch.fcisolver.epsilon2 = SHCI_args.hci_cutoff
            else:
                mch.fcisolver.stochastic = (
                    True  # this is for PT and doesnt add PT to rdm
                )
                mch.fcisolver.nPTiter = 0
            mch.fcisolver.sweep_iter = [0]
            mch.fcisolver.DoRDM = True
            mch.fcisolver.sweep_epsilon = [SHCI_args.hci_cutoff]
            mch.fcisolver.scratchDirectory = scratch_dir
            mch.mc1step()
            rdm1_tmp, rdm2s = mch.fcisolver.make_rdm12(0, nmo, nelec)
            """

        elif solver == "SCI":
            # pylint: disable-next=E0611
            from pyscf import cornell_shci  # noqa: PLC0415  # optional module

            assert isinstance(solver_args, SHCI_ArgsUser)
            SHCI_args = _SHCI_Args.from_user_input(solver_args)

            assert isinstance(fobj.dname, str)

            nmo = fobj._mf.mo_coeff.shape[1]
            nelec = (fobj.nsocc, fobj.nsocc)
            cas = mcscf.CASCI(fobj._mf, nmo, nelec)
            h1, ecore = cas.get_h1eff(mo_coeff=fobj._mf.mo_coeff)
            unused(ecore)
            eri = ao2mo.kernel(
                fobj._mf._eri, fobj._mf.mo_coeff, aosym="s4", compact=False
            ).reshape(4 * ((nmo),))

            if SHCI_args.return_frag_data:
                warn(
                    "If return_frag_data is True, RDMs and other data"
                    "are written into a directory which is not"
                    "cleaned: cleanup_at_end is False"
                )
                iter = 0
                frag_name = (
                    Path(f"{scratch_dir}-frag_data") / f"{fobj.dname}_iter{iter}"
                )
                while frag_name.exists():
                    iter += 1
                    frag_name = (
                        Path(f"{scratch_dir}-frag_data") / f"{fobj.dname}_iter{iter}"
                    )
                frag_scratch = WorkDir(frag_name, cleanup_at_end=False)
                print("Fragment Scratch Directory:", frag_scratch)
            else:
                frag_scratch = WorkDir(scratch_dir / fobj.dname)
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

        elif solver in ["block2", "DMRG", "DMRGCI", "DMRGSCF"]:
            assert isinstance(fobj.dname, str)
            frag_scratch = WorkDir(scratch_dir / fobj.dname)

            assert isinstance(solver_args, DMRG_ArgsUser)
            DMRG_args = _DMRG_Args.from_user_input(solver_args, fobj._mf)

            assert fobj.nsocc is not None
            try:
                rdm1_tmp, rdm2s = solve_block2(
                    fobj._mf,
                    fobj.nsocc,
                    frag_scratch=frag_scratch,
                    DMRG_args=DMRG_args,
                    use_cumulant=use_cumulant,
                )
            except Exception as inst:
                raise inst
            finally:
                if DMRG_args.force_cleanup:
                    delete_multiple_files(
                        frag_scratch.path.glob("F.*"),
                        frag_scratch.path.glob("FCIDUMP*"),
                        frag_scratch.path.glob("node*"),
                    )

        else:
            raise ValueError("Solver not implemented")

        fobj.rdm1__ = rdm1_tmp.copy()

        assert fobj.mo_coeffs is not None
        fobj._rdm1 = (
            multi_dot(
                (
                    fobj.mo_coeffs,
                    rdm1_tmp,
                    fobj.mo_coeffs.T,
                ),
            )
            * 0.5
        )

        if eeval:
            if solver == "FCI" or solver == "SCI":
                if solver == "FCI":
                    rdm2s = mc.make_rdm2(civec, mc.norb, mc.nelec)
                if use_cumulant:
                    assert fobj.nsocc is not None
                    hf_dm = zeros_like(rdm1_tmp)
                    hf_dm[diag_indices(fobj.nsocc)] += 2.0
                    del_rdm1 = rdm1_tmp.copy()
                    del_rdm1[diag_indices(fobj.nsocc)] -= 2.0
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
            fobj.rdm2__ = rdm2s.copy()
            # Find the energy of a given fragment.
            # Return [e1, e2, ec] as e_f and add to the running total_e.
            e_f = get_frag_energy(
                mo_coeffs=fobj.mo_coeffs,
                nsocc=fobj.nsocc,
                n_frag=fobj.n_frag,
                weight_and_relAO_per_center=fobj.weight_and_relAO_per_center,
                TA=fobj.TA,
                h1=fobj.h1,
                rdm1=rdm1_tmp,
                rdm2s=rdm2s,
                dname=fobj.dname,
                veff0=fobj.veff0,
                veff=None if use_cumulant else fobj.veff,
                use_cumulant=use_cumulant,
                eri_file=fobj.eri_file,
            )
            total_e = [sum(x) for x in zip(total_e, e_f)]
            fobj.update_ebe_hf()
    if eeval:
        Ecorr = sum(total_e)
        if not return_vec:
            return (Ecorr, total_e)

    ernorm, ervec = solve_error(Fobjs, Nocc, only_chem=only_chem, Delta_n_el=Delta_n_el)

    if return_vec:
        return (ernorm, ervec, [Ecorr, total_e])

    return ernorm


def be_func_u(
    pot,  # noqa: ARG001
    Fobjs: list[tuple[Frags, Frags]],
    solver: USolvers,
    enuc,  # noqa: ARG001
    hf_veff=None,
    eeval=False,
    relax_density=False,
    use_cumulant=True,
    frozen=False,
):
    """Perform bootstrap embedding calculations for each fragment with UCCSD.

    This function computes the energy and/or error for each fragment in a
    molecular system using various quantum chemistry solvers.

    Parameters
    ----------
    pot : list
        List of potentials.
    Fobjs : list
        zip list of :class:`quemb.molbe.autofrag.FragPart`, alpha and beta
        List of fragment objects. Each element is a tuple with the alpha and
        beta components
    solver :
        Quantum chemistry solver to use ('UCCSD').
    enuc : float
        Nuclear energy.
    hf_veff : tuple of numpy.ndarray, optional
        Hartree-Fock effective potential. Defaults to None.
    eeval : bool, optional
        Whether to evaluate the energy. Defaults to False.
    relax_density : bool, optional
        Whether to relax the density. Defaults to False.
    return_vec : bool, optional
        Whether to return the error vector. Defaults to False.
    ebe_hf : float, optional
        Hartree-Fock energy. Defaults to 0.
    use_cumulant : bool, optional
        Whether to use the cumulant-based energy expression. Defaults to True.
    frozen : bool, optional
        Frozen core. Defaults to False
    Returns
    -------
    float or tuple
        Depending on the options, it returns the norm of the error vector, the energy,
        or a combination of these values.
    """
    E = 0.0
    if eeval:
        total_e = [0.0, 0.0, 0.0]

    # Loop over each fragment and solve using the specified solver
    for fobj_a, fobj_b in Fobjs:
        fobj_a.scf(unrestricted=True, spin_ind=0)
        fobj_b.scf(unrestricted=True, spin_ind=1)

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
            raise ValueError("Solver not implemented")

        assert fobj_a._mf is not None and fobj_b._mf is not None
        fobj_a.rdm1__ = rdm1_tmp[0].copy()
        fobj_b._rdm1 = (
            multi_dot((fobj_a._mf.mo_coeff, rdm1_tmp[0], fobj_a._mf.mo_coeff.T)) * 0.5
        )

        fobj_b.rdm1__ = rdm1_tmp[1].copy()
        fobj_b._rdm1 = (
            multi_dot((fobj_b._mf.mo_coeff, rdm1_tmp[1], fobj_b._mf.mo_coeff.T)) * 0.5
        )

        if eeval:
            fobj_a.rdm2__ = rdm2s[0].copy()
            fobj_b.rdm2__ = rdm2s[1].copy()

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
            total_e = [sum(x) for x in zip(total_e, e_f)]

    E = sum(total_e)
    return (E, total_e)


def solve_error(
    Fobjs: Sequence[Frags] | Sequence[pFrags],
    Nocc: float,
    only_chem: bool = False,
    Delta_n_el: float = 0.0,
) -> tuple[float, Vector[np.float64]]:
    """
    Compute the error for self-consistent fragment density matrix matching.

    This function calculates the error in the one-particle density matrix
    for a given fragment, matching the density matrix elements of the edges and centers.
    It returns the norm of the error vector and the error vector itself.

    Parameters
    ----------
    Fobjs :
        List of fragment objects.
    Nocc :
        Number of occupied orbitals.
    Delta_n_el :
        Additional deviation of the particle number.

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
            assert fobj._rdm1 is not None
            for i in fobj.weight_and_relAO_per_center[1]:
                err_chempot += fobj._rdm1[i, i]

        print(">>>>>", Fobjs[0].unitcell_nkpt)
        err_chempot /= Fobjs[0].unitcell_nkpt
        print(Nocc)
        err = err_chempot - (Nocc + Delta_n_el / 2)

        return abs(err), cast(Vector[np.float64], asarray([err]))

    # Compute edge and chemical potential errors
    for fobj in Fobjs:
        # match rdm-edge
        assert fobj._rdm1 is not None
        for edge in fobj.relAO_per_edge:
            for j_ in range(len(edge)):
                for k_ in range(len(edge)):
                    if j_ > k_:
                        continue
                    err_edge.append(fobj._rdm1[edge[j_], edge[k_]])
        # chem potential
        for i in fobj.weight_and_relAO_per_center[1]:
            err_chempot += fobj._rdm1[i, i]

    err_chempot /= Fobjs[0].unitcell_nkpt
    err_edge.append(err_chempot)  # far-end edges are included as err_chempot

    # Compute center errors
    err_cen = []
    for findx, fobj in enumerate(Fobjs):  # type: ignore[assignment]
        # Match RDM for centers
        for cindx, cens in enumerate(fobj.relAO_in_ref_per_edge):
            lenc = len(cens)
            for j_ in range(lenc):
                for k_ in range(lenc):
                    if j_ > k_:
                        continue
                    err_cen.append(
                        Fobjs[fobj.ref_frag_idx_per_edge[cindx]]._rdm1[  # type: ignore[call-overload]
                            cens[j_], cens[k_]
                        ]
                    )

    err_cen.append((Nocc + Delta_n_el / 2))

    # Compute the error vector
    err_vec = array(err_edge) - array(err_cen)

    # Compute the norm of the error vector
    norm_ = mean(err_vec * err_vec) ** 0.5

    return norm_, err_vec


def solve_mp2(
    mf: RHF,
    frozen: int | list[int] | None = None,
    mo_coeff: Matrix[floating] | None = None,
    mo_occ: Vector[floating] | None = None,
    mo_energy: Vector[floating] | None = None,
) -> MP2:
    """
    Perform an MP2 (2nd order Moller-Plesset perturbation theory) calculation.

    This function sets up and runs an MP2 calculation using the provided
    mean-field object.  It returns the MP2 object after the calculation.

    Parameters
    ----------
    mf :
        Mean-field object from PySCF.
    frozen :
        List of frozen orbitals or number of frozen core orbitals. Defaults to None.
    mo_coeff :
        Molecular orbital coefficients. Defaults to None.
    mo_occ :
        Molecular orbital occupations. Defaults to None.
    mo_energy :
        Molecular orbital energies. Defaults to None.

    Returns
    -------
        The MP2 object after running the calculation.
    """
    # Set default values for optional parameters
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    if mo_energy is None:
        mo_energy = mf.mo_energy
    if mo_occ is None:
        mo_occ = mf.mo_occ

    # Initialize the MP2 object
    pt__ = mp.MP2(mf, frozen=frozen, mo_coeff=mo_coeff, mo_occ=mo_occ)
    pt__.verbose = 0

    # Run the MP2 calculation
    pt__.kernel(mo_energy=mo_energy)

    return pt__


def solve_ccsd(
    mf,
    frozen=None,
    mo_coeff=None,
    relax=False,
    use_cumulant=True,
    rdm_return=False,
    rdm2_return=False,
    mo_occ=None,
    mo_energy=None,
    verbose=0,
):
    """
    Solve the CCSD (Coupled Cluster with Single and Double excitations) equations.

    This function sets up and solves the CCSD equations using the provided
    mean-field object.  It can return the CCSD amplitudes (t1, t2),
    the one- and two-particle density matrices, and the CCSD object.

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
        Whether to use cumulant-based energy expression. When using the cumulant, the
        one-particle density matrix is not included in the two-particle density matrix
        calculation (with_dm1 = False). Defaults to True.
    rdm_return : bool, optional
        Whether to return the one-particle density matrix. Defaults to False.
    rdm2_return : bool, optional
        Whether to return the two-particle density matrix. Defaults to False.
    mo_occ : numpy.ndarray, optional
        Molecular orbital occupations. Defaults to None.
    mo_energy : numpy.ndarray, optional
        Molecular orbital energies. Defaults to None.
    verbose : int, optional
        Verbosity level. Defaults to 0.

    Returns
    -------
    tuple
        - t1 (numpy.ndarray): Single excitation amplitudes.
        - t2 (numpy.ndarray): Double excitation amplitudes.
        - rdm1a (numpy.ndarray, optional): One-particle density matrix
            (if rdm_return is True).
        - rdm2s (numpy.ndarray, optional): Two-particle density matrix
            (if rdm2_return is True and rdm_return is True).
        - mycc (pyscf.cc.ccsd.CCSD, optional): CCSD object
            (if rdm_return is True and rdm2_return is False).
    """
    # Set default values for optional parameters
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    if mo_energy is None:
        mo_energy = mf.mo_energy
    if mo_occ is None:
        mo_occ = mf.mo_occ

    # Initialize the CCSD object
    mycc = cc.CCSD(mf, frozen=frozen, mo_coeff=mo_coeff, mo_occ=mo_occ)
    mycc.verbose = 0
    mf = None
    mycc.incore_complete = True

    # Prepare the integrals and Fock matrix
    eris = mycc.ao2mo()
    eris.mo_energy = mo_energy
    eris.fock = diag(mo_energy)

    # Solve the CCSD equations
    try:
        mycc.verbose = verbose
        mycc.kernel(eris=eris)
    except Exception as e:
        print(flush=True)
        print("Exception in CCSD, play with different CC options.", flush=True)
        print(flush=True)
        raise e

    # Extract the CCSD amplitudes
    t1 = mycc.t1
    t2 = mycc.t2

    # Compute and return the density matrices if requested
    if rdm_return:
        if not relax:
            # use PySCF function to make unrelaxed RDMs
            l1 = zeros_like(t1)
            l2 = zeros_like(t2)
            rdm1a = cc.ccsd_rdm.make_rdm1(mycc, t1, t2, l1, l2)
        else:
            rdm1a = mycc.make_rdm1(with_frozen=False)

        if rdm2_return:
            if relax:
                rdm2s = make_rdm2(
                    mycc,
                    t1,
                    t2,
                    mycc.l1,
                    mycc.l2,
                    with_frozen=False,
                    ao_repr=False,
                    with_dm1=not use_cumulant,
                )
            else:
                rdm2s = make_rdm2_urlx(t1, t2, with_dm1=not use_cumulant)
            return (t1, t2, rdm1a, rdm2s)

        return (t1, t2, rdm1a, mycc)

    return (t1, t2)


def solve_block2(
    mf: RHF,
    nocc: int,
    frag_scratch: WorkDir,
    DMRG_args: DMRG_ArgsUser,
    use_cumulant: bool,
):
    """DMRG fragment solver using the pyscf.dmrgscf wrapper.

    Parameters
    ----------
        mf:
            Mean field object or similar following the data signature of the
            pyscf.RHF class.
        nocc:
            Number of occupied MOs in the fragment, used for constructing the
            fragment 1- and 2-RDMs.
        frag_scratch:
            Fragment-level DMRG scratch directory.
        use_cumulant:
            Use the cumulant energy expression.

    Returns
    -------
        rdm1: numpy.ndarray
            1-Particle reduced density matrix for fragment.
        rdm2: numpy.ndarray
            2-Particle reduced density matrix for fragment.
    """
    # pylint: disable-next=E0611
    from pyscf import dmrgscf  # type: ignore[attr-defined]  # noqa: PLC0415

    orbs = mf.mo_coeff

    mc = mcscf.CASCI(mf, DMRG_args.norb, DMRG_args.nelec)
    mc.fcisolver = dmrgscf.DMRGCI(mf.mol)
    # Sweep scheduling
    mc.fcisolver.scheduleSweeps = DMRG_args.schedule_kwargs["scheduleSweeps"]
    mc.fcisolver.scheduleMaxMs = DMRG_args.schedule_kwargs["scheduleMaxMs"]
    mc.fcisolver.scheduleTols = DMRG_args.schedule_kwargs["scheduleTols"]
    mc.fcisolver.scheduleNoises = DMRG_args.schedule_kwargs["scheduleNoises"]

    # Other DMRG parameters
    mc.fcisolver.threads = int(os.environ.get("OMP_NUM_THREADS", "8"))
    mc.fcisolver.twodot_to_onedot = DMRG_args.twodot_to_onedot
    mc.fcisolver.maxIter = DMRG_args.max_iter
    mc.fcisolver.block_extra_keyword = DMRG_args.block_extra_keyword
    mc.fcisolver.scratchDirectory = frag_scratch.path
    mc.fcisolver.runtimeDir = frag_scratch.path
    mc.fcisolver.memory = DMRG_args.max_mem
    os.chdir(frag_scratch)

    mc.kernel(orbs)
    rdm1, rdm2 = dmrgscf.DMRGCI.make_rdm12(
        mc.fcisolver, DMRG_args.root, DMRG_args.norb, DMRG_args.nelec
    )

    # Subtract off non-cumulant contribution to correlated 2RDM.
    if use_cumulant:
        hf_dm = zeros_like(rdm1)
        hf_dm[diag_indices(nocc)] += 2.0

        del_rdm1 = rdm1.copy()
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

        rdm2 -= nc

    return rdm1, rdm2


def solve_uccsd(
    mf,
    eris_inp,
    frozen=None,
    relax=False,
    use_cumulant=True,
    rdm_return=False,
    rdm2_return=False,
    verbose=0,
):
    """
    Solve the U-CCSD (Unrestricted Coupled Cluster with Single and Double excitations)
    equations.

    This function sets up and solves the UCCSD equations using the provided
    mean-field object.
    It can return the one- and two-particle density matrices and the UCCSD object.

    Parameters
    ----------
    mf : pyscf.scf.uhf.UHF
        Mean-field object from PySCF. Constructed with make_uhf_obj
    eris_inp :
        Custom fragment ERIs object
    frozen : list or int, optional
        List of frozen orbitals or number of frozen core orbitals. Defaults to None.
    relax : bool, optional
        Whether to use relaxed density matrices. Defaults to False.
    use_cumulant : bool, optional
        Whether to use cumulant-based energy expression. Defaults to True.
    rdm_return : bool, optional
        Whether to return the one-particle density matrix. Defaults to False.
    rdm2_return : bool, optional
        Whether to return the two-particle density matrix. Defaults to False.
    verbose : int, optional
        Verbosity level. Defaults to 0.

    Returns
    -------
    tuple
        - ucc (pyscf.cc.ccsd.UCCSD): UCCSD object
        - rdm1 (tuple, numpy.ndarray, optional): One-particle density matrix
            (if rdm_return is True).
        - rdm2 (tuple, numpy.ndarray, optional): Two-particle density matrix
            (if rdm2_return is True and rdm_return is True).
    """
    C = mf.mo_coeff

    Vss = eris_inp[:2]
    Vos = eris_inp[-1]

    def ao2mofn(moish):
        if isinstance(moish, ndarray):
            # Since inside '_make_eris_incore' it does not differentiate spin
            # for the two same-spin components, we here brute-forcely determine
            # what spin component we are dealing with by comparing the first
            # 2-by-2 block of the mo coeff matrix.
            # Note that this assumes we have at least two basis functions
            moish_feature = moish[:2, :2]
            s = -1
            for ss in [0, 1]:
                if allclose(moish_feature, C[ss][:2, :2]):
                    s = ss
                    break
            if s < 0:
                raise ValueError("Input mo coeff matrix matches neither moa nor mob.")
            return ao2mo.incore.full(Vss[s], moish, compact=False)
        elif isinstance(moish, list) or isinstance(moish, tuple):
            if len(moish) != 4:
                raise ValueError(
                    "Expect a list/tuple of 4 numpy arrays but get %d of them."
                    % len(moish)
                )
            moish_feature = [mo[:2, :2] for mo in moish]
            for s in [0, 1]:
                Cs_feature = C[s][:2, :2]
                if not (
                    allclose(moish_feature[2 * s], Cs_feature)
                    and allclose(moish_feature[2 * s + 1], Cs_feature)
                ):
                    raise ValueError(
                        "Expect a list/tuple of 4 numpy arrays in the order "
                        "(moa,moa,mob,mob)."
                    )
            try:
                return ao2mo.incore.general(Vos, moish, compact=False)
            except NotImplementedError:
                # ao2mo.incore.general is not implemented for complex numbers
                return einsum(
                    "ijkl,ip,jq,kr,ls->pqrs",
                    Vos,
                    moish[0],
                    moish[1],
                    moish[2],
                    moish[3],
                    optimize=True,
                )
        else:
            raise TypeError(
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
            rdm2 = make_rdm2_uccsd(ucc, relax=relax, with_dm1=not use_cumulant)
            return (ucc, rdm1, rdm2)
        return (ucc, rdm1, None)
    return ucc
