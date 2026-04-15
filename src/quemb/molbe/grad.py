import uuid
from os import system
from os.path import dirname, join
from re import findall
from typing import Literal, TypeAlias

import h5py
from numpy import array, diag, float64, zeros, zeros_like
from numpy.linalg import inv, multi_dot
from pathos.pools import ProcessPool
from pyscf import gto
from pyscf.ao2mo import restore
from pyscf.cc import CCSD
from pyscf.scf import RHF

from quemb.molbe.helper import get_eri, get_scfObj
from quemb.molbe.mbe import BE
from quemb.molbe.pfrag import Frags
from quemb.molbe.solver import Solvers, UserSolverArgs
from quemb.shared.external.lo_helper import symm_orth
from quemb.shared.manage_scratch import WorkDir
from quemb.shared.typing import Matrix, PathLike

Grad_Method: TypeAlias = Literal[
    "force_fd1_ctr_cart",
    # Force Embedding, First-order central finite diff. (Cartesian coord)
    "energy_fd1_ctr_cart",
    # Energy Embedding, First-order central finite diff. (Cartesian coord)
]


class BEGrad:
    """
    Gradient routine for Bootstrap Embedding
    """

    def __init__(self, ref_be_obj: BE):
        self.ref_be_obj: BE = ref_be_obj

        self.delta = 1e-4  # in Angstroms

        # No support for IAO yet
        if self.ref_be_obj.lo_method == "IAO":
            raise NotImplementedError(
                "Gradient calculation with IAO is not supported yet."
            )

    @property
    def grad_method(self) -> Grad_Method:
        return self._grad_method

    def set_grad_method(
        self,
        grad_method: Grad_Method,
    ):
        self._grad_method = grad_method
        self.displacement_vector_list = self._displacement_vector_list(
            grad_method, self.delta
        )
        if "force" in grad_method:
            self.displaced_pfrags = self._force_displaced_pfrags(
                self.displacement_vector_list
            )
        elif "energy" in grad_method:
            pass  # energy embedding does not require preparation of displaced pfrags
        else:
            raise NotImplementedError(f"Unsupported gradient method: {grad_method}")

    def compute_grad(self, solver="MP2", nproc=None, ompnum=None, basis_proj=False):
        if nproc is not None:
            self.ref_be_obj.nproc = nproc
        if ompnum is not None:
            self.ref_be_obj.ompnum = ompnum
        if "force" in self.grad_method:
            self.de = self._compute_force_grad(solver)
        elif "energy" in self.grad_method:
            self.de = self._compute_energy_grad(solver, basis_proj=basis_proj)
        else:
            raise NotImplementedError(
                f"Unsupported gradient method: {self.grad_method}"
            )
        return self.de

    def _compute_force_grad(self, solver):
        """Gradient using force embedding"""
        # HF contribution (analytical)
        grad_hf = self.ref_be_obj.mf.Gradients()
        grad_hf.verbose = 0
        grad_hf.kernel()

        # Evaluate fragment objects in parallel
        system(f"export OMP_NUM_THREADS={self.ref_be_obj.ompnum}")
        nprocs = max(1, self.ref_be_obj.nproc // self.ref_be_obj.ompnum)
        with ProcessPool(nprocs) as pool_:
            results = []
            for fobj in self.displaced_pfrags:
                result = pool_.apipe(
                    _compute_frag,
                    fobj.fock,
                    fobj.dm0.copy(),
                    fobj.dname,
                    fobj.nao,
                    fobj.nsocc,
                    solver,
                    fobj.eri_file,
                    None,
                )
                results.append(result)

            frag_energies = [result.get() for result in results]

        if "fd1" in self.grad_method:
            # First-order central finite difference
            # f'(x) ≈ (f(x + δ) - f(x - δ)) / (2δ)
            grad = zeros((self.ref_be_obj.mf.mol.natm * 3))
            for idx, _ in enumerate(grad):
                # +δ
                grad[idx] += frag_energies[2 * idx]
                # -δ
                grad[idx] -= frag_energies[2 * idx + 1]
            grad /= 2 * self.delta
            return (
                grad.reshape((self.ref_be_obj.mf.mol.natm, 3)) + grad_hf.de / 0.529177
            )

    def _compute_energy_grad(self, solver, basis_proj=False):
        """Gradient using energy embedding"""
        if "fd1" in self.grad_method:
            # First-order central finite difference
            # f'(x) ≈ (f(x + δ) - f(x - δ)) / (2δ)
            grad = zeros((self.ref_be_obj.mf.mol.natm, 3))
            for atomidx in range(self.ref_be_obj.mf.mol.natm):
                for idx, disp in enumerate(self.displacement_vector_list):
                    displaced_be_obj = self._build_displaced_be_objs(atomidx, disp)
                    if basis_proj:
                        s2inv = inv(displaced_be_obj.S)
                        cross_ovlp = gto.intor_cross(
                            "int1e_ovlp",
                            displaced_be_obj.mf.mol,
                            self.ref_be_obj.mf.mol,
                        )
                        for fragidx in range(len(self.ref_be_obj.Fobjs)):
                            basis_proj_TA = multi_dot(
                                (
                                    s2inv,
                                    cross_ovlp,
                                    self.ref_be_obj.Fobjs[fragidx].TA,
                                )
                            )
                            # Orthonormalize
                            basis_proj_TA = symm_orth(
                                basis_proj_TA, 1e-6, displaced_be_obj.S
                            )
                            displaced_be_obj.Fobjs[fragidx] = (
                                self._update_pfrag_with_TA(
                                    displaced_be_obj.Fobjs[fragidx],
                                    basis_proj_TA,
                                    displaced_be_obj.eri_file,
                                    displaced_be_obj.hcore,
                                    displaced_be_obj.hf_veff,
                                    displaced_be_obj.S,
                                    displaced_be_obj.hf_dm,
                                    displaced_be_obj.mf._eri,
                                )
                            )

                    displaced_be_obj.oneshot(
                        solver=solver,
                        use_cumulant=True,
                        nproc=self.ref_be_obj.nproc,
                        ompnum=self.ref_be_obj.ompnum,
                    )
                    if idx % 2 == 0:  # +δ
                        grad[atomidx, idx // 2] += displaced_be_obj.ebe_tot
                    else:  # -δ
                        grad[atomidx, idx // 2] -= displaced_be_obj.ebe_tot
            grad /= 2 * self.delta
            return grad

    def _displacement_vector_list(
        self, grad_method: Grad_Method, delta: float
    ) -> list[list[float]]:
        """Get the displacement vector for finite difference"""
        fd_degree = int(findall(r"fd([0-9]*)", grad_method)[0])
        if fd_degree == 1:
            if "ctr" in grad_method and "cart" in grad_method:
                # First-order central finite difference in Cartesian coordinates
                # f'(x) ≈ (f(x + δ) - f(x - δ)) / (2δ)
                displacement_vector_list = []
                for dim in range(3):  # x, y, z
                    displacement_vector = [0.0, 0.0, 0.0]
                    displacement_vector[dim] = delta  # plus delta
                    displacement_vector_list.append(displacement_vector)
                    displacement_vector = [0.0, 0.0, 0.0]
                    displacement_vector[dim] = -delta  # minus delta
                    displacement_vector_list.append(displacement_vector)
                return displacement_vector_list
                # six displacements for each atom (x+, x-, y+, y-, z+, z-)
        else:
            raise NotImplementedError(f"Unsupported gradient method: {grad_method}")

    def _force_displaced_pfrags(
        self,
        displacement_vector_list: list[list[float]],
    ) -> list[Frags]:
        """Prepare displaced BE objects"""
        displaced_pfrags = []
        # [Fobj for atom 0 x+, Fobj for atom 0 x-, ..., Fobj for atom N z-]
        for atomidx in range(self.ref_be_obj.mf.mol.natm):
            fragidx = self.ref_be_obj.fobj.fragmented.get_frag_per_atom()[atomidx]
            for disp in displacement_vector_list:
                displaced_be_obj = self._build_displaced_be_objs(
                    atomidx, disp, initialize_fragment_idx=[fragidx]
                )
                s2inv = inv(displaced_be_obj.S)
                # Basis projection of reference TA to perturbed geometry
                cross_ovlp = gto.intor_cross(
                    "int1e_ovlp", displaced_be_obj.mf.mol, self.ref_be_obj.mf.mol
                )
                basis_proj_TA = multi_dot(
                    (
                        s2inv,
                        cross_ovlp,
                        self.ref_be_obj.Fobjs[fragidx].TA,
                    )
                )
                # Orthonormalize
                basis_proj_TA = symm_orth(basis_proj_TA, 1e-6, displaced_be_obj.S)
                displaced_pfrags.append(
                    self._update_pfrag_with_TA(
                        displaced_be_obj,
                        fragidx,
                        basis_proj_TA,
                    )
                )
        return displaced_pfrags

    def _update_pfrag_with_TA(
        self,
        be_obj: Frags,
        fidx: int,
        TA: Matrix[float64],
    ) -> Frags:
        """Update the pfrag object with the aligned TA"""
        # TODO: Parallelize after checking pickle-ability
        be_obj.Fobjs[fidx].TA = TA
        be_obj.Fobjs[fidx].nao = TA.shape[1]
        # ERI Transform
        with h5py.File(be_obj.eri_file, "w+") as file_eri:
            del file_eri[
                be_obj.Fobjs[fidx].dname
            ]  # delete the existing ERI dataset for the fragment
            be_obj._eri_transform(
                be_obj.integral_transform,
                be_obj.mf._eri,
                file_eri,
                initialize_fragment_idx=[fidx],
            )
        eri = array(file_eri.get(be_obj.Fobjs[fidx].dname))
        eri = restore(8, eri, be_obj.Fobjs[fidx].nao)
        be_obj.Fobjs[fidx].cons_fock(be_obj.hf_veff, be_obj.S, be_obj.hf_dm, eri_=eri)
        # this implicitly changes P_env
        be_obj.Fobjs[fidx].heff = zeros_like(be_obj.Fobjs[fidx].h1)
        # TODO: Set _mo_coeffs appropriately (get_nsocc)
        be_obj.Fobjs[fidx].scf(fs=True, eri=eri)
        be_obj.Fobjs[fidx].dm0 = 2.0 * (
            be_obj.Fobjs[fidx]._mo_coeffs[:, : be_obj.Fobjs[fidx].nsocc]
            @ be_obj.Fobjs[fidx]._mo_coeffs[:, : be_obj.Fobjs[fidx].nsocc].T
        )
        be_obj.Fobjs[fidx].update_ebe_hf()
        return be_obj.Fobjs[fidx]

    def _build_displaced_be_objs(
        self,
        atomidx: int,
        displacement_vector: list[float],
        initialize_fragment_idx: list[int] | None = None,
    ) -> BE:
        """Build BE object for the displaced molecule"""
        displaced_mol = self.ref_be_obj.mf.mol.copy()
        displaced_mol.atom = [
            [
                self.ref_be_obj.mf.mol.atom_symbol(i),
                self.ref_be_obj.mf.mol.atom_coord(i, unit="Angstrom")
                + displacement_vector
                if i == atomidx
                else self.ref_be_obj.mf.mol.atom_coord(i, unit="Angstrom"),
            ]
            for i in range(self.ref_be_obj.mf.mol.natm)
        ]
        displaced_mol.unit = "Angstrom"
        displaced_mol.build()

        displaced_mf = RHF(displaced_mol)
        displaced_mf.kernel(
            self.ref_be_obj.hf_dm
        )  # use reference 1-RDM as initial guess

        # use randomized ERI file name for each displaced BE obj to avoid conflict
        scratch_dir = WorkDir(
            join(
                dirname(self.ref_be_obj.eri_file),
                f"eri_{id(self)}_{id(displaced_mf)}",
                str(uuid.uuid4()),
            )
        )
        displaced_be_obj = BE(
            displaced_mf,
            self.ref_be_obj.fobj,
            eri_file=scratch_dir / "eri_file.h5",
            lo_method=self.ref_be_obj.lo_method,
            nproc=self.ref_be_obj.nproc,
            ompnum=self.ref_be_obj.ompnum,
            thr_bath=self.ref_be_obj.thr_bath,
            scratch_dir=scratch_dir,
            int_transform=self.ref_be_obj.integral_transform,
            auxbasis=self.ref_be_obj.auxbasis,
            MO_coeff_epsilon=self.ref_be_obj.MO_coeff_epsilon,
            AO_coeff_epsilon=self.ref_be_obj.AO_coeff_epsilon,
            initialize_fragment_idx=initialize_fragment_idx,
        )

        return displaced_be_obj


def _compute_frag(
    h1: Matrix[float64],
    dm0: Matrix[float64],
    dname: str,
    nao: int,
    nocc: int,
    solver: Solvers = "CCSD",
    eri_file: str = "eri_file.h5",
    solver_args: UserSolverArgs | None = None,  # noqa: ARG001
):
    if solver != "CCSD":
        raise NotImplementedError
    eri = get_eri(dname, nao, eri_file=eri_file)
    mf = get_scfObj(h1, eri, nocc, dm0=dm0)

    if solver == "CCSD":
        mc = CCSD(mf, mo_coeff=mf.mo_coeff, mo_occ=mf.mo_occ)
        mc.verbose = 0
        mc.incore_complete = True
        eri_embmo = mc.ao2mo()
        eri_embmo.mo_energy = mf.mo_energy
        eri_embmo.fock = diag(mf.mo_energy)

        try:
            mc.kernel(eris=eri_embmo)
        except Exception as e:
            print(f"Exception during CCSD in Fragment {dname}")
            raise e

        return mc.e_tot - mf.e_tot
