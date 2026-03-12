import uuid
from os import system
from os.path import dirname, join
from re import findall
from typing import Literal, TypeAlias

import h5py
from numpy import diag, float64, trace, zeros, zeros_like
from numpy.linalg import eigh, multi_dot
from pathos.pools import ProcessPool
from pyscf import gto
from pyscf.ao2mo import incore, restore
from pyscf.cc import CCSD
from pyscf.scf import RHF

from quemb.molbe.helper import corr_orbital, corr_orbital_frag_idx, get_eri, get_scfObj
from quemb.molbe.mbe import BE
from quemb.molbe.pfrag import Frags
from quemb.molbe.solver import Solvers, UserSolverArgs
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
        alignment: Literal["corr_full", "no_alignment"] | None = "corr_full",
    ):
        self._grad_method = grad_method
        self.displacement_vector_list = self._displacement_vector_list(
            grad_method, self.delta
        )
        if "force" in grad_method:
            self.displaced_pfrags = self._force_displaced_pfrags(
                self.displacement_vector_list, alignment=alignment
            )
        elif "energy" in grad_method:
            pass  # energy embedding does not require preparation of displaced pfrags
        else:
            raise NotImplementedError(f"Unsupported gradient method: {grad_method}")

    def compute_grad(self, solver="MP2", nproc=None, ompnum=None):
        if nproc is not None:
            self.ref_be_obj.nproc = nproc
        if ompnum is not None:
            self.ref_be_obj.ompnum = ompnum
        if "force" in self.grad_method:
            self.de = self._compute_force_grad(solver)
        elif "energy" in self.grad_method:
            self.de = self._compute_energy_grad(solver)
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

    def _compute_energy_grad(self, solver):
        """Gradient using energy embedding"""
        if "fd1" in self.grad_method:
            # First-order central finite difference
            # f'(x) ≈ (f(x + δ) - f(x - δ)) / (2δ)
            grad = zeros((self.ref_be_obj.mf.mol.natm, 3))
            for atomidx in range(self.ref_be_obj.mf.mol.natm):
                for idx, disp in enumerate(self.displacement_vector_list):
                    displaced_be_obj = self._build_displaced_be_objs(atomidx, disp)
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
        alignment: Literal["corr_full", "no_alignment"] = "corr_full",
    ) -> list[Frags]:
        """Prepare displaced BE objects"""
        displaced_pfrags = []
        # [Fobj for atom 0 x+, Fobj for atom 0 x-, ..., Fobj for atom N z-]
        for atomidx in range(self.ref_be_obj.mf.mol.natm):
            fragidx = self.ref_be_obj.fobj.fragmented.get_frag_per_atom()[atomidx]
            for disp in displacement_vector_list:
                displaced_be_obj = self._build_displaced_be_objs(atomidx, disp)
                if alignment == "corr_full":
                    # Corresponding Orbital Transformation to reference geometry
                    cot = corr_orbital_frag_idx(
                        displaced_be_obj,
                        self.ref_be_obj,
                        idx_list=[fragidx],
                    )[0]
                    cot = cot[0] @ cot[2]  # Σ = U @ V^T
                    aligned_TA = displaced_be_obj.Fobjs[fragidx].TA @ cot
                    displaced_pfrags.append(
                        self._update_pfrag_with_TA(
                            displaced_be_obj.Fobjs[fragidx],
                            aligned_TA,
                            displaced_be_obj.eri_file,
                            displaced_be_obj.hcore,
                            displaced_be_obj.hf_veff,
                            displaced_be_obj.S,
                            displaced_be_obj.hf_dm,
                            displaced_be_obj.mf._eri,
                        )
                    )
                elif alignment == "mop_noo":  # MO(perturbed) x NO(ref)
                    # This version tracks Carina's original implementation.
                    # It performs COT between the perturbed MOs and
                    # reference embedding space NOs
                    cross_ovlp = gto.intor_cross(
                        "int1e_ovlp", displaced_be_obj.mf.mol, self.ref_be_obj.mf.mol
                    )

                    # Evaluate natural orbitals for the reference fragment
                    emb_nelec = trace(self.ref_be_obj.Fobjs[fragidx].dm0)
                    no_occ, no_coeff_in_EObasis = eigh(
                        self.ref_be_obj.Fobjs[fragidx].dm0
                    )  # numpy eigh returns ascending order
                    no_occ = no_occ[::-1]
                    no_coeff_in_EObasis = no_coeff_in_EObasis[
                        :, ::-1
                    ]  # reverse to descending order
                    no_occ = [
                        2 * (x < (emb_nelec / 2)) for x in range(len(no_occ))
                    ]  # peg to exactly 2
                    no_in_AObasis = (
                        self.ref_be_obj.Fobjs[fragidx].TA @ no_coeff_in_EObasis
                    )

                    # Note that Carina's version enforces block diagonal form for
                    # occ-virt separation unlike what is done here.
                    cot = corr_orbital(displaced_be_obj.C, no_in_AObasis, cross_ovlp)
                    cot = cot[0] @ cot[2]  # Σ = U @ V^T

                    aligned_TA = displaced_be_obj.C @ cot @ no_coeff_in_EObasis.T
                    displaced_pfrags.append(
                        self._update_pfrag_with_TA(
                            displaced_be_obj.Fobjs[fragidx],
                            aligned_TA,
                            displaced_be_obj.eri_file,
                            displaced_be_obj.hcore,
                            displaced_be_obj.hf_veff,
                            displaced_be_obj.S,
                            displaced_be_obj.hf_dm,
                            displaced_be_obj.mf._eri,
                        )
                    )
                elif alignment == "no_alignment":
                    displaced_pfrags.append(displaced_be_obj.Fobjs[fragidx])
                else:
                    raise NotImplementedError(
                        f"Unsupported alignment method: {alignment}"
                    )
        return displaced_pfrags

    def _update_pfrag_with_TA(
        self,
        frag_obj: Frags,
        TA: Matrix[float64],
        eri_file: PathLike,
        h1: Matrix[float64],
        hf_veff: Matrix[float64],
        ao_ovlp: Matrix[float64],
        hf_dm: Matrix[float64],
        ao_eri: Matrix[float64] | None = None,
    ) -> Frags:
        """Update the pfrag object with the aligned TA"""
        # TODO: Parallelize after checking pickle-ability
        frag_obj.TA = TA
        frag_obj.nao = TA.shape[1]
        # ERI Transform
        # TODO: Support other types of ERI transformation.
        file_eri = h5py.File(eri_file, "r+")
        del file_eri[frag_obj.dname]  # delete the existing ERI dataset for the fragment
        eri_eo = incore.full(ao_eri, TA, compact=True)
        file_eri.create_dataset(frag_obj.dname, data=eri_eo)
        frag_obj.h1 = multi_dot((TA.T, h1, TA))
        eri_eo = restore(8, eri_eo, frag_obj.nao)
        frag_obj.cons_fock(hf_veff, ao_ovlp, hf_dm, eri_=eri_eo)
        # this implicitly changes P_env
        frag_obj.heff = zeros_like(frag_obj.h1)
        # TODO: Set _mo_coeffs appropriately (get_nsocc)
        frag_obj.scf(fs=True, eri=eri_eo)
        frag_obj.dm0 = 2.0 * (
            frag_obj._mo_coeffs[:, : frag_obj.nsocc]
            @ frag_obj._mo_coeffs[:, : frag_obj.nsocc].T
        )
        frag_obj.update_ebe_hf()
        file_eri.close()
        return frag_obj

    def _build_displaced_be_objs(
        self, atomidx: int, displacement_vector: list[float]
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
        )  # TODO: avoid unnecessary integral transformation in the future.
        #       This would require modification of the BE class to allow
        #       lazy initialization.

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
