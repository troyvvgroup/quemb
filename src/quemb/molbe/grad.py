from os.path import dirname, join
from re import findall
from typing import Literal, TypeAlias

from pyscf.scf import RHF

from quemb.molbe.mbe import BE

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

    def set_grad_method(self, grad_method: Grad_Method):
        self._grad_method = grad_method
        self.displacement_vector_list = self._displacement_vector_list(
            grad_method, self.delta
        )
        if "force" in grad_method:
            self.displaced_pfrags = self._force_displaced_pfrags(
                self.displacement_vector_list
            )
        else:
            raise NotImplementedError(f"Unsupported gradient method: {grad_method}")

    def compute_grad(self):
        if "force" in self.grad_method:
            return self._compute_force_grad()
        else:
            raise NotImplementedError(
                f"Unsupported gradient method: {self.grad_method}"
            )

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

    def _force_displaced_pfrags(self, displacement_vector_list: list[list[float]]):
        """Prepare displaced BE objects"""
        displaced_pfrags = []
        # [Fobj for atom 0 x+, Fobj for atom 0 x-, ..., Fobj for atom N z-]
        for atomidx in range(self.ref_be_obj.mf.mol.natm):
            fragidx = self.ref_be_obj.fobj.fragmented.get_frag_per_atom()[atomidx]
            for disp in displacement_vector_list:
                displaced_be_obj = self._build_displaced_be_objs(disp)
                displaced_pfrags.append(displaced_be_obj.Fobjs[fragidx])
        return displaced_pfrags

    def _build_displaced_be_objs(self, displacement_vector: list[float]) -> BE:
        """Build BE object for the displaced molecule"""
        displaced_mol = self.ref_be_obj.mf.mol.copy()
        displaced_mol.atom = [
            [
                self.ref_be_obj.mf.mol.atom_symbol(i),
                self.ref_be_obj.mf.mol.atom_coord(i, unit="Angstrom")
                + displacement_vector,
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
        scratch_dir = join(
            dirname(self.ref_be_obj.eri_file), f"eri_{id(self)}_{id(displaced_mf)}"
        )
        displaced_be_obj = BE(
            displaced_mf,
            self.ref_be_obj.fobj,
            eri_file=join(scratch_dir, "eri_file.h5"),
            lo_method=self.ref_be_obj.lo_method,
            nproc=self.ref_be_obj.nproc,
            ompnum=self.ref_be_obj.ompnum,
            thr_bath=self.ref_be_obj.thr_bath,
            scratch_dir=scratch_dir,
            int_transform=self.ref_be_obj.int_transform,
            auxbasis=self.ref_be_obj.auxbasis,
            MO_coeff_epsilon=self.ref_be_obj.MO_coeff_epsilon,
            AO_coeff_epsilon=self.ref_be_obj.AO_coeff_epsilon,
        )  # TODO: avoid unnecessary integral transformation in the future.
        #       This would require modification of the BE class to allow
        #       lazy initialization.

        return displaced_be_obj
