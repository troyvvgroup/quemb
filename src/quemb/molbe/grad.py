from re import findall
from typing import Literal, TypeAlias

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

    def __init__(self, ref_be_obj: BE, grad_method: Grad_Method):
        self.ref_be_obj: BE = ref_be_obj
        self.grad_method = grad_method

        self.delta = 1e-4  # in Angstroms

    def _displacement_vector(self, grad_method: Grad_Method, delta: float):
        """Get the displacement vector for finite difference"""
        fd_degree = int(findall(r"fd([0-9]*)", grad_method)[0])
        if fd_degree == 1:
            if "ctr" in grad_method and "cart" in grad_method:
                # First-order central finite difference in Cartesian coordinates
                # f'(x) ≈ (f(x + δ) - f(x - δ)) / (2δ)
                displacement_vector = []
                for dim in range(3):  # x, y, z
                    displacement_vector.append([0.0, 0.0, 0.0])
                    displacement_vector[-1][dim] = delta  # plus delta
                    displacement_vector.append([0.0, 0.0, 0.0])
                    displacement_vector[-1][dim] = -delta  # minus delta
                return displacement_vector
        else:
            raise NotImplementedError(f"Unsupported gradient method: {grad_method}")

    def _prepare_displaced_beobjs(self, displacement_vector: list[list[float]]):
        """Prepare displaced BE objects"""
        # Assume 2 point finite difference (+- delta) for now
        self.displaced_be_objs = []
        for i in range(self.ref_be_obj.mf.mol.natm):  # each atom
            for disp in displacement_vector:  # each displacement vector
                displaced_mol = self.ref_be_obj.mf.mol.copy()
                displaced_mol.atom = [
                    [
                        self.ref_be_obj.mf.mol.atom_symbol(i),
                        self.ref_be_obj.mf.mol.atom_coord(i, unit="Angstrom") + disp,
                    ]
                    for i in range(self.ref_be_obj.mf.mol.natm)
                ]
                displaced_mol.unit = "Angstrom"
                displaced_mol.build()
