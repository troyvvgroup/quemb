import pdb

import numpy as np
from dftd3.interface import DispersionModel, RationalDampingParam


class D3:
    def __init__(
        self,
        atomic_numbers: list[int],
        fragments: list[list[int]],
        method: str = "hf",
        **kwargs,
    ):
        """
        Initialize the D3 dispersion correction.

        Parameters
        ----------
        atomic_numbers : list[int]
            A list of atomic numbers for the system.
        fragments : list[list[int]]
            A list of fragments, where each fragment is defined by
            a list of zero-indexed atom indices.
        method : str
            The D3 method to use (e.g., "hf", "b3lyp").
        **kwargs
            Additional keyword arguments for the damping function.
        """
        self.atomic_numbers = np.asarray(atomic_numbers)
        self.n_atoms = len(atomic_numbers)
        self.fragments = fragments
        self.n_fragments = len(fragments)
        self.method = method
        self.inverted_fragments = self._invert_fragments()
        if method is not None:
            self.damping = RationalDampingParam(method=method)
        else:
            self.damping = RationalDampingParam(**kwargs)

    def _invert_fragments(self) -> list[list[int]]:
        """
        For each fragment, compute the list of all atoms *not* in that fragment.

        Returns
        -------
        list[list[int]]
            A list where each sublist contains the atom indices for a fragment.
        """
        inverted_fragments = []
        for fragment in self.fragments:
            inverted_fragments.append(
                [i for i in range(self.n_atoms) if i not in fragment]
            )
        return inverted_fragments

    def _call_d3(self, atomic_numbers: np.ndarray, coordinates: np.ndarray, grad=False):
        """
        Call the D3 dispersion energy calculation.

        Parameters
        ----------
        atomic_numbers : np.ndarray
            A list of atomic numbers for the system.
        coordinates : np.ndarray
            A 2D array of shape (N, 3) where N is the number of atoms
            and each row contains the x, y, z coordinates of an atom.

        Returns
        -------
        energy : float
            The D3 dispersion energy.
        gradient : np.ndarray, optional
            The gradient of the D3 dispersion energy with respect to atomic coordinates,
            returned if grad=True
        """
        model = DispersionModel(atomic_numbers, coordinates)

        pdb.set_trace()
        res = model.get_dispersion(self.damping, grad=grad)
        energy = res["energy"]
        if grad:
            gradient = res["gradient"]
            return energy, gradient
        return energy

    def _fragment_call_d3(
        self, fragment: list[int], coordinates: np.ndarray, grad=False
    ):
        """
        Calculate the D3 dispersion energy for a specific fragment.

        Parameters
        ----------
        fragment : list[int]
            A list of zero-indexed atom indices defining the fragment.
        coordinates : np.ndarray
            A 2D array of shape (N, 3) where N is the number of atoms
            and each row contains the x, y, z coordinates of an atom.
        grad : bool
            If True, also compute the gradient of the D3 dispersion energy
            with respect to atomic coordinates

        Returns
        -------
        energy : float
            The D3 dispersion energy for the specified fragment.
        gradient : np.ndarray, optional
            The gradient of the D3 dispersion energy with respect to atomic coordinates,
            returned if grad=True
        """
        fragment_coords = coordinates[fragment]
        fragment_atomic_numbers = self.atomic_numbers[fragment]
        return self._call_d3(fragment_atomic_numbers, fragment_coords, grad=grad)

    def energy(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Calculate the D3 dispersion energy for a given set of atomic coordinates.

        Parameters
        ----------
        coordinates : np.ndarray
            A 2D array of shape (N, 3) where N is the number of atoms
            and each row contains the x, y, z coordinates of an atom.

        Returns
        -------
        np.ndarray[float]
            The external D3 dispersion energy for each fragment.
        """
        total_energy = self._call_d3(self.atomic_numbers, coordinates)
        fragment_energies = np.zeros(self.n_fragments)
        for i, fragment in enumerate(self.fragments):
            fragment_energy = self._fragment_call_d3(fragment, coordinates)
            inverted_energy = self._fragment_call_d3(
                self.inverted_fragments[i], coordinates
            )
            fragment_energies[i] = total_energy - fragment_energy - inverted_energy
        return fragment_energies

    def energy_and_gradient(
        self, coordinates: np.ndarray
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """
        Calculate the D3 dispersion energy and its gradient
        for a given set of atomic coordinates.

        Parameters
        ----------
        coordinates : np.ndarray
            A 2D array of shape (N, 3) where N is the number of atoms
            and each row contains the x, y, z coordinates of an atom.

        Returns
        -------
        fragment_energies : np.ndarray[float]
            The external D3 dispersion energy for each fragment.
        fragment_gradients : list[np.ndarray[float]]
            The gradient of the external D3 dispersion energy
            with respect to atomic coordinates for each fragment.
        """
        total_energy, total_gradient = self._call_d3(
            self.atomic_numbers, coordinates, grad=True
        )
        fragment_energies = np.zeros(self.n_fragments)
        fragment_gradients = []

        for i, fragment in enumerate(self.fragments):
            fragment_energy, fragment_gradient = self._fragment_call_d3(
                fragment, coordinates, grad=True
            )
            inverted_energy, inverted_gradient = self._fragment_call_d3(
                self.inverted_fragments[i], coordinates, grad=True
            )
            fragment_energies[i] = total_energy - fragment_energy - inverted_energy
            fragment_gradients.append(total_gradient[fragment] - fragment_gradient)

        return fragment_energies, fragment_gradients

    def fragment_energy_and_gradient(
        self, fragment: int, coordinates: np.ndarray
    ) -> tuple[float, np.ndarray]:
        """
        Calculate the D3 dispersion energy and its gradient for a specific fragment.

        Parameters
        ----------
        fragment : list[int]
            A list of zero-indexed atom indices defining the fragment.
        coordinates : np.ndarray
            A 2D array of shape (N, 3) where N is the number of atoms
            and each row contains the x, y, z coordinates of an atom.

        Returns
        -------
        energy : float
            The D3 dispersion energy for the specified fragment.
        gradient : np.ndarray
            The gradient of the D3 dispersion energy with respect to atomic coordinates
            for the specified fragment.
        """
        fragment_idxs = self.fragments[fragment]
        total_energy, total_gradient = self._call_d3(
            self.atomic_numbers, coordinates, grad=True
        )
        fragment_energy, fragment_gradient = self._fragment_call_d3(
            fragment_idxs, coordinates, grad=True
        )
        inverted_energy, inverted_gradient = self._fragment_call_d3(
            self.inverted_fragments[fragment], coordinates, grad=True
        )
        energy = total_energy - fragment_energy - inverted_energy
        gradient = total_gradient[fragment_idxs] - fragment_gradient
        return energy, gradient


# Example of how to use this
# if __name__ == "__main__":
#    atomic_numbers = [6, 1, 1, 6, 1, 6, 1, 1, 1, 6, 1, 6, 1,
#                      1, 1, 6, 1, 6, 1, 1, 1, 6, 1, 1, 1, 1]
#    coordinates = np.loadtxt("geom.xyz", skiprows=2, usecols=(1, 2, 3))
#    fragments = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 25],
#                 [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]]
#    d3 = D3(atomic_numbers, fragments, method="hf")
#    energies, gradients = d3.energy_and_gradient(coordinates)
#    print("Fragment Energies:", energies)
#    print("Fragment Gradients:", gradients)
