import numpy as np
from dftd3.interface import DispersionModel, RationalDampingParam
from pyscf.scf import hf
from scipy.linalg import eigh


def embedding_density_matrix(TA):
    """Build a density-like matrix for the embedding orbital subspace."""
    return TA @ TA.conj().T


def atomic_populations_from_ao_values(mol, ao_values):
    """Sum one AO-resolved population value over the AOs on each atom."""
    populations = np.zeros(mol.natm)

    for atom_idx, (_, _, ao0, ao1) in enumerate(mol.aoslice_by_atom()):
        populations[atom_idx] = ao_values[ao0:ao1].sum()

    return populations


def mulliken_embedding_populations(mol, density_matrix, overlap):
    """Atomic Mulliken populations of the embedding-space density matrix."""
    ao_populations = np.einsum(
        "ij,ji->i",
        density_matrix,
        overlap,
    ).real
    return atomic_populations_from_ao_values(mol, ao_populations)


def lowdin_embedding_populations(mol, density_matrix, overlap):
    """Atomic symmetric-Löwdin populations of the embedding-space density."""
    eigenvalues, eigenvectors = eigh(overlap)

    if np.any(eigenvalues < 1.0e-12):
        raise ValueError("AO overlap matrix is numerically linearly dependent.")

    overlap_sqrt = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.conj().T

    lowdin_density = overlap_sqrt @ density_matrix @ overlap_sqrt
    ao_populations = np.diag(lowdin_density).real

    return atomic_populations_from_ao_values(mol, ao_populations)


def meta_lowdin_embedding_populations(mol, density_matrix, overlap):
    """Atomic meta-Löwdin populations using PySCF's implementation."""
    ao_populations, _ = hf.mulliken_meta(
        mol,
        density_matrix,
        s=overlap,
        verbose=0,
    )
    return atomic_populations_from_ao_values(mol, ao_populations)


def get_atoms_in_embedding_space(
    mybe,
    mol,
    scheme="ao_density",
    coverage=0.99,
):
    """
    Return the smallest set of atoms that captures a chosen fraction of each
    fragment's embedding-space population.

    Parameters
    ----------
    mybe
        Bootstrap-embedding object containing fragment objects in ``Fobjs``.
    mol
        PySCF Mole object.
    scheme : {"ao_density", "mulliken", "lowdin", "meta_lowdin"}, optional
        Method used to assign the embedding space to atoms.
    coverage : float, optional
        Fraction of the total nonnegative atom score to capture. Must lie in
        the interval ``(0, 1]``.

    Returns
    -------
    list[list[int]]
        Zero-based atom indices included in each fragment's embedding space.
    """
    if not 0.0 < coverage <= 1.0:
        raise ValueError("coverage must be in the interval (0, 1].")

    ao_slices = mol.aoslice_by_atom()
    overlap = None
    atoms_in_embedding_space = []

    for fobj in mybe.Fobjs:
        TA = np.asarray(fobj.TA)

        if TA.ndim != 2:
            raise ValueError(
                f"Expected TA to be two-dimensional, but obtained shape {TA.shape}."
            )

        if TA.shape[0] != mol.nao_nr():
            raise ValueError(
                f"TA contains {TA.shape[0]} AO rows, but mol has {mol.nao_nr()} AOs."
            )

        if scheme == "ao_density":
            atom_scores = np.array(
                [np.sum(np.abs(TA[ao0:ao1, :]) ** 2) for _, _, ao0, ao1 in ao_slices]
            )
        else:
            if overlap is None:
                overlap = mol.intor_symmetric("int1e_ovlp")

            density_matrix = embedding_density_matrix(TA)

            if scheme == "mulliken":
                atom_scores = mulliken_embedding_populations(
                    mol, density_matrix, overlap
                )
            elif scheme == "lowdin":
                atom_scores = lowdin_embedding_populations(mol, density_matrix, overlap)
            elif scheme == "meta_lowdin":
                atom_scores = meta_lowdin_embedding_populations(
                    mol, density_matrix, overlap
                )
            else:
                raise ValueError(
                    "scheme must be one of "
                    "'ao_density', 'mulliken', 'lowdin', or 'meta_lowdin'."
                )

        for atom_index, score in enumerate(atom_scores):
            print(f"Atom {atom_index}: {score:.8f}")

        # Clip tiny negative numerical artifacts from population analyses.
        positive_scores = np.clip(atom_scores, 0.0, None)
        total_score = positive_scores.sum()

        if total_score == 0.0:
            fragment_atoms = []
            print("No positive embedding-space population found.")
        else:
            order = np.argsort(positive_scores)[::-1]
            cumulative_scores = np.cumsum(positive_scores[order])
            target = coverage * total_score
            n_atoms = np.searchsorted(cumulative_scores, target, side="left") + 1

            fragment_atoms = np.sort(order[:n_atoms]).tolist()

            captured_coverage = cumulative_scores[n_atoms - 1] / total_score
            print(
                f"Selected {n_atoms} atoms to capture {captured_coverage:.1%} "
                f"of the embedding-space population "
                f"(requested coverage: {coverage:.1%})."
            )

        atoms_in_embedding_space.append(fragment_atoms)

    return atoms_in_embedding_space


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
        atomic_numbers : numpy.ndarray
            A list of atomic numbers for the system.
        coordinates : numpy.ndarray
            A 2D array of shape (N, 3) where N is the number of atoms
            and each row contains the x, y, z coordinates of an atom.

        Returns
        -------
        energy : float
            The D3 dispersion energy.
        gradient : numpy.ndarray, optional
            The gradient of the D3 dispersion energy with respect to atomic coordinates,
            returned if grad=True
        """
        model = DispersionModel(atomic_numbers, coordinates)

        # pdb.set_trace()
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
        coordinates : numpy.ndarray
            A 2D array of shape (N, 3) where N is the number of atoms
            and each row contains the x, y, z coordinates of an atom.
        grad : bool
            If True, also compute the gradient of the D3 dispersion energy
            with respect to atomic coordinates

        Returns
        -------
        energy : float
            The D3 dispersion energy for the specified fragment.
        gradient : numpy.ndarray, optional
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
        coordinates : numpy.ndarray
            A 2D array of shape (N, 3) where N is the number of atoms
            and each row contains the x, y, z coordinates of an atom.

        Returns
        -------
        fragment_energies : numpy.ndarray[float]
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
        coordinates : numpy.ndarray
            A 2D array of shape (N, 3) where N is the number of atoms
            and each row contains the x, y, z coordinates of an atom.

        Returns
        -------
        fragment_energies : numpy.ndarray[float]
            The external D3 dispersion energy for each fragment.
        fragment_gradients : list[numpy.ndarray[float]]
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
        coordinates : numpy.ndarray
            A 2D array of shape (N, 3) where N is the number of atoms
            and each row contains the x, y, z coordinates of an atom.

        Returns
        -------
        energy : float
            The D3 dispersion energy for the specified fragment.
        gradient : numpy.ndarray
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


def finite_difference_fragment_gradients(
    d3: D3,
    coordinates: np.ndarray,
    step: float = 1.0e-4,
) -> list[np.ndarray]:
    """
    Compute fragment gradients by central finite differences.

    Parameters
    ----------
    d3
        Initialized D3 object.
    coordinates
        Coordinates in the same units expected by D3.
    step
        Finite-difference displacement in the same units as coordinates.

    Returns
    -------
    list[numpy.ndarray]
        One gradient array of shape (n_fragment_atoms, 3) per fragment.
    """
    fd_gradients: list[np.ndarray] = []

    for fragment_index, atom_indices in enumerate(d3.fragments):
        gradient = np.zeros((len(atom_indices), 3), dtype=float)

        for local_atom_index, global_atom_index in enumerate(atom_indices):
            for xyz in range(3):
                coordinates_plus = coordinates.copy()
                coordinates_minus = coordinates.copy()

                coordinates_plus[global_atom_index, xyz] += step
                coordinates_minus[global_atom_index, xyz] -= step

                energies_plus = d3.energy(coordinates_plus)
                energies_minus = d3.energy(coordinates_minus)

                gradient[local_atom_index, xyz] = (
                    energies_plus[fragment_index] - energies_minus[fragment_index]
                ) / (2.0 * step)

        fd_gradients.append(gradient)

    return fd_gradients
