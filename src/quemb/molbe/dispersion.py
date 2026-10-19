import re
from functools import cache
from importlib.resources import files

import numpy as np
import torch
from dftd3.interface import DispersionModel, RationalDampingParam
from pyscf.scf import hf
from scipy.linalg import eigh

ANGSTROM_TO_BOHR = 1.8897261254578281


@cache
def _d3_r4r2_table() -> torch.Tensor:
    """Return simple-dftd3 C8 factors, indexed by atomic number."""
    source = files("quemb.molbe").joinpath("data", "r4r2.f90").read_text()

    match = re.search(
        r"r4_over_r2\(max_elem\)\s*=\s*\[\s*&(?P<data>.*?)\]",
        source,
        flags=re.DOTALL,
    )
    if match is None:
        raise RuntimeError("Could not find r4_over_r2 in r4r2.f90.")

    data_block = re.sub(r"!.*", "", match.group("data"))
    raw_r4r2 = [
        float(value)
        for value in re.findall(
            r"([+-]?(?:\d+\.\d*|\.\d+)(?:[Ee][+-]?\d+)?)_wp",
            data_block,
        )
    ]

    if len(raw_r4r2) != 118:
        raise RuntimeError(
            f"Expected 118 r4_over_r2 values in r4r2.f90, but found {len(raw_r4r2)}."
        )

    atomic_numbers = torch.arange(1, 119, dtype=torch.float64)
    r4r2_factors = torch.sqrt(
        0.5 * torch.tensor(raw_r4r2, dtype=torch.float64) * torch.sqrt(atomic_numbers)
    )

    return torch.cat((torch.zeros(1, dtype=torch.float64), r4r2_factors))


def d3_r4r2(
    atomic_numbers: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return simple-dftd3 C8 factors for each atomic number."""
    if atomic_numbers.ndim != 1:
        raise ValueError("atomic_numbers must have shape (n_atoms,).")

    if atomic_numbers.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise TypeError("atomic_numbers must have an integer dtype.")

    if torch.any(atomic_numbers < 1) or torch.any(atomic_numbers > 118):
        raise ValueError("atomic_numbers must lie between 1 and 118.")

    table = _d3_r4r2_table().to(dtype=dtype, device=device)
    return table[atomic_numbers.to(device=device)]


def d3_bj_energy(
    coordinates: torch.Tensor,
    atomic_numbers: torch.Tensor,
    *,
    s6: float,
    s8: float,
    a1: float,
    a2: float,
) -> torch.Tensor:
    """Compute the two-body D3(BJ) dispersion energy.

    Coordinates must be expressed in bohr.  The supported elements are those
    for which the bundled D3 reference-CN and C6 data are available.
    """
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError(
            "coordinates must have shape (n_atoms, 3), "
            f"but has shape {tuple(coordinates.shape)}."
        )
    if atomic_numbers.ndim != 1 or atomic_numbers.shape[0] != coordinates.shape[0]:
        raise ValueError(
            "atomic_numbers must have shape (n_atoms,), with n_atoms equal "
            "to coordinates.shape[0]."
        )

    coordination_numbers = d3_coordination_numbers(
        coordinates,
        atomic_numbers,
    )
    c6 = d3_pairwise_c6(
        atomic_numbers,
        coordination_numbers,
    )
    r4r2 = d3_r4r2(
        atomic_numbers,
        dtype=coordinates.dtype,
        device=coordinates.device,
    )
    c8 = d3_pairwise_c8(
        c6,
        r4r2,
    )
    return d3_bj_two_body_energy(
        coordinates,
        c6,
        c8,
        s6=s6,
        s8=s8,
        a1=a1,
        a2=a2,
    )


@cache
def d3_reference_data() -> tuple[
    dict[int, torch.Tensor],
    dict[tuple[int, int], torch.Tensor],
]:
    """Read all D3 reference-CN and C6 data from reference.f90."""
    source = files("quemb.molbe").joinpath("data", "reference.f90").read_text()

    def without_comments(text: str) -> str:
        return re.sub(r"!.*", "", text)

    def fortran_float_values(text: str) -> list[float]:
        return [
            float(value)
            for value in re.findall(
                r"([+-]?(?:\d+\.\d*|\.\d+)(?:[Ee][+-]?\d+)?)_wp",
                text,
            )
        ]

    reference_count_match = re.search(
        r"number_of_references\(max_elem\)\s*=\s*\[\s*&(?P<data>.*?)\]",
        source,
        flags=re.DOTALL,
    )
    if reference_count_match is None:
        raise RuntimeError("Could not find number_of_references in reference.f90.")

    number_of_references = [
        int(value)
        for value in re.findall(
            r"\d+",
            without_comments(reference_count_match.group("data")),
        )
    ]

    max_elem = len(number_of_references)
    max_ref = 7

    reference_cn_match = re.search(
        r"reference_cn\(max_ref,\s*max_elem\)\s*=\s*reshape\(\[\s*&"
        r"(?P<data>.*?)\[\s*max_ref,\s*max_elem\s*\]\s*\)",
        source,
        flags=re.DOTALL,
    )
    if reference_cn_match is None:
        raise RuntimeError("Could not find reference_cn in reference.f90.")

    reference_cn_values = fortran_float_values(
        without_comments(reference_cn_match.group("data"))
    )
    if len(reference_cn_values) != max_ref * max_elem:
        raise RuntimeError(
            "reference_cn has an unexpected number of entries: "
            f"{len(reference_cn_values)}."
        )

    # reference_cn is reshaped in Fortran column-major order.
    reference_cn_table = np.asarray(reference_cn_values).reshape(
        (max_ref, max_elem),
        order="F",
    )

    c6_chunks = re.finditer(
        r"c6ab_view\((?P<start>\d+):(?P<stop>\d+)\)\s*=\s*\[\s*&"
        r"(?P<data>.*?)\]",
        source,
        flags=re.DOTALL,
    )

    c6_values: list[float] = []
    expected_start = 1

    for chunk in c6_chunks:
        start = int(chunk.group("start"))
        stop = int(chunk.group("stop"))

        if start != expected_start:
            raise RuntimeError(
                f"Expected C6 data to start at {expected_start}, but found {start}."
            )

        values = fortran_float_values(without_comments(chunk.group("data")))
        if len(values) != stop - start + 1:
            raise RuntimeError(
                f"C6 block {start}:{stop} contains {len(values)} values."
            )

        c6_values.extend(values)
        expected_start = stop + 1

    n_pairs = max_elem * (max_elem + 1) // 2
    expected_c6_count = max_ref * max_ref * n_pairs

    if len(c6_values) != expected_c6_count:
        raise RuntimeError(
            f"Expected {expected_c6_count} C6 values, but found {len(c6_values)}."
        )

    # This reproduces the Fortran c6ab(max_ref, max_ref, n_pairs) layout.
    c6_table = np.asarray(c6_values).reshape(
        (max_ref, max_ref, n_pairs),
        order="F",
    )

    reference_coordination_numbers: dict[int, torch.Tensor] = {}
    c6_reference: dict[tuple[int, int], torch.Tensor] = {}

    for atomic_number, n_ref in enumerate(number_of_references, start=1):
        reference_coordination_numbers[atomic_number] = torch.tensor(
            reference_cn_table[:n_ref, atomic_number - 1],
            dtype=torch.float64,
        )

    pair_index = 0
    for atomic_number_j in range(1, max_elem + 1):
        n_ref_j = number_of_references[atomic_number_j - 1]

        for atomic_number_i in range(1, atomic_number_j + 1):
            n_ref_i = number_of_references[atomic_number_i - 1]

            # simple-dftd3 stores the upper-triangular pair block with its
            # reference-state axes reversed; transpose it into (i, j) order.
            c6_reference[(atomic_number_i, atomic_number_j)] = torch.tensor(
                c6_table[:n_ref_j, :n_ref_i, pair_index].T,
                dtype=torch.float64,
            )
            pair_index += 1

    return reference_coordination_numbers, c6_reference


@cache
def _d3_covalent_radius_table() -> torch.Tensor:
    """Return mctc-data D3 covalent radii indexed by atomic number."""
    source = files("quemb.molbe").joinpath("data", "covrad.f90").read_text()

    match = re.search(
        r"covalent_rad_2009\(max_elem\)\s*=\s*aatoau\s*\*\s*\[(.*?)\]",
        source,
        flags=re.DOTALL,
    )
    if match is None:
        raise RuntimeError("Could not find covalent_rad_2009 in covrad.f90.")

    radii_block = re.sub(r"!.*", "", match.group(1))
    radii_angstrom = [
        float(value) for value in re.findall(r"(\d+\.\d+)_wp", radii_block)
    ]

    if len(radii_angstrom) != 118:
        raise RuntimeError(
            "Expected 118 covalent radii in covrad.f90, "
            f"but found {len(radii_angstrom)}."
        )

    # Prepend a dummy entry: mctc-data uses one-based atomic numbers.
    return (
        torch.tensor(
            [0.0, *radii_angstrom],
            dtype=torch.float64,
        )
        * (4.0 / 3.0)
        * ANGSTROM_TO_BOHR
    )


def d3_covalent_radii(
    atomic_numbers: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return mctc-data D3 covalent radii for atomic_numbers, in bohr."""
    if atomic_numbers.ndim != 1:
        raise ValueError("atomic_numbers must have shape (n_atoms,).")

    if atomic_numbers.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise TypeError("atomic_numbers must have an integer dtype.")

    if torch.any(atomic_numbers < 1) or torch.any(atomic_numbers > 118):
        raise ValueError("atomic_numbers must lie between 1 and 118.")

    radius_table = _d3_covalent_radius_table().to(
        dtype=dtype,
        device=device,
    )
    return radius_table[atomic_numbers.to(device=device)]


def d3_bj_two_body_energy(
    coordinates: torch.Tensor,
    c6: torch.Tensor,
    c8: torch.Tensor,
    *,
    s6: float,
    s8: float,
    a1: float,
    a2: float,
) -> torch.Tensor:
    """Compute the pairwise D3(BJ) dispersion energy.

    Parameters
    ----------
    coordinates
        Cartesian coordinates with shape ``(n_atoms, 3)``.
    c6, c8
        Pairwise coefficient matrices, each with shape
        ``(n_atoms, n_atoms)``.
    s6, s8, a1, a2
        D3(BJ) damping parameters.

    Returns
    -------
    torch.Tensor
        Scalar D3(BJ) two-body dispersion energy.
    """
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError(
            "coordinates must have shape (n_atoms, 3), "
            f"but has shape {tuple(coordinates.shape)}."
        )

    n_atoms = coordinates.shape[0]
    expected_shape = (n_atoms, n_atoms)
    if c6.shape != expected_shape or c8.shape != expected_shape:
        raise ValueError(
            "c6 and c8 must both have shape "
            f"{expected_shape}, but have shapes {tuple(c6.shape)} "
            f"and {tuple(c8.shape)}."
        )
    if c6.dtype != coordinates.dtype or c8.dtype != coordinates.dtype:
        raise TypeError("c6 and c8 must have the same dtype as coordinates.")
    if c6.device != coordinates.device or c8.device != coordinates.device:
        raise ValueError("c6 and c8 must be on the same device as coordinates.")

    distances = pairwise_distances(coordinates)
    pair_indices = torch.triu_indices(
        n_atoms,
        n_atoms,
        offset=1,
        device=coordinates.device,
    )
    atom_i, atom_j = pair_indices

    pair_distances = distances[atom_i, atom_j]
    pair_c6 = c6[atom_i, atom_j]
    pair_c8 = c8[atom_i, atom_j]

    if torch.any(pair_c6 <= 0.0) or torch.any(pair_c8 <= 0.0):
        raise ValueError("Off-diagonal C6 and C8 coefficients must be positive.")

    r0 = torch.sqrt(pair_c8 / pair_c6)
    damping_radius = a1 * r0 + a2

    c6_term = s6 * pair_c6 / (pair_distances**6 + damping_radius**6)
    c8_term = s8 * pair_c8 / (pair_distances**8 + damping_radius**8)

    return -(c6_term + c8_term).sum()


def d3_reference_weights(
    coordination_numbers: torch.Tensor,
    reference_coordination_numbers: torch.Tensor,
) -> torch.Tensor:
    """Return normalized D3 reference-state weights for each atom."""
    differences = (
        coordination_numbers[:, None] - reference_coordination_numbers[None, :]
    )
    unnormalized_weights = torch.exp(-4.0 * differences**2)
    return unnormalized_weights / unnormalized_weights.sum(
        dim=1,
        keepdim=True,
    )


def d3_pairwise_c6(
    atomic_numbers: torch.Tensor,
    coordination_numbers: torch.Tensor,
) -> torch.Tensor:
    """Interpolate element-specific pairwise D3 C6 coefficients.

    Parameters
    ----------
    atomic_numbers
        Atomic numbers with shape ``(n_atoms,)``.
    coordination_numbers
        D3 coordination numbers with shape ``(n_atoms,)``.
    Returns
    -------
    torch.Tensor
        Interpolated C6 matrix with shape ``(n_atoms, n_atoms)``. Its
        diagonal is zero because self-interactions are not used in D3.
    """
    if atomic_numbers.ndim != 1:
        raise ValueError("atomic_numbers must have shape (n_atoms,).")
    if atomic_numbers.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise TypeError("atomic_numbers must have an integer dtype.")

    n_atoms = atomic_numbers.shape[0]
    if coordination_numbers.shape != (n_atoms,):
        raise ValueError(
            "coordination_numbers must have shape (n_atoms,), "
            f"but has shape {tuple(coordination_numbers.shape)}."
        )

    reference_coordination_numbers, c6_reference = d3_reference_data()

    max_atomic_number = max(reference_coordination_numbers)
    if torch.any(atomic_numbers < 1) or torch.any(atomic_numbers > max_atomic_number):
        raise ValueError(
            "atomic_numbers must lie between 1 and "
            f"{max_atomic_number}, the range covered by reference.f90."
        )

    elements = [int(atomic_number) for atomic_number in atomic_numbers.tolist()]
    reference_weights: list[torch.Tensor] = []

    for atom_index, element in enumerate(elements):
        if element not in reference_coordination_numbers:
            raise ValueError(f"No reference coordination data for Z={element}.")

        reference_cn = reference_coordination_numbers[element].to(
            device=coordination_numbers.device,
            dtype=coordination_numbers.dtype,
        )
        if reference_cn.ndim != 1 or reference_cn.numel() == 0:
            raise ValueError(
                f"Reference coordination data for Z={element} must be nonempty "
                "and one-dimensional."
            )

        weights = d3_reference_weights(
            coordination_numbers[atom_index : atom_index + 1],
            reference_cn,
        )
        reference_weights.append(weights[0])

    rows: list[torch.Tensor] = []
    zero = torch.zeros(
        (), dtype=coordination_numbers.dtype, device=coordination_numbers.device
    )

    for atom_i, element_i in enumerate(elements):
        row: list[torch.Tensor] = []

        for atom_j, element_j in enumerate(elements):
            if atom_i == atom_j:
                row.append(zero)
                continue

            pair_key = tuple(sorted((element_i, element_j)))
            if pair_key not in c6_reference:
                raise ValueError(f"No C6 reference data for element pair {pair_key}.")

            c6_block = c6_reference[pair_key].to(
                device=coordination_numbers.device,
                dtype=coordination_numbers.dtype,
            )
            weights_i = reference_weights[atom_i]
            weights_j = reference_weights[atom_j]

            # Stored blocks use the sorted element order.
            if element_i > element_j:
                c6_block = c6_block.T

            expected_shape = (weights_i.numel(), weights_j.numel())
            if c6_block.shape != expected_shape:
                raise ValueError(
                    f"C6 block for {pair_key} has shape {tuple(c6_block.shape)}, "
                    f"but expected {expected_shape}."
                )

            pair_weights = weights_i[:, None] * weights_j[None, :]
            valid_reference_pairs = c6_block > 0.0

            if not bool(torch.any(valid_reference_pairs)):
                raise ValueError(f"C6 block for {pair_key} has no valid entries.")

            # Real D3 normalizes over valid reference-state combinations only.
            numerator = (c6_block * pair_weights).sum()
            denominator = pair_weights[valid_reference_pairs].sum()
            row.append(numerator / denominator)

        rows.append(torch.stack(row))

    return torch.stack(rows)


def d3_pairwise_c8(
    c6: torch.Tensor,
    r4r2: torch.Tensor,
) -> torch.Tensor:
    """Construct pairwise D3 C8 coefficients from C6 and atomic factors."""
    n_atoms = c6.shape[0]

    if c6.shape != (n_atoms, n_atoms):
        raise ValueError("c6 must have shape (n_atoms, n_atoms).")

    if r4r2.shape != (n_atoms,):
        raise ValueError("r4r2 must have shape (n_atoms,).")

    return 3.0 * c6 * r4r2[:, None] * r4r2[None, :]


def d3_coordination_numbers(
    coordinates: torch.Tensor,
    atomic_numbers: torch.Tensor,
) -> torch.Tensor:
    """Compute D3 coordination numbers from nuclear coordinates.

    Parameters
    ----------
    coordinates
        Cartesian coordinates with shape ``(n_atoms, 3)``, in bohr.
    atomic_numbers
        Atomic numbers with shape ``(n_atoms,)``.

    Returns
    -------
    torch.Tensor
        Coordination numbers with shape ``(n_atoms,)``.

    Notes
    -----
    Uses the simple-dftd3 coordination-number switching function:

    ``CN_i = sum_{j != i} sigmoid[16 * ((R_cov,i + R_cov,j) / R_ij - 1)]``.
    """
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError(
            "coordinates must have shape (n_atoms, 3), "
            f"but has shape {tuple(coordinates.shape)}."
        )

    covalent_radii = d3_covalent_radii(
        atomic_numbers,
        dtype=coordinates.dtype,
        device=coordinates.device,
    )

    n_atoms = coordinates.shape[0]
    if covalent_radii.shape != (n_atoms,):
        raise ValueError(
            "atomic_numbers must have shape (n_atoms,), "
            f"but has shape {tuple(atomic_numbers.shape)}."
        )

    if torch.any(covalent_radii <= 0):
        raise ValueError("All covalent radii must be positive.")

    distances = pairwise_distances(coordinates)
    pair_mask = ~torch.eye(
        n_atoms,
        dtype=torch.bool,
        device=coordinates.device,
    )

    # Replace the diagonal only before division; self-pairs are excluded below.
    safe_distances = torch.where(pair_mask, distances, torch.ones_like(distances))
    if torch.any(safe_distances[pair_mask] <= 0.0):
        raise ValueError("coordinates must not contain coincident atoms.")
    radii_sums = covalent_radii[:, None] + covalent_radii[None, :]

    switching_argument = 16.0 * (radii_sums / safe_distances - 1.0)
    pair_contributions = torch.sigmoid(switching_argument)

    return torch.where(pair_mask, pair_contributions, torch.zeros_like(distances)).sum(
        dim=1
    )


def pairwise_distances(coordinates: torch.Tensor) -> torch.Tensor:
    """Return the matrix R_ij of interatomic distances."""
    displacements = coordinates[:, None, :] - coordinates[None, :, :]
    return torch.linalg.vector_norm(displacements, dim=-1)


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
    verbose=False,
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
    verbose : bool, optional
        If true, print per-atom scores and the selected coverage summary.

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

        if verbose:
            for atom_index, score in enumerate(atom_scores):
                print(f"Atom {atom_index}: {score:.8f}")

        # Clip tiny negative numerical artifacts from population analyses.
        positive_scores = np.clip(atom_scores, 0.0, None)
        total_score = positive_scores.sum()

        if total_score == 0.0:
            fragment_atoms = []
            if verbose:
                print("No positive embedding-space population found.")
        else:
            order = np.argsort(positive_scores)[::-1]
            cumulative_scores = np.cumsum(positive_scores[order])
            target = coverage * total_score
            n_atoms = np.searchsorted(cumulative_scores, target, side="left") + 1

            fragment_atoms = np.sort(order[:n_atoms]).tolist()

            captured_coverage = cumulative_scores[n_atoms - 1] / total_score
            if verbose:
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
