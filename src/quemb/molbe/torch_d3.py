import re
from dataclasses import dataclass
from functools import cache
from importlib.resources import files

import numpy as np
import tomli
import torch

ANGSTROM_TO_BOHR = 1.8897261254578281


@dataclass(frozen=True)
class D3ReferenceData:
    coordination_numbers: dict[int, torch.Tensor]
    c6_coefficients: dict[tuple[int, int], torch.Tensor]
    covalent_radii: torch.Tensor
    r4r2_factors: torch.Tensor

    @staticmethod
    def reference_weights(
        coordination_numbers: torch.Tensor,
        reference_coordination_numbers: torch.Tensor,
    ) -> torch.Tensor:
        """Return normalized weights for D3 reference coordination states."""
        differences = (
            coordination_numbers[:, None] - reference_coordination_numbers[None, :]
        )
        unnormalized_weights = torch.exp(-4.0 * differences**2)
        return unnormalized_weights / unnormalized_weights.sum(
            dim=1,
            keepdim=True,
        )

    def covalent_radii_for(
        self,
        atomic_numbers: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Return D3 covalent radii for ``atomic_numbers``, in bohr."""
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

        radius_table = self.covalent_radii.to(dtype=dtype, device=device)
        return radius_table[atomic_numbers.to(device=device)]

    def r4r2_for(
        self,
        atomic_numbers: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Return D3 C8 factors for ``atomic_numbers``."""
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

        table = self.r4r2_factors.to(dtype=dtype, device=device)
        return table[atomic_numbers.to(device=device)]

    @classmethod
    @cache
    def from_files(cls) -> "D3ReferenceData":
        """Read reference coordination numbers and C6 data from reference.f90."""
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

        covalent_radii = cls._read_covalent_radii()
        r4r2_factors = cls._read_r4r2_factors()

        return cls(
            coordination_numbers=reference_coordination_numbers,
            c6_coefficients=c6_reference,
            covalent_radii=covalent_radii,
            r4r2_factors=r4r2_factors,
        )

    @staticmethod
    def _read_covalent_radii() -> torch.Tensor:
        """Read D3 covalent radii from covrad.f90, in bohr."""
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

        return (
            torch.tensor([0.0, *radii_angstrom], dtype=torch.float64)
            * (4.0 / 3.0)
            * ANGSTROM_TO_BOHR
        )

    @staticmethod
    def _read_r4r2_factors() -> torch.Tensor:
        """Read simple-dftd3 C8 factors from r4r2.f90."""
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
                "Expected 118 r4_over_r2 values in r4r2.f90, "
                f"but found {len(raw_r4r2)}."
            )

        atomic_numbers = torch.arange(1, 119, dtype=torch.float64)
        factors = torch.sqrt(
            0.5
            * torch.tensor(raw_r4r2, dtype=torch.float64)
            * torch.sqrt(atomic_numbers)
        )

        return torch.cat((torch.zeros(1, dtype=torch.float64), factors))


@dataclass(frozen=True)
class D3BJParameters:
    s6: float
    s8: float
    a1: float
    a2: float

    @classmethod
    @cache
    def from_method(cls, method: str) -> "D3BJParameters":
        source = files("quemb.molbe").joinpath("data", "parameters.toml").read_text()
        data = tomli.loads(source)

        try:
            defaults = data["default"]["parameter"]["d3"]["bj"]
            method_parameters = data["parameter"][method.lower()]["d3"]["bj"]
        except KeyError as exc:
            raise ValueError(
                f"No D3(BJ) parameters are available for method {method!r}."
            ) from exc

        values = {**defaults, **method_parameters}
        return cls(
            s6=float(values["s6"]),
            s8=float(values["s8"]),
            a1=float(values["a1"]),
            a2=float(values["a2"]),
        )


class TorchD3BJ:
    """Differentiable two-body D3(BJ) correction."""

    def __init__(self, atomic_numbers: list[int], method: str = "hf"):
        self.atomic_numbers = torch.tensor(atomic_numbers, dtype=torch.int64)
        self.method = method
        self.parameters = D3BJParameters.from_method(method)
        self.reference_data = D3ReferenceData.from_files()

    def energy(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Compute the two-body D3(BJ) dispersion energy for ``method``.

        Coordinates must be expressed in bohr. The D3(BJ) damping parameters are
        loaded from the bundled simple-dftd3-compatible ``parameters.toml`` file.
        """
        self._validate_coordinates(coordinates)
        c6, c8 = self._pairwise_coefficients(coordinates)

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

        pair_energies, _ = self._two_body_pair_energies(
            coordinates,
            c6,
            c8,
        )
        return pair_energies.sum()

    def energy_and_gradient(
        self,
        coordinates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not coordinates.requires_grad:
            coordinates = coordinates.detach().clone().requires_grad_(True)

        energy = self.energy(coordinates)
        gradient = torch.autograd.grad(energy, coordinates)[0]
        return energy, gradient

    def _two_body_pair_energies(
        self,
        coordinates: torch.Tensor,
        c6: torch.Tensor,
        c8: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return D3(BJ) pair-energy contributions and atom-pair indices."""
        distances = self._pairwise_distances(coordinates)

        pair_indices = torch.triu_indices(
            coordinates.shape[0],
            coordinates.shape[0],
            offset=1,
            device=coordinates.device,
        )
        atom_i, atom_j = pair_indices

        pair_distances = distances[atom_i, atom_j]
        pair_c6 = c6[atom_i, atom_j]
        pair_c8 = c8[atom_i, atom_j]

        r0 = torch.sqrt(pair_c8 / pair_c6)
        damping_radius = self.parameters.a1 * r0 + self.parameters.a2

        c6_term = self.parameters.s6 * pair_c6 / (pair_distances**6 + damping_radius**6)
        c8_term = self.parameters.s8 * pair_c8 / (pair_distances**8 + damping_radius**8)

        pair_energies = -(c6_term + c8_term)
        return pair_energies, pair_indices

    def _coordination_numbers(
        self,
        coordinates: torch.Tensor,
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

        atomic_numbers = self.atomic_numbers.to(coordinates.device)
        covalent_radii = self.reference_data.covalent_radii_for(
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

        distances = self._pairwise_distances(coordinates)

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

        return torch.where(
            pair_mask, pair_contributions, torch.zeros_like(distances)
        ).sum(dim=1)

    def _pairwise_c6(
        self,
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
        atomic_numbers = self.atomic_numbers.to(coordination_numbers.device)

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

        reference_data = self.reference_data
        reference_coordination_numbers = self.reference_data.coordination_numbers
        c6_reference = reference_data.c6_coefficients

        max_atomic_number = max(reference_coordination_numbers)
        if torch.any(atomic_numbers < 1) or torch.any(
            atomic_numbers > max_atomic_number
        ):
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

            weights = D3ReferenceData.reference_weights(
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
                    raise ValueError(
                        f"No C6 reference data for element pair {pair_key}."
                    )

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

    @staticmethod
    def _pairwise_c8(
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

    @staticmethod
    def _pairwise_distances(coordinates: torch.Tensor) -> torch.Tensor:
        """Return the matrix R_ij of interatomic distances."""
        displacements = coordinates[:, None, :] - coordinates[None, :, :]
        return torch.linalg.vector_norm(displacements, dim=-1)

    @staticmethod
    def _validate_coordinates(coordinates: torch.Tensor) -> None:
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise ValueError(
                "coordinates must have shape (n_atoms, 3), "
                f"but has shape {tuple(coordinates.shape)}."
            )

    def _pairwise_coefficients(
        self,
        coordinates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        atomic_numbers = self.atomic_numbers.to(coordinates.device)

        coordination_numbers = self._coordination_numbers(coordinates)
        c6 = self._pairwise_c6(coordination_numbers)
        r4r2 = self.reference_data.r4r2_for(
            atomic_numbers,
            dtype=coordinates.dtype,
            device=coordinates.device,
        )
        c8 = self._pairwise_c8(c6, r4r2)
        return c6, c8
