#!/usr/bin/env python3
"""Scan D3(BJ) a1/a2 damping parameters using a Slurm job array.

Each array task evaluates a disjoint subset of the parameter grid and writes
``d3_damping_scan_<array-task-id>.csv``.  Run without Slurm to evaluate the
entire grid locally and write ``d3_damping_scan_local.csv``.
"""

import csv
import os
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from pyscf import gto

from quemb.molbe.chemfrag import Fragmented
from quemb.molbe.torch_d3 import (
    BeckeJohnsonDamping,
    D3BJParameters,
    D3Parameters,
    D3TTParameters,
    DampingModel,
    TangToenniesDamping,
    TorchExternalD3,
)

GEOMETRY_FILE = Path("geometry.xyz")
FORCE_EMBEDDING_FILE = Path("force_embedding_C3A.txt")
REFERENCE_FILE = Path("ref_C3A.txt")
BASIS = "sto-3g"

METHOD = "hf"
DAMPING = "tt"

A1_VALUES = [0.2, 0.3, 0.4, 0.5, 0.6]
A2_VALUES = [3.0, 4.0, 5.0, 6.0, 7.0]

# Units: bohr^-1.
TT_B_VALUES = np.linspace(0.0001, 0.0003, 100)

Point = tuple[float, ...]
AtomIndices = Sequence[int]


@dataclass(frozen=True)
class ScanConfiguration:
    """Everything that varies between damping-model scans."""

    name: str
    damping: DampingModel
    default_parameters: D3Parameters
    parameter_names: tuple[str, ...]
    parameter_units: tuple[str, ...]
    points: tuple[Point, ...]

    def parameters_for(self, point: Point) -> D3Parameters:
        """Construct the D3 parameters represented by one scan point."""
        if len(point) != len(self.parameter_names):
            raise ValueError(
                f"Expected {len(self.parameter_names)} scan values, got {len(point)}."
            )

        # ``dataclasses.replace`` cannot type-check a dynamically constructed
        # ``**dict`` against a union of two dataclass types.  Narrow the
        # parameter type first, then name its fields explicitly.
        if isinstance(self.default_parameters, D3BJParameters):
            if self.parameter_names != ("a1", "a2"):
                raise ValueError("D3(BJ) scans must vary a1 and a2.")
            a1, a2 = point
            return replace(self.default_parameters, a1=a1, a2=a2)

        if isinstance(self.default_parameters, D3TTParameters):
            if self.parameter_names != ("b",):
                raise ValueError("Tang-Toennies scans must vary b.")
            (b,) = point
            return replace(self.default_parameters, b=b)

        raise TypeError("Unsupported D3 parameter type.")

    def row_for(self, point: Point) -> dict[str, float]:
        """Return the CSV columns for one scan point, excluding MSE."""
        return dict(zip(self.parameter_names, point, strict=True))

    def label_for(self, point: Point) -> str:
        """Return a readable progress label for one scan point."""
        fields = []

        for name, value, unit in zip(
            self.parameter_names,
            point,
            self.parameter_units,
            strict=True,
        ):
            suffix = f" {unit}" if unit else ""
            fields.append(f"{name}={value:.6g}{suffix}")

        return "  ".join(fields)


def make_bj_scan(method: str) -> ScanConfiguration:
    damping = BeckeJohnsonDamping()
    default_parameters = damping.default_parameters(method)

    return ScanConfiguration(
        name="bj",
        damping=damping,
        default_parameters=default_parameters,
        parameter_names=("a1", "a2"),
        parameter_units=("", ""),
        points=tuple(product(A1_VALUES, A2_VALUES)),
    )


def make_tt_scan(method: str) -> ScanConfiguration:
    damping = TangToenniesDamping()

    # Temporary choice: reuse the HF-D3(BJ) scale factors.
    # Replace this when you establish TT-specific fitted defaults.
    bj_defaults = D3BJParameters.from_method(method)
    default_parameters = D3TTParameters(
        s6=bj_defaults.s6,
        s8=bj_defaults.s8,
        b=float(TT_B_VALUES[0]),
    )

    return ScanConfiguration(
        name="tt",
        damping=damping,
        default_parameters=default_parameters,
        parameter_names=("b",),
        parameter_units=("",),
        points=tuple((float(b),) for b in TT_B_VALUES),
    )


SCAN_CONFIG_FACTORIES: dict[
    str,
    Callable[[str], ScanConfiguration],
] = {
    "bj": make_bj_scan,
    "tt": make_tt_scan,
}


def get_scan_configuration(name: str, method: str) -> ScanConfiguration:
    """Construct the selected damping-model scan configuration."""
    try:
        factory = SCAN_CONFIG_FACTORIES[name.lower()]
    except KeyError as exc:
        available = ", ".join(sorted(SCAN_CONFIG_FACTORIES))
        raise ValueError(
            f"Unknown damping model {name!r}; choose one of: {available}."
        ) from exc

    return factory(method)


def load_gradient(path: Path, n_atoms: int) -> torch.Tensor:
    """Load a Cartesian gradient and validate its dimensions."""
    gradient = torch.tensor(np.loadtxt(path), dtype=torch.float64)
    if gradient.shape != (n_atoms, 3):
        raise ValueError(f"{path} must have shape ({n_atoms}, 3).")
    return gradient


def build_system() -> tuple[
    gto.Mole,
    list[list[int]],
    list[list[int]],
    list[int],
]:
    """Build the molecule and the fragment data used for the D3 correction."""
    mol: gto.Mole = gto.M(atom=GEOMETRY_FILE.read_text(), basis=BASIS)
    fragmented = Fragmented.from_mole(mol, n_BE=2, h_treatment="treat_H_diff")
    # QuEmb exposes distinct ``AtomIdx`` and ``FragmentIdx`` types.  Convert
    # them at this boundary because ``TorchExternalD3`` deliberately accepts
    # ordinary mutable ``list[list[int]]`` fragment definitions.
    fragments = [
        [int(atom_index) for atom_index in fragment]
        for fragment in fragmented.get_atom_in_frag()
    ]
    frag_per_atom = [
        int(fragment_index) for fragment_index in fragmented.get_frag_per_atom()
    ]
    inverted_fragments = [
        [atom for atom in range(mol.natm) if atom not in fragment]
        for fragment in fragments
    ]
    return mol, fragments, inverted_fragments, frag_per_atom


def d3_gradient(
    coordinates: torch.Tensor,
    d3: TorchExternalD3,
    fragments: Sequence[AtomIndices],
    frag_per_atom: Sequence[int],
    parameters: D3Parameters,
) -> torch.Tensor:
    """Assemble the force-embedding D3 gradient from owned fragment atoms."""
    fragment_gradients = d3.fragment_gradients(
        coordinates,
        parameters=parameters,
        create_graph=False,
    )

    correction = torch.zeros_like(coordinates)
    frag_per_atom_tensor = torch.as_tensor(
        frag_per_atom, device=coordinates.device, dtype=torch.long
    )

    for fragment_number, (fragment_atoms, gradient) in enumerate(
        zip(fragments, fragment_gradients, strict=True)
    ):
        fragment_indices = torch.as_tensor(
            fragment_atoms,
            device=coordinates.device,
            dtype=torch.long,
        )
        owned_mask = frag_per_atom_tensor[fragment_indices] == fragment_number

        if owned_mask.any():
            correction.index_copy_(
                0,
                fragment_indices[owned_mask],
                gradient[owned_mask],
            )

    return correction


def mse(
    coordinates: torch.Tensor,
    force_embedding: torch.Tensor,
    reference: torch.Tensor,
    d3: TorchExternalD3,
    fragments: Sequence[AtomIndices],
    frag_per_atom: Sequence[int],
    parameters: D3Parameters,
) -> float:
    """Return the mean-squared error for one D3(BJ) parameter combination."""
    residual = (
        force_embedding
        + d3_gradient(coordinates, d3, fragments, frag_per_atom, parameters)
        - reference
    )
    return torch.mean(residual.square()).item()


def assigned_points(
    all_points: tuple[Point, ...],
) -> tuple[int, int, tuple[Point, ...]]:
    """Return this Slurm array task's round-robin share of the grid."""
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    task_count = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1"))

    if not 0 <= task_id < task_count:
        raise ValueError(
            f"Invalid Slurm array task {task_id} for {task_count} array tasks."
        )

    return task_id, task_count, all_points[task_id::task_count]


def write_results(
    rows: list[dict[str, float]],
    task_label: str,
    config: ScanConfiguration,
) -> Path:
    filename = Path(f"d3_{config.name}_damping_scan_{task_label}.csv")
    fieldnames = [*config.parameter_names, "mse"]

    with filename.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return filename

def write_tt_damping_curves(
    points: tuple[Point, ...],
    task_label: str,
    config: ScanConfiguration,
) -> Path:
    """Write f6 and f8 versus the reduced TT distance for each scanned b."""
    if config.name != "tt":
        raise ValueError("TT damping curves are only defined for the TT scan.")

    filename = Path(f"d3_tt_damping_function_{task_label}.csv")

    # x is the dimensionless argument used by TangToenniesDamping._factor.
    x_values = torch.linspace(0.0, 10.0, 1000, dtype=torch.float64)

    with filename.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["b", "reduced_distance", "distance_over_reference", "f6", "f8"],
        )
        writer.writeheader()

        for point in points:
            parameters = config.parameters_for(point)
            assert isinstance(parameters, D3TTParameters)

            # In the energy code: x = b * R / R_ref.
            f6 = TangToenniesDamping._factor(x_values, order=6)
            f8 = TangToenniesDamping._factor(x_values, order=8)
            distance_over_reference = x_values / parameters.b

            for x, r_scaled, value6, value8 in zip(
                x_values.tolist(),
                distance_over_reference.tolist(),
                f6.tolist(),
                f8.tolist(),
                strict=True,
            ):
                writer.writerow(
                    {
                        "b": parameters.b,
                        "reduced_distance": x,
                        "distance_over_reference": r_scaled,
                        "f6": value6,
                        "f8": value8,
                    }
                )

    return filename

def main() -> None:
    # One process per array task: prevent PyTorch from creating extra CPU threads.
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))

    mol, fragments, inverted_fragments, frag_per_atom = build_system()
    coordinates = torch.tensor(mol.atom_coords(), dtype=torch.float64)
    force_embedding = load_gradient(FORCE_EMBEDDING_FILE, mol.natm)
    reference = load_gradient(REFERENCE_FILE, mol.natm)

    config = get_scan_configuration(DAMPING, METHOD)

    d3 = TorchExternalD3(
        atomic_numbers=mol.atom_charges().tolist(),
        fragments=fragments,
        inverted_fragments=inverted_fragments,
        method=METHOD,
        damping=config.damping,
        parameters=config.default_parameters,
    )

    task_id, task_count, points = assigned_points(config.points)
    task_label = os.environ.get("SLURM_ARRAY_TASK_ID", "local")

    print(
        f"Array task {task_id + 1}/{task_count}: evaluating {len(points)} "
        f"of {len(config.points)} points with default "
        f"s6={config.default_parameters.s6:.4f} and "
        f"s8={config.default_parameters.s8:.4f}.",
        flush=True,
    )

    rows: list[dict[str, float]] = []

    for point in points:
        parameters = config.parameters_for(point)
        row = config.row_for(point)
        label = config.label_for(point)

        value = mse(
            coordinates,
            force_embedding,
            reference,
            d3,
            fragments,
            frag_per_atom,
            parameters,
        )

        row["mse"] = value
        rows.append(row)
        print(f"{label}  MSE={value:.8e}", flush=True)

    results_file = write_results(rows, task_label, config)
    print(f"Wrote {results_file}.", flush=True)

    if config.name == "tt":
        damping_file = write_tt_damping_curves(points, task_label, config)
        print(f"Wrote {damping_file}.", flush=True)


if __name__ == "__main__":
    main()
