#!/usr/bin/env python3
"""Scan D3 damping parameters using a Slurm job array.
Each array task evaluates a disjoint subset of the parameter grid and writes
``d3_<model>_damping_scan_<array-task-id>.csv``. Run without Slurm to
evaluate the entire grid locally and write a corresponding ``*_local.csv``.
Set ``BJ_SWITCH = "r"`` to fit the existing distance-dependent Johnson sine
factor, or ``BJ_SWITCH = "population"`` to use the population factor from
the slide.  Both choices multiply a D3(BJ) pair energy, but only the
distance-dependent version has the additional parameter ``d``.
"""

import csv
import os
import re
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
    BeckeJohnsonSineDamping,
    D3BJParameters,
    D3BJSineParameters,
    D3JohnsonSineParameters,
    D3Parameters,
    D3TTParameters,
    DampingModel,
    JohnsonSineDamping,
    TangToenniesDamping,
    TorchExternalD3,
)

# "r": D3(BJ) times the existing Johnson sine factor f(R_AB / R0_AB).
BJ_SWITCH = "population"
A1_VALUES = [0.2, 0.3, 0.4, 0.5, 0.6]
A2_VALUES = [5.0, 6.0, 7.0, 8.0, 9.0]
# Dimensionless coefficient in x = b * R_AB / R0_AB.
TT_B_VALUES = np.linspace(0.0001, 0.0003, 100)
# Dimensionless onset in x = R_AB / (r_cov,A + r_cov,B).  The sine transition
# runs from x = d to x = d + 2.
SINE_D_VALUES = np.linspace(0.01, 2.0, 100)
Point = tuple[float, ...]
AtomIndices = Sequence[int]


def rmse_for_both_signs(
    coordinates: torch.Tensor,
    be_gradient: torch.Tensor,
    reference_gradient: torch.Tensor,
    d3: TorchExternalD3,
    fragments: Sequence[AtomIndices],
    frag_per_atom: Sequence[int],
    parameters: D3Parameters,
) -> tuple[float, float]:
    """Return RMSEs for BE - D3 and BE + D3 corrections, respectively."""
    correction = d3_gradient(
        coordinates,
        d3,
        fragments,
        frag_per_atom,
        parameters,
    )

    rmse_minus = torch.sqrt(
        torch.mean((be_gradient - correction - reference_gradient).square())
    ).item()

    rmse_plus = torch.sqrt(
        torch.mean((be_gradient + correction - reference_gradient).square())
    ).item()

    return rmse_minus, rmse_plus


def atom_populations_from_ta(mol: gto.Mole, ta: np.ndarray) -> torch.Tensor:
    """Return atom populations from an S-orthonormal TA matrix.
    If TA satisfies TA^† S TA = I in the nonorthogonal AO basis, first form
    T_orth = S^(1/2) TA. The AO-site populations are then
        p_mu = sum_k |T_orth[mu, k]|^2,
    which lie in [0, 1]. D3 is atom-pair based, so each atom's population is
    the mean of the populations of its AO sites.
    """
    ta = np.asarray(ta, dtype=np.complex128)
    nao = mol.nao_nr()
    if ta.ndim != 2 or ta.shape[0] != nao:
        raise ValueError(
            "Each fragment TA must have shape "
            "(mol.nao_nr(), n_embedding_orbitals); "
            f"got {ta.shape}, expected first dimension {nao}."
        )
    overlap = mol.intor("int1e_ovlp")
    overlap = 0.5 * (overlap + overlap.T.conj())
    eigenvalues, eigenvectors = np.linalg.eigh(overlap)
    overlap_tolerance = 1.0e-12
    if np.any(eigenvalues <= overlap_tolerance):
        raise ValueError(
            "The AO overlap matrix is singular or nearly singular; "
            "cannot construct S^(1/2)."
        )
    sqrt_overlap = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T.conj()
    # Optional but useful: verify that the supplied matrix is S-orthonormal.
    metric_overlap = ta.T.conj() @ overlap @ ta
    identity = np.eye(ta.shape[1], dtype=metric_overlap.dtype)
    if not np.allclose(metric_overlap, identity, rtol=1.0e-8, atol=1.0e-10):
        max_error = np.max(np.abs(metric_overlap - identity))
        raise ValueError(
            "TA is not S-orthonormal: TA^† S TA differs from I by up to "
            f"{max_error:.3e}."
        )
    ta_orth = sqrt_overlap @ ta
    p_ao = np.einsum("ik,ik->i", ta_orth.conj(), ta_orth).real
    tolerance = 1.0e-10
    if np.any(p_ao < -tolerance) or np.any(p_ao > 1.0 + tolerance):
        raise ValueError("Löwdin-orthogonalized TA row norms must lie in [0, 1].")
    p_ao = np.clip(p_ao, 0.0, 1.0)
    # PySCF returns AO start/stop indices in columns 2 and 3.
    p_atom = np.array(
        [
            p_ao[ao_start:ao_stop].mean()
            for _, _, ao_start, ao_stop in mol.aoslice_by_atom()
        ],
        dtype=float,
    )
    return torch.tensor(p_atom, dtype=torch.float64)


def atom_populations_from_saved_tas(
    mol: gto.Mole,
    ta_directory: str | Path,
    n_fragments: int,
) -> list[torch.Tensor]:
    """Build one full-system population vector from each saved fragment TA."""
    ta_directory = Path(ta_directory)

    def fragment_index(path: Path) -> int:
        match = re.fullmatch(r"frag(\d+)_TA\.txt", path.name)
        if match is None:
            raise ValueError(f"Unexpected TA filename: {path.name}")
        return int(match.group(1))

    ta_paths = sorted(
        ta_directory.glob("frag*_TA.txt"),
        key=fragment_index,
    )

    if len(ta_paths) != n_fragments:
        raise ValueError(
            f"Found {len(ta_paths)} TA files in {ta_directory}, "
            f"but expected {n_fragments}."
        )
    expected_names = [f"frag{i}_TA.txt" for i in range(n_fragments)]
    actual_names = [path.name for path in ta_paths]
    if actual_names != expected_names:
        raise ValueError(
            "TA files must be consecutively numbered from "
            f"frag0_TA.txt through frag{n_fragments - 1}_TA.txt. "
            f"Found: {actual_names}"
        )
    return [
        atom_populations_from_ta(mol, np.loadtxt(ta_path, ndmin=2))
        for ta_path in ta_paths
    ]


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
        # ``**dict`` against a union of dataclass types.  Narrow the
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
        if isinstance(self.default_parameters, D3JohnsonSineParameters):
            if self.parameter_names != ("d",):
                raise ValueError("Johnson sine scans must vary d.")
            (d,) = point
            return replace(self.default_parameters, d=d)
        if isinstance(self.default_parameters, D3BJSineParameters):
            if self.parameter_names != ("a1", "a2", "d"):
                raise ValueError("D3(BJ+sine) scans must vary a1, a2, and d.")
            a1, a2, d = point
            return replace(self.default_parameters, a1=a1, a2=a2, d=d)
        raise TypeError("Unsupported D3 parameter type.")

    def row_for(self, point: Point) -> dict[str, float]:
        """Return the CSV columns for one scan point, excluding RMSE."""
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


def make_sine_scan(method: str) -> ScanConfiguration:
    """Scan the onset of covalent-radius-scaled Johnson sine damping."""
    damping = JohnsonSineDamping()
    # Temporary choice: reuse the HF-D3(BJ) scale factors.
    # Replace this when you establish sine-specific fitted defaults.
    bj_defaults = D3BJParameters.from_method(method)
    default_parameters = D3JohnsonSineParameters(
        s6=bj_defaults.s6,
        s8=bj_defaults.s8,
        d=float(SINE_D_VALUES[0]),
    )
    return ScanConfiguration(
        name="sine",
        damping=damping,
        default_parameters=default_parameters,
        parameter_names=("d",),
        parameter_units=("",),
        points=tuple((float(d),) for d in SINE_D_VALUES),
    )


def make_bj_sine_scan(method: str) -> ScanConfiguration:
    """Scan the custom D3(BJ+sine) damping model.
    The Johnson sine factor is applied multiplicatively to the D3(BJ) pair
    energy, so it supplies extra short-range damping without double-counting
    the asymptotic dispersion energy.
    """
    damping = BeckeJohnsonSineDamping()
    bj_defaults = D3BJParameters.from_method(method)
    default_parameters = D3BJSineParameters(
        s6=bj_defaults.s6,
        s8=bj_defaults.s8,
        a1=float(A1_VALUES[0]),
        a2=float(A2_VALUES[0]),
        d=float(SINE_D_VALUES[0]),
    )
    return ScanConfiguration(
        name="bj_sine",
        damping=damping,
        default_parameters=default_parameters,
        parameter_names=("a1", "a2", "d"),
        parameter_units=("", "", ""),
        points=tuple(
            (float(a1), float(a2), float(d))
            for a1, a2, d in product(A1_VALUES, A2_VALUES, SINE_D_VALUES)
        ),
    )


SCAN_CONFIG_FACTORIES: dict[
    str,
    Callable[[str], ScanConfiguration],
] = {
    "bj": make_bj_scan,
    "tt": make_tt_scan,
    "sine": make_sine_scan,
    "johnson_sine": make_sine_scan,
    "bj_sine": make_bj_sine_scan,
    "becke_johnson_sine": make_bj_sine_scan,
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


def build_system(
    xyz_file: str | Path,
) -> tuple[
    gto.Mole,
    list[list[int]],
    list[list[int]],
    list[int],
]:
    xyz_file = Path(xyz_file)
    lines = xyz_file.read_text().splitlines()
    try:
        natoms = int(lines[0].strip())
    except (IndexError, ValueError) as exc:
        raise ValueError(f"{xyz_file} is not a valid XYZ file.") from exc
    geometry = "\n".join(lines[2 : 2 + natoms])
    if len(geometry.splitlines()) != natoms:
        raise ValueError(
            f"{xyz_file} declares {natoms} atoms but not enough coordinates."
        )
    mol = gto.M(
        atom=geometry,
        basis="sto-3g",
        unit="Angstrom",  # standard XYZ-coordinate convention
    )
    fragmented = Fragmented.from_mole(
        mol,
        n_BE=2,
        h_treatment="treat_H_diff",
    )
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
    fieldnames = [*config.parameter_names, "rmse_minus_d3", "rmse_plus_d3"]
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
    mol, fragments, inverted_fragments, frag_per_atom = build_system("C2C2PD.xyz")
    print("Done building system", flush=True)
    coordinates = torch.tensor(mol.atom_coords(), dtype=torch.float64)
    be_gradient = load_gradient(Path("be_gradient_C2C2PD.txt"), mol.natm)
    reference_gradient = load_gradient(
        Path("reference_gradient_C2C2PD.txt"),
        mol.natm,
    )
    uncorrected_rmse = torch.sqrt(
        torch.mean((be_gradient - reference_gradient).square())
    ).item()
    print(f"Uncorrected RMSE={uncorrected_rmse:.8e}", flush=True)
    if BJ_SWITCH == "r":
        # Fit a1, a2, and the distance-based sine onset d.
        config = get_scan_configuration("bj_sine", "hf")
        fragment_atom_populations = None
    elif BJ_SWITCH == "population":
        print("Doing Johnson sine damping based on population", flush=True)
        # Fit a1 and a2.  The population factor has no adjustable d parameter.
        config = get_scan_configuration("bj", "hf")
        fragment_atom_populations = atom_populations_from_saved_tas(
            mol,
            ta_directory="TA_matrices/C2C2PD",
            n_fragments=len(fragments),
        )

    else:
        raise ValueError("BJ_SWITCH must be either 'r' or 'population'.")
    d3 = TorchExternalD3(
        atomic_numbers=mol.atom_charges().tolist(),
        fragments=fragments,
        inverted_fragments=inverted_fragments,
        method="hf",
        damping=config.damping,
        parameters=config.default_parameters,
        fragment_atom_populations=fragment_atom_populations,
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

        rmse_minus, rmse_plus = rmse_for_both_signs(
            coordinates,
            be_gradient,
            reference_gradient,
            d3,
            fragments,
            frag_per_atom,
            parameters,
        )

        row["rmse_minus_d3"] = rmse_minus
        row["rmse_plus_d3"] = rmse_plus

        print(
            f"{label}  RMSE(BE - D3)={rmse_minus:.8e}  RMSE(BE + D3)={rmse_plus:.8e}",
            flush=True,
        )

        rows.append(row)
    results_file = write_results(rows, task_label, config)
    print(f"Wrote {results_file}.", flush=True)
    if config.name == "tt":
        damping_file = write_tt_damping_curves(points, task_label, config)
        print(f"Wrote {damping_file}.", flush=True)


if __name__ == "__main__":
    main()
