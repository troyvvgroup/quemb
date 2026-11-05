#!/usr/bin/env python3
"""Scan D3(BJ) a1/a2 damping parameters using a Slurm job array.

Each array task evaluates a disjoint subset of the parameter grid and writes
``d3_damping_scan_<array-task-id>.csv``.  Run without Slurm to evaluate the
entire grid locally and write ``d3_damping_scan_local.csv``.
"""

import csv
import os
from dataclasses import replace
from itertools import product
from pathlib import Path
from quemb.molbe.torch_d3 import (
    D3BJParameters,
    D3TTParameters,
    TorchExternalD3BJ,
)
import numpy as np
import torch
from pyscf import gto

from quemb.molbe.chemfrag import Fragmented
from quemb.molbe.torch_d3 import TorchExternalD3BJ


GEOMETRY_FILE = "geometry.xyz"
FORCE_EMBEDDING_FILE = "force_embedding_C3A.txt"
REFERENCE_FILE = "ref_C3A.txt"
BASIS = "sto-3g"

DAMPING = "tt"  # "bj" or "tt"

A1_VALUES = [0.2, 0.3, 0.4, 0.5, 0.6]
A2_VALUES = [3.0, 4.0, 5.0, 6.0, 7.0]

# Units: bohr^-1. These are trial values; calibrate the range for your system.
TT_B_VALUES = [0.003030, 0.006061, 0.009091, 0.012121, 0.015152, 0.018182, 0.021212, 0.024242, 0.027273,
0.030303, 0.033333, 0.036364, 0.039394, 0.042424, 0.045455, 0.048485, 0.051515, 0.054545, 0.057576,
0.060606, 0.063636, 0.066667, 0.069697, 0.072727, 0.075758, 0.078788, 0.081818, 0.084848, 0.087879,
0.090909, 0.093939, 0.096970, 0.100000, 0.103030, 0.106061, 0.109091, 0.112121, 0.115152, 0.118182,
0.121212, 0.124242, 0.127273, 0.130303, 0.133333, 0.136364, 0.139394, 0.142424, 0.145455, 0.148485,
0.151515, 0.154545, 0.157576, 0.160606, 0.163636, 0.166667, 0.169697, 0.172727, 0.175758, 0.178788,
0.181818, 0.184848, 0.187879, 0.190909, 0.193939, 0.196970, 0.200000, 0.203030, 0.206061, 0.209091,
0.212121, 0.215152, 0.218182, 0.221212, 0.224242, 0.227273, 0.230303, 0.233333, 0.236364, 0.239394,
0.242424, 0.245455, 0.248485, 0.251515, 0.254545, 0.257576, 0.260606, 0.263636, 0.266667, 0.269697,
0.272727, 0.275758, 0.278788, 0.281818, 0.284848, 0.287879, 0.290909, 0.293939, 0.296970, 0.300000]


def load_gradient(filename: str, n_atoms: int) -> torch.Tensor:
    """Load a Cartesian gradient and validate its dimensions."""
    gradient = torch.tensor(np.loadtxt(filename), dtype=torch.float64)
    if gradient.shape != (n_atoms, 3):
        raise ValueError(f"{filename} must have shape ({n_atoms}, 3).")
    return gradient


def build_system():
    """Build the molecule and the fragment data used for the D3 correction."""
    mol = gto.M(atom=Path(GEOMETRY_FILE).read_text(), basis=BASIS)
    fragmented = Fragmented.from_mole(mol, n_BE=2, h_treatment="treat_H_diff")
    fragments = fragmented.get_atom_in_frag()
    frag_per_atom = fragmented.get_frag_per_atom()
    inverted_fragments = [
        [atom for atom in range(mol.natm) if atom not in fragment]
        for fragment in fragments
    ]
    return mol, fragments, inverted_fragments, frag_per_atom


def d3_gradient(
    coordinates: torch.Tensor,
    d3: TorchExternalD3BJ,
    fragments,
    frag_per_atom,
    parameters,
) -> torch.Tensor:
    """Assemble the force-embedding D3 gradient from owned fragment atoms."""
    fragment_gradients = d3.fragment_gradients(
        coordinates,
        parameters=parameters,
        create_graph=False,
    )

    correction = torch.zeros_like(coordinates)
    frag_per_atom = torch.as_tensor(
        frag_per_atom, device=coordinates.device, dtype=torch.long
    )

    for fragment_number, (fragment, gradient) in enumerate(
        zip(fragments, fragment_gradients)
    ):
        fragment = torch.as_tensor(
            fragment, device=coordinates.device, dtype=torch.long
        )
        owned_mask = frag_per_atom[fragment] == fragment_number
        if owned_mask.any():
            correction.index_copy_(0, fragment[owned_mask], gradient[owned_mask])

    return correction


def mse(
    coordinates: torch.Tensor,
    force_embedding: torch.Tensor,
    reference: torch.Tensor,
    d3: TorchExternalD3BJ,
    fragments,
    frag_per_atom,
    parameters,
) -> float:
    """Return the mean-squared error for one D3(BJ) parameter combination."""
    residual = force_embedding + d3_gradient(
        coordinates, d3, fragments, frag_per_atom, parameters
    ) - reference
    return torch.mean(residual.square()).item()


def assigned_points(all_points: list[tuple[float, float]]):
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
    damping: str,
) -> Path:
    filename = Path(f"d3_{damping}_damping_scan_{task_label}.csv")

    fieldnames = (
        ["a1", "a2", "mse"]
        if damping == "bj"
        else ["b", "mse"]
    )

    with filename.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return filename


def main() -> None:
    # One process per array task: prevent PyTorch from creating extra CPU threads.
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))

    mol, fragments, inverted_fragments, frag_per_atom = build_system()
    coordinates = torch.tensor(mol.atom_coords(), dtype=torch.float64)
    force_embedding = load_gradient(FORCE_EMBEDDING_FILE, mol.natm)
    reference = load_gradient(REFERENCE_FILE, mol.natm)


    bj_default = D3BJParameters.from_method("hf")
    
    if DAMPING == "bj":
        d3 = TorchExternalD3BJ(
            atomic_numbers=mol.atom_charges().tolist(),
            fragments=fragments,
            inverted_fragments=inverted_fragments,
            method="hf",
            damping="bj",
        )
        default = d3.total_d3.parameters
        all_points = list(product(A1_VALUES, A2_VALUES))
    
    elif DAMPING == "tt":
        # This borrows HF-D3(BJ)'s s6/s8 as initial fixed scale factors.
        # For a production TT model, consider fitting them with b.
        default = D3TTParameters(
            s6=bj_default.s6,
            s8=bj_default.s8,
            b=TT_B_VALUES[0],
        )
        d3 = TorchExternalD3BJ(
            atomic_numbers=mol.atom_charges().tolist(),
            fragments=fragments,
            inverted_fragments=inverted_fragments,
            method="hf",
            damping="tt",
            tt_parameters=default,
        )
        all_points = [(b,) for b in TT_B_VALUES]

    else:
        raise ValueError(f"Unsupported DAMPING={DAMPING!r}.")


    task_id, task_count, points = assigned_points(all_points)
    task_label = os.environ.get("SLURM_ARRAY_TASK_ID", "local")

    print(
        f"Array task {task_id + 1}/{task_count}: evaluating {len(points)} "
        f"of {len(all_points)} points with default "
        f"s6={default.s6:.4f} and s8={default.s8:.4f}.",
        flush=True,
    )

    
    rows = []

    for point in points:
        if DAMPING == "bj":
            a1, a2 = point
            parameters = replace(default, a1=a1, a2=a2)
            row = {"a1": a1, "a2": a2}
            label = f"a1={a1:.4f}  a2={a2:.4f}"

        else:
            (b,) = point
            parameters = replace(default, b=b)
            row = {"b": b}
            label = f"b={b:.4f} bohr^-1"

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
    
    results_file = write_results(rows, task_label, DAMPING)
    print(f"Wrote {results_file}.", flush=True)


if __name__ == "__main__":
    main()
