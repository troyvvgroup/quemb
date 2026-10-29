import numpy as np
import torch
from pyscf import gto

from quemb.molbe.chemfrag import Fragmented
from quemb.molbe.torch_d3 import TorchExternalD3BJ, TrainableD3BJParameters

GEOMETRY = """
    C -2.230000 -0.152000 2.591000
    C -0.950000 -0.739000 2.643000
    C 0.131000 0.153000 2.667000
    C -1.196000 1.917000 2.639000
    C 0.791000 -1.962000 2.624000
    N -2.328000 1.192000 2.594000
    N 0.075000 1.492000 2.674000
    N -0.520000 -2.058000 2.616000
    N 1.248000 -0.659000 2.668000
    N -3.378000 -0.885000 2.563000
    H -1.335000 2.995000 2.632000
    H 1.478000 -2.796000 2.584000
    H 2.209000 -0.350000 2.597000
    H -4.215000 -0.388000 2.285000
    H -3.310000 -1.855000 2.283000
    C 6.220000 -1.272000 -0.103000
    C 6.356000 0.086000 -0.067000
    C 5.220000 0.956000 -0.165000
    C 3.923000 0.366000 -0.304000
    C 3.782000 -1.049000 -0.351000
    C 4.934000 -1.892000 -0.240000
    C 2.493000 -1.631000 -0.480000
    C 4.770000 -3.283000 -0.265000
    C 3.506000 -3.877000 -0.385000
    C 2.348000 -3.043000 -0.500000
    C 3.325000 -5.299000 -0.390000
    C 2.082000 -5.857000 -0.476000
    C 0.902000 -5.047000 -0.564000
    C 1.053000 -3.625000 -0.579000
    C -0.094000 -2.794000 -0.668000
    C 0.051000 -1.376000 -0.669000
    C 1.342000 -0.795000 -0.596000
    C 2.777000 1.198000 -0.411000
    C 1.483000 0.616000 -0.539000
    C 0.335000 1.447000 -0.612000
    C -0.954000 0.866000 -0.706000
    C -1.098000 -0.545000 -0.714000
    C -1.388000 -3.374000 -0.698000
    C -2.540000 -2.541000 -0.725000
    C -2.397000 -1.129000 -0.764000
    C -0.384000 -5.600000 -0.613000
    C -1.531000 -4.796000 -0.667000
    C -3.848000 -3.122000 -0.735000
    C -4.970000 -2.283000 -0.758000
    C -4.850000 -0.887000 -0.776000
    C -3.547000 -0.297000 -0.768000
    C -2.110000 1.699000 -0.715000
    C -3.403000 1.117000 -0.763000
    C -2.853000 -5.350000 -0.684000
    C -3.959000 -4.551000 -0.713000
    C -4.561000 1.957000 -0.756000
    C -5.853000 1.338000 -0.775000
    C -5.992000 -0.020000 -0.782000
    C -4.398000 3.348000 -0.723000
    C -3.131000 3.942000 -0.678000
    C -1.967000 3.110000 -0.676000
    C -0.674000 3.692000 -0.579000
    C 0.478000 2.863000 -0.555000
    C 1.768000 3.443000 -0.435000
    C 2.916000 2.610000 -0.345000
    C 4.218000 3.190000 -0.215000
    C 5.337000 2.351000 -0.126000
    C 4.325000 4.619000 -0.169000
    C 1.904000 4.865000 -0.368000
    C 3.221000 5.418000 -0.243000
    C -0.527000 5.114000 -0.521000
    C 0.756000 5.667000 -0.421000
    C -2.953000 5.364000 -0.625000
    C -1.710000 5.923000 -0.552000
    H 7.095000 -1.912000 -0.024000
    H 7.340000 0.535000 0.037000
    H 5.648000 -3.920000 -0.181000
    H 4.205000 -5.932000 -0.315000
    H 1.965000 -6.939000 -0.470000
    H -0.497000 -6.682000 -0.594000
    H -5.963000 -2.729000 -0.763000
    H -2.959000 -6.432000 -0.666000
    H -4.953000 -4.992000 -0.720000
    H -6.732000 1.979000 -0.778000
    H -6.982000 -0.470000 -0.790000
    H -5.281000 3.984000 -0.719000
    H 6.324000 2.797000 -0.023000
    H 5.314000 5.059000 -0.070000
    H 3.324000 6.500000 -0.201000
    H 0.865000 6.749000 -0.375000
    H -3.838000 5.995000 -0.639000
    H -1.596000 7.004000 -0.509000
         """

BASIS = "sto-3g"
FORCE_EMBEDDING_GRADIENT_PATH = "force_embedding_C3A.txt"
REFERENCE_GRADIENT_PATH = "ref_C3A.txt"


def load_gradient(path: str, n_atoms: int) -> torch.Tensor:
    """Load a Cartesian gradient and validate its ``(n_atoms, 3)`` shape."""
    gradient = torch.as_tensor(
        np.loadtxt(path),
        dtype=torch.float64,
    )

    if gradient.shape != (n_atoms, 3):
        raise ValueError(
            f"{path} must have shape {(n_atoms, 3)}, "
            f"but has shape {tuple(gradient.shape)}."
        )

    return gradient


def build_system():
    """Build the PySCF molecule and its BE fragmentation."""
    mol = gto.M(atom=GEOMETRY, basis=BASIS)
    coordinates = torch.as_tensor(mol.atom_coords(), dtype=torch.float64)
    atomic_numbers = mol.atom_charges()

    fragmented = Fragmented.from_mole(
        mol,
        n_BE=2,
        h_treatment="treat_H_diff",
    )
    fragments = fragmented.get_atom_in_frag()
    frag_per_atom = fragmented.get_frag_per_atom()
    inverted_fragments = [
        [atom for atom in range(mol.natm) if atom not in fragment]
        for fragment in fragments
    ]

    return (
        mol,
        coordinates,
        atomic_numbers,
        fragments,
        inverted_fragments,
        frag_per_atom,
    )


def load_training_gradients(n_atoms: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Load the force-embedding and reference gradients used for fitting."""
    return (
        load_gradient(FORCE_EMBEDDING_GRADIENT_PATH, n_atoms),
        load_gradient(REFERENCE_GRADIENT_PATH, n_atoms),
    )


def external_d3_gradient(
    coordinates: torch.Tensor,
    *,
    external_d3: TorchExternalD3BJ,
    fragments: list[list[int]],
    frag_per_atom: list[int],
    parameters: TrainableD3BJParameters,
    create_graph: bool,
) -> torch.Tensor:
    """Assemble one external-D3 gradient per atom from its owning fragment."""
    _, fragment_gradients = external_d3.energies_and_gradients(
        coordinates,
        parameters=parameters,
        create_graph=create_graph,
    )

    correction_gradient = torch.zeros_like(coordinates)

    for fragment_index, (fragment, fragment_gradient) in enumerate(
        zip(fragments, fragment_gradients)
    ):
        # Atoms geometrically present in this fragment but *owned* by another
        # fragment are buffer atoms: do not use this fragment's gradient for them.
        owned_positions = [
            local_index
            for local_index, atom_index in enumerate(fragment)
            if frag_per_atom[atom_index] == fragment_index
        ]

        if not owned_positions:
            continue

        local_indices = torch.tensor(
            owned_positions,
            dtype=torch.int64,
            device=coordinates.device,
        )
        atom_indices = torch.tensor(
            [fragment[index] for index in owned_positions],
            dtype=torch.int64,
            device=coordinates.device,
        )

        correction_gradient = correction_gradient.index_copy(
            dim=0,
            index=atom_indices,
            source=fragment_gradient[local_indices],
        )

    return correction_gradient


def evaluate_loss(
    coordinates: torch.Tensor,
    force_embedding_gradient: torch.Tensor,
    reference_gradient: torch.Tensor,
    external_d3: TorchExternalD3BJ,
    fragments: list[list[int]],
    frag_per_atom: list[int],
    parameters: TrainableD3BJParameters,
) -> torch.Tensor:
    """Return the MSE without building a graph or accumulating gradients."""
    d3_correction_gradient = external_d3_gradient(
        coordinates,
        external_d3=external_d3,
        fragments=fragments,
        frag_per_atom=frag_per_atom,
        parameters=parameters,
        create_graph=False,
    )

    residual = force_embedding_gradient + d3_correction_gradient - reference_gradient
    return torch.mean(residual**2)


def optimize_d3_parameters(
    coordinates: torch.Tensor,
    force_embedding_gradient: torch.Tensor,
    reference_gradient: torch.Tensor,
    external_d3: TorchExternalD3BJ,
    fragments: list[list[int]],
    frag_per_atom: list[int],
    fit_parameters: TrainableD3BJParameters,
) -> tuple[object, int, float]:
    """Fit D3(BJ) parameters using L-BFGS and report periodic diagnostics."""
    optimizer = torch.optim.LBFGS(
        fit_parameters.parameters(),
        lr=1.0,
        max_iter=100,
        tolerance_grad=1.0e-10,
        tolerance_change=1.0e-12,
        line_search_fn="strong_wolfe",
    )
    closure_calls = 0

    def closure() -> torch.Tensor:
        nonlocal closure_calls
        closure_calls += 1
        optimizer.zero_grad()

        d3_correction_gradient = external_d3_gradient(
            coordinates,
            external_d3=external_d3,
            fragments=fragments,
            frag_per_atom=frag_per_atom,
            parameters=fit_parameters,
            create_graph=True,
        )

        gradient_residual = (
            force_embedding_gradient + d3_correction_gradient - reference_gradient
        )

        loss = torch.mean(gradient_residual**2)
        loss.backward()

        if closure_calls == 1 or closure_calls % 10 == 0:
            current = fit_parameters.as_fixed_parameters()
            print(
                f"[closure {closure_calls:3d}] "
                f"MSE = {loss.item():.8e} | "
                f"RMS residual = {torch.sqrt(loss).item():.8e} | "
                f"s6={current.s6:.6f}, s8={current.s8:.6f}, "
                f"a1={current.a1:.6f}, a2={current.a2:.6f}"
            )

        return loss

    print("Starting L-BFGS optimization...")
    initial_loss = evaluate_loss(
        coordinates,
        force_embedding_gradient,
        reference_gradient,
        external_d3,
        fragments,
        frag_per_atom,
        fit_parameters,
    )
    print(f"Initial MSE: {initial_loss.item():.8e}")
    optimizer.step(closure)

    final_loss = evaluate_loss(
        coordinates,
        force_embedding_gradient,
        reference_gradient,
        external_d3,
        fragments,
        frag_per_atom,
        fit_parameters,
    )
    return fit_parameters.as_fixed_parameters(), closure_calls, final_loss.item()


def main() -> None:
    print("Building molecular system...")
    (
        mol,
        coordinates,
        atomic_numbers,
        fragments,
        inverted_fragments,
        frag_per_atom,
    ) = build_system()

    print(f"Atoms: {mol.natm}")
    print(f"Fragments: {len(fragments)}")
    print(f"Fragment sizes: {[len(fragment) for fragment in fragments]}")

    print("Loading reference gradients...")
    force_embedding_gradient, reference_gradient = load_training_gradients(mol.natm)

    print("Initializing external D3(BJ) model...")
    external_d3 = TorchExternalD3BJ(
        atomic_numbers=atomic_numbers.tolist(),
        fragments=fragments,
        inverted_fragments=inverted_fragments,
        method="hf",
    )
    fit_parameters = TrainableD3BJParameters(external_d3.total_d3.parameters)
    initial_parameters = external_d3.total_d3.parameters
    print("Initial D3(BJ) parameters:")
    print(f"  s6 = {initial_parameters.s6:.10f}")
    print(f"  s8 = {initial_parameters.s8:.10f}")
    print(f"  a1 = {initial_parameters.a1:.10f}")
    print(f"  a2 = {initial_parameters.a2:.10f}")

    optimized_parameters, closure_calls, final_loss = optimize_d3_parameters(
        coordinates,
        force_embedding_gradient,
        reference_gradient,
        external_d3,
        fragments,
        frag_per_atom,
        fit_parameters,
    )

    print("\nOptimization complete.")
    print(f"Closure evaluations: {closure_calls}")
    print(f"Final MSE: {final_loss:.12e}")
    print("Optimized D3(BJ) parameters:")
    print(f"  s6 = {optimized_parameters.s6:.10f}")
    print(f"  s8 = {optimized_parameters.s8:.10f}")
    print(f"  a1 = {optimized_parameters.a1:.10f}")
    print(f"  a2 = {optimized_parameters.a2:.10f}")


if __name__ == "__main__":
    main()
