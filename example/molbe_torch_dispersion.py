import numpy as np
import torch


# Todo
# extend to multiple geometries??
# ensure units are compatible
def load_gradient(path: str, n_atoms: int) -> torch.Tensor:
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


force_embedding_gradient = load_gradient(
    "force_embedding_gradient.txt",
    n_atoms=len(atomic_numbers),
)
reference_gradient = load_gradient(
    "reference_gradient.txt",
    n_atoms=len(atomic_numbers),
)

external_d3 = TorchExternalD3BJ(
    atomic_numbers=atomic_numbers.tolist(),
    fragments=fragments,
    inverted_fragments=inverted_fragments,
    method="hf",
)

fit_parameters = TrainableD3BJParameters(external_d3.parameters)


def external_d3_gradient(
    coordinates: torch.Tensor,
    *,
    parameters: TrainableD3BJParameters,
    create_graph: bool,
) -> torch.Tensor:
    """Return the external D3 gradient for every atom in the full system."""
    _, fragment_gradients = external_d3.energies_and_gradients(
        coordinates,
        parameters=parameters,
        create_graph=create_graph,
    )

    correction_gradient = torch.zeros_like(coordinates)
    atom_assignment_count = torch.zeros(
        coordinates.shape[0],
        dtype=torch.int64,
        device=coordinates.device,
    )

    for fragment_index, (fragment, fragment_gradient) in enumerate(
        zip(fragments, fragment_gradients)
    ):
        if torch.count_nonzero(fragment_gradient).item() == 0:
            raise RuntimeError(
                f"External D3 gradient for fragment {fragment_index} "
                "is identically zero."
            )

        atom_indices = torch.tensor(
            fragment,
            dtype=torch.int64,
            device=coordinates.device,
        )

        # Each atom should receive the correction associated with its
        # unique owning fragment.
        correction_gradient = correction_gradient.index_copy(
            dim=0,
            index=atom_indices,
            source=fragment_gradient,
        )
        atom_assignment_count[atom_indices] += 1

    if not torch.all(atom_assignment_count == 1):
        raise ValueError(
            "fragments must assign every system atom to exactly one fragment."
        )

    return correction_gradient


def closure() -> torch.Tensor:
    optimizer.zero_grad()

    d3_correction_gradient = external_d3_gradient(
        coordinates,
        parameters=fit_parameters,
        create_graph=True,
    )

    gradient_residual = (
        force_embedding_gradient + d3_correction_gradient - reference_gradient
    )

    loss = torch.mean(gradient_residual**2)
    loss.backward()

    return loss


optimizer = torch.optim.LBFGS(
    fit_parameters.parameters(),
    lr=1.0,
    max_iter=100,
    tolerance_grad=1.0e-10,
    tolerance_change=1.0e-12,
    line_search_fn="strong_wolfe",
)

optimizer.step(closure)
optimized_parameters = fit_parameters.as_fixed_parameters()

print(f"Final loss: {closure().item():.12e}")
print(f"s6 = {optimized_parameters.s6:.10f}")
print(f"s8 = {optimized_parameters.s8:.10f}")
print(f"a1 = {optimized_parameters.a1:.10f}")
print(f"a2 = {optimized_parameters.a2:.10f}")
