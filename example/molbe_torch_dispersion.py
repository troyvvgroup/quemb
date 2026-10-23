import numpy as np
import torch


# Todo
# extend to multiple geometries??
# correction should only be for the atoms outside the fragment
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

fit_parameters = TrainableD3BJParameters(d3.parameters)

optimizer = torch.optim.LBFGS(
    fit_parameters.parameters(),
    lr=1.0,
    max_iter=100,
    tolerance_grad=1.0e-10,
    tolerance_change=1.0e-12,
    line_search_fn="strong_wolfe",
)


def closure() -> torch.Tensor:
    optimizer.zero_grad()

    _, d3_gradient = d3.energy_and_gradient(
        coordinates,
        parameters=fit_parameters,
        create_graph=True,
    )

    gradient_residual = force_embedding_gradient + d3_gradient - reference_gradient

    loss = torch.mean(gradient_residual**2)
    loss.backward()

    return loss


optimizer.step(closure)
optimized_parameters = fit_parameters.as_fixed_parameters()

print(f"Final loss: {closure().item():.12e}")
print(f"s6 = {optimized_parameters.s6:.10f}")
print(f"s8 = {optimized_parameters.s8:.10f}")
print(f"a1 = {optimized_parameters.a1:.10f}")
print(f"a2 = {optimized_parameters.a2:.10f}")
