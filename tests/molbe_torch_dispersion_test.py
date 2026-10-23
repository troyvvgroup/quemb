import numpy as np
import torch
from dftd3.interface import DispersionModel, RationalDampingParam

from quemb.molbe.torch_d3 import (
    TorchD3BJ,
    TorchExternalD3BJ,
)


def reference_energy_and_gradient(
    atomic_numbers: torch.Tensor,
    coordinates: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return detached simple-dftd3 HF-D3(BJ) energy and gradient references."""

    model = DispersionModel(
        atomic_numbers.detach().cpu().numpy(),
        coordinates.detach().cpu().numpy(),
    )
    parameter = RationalDampingParam(method="hf", atm=False)

    result = model.get_dispersion(parameter, grad=True)

    # Confirm that the reference contains no three-body Axilrod--Teller--Muto
    # contribution, so it is comparable to the two-body implementation under test.
    pairwise = model.get_pairwise_dispersion(parameter)
    np.testing.assert_allclose(
        pairwise["non-additive pairwise energy"],
        0.0,
        atol=1.0e-14,
    )

    return (
        torch.tensor(
            result["energy"],
            dtype=coordinates.dtype,
            device=coordinates.device,
        ),
        torch.tensor(
            result["gradient"],
            dtype=coordinates.dtype,
            device=coordinates.device,
        ),
    )


def test_torch_hf_d3bj_two_body_energy_and_gradient_match_expectations():
    atomic_numbers = torch.tensor(
        [6, 1, 1, 1, 1, 6, 1, 1, 1, 1],
        dtype=torch.int64,
    )

    d3 = TorchD3BJ(
        atomic_numbers=atomic_numbers.tolist(),
        method="hf",
    )

    # Two methane molecules, C--C separation = 7 bohr (~3.7 Å).
    # C--H distance is ~2.05 bohr.
    coordinates = torch.tensor(
        [
            # CH4 molecule 1
            [0.000000, 0.000000, 0.000000],
            [1.186000, 1.186000, 1.186000],
            [1.186000, -1.186000, -1.186000],
            [-1.186000, 1.186000, -1.186000],
            [-1.186000, -1.186000, 1.186000],
            # CH4 molecule 2, translated by 7 bohr along z
            [0.000000, 0.000000, 7.000000],
            [1.186000, 1.186000, 8.186000],
            [1.186000, -1.186000, 5.814000],
            [-1.186000, 1.186000, 5.814000],
            [-1.186000, -1.186000, 8.186000],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )

    torch_energy = d3.energy(coordinates)

    reference_energy, reference_gradient = reference_energy_and_gradient(
        atomic_numbers,
        coordinates,
    )

    torch.testing.assert_close(
        torch_energy,
        reference_energy,
        rtol=1.0e-10,
        atol=1.0e-12,
    )

    torch_energy.backward()

    assert coordinates.grad is not None
    assert torch.count_nonzero(coordinates.grad).item() > 0

    torch.testing.assert_close(
        coordinates.grad,
        reference_gradient,
        rtol=1.0e-9,
        atol=1.0e-11,
    )


def test_torch_external_d3bj_energies_and_gradients_match_dftd3():
    atomic_numbers = torch.tensor(
        [6, 1, 1, 1, 1, 6, 1, 1, 1, 1],
        dtype=torch.int64,
    )

    coordinates = torch.tensor(
        [
            # Methane 1
            [0.000000, 0.000000, 0.000000],
            [1.186000, 1.186000, 1.186000],
            [1.186000, -1.186000, -1.186000],
            [-1.186000, 1.186000, -1.186000],
            [-1.186000, -1.186000, 1.186000],
            # Methane 2
            [0.000000, 0.000000, 7.000000],
            [1.186000, 1.186000, 8.186000],
            [1.186000, -1.186000, 5.814000],
            [-1.186000, 1.186000, 5.814000],
            [-1.186000, -1.186000, 8.186000],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )

    fragments = [
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
    ]
    inverted_fragments = [
        [5, 6, 7, 8, 9],
        [0, 1, 2, 3, 4],
    ]

    external_d3 = TorchExternalD3BJ(
        atomic_numbers=atomic_numbers.tolist(),
        fragments=fragments,
        inverted_fragments=inverted_fragments,
        method="hf",
    )

    torch_energies, torch_gradients = external_d3.energies_and_gradients(coordinates)

    for index, gradient in enumerate(torch_gradients):
        assert torch.count_nonzero(gradient).item() > 0, (
            f"External D3 gradient for fragment {index} is identically zero."
        )

    total_energy, total_gradient = reference_energy_and_gradient(
        atomic_numbers,
        coordinates,
    )

    for index, (fragment, inverted_fragment) in enumerate(
        zip(fragments, inverted_fragments)
    ):
        fragment_indices = torch.tensor(fragment, dtype=torch.int64)
        inverted_indices = torch.tensor(inverted_fragment, dtype=torch.int64)

        fragment_energy, fragment_gradient = reference_energy_and_gradient(
            atomic_numbers[fragment_indices],
            coordinates[fragment_indices],
        )
        inverted_energy, inverted_gradient = reference_energy_and_gradient(
            atomic_numbers[inverted_indices],
            coordinates[inverted_indices],
        )

        reference_external_energy = total_energy - fragment_energy - inverted_energy

        # The full expression is used so this remains correct if fragments
        # ever overlap in a future embedding construction.
        reference_full_gradient = total_gradient.clone()
        reference_full_gradient[fragment_indices] -= fragment_gradient
        reference_full_gradient[inverted_indices] -= inverted_gradient
        reference_fragment_gradient = reference_full_gradient[fragment_indices]

        torch.testing.assert_close(
            torch_energies[index],
            reference_external_energy,
            rtol=1.0e-10,
            atol=1.0e-12,
        )
        torch.testing.assert_close(
            torch_gradients[index],
            reference_fragment_gradient,
            rtol=1.0e-9,
            atol=1.0e-11,
        )
