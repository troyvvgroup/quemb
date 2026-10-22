import numpy as np
import torch
from dftd3.interface import DispersionModel, RationalDampingParam

from quemb.molbe.dispersion import (
    d3_bj_energy,
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

    # The production result remains connected to ``coordinates`` so that this
    # test exercises both the energy expression and its analytic gradient.
    torch_energy = d3_bj_energy(
        coordinates,
        atomic_numbers,
        method="hf",
    )
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
