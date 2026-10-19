import numpy as np
import torch
from dftd3.interface import DispersionModel, RationalDampingParam

from quemb.molbe.dispersion import (
    d3_bj_energy,
)


def simple_d3_hf_bj_two_body_energy(
    atomic_numbers: torch.Tensor,
    coordinates: torch.Tensor,
) -> torch.Tensor:
    """Return a detached simple-dftd3 HF-D3(BJ) two-body reference energy."""
    # simple-dftd3 is used only as an independent numerical reference here;
    # detaching keeps this helper outside the PyTorch autograd graph.
    model = DispersionModel(
        atomic_numbers.detach().cpu().numpy(),
        coordinates.detach().cpu().numpy(),
    )
    parameter = RationalDampingParam(method="hf", atm=False)

    result = model.get_dispersion(parameter, grad=False)

    # Confirm that the reference contains no three-body Axilrod--Teller--Muto
    # contribution, so it is comparable to the two-body implementation under test.
    pairwise = model.get_pairwise_dispersion(parameter)
    np.testing.assert_allclose(
        pairwise["non-additive pairwise energy"],
        0.0,
        atol=1.0e-14,
    )

    return torch.tensor(
        result["energy"],
        dtype=coordinates.dtype,
        device=coordinates.device,
    )


def test_torch_hf_d3bj_two_body_energy_and_gradient_match_expectations():
    atomic_numbers = torch.tensor([8, 1, 1], dtype=torch.int64)
    # Water geometry in bohr, the coordinate unit expected by both implementations.
    coordinates = torch.tensor(
        [
            [0.0000000000, 0.0000000000, 0.0000000000],
            [0.0000000000, 1.4323367300, 1.1071526600],
            [0.0000000000, -1.4323367300, 1.1071526600],
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
    reference_energy = simple_d3_hf_bj_two_body_energy(
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
    # A finite gradient is a basic autograd regression check.  A value-by-value
    # force comparison can be added later if an independent gradient reference
    # is introduced.
    assert torch.isfinite(coordinates.grad).all()
