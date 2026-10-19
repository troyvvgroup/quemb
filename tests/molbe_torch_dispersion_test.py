import torch

from quemb.molbe.dispersion import (
    d3_coordination_numbers,
    pairwise_distances,
)


def test_pairwise_distances_preserve_autograd():
    coordinates = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]],
        dtype=torch.float64,
        requires_grad=True,
    )

    distances = pairwise_distances(coordinates)

    expected_distances = torch.tensor(
        [[0.0, 3.0], [3.0, 0.0]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(distances, expected_distances)

    # A temporary toy dispersion-like energy.
    energy = -(distances[0, 1] ** -6)
    energy.backward()

    expected_gradient = torch.tensor(
        [
            [0.0, 0.0, -6.0 / 3.0**7],
            [0.0, 0.0, 6.0 / 3.0**7],
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(coordinates.grad, expected_gradient)


def test_d3_coordination_numbers_are_symmetric_for_two_identical_atoms():
    coordinates = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]],
        dtype=torch.float64,
    )
    covalent_radii = torch.tensor([1.0, 1.0], dtype=torch.float64)

    coordination_numbers = d3_coordination_numbers(coordinates, covalent_radii)

    torch.testing.assert_close(
        coordination_numbers[0],
        coordination_numbers[1],
    )
    assert torch.all(coordination_numbers > 0)
    assert torch.all(coordination_numbers < 1)


def test_d3_coordination_numbers_decrease_with_distance():
    covalent_radii = torch.tensor([1.0, 1.0], dtype=torch.float64)

    close_coordinates = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]],
        dtype=torch.float64,
    )
    far_coordinates = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 6.0]],
        dtype=torch.float64,
    )

    close_coordination = d3_coordination_numbers(
        close_coordinates,
        covalent_radii,
    )
    far_coordination = d3_coordination_numbers(
        far_coordinates,
        covalent_radii,
    )

    assert torch.all(close_coordination > far_coordination)


def test_d3_coordination_numbers_pass_gradcheck():
    coordinates = torch.tensor(
        [[0.0, 0.0, 0.0], [0.2, -0.1, 3.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    covalent_radii = torch.tensor([1.0, 1.0], dtype=torch.float64)

    def coordination_from_coordinates(coordinates):
        return d3_coordination_numbers(coordinates, covalent_radii)

    assert torch.autograd.gradcheck(
        coordination_from_coordinates,
        (coordinates,),
        eps=1.0e-6,
        atol=1.0e-5,
        rtol=1.0e-3,
    )
