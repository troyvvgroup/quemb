import torch

from quemb.molbe.dispersion import (
    d3_coordination_numbers,
    d3_pairwise_c6,
    d3_reference_weights,
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


def test_d3_reference_weights_are_normalized():
    coordination_numbers = torch.tensor(
        [0.5, 2.0, 3.5],
        dtype=torch.float64,
    )
    reference_coordination_numbers = torch.tensor(
        [0.0, 2.0, 4.0],
        dtype=torch.float64,
    )

    weights = d3_reference_weights(
        coordination_numbers,
        reference_coordination_numbers,
    )

    expected_sums = torch.ones(3, dtype=torch.float64)
    torch.testing.assert_close(weights.sum(dim=1), expected_sums)
    assert torch.all(weights > 0)


def test_d3_reference_weights_prefer_nearest_reference_state():
    coordination_numbers = torch.tensor(
        [0.0, 2.0, 4.0],
        dtype=torch.float64,
    )
    reference_coordination_numbers = torch.tensor(
        [0.0, 2.0, 4.0],
        dtype=torch.float64,
    )

    weights = d3_reference_weights(
        coordination_numbers,
        reference_coordination_numbers,
    )

    expected_preferred_states = torch.tensor([0, 1, 2])
    torch.testing.assert_close(
        torch.argmax(weights, dim=1),
        expected_preferred_states,
    )


def test_d3_reference_weights_pass_gradcheck():
    coordination_numbers = torch.tensor(
        [0.4, 1.7, 3.3],
        dtype=torch.float64,
        requires_grad=True,
    )
    reference_coordination_numbers = torch.tensor(
        [0.0, 2.0, 4.0],
        dtype=torch.float64,
    )

    def weights_from_coordination_numbers(coordination_numbers):
        return d3_reference_weights(
            coordination_numbers,
            reference_coordination_numbers,
        )

    assert torch.autograd.gradcheck(
        weights_from_coordination_numbers,
        (coordination_numbers,),
        eps=1.0e-6,
        atol=1.0e-5,
        rtol=1.0e-3,
    )


def test_d3_pairwise_c6_interpolates_reference_values():
    reference_weights = torch.tensor(
        [
            [0.25, 0.75],
            [0.60, 0.40],
        ],
        dtype=torch.float64,
    )
    c6_reference = torch.tensor(
        [
            [10.0, 20.0],
            [20.0, 40.0],
        ],
        dtype=torch.float64,
    )

    c6 = d3_pairwise_c6(reference_weights, c6_reference)

    # [0.25, 0.75] @ C6_reference @ [0.60, 0.40]^T
    assert c6[0, 1].item() == 24.5
    assert c6[1, 0].item() == 24.5


def test_d3_pairwise_c6_is_symmetric_for_symmetric_reference_table():
    reference_weights = torch.tensor(
        [
            [0.20, 0.80],
            [0.50, 0.50],
            [0.90, 0.10],
        ],
        dtype=torch.float64,
    )
    c6_reference = torch.tensor(
        [
            [10.0, 20.0],
            [20.0, 40.0],
        ],
        dtype=torch.float64,
    )

    c6 = d3_pairwise_c6(reference_weights, c6_reference)

    torch.testing.assert_close(c6, c6.T)


def test_d3_pairwise_c6_passes_gradcheck():
    reference_weights = torch.tensor(
        [
            [0.25, 0.75],
            [0.60, 0.40],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    c6_reference = torch.tensor(
        [
            [10.0, 20.0],
            [20.0, 40.0],
        ],
        dtype=torch.float64,
    )

    def c6_from_weights(weights):
        return d3_pairwise_c6(weights, c6_reference)

    assert torch.autograd.gradcheck(
        c6_from_weights,
        (reference_weights,),
        eps=1.0e-6,
        atol=1.0e-5,
        rtol=1.0e-3,
    )
