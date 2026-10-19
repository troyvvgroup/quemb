import torch

from quemb.molbe.dispersion import pairwise_distances


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
