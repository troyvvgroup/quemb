# Validates analytic D3 fragment gradients against central finite differences.

import numpy as np
from pyscf import gto

from quemb.molbe.chemfrag import Fragmented
from quemb.molbe.dispersion import D3, finite_difference_fragment_gradients


def test_d3_fragment_gradients():
    mol = gto.M(
        atom="""
            C         -9.34365       -1.15645       -0.59044
            H         -9.69311       -0.40929       -1.31014
            H         -9.68356       -0.85885        0.40667
            C         -9.85706       -2.54152       -0.95028
            H         -9.47312       -2.82186       -1.93806
            C        -11.38479       -2.58895       -0.95421
            H         -9.46363       -3.26894       -0.23071
            H        -11.77324       -1.85666       -1.67249
            H        -11.76375       -2.30372        0.03478
            C        -11.89717       -3.98384       -1.31663
            H        -11.51823       -4.26884       -2.30567
            C        -13.42604       -4.03140       -1.32061
            H        -11.50876       -4.71596       -0.59820
            H        -13.81445       -3.29928       -2.03904
            H        -13.80498       -3.74641       -0.33157
            C        -13.93843       -5.42628       -1.68303
            H        -13.55946       -5.71152       -2.67202
            C        -15.46616       -5.47368       -1.68698
            H        -13.55001       -6.15859       -0.96475
            H        -15.85957       -4.74625       -2.40655
            H        -15.85012       -5.19334       -0.69921
            C        -15.97962       -6.85873       -2.04684
            H        -15.63969       -7.15633       -3.04395
            H        -17.07405       -6.86988       -2.04370
            H        -15.63019       -7.60591       -1.32714
            H         -8.24922       -1.14526       -0.59360
        """,
        basis="3-21g",
    )

    coordinates = mol.atom_coords()
    atomic_numbers = mol.atom_charges()

    fragmented = Fragmented.from_mole(
        mol,
        n_BE=2,
        h_treatment="treat_H_diff",
    )
    atom_in_frag = fragmented.get_atom_in_frag()

    d3 = D3(atomic_numbers, atom_in_frag, method="hf")

    # Analytic fragment gradients.
    _, analytic_gradients = d3.energy_and_gradient(coordinates)

    # Numerical fragment gradients from central finite differences.
    fd_gradients = finite_difference_fragment_gradients(
        d3,
        coordinates,
        step=1.0e-4,
    )

    # expected_max_errors = [
    #    5.1072904586058065e-12,
    #    7.5715536271281360e-12,
    #    7.8630244676469730e-12,
    #    7.8773784370489430e-12,
    #    7.3666567637631350e-12,
    #    4.7960554798442970e-12,
    # ]

    # expected_rms_errors = [
    #    1.2602107983116170e-12,
    #    1.9620500828492450e-12,
    #    2.1084509790929532e-12,
    #    2.1192960767113310e-12,
    #    1.9483748577899804e-12,
    #    1.2267847276724338e-12,
    # ]

    assert len(analytic_gradients) == len(fd_gradients)

    # Validate each fragment independently.
    for fragment_index, (analytic, finite_difference) in enumerate(
        zip(analytic_gradients, fd_gradients, strict=True)
    ):
        difference = finite_difference - analytic
        max_error = np.max(np.abs(difference))
        rms_error = np.sqrt(np.mean(difference**2))

        assert max_error < 1.0e-10, (
            f"Fragment {fragment_index} maximum absolute error is too large: "
            f"{max_error:.12e}"
        )

        assert rms_error < 1.0e-11, (
            f"Fragment {fragment_index} RMS error is too large: {rms_error:.12e}"
        )

    del d3


if __name__ == "__main__":
    test_d3_fragment_gradients()
