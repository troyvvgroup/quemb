# Author(s): Alexa Alexiu

import re

from numpy import array, linalg


def dyson_parser(fobj, output="eom.out", n_ex=15):
    """
    This function parses the Q-Chem EOM-IP output file and reads the
    excitation energies, as well as left- and right-Dyson orbitals.

    Parameters
    ----------
    output : str
        Q-Chem output file
    n_ex : int
        Number of excited states computed
    """

    with open(output, "r") as f:
        content = f.read()

    # extract excitation energies
    energy_pattern = re.compile(
        r"EOMIP transition\s+\d+/\w+\s+"
        r"Total energy = ([\-\d\.]+) a\.u\.\s+"
        r"Excitation energy = ([\d\.]+) eV\.",
        re.MULTILINE,
    )

    excitation_energies = [float(match[1]) for match in energy_pattern.findall(content)]
    excitation_energies = array(excitation_energies)

    # fixes bug FOR MO BASIS DYSON ORBITALS (where NAO>100)

    # extract left Dyson orbitals
    # match everything until "Excitation energy" or end of file
    dyson_pattern = re.compile(
        r"Left alpha Dyson orbital in the MO basis "
        r"\(canonical Q-Chem's ordering\):\s*\n(.*?)(?:\n\n|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    dyson_matches_left = dyson_pattern.findall(content)

    # extract right Dyson orbitals
    dyson_pattern = re.compile(
        r"Right alpha Dyson orbital in the MO basis "
        r"\(canonical Q-Chem's ordering\):\s*\n(.*?)(?:\n\n|\Z)",
        re.MULTILINE | re.DOTALL,
    )

    dyson_matches_right = dyson_pattern.findall(content)

    coeff_matrix_left = []

    line_regex = re.compile(r"(\d+)\s*([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)")

    for block in dyson_matches_left[:n_ex]:
        block_coeffs = []
        for line in block.strip().splitlines():
            m = line_regex.match(line.strip())
            if m:
                index, coeff = m.groups()
                block_coeffs.append(float(coeff))
            else:
                print("Could not parse line:", line)
        coeff_matrix_left.append(block_coeffs)

    coeff_matrix_left = array(coeff_matrix_left)

    coeff_matrix_right = []

    line_regex = re.compile(r"(\d+)\s*([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)")

    for block in dyson_matches_right[:n_ex]:
        block_coeffs = []
        for line in block.strip().splitlines():
            m = line_regex.match(line.strip())
            if m:
                index, coeff = m.groups()
                block_coeffs.append(float(coeff))
            else:
                print("Could not parse line:", line)
        coeff_matrix_right.append(block_coeffs)

    coeff_matrix_right = array(coeff_matrix_right)

    # extract Dyson orbitals (AO basis)

    blocks = []
    lines = content.splitlines()

    i = 0
    while i < len(lines):
        if lines[i].startswith(
            "Decomposition over AOs for the left alpha Dyson orbital:"
        ):
            i += 1
            coeffs = []

            while i < len(lines) and lines[i].strip() != "*****":
                coeffs.append(float(lines[i].strip()))
                i += 1

            blocks.append(coeffs)

        i += 1

    coeff_matrix_ao = array(blocks[:n_ex])

    ###NEW: normalize MO basis Dyson orbitals!
    for i in range(n_ex):
        coeff_matrix_left[i, :] = coeff_matrix_left[i, :] / linalg.norm(
            coeff_matrix_left[i, :]
        )
        coeff_matrix_right[i, :] = coeff_matrix_right[i, :] / linalg.norm(
            coeff_matrix_right[i, :]
        )

    # save results to fobj
    fobj.ex_e = excitation_energies
    fobj.dyson_left = coeff_matrix_left
    fobj.dyson_right = coeff_matrix_right
    fobj.dyson_ao = coeff_matrix_ao

    return
