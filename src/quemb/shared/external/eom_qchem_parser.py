# Author(s): Alexa Alexiu

import re

from numpy import array


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
        re.MULTILINE)

    excitation_energies = [float(match[1]) for match in energy_pattern.findall(content)]
    excitation_energies = array(excitation_energies)

    # extract left Dyson orbitals
    dyson_pattern = re.compile(
        r"Left alpha Dyson orbital in the MO basis "
        r"\(canonical Q-Chem's ordering\):\s*\n((?:\s*\d+\s+[\-\d\.Ee+]+\s*\n)+)",
        re.MULTILINE)
    dyson_matches_left = dyson_pattern.findall(content)

    # extract right Dyson orbitals
    dyson_pattern = re.compile(
        r"Right alpha Dyson orbital in the MO basis "
        r"\(canonical Q-Chem's ordering\):\s*\n((?:\s*\d+\s+[\-\d\.Ee+]+\s*\n)+)",
        re.MULTILINE)
    dyson_matches_right = dyson_pattern.findall(content)

    # extract Dyson orbitals (AO basis)
    dyson_pattern_ao = re.compile(
        r"Decomposition over AOs for the left alpha Dyson orbital:"
        r"\s*\n((?:\s*\d+\s+[\-\d\.Ee+]+\s*\n)+)",
        re.MULTILINE)
    dyson_matches_ao = dyson_pattern_ao.findall(content)

    coeff_matrix_left = array([
        [float(line.strip().split()[1]) for line in block.strip().splitlines()]
        for block in dyson_matches_left[:n_ex]
    ])

    coeff_matrix_right = array([
        [float(line.strip().split()[1]) for line in block.strip().splitlines()]
        for block in dyson_matches_right[:n_ex]
    ])

    coeff_matrix_ao = array([
        [float(line.strip().split()[1]) for line in block.strip().splitlines()]
        for block in dyson_matches_ao[:n_ex]
    ])

    # save results to fobj
    fobj.ex_e = excitation_energies
    fobj.dyson_left = coeff_matrix_left
    fobj.dyson_right = coeff_matrix_right
    fobj.dyson_ao = coeff_matrix_ao

    return