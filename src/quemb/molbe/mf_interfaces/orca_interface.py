from collections.abc import Sequence
from typing import cast

import numpy as np
from pyscf.scf import RHF

from quemb.molbe.mf_interfaces.pyscf_orbs import Orbital
from quemb.shared.helper import argsort, normalize_column_signs
from quemb.shared.typing import Matrix, Vector


def _get_orca_mo_coeff(
    json_data: dict, orbitals: Sequence[Orbital], idx: Sequence[int]
) -> Matrix[np.float64]:
    orca_MOs = np.array(
        [x["MOCoefficients"] for x in json_data["Molecule"]["MolecularOrbitals"]["MOs"]]
    ).T
    # The +-3 and +-4 m_l values of the f, g, and h orbitals
    # use an opposite sign convention
    switch_sign = [
        i
        for i, o in enumerate(orbitals)
        if o.l in {"f", "g", "h"} and o.m_l[-2:] in {"-4", "-3", "+3", "+4"}
    ]
    orca_MOs[switch_sign, :] *= -1
    return normalize_column_signs(orca_MOs[idx, :])


def _get_orca_mo_occ(json_data: dict, idx: Sequence[int]) -> Vector[np.float64]:
    orca_occ = np.array(
        [x["Occupancy"] for x in json_data["Molecule"]["MolecularOrbitals"]["MOs"]]
    )
    return cast(Vector[np.float64], orca_occ[idx])


def _get_orca_mo_energy(json_data: dict, idx: Sequence[int]) -> Vector[np.float64]:
    if json_data["Molecule"]["MolecularOrbitals"]["EnergyUnit"] != "Eh":
        raise ValueError("Inconsistent Error Unit")
    mo_energy = np.array(
        [x["OrbitalEnergy"] for x in json_data["Molecule"]["MolecularOrbitals"]["MOs"]]
    )
    return cast(Vector[np.float64], mo_energy[idx])


def _get_mf_from_orca(mol, json_data) -> RHF:
    orbitals = [
        Orbital.from_orca_label(label)
        for label in json_data["Molecule"]["MolecularOrbitals"]["OrbitalLabels"]
    ]
    idx = argsort(orbitals)

    mf = RHF(mol)

    mf.mo_coeff = _get_orca_mo_coeff(json_data, orbitals, idx)
    mf.mo_energy = _get_orca_mo_energy(json_data, idx)
    mf.mo_occ = _get_orca_mo_occ(json_data, idx)
    return mf
