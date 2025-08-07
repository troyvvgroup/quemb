# mypy: disable-error-code=import-not-found

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Final, cast

import numpy as np
from attrs import define, field
from chemcoord import Cartesian
from pyscf.gto import Mole
from pyscf.scf.hf import RHF

from quemb.molbe.mf_interfaces._pyscf_interface import create_mf
from quemb.molbe.mf_interfaces._pyscf_orbital_order import Orbital
from quemb.shared.helper import argsort, normalize_column_signs, unused
from quemb.shared.manage_scratch import WorkDir
from quemb.shared.typing import Matrix, Vector

logger: Final = logging.getLogger(__name__)


@define
class OrcaArgs:
    kwargs: dict = field(factory=dict)


try:
    from opi.core import Calculator
    from opi.input.simple_keywords import (
        BasisSet,
        SimpleKeyword,
    )
    from opi.input.simple_keywords.method import Method
    from opi.input.structures.structure import Structure
    from opi.output.core import Output

    PYSCF_TO_ORCA_BASIS: Final[Mapping[str, SimpleKeyword]] = {
        "sto-3g": BasisSet.STO_3G,
        "3-21g": BasisSet.G3_21G,
        "6-31g": BasisSet.G6_31G,
        "6-311g": BasisSet.G6_311G,
        "def2-svp": BasisSet.DEF2_SVP,
        "def2-tzvp": BasisSet.DEF2_TZVP,
        "def2-qzvp": BasisSet.DEF2_QZVP,
        "def2-svpd": BasisSet.DEF2_SVPD,
        "def2-tzvpd": BasisSet.DEF2_TZVPD,
        "def2-qzvpd": BasisSet.DEF2_QZVPD,
        "def2-tzvpp": BasisSet.DEF2_TZVPP,
        "def2-qzvpp": BasisSet.DEF2_QZVPP,
        "def2-tzvppd": BasisSet.DEF2_TZVPPD,
        "def2-qzvppd": BasisSet.DEF2_QZVPPD,
        "cc-pvdz": BasisSet.CC_PVDZ,
        "cc-pvtz": BasisSet.CC_PVTZ,
        "cc-pvqz": BasisSet.CC_PVQZ,
        "cc-pv5z": BasisSet.CC_PV5Z,
        "aug-cc-pvdz": BasisSet.AUG_CC_PVDZ,
        "aug-cc-pvtz": BasisSet.AUG_CC_PVTZ,
        "aug-cc-pvqz": BasisSet.AUG_CC_PVQZ,
        "aug-cc-pv5z": BasisSet.AUG_CC_PV5Z,
    }

    def _get_orca_mo_coeff(json_data: dict) -> Matrix[np.float64]:
        orca_MOs = np.array(
            [
                x["MOCoefficients"]
                for x in json_data["Molecule"]["MolecularOrbitals"]["MOs"]
            ]
        ).T
        orbital_labels = [
            Orbital.from_orca_label(label)
            for label in json_data["Molecule"]["MolecularOrbitals"]["OrbitalLabels"]
        ]
        # The +-3 and +-4 m_l values of the f, g, and h orbitals
        # use an opposite sign convention as compared to pyscf
        switch_sign = [
            i
            for i, orbital in enumerate(orbital_labels)
            if orbital.l in {"f", "g", "h"}
            and orbital.m_l[-2:] in {"-4", "-3", "+3", "+4"}
        ]
        orca_MOs[switch_sign, :] *= -1
        AOs_pyscf_order = argsort(orbital_labels)
        return normalize_column_signs(orca_MOs[AOs_pyscf_order, :])

    def _get_orca_mo_occ(json_data: dict) -> Vector[np.float64]:
        return cast(
            Vector[np.float64],
            np.array(
                [
                    x["Occupancy"]
                    for x in json_data["Molecule"]["MolecularOrbitals"]["MOs"]
                ],
            ),
        )

    def _get_orca_mo_energy(json_data: dict) -> Vector[np.float64]:
        if json_data["Molecule"]["MolecularOrbitals"]["EnergyUnit"] != "Eh":
            raise ValueError("Inconsistent Error Unit")
        return cast(
            Vector[np.float64],
            np.array(
                [
                    x["OrbitalEnergy"]
                    for x in json_data["Molecule"]["MolecularOrbitals"]["MOs"]
                ]
            ),
        )

    def _parse_energy(output: Output) -> float:
        output.parse()
        return output.results_properties.geometries[0].single_point_data.finalenergy  # type: ignore[union-attr,index,return-value]

    def _parse_orca_into_mf(mol: Mole, output: Output) -> RHF:
        with open(output.gbw_json_file, "r") as f:
            json_data = json.load(f)

        return create_mf(
            mol=mol,
            mo_coeff=_get_orca_mo_coeff(json_data),
            mo_energy=_get_orca_mo_energy(json_data),
            mo_occ=_get_orca_mo_occ(json_data),
            e_tot=_parse_energy(output),
        )

    def _prepare_orca_calc(
        mol: Mole,
        work_dir: WorkDir,
        n_procs: int,
        simple_keywords: list[SimpleKeyword],
    ) -> Calculator:
        orca_work_dir: Final = WorkDir(work_dir / "orca_mf")
        geometry_path: Final = orca_work_dir / "geometry.xyz"
        if mol.unit != "angstrom":
            raise ValueError("mol has to be in Angstrom.")
        Cartesian.from_pyscf(mol).to_xyz(geometry_path)

        calc = Calculator(basename="mf_calculation", working_dir=orca_work_dir.path)
        calc.structure = Structure.from_xyz(
            geometry_path, charge=mol.charge, multiplicity=mol.multiplicity
        )

        try:
            basis: Final = PYSCF_TO_ORCA_BASIS[mol.basis]
        except KeyError:
            raise NotImplementedError(
                f"PYSCF basis set {mol.basis} is not supported.\n"
                "This is either because it genuinely does not exist in ORCA,\n"
                "or because the translation is not defined.\n"
                "If you think that the basis exists in\n"
                "pyscf and ORCA, then raise an issue at the quemb GitHub repository."
            )
        sk_list: Final = [
            Method.HF,
            basis,
        ]

        calc.input.add_simple_keywords(*(sk_list + simple_keywords))

        # > Define number of CPUs for the calcualtion
        calc.input.ncores = n_procs

        return calc

    def get_mf_orca(
        mol: Mole, work_dir: WorkDir, n_procs: int, simple_keywords: list[SimpleKeyword]
    ) -> RHF:
        calc = _prepare_orca_calc(mol, work_dir, n_procs, simple_keywords)
        logger.debug("Writing ORCA input")
        calc.write_input()
        logger.info("Starting ORCA calculation")
        calc.run()
        output = calc.get_output()
        if not output.terminated_normally():
            raise RuntimeError("ORCA terminated with errors")
        logger.info("ORCA calculation finished successfully")

        return _parse_orca_into_mf(mol, output)

except ImportError:

    def get_mf_orca(
        mol: Mole, work_dir: WorkDir, n_procs: int, simple_keywords: list[SimpleKeyword]
    ) -> RHF:  # type: ignore[return-type]
        unused(mol, work_dir, n_procs, simple_keywords)
        raise ImportError("ORCA and the ORCA python interface have to be available.")
