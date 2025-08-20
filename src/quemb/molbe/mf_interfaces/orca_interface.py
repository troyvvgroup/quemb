from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from typing import Final, cast

import numpy as np
from attrs import define
from chemcoord import Cartesian
from pyscf.gto import Mole
from pyscf.scf.hf import RHF

from quemb.molbe.mf_interfaces._pyscf_orbital_order import Orbital
from quemb.molbe.mf_interfaces.pyscf_interface import create_mf
from quemb.shared.helper import argsort, normalize_column_signs, unused
from quemb.shared.manage_scratch import WorkDir
from quemb.shared.typing import Matrix, Vector

logger: Final = logging.getLogger(__name__)


@define(frozen=True, kw_only=True)
class OrcaArgs:
    """Use to pass information to ORCA.
    Follows the ORCA python interface. (https://www.faccts.de/docs/opi/1.0/docs/index.html)

    You can use the :func:`~get_orca_basis` function to translate
    a pyscf basis label to ORCA
    A "normal" Hartree Fock calculation can be invoked via

    >>> OrcaArgs(
    >>>     simple_keywords=[],
    >>>     blocks=[BlockBasis(basis=get_orca_basis(mol))],
    >>> )

    While a ``RIJK`` calculation with parallelisation would be:

    >>> from opi.input.blocks.block_basis import BlockBasis
    >>> from opi.input.simple_keywords import Approximation, SimpleKeyword
    >>> OrcaArgs(
    >>>     n_procs=4,
    >>>     memory_MB=16000,
    >>>     simple_keywords=[Approximation.RIJK],
    >>>     blocks=[
    >>>         BlockBasis(basis=get_orca_basis(mol), auxjk=SimpleKeyword("def2/jk"))
    >>>     ],
    >>> )

    Note that memory is for each core in MBs. i.e., you need physical memory of at
    least n_procs x memory (in MB). See: (https://www.faccts.de/docs/orca/6.0/manual/contents/structure.html#global-memory-use)
    """

    n_procs: Final[int] = 1
    memory_MB: Final[int] = 4000
    simple_keywords: Final[Sequence[SimpleKeyword]]
    blocks: Final[Sequence[Block]]


try:
    from opi.core import Calculator  # type: ignore[import-not-found]
    from opi.input.blocks import Block  # type: ignore[import-not-found]
    from opi.input.simple_keywords import (  # type: ignore[import-not-found]
        BasisSet,
        SimpleKeyword,
    )
    from opi.input.simple_keywords.method import (  # type: ignore[import-not-found]
        Method,
    )
    from opi.input.structures.structure import (  # type: ignore[import-not-found]
        Structure,
    )
    from opi.output.core import Output  # type: ignore[import-not-found]

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
        memory_MB: int,
        simple_keywords: Sequence[SimpleKeyword],
        blocks: Sequence[Block],
    ) -> Calculator:
        orca_work_dir: Final = work_dir.make_subdir("orca_mf")
        geometry_path: Final = orca_work_dir / "geometry.xyz"

        # Call to `Cartesian.from_pyscf` specifies unit = "angstrom" when calling
        # `atom_coords`, ensuring that the coordinates are converted appropriately.
        Cartesian.from_pyscf(mol).to_xyz(geometry_path)

        calc = Calculator(basename="mf_calculation", working_dir=orca_work_dir.path)
        calc.structure = Structure.from_xyz(
            geometry_path, charge=mol.charge, multiplicity=mol.multiplicity
        )

        calc.input.add_simple_keywords(*([Method.HF] + list(simple_keywords)))
        calc.input.add_blocks(*blocks)

        # > Define number of CPUs for the calculation
        calc.input.ncores = n_procs
        # > Define memory (in MBs) per CPU for the calculation (%maxcore keyword)
        calc.input.memory = memory_MB

        return calc

    def get_orca_basis(mol: Mole) -> SimpleKeyword:
        """Translate the basis in ``mol`` to an ORCA basis label."""
        try:
            return PYSCF_TO_ORCA_BASIS[mol.basis]
        except KeyError:
            raise NotImplementedError(
                f"PYSCF basis set {mol.basis} is not supported.\n"
                "This is either because it genuinely does not exist in ORCA,\n"
                "or because the translation is not defined.\n"
                "If you think that the basis exists in\n"
                "pyscf and ORCA, then raise an issue at the quemb GitHub repository."
            )

    def get_mf_orca(
        mol: Mole,
        work_dir: WorkDir,
        n_procs: int,
        memory_MB: int,
        simple_keywords: Sequence[SimpleKeyword],
        blocks: Sequence[Block],
    ) -> RHF:
        """Compute a mean field object via orca.

        Use the


        You can use the :func:`~get_orca_basis` function to translate
        a pyscf basis label to ORCA
        A "normal" Hartree Fock calculation can be invoked via

        >>> get_mf_orca(
        >>>     mol,
        >>>     workdir,
        >>>     n_procs=1,
        >>>     memory_MB=16000,
        >>>     simple_keywords=[],
        >>>     blocks=[BlockBasis(basis=get_orca_basis(mol))],
        >>> )

        While a ``RIJK`` calculation with parallelisation would be:

        >>> from opi.input.blocks.block_basis import BlockBasis
        >>> from opi.input.simple_keywords import Approximation, SimpleKeyword
        >>> get_mf_orca(
        >>>     mol,
        >>>     workdir,
        >>>     n_procs=4,
        >>>     memory_MB=16000,
        >>>     simple_keywords=[Approximation.RIJK],
        >>>     blocks=[
        >>>         BlockBasis(basis=get_orca_basis(mol), auxjk=SimpleKeyword("def2/jk"))
        >>>     ],
        >>> )
        """  # noqa: E501
        calc = _prepare_orca_calc(
            mol, work_dir, n_procs, memory_MB, simple_keywords, blocks
        )
        logger.debug("Writing ORCA input")
        calc.write_input()
        logger.info("Starting ORCA calculation")
        calc.run()
        output = calc.get_output()

        if not output.terminated_normally():
            raise RuntimeError(
                "ORCA terminated with errors.\n"
                f"The output file can be found at {output.get_file('.out')}"
            )
        logger.info("ORCA calculation finished successfully")

        return _parse_orca_into_mf(mol, output)

    ORCA_AVAILABLE = True

except ImportError:

    def get_mf_orca(
        mol: Mole,
        work_dir: WorkDir,
        n_procs: int,
        memory_MB: int,
        simple_keywords: Sequence[SimpleKeyword],
        blocks: Sequence[Block],
    ) -> RHF:  # type: ignore[return-type]
        unused(mol, work_dir, n_procs, memory_MB, simple_keywords, blocks)
        raise ImportError("ORCA and the ORCA python interface have to be available.")

    def get_orca_basis(mol: Mole) -> SimpleKeyword:
        unused(mol)
        raise ImportError("ORCA and the ORCA python interface have to be available.")

    ORCA_AVAILABLE = False
