# Author(s): Beck Hanscam

import copy
import inspect
import os
from dataclasses import dataclass, field
import shutil # HYL added
import h5py
import numpy as np
from pyscf import cc, gto, lib, scf

from quemb.molbe import BE, fragmentate
from quemb.molbe.chemfrag import Fragmented
from quemb.molbe.helper import get_eri, get_scfObj
from quemb.molbe.mbe import BEArgs
from quemb.shared.manage_scratch import WorkDir
from quemb.molbe.solver import be_func # HYL added
from quemb.molbe.be_parallel import be_func_parallel # HYL added

# HYL added below -- likely not all needed, but just in case for now
import logging
import os
import pickle
from typing import Dict, Final, List, Literal, TypeAlias
from warnings import warn

import h5py
import numpy as np
from attrs import define
from numpy import (
    allclose,
    argmax,
    array,
    concatenate,
    delete,
    diag,
    diag_indices,
    einsum,
    eye,
    float64,
    floating,
    hstack,
    isnan,
    load,
    ndarray,
    save,
    shape,
    sqrt,
    where,
    zeros,
    zeros_like,
)
from numpy.linalg import eigh, multi_dot, norm, svd
from pyscf import ao2mo, scf
from pyscf.gto import Mole
from scipy.optimize import linear_sum_assignment
from typing_extensions import assert_never

from quemb.molbe.be_parallel import be_func_parallel
from quemb.molbe.chemfrag import ChemGenArgs
from quemb.molbe.eri_onthefly import integral_direct_DF
from quemb.molbe.eri_sparse_DF import (
    transform_sparse_DF_integral_cpu,
)
from quemb.molbe.fragment import FragPart
from quemb.molbe.helper import get_eri, get_scfObj
from quemb.molbe.lo import (
    IAO_LocMethods,
    LocMethods,
    get_iao,
    get_loc,
    get_pao,
    get_xovlp,
    remove_core_mo,
)
from quemb.molbe.misc import print_energy_cumulant, print_energy_noncumulant
from quemb.molbe.numerical_jac import compute_numerical_jacobian
from quemb.molbe.opt import BEOPT
from quemb.molbe.pfrag import Frags, union_of_frag_MOs_and_index
from quemb.molbe.solver import Solvers, UserSolverArgs, be_func
from quemb.shared.external.eom_qchem_parser import dyson_parser, dyson_parser_ea
from quemb.shared.external.lo_helper import (
    get_aoind_by_atom,
    reorder_by_atom_,
)
from quemb.shared.external.optqn import (
    get_be_error_jacobian as _ext_get_be_error_jacobian,
)
from quemb.shared.helper import copy_docstring, ensure, ncore_, timer, unused
from quemb.shared.manage_scratch import WorkDir
from quemb.shared.typing import Matrix, PathLike
# HYL added above -- likely not all needed, but just in case for now

def energy_hf(mol, energy_args=None, fd_info=None):
    r"""Compute the restricted Hartree-Fock total energy

    Parameters
    ----------
    mol : object
        Molecule object defining the geometry, basis, charge, and spin.
    energy_args: optional
        User defined arguments for energy calculation.
    fd_info: FDinfo, optional
        Finite difference metadata describing the displacement relative
        to the current reference geometry.

    Returns
    ------
    float
        Converged RHF total energy in Hartree
    """
    if energy_args is None:
        pass
    if fd_info is not None:
        if fd_info.ref_mol is None:
            raise RuntimeError("missing finite difference reference geometry.")

    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    return mf.e_tot


def be_ref_data(mol, energy_args=None):
    r"""Build reference-geometry data needed by BE energy functions.

    Parameters
    ----------
    mol : object
        Molecule object defining the geometry, basis, charge, and spin.
    energy_args: BEArgs, optional
        User defined arguments for BE calculation.

    Returns
    ------
    dict
        Dictionary containing reference-geometry data needed by ``energy_be``.
        The ``"ref_fobj"`` entry stores the fragmentate object built from ``mol``.
    """
    if energy_args is None:
        energy_args = BEArgs()

    ref_fobj = fragmentate(
        mol=mol,
        n_BE=energy_args.n_BE,
        frag_type=energy_args.frag_type,
        iao_valence_basis=energy_args.iao_valence_basis,
        frozen_core=energy_args.frozen_core,
        additional_args=energy_args.additional_args,
    )

    return {"ref_fobj": ref_fobj}


def be_frag_ref_data(
        mol, 
        energy_args=None,
        mf=None, # HYL added: this is default, but needed to use density_fit()
        conv_tol=1e-6, # HYL added: this is default to not affect others' code; I'll tighten when calling
        conv_tol_normt=1e-5, # HYL added: this is default to not affect others' code; I'll tighten when calling
        max_iter=500 # HYL added: this is default to not affect others' code; I'll tighten when calling
        ):
    r"""Build reference-geometry data needed by force embedding energy functions.

    Parameters
    ----------
    mol : object
        Molecule object defining the geometry, basis, charge, and spin.
    energy_args: BEArgs, optional
        User defined arguments for BE calculation.

    Returns
    ------
    dict
        Dictionary containing reference-geometry data needed by ``energy_be_frag``.
        The ``"ref_fobj"`` entry stores the fragmentate object built from ``mol``.
        The ``"ref_mybe"`` entry stores the BE object.
        The ``"frag_per_atom"`` entry stores the fragment index for each atom.
    """
    if energy_args is None:
        energy_args = BEArgs()

    if mf is None:
        mf = scf.RHF(mol)
        mf.verbose = 0
        mf.kernel()

    ref_fobj = fragmentate(
        mol=mol,
        n_BE=energy_args.n_BE,
        frag_type=energy_args.frag_type,
        iao_valence_basis=energy_args.iao_valence_basis,
        frozen_core=energy_args.frozen_core,
        additional_args=energy_args.additional_args,
    )

    ref_mybe = BE(
        mf,
        ref_fobj,
        lo_method=energy_args.lo_method,
        int_transform=energy_args.int_transform,
        auxbasis=energy_args.auxbasis,
        nproc=energy_args.nproc,
        ompnum=energy_args.ompnum,
        initialize_fragment_idx=energy_args.initialize_fragment_idx,
    )

    # HYL added begins here
    ref_mybe.conv_tol = conv_tol
    ref_mybe.max_iter = max_iter 
    ref_mybe.conv_tol_normt = conv_tol_normt
    # HYL added ends here
    
    fragmented = Fragmented.from_mole(
        mol, n_BE=energy_args.n_BE, h_treatment=energy_args.additional_args.h_treatment
    )
    frag_per_atom = fragmented.get_frag_per_atom()

    ref_dict = {
        "ref_mol": mol.copy(),
        "ref_fobj": ref_fobj,
        "ref_mybe": ref_mybe,
        "frag_per_atom": frag_per_atom,
    }

    return ref_dict


def energy_be(mol, energy_args=None, fd_info=None):
    r"""Compute the BEn total energy

    Parameters
    ----------
    mol : object
        Molecule object defining the geometry, basis, charge, and spin.
    energy_args: BEArgs, optional
        User defined arguments for BE calculation.
    fd_info: FDinfo, optional
        Finite difference metadata describing the displacement relative
        to the current reference geometry.

    Returns
    ------
    float
        Converged BE total energy in Hartree
    """
    if energy_args is None:
        energy_args = BEArgs()

    if fd_info is None:
        fobj = fragmentate(
            mol=mol,
            n_BE=energy_args.n_BE,
            frag_type=energy_args.frag_type,
            iao_valence_basis=energy_args.iao_valence_basis,
            frozen_core=energy_args.frozen_core,
            additional_args=energy_args.additional_args,
        )
    else:
        if fd_info.ref_mol is None:
            raise RuntimeError("missing finite difference reference geometry.")

        try:
            fobj = fd_info.ref_data["ref_fobj"]
        except KeyError as exc:
            raise RuntimeError("missing reference BE fragmentate object.") from exc

    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()

    mybe = BE(
        mf,
        fobj,
        lo_method=energy_args.lo_method,
        int_transform=energy_args.int_transform,
        auxbasis=energy_args.auxbasis,
        nproc=energy_args.nproc,
        ompnum=energy_args.ompnum,
        initialize_fragment_idx=energy_args.initialize_fragment_idx,
    )

    if energy_args.optimize:
        mybe.optimize(
            solver=energy_args.solver,
            use_cumulant=energy_args.use_cumulant,
            nproc=energy_args.nproc,
            ompnum=energy_args.ompnum,
            only_chem=energy_args.only_chem,
            method=energy_args.method,
            conv_tol=energy_args.conv_tol,
            relax_density=energy_args.relax_density,
            jac_solver=energy_args.jac_solver,
            max_iter=energy_args.max_iter,
            trust_region=energy_args.trust_region,
            step_size=energy_args.step_size,
        )
    else:
        mybe.oneshot(
            solver=energy_args.solver,
            use_cumulant=energy_args.use_cumulant,
            nproc=energy_args.nproc,
            ompnum=energy_args.ompnum,
        )

    return mybe.ebe_tot


def energy_be_frag(mol, energy_args=None, fd_info=None):
    r"""Compute the BEn fragment energy

    Parameters
    ----------
    mol : object
        Molecule object defining the geometry, basis, charge, and spin.
    energy_args: BEArgs, optional
        User defined arguments for BE calculation.
    fd_info: FDinfo, optional
        Finite difference metadata describing the displacement relative
        to the current reference geometry.

    Returns
    ------
    float
        Converged BE fragment energy in Hartree
    """
    if energy_args is None:
        energy_args = BEArgs()

    if fd_info.kind == "multi_displacement":
        raise RuntimeError(
            "energy_be_frag currently supports only single finite-difference "
            f"displacements; got {fd_info.kind!r}"
        )
    if fd_info is None:
        raise RuntimeError("missing finite difference displacement info.")
    else:
        if fd_info.ref_mol is None:
            raise RuntimeError("missing finite difference reference geometry.")

        try:
            ref_fobj = fd_info.ref_data["ref_fobj"]
            ref_mybe = fd_info.ref_data["ref_mybe"]
            frag_per_atom = fd_info.ref_data["frag_per_atom"]
        except KeyError as exc:
            raise RuntimeError("missing reference BE info.") from exc

    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()

    if fd_info.kind in ("reference", "scanner_point"):
        # placeholder energy to allow Gradients.as_scanner()
        # to return (energy, gradient)
        return mf.e_tot

    assert len(fd_info.atom_idx) == 1, (
        "Expected fd_info.atom_idx to have length 1, "
        f"but got {len(fd_info.atom_idx)}: {fd_info.atom_idx}"
    )

    assert len(fd_info.axis_idx) == 1, (
        "Expected fd_info.axis_idx to have length 1, "
        f"but got {len(fd_info.axis_idx)}: {fd_info.axis_idx}"
    )

    assert len(fd_info.delta_bohr) == 1, (
        "Expected fd_info.delta_bohr to have length 1, "
        f"but got {len(fd_info.delta_bohr)}: {fd_info.delta_bohr}"
    )

    atom_idx = fd_info.atom_idx[0]
    axis_idx = fd_info.axis_idx[0]
    delta = fd_info.delta_bohr[0]
    frag_idx = frag_per_atom[fd_info.atom_idx[0]]

    axis_label = ("x", "y", "z")[axis_idx]
    sign_label = "p" if delta > 0 else "m"

    redo_tag = f"redo_frag{frag_idx}_atom{atom_idx}_{sign_label}{axis_label}"

    mybe = BE(
        mf,
        ref_fobj,
        lo_method=energy_args.lo_method,
        int_transform=energy_args.int_transform,
        auxbasis=energy_args.auxbasis,
        nproc=energy_args.nproc,
        ompnum=energy_args.ompnum,
        initialize_fragment_idx=[frag_idx],
    )

    S_cross = gto.intor_cross("int1e_ovlp", mol, fd_info.ref_mol)

    # keep the original fragment object untouched
    orig_fobj = mybe.Fobjs[frag_idx]

    # Create a dedicated temporary scratch directory for the re-done ERIs
    redo_scratch = WorkDir(
        mybe.scratch_dir / redo_tag,
        cleanup_at_end=True,
        ensure_empty=True,
    )
    tmp_eri_file = redo_scratch / f"{redo_tag}.h5"

    # Work on an independent fragment object for this displaced geometry
    # Use deepcopy since _initialize_fragments changes mybe.ebe_hf
    fobj = copy.deepcopy(orig_fobj)

    try:
        fobj.TA = np.linalg.inv(mybe.S) @ S_cross @ ref_mybe.Fobjs[frag_idx].TA
        fobj.dname = redo_tag
        fobj.eri_file = tmp_eri_file

        # _eri_transform() and _initialize_fragments() operate through mybe.Fobjs,
        # so temporarily point this fragment index to the copied fragment
        mybe.Fobjs[frag_idx] = fobj

        with h5py.File(tmp_eri_file, "w") as file_eri:
            mybe._eri_transform(
                energy_args.int_transform,
                mf._eri,
                file_eri,
                [frag_idx],
            )
            mybe._initialize_fragments(file_eri, False, [frag_idx])

        # Use the reinitialized copied fragment
        fobj = mybe.Fobjs[frag_idx]

        eri = get_eri(fobj.dname, fobj.nao, eri_file=tmp_eri_file)
        fobj._mf = get_scfObj(
            fobj.fock + fobj.heff,
            eri,
            fobj.nsocc,
            dm0=fobj.dm0.copy(),
        )

        mc = cc.CCSD(fobj._mf)
        mc.verbose = 0
        mc.incore_complete = True

        eri_embmo = mc.ao2mo()
        eri_embmo.mo_energy = fobj._mf.mo_energy
        eri_embmo.fock = np.diag(fobj._mf.mo_energy)

        mc.kernel(eris=eri_embmo)
        energy = mf.e_tot + mc.e_tot - fobj._mf.e_tot

    finally:
        mybe.Fobjs[frag_idx] = orig_fobj

        redo_scratch.cleanup(ignore_error=True)

    return energy

# HYL added begins -- create_qchem_input_directories()
def create_qchem_input_directories(
    frag_number,
    suffix,
    n_occ,
    n_env,
    mol_perturbed,
    template_in="eom.in",
    template_submit="submit.sh"
):
    """
    Generates isolated QChem calculation directories for batch execution.
    Adds perturbed coordinates via placeholder substitution and sets frozen core/virtual.
    """
    # Create the unique QChem calculation folder
    qchem_dir = f"qchem_fragment_{frag_number}_{suffix}"
    os.makedirs(qchem_dir, exist_ok=True)

    # Format the physical coordinates from Bohr to Angstroms for QChem
    perturbed_coordinates = []
    for i in range(mol_perturbed.natm):
        symb = mol_perturbed.atom_symbol(i)
        coords = mol_perturbed.atom_coord(i) * 0.5291772109  # Bohr to Angstrom conversion factor
        # Format: Element   X_coord   Y_coord   Z_coord
        perturbed_coordinates.append(f"   {symb}   {coords[0]:.10f}   {coords[1]:.10f}   {coords[2]:.10f}")

    molecule_block = "\n".join(perturbed_coordinates)

    # Read template input file
    with open(template_in, "r") as f:
        content = f.read()

    # Add the coordinates directly into the placeholder slot
    # NOTE: this placeholder must exist in template for correct substitution
    content = content.replace("__COORDINATES_PLACEHOLDER__", molecule_block)

    # Rewrite the n_frozen_core and n_frozen_virtual in QChem input
    lines = content.split("\n")
    for idx, line in enumerate(lines):
        if "n_frozen_core" in line:
            lines[idx] = f"n_frozen_core {n_occ}"
        elif "n_frozen_virtual" in line:
            lines[idx] = f"n_frozen_virtual {n_env}"
    content = "\n".join(lines)

    # Write out the completed input file
    with open(os.path.join(qchem_dir, "eom.in"), "w") as f:
        f.write(content)

    # Prepare the submit script with updated scratch folder name
    with open(template_submit, "r") as f:
        submit_content = f.read()

    unique_scratch_name = f"scratch_{frag_number}_{suffix}"
    submit_content = submit_content.replace("scratch_template", unique_scratch_name)

    with open(os.path.join(qchem_dir, "submit.sh"), "w") as f:
        f.write(submit_content)

    # Copy the orbital-aligned binary files (99.0, 58.0, 53.0) into the folder
    # TO DO: HYL currently copying to be safe, but for storage purposes, I'd like to move it or delete after successful copying
    source_scratch = f"files_EOM/scratch_fragment_{frag_number}_{suffix}"
    destination_scratch = os.path.join(qchem_dir, unique_scratch_name)

    if os.path.exists(destination_scratch):
        shutil.rmtree(destination_scratch)
    shutil.copytree(source_scratch, destination_scratch)

    print(f"Created isolated input directory: {qchem_dir}")
# HYL added ends -- create_qchem_input_directories()

# HYL added begins -- run_qchem_alignment_driver()
def run_qchem_alignment_driver(
    mol,
    energy_args=None,
    fd_info=None,
    conv_tol=1e-12,
    direct_scf_tol=1e-13,
    max_cycle=500,
    oneshot=False,
    solver="CCSD",
    only_chem=True,
    template_in="eom.in",
    template_submit="submit.sh"
):
    """
    Generates aligned Q-Chem inputs for finite difference steps.

    Parameters:
    mol (gto.Mole): The active molecule object (either unperturbed baseline or displaced coordinate state).
    energy_args (BEArgs): Contains BE arguments.
    fd_info (FDinfo): Tracks displacement indices, unperturbed references, and mapping arrays.
    conv_tol (float): Tightened energy convergence metric assigned to all global and fragment SCF steps.
    direct_scf_tol (float): Tightened two-electron integral screening threshold preventing sub-coordinate data loss.
    max_cycle (int): Maximum number of permissible iteration cycles allocated for the SCF solvers.
    template_in: name of template QChem input file passed to create_qchem_input_directories
    template_submit: name of template QChem submission file passed to create_qchem_input_directories

    Returns
    -------
    boolean:
        True upon successful creation of QChem input files
        False if reference fd_info as input (no inputs created)

    NOTE: only supports single-atom displacement now
    """
    # ========================
    # same as energy_be_frag()
    # ========================
    if energy_args is None:
        energy_args = BEArgs()

    if fd_info.kind == "multi_displacement":
        raise RuntimeError(
            "energy_be_frag currently supports only single finite-difference "
            f"displacements; got {fd_info.kind!r}"
        )
    if fd_info is None:
        raise RuntimeError("missing finite difference displacement info.")
    else:
        if fd_info.ref_mol is None:
            raise RuntimeError("missing finite difference reference geometry.")

        try:
            ref_fobj = fd_info.ref_data["ref_fobj"]
            ref_mybe = fd_info.ref_data["ref_mybe"]
            frag_per_atom = fd_info.ref_data["frag_per_atom"]
        except KeyError as exc:
            raise RuntimeError("missing reference BE info.") from exc
    
    # ====================
    # HYL added / modified
    # ====================
    # energy_be_frag() has placeholder energy for reference,
    # but since I want it to generate QChem files,
    # returning False

    if fd_info.kind == "reference":
        print("Warning: Calling reference on run_qchem_alignment_driver -- this should be done with build_tight_reference_data.")
        return False

    # If ends up here, PERTURBED fragment
    # adds tightened threshold & density_fit() option compared to energy_be_frag()

    mf = scf.RHF(mol) # perturbed mol will be passed in here
    is_df = energy_args is not None and "DF" in getattr(energy_args, "int_transform", "")
    if is_df:
        mf = mf.density_fit()
    mf.conv_tol = conv_tol
    mf.direct_scf_tol = direct_scf_tol
    mf.max_cycle = max_cycle
    mf.verbose = 0
    mf.kernel()

    # ========================
    # same as energy_be_frag()
    # ========================

    assert len(fd_info.atom_idx) == 1, (
        "Expected fd_info.atom_idx to have length 1, "
        f"but got {len(fd_info.atom_idx)}: {fd_info.atom_idx}"
    )

    assert len(fd_info.axis_idx) == 1, (
        "Expected fd_info.axis_idx to have length 1, "
        f"but got {len(fd_info.axis_idx)}: {fd_info.axis_idx}"
    )

    assert len(fd_info.delta_bohr) == 1, (
        "Expected fd_info.delta_bohr to have length 1, "
        f"but got {len(fd_info.delta_bohr)}: {fd_info.delta_bohr}"
    )

    # ====================
    # HYL added / modified
    # ====================
    
    # allows for 1 in addition to [1]
    '''
    atom_idx = fd_info.atom_idx[0]
    axis_idx = fd_info.axis_idx[0]
    delta = fd_info.delta_bohr[0]
    '''

    atom_idx = fd_info.atom_idx[0] if isinstance(fd_info.atom_idx, list) else fd_info.atom_idx
    axis_idx = fd_info.axis_idx[0] if isinstance(fd_info.axis_idx, list) else fd_info.axis_idx
    delta = fd_info.delta_bohr[0] if isinstance(fd_info.delta_bohr, list) else fd_info.delta_bohr

    frag_idx = frag_per_atom[atom_idx]

    # ========================
    # same as energy_be_frag()
    # ========================

    axis_label = ("x", "y", "z")[axis_idx]
    sign_label = "p" if delta > 0 else "m"

    # ====================
    # HYL added / modified
    # ====================

    # suffix to create separate dir for diff atom displacement
    # I think energy_be_frag() uses this naming convention too, 
    # just not defined as a variable there
    displacement_suffix = f"atom{atom_idx}_{sign_label}{axis_label}"

    c_thr_bath = getattr(energy_args, 'thr_bath', 1.0e-10) # read the value from energy_args, falling back to 1.0e-10 if it doesn't exist

    redo_tag = f"redo_frag{frag_idx}_{displacement_suffix}" # to create separate folder for new ERIs

    mybe = BE(
        mf,
        ref_fobj,
        lo_method=energy_args.lo_method,
        int_transform=energy_args.int_transform,
        auxbasis=energy_args.auxbasis,
        nproc=energy_args.nproc,
        ompnum=energy_args.ompnum,
        initialize_fragment_idx=[frag_idx],
        thr_bath=c_thr_bath # added in case want to tighten later
    )

    # ========================
    # same as energy_be_frag()
    # ========================

    S_cross = gto.intor_cross("int1e_ovlp", mol, fd_info.ref_mol)

    # keep the original fragment object untouched
    orig_fobj = mybe.Fobjs[frag_idx]

    # Create a dedicated temporary scratch directory for the re-done ERIs
    redo_scratch = WorkDir(
        mybe.scratch_dir / redo_tag,
        cleanup_at_end=True,
        ensure_empty=True,
    )
    tmp_eri_file = redo_scratch / f"{redo_tag}.h5"

    # Work on an independent fragment object for this displaced geometry
    # Use deepcopy since _initialize_fragments changes mybe.ebe_hf
    fobj = copy.deepcopy(orig_fobj)

    try:
        fobj.TA = np.linalg.inv(mybe.S) @ S_cross @ ref_mybe.Fobjs[frag_idx].TA
        fobj.dname = redo_tag
        fobj.eri_file = tmp_eri_file

        # _eri_transform() and _initialize_fragments() operate through mybe.Fobjs,
        # so temporarily point this fragment index to the copied fragment
        mybe.Fobjs[frag_idx] = fobj

        with h5py.File(tmp_eri_file, "w") as file_eri:
            mybe._eri_transform(
                energy_args.int_transform,
                mf._eri,
                file_eri,
                [frag_idx],
            )
            mybe._initialize_fragments(file_eri, False, [frag_idx])

        # Use the reinitialized copied fragment
        fobj = mybe.Fobjs[frag_idx]

        # NOTE: the following are needed for qchem_setup_gen()
        eri = get_eri(fobj.dname, fobj.nao, eri_file=tmp_eri_file)
        fobj._mf = get_scfObj(
            fobj.fock + fobj.heff,
            eri,
            fobj.nsocc,
            dm0=fobj.dm0.copy(),
        )

        # ==================================================
        # HYL added -- mostly borrowed from qchem_setup()
        # qchem_setup() conveniently loops through all
        # fragments to create input files for each
        # so I can borrow the code inside the for-loop:
        # qchem_setup() calls fobj._mf above just mf
        # qchem_setup() uses eri name same as above
        # qchem_setup() fobj loops through self.Fobjs,
        # which conveniently is the fobj defined above
        # qchem_setup() self is mybe in this case
        # qchem_setup() frag_number is frag_idx in this case
        # ==================================================

        # I belive fragment CCSD energy is computed here in energy_be_frag(), which I don't need for the purposes of generating QChem scratch files -- copied and commented out below for reference
        '''
        mc = cc.CCSD(fobj._mf)
        mc.verbose = 0
        mc.incore_complete = True

        eri_embmo = mc.ao2mo()
        eri_embmo.mo_energy = fobj._mf.mo_energy
        eri_embmo.fock = np.diag(fobj._mf.mo_energy)

        mc.kernel(eris=eri_embmo)
        energy = mf.e_tot + mc.e_tot - fobj._mf.e_tot
        '''
        # Instead, I need to create the QChem scratch files

        # Note: Alexa reads in full system fock matrix from files_EOM in qchem_setup(self) code, and she saved it in class BE def __init__ via the following
        '''
        if not os.path.exists("files_EOM"):
            os.makedirs("files_EOM")
        with open("files_EOM/full_syst_fock.npy", "wb") as f:
            save(f, mf.get_fock())
        '''
        # Thus, I will use the equivalence of mf.get_fock() for obtaining the full system fock matrix
        # Note: since this is in class BE def __init__, the mf in mf.get_fock() is the mf object (also called mf above) used to create the BE object (in this case mybe) -- note this mf is NOT the same as the mf in qchem_setup()

        print("QChem:")
        print("Exporting files 99.0, 58.0, 53.0 to Q-Chem for EOM-CCSD calculation")
        print("Fragment number: ", frag_idx)

        ###numbers of electrons in:
        n_mo_full_syst = shape(fobj.TA)[0]
        occ_tot = mybe.Nocc  ###full system
        SO_tot = shape(fobj.TA)[1]
        SO_occ = fobj.nsocc  ###Schmidt space

        print("n_mo_full_syst is:")
        print(n_mo_full_syst)
        print("Number of electrons in full system (occ_tot):")
        print(occ_tot)
        print("Number of electrons in Schmidt space (SO_tot):")
        print(SO_tot)

        env_occ = occ_tot - SO_occ + mybe.ncore
        print("Qchem: set n_frozen_core")
        print("Number occupied environment: ", env_occ)
        
        env_virt = n_mo_full_syst - SO_tot - env_occ
        print("Qchem: set n_frozen_virtual")
        print("Number virtual environment: ", env_virt)

        ###File 99.0 - energy file in Qchem
        
        energy = fobj._mf.kernel()
        
        print("SCF energy is: ", energy)
        
        energy_array = zeros(12)
        # placeholder value - exact value doesn't matter
        energy_array[0] = 3.7617453591977221e02
        energy_array[1] = energy
        energy_array[11] = energy
        
        ###File 58.0 - Fock matrix file in Qchem
        ###Full system Fock matrix (AO basis)
        # HYL added
        # instead of the following code in qchem_setup() 
        # fock_full_syst = load("files_EOM/full_syst_fock.npy")
        # I'm just going to call mf.get_fock() -- see above

        fock_full_syst = mf.get_fock()

        flat_fock = array(fock_full_syst.flatten(), dtype=float64)
        full_fock = concatenate((flat_fock, flat_fock), axis=None)

        ###File 53.0 - MO coefficient matrix
        ###TA: AOxSO; mf.mo_coeff: SOxMO
        TA_after_HF = fobj.TA @ fobj._mf.mo_coeff
        
        ###Pad TA matrix with orthogonal vectors
        ###use it as MO coefficient matrix for Qchem
        
        # m=n_orb_total (frag+bath+env)
        # n=n_frag+n_bath
        m, n = TA_after_HF.shape
        
        # compute the orthonormal basis for the null space of TA.T
        # do SVD
        _, _, vh = svd(TA_after_HF.T, full_matrices=True)
        
        # take the (m-n) right singular vectors orthogonal to TA.T
        orthogonal_vectors = vh[n:m].T
        
        # pad the original matrix with the orthogonal vectors
        TA_full_pyscf = hstack(
            (
                orthogonal_vectors[:, :env_occ],
                TA_after_HF,
                orthogonal_vectors[:, env_occ:],
            )
        )
        
        TA_full_qchem = TA_full_pyscf.T
        
        flat_mos = array(TA_full_qchem.flatten(), dtype=float64)

        ###MO energies needed at the end of file 53.0
        mo_energies = zeros(n_mo_full_syst)
        # set to arbitrary low number to avoid recanonicalization in Qchem
        mo_energies[:env_occ] = -1000
        mo_energies[env_occ : env_occ + SO_tot] = fobj._mf.mo_energy
        mo_energies[env_occ + SO_tot :] = 1000
        
        print("Fragment: ", frag_idx)
        print(fobj._mf.mo_energy)
        print("SO occ: ", SO_occ)
        
        full_mo_array = concatenate(
            (flat_mos, flat_mos, mo_energies, mo_energies), axis=None
        )
        
        print("MOs: ")
        print(flat_mos)

        # Since my naming convention is slightly different for finite difference purposes, the following from qchem_setup() isn't used (note: if every need to use in the future, change frag_number to frag_idx for consistency with the current function)
        '''
        if not os.path.exists("files_EOM/scratch_fragment_" + str(frag_number)):
            os.makedirs("files_EOM/scratch_fragment_" + str(frag_number))
        energy_array.tofile(
            "files_EOM/scratch_fragment_" + str(frag_number) + "/99.0"
        )
        full_fock.tofile("files_EOM/scratch_fragment_" + str(frag_number) + "/58.0")
        full_mo_array.tofile(
            "files_EOM/scratch_fragment_" + str(frag_number) + "/53.0"
        )
        '''
        # Instead, I use these paths
        if displacement_suffix is not None:
            scratch_path = f"files_EOM/scratch_fragment_{frag_idx}_{displacement_suffix}"
        else: 
            print("Warning: suffix is None, so default to reference -- note even for reference, one should assign suffix as reference rather than None to avoid ambiguity, although they result in the same folder created.")
            scratch_path = f"files_EOM/scratch_fragment_{frag_idx}_reference"

        if not os.path.exists(scratch_path):
            os.makedirs(scratch_path)

        # Update to use the new unique scratch_path
        energy_array.tofile(os.path.join(scratch_path, "99.0"))
        full_fock.tofile(os.path.join(scratch_path, "58.0"))
        full_mo_array.tofile(os.path.join(scratch_path, "53.0"))

        create_qchem_input_directories(
            frag_number=frag_idx,
            suffix=displacement_suffix,
            n_occ=env_occ,
            n_env=env_virt,
            mol_perturbed=mol,
            template_in=template_in,
            template_submit=template_submit
        )

    # ========================
    # same as energy_be_frag()
    # ========================

    finally:
        # restore the unperturbed, original fragment for next cycle
        mybe.Fobjs[frag_idx] = orig_fobj
        # clean up temporary ERI .h5
        redo_scratch.cleanup(ignore_error=True)
        print(f"Successfully exported aligned QChem files with suffix '{displacement_suffix}'.") # HYL added
    return True # HYL added / changed; energy_be_frag() returns energy

# HYL added ends -- run_qchem_alignment_driver()

@dataclass
class FDinfo:
    """Container for finite difference metadata."""

    kind: str = "reference"

    atom_idx: list[int] | None = field(default_factory=list)
    axis_idx: list[int] | None = field(default_factory=list)

    delta_bohr: list[float] | None = field(default_factory=list)

    ref_mol: gto.Mole | None = None
    ref_data: dict = field(default_factory=dict)


class Energy(lib.StreamObject):
    r"""PySCF-style wrapper for a custom molecular energy function.

    This class provides a minimal interface around an arbitrary energy function
    ``energy_func(mol)``. It is intended to be compatible with PySCF utilities that
    expect an object with ``mol``, ``kernel()``, and ``as_scanner()`` methods, such as
    ``pyscf.tools.finite_diff``.

    This class supplies repeated single-point energies that can be used by finite
    difference gradient or Hessian drivers.
    """

    def __init__(
        self,
        mol,
        energy_func,
        displacement=1e-4,
        energy_args=None,
        ref_data_func=None,
    ):
        r"""Initialize the custom energy wrapper.

        Parameters
        ----------
        mol : object
            Reference molecule.
        energy_func :
            Callable function with signature ``energy_func(mol) -> float`` returning
            the total energy in Hartree. Should optionally accept ``fd_info`` and
            ``energy_args`` for additional keyword arguments.
        displacement : float, optional
            Finite difference displacement in Bohr, default is 1e-4.
        energy_args : optional
            Additional keyword arguments passed to ``energy_func``.
        ref_data_func: optional
            Callable function with signature ``ref_data_func(mol) -> dict`` returning
            a dictionary containing the necessary reference geometry info for
            ``energy_func``. Should optionally accept additional keyword arguments.
        """
        self.mol = mol
        self.energy_func = energy_func
        self.energy_args = energy_args
        self.e_tot = None
        self.displacement = displacement
        self.ref_data_func = ref_data_func

        # Attributes expected by PySCF finite-difference assertions
        # These do not control convergence for the custom method
        self.conv_tol = 1e-12
        self.converged = True

    def kernel(self, mol=None, fd_info=None):
        r"""Evaluate the energy for a molecule.

        Parameters
        ----------
        mol : object, optional
            Molecule at which to evaluate the energy.
        fd_info: FDinfo, optional
            Finite difference metadata describing the displacement relative
            to the current reference geometry.

        Returns
        ------
        float
            Total energy in Hartree
        """
        if mol is not None:
            self.mol = mol

        if fd_info is None:
            ref_data = (
                self.ref_data_func(self.mol, energy_args=self.energy_args)
                if self.ref_data_func is not None
                else {}
            )
            fd_info = FDinfo(
                kind="reference",
                atom_idx=[],
                axis_idx=[],
                delta_bohr=[0],
                ref_mol=self.mol.copy(),
                ref_data=ref_data,
            )

        self.e_tot = self.energy_func(
            self.mol, energy_args=self.energy_args, fd_info=fd_info
        )
        return self.e_tot

    def as_scanner(self):
        r"""Return a PySCF-compatible energy scanner.

        The returned scanner is callable as ``scanner(mol)`` and evaluates
        ``energy_func`` for each supplied geometry. This mirrors the behavor of
        PySCF scanner objects used in geometry optimization and finite difference
        calculations.

        Returns
        ------
        object
            Callable scanner object returning total energies in Hartree.
        """
        parent = self

        def called_from_pyscf_finite_diff():
            for frame in inspect.stack()[1:8]:
                fname = os.path.basename(frame.filename)
                if fname == "finite_diff.py":
                    return True
            return False

        class Scanner(lib.SinglePointScanner, lib.StreamObject):
            def __init__(self):
                self.ref_coords = parent.mol.atom_coords().copy()
                self.ref_mol = parent.mol.copy()
                self.ref_data = (
                    parent.ref_data_func(self.ref_mol, energy_args=parent.energy_args)
                    if parent.ref_data_func is not None
                    else {}
                )

            def is_fd_probe(self, diff, tol=1e-8):
                """
                Return True if coords look like a finite-difference probe of ref_coords.
                """
                diff_flat = np.ravel(diff)
                nonzero_idx = np.where(np.abs(diff_flat) > 1e-12)[0]

                if len(nonzero_idx) == 0:
                    return True

                for x in diff_flat[nonzero_idx]:
                    n = x / parent.displacement
                    if abs(n - round(n)) > tol:
                        return False

                return True

            def __call__(self, mol):
                coords = mol.atom_coords()
                in_fd = called_from_pyscf_finite_diff()

                if not in_fd:
                    # scanner point: new ref geometry
                    self.ref_coords = coords.copy()
                    self.ref_mol = mol.copy()
                    self.ref_data = (
                        parent.ref_data_func(
                            self.ref_mol, energy_args=parent.energy_args
                        )
                        if parent.ref_data_func is not None
                        else {}
                    )
                    fd_info = FDinfo(
                        kind="scanner_point",
                        atom_idx=[],
                        axis_idx=[],
                        delta_bohr=[0],
                        ref_mol=self.ref_mol.copy(),
                        ref_data=self.ref_data,
                    )
                else:
                    diff = coords - self.ref_coords

                    # check if finite_diff is probing current geometry
                    # reset ref_coords and mol_ref if new geometry
                    if not self.is_fd_probe(diff):
                        self.ref_mol = mol.copy()
                        self.ref_coords = coords.copy()
                        diff = coords - self.ref_coords
                        self.ref_data = (
                            parent.ref_data_func(
                                self.ref_mol, energy_args=parent.energy_args
                            )
                            if parent.ref_data_func is not None
                            else {}
                        )

                    displaced = np.reshape(diff, -1)
                    displaced_idx = np.where(np.abs(displaced) > 1e-12)[0]

                    fd_info = FDinfo(
                        kind="reference",
                        atom_idx=[idx // 3 for idx in displaced_idx],
                        axis_idx=[idx % 3 for idx in displaced_idx],
                        delta_bohr=[displaced[idx] for idx in displaced_idx],
                        ref_mol=self.ref_mol.copy(),
                        ref_data=self.ref_data,
                    )
                    if len(displaced_idx) == 1:
                        fd_info.kind = "single_displacement"
                    elif len(displaced_idx) > 1:
                        fd_info.kind = "multi_displacement"

                parent.mol = mol
                parent.e_tot = parent.energy_func(
                    mol, fd_info=fd_info, energy_args=parent.energy_args
                )

                self.mol = mol
                self.e_tot = parent.e_tot

                return self.e_tot

        scanner = Scanner()
        scanner.mol = parent.mol
        parent._scanner = scanner

        # Attributes expected by PySCF finite-difference assertions
        # These do not control convergence for the custom method
        scanner.conv_tol = 1e-12
        scanner.converged = True

        return scanner
