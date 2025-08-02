import h5py
import numpy as np
from pyscf.gto import Mole
from pyscf.scf.hf import RHF

from quemb.shared.typing import Matrix, PathLike, Vector


def create_mf(
    mol: Mole,
    mo_coeff: Matrix[np.floating],
    mo_energy: Vector[np.floating],
    mo_occ: Vector[np.floating],
    e_tot: float,
) -> RHF:
    """Create a pyscf mean-field object from data **without** running a calculation"""
    mf = RHF(mol)

    mf.mo_coeff = mo_coeff
    mf.mo_energy = mo_energy
    mf.mo_occ = mo_occ
    mf.e_tot = e_tot
    mf.converged = True
    return mf


def get_mf_psycf(mol: Mole) -> RHF:
    "Run an RHF calculation in pyscf"
    mf = RHF(mol)
    mf.kernel()
    return mf


def store_to_hdf5(path: PathLike, mf: RHF) -> None:
    """Store a PySCF RHF object to an HDF5 file."""
    with h5py.File(path, "w") as f:
        mol_group = f.create_group("mol")
        mol = mf.mol

        mol_group.attrs["atom"] = mol.atom
        mol_group.attrs["basis"] = mol.basis
        mol_group.attrs["unit"] = mol.unit
        mol_group.attrs["charge"] = mol.charge
        mol_group.attrs["spin"] = mol.spin

        f.create_dataset("mo_coeff", data=mf.mo_coeff)
        f.create_dataset("mo_energy", data=mf.mo_energy)
        f.create_dataset("mo_occ", data=mf.mo_occ)
        f.attrs["e_tot"] = mf.e_tot


def read_hdf5(path: PathLike) -> RHF:
    """Recreate a PySCF RHF object from an HDF5 file."""
    with h5py.File(path, "r") as f:
        mol_data = f["mol"].attrs
        mol = Mole()
        mol.atom = mol_data["atom"]
        mol.basis = mol_data["basis"]
        mol.unit = mol_data["unit"]
        mol.charge = int(mol_data["charge"])
        mol.spin = int(mol_data["spin"])
        mol.build()

        mo_coeff = f["mo_coeff"][()]
        mo_energy = f["mo_energy"][()]
        mo_occ = f["mo_occ"][()]
        e_tot = float(f.attrs["e_tot"])
    return create_mf(mol, mo_coeff, mo_energy, mo_occ, e_tot)
