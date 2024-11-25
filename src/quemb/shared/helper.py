# Author(s): Oskar Weser

from collections.abc import Callable
from functools import partial

import h5py
import numpy
from numpy import float64
from pyscf import ao2mo, gto, lib, scf
from typing_extensions import Any, List, Optional, ParamSpec, Tuple

from quemb.shared.typing import Matrix, Tensor4D

Parameters = ParamSpec("Parameters")


def unused(*args: Any) -> None:
    for arg in args:
        del arg


def add_docstring[T, **Parameters](
    f: Callable[Parameters, T], doc: str
) -> Callable[Parameters, T]:
    f.__doc__ = doc
    return f


def ncore_(z: int) -> int:
    if 1 <= z <= 2:
        nc = 0
    elif 2 <= z <= 5:
        nc = 1
    elif 5 <= z <= 12:
        nc = 1
    elif 12 <= z <= 30:
        nc = 5
    elif 31 <= z <= 38:
        nc = 9
    elif 39 <= z <= 48:
        nc = 14
    elif 49 <= z <= 56:
        nc = 18
    else:
        raise ValueError("Ncore not computed in helper.ncore(), add it yourself!")
    return nc


def get_core(mol: gto.Mole) -> Tuple[int, List[int], List[int]]:
    """
    Calculate the number of cores for each atom in the molecule.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecule object from PySCF.

    Returns
    -------
    tuple
        (Ncore, idx, corelist)
    """
    idx = []
    corelist = []
    Ncore = 0
    for ix, bas in enumerate(mol.aoslice_by_atom()):
        ncore = ncore_(mol.atom_charge(ix))
        corelist.append(ncore)
        Ncore += ncore
        idx.extend([k for k in range(bas[2] + ncore, bas[3])])
    return (Ncore, idx, corelist)


# create pyscf pbc scf object
def get_scfObj(
    h1: Matrix[float64],
    Eri: Tensor4D[float64],
    nocc: int,
    dm0: Optional[Matrix[float64]] = None,
    enuc: float = 0.0,
    pert_h: bool = False,
    save_chkfile: bool = False,
    fname: str = "f0",
) -> scf.hf.RHF:
    """
    Initialize and run a restricted Hartree-Fock (RHF) calculation.

    This function sets up an SCF (Self-Consistent Field) object using the provided
    one-electron Hamiltonian, electron repulsion integrals, and number
    of occupied orbitals. It then runs the SCF procedure, optionally using an initial
    density matrix.

    Parameters
    ----------
    h1 : numpy.ndarray
        One-electron Hamiltonian matrix.
    Eri : numpy.ndarray
        Electron repulsion integrals.
    nocc : int
        Number of occupied orbitals.
    dm0 : numpy.ndarray, optional
        Initial density matrix. If not provided, the SCF calculation will start
        from scratch. Defaults to None.
    enuc : float, optional
        Nuclear repulsion energy. Defaults to 0.0.

    Returns
    -------
    mf_ : pyscf.scf.hf.RHF
        The SCF object after running the Hartree-Fock calculation.
    """
    # from 40-customizing_hamiltonian.py in pyscf examples
    nao = h1.shape[0]

    # Initialize a dummy molecule with the required number of electrons
    S = numpy.eye(nao)
    mol = gto.M()
    mol.nelectron = nocc * 2
    mol.incore_anyway = True

    # Initialize an RHF object
    mf_ = scf.RHF(mol)
    mf_.get_hcore = lambda *args: h1
    mf_.get_ovlp = lambda *args: S
    mf_._eri = Eri
    mf_.incore_anyway = True
    mf_.max_cycle = 50
    mf_.verbose = 0

    # Run the SCF calculation
    if dm0 is None:
        mf_.kernel()
    else:
        mf_.kernel(dm0=dm0)

    # Check if the SCF calculation converged
    if not mf_.converged:
        print(flush=True)
        print(
            "WARNING!!! SCF not convereged - applying level_shift=0.2, diis_space=25 ",
            flush=True,
        )
        print(flush=True)
        mf_.verbose = 0
        mf_.level_shift = 0.2
        mf_.diis_space = 25
        if dm0 is None:
            mf_.kernel()
        else:
            mf_.kernel(dm0=dm0)
        if not mf_.converged:
            print(flush=True)
            print("WARNING!!! SCF still not convereged!", flush=True)
            print(flush=True)
        else:
            print(flush=True)
            print("SCF Converged!", flush=True)
            print(flush=True)
    return mf_


def get_eri(
    i_frag: str,
    Nao: int,
    symm: int = 8,
    ignore_symm: bool = False,
    eri_file: str = "eri_file.h5",
) -> Tensor4D[float64]:
    """
    Retrieve and optionally restore electron repulsion integrals (ERI)
    from an HDF5 file.

    This function reads the ERI for a given fragment from an HDF5 file, and optionally
    restores the symmetry of the ERI.

    Parameters
    ----------
    i_frag : str
        Fragment identifier used to locate the ERI data in the HDF5 file.
    Nao : int
        Number of atomic orbitals.
    symm : int, optional
        Symmetry of the ERI. Defaults to 8.
    ignore_symm : bool, optional
        If True, the symmetry step is skipped. Defaults to False.
    eri_file : str, optional
        Filename of the HDF5 file containing the electron repulsion integrals.
        Defaults to 'eri_file.h5'.

    Returns
    -------
    numpy.ndarray
        Electron repulsion integrals, possibly restored with symmetry.
    """
    # Open the HDF5 file and read the ERI for the specified fragment
    with h5py.File(eri_file, "r") as file:
        eri__ = numpy.array(file.get(i_frag))

    # Optionally restore the symmetry of the ERI
    if not ignore_symm:
        # Set the number of threads for the library to 1
        lib.num_threads(1)
        eri__ = ao2mo.restore(symm, eri__, Nao)

    return eri__


def get_veff(
    eri_: Tensor4D[float64],
    dm: Matrix[float64],
    S: Matrix[float64],
    TA: Matrix[float64],
    hf_veff: Matrix[float64],
) -> Matrix[float64]:
    """
    Calculate the effective HF potential (Veff) for a given density matrix and
    electron repulsion integrals.

    This function computes the effective potential by transforming the density matrix,
    computing the Coulomb (J) and exchange (K) integrals.

    Parameters
    ----------
    eri_ : numpy.ndarray
        Electron repulsion integrals.
    dm : numpy.ndarray
        Density matrix. 2D array.
    S : numpy.ndarray
        Overlap matrix.
    TA : numpy.ndarray
        Transformation matrix.
    hf_veff : numpy.ndarray
        Hartree-Fock effective potential for the full system.

    Returns
    -------
    numpy.ndarray
        Effective HF potential in the embedding basis.
    """
    as_f8_array = partial(numpy.asarray, dtype=numpy.float64)
    # Transform the density matrix
    # Ensure the transformed density matrix and ERI are real and double-precision
    ST = S @ TA
    P_ = as_f8_array(ST.T @ dm @ ST)
    eri_ = as_f8_array(eri_)

    # Compute the Coulomb (J) and exchange (K) integrals
    vj, vk = scf.hf.dot_eri_dm(eri_, P_, hermi=1, with_j=True, with_k=True)
    return TA.T @ hf_veff @ TA - vj - 0.5 * vk
