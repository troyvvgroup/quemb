# Illustrates how fcidump file containing fragment hamiltonian
# can be generated using be2fcidump

import os
from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp

import h5py
import numpy as np
import pytest
from numpy import einsum
from pyscf import ao2mo
from pyscf.lib.misc import with_omp_threads
from pyscf.tools import fcidump

from quemb.molbe import BE, fragmentate
from quemb.molbe.misc import be2fcidump, libint2pyscf


def prepare_system() -> BE:
    # Read in molecular integrals expressed in libint basis ordering
    # numpy.loadtxt takes care of the input under the hood
    mol, mf = libint2pyscf(
        "xyz/distorted_octane.xyz",
        "data/hcore_libint_octane.dat",
        "STO-3G",
        hcore_skiprows=1,
    )
    with with_omp_threads(1):
        # multi-threaded HF execution can lead to non-deterministic
        # MO-coefficients, if the orbitals are degenerate.
        # https://github.com/pyscf/pyscf/issues/2243
        mf.kernel()

    # Construct fragments for BE
    fobj = fragmentate(n_BE=2, mol=mol)
    oct_be = BE(mf, fobj)
    return oct_be


def verify_fcidump_writing(kind_of_MO: str) -> None:
    oct_be = prepare_system()
    tmp_dir = Path(mkdtemp())
    data_dir = Path("data/octane_FCIDUMPs/")
    (tmp_dir / kind_of_MO).mkdir()

    # Write out fcidump file for each fragment
    be2fcidump(oct_be, str(tmp_dir / kind_of_MO / "octane"), kind_of_MO)

    for i in range(6):
        reference = fcidump.read(data_dir / kind_of_MO / f"octanef{i}")
        new = fcidump.read(tmp_dir / kind_of_MO / f"octanef{i}")

        for key in ["H1", "H2"]:
            if not np.allclose(new[key], reference[key]):
                print(abs(new[key] - reference[key]).max())
                raise ValueError("too large difference")
    rmtree(tmp_dir)


def verify_fcidump_io(kind_of_MO: str) -> None:
    oct_be = prepare_system()
    tmp_dir = Path(mkdtemp())
    (tmp_dir / kind_of_MO).mkdir()

    # Write out fcidump file for each fragment
    be2fcidump(oct_be, str(tmp_dir / kind_of_MO / "octane"), kind_of_MO)

    # Obtain the 1- and 2-body matrices
    for fidx, frag in enumerate(oct_be.Fobjs):
        with h5py.File(frag.eri_file, "r") as read:
            eri = read[frag.dname][()]
        eri = ao2mo.restore(1, eri, frag.nao)
        if kind_of_MO == "embedding":
            h1e = frag.fock
            h2e = eri
        elif kind_of_MO == "fragment_mo":
            frag.scf()  # make sure that we have mo coefficients

            assert frag.fock is not None
            assert frag.mo_coeffs is not None

            h1e = einsum(
                "ij,ia,jb->ab", frag.fock, frag.mo_coeffs, frag.mo_coeffs, optimize=True
            )
            h2e = einsum(
                "ijkl,ia,jb,kc,ld->abcd",
                eri,
                frag.mo_coeffs,
                frag.mo_coeffs,
                frag.mo_coeffs,
                frag.mo_coeffs,
                optimize=True,
            )
        else:
            raise Exception("kind_of_MO should be either embedding or fragment_mo")

        assert h1e is not None
        assert h2e is not None

        new = fcidump.read(tmp_dir / kind_of_MO / f"octanef{fidx}")
        # Check that integrals survived I/O

        if not np.allclose(new["H1"], h1e):
            print(abs(new["H1"] - h1e).max())
            raise ValueError("H1 mismatch after FCIDUMP IO")

        new_h2e = ao2mo.restore(1, new["H2"], h1e.shape[0])
        if not np.allclose(new_h2e, h2e):
            print(abs(new_h2e - h2e).max())
            raise ValueError("H2 mismatch after FCIDUMP IO")

    rmtree(tmp_dir)


@pytest.mark.skipif(
    not os.getenv("QUEMB_DO_KNOWN_TO_FAIL_TESTS") == "true",
    reason="This test is known to fail.",
)
def test_embedding() -> None:
    verify_fcidump_writing("embedding")


@pytest.mark.skipif(
    not os.getenv("QUEMB_DO_KNOWN_TO_FAIL_TESTS") == "true",
    reason="This test is known to fail.",
)
def test_fragment_mo() -> None:
    verify_fcidump_writing("fragment_mo")


def test_fcidump_io_embedding() -> None:
    verify_fcidump_io("embedding")


def test_fcidump_io_fragment_mo() -> None:
    verify_fcidump_io("fragment_mo")


if __name__ == "__main__":
    test_embedding()
    test_fragment_mo()
    test_fcidump_io_embedding()
    test_fcidump_io_fragment_mo()
