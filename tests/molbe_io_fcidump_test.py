# Illustrates how fcidump file containing fragment hamiltonian
# can be generated using be2fcidump

from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp

import numpy as np
from pyscf.tools import fcidump

from molbe import BE
from molbe.misc import be2fcidump, libint2pyscf


def prepare_system():
    # Read in molecular integrals expressed in libint basis ordering
    # numpy.loadtxt takes care of the input under the hood
    mol, mf = libint2pyscf(
        "data/octane.xyz", "data/hcore_libint_octane.dat", "STO-3G", hcore_skiprows=1
    )
    mf.kernel()

    # Construct fragments for BE
    fobj = fragpart(be_type="be2", mol=mol)
    oct_be = BE(mf, fobj)
    return oct_be

def test_fcidump_writing():
    oct_be = prepare_system()
    tmp_dir = Path(mkdtemp())
    data_dir = Path("data/")

    # Write out fcidump file for each fragment
    be2fcidump(oct_be, tmp_dir / "octane", "fragment_mo")

    for i in range(6):
        reference = fcidump.read(data_dir / f'octanef{i}')
        new = fcidump.read(tmp_dir / f'octanef{i}')

        assert np.allclose(new["H1"], reference["H1"])
        assert np.allclose(new["H2"], reference["H2"])


    rmtree(tmp_dir)



