# mypy: disable-error-code="import-not-found"

from opi.input.blocks.block_basis import BlockBasis
from opi.input.simple_keywords import Approximation, SimpleKeyword
from pyscf.gto import M

from quemb.molbe.mf_interfaces import OrcaArgs, get_orca_basis
from quemb.molbe.mf_interfaces.main import AVAILABLE_BACKENDS, get_mf

# You have to have ``orca`` in your path and the ORCA python interface installed.
# Follow the instructions here https://vanvoorhisgroup.mit.edu/quemb/main/install.html
assert AVAILABLE_BACKENDS["orca"]

mol = M("./data/octane.xyz", basis="sto-3g")

# You can do a normal Hartree-Fock calculation with pyscf (the default)
pyscf_mf = get_mf(mol)


# Or, you can do a normal Hartree-Fock calculation with ORCA
orca_mf = get_mf(
    mol,
    backend="orca",
)

# Or you do a more "special" Hartree-Fock calculation using for example RIJK.
# For this you can pass arguments of the ORCA python interface (opi)
# via the ``OrcaArgs`` wrapper class.
# In addition you can use ``get_orca_basis`` to translate from pyscf bases
# to ORCA bases.
orca_RIJK_mf = get_mf(
    mol,
    backend="orca",
    additional_args=OrcaArgs(
        n_procs=4,
        simple_keywords=[Approximation.RIJK],
        blocks=[BlockBasis(basis=get_orca_basis(mol), auxjk=SimpleKeyword("def2/jk"))],
    ),
)


# The output of get_mf is always a pyscf RHF class that can be passed on to BE.
