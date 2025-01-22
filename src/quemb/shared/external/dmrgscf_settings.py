#
# This file sets the environment.
# Configs will be used in dmrgsci.py and chemps2.py
#

import os

# For Block and Block2 solvers
BLOCKEXE = "block2main"
BLOCKEXE_COMPRESS_NEVPT = "block2main"
BLOCKSCRATCHDIR = str(os.getpid())
BLOCKRUNTIMEDIR = str(os.getpid())
MPIPREFIX = ""
BLOCKVERSION = None

# For chemps2 solvers
# PYCHEMPS2BIN = '/path/to/CheMPS2/build/PyCheMPS2/PyCheMPS2.so'
