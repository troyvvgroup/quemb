from pathlib import Path
from typing import Final

PRINT_LEVEL = 5
WRITE_FILE = 0
SOLVER_CALL = 0
SCRATCH: Final = Path("/tmp")
CREATE_SCRATCH_DIR = False
INTEGRAL_TRANSFORM_MAX_MEMORY = 50  # in GB