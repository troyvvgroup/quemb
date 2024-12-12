from attrs import define


@define
class settings:
    PRINT_LEVEL: int = 5
    WRITE_FILE: int = 0
    SOLVER_CALL: int = 0
    SCRATCH: str = ""
    CREATE_SCRATCH_DIR: bool = False
    INTEGRAL_TRANSFORM_MAX_MEMORY: float = 50  # in GB
