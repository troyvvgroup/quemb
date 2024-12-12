from attrs import define


@define
class Settings:
    PRINT_LEVEL: int = 5
    SCRATCH: str = ""
    CREATE_SCRATCH_DIR: bool = False
    INTEGRAL_TRANSFORM_MAX_MEMORY: float = 50  # in GB


settings = Settings()
