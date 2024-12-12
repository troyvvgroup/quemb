from pathlib import Path
from typing import Final

import yaml
from attrs import define
from cattrs import structure, unstructure

from quemb.shared.helper import add_docstring

DEFAULT_RC_PATH: Final = Path("~/.quembrc")


@define
class Settings:
    PRINT_LEVEL: int = 5
    SCRATCH: str = ""
    CREATE_SCRATCH_DIR: bool = False
    INTEGRAL_TRANSFORM_MAX_MEMORY: float = 50  # in GB


def _write_settings(settings: Settings, path: Path) -> None:
    with open(path, "w+") as f:
        yaml.dump(unstructure(settings), stream=f, default_flow_style=False)


def _read_settings(path: Path) -> Settings:
    with open(path, "r") as f:
        return structure(yaml.safe_load(stream=f), Settings)


@add_docstring(f"Writes settings to {DEFAULT_RC_PATH}")
def dump_settings() -> None:
    _write_settings(settings, DEFAULT_RC_PATH)


if DEFAULT_RC_PATH.exists():
    settings = _read_settings(DEFAULT_RC_PATH)
else:
    settings = Settings()
