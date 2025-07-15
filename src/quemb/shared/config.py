"""Configure :python:`quemb`

One can modify settings in one session or create an RC-file.
See examples below.

Examples
--------
>>> from quemb.shared.config import settings
>>>
>>> settings.SCRATCH_ROOT = "/scratch"
Changes the default root for the scratch directory
for this python session.

>>> from quemb.shared.config import dump_settings
>>>
>>> dump_settings()
Creates ~/.quembrc.yml file that allows changes to persist.
"""

from pathlib import Path
from tempfile import gettempdir
from typing import Final

import yaml
from attrs import define
from cattrs import structure, unstructure

from quemb.shared.helper import add_docstring

DEFAULT_RC_PATH: Final = Path("~/.quembrc.yml")


@define
class Settings:
    SCRATCH_ROOT: Path = Path(gettempdir())
    INTEGRAL_TRANSFORM_MAX_MEMORY: float = 50  # in GB


def _write_settings(settings: Settings, path: Path) -> None:
    with open(path, "w+") as f:
        f.write("# Settings files for `quemb`.\n")
        f.write("# You can delete keys; in this case the default is taken.\n")
        yaml.dump(unstructure(settings), stream=f, default_flow_style=False)


def _read_settings(path: Path) -> Settings:
    with open(path) as f:
        return structure(yaml.safe_load(stream=f), Settings)


@add_docstring(f"Writes settings to :code:`{DEFAULT_RC_PATH}`")
def dump_settings() -> None:
    _write_settings(settings, DEFAULT_RC_PATH.expanduser())


if DEFAULT_RC_PATH.expanduser().exists():
    settings = _read_settings(DEFAULT_RC_PATH.expanduser())
else:
    settings = Settings()
