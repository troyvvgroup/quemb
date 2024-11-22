from __future__ import annotations

import os
from pathlib import Path
from shutil import rmtree
from typing import TypeAlias

from attrs import define

from general.be_var import SCRATCH

PathLike: TypeAlias = str | os.PathLike


@define(order=False)
class ScratchManager:
    scratch_area: Path
    do_cleanup: bool

    def __init__()

    @classmethod
    def from_environment(
        cls,
        user_defined_name: PathLike | None,
        user_defined_root: PathLike | None,
        do_cleanup: bool,
    ) -> ScratchManager:
        if user_defined_name and user_defined_root:
            raise TypeError("Don't use both `user_defined_name` and `user_defined_root`")

        if user_defined_name:
            return cls(Path(user_defined_name), do_cleanup)

        if user_defined_root:
            scratch_root = Path(user_defined_root)
        else:
            scratch_root = SCRATCH



        return cls(scratch_root, do_cleanup)


    def cleanup(self, force_cleanup: bool | None):
        if self.do_cleanup or force_cleanup:
            rmtree(self.scratch_area)






ScratchManager.from_environment('.', None, True)
