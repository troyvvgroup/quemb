from __future__ import annotations

import os
from pathlib import Path
from shutil import rmtree
from types import TracebackType
from typing import TypeAlias

from attr import define
from typing_extensions import Literal

from general.be_var import SCRATCH

PathLike: TypeAlias = str | os.PathLike


@define(order=False)
class ScratchManager:
    """Manage a scratch area.

    Upon initialisation of the object the `scratch_area` is created,
    if it does not exist yet.
    If it already exists, it is ensured, that it is empty.

    If `do_cleanup` is true, then `scratch_area` is deleted,
    when the ScratchManager goes out of scope or
    if `self.cleanup` is called.

    The ScratchManager also exists as a ContextManager;
    then the cleanup is performed when leaving the ContextManager.
    """

    scratch_area: Path
    cleanup_at_end: bool

    def __init__(self, scratch_area: PathLike, cleanup_at_end: bool = True) -> None:
        self.scratch_area = Path(scratch_area)
        self.cleanup_at_end = cleanup_at_end

        self.scratch_area.mkdir(parents=True, exist_ok=True)
        if any(self.scratch_area.iterdir()):
            self.cleanup_at_end = False
            raise ValueError("scratch_area has to be empty.")

    def __enter__(self) -> ScratchManager:
        return self

    def __exit__(
        self,
        type_: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        if value is None:
            self.__del__()
        else:
            self.cleanup_at_end = False
        return False

    @classmethod
    def from_environment(
        cls,
        *,
        user_defined_name: PathLike | None = None,
        user_defined_root: PathLike | None = None,
        prefix: str = "QuEmb_",
        do_cleanup: bool = True,
    ) -> ScratchManager:
        if user_defined_name and user_defined_root:
            raise TypeError(
                "Don't use both `user_defined_name` and `user_defined_root`"
            )

        if user_defined_name:
            return cls(Path(user_defined_name), do_cleanup)

        if user_defined_root:
            scratch_root = Path(user_defined_root)
        else:
            scratch_root = Path(SCRATCH)

        if "SLURM_JOB_ID" in os.environ:
            # we can safely assume that the SLURM_JOB_ID is unique
            subdir = Path(f"{prefix}{os.environ['SLURM_JOB_ID']}/")
        else:
            # We cannot safely assume that PIDs are unique
            id = os.getpid()
            subdir = Path(f"{prefix}{id}/")
            while (scratch_root / subdir).exists():
                id = id + 1
                subdir = Path(f"{prefix}{id}/")
        return cls(scratch_root / subdir, do_cleanup)

    def cleanup(self, force_cleanup: bool | None = False) -> None:
        if self.cleanup_at_end or force_cleanup:
            rmtree(self.scratch_area)

    def __del__(self) -> None:
        try:
            self.cleanup()
        except FileNotFoundError:
            pass
