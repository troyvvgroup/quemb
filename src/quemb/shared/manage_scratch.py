from __future__ import annotations

import os
from pathlib import Path
from shutil import rmtree
from types import TracebackType
from typing import Annotated, Final, Literal, Optional, TypeAlias

from attr import define, field

from quemb.shared.be_var import SCRATCH

PathLike: TypeAlias = str | os.PathLike


def _to_abs_path(pathlike: PathLike) -> Path:
    return Path(pathlike).resolve()


def _get_absolute_path(pathlike: PathLike) -> Path:
    return Path(pathlike).resolve()


@define(order=False)
class WorkDir:
    """Manage a scratch area.

    Upon initialisation of the object the workdir `path` is created,
    if it does not exist yet.
    If it already exists, it is ensured, that it is empty.
    Internally the `path` will be stored as absolute.

    If `do_cleanup` is true, then the scratch area is deleted,
    when if `self.cleanup` is called.

    Not that the `/` is overloaded for this class and it can be used
    as `pathlib.Path` in that regard, see example below.


    The `WorkDir` also exists as a ContextManager;
    then the cleanup is performed when leaving the ContextManager.
    See an example below.

    Examples
    --------
    >>> with WorkDir('./test_dir') as scratch:
    >>>     with open(scratch / 'test.txt', 'w') as f:
    >>>         f.write('hello world')
    './test_dir' does not exist anymore, if the outer contextmanager is left
    without errors.
    """

    path: Final[Annotated[Path, "An absolute path"]] = field(converter=_to_abs_path)
    cleanup_at_end: Final[bool] = True

    # The __init__ is automatically created
    # the values `self.path` and `self.cleanup_at_end` are already filled.
    # we define the __attrs_post_init__ to create the directory
    def __attrs_post_init__(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        if any(self.path.iterdir()):
            raise ValueError("scratch_area has to be empty.")

    def __enter__(self) -> WorkDir:
        return self

    def __exit__(
        self,
        type_: Optional[type[BaseException]],
        value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> Literal[False]:
        if value is None:
            self.cleanup()
        return False

    @classmethod
    def from_environment(
        cls,
        *,
        user_defined_root: Optional[PathLike] = None,
        prefix: str = "QuEmb_",
        do_cleanup: bool = True,
    ) -> WorkDir:
        """Create a WorkDir based on the environment.

        The naming scheme is `${user_defined_root}/${prefix}${SLURM_JOB_ID}`
        on systems with `SLURM`.
        If `SLURM` is not available, then the process ID is used instead.

        Parameters
        ----------
        user_defined_root: PathLike, optional
            The root directory where to create temporary directories
            e.g. `/tmp` or `/scratch`.
            If `None`, then the value from `quemb.config.SCRATCH` is taken.
        prefix: str, default: "QuEmb_"
            The prefix for the subdirectory.
        do_cleanup: bool, default: True
            Perform cleanup when calling `self.cleanup`.

        Returns
        -------
        WorkDir
            A ready to use `WorkDir`
        """
        scratch_root = Path(user_defined_root) if user_defined_root else Path(SCRATCH)

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

    def cleanup(self, force_cleanup: bool = False) -> None:
        """Conditionally cleanup the working directory.

        Parameters
        ----------
        force_cleanup : bool, optional
            If the instance was initialized with `cleanup_at_end=True`,
            or the argument `force_cleanup` is given, then
            the working directory is deleted.
            Otherwise nothing happens.
        """
        if self.cleanup_at_end or force_cleanup:
            rmtree(self.path)

    def __fspath__(self) -> str:
        return self.path.__fspath__()

    def __truediv__(self, other_path: PathLike) -> Path:
        return self.path / other_path
