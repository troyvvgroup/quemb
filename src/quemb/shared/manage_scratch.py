from __future__ import annotations

import atexit
import os
from functools import partial
from pathlib import Path
from shutil import rmtree
from types import TracebackType
from typing import Annotated, Final, Literal

from attrs import define, field

from quemb.shared.config import settings
from quemb.shared.typing import PathLike


def _determine_path(
    root: PathLike | None = None,
    subdir_prefix: str | None = None,
) -> Path:
    """Find a good path name for the scratch directory.

    The naming scheme is :python:`f"{root}/{subdir_prefix}{SLURM_JOB_ID}"`
    on systems with :python:`SLURM`.
    If :python:`SLURM` is not available, then the process ID is used instead.
    """
    scratch_root = Path(root) if root else Path(settings.SCRATCH_ROOT)
    subdir_prefix = "QuEmb_" if subdir_prefix is None else subdir_prefix
    if "SLURM_JOB_ID" in os.environ:
        # we can safely assume that the SLURM_JOB_ID is unique
        subdir = Path(f"{subdir_prefix}{os.environ['SLURM_JOB_ID']}/")
    else:
        # We cannot safely assume that PIDs are unique
        id = os.getpid()
        subdir = Path(f"{subdir_prefix}{id}/")
        while (scratch_root / subdir).exists():
            id = id + 1
            subdir = Path(f"{subdir_prefix}{id}/")
    return scratch_root / subdir


def _get_abs_path(pathlike: PathLike | None) -> Path:
    """Return valid path names for the :class:`WorkDir`

    Ensure that absolute paths are returned.
    if :class:`None` is given as argument, then the path name is automatically
    determined.
    """
    if pathlike is None:
        return _determine_path().resolve()
    else:
        return Path(pathlike).resolve()


@define(order=False)
class WorkDir:
    """Manage a scratch area.

    Upon initialisation of the object the workdir :python:`path` is created,
    if it does not exist yet.
    One can either enter a :python:`path` themselves, or if it is :class:`None`,
    then the path is automatically determined by the environment,
    i.e. are we on SLURM, where is the scratch etc.
    Read more at :func:`from_environment` for more details.

    Note that the :python:`/` is overloaded for this class and it can be used
    as :python:`pathlib.Path` in that regard, see example below.

    If :python:`cleanup_at_end` is true,
    then :func:`cleanup` method is registered to be called when
    the program terminates with no errors and deletes the scratch directory.
    The :python:`WorkDir` also exists as a ContextManager;
    then the cleanup is performed when leaving the ContextManager.
    Again, assuming that :python:`cleanup_at_end` is true.

    Examples
    --------
    >>> with WorkDir('./test_dir', cleanup_at_end=True) as scratch:
    >>>     with open(scratch / 'test.txt', 'w') as f:
    >>>         f.write('hello world')
    './test_dir' does not exist anymore, if the outer contextmanager is left
    without errors.
    """

    path: Final[Annotated[Path, "An absolute path"]] = field(converter=_get_abs_path)
    cleanup_at_end: Final[bool] = True

    # The __init__ is automatically created
    # the values `self.path` and `self.cleanup_at_end` are already filled.
    # we define the __attrs_post_init__ to create the directory
    def __attrs_post_init__(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        if self.cleanup_at_end:
            atexit.register(partial(self.cleanup, ignore_error=True))

    def __enter__(self) -> WorkDir:
        return self

    def __exit__(
        self,
        type_: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        if self.cleanup_at_end:
            self.cleanup()
        return False

    @classmethod
    def from_environment(
        cls,
        *,
        user_defined_root: PathLike | None = None,
        prefix: str | None = None,
        cleanup_at_end: bool = True,
    ) -> WorkDir:
        """Create a WorkDir based on the environment.

        The naming scheme is :python:`f"{user_defined_root}/{prefix}{SLURM_JOB_ID}"`
        on systems with :python:`SLURM`.
        If :python:`SLURM` is not available, then the process ID is used instead.

        Parameters
        ----------
        user_defined_root:
            The root directory where to create temporary directories
            e.g. :bash:`/tmp` or :bash:`/scratch`.
            If :class:`None`, then the :python:`SCRATCH_ROOT`
            value from :class:`quemb.shared.config.Settings`
            is taken.
        prefix:
            The prefix for the subdirectory.
        cleanup_at_end:
            Perform cleanup when calling :python:`self.cleanup`.
        """
        return cls(_determine_path(user_defined_root, prefix), cleanup_at_end)

    def cleanup(self, ignore_error: bool = False) -> None:
        """Conditionally cleanup the working directory.

        Parameters
        ----------
        ignore_errors :
            Ignore :class:`FileNotFoundError`, and only that exception, when deleting.
        """
        try:
            rmtree(self.path)
        except FileNotFoundError as e:
            if not ignore_error:
                raise e

    def __fspath__(self) -> str:
        return self.path.__fspath__()

    def __format__(self, format_spec: str) -> str:
        return self.path.__format__(format_spec)

    def __str__(self) -> str:
        return self.path.__str__()

    def __truediv__(self, other_path: PathLike) -> Path:
        return self.path / other_path
