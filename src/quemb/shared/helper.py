from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar, Dict

from pyscf.tools.cubegen import orbital

from quemb import molbe
from quemb.shared.manage_scratch import PathLike

Function = TypeVar("Function", bound=Callable)


# Note that we have once Callable and once Function.
# This is **intentional**.
# The inner function `update_doc` takes a function
# and returns a function **with** the exact same signature.
def add_docstring(doc: Optional[str]) -> Callable[[Function], Function]:
    """Add a docstring to a function as decorator.

    Is useful for programmatically generating docstrings.

    Parameters
    ----------
    doc: str
        A docstring.

    Example
    ----------
    >>> @add_docstring("Returns 'asdf'")
    >>> def f():
    >>>     return 'asdf'
    is equivalent to
    >>> def f():
    >>>     "Returns 'asdf'"
    >>>     return 'asdf'
    """

    def update_doc(f: Function) -> Function:
        f.__doc__ = doc
        return f

    return update_doc


def copy_docstring(f: Callable) -> Callable[[Function], Function]:
    """Copy docstring from another function as decorator."""
    return add_docstring(f.__doc__)


def unused(*args: Any) -> None:
    pass


def ncore_(z: int) -> int:
    if 1 <= z <= 2:
        nc = 0
    elif 2 <= z <= 5:
        nc = 1
    elif 5 <= z <= 12:
        nc = 1
    elif 12 <= z <= 30:
        nc = 5
    elif 31 <= z <= 38:
        nc = 9
    elif 39 <= z <= 48:
        nc = 14
    elif 49 <= z <= 56:
        nc = 18
    else:
        raise ValueError("Ncore not computed in helper.ncore(), add it yourself!")
    return nc


def write_cube(
    be_object: molbe.BE,
    cube_file_path: PathLike,
    fragment_idx: Optional[Sequence[int]] = None,
    cubegen_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    """Write cube files of embedding orbitals from a BE object.

    Parameters
    ----------
    be_object
        BE object containing the fragments, each of which contains embedding orbitals.
    cube_file_path
        Directory to write the cube files to.
    fragment_idx
        Index of the fragments to write the cube files for.
        If None, write all fragments.
    cubegen_kwargs
        Keyword arguments passed to cubegen.orbital.
    """
    cube_file_path = Path(cube_file_path)
    cubegen_kwargs = cubegen_kwargs if cubegen_kwargs else {}
    if not isinstance(be_object, molbe.BE):
        raise NotImplementedError("Support for Periodic BE not implemented yet.")
    if fragment_idx is None:
        fragment_idx = range(be_object.Nfrag)
    for idx in fragment_idx:
        for emb_orb_idx in range(be_object.Fobjs[idx].TA.shape[1]):
            orbital(
                be_object.mol,
                cube_file_path / f"frag_{idx}_orb_{emb_orb_idx}.cube",
                be_object.Fobjs[idx].TA[:, emb_orb_idx],
                **cubegen_kwargs,
            )
