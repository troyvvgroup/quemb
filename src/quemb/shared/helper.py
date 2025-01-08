from collections.abc import Iterable
from inspect import signature
from itertools import islice
from pathlib import Path
from typing import Any, Callable, TypeVar

Function = TypeVar("Function", bound=Callable)


# Note that we have once Callable and once Function.
# This is **intentional**.
# The inner function `update_doc` takes a function
# and returns a function **with** the exact same signature.
def add_docstring(doc: str | None) -> Callable[[Function], Function]:
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


def _get_init_docstring(obj: type) -> str:
    sig = signature(obj.__init__)  # type: ignore[misc]
    docstring = """Initialization

Parameters
----------
"""
    # we want to skip `self`
    for var in islice(sig.parameters.values(), 1, None):
        docstring += f"{var.name}: {var.annotation}\n"
    return docstring


def add_init_docstring(obj: type) -> type:
    """Add a sensible docstring to the __init__ method of an attrs class

    Makes only sense if the attributes are type-annotated.
    Is a stopgap measure until https://github.com/sphinx-doc/sphinx/issues/10682
    is solved.
    """
    obj.__init__.__doc__ = _get_init_docstring(obj)  # type: ignore[misc]
    return obj


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


def delete_multiple_files(*args: Iterable[Path]) -> None:
    for files in args:
        for file in files:
            file.unlink()
