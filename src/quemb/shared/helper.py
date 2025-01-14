from typing import Any, Callable, Optional, TypeVar

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


def delete_multiple_files(*args: Iterable[Path]) -> None:
    for file in args:
        if file.is_file():
            file.unlink()
        else:
            delete_multiple_files(file.iterdir())
