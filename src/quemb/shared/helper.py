from collections.abc import Iterable
from inspect import signature
from itertools import islice
from pathlib import Path
from time import time
from typing import Any, Callable, TypeVar

from attr import define, field
from numba import njit

from quemb.shared.typing import Integral

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


@define(frozen=True)
class Timer:
    """Simple class to time code execution"""

    message: str = "Elapsed time"
    start: float = field(init=False, factory=time)

    def elapsed(self) -> float:
        return time() - self.start

    def str_elapsed(self, message: str | None = None) -> str:
        return f"{self.message if message is None else message}: {self.elapsed():.5f}"


@njit(cache=True)
def gauss_sum(n: Integral) -> Integral:
    """Return the sum :math:`\sum_{i=1}^n i`

    Parameters
    ----------
    n :
    """
    return (n * (n + 1)) // 2


@njit(cache=True)
def symmetric_index(a: Integral, b: Integral) -> Integral:
    """Flatten the index a, b assuming symmetry.

    The resulting indexation for a matrix looks like this::

        0
        1   2
        3   4   5
        6   7   8   9

    Parameters
    ----------
    a :
    b :
    """
    return gauss_sum(a) + b if a > b else gauss_sum(b) + a


@njit(cache=True)
def symmetric_different_size(m: Integral, n: Integral) -> Integral:
    r"""Return the number of unique elements in a symmetric matrix of different row
    and column length

    This is for example the situation for pairs :math:`\mu, i` where :math:`\mu`
    is an AO and :math:`i` is a fragment orbital.

    The assumed structure of the symmetric matrix is::

        *   *   *   *
        *   *   *   *
        *   *   0   0
        *   *   0   0

    where the stars denote non-zero elements.

    Parameters
    ----------
    m:
    n:
    """

    m, n = min(m, n), max(m, n)  # type: ignore[type-var]
    return gauss_sum(m) + m * (n - m)


@njit(cache=True)
def get_flexible_n_eri(
    p_max: Integral, q_max: Integral, r_max: Integral, s_max: Integral
) -> Integral:
    r"""Return the number of unique ERIs but allowing different number of orbitals.

    This is for example the situation for a tuple :math:`\mu, \nu, \kappa, i`,
    where :math:`\mu, \nu, \kappa` are AOs and :math:`i` is a fragment orbital.
    This function returns the number of unique ERIs :math:`g_{\mu, \nu, \kappa, i}`.

    Parameters
    ----------
    p_max:
    q_max:
    r_max:
    s_max:
    """

    return symmetric_different_size(
        symmetric_different_size(p_max, q_max), symmetric_different_size(r_max, s_max)
    )
