import inspect
from collections.abc import Callable, Iterable, Sequence
from inspect import signature
from itertools import islice
from pathlib import Path
from time import time
from typing import Any, TypeVar, overload

import numba as nb
import numpy as np
from attr import define, field
from ordered_set import OrderedSet

from quemb.shared.typing import Integral, Matrix, T

_Function = TypeVar("_Function", bound=Callable)
_T_Integral = TypeVar("_T_Integral", bound=Integral)


# Note that we have once Callable and once Function.
# This is **intentional**.
# The inner function `update_doc` takes a function
# and returns a function **with** the exact same signature.
def add_docstring(doc: str | None) -> Callable[[_Function], _Function]:
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

    def update_doc(f: _Function) -> _Function:
        f.__doc__ = doc
        return f

    return update_doc


def ensure(condition: bool, message: str = "") -> None:
    """This function can be used instead of :python:`assert`,
    if the test should be always executed.
    """
    if not condition:
        message = message if message else "Invariant condition was violated."
        raise ValueError(message)


def copy_docstring(f: Callable) -> Callable[[_Function], _Function]:
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

    def __attrs_post_init__(self) -> None:
        from quemb.shared.config import settings  # noqa: PLC0415

        if settings.PRINT_LEVEL >= 10:
            print(f"Timer with message '{self.message}' started.", flush=True)

    def elapsed(self) -> float:
        return time() - self.start

    def str_elapsed(self, message: str | None = None) -> str:
        return f"{self.message if message is None else message}: {self.elapsed():.5f}"


@overload
def njit(f: _Function, *, nogil: bool) -> _Function: ...
@overload
def njit(*, nogil: bool, **kwargs: Any) -> Callable[[_Function], _Function]: ...


def njit(
    f: _Function | None = None, *, nogil: bool, **kwargs: Any
) -> _Function | Callable[[_Function], _Function]:
    """Type-safe jit wrapper that caches the compiled function

    With this jit wrapper, you can actually use static typing together with numba.
    The crucial declaration is that the decorated function's interface is preserved,
    i.e. mapping :class:`Function` to :class:`Function`.
    Otherwise the following example would not raise a type error:

    .. code-block:: python

        @numba.njit
        def f(x: int) -> int:
            return x

        f(2.0)   # No type error

    While the same example, using this custom :func:`njit` would raise a type error.

    In addition to type safety, this wrapper also sets :code:`cache=True` by default.
    """
    if f is None:
        return nb.njit(cache=True, nogil=nogil, **kwargs)
    else:
        return nb.njit(f, cache=True, nogil=nogil, **kwargs)


@overload
def jitclass(cls_or_spec: T, spec: list[tuple[str, Any]] | None = ...) -> T: ...


@overload
def jitclass(
    cls_or_spec: list[tuple[str, Any]] | None = None, spec: None = None
) -> Callable[[T], T]: ...


def jitclass(
    cls_or_spec: T | list[tuple[str, Any]] | None = None,
    spec: list[tuple[str, Any]] | None = None,
) -> T | Callable[[T], T]:
    """Decorator to make a class jit-able.

    The rationale is the same as for :func:`njit`, and described there.

    For a more detailed explanation of numba jitclasses,
    see https://numba.readthedocs.io/en/stable/user/jitclass.html
    """
    return nb.experimental.jitclass(cls_or_spec, spec)


@njit(nogil=True)
def gauss_sum(n: _T_Integral) -> _T_Integral:
    r"""Return the sum :math:`\sum_{i=1}^n i`

    Parameters
    ----------
    n :
    """
    return (n * (n + 1)) // 2  # type: ignore[return-value]


@njit(nogil=True)
def ravel_symmetric(a: _T_Integral, b: _T_Integral) -> _T_Integral:
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
    return gauss_sum(a) + b if a > b else gauss_sum(b) + a  # type: ignore[return-value,operator]


@njit(nogil=True)
def n_symmetric(n: _T_Integral) -> _T_Integral:
    "The number if symmetry-equivalent pairs i <= j, for i <= n and j <= n"
    return ravel_symmetric(n - 1, n - 1) + 1  # type: ignore[return-value]


@njit(nogil=True)
def unravel_symmetric(i: Integral) -> tuple[int, int]:
    a = int((np.sqrt(8 * i + 1) - 1) // 2)
    offset = gauss_sum(a)
    b = i - offset
    if b > a:
        a, b = b, a
    return a, b


@njit(nogil=True)
def ravel_eri_idx(
    a: _T_Integral, b: _T_Integral, c: _T_Integral, d: _T_Integral
) -> _T_Integral:
    """Return compound index given four indices using Yoshimine sort and
    assuming 8-fold permutational symmetry"""
    return ravel_symmetric(ravel_symmetric(a, b), ravel_symmetric(c, d))


@njit(nogil=True)
def unravel_eri_idx(i: _T_Integral) -> tuple[int, int, int, int]:
    """Invert :func:`ravel_eri_idx`"""
    ab, cd = unravel_symmetric(i)
    a, b = unravel_symmetric(ab)
    c, d = unravel_symmetric(cd)
    return a, b, c, d


@njit(nogil=True)
def n_eri(n):
    return ravel_eri_idx(n - 1, n - 1, n - 1, n - 1) + 1


@njit(nogil=True)
def ravel_C(a: _T_Integral, b: _T_Integral, n_cols: _T_Integral) -> _T_Integral:
    """Flatten the index a, b assuming row-mayor/C indexing

    The resulting indexation for a 3 by 4 matrix looks like this::

        0   1   2   3
        4   5   6   7
        8   9  10  11


    Parameters
    ----------
    a :
    b :
    n_cols :
    """
    assert b < n_cols  # type: ignore[operator]
    return (a * n_cols) + b  # type: ignore[return-value,operator]


@njit(nogil=True)
def ravel_Fortran(a: _T_Integral, b: _T_Integral, n_rows: _T_Integral) -> _T_Integral:
    """Flatten the index a, b assuming column-mayor/Fortran indexing

    The resulting indexation for a 3 by 4 matrix looks like this::

        0   3   6   9
        1   4   7  10
        2   5   8  11


    Parameters
    ----------
    a :
    b :
    n_rows :
    """
    assert a < n_rows  # type: ignore[operator]
    return a + (b * n_rows)  # type: ignore[return-value,operator]


@njit(nogil=True)
def symmetric_different_size(m: _T_Integral, n: _T_Integral) -> _T_Integral:
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
    return gauss_sum(m) + m * (n - m)  # type: ignore[operator,return-value]


@njit(nogil=True)
def get_flexible_n_eri(
    p_max: _T_Integral, q_max: _T_Integral, r_max: _T_Integral, s_max: _T_Integral
) -> _T_Integral:
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


def union_of_seqs(*seqs: Sequence[T]) -> OrderedSet[T]:
    """Merge multiple sequences into a single :class:`OrderedSet`.

    This preserves the order of the elements in each sequence,
    and of the arguments to this function, but removes duplicates.
    (Always the first occurrence of an element is kept.)

    .. code-block:: python

        merge_seq([1, 2], [2, 3], [1, 4]) -> OrderedSet([1, 2, 3, 4])
    """
    # mypy wrongly complains that the arg type is not valid, which it is.
    return OrderedSet().union(*seqs)  # type: ignore[arg-type]


def get_calling_function_name() -> str:
    """Do stack inspection shenanigan to obtain the name
    of the calling function"""
    return inspect.stack()[1][3]


def clean_overlap(M: Matrix[np.float64], epsilon: float = 1e-12) -> Matrix[np.int64]:
    """We assume that M is a (not necessarily square) overlap matrix
    between ortho-normal vectors. We clean for floating point noise and return
    an integer matrix with only 0s and 1s."""
    M = M.copy()
    very_small = np.abs(M) < epsilon
    M[very_small] = 0
    assert (np.abs(1 - M[~very_small]) < epsilon).all()
    M[~very_small] = 1
    return M.astype(np.int64)
