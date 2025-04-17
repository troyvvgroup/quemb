"""Define some types that do not fit into one particular module

In particular it enables barebone typechecking for the shape of numpy arrays

Inspired by
https://stackoverflow.com/questions/75495212/type-hinting-numpy-arrays-and-batches

Note that most numpy functions return :python:`ndarray[Any, Any]`
i.e. the type is mostly useful to document intent to the developer.
"""

import os
from typing import Any, NewType, TypeAlias, TypeVar

import numpy as np

# We just reexpose the AtomIdx type from chemcoord here
from chemcoord.typing import AtomIdx

# We want the dtype to behave covariant, i.e. if a
#  Vector[float] is allowed, then the more specific
#  Vector[float64] should also be allowed.
# Also see here:
# https://stackoverflow.com/questions/61568462/what-does-typevara-b-covariant-true-mean
#: Type annotation of a generic covariant type.
T_dtype_co = TypeVar("T_dtype_co", bound=np.generic, covariant=True)

# Currently we can define :code:`Matrix` and higher order tensors
# only with shape :code`Tuple[int, ...]` because of
# https://github.com/numpy/numpy/issues/27957
# make the typechecks more strict over time, when shape checking finally comes to numpy.

#: Type annotation of a vector.
Vector = np.ndarray[tuple[int], np.dtype[T_dtype_co]]
#: Type annotation of a matrix.
Matrix = np.ndarray[tuple[int, ...], np.dtype[T_dtype_co]]
#: Type annotation of a tensor.
Tensor3D = np.ndarray[tuple[int, ...], np.dtype[T_dtype_co]]
#: Type annotation of a tensor.
Tensor4D = np.ndarray[tuple[int, ...], np.dtype[T_dtype_co]]
#: Type annotation of a tensor.
Tensor5D = np.ndarray[tuple[int, ...], np.dtype[T_dtype_co]]
#: Type annotation of a tensor.
Tensor6D = np.ndarray[tuple[int, ...], np.dtype[T_dtype_co]]
#: Type annotation of a tensor.
Tensor7D = np.ndarray[tuple[int, ...], np.dtype[T_dtype_co]]
#: Type annotation of a tensor.
Tensor = np.ndarray[tuple[int, ...], np.dtype[T_dtype_co]]

#: Type annotation for pathlike objects.
PathLike: TypeAlias = str | os.PathLike
#: Type annotation for dictionaries holding keyword arguments.
KwargDict: TypeAlias = dict[str, Any]


#: A generic type variable, without any constraints.
T = TypeVar("T")


#: The index of an atomic orbital. This is the global index, i.e. not per fragment.
AOIdx = NewType("AOIdx", int)

#: The global index of an atomic orbital, i.e. not per fragment.
#: This is basically the result of
#: the `func:pyscf.gto.mole.Mole.aoslice_by_atom` method.
GlobalAOIdx = NewType("GlobalAOIdx", AOIdx)

#: The relative AO index.
#: This is relative to the own fragment.
RelAOIdx = NewType("RelAOIdx", AOIdx)

#: The relative AO index, relative to the reference fragment.
#: For example for an edge in fragment 1 it is the AO index of the same atom
#: interpreted as center in fragment 2.
RelAOIdxInRef = NewType("RelAOIdxInRef", AOIdx)

#: The index of a Fragment.
FragmentIdx = NewType("FragmentIdx", int)

#: The index of a heavy atom, i.e. of a motif.
#: If hydrogen atoms are not treated differently, then every atom
#: is a motif, and this type is equivalent to :class:`AtomIdx`.
MotifIdx = NewType("MotifIdx", AtomIdx)
#: In a given fragment, this is the index of a center.
#: A center was used to generate a fragment around it.
#: Since a fragment can swallow other smaller fragments, there
#: is only one origin per fragment but multiple centers,
#: which are the origins of the swallowed fragments.
CenterIdx = NewType("CenterIdx", MotifIdx)
#: An edge is the complement of the set of centers in a fragment.
EdgeIdx = NewType("EdgeIdx", MotifIdx)

#: In a given BE fragment, this is the origin of the remaining
#: fragment after subsets have been removed.
#: Since a fragment can swallow other smaller fragments, there
#: is only one origin per fragment but multiple centers,
#: which are the origins of the swallowed fragments.
#:
#: In the following example, we have drawn two fragments,
#: one around atom A and one around atom B. The fragment
#: around atom A is completely contained in the fragment around B,
#: hence we remove it. The remaining fragment around B has the origin B,
#: and swallowed the fragment around A.
#: Hence its centers are A and B.
#: The set of edges is the complement of the set of centers,
#: hence in this case for the fragment around B
#: the set of edges is the one-element set {C}.
#:
#: .. code-block::
#:
#:    __________
#:    |        |  BE2 fragment around A
#:    |        |
#:    ___________________
#:    |        |        |   BE2 fragment around B
#:    |        |        |
#:    A ------ B ------ C ------ D
#:    |        |        |        |
#:
OriginIdx = NewType("OriginIdx", CenterIdx)


ListOverFrag: TypeAlias = list
ListOverEdge: TypeAlias = list
ListOverCenter: TypeAlias = list
ListOverMotif: TypeAlias = list
