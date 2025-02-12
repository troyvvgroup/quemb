"""This module implements the fragmentation of a molecule based
on chemical connectivity that uses the overlap of tabulated van der Waals radii.

There are three main classes:

* :class:`ConnectivityData` contains the connectivity data of a molecule
    and is fully independent of the BE fragmentation level or used basis sets.
    After construction the knowledge about motifs in the molecule are available,
    if hydrogen atoms are treated differently then the motifs are all
    non-hydrogen atoms, while if hydrogen atoms are treated equal then
    all atoms are motifs.
* :class:`FragmentedStructure` is depending on the :class:`ConnectivityData`
    and performs the fragmentation depending on the BE fragmentation level, but is still
    independent of the used basis set.
    After construction this class knows about the assignment of origins, centers,
    and edges.
* :class:`FragmentedMolecule` is depending on the :class:`FragmentedStructure`
    and assigns the AO indices to each fragment and is responsible for the book keeping
    of which AO index belongs to which center and edge.
"""

from collections import defaultdict
from collections.abc import Hashable, Mapping, Sequence
from numbers import Real
from typing import Callable, Final, NewType, TypeAlias, TypeVar, cast

import chemcoord as cc
import numpy as np
from attr import define
from chemcoord import Cartesian
from chemcoord.constants import elements
from chemcoord.typing import AtomIdx
from ordered_set import OrderedSet
from pyscf.gto import Mole
from typing_extensions import Self, assert_never

from quemb.shared.typing import T

# Note that most OrderedSet types could be a non-mutable equivalent here,
# unfortunately this is not that easy to declare
# see https://stackoverflow.com/questions/79401030/good-way-to-define-new-protocol-for-collection-types


#: The index of an atomic orbital. This is the global index, i.e. not per fragment.
AOIdx = NewType("AOIdx", int)

#: The global index of an atomic orbital, i.e. not per fragment.
#: This is basically the result of
#: the `func:pyscf.gto.mole.Mole.aoslice_by_atom` method.
GlobalAOIdx = NewType("GlobalAOIdx", AOIdx)

#: The relative AO index.
#: This is relative to the own fragment.
OwnRelAOIdx = NewType("OwnRelAOIdx", AOIdx)

#: The relative AO index, relative to another fragment.
#: For example for an edge in fragment 1 it is the AO index of the same atom
#: interpreted as center in fragment 2.
OtherRelAOIdx = NewType("OtherRelAOIdx", AOIdx)

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

# We would like to have a subtype of Sequence that also behaves generically
# so that we could write
# `SeqOverFrag[AOIdx]` for a sequence that contains all AO indices in a fragment
# or `SeqOverAtom[AOIdx]` for a sequence that contains all AO indices in an atom,
# where the Sequence types are different, i.e. a function that takes a `SeqOverFrag`
# would neither accept a `SeqOverAtom` nor a generic `Sequence`.
# However, this is (currently) not possible in Python, see this issue:
# https://github.com/python/mypy/issues/3331
#
# Hence we have to use just TypeAliases, which means that `SeqOverFrag`, `SeqOverAtom`
# and `Sequence` are all the same type.
# This is not ideal, but it is the best we can do at the moment.
SeqOverFrag: TypeAlias = Sequence
SeqOverAtom: TypeAlias = Sequence
SeqOverMotif: TypeAlias = Sequence
SeqOverCenter: TypeAlias = Sequence
SeqOverEdge: TypeAlias = Sequence
SeqOverOrigin: TypeAlias = Sequence


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


# We want to express the idea in the type system that restricting
# the keys of a Mapping to a subset can also narrow down
# the type of the keys, hence of the Mapping itself.
# The direct approach, i.e.
#   Key = TypeVar("Key", bound=Hashable)
#   SubKey = TypeVar("SubKey", bound=Key)
# is not possible, because one cannot bind a TypeVar
# to another TypeVar, hence we form a union type of the subset key type
# with its complement to obain the super type of the key.
Key = TypeVar("Key", bound=Hashable)
ComplementKey = TypeVar("ComplementKey", bound=Hashable)
Val = TypeVar("Val")


def restrict_keys(
    D: Mapping[Key | ComplementKey, Val], keys: Sequence[Key]
) -> Mapping[Key, Val]:
    """Restrict the keys of a dictionary to a subset.

    The function has the interface declared in such a way that if the subset of
    keys is actually a subtype of the type of the keys, the type of the keys
    of the returned dictionary is narrowed down."""
    return {k: D[k] for k in keys}


# The following can be passed van der Waals radius alternative.
InVdWRadius: TypeAlias = Real | Callable[[Real], Real] | Mapping[str, Real]


@define(frozen=True)
class ConnectivityData:
    """Data structure to store the connectivity data of a molecule.

    This collects all information that is independent of the chosen
    fragmentation scheme, i.e. BE1, BE2, etc., and is independent
    of the basis set, i.e. STO-3G, 6-31G, etc.
    """

    #: The connectivity graph of the molecule.
    bonds_atoms: Final[Mapping[AtomIdx, OrderedSet[AtomIdx]]]
    #: The heavy atoms/motifs in the molecule. If hydrogens are not treated differently
    #: then every hydrogen is also a motif on its own.
    motifs: Final[OrderedSet[MotifIdx]]
    #: The connectivity graph solely of the motifs,
    # i.e. of the heavy atoms when ignoring the hydrogen atoms.
    bonds_motifs: Final[Mapping[MotifIdx, OrderedSet[MotifIdx]]]
    #: The hydrogen atoms in the molecule. If hydrogens are not treated differently,
    #: then this is an empty set.
    H_atoms: Final[OrderedSet[AtomIdx]]
    #: The hydrogen atoms per motif. If hydrogens are not treated differently,
    #: then the values of the dictionary are empty sets.
    H_per_motif: Final[Mapping[MotifIdx, OrderedSet[AtomIdx]]]
    #: All atoms per motif. Lists the motif/heavy atom first.
    atoms_per_motif: Final[Mapping[MotifIdx, OrderedSet[AtomIdx]]]
    #: Do we treat hydrogens differently?
    treat_H_different: Final[bool] = True

    @classmethod
    def from_cartesian(
        cls,
        m: Cartesian,
        *,
        bonds_atoms: Mapping[int, set[int]] | None = None,
        vdW_radius: InVdWRadius | None = None,
        treat_H_different: bool = True,
    ) -> Self:
        """Create a :class:`ConnectivityData` from a :class:`chemcoord.Cartesian`.

        Parameters
        ----------
        m :
            The Cartesian object to extract the connectivity data from.
        bonds_atoms : Mapping[int, OrderedSet[int]]
            Can be used to specify the connectivity graph of the molecule.
            Has exactly the same format as the output of
            :meth:`chemcoord.Cartesian.get_bonds`,
            which is called internally if this argument is not specified.
            Allows it to manually change the connectivity by modifying the output of
            :meth:`chemcoord.Cartesian.get_bonds`.
            The keyword is mutually exclusive with :python:`vdW_radius`.
        vdW_radius : Real | Callable[[Real], Real] | Mapping[str, Real]
            If :python:`bonds_atoms` is :class:`None`, then the connectivity graph is
            determined by the van der Waals radius of the atoms.
            It is possible to pass:

            * a single number which is used as radius for all atoms,
            * a callable which is applied to all radii
              and can be used to e.g. scale via :python:`lambda r: r * 1.1`,
            * a dictionary which maps the element symbol to the van der Waals radius,
              to change the radius of individual elements, e.g. :python:`{"C": 1.5}`.

            The keyword is mutually exclusive with :python:`bonds_atoms`.
        treat_H_different :
            If True, we treat hydrogen atoms differently from heavy atoms.
        """
        if not (m.index.min() == 0 and m.index.max() == len(m) - 1):
            raise ValueError("We assume 0-indexed data for the rest of the code.")
        m = m.sort_index()

        if bonds_atoms is not None and vdW_radius is not None:
            raise ValueError("Cannot specify both bonds_atoms and vdW_radius.")

        if bonds_atoms is not None:
            processed_bonds_atoms = {
                AtomIdx(k): OrderedSet([AtomIdx(j) for j in sorted(v)])
                for k, v in bonds_atoms.items()
            }
        else:
            with cc.constants.RestoreElementData():
                used_vdW_r = elements.loc[:, "atomic_radius_cc"]
                if isinstance(vdW_radius, Real):
                    elements.loc[:, "atomic_radius_cc"] = used_vdW_r.map(
                        lambda _: float(vdW_radius)
                    )
                elif callable(vdW_radius):
                    elements.loc[:, "atomic_radius_cc"] = used_vdW_r.map(vdW_radius)  # type: ignore[arg-type]
                elif isinstance(vdW_radius, Mapping):
                    elements.loc[:, "atomic_radius_cc"].update(vdW_radius)  # type: ignore[arg-type]
                elif vdW_radius is None:
                    # To avoid false-negatives we set all vdW radii to
                    # at least 0.55 Å
                    # or 20 % larger than the tabulated value.
                    elements.loc[:, "atomic_radius_cc"] = np.maximum(
                        0.55, used_vdW_r * 1.20
                    )
                else:
                    assert_never(vdW_radius)
                processed_bonds_atoms = {
                    k: OrderedSet(sorted(v)) for k, v in m.get_bonds().items()
                }

        if treat_H_different:
            motifs = OrderedSet(m.loc[m.atom != "H", :].index)
        else:
            motifs = OrderedSet(m.index)

        bonds_motif: Mapping[MotifIdx, OrderedSet[MotifIdx]] = {
            motif: motifs & processed_bonds_atoms[motif] for motif in motifs
        }
        H_atoms = OrderedSet(m.index).difference(motifs)
        H_per_motif = {
            motif: processed_bonds_atoms[motif] & H_atoms for motif in motifs
        }
        atoms_per_motif = {
            motif: union_of_seqs([motif], H_atoms)
            for motif, H_atoms in H_per_motif.items()
        }

        def motifs_share_H() -> bool:
            for i_motif, i_H_atoms in H_per_motif.items():
                for j_motif, j_H_atoms in H_per_motif.items():
                    if i_motif == j_motif:
                        continue
                    if i_H_atoms & j_H_atoms:
                        return True
            return False

        if treat_H_different and motifs_share_H():
            raise ValueError(
                "Cannot treat hydrogens differently if motifs share hydrogens."
            )

        return cls(
            processed_bonds_atoms,
            motifs,
            bonds_motif,
            H_atoms,
            H_per_motif,
            atoms_per_motif,
            treat_H_different,
        )

    def get_BE_fragment(self, i_center: MotifIdx, n_BE: int) -> OrderedSet[MotifIdx]:
        """Return the BE fragment around atom :code:`i_center`.

        The BE fragment is the set of motifs (heavy atoms if hydrogens are different)
        that are reachable from the center atom within :code:`(n_BE - 1)` bonds.
        This means that :code:`n_BE == 1` returns only the center atom itself.

        Parameters
        ----------
        i_center :
            The index of the center atom.
        n_BE :
            The coordination sphere to consider.
        """
        if n_BE < 1:
            raise ValueError("n_BE must greater than or equal to 1.")

        result = OrderedSet({i_center})
        new = result.copy()
        for _ in range(n_BE - 1):
            new = union_of_seqs(*(self.bonds_motifs[i] for i in new)).difference(result)
            if not new:
                break
            result = result.union(new)
        return result

    def get_all_BE_fragments(self, n_BE: int) -> dict[MotifIdx, OrderedSet[MotifIdx]]:
        """Return all BE-fragments

        Parameters
        ----------
        n_BE :
            The coordination sphere to consider.

        Returns
        -------
        dict
            A dictionary mapping the center atom to the BE-fragment around it.
        """
        return {i: self.get_BE_fragment(i, n_BE) for i in self.motifs}


@define(frozen=True)
class _SubsetsCleaned:
    """Data class to contain the results of the :func:`_cleanup_if_subset` function.

    Currently, this data class is only used internally
    and strictly assumes that there is exactly one unique origin per
    fragment and one unique fragment per origin.
    Otherwise the data structure of a
    :python:`typing.Mapping[OriginIdx, OrderedSet[MotifIdx]]`
    would not make sense.
    This assumption makes the code in :func:`_cleanup_if_subset`
    much easier to write and more performant,
    but it is not true for all possible fragmentations.
    For example pair-wise fragmentations, where there are multiple
    fragments for one origin, are not supported.

    The rest of the code, however, is written in a way that would fully support
    pair-wise fragmentations, if we would use a different data structure
    here and rewrote :func:`_cleanup_if_subset` accordingly.
    """

    #: The remaining fragments after removing subsets.
    #: This is a dictionary mapping the origin index to the set of motif indices.
    motif_per_frag: Final[Mapping[OriginIdx, OrderedSet[MotifIdx]]]
    #: The centers that are swallowed by the larger fragment whose center index
    #: becomes the origin index.
    swallowed_centers: Final[Mapping[OriginIdx, OrderedSet[CenterIdx]]]


def _cleanup_if_subset(
    fragment_indices: Mapping[MotifIdx, OrderedSet[MotifIdx]],
) -> _SubsetsCleaned:
    """Remove fragments that are subsets of other fragments.

    We also keep track of the Center indices that are swallowed by the
    larger fragment whose center index becomes the origin index.

    Parameters
    ----------
    fragment_indices :
        A dictionary mapping the center index to the set of motif indices.

    Returns
    -------
    _SubsetsCleaned :
        The cleaned fragments and a dictionary to keep track of swallowed centers.
    """
    # We actually need mutability here, hence it is not a Mapping.
    contain_others: dict[OriginIdx, OrderedSet[CenterIdx]] = defaultdict(OrderedSet)
    subset_of_others: set[CenterIdx] = set()

    for i_center, i_fragment in fragment_indices.items():
        if i_center in subset_of_others:
            continue
        for j_center in i_fragment:
            if i_center == j_center:
                continue
            # Now we treat j_center not as a mere MotifIdx, but as a CenterIdx.
            # Hence cast it.
            j_center = cast(CenterIdx, j_center)
            if fragment_indices[j_center].issubset(i_fragment):
                # Now we know that i_center is actually an origin,
                # because it contains the fragment around j_center.
                # Hence cast it.
                i_center = cast(OriginIdx, i_center)
                subset_of_others.add(j_center)
                contain_others[i_center] |= {j_center}
                if j_center in contain_others:
                    j_center = cast(OriginIdx, j_center)
                    contain_others[i_center] |= contain_others[j_center]
                    del contain_others[j_center]

    # We know that the first element of motifs is the center, which should
    # stay at the first position. The rest of the motifs should be sorted.
    # We also remove the swallowed centers, i.e. only origins are left.
    cleaned_fragments = {
        OriginIdx(CenterIdx(i_center)): union_of_seqs([i_center], sorted(motifs[1:]))
        for i_center, motifs in fragment_indices.items()
        if i_center not in subset_of_others
    }
    return _SubsetsCleaned(cleaned_fragments, contain_others)


@define(frozen=True, kw_only=True)
class FragmentedStructure:
    """Data structure to store the fragments of a molecule.

    This takes into account only the connectivity data and the fragmentation
    scheme but is independent of the basis sets or the electronic structure.
    """

    #: The motifs per fragment.
    #: Note that the set of motifs in the fragment
    #: is the union of centers and edges.
    #: The order is guaranteed to be first
    #: origin, centers, then edges
    #: and in each category the motif index is ascending.
    motifs_per_frag: Final[SeqOverFrag[OrderedSet[MotifIdx]]]
    #: The centers per fragment.
    #: Note that the set of centers is the complement of the edges.
    #: The order is guaranteed to be ascending.
    centers_per_frag: Final[SeqOverFrag[OrderedSet[CenterIdx]]]
    #: The edges per fragment.
    #: Note that the set of edges is the complement of the centers.
    #: The order is guaranteed to be ascending.
    edges_per_frag: Final[SeqOverFrag[OrderedSet[EdgeIdx]]]
    #: The origins per frag. Note that for "normal" BE calculations
    #: there is exacctly one origin per fragment, i.e. the
    #: `SeqOverOrigin` has one element.
    #: The order is guaranteed to be ascending.
    origin_per_frag: Final[SeqOverFrag[OrderedSet[OriginIdx]]]

    #: The atom indices per fragment.
    #: The order of the motifs is the same as in :attr:`motifs_per_frag`
    #: and hydrogen atoms directly follow their motif and are then ascending.
    atoms_per_frag: Final[SeqOverFrag[OrderedSet[AtomIdx]]]

    #: For each edge in a fragment it points to the index
    #: of the fragment where this fragment is a center, i.e.
    #: where this edge is correctly described and should be matched against.
    #: Variable was formerly known as `center`.
    frag_idx_per_edge: Final[SeqOverFrag[Mapping[EdgeIdx, FragmentIdx]]]

    #: Connectivity data of the molecule.
    conn_data: Final[ConnectivityData]
    n_BE: Final[int]

    @classmethod
    def from_conn_data(cls, conn_data: ConnectivityData, n_BE: int) -> Self:
        fragments = _cleanup_if_subset(
            {
                i_center: conn_data.get_BE_fragment(i_center, n_BE)
                for i_center in conn_data.motifs
            }
        )
        centers_per_frag = {
            i_origin: union_of_seqs(
                [i_origin], sorted(fragments.swallowed_centers.get(i_origin, []))
            )
            for i_origin in fragments.motif_per_frag
        }

        def get_edges(i_origin: OriginIdx) -> OrderedSet[EdgeIdx]:
            # the complement of the center set is the edge set,
            # we can rightfully cast the result to EdgeIdx.
            return cast(
                OrderedSet[EdgeIdx],
                OrderedSet(
                    sorted(
                        fragments.motif_per_frag[i_origin].difference(
                            centers_per_frag[i_origin]
                        )
                    )
                ),
            )

        edges_per_frag = [get_edges(i_origin) for i_origin in fragments.motif_per_frag]

        def frag_idx(edge: EdgeIdx) -> FragmentIdx:
            for i_frag, centers in enumerate(centers_per_frag.values()):
                if edge in centers:
                    return FragmentIdx(i_frag)
            raise ValueError(f"Edge {edge} not found in any fragment.")

        origin_per_frag = [
            OrderedSet([i_origin]) for i_origin in fragments.motif_per_frag
        ]

        # The final reordered motifs per frag
        motifs_per_frag = [
            union_of_seqs(cast(Sequence[MotifIdx], origin), centers, edges)
            for origin, centers, edges in zip(
                origin_per_frag, centers_per_frag.values(), edges_per_frag
            )
        ]

        atoms_per_frag = [
            union_of_seqs(
                *[conn_data.atoms_per_motif[i_motif] for i_motif in i_fragment]
            )
            for i_fragment in motifs_per_frag
        ]

        frag_idx_per_edge = [
            {edge: frag_idx(edge) for edge in edges} for edges in edges_per_frag
        ]

        return cls(
            atoms_per_frag=atoms_per_frag,
            motifs_per_frag=motifs_per_frag,
            centers_per_frag=list(centers_per_frag.values()),
            edges_per_frag=edges_per_frag,
            origin_per_frag=origin_per_frag,
            frag_idx_per_edge=frag_idx_per_edge,
            conn_data=conn_data,
            n_BE=n_BE,
        )

    @classmethod
    def from_cartesian(
        cls,
        mol: Cartesian,
        n_BE: int,
        *,
        treat_H_different: bool = True,
        bonds_atoms: Mapping[int, set[int]] | None = None,
        vdW_radius: InVdWRadius | None = None,
    ) -> Self:
        """Construct a :class:`FragmentedStructure` from a :class:`chemcoord.Cartesian`.

        Parameters
        ----------
        mol :
            The Cartesian object to extract the connectivity data from.
        n_BE :
            The coordination sphere to consider.
        treat_H_different :
            If True, we treat hydrogen atoms differently from heavy atoms.
        """
        return cls.from_conn_data(
            ConnectivityData.from_cartesian(
                mol,
                treat_H_different=treat_H_different,
                bonds_atoms=bonds_atoms,
                vdW_radius=vdW_radius,
            ),
            n_BE,
        )

    @classmethod
    def from_Mol(
        cls,
        mol: Mole,
        n_BE: int,
        *,
        treat_H_different: bool = True,
        bonds_atoms: Mapping[int, set[int]] | None = None,
        vdW_radius: InVdWRadius | None = None,
    ) -> Self:
        """Construct a :class:`FragmentedStructure` from a :class:`pyscf.gto.mole.Mole`.

        Parameters
        ----------
        mol :
            The Molecule to extract the connectivity data from.
        n_BE :
            The coordination sphere to consider.
        treat_H_different :
            If True, we treat hydrogen atoms differently from heavy atoms.
        """
        return cls.from_cartesian(
            Cartesian.from_pyscf(mol),
            n_BE,
            treat_H_different=treat_H_different,
            bonds_atoms=bonds_atoms,
            vdW_radius=vdW_radius,
        )

    def is_ordered(self) -> bool:
        """Return if :python:`self` is ordered.

        Ordered in this context means, that first the
        origins, then centers, then edges appear in the motif.
        """
        # note that centers is a subset of origins.
        # All centers that are origins appear first due to
        # how the union of OrderedSet works.
        return all(
            origins | centers | edges == motifs
            for origins, centers, edges, motifs in zip(
                self.origin_per_frag,
                self.centers_per_frag,
                self.edges_per_frag,
                self.motifs_per_frag,
            )
        )


ListOverFrag: TypeAlias = list
ListOverEdge: TypeAlias = list
ListOverMotif: TypeAlias = list


@define(frozen=True, kw_only=True)
class AutogenOutput:
    """Data structure to match explicitly the output of autogen."""

    fsites: Final[ListOverFrag[list[GlobalAOIdx]]]
    edgesites: Final[ListOverFrag[ListOverEdge[list[GlobalAOIdx]]]]
    center: Final[ListOverFrag[ListOverEdge[FragmentIdx]]]
    edge_idx: Final[ListOverFrag[ListOverEdge[list[OwnRelAOIdx]]]]
    center_idx: Final[ListOverFrag[ListOverEdge[list[OtherRelAOIdx]]]]
    centerf_idx: Final[ListOverFrag[list[OwnRelAOIdx]]]
    ebe_weight: Final[ListOverFrag[tuple[float, list[OwnRelAOIdx]]]]
    Frag_atom: Final[ListOverFrag[ListOverMotif[MotifIdx]]]
    center_atom: Final[ListOverFrag[OriginIdx]]
    hlist_atom: Final[SeqOverAtom[list[AtomIdx]]]
    add_center_atom: Final[ListOverFrag[list[CenterIdx]]]
    Nfrag: Final[int]


@define(frozen=True, kw_only=True)
class FragmentedMolecule:
    """Data structure to store the fragments, including AO indices.

    This takes into account the geometrical data and the used
    basis sets, hence it "knows" which AO index belongs to which atom
    and which fragment.
    Hence, it depends on :class:`FragmentedStructure`.
    """

    #: The actual molecule
    mol: Final[Mole]
    # yes, it is a bit redundant, because it is also contained in
    # fragmented_structure, but it is very convenient to have it here
    # as well. Due to the immutability the two views are also not a problem.
    conn_data: Final[ConnectivityData]
    frag_structure: Final[FragmentedStructure]

    #: The atomic orbital indices per atom
    AO_per_atom: Final[SeqOverAtom[OrderedSet[GlobalAOIdx]]]

    #: The atomic orbital indices per fragment
    AO_per_frag: Final[SeqOverFrag[OrderedSet[GlobalAOIdx]]]

    #: The atomic orbital indices per motif
    AO_per_motif: Final[Mapping[MotifIdx, OrderedSet[GlobalAOIdx]]]

    #: The atomic orbital indices per edge per fragment.
    #: The AO index is global.
    #: This variable was formerly known as :python:`edgesites`.
    AO_per_edge_per_frag: Final[SeqOverFrag[Mapping[EdgeIdx, OrderedSet[GlobalAOIdx]]]]

    #: The relative atomic orbital indices per motif per fragment.
    #: Relative means that the AO indices are relative to
    #: the **own** fragment.
    rel_AO_per_motif_per_frag: Final[
        SeqOverFrag[Mapping[MotifIdx, OrderedSet[OwnRelAOIdx]]]
    ]

    #: The relative atomic orbital indices per edge per fragment.
    #: Relative means that the AO indices are relative to
    #: the **own** fragment.
    #: This variable is a strict subset of :attr:`rel_AO_per_motif_per_frag`,
    #: in the sense that the motif indices, the keys in the mapping,
    #: are restricted to the edges of the fragment.
    #: This variable was formerly known as :python:`edge_idx`.
    rel_AO_per_edge_per_frag: Final[
        SeqOverFrag[Mapping[EdgeIdx, OrderedSet[OwnRelAOIdx]]]
    ]

    #: Is the complement of :attr:`rel_AO_per_edge_per_frag`.
    #: This variable was formerly known as :python:`ebe_weight`.
    #: Note that :python:`ebe_weight` also contained the weight
    #: for democratic matching. This was always 1.0 in
    #: :func:`quemb.molbe.autofrag.autogen`
    #: so it did actually not matter.
    rel_AO_per_center_per_frag: Final[
        SeqOverFrag[Mapping[CenterIdx, OrderedSet[OwnRelAOIdx]]]
    ]

    #: The relative atomic orbital indices per origin per fragment.
    #: Relative means that the AO indices are relative to
    #: the **own** fragment.
    #: This variable is a strict subset of :attr:`rel_AO_per_center_per_frag`,
    #: in the sense that the motif indices, the keys in the mapping,
    #: are restricted to the origins of the fragment.
    #: This variable was formerly known as :python:`centerf_idx`.
    rel_AO_per_origin_per_frag: Final[
        SeqOverFrag[Mapping[OriginIdx, OrderedSet[OwnRelAOIdx]]]
    ]

    #: The relative atomic orbital indices per edge per fragment.
    #: Relative means that the AO indices are relative to the **other**
    #: fragment where the edge is a center.
    #: This variable was formerly known as :python:`center_idx`.
    other_rel_AO_per_edge_per_frag: Final[
        SeqOverFrag[Mapping[EdgeIdx, OrderedSet[OtherRelAOIdx]]]
    ]

    @classmethod
    def from_frag_structure(
        cls, mol: Mole, frag_structure: FragmentedStructure
    ) -> Self:
        """Construct a :class:`FragmentedMolecule`

        Parameters
        ----------
        mol :
            The Molecule to extract the connectivity data from.
        frag_structure :
            The fragmented structure to use.
        """
        conn_data: Final = frag_structure.conn_data
        AO_per_atom: Final = get_AOidx_per_atom(mol)
        AO_per_frag: Final = [
            union_of_seqs(*(AO_per_atom[i_atom] for i_atom in i_frag))
            for i_frag in frag_structure.atoms_per_frag
        ]
        AO_per_motif: Final = {
            motif: union_of_seqs(
                *(AO_per_atom[atom] for atom in conn_data.atoms_per_motif[motif])
            )
            for motif in conn_data.motifs
        }

        AO_per_edge_per_frag: Final = [
            restrict_keys(AO_per_motif, edges)
            for edges in frag_structure.edges_per_frag
        ]

        rel_AO_per_motif_per_frag: list[Mapping[MotifIdx, OrderedSet[OwnRelAOIdx]]] = []
        for motifs in frag_structure.motifs_per_frag:
            rel_AO_per_motif = {}
            previous = 0
            for motif in motifs:
                indices = range(
                    previous, (previous := previous + len(AO_per_motif[motif]))
                )
                rel_AO_per_motif[motif] = OrderedSet(
                    OwnRelAOIdx(AOIdx(i)) for i in indices
                )
            rel_AO_per_motif_per_frag.append(rel_AO_per_motif)

        rel_AO_per_edge_per_frag: Final = [
            restrict_keys(rel_AO_per_motif, edges)
            for (edges, rel_AO_per_motif) in zip(
                frag_structure.edges_per_frag, rel_AO_per_motif_per_frag
            )
        ]

        rel_AO_per_center_per_frag: Final = [
            restrict_keys(rel_AO_per_motif, centers)
            for (centers, rel_AO_per_motif) in zip(
                frag_structure.centers_per_frag, rel_AO_per_motif_per_frag
            )
        ]

        rel_AO_per_origin_per_frag: Final = [
            restrict_keys(rel_AO_per_motif, origin)
            for (origin, rel_AO_per_motif) in zip(
                frag_structure.origin_per_frag, rel_AO_per_motif_per_frag
            )
        ]

        other_rel_AO_per_edge_per_frag: list[
            Mapping[EdgeIdx, OrderedSet[OtherRelAOIdx]]
        ] = [
            {
                i_edge: OrderedSet(
                    # We correctly reinterpet the AO indices as
                    # indices of the other fragment
                    cast(OtherRelAOIdx, i)
                    for i in rel_AO_per_motif_per_frag[frag_per_edge[i_edge]][i_edge]
                )
                for i_edge in edges
            }
            for edges, frag_per_edge in zip(
                frag_structure.edges_per_frag, frag_structure.frag_idx_per_edge
            )
        ]

        return cls(
            frag_structure=frag_structure,
            conn_data=conn_data,
            mol=mol,
            AO_per_atom=AO_per_atom,
            AO_per_frag=AO_per_frag,
            AO_per_motif=AO_per_motif,
            AO_per_edge_per_frag=AO_per_edge_per_frag,
            rel_AO_per_motif_per_frag=rel_AO_per_motif_per_frag,
            rel_AO_per_edge_per_frag=rel_AO_per_edge_per_frag,
            rel_AO_per_center_per_frag=rel_AO_per_center_per_frag,
            rel_AO_per_origin_per_frag=rel_AO_per_origin_per_frag,
            other_rel_AO_per_edge_per_frag=other_rel_AO_per_edge_per_frag,
        )

    @classmethod
    def from_mol(
        cls,
        mol: Mole,
        n_BE: int,
        *,
        treat_H_different: bool = True,
        bonds_atoms: Mapping[int, set[int]] | None = None,
        vdW_radius: InVdWRadius | None = None,
    ) -> Self:
        """Construct a :class:`FragmentedMolecule` from :class:`pyscf.gto.mole.Mole`."""
        return cls.from_frag_structure(
            mol,
            FragmentedStructure.from_Mol(
                mol,
                n_BE=n_BE,
                treat_H_different=treat_H_different,
                bonds_atoms=bonds_atoms,
                vdW_radius=vdW_radius,
            ),
        )

    def __len__(self) -> int:
        """The number of fragments."""
        return len(self.AO_per_frag)

    def match_autogen_output(self) -> AutogenOutput:
        """Match the output of :func:`quemb.molbe.autofrag.autogen`."""

        def extract_values(
            nested: Sequence[Mapping[Key, Sequence[Val]]],
        ) -> list[list[list[Val]]]:
            """Extract the values of a mapping from a sequence of mappings"""
            return [[list(v) for v in D.values()] for D in nested]

        # We cannot use the `extract_values(self.rel_AO_per_origin_per_frag)`
        # alone, because the structure in `self.rel_AO_per_origin_per_frag`
        # is more flexible and allows multiple origins per fragment.
        # extracting the values from this structure would give one nesting
        # level too much. We therefore need to merge over all origins,
        # (which there is usually only one per fragment).
        centerf_idx = [
            union_of_seqs(*idx_per_origin)
            for idx_per_origin in extract_values(self.rel_AO_per_origin_per_frag)
        ]
        # A similar issue occurs for ebe_weight, where the output
        # of autogen is a union over all centers.
        ebe_weight = [
            (1.0, list(union_of_seqs(*idx_per_center)))
            for idx_per_center in extract_values(self.rel_AO_per_center_per_frag)
        ]
        # Again, we have to account for the fact that
        # autogen assumes a single origin per fragment.
        # Check with an assert as well
        center_atom = list(union_of_seqs(*self.frag_structure.origin_per_frag))
        assert len(center_atom) == len(self)

        return AutogenOutput(
            fsites=[list(AO_indices) for AO_indices in self.AO_per_frag],
            edgesites=extract_values(self.AO_per_edge_per_frag),
            center=[list(D.values()) for D in self.frag_structure.frag_idx_per_edge],
            edge_idx=extract_values(self.rel_AO_per_edge_per_frag),
            center_idx=extract_values(self.other_rel_AO_per_edge_per_frag),
            centerf_idx=[list(seq) for seq in centerf_idx],
            ebe_weight=ebe_weight,
            Frag_atom=[list(motifs) for motifs in self.frag_structure.motifs_per_frag],
            center_atom=center_atom,
            hlist_atom=[
                list(self.conn_data.H_per_motif.get(MotifIdx(atom), []))
                for atom in self.conn_data.bonds_atoms
            ],
            add_center_atom=[
                list(centers.difference(origins))
                for (centers, origins) in zip(
                    self.frag_structure.centers_per_frag,
                    self.frag_structure.origin_per_frag,
                )
            ],
            Nfrag=len(self),
        )


def get_AOidx_per_atom(mol: Mole) -> list[OrderedSet[GlobalAOIdx]]:
    """Get the range of atomic orbital indices per atom.

    Parameters
    ----------
    mol :
        The molecule to get the atomic orbital indices from.

    Returns
    -------
    list
        A list of ranges of atomic orbital indices per atom.
    """
    return [
        OrderedSet(GlobalAOIdx(AOIdx(i)) for i in range(AO_offsets[2], AO_offsets[3]))
        for AO_offsets in mol.aoslice_by_atom()
    ]
