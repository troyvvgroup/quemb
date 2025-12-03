"""This module implements the fragmentation of molecular and periodic systems
based on chemical connectivity that uses the overlap of tabulated van der Waals radii.

There are three main classes:

* :class:`BondConnectivity` contains the connectivity data of a chemical system
    and is fully independent of the BE fragmentation level or used basis sets.
    After construction the knowledge about motifs in the system are available,
    if hydrogen atoms are treated differently then the motifs are all
    non-hydrogen atoms, while if hydrogen atoms are treated equal then
    all atoms are motifs.
* :class:`PurelyStructureFragmented` is depending on the :class:`BondConnectivity`
    and performs the fragmentation depending on the BE fragmentation level, but is still
    independent of the used basis set.
    After construction this class knows about the assignment of origins, centers,
    and edges.
* :class:`Fragmented` is depending on the :class:`PurelyStructureFragmented`
    and assigns the AO indices to each fragment and is responsible for the book keeping
    of which AO index belongs to which center and edge.
"""

# Author(s): Oskar Weser

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence, Set
from itertools import chain
from pathlib import Path
from typing import Any, Final, Generic, Literal, TypeAlias, TypeVar, cast

import numpy as np
from attrs import cmp_using, define, field, fields
from chemcoord import Cartesian
from networkx.algorithms.shortest_paths.generic import shortest_path_length
from networkx.classes.graph import Graph
from ordered_set import OrderedSet
from pyscf.gto import M, Mole, is_au
from pyscf.lib import param
from pyscf.pbc.gto import Cell
from typing_extensions import Self, assert_never, override

from quemb.molbe.autofrag import FragPart
from quemb.molbe.helper import are_equal, get_core
from quemb.shared.helper import union_of_seqs, unused
from quemb.shared.typing import (
    AOIdx,
    AtomIdx,
    CenterIdx,
    EdgeIdx,
    FragmentIdx,
    GlobalAOIdx,
    MotifIdx,
    OriginIdx,
    Real,
    RelAOIdx,
    RelAOIdxInRef,
    T,
    Vector,
)

# NOTE: We rely on the fact that post python 3.7 dictionaries preserve insertion order!

# Note that most OrderedSet types could be a non-mutable equivalent here,
# unfortunately this is not that easy to declare
# see https://stackoverflow.com/questions/79401030/good-way-to-define-new-protocol-for-collection-types


# We would like to have a subtype of Sequence that also behaves generically
# so that we could write
# `SeqOverFrag[AOIdx]` for a sequence that contains all AO indices in a fragment
# or `SeqOverAtom[AOIdx]` for a sequence that contains all AO indices in an atom,
# where the Sequence types are different, i.e. a function that takes a `SeqOverFrag`
# would neither accept a `SeqOverAtom` nor a generic `Sequence`.
# However, this is (currently) not possible in Python, see this issue:
# https://github.com/python/mypy/issues/3331
# For now just stick to ``Sequence``.

HTreatment: TypeAlias = Literal[
    "treat_H_diff",  # Default, treat H and heavy atoms differently, with bond dict
    "treat_H_like_heavy_atom",  # Treat all H as the same as a heavy atom
    "at_most_one_H",  # Enforce each H belonging to at most 1 motif
]


def _iloc(view: Iterable[T], n: int) -> T:
    """Get the n-th element of an iterable.

    Scales linearly! Do not use for large n!
    """
    return next(x for i, x in enumerate(iter(view)) if i == n)


def _reorder(seq: Sequence[T], idx: Sequence[int] | Vector[np.integer]) -> list[T]:
    return [seq[i] for i in idx]  # type: ignore[index]


def _flatten(nested: Iterable[Iterable[T]]) -> list[T]:
    return list(chain(*nested))


# We want to express the idea in the type system that restricting
# the keys of a Mapping to a subset can also narrow down
# the type of the keys, hence of the Mapping itself.
# The direct approach, i.e.
#   Key = TypeVar("Key", bound=Hashable)
#   SubKey = TypeVar("SubKey", bound=Key)
# is not possible, because one cannot bind a TypeVar
# to another TypeVar, hence we form a union type of the subset key type
# with its complement to obain the super type of the key.
_T_Key = TypeVar("_T_Key", bound=Hashable)
_T_ComplementKey = TypeVar("_T_ComplementKey", bound=Hashable)
_T_Val = TypeVar("_T_Val")


def restrict_keys(
    D: Mapping[_T_Key | _T_ComplementKey, _T_Val], keys: Sequence[_T_Key]
) -> Mapping[_T_Key, _T_Val]:
    """Restrict the keys of a dictionary to a subset.

    The function has the interface declared in such a way that if the subset of
    keys is actually a subtype of the type of the keys, the type of the keys
    of the returned dictionary is narrowed down."""
    return {k: D[k] for k in keys}


_T_motif = TypeVar("_T_motif", bound=MotifIdx)
_T_AOIdx = TypeVar("_T_AOIdx", bound=AOIdx)


def _restrict(
    AO_per_motif_per_frag: Sequence[
        Mapping[MotifIdx, Mapping[AtomIdx, OrderedSet[_T_AOIdx]]]
    ],
    subsets_motifs: Sequence[OrderedSet[_T_motif]],
) -> Sequence[Mapping[_T_motif, Mapping[AtomIdx, OrderedSet[_T_AOIdx]]]]:
    return [
        restrict_keys(AO_per_motif, motif_subset)
        for (motif_subset, AO_per_motif) in zip(subsets_motifs, AO_per_motif_per_frag)
    ]


# The following can be passed van der Waals radius alternative.
InVdWRadius: TypeAlias = Real | Callable[[Real], Real] | Mapping[str, Real]


@define(frozen=True)
class BondConnectivity:
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
    #: How do we treat hydrogen atoms?
    h_treatment: Final[HTreatment] = "treat_H_diff"

    @classmethod
    def from_cartesian(
        cls,
        m: Cartesian,
        *,
        bonds_atoms: Mapping[int, Set[int]]
        | Mapping[AtomIdx, OrderedSet[AtomIdx]]
        | None = None,
        vdW_radius: InVdWRadius | None = None,
        h_treatment: HTreatment = "treat_H_diff",
    ) -> Self:
        """Create a :class:`BondConnectivity` from a :class:`chemcoord.Cartesian`.

        Parameters
        ----------
        m :
            The Cartesian object to extract the connectivity data from.
        bonds_atoms :
            Can be used to specify the connectivity graph of the molecule.
            Has exactly the same format as the output of
            :meth:`chemcoord.Cartesian.get_bonds`,
            which is called internally if this argument is not specified.
            Allows it to manually change the connectivity by modifying the output of
            :meth:`chemcoord.Cartesian.get_bonds`.
            The keyword is mutually exclusive with :python:`vdW_radius`.
            This can also take in a modified processed_bonds_atoms object (dict) when
            specifying certain h_treatment options
        vdW_radius :
            If :python:`bonds_atoms` is :class:`None`, then the connectivity graph is
            determined by the van der Waals radius of the atoms.
            It is possible to pass:

            * a single number which is used as radius for all atoms,
            * a callable which is applied to all radii
              and can be used to e.g. scale via :python:`lambda r: r * 1.1`,
            * a dictionary which maps the element symbol to the van der Waals radius,
              to change the radius of individual elements, e.g. :python:`{"C": 1.5}`.

            The keyword is mutually exclusive with :python:`bonds_atoms`.
        h_treatment :
            How do we treat the hydrogen atoms? Options include:

            * :python:`"treat_H_diff"`: Default, treating each H different from heavy
              atoms. Using the given vdW_radius to determine which H belong to which
              motif
            * :python:`"treat_H_like_heavy_atom"`: Treating each H the same as the heavy
              atoms when determining fragments
            * :python:`"at_most_one_H"`: Enforcing that each H can belong to at most one
              H, if a H is assigned to multiple motifs

        """
        if not (m.index.min() == 0 and m.index.max() == len(m) - 1):
            raise ValueError("We assume 0-indexed data for the rest of the code.")
        m = m.sort_index()

        if bonds_atoms is not None and vdW_radius is not None:
            raise ValueError("Cannot specify both bonds_atoms and vdW_radius.")

        if bonds_atoms is not None:
            processed_bonds_atoms = {
                cast(AtomIdx, k): OrderedSet([cast(AtomIdx, j) for j in sorted(v)])
                for k, v in bonds_atoms.items()
            }
        else:
            # To avoid false-negatives we set all vdW radii to
            # at least 0.55 Å
            # or 20 % larger than the tabulated value.
            processed_bonds_atoms = {
                k: OrderedSet(sorted(v))  # type: ignore[type-var]
                for k, v in m.get_bonds(
                    modify_element_data=(lambda r: np.maximum(0.55, 1.2 * r))
                    if vdW_radius is None
                    else vdW_radius
                ).items()
            }

        if h_treatment == "treat_H_like_heavy_atom":
            motifs = OrderedSet(m.index)
        else:
            motifs = OrderedSet(m.loc[m.atom != "H", :].index)

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

        def all_H_belong_to_motif() -> bool:
            return H_atoms.issubset(union_of_seqs(*(H_per_motif.values())))

        def enforce_one_H_per_motif() -> dict[AtomIdx, OrderedSet[AtomIdx]]:
            shared_H = []
            for i_motif, i_H_atoms in H_per_motif.items():
                for j_motif, j_H_atoms in H_per_motif.items():
                    if i_motif == j_motif:
                        continue
                    if i_H_atoms & j_H_atoms:
                        for x in i_H_atoms & j_H_atoms:
                            if x not in shared_H:
                                shared_H.append(x)

            for h in shared_H:
                h_dists = {}
                for i in processed_bonds_atoms[h]:
                    h_dists[i] = m.get_bond_lengths(np.asarray((h, i)))[0]
                min_dist = min(h_dists.values())

                remove_bonds = [atm for atm, d in h_dists.items() if d != min_dist]
                min_bonds = [atm for atm, d in h_dists.items() if d == min_dist]

                for b in remove_bonds:
                    processed_bonds_atoms[h].remove(b)
                    processed_bonds_atoms[b].remove(h)

                if len(min_bonds) > 1:
                    print(
                        f"H{h} is equidistant from >=2 heavy atoms. Choosing "
                        f"to be bound to the lowest index heavy atom {min_bonds[0]} "
                        f"of equidistant atoms {min_bonds}"
                    )
                    for b in min_bonds[1:]:
                        processed_bonds_atoms[h].remove(b)
                        processed_bonds_atoms[b].remove(h)

            return processed_bonds_atoms

        if h_treatment == "treat_H_diff":
            if not all_H_belong_to_motif():
                raise ValueError(
                    "Not all H belong to a motif. Modify the bond dictionary or"
                    "change `h_treatment` assign all H atoms a motif"
                )
            if motifs_share_H():
                raise ValueError(
                    "Motifs share H. Modify the bond dictionary or change "
                    "h_treatment so that no motifs share a H."
                )

            return cls(
                processed_bonds_atoms,
                motifs,
                bonds_motif,
                H_atoms,
                H_per_motif,
                atoms_per_motif,
                h_treatment,
            )
        elif h_treatment == "treat_H_like_heavy_atom":
            return cls(
                processed_bonds_atoms,
                motifs,
                bonds_motif,
                H_atoms,
                H_per_motif,
                atoms_per_motif,
                h_treatment,
            )
        elif h_treatment == "at_most_one_H":
            if not all_H_belong_to_motif():
                raise ValueError(
                    "Not all H belong to a motif. Modify the bond dictionary or"
                    "change `h_treatment` assign all H atoms a motif"
                )

            if motifs_share_H():
                mod_bonds_atoms = enforce_one_H_per_motif()
                # Modify the bond dictionary, then call from_cartesian with the standard
                # "treat_H_diff" option
                return cls.from_cartesian(
                    m,
                    bonds_atoms=mod_bonds_atoms,
                    h_treatment="treat_H_diff",
                )
            else:
                return cls(
                    processed_bonds_atoms,
                    motifs,
                    bonds_motif,
                    H_atoms,
                    H_per_motif,
                    atoms_per_motif,
                    h_treatment,
                )
        else:
            raise NotImplementedError(f"h_treatment = {h_treatment} is not implemented")
            assert_never(h_treatment)

    @classmethod
    def from_mole(
        cls,
        mol: Mole,
        *,
        bonds_atoms: Mapping[int, set[int]] | None = None,
        vdW_radius: InVdWRadius | None = None,
        h_treatment: HTreatment = "treat_H_diff",
    ) -> Self:
        """Create a :class:`BondConnectivity` from a :class:`pyscf.gto.mole.Mole`.

        Parameters
        ----------
        mol :
            The :class:`pyscf.gto.mole.Mole` to extract the connectivity data from.
        bonds_atoms :
            Can be used to specify the connectivity graph of the molecule.
            Has exactly the same format as the output of
            :meth:`chemcoord.Cartesian.get_bonds`,
            which is called internally if this argument is not specified.
            Allows it to manually change the connectivity by modifying the output of
            :meth:`chemcoord.Cartesian.get_bonds`.
            The keyword is mutually exclusive with :python:`vdW_radius`.
        vdW_radius :
            If :python:`bonds_atoms` is :class:`None`, then the connectivity graph is
            determined by the van der Waals radius of the atoms.
            It is possible to pass:

            * a single number which is used as radius for all atoms,
            * a callable which is applied to all radii
              and can be used to e.g. scale via :python:`lambda r: r * 1.1`,
            * a dictionary which maps the element symbol to the van der Waals radius,
              to change the radius of individual elements, e.g. :python:`{"C": 1.5}`.

            The keyword is mutually exclusive with :python:`bonds_atoms`.
        h_treatment :
            How do we treat the hydrogen atoms? Options include:

            * :python:`"treat_H_diff"`: Default, treating each H different from heavy
              atoms. Using the given vdW_radius to determine which H belong to which
              motif
            * :python:`"treat_H_like_heavy_atom"`: Treating each H the same as the heavy
              atoms when determining fragments
            * :python:`"at_most_one_H"`: Enforcing that each H can belong to at most one
              H, if a H is assigned to multiple motifs

        """
        return cls.from_cartesian(
            Cartesian.from_pyscf(mol),
            bonds_atoms=bonds_atoms,
            vdW_radius=vdW_radius,
            h_treatment=h_treatment,
        )

    @classmethod
    def from_cell(
        cls,
        cell: Cell,
        *,
        bonds_atoms: Mapping[int, set[int]] | None = None,
        vdW_radius: InVdWRadius | None = None,
        h_treatment: HTreatment = "treat_H_diff",
    ) -> Self:
        """Create a :class:`BondConnectivity` from a :class:`pyscf.pbc.gto.cell.Cell`.
        This function considers the periodic boundary conditions by adding periodic
        copies of the cell to the molecule. The connectivity graph from the open
        boundary condition supercell is then used to determine the connectivity graph
        of the original periodic cell.

        Parameters
        ----------
        cell :
            The :class:`pyscf.pbc.gto.cell.Cell` to extract the connectivity data from.
        bonds_atoms : Mapping[int, OrderedSet[int]]
            Can be used to specify the connectivity graph of the molecule.
            Has exactly the same format as the output of
            :meth:`chemcoord.Cartesian.get_bonds`,
            which is called internally if this argument is not specified.
            Allows it to manually change the connectivity by modifying the output of
            :meth:`chemcoord.Cartesian.get_bonds`.
            The keyword is mutually exclusive with :python:`vdW_radius`.
        vdW_radius :
            If :python:`bonds_atoms` is :class:`None`, then the connectivity graph is
            determined by the van der Waals radius of the atoms.
            It is possible to pass:

            * a single number which is used as radius for all atoms,
            * a callable which is applied to all radii
              and can be used to e.g. scale via :python:`lambda r: r * 1.1`,
            * a dictionary which maps the element symbol to the van der Waals radius,
              to change the radius of individual elements, e.g. :python:`{"C": 1.5}`.

            The keyword is mutually exclusive with :python:`bonds_atoms`.
        h_treatment :
            How do we treat the hydrogen atoms? Options include:

            * :python:`"treat_H_diff"`: Default, treating each H different from heavy
              atoms. Using the given vdW_radius to determine which H belong to which
              motif
            * :python:`"treat_H_like_heavy_atom"`: Treating each H the same as the heavy
              atoms when determining fragments
            * :python:`"at_most_one_H"`: Enforcing that each H can belong to at most one
              H, if a H is assigned to multiple motifs

        """
        # If bonds_atoms was given, use the information.
        # Otherwise, use chemcoord to get the connectivity graph.
        if h_treatment in ("at_most_one_H"):
            raise NotImplementedError("H treament not implemented for periodic systems")
        if bonds_atoms is None:
            # Add periodic copies to a fake mol object
            # Eight copies of the original cell to account for periodicity
            lattice_vectors = (
                np.array(cell.a) if is_au(cell.unit) else np.array(cell.a) / param.BOHR
            )
            offsets = np.array(
                [
                    [
                        0.0,
                        0.0,
                        0.0,
                    ],
                    lattice_vectors[0],
                    lattice_vectors[1],
                    lattice_vectors[2],
                    lattice_vectors[0] + lattice_vectors[1],
                    lattice_vectors[0] + lattice_vectors[2],
                    lattice_vectors[1] + lattice_vectors[2],
                    lattice_vectors[0] + lattice_vectors[1] + lattice_vectors[2],
                ]
            )
            supercell_mol = M(
                atom=[
                    (element, (coords + offset).tolist())
                    for offset in offsets
                    for element, coords in zip(
                        cell.elements, cell.atom_coords(unit="Bohr")
                    )
                ],
                basis=cell.basis,
                unit="bohr",
            )
            # Reuse molecular code with periodic copies
            supercell_connectivity = cls.from_cartesian(
                Cartesian.from_pyscf(supercell_mol),
                bonds_atoms=bonds_atoms,
                vdW_radius=vdW_radius,
                h_treatment=h_treatment,
            )
            # We have to choose unique pairs from the whole
            # supercell connectivity graph,
            # because we cannot guarantee that we are in the middle of the supercell.
            bonds_atoms = defaultdict(set)
            for idx, connected in supercell_connectivity.bonds_atoms.items():
                bonds_atoms[idx % cell.natm] |= {j % cell.natm for j in connected}

        return cls.from_cartesian(
            Cartesian.from_pyscf(cell.to_mol()),
            bonds_atoms=bonds_atoms,  # always set (from input or molecular code)
            vdW_radius=None,
            h_treatment=h_treatment,
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
            Defines the :python:`(n_BE - 1)`-th coordination sphere to consider.
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

    The rest of the code, however, is written in a way that fully supports
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
    swallow_replace: bool = False,
) -> _SubsetsCleaned:
    """Remove fragments that are subsets of other fragments.

    We also keep track of the Center indices that are swallowed by the
    larger fragment whose center index becomes the origin index.

    Parameters
    ----------
    fragment_indices :
        A dictionary mapping the center index to the set of motif indices.
    swallow_replace :
        If a fragment would be swallowed, it is instead replaced by the largest
        fragment that contains the smaller fragment. The definition of the origin
        is taken from the smaller fragment.
        This means, there will be no centers other than origins.

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
        cast(OriginIdx, i_center): union_of_seqs([i_center], sorted(motifs[1:]))  # type: ignore[type-var]
        for i_center, motifs in fragment_indices.items()
        if i_center not in subset_of_others
    }

    if swallow_replace:
        for i_origin, centers in contain_others.items():
            for center in centers:
                cleaned_fragments[cast(OriginIdx, center)] = cleaned_fragments[i_origin]
        contain_others = {k: OrderedSet() for k in contain_others}
    return _SubsetsCleaned(cleaned_fragments, contain_others)


_T_chemsystem = TypeVar("_T_chemsystem", Mole, Cell)


@define(frozen=True, kw_only=True)
class PurelyStructureFragmented(Generic[_T_chemsystem]):
    """Data structure to store the fragments of a molecule.

    This takes into account only the connectivity data and the fragmentation
    scheme but is independent of the basis sets or the electronic structure.
    """

    #: The full molecule
    mol: _T_chemsystem = field(eq=cmp_using(are_equal))

    #: The motifs per fragment.
    #: Note that the full set of motifs for a fragment is the union of all center motifs
    #: and edge motifs.
    #: The order is guaranteed to be first
    #: origin, centers, then edges
    #: and in each category the motif index is ascending.
    motifs_per_frag: Final[Sequence[OrderedSet[MotifIdx]]]
    #: The centers per fragment.
    #: Note that the set of centers is the complement of the edges.
    #: The order is guaranteed to be ascending.
    #: Every motif is at least once a center, but a given motif can appear
    #: multiple times as center in different fragments.
    #: By using :meth:`get_autocratically_matched` we can ensure that a given center
    #: appears exactly once as a center.
    centers_per_frag: Final[Sequence[OrderedSet[CenterIdx]]]
    #: The edges per fragment.
    #: Note that the set of edges is the complement of the centers.
    #: The order is guaranteed to be ascending.
    edges_per_frag: Final[Sequence[OrderedSet[EdgeIdx]]]
    #: The origins per frag. Note that for "normal" BE calculations
    #: there is exacctly one origin per fragment, i.e. the
    #: `Sequence` has one element.
    #: The order is guaranteed to be ascending.
    origin_per_frag: Final[Sequence[OrderedSet[OriginIdx]]]

    #: The atom indices per fragment. it contains both
    #: motif **and** hydrogen indices.
    #: The order of the motifs is the same as in :attr:`motifs_per_frag`.
    #: The hydrogen atoms directly follow the motif to which they are attached,
    #: and are then ascendingly sorted.
    #: To given an example: if 1 and 4 are motif indices and
    #: hydrogen atoms 5, 6 are connected to 1, while hydrogen atoms 2, 3
    #: are connected to 4, then the order is: :python:`[1, 5, 6, 4, 2, 3]`.
    atoms_per_frag: Final[Sequence[OrderedSet[AtomIdx]]]

    #: For each edge in a fragment it points to the index
    #: of the fragment where this fragment is a center, i.e.
    #: where this edge is correctly described and should be matched against.
    #: Variable was formerly known as `center`.
    ref_frag_idx_per_edge: Final[Sequence[Mapping[EdgeIdx, FragmentIdx]]]

    #: Connectivity data of the molecule.
    conn_data: Final[BondConnectivity]
    n_BE: Final[int]

    @classmethod
    def from_conn_data(
        cls,
        mol: _T_chemsystem,
        conn_data: BondConnectivity,
        n_BE: int,
        swallow_replace: bool,
    ) -> Self:
        fragments = _cleanup_if_subset(
            {
                i_center: conn_data.get_BE_fragment(i_center, n_BE)
                for i_center in conn_data.motifs
            },
            swallow_replace=swallow_replace,
        )
        centers_per_frag = {
            i_origin: union_of_seqs(
                [i_origin],
                sorted(fragments.swallowed_centers.get(i_origin, [])),  # type: ignore[type-var]
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
                    return cast(FragmentIdx, i_frag)
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

        ref_frag_idx_per_edge = [
            {edge: frag_idx(edge) for edge in edges} for edges in edges_per_frag
        ]

        return cls(
            mol=mol,
            atoms_per_frag=atoms_per_frag,
            motifs_per_frag=motifs_per_frag,
            centers_per_frag=list(centers_per_frag.values()),
            edges_per_frag=edges_per_frag,
            origin_per_frag=origin_per_frag,
            ref_frag_idx_per_edge=ref_frag_idx_per_edge,
            conn_data=conn_data,
            n_BE=n_BE,
        )

    @classmethod
    def from_mole(
        cls,
        mol: _T_chemsystem,
        n_BE: int,
        *,
        h_treatment: HTreatment = "treat_H_diff",
        bonds_atoms: Mapping[int, set[int]] | None = None,
        vdW_radius: InVdWRadius | None = None,
        autocratic_matching: bool = True,
        swallow_replace: bool = False,
    ) -> Self:
        """Construct a :class:`PurelyStructureFragmented`
        from a :class:`pyscf.gto.mole.Mole`.

        Parameters
        ----------
        mol :
            The Molecule to extract the connectivity data from.
        n_BE :
            The coordination sphere to consider.
        h_treatment :
            How do we treat the hydrogen atoms? Options include:

            * :python:`"treat_H_diff"`: Default, treating each H different from heavy
              atoms. Using the given vdW_radius to determine which H belong to which
              motif
            * :python:`"treat_H_like_heavy_atom"`: Treating each H the same as the heavy
              atoms when determining fragments
            * :python:`"at_most_one_H"`: Enforcing that each H can belong to at most one
              H, if a H is assigned to multiple motifs

        autocratic_matching :
            Assume autocratic matching for possibly shared centers.
            Will call :meth:`get_autocratically_matched` upon construction.
            Look there for more details.
        swallow_replace :
            If a fragment would be swallowed, it is instead replaced by the largest
            fragment that contains the smaller fragment. The definition of the origin
            is taken from the smaller fragment.
            This means, there will be no centers other than origins.
        """
        if isinstance(mol, Mole):
            fragments = cls.from_conn_data(
                mol,
                BondConnectivity.from_mole(
                    mol,
                    h_treatment=h_treatment,
                    bonds_atoms=bonds_atoms,
                    vdW_radius=vdW_radius,
                ),
                n_BE,
                swallow_replace=swallow_replace,
            )
        elif isinstance(mol, Cell):
            fragments = cls.from_conn_data(
                mol,
                BondConnectivity.from_cell(
                    mol,
                    h_treatment=h_treatment,
                    bonds_atoms=bonds_atoms,
                    vdW_radius=vdW_radius,
                ),
                n_BE,
                swallow_replace=swallow_replace,
            )
        if autocratic_matching:
            return fragments.get_autocratically_matched()
        return fragments

    def is_ordered(self) -> bool:
        """Return if :python:`self` is ordered.

        Ordered in this context means, that first the
        origins, then centers, then edges appear in the motif.
        """
        # note that origins is a subset of centers.
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

    def write_geom(self, prefix: str = "f", dir: Path = Path(".")) -> None:
        """Write the structures of the fragments to files."""
        if isinstance(self.mol, Cell):
            raise NotImplementedError(
                "Writing the structures of fragments from periodic systems"
                " is not implemented."
            )
        mol = Cartesian.from_pyscf(self.mol)
        for i_frag, atoms in enumerate(self.atoms_per_frag):
            mol.loc[atoms, :].to_xyz(dir / f"{prefix}{i_frag}.xyz")

    def get_string(self) -> str:
        """Get a long string representation of the fragments.

        One can also call :python:`str(self)` to get a short string representation.
        """

        def to_comma_output(seq: Sequence) -> str:
            return ", ".join(str(x + 1) for x in seq)

        n_col_centers: Final = max(
            10, *(len(to_comma_output(centers)) for centers in self.centers_per_frag)
        )
        n_col_edges: Final = max(
            10, *(len(to_comma_output(edges)) for edges in self.edges_per_frag)
        )
        separator_line: Final = (28 + n_col_edges + n_col_centers) * "-"

        output = (
            "Atom indices of motifs (1-indexed)\n"
            f"{separator_line}\n"
            f" Fragment |    Origin | {'Centers':>{n_col_centers}} | {'Edges':>{n_col_edges}}\n"  # noqa: E501
            f"{separator_line}\n"
        )
        for i_frag, (motifs, centers, edges, origins) in enumerate(
            zip(
                self.motifs_per_frag,
                self.centers_per_frag,
                self.edges_per_frag,
                self.origin_per_frag,
            )
        ):
            output += f"{i_frag + 1:>9} | {to_comma_output(origins):>9} | {to_comma_output(centers):>{n_col_centers}} | {to_comma_output(edges):>{n_col_edges}}\n"  # noqa: E501

        output += f"{separator_line}\n"
        return output

    def _get_shared_centers(self) -> dict[CenterIdx, OrderedSet[FragmentIdx]]:
        """Get a dictionary of centers which are shared among multiple fragments.

        The returned dictionary contains the shared centers pointing to
        the containing fragments.
        """
        result: dict[CenterIdx, OrderedSet[FragmentIdx]] = defaultdict(OrderedSet)
        # We assume that every ``MotifIdx`` is a ``CenterIdx``
        # at least once in at least one fragment, hence cast to ``Sequence[CenterIdx]```
        for center in cast(Sequence[CenterIdx], self.conn_data.motifs):
            # Find all fragments where ``center`` is a center
            for i_frag, centers_in_frag in enumerate(self.centers_per_frag):
                if center in centers_in_frag:
                    result[center].add(cast(FragmentIdx, i_frag))
        # Retain only those center indices which appear in more than one fragment
        return {k: v for k, v in result.items() if len(v) > 1}

    def _best_repr_fragment(
        self, center: CenterIdx, fragments: Set[FragmentIdx]
    ) -> FragmentIdx:
        """Assuming that :python:``center`` is shared across :python:``fragments``,
        return the index of the fragment which represents it best.

        This is done by finding the shortest path to each fragment's origin
        and taking the fragment with the closest origin.
        """
        nx_graph: Graph = Graph(self.conn_data.bonds_motifs)  # type: ignore[arg-type]

        def distance_to_fragment(i_frag: FragmentIdx) -> tuple[int, FragmentIdx]:
            """Return the distance to the fragment with index ``i_frag``,
            as measured by the edge-number of the shortest path to the closest origin.
            Additionally return the index itsef to achieve unique ordering, if the
            distance is degenerate."""
            return (
                min(
                    shortest_path_length(nx_graph, source=center, target=i_origin)
                    for i_origin in self.origin_per_frag[i_frag]  # type: ignore[call-overload]
                ),
                i_frag,
            )

        return sorted(fragments, key=distance_to_fragment)[0]

    def get_autocratically_matched(self) -> Self:
        """Ensure that no centers exist, that are shared among fragments.

        This is the same as using autocratic matching.
        If there is a motif, which appears as center in multiple fragments,
        we will choose a fragment whose origin is closest to the center.
        In this fragment it will stay a center, while the same motif
        will become an edge in the other fragments.

        For example, if we have the following nested structure
        (part of a larger molecule that continues left and right)

        .. code-block:: text

            --- 1 - 2 - 3 - 4 - 5 ---
                        |
                        6
                        |
                        7

        and assume BE(3) fragmentation then the atom 6 appears as center
        in the BE(3)-fragments around atoms 2, 3, and 4.
        Since atom 6 is closest to atom 3,
        it will stay a center in the fragment around atom 3
        and will be re-declared as edge in the fragments
        around 2 and 4.
        """
        shared_centers = self._get_shared_centers()

        best_fragment = {
            center: self._best_repr_fragment(center, idx_fragments)
            for center, idx_fragments in shared_centers.items()
        }
        bad_fragments = {
            center: idx_fragments - {best_fragment[center]}
            for center, idx_fragments in shared_centers.items()
        }

        def invert(
            D: Mapping[CenterIdx, Set[FragmentIdx]],
        ) -> defaultdict[FragmentIdx, set[CenterIdx]]:
            result = defaultdict(set)
            for k, values in D.items():
                for v in values:
                    result[v].add(k)
            return result

        becomes_an_edge = invert(bad_fragments)

        return self.__class__(
            mol=self.mol,
            motifs_per_frag=self.motifs_per_frag,
            origin_per_frag=self.origin_per_frag,
            atoms_per_frag=self.atoms_per_frag,
            conn_data=self.conn_data,
            n_BE=self.n_BE,
            centers_per_frag=[
                centers.difference(becomes_an_edge[cast(FragmentIdx, i_frag)])
                for i_frag, centers in enumerate(self.centers_per_frag)
            ],
            # In the following we re-declare some of the centers as edges,
            # hence we have to cast to ``EdgeIdx``.
            edges_per_frag=[
                OrderedSet(
                    sorted(
                        edges.union(
                            cast(
                                Set[EdgeIdx], becomes_an_edge[cast(FragmentIdx, i_frag)]
                            )
                        )
                    )  # type: ignore[type-var]
                )
                for i_frag, edges in enumerate(self.edges_per_frag)
            ],
            ref_frag_idx_per_edge=[
                _sort_by_keys(
                    dict(edges)
                    | {
                        cast(EdgeIdx, center): best_fragment[center]
                        for center in becomes_an_edge.get(
                            cast(FragmentIdx, i_frag), set()
                        )
                    }
                )  # type: ignore[type-var]
                for i_frag, edges in enumerate(self.ref_frag_idx_per_edge)
            ],
        )

    def shared_centers_exist(self) -> bool:
        """Check if shared centers exist.

        Using :meth:`get_autocratically_matched` it is possible to re-declare shared
        centers as edges.
        """
        return len(self.conn_data.motifs) != sum(len(x) for x in self.centers_per_frag)

    def reorder_frags(self, idx: Sequence[int] | Vector[np.integer]) -> Self:
        """Reorder the fragments of self.

        Can, for example, be used to order the fragments by size.

        Parameters
        ----------
        idx :
            The new index, has to be a permutation of :python:`[0, ..., len(self)]`.
        """
        assert set(idx) == set(range(len(idx)))
        return self.__class__(
            mol=self.mol,
            motifs_per_frag=_reorder(self.motifs_per_frag, idx),
            centers_per_frag=_reorder(self.centers_per_frag, idx),
            edges_per_frag=_reorder(self.edges_per_frag, idx),
            origin_per_frag=_reorder(self.origin_per_frag, idx),
            atoms_per_frag=_reorder(self.atoms_per_frag, idx),
            ref_frag_idx_per_edge=_reorder(self.ref_frag_idx_per_edge, idx),
            conn_data=self.conn_data,
            n_BE=self.n_BE,
        )


@define(frozen=True, kw_only=True)
class Fragmented(Generic[_T_chemsystem]):
    """Contains the whole BE fragmentation information, including AO indices.

    This takes into account the geometrical data and the used
    basis sets, hence it "knows" which AO index belongs to which atom
    and which fragment.
    It depends on :class:`PurelyStructureFragmented` to store structural data,
    but contains more information.
    """

    #: The full molecule
    mol: _T_chemsystem = field(eq=cmp_using(are_equal))

    # yes, it is a bit redundant, because `conn_data` is also contained in
    # `frag_structure`, but it is very convenient to have it here
    # as well. Due to the immutability the two views are also not a problem.
    conn_data: Final[BondConnectivity]

    frag_structure: Final[PurelyStructureFragmented] = field()

    @frag_structure.validator
    def _ensure_no_shared_centers(
        self, attribute: Any, value: PurelyStructureFragmented
    ) -> None:
        unused(attribute)
        if value.shared_centers_exist():
            raise ValueError(
                "Shared centers not supported. Use autocratic matching instead."
            )

    #: The atomic orbital indices per atom
    AO_per_atom: Final[Sequence[OrderedSet[GlobalAOIdx]]]

    #: The atomic orbital indices per fragment
    AO_per_frag: Final[Sequence[OrderedSet[GlobalAOIdx]]]

    #: The atomic orbital indices per motif
    AO_per_motif: Final[Mapping[MotifIdx, Mapping[AtomIdx, OrderedSet[GlobalAOIdx]]]]

    #: The atomic orbital indices per edge per fragment.
    #: The AO index is global.
    AO_per_edge_per_frag: Final[
        Sequence[Mapping[EdgeIdx, Mapping[AtomIdx, OrderedSet[GlobalAOIdx]]]]
    ]

    #: The relative atomic orbital indices per motif per fragment.
    #: Relative means that the AO indices are relative to
    #: the **own** fragment.
    relAO_per_motif_per_frag: Final[
        Sequence[Mapping[MotifIdx, Mapping[AtomIdx, OrderedSet[RelAOIdx]]]]
    ]

    #: The relative atomic orbital indices per edge per fragment.
    #: Relative means that the AO indices are relative to
    #: the **own** fragment.
    #: This variable is a strict subset of :attr:`relAO_per_motif_per_frag`,
    #: in the sense that the motif indices, the keys in the Mapping,
    #: are restricted to the edges of the fragment.
    relAO_per_edge_per_frag: Final[
        Sequence[Mapping[EdgeIdx, Mapping[AtomIdx, OrderedSet[RelAOIdx]]]]
    ]

    #: The relative atomic orbital indices per edge per fragment.
    #: Relative means that the AO indices are relative to
    #: the **own** fragment.
    #: This variable is a subset of :attr:`relAO_per_motif_per_frag`,
    #: in the sense that the motif indices, the keys in the Mapping,
    #: are restricted to the centers of the fragment.
    relAO_per_center_per_frag: Final[
        Sequence[Mapping[CenterIdx, Mapping[AtomIdx, OrderedSet[RelAOIdx]]]]
    ]

    #: The relative atomic orbital indices per origin per fragment.
    #: Relative means that the AO indices are relative to
    #: the **own** fragment.
    #: This variable is a subset of :attr:`relAO_per_center_per_frag`,
    #: in the sense that the motif indices, the keys in the Mapping,
    #: are restricted to the origins of the fragment.
    #: This variable was formerly known as :python:`centerf_idx`.
    relAO_per_origin_per_frag: Final[
        Sequence[Mapping[OriginIdx, Mapping[AtomIdx, OrderedSet[RelAOIdx]]]]
    ]

    #: The relative atomic orbital indices per edge per fragment.
    #: Relative means that the AO indices are relative to the **other**
    #: fragment where the edge is a center.
    relAO_in_ref_per_edge_per_frag: Final[
        Sequence[Mapping[EdgeIdx, Mapping[AtomIdx, OrderedSet[RelAOIdxInRef]]]]
    ]

    #: Do we have frozen_core AO index offsets?
    frozen_core: Final[bool]

    #: The molecule with the valence/minimal basis, if we use IAO.
    iao_valence_mol: _T_chemsystem | None = field(
        eq=cmp_using(
            lambda x, y: (x is None and y is None)
            or (x is not None and y is not None and are_equal(x, y))
        )
    )

    @classmethod
    def from_frag_structure(
        cls,
        mol: _T_chemsystem,
        frag_structure: PurelyStructureFragmented,
        frozen_core: bool,
        iao_valence_basis: str | None = None,
    ) -> Self:
        """Construct a :class:`Fragmented`

        Parameters
        ----------
        mol :
            The Molecule to extract the connectivity data from.
        frag_structure :
            The fragmented structure to use.
        """
        conn_data: Final = frag_structure.conn_data
        AO_per_atom: Final = _get_AOidx_per_atom(mol, frozen_core)
        AO_per_frag: Final = [
            union_of_seqs(*(AO_per_atom[i_atom] for i_atom in i_frag))
            for i_frag in frag_structure.atoms_per_frag
        ]
        AO_per_motif: Final = {
            motif: {
                atom: AO_per_atom[atom] for atom in conn_data.atoms_per_motif[motif]
            }
            for motif in conn_data.motifs
        }

        AO_per_edge_per_frag: Final = [
            restrict_keys(AO_per_motif, edges)
            for edges in frag_structure.edges_per_frag
        ]

        relAO_per_motif_per_frag: list[
            Mapping[MotifIdx, Mapping[AtomIdx, OrderedSet[RelAOIdx]]]
        ] = []
        for motifs in frag_structure.motifs_per_frag:
            rel_AO_per_motif: dict[MotifIdx, dict[AtomIdx, OrderedSet[RelAOIdx]]] = {}
            previous = 0
            for motif in motifs:
                rel_AO_per_motif[motif] = {}
                for atom in conn_data.atoms_per_motif[motif]:
                    indices = range(
                        previous,
                        (previous := previous + len(AO_per_motif[motif][atom])),
                    )
                    rel_AO_per_motif[motif][atom] = OrderedSet(
                        cast(RelAOIdx, i) for i in indices
                    )
            relAO_per_motif_per_frag.append(rel_AO_per_motif)

        relAO_per_edge_per_frag: Final = _restrict(
            relAO_per_motif_per_frag, frag_structure.edges_per_frag
        )

        relAO_per_center_per_frag: Final = _restrict(
            relAO_per_motif_per_frag, frag_structure.centers_per_frag
        )

        relAO_per_origin_per_frag: Final = _restrict(
            relAO_per_motif_per_frag, frag_structure.origin_per_frag
        )

        relAO_in_ref_per_edge_per_frag: list[
            Mapping[EdgeIdx, Mapping[AtomIdx, OrderedSet[RelAOIdxInRef]]]
        ] = [
            {
                i_edge: {
                    atom: cast(OrderedSet[RelAOIdxInRef], indices)
                    # We correctly reinterpet the AO indices as
                    # indices of the other fragment
                    for atom, indices in relAO_per_motif_per_frag[
                        frag_per_edge[i_edge]
                    ][i_edge].items()
                }
                for i_edge in edges
            }
            for edges, frag_per_edge in zip(
                frag_structure.edges_per_frag, frag_structure.ref_frag_idx_per_edge
            )
        ]

        if iao_valence_basis is not None:
            small_mol = mol.copy()
            small_mol.basis = iao_valence_basis
            small_mol.build()
        else:
            small_mol = None

        return cls(
            frag_structure=frag_structure,
            conn_data=conn_data,
            mol=mol,
            AO_per_atom=AO_per_atom,
            AO_per_frag=AO_per_frag,
            AO_per_motif=AO_per_motif,
            AO_per_edge_per_frag=AO_per_edge_per_frag,
            relAO_per_motif_per_frag=relAO_per_motif_per_frag,
            relAO_per_edge_per_frag=relAO_per_edge_per_frag,
            relAO_per_center_per_frag=relAO_per_center_per_frag,
            relAO_per_origin_per_frag=relAO_per_origin_per_frag,
            relAO_in_ref_per_edge_per_frag=relAO_in_ref_per_edge_per_frag,
            frozen_core=frozen_core,
            iao_valence_mol=small_mol,
        )

    @classmethod
    def from_mole(
        cls,
        mol: _T_chemsystem,
        n_BE: int,
        *,
        frozen_core: bool = False,
        h_treatment: HTreatment = "treat_H_diff",
        bonds_atoms: Mapping[int, set[int]] | None = None,
        vdW_radius: InVdWRadius | None = None,
        iao_valence_basis: str | None = None,
        autocratic_matching: bool = True,
        swallow_replace: bool = False,
    ) -> Self:
        """Construct a :class:`Fragmented` from :class:`pyscf.gto.mole.Mole`
         or :class:`pyscf.pbc.gto.cell.Cell`.

        Parameters
        ----------
        mol :
            The :class:`pyscf.gto.mole.Mole` or :class:`pyscf.pbc.gto.cell.Cell`
            to extract the connectivity data from.
        n_BE :
            The BE fragmentation level.
        h_treatment :
            How do we treat the hydrogen atoms? Options include:

            * :python:`"treat_H_diff"`: Default, treating each H different from heavy
              atoms. Using the given vdW_radius to determine which H belong to which
              motif
            * :python:`"treat_H_like_heavy_atom"`: Treating each H the same as the heavy
              atoms when determining fragments
            * :python:`"at_most_one_H"`: Enforcing that each H can belong to at most one
              H, if a H is assigned to multiple motifs

        bonds_atoms :
            Can be used to specify the connectivity graph of the molecule.
            Has exactly the same format as the output of
            :meth:`chemcoord.Cartesian.get_bonds`,
            which is called internally if this argument is not specified.
            Allows it to manually change the connectivity by modifying the output of
            :meth:`chemcoord.Cartesian.get_bonds`.
            The keyword is mutually exclusive with :python:`vdW_radius`.
        vdW_radius :
            If :python:`bonds_atoms` is :class:`None`, then the connectivity graph is
            determined by the van der Waals radius of the atoms.
            It is possible to pass:

            * a single number which is used as radius for all atoms,
            * a callable which is applied to all radii
              and can be used to e.g. scale via :python:`lambda r: r * 1.1`,
            * a dictionary which maps the element symbol to the van der Waals radius,
              to change the radius of individual elements, e.g. :python:`{"C": 1.5}`.

            The keyword is mutually exclusive with :python:`bonds_atoms`.
        autocratic_matching :
            Assume autocratic matching for possibly shared centers.
            Will call :meth:`PurelyStructureFragmented.get_autocratically_matched`
            upon construction.  Look there for more details.
        swallow_replace :
            If a fragment would be swallowed, it is instead replaced by the largest
            fragment that contains the smaller fragment. The definition of the origin
            is taken from the smaller fragment.
            This means, there will be no centers other than origins.
        """
        return cls.from_frag_structure(
            mol,
            PurelyStructureFragmented.from_mole(
                mol,
                n_BE=n_BE,
                h_treatment=h_treatment,
                bonds_atoms=bonds_atoms,
                vdW_radius=vdW_radius,
                autocratic_matching=autocratic_matching,
                swallow_replace=swallow_replace,
            ),
            frozen_core=frozen_core,
            iao_valence_basis=iao_valence_basis,
        )

    def __len__(self) -> int:
        """The number of fragments."""
        return len(self.AO_per_frag)

    def _get_FragPart_no_iao(self) -> ChemFragPart:
        """Transform into a :class:`quemb.molbe.chemfrag.ChemFragPart`
        for further use in quemb.

        Matches the output of :func:`quemb.molbe.autofrag.autogen`.
        """

        # We cannot use the `extract_values(self.relAO_per_origin)`
        # alone, because the structure in `self.relAO_per_origin`
        # is more flexible than in FragPart and allows multiple origins per fragment.
        # extracting the values from this structure would give one nesting
        # level too much. We therefore need to merge over all origins,
        # (which there is usually only one per fragment).
        relAO_per_origin = [
            union_of_seqs(*idx_per_origin)
            for idx_per_origin in _extract_values(self.relAO_per_origin_per_frag)
        ]
        weight_and_relAO_per_center_per_frag = [
            (1.0, list(union_of_seqs(*idx_per_center)))
            for idx_per_center in _extract_values(self.relAO_per_center_per_frag)
        ]
        # Again, we have to account for the fact that
        # autogen assumes a single origin per fragment.
        # Check with an assert as well
        origin_per_frag = list(union_of_seqs(*self.frag_structure.origin_per_frag))
        assert len(origin_per_frag) == len(self)

        return ChemFragPart(
            mol=self.mol,
            frag_type="chemgen",
            n_BE=self.frag_structure.n_BE,
            AO_per_frag=[list(AO_indices) for AO_indices in self.AO_per_frag],
            AO_per_edge_per_frag=_extract_values(self.AO_per_edge_per_frag),
            ref_frag_idx_per_edge_per_frag=[
                list(D.values()) for D in self.frag_structure.ref_frag_idx_per_edge
            ],
            relAO_per_edge_per_frag=_extract_values(self.relAO_per_edge_per_frag),
            relAO_in_ref_per_edge_per_frag=_extract_values(
                self.relAO_in_ref_per_edge_per_frag
            ),
            relAO_per_origin_per_frag=[list(seq) for seq in relAO_per_origin],
            weight_and_relAO_per_center_per_frag=weight_and_relAO_per_center_per_frag,
            motifs_per_frag=[
                list(motifs) for motifs in self.frag_structure.motifs_per_frag
            ],
            origin_per_frag=origin_per_frag,
            H_per_motif=[
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
            frozen_core=self.frozen_core,
            iao_valence_basis=None,
            iao_valence_only=False,
            fragmented=self,
        )

    def _get_FragPart_with_iao(self, wrong_iao_indexing: bool) -> ChemFragPart:
        """Transform into a :class:`quemb.molbe.autofrag.FragPart`
        for further use in quemb.

        Matches the output of :func:`quemb.molbe.autofrag.autogen`.

        Parameters
        ----------
        wrong_iao_indexing:
            There is a suspected error in how autogen treats the ordering
            of AOs for H atoms. Do we fix it, or adhere to the wrong indexing?
        """

        assert self.iao_valence_mol is not None
        valence_frags: Final = Fragmented.from_frag_structure(
            self.iao_valence_mol, self.frag_structure, self.frozen_core, None
        )
        H_per_motif: Final = self.conn_data.H_per_motif

        #: The number of H atoms connected to motif.
        n_conn_H: Final = {
            motif: len(H_atoms) for motif, H_atoms in H_per_motif.items()
        }

        #: The number of AO indices per H atom in the small basis.
        #: Set to zero if there are no H atoms.
        n_small_AO_H: Final = (
            len(
                valence_frags.AO_per_atom[
                    # We take exemplary the first H atom
                    self.frag_structure.conn_data.H_atoms[0]  # type: ignore[call-overload]
                ]
            )
            if self.frag_structure.conn_data.H_atoms
            else 0
        )

        def _extract_with_iao_offset(
            AO_small_basis: Sequence[
                Mapping[_T_motif, Mapping[AtomIdx, OrderedSet[_T_AOIdx]]]
            ],
            AO_full_basis: Sequence[
                Mapping[_T_motif, Mapping[AtomIdx, OrderedSet[_T_AOIdx]]]
            ],
            wrong_iao_indexing: bool,
        ) -> list[list[list[_T_AOIdx]]]:
            result = []
            for fragment, fragment_big_basis in zip(AO_small_basis, AO_full_basis):
                AO_indices: list[list[_T_AOIdx]] = []
                for motif in fragment:
                    if wrong_iao_indexing:
                        offset = _iloc(fragment_big_basis[motif].values(), 1)[0]
                        H_offsets = [
                            OrderedSet(
                                cast(
                                    Sequence[_T_AOIdx],
                                    range(start, start + n_small_AO_H),
                                )
                            )
                            for start in range(
                                offset,
                                offset + n_conn_H[motif] * n_small_AO_H,
                                n_small_AO_H,
                            )
                        ]
                    else:
                        H_offsets = [
                            fragment_big_basis[motif][H_atom][:n_small_AO_H]
                            for H_atom in H_per_motif[motif]
                        ]

                    AO_index_heavy_atom = _iloc(fragment_big_basis[motif].values(), 0)[
                        : len(_iloc(fragment[motif].values(), 0))
                    ]

                    AO_indices.append(_flatten([AO_index_heavy_atom] + H_offsets))
                result.append(AO_indices)
            return result

        relAO_in_ref_per_edge: Final = _extract_with_iao_offset(
            valence_frags.relAO_in_ref_per_edge_per_frag,
            self.relAO_in_ref_per_edge_per_frag,
            wrong_iao_indexing=wrong_iao_indexing,
        )
        AO_per_edge: Final = _extract_with_iao_offset(
            valence_frags.AO_per_edge_per_frag,
            self.AO_per_edge_per_frag,
            wrong_iao_indexing=wrong_iao_indexing,
        )
        relAO_per_edge: Final = _extract_with_iao_offset(
            valence_frags.relAO_per_edge_per_frag,
            self.relAO_per_edge_per_frag,
            wrong_iao_indexing=wrong_iao_indexing,
        )

        # We have to flatten one nesting level since it is assumed in the output
        # of autogen that there is always only one origin per fragment.,
        relAO_per_origin: Final = [
            L[0]
            for L in _extract_with_iao_offset(
                valence_frags.relAO_per_origin_per_frag,
                self.relAO_per_origin_per_frag,
                wrong_iao_indexing=wrong_iao_indexing,
            )
        ]

        matched_output_no_iao = self._get_FragPart_no_iao()

        # Only are actually different when doing IAOs
        return ChemFragPart(
            mol=self.mol,
            frag_type="chemgen",
            n_BE=self.frag_structure.n_BE,
            AO_per_edge_per_frag=AO_per_edge,
            relAO_per_edge_per_frag=relAO_per_edge,
            relAO_in_ref_per_edge_per_frag=relAO_in_ref_per_edge,
            relAO_per_origin_per_frag=relAO_per_origin,
            AO_per_frag=matched_output_no_iao.AO_per_frag,
            ref_frag_idx_per_edge_per_frag=matched_output_no_iao.ref_frag_idx_per_edge_per_frag,
            weight_and_relAO_per_center_per_frag=matched_output_no_iao.weight_and_relAO_per_center_per_frag,
            motifs_per_frag=matched_output_no_iao.motifs_per_frag,
            origin_per_frag=matched_output_no_iao.origin_per_frag,
            H_per_motif=matched_output_no_iao.H_per_motif,
            add_center_atom=matched_output_no_iao.add_center_atom,
            frozen_core=self.frozen_core,
            iao_valence_basis=self.iao_valence_mol.basis,
            iao_valence_only=False,
            fragmented=self,
        )

    def get_FragPart(self, wrong_iao_indexing: bool | None = None) -> ChemFragPart:
        """Match the output of :func:`quemb.molbe.autofrag.autogen`."""
        if self.iao_valence_mol is None:
            return self._get_FragPart_no_iao()
        else:
            assert wrong_iao_indexing is not None
            return self._get_FragPart_with_iao(wrong_iao_indexing)

    def reorder_frags(self, idx: Sequence[int] | Vector[np.integer]) -> Self:
        """Reorder the fragments of self.

        Can, for example, be used to order the fragments by size.

        Parameters
        ----------
        idx :
            The new index, has to be a permutation of :python:`[0, ..., len(self)]`.
        """
        assert set(idx) == set(range(len(idx)))
        return self.__class__(
            mol=self.mol,
            conn_data=self.conn_data,
            frag_structure=self.frag_structure.reorder_frags(idx),
            AO_per_atom=self.AO_per_atom,
            AO_per_frag=_reorder(self.AO_per_frag, idx),
            AO_per_motif=self.AO_per_motif,
            AO_per_edge_per_frag=_reorder(self.AO_per_edge_per_frag, idx),
            relAO_per_center_per_frag=_reorder(self.relAO_per_center_per_frag, idx),
            relAO_per_origin_per_frag=_reorder(self.relAO_per_origin_per_frag, idx),
            relAO_in_ref_per_edge_per_frag=_reorder(
                self.relAO_in_ref_per_edge_per_frag, idx
            ),
            relAO_per_motif_per_frag=_reorder(self.relAO_per_motif_per_frag, idx),
            relAO_per_edge_per_frag=_reorder(self.relAO_per_edge_per_frag, idx),
            frozen_core=self.frozen_core,
            iao_valence_mol=self.iao_valence_mol,
        )

    def get_frag_per_atom(self) -> list[FragmentIdx]:
        """Return the fragment index where each atom is best described.

        Returns
        -------
        frag_per_atom:
            A list of fragment indices of length ``self.mol.natm``, where the
            i-th entry gives the fragment index to which atom ``i`` belongs.
        """
        frag_per_atom = {
            atom: cast(FragmentIdx, i_frag)
            for i_frag, centers in enumerate(self.frag_structure.centers_per_frag)
            for center in centers
            for atom in self.conn_data.atoms_per_motif[center]
        }
        return [
            frag_per_atom[i_atom]
            for i_atom in cast(Sequence[AtomIdx], range(self.mol.natm))
        ]


@define(kw_only=True, hash=False)
class ChemFragPart(FragPart):
    """Expose chemgen info, but retain old FragPart behaviour for outside code.

    In particular, the equality comparison uses only FragPart info."""

    fragmented: Final[Fragmented] = field(eq=False)

    __hash__ = None  # explicitly mark as unhashable

    def _get_FragPart(self) -> FragPart:
        return FragPart(
            **{f.name: getattr(self, f.name) for f in fields(FragPart) if f.init}
        )

    @override
    def __eq__(self, other: Any) -> bool:
        return self._get_FragPart() == other

    @override
    def __ne__(self, other: Any) -> bool:
        return not (self == other)

    @override
    def reorder_frags(self, idx: Sequence[int] | Vector[np.integer]) -> Self:
        """Reorder the fragments of self.

        Can, for example, be used to order the fragments by size.

        Parameters
        ----------
        idx :
            The new index, has to be a permutation of :python:`[0, ..., len(self)]`.
        """
        assert set(idx) == set(range(len(idx)))
        return self.__class__(
            mol=self.mol,
            frag_type=self.frag_type,
            n_BE=self.n_BE,
            AO_per_frag=_reorder(self.AO_per_frag, idx),
            AO_per_edge_per_frag=_reorder(self.AO_per_edge_per_frag, idx),
            ref_frag_idx_per_edge_per_frag=_reorder(
                self.ref_frag_idx_per_edge_per_frag, idx
            ),
            relAO_per_edge_per_frag=_reorder(self.relAO_per_edge_per_frag, idx),
            relAO_in_ref_per_edge_per_frag=_reorder(
                self.relAO_in_ref_per_edge_per_frag, idx
            ),
            relAO_per_origin_per_frag=_reorder(self.relAO_per_origin_per_frag, idx),
            weight_and_relAO_per_center_per_frag=_reorder(
                self.weight_and_relAO_per_center_per_frag, idx
            ),
            motifs_per_frag=_reorder(self.motifs_per_frag, idx),
            origin_per_frag=_reorder(self.origin_per_frag, idx),
            H_per_motif=self.H_per_motif,
            add_center_atom=_reorder(self.add_center_atom, idx),
            frozen_core=self.frozen_core,
            iao_valence_basis=self.iao_valence_basis,
            iao_valence_only=self.iao_valence_only,
            fragmented=self.fragmented.reorder_frags(idx),
        )


def _get_AOidx_per_atom(
    mol: _T_chemsystem, frozen_core: bool
) -> list[OrderedSet[GlobalAOIdx]]:
    """Get the range of atomic orbital indices per atom.

    Parameters
    ----------
    mol :
        The molecule to get the atomic orbital indices from.
    frozen_core :
        Do we perform a frozen core calculation?

    Returns
    -------
    list
        A list of ordered sets of atomic orbital indices per atom.
    """
    if frozen_core:
        core_offset = 0
        result = []
        core_list = get_core(mol)[2]
        for n_core, (_, _, start, stop) in zip(core_list, mol.aoslice_by_atom()):
            result.append(
                cast(
                    OrderedSet[GlobalAOIdx],
                    OrderedSet(
                        range(start - core_offset, stop - (core_offset + n_core))
                    ),
                )
            )
            core_offset += n_core
        return result
    else:
        return [
            cast(
                OrderedSet[GlobalAOIdx], OrderedSet(range(AO_offsets[2], AO_offsets[3]))
            )
            for AO_offsets in mol.aoslice_by_atom()
        ]


_T_Key2 = TypeVar("_T_Key2", bound=Hashable)


def _extract_values(
    nested: Sequence[Mapping[_T_Key, Mapping[_T_Key2, Sequence[_T_Val]]]],
) -> list[list[list[_T_Val]]]:
    """Extract the values of a mapping from a sequence of mappings"""
    return [[list(union_of_seqs(*v.values())) for v in D.values()] for D in nested]


@define(frozen=True, kw_only=True)
class ChemGenArgs:
    """Additional arguments for ChemGen fragmentation.

    These are passed on to
    :func:`quemb.molbe.chemfrag.PurelyStructureFragmented.from_mole`
    and documented there.
    """

    h_treatment: Final[HTreatment] = "treat_H_diff"
    bonds_atoms: Mapping[int, set[int]] | None = None
    vdW_radius: InVdWRadius | None = None

    #: If a fragment would be swallowed, it is instead replaced by the largest
    #: fragment that contains the smaller fragment. The definition of the origin
    #: is taken from the smaller fragment.
    #: This means, there will be no centers other than origins.
    swallow_replace: bool = False

    #: Option for debugging.
    #: If it is true, then chemgen adheres to the old **wrong** indexing
    #: of :python:`"autogen"``.
    _wrong_iao_indexing: bool = False


def chemgen(
    mol: _T_chemsystem,
    n_BE: int,
    args: ChemGenArgs | None,
    frozen_core: bool,
    iao_valence_basis: str | None,
) -> Fragmented:
    """Fragment a molecule based on chemical connectivity.

    Parameters
    ----------
    mol :
        Molecule or Cell to be fragmented.
    n_BE :
        BE fragmentation level.
    args :
        Additional arguments for ChemGen fragmentation.
        These are passed on to
        :func:`quemb.molbe.chemfrag.PurelyStructureFragmented.from_mole`
        and documented there.
    frozen_core :
        Do we perform a frozen core calculation?
    iao_valuence_basis :
        The minimal basis used for the IAO definition.
    swallow_replace :
        If a fragment would be swallowed, it is instead replaced by the largest
        fragment that contains the smaller fragment. The definition of the origin
        is taken from the smaller fragment.
        This means, there will be no centers other than origins.
    """
    if args is None:
        return Fragmented.from_mole(
            mol, n_BE=n_BE, frozen_core=frozen_core, iao_valence_basis=iao_valence_basis
        )
    else:
        return Fragmented.from_mole(
            mol,
            n_BE=n_BE,
            frozen_core=frozen_core,
            h_treatment=args.h_treatment,
            bonds_atoms=args.bonds_atoms,
            vdW_radius=args.vdW_radius,
            iao_valence_basis=iao_valence_basis,
            swallow_replace=args.swallow_replace,
        )


_T_int = TypeVar("_T_int", bound=int)


def _sort_by_keys(D: Mapping[_T_int, T]) -> dict[_T_int, T]:
    return {key: D[key] for key in sorted(D.keys())}
