from collections import defaultdict
from collections.abc import Sequence
from typing import Final, NewType, cast

import chemcoord as cc
from attr import define
from chemcoord import Cartesian
from ordered_set import OrderedSet
from pyscf.gto import Mole
from typing_extensions import Self

from quemb.shared.typing import T

#: The index of an atom.
AtomIdx = NewType("AtomIdx", int)

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


def merge_seqs(*seqs: Sequence[T]) -> OrderedSet[T]:
    """Merge multiple sequences into a single :class:`OrderedSet`.

    This preserves the order of the elements in each sequence,
    and of the arguments to this function, but removes duplicates.
    (Always the first occurrence of an element is kept.)

    .. code-block:: python

        merge_seq([1, 2], [2, 3], [1, 4]) -> OrderedSet([1, 2, 3, 4])
    """
    # mypy wrongly complains that the arg type is not valid, which it is.
    return OrderedSet().union(*seqs)  # type: ignore[arg-type]


@define(frozen=True)
class ConnectivityData:
    """Data structure to store the connectivity data of a molecule."""

    #: The connectivity graph of the molecule.
    bonds: Final[dict[AtomIdx, OrderedSet[AtomIdx]]]
    #: The heavy atoms/motifs in the molecule. If hydrogens are not treated differently
    #: then every hydrogen is also a motif on its own.
    motifs: Final[OrderedSet[MotifIdx]]
    #: The connectivity graph solely of the motifs,
    # i.e. of the heavy atoms when ignoring the hydrogen atoms.
    motif_bonds: Final[dict[MotifIdx, OrderedSet[MotifIdx]]]
    #: The hydrogen atoms in the molecule. If hydrogens are not treated differently,
    #: then this is an empty set.
    H_atoms: Final[OrderedSet[AtomIdx]]
    #: The hydrogen atoms per motif. If hydrogens are not treated differently,
    #: then the values of the dictionary are empty sets.
    H_per_motif: Final[dict[MotifIdx, OrderedSet[AtomIdx]]]
    #: All atoms per motif. Lists the motif/heavy atom first.
    atoms_per_motif: Final[dict[MotifIdx, OrderedSet[AtomIdx]]]
    #: Do we treat hydrogens differently?
    treat_H_different: Final[bool] = True

    @classmethod
    def from_cartesian(cls, m: Cartesian, treat_H_different: bool = True) -> Self:
        """Create a :class:`ConnectivityData` from a :class:`chemcoord.Cartesian`.

        Parameters
        ----------
        m :
            The Cartesian object to extract the connectivity data from.
        treat_H_different :
            If True, we treat hydrogen atoms differently from heavy atoms.
        """
        if not (m.index.min() == 0 and m.index.max() == len(m) - 1):
            raise ValueError("We assume 0-indexed data for the rest of the code.")
        m = m.sort_index()

        with cc.constants.RestoreElementData():
            # temporarily increase van der Waals radius by 15 %
            cc.constants.elements.loc[:, "atomic_radius_cc"] *= 1.15
            bonds = {k: OrderedSet(sorted(v)) for k, v in m.get_bonds().items()}

        if treat_H_different:
            motifs = OrderedSet(m.loc[m.atom != "H", :].index)
        else:
            motifs = OrderedSet(m.index)
        site_bonds = {site: bonds[site] & motifs for site in motifs}
        H_atoms = OrderedSet(m.index).difference(motifs)
        H_per_motif = {i_site: bonds[i_site] & H_atoms for i_site in motifs}
        atoms_per_motif = {
            i_site: merge_seqs([i_site], H_atoms)
            for i_site, H_atoms in H_per_motif.items()
        }
        return cls(
            bonds,
            motifs,
            site_bonds,
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
            new = merge_seqs(*(self.motif_bonds[i] for i in new)).difference(result)
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
class SubsetsCleaned:
    """Data class to contain the results of the :func:`cleanup_if_subset` function.

    Currently, this data class is only used internally
    and strictly assumes that there is exactly one unique origin per
    fragment and one unique fragment per origin.
    Otherwise the data structure of a
    :python:`dict[OriginIdx, OrderedSet[MotifIdx]]`
    would not make sense.
    This assumption makes the code in :func:`cleanup_if_subset`
    much easier to write and more performant,
    but it is not true for all possible fragmentations.
    For example pair-wise fragmentations, where there are multiple
    fragments for one origin, are not supported.

    The rest of the code, however, is written in a way that would fully support
    pair-wise fragmentations, if we would use a different data structure
    here and rewrote :func:`cleanup_if_subset` accordingly.
    """

    #: The remaining fragments after removing subsets.
    #: This is a dictionary mapping the origin index to the set of motif indices.
    motif_per_frag: Final[dict[OriginIdx, OrderedSet[MotifIdx]]]
    #: The centers that are swallowed by the larger fragment whose center index
    #: becomes the origin index.
    swallowed_centers: Final[dict[OriginIdx, OrderedSet[CenterIdx]]]


def cleanup_if_subset(
    fragment_indices: dict[MotifIdx, OrderedSet[MotifIdx]],
) -> SubsetsCleaned:
    """Remove fragments that are subsets of other fragments.

    We also keep track of the Center indices that are swallowed by the
    larger fragment whose center index becomes the origin index.

    Parameters
    ----------
    fragment_indices :
        A dictionary mapping the center index to the set of motif indices.

    Returns
    -------
    SubsetsCleaned :
        The cleaned fragments and a dictionary to keep track of swallowed centers.
    """
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

    # We know that the first element of motifs is the cetner, which should
    # stay at the first position. The rest of the motifs should be sorted.
    # We also remove the swallowed centers, i.e. only origins are left.
    cleaned_fragments = {
        OriginIdx(CenterIdx(i_center)): merge_seqs([i_center], sorted(motifs[1:]))
        for i_center, motifs in fragment_indices.items()
        if i_center not in subset_of_others
    }
    return SubsetsCleaned(cleaned_fragments, contain_others)


@define(frozen=True)
class FragmentedStructure:
    """Data structure to store the fragments of a molecule.

    This takes into account only the geometrical data and is
    independent of the basis sets or the electronic structure.
    """

    #: The atomic orbital indices per fragment
    atoms_per_frag: Final[Sequence[OrderedSet[AtomIdx]]]
    #: The motifs per fragment.
    #: Note that the set of motifs in the fragment
    #: is the union of centers and edges.
    motifs_per_frag: Final[Sequence[OrderedSet[MotifIdx]]]
    #: The centers per fragment.
    #: Note that the set of centers is the complement of the edges.
    center_per_frag: Final[Sequence[OrderedSet[CenterIdx]]]
    #: The edges per fragment.
    #: Note that the set of edges is the complement of the centers.
    edge_per_frag: Final[Sequence[OrderedSet[EdgeIdx]]]
    #: The origins per frag
    origin_per_frag: Final[Sequence[OrderedSet[OriginIdx]]]
    #: Connectivity data of the molecule.
    conn_data: Final[ConnectivityData]
    n_BE: Final[int]

    @classmethod
    def from_motifs(cls, conn_data: ConnectivityData, n_BE: int) -> Self:
        fragments = cleanup_if_subset(
            {
                i_center: conn_data.get_BE_fragment(i_center, n_BE)
                for i_center in conn_data.motifs
            }
        )
        atoms_per_frag = [
            merge_seqs(*[conn_data.atoms_per_motif[i_motif] for i_motif in i_fragment])
            for i_fragment in fragments.motif_per_frag.values()
        ]
        center_per_frag = {
            i_origin: merge_seqs(
                [i_origin], fragments.swallowed_centers.get(i_origin, [])
            )
            for i_origin in fragments.motif_per_frag
        }

        def get_edges(i_origin: OriginIdx) -> OrderedSet[EdgeIdx]:
            # the complement of the center set is the edge set,
            # we can rightfully cast the result to EdgeIdx.
            return cast(
                OrderedSet[EdgeIdx],
                fragments.motif_per_frag[i_origin].difference(
                    center_per_frag[i_origin]
                ),
            )

        return cls(
            atoms_per_frag,
            list(fragments.motif_per_frag.values()),
            list(center_per_frag.values()),
            [get_edges(i_origin) for i_origin in fragments.motif_per_frag],
            [OrderedSet([i_origin]) for i_origin in fragments.motif_per_frag],
            conn_data,
            n_BE,
        )

    @classmethod
    def from_cartesian(
        cls, mol: Cartesian, n_BE: int, treat_H_different: bool = True
    ) -> Self:
        """Construct a :class:`FragmentedMolecule` from a :class:`chemcoord.Cartesian`.

        Parameters
        ----------
        mol :
            The Cartesian object to extract the connectivity data from.
        n_BE :
            The coordination sphere to consider.
        treat_H_different :
            If True, we treat hydrogen atoms differently from heavy atoms.
        """
        return cls.from_motifs(
            ConnectivityData.from_cartesian(mol, treat_H_different), n_BE
        )

    @classmethod
    def from_Mol(cls, mol: Mole, n_BE: int, treat_H_different: bool = True) -> Self:
        """Construct a :class:`FragmentedMolecule` from a :class:`pyscf.gto.mole.Mole`.

        Parameters
        ----------
        mol :
            The Molecule to extract the connectivity data from.
        n_BE :
            The coordination sphere to consider.
        treat_H_different :
            If True, we treat hydrogen atoms differently from heavy atoms.
        """
        return cls.from_cartesian(Cartesian.from_pyscf(mol), n_BE, treat_H_different)
