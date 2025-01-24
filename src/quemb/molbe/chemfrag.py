from collections import defaultdict
from collections.abc import Sequence
from typing import Final, NewType, TypeAlias, cast

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
#: In the following example we have drawn two fragments,
#: one around atom A and one around atom B. The fragment
#: around atom A is completely contained in the fragment around B,
#: hence we remove it. The remaining fragment around B has the origin B,
#: and swallowed the fragment around A.
#: Hence its centers are A and B.
#: The edge is the complement of the centers, hence for the fragment around B
#: the set of edge is the one-element set {C}.
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


CenterPerFrag: TypeAlias = dict[OriginIdx, OrderedSet[CenterIdx]]
EdgePerFrag: TypeAlias = dict[OriginIdx, OrderedSet[EdgeIdx]]


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


@define
class ConnectivityData:
    """Data structure to store the connectivity data of a molecule."""

    #: The connectivity graph of the molecule.
    bonds: Final[dict[AtomIdx, OrderedSet[AtomIdx]]]
    #: The heavy atoms/motifs in the molecule.
    heavy_atoms: Final[OrderedSet[MotifIdx]]
    #: The connectivity graph of the heavy atoms, i.e. ignoring the hydrogen atoms.
    heavy_atom_bonds: Final[dict[MotifIdx, OrderedSet[MotifIdx]]]
    #: The hydrogen atoms in the molecule.
    H_atoms: Final[OrderedSet[AtomIdx]]
    #: The hydrogen atoms per motif.
    H_per_motif: Final[dict[MotifIdx, OrderedSet[AtomIdx]]]
    #: All atoms per motif. Lists the heavy atom first.
    atoms_per_motif: Final[dict[MotifIdx, OrderedSet[AtomIdx]]]

    @classmethod
    def from_cartesian(cls, m: Cartesian) -> Self:
        if not (m.index.min() == 0 and m.index.max() == len(m) - 1):
            raise ValueError("We assume 0-indexed data for the rest of the code.")
        m = m.sort_index()

        bonds = {k: OrderedSet(sorted(v)) for k, v in m.get_bonds().items()}
        heavy_atoms = OrderedSet(m.loc[m.atom != "H", :].index)
        site_bonds = {site: bonds[site] & heavy_atoms for site in heavy_atoms}
        H_atoms = OrderedSet(m.index).difference(heavy_atoms)
        H_per_motif = {i_site: bonds[i_site] & H_atoms for i_site in heavy_atoms}
        atoms_per_motif = {
            i_site: merge_seqs([i_site], H_atoms)
            for i_site, H_atoms in H_per_motif.items()
        }
        return cls(
            bonds, heavy_atoms, site_bonds, H_atoms, H_per_motif, atoms_per_motif
        )

    def get_BE_fragment(self, i_center: MotifIdx, n_BE: int) -> OrderedSet[MotifIdx]:
        """Return the BE fragment around atom :code:`i_center`.

        Return the index of the site atoms of the fragment that
        contains the i_center atom and its (n_BE - 1) coordination sphere.
        """
        if n_BE < 1:
            raise ValueError("n_BE must greater than or equal to 1.")

        result = OrderedSet({i_center})
        new = result.copy()
        for _ in range(n_BE - 1):
            new = merge_seqs(*(self.heavy_atom_bonds[i] for i in new)).difference(
                result
            )
            if not new:
                break
            result = result.union(new)
        return result

    def all_fragments_sites_only(
        self, n_BE: int
    ) -> dict[MotifIdx, OrderedSet[MotifIdx]]:
        return {i: self.get_BE_fragment(i, n_BE) for i in self.heavy_atoms}


@define
class SubsetsCleaned:
    motif_per_frag: Final[dict[OriginIdx, OrderedSet[MotifIdx]]]
    swallowed_centers: Final[CenterPerFrag]


def cleanup_if_subset(
    fragment_indices: dict[MotifIdx, OrderedSet[MotifIdx]],
) -> SubsetsCleaned:
    """Remove fragments that are subsets of other fragments."""
    contain_others: CenterPerFrag = defaultdict(OrderedSet)
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
    return SubsetsCleaned(
        {
            OriginIdx(CenterIdx(k)): v
            for k, v in fragment_indices.items()
            if k not in subset_of_others
        },
        contain_others,
    )


@define
class FragmentedMolecule:
    atoms_per_frag: Final[dict[OriginIdx, OrderedSet[AtomIdx]]]
    motifs_per_frag: Final[dict[OriginIdx, OrderedSet[MotifIdx]]]
    center_per_frag: Final[CenterPerFrag]
    edge_per_frag: Final[EdgePerFrag]
    conn_data: Final[ConnectivityData]
    n_BE: Final[int]

    @classmethod
    def from_motifs(cls, conn_data: ConnectivityData, n_BE: int) -> Self:
        fragments = cleanup_if_subset(
            {
                i_center: conn_data.get_BE_fragment(i_center, n_BE)
                for i_center in conn_data.heavy_atoms
            }
        )
        atoms_per_frag = {
            i_origin: merge_seqs(
                *[conn_data.atoms_per_motif[i_motif] for i_motif in i_fragment]
            )
            for i_origin, i_fragment in fragments.motif_per_frag.items()
        }
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
            fragments.motif_per_frag,
            center_per_frag,
            {i_origin: get_edges(i_origin) for i_origin in fragments.motif_per_frag},
            conn_data,
            n_BE,
        )

    @classmethod
    def from_cartesian(cls, mol: Cartesian, n_BE: int) -> Self:
        return cls.from_motifs(ConnectivityData.from_cartesian(mol), n_BE)

    @classmethod
    def from_Mol(cls, mol: Mole, n_BE: int) -> Self:
        return cls.from_cartesian(mol.to_pysf(), n_BE)
