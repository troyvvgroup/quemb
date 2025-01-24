from collections import defaultdict
from collections.abc import Sequence
from typing import Final, NewType, Self, TypeAlias, cast

from attr import define
from chemcoord import Cartesian
from ordered_set import OrderedSet
from pyscf.gto import Mole

from quemb.shared.typing import T

#: The index of an atomic orbital.
AOIdx = NewType("AOIdx", int)


#: The index of an atom.
AtomIdx = NewType("AtomIdx", int)

CenterIdx = NewType("CenterIdx", AtomIdx)
EdgeIdx = NewType("EdgeIdx", AtomIdx)
MotifIdx: TypeAlias = CenterIdx | EdgeIdx

OriginIdx = NewType("OriginIdx", CenterIdx)


#: A dictionary that maps an atom index, the center,
#: to a set of center indices.
#: At the minimum this is just a dictionary that maps a center index
#: to a one-element set containing itself,
#: but if other fragments are contained in the fragment of the key :code:`i_center`,
#: then the set can also contain other center indices.
#: I.e.
#: .. code-block:: python
#:
#:      j_center in contained_center_indices[i_center]
#:      # implies
#:      fragments[j_center] <= fragments[i_center]
#:
#: Note the following property for the case of exactly equal fragments:
#:
#: .. code-block:: python
#:
#:      if (j_center in contained_center_indices[i_center])
#            and fragments[j_center] == fragments[i_center]):
#:          i_center <= j_center
ContainedCenterIdx = NewType(
    "ContainedCenterIdx", dict[OriginIdx, OrderedSet[CenterIdx]]
)

#: A dictionary that maps an atom index, the center,
#: to a set of AO indices in the corresponding BE fragment.
AOPerFrag = NewType("AOPerFrag", dict[CenterIdx, OrderedSet[AOIdx]])


def merge_sets(*sets: Sequence[T]) -> OrderedSet[T]:
    # mypy wrongly complains that the arg type is not valid, which it is.
    return OrderedSet().union(*sets)  # type: ignore[arg-type]


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
            i_site: merge_sets([i_site], H_atoms)
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
            new = merge_sets(*(self.heavy_atom_bonds[i] for i in new)).difference(
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
    contained_center_indices: Final[ContainedCenterIdx]


def cleanup_if_subset(
    fragment_indices: dict[MotifIdx, OrderedSet[MotifIdx]],
) -> SubsetsCleaned:
    """Remove fragments that are subsets of other fragments."""
    contain_others = ContainedCenterIdx(defaultdict(OrderedSet))
    subset_of_others: set[CenterIdx] = set()

    for i_center, i_fragment in fragment_indices.items():
        if i_center in subset_of_others:
            continue
        for j_center in i_fragment:
            if i_center == j_center:
                continue
            j_center = cast(CenterIdx, j_center)
            if fragment_indices[j_center].issubset(i_fragment):
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
    motif_per_frag: Final[dict[OriginIdx, OrderedSet[MotifIdx]]]
    contained_center_indices: Final[ContainedCenterIdx]
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
            i_origin: merge_sets(
                *[conn_data.atoms_per_motif[i_motif] for i_motif in i_fragment]
            )
            for i_origin, i_fragment in fragments.motif_per_frag.items()
        }
        return cls(
            atoms_per_frag,
            fragments.motif_per_frag,
            fragments.contained_center_indices,
            conn_data,
            n_BE,
        )

    @classmethod
    def from_cartesian(cls, mol: Cartesian, n_BE: int) -> Self:
        return cls.from_motifs(ConnectivityData.from_cartesian(mol), n_BE)

    @classmethod
    def from_Mol(cls, mol: Mole, n_BE: int) -> Self:
        return cls.from_cartesian(mol.to_pysf(), n_BE)


# def add_back_H(m: Cartesian, n_BE: int, fragments: AtomPerFrag) -> AtomPerFrag:
#     """Add back the non-considered hydrogens

#     If we considered only non-hydrogen atoms before, i.e. :code:`pure_heavy is True`,
#     then add back the hydrogens."""
#     only_H = OrderedSet(m.loc[m.atom == "H", :].index)
#     m.get_bonds(set_lookup=True)

#     def get_BE_coord_sphere(i_center: CenterIdx) -> OrderedSet[AtomIdx]:
#         return OrderedSet(
#             m.get_coordination_sphere(
#                 i_center, n_sphere=n_BE, only_surface=False, use_lookup=True
#             ).index
#         )

#     return AtomPerFrag(
#         {
#             i_center: fragment_index.union(get_BE_coord_sphere(i_center) & only_H)
#             for i_center, fragment_index in fragments.items()
#         }
#     )


# # def get_BE_fragments(
# #     m: Cartesian, n_BE: int = 3, pure_heavy: bool = True
# # ) -> FragmentedMolecule:
# #     """Create the BE fragments for the molecule m.

# #     Adhere to BE literature nomenclature,
# #     i.e. BE(n) takes the n - 1 coordination sphere."""
# #     m_considered = m.loc[m.atom != "H", :] if pure_heavy else m
# #     fragments = cleanup_if_subset(
# #         AtomPerFrag(
# #             {
# #                 i_center: get_BE_fragment(m_considered, i_center, n_BE)
# #                 for i_center in m_considered.index
# #             }
# #         )
# #     )
# #     if pure_heavy:
# #         return FragmentedMolecule(
# #             add_back_H(m, n_BE, fragments.atom_per_frag),
# #             fragments.contained_center_indices,
# #         )
# #     return fragments


# def get_fs(
#     mol: Mole, fragments: AtomPerFrag
# ) -> dict[CenterIdx, dict[AtomIdx, OrderedSet[AOIdx]]]:
#     atom_to_AO = [
#         OrderedSet(AOIdx(i) for i in range(AO_offsets[2], AO_offsets[3]))
#         for AO_offsets in mol.aoslice_by_atom()
#     ]

#     def AO_indices_of_fragment(
#         fragment: OrderedSet[AtomIdx],
#     ) -> dict[AtomIdx, OrderedSet[AOIdx]]:
#         # mypy wrongly complains that set(range(n)) is not valid, which it is.
#         return {i_atom: atom_to_AO[i_atom] for i_atom in fragment}

#     return {
#         i_center: AO_indices_of_fragment(fragment)
#         for i_center, fragment in fragments.items()
#     }


# def get_fsites(mol: Mole, fragments: AtomPerFrag) -> AOPerFrag:
#     return AOPerFrag(
#         {
#             i_center: merge_sets(*i_fragment.values())
#             for i_center, i_fragment in get_fs(mol, fragments).items()
#         }
#     )


# # def get_edge_idx(fragments: AtomPerFrag) -> dict[CenterIdx, OrderedSet[CenterIdx]]:
# #     return {
# #         i_center: {
# #             j_center
# #             for j_center, j_fragment in fragments.items()
# #             if (i_fragment & j_fragment)
# #         }
# #         for i_center, i_fragment in fragments.items()
# #     }
