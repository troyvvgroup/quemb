from typing import Final, NewType

from attr import define
from chemcoord import Cartesian
from pyscf.gto import Mole

from quemb.shared.typing import T

AtomIdx = NewType("AtomIdx", int)

AOIdx = NewType("AOIdx", int)

CenterIdx = NewType("CenterIdx", AtomIdx)

#: A dictionary that maps an atom index, the center,
#: to a set of atom indices, the corresponding fragment.
AtomPerFrag = NewType("AtomPerFrag", dict[CenterIdx, set[AtomIdx]])


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
ContainedCenterIdx = NewType("ContainedCenterIdx", dict[CenterIdx, set[CenterIdx]])


def merge_sets(*sets: set[T]) -> set[T]:
    return set().union(*sets)


@define
class FragmentedMolecule:
    atom_per_frag: Final[AtomPerFrag]
    contained_center_indices: Final[ContainedCenterIdx]


#: A dictionary that maps an atom index, the center,
#: to a set of AO indices in the corresponding BE fragment.
AOPerFrag = NewType("AOPerFrag", dict[CenterIdx, set[AOIdx]])


def get_BE_fragment(m: Cartesian, i: int, n_BE: int) -> set[AtomIdx]:
    """Return the BE fragment around atom :code:`i`.

    Return the index of the atoms of the fragment that
    contains the i_center atom and its (n_BE - 1) coordination sphere.
    """
    if m.index.min() != 0:
        raise ValueError("We assume 0-indexed data for the rest of the code.")
    m.get_bonds(set_lookup=True)
    return m.get_coordination_sphere(
        i, n_sphere=n_BE - 1, only_surface=False, give_only_index=True, use_lookup=True
    )


def cleanup_if_subset(fragment_indices: AtomPerFrag) -> FragmentedMolecule:
    """Remove fragments that are subsets of other fragments."""
    result = AtomPerFrag({})
    contained_center_indices = ContainedCenterIdx(
        {i_center: {i_center} for i_center in fragment_indices}
    )
    for i_center, i_fragment in fragment_indices.items():
        for j_center in i_fragment:
            if j_center == i_center:
                continue
            if i_fragment <= fragment_indices[CenterIdx(j_center)]:
                if i_center > j_center:
                    contained_center_indices[CenterIdx(j_center)].add(i_center)
                    break
        else:
            result[i_center] = i_fragment

    return FragmentedMolecule(
        AtomPerFrag(result),
        ContainedCenterIdx(
            {k: v for k, v in contained_center_indices.items() if k in result}
        ),
    )


def add_back_H(m: Cartesian, n_BE: int, fragments: AtomPerFrag) -> AtomPerFrag:
    """Add back the non-considered hydrogens

    If we considered only non-hydrogen atoms before, i.e. :code:`pure_heavy is True`,
    then add back the hydrogens."""
    m.get_bonds(set_lookup=True)
    return AtomPerFrag(
        {
            i_center: fragment_index
            | set(
                m.get_coordination_sphere(
                    i_center, n_sphere=n_BE, only_surface=False, use_lookup=True
                )
                .loc[m.atom == "H", :]
                .index
            )
            for i_center, fragment_index in fragments.items()
        }
    )


def get_BE_fragments(
    m: Cartesian, n_BE: int = 3, pure_heavy: bool = True
) -> FragmentedMolecule:
    """Create the BE fragments for the molecule m.

    Adhere to BE literature nomenclature,
    i.e. BE(n) takes the n - 1 coordination sphere."""
    m_considered = m.loc[m.atom != "H", :] if pure_heavy else m
    fragments = cleanup_if_subset(
        AtomPerFrag(
            {
                i_center: get_BE_fragment(m_considered, i_center, n_BE)
                for i_center in m_considered.index
            }
        )
    )
    if pure_heavy:
        return FragmentedMolecule(
            add_back_H(m, n_BE, fragments.atom_per_frag),
            fragments.contained_center_indices,
        )
    return fragments


def get_fs(
    mol: Mole, fragments: AtomPerFrag
) -> dict[CenterIdx, dict[AtomIdx, set[AOIdx]]]:
    atom_to_AO = [
        range(AO_offsets[2], AO_offsets[3]) for AO_offsets in mol.aoslice_by_atom()
    ]

    def AO_indices_of_fragment(fragment: set[AtomIdx]) -> dict[AtomIdx, set[AOIdx]]:
        # mypy wrongly complains that set(range(n)) is not valid, which it is.
        return {i_atom: set(atom_to_AO[i_atom]) for i_atom in fragment}  # type: ignore[arg-type]

    return {
        i_center: AO_indices_of_fragment(fragment)
        for i_center, fragment in fragments.items()
    }


def get_fsites(mol: Mole, fragments: AtomPerFrag) -> AOPerFrag:
    return AOPerFrag(
        {
            i_center: merge_sets(*i_fragment.values())
            for i_center, i_fragment in get_fs(mol, fragments).items()
        }
    )


def get_edge_idx(fragments: AtomPerFrag) -> dict[CenterIdx, set[CenterIdx]]:
    return {
        i_center: {
            j_center
            for j_center, j_fragment in fragments.items()
            if (i_fragment & j_fragment)
        }
        for i_center, i_fragment in fragments.items()
    }
