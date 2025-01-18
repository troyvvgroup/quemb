from itertools import chain
from typing import NewType

from chemcoord import Cartesian
from pyscf.gto import Mole

AtomIdx = NewType("AtomIdx", int)

AOIdx = NewType("AOIdx", int)

CenterIdx = NewType("CenterIdx", AtomIdx)

#: A dictionary that maps an atom index, the center,
#: to a set of atom indices, the fragment.
AtomPerFrag = NewType("AtomPerFrag", dict[CenterIdx, set[AtomIdx]])

#: A dictionary that maps an atom index, the center,
#: to a set of AO indices in the corresponding BE fragment.
AOPerFrag = NewType("AOPerFrag", dict[CenterIdx, set[AOIdx]])


def get_BE_fragment(m: Cartesian, i: int, n_BE: int) -> set[AtomIdx]:
    """Return the BE fragment around atom :code:`i`.

    Return the index of the atoms of the fragment that
    contains the i_center atom and its (n_BE - 1) coordination sphere."""
    if m.index.min() != 0:
        raise ValueError("We assume 0-indexed data for the rest of the code.")
    m.get_bonds(set_lookup=True)
    return m.get_coordination_sphere(
        i, n_sphere=n_BE - 1, only_surface=False, give_only_index=True, use_lookup=True
    )


def cleanup_if_subset(fragment_indices: AtomPerFrag) -> AtomPerFrag:
    """Remove fragments that are subsets of other fragments."""
    result = {}
    for i_center, connected in fragment_indices.items():
        for j in connected:
            if j == i_center:
                continue
            if connected <= fragment_indices[CenterIdx(j)]:
                break
        else:
            result[i_center] = connected
    return AtomPerFrag(result)


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
) -> AtomPerFrag:
    """Create the BE fragments for the molecule m.

    Adhere to BE literature nomenclature,
    i.e. BE(n) takes the n - 1 coordination sphere."""
    m_considered = m.loc[m.atom != "H", :] if pure_heavy else m
    fragments = AtomPerFrag(
        {
            i_center: get_BE_fragment(m_considered, i_center, n_BE)
            for i_center in m_considered.index
        }
    )

    if pure_heavy:
        return add_back_H(m, n_BE, cleanup_if_subset((fragments)))
    return cleanup_if_subset((fragments))


def get_fsites(mol: Mole, fragments: AtomPerFrag) -> AOPerFrag:
    atom_to_AO = [
        range(AO_offsets[2], AO_offsets[3]) for AO_offsets in mol.aoslice_by_atom()
    ]

    def AO_indices_of_fragment(fragment: set[AtomIdx]) -> set[AOIdx]:
        return {
            AOIdx(AO_idx)
            for AO_idx in chain(*(atom_to_AO[i_atom] for i_atom in fragment))
        }

    return AOPerFrag(
        {
            i_center: AO_indices_of_fragment(fragment)
            for i_center, fragment in fragments.items()
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
