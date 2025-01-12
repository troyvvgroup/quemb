from chemcoord import Cartesian


def get_BE_fragment(m: Cartesian, i: int, n_BE: int) -> set[int]:
    """Return the index of the atoms of the fragment that
    contains the i_center atom and its (n_BE - 1) coordination sphere."""
    return m.get_coordination_sphere(
        i, n_sphere=n_BE - 1, only_surface=False, give_only_index=True
    )


def cleanup_if_subset(fragment_indices: dict[int, set[int]]) -> dict[int, set[int]]:
    """Remove fragments that are subsets of other fragments."""
    result = {}
    for i_center, connected in fragment_indices.items():
        for j in connected - {i_center}:
            if connected.issubset(fragment_indices[j]):
                break
        else:
            result[i_center] = connected
    return result


def add_back_H(
    m: Cartesian, i_center: int, n_BE: int, fragment_index: set[int]
) -> Cartesian:
    """Convert to Cartesian and add back the hydrogens
    that are connected to atoms in the fragment."""
    H_index = set(
        m.get_coordination_sphere(i_center, n_sphere=n_BE, only_surface=False)
        .loc[m.atom == "H", :]
        .index
    )
    return m.loc[fragment_index | H_index, :]


def get_BE_fragments(
    m: Cartesian, n_BE: int = 3, pure_heavy: bool = True
) -> dict[int, Cartesian]:
    """Create the BE fragments for the molecule m.

    Adhere to BE literature nomenclature,
    i.e. BE(n) takes the n - 1 coordination sphere."""
    m_considered = m.loc[m.atom != "H", :] if pure_heavy else m

    fragments = {
        i_center: fragment_indices
        for i_center in m_considered.index
        if (fragment_indices := get_BE_fragment(m_considered, i_center, n_BE))
    }

    return {
        i_center: add_back_H(m, i_center, n_BE, index)
        for i_center, index in cleanup_if_subset(fragments).items()
    }
