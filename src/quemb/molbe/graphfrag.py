# Author(s): Shaun Weatherly

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import Final, Literal

import networkx as nx
import numpy as np
from attrs import define
from networkx import shortest_path
from numpy.linalg import norm
from pyscf import gto
from pyscf.gto import Mole

from quemb.molbe.autofrag import FragPart
from quemb.molbe.helper import get_core
from quemb.shared.typing import (
    Vector,
)


@define(frozen=True, kw_only=True)
class GraphGenArgs:
    """Graphgen specific arguments.

    Parameters
    ----------

    connectivity:
        Keyword string specifying the distance metric to be used for edge
        weights in the fragment adjacency graph. Currently supports "euclidean"
        (which uses the square of the distance between atoms in real
        space to determine connectivity within a fragment.)
    cutoff:
        Atoms with an edge weight beyond `cutoff` will be excluded from the
        `shortest_path` calculation. This is crucial when handling very large
        systems, where computing the shortest paths from all to all becomes
        non-trivial. Defaults to 20.0.
    remove_nonunique_frags:
        Whether to remove fragments which are strict subsets of another
        fragment in the system. True by default.
    """

    connectivity: Final[Literal["euclidean"]] = "euclidean"
    cutoff: Final[float] = 20.0
    remove_nonnunique_frags: Final[bool] = True


@define
class FragmentMap:
    """Dataclass for fragment bookkeeping.

    Parameters
    ----------
    AO_per_frag:
        List whose entries are sequences (tuple or list) containing
        all AO indices for a fragment.
    fs :
        List whose entries are sequences of sequences, containing AO indices per atom
        per fragment.
    AO_per_edge :
        List whose entries are sequences of sequences, containing edge AO
        indices per atom (inner tuple) per fragment (outer tuple).
    center :
        List whose entries are sequences of sequences, containing all fragment AO
        indices per atom (inner tuple) and per fragment (outer tuple).
    centerf_idx :
        List whose entries are sequences containing the relative AO index of the
        origin site within a fragment.
        Relative is to the own fragment; since the origin site is at the beginning
        of the motif list for each fragment, this is always a Sequence
        :python:`range(0, n)`.
    weight_and_relAO_per_center :
        Weights determining the energy contributions from each center site
        (ie, with respect to centerf_idx).
    sites :
        List whose entries are sequences containing all AO indices per atom
        (excluding frozen core indices, if applicable).
    dnames :
        List of strings giving fragment data names. Useful for bookkeeping and
        for constructing fragment scratch directories.
    motifs_per_frag :
        List whose entries are sequences containing all atom indices for a fragment.
    origin_per_frag :
        List whose entries are sequences giving the center atom indices per fragment.
    edge_atoms :
        List whose entries are sequences giving the edge atom indices per fragment.
    adjacency_mat :
        The adjacency matrix for all sites (atoms) in the system.
    adjacency_graph :
        The adjacency graph corresponding to `adjacency_mat`.
    """

    AO_per_frag: list[Sequence[int]]
    fs: list[Sequence[Sequence[int]]]
    AO_per_edge: list[Sequence[Sequence[int]]]
    ref_frag_idx_per_edge: list[Sequence[int]]
    relAO_per_origin: list[Sequence[int]]
    weight_and_relAO_per_center: list[Sequence]
    sites: list[Sequence]
    dnames: list[str]
    motifs_per_frag: list[Sequence[int]]
    origin_per_frag: list[Sequence[int]]
    edge_atoms: list[Sequence[int]]
    adjacency_mat: np.ndarray
    adjacency_graph: nx.Graph

    def remove_nonnunique_frags(self, natm: int) -> None:
        """Remove all fragments which are strict subsets of another.

        Remove all fragments whose AO indices can be identified as subsets of
        another fragment's. The center site for the removed frag is then
        added to that of the superset. Because doing so will necessarily
        change the definition of fragments, we repeat it up to `natm` times
        such that all fragments are guaranteed to be distinct sets.
        another fragment's. The center site for the removed frag is then
        added to that of the superset. Because doing so will necessarily
        change the definition of fragments, we repeat it up to `natm` times
        such that all fragments are guaranteed to be distinct sets.
        """
        for _ in range(0, natm):
            subsets = set()
            for adx, basa in enumerate(self.AO_per_frag):
                for bdx, basb in enumerate(self.AO_per_frag):
                    if adx == bdx:
                        pass
                    elif set(basb).issubset(set(basa)):
                        subsets.add(bdx)
                        self.ref_frag_idx_per_edge[adx] = tuple(
                            set(
                                list(self.ref_frag_idx_per_edge[adx])
                                + list(deepcopy(self.ref_frag_idx_per_edge[bdx]))
                            )
                        )
                        self.origin_per_frag[adx] = tuple(
                            set(
                                list(self.origin_per_frag[adx])
                                + list(deepcopy(self.origin_per_frag[bdx]))
                            )
                        )
            if subsets:
                sorted_subsets = sorted(subsets, reverse=True)
                for bdx in sorted_subsets:
                    del self.ref_frag_idx_per_edge[bdx]
                    del self.AO_per_frag[bdx]
                    del self.fs[bdx]
                    del self.origin_per_frag[bdx]
                    del self.motifs_per_frag[bdx]

        return None

    def to_FragPart(self, mol: Mole, n_BE: int, frozen_core: bool) -> FragPart:
        MISSING = []  # type: ignore[var-annotated]
        MISSING_PER_FRAG = [[] for _ in range(len(self.AO_per_frag))]  # type: ignore[var-annotated]
        return FragPart(
            mol=mol,
            frag_type="graphgen",
            n_BE=n_BE,
            AO_per_edge_per_frag=self.AO_per_edge,  # type: ignore[arg-type]
            relAO_per_edge_per_frag=MISSING_PER_FRAG,
            relAO_in_ref_per_edge_per_frag=MISSING_PER_FRAG,
            relAO_per_origin_per_frag=self.relAO_per_origin,  # type: ignore[arg-type]
            AO_per_frag=self.AO_per_frag,  # type: ignore[arg-type]
            ref_frag_idx_per_edge_per_frag=self.ref_frag_idx_per_edge,  # type: ignore[arg-type]
            weight_and_relAO_per_center_per_frag=self.weight_and_relAO_per_center,  # type: ignore[arg-type]
            motifs_per_frag=self.motifs_per_frag,  # type: ignore[arg-type]
            origin_per_frag=self.origin_per_frag,  # type: ignore[arg-type]
            H_per_motif=MISSING,
            add_center_atom=MISSING,
            frozen_core=frozen_core,
            iao_valence_basis=None,
            iao_valence_only=False,
        )


def euclidean_distance(
    i_coord: Vector,
    j_coord: Vector,
) -> np.floating:
    return norm(i_coord - j_coord)


def graphgen(
    mol: gto.Mole,
    n_BE: int = 2,
    frozen_core: bool = True,
    remove_nonunique_frags: bool = True,
    frag_prefix: str = "f",
    connectivity: str = "euclidean",
    iao_valence_basis: str | None = None,
    cutoff: float = 20.0,
) -> FragmentMap:
    """Generate fragments via adjacency graph.

    Generalizes the BEn fragmentation scheme to arbitrary fragment sizes using a
    graph theoretic heuristic. In brief: atoms are assigned to nodes in an
    adjacency graph and edges are weighted by some distance metric. For a given
    fragment center site, Dijkstra's algorithm is used to find the shortest path
    from that center to its neighbors. The number of nodes visited on that shortest
    path determines the degree of separation of the corresponding neighbor. I.e.,
    all atoms whose shortest paths from the center site visit at most 1 node must
    be direct neighbors to the center site, which gives BE2-type fragments; all
    atoms whose shortest paths visit at most 2 nodes must then be second-order
    neighbors, hence BE3; and so on.

    Currently does not support periodic calculations.

    Parameters
    ----------
    mol :
        The molecule object.
    n_BE:
        The order of nearest neighbors (with respect to the center atom)
        included in a fragment. Supports all ``n_BE``, with ``n_BE`` in
        ``[1, 2, 3, 4, 5, 6, 7, 8, 9]`` having been tested.
    frozen_core:
        Whether to exclude core AO indices from the fragmentation process.
        True by default.
    remove_nonunique_frags:
        Whether to remove fragments which are strict subsets of another
        fragment in the system. True by default.
    frag_prefix:
        Prefix to be appended to the fragment datanames. Useful for managing
        fragment scratch directories.
    connectivity:
        Keyword string specifying the distance metric to be used for edge
        weights in the fragment adjacency graph. Currently supports "euclidean"
        (which uses the square of the distance between atoms in real
        space to determine connectivity within a fragment.)
    cutoff:
        Atoms with an edge weight beyond `cutoff` will be excluded from the
        `shortest_path` calculation. This is crucial when handling very large
        systems, where computing the shortest paths from all to all becomes
        non-trivial. Defaults to 20.0.

    Returns
    -------
    FragmentMap :
        FragmentMap mapping various fragment components to AO indices, data names,
        and other info.
    """
    assert mol is not None
    if iao_valence_basis is not None:
        raise NotImplementedError("IAOs not yet implemented for graphgen.")

    natm = mol.natm

    adx_map = {
        adx: {
            "bas": bas,
            "label": mol.atom_symbol(adx),
            "coord": mol.atom_coord(adx),
            "shortest_paths": dict(),
        }
        for adx, bas in enumerate(mol.aoslice_by_atom())
    }

    fragment_map = FragmentMap(
        AO_per_frag=(list(tuple())),
        fs=list(tuple(tuple())),
        AO_per_edge=list(tuple(tuple())),
        ref_frag_idx_per_edge=list(tuple()),
        relAO_per_origin=list(tuple()),
        weight_and_relAO_per_center=list(tuple()),
        sites=list(tuple()),
        dnames=list(),
        motifs_per_frag=list(),
        origin_per_frag=list(),
        edge_atoms=list(),
        adjacency_mat=np.zeros((natm, natm), np.float64),
        adjacency_graph=nx.Graph(),
    )
    fragment_map.adjacency_graph.add_nodes_from(adx_map)

    _core_offset = 0
    for adx, map in adx_map.items():
        start_ = map["bas"][2]
        stop_ = map["bas"][3]
        if frozen_core:
            _, _, core_list = get_core(mol)
            start_ -= _core_offset
            ncore_ = int(core_list[adx])
            stop_ -= _core_offset + ncore_
            _core_offset += ncore_
            fragment_map.sites.append(tuple([i for i in range(start_, stop_)]))
        else:
            fragment_map.sites.append(tuple([i for i in range(start_, stop_)]))

    if connectivity.lower() in ["euclidean_distance", "euclidean"]:
        # Begin by constructing the adjacency matrix and adjacency graph
        # for the system. Each node corresponds to an atom, such that each
        # pair of nodes can be assigned an edge weighted by the square of
        # their distance in real space.
        for adx in range(natm):
            for bdx in range(adx + 1, natm):
                dr = (
                    euclidean_distance(
                        adx_map[adx]["coord"],
                        adx_map[bdx]["coord"],
                    )
                    ** 2
                )
                fragment_map.adjacency_mat[adx, bdx] = dr
                if dr <= cutoff:
                    fragment_map.adjacency_graph.add_edge(adx, bdx, weight=dr)

        # For a given center site (adx), find the set of shortest
        # paths to all other sites. The number of nodes visited
        # on that path gives the degree of separation of the
        # sites.
        for adx, map in adx_map.items():
            fragment_map.origin_per_frag.append((adx,))
            fragment_map.ref_frag_idx_per_edge.append(deepcopy(fragment_map.sites[adx]))
            AO_per_frag_tmp = deepcopy(list(fragment_map.sites[adx]))
            fatoms_temp = [adx]
            fs_temp = []
            fs_temp.append(deepcopy(fragment_map.sites[adx]))

            for bdx, _ in adx_map.items():
                if fragment_map.adjacency_graph.has_edge(adx, bdx):
                    map["shortest_paths"].update(
                        {
                            bdx: shortest_path(
                                fragment_map.adjacency_graph,
                                source=adx,
                                target=bdx,
                                weight=lambda a, b, _: (
                                    fragment_map.adjacency_graph[a][b]["weight"]
                                ),
                                method="dijkstra",
                            )
                        }
                    )

            # If the degree of separation is smaller than the *n*
            # in your fragment type, BE*n*, then that site is appended to
            # the set of fragment sites for adx.
            for bdx, path in map["shortest_paths"].items():
                if 0 < (len(path) - 1) < n_BE:
                    AO_per_frag_tmp = AO_per_frag_tmp + deepcopy(
                        list(fragment_map.sites[bdx])
                    )
                    fs_temp.append(deepcopy(fragment_map.sites[bdx]))
                    fatoms_temp.append(bdx)

            fragment_map.AO_per_frag.append(tuple(AO_per_frag_tmp))
            fragment_map.fs.append(tuple(fs_temp))
            fragment_map.motifs_per_frag.append(tuple(fatoms_temp))

    elif connectivity.lower() in ["resistance_distance", "resistance"]:
        raise NotImplementedError("Work in progress...")

    elif connectivity.lower() in ["entanglement"]:
        raise NotImplementedError("Work in progress...")

    else:
        raise AttributeError(f"Connectivity metric not recognized: '{connectivity}'")

    if remove_nonunique_frags:
        fragment_map.remove_nonnunique_frags(natm)

    # Define the 'edges' for fragment A as the intersect of its sites
    # with the set of all center sites outside of A:
    for adx, fs in enumerate(fragment_map.fs):
        edge_temp: set[tuple] = set()
        eatoms_temp: set[tuple[int, ...]] = set()
        for bdx, center in enumerate(fragment_map.ref_frag_idx_per_edge):
            if adx == bdx:
                pass
            else:
                for f in fs:
                    overlap = set(f).intersection(set(center))
                    if overlap:
                        f_temp = set(fragment_map.motifs_per_frag[adx])
                        c_temp = set(fragment_map.origin_per_frag[bdx])
                        edge_temp.add(tuple(overlap))
                        eatoms_temp.add(tuple(i for i in f_temp.intersection(c_temp)))
        fragment_map.AO_per_edge.append(tuple(edge_temp))
        fragment_map.edge_atoms.extend(tuple(eatoms_temp))

    # Update relative center site indices (centerf_idx) and weights
    # for center site contributions to the energy ():
    for adx, center in enumerate(fragment_map.ref_frag_idx_per_edge):
        centerf_idx = tuple(fragment_map.AO_per_frag[adx].index(cdx) for cdx in center)
        fragment_map.relAO_per_origin.append(centerf_idx)
        fragment_map.weight_and_relAO_per_center.append((1.0, tuple(centerf_idx)))

    # Finally, set fragment data names for scratch and bookkeeping:
    for adx, _ in enumerate(fragment_map.fs):
        fragment_map.dnames.append(str(frag_prefix) + str(adx))

    return fragment_map