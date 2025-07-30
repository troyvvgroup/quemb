# Author(s): Shaun Weatherly
from __future__ import annotations

import re
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Final, Generator, Literal

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from attrs import define
from networkx import shortest_path
from numpy.linalg import norm
from pyscf import gto

from quemb.molbe.autofrag import FragPart
from quemb.molbe.helper import get_core
from quemb.shared.typing import Vector


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
    cutoff: Final[float] = 0.0
    remove_nonnunique_frags: Final[bool] = True


@define(frozen=False, kw_only=True)
class GraphGenUtility:
    """Utility functions for handling graphs in `graphgen()`."""

    @staticmethod
    def _euclidean_distance(
        i_coord: Vector,
        j_coord: Vector,
    ) -> np.floating:
        return norm(i_coord - j_coord)

    @staticmethod
    def _graph_to_string(
        graph: nx.Graph,
        options: dict | None = None,
    ) -> Generator:
        options = {"with_labels": True} if (options is None) else options
        for element in nx.generate_network_text(graph, **options):
            yield element

    @staticmethod
    def _remove_nonnunique_frags(
        natm: int,
        AO_per_frag: list[Sequence[int]],
        center: list[Sequence[int]],
        origin_per_frag: list[Sequence[int]],
        motifs_per_frag: list[Sequence[int]],
        edge_list: list[Sequence],
        fsites_by_atom: list[Sequence[Sequence[int]]],
        add_center_atom: list[Sequence[int]],
    ) -> None:
        """Remove all fragments which are strict subsets of another.
        Remove all fragments whose AO indices can be identified as subsets of
        another fragment's. The center site for the removed frag is
        added to that of the superset. Because doing so will necessarily
        change the definition of fragments, we repeat it up to `natm` times
        such that all fragments are guaranteed to be distinct sets.
        another fragment's. The center site for the removed frag is then
        added to that of the superset. Because doing so will necessarily
        change the definition of fragments, we repeat it up to `natm` times
        such that all fragments are guaranteed to be distinct sets.
        NOTE: The arguments passed to this function are edited
        in-place, meaning `_remove_nonnunique_frags()` is irreversible.
        """
        for _ in range(0, natm):
            subsets = set()
            for adx, basa in enumerate(AO_per_frag):
                for bdx, basb in enumerate(AO_per_frag):
                    if adx == bdx:
                        pass
                    elif set(basb).issubset(set(basa)):
                        if bdx in subsets:
                            pass
                        else:
                            subsets.add(bdx)
                            center[adx] = tuple(
                                set(list(center[adx]) + list(deepcopy(center[bdx])))
                            )
                            origin_per_frag[adx] = tuple(
                                set(
                                    list(origin_per_frag[adx])
                                    + list(deepcopy(origin_per_frag[bdx]))
                                )
                            )
                            add_center_atom[adx] = tuple(
                                set(
                                    list(add_center_atom[adx])
                                    + list(deepcopy(origin_per_frag[bdx]))
                                )
                            )
            if subsets:
                sorted_subsets = sorted(subsets, reverse=True)
                for bdx in sorted_subsets:
                    if len(AO_per_frag) == 1:
                        # If all fragments are identified as subsets,
                        # this stops the loop from deleting the final fragment.
                        break
                    else:
                        # Otherwise, delete the subset fragment.
                        del center[bdx]
                        del AO_per_frag[bdx]
                        del fsites_by_atom[bdx]
                        del origin_per_frag[bdx]
                        del motifs_per_frag[bdx]
                        del edge_list[bdx]
                        del add_center_atom[bdx]
        return None

    @staticmethod
    def export_graph(
        edge_list: list[Sequence],
        adx_map: dict,
        adjacency_graph: nx.Graph,
        origin_per_frag: list[Sequence[int]],
        dnames: list[str],
        outdir: Path | None = None,
        outname: str = "AdjGraph",
        cmap: str = "cubehelix",
        node_position: str = "coordinates",
    ) -> tuple[object, object]:
        """
        Export a visual representation of a fragment-based adjacency graph to a PNG.

        This function draws a directed network graph using a provided adjacency
        structure, with fragments color-coded and displayed using either
        coordinate-based or spring-layout node positioning, and saves the
        resulting image to the specified output directory.

        Parameters
        ----------
        edge_list
            A list of edge groupings, each corresponding to a fragment's set
            of directed edges.
        adx_map
            Mapping from node indices to node metadata
            (must include 'coord' and 'label' keys).
        adjacency_graph
            The NetworkX graph containing node and edge connectivity.
        origin_per_frag
            A list of node index groups, each corresponding to the origin nodes
            of a fragment.
        dnames
            Names for each fragment, used in the legend.
        outdir
            If specified as `Path`: export .png to `outdir`.
            If `None`: no .png is exported, but (fix, ax) are still returned.
        outname
            Filename (without extension) for the output image.
            Default is "AdjGraph".
        cmap
            Matplotlib colormap name used to color-code fragments.
            Default is "cubehelix".
        node_position
            Node positioning strategy: "coordinates" for z-shifted spatial
            coordinates, or "spring" for force-directed layout.
            Default is "coordinates".

        Returns
        -------
        (fix, ax) : tuple[object, object]
            Returns the corresponding `pyplot` objects.

        Notes
        -----
        - Assumes that `adx_map` contains a `"coord"` field for all nodes: `[x, y, z]`
          coordinates when using `"coordinates"` positioning.
        - Colors are assigned per fragment based on the chosen colormap.
        - Nodes are drawn in two layers to produce a bordered effect.
        - Arcs between nodes are radially offset to distinguish overlapping edges.
        """
        z_offset = 0.5
        c_ = plt.cm.get_cmap(cmap)
        c = [c_(fdx / len(edge_list))[0:3] for fdx in range(0, len(edge_list))]
        patches = [mpatches.Patch(color=color, alpha=0.9) for color in c]
        labels = {adx: (map["label"] + str(adx)) for adx, map in adx_map.items()}

        G = adjacency_graph

        if node_position in ["coordinates"]:
            pos = [
                (
                    map["coord"][0] + (map["coord"][2] * z_offset),
                    map["coord"][1] + (map["coord"][2] * z_offset),
                )
                for map in adx_map.values()
            ]
        elif node_position in ["spring"]:
            pos = nx.spring_layout(G, seed=3068)  # type: ignore[assignment]

        fig, ax = plt.subplots()
        arc_rads = np.arange(-0.3, 0.3, 0.6 / len(c), dtype=float)

        for fdx, color in enumerate(c):
            edges = edge_list[fdx]
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=origin_per_frag[fdx],
                node_color=[color for _ in origin_per_frag[fdx]],  # type: ignore[misc]
                edgecolors="tab:gray",
                node_size=850,
                alpha=1.0,
            )
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=origin_per_frag[fdx],
                node_color="whitesmoke",  # type: ignore[arg-type]
                edgecolors=color,
                node_size=700,
                alpha=0.6,
            )
            nx.draw_networkx_edges(
                G,
                pos,
                arrows=True,
                edgelist=edges,
                width=5,
                alpha=0.8,
                edge_color=color,  # type: ignore[arg-type]
                connectionstyle=f"arc3,rad={arc_rads[fdx]}",
            )
        nx.draw_networkx_labels(
            G, pos, labels, font_size=10, font_color="black", alpha=1
        )
        plt.tight_layout()
        plt.legend(patches, dnames, loc="upper left", fontsize=8)
        plt.axis("off")

        if outdir is not None:
            plt.savefig(outdir / f"{outname}.png", dpi=1500)

        return (fig, ax)

    @staticmethod
    def get_subgraphs(
        motifs_per_frag: list[Sequence[int]],
        edge_list: list[Sequence],
        origin_per_frag: list[Sequence[int]],
        adx_map: dict,
        fdx: int | None = None,
        options: dict = {},
    ) -> dict:
        """
        Construct labeled subgraphs for fragments using motif, edge, and label data.

        If a fragment index (`fdx`) is provided, returns a dictionary containing the
        subgraph for that single fragment. Otherwise, returns a dictionary of subgraphs
        for all fragments, keyed by their respective fragment indices (`fdx`).

        Parameters
        ----------
        motifs_per_frag
            A list where each item is a sequence of node indices (motifs) per fragment.
        edge_list
            A list of edge groupings, each corresponding to a fragment's set of edges.
        origin_per_frag
            A list where each item is a sequence of node indices marking origin nodes
            within each fragment.
        adx_map
            A mapping from node index to metadata, where each value must contain a
            `"label"` field for node annotation.
        fdx
            Index of a specific fragment to extract. If None (default), subgraphs for
            all fragments are returned.
        options
            Optional keyword arguments passed to the `nx.Graph` constructor for each
            subgraph.

        Returns
        -------
        subgraph_dict : dict
            a dictionary mapping fragment indices to `networkx.graph` objects. If `fdx`
            is given, the dictionary will contain only one entry for that fragment.

        Notes
        -----
        - Origin nodes are highlighted by enclosing their labels in square brackets.
        - Returned graphs include labeled nodes and edges per fragment.
        - The structure and labels are derived from `adx_map` and `origin_per_frag`.
        """
        labels = {adx: (map["label"] + str(adx)) for adx, map in adx_map.items()}

        if fdx is not None:
            f_labels = []
            subgraph: nx.Graph = nx.Graph(**options)
            nodelist = motifs_per_frag[fdx]
            edgelist = edge_list[fdx]
            for adx in nodelist:
                if adx in origin_per_frag[fdx]:
                    f_labels.append((adx, {"label": f"[{labels[adx]}]"}))
                else:
                    f_labels.append((adx, {"label": labels[adx]}))
            subgraph.add_nodes_from(f_labels)
            subgraph.add_edges_from(edgelist)

            return {fdx: subgraph}
        else:
            subgraph_dict: dict[int, nx.Graph] = {}
            for fdx, edge in enumerate(edge_list):
                f_labels = []
                subgraph_dict[fdx] = nx.Graph(**options)
                nodelist = motifs_per_frag[fdx]
                for adx in nodelist:
                    if adx in origin_per_frag[fdx]:
                        f_labels.append((adx, {"label": f"[{labels[adx]}]"}))
                    else:
                        f_labels.append((adx, {"label": labels[adx]}))
                subgraph_dict[fdx].add_nodes_from(f_labels)
                subgraph_dict[fdx].add_edges_from(edge)

            return subgraph_dict


def graphgen(
    mol: gto.Mole,
    n_BE: str | int = 2,
    frozen_core: bool = True,
    remove_nonunique_frags: bool = True,
    frag_prefix: str = "f",
    connectivity: Literal["euclidean"] = "euclidean",
    iao_valence_basis: str | None = None,
    cutoff: float = 0.0,
    export_graph_to: Path | None = None,
    print_frags: bool = True,
) -> FragPart:
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
    NOTE: Currently does not support periodic calculations or IAOs.

    Parameters
    ----------
    mol :
        The gto.Mole object.
    n_BE :
        The order of nearest neighbors (with respect to the center atom)
        included in a fragment. Supports all 'n' in 'BEn'. Defaults to 2.
    frozen_core:
        Whether to exclude core AO indices from the fragmentation process.
        True by default.
    remove_nonunique_frags:
        Whether to remove fragments which are strict subsets of another
        fragment in the system. Defaults to True.
    frag_prefix:
        Prefix to be appended to the fragment datanames. Useful for managing
        fragment scratch directories. Defaults to "f".
    connectivity:
        Keyword string specifying the distance metric to be used for edge
        weights in the fragment adjacency graph. Currently supports "euclidean"
        (which uses the square of the distance between atoms in real
        space to determine connectivity within a fragment.) Defaults to
        "euclidean".
    cutoff:
        Atoms with an edge weight beyond `cutoff` will be excluded from the
        `shortest_path` calculation. When set to 0.0, `cutoff` will be
        determined dynamically based on the magnitude of `n_BE`. Defaults
        to 0.0.
        NOTE: For very large systems a smaller `cutoff` often significantly
        improves runtime, but can sometimes affect the fragmentation pattern
        if set *too* small.
    export_graph_to:
        If not `None`, specifies the path to which the fragment connectivity
        graph will be saved. Defaults to None.
    print_frags:
        Whether to print simplified string representations of fragment
        connectivity graphs. Defaults to True.

    """
    assert mol is not None
    if iao_valence_basis is not None:
        raise NotImplementedError("IAOs not yet implemented for graphgen.")
    if isinstance(n_BE, str):
        fragment_type_order = int(re.findall(r"\d+", n_BE)[0])
    else:
        fragment_type_order = n_BE
    if cutoff == 0.0 and fragment_type_order <= 3:
        cutoff = 4.5
    elif cutoff == 0.0:
        cutoff = 4.5 * fragment_type_order

    natm: int = mol.natm

    # Used to be `fsites`
    AO_per_frag: list[Sequence[int]] = list(tuple())

    # Similar to `AO_per_frag`: Ao indices per fragment per atom.
    fsites_by_atom: list[Sequence[Sequence[int]]] = list(tuple(tuple()))

    # Used to be `edge_sites`
    AO_per_edge_per_frag: list[Sequence[Sequence[int]]] = list(tuple(tuple()))

    # Center site AO indices per fragment.
    center: list[Sequence[int]] = list(tuple())

    # Used to be `centerf_idx`
    relAO_per_origin_per_frag: list[Sequence[int]] = list(tuple())

    # Used to be `ebe_weight`
    weight_and_relAO_per_center_per_frag: list[Sequence] = list(tuple())
    sites: list[Sequence] = list(tuple())
    dnames: list[str] = list()
    motifs_per_frag: list[Sequence[int]] = list()
    origin_per_frag: list[Sequence[int]] = list()

    # List of edge atom indices per fragment
    edge_atoms: list[Sequence[int]] = list()

    # The full molecular adjacency matrix
    adjacency_mat: np.ndarray = np.zeros((natm, natm), np.float64)

    # The molecular adjacency graph, constructed from `adjacency_mat`
    adjacency_graph: nx.Graph = nx.Graph()

    # List of graph edges to added to `adjacency_graph`
    edge_list: list[Sequence] = list()

    # Center atom indices that have been added to each fragment via
    # `remove_nonnunique_frags`
    add_center_atom: list[Sequence] = list()

    # Dictionary mapping atom index (adx) to relevant atomic info.
    # Each entry to `adx_map` is a node in the adjacency graph.
    adx_map: dict = {
        adx: {
            "bas": bas,
            "label": mol.atom_symbol(adx),
            "coord": mol.atom_coord(adx),
            "shortest_paths": dict(),
            "attached_hydrogens": list(),
        }
        for adx, bas in enumerate(mol.aoslice_by_atom())
    }

    adjacency_graph.add_nodes_from(adx_map)

    _core_offset = 0
    for adx, map in adx_map.items():
        start_ = map["bas"][2]
        stop_ = map["bas"][3]
        if frozen_core:
            # When frozen_core=True, offset all AO indices by the
            # number of core orbitals per atom.
            _, _, core_list = get_core(mol)
            start_ -= _core_offset
            ncore_ = int(core_list[adx])
            stop_ -= _core_offset + ncore_
            _core_offset += ncore_

        sites.append(tuple([i for i in range(start_, stop_)]))

    if connectivity.lower() in ["euclidean"]:
        # Begin by constructing the adjacency matrix and adjacency graph
        # for the system. Each node corresponds to an atom, such that each
        # pair of nodes can be assigned an edge weighted by the square of
        # their euclidean distance.
        for adx in range(natm):
            for bdx in range(adx + 1, natm):
                dr = GraphGenUtility._euclidean_distance(
                    adx_map[adx]["coord"],
                    adx_map[bdx]["coord"],
                )
                adjacency_mat[adx, bdx] = dr**2
                if dr <= cutoff:
                    adjacency_graph.add_edge(adx, bdx, weight=dr**2)
                # For bookkeeping, also keep track of attached Hydrogens:
                if dr <= 2.5 and adx_map[bdx]["label"] == "H":
                    if adx_map[adx]["label"] != "H":
                        adx_map[adx]["attached_hydrogens"].append(bdx)

        # For each center site (adx), find the set of shortest
        # paths to all(*) other sites. The number of nodes visited
        # on that path gives the degree of separation of the
        # sites.
        # (*)-To save runtime, we only compute the shortest paths for
        # sites within some cutoff radius from the center, specified
        # by `cutoff`.)
        for adx, map in adx_map.items():
            origin_per_frag.append((adx,))
            center.append(deepcopy(sites[adx]))
            add_center_atom.append(list())
            fsites_temp = deepcopy(list(sites[adx]))
            fatoms_temp = [adx]
            edges_temp: list[tuple[int, int]] = []
            fs_temp = []
            fs_temp.append(deepcopy(sites[adx]))

            for bdx, _ in adx_map.items():
                if adjacency_graph.has_edge(adx, bdx):
                    map["shortest_paths"].update(
                        {
                            bdx: shortest_path(
                                adjacency_graph,
                                source=adx,
                                target=bdx,
                                weight=lambda a, b, _: (
                                    adjacency_graph[a][b]["weight"]
                                ),
                                method="dijkstra",
                            )
                        }
                    )

            # If the degree of separation is smaller than the *n*
            # in your fragment type, BE*n*, then that site is appended to
            # the set of fragment sites for adx.
            for bdx, path in map["shortest_paths"].items():
                if 0 < (len(path) - 1) < fragment_type_order:
                    fsites_temp = fsites_temp + deepcopy(list(sites[bdx]))
                    fs_temp.append(deepcopy(sites[bdx]))
                    fatoms_temp.append(bdx)
                    edges_temp = edges_temp + list(nx.utils.pairwise(path))

            AO_per_frag.append(tuple(fsites_temp))
            fsites_by_atom.append(tuple(fs_temp))
            edge_list.append(edges_temp)
            motifs_per_frag.append(tuple(fatoms_temp))

    elif connectivity.lower() in ["resistance_distance", "resistance"]:
        raise NotImplementedError("Work in progress...")

    elif connectivity.lower() in ["entanglement"]:
        raise NotImplementedError("Work in progress...")

    else:
        raise AttributeError(f"Connectivity metric not recognized: '{connectivity}'")

    if remove_nonunique_frags:
        # Up to this point, there are as many fragments as there are
        # atoms in the system, and each fragment has just *1* center.
        # Many of these fragments are redundant or non-unique, so it
        # is convention to "absorb" them into nearby larger fragments.
        # The redundant fragment are then deleted.
        GraphGenUtility._remove_nonnunique_frags(
            natm=natm,
            AO_per_frag=AO_per_frag,
            center=center,
            origin_per_frag=origin_per_frag,
            motifs_per_frag=motifs_per_frag,
            edge_list=edge_list,
            fsites_by_atom=fsites_by_atom,
            add_center_atom=add_center_atom,
        )

    # Define the 'edges' for fragment A as the intersect of its sites
    # with the set of all center sites outside of A.
    for adx, fs in enumerate(fsites_by_atom):
        edge_temp: set[tuple] = set()
        eatoms_temp: set[tuple[int, ...]] = set()
        for bdx, c in enumerate(center):
            if adx == bdx:
                pass
            else:
                for f in fs:
                    overlap = set(f).intersection(set(c))
                    if overlap:
                        f_temp = set(motifs_per_frag[adx])
                        c_temp = set(origin_per_frag[bdx])
                        edge_temp.add(tuple(overlap))
                        eatoms_temp.add(tuple(i for i in f_temp.intersection(c_temp)))
        AO_per_edge_per_frag.append(tuple(edge_temp))
        edge_atoms.extend(tuple(eatoms_temp))

    # Update relative center site indices and weights
    # for center site contributions to the energy.
    for adx, c in enumerate(center):
        centerf_idx_entry = tuple(AO_per_frag[adx].index(cdx) for cdx in c)
        relAO_per_origin_per_frag.append(centerf_idx_entry)
        weight_and_relAO_per_center_per_frag.append((1.0, tuple(centerf_idx_entry)))

    # The atomic indices of any hydrogens attached to each atom.
    # Exists for posterity.
    H_per_motif: list[Sequence] = [
        map["attached_hydrogens"] for map in adx_map.values()
    ]

    # Exists for posterity.
    relAO_per_edge_per_frag = [
        [[AO_per_frag[fidx].index(ao_idx) for ao_idx in edge] for edge in frag]
        for fidx, frag in enumerate(AO_per_edge_per_frag)
    ]

    # Exists for posterity.
    ref_frag_idx_per_edge_per_frag = []
    for frag_edges in AO_per_edge_per_frag:
        _flattened_edges = [e for es in frag_edges for e in es]
        ref_idx = []
        for frag_idx, frag_centers in enumerate(center):
            if bool(set(frag_centers) & set(_flattened_edges)):
                ref_idx.append(frag_idx)
        ref_frag_idx_per_edge_per_frag.append(ref_idx)

    # Exists for posterity.
    relAO_in_ref_per_edge_per_frag = [
        [relAO_per_origin_per_frag[frag_idx] for frag_idx in frag]
        for frag in ref_frag_idx_per_edge_per_frag
    ]

    # Finally, set fragment data names for scratch and bookkeeping:
    for adx, _ in enumerate(fsites_by_atom):
        dnames.append(frag_prefix + str(adx))

    # Optionally export a visualization of fragment connectivity
    # graphs. Useful for better understanding the shape and size
    # of the generated fragments.
    if export_graph_to:
        GraphGenUtility.export_graph(
            outdir=export_graph_to,
            edge_list=edge_list,
            adx_map=adx_map,
            adjacency_graph=adjacency_graph,
            origin_per_frag=origin_per_frag,
            dnames=dnames,
            outname=f"AdjGraph_BE{fragment_type_order}",
        )

    # Print an ASCII representation of each fragment connectivity
    # graph. All center sites are [bracketed].
    if print_frags:
        title = "VERBOSE: Fragment Connectivity Graphs"
        print(title, "-" * (80 - len(title)))
        print("(Center sites within a fragment are [bracketed])")
        subgraphs = GraphGenUtility.get_subgraphs(
            motifs_per_frag=motifs_per_frag,
            edge_list=edge_list,
            origin_per_frag=origin_per_frag,
            adx_map=adx_map,
        )
        for fdx, sg in subgraphs.items():
            print(
                f"Frag `{dnames[fdx]}`:",
            )
            for st in GraphGenUtility._graph_to_string(sg):
                print(st, flush=True)

    return FragPart(
        mol=mol,
        n_BE=fragment_type_order,
        frag_type="graphgen",
        AO_per_frag=AO_per_frag,  # type: ignore[arg-type]
        AO_per_edge_per_frag=AO_per_edge_per_frag,  # type: ignore[arg-type]
        ref_frag_idx_per_edge_per_frag=ref_frag_idx_per_edge_per_frag,  # type: ignore[arg-type]
        relAO_per_edge_per_frag=relAO_per_edge_per_frag,  # type: ignore[arg-type]
        relAO_in_ref_per_edge_per_frag=relAO_in_ref_per_edge_per_frag,  # type: ignore[arg-type]
        relAO_per_origin_per_frag=relAO_per_origin_per_frag,  # type: ignore[arg-type]
        weight_and_relAO_per_center_per_frag=weight_and_relAO_per_center_per_frag,  # type: ignore[arg-type]
        motifs_per_frag=motifs_per_frag,  # type: ignore[arg-type]
        origin_per_frag=origin_per_frag,  # type: ignore[arg-type]
        H_per_motif=H_per_motif,  # type: ignore[arg-type]
        add_center_atom=add_center_atom,  # type: ignore[arg-type]
        frozen_core=frozen_core,
        iao_valence_basis=iao_valence_basis,
        iao_valence_only=False,
    )
