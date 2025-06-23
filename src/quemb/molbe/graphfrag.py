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
    cutoff: Final[float] = 20.0
    remove_nonnunique_frags: Final[bool] = True


@define(frozen=False, kw_only=True)
class GraphGenUtility:
    """Utilitity functions for handling graphs in `graphgen()`."""

    @staticmethod
    def euclidean_distance(
        i_coord: Vector,
        j_coord: Vector,
    ) -> np.floating:
        return norm(i_coord - j_coord)

    @staticmethod
    def graph_to_string(
        graph: nx.Graph,
        options: dict = {"with_labels": True},
    ) -> Generator:
        for element in nx.generate_network_text(graph, **options):
            yield element

    @staticmethod
    def remove_nonnunique_frags(
        natm: int,
        fsites: list[Sequence[int]],
        center: list[Sequence[int]],
        center_atom: list[Sequence[int]],
        Frag_atom: list[Sequence[int]],
        edge_list: list[Sequence],
        fsites_by_atom: list[Sequence[Sequence[int]]],
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
        in-place, meaning `remove_nonnunique_frags` is irreversible.
        """
        for _ in range(0, natm):
            subsets = set()
            for adx, basa in enumerate(fsites):
                for bdx, basb in enumerate(fsites):
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
                            center_atom[adx] = tuple(
                                set(
                                    list(center_atom[adx])
                                    + list(deepcopy(center_atom[bdx]))
                                )
                            )
            if subsets:
                sorted_subsets = sorted(subsets, reverse=True)
                for bdx in sorted_subsets:
                    if len(fsites) == 1:
                        # If all fragments are identified as subsets,
                        # this stops the loop from deleting the final fragment.
                        break
                    else:
                        # Otherwise, delete the subset fragment.
                        del center[bdx]
                        del fsites[bdx]
                        del fsites_by_atom[bdx]
                        del center_atom[bdx]
                        del Frag_atom[bdx]
                        del edge_list[bdx]
        return None

    @staticmethod
    def export_graph(
        outdir: Path,
        edge_list: list[Sequence],
        adx_map: dict,
        adjacency_graph: nx.Graph,
        center_atom: list[Sequence[int]],
        dnames: list[str],
        outname: str = "AdjGraph",
        cmap: str = "cubehelix",
        node_position: str = "coordinates",
    ) -> None:
        outdir = Path(outdir)
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
            pos = nx.spring_layout(G, seed=3068)

        __, _ = plt.subplots()
        arc_rads = np.arange(-0.3, 0.3, 0.6 / len(c), dtype=float)

        for fdx, color in enumerate(c):
            edges = edge_list[fdx]
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=center_atom[fdx],
                node_color=[color for _ in center_atom[fdx]],  # type: ignore[arg-type]
                edgecolors="tab:gray",
                node_size=850,
                alpha=1.0,
            )
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=center_atom[fdx],
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
        plt.savefig(outdir / f"{outname}.png", dpi=1500)

    @staticmethod
    def get_subgraphs(
        Frag_atom: list[Sequence[int]],
        edge_list: list[Sequence],
        center_atom: list[Sequence[int]],
        adx_map: dict,
        fdx: int | None = None,
        options: dict = {},
    ) -> dict:
        """
        Return the subgraph for a fragment indexed by `fdx`.
        If `fdx=None`, returns subgraphs for all fragments as a dictionary keyed
        by each fragment index (`fdx`).
        """
        labels = {adx: (map["label"] + str(adx)) for adx, map in adx_map.items()}

        if fdx is not None:
            f_labels = []
            subgraph: nx.Graph = nx.Graph(**options)
            nodelist = Frag_atom[fdx]
            edgelist = edge_list[fdx]
            for adx in nodelist:
                if adx in center_atom[fdx]:
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
                nodelist = Frag_atom[fdx]
                for adx in nodelist:
                    if adx in center_atom[fdx]:
                        f_labels.append((adx, {"label": f"[{labels[adx]}]"}))
                    else:
                        f_labels.append((adx, {"label": labels[adx]}))
                subgraph_dict[fdx].add_nodes_from(f_labels)
                subgraph_dict[fdx].add_edges_from(edge)

            return subgraph_dict


def graphgen(
    mol: gto.Mole,
    n_BE: str | int = "BE2",
    frozen_core: bool = True,
    remove_nonunique_frags: bool = True,
    frag_prefix: str = "f",
    connectivity: str = "euclidean",
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
        The molecule object.
    n_BE :
        The order of nearest neighbors (with respect to the center atom)
        included in a fragment. Supports all 'BEn', with 'n' in -
        [1, 2, 3, 4, 5, 6, 7, 8, 9] having been tested.
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
    dict :
        Keyword dictionary containing all fragmentation results. These are
        later used to directly instantiate the `FragPart` object.
        More specifically:
            fsites :
            List whose entries are sequences (tuple or list) containing
            all AO indices for a fragment.
            fsites_by_atom :
            List whose entries are sequences of sequences, containing AO indices per
            atom per fragment.
            edge_sites :
            List whose entries are sequences of sequences, containing edge AO
            indices per atom (inner tuple) per fragment (outer tuple).
            center :
            List whose entries are sequences of sequences, containing all fragment
            AO indices per atom (inner tuple) and per fragment (outer tuple).
            centerf_idx :
            List whose entries are sequences containing the relative AO index of the
            origin site within a fragment.
            Relative is to the own fragment; since the origin site is at the
            beginning of the motif list for each fragment, this is always a
            Sequence :python:`range(0, n)`.
            ebe_weight :
            Weights determining the energy contributions from each center site
            (ie, with respect to centerf_idx).
            sites :
            List whose entries are sequences containing all AO indices per atom
            (excluding frozen core indices, if applicable).
            dnames :
            List of strings giving fragment data names. Useful for bookkeeping and
            for constructing fragment scratch directories.
            Frag_atom :
            List whose entries are sequences containing all atom indices for a
            fragment.
            center_atom :
            List whose entries are sequences giving the center atom indices per
            fragment.
            edge_atoms :
            List whose entries are sequences giving the edge atom indices per
            fragment.
            adjacency_mat :
            The adjacency matrix for all sites (atoms) in the system.
            adjacency_graph :
            The adjacency graph corresponding to `adjacency_mat`.
            edge_list:
            Sequences of edge pairs per fragment (these correspond to edges in
            `adjacency_graph`).
    """
    assert mol is not None
    if iao_valence_basis is not None:
        raise NotImplementedError("IAOs not yet implemented for graphgen.")
    if isinstance(n_BE, str):
        fragment_type_order = int(re.findall(r"\d+", str(n_BE))[0])
    else:
        fragment_type_order = n_BE
    if cutoff == 0.0 and fragment_type_order <= 3:
        cutoff = 4.5
    elif cutoff == 0.0:
        cutoff = 4.5 * fragment_type_order

    natm: int = mol.natm
    fsites: list[Sequence[int]] = list(tuple())
    fsites_by_atom: list[Sequence[Sequence[int]]] = list(tuple(tuple()))
    edge_sites: list[Sequence[Sequence[int]]] = list(tuple(tuple()))
    center: list[Sequence[int]] = list(tuple())
    centerf_idx: list[Sequence[int]] = list(tuple())
    ebe_weight: list[Sequence] = list(tuple())
    sites: list[Sequence] = list(tuple())
    dnames: list[str] = list()
    Frag_atom: list[Sequence[int]] = list()
    center_atom: list[Sequence[int]] = list()
    edge_atoms: list[Sequence[int]] = list()
    adjacency_mat: np.ndarray = np.zeros((natm, natm), np.float64)
    adjacency_graph: nx.Graph = nx.Graph()
    edge_list: list[Sequence] = list()
    adx_map: dict = {
        adx: {
            "bas": bas,
            "label": mol.atom_symbol(adx),
            "coord": mol.atom_coord(adx),
            "shortest_paths": dict(),
        }
        for adx, bas in enumerate(mol.aoslice_by_atom())
    }

    adjacency_graph.add_nodes_from(adx_map)

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
            sites.append(tuple([i for i in range(start_, stop_)]))
        else:
            sites.append(tuple([i for i in range(start_, stop_)]))

    if connectivity.lower() in ["euclidean_distance", "euclidean"]:
        # Begin by constructing the adjacency matrix and adjacency graph
        # for the system. Each node corresponds to an atom, such that each
        # pair of nodes can be assigned an edge weighted by the square of
        # their distance in real space.
        for adx in range(natm):
            for bdx in range(adx + 1, natm):
                dr = GraphGenUtility.euclidean_distance(
                    adx_map[adx]["coord"],
                    adx_map[bdx]["coord"],
                )
                adjacency_mat[adx, bdx] = dr**2
                if dr <= cutoff:
                    adjacency_graph.add_edge(adx, bdx, weight=dr**2)

        # For a given center site (adx), find the set of shortest
        # paths to all other sites. The number of nodes visited
        # on that path gives the degree of separation of the
        # sites.
        for adx, map in adx_map.items():
            center_atom.append((adx,))
            center.append(deepcopy(sites[adx]))
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

            fsites.append(tuple(fsites_temp))
            fsites_by_atom.append(tuple(fs_temp))
            edge_list.append(edges_temp)
            Frag_atom.append(tuple(fatoms_temp))

    elif connectivity.lower() in ["resistance_distance", "resistance"]:
        raise NotImplementedError("Work in progress...")

    elif connectivity.lower() in ["entanglement"]:
        raise NotImplementedError("Work in progress...")

    else:
        raise AttributeError(f"Connectivity metric not recognized: '{connectivity}'")

    if remove_nonunique_frags:
        GraphGenUtility.remove_nonnunique_frags(
            natm=natm,
            fsites=fsites,
            center=center,
            center_atom=center_atom,
            Frag_atom=Frag_atom,
            edge_list=edge_list,
            fsites_by_atom=fsites_by_atom,
        )

    # Define the 'edges' for fragment A as the intersect of its sites
    # with the set of all center sites outside of A:
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
                        f_temp = set(Frag_atom[adx])
                        c_temp = set(center_atom[bdx])
                        edge_temp.add(tuple(overlap))
                        eatoms_temp.add(tuple(i for i in f_temp.intersection(c_temp)))
        edge_sites.append(tuple(edge_temp))
        edge_atoms.extend(tuple(eatoms_temp))

    # Update relative center site indices (centerf_idx) and weights
    # for center site contributions to the energy (ebe_weights):
    for adx, c in enumerate(center):
        centerf_idx_entry = tuple(fsites[adx].index(cdx) for cdx in c)
        centerf_idx.append(centerf_idx_entry)
        ebe_weight.append((1.0, tuple(centerf_idx_entry)))

    # Finally, set fragment data names for scratch and bookkeeping:
    for adx, _ in enumerate(fsites_by_atom):
        dnames.append(frag_prefix + str(adx))

    if export_graph_to:
        GraphGenUtility.export_graph(
            outdir=export_graph_to,
            edge_list=edge_list,
            adx_map=adx_map,
            adjacency_graph=adjacency_graph,
            center_atom=center_atom,
            dnames=dnames,
            outname=f"AdjGraph_BE{fragment_type_order}",
        )

    if print_frags:
        title = "VERBOSE: Fragment Connectivity Graphs"
        print(title, "-" * (80 - len(title)))
        print("(Center sites within a fragment are [bracketed])")
        subgraphs = GraphGenUtility.get_subgraphs(
            Frag_atom=Frag_atom,
            edge_list=edge_list,
            center_atom=center_atom,
            adx_map=adx_map,
        )
        for fdx, sg in subgraphs.items():
            print(
                f"Frag `{dnames[fdx]}`:",
            )
            for st in GraphGenUtility.graph_to_string(sg):
                print(st, flush=True)
    """
    return {
        "mol": mol,
        "frag_type": "graphgen",
        "n_BE": fragment_type_order,
        "fsites": fsites,
        "edge_sites": edge_sites,
        "center": center,
        "edge_idx": MISSING,
        "center_idx": MISSING,
        "centerf_idx": centerf_idx,
        "ebe_weight": ebe_weight,
        "Frag_atom": Frag_atom,
        "center_atom": center_atom,
        "hlist_atom": MISSING,
        "add_center_atom": MISSING,
        "frozen_core": frozen_core,
        "iao_valence_basis": iao_valence_basis,
        "iao_valence_only": False,
    }
    """
    MISSING = []  # type: ignore[var-annotated]
    MISSING_PER_FRAG = [[] for _ in range(len(fsites))]  # type: ignore[var-annotated]
    return FragPart(
        mol=mol,
        n_BE=fragment_type_order,
        frag_type="graphgen",
        AO_per_frag=fsites,  # type: ignore[arg-type]
        AO_per_edge_per_frag=edge_sites,  # type: ignore[arg-type]
        ref_frag_idx_per_edge_per_frag=center,  # type: ignore[arg-type]
        relAO_per_edge_per_frag=MISSING_PER_FRAG,  # type: ignore[arg-type]
        relAO_in_ref_per_edge_per_frag=MISSING_PER_FRAG,
        relAO_per_origin_per_frag=centerf_idx,  # type: ignore[arg-type]
        weight_and_relAO_per_center_per_frag=ebe_weight,  # type: ignore[arg-type]
        motifs_per_frag=Frag_atom,  # type: ignore[arg-type]
        origin_per_frag=center_atom,  # type: ignore[arg-type]
        H_per_motif=MISSING,
        add_center_atom=MISSING,
        frozen_core=frozen_core,
        iao_valence_basis=iao_valence_basis,
        iao_valence_only=False,
    )
