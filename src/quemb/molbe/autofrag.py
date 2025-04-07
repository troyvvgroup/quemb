# Author: Oinam Romesh Meitei, Shaun Weatherly

import re
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Final, Generator, Literal, TypeAlias
from warnings import warn

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from attrs import cmp_using, define, field
from networkx import shortest_path
from numpy.linalg import norm
from pyscf import gto
from pyscf.gto import Mole

from quemb.molbe.helper import are_equal, get_core
from quemb.shared.helper import unused
from quemb.shared.typing import (
    AtomIdx,
    CenterIdx,
    FragmentIdx,
    GlobalAOIdx,
    MotifIdx,
    OriginIdx,
    OtherRelAOIdx,
    OwnRelAOIdx,
    Vector,
)

FragType: TypeAlias = Literal["chemgen", "graphgen", "autogen"]

ListOverFrag: TypeAlias = list
ListOverEdge: TypeAlias = list
ListOverMotif: TypeAlias = list


def euclidean_distance(
    i_coord: Vector,
    j_coord: Vector,
) -> np.floating:
    return norm(i_coord - j_coord)


def graph_to_string(
    graph: nx.Graph,
    options: dict = {"with_labels": True},
) -> Generator:
    for element in nx.generate_network_text(graph, **options):
        yield element


@define
class FragPart:
    """Data structure to hold the result of BE fragmentations."""

    #: The full molecule.
    mol: Mole = field(eq=cmp_using(are_equal))
    #: The algorithm used for fragmenting.
    frag_type: FragType
    #: The level of BE fragmentation, i.e. "be1", "be2", ...
    be_type: str

    #: This is a list over fragments  and gives the global orbital indices of all atoms
    #: in the fragment. These are ordered by the atoms in the fragment.
    fsites: ListOverFrag[list[GlobalAOIdx]]

    #: Contains the same information as `fsites`, except AO indices are 
    #: further organized by motif (atom) within each fragment. 
    fsites_by_atom: ListOverFrag[ListOverMotif[list[GlobalAOIdx]]]

    #: The global orbital indices, including hydrogens, per edge per fragment.
    edge_sites: ListOverFrag[ListOverEdge[list[GlobalAOIdx]]]

    # A list over fragments: list of indices of the fragments in which an edge
    # of the fragment is actually a center:
    # For fragments A, B: the A’th element of :python:`.center`,
    # if the edge of A is the center of B, will be B.
    center: ListOverFrag[ListOverEdge[FragmentIdx]]

    #: The relative orbital indices, including hydrogens, per edge per fragment.
    #: The index is relative to the own fragment.
    edge_idx: ListOverFrag[ListOverEdge[list[OwnRelAOIdx]]]

    #: The relative atomic orbital indices per edge per fragment.
    #: **Note** for this variable relative means that the AO indices
    #: are relative to the other fragment where the edge is a center.
    center_idx: ListOverFrag[ListOverEdge[list[OtherRelAOIdx]]]

    #: List whose entries are lists containing the relative orbital index of the
    #: origin site within a fragment. Relative is to the own fragment.
    #  Since the origin site is at the beginning
    #: of the motif list for each fragment, this is always a ``list(range(0, n))``
    centerf_idx: ListOverFrag[list[OwnRelAOIdx]]

    #: The first element is a float, the second is the list
    #: The float weight makes only sense for democratic matching and is currently 1.0
    #: everywhere anyway. We concentrate only on the second part,
    #: i.e. the list of indices.
    #: This is a list whose entries are sequences containing the relative orbital index
    #  of the center sites within a fragment. Relative is to the own fragment.
    ebe_weight: ListOverFrag[list[float | list[OwnRelAOIdx]]]

    #: The heavy atoms in each fragment, in order.
    #: Each are labeled based on the global atom index.
    #: It is ordered by origin, centers, edges!
    Frag_atom: ListOverFrag[ListOverMotif[MotifIdx]]

    #: The origin for each fragment.
    #: (Note that for conventional BE there is just one origin per fragment)
    center_atom: ListOverFrag[OriginIdx]

    #: A list over atoms (not over motifs!)
    #: For each atom it contains a list of the attached hydrogens.
    #: This means that there are a lot of empty sets for molecular systems,
    # because hydrogens have no attached hydrogens (usually).
    hlist_atom: Sequence[list[AtomIdx]]

    #: A list over fragments.
    #: For each fragment a list of centers that are not the origin of that fragment.
    add_center_atom: ListOverFrag[list[CenterIdx]]

    frozen_core: bool
    iao_valence_basis: str | None

    #: If this option is set to True, all calculation will be performed in
    #: the valence basis in the IAO partitioning.
    #: This is an experimental feature.
    iao_valence_only: bool

    Nfrag: int = field()
    ncore: int | None = field()
    no_core_idx: list[int] | None = field()
    core_list: list[int] | None = field()

    @Nfrag.default
    def _get_default_Nfrag(self) -> int:
        return len(self.fsites)

    @ncore.default
    def _get_default_ncore(self) -> int | None:
        return get_core(self.mol)[0] if self.frozen_core else None

    @no_core_idx.default
    def _get_default_no_core_idx(self) -> list[int] | None:
        return get_core(self.mol)[1] if self.frozen_core else None

    @core_list.default
    def _get_default_core_list(self) -> list[int] | None:
        return get_core(self.mol)[2] if self.frozen_core else None

    def __len__(self) -> int:
        return self.Nfrag


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


@define
class FragmentMap:
    """Dataclass for fragment bookkeeping.

    Parameters
    ----------
    fsites :
        List whose entries are sequences (tuple or list) containing
        all AO indices for a fragment.
    fsites_by_atom :
        List whose entries are sequences of sequences, containing AO indices per atom
        per fragment.
    edge_sites :
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
        List whose entries are sequences containing all atom indices for a fragment.
    center_atom :
        List whose entries are sequences giving the center atom indices per fragment.
    edge_atoms :
        List whose entries are sequences giving the edge atom indices per fragment.
    adjacency_mat :
        The adjacency matrix for all sites (atoms) in the system.
    adjacency_graph :
        The adjacency graph corresponding to `adjacency_mat`.
    edge_list:
        Sequences of edge pairs per fragment (these correspond to edges in
        `adjacency_graph`).
    """

    fsites: list[Sequence[int]]
    fsites_by_atom: list[Sequence[Sequence[int]]]
    edge_sites: list[Sequence[Sequence[int]]]
    center: list[Sequence[int]]
    centerf_idx: list[Sequence[int]]
    ebe_weight: list[Sequence]
    sites: list[Sequence]
    dnames: list[str]
    Frag_atom: list[Sequence[int]]
    center_atom: list[Sequence[int]]
    edge_atoms: list[Sequence[int]]
    adjacency_mat: np.ndarray
    adjacency_graph: nx.Graph
    edge_list: list[Sequence]
    adx_map: dict

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
            for adx, basa in enumerate(self.fsites):
                for bdx, basb in enumerate(self.fsites):
                    if adx == bdx:
                        pass
                    elif set(basb).issubset(set(basa)):
                        subsets.add(bdx)
                        self.center[adx] = tuple(
                            set(
                                list(self.center[adx])
                                + list(deepcopy(self.center[bdx]))
                            )
                        )
                        self.center_atom[adx] = tuple(
                            set(
                                list(self.center_atom[adx])
                                + list(deepcopy(self.center_atom[bdx]))
                            )
                        )
            if subsets:
                sorted_subsets = sorted(subsets, reverse=True)
                for bdx in sorted_subsets:
                    if len(self.fsites) == 1:
                        # If all fragments are identified as subsets,
                        # this stops the loop from deleting the final fragment.
                        break
                    else:
                        # Otherwise, delete the subset fragment.
                        del self.center[bdx]
                        del self.fsites[bdx]
                        del self.fsites_by_atom[bdx]
                        del self.center_atom[bdx]
                        del self.Frag_atom[bdx]
                        del self.edge_list[bdx]
        return None

    def export_graph(
        self,
        outdir: Path,
        outname: str = "AdjGraph",
        cmap: str = "cubehelix",
        node_position: str = "coordinates",
    ) -> None:
        outdir = Path(outdir)
        z_offset = 0.5
        c_ = plt.cm.get_cmap(cmap)
        c = [
            c_(fdx / len(self.edge_list))[0:3] for fdx in range(0, len(self.edge_list))
        ]
        patches = [mpatches.Patch(color=color, alpha=0.9) for color in c]
        labels = {adx: (map["label"] + str(adx)) for adx, map in self.adx_map.items()}

        G = self.adjacency_graph

        if node_position in ["coordinates"]:
            pos = [
                (
                    map["coord"][0] + (map["coord"][2] * z_offset),
                    map["coord"][1] + (map["coord"][2] * z_offset),
                )
                for map in self.adx_map.values()
            ]
        elif node_position in ["spring"]:
            pos = nx.spring_layout(G, seed=3068)

        __, _ = plt.subplots()
        arc_rads = np.arange(-0.3, 0.3, 0.6 / len(c), dtype=float)

        for fdx, color in enumerate(c):
            edges = self.edge_list[fdx]
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=self.center_atom[fdx],
                node_color=[color for _ in self.center_atom[fdx]],  # type: ignore[arg-type]
                edgecolors="tab:gray",
                node_size=850,
                alpha=1.0,
            )
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=self.center_atom[fdx],
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
        plt.legend(patches, self.dnames, loc="upper left", fontsize=8)
        plt.axis("off")
        plt.savefig(outdir / f"{outname}.png", dpi=1500)

    def to_FragPart(self, mol: Mole, be_type: str, frozen_core: bool) -> FragPart:
        MISSING = []  # type: ignore[var-annotated]
        return FragPart(
            mol=mol,
            frag_type="graphgen",
            be_type=be_type,
            edge_sites=self.edge_sites,  # type: ignore[arg-type]
            edge_idx=MISSING,
            center_idx=MISSING,
            centerf_idx=self.centerf_idx,  # type: ignore[arg-type]
            fsites=self.fsites,  # type: ignore[arg-type]
            fsites_by_atom=self.fsites_by_atom,  # type: ignore[arg-type]
            center=self.center,  # type: ignore[arg-type]
            ebe_weight=self.ebe_weight,  # type: ignore[arg-type]
            Frag_atom=self.Frag_atom,  # type: ignore[arg-type]
            center_atom=self.center_atom,  # type: ignore[arg-type]
            hlist_atom=MISSING,
            add_center_atom=MISSING,
            frozen_core=frozen_core,
            iao_valence_basis=None,
            iao_valence_only=False,
        )

    def get_subgraphs(
        self,
        fdx: int | None = None,
        options: dict = {},
    ) -> dict:
        """
        Return the subgraph for a fragment indexed by `fdx`.

        If `fdx=None`, returns subgraphs for all fragments as a dictionary keyed
        by each fragment index (`fdx`).
        """
        labels = {adx: (map["label"] + str(adx)) for adx, map in self.adx_map.items()}

        if fdx is not None:
            f_labels = []
            subgraph: nx.Graph = nx.Graph(**options)
            nodelist = self.Frag_atom[fdx]
            edgelist = self.edge_list[fdx]
            for adx in nodelist:
                if adx in self.center_atom[fdx]:
                    f_labels.append((adx, {"label": f"[{labels[adx]}]"}))
                else:
                    f_labels.append((adx, {"label": labels[adx]}))
            subgraph.add_nodes_from(f_labels)
            subgraph.add_edges_from(edgelist)

            return {fdx: subgraph}
        else:
            subgraph_dict: dict[int, nx.Graph] = {}
            for fdx, edge in enumerate(self.edge_list):
                f_labels = []
                subgraph_dict[fdx] = nx.Graph(**options)
                nodelist = self.Frag_atom[fdx]
                for adx in nodelist:
                    if adx in self.center_atom[fdx]:
                        f_labels.append((adx, {"label": f"[{labels[adx]}]"}))
                    else:
                        f_labels.append((adx, {"label": labels[adx]}))
                subgraph_dict[fdx].add_nodes_from(f_labels)
                subgraph_dict[fdx].add_edges_from(edge)

            return subgraph_dict


def graphgen(
    mol: gto.Mole,
    be_type: str = "BE2",
    frozen_core: bool = True,
    remove_nonunique_frags: bool = True,
    frag_prefix: str = "f",
    connectivity: str = "euclidean",
    iao_valence_basis: str | None = None,
    cutoff: float = 0.0,
    export_graph_to: Path | None = None,
    print_frags: bool = True,
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
    be_type :
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
    FragmentMap :
        FragmentMap mapping various fragment components to AO indices, data names,
        and other info.
    """
    assert mol is not None
    if iao_valence_basis is not None:
        raise NotImplementedError("IAOs not yet implemented for graphgen.")

    fragment_type_order = int(re.findall(r"\d+", str(be_type))[0])
    if cutoff == 0.0 and fragment_type_order <= 3:
        cutoff = 4.5
    elif cutoff == 0.0:
        cutoff = 4.5 * fragment_type_order

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
        fsites=(list(tuple())),
        fsites_by_atom=list(tuple(tuple())),
        edge_sites=list(tuple(tuple())),
        center=list(tuple()),
        centerf_idx=list(tuple()),
        ebe_weight=list(tuple()),
        sites=list(tuple()),
        dnames=list(),
        Frag_atom=list(),
        center_atom=list(),
        edge_atoms=list(),
        adjacency_mat=np.zeros((natm, natm), np.float64),
        adjacency_graph=nx.Graph(),
        edge_list=list(),
        adx_map=adx_map,
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
                dr = euclidean_distance(
                    adx_map[adx]["coord"],
                    adx_map[bdx]["coord"],
                )
                fragment_map.adjacency_mat[adx, bdx] = dr**2
                if dr <= cutoff:
                    fragment_map.adjacency_graph.add_edge(adx, bdx, weight=dr**2)

        # For a given center site (adx), find the set of shortest
        # paths to all other sites. The number of nodes visited
        # on that path gives the degree of separation of the
        # sites.
        for adx, map in adx_map.items():
            fragment_map.center_atom.append((adx,))
            fragment_map.center.append(deepcopy(fragment_map.sites[adx]))
            fsites_temp = deepcopy(list(fragment_map.sites[adx]))
            fatoms_temp = [adx]
            edges_temp: list[tuple[int, int]] = []
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
                if 0 < (len(path) - 1) < fragment_type_order:
                    fsites_temp = fsites_temp + deepcopy(list(fragment_map.sites[bdx]))
                    fs_temp.append(deepcopy(fragment_map.sites[bdx]))
                    fatoms_temp.append(bdx)
                    edges_temp = edges_temp + list(nx.utils.pairwise(path))

            fragment_map.fsites.append(tuple(fsites_temp))
            fragment_map.fsites_by_atom.append(tuple(fs_temp))
            fragment_map.edge_list.append(edges_temp)
            fragment_map.Frag_atom.append(tuple(fatoms_temp))

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
    for adx, fs in enumerate(fragment_map.fsites_by_atom):
        edge_temp: set[tuple] = set()
        eatoms_temp: set[tuple[int, ...]] = set()
        for bdx, center in enumerate(fragment_map.center):
            if adx == bdx:
                pass
            else:
                for f in fs:
                    overlap = set(f).intersection(set(center))
                    if overlap:
                        f_temp = set(fragment_map.Frag_atom[adx])
                        c_temp = set(fragment_map.center_atom[bdx])
                        edge_temp.add(tuple(overlap))
                        eatoms_temp.add(tuple(i for i in f_temp.intersection(c_temp)))
        fragment_map.edge_sites.append(tuple(edge_temp))
        fragment_map.edge_atoms.extend(tuple(eatoms_temp))

    # Update relative center site indices (centerf_idx) and weights
    # for center site contributions to the energy (ebe_weights):
    for adx, center in enumerate(fragment_map.center):
        centerf_idx = tuple(fragment_map.fsites[adx].index(cdx) for cdx in center)
        fragment_map.centerf_idx.append(centerf_idx)
        fragment_map.ebe_weight.append((1.0, tuple(centerf_idx)))

    # Finally, set fragment data names for scratch and bookkeeping:
    for adx, _ in enumerate(fragment_map.fsites_by_atom):
        fragment_map.dnames.append(frag_prefix + str(adx))

    if export_graph_to:
        fragment_map.export_graph(
            outdir=export_graph_to,
            outname=f"AdjGraph_{be_type}",
        )

    if print_frags:
        title = "VERBOSE: Fragment Connectivity Graphs"
        print(title, "-" * (80 - len(title)))
        print("(Center sites within a fragment are [bracketed])")
        subgraphs = fragment_map.get_subgraphs()
        for fdx, sg in subgraphs.items():
            print(
                f"Frag `{fragment_map.dnames[fdx]}`:",
            )
            for st in graph_to_string(sg):
                print(st, flush=True)

    return fragment_map


@define
class AutogenArgs:
    """Additional arguments for autogen

    Parameters
    ----------
    iao_valence_only:
        If this option is set to True, all calculation will be performed in
        the valence basis in the IAO partitioning.
        This is an experimental feature.
    """

    iao_valence_only: bool = False


def autogen(
    mol,
    frozen_core=True,
    be_type="be2",
    write_geom=False,
    iao_valence_basis=None,
    iao_valence_only=False,
    print_frags=True,
):
    """Automatic molecular partitioning

    Partitions a molecule into overlapping fragments as defined in BE atom-based
    fragmentations.  It automatically detects branched chemical chains and ring systems
    and partitions accordingly.  For efficiency, it only checks two atoms
    for connectivity (chemical bond) if they are within 3.5 Angstrom.
    This value is hardcoded as normdist. Two atoms are defined as bonded
    if they are within 1.8 Angstrom (1.2 for Hydrogen atom).
    This is also hardcoded as bond & hbond.

    Parameters
    ----------
    mol : pyscf.gto.mole.Mole
        This is required for the options, 'autogen',
        and 'chain' as frag_type.
    frozen_core : bool, optional
        Whether to invoke frozen core approximation. Defaults to True.
    be_type : str, optional
        Specifies the order of bootstrap calculation in the atom-based fragmentation.
        Supported values are 'be1', 'be2', 'be3', and 'be4'.
        Defaults to 'be2'.
    write_geom : bool, optional
        Whether to write a 'fragment.xyz' file which contains all the fragments in
        Cartesian coordinates. Defaults to False.
    iao_valence_basis : str, optional
        Name of minimal basis set for IAO scheme. 'sto-3g' is sufficient for most cases.
        Defaults to None.
    iao_valence_only : bool, optional
        If True, all calculations will be performed in the valence basis in
        the IAO partitioning. This is an experimental feature. Defaults to False.
    print_frags : bool, optional
        Whether to print out the list of resulting fragments. Defaults to True.
    """

    if iao_valence_basis is not None:
        warn(
            'IAO indexing is not working correctly for "autogen". '
            'It is recommended to use "chemgen" instead.'
        )

    cell = mol.copy()
    if iao_valence_only:
        cell.basis = iao_valence_basis
        cell.build()

    ncore, no_core_idx, core_list = get_core(cell)
    unused(ncore, no_core_idx)
    coord = cell.atom_coords()
    ang2bohr = 1.88973
    normdist = 3.5 * ang2bohr
    bond = 1.8 * ang2bohr
    hbond = 1.2 * ang2bohr

    # Compute the norm (magnitude) of each atomic coordinate
    normlist = []
    for i in coord:
        normlist.append(norm(i))
    Frag_atom = []
    pedge = []
    center_atom = []

    # Check if the molecule is a hydrogen chain
    hchain = True
    for i in range(cell.natm):
        if not cell.atom_pure_symbol(i) == "H":
            hchain = False
            break

    open_frag = []
    open_frag_cen = []

    # Assumes that there can be only 5 member connected system
    for idx, i in enumerate(normlist):
        if cell.atom_pure_symbol(idx) == "H" and not hchain:
            continue

        tmplist = normlist - i
        tmplist = list(tmplist)

        clist = []
        for jdx, j in enumerate(tmplist):
            if not idx == jdx and (not cell.atom_pure_symbol(jdx) == "H" or hchain):
                if abs(j) < normdist:
                    clist.append(jdx)
        pedg = []
        flist = []
        flist.append(idx)

        if not be_type == "be1":
            for jdx in clist:
                dist = norm(coord[idx] - coord[jdx])
                if dist <= bond:
                    flist.append(jdx)
                    pedg.append(jdx)
                    if be_type == "be3" or be_type == "be4":
                        for kdx in clist:
                            if not kdx == jdx:
                                dist = norm(coord[jdx] - coord[kdx])
                                if dist <= bond:
                                    if kdx not in pedg:
                                        flist.append(kdx)
                                        pedg.append(kdx)
                                    if be_type == "be4":
                                        for ldx, l in enumerate(coord):
                                            if (
                                                ldx == kdx
                                                or ldx == jdx
                                                or (
                                                    cell.atom_pure_symbol(ldx) == "H"
                                                    and not hchain
                                                )
                                                or ldx in pedg
                                            ):
                                                continue
                                            dist = norm(coord[kdx] - l)
                                            if dist <= bond:
                                                flist.append(ldx)
                                                pedg.append(ldx)

            # Update fragment and edge lists based on current partitioning
            for pidx, frag_ in enumerate(Frag_atom):
                if set(flist).issubset(frag_):
                    open_frag.append(pidx)
                    open_frag_cen.append(idx)
                    break
                elif set(frag_).issubset(flist):
                    open_frag = [
                        oidx - 1 if oidx > pidx else oidx for oidx in open_frag
                    ]
                    open_frag.append(len(Frag_atom) - 1)
                    open_frag_cen.append(center_atom[pidx])
                    del center_atom[pidx]
                    del Frag_atom[pidx]
                    del pedge[pidx]
            else:
                Frag_atom.append(flist)
                pedge.append(pedg)
                center_atom.append(idx)
        else:
            Frag_atom.append(flist)
            center_atom.append(idx)

    hlist_atom = [[] for i in coord]
    if not hchain:
        for idx, i in enumerate(normlist):
            if cell.atom_pure_symbol(idx) == "H":
                tmplist = normlist - i
                tmplist = list(tmplist)
                clist = []
                for jdx, j in enumerate(tmplist):
                    if not idx == jdx and not cell.atom_pure_symbol(jdx) == "H":
                        if abs(j) < normdist:
                            clist.append(jdx)
                for jdx in clist:
                    dist = norm(coord[idx] - coord[jdx])
                    if dist <= hbond:
                        hlist_atom[jdx].append(idx)

    # Print fragments if requested
    if print_frags:
        print(flush=True)
        print("Fragment sites", flush=True)
        print("--------------------------", flush=True)
        print("Fragment |   Origin | Atoms ", flush=True)
        print("--------------------------", flush=True)

        for idx, i in enumerate(Frag_atom):
            print(
                "   {:>4}  |   {:>5}  |".format(
                    idx,
                    cell.atom_pure_symbol(center_atom[idx]) + str(center_atom[idx] + 1),
                ),
                end=" ",
                flush=True,
            )
            for j in hlist_atom[center_atom[idx]]:
                print(
                    " {:>5} ".format("*" + cell.atom_pure_symbol(j) + str(j + 1)),
                    end=" ",
                    flush=True,
                )
            for j in i:
                if j == center_atom[idx]:
                    continue
                print(
                    f" {cell.atom_pure_symbol(j) + str(j + 1):>5} ",
                    end=" ",
                    flush=True,
                )
                for k in hlist_atom[j]:
                    print(
                        f" {cell.atom_pure_symbol(k) + str(k + 1):>5} ",
                        end=" ",
                        flush=True,
                    )
            print(flush=True)
        print("--------------------------", flush=True)
        print(" No. of fragments : ", len(Frag_atom), flush=True)
        print("*H : Center H atoms (printed as Edges above.)", flush=True)
        print(flush=True)

    # Write fragment geometry to a file if requested
    if write_geom:
        w = open("fragments.xyz", "w")
        for idx, i in enumerate(Frag_atom):
            w.write(
                str(len(i) + len(hlist_atom[center_atom[idx]]) + len(hlist_atom[j]))
                + "\n"
            )
            w.write("Fragment - " + str(idx) + "\n")
            for j in hlist_atom[center_atom[idx]]:
                w.write(
                    " {:>3}   {:>10.7f}   {:>10.7f}   {:>10.7f} \n".format(
                        cell.atom_pure_symbol(j),
                        coord[j][0] / ang2bohr,
                        coord[j][1] / ang2bohr,
                        coord[j][2] / ang2bohr,
                    )
                )
            for j in i:
                w.write(
                    " {:>3}   {:>10.7f}   {:>10.7f}   {:>10.7f} \n".format(
                        cell.atom_pure_symbol(j),
                        coord[j][0] / ang2bohr,
                        coord[j][1] / ang2bohr,
                        coord[j][2] / ang2bohr,
                    )
                )
                for k in hlist_atom[j]:
                    w.write(
                        " {:>3}   {:>10.7f}   {:>10.7f}   {:>10.7f} \n".format(
                            cell.atom_pure_symbol(k),
                            coord[k][0] / ang2bohr,
                            coord[k][1] / ang2bohr,
                            coord[k][2] / ang2bohr,
                        )
                    )
        w.close()

    # Prepare for PAO basis if requested
    pao = bool(iao_valence_basis and not iao_valence_only)

    if pao:
        cell2 = cell.copy()
        cell2.basis = iao_valence_basis
        cell2.build()

        bas2list = cell2.aoslice_by_atom()
        nbas2 = [0 for i in range(cell.natm)]

    baslist = cell.aoslice_by_atom()
    sites__ = [[] for i in coord]
    coreshift = 0
    hshift = [0 for i in coord]

    # Process each atom to determine core and valence basis sets
    for adx in range(cell.natm):
        if hchain:
            bas = baslist[adx]
            start_ = bas[2]
            stop_ = bas[3]
            if pao:
                bas2 = bas2list[adx]
                nbas2[adx] += bas2[3] - bas2[2]
            b1list = [i for i in range(start_, stop_)]
            sites__[adx] = b1list
            continue

        if not cell.atom_pure_symbol(adx) == "H" and not hchain:
            bas = baslist[adx]
            start_ = bas[2]
            stop_ = bas[3]
            if pao:
                bas2 = bas2list[adx]
                nbas2[adx] += bas2[3] - bas2[2]

            if frozen_core:
                start_ -= coreshift
                ncore_ = core_list[adx]
                stop_ -= coreshift + ncore_
                if pao:
                    nbas2[adx] -= ncore_
                coreshift += ncore_

            b1list = [i for i in range(start_, stop_)]
            sites__[adx] = b1list
        else:
            hshift[adx] = coreshift

    hsites = [[] for i in coord]
    nbas2H = [0 for i in coord]
    for hdx, h in enumerate(hlist_atom):
        for hidx in h:
            basH = baslist[hidx]
            startH = basH[2]
            stopH = basH[3]

            if pao:
                bas2H = bas2list[hidx]
                nbas2H[hdx] += bas2H[3] - bas2H[2]

            if frozen_core:
                startH -= hshift[hidx]
                stopH -= hshift[hidx]

            b1list = [i for i in range(startH, stopH)]
            hsites[hdx].extend(b1list)

    fsites = []
    edge_sites = []
    edge_idx = []
    centerf_idx = []
    edge = []

    # Create fragments and edges based on partitioning
    for idx, i in enumerate(Frag_atom):
        ftmp = []
        ftmpe = []
        indix = 0
        edind = []
        edg = []

        frglist = sites__[center_atom[idx]].copy()
        frglist.extend(hsites[center_atom[idx]])

        ls = len(sites__[center_atom[idx]]) + len(hsites[center_atom[idx]])
        if idx in open_frag:
            for pidx__, pid__ in enumerate(open_frag):
                if idx == pid__:
                    frglist.extend(sites__[open_frag_cen[pidx__]])
                    frglist.extend(hsites[open_frag_cen[pidx__]])
                    ls += len(sites__[open_frag_cen[pidx__]]) + len(
                        hsites[open_frag_cen[pidx__]]
                    )

        ftmp.extend(frglist)
        if not pao:
            ls_ = len(sites__[center_atom[idx]]) + len(hsites[center_atom[idx]])
            centerf_idx.append([pq for pq in range(indix, indix + ls_)])
        else:
            cntlist = sites__[center_atom[idx]].copy()[: nbas2[center_atom[idx]]]
            cntlist.extend(hsites[center_atom[idx]][: nbas2H[center_atom[idx]]])
            ind__ = [indix + frglist.index(pq) for pq in cntlist]
            centerf_idx.append(ind__)
        indix += ls

        if not be_type == "be1":
            for jdx in pedge[idx]:
                if idx in open_frag:
                    if jdx == open_frag_cen[open_frag.index(idx)]:
                        continue
                    if jdx in open_frag_cen:
                        continue
                edg.append(jdx)
                frglist = sites__[jdx].copy()
                frglist.extend(hsites[jdx])

                ftmp.extend(frglist)
                ls = len(sites__[jdx]) + len(hsites[jdx])
                if not pao:
                    edglist = sites__[jdx].copy()
                    edglist.extend(hsites[jdx])
                    ftmpe.append(edglist)
                    edind.append([pq for pq in range(indix, indix + ls)])
                else:
                    edglist = sites__[jdx][: nbas2[jdx]].copy()
                    edglist.extend(hsites[jdx][: nbas2H[jdx]])

                    ftmpe.append(edglist)
                    ind__ = [indix + frglist.index(pq) for pq in edglist]
                    edind.append(ind__)
                indix += ls
            edge.append(edg)
            edge_sites.append(ftmpe)
            edge_idx.append(edind)
        fsites.append(ftmp)
    center = []
    for ix in edge:
        cen_ = []
        for jx in ix:
            if jx in center_atom:
                cen_.append(center_atom.index(jx))
            elif jx in open_frag_cen:
                cen_.append(open_frag[open_frag_cen.index(jx)])
            else:
                raise ValueError("This is more complicated than I can handle.")

        center.append(cen_)

    Nfrag = len(fsites)

    add_center_atom = [[] for x in range(Nfrag)]  # additional centers for mixed-basis
    ebe_weight = []

    # Compute weights for each fragment
    for ix, i in enumerate(fsites):
        tmp_ = [i.index(pq) for pq in sites__[center_atom[ix]]]
        tmp_.extend([i.index(pq) for pq in hsites[center_atom[ix]]])
        if ix in open_frag:
            for pidx__, pid__ in enumerate(open_frag):
                if ix == pid__:
                    add_center_atom[pid__].append(open_frag_cen[pidx__])
                    tmp_.extend([i.index(pq) for pq in sites__[open_frag_cen[pidx__]]])
                    tmp_.extend([i.index(pq) for pq in hsites[open_frag_cen[pidx__]]])
        ebe_weight.append([1.0, tmp_])

    center_idx = []
    if not be_type == "be1":
        for i in range(Nfrag):
            idx = []
            for jdx, j in enumerate(center[i]):
                jdx_continue = False
                if j in open_frag:
                    for kdx, k in enumerate(open_frag):
                        if j == k:
                            if edge[i][jdx] == open_frag_cen[kdx]:
                                if not pao:
                                    cntlist = sites__[open_frag_cen[kdx]].copy()
                                    cntlist.extend(hsites[open_frag_cen[kdx]])
                                    idx.append([fsites[j].index(k) for k in cntlist])
                                else:
                                    cntlist = sites__[open_frag_cen[kdx]].copy()[
                                        : nbas2[center_atom[j]]
                                    ]
                                    cntlist.extend(
                                        hsites[open_frag_cen[kdx]][
                                            : nbas2H[center_atom[j]]
                                        ]
                                    )
                                    idx.append([fsites[j].index(k) for k in cntlist])
                                jdx_continue = True
                                break

                if jdx_continue:
                    continue
                if not pao:
                    cntlist = sites__[center_atom[j]].copy()
                    cntlist.extend(hsites[center_atom[j]])
                    idx.append([fsites[j].index(k) for k in cntlist])
                else:
                    cntlist = sites__[center_atom[j]].copy()[: nbas2[center_atom[j]]]
                    cntlist.extend(hsites[center_atom[j]][: nbas2H[center_atom[j]]])
                    idx.append([fsites[j].index(k) for k in cntlist])

            center_idx.append(idx)

    return FragPart(
        mol=mol,
        frag_type="autogen",
        be_type=be_type,
        fsites=fsites,
        edge_sites=edge_sites,
        center=center,
        edge_idx=edge_idx,
        center_idx=center_idx,
        centerf_idx=centerf_idx,
        ebe_weight=ebe_weight,
        Frag_atom=Frag_atom,
        center_atom=center_atom,
        hlist_atom=hlist_atom,
        add_center_atom=add_center_atom,
        frozen_core=frozen_core,
        iao_valence_basis=iao_valence_basis,
        iao_valence_only=iao_valence_only,
    )
