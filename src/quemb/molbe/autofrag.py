# Author: Oinam Romesh Meitei, Shaun Weatherly

from collections.abc import Sequence
from copy import deepcopy
from typing import Final, Literal, TypeAlias
from warnings import warn

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
    ListOverEdge,
    ListOverFrag,
    ListOverMotif,
    MotifIdx,
    OriginIdx,
    OtherRelAOIdx,
    OwnRelAOIdx,
    Vector,
)

FragType: TypeAlias = Literal["chemgen", "graphgen", "autogen"]


@define(kw_only=True)
class FragPart:
    """Data structure to hold the result of BE fragmentations."""

    #: The full molecule.
    mol: Mole = field(eq=cmp_using(are_equal))
    #: The algorithm used for fragmenting.
    frag_type: FragType
    #: The level of BE fragmentation, i.e. 1, 2, ...
    n_BE: int

    #: This is a list over fragments  and gives the global orbital indices of all atoms
    #: in the fragment. These are ordered by the atoms in the fragment.
    AO_per_frag: ListOverFrag[list[GlobalAOIdx]]

    #: The global orbital indices, including hydrogens, per edge per fragment.
    AO_per_edge_per_frag: ListOverFrag[ListOverEdge[list[GlobalAOIdx]]]

    #: Reference fragment index per edge:
    #: A list over fragments: list of indices of the fragments in which an edge
    #: of the fragment is actually a center.
    #: The edge will be matched against this center.
    #: For fragments A, B: the A’th element of :python:`.center`,
    #: if the edge of A is the center of B, will be B.
    ref_frag_idx_per_edge: ListOverFrag[ListOverEdge[FragmentIdx]]

    #: The relative orbital indices, including hydrogens, per edge per fragment.
    #: The index is relative to the own fragment.
    rel_AO_per_edge_per_frag: ListOverFrag[ListOverEdge[list[OwnRelAOIdx]]]

    #: The relative atomic orbital indices per edge per fragment.
    #: **Note** for this variable relative means that the AO indices
    #: are relative to the other fragment where the edge is a center.
    other_rel_AO_per_edge_per_frag: ListOverFrag[ListOverEdge[list[OtherRelAOIdx]]]

    #: List whose entries are lists containing the relative orbital index of the
    #: origin site within a fragment. Relative is to the own fragment.
    #  Since the origin site is at the beginning
    #: of the motif list for each fragment, this is always a ``list(range(0, n))``
    rel_AO_per_origin_per_frag: ListOverFrag[list[OwnRelAOIdx]]

    #: The first element is a float, the second is the list
    #: The float weight makes only sense for democratic matching and is currently 1.0
    #: everywhere anyway. We concentrate only on the second part,
    #: i.e. the list of indices.
    #: This is a list whose entries are sequences containing the relative orbital index
    #  of the center sites within a fragment. Relative is to the own fragment.
    scale_rel_AO_per_center_per_frag: ListOverFrag[tuple[float, list[OwnRelAOIdx]]]

    #: The motifs/heavy atoms in each fragment, in order.
    #: Each are labeled based on the global atom index.
    #: It is ordered by origin, centers, edges!
    motifs_per_frag: ListOverFrag[ListOverMotif[MotifIdx]]

    #: The origin for each fragment.
    #: (Note that for conventional BE there is just one origin per fragment)
    origin_per_frag: ListOverFrag[OriginIdx]

    #: A list over atoms (not over motifs!)
    #: For each atom it contains a list of the attached hydrogens.
    #: This means that there are a lot of empty sets for molecular systems,
    # because hydrogens have no attached hydrogens (usually).
    H_per_motif: Sequence[list[AtomIdx]]

    #: A list over fragments.
    #: For each fragment a list of centers that are not the origin of that fragment.
    add_center_atom: ListOverFrag[list[CenterIdx]]

    frozen_core: bool
    iao_valence_basis: str | None

    #: If this option is set to True, all calculation will be performed in
    #: the valence basis in the IAO partitioning.
    #: This is an experimental feature.
    iao_valence_only: bool

    n_frag: int = field(init=False)
    ncore: int | None = field(init=False)
    no_core_idx: list[int] | None = field(init=False)
    core_list: list[int] | None = field(init=False)

    @n_frag.default
    def _get_default_n_frag(self) -> int:
        return len(self.AO_per_frag)

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
        return self.n_frag


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
    AO_per_edge_per_frag :
        List whose entries are sequences of sequences, containing edge AO
        indices per atom (inner tuple) per fragment (outer tuple).
    center :
        List whose entries are sequences of sequences, containing all fragment AO
        indices per atom (inner tuple) and per fragment (outer tuple).
    rel_AO_per_origin_per_frag :
        List whose entries are sequences containing the relative AO index of the
        origin site within a fragment.
        Relative is to the own fragment; since the origin site is at the beginning
        of the motif list for each fragment, this is always a Sequence
        :python:`range(0, n)`.
    scale_rel_AO_per_center_per_frag :
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
    AO_per_edge_per_frag: list[Sequence[Sequence[int]]]
    ref_frag_idx_per_edge: list[Sequence[int]]
    rel_AO_per_origin_per_frag: list[Sequence[int]]
    scale_rel_AO_per_center_per_frag: list[Sequence]
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
        return FragPart(
            mol=mol,
            frag_type="graphgen",
            n_BE=n_BE,
            AO_per_edge_per_frag=self.AO_per_edge_per_frag,  # type: ignore[arg-type]
            rel_AO_per_edge_per_frag=MISSING,
            other_rel_AO_per_edge_per_frag=MISSING,
            rel_AO_per_origin_per_frag=self.rel_AO_per_origin_per_frag,  # type: ignore[arg-type]
            AO_per_frag=self.AO_per_frag,  # type: ignore[arg-type]
            ref_frag_idx_per_edge=self.ref_frag_idx_per_edge,  # type: ignore[arg-type]
            scale_rel_AO_per_center_per_frag=self.scale_rel_AO_per_center_per_frag,  # type: ignore[arg-type]
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
        AO_per_edge_per_frag=list(tuple(tuple())),
        ref_frag_idx_per_edge=list(tuple()),
        rel_AO_per_origin_per_frag=list(tuple()),
        scale_rel_AO_per_center_per_frag=list(tuple()),
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
        fragment_map.AO_per_edge_per_frag.append(tuple(edge_temp))
        fragment_map.edge_atoms.extend(tuple(eatoms_temp))

    # Update relative center site indices (centerf_idx) and weights
    # for center site contributions to the energy ():
    for adx, center in enumerate(fragment_map.ref_frag_idx_per_edge):
        rel_AO_per_origin_per_frag = tuple(
            fragment_map.AO_per_frag[adx].index(cdx) for cdx in center
        )
        fragment_map.rel_AO_per_origin_per_frag.append(rel_AO_per_origin_per_frag)
        fragment_map.scale_rel_AO_per_center_per_frag.append(
            (1.0, tuple(rel_AO_per_origin_per_frag))
        )

    # Finally, set fragment data names for scratch and bookkeeping:
    for adx, _ in enumerate(fragment_map.fs):
        fragment_map.dnames.append(str(frag_prefix) + str(adx))

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
    n_BE=2,
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
    n_BE: int, optional
        Specifies the order of bootstrap calculation in the atom-based fragmentation,
        i.e. BE(n).
        Supported values are 1, 2, 3, and 4
        Defaults to 2.
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
    motifs_per_frag = []
    pedge = []
    origin_per_frag = []

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

        if n_BE != 1:
            for jdx in clist:
                dist = norm(coord[idx] - coord[jdx])
                if dist <= bond:
                    flist.append(jdx)
                    pedg.append(jdx)
                    if 3 <= n_BE <= 4:
                        for kdx in clist:
                            if not kdx == jdx:
                                dist = norm(coord[jdx] - coord[kdx])
                                if dist <= bond:
                                    if kdx not in pedg:
                                        flist.append(kdx)
                                        pedg.append(kdx)
                                    if n_BE == 4:
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
            for pidx, frag_ in enumerate(motifs_per_frag):
                if set(flist).issubset(frag_):
                    open_frag.append(pidx)
                    open_frag_cen.append(idx)
                    break
                elif set(frag_).issubset(flist):
                    open_frag = [
                        oidx - 1 if oidx > pidx else oidx for oidx in open_frag
                    ]
                    open_frag.append(len(motifs_per_frag) - 1)
                    open_frag_cen.append(origin_per_frag[pidx])
                    del origin_per_frag[pidx]
                    del motifs_per_frag[pidx]
                    del pedge[pidx]
            else:
                motifs_per_frag.append(flist)
                pedge.append(pedg)
                origin_per_frag.append(idx)
        else:
            motifs_per_frag.append(flist)
            origin_per_frag.append(idx)

    H_per_motif = [[] for i in coord]
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
                        H_per_motif[jdx].append(idx)

    # Print fragments if requested
    if print_frags:
        print(flush=True)
        print("Fragment sites", flush=True)
        print("--------------------------", flush=True)
        print("Fragment |   Origin | Atoms ", flush=True)
        print("--------------------------", flush=True)

        for idx, i in enumerate(motifs_per_frag):
            print(
                "   {:>4}  |   {:>5}  |".format(
                    idx,
                    cell.atom_pure_symbol(origin_per_frag[idx])
                    + str(origin_per_frag[idx] + 1),
                ),
                end=" ",
                flush=True,
            )
            for j in H_per_motif[origin_per_frag[idx]]:
                print(
                    " {:>5} ".format("*" + cell.atom_pure_symbol(j) + str(j + 1)),
                    end=" ",
                    flush=True,
                )
            for j in i:
                if j == origin_per_frag[idx]:
                    continue
                print(
                    f" {cell.atom_pure_symbol(j) + str(j + 1):>5} ",
                    end=" ",
                    flush=True,
                )
                for k in H_per_motif[j]:
                    print(
                        f" {cell.atom_pure_symbol(k) + str(k + 1):>5} ",
                        end=" ",
                        flush=True,
                    )
            print(flush=True)
        print("--------------------------", flush=True)
        print(" No. of fragments : ", len(motifs_per_frag), flush=True)
        print("*H : Center H atoms (printed as Edges above.)", flush=True)
        print(flush=True)

    # Write fragment geometry to a file if requested
    if write_geom:
        w = open("fragments.xyz", "w")
        for idx, i in enumerate(motifs_per_frag):
            w.write(
                str(
                    len(i)
                    + len(H_per_motif[origin_per_frag[idx]])
                    + len(H_per_motif[j])
                )
                + "\n"
            )
            w.write("Fragment - " + str(idx) + "\n")
            for j in H_per_motif[origin_per_frag[idx]]:
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
                for k in H_per_motif[j]:
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
    for hdx, h in enumerate(H_per_motif):
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

    AO_per_frag = []
    AO_per_edge_per_frag = []
    rel_AO_per_edge_per_frag = []
    rel_AO_per_origin_per_frag = []
    edge = []

    # Create fragments and edges based on partitioning
    for idx, i in enumerate(motifs_per_frag):
        ftmp = []
        ftmpe = []
        indix = 0
        edind = []
        edg = []

        frglist = sites__[origin_per_frag[idx]].copy()
        frglist.extend(hsites[origin_per_frag[idx]])

        ls = len(sites__[origin_per_frag[idx]]) + len(hsites[origin_per_frag[idx]])
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
            ls_ = len(sites__[origin_per_frag[idx]]) + len(hsites[origin_per_frag[idx]])
            rel_AO_per_origin_per_frag.append([pq for pq in range(indix, indix + ls_)])
        else:
            cntlist = sites__[origin_per_frag[idx]].copy()[
                : nbas2[origin_per_frag[idx]]
            ]
            cntlist.extend(hsites[origin_per_frag[idx]][: nbas2H[origin_per_frag[idx]]])
            ind__ = [indix + frglist.index(pq) for pq in cntlist]
            rel_AO_per_origin_per_frag.append(ind__)
        indix += ls

        if n_BE != 1:
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
            AO_per_edge_per_frag.append(ftmpe)
            rel_AO_per_edge_per_frag.append(edind)
        AO_per_frag.append(ftmp)
    ref_frag_idx_per_edge = []
    for ix in edge:
        cen_ = []
        for jx in ix:
            if jx in origin_per_frag:
                cen_.append(origin_per_frag.index(jx))
            elif jx in open_frag_cen:
                cen_.append(open_frag[open_frag_cen.index(jx)])
            else:
                raise ValueError("This is more complicated than I can handle.")

        ref_frag_idx_per_edge.append(cen_)

    n_frag = len(AO_per_frag)

    add_center_atom = [[] for x in range(n_frag)]  # additional centers for mixed-basis
    scale_rel_AO_per_center_per_frag = []

    # Compute weights for each fragment
    for ix, i in enumerate(AO_per_frag):
        tmp_ = [i.index(pq) for pq in sites__[origin_per_frag[ix]]]
        tmp_.extend([i.index(pq) for pq in hsites[origin_per_frag[ix]]])
        if ix in open_frag:
            for pidx__, pid__ in enumerate(open_frag):
                if ix == pid__:
                    add_center_atom[pid__].append(open_frag_cen[pidx__])
                    tmp_.extend([i.index(pq) for pq in sites__[open_frag_cen[pidx__]]])
                    tmp_.extend([i.index(pq) for pq in hsites[open_frag_cen[pidx__]]])
        scale_rel_AO_per_center_per_frag.append((1.0, tmp_))

    other_rel_AO_per_edge_per_frag = []
    if n_BE != 1:
        for i in range(n_frag):
            idx = []
            for jdx, j in enumerate(ref_frag_idx_per_edge[i]):
                jdx_continue = False
                if j in open_frag:
                    for kdx, k in enumerate(open_frag):
                        if j == k:
                            if edge[i][jdx] == open_frag_cen[kdx]:
                                if not pao:
                                    cntlist = sites__[open_frag_cen[kdx]].copy()
                                    cntlist.extend(hsites[open_frag_cen[kdx]])
                                    idx.append(
                                        [AO_per_frag[j].index(k) for k in cntlist]
                                    )
                                else:
                                    cntlist = sites__[open_frag_cen[kdx]].copy()[
                                        : nbas2[origin_per_frag[j]]
                                    ]
                                    cntlist.extend(
                                        hsites[open_frag_cen[kdx]][
                                            : nbas2H[origin_per_frag[j]]
                                        ]
                                    )
                                    idx.append(
                                        [AO_per_frag[j].index(k) for k in cntlist]
                                    )
                                jdx_continue = True
                                break

                if jdx_continue:
                    continue
                if not pao:
                    cntlist = sites__[origin_per_frag[j]].copy()
                    cntlist.extend(hsites[origin_per_frag[j]])
                    idx.append([AO_per_frag[j].index(k) for k in cntlist])
                else:
                    cntlist = sites__[origin_per_frag[j]].copy()[
                        : nbas2[origin_per_frag[j]]
                    ]
                    cntlist.extend(
                        hsites[origin_per_frag[j]][: nbas2H[origin_per_frag[j]]]
                    )
                    idx.append([AO_per_frag[j].index(k) for k in cntlist])

            other_rel_AO_per_edge_per_frag.append(idx)

    return FragPart(
        mol=mol,
        frag_type="autogen",
        n_BE=n_BE,
        AO_per_frag=AO_per_frag,
        AO_per_edge_per_frag=AO_per_edge_per_frag,
        ref_frag_idx_per_edge=ref_frag_idx_per_edge,
        rel_AO_per_edge_per_frag=rel_AO_per_edge_per_frag,
        other_rel_AO_per_edge_per_frag=other_rel_AO_per_edge_per_frag,
        rel_AO_per_origin_per_frag=rel_AO_per_origin_per_frag,
        scale_rel_AO_per_center_per_frag=scale_rel_AO_per_center_per_frag,
        motifs_per_frag=motifs_per_frag,
        origin_per_frag=origin_per_frag,
        H_per_motif=H_per_motif,
        add_center_atom=add_center_atom,
        frozen_core=frozen_core,
        iao_valence_basis=iao_valence_basis,
        iao_valence_only=iao_valence_only,
    )
