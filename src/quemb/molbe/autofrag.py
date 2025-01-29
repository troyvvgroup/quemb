# Author: Oinam Romesh Meitei, Shaun Weatherly

from copy import deepcopy
from typing import Sequence

import networkx as nx
import numpy as np
from attrs import define
from networkx import shortest_path
from numpy.linalg import norm
from pyscf import gto

from quemb.molbe.helper import get_core
from quemb.shared.helper import unused
from quemb.shared.typing import Vector


@define
class FragmentMap:
    """Dataclass for fragment bookkeeping.

    Parameters
    ----------
    fsites:
        List whose entries are sequences (tuple or list) containing
        all AO indices for a fragment.
    fs:
        List whose entries are sequences of sequences, containing AO indices per atom
        per fragment.
    edge:
        List whose entries are sequences of sequences, containing edge AO
        indices per atom (inner tuple) per fragment (outer tuple).
    center:
        List whose entries are sequences of sequences, containing all fragment AO
        indices per atom (inner tuple) and per fragment (outer tuple).
    centerf_idx:
        List whose entries are sequences containing the relative index of all
        center sites within a fragment (ie, with respect to fsites).
    ebe_weights:
        Weights determining the energy contributions from each center site
        (ie, with respect to centerf_idx).
    sites:
        List whose entries are sequences containing all AO indices per atom
        (excluding frozen core indices, if applicable).
    dnames:
        List of strings giving fragment data names. Useful for bookkeeping and
        for constructing fragment scratch directories.
    fragment_atoms:
        List whose entries are sequences containing all atom indices for a fragment.
    center_atoms:
        List whose entries are sequences giving the center atom indices per fragment.
    edge_atoms:
        List whose entries are sequences giving the edge atom indices per fragment.
    adjacency_mat:
        The adjacency matrix for all sites (atoms) in the system.
    adjacency_graph:
        The adjacency graph corresponding to `adjacency_mat`.
    """

    fsites: list[Sequence[int]]
    fs: list[Sequence[Sequence[int]]]
    edge: list[Sequence[Sequence[int]]]
    center: list[Sequence[int]]
    centerf_idx: list[Sequence[int]]
    ebe_weights: list[Sequence]
    sites: list[Sequence]
    dnames: list[str]
    fragment_atoms: list[Sequence[int]]
    center_atoms: list[Sequence[int]]
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
                        self.center_atoms[adx] = tuple(
                            set(
                                list(self.center_atoms[adx])
                                + list(deepcopy(self.center_atoms[bdx]))
                            )
                        )
            if subsets:
                sorted_subsets = sorted(subsets, reverse=True)
                for bdx in sorted_subsets:
                    del self.center[bdx]
                    del self.fsites[bdx]
                    del self.fs[bdx]
                    del self.center_atoms[bdx]
                    del self.fragment_atoms[bdx]

        return None


def euclidean_distance(
    i_coord: Vector,
    j_coord: Vector,
) -> np.floating:
    return norm(i_coord - j_coord)


def graphgen(
    mol: gto.Mole,
    be_type: str = "BE2",
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

    fragment_type_order = int(be_type[-1])
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
        fs=list(tuple(tuple())),
        edge=list(tuple(tuple())),
        center=list(tuple()),
        centerf_idx=list(tuple()),
        ebe_weights=list(tuple()),
        sites=list(tuple()),
        dnames=list(),
        fragment_atoms=list(),
        center_atoms=list(),
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
            fragment_map.center_atoms.append((adx,))
            fragment_map.center.append(deepcopy(fragment_map.sites[adx]))
            fsites_temp = deepcopy(list(fragment_map.sites[adx]))
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
                if 0 < (len(path) - 1) < fragment_type_order:
                    fsites_temp = fsites_temp + deepcopy(list(fragment_map.sites[bdx]))
                    fs_temp.append(deepcopy(fragment_map.sites[bdx]))
                    fatoms_temp.append(bdx)

            fragment_map.fsites.append(tuple(fsites_temp))
            fragment_map.fs.append(tuple(fs_temp))
            fragment_map.fragment_atoms.append(tuple(fatoms_temp))

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
        for bdx, center in enumerate(fragment_map.center):
            if adx == bdx:
                pass
            else:
                for f in fs:
                    overlap = set(f).intersection(set(center))
                    if overlap:
                        f_temp = set(fragment_map.fragment_atoms[adx])
                        c_temp = set(fragment_map.center_atoms[bdx])
                        edge_temp.add(tuple(overlap))
                        eatoms_temp.add(tuple(i for i in f_temp.intersection(c_temp)))
        fragment_map.edge.append(tuple(edge_temp))
        fragment_map.edge_atoms.extend(tuple(eatoms_temp))

    # Update relative center site indices (centerf_idx) and weights
    # for center site contributions to the energy (ebe_weights):
    for adx, center in enumerate(fragment_map.center):
        centerf_idx = tuple(fragment_map.fsites[adx].index(cdx) for cdx in center)
        ebe_weight = (1.0, tuple(centerf_idx))
        fragment_map.centerf_idx.append(centerf_idx)
        fragment_map.ebe_weights.append(ebe_weight)

    # Finally, set fragment data names for scratch and bookkeeping:
    for adx, _ in enumerate(fragment_map.fs):
        fragment_map.dnames.append(str(frag_prefix) + str(adx))

    return fragment_map


def autogen(
    mol,
    frozen_core=True,
    be_type="be2",
    write_geom=False,
    iao_valence_basis=None,
    valence_only=False,
    print_frags=True,
):
    """
    Automatic molecular partitioning

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
    valence_only : bool, optional
        If True, all calculations will be performed in the valence basis in
        the IAO partitioning. This is an experimental feature. Defaults to False.
    print_frags : bool, optional
        Whether to print out the list of resulting fragments. Defaults to True.

    Returns
    -------
    fsites : list of list of int
        List of fragment sites where each fragment is a list of LO indices.
    edgsites : list of list of list of int
        List of edge sites for each fragment where each edge is a list of LO indices.
    center : list of list of int
        List of the fragment index of each edge site for all fragments.
    edge_idx : list of list of list of int
        List of edge indices for each fragment where each edge index is a list of
        LO indices.
    center_idx : list of list of list of int
        List of center indices for each fragment where each center index is a list of
        LO indices.
    centerf_idx : list of list of int
        List of center fragment indices.
    ebe_weight : list of list
        Weights for each fragment. Each entry contains a weight and a list of
        LO indices.
    Frag_atom: list of lists
        Heavy atom indices for each fragment, per fragment
    cen: list
        Atom indices of all centers
    """

    if not valence_only:
        cell = mol.copy()
    else:
        cell = mol.copy()
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

    hlist = [[] for i in coord]
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
                        hlist[jdx].append(idx)

    # Print fragments if requested
    if print_frags:
        print(flush=True)
        print("Fragment sites", flush=True)
        print("--------------------------", flush=True)
        print("Fragment |   Center | Edges ", flush=True)
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
            for j in hlist[center_atom[idx]]:
                print(
                    " {:>5} ".format("*" + cell.atom_pure_symbol(j) + str(j + 1)),
                    end=" ",
                    flush=True,
                )
            for j in i:
                if j == center_atom[idx]:
                    continue
                print(
                    " {:>5} ".format(cell.atom_pure_symbol(j) + str(j + 1)),
                    end=" ",
                    flush=True,
                )
                for k in hlist[j]:
                    print(
                        " {:>5} ".format(cell.atom_pure_symbol(k) + str(k + 1)),
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
            w.write(str(len(i) + len(hlist[center_atom[idx]]) + len(hlist[j])) + "\n")
            w.write("Fragment - " + str(idx) + "\n")
            for j in hlist[center_atom[idx]]:
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
                for k in hlist[j]:
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
    pao = bool(iao_valence_basis and not valence_only)

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
    for hdx, h in enumerate(hlist):
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
    edgsites = []
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
            edgsites.append(ftmpe)
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

    add_centers = [[] for x in range(Nfrag)]  # additional centers for mixed-basis
    ebe_weight = []

    # Compute weights for each fragment
    for ix, i in enumerate(fsites):
        tmp_ = [i.index(pq) for pq in sites__[center_atom[ix]]]
        tmp_.extend([i.index(pq) for pq in hsites[center_atom[ix]]])
        if ix in open_frag:
            for pidx__, pid__ in enumerate(open_frag):
                if ix == pid__:
                    add_centers[pid__].append(open_frag_cen[pidx__])
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

    return (
        fsites,
        edgsites,
        center,
        edge_idx,
        center_idx,
        centerf_idx,
        ebe_weight,
        Frag_atom,
        center_atom,
    )
