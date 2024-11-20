# Author: Oinam Romesh Meitei

import sys
import numpy
from .helper import get_core


def autogen(
    mol,
    frozen_core=True,
    be_type="be2",
    write_geom=False,
    valence_basis=None,
    valence_only=False,
    print_frags=True,
):
    """
    Automatic molecular partitioning

    Partitions a molecule into overlapping fragments as defined in BE atom-based fragmentations.
    It automatically detects branched chemical chains and ring systems and partitions accordingly.
    For efficiency, it only checks two atoms for connectivity (chemical bond) if they are within 3.5 Angstrom.
    This value is hardcoded as normdist. Two atoms are defined as bonded if they are within 1.8 Angstrom (1.2 for Hydrogen atom).
    This is also hardcoded as bond & hbond.

    Parameters
    ----------
    mol : pyscf.gto.Molecule
        pyscf.gto.Molecule object. This is required for the options, 'autogen' and 'chain' as frag_type.
    frozen_core : bool, optional
        Whether to invoke frozen core approximation. Defaults to True.
    be_type : str, optional
        Specifies the order of bootstrap calculation in the atom-based fragmentation. Supported values are 'be1', 'be2', 'be3', and 'be4'.
        Defaults to 'be2'.
    write_geom : bool, optional
        Whether to write a 'fragment.xyz' file which contains all the fragments in Cartesian coordinates. Defaults to False.
    valence_basis : str, optional
        Name of minimal basis set for IAO scheme. 'sto-3g' is sufficient for most cases. Defaults to None.
    valence_only : bool, optional
        If True, all calculations will be performed in the valence basis in the IAO partitioning. This is an experimental feature. Defaults to False.
    print_frags : bool, optional
        Whether to print out the list of resulting fragments. Defaults to True.

    Returns
    -------
    fsites : list of list of int
        List of fragment sites where each fragment is a list of LO indices.
    edgsites : list of list of list of int
        List of edge sites for each fragment where each edge is a list of LO indices.
    center : list of list of int
        List of center indices for each edge.
    edge_idx : list of list of list of int
        List of edge indices for each fragment where each edge index is a list of LO indices.
    center_idx : list of list of list of int
        List of center indices for each fragment where each center index is a list of LO indices.
    centerf_idx : list of list of int
        List of center fragment indices.
    ebe_weight : list of list
        Weights for each fragment. Each entry contains a weight and a list of LO indices.
    Frag: list of lists
        Heavy atom indices for each fragment, per fragment
    cen: list
        Atom indices of all centers
    hlist: list of lists
        All hydrogen atom indices for each fragment, per fragment
    add_centers: list of lists
        "additional centers" for all fragments, per fragment: contains heavy atoms which are not
        centers in any other fragments

    """
    from pyscf import lib

    if not valence_only:
        cell = mol.copy()
    else:
        cell = mol.copy()
        cell.basis = valence_basis
        cell.build()

    ncore, no_core_idx, core_list = get_core(cell)
    coord = cell.atom_coords()
    ang2bohr = 1.88973
    normdist = 3.5 * ang2bohr
    bond = 1.8 * ang2bohr
    hbond = 1.2 * ang2bohr

    # Compute the norm (magnitude) of each atomic coordinate
    normlist = []
    for i in coord:
        normlist.append(numpy.linalg.norm(i))
    Frag = []
    pedge = []
    cen = []

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
        cout = 0
        for jdx, j in enumerate(tmplist):
            if not idx == jdx and (not cell.atom_pure_symbol(jdx) == "H" or hchain):
                if abs(j) < normdist:
                    clist.append(jdx)
        pedg = []
        flist = []
        flist.append(idx)

        if not be_type == "be1":
            for jdx in clist:
                dist = numpy.linalg.norm(coord[idx] - coord[jdx])
                if dist <= bond:
                    flist.append(jdx)
                    pedg.append(jdx)
                    if be_type == "be3" or be_type == "be4":
                        for kdx in clist:
                            if not kdx == jdx:
                                dist = numpy.linalg.norm(coord[jdx] - coord[kdx])
                                if dist <= bond:
                                    if not kdx in pedg:
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
                                            dist = numpy.linalg.norm(coord[kdx] - l)
                                            if dist <= bond:
                                                flist.append(ldx)
                                                pedg.append(ldx)

            # Update fragment and edge lists based on current partitioning
            for pidx, frag_ in enumerate(Frag):
                if set(flist).issubset(frag_):
                    open_frag.append(pidx)
                    open_frag_cen.append(idx)
                    break
                elif set(frag_).issubset(flist):
                    open_frag = [
                        oidx - 1 if oidx > pidx else oidx for oidx in open_frag
                    ]
                    open_frag.append(len(Frag) - 1)
                    open_frag_cen.append(cen[pidx])
                    del cen[pidx]
                    del Frag[pidx]
                    del pedge[pidx]
            else:
                Frag.append(flist)
                pedge.append(pedg)
                cen.append(idx)
        else:
            Frag.append(flist)
            cen.append(idx)

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
                    dist = numpy.linalg.norm(coord[idx] - coord[jdx])
                    if dist <= hbond:
                        hlist[jdx].append(idx)

    # Print fragments if requested
    if print_frags:
        print(flush=True)
        print("Fragment sites", flush=True)
        print("--------------------------", flush=True)
        print("Fragment |   Center | Edges ", flush=True)
        print("--------------------------", flush=True)

        for idx, i in enumerate(Frag):
            print(
                "   {:>4}  |   {:>5}  |".format(
                    idx, cell.atom_pure_symbol(cen[idx]) + str(cen[idx] + 1)
                ),
                end=" ",
                flush=True,
            )
            for j in hlist[cen[idx]]:
                print(
                    " {:>5} ".format("*" + cell.atom_pure_symbol(j) + str(j + 1)),
                    end=" ",
                    flush=True,
                )
            for j in i:
                if j == cen[idx]:
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
        print(" No. of fragments : ", len(Frag), flush=True)
        print("*H : Center H atoms (printed as Edges above.)", flush=True)
        print(flush=True)

    # Write fragment geometry to a file if requested
    if write_geom:
        w = open("fragments.xyz", "w")
        for idx, i in enumerate(Frag):
            w.write(str(len(i) + len(hlist[cen[idx]]) + len(hlist[j])) + "\n")
            w.write("Fragment - " + str(idx) + "\n")
            for j in hlist[cen[idx]]:
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
    pao = bool(valence_basis and not valence_only)

    if pao:
        cell2 = cell.copy()
        cell2.basis = valence_basis
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
    for idx, i in enumerate(Frag):
        ftmp = []
        ftmpe = []
        indix = 0
        edind = []
        edg = []

        frglist = sites__[cen[idx]].copy()
        frglist.extend(hsites[cen[idx]])

        ls = len(sites__[cen[idx]]) + len(hsites[cen[idx]])
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
            ls_ = len(sites__[cen[idx]]) + len(hsites[cen[idx]])
            centerf_idx.append([pq for pq in range(indix, indix + ls_)])
        else:
            cntlist = sites__[cen[idx]].copy()[: nbas2[cen[idx]]]
            cntlist.extend(hsites[cen[idx]][: nbas2H[cen[idx]]])
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
            if jx in cen:
                cen_.append(cen.index(jx))
            elif jx in open_frag_cen:
                cen_.append(open_frag[open_frag_cen.index(jx)])
            else:
                print(" This is more complicated than I can handle")
                sys.exit()
        center.append(cen_)

    Nfrag = len(fsites)

    add_centers = [[] for x in range(Nfrag)]  # additional centers for mixed-basis
    ebe_weight = []

    # Compute weights for each fragment
    for ix, i in enumerate(fsites):
        tmp_ = [i.index(pq) for pq in sites__[cen[ix]]]
        tmp_.extend([i.index(pq) for pq in hsites[cen[ix]]])
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
                                        : nbas2[cen[j]]
                                    ]
                                    cntlist.extend(
                                        hsites[open_frag_cen[kdx]][: nbas2H[cen[j]]]
                                    )
                                    idx.append([fsites[j].index(k) for k in cntlist])
                                jdx_continue = True
                                break

                if jdx_continue:
                    continue
                if not pao:
                    cntlist = sites__[cen[j]].copy()
                    cntlist.extend(hsites[cen[j]])
                    idx.append([fsites[j].index(k) for k in cntlist])
                else:
                    cntlist = sites__[cen[j]].copy()[: nbas2[cen[j]]]
                    cntlist.extend(hsites[cen[j]][: nbas2H[cen[j]]])
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
        Frag,
        cen,
        hlist,
        add_centers,
    )
