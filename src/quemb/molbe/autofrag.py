# Author: Oinam Romesh Meitei
# type: ignore

from warnings import warn

from attrs import define
from numpy.linalg import norm

from quemb.molbe.helper import get_core
from quemb.shared.helper import unused


@define
class AutoGenArgs:
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
) -> dict:
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
    if n_BE != 1:
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

    return {
        "mol": mol,
        "frag_type": "autogen",
        "n_BE": n_BE,
        "fsites": fsites,
        "edge_sites": edge_sites,
        "center": center,
        "edge_idx": edge_idx,
        "center_idx": center_idx,
        "centerf_idx": centerf_idx,
        "ebe_weight": ebe_weight,
        "Frag_atom": Frag_atom,
        "center_atom": center_atom,
        "hlist_atom": hlist_atom,
        "add_center_atom": add_center_atom,
        "frozen_core": frozen_core,
        "iao_valence_basis": iao_valence_basis,
        "iao_valence_only": iao_valence_only,
    }
