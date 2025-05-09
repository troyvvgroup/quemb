# Author(s): Oinam Romesh Meitei

from warnings import warn

from attrs import define
from numpy import arange, asarray, where
from numpy.linalg import norm
from pyscf import lib

from quemb.kbe.misc import sgeom
from quemb.molbe.helper import get_core


@define
class AutogenArgs:
    """Additional arguments for autogen

    Parameters
    ----------
    gamma_2d:
    gamma_1d:
    interlayer :
        Whether the periodic system has two stacked monolayers.
    long_bond :
        For systems with longer than 1.8 Angstrom covalent bond, set this to True
        otherwise the fragmentation might fail.
    perpend_dist:
    perpend_dist_tol:
    nx:
    ny:
    nz:
    """

    gamma_2d: bool = False
    gamma_1d: bool = False
    interlayer: bool = False
    long_bond: bool = False
    perpend_dist: float = 4.0
    perpend_dist_tol: float = 1e-3
    nx: bool = False
    ny: bool = False
    nz: bool = False


def warn_large_fragment():
    raise ValueError(
        "Fragments that spans more than 2 unit-cells are not supported"
        "Try with larger unit-cell(or super-cell)"
    )


def add_check_k(min1, flist, sts, ksts, nk_):
    sts_ = sts.copy()
    for q_ in min1:
        if q_ in sts_:
            for qjdx, qj in enumerate(sts_):
                if qj == q_ and ksts[qjdx] == nk_:
                    break
            else:
                flist.append(q_)
                sts.append(q_)
                ksts.append(nk_)
        else:
            flist.append(q_)
            sts.append(q_)
            ksts.append(nk_)


def nearestof2coord(coord1, coord2, bond=2.6 * 1.88973):
    mind = 50000
    lmin = ()
    for idx, i in enumerate(coord1):
        for jdx, j in enumerate(coord2):
            if idx == jdx:
                continue
            dist = norm(i - j)

            if dist < mind or dist - mind < 0.1:
                if dist <= bond:
                    lmin = (idx, jdx)
                    mind = dist
    if any(lmin):
        lunit_ = [lmin[0]]
        runit_ = [lmin[1]]
    else:
        return ([], [])

    for idx, i in enumerate(coord1):
        for jdx, j in enumerate(coord2):
            if idx == jdx:
                continue

            if idx == lmin[0] and jdx == lmin[1]:
                continue
            dist = norm(i - j)
            if dist - mind < 0.1 and dist <= bond:
                lunit_.append(idx)
                runit_.append(jdx)

    return (lunit_, runit_)


def sidefunc(
    cell,
    Idx,
    unit1,
    unit2,
    main_list,
    sub_list,
    coord,
    n_BE: int,
    bond=2.6 * 1.88973,
    klist=[],
    ext_list=[],
    NK=None,
    rlist=[],
):
    if ext_list == []:
        main_list.extend(unit2[where(unit1 == Idx)[0]])
        sub_list.extend(unit2[where(unit1 == Idx)[0]])
    else:
        for sub_i in unit2[where(unit1 == Idx)[0]]:
            if sub_i in rlist:
                continue
            if sub_i in ext_list:
                for tmp_jdx, tmpj in enumerate(ext_list):
                    if sub_i == tmpj and klist[tmp_jdx] == NK:
                        break
                else:
                    main_list.append(sub_i)
                    sub_list.append(sub_i)
            else:
                main_list.append(sub_i)
                sub_list.append(sub_i)

    closest = sub_list.copy()
    close_be3 = []

    if 3 <= n_BE <= 4:
        for lmin1 in unit2[where(unit1 == Idx)[0]]:
            for jdx, j in enumerate(coord):
                if (
                    jdx not in unit1
                    and jdx not in unit2
                    and not cell.atom_pure_symbol(jdx) == "H"
                ):
                    dist = norm(coord[lmin1] - j)
                    if dist <= bond:
                        if jdx not in sub_list:  # avoid repeated occurence
                            main_list.append(jdx)
                            sub_list.append(jdx)
                            close_be3.append(jdx)

                            if n_BE == 4:
                                for kdx, k in enumerate(coord):
                                    if kdx == jdx:
                                        continue
                                    if (
                                        kdx not in unit1
                                        and kdx not in unit2
                                        and not cell.atom_pure_symbol(kdx) == "H"
                                    ):
                                        dist = norm(coord[jdx] - k)
                                        if dist <= bond:
                                            main_list.append(kdx)
                                            sub_list.append(kdx)
    return closest, close_be3


def surround(
    cell,
    sidx,
    unit1,
    unit2,
    flist,
    coord,
    n_BE: int,
    ext_list,
    klist,
    NK,
    rlist=[],
    bond=1.8 * 1.88973,
):
    n_BE_ = be_reduce(n_BE)
    if not rlist == [] and n_BE_ == 3:
        n_BE_ = 2
    sublist_ = []  # type: ignore[var-annotated]
    if not n_BE_ == 0:
        sidefunc(
            cell,
            sidx,
            unit1,
            unit2,
            flist,
            sublist_,
            coord,
            n_BE_,
            bond=bond,
            klist=klist,
            ext_list=ext_list,
            NK=NK,
            rlist=rlist,
        )
        sublist = [tmpi for tmpi in sublist_ if tmpi not in rlist]
        sublist = []
        for tmpi in sublist_:
            if tmpi not in rlist:
                for tmp_jdx, tmpj in enumerate(ext_list):
                    if tmpj == tmpi and klist[tmp_jdx] == NK:
                        break
                else:
                    sublist.append(tmpi)

        ext_list.extend(sublist)
        for kdx in sublist:
            klist.append(NK)


def kfrag_func(
    site_list,
    numk,
    nk1,
    uNs,
    Ns,
    nk2=None,
):
    if nk2 is None:
        nk2 = nk1
    frglist = []

    for pq in site_list:
        if numk > nk1 * 2:
            if not uNs == Ns:
                nk_ = numk - (nk1 * 2)
                frglist.append(uNs * nk1 * 2 + (nk_ * uNs) + pq)
            else:
                frglist.append(uNs * numk + pq)
        elif numk == nk1:
            frglist.append(uNs * numk + pq)

        elif numk > nk1:
            if uNs == Ns:
                frglist.append(uNs * numk + pq)
            else:
                nk_ = numk - nk1
                frglist.append(uNs * nk1 + (nk_ * uNs) + pq)
        else:
            frglist.append(uNs * numk + pq)
    return frglist


def be_reduce(n_BE):
    if n_BE == 2:
        return 0
    elif 3 <= n_BE <= 4:
        return n_BE - 1
    else:
        raise ValueError("Should not be here")


def autogen(
    mol,
    kpt,
    frozen_core=True,
    n_BE=2,
    write_geom=False,
    unitcell=1,
    gamma_2d=False,
    gamma_1d=False,
    long_bond=False,
    perpend_dist=4.0,
    perpend_dist_tol=1e-3,
    nx=False,
    ny=False,
    nz=False,
    iao_valence_basis=None,
    interlayer=False,
    print_frags=True,
):
    """
    Automatic cell partitioning

    Partitions a unitcell into overlapping fragments as defined in
    BE atom-based fragmentations.  It automatically detects branched chemical chains
    and ring systems and partitions accordingly.
    For efficiency, it only checks two atoms for connectivity (chemical bond)
    if they are within 3.5 Angstrom.  This value is hardcoded as normdist.
    Two atoms are defined as bonded if they are within 1.8 Angstrom
    (1.2 for Hydrogen atom).
    This is also hardcoded as bond & hbond. Neighboring unitcells are used in the
    fragmentation, exploiting translational symmetry.

    Parameters
    ----------
    mol : pyscf.pbc.gto.cell.Cell
        pyscf.pbc.gto.cell.Cell object. This is required for the options, 'autogen',
        and 'chain' as frag_type.
    kpt : list of int
        Number of k-points in each lattice vector dimension.
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
    interlayer : bool
        Whether the periodic system has two stacked monolayers.
    long_bond : bool
        For systems with longer than 1.8 Angstrom covalent bond,
        set this to True otherwise the fragmentation might fail.
    """
    if not float(unitcell).is_integer():
        raise ValueError("Fractional unitcell is not supported!")
    elif unitcell > 1:
        if not nx and not ny and not nz:
            nx_ = unitcell if kpt[0] > 1 else 1
            ny_ = unitcell if kpt[1] > 1 else 1
            nz_ = unitcell if kpt[2] > 1 else 1
        else:
            nx_ = unitcell if nx else 1
            ny_ = unitcell if ny else 1
            nz_ = unitcell if nz else 1
        kmesh = [nx_, ny_, nz_]

        cell = sgeom(mol, kmesh=kmesh)
        cell.build()

        print(flush=True)
        print(
            f"No. of cells used in building fragments : {unitcell:>3}",
            flush=True,
        )

        for idx, i in enumerate(kpt):
            if not i == 1:
                n_ = i / float(unitcell)
                if n_ > kpt[idx]:
                    raise ValueError(
                        "Use a larger number of k-points; "
                        "Fragment cell larger than all k-points combined "
                        "is not supported"
                    )
    else:
        cell = mol.copy()

    _, _, core_list = get_core(cell)

    coord = cell.atom_coords()

    ang2bohr = 1.88973
    normdist = 3.5 * ang2bohr
    bond = 1.8 * ang2bohr
    if long_bond:
        normdist = 6.0 * ang2bohr
        bond = 2.6 * ang2bohr
    hbond = 1.2 * ang2bohr
    lunit = []
    runit = []
    uunit = []
    dunit = []
    munit = []
    tunit = []

    lnext = [i for i in kpt if i > 1]

    if not len(lnext) == 0:
        nk1 = lnext[0]
        twoD = False
        if len(lnext) > 1:
            nk2 = lnext[1]
            twoD = True
        lkpt = [2 if i > 1 else 1 for i in kpt]
    else:
        if gamma_1d:
            nk1 = 1
            nk2 = 1
            lkpt = [1, 1, 2]
            twoD = False
        else:
            nk1 = 1
            nk2 = 1
            twoD = True
            lkpt = [2, 2, 1]

    if frozen_core:
        ncore__, _, _ = get_core(mol)
        Nsite = mol.aoslice_by_atom()[-1][3] - ncore__ - 1
    else:
        Nsite = mol.aoslice_by_atom()[-1][3] - 1
    # kmesh is lkpt
    # Building neighbouring cells

    if twoD:
        lnk2 = 2
    else:
        lnk2 = 1

    lattice_vector = cell.lattice_vectors()
    Ts = lib.cartesian_prod((arange(lkpt[0]), arange(lkpt[1]), arange(lkpt[2])))

    Ls = Ts @ lattice_vector

    # 1-2-(1-2)-1-2
    #   *       *

    lcoord = Ls.reshape(-1, 1, 3)[lnk2] * -1 + coord
    rcoord = Ls.reshape(-1, 1, 3)[lnk2] + coord

    lunit, runit = nearestof2coord(coord, lcoord, bond=bond)
    lunit_, runit_ = nearestof2coord(coord, rcoord, bond=bond)

    if not set(lunit) == set(runit_) or not set(runit) == set(lunit_):
        raise ValueError("Fragmentation error : wrong connection of unit cells ")

    if sum(i > 1 for i in kpt) > 1 or gamma_2d:
        # only 2D is supported
        # neighbours up-down or right-left

        #  *1-2
        #   |
        #  (1-2)
        #   |
        #  *1-2
        lcoord2 = Ls.reshape(-1, 1, 3)[1] * -1 + coord
        rcoord2 = Ls.reshape(-1, 1, 3)[1] + coord

        dunit, uunit = nearestof2coord(coord, lcoord2, bond=bond)
        dunit_, uunit_ = nearestof2coord(coord, rcoord2, bond=bond)

        if not set(uunit) == set(dunit_) or not set(dunit) == set(uunit_):
            raise ValueError("Fragmentation error : wrong connection of unit cells ")

        # diagonal
        # 1-2    1-2*
        #    \   /
        #    (1-2)
        #    /   \
        # 1-2*     1-2
        lcoord3 = Ls.reshape(-1, 1, 3)[lnk2 + 1] * -1 + coord
        rcoord3 = Ls.reshape(-1, 1, 3)[lnk2 + 1] + coord

        munit, tunit = nearestof2coord(coord, lcoord3, bond=bond)
        munit_, tunit_ = nearestof2coord(coord, rcoord3, bond=bond)

        if not set(munit) == set(tunit_) or not set(tunit) == set(munit_):
            raise ValueError("Fragmentation error : wrong connection of unit cells ")
        # kmesh lkpt ends

    # starts here
    normlist = []
    for i in coord:
        normlist.append(norm(i))
    Frag = []
    pedge = []
    cen = []
    lsites = []
    rsites = []
    usites = []
    dsites = []
    msites = []
    tsites = []

    klsites = []
    krsites = []
    kusites = []
    kdsites = []
    kmsites = []
    ktsites = []

    lunit = asarray(lunit)
    runit = asarray(runit)
    uunit = asarray(uunit)
    dunit = asarray(dunit)
    munit = asarray(munit)
    tunit = asarray(tunit)

    inter_dist = 1000.0
    if twoD and interlayer:
        inter_layer_axis = 2
        inter_layer_dict = []
        for aidx, ai in enumerate(coord):
            inter_dist = 1000
            for ajdx, aj in enumerate(coord):
                if aidx == ajdx:
                    continue
                if ai[inter_layer_axis] == aj[inter_layer_axis]:
                    continue
                dist = norm(ai - aj)
                if dist > bond:
                    if inter_dist > dist:
                        inter_dist = dist
            inter_dist_ = []
            inter_idx_ = []
            for ajdx, aj in enumerate(coord):
                if aidx == ajdx:
                    continue
                if ai[inter_layer_axis] == aj[inter_layer_axis]:
                    continue
                dist = norm(ai - aj)
                if abs(dist - inter_dist) < perpend_dist_tol:
                    inter_dist_.append(dist)
                    inter_idx_.append(ajdx)
            inter_layer_dict.append([inter_idx_, inter_dist_])

    # Assumes - minimum atom in a ring is 5
    if n_BE == 4:
        if twoD:
            warn("*********************\n"
                 "USE BE4  WITH CAUTION\n"
                 "*********************")  # fmt: skip

    for idx, i in enumerate(normlist):
        if cell.atom_pure_symbol(idx) == "H":
            continue

        lsts = []
        rsts = []
        usts = []
        dsts = []
        msts = []
        tsts = []

        klsts = []
        krsts = []
        kusts = []
        kdsts = []
        kmsts = []
        ktsts = []

        tmplist = normlist - i
        tmplist = list(tmplist)

        clist = []
        for jdx, j in enumerate(tmplist):
            if not idx == jdx and not cell.atom_pure_symbol(jdx) == "H":
                if abs(j) < normdist:
                    clist.append(jdx)

        pedg = []
        flist = []

        clist_check = []  # contains atoms from fragment that are
        # part of l,r,d,u,m,t but are in the
        # unit cell that is their k is 1
        # for example, klsts[i] = 1
        # WARNING !!!
        # For systems like Graphene, BN, SiC, hexagonal 2D sheets,
        # BE4 can give wrong fragmentations
        # the following code adds in redundant atoms

        if idx in lunit:
            closest, close_be3 = sidefunc(
                cell, idx, lunit, runit, flist, lsts, coord, n_BE, bond=bond
            )
            for zdx in lsts:
                if twoD:
                    klsts.append(nk1 * (nk2 - 1) + 1)
                else:
                    klsts.append(nk1)

            for lsidx in closest:
                if lsidx in lunit or lsidx in munit:
                    warn_large_fragment()
                if lsidx in runit:
                    surround(
                        cell,
                        lsidx,
                        runit,
                        lunit,
                        flist,
                        coord,
                        n_BE,
                        lsts,
                        klsts,
                        1,
                        rlist=[idx] + clist_check,
                        bond=bond,
                    )  # 1
                if lsidx in uunit:
                    surround(
                        cell,
                        lsidx,
                        uunit,
                        dunit,
                        flist,
                        coord,
                        n_BE,
                        lsts,
                        klsts,
                        nk1 * (nk2 - 1) + 2,
                        bond=bond,
                    )
                if lsidx in dunit:
                    surround(
                        cell,
                        lsidx,
                        dunit,
                        uunit,
                        flist,
                        coord,
                        n_BE,
                        lsts,
                        klsts,
                        nk1 * nk2,
                        bond=bond,
                    )
                if lsidx in tunit:
                    surround(
                        cell,
                        lsidx,
                        tunit,
                        munit,
                        flist,
                        coord,
                        n_BE,
                        lsts,
                        klsts,
                        2,
                        bond=bond,
                    )

            if n_BE == 4:
                for lsidx in close_be3:
                    if lsidx in lunit or lsidx in munit:
                        warn_large_fragment()
                    if lsidx in runit:
                        surround(
                            cell,
                            lsidx,
                            runit,
                            lunit,
                            flist,
                            coord,
                            3,
                            lsts,
                            klsts,
                            1,
                            rlist=[idx],
                            bond=bond,
                        )  # 1
                    if lsidx in uunit:
                        surround(
                            cell,
                            lsidx,
                            uunit,
                            dunit,
                            flist,
                            coord,
                            3,
                            lsts,
                            klsts,
                            nk1 * (nk2 - 1) + 2,
                            bond=bond,
                        )
                    if lsidx in dunit:
                        surround(
                            cell,
                            lsidx,
                            dunit,
                            uunit,
                            flist,
                            coord,
                            3,
                            lsts,
                            klsts,
                            nk1 * nk2,
                            bond=bond,
                        )
                    if lsidx in tunit:
                        surround(
                            cell,
                            lsidx,
                            tunit,
                            munit,
                            flist,
                            coord,
                            3,
                            lsts,
                            klsts,
                            2,
                            bond=bond,
                        )
            for i1dx, i1 in enumerate(klsts):
                if i1 == 1:
                    clist_check.append(lsts[i1dx])

        if idx in uunit:
            closest, close_be3 = sidefunc(
                cell, idx, uunit, dunit, flist, usts, coord, n_BE, bond=bond
            )
            for kdx in usts:
                kusts.append(2)
            for usidx in closest:
                if usidx in uunit or usidx in tunit:
                    warn_large_fragment()
                if usidx in lunit:
                    surround(
                        cell,
                        usidx,
                        lunit,
                        runit,
                        flist,
                        coord,
                        n_BE,
                        usts,
                        kusts,
                        nk1 * (nk2 - 1) + 2,
                        bond=bond,
                    )
                if usidx in runit:
                    surround(
                        cell,
                        usidx,
                        runit,
                        lunit,
                        flist,
                        coord,
                        n_BE,
                        usts,
                        kusts,
                        nk1 + 2,
                        bond=bond,
                    )
                if usidx in munit:
                    surround(
                        cell,
                        usidx,
                        munit,
                        tunit,
                        flist,
                        coord,
                        n_BE,
                        usts,
                        kusts,
                        nk1 * (nk2 - 1) + 1,
                        bond=bond,
                    )
                if usidx in dunit:
                    surround(
                        cell,
                        usidx,
                        dunit,
                        uunit,
                        flist,
                        coord,
                        n_BE,
                        usts,
                        kusts,
                        1,
                        rlist=[idx] + clist_check,
                        bond=bond,
                    )
            if n_BE == 4:
                for usidx in close_be3:
                    if usidx in uunit or usidx in tunit:
                        warn_large_fragment()
                    if usidx in lunit:
                        surround(
                            cell,
                            usidx,
                            lunit,
                            runit,
                            flist,
                            coord,
                            3,
                            usts,
                            kusts,
                            nk1 * (nk2 - 1) + 2,
                            bond=bond,
                        )
                    if usidx in runit:
                        surround(
                            cell,
                            usidx,
                            runit,
                            lunit,
                            flist,
                            coord,
                            3,
                            usts,
                            kusts,
                            nk1 + 2,
                            bond=bond,
                        )
                    if usidx in munit:
                        surround(
                            cell,
                            usidx,
                            munit,
                            tunit,
                            flist,
                            coord,
                            3,
                            usts,
                            kusts,
                            nk1 * (nk2 - 1) + 1,
                            bond=bond,
                        )
                    if usidx in dunit:
                        surround(
                            cell,
                            usidx,
                            dunit,
                            uunit,
                            flist,
                            coord,
                            3,
                            usts,
                            kusts,
                            1,
                            rlist=[idx],
                            bond=bond,
                        )
            for i1dx, i1 in enumerate(kusts):
                if i1 == 1:
                    clist_check.append(usts[i1dx])

        if idx in munit:
            closest, close_be3 = sidefunc(
                cell, idx, munit, tunit, flist, msts, coord, n_BE, bond=bond
            )
            for kdx in msts:
                kmsts.append(nk1 * nk2)
            for msidx in closest:
                if msidx in lunit or msidx in munit or msidx in dunit:
                    warn_large_fragment()
                if msidx in runit:
                    surround(
                        cell,
                        msidx,
                        runit,
                        lunit,
                        flist,
                        coord,
                        n_BE,
                        msts,
                        kmsts,
                        nk1,
                        bond=bond,
                    )
                if msidx in uunit:
                    surround(
                        cell,
                        msidx,
                        uunit,
                        dunit,
                        flist,
                        coord,
                        n_BE,
                        msts,
                        kmsts,
                        nk1 * (nk2 - 1) + 1,
                        bond=bond,
                    )
                if msidx in tunit:
                    surround(
                        cell,
                        msidx,
                        tunit,
                        munit,
                        flist,
                        coord,
                        n_BE,
                        msts,
                        kmsts,
                        1,
                        rlist=[idx] + clist_check,
                        bond=bond,
                    )
            if n_BE == 4:
                for msidx in close_be3:
                    if msidx in lunit or msidx in munit or msidx in dunit:
                        warn_large_fragment()
                    if msidx in runit:
                        surround(
                            cell,
                            msidx,
                            runit,
                            lunit,
                            flist,
                            coord,
                            3,
                            msts,
                            kmsts,
                            nk1,
                            bond=bond,
                        )
                    if msidx in uunit:
                        surround(
                            cell,
                            msidx,
                            uunit,
                            dunit,
                            flist,
                            coord,
                            3,
                            msts,
                            kmsts,
                            nk1 * (nk2 - 1) + 1,
                            bond=bond,
                        )
                    if msidx in tunit:
                        surround(
                            cell,
                            msidx,
                            tunit,
                            munit,
                            flist,
                            coord,
                            3,
                            msts,
                            kmsts,
                            1,
                            rlist=[idx],
                            bond=bond,
                        )
            for i1dx, i1 in enumerate(kmsts):
                if i1 == 1:
                    clist_check.append(msts[i1dx])

        if idx in runit:
            closest, close_be3 = sidefunc(
                cell, idx, runit, lunit, flist, rsts, coord, n_BE, bond=bond
            )
            for kdx in rsts:
                if twoD:
                    krsts.append(nk1 + 1)
                else:
                    krsts.append(2)
            for rsidx in closest:
                if rsidx in runit or rsidx in tunit:
                    warn_large_fragment()
                if rsidx in lunit:
                    surround(
                        cell,
                        rsidx,
                        lunit,
                        runit,
                        flist,
                        coord,
                        n_BE,
                        rsts,
                        krsts,
                        1,
                        rlist=[idx] + clist_check,
                        bond=bond,
                    )  # 1
                if rsidx in munit:
                    surround(
                        cell,
                        rsidx,
                        munit,
                        tunit,
                        flist,
                        coord,
                        n_BE,
                        rsts,
                        krsts,
                        nk1,
                        bond=bond,
                    )
                if rsidx in uunit:
                    surround(
                        cell,
                        rsidx,
                        uunit,
                        dunit,
                        flist,
                        coord,
                        n_BE,
                        rsts,
                        krsts,
                        nk1 + 2,
                        bond=bond,
                    )
                if rsidx in dunit:
                    surround(
                        cell,
                        rsidx,
                        dunit,
                        uunit,
                        flist,
                        coord,
                        n_BE,
                        rsts,
                        krsts,
                        nk1 * 2,
                        bond=bond,
                    )
            if n_BE == 4:
                for rsidx in close_be3:
                    if rsidx in runit or rsidx in tunit:
                        warn_large_fragment()
                    if rsidx in lunit:
                        surround(
                            cell,
                            rsidx,
                            lunit,
                            runit,
                            flist,
                            coord,
                            3,
                            rsts,
                            krsts,
                            1,
                            rlist=[idx],
                            bond=bond,
                        )  # 1
                    if rsidx in munit:
                        surround(
                            cell,
                            rsidx,
                            munit,
                            tunit,
                            flist,
                            coord,
                            3,
                            rsts,
                            krsts,
                            nk1,
                            bond=bond,
                        )
                    if rsidx in uunit:
                        surround(
                            cell,
                            rsidx,
                            uunit,
                            dunit,
                            flist,
                            coord,
                            3,
                            rsts,
                            krsts,
                            nk1 + 2,
                            bond=bond,
                        )
                    if rsidx in dunit:
                        surround(
                            cell,
                            rsidx,
                            dunit,
                            uunit,
                            flist,
                            coord,
                            3,
                            rsts,
                            krsts,
                            nk1 * 2,
                            bond=bond,
                        )
            for i1dx, i1 in enumerate(krsts):
                if i1 == 1:
                    clist_check.append(rsts[i1dx])
        if idx in dunit:
            closest, close_be3 = sidefunc(
                cell, idx, dunit, uunit, flist, dsts, coord, n_BE, bond=bond
            )
            for kdx in dsts:
                kdsts.append(nk1)
            for dsidx in closest:
                if dsidx in munit or dsidx in dunit:
                    warn_large_fragment()
                if dsidx in lunit:
                    surround(
                        cell,
                        dsidx,
                        lunit,
                        runit,
                        flist,
                        coord,
                        n_BE,
                        dsts,
                        kdsts,
                        nk1 * nk2,
                        bond=bond,
                    )
                if dsidx in runit:
                    surround(
                        cell,
                        dsidx,
                        runit,
                        lunit,
                        flist,
                        coord,
                        n_BE,
                        dsts,
                        kdsts,
                        nk1 * 2,
                        bond=bond,
                    )
                if dsidx in uunit:
                    surround(
                        cell,
                        dsidx,
                        uunit,
                        dunit,
                        flist,
                        coord,
                        n_BE,
                        dsts,
                        kdsts,
                        1,
                        rlist=[idx] + clist_check,
                        bond=bond,
                    )
                if dsidx in tunit:
                    surround(
                        cell,
                        dsidx,
                        tunit,
                        munit,
                        flist,
                        coord,
                        n_BE,
                        dsts,
                        kdsts,
                        nk1 + 1,
                        bond=bond,
                    )
            if n_BE == 4:
                for dsidx in close_be3:
                    if dsidx in munit or dsidx in dunit:
                        warn_large_fragment()
                    if dsidx in lunit:
                        surround(
                            cell,
                            dsidx,
                            lunit,
                            runit,
                            flist,
                            coord,
                            3,
                            dsts,
                            kdsts,
                            nk1 * nk2,
                            bond=bond,
                        )
                    if dsidx in runit:
                        surround(
                            cell,
                            dsidx,
                            runit,
                            lunit,
                            flist,
                            coord,
                            3,
                            dsts,
                            kdsts,
                            nk1 * 2,
                            bond=bond,
                        )
                    if dsidx in uunit:
                        surround(
                            cell,
                            dsidx,
                            uunit,
                            dunit,
                            flist,
                            coord,
                            3,
                            dsts,
                            kdsts,
                            1,
                            rlist=[idx],
                            bond=bond,
                        )
                    if dsidx in tunit:
                        surround(
                            cell,
                            dsidx,
                            tunit,
                            munit,
                            flist,
                            coord,
                            3,
                            dsts,
                            kdsts,
                            nk1 + 1,
                            bond=bond,
                        )
            for i1dx, i1 in enumerate(kdsts):
                if i1 == 1:
                    clist_check.append(dsts[i1dx])

        if idx in tunit:
            closest, close_be3 = sidefunc(
                cell, idx, tunit, munit, flist, tsts, coord, n_BE, bond=bond
            )
            for kdx in tsts:
                ktsts.append(nk1 + 2)
            for tsidx in closest:
                if tsidx in runit or tsidx in uunit or tsidx in tunit:
                    warn_large_fragment()
                if tsidx in lunit:
                    surround(
                        cell,
                        tsidx,
                        lunit,
                        runit,
                        flist,
                        coord,
                        n_BE,
                        tsts,
                        ktsts,
                        2,
                        bond=bond,
                    )
                if tsidx in munit:
                    surround(
                        cell,
                        tsidx,
                        munit,
                        tunit,
                        flist,
                        coord,
                        n_BE,
                        tsts,
                        ktsts,
                        1,
                        rlist=[idx] + clist_check,
                        bond=bond,
                    )
                if tsidx in dunit:
                    surround(
                        cell,
                        tsidx,
                        dunit,
                        uunit,
                        flist,
                        coord,
                        n_BE,
                        tsts,
                        ktsts,
                        nk1 + 1,
                        bond=bond,
                    )
            if n_BE == 4:
                for tsidx in close_be3:
                    if tsidx in runit or tsidx in uunit or tsidx in tunit:
                        warn_large_fragment()
                    if tsidx in lunit:
                        surround(
                            cell,
                            tsidx,
                            lunit,
                            runit,
                            flist,
                            coord,
                            3,
                            tsts,
                            ktsts,
                            2,
                            bond=bond,
                        )
                    if tsidx in munit:
                        surround(
                            cell,
                            tsidx,
                            munit,
                            tunit,
                            flist,
                            coord,
                            3,
                            tsts,
                            ktsts,
                            1,
                            rlist=[idx],
                            bond=bond,
                        )
                    if tsidx in dunit:
                        surround(
                            cell,
                            tsidx,
                            dunit,
                            uunit,
                            flist,
                            coord,
                            3,
                            tsts,
                            ktsts,
                            nk1 + 1,
                            bond=bond,
                        )
            for i1dx, i1 in enumerate(ktsts):
                if i1 == 1:
                    clist_check.append(tsts[i1dx])

        flist.append(idx)
        cen.append(idx)

        for jdx in clist:
            dist = norm(coord[idx] - coord[jdx])

            if (dist <= bond) or (
                interlayer
                and dist in inter_layer_dict[idx][1]
                and jdx in inter_layer_dict[idx][0]
                and dist < perpend_dist * ang2bohr
                and jdx not in pedg
            ):
                if jdx not in clist_check:
                    flist.append(jdx)
                    pedg.append(jdx)
                if dist > bond:
                    continue
                if 3 <= n_BE <= 4:
                    if jdx in lunit:
                        lmin1 = runit[where(lunit == jdx)[0]]
                        if not twoD:
                            flist.extend(lmin1)
                            lsts.extend(lmin1)
                            for kdx in lmin1:
                                if twoD:
                                    klsts.append(nk1 * (nk2 - 1) + 1)
                                else:
                                    klsts.append(nk1)
                        else:
                            add_check_k(lmin1, flist, lsts, klsts, nk1 * (nk2 - 1) + 1)
                        if n_BE == 4:
                            for kdx, k in enumerate(coord):
                                if (
                                    kdx == jdx
                                    or kdx in lmin1
                                    or cell.atom_pure_symbol(kdx) == "H"
                                ):
                                    continue
                                dist = norm(coord[lmin1] - k)
                                if dist <= bond:
                                    if (
                                        kdx in lsts
                                        and klsts[lsts.index(kdx)]
                                        == nk1 * (nk2 - 1) + 1
                                        and twoD
                                    ):
                                        continue
                                    flist.append(kdx)
                                    lsts.append(kdx)
                                    if twoD:
                                        klsts.append(nk1 * (nk2 - 1) + 1)
                                    else:
                                        klsts.append(nk1)
                            for lsidx in lmin1:
                                if lsidx in lunit or lsidx in munit:
                                    warn_large_fragment()
                                if lsidx in uunit:
                                    surround(
                                        cell,
                                        lsidx,
                                        uunit,
                                        dunit,
                                        flist,
                                        coord,
                                        3,
                                        lsts,
                                        klsts,
                                        nk1 * (nk2 - 1) + 2,
                                        bond=bond,
                                    )
                                if lsidx in dunit:
                                    surround(
                                        cell,
                                        lsidx,
                                        dunit,
                                        uunit,
                                        flist,
                                        coord,
                                        3,
                                        lsts,
                                        klsts,
                                        nk1 * nk2,
                                        bond=bond,
                                    )
                                if lsidx in tunit:
                                    surround(
                                        cell,
                                        lsidx,
                                        tunit,
                                        munit,
                                        flist,
                                        coord,
                                        3,
                                        lsts,
                                        klsts,
                                        2,
                                        bond=bond,
                                    )
                    if jdx in runit:
                        rmin1 = lunit[where(runit == jdx)[0]]
                        if not twoD:
                            flist.extend(rmin1)
                            rsts.extend(rmin1)
                            for kdx in rmin1:
                                if twoD:
                                    krsts.append(nk1 + 1)
                                else:
                                    krsts.append(2)
                        else:
                            add_check_k(rmin1, flist, rsts, krsts, nk1 + 1)
                        if n_BE == 4:
                            for kdx, k in enumerate(coord):
                                if (
                                    kdx == jdx
                                    or kdx in rmin1
                                    or cell.atom_pure_symbol(kdx) == "H"
                                ):
                                    continue
                                dist = norm(coord[rmin1] - k)
                                if dist <= bond:
                                    if (
                                        kdx in rsts
                                        and krsts[rsts.index(kdx)] == nk1 + 1
                                        and twoD
                                    ):
                                        continue
                                    flist.append(kdx)
                                    rsts.append(kdx)
                                    if twoD:
                                        krsts.append(nk1 + 1)
                                    else:
                                        krsts.append(2)
                            for rsidx in rmin1:
                                if rsidx in runit or rsidx in tunit:
                                    warn_large_fragment()
                                if rsidx in munit:
                                    surround(
                                        cell,
                                        rsidx,
                                        munit,
                                        tunit,
                                        flist,
                                        coord,
                                        3,
                                        rsts,
                                        krsts,
                                        nk1,
                                        bond=bond,
                                    )
                                if rsidx in uunit:
                                    surround(
                                        cell,
                                        rsidx,
                                        uunit,
                                        dunit,
                                        flist,
                                        coord,
                                        3,
                                        rsts,
                                        krsts,
                                        nk1 + 2,
                                        bond=bond,
                                    )
                                if rsidx in dunit:
                                    surround(
                                        cell,
                                        rsidx,
                                        dunit,
                                        uunit,
                                        flist,
                                        coord,
                                        3,
                                        rsts,
                                        krsts,
                                        nk1 * 2,
                                        bond=bond,
                                    )

                    if jdx in uunit:
                        umin1 = dunit[where(uunit == jdx)[0]]
                        add_check_k(umin1, flist, usts, kusts, 2)
                        if n_BE == 4:
                            for kdx, k in enumerate(coord):
                                if (
                                    kdx == jdx
                                    or kdx in umin1
                                    or cell.atom_pure_symbol(kdx) == "H"
                                ):
                                    continue
                                dist = norm(coord[umin1] - k)
                                if dist <= bond:
                                    if (
                                        kdx in usts
                                        and kusts[usts.index(kdx)] == 2
                                        and twoD
                                    ):
                                        continue
                                    flist.append(kdx)
                                    usts.append(kdx)
                                    kusts.append(2)
                            for usidx in umin1:
                                if usidx in uunit or usidx in tunit:
                                    warn_large_fragment()
                                if usidx in lunit:
                                    surround(
                                        cell,
                                        usidx,
                                        lunit,
                                        runit,
                                        flist,
                                        coord,
                                        3,
                                        usts,
                                        kusts,
                                        nk1 * (nk2 - 1) + 2,
                                        bond=bond,
                                    )
                                if usidx in runit:
                                    surround(
                                        cell,
                                        usidx,
                                        runit,
                                        lunit,
                                        flist,
                                        coord,
                                        3,
                                        usts,
                                        kusts,
                                        nk1 + 2,
                                        bond=bond,
                                    )
                                if usidx in munit:
                                    surround(
                                        cell,
                                        usidx,
                                        munit,
                                        tunit,
                                        flist,
                                        coord,
                                        3,
                                        usts,
                                        kusts,
                                        nk1 * (nk2 - 1) + 1,
                                        bond=bond,
                                    )
                    if jdx in dunit:
                        dmin1 = uunit[where(dunit == jdx)[0]]
                        add_check_k(dmin1, flist, dsts, kdsts, nk1)

                        if n_BE == 4:
                            for kdx, k in enumerate(coord):
                                if (
                                    kdx == jdx
                                    or kdx in dmin1
                                    or cell.atom_pure_symbol(kdx) == "H"
                                ):
                                    continue
                                dist = norm(coord[dmin1] - k)
                                if dist <= bond:
                                    if (
                                        kdx in dsts
                                        and kdsts[dsts.index(kdx)] == nk1
                                        and twoD
                                    ):
                                        continue
                                    flist.append(kdx)
                                    dsts.append(kdx)
                                    kdsts.append(nk1)
                            for dsidx in dmin1:
                                if dsidx in munit or dsidx in dunit:
                                    warn_large_fragment()
                                if dsidx in lunit:
                                    surround(
                                        cell,
                                        dsidx,
                                        lunit,
                                        runit,
                                        flist,
                                        coord,
                                        3,
                                        dsts,
                                        kdsts,
                                        nk1 * nk2,
                                        bond=bond,
                                    )
                                if dsidx in runit:
                                    surround(
                                        cell,
                                        dsidx,
                                        runit,
                                        lunit,
                                        flist,
                                        coord,
                                        3,
                                        dsts,
                                        kdsts,
                                        nk1 * 2,
                                        bond=bond,
                                    )
                                if dsidx in tunit:
                                    surround(
                                        cell,
                                        dsidx,
                                        tunit,
                                        munit,
                                        flist,
                                        coord,
                                        3,
                                        dsts,
                                        kdsts,
                                        nk1 + 1,
                                        bond=bond,
                                    )
                    if jdx in munit:  #
                        mmin1 = tunit[where(munit == jdx)[0]]
                        add_check_k(mmin1, flist, msts, kmsts, nk1 * nk2)
                        if n_BE == 4:
                            for kdx, k in enumerate(coord):
                                if (
                                    kdx == jdx
                                    or kdx in mmin1
                                    or cell.atom_pure_symbol(kdx) == "H"
                                ):
                                    continue
                                dist = norm(coord[mmin1] - k)
                                if dist <= bond:
                                    if (
                                        kdx in msts
                                        and kmsts[msts.index(kdx)] == nk1 * nk2
                                        and twoD
                                    ):
                                        continue
                                    flist.append(kdx)
                                    msts.append(kdx)
                                    kmsts.append(nk1 * nk2)
                            for msidx in mmin1:
                                if msidx in lunit or msidx in munit or msidx in dunit:
                                    warn_large_fragment()
                                if msidx in runit:
                                    surround(
                                        cell,
                                        msidx,
                                        runit,
                                        lunit,
                                        flist,
                                        coord,
                                        3,
                                        msts,
                                        kmsts,
                                        nk1,
                                        bond=bond,
                                    )
                                if msidx in uunit:
                                    surround(
                                        cell,
                                        msidx,
                                        uunit,
                                        dunit,
                                        flist,
                                        coord,
                                        3,
                                        msts,
                                        kmsts,
                                        nk1 * (nk2 - 1) + 1,
                                        bond=bond,
                                    )

                    if jdx in tunit:
                        tmin1 = munit[where(tunit == jdx)[0]]
                        add_check_k(tmin1, flist, tsts, ktsts, nk1 + 2)

                        if n_BE == 4:
                            for kdx, k in enumerate(coord):
                                if (
                                    kdx == jdx
                                    or kdx in tmin1
                                    or cell.atom_pure_symbol(kdx) == "H"
                                ):
                                    continue
                                dist = norm(coord[tmin1] - k)
                                if dist <= bond:
                                    if (
                                        kdx in tsts
                                        and ktsts[tsts.index(kdx)] == nk1 + 2
                                        and twoD
                                    ):
                                        continue
                                    flist.append(kdx)
                                    tsts.append(kdx)
                                    ktsts.append(nk1 + 2)
                            for tsidx in tmin1:
                                if tsidx in runit or tsidx in uunit or tsidx in tunit:
                                    warn_large_fragment()
                                if tsidx in lunit:
                                    surround(
                                        cell,
                                        tsidx,
                                        lunit,
                                        runit,
                                        flist,
                                        coord,
                                        3,
                                        tsts,
                                        ktsts,
                                        2,
                                        bond=bond,
                                    )
                                if tsidx in dunit:
                                    surround(
                                        cell,
                                        tsidx,
                                        dunit,
                                        uunit,
                                        flist,
                                        coord,
                                        3,
                                        tsts,
                                        ktsts,
                                        nk1 + 1,
                                        bond=bond,
                                    )

                    for kdx in clist:
                        if not kdx == jdx:
                            dist = norm(coord[jdx] - coord[kdx])
                            if (dist <= bond) or (
                                interlayer
                                and dist in inter_layer_dict[jdx][1]
                                and kdx in inter_layer_dict[jdx][0]
                                and dist < perpend_dist * ang2bohr
                            ):
                                if kdx not in pedg and kdx not in clist_check:
                                    flist.append(kdx)
                                    pedg.append(kdx)
                                if n_BE == 4:
                                    if kdx in lunit:
                                        lmin1 = runit[where(lunit == kdx)[0]]
                                        for zdx in lmin1:
                                            if (
                                                zdx in lsts
                                                and klsts[lsts.index(zdx)]
                                                == nk1 * (nk2 - 1) + 1
                                                and twoD
                                            ):
                                                continue
                                            flist.append(zdx)
                                            lsts.append(zdx)
                                            if twoD:
                                                klsts.append(nk1 * (nk2 - 1) + 1)
                                            else:
                                                klsts.append(nk1)
                                    if kdx in runit:
                                        rmin1 = lunit[where(runit == kdx)[0]]
                                        for zdx in rmin1:
                                            if (
                                                zdx in rsts
                                                and krsts[rsts.index(zdx)] == nk1 + 1
                                                and twoD
                                            ):
                                                continue
                                            flist.append(zdx)
                                            rsts.append(zdx)
                                            if twoD:
                                                krsts.append(nk1 + 1)
                                            else:
                                                krsts.append(2)
                                    if kdx in uunit:
                                        umin1 = dunit[where(uunit == kdx)[0]]
                                        for zdx in umin1:
                                            if (
                                                zdx in usts
                                                and kusts[usts.index(zdx)] == 2
                                                and twoD
                                            ):
                                                continue
                                            flist.append(zdx)
                                            usts.append(zdx)
                                            kusts.append(2)
                                    if kdx in dunit:
                                        dmin1 = uunit[where(dunit == kdx)[0]]
                                        for zdx in dmin1:
                                            if (
                                                zdx in dsts
                                                and kdsts[dsts.index(zdx)] == nk1
                                                and twoD
                                            ):
                                                continue
                                            flist.append(zdx)
                                            dsts.append(zdx)
                                            kdsts.append(nk1)
                                    if kdx in munit:
                                        mmin1 = tunit[where(munit == kdx)[0]]
                                        for zdx in mmin1:
                                            if (
                                                zdx in msts
                                                and kmsts[msts.index(zdx)] == nk1 * nk2
                                                and twoD
                                            ):
                                                continue
                                            flist.append(zdx)
                                            msts.append(zdx)
                                            kmsts.append(nk1 * nk2)
                                    if kdx in tunit:
                                        tmin1 = munit[where(tunit == kdx)[0]]
                                        for zdx in tmin1:
                                            if (
                                                zdx in tsts
                                                and ktsts[tsts.index(zdx)] == nk1 + 2
                                                and twoD
                                            ):
                                                continue
                                            flist.append(zdx)
                                            tsts.append(zdx)
                                            ktsts.append(nk1 + 2)

                                    for ldx, l in enumerate(coord):
                                        if (
                                            ldx == kdx
                                            or ldx == jdx
                                            or cell.atom_pure_symbol(ldx) == "H"
                                            or ldx in pedg
                                        ):
                                            continue
                                        dist = norm(coord[kdx] - l)
                                        if dist <= bond:
                                            flist.append(ldx)
                                            pedg.append(ldx)

        lsites.append(lsts)
        rsites.append(rsts)
        usites.append(usts)
        dsites.append(dsts)
        msites.append(msts)
        tsites.append(tsts)
        klsites.append(klsts)
        krsites.append(krsts)
        kusites.append(kusts)
        kdsites.append(kdsts)
        kmsites.append(kmsts)
        ktsites.append(ktsts)

        Frag.append(flist)
        pedge.append(pedg)

    hlist = [[] for i in coord]
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
                    f" {cell.atom_pure_symbol(j) + str(j + 1):>5} ",
                    end=" ",
                    flush=True,
                )
                for k in hlist[j]:
                    print(
                        f" {cell.atom_pure_symbol(k) + str(k + 1):>5} ",
                        end=" ",
                        flush=True,
                    )
            print(flush=True)
        print("--------------------------", flush=True)
        print(" No. of fragments : ", len(Frag), flush=True)
        print("*H : Center H atoms (printed as Edges above.)", flush=True)
        print(flush=True)

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

    pao = iao_valence_basis is not None

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

    for adx in range(cell.natm):
        if not cell.atom_pure_symbol(adx) == "H":
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

    max_site = max([j for i in sites__ for j in i])
    if any(hsites):
        maxH = max([j for i in hsites for j in i])
        max_site = max(max_site, maxH)

    AO_per_frag = []
    AO_per_edge_per_frag = []
    relAO_per_edge_per_frag = []
    relAO_per_origin_per_frag = []
    edge = []

    nkcon = True
    conmax = not (gamma_2d or gamma_1d)
    Ns = Nsite + 1
    uNs = Ns * unitcell
    Sites = sites__.copy()

    for j_ in range(unitcell - 1):
        for idx, i in enumerate(Sites):
            if any(i >= unitcell * Ns + unitcell * Ns * j_):
                sites__[idx] = [
                    (nk1 * Ns)
                    + j
                    + (nk1 * Ns * j_ - unitcell * Ns * j_)
                    - (unitcell * Ns)
                    for j in i
                ]

    for idx, i in enumerate(Frag):
        ftmp = []
        ftmpe = []
        indix = 0
        edind = []
        edg = []
        for jdx_, jdx in enumerate(lsites[idx]):
            edg.append(jdx)
            if not conmax:
                frglist = sites__[jdx].copy()
                frglist.extend(hsites[jdx])
            elif nkcon:
                numk = klsites[idx][jdx_] - 1
                frglist = kfrag_func(sites__[jdx] + hsites[jdx], numk, nk1, uNs, Ns)
            else:
                frglist = [pq - max_site - 1 for pq in sites__[jdx]]
                frglist.extend([pq - max_site - 1 for pq in hsites[jdx]])
            ftmp.extend(frglist)
            ls = len(sites__[jdx]) + len(hsites[jdx])
            if not pao:
                if not conmax:
                    edglist = sites__[jdx].copy()
                    edglist.extend(hsites[jdx])
                elif nkcon:
                    edglist = kfrag_func(sites__[jdx] + hsites[jdx], numk, nk1, uNs, Ns)
                else:
                    edglist = [pq - max_site - 1 for pq in sites__[jdx]]
                    edglist.extend([pq - max_site - 1 for pq in hsites[jdx]])
                ftmpe.append(edglist)
                edind.append([pq for pq in range(indix, indix + ls)])
            else:
                if not conmax:
                    edglist = sites__[jdx].copy()[: nbas2[jdx]]
                    edglist.extend(hsites[jdx][: nbas2H[jdx]])
                elif nkcon:
                    edglist = kfrag_func(
                        sites__[jdx][: nbas2[jdx]] + hsites[jdx][: nbas2H[jdx]],
                        numk,
                        nk1,
                        uNs,
                        Ns,
                    )
                else:
                    edglist = [pq - max_site - 1 for pq in sites__[jdx]][: nbas2[jdx]]
                    edglist.extend(
                        [pq - max_site - 1 for pq in hsites[jdx]][: nbas2H[jdx]]
                    )

                ftmpe.append(edglist)
                ind__ = [indix + frglist.index(pq) for pq in edglist]
                edind.append(ind__)
            indix += ls

        for jdx_, jdx in enumerate(usites[idx]):
            edg.append(jdx)
            if not conmax:
                frglist = sites__[jdx].copy()
                frglist.extend(hsites[jdx])
            elif nkcon:
                numk = kusites[idx][jdx_] - 1
                frglist = kfrag_func(sites__[jdx] + hsites[jdx], numk, nk1, uNs, Ns)
            else:
                frglist = [pq - max_site - 1 for pq in sites__[jdx]]
                frglist.extend([pq - max_site - 1 for pq in hsites[jdx]])

            ftmp.extend(frglist)
            ls = len(sites__[jdx]) + len(hsites[jdx])
            if not pao:
                if not conmax:
                    edglist = sites__[jdx].copy()
                    edglist.extend(hsites[jdx])
                elif nkcon:
                    edglist = kfrag_func(sites__[jdx] + hsites[jdx], numk, nk1, uNs, Ns)
                else:
                    edglist = [pq - max_site - 1 for pq in sites__[jdx]]
                    edglist.extend([pq - max_site - 1 for pq in hsites[jdx]])

                ftmpe.append(edglist)
                edind.append([pq for pq in range(indix, indix + ls)])
            else:
                if not conmax:
                    edglist = sites__[jdx].copy()[: nbas2[jdx]]
                    edglist.extend(hsites[jdx][: nbas2H[jdx]])
                elif nkcon:
                    edglist = kfrag_func(
                        sites__[jdx][: nbas2[jdx]] + hsites[jdx][: nbas2H[jdx]],
                        numk,
                        nk1,
                        uNs,
                        Ns,
                    )
                else:
                    edglist = [pq - max_site - 1 for pq in sites__[jdx]][: nbas2[jdx]]
                    edglist.extend(
                        [pq - max_site - 1 for pq in hsites[jdx]][: nbas2H[jdx]]
                    )

                ftmpe.append(edglist)
                ind__ = [indix + frglist.index(pq) for pq in edglist]
                edind.append(ind__)
            indix += ls

        for jdx_, jdx in enumerate(msites[idx]):
            edg.append(jdx)
            if not conmax:
                frglist = sites__[jdx].copy()
                frglist.extend(hsites[jdx])
            elif nkcon:
                numk = kmsites[idx][jdx_] - 1
                frglist = kfrag_func(sites__[jdx] + hsites[jdx], numk, nk1, uNs, Ns)

            else:
                frglist = [pq - max_site - 1 for pq in sites__[jdx]]
                frglist.extend([pq - max_site - 1 for pq in hsites[jdx]])

            ftmp.extend(frglist)
            ls = len(sites__[jdx]) + len(hsites[jdx])
            if not pao:
                if not conmax:
                    edglist = sites__[jdx].copy()
                    edglist.extend(hsites[jdx])
                elif nkcon:
                    edglist = kfrag_func(sites__[jdx] + hsites[jdx], numk, nk1, uNs, Ns)
                else:
                    edglist = [pq - max_site - 1 for pq in sites__[jdx]]
                    edglist.extend([pq - max_site - 1 for pq in hsites[jdx]])
                ftmpe.append(edglist)
                edind.append([pq for pq in range(indix, indix + ls)])
            else:
                if not conmax:
                    edglist = sites__[jdx].copy()[: nbas2[jdx]]
                    edglist.extend(hsites[jdx][: nbas2H[jdx]])
                elif nkcon:
                    edglist = kfrag_func(
                        sites__[jdx][: nbas2[jdx]] + hsites[jdx][: nbas2H[jdx]],
                        numk,
                        nk1,
                        uNs,
                        Ns,
                    )
                else:
                    edglist = [pq - max_site - 1 for pq in sites__[jdx]][: nbas2[jdx]]
                    edglist.extend(
                        [pq - max_site - 1 for pq in hsites[jdx]][: nbas2H[jdx]]
                    )

                ftmpe.append(edglist)
                ind__ = [indix + frglist.index(pq) for pq in edglist]
                edind.append(ind__)
            indix += ls

        frglist = sites__[cen[idx]].copy()
        frglist.extend(hsites[cen[idx]])
        ftmp.extend(frglist)

        ls = len(sites__[cen[idx]]) + len(hsites[cen[idx]])
        if not pao:
            relAO_per_origin_per_frag.append([pq for pq in range(indix, indix + ls)])
        else:
            cntlist = sites__[cen[idx]].copy()[: nbas2[cen[idx]]]
            cntlist.extend(hsites[cen[idx]][: nbas2H[cen[idx]]])
            ind__ = [indix + frglist.index(pq) for pq in cntlist]
            relAO_per_origin_per_frag.append(ind__)
        indix += ls

        for jdx in pedge[idx]:
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

        for jdx_, jdx in enumerate(rsites[idx]):
            edg.append(jdx)
            if not conmax:
                frglist = sites__[jdx].copy()
                frglist.extend(hsites[jdx])
            elif nkcon:
                numk = krsites[idx][jdx_] - 1
                frglist = kfrag_func(sites__[jdx] + hsites[jdx], numk, nk1, uNs, Ns)
            else:
                frglist = [pq + max_site + 1 for pq in sites__[jdx]]
                frglist.extend([pq + max_site + 1 for pq in hsites[jdx]])
            ftmp.extend(frglist)

            ls = len(sites__[jdx]) + len(hsites[jdx])
            if not pao:
                if not conmax:
                    edglist = sites__[jdx].copy()
                    edglist.extend(hsites[jdx])
                elif nkcon:
                    edglist = kfrag_func(sites__[jdx] + hsites[jdx], numk, nk1, uNs, Ns)
                else:
                    edglist = [pq - max_site + 1 for pq in sites__[jdx]]
                    edglist.extend([pq - max_site + 1 for pq in hsites[jdx]])
                ftmpe.append(edglist)
                edind.append([pq for pq in range(indix, indix + ls)])
            else:
                if not conmax:
                    edglist = sites__[jdx].copy()[: nbas2[jdx]]
                    edglist.extend(hsites[jdx][: nbas2H[jdx]])
                elif nkcon:
                    edglist = kfrag_func(
                        sites__[jdx][: nbas2[jdx]] + hsites[jdx][: nbas2H[jdx]],
                        numk,
                        nk1,
                        uNs,
                        Ns,
                    )
                else:
                    edglist = [pq + max_site + 1 for pq in sites__[jdx]][: nbas2[jdx]]
                    edglist.extend(
                        [pq + max_site + 1 for pq in hsites[jdx]][: nbas2H[jdx]]
                    )

                ftmpe.append(edglist)
                ind__ = [indix + frglist.index(pq) for pq in edglist]
                edind.append(ind__)
            indix += ls

        for jdx_, jdx in enumerate(dsites[idx]):
            edg.append(jdx)
            if not conmax:
                frglist = sites__[jdx].copy()
                frglist.extend(hsites[jdx])
            elif nkcon:
                numk = kdsites[idx][jdx_] - 1
                frglist = kfrag_func(sites__[jdx] + hsites[jdx], numk, nk1, uNs, Ns)
            else:
                frglist = [pq + max_site + 1 for pq in sites__[jdx]]
                frglist.extend([pq + max_site + 1 for pq in hsites[jdx]])

            ftmp.extend(frglist)
            ls = len(sites__[jdx]) + len(hsites[jdx])
            if not pao:
                if not conmax:
                    edglist = sites__[jdx].copy()
                    edglist.extend(hsites[jdx])
                elif nkcon:
                    edglist = kfrag_func(sites__[jdx] + hsites[jdx], numk, nk1, uNs, Ns)
                else:
                    edglist = [pq - max_site + 1 for pq in sites__[jdx]]
                    edglist.extend([pq - max_site + 1 for pq in hsites[jdx]])

                ftmpe.append(edglist)
                edind.append([pq for pq in range(indix, indix + ls)])
            else:
                if not conmax:
                    edglist = sites__[jdx].copy()[: nbas2[jdx]]
                    edglist.extend(hsites[jdx][: nbas2H[jdx]])
                elif nkcon:
                    edglist = kfrag_func(
                        sites__[jdx][: nbas2[jdx]] + hsites[jdx][: nbas2H[jdx]],
                        numk,
                        nk1,
                        uNs,
                        Ns,
                    )
                else:
                    edglist = [pq + max_site + 1 for pq in sites__[jdx]][: nbas2[jdx]]
                    edglist.extend(
                        [pq + max_site + 1 for pq in hsites[jdx]][: nbas2H[jdx]]
                    )

                ftmpe.append(edglist)
                ind__ = [indix + frglist.index(pq) for pq in edglist]
                edind.append(ind__)
            indix += ls

        for jdx_, jdx in enumerate(tsites[idx]):
            edg.append(jdx)
            if not conmax:
                frglist = sites__[jdx].copy()
                frglist.extend(hsites[jdx])
            elif nkcon:
                numk = ktsites[idx][jdx_] - 1
                frglist = kfrag_func(sites__[jdx] + hsites[jdx], numk, nk1, uNs, Ns)
            else:
                frglist = [pq + max_site + 1 for pq in sites__[jdx]]
                frglist.extend([pq + max_site + 1 for pq in hsites[jdx]])

            ftmp.extend(frglist)
            ls = len(sites__[jdx]) + len(hsites[jdx])
            if not pao:
                if not conmax:
                    edglist = sites__[jdx].copy()
                    edglist.extend(hsites[jdx])
                elif nkcon:
                    edglist = kfrag_func(sites__[jdx] + hsites[jdx], numk, nk1, uNs, Ns)
                else:
                    edglist = [pq - max_site + 1 for pq in sites__[jdx]]
                    edglist.extend([pq - max_site + 1 for pq in hsites[jdx]])
                ftmpe.append(edglist)
                edind.append([pq for pq in range(indix, indix + ls)])
            else:
                if not conmax:
                    edglist = sites__[jdx].copy()[: nbas2[jdx]]
                    edglist.extend(hsites[jdx][: nbas2H[jdx]])
                elif nkcon:
                    edglist = kfrag_func(
                        sites__[jdx][: nbas2[jdx]] + hsites[jdx][: nbas2H[jdx]],
                        numk,
                        nk1,
                        uNs,
                        Ns,
                    )
                else:
                    edglist = [pq + max_site + 1 for pq in sites__[jdx]][: nbas2[jdx]]
                    edglist.extend(
                        [pq + max_site + 1 for pq in hsites[jdx]][: nbas2H[jdx]]
                    )

                ftmpe.append(edglist)
                ind__ = [indix + frglist.index(pq) for pq in edglist]
                edind.append(ind__)
            indix += ls

        edge.append(edg)
        AO_per_frag.append(ftmp)
        AO_per_edge_per_frag.append(ftmpe)
        relAO_per_edge_per_frag.append(edind)

    ref_frag_idx_per_edge_per_frag = []
    for ix in edge:
        cen_ = []
        for jx in ix:
            cen_.append(cen.index(jx))
        ref_frag_idx_per_edge_per_frag.append(cen_)

    n_frag = len(AO_per_frag)
    weight_and_relAO_per_center_per_frag = []
    # Use IAO+PAO for computing energy
    for ix, i in enumerate(AO_per_frag):
        tmp_ = [i.index(pq) for pq in sites__[cen[ix]]]
        tmp_.extend([i.index(pq) for pq in hsites[cen[ix]]])
        weight_and_relAO_per_center_per_frag.append([1.0, tmp_])

    relAO_in_ref_per_edge_per_frag = []
    for i in range(n_frag):
        idx = []
        for j in ref_frag_idx_per_edge_per_frag[i]:
            if not pao:
                cntlist = sites__[cen[j]].copy()
                cntlist.extend(hsites[cen[j]])
                idx.append([AO_per_frag[j].index(k) for k in cntlist])
            else:
                cntlist = sites__[cen[j]].copy()[: nbas2[cen[j]]]
                cntlist.extend(hsites[cen[j]][: nbas2H[cen[j]]])
                idx.append([AO_per_frag[j].index(k) for k in cntlist])

        relAO_in_ref_per_edge_per_frag.append(idx)

    if not AO_per_edge_per_frag:
        AO_per_edge_per_frag = [[] for _ in range(n_frag)]
        ref_frag_idx_per_edge_per_frag = [[] for _ in range(n_frag)]
        relAO_per_edge_per_frag = [[] for _ in range(n_frag)]
        relAO_in_ref_per_edge_per_frag = [[] for _ in range(n_frag)]

    return (
        AO_per_frag,
        AO_per_edge_per_frag,
        ref_frag_idx_per_edge_per_frag,
        relAO_per_edge_per_frag,
        relAO_in_ref_per_edge_per_frag,
        relAO_per_origin_per_frag,
        weight_and_relAO_per_center_per_frag,
    )
