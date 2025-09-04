# Authors: Oinam Romesh Meitei, Shaun Weatherly, Oskar Weser

from collections.abc import Sequence
from typing import Generic, Literal, TypeAlias, TypeVar
from warnings import warn

from attrs import cmp_using, define, field
from numpy.linalg import norm
from pyscf.gto import Mole
from pyscf.pbc.gto import Cell
from typing_extensions import Self

from quemb.molbe.helper import are_equal, get_core
from quemb.molbe.pfrag import Frags
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
    PathLike,
    RelAOIdx,
    RelAOIdxInRef,
    T,
    Vector,
)

FragType: TypeAlias = Literal["chemgen", "graphgen", "autogen"]

_T_chemsystem = TypeVar("_T_chemsystem", Mole, Cell)


@define(kw_only=True)
class FragPart(Generic[_T_chemsystem]):
    """Data structure to hold the result of BE fragmentations."""

    #: The full molecule.
    mol: _T_chemsystem = field(eq=cmp_using(are_equal))
    #: The algorithm used for fragmenting.
    frag_type: FragType
    #: The level of BE fragmentation, i.e. 1, 2, ...
    n_BE: int

    #: This is a list over fragments  and gives the global orbital indices of all atoms
    #: in the fragment. These are ordered by the atoms in the fragment.
    #:
    #: When using IAOs this refers to the large/working basis.
    AO_per_frag: ListOverFrag[list[GlobalAOIdx]]

    #: The global orbital indices, including hydrogens, per edge per fragment.
    #:
    #: When using IAOs this refers to the valence/small basis.
    AO_per_edge_per_frag: ListOverFrag[ListOverEdge[list[GlobalAOIdx]]]

    #: Reference fragment index per edge:
    #: A list over fragments: list of indices of the fragments in which an edge
    #: of the fragment is actually a center.
    #: The edge will be matched against this center.
    #: For fragments A, B: the A’th element of :python:`.center`,
    #: if the edge of A is the center of B, will be B.
    ref_frag_idx_per_edge_per_frag: ListOverFrag[ListOverEdge[FragmentIdx]]

    #: The relative orbital indices, including hydrogens, per edge per fragment.
    #: The index is relative to the own fragment.
    #:
    #: When using IAOs this refers to the valence/small basis.
    relAO_per_edge_per_frag: ListOverFrag[ListOverEdge[list[RelAOIdx]]]

    #: The relative atomic orbital indices per edge per fragment.
    #: **Note** for this variable relative means that the AO indices
    #: are relative to the other fragment where the edge is a center.
    #:
    #: When using IAOs this refers to the valence/small basis.
    relAO_in_ref_per_edge_per_frag: ListOverFrag[ListOverEdge[list[RelAOIdxInRef]]]

    #: List whose entries are lists containing the relative orbital index of the
    #: origin site within a fragment. Relative is to the own fragment.
    #  Since the origin site is at the beginning
    #: of the motif list for each fragment, this is always a ``list(range(0, n))``
    #:
    #: When using IAOs this refers to the valence/small basis.
    relAO_per_origin_per_frag: ListOverFrag[list[RelAOIdx]]

    #: The first element is a float, the second is the list
    #: The float weight makes only sense for democratic matching and is currently 1.0
    #: everywhere anyway. We concentrate only on the second part,
    #: i.e. the list of indices.
    #: This is a list whose entries are sequences containing the relative orbital index
    #: of the center sites within a fragment. Relative is to the own fragment.
    #:
    #: When using IAOs this refers to the large/working basis.
    weight_and_relAO_per_center_per_frag: ListOverFrag[tuple[float, list[RelAOIdx]]]

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

    def all_centers_are_origins(self) -> bool:
        if self.iao_valence_basis:
            raise ValueError("Test is only defined if IAO is not used.")
            # This is because relAO_per_center uses the large basis
            # and relAO_per_origin the small/valence basis
        return all(
            relAO_per_center == relAO_per_origin
            for (_, relAO_per_center), relAO_per_origin in zip(
                self.weight_and_relAO_per_center_per_frag,
                self.relAO_per_origin_per_frag,
            )
        )

    def to_Frags(self, I: int, eri_file: PathLike, unrestricted: bool = False) -> Frags:
        return Frags(
            self.AO_per_frag[I],
            I,
            AO_per_edge=self.AO_per_edge_per_frag[I],
            eri_file=eri_file,
            ref_frag_idx_per_edge=self.ref_frag_idx_per_edge_per_frag[I],
            relAO_per_edge=self.relAO_per_edge_per_frag[I],
            relAO_in_ref_per_edge=self.relAO_in_ref_per_edge_per_frag[I],
            weight_and_relAO_per_center=self.weight_and_relAO_per_center_per_frag[I],
            relAO_per_origin=self.relAO_per_origin_per_frag[I],
            unrestricted=unrestricted,
        )

    def reindex(self, idx: Sequence[int] | Vector) -> Self:
        def _get_elements(seq: Sequence[T], idx: Sequence[int] | Vector) -> list[T]:
            return [seq[i] for i in idx]  # type: ignore[index]

        return self.__class__(
            mol=self.mol,
            frag_type=self.frag_type,
            n_BE=self.n_BE,
            AO_per_frag=_get_elements(self.AO_per_frag, idx),
            AO_per_edge_per_frag=_get_elements(self.AO_per_edge_per_frag, idx),
            ref_frag_idx_per_edge_per_frag=_get_elements(
                self.ref_frag_idx_per_edge_per_frag, idx
            ),
            relAO_per_edge_per_frag=_get_elements(self.relAO_per_edge_per_frag, idx),
            relAO_in_ref_per_edge_per_frag=_get_elements(
                self.relAO_in_ref_per_edge_per_frag, idx
            ),
            relAO_per_origin_per_frag=_get_elements(
                self.relAO_per_origin_per_frag, idx
            ),
            weight_and_relAO_per_center_per_frag=_get_elements(
                self.weight_and_relAO_per_center_per_frag, idx
            ),
            motifs_per_frag=_get_elements(self.motifs_per_frag, idx),
            origin_per_frag=_get_elements(self.origin_per_frag, idx),
            H_per_motif=self.H_per_motif,
            add_center_atom=_get_elements(self.add_center_atom, idx),
            frozen_core=self.frozen_core,
            iao_valence_basis=self.iao_valence_basis,
            iao_valence_only=self.iao_valence_only,
        )


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
    if n_BE > 4:
        raise ValueError("n_BE > 4 not supported, use 'chemgen' or 'graphgen' instead.")
    if n_BE < 1:
        raise ValueError("n_BE < 1 does not make sense.")

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
    AO_per_edge = []
    relAO_per_edge = []
    relAO_per_origin = []
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
            relAO_per_origin.append([pq for pq in range(indix, indix + ls_)])
        else:
            cntlist = sites__[origin_per_frag[idx]].copy()[
                : nbas2[origin_per_frag[idx]]
            ]
            cntlist.extend(hsites[origin_per_frag[idx]][: nbas2H[origin_per_frag[idx]]])
            ind__ = [indix + frglist.index(pq) for pq in cntlist]
            relAO_per_origin.append(ind__)
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
            AO_per_edge.append(ftmpe)
            relAO_per_edge.append(edind)
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
    weight_and_relAO_per_center = []

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
        weight_and_relAO_per_center.append((1.0, tmp_))

    relAO_in_ref_per_edge = []
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

            relAO_in_ref_per_edge.append(idx)

    if not AO_per_edge:
        AO_per_edge = [[] for _ in range(n_frag)]
        ref_frag_idx_per_edge = [[] for _ in range(n_frag)]
        relAO_per_edge = [[] for _ in range(n_frag)]
        relAO_in_ref_per_edge = [[] for _ in range(n_frag)]

    return FragPart(
        mol=mol,
        frag_type="autogen",
        n_BE=n_BE,
        AO_per_frag=AO_per_frag,
        AO_per_edge_per_frag=AO_per_edge,
        ref_frag_idx_per_edge_per_frag=ref_frag_idx_per_edge,
        relAO_per_edge_per_frag=relAO_per_edge,
        relAO_in_ref_per_edge_per_frag=relAO_in_ref_per_edge,
        relAO_per_origin_per_frag=relAO_per_origin,
        weight_and_relAO_per_center_per_frag=weight_and_relAO_per_center,
        motifs_per_frag=motifs_per_frag,
        origin_per_frag=origin_per_frag,
        H_per_motif=H_per_motif,
        add_center_atom=add_center_atom,
        frozen_core=frozen_core,
        iao_valence_basis=iao_valence_basis,
        iao_valence_only=iao_valence_only,
    )
