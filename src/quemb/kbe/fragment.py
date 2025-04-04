# Author(s): Oinam Romesh Meitei

from typing import Literal

from attrs import define, field
from pyscf.pbc.gto.cell import Cell

from quemb.kbe.autofrag import autogen
from quemb.molbe.helper import get_core
from quemb.shared.typing import (
    FragmentIdx,
    GlobalAOIdx,
    ListOverEdge,
    ListOverFrag,
    OtherRelAOIdx,
    OwnRelAOIdx,
)


@define(kw_only=True)
class FragPart:
    unitcell: int
    mol: Cell
    frag_type: str
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

    #: The first element is a float, the second is the list
    #: The float weight makes only sense for democratic matching and is currently 1.0
    #: everywhere anyway. We concentrate only on the second part,
    #: i.e. the list of indices.
    #: This is a list whose entries are sequences containing the relative orbital index
    #  of the center sites within a fragment. Relative is to the own fragment.
    rel_AO_per_center_per_frag: ListOverFrag[tuple[float, list[OwnRelAOIdx]]]

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

    n_BE: int
    natom: int
    frozen_core: bool
    self_match: bool
    allcen: bool
    iao_valence_basis: str | None
    kpt: list[int] | tuple[int, int, int]

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


def fragmentate(
    mol: Cell,
    kpt: list[int] | tuple[int, int, int],
    *,
    natom: int = 0,
    frag_type: Literal["autogen"] = "autogen",
    unitcell: int = 1,
    gamma_2d: bool = False,
    gamma_1d: bool = False,
    interlayer: bool = False,
    long_bond: bool = False,
    perpend_dist: float = 4.0,
    perpend_dist_tol: float = 1e-3,
    nx: bool = False,
    ny: bool = False,
    nz: bool = False,
    iao_valence_basis: str | None = None,
    n_BE: int = 2,
    frozen_core: bool = False,
    self_match: bool = False,
    allcen: bool = True,
) -> FragPart:
    """Fragment/partitioning definition

    Interfaces the main fragmentation function (autogen) in MolBE.
    It defines edge & center for density matching and energy estimation.
    It also forms the base for IAO/PAO partitioning for
    a large basis set bootstrap calculation. Fragments are constructed based
    on atoms within a unitcell.

    Parameters
    ----------
    frag_type : str
        Name of fragmentation function. 'autogen', 'hchain_simple', and 'chain' are
        supported. Defaults to 'autogen'
        For systems with only hydrogen, use 'chain';
        everything else should use 'autogen'
    n_BE: int, optional
        Specifies the order of bootstrap calculation in the atom-based fragmentation,
        i.e. BE(n).
        For a simple linear system A-B-C-D,
        BE(1) only has fragments [A], [B], [C], [D]
        BE(2) has [A, B, C], [B, C, D]
        ben ...
    mol : pyscf.pbc.gto.cell.Cell
        pyscf.pbc.gto.cell.Cell object. This is required for the options, 'autogen',
        and 'chain' as frag_type.
    iao_valence_basis: str
        Name of minimal basis set for IAO scheme. 'sto-3g' suffice for most cases.
    frozen_core: bool
        Whether to invoke frozen core approximation. This is set to False by default
    print_frags: bool
        Whether to print out list of resulting fragments. True by default
    write_geom: bool
        Whether to write 'fragment.xyz' file which contains all the fragments in
        cartesian coordinates.
    kpt : list of int
        No. of k-points in each lattice vector direction. This is the same as kmesh.
    interlayer : bool
        Whether the periodic system has two stacked monolayers.
    long_bond : bool
        For systems with longer than 1.8 Angstrom covalent bond, set this to True
        otherwise the fragmentation might fail.
    """
    if frag_type == "autogen":
        if kpt is None:
            raise ValueError("Provide kpt mesh in fragmentate() and restart!")

        (
            AO_per_frag,
            AO_per_edge_per_frag,
            ref_frag_idx_per_edge,
            rel_AO_per_edge_per_frag,
            other_rel_AO_per_edge_per_frag,
            rel_AO_per_origin_per_frag,
            scale_rel_AO_per_center_per_frag,
        ) = autogen(
            mol,
            kpt,
            n_BE=n_BE,
            frozen_core=frozen_core,
            iao_valence_basis=iao_valence_basis,
            unitcell=unitcell,
            nx=nx,
            ny=ny,
            nz=nz,
            long_bond=long_bond,
            perpend_dist=perpend_dist,
            perpend_dist_tol=perpend_dist_tol,
            gamma_2d=gamma_2d,
            gamma_1d=gamma_1d,
            interlayer=interlayer,
        )

        return FragPart(
            unitcell=unitcell,
            mol=mol,
            frag_type=frag_type,
            AO_per_frag=AO_per_frag,
            AO_per_edge_per_frag=AO_per_edge_per_frag,
            ref_frag_idx_per_edge=ref_frag_idx_per_edge,
            rel_AO_per_center_per_frag=scale_rel_AO_per_center_per_frag,
            rel_AO_per_edge_per_frag=rel_AO_per_edge_per_frag,
            other_rel_AO_per_edge_per_frag=other_rel_AO_per_edge_per_frag,
            rel_AO_per_origin_per_frag=rel_AO_per_origin_per_frag,
            n_BE=n_BE,
            natom=natom,
            frozen_core=frozen_core,
            self_match=self_match,
            allcen=allcen,
            iao_valence_basis=iao_valence_basis,
            kpt=kpt,
        )

    else:
        raise ValueError(f"Fragmentation type = {frag_type} not implemented!")
