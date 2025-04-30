# Author(s): Oinam Romesh Meitei, Minsik Cho, Oskar Weser

from typing import Literal

from attrs import define, field
from pyscf.pbc.gto.cell import Cell

from quemb.kbe.autofrag import AutogenArgs, autogen
from quemb.kbe.pfrag import Frags
from quemb.molbe.chemfrag import ChemGenArgs, chemgen
from quemb.molbe.helper import get_core
from quemb.shared.typing import (
    FragmentIdx,
    GlobalAOIdx,
    ListOverEdge,
    ListOverFrag,
    PathLike,
    RelAOIdx,
    RelAOIdxInRef,
)


@define(kw_only=True)
class FragPart:
    unitcell: int
    mol: Cell
    frag_type: str
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

    #: The first element is a float, the second is the list
    #: The float weight makes only sense for democratic matching and is currently 1.0
    #: everywhere anyway. We concentrate only on the second part,
    #: i.e. the list of indices.
    #: This is a list whose entries are sequences containing the relative orbital index
    #: of the center sites within a fragment. Relative is to the own fragment.
    #:
    #: When using IAOs this refers to the large/working basis.
    weight_and_relAO_per_center_per_frag: ListOverFrag[tuple[float, list[RelAOIdx]]]

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

    def to_Frags(self, I: int, eri_file: PathLike, unitcell_nkpt: int) -> Frags:
        return Frags(
            AO_in_frag=self.AO_per_frag[I],
            ifrag=I,
            AO_per_edge=self.AO_per_edge_per_frag[I],
            eri_file=eri_file,
            ref_frag_idx_per_edge=self.ref_frag_idx_per_edge_per_frag[I],
            relAO_per_edge=self.relAO_per_edge_per_frag[I],
            relAO_in_ref_per_edge=self.relAO_in_ref_per_edge_per_frag[I],
            weight_and_relAO_per_center=self.weight_and_relAO_per_center_per_frag[I],
            relAO_per_origin=self.relAO_per_origin_per_frag[I],
            unitcell=self.unitcell,
            unitcell_nkpt=unitcell_nkpt,
        )


def fragmentate(
    mol: Cell,
    kpt: list[int] | tuple[int, int, int],
    *,
    natom: int = 0,
    frag_type: Literal["autogen"] = "autogen",
    unitcell: int = 1,
    iao_valence_basis: str | None = None,
    n_BE: int = 2,
    frozen_core: bool = False,
    self_match: bool = False,
    allcen: bool = True,
    print_frags: bool = True,
    additional_args: ChemGenArgs | AutogenArgs | None = None,
) -> FragPart:
    """Fragment/partitioning definition

    Interfaces the main fragmentation function (autogen) in MolBE.
    It defines edge & center for density matching and energy estimation.
    It also forms the base for IAO/PAO partitioning for
    a large basis set bootstrap calculation. Fragments are constructed based
    on atoms within a unitcell.

    Parameters
    ----------
    frag_type :
        Name of fragmentation function. 'autogen' and 'chemgen' are supported.
        Defaults to 'autogen'
    n_BE :
        Specifies the order of bootstrap calculation in the atom-based fragmentation,
        i.e. BE(n).
        For a simple linear system A-B-C-D,
        BE(1) only has fragments [A], [B], [C], [D]
        BE(2) has [A, B, C], [B, C, D]
    mol :
        pyscf.pbc.gto.cell.Cell object. This is required for the options, 'autogen',
        and 'chain' as frag_type.
    iao_valence_basis :
        Name of minimal basis set for IAO scheme. 'sto-3g' suffice for most cases.
    frozen_core :
        Whether to invoke frozen core approximation. This is set to False by default
    print_frags :
        Whether to print out list of resulting fragments. True by default
    kpt :
        No. of k-points in each lattice vector direction. This is the same as kmesh.
    additional_args :
        Additional arguments for different fragmentation functions.
    """
    if frag_type == "autogen":
        if additional_args is None:
            additional_args = AutogenArgs()
        else:
            assert isinstance(additional_args, AutogenArgs)

        (
            AO_per_frag,
            AO_per_edge_per_frag,
            ref_frag_idx_per_edge_per_frag,
            relAO_per_edge_per_frag,
            relAO_in_ref_per_edge_per_frag,
            relAO_per_origin_per_frag,
            weight_and_relAO_per_center_per_frag,
        ) = autogen(
            mol,
            kpt,
            n_BE=n_BE,
            frozen_core=frozen_core,
            iao_valence_basis=iao_valence_basis,
            unitcell=unitcell,
            nx=additional_args.nx,
            ny=additional_args.ny,
            nz=additional_args.nz,
            long_bond=additional_args.long_bond,
            perpend_dist=additional_args.perpend_dist,
            perpend_dist_tol=additional_args.perpend_dist_tol,
            gamma_2d=additional_args.gamma_2d,
            gamma_1d=additional_args.gamma_1d,
            interlayer=additional_args.interlayer,
            print_frags=print_frags,
        )

        return FragPart(
            unitcell=unitcell,
            mol=mol,
            frag_type=frag_type,
            AO_per_frag=AO_per_frag,
            AO_per_edge_per_frag=AO_per_edge_per_frag,
            ref_frag_idx_per_edge_per_frag=ref_frag_idx_per_edge_per_frag,
            weight_and_relAO_per_center_per_frag=weight_and_relAO_per_center_per_frag,
            relAO_per_edge_per_frag=relAO_per_edge_per_frag,
            relAO_in_ref_per_edge_per_frag=relAO_in_ref_per_edge_per_frag,
            relAO_per_origin_per_frag=relAO_per_origin_per_frag,
            n_BE=n_BE,
            natom=natom,
            frozen_core=frozen_core,
            self_match=self_match,
            allcen=allcen,
            iao_valence_basis=iao_valence_basis,
            kpt=kpt,
        )
    elif frag_type == "chemgen":
        if additional_args is None:
            additional_args = ChemGenArgs()
        else:
            assert isinstance(additional_args, ChemGenArgs)
        fragments = chemgen(
            mol,
            n_BE=n_BE,
            frozen_core=frozen_core,
            args=additional_args,
            iao_valence_basis=iao_valence_basis,
        )
        if print_frags:
            print(fragments.frag_structure.get_string())
        # Once periodic FragPart API is fixed,
        # add _get_FragPart_no_iao equivalent in quemb.molbe.chemgen
        mol_fragments = fragments.get_FragPart(
            wrong_iao_indexing=additional_args._wrong_iao_indexing
        )
        return FragPart(
            unitcell=unitcell,
            mol=mol,
            frag_type=frag_type,
            AO_per_frag=mol_fragments.AO_per_frag,
            AO_per_edge_per_frag=mol_fragments.AO_per_edge_per_frag,
            ref_frag_idx_per_edge_per_frag=mol_fragments.ref_frag_idx_per_edge_per_frag,
            weight_and_relAO_per_center_per_frag=mol_fragments.weight_and_relAO_per_center_per_frag,
            relAO_per_edge_per_frag=mol_fragments.relAO_per_edge_per_frag,
            relAO_in_ref_per_edge_per_frag=mol_fragments.relAO_in_ref_per_edge_per_frag,
            relAO_per_origin_per_frag=mol_fragments.relAO_per_origin_per_frag,
            n_BE=mol_fragments.n_BE,
            natom=natom,
            frozen_core=frozen_core,
            iao_valence_basis=iao_valence_basis,
            kpt=kpt,
            self_match=self_match,
            allcen=allcen,
        )
    else:
        raise ValueError(f"Fragmentation type = {frag_type} not implemented!")
