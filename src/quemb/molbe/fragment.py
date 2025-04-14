# Author: Oinam Romesh Meitei

from collections.abc import Sequence
from typing import Literal, TypeAlias
from warnings import warn

from attrs import cmp_using, define, field
from pyscf.gto import Mole
from typing_extensions import assert_never

from quemb.molbe.autofrag import AutoGenArgs, autogen  # type: ignore[attr-defined]
from quemb.molbe.chemfrag import ChemGenArgs, chemgen
from quemb.molbe.graphfrag import GraphGenArgs, graphgen
from quemb.molbe.helper import are_equal, get_core
from quemb.shared.typing import (
    AtomIdx,
    CenterIdx,
    FragmentIdx,
    GlobalAOIdx,
    MotifIdx,
    OriginIdx,
    OtherRelAOIdx,
    OwnRelAOIdx,
)

AdditionalArgs: TypeAlias = AutoGenArgs | ChemGenArgs | GraphGenArgs

ListOverFrag: TypeAlias = list
ListOverEdge: TypeAlias = list
ListOverMotif: TypeAlias = list

FragType: TypeAlias = Literal["chemgen", "graphgen", "autogen"]


@define
class FragPart:
    """Data structure to hold the result of BE fragmentations."""

    #: The full molecule.
    mol: Mole = field(eq=cmp_using(are_equal))
    #: The algorithm used for fragmenting.
    frag_type: FragType
    #: The level of BE fragmentation, i.e. "be1", "be2", ...
    n_BE: int

    #: This is a list over fragments  and gives the global orbital indices of all atoms
    #: in the fragment. These are ordered by the atoms in the fragment.
    fsites: ListOverFrag[list[GlobalAOIdx]]

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


def fragmentate(
    mol: Mole,
    *,
    frag_type: FragType = "autogen",
    iao_valence_basis: str | None = None,
    print_frags: bool = True,
    write_geom: bool = False,
    n_BE: int = 2,
    frag_prefix: str = "f",
    frozen_core: bool = False,
    additional_args: AdditionalArgs | None = None,
) -> FragPart:
    """Fragment/partitioning definition

    Interfaces the fragmentation functions in MolBE. It defines
    edge & center for density matching and energy estimation. It also forms the base
    for IAO/PAO partitioning for a large basis set bootstrap calculation.

    Parameters
    ----------
    frag_type :
        Name of fragmentation function. 'chemgen', 'autogen', and 'graphgen'
        are supported. Defaults to 'autogen'.
    n_BE: int, optional
        Specifies the order of bootstrap calculation in the atom-based fragmentation,
        i.e. BE(n).
        For a simple linear system A-B-C-D,
        BE(1) only has fragments [A], [B], [C], [D]
        BE(2) has [A, B, C], [B, C, D]
        ben ...
    mol :
        This is required for the following :python:`frag_type` options:
        :python:`"chemgen", "graphgen", "autogen"`
    iao_valence_basis:
        Name of minimal basis set for IAO scheme. 'sto-3g' suffice for most cases.
    frozen_core:
        Whether to invoke frozen core approximation. This is set to False by default
    print_frags:
        Whether to print out list of resulting fragments. True by default
    write_geom:
        Whether to write 'fragment.xyz' file which contains all the fragments
        in cartesian coordinates.
    frag_prefix:
        Prefix to be appended to the fragment datanames. Useful for managing
        fragment scratch directories.
    additional_args:
        Additional arguments for different fragmentation functions.
    """

    if frag_type == "graphgen":
        if additional_args is None:
            additional_args = GraphGenArgs()
        else:
            assert isinstance(additional_args, GraphGenArgs)
        if iao_valence_basis:
            raise ValueError("iao_valence_basis not yet supported for 'graphgen'")
        graphgen_output = graphgen(
            mol=mol.copy(),
            n_BE=n_BE,
            frozen_core=frozen_core,
            remove_nonunique_frags=additional_args.remove_nonnunique_frags,
            frag_prefix=frag_prefix,
            connectivity=additional_args.connectivity,
            iao_valence_basis=iao_valence_basis,
            cutoff=additional_args.cutoff,
        )
        result = FragPart(**graphgen_output)

    elif frag_type == "autogen":
        if additional_args is None:
            additional_args = AutoGenArgs()
        else:
            assert isinstance(additional_args, AutoGenArgs)

        autogen_output = autogen(
            mol,
            n_BE=n_BE,
            frozen_core=frozen_core,
            write_geom=write_geom,
            iao_valence_basis=iao_valence_basis,
            print_frags=print_frags,
            iao_valence_only=additional_args.iao_valence_only,  # type: ignore[union-attr]
        )
        result = FragPart(**autogen_output)

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
        if write_geom:
            fragments.frag_structure.write_geom(prefix=frag_prefix)
        if print_frags:
            print(fragments.frag_structure.get_string())
        chemgen_output = fragments.get_FragPart(
            wrong_iao_indexing=additional_args._wrong_iao_indexing
        )
        result = FragPart(**chemgen_output)

    else:
        assert_never(f"Fragmentation type = {frag_type} not implemented!")

    if not _correct_number_of_centers(result) and frag_type != "graphgen":
        warn(
            "Strange number of centers detected. "
            'It is advised to use "chemgen" instead.'
        )
    return result


def _correct_number_of_centers(fragpart: FragPart) -> bool:
    """By default we assume autocratic matching
    and the same number of centers and motifs."""
    if any(atom != "H" for atom in fragpart.mol.elements):
        n_motifs = sum(atom != "H" for atom in fragpart.mol.elements)
    else:
        n_motifs = fragpart.mol.natm

    n_centers = sum(
        [
            len(motifs) - len(edges)
            for motifs, edges in zip(fragpart.Frag_atom, fragpart.center)
        ]
    )
    return n_centers == n_motifs
