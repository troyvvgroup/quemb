# Author: Oinam Romesh Meitei

from typing import Literal, TypeAlias

from pyscf.gto.mole import Mole
from typing_extensions import assert_never

from quemb.molbe.autofrag import (
    AutogenArgs,
    FragPart,
    GraphGenArgs,
    autogen,
    graphgen,
)
from quemb.molbe.chemfrag import ChemGenArgs, chemgen
from quemb.molbe.helper import get_core

FragType: TypeAlias = Literal["chemgen", "graphgen", "autogen"]


AdditionalArgs: TypeAlias = AutogenArgs | ChemGenArgs | GraphGenArgs


def fragpart(
    mol: Mole,
    *,
    frag_type: FragType = "autogen",
    iao_valence_basis: str | None = None,
    print_frags: bool = True,
    write_geom: bool = False,
    be_type: str = "be2",
    frag_prefix: str = "f",
    frozen_core: bool = False,
    additional_args: AdditionalArgs | None = None,
) -> FragPart:
    if frag_type == "autogen":
        if additional_args is None:
            additional_args = AutogenArgs()
        else:
            assert isinstance(additional_args, AutogenArgs)

        return autogen(
            mol,
            be_type=be_type,
            frozen_core=frozen_core,
            write_geom=write_geom,
            iao_valence_basis=iao_valence_basis,
            print_frags=print_frags,
            iao_valence_only=additional_args.iao_valence_only,
        )

    elif frag_type == "chemgen":
        if additional_args is None:
            additional_args = ChemGenArgs()
        else:
            assert isinstance(additional_args, ChemGenArgs)
        fragments = chemgen(
            mol,
            n_BE=int(be_type[2:]),
            frozen_core=frozen_core,
            args=additional_args,
            iao_valence_basis=iao_valence_basis,
        )
        if write_geom:
            fragments.frag_structure.write_geom(prefix=frag_prefix)
        if print_frags:
            print(fragments.frag_structure.get_string())
        return fragments.match_autogen_output(
            wrong_iao_indexing=additional_args._wrong_iao_indexing
        )
    # else:
    #     pass
    # assert_never(f"Fragmentation type = {frag_type} not implemented!")


# class fragpart:
#     """Fragment/partitioning definition

#     Interfaces the fragmentation functions in MolBE. It defines
#     edge & center for density matching and energy estimation. It also forms the base
#     for IAO/PAO partitioning for a large basis set bootstrap calculation.

#     Parameters
#     ----------
#     frag_type :
#         Name of fragmentation function. 'chemgen', 'autogen', and 'graphgen'
#         are supported. Defaults to 'autogen'.
#     be_type :
#         Specifies order of bootsrap calculation in the atom-based fragmentation.
#         'be1', 'be2', 'be3', & 'be4' are supported.
#         Defaults to 'be2'
#         For a simple linear system A-B-C-D,
#         be1 only has fragments [A], [B], [C], [D]
#         be2 has [A, B, C], [B, C, D]
#         ben ...
#     mol :
#         This is required for the following :python:`frag_type` options:
#         :python:`"chemgen", "graphgen", "autogen"`
#     iao_valence_basis:
#         Name of minimal basis set for IAO scheme. 'sto-3g' suffice for most cases.
#     frozen_core:
#         Whether to invoke frozen core approximation. This is set to False by default
#     print_frags:
#         Whether to print out list of resulting fragments. True by default
#     write_geom:
#         Whether to write 'fragment.xyz' file which contains all the fragments
#         in cartesian coordinates.
#     remove_nonunique_frags:
#         Whether to remove fragments which are strict subsets of another
#         fragment in the system. True by default.
#     frag_prefix:
#         Prefix to be appended to the fragment datanames. Useful for managing
#         fragment scratch directories.
#     connectivity:
#         Keyword string specifying the distance metric to be used for edge
#         weights in the fragment adjacency graph. Currently supports "euclidean"
#         (which uses the square of the distance between atoms in real
#         space to determine connectivity within a fragment.)
#     cutoff:
#         Atoms with an edge weight beyond `cutoff` will be excluded from the
#         `shortest_path` calculation. This is crucial when handling very large
#         systems, where computing the shortest paths from all to all becomes
#         non-trivial. Defaults to 20.0.
#     additional_args:
#         Additional arguments for different fragmentation functions.
#     """

#     def __init__(
#         self,
#         mol: Mole,
#         *,
#         frag_type: FragType = "autogen",
#         iao_valence_basis: str | None = None,
#         print_frags: bool = True,
#         write_geom: bool = False,
#         be_type: str = "be2",
#         frag_prefix: str = "f",
#         frozen_core: bool = False,
#         additional_args: AdditionalArgs | None = None,
#     ) -> None:
#         self.mol = mol
#         self.frag_type = frag_type
#         self.fsites = []
#         self.Nfrag = 0
#         self.edge_sites = []
#         self.center = []
#         self.ebe_weight = []
#         self.edge_idx = []
#         self.center_idx = []
#         self.centerf_idx = []
#         self.be_type = be_type
#         self.frag_prefix = frag_prefix
#         self.frozen_core = frozen_core
#         self.iao_valence_basis = iao_valence_basis
#         self.Frag_atom = []
#         self.center_atom = []
#         self.hlist_atom = []
#         self.add_center_atom = []
#         self.iao_valence_only = False

#         # Check for frozen core approximation
#         if frozen_core:
#             self.ncore, self.no_core_idx, self.core_list = get_core(self.mol)

#         if frag_type == "graphgen":
#             if additional_args is None:
#                 additional_args = GraphGenArgs()
#             else:
#                 assert isinstance(additional_args, GraphGenArgs)
#             if iao_valence_basis:
#                 raise ValueError("iao_valence_basis not yet supported for 'graphgen'")
#             fragment_map = graphgen(
#                 mol=self.mol.copy(),
#                 be_type=self.be_type,
#                 frozen_core=self.frozen_core,
#                 remove_nonunique_frags=additional_args.remove_nonnunique_frags,
#                 frag_prefix=self.frag_prefix,
#                 connectivity=additional_args.connectivity,
#                 iao_valence_basis=self.iao_valence_basis,
#                 cutoff=additional_args.cutoff,
#             )

#             self.fsites = fragment_map.fsites
#             self.edge_sites = fragment_map.edge
#             self.center = fragment_map.center
#             self.Frag_atom = fragment_map.fragment_atoms
#             self.center_atom = fragment_map.center_atoms
#             self.centerf_idx = fragment_map.centerf_idx
#             self.ebe_weight = fragment_map.ebe_weights
#             self.Nfrag = len(self.fsites)

#         elif frag_type == "autogen":
#             if additional_args is None:
#                 additional_args = AutogenArgs()
#             else:
#                 assert isinstance(additional_args, AutogenArgs)

#             self.iao_valence_only = additional_args.iao_valence_only
#             fgs = autogen(
#                 mol,
#                 be_type=be_type,
#                 frozen_core=frozen_core,
#                 write_geom=write_geom,
#                 iao_valence_basis=iao_valence_basis,
#                 print_frags=print_frags,
#                 iao_valence_only=additional_args.iao_valence_only,
#             )

#             (
#                 self.fsites,
#                 self.edge_sites,
#                 self.center,
#                 self.edge_idx,
#                 self.center_idx,
#                 self.centerf_idx,
#                 self.ebe_weight,
#                 self.Frag_atom,
#                 self.center_atom,
#                 self.hlist_atom,
#                 self.add_center_atom,
#             ) = fgs
#             self.Nfrag = len(self.fsites)

#         elif frag_type == "chemgen":
#             if additional_args is None:
#                 additional_args = ChemGenArgs()
#             else:
#                 assert isinstance(additional_args, ChemGenArgs)
#             fragments = chemgen(
#                 mol,
#                 n_BE=int(be_type[2:]),
#                 frozen_core=frozen_core,
#                 args=additional_args,
#                 iao_valence_basis=iao_valence_basis,
#             )
#             if write_geom:
#                 fragments.frag_structure.write_geom(prefix=frag_prefix)
#             if print_frags:
#                 print(fragments.frag_structure.get_string())
#             fgs = fragments.match_autogen_output(
#                 wrong_iao_indexing=additional_args._wrong_iao_indexing
#             )
#             self.fsites = fgs.fsites
#             self.edge_sites = fgs.edge_sites
#             self.center = fgs.center
#             self.edge_idx = fgs.edge_idx
#             self.center_idx = fgs.center_idx
#             self.centerf_idx = fgs.centerf_idx
#             self.ebe_weight = fgs.ebe_weight
#             self.Frag_atom = fgs.Frag_atom
#             self.center_atom = fgs.center_atom
#             self.hlist_atom = fgs.hlist_atom
#             self.add_center_atom = fgs.add_center_atom
#             self.Nfrag = fgs.Nfrag

#         else:
#             assert_never(f"Fragmentation type = {frag_type} not implemented!")
