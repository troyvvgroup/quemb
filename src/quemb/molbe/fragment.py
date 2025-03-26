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
    """Fragment/partitioning definition

    Interfaces the fragmentation functions in MolBE. It defines
    edge & center for density matching and energy estimation. It also forms the base
    for IAO/PAO partitioning for a large basis set bootstrap calculation.

    Parameters
    ----------
    frag_type :
        Name of fragmentation function. 'chemgen', 'autogen', and 'graphgen'
        are supported. Defaults to 'autogen'.
    be_type :
        Specifies order of bootsrap calculation in the atom-based fragmentation.
        'be1', 'be2', 'be3', & 'be4' are supported.
        Defaults to 'be2'
        For a simple linear system A-B-C-D,
        be1 only has fragments [A], [B], [C], [D]
        be2 has [A, B, C], [B, C, D]
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
        return graphgen(
            mol=mol.copy(),
            be_type=be_type,
            frozen_core=frozen_core,
            remove_nonunique_frags=additional_args.remove_nonnunique_frags,
            frag_prefix=frag_prefix,
            connectivity=additional_args.connectivity,
            iao_valence_basis=iao_valence_basis,
            cutoff=additional_args.cutoff,
        ).to_FragPart(mol, be_type, frozen_core)
    elif frag_type == "autogen":
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
    else:
        assert_never(f"Fragmentation type = {frag_type} not implemented!")
