# Authors: Oinam Romesh Meitei, Oskar Weser

from typing import TypeAlias
from warnings import warn

import numpy as np
from pyscf.gto.mole import Mole
from typing_extensions import assert_never

from quemb.molbe.autofrag import (
    AutogenArgs,
    FragPart,
    FragType,
    autogen,
)
from quemb.molbe.chemfrag import ChemGenArgs, chemgen
from quemb.molbe.graphfrag import GraphGenArgs, graphgen

AdditionalArgs: TypeAlias = AutogenArgs | ChemGenArgs | GraphGenArgs


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
    order_by_size: bool = False,
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
        :python:`"chemgen", "graphgen", "autogen"`.
        If you use :python:`"chemgen"` in your work
        please credit :cite:`weser_automated_2023`.
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
    order_by_size:
        Order the fragments by descending size.
        This can be beneficial for better load-balancing.
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
        result = graphgen(
            mol=mol.copy(),
            n_BE=n_BE,
            frozen_core=frozen_core,
            remove_nonunique_frags=additional_args.remove_nonnunique_frags,
            frag_prefix=frag_prefix,
            connectivity=additional_args.connectivity,
            iao_valence_basis=iao_valence_basis,
            cutoff=additional_args.cutoff,
        )
    elif frag_type == "autogen":
        if additional_args is None:
            additional_args = AutogenArgs()
        else:
            assert isinstance(additional_args, AutogenArgs)

        result = autogen(
            mol,
            n_BE=n_BE,
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
            n_BE=n_BE,
            frozen_core=frozen_core,
            args=additional_args,
            iao_valence_basis=iao_valence_basis,
        )
        if write_geom:
            fragments.frag_structure.write_geom(prefix=frag_prefix)
        if print_frags:
            print(fragments.frag_structure.get_string())
        result = fragments.get_FragPart(
            wrong_iao_indexing=additional_args._wrong_iao_indexing
        )
    else:
        assert_never(f"Fragmentation type = {frag_type} not implemented!")

    if not _correct_number_of_centers(result) and frag_type != "graphgen":
        warn(
            "Strange number of centers detected. "
            'It is advised to use "chemgen" instead.'
        )
    if order_by_size:
        result = _order_by_decreasing_size(result)
    return result


def _order_by_decreasing_size(fragments: FragPart) -> FragPart:
    """Order by decreasing fragment size"""
    idx = np.argsort([-len(motifs) for motifs in fragments.AO_per_frag], stable=True)
    return fragments.reindex(idx)  # type: ignore[arg-type]


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
            for motifs, edges in zip(
                fragpart.motifs_per_frag, fragpart.ref_frag_idx_per_edge_per_frag
            )
        ]
    )
    return n_centers == n_motifs
