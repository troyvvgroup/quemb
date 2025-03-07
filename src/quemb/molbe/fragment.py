# Author: Oinam Romesh Meitei

from typing import Literal, TypeAlias

from pyscf.gto.mole import Mole
from typing_extensions import assert_never

from quemb.molbe.autofrag import ChemGenArgs, autogen, chemgen, graphgen
from quemb.molbe.helper import get_core
from quemb.molbe.lchain import chain as _ext_chain
from quemb.shared.helper import copy_docstring

FragType: TypeAlias = Literal[
    "chemgen", "graphgen", "autogen", "hchain_simple", "chain"
]


class fragpart:
    """Fragment/partitioning definition

    Interfaces two main fragmentation functions (autogen & chain) in MolBE. It defines
    edge & center for density matching and energy estimation. It also forms the base
    for IAO/PAO partitioning for a large basis set bootstrap calculation.

    Parameters
    ----------
    frag_type :
        Name of fragmentation function. 'chemgen', 'autogen', 'graphgen',
        'hchain_simple', and 'chain' are supported. Defaults to 'autogen'.
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
    iao_valence_only:
        If this option is set to True, all calculation will be performed in
        the valence basis in the IAO partitioning.
        This is an experimental feature.
    frozen_core:
        Whether to invoke frozen core approximation. This is set to False by default
    print_frags:
        Whether to print out list of resulting fragments. True by default
    write_geom:
        Whether to write 'fragment.xyz' file which contains all the fragments
        in cartesian coordinates.
    remove_nonunique_frags:
        Whether to remove fragments which are strict subsets of another
        fragment in the system. True by default.
    frag_prefix:
        Prefix to be appended to the fragment datanames. Useful for managing
        fragment scratch directories.
    connectivity:
        Keyword string specifying the distance metric to be used for edge
        weights in the fragment adjacency graph. Currently supports "euclidean"
        (which uses the square of the distance between atoms in real
        space to determine connectivity within a fragment.)
    cutoff:
        Atoms with an edge weight beyond `cutoff` will be excluded from the
        `shortest_path` calculation. This is crucial when handling very large
        systems, where computing the shortest paths from all to all becomes
        non-trivial. Defaults to 20.0.
    additional_args:
        Additional arguments for different fragmentation functions.
    """

    def __init__(
        self,
        frag_type: FragType = "autogen",
        closed: bool = False,
        iao_valence_basis: str | None = None,
        iao_valence_only: bool = False,
        print_frags: bool = True,
        write_geom: bool = False,
        be_type: str = "be2",
        frag_prefix: str = "f",
        connectivity: str = "euclidean",
        mol: Mole | None = None,
        frozen_core: bool = False,
        cutoff: float = 20.0,
        remove_nonnunique_frags: bool = True,
        additional_args: ChemGenArgs | None = None,
    ) -> None:
        self.mol = mol
        self.frag_type = frag_type
        self.fsites = []
        self.Nfrag = 0
        self.edge_sites = []
        self.center = []
        self.ebe_weight = []
        self.edge_idx = []
        self.center_idx = []
        self.centerf_idx = []
        self.be_type = be_type
        self.frag_prefix = frag_prefix
        self.connectivity = connectivity
        self.frozen_core = frozen_core
        self.iao_valence_basis = iao_valence_basis
        self.iao_valence_only = iao_valence_only
        self.cutoff = cutoff
        self.remove_nonnunique_frags = remove_nonnunique_frags
        self.Frag_atom = []
        self.center_atom = []
        self.hlist_atom = []
        self.add_center_atom = []

        # Check for frozen core approximation
        if frozen_core:
            assert self.mol is not None
            self.ncore, self.no_core_idx, self.core_list = get_core(self.mol)

        if frag_type != "hchain_simple" and self.mol is None:
            raise ValueError("Provide pyscf gto.M object in fragpart() and restart!")

        # Check type of fragmentation function
        if frag_type == "hchain_simple":
            # This is an experimental feature.
            self.hchain_simple()

        elif frag_type == "chain":
            self.chain(self.mol, frozen_core=frozen_core, closed=closed)

        elif frag_type == "graphgen":
            assert self.mol is not None
            fragment_map = graphgen(
                mol=self.mol.copy(),
                be_type=self.be_type,
                frozen_core=self.frozen_core,
                remove_nonunique_frags=self.remove_nonnunique_frags,
                frag_prefix=self.frag_prefix,
                connectivity=self.connectivity,
                iao_valence_basis=self.iao_valence_basis,
                cutoff=self.cutoff,
            )

            self.fsites = fragment_map.fsites
            self.edge_sites = fragment_map.edge
            self.center = fragment_map.center
            self.Frag_atom = fragment_map.fragment_atoms
            self.center_atom = fragment_map.center_atoms
            self.centerf_idx = fragment_map.centerf_idx
            self.ebe_weight = fragment_map.ebe_weights
            self.Nfrag = len(self.fsites)

        elif frag_type == "autogen":
            fgs = autogen(
                mol,
                be_type=be_type,
                frozen_core=frozen_core,
                write_geom=write_geom,
                iao_valence_basis=iao_valence_basis,
                iao_valence_only=iao_valence_only,
                print_frags=print_frags,
            )

            (
                self.fsites,
                self.edge_sites,
                self.center,
                self.edge_idx,
                self.center_idx,
                self.centerf_idx,
                self.ebe_weight,
                self.Frag_atom,
                self.center_atom,
                self.hlist_atom,
                self.add_center_atom,
            ) = fgs
            self.Nfrag = len(self.fsites)

        elif frag_type == "chemgen":
            if iao_valence_only:
                raise NotImplementedError(
                    "iao_valence_only is not implemented for chemgen"
                )
            assert isinstance(additional_args, (type(None), ChemGenArgs))
            assert mol is not None
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
            fgs = fragments.match_autogen_output(
                wrong_iao_indexing=False
                if additional_args is None
                else additional_args._wrong_iao_indexing
            )
            self.fsites = fgs.fsites
            self.edge_sites = fgs.edge_sites
            self.center = fgs.center
            self.edge_idx = fgs.edge_idx
            self.center_idx = fgs.center_idx
            self.centerf_idx = fgs.centerf_idx
            self.ebe_weight = fgs.ebe_weight
            self.Frag_atom = fgs.Frag_atom
            self.center_atom = fgs.center_atom
            self.hlist_atom = fgs.hlist_atom
            self.add_center_atom = fgs.add_center_atom
            self.Nfrag = fgs.Nfrag

        else:
            assert_never(f"Fragmentation type = {frag_type} not implemented!")

    @copy_docstring(_ext_chain)
    def chain(self, mol, frozen_core=False, closed=False):
        return _ext_chain(self, mol, frozen_core=frozen_core, closed=closed)

    def hchain_simple(self):
        """Hard coded fragmentation feature"""
        self.natom = self.mol.natm
        if self.be_type == "be1":
            for i in range(self.natom):
                self.fsites.append([i])
                self.edge_sites.append([])
            self.Nfrag = len(self.fsites)

        elif self.be_type == "be2":
            for i in range(self.natom - 2):
                self.fsites.append([i, i + 1, i + 2])
                self.centerf_idx.append([1])
            self.Nfrag = len(self.fsites)

            self.edge_sites.append([[2]])
            for i in self.fsites[1:-1]:
                self.edge_sites.append([[i[0]], [i[-1]]])
            self.edge_sites.append([[self.fsites[-1][0]]])

            self.center.append([1])
            for i in range(self.Nfrag - 2):
                self.center.append([i, i + 2])
            self.center.append([self.Nfrag - 2])

        elif self.be_type == "be3":
            for i in range(self.natom - 4):
                self.fsites.append([i, i + 1, i + 2, i + 3, i + 4])
                self.centerf_idx.append([2])
            self.Nfrag = len(self.fsites)

            self.edge_sites.append([[3], [4]])
            for i in self.fsites[1:-1]:
                self.edge_sites.append([[i[0]], [i[1]], [i[-2]], [i[-1]]])
            self.edge_sites.append([[self.fsites[-1][0]], [self.fsites[-1][1]]])

            self.center.append([1, 2])
            self.center.append([0, 0, 2, 3])
            for i in range(self.Nfrag - 4):
                self.center.append([i, i + 1, i + 3, i + 4])

            self.center.append(
                [self.Nfrag - 4, self.Nfrag - 3, self.Nfrag - 1, self.Nfrag - 1]
            )
            self.center.append([self.Nfrag - 3, self.Nfrag - 2])

        for ix, i in enumerate(self.fsites):
            tmp_ = []
            elist_ = [xx for yy in self.edge_sites[ix] for xx in yy]
            for j in i:
                if j not in elist_:
                    tmp_.append(i.index(j))
            self.ebe_weight.append([1.0, tmp_])

        if not self.be_type == "be1":
            for i in range(self.Nfrag):
                idx = []
                for j in self.edge_sites[i]:
                    idx.append([self.fsites[i].index(k) for k in j])
                self.edge_idx.append(idx)

            for i in range(self.Nfrag):
                idx = []
                for j in range(len(self.center[i])):
                    idx.append(
                        [
                            self.fsites[self.center[i][j]].index(k)
                            for k in self.edge_sites[i][j]
                        ]
                    )
                self.center_idx.append(idx)
