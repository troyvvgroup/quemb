# Author: Oinam Romesh Meitei


from quemb.molbe.autofrag import autogen, graphgen
from quemb.molbe.helper import get_core
from quemb.molbe.lchain import chain as _ext_chain
from quemb.shared.helper import copy_docstring


class fragpart:
    """Fragment/partitioning definition

    Interfaces two main fragmentation functions (autogen & chain) in MolBE. It defines
    edge & center for density matching and energy estimation. It also forms the base
    for IAO/PAO partitioning for a large basis set bootstrap calculation.

    Parameters
    ----------
    frag_type : str
        Name of fragmentation function. 'autogen', 'graphgen', 'hchain_simple',
        and 'chain' are supported. Defaults to 'autogen'.
    be_type : str
        Specifies order of bootsrap calculation in the atom-based fragmentation.
        'be1', 'be2', 'be3', & 'be4' are supported.
        Defaults to 'be2'
        For a simple linear system A-B-C-D,
        be1 only has fragments [A], [B], [C], [D]
        be2 has [A, B, C], [B, C, D]
        ben ...
    mol : pyscf.gto.mole.Mole
        This is required for the options, 'autogen'
        and 'chain' as frag_type.
    iao_valence_basis: str
        Name of minimal basis set for IAO scheme. 'sto-3g' suffice for most cases.
    valence_only: bool
        If this option is set to True, all calculation will be performed in
        the valence basis in the IAO partitioning.
        This is an experimental feature.
    frozen_core: bool
        Whether to invoke frozen core approximation. This is set to False by default
    print_frags: bool
        Whether to print out list of resulting fragments. True by default
    write_geom: bool
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
    """

    def __init__(
        self,
        frag_type="autogen",
        closed=False,
        iao_valence_basis=None,
        valence_only=False,
        print_frags=True,
        write_geom=False,
        be_type="be2",
        frag_prefix="f",
        connectivity="euclidean",
        mol=None,
        frozen_core=False,
        cutoff=20,
        remove_nonnunique_frags=True,
    ):
        self.mol = mol
        self.frag_type = frag_type
        self.fsites = []
        self.Nfrag = 0
        self.edgesites = []
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
        self.valence_only = valence_only
        self.cutoff = cutoff
        self.remove_nonnunique_frags = remove_nonnunique_frags
        self.Frag_atom = []
        self.center_atom = []

        # Check for frozen core approximation
        if frozen_core:
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
            self.edgesites = fragment_map.edge
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
                valence_only=valence_only,
                print_frags=print_frags,
            )

            (
                self.fsites,
                self.edgesites,
                self.center,
                self.edge_idx,
                self.center_idx,
                self.centerf_idx,
                self.ebe_weight,
                self.Frag_atom,
                self.center_atom,
            ) = fgs
            self.Nfrag = len(self.fsites)

        else:
            raise ValueError(f"Fragmentation type = {frag_type} not implemented!")

    @copy_docstring(_ext_chain)
    def chain(self, mol, frozen_core=False, closed=False):
        return _ext_chain(self, mol, frozen_core=frozen_core, closed=closed)

    def hchain_simple(self):
        """Hard coded fragmentation feature"""
        self.natom = self.mol.natm
        if self.be_type == "be1":
            for i in range(self.natom):
                self.fsites.append([i])
                self.edgesites.append([])
            self.Nfrag = len(self.fsites)

        elif self.be_type == "be2":
            for i in range(self.natom - 2):
                self.fsites.append([i, i + 1, i + 2])
                self.centerf_idx.append([1])
            self.Nfrag = len(self.fsites)

            self.edgesites.append([[2]])
            for i in self.fsites[1:-1]:
                self.edgesites.append([[i[0]], [i[-1]]])
            self.edgesites.append([[self.fsites[-1][0]]])

            self.center.append([1])
            for i in range(self.Nfrag - 2):
                self.center.append([i, i + 2])
            self.center.append([self.Nfrag - 2])

        elif self.be_type == "be3":
            for i in range(self.natom - 4):
                self.fsites.append([i, i + 1, i + 2, i + 3, i + 4])
                self.centerf_idx.append([2])
            self.Nfrag = len(self.fsites)

            self.edgesites.append([[3], [4]])
            for i in self.fsites[1:-1]:
                self.edgesites.append([[i[0]], [i[1]], [i[-2]], [i[-1]]])
            self.edgesites.append([[self.fsites[-1][0]], [self.fsites[-1][1]]])

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
            elist_ = [xx for yy in self.edgesites[ix] for xx in yy]
            for j in i:
                if j not in elist_:
                    tmp_.append(i.index(j))
            self.ebe_weight.append([1.0, tmp_])

        if not self.be_type == "be1":
            for i in range(self.Nfrag):
                idx = []
                for j in self.edgesites[i]:
                    idx.append([self.fsites[i].index(k) for k in j])
                self.edge_idx.append(idx)

            for i in range(self.Nfrag):
                idx = []
                for j in range(len(self.center[i])):
                    idx.append(
                        [
                            self.fsites[self.center[i][j]].index(k)
                            for k in self.edgesites[i][j]
                        ]
                    )
                self.center_idx.append(idx)
