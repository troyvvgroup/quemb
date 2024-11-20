# Author: Oinam Romesh Meitei

import sys

from .autofrag import autogen
from .helper import get_core


class fragpart:
    """Fragment/partitioning definition

    Interfaces two main fragmentation functions (autogen & chain) in MolBE. It defines edge &
    center for density matching and energy estimation. It also forms the base for IAO/PAO partitioning for
    a large basis set bootstrap calculation.

    Parameters
    ----------
    frag_type : str
        Name of fragmentation function. 'autogen', 'hchain_simple', and 'chain' are supported. Defaults to 'autogen'
        For systems with only hydrogen, use 'chain'; everything else should use 'autogen'
    be_type : str
        Specifies order of bootsrap calculation in the atom-based fragmentation. 'be1', 'be2', 'be3', & 'be4' are supported.
        Defaults to 'be2'
        For a simple linear system A-B-C-D,
        be1 only has fragments [A], [B], [C], [D]
        be2 has [A, B, C], [B, C, D]
        ben ...
    mol : pyscf.gto.Molecule
        pyscf.gto.Molecule object. This is required for the options, 'autogen' and 'chain' as frag_type.
    valence_basis: str
        Name of minimal basis set for IAO scheme. 'sto-3g' suffice for most cases.
    valence_only: bool
        If this option is set to True, all calculation will be performed in the valence basis in the IAO partitioning.
        This is an experimental feature.
    frozen_core: bool
        Whether to invoke frozen core approximation. This is set to False by default
    print_frags: bool
        Whether to print out list of resulting fragments. True by default
    write_geom: bool
        Whether to write 'fragment.xyz' file which contains all the fragments in cartesian coordinates.
    """

    def __init__(
        self,
        frag_type="autogen",
        closed=False,
        valence_basis=None,
        valence_only=False,
        print_frags=True,
        write_geom=False,
        be_type="be2",
        mol=None,
        frozen_core=False,
    ):
        # Initialize class attributes
        self.mol = mol
        self.frag_type = frag_type
        self.fsites = []
        self.Nfrag = 0
        self.edge = []
        self.center = []
        self.ebe_weight = []
        self.edge_idx = []
        self.center_idx = []
        self.centerf_idx = []
        self.be_type = be_type
        self.frozen_core = frozen_core
        self.valence_basis = valence_basis
        self.valence_only = valence_only

        # Initialize class attributes necessary for mixed-basis BE
        self.Frag_atom = []
        self.center_atom = []
        self.hlist_atom = []
        self.add_center_atom = []

        # Check for frozen core approximation
        if frozen_core:
            self.ncore, self.no_core_idx, self.core_list = get_core(mol)

        # Check type of fragmentation function
        if frag_type == "hchain_simple":
            # This is an experimental feature.
            self.hchain_simple()
        elif frag_type == "chain":
            if mol is None:
                print(
                    "Provide pyscf gto.M object in fragpart() and restart!", flush=True
                )
                print("exiting", flush=True)
                sys.exit()
            self.chain(mol, frozen_core=frozen_core, closed=closed)
        elif frag_type == "autogen":
            if mol is None:
                print(
                    "Provide pyscf gto.M object in fragpart() and restart!", flush=True
                )
                print("exiting", flush=True)
                sys.exit()

            fgs = autogen(
                mol,
                be_type=be_type,
                frozen_core=frozen_core,
                write_geom=write_geom,
                valence_basis=valence_basis,
                valence_only=valence_only,
                print_frags=print_frags,
            )

            (
                self.fsites,
                self.edge,
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

        else:
            print("Fragmentation type = ", frag_type, " not implemented!", flush=True)
            print("exiting", flush=True)
            sys.exit()

    from .lchain import chain

    def hchain_simple(self):
        """Hard coded fragmentation feature"""
        self.natom = self.mol.natm
        if self.be_type == "be1":
            for i in range(self.natom):
                self.fsites.append([i])
                self.edge.append([])
            self.Nfrag = len(self.fsites)

        elif self.be_type == "be2":
            for i in range(self.natom - 2):
                self.fsites.append([i, i + 1, i + 2])
                self.centerf_idx.append([1])
            self.Nfrag = len(self.fsites)

            self.edge.append([[2]])
            for i in self.fsites[1:-1]:
                self.edge.append([[i[0]], [i[-1]]])
            self.edge.append([[self.fsites[-1][0]]])

            self.center.append([1])
            for i in range(self.Nfrag - 2):
                self.center.append([i, i + 2])
            self.center.append([self.Nfrag - 2])

        elif self.be_type == "be3":
            for i in range(self.natom - 4):
                self.fsites.append([i, i + 1, i + 2, i + 3, i + 4])
                self.centerf_idx.append([2])
            self.Nfrag = len(self.fsites)

            self.edge.append([[3], [4]])
            for i in self.fsites[1:-1]:
                self.edge.append([[i[0]], [i[1]], [i[-2]], [i[-1]]])
            self.edge.append([[self.fsites[-1][0]], [self.fsites[-1][1]]])

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
            elist_ = [xx for yy in self.edge[ix] for xx in yy]
            for j in i:
                if j not in elist_:
                    tmp_.append(i.index(j))
            self.ebe_weight.append([1.0, tmp_])

        if not self.be_type == "be1":
            for i in range(self.Nfrag):
                idx = []
                for j in self.edge[i]:
                    idx.append([self.fsites[i].index(k) for k in j])
                self.edge_idx.append(idx)

            for i in range(self.Nfrag):
                idx = []
                for j in range(len(self.center[i])):
                    idx.append(
                        [
                            self.fsites[self.center[i][j]].index(k)
                            for k in self.edge[i][j]
                        ]
                    )
                self.center_idx.append(idx)
