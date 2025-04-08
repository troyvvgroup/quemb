# Author(s): Oinam Romesh Meitei


from warnings import warn

from attrs import define, field
from pyscf.pbc.gto.cell import Cell

from quemb.kbe.autofrag import autogen
from quemb.molbe.chemfrag import ChemGenArgs, chemgen
from quemb.molbe.helper import get_core


@define
class FragPart:
    unitcell: int
    mol: Cell
    frag_type: str
    fsites: list
    edge_sites: list
    center: list
    ebe_weight: list
    edge_idx: list
    center_idx: list
    centerf_idx: list
    be_type: str
    natom: int
    frozen_core: bool
    self_match: bool
    allcen: bool
    iao_valence_basis: str
    kpt: list[int] | tuple[int, int, int]

    Nfrag: int = field(init=False)
    ncore: int | None = field(init=False)
    no_core_idx: list[int] | None = field(init=False)
    core_list: list[int] | None = field(init=False)

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
    mol: Cell,
    kpt: list[int] | tuple[int, int, int],
    *,
    natom=0,
    frag_type="autogen",
    unitcell=1,
    gamma_2d=False,
    gamma_1d=False,
    interlayer=False,
    long_bond=False,
    perpend_dist=4.0,
    perpend_dist_tol=1e-3,
    nx=False,
    ny=False,
    nz=False,
    iao_valence_basis=None,
    be_type="be2",
    frozen_core=False,
    self_match=False,
    allcen=True,
    print_frags=True,
):
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
    be_type : str
        Specifies order of bootsrap calculation in the atom-based fragmentation.
        'be1', 'be2', 'be3', & 'be4' are supported.
        Defaults to 'be2'
        For a simple linear system A-B-C-D,
        be1 only has fragments [A], [B], [C], [D]
        be2 has [A, B, C], [B, C, D]
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
            fsites,
            edge_sites,
            center,
            edge_idx,
            center_idx,
            centerf_idx,
            ebe_weight,
        ) = autogen(
            mol,
            kpt,
            be_type=be_type,
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
            fsites=fsites,
            edge_sites=edge_sites,
            center=center,
            ebe_weight=ebe_weight,
            edge_idx=edge_idx,
            center_idx=center_idx,
            centerf_idx=centerf_idx,
            be_type=be_type,
            natom=natom,
            frozen_core=frozen_core,
            self_match=self_match,
            allcen=allcen,
            iao_valence_basis=iao_valence_basis,
            kpt=kpt,
        )
    elif frag_type == "chemgen":
        warn("Periodic BE1 with chemgen is a temporary solution.")
        if kpt is None:
            raise ValueError("Provide kpt mesh in fragmentate() and restart!")
        if be_type != "be1":
            raise ValueError("Only be_type='be1' is supported for periodic chemgen!")
        fragments = chemgen(
            mol.to_mol(),
            n_BE=int(be_type[2:]),
            frozen_core=frozen_core,
            args=ChemGenArgs(),
            iao_valence_basis=iao_valence_basis,
        )
        molecular_FragPart = fragments.get_FragPart()
        if print_frags:
            print(fragments.frag_structure.get_string())
        return FragPart(
            unitcell=unitcell,
            mol=mol,
            frag_type=frag_type,
            fsites=molecular_FragPart.fsites,
            edge_sites=molecular_FragPart.edge_sites,
            center=molecular_FragPart.center,
            ebe_weight=molecular_FragPart.ebe_weight,
            edge_idx=molecular_FragPart.edge_idx,
            center_idx=molecular_FragPart.center_idx,
            centerf_idx=molecular_FragPart.centerf_idx,
            be_type=molecular_FragPart.be_type,
            natom=natom,
            frozen_core=frozen_core,
            iao_valence_basis=iao_valence_basis,
            kpt=kpt,
            self_match=self_match,
            allcen=allcen,
        )
    else:
        raise ValueError(f"Fragmentation type = {frag_type} not implemented!")
