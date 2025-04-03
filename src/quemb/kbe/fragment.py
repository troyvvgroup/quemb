# Author(s): Oinam Romesh Meitei


from attrs import define, field
from pyscf.pbc.gto.cell import Cell

from quemb.kbe.autofrag import autogen
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
    n_BE: int
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
    n_BE: int = 2,
    frozen_core=False,
    self_match=False,
    allcen=True,
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
            fsites=fsites,
            edge_sites=edge_sites,
            center=center,
            ebe_weight=ebe_weight,
            edge_idx=edge_idx,
            center_idx=center_idx,
            centerf_idx=centerf_idx,
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
