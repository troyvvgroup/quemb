from collections.abc import Sequence
from pathlib import Path

from pyscf.tools.cubegen import orbital

from quemb import molbe
from quemb.shared.typing import KwargDict, PathLike


def write_cube(
    be_object: molbe.BE,
    cube_file_path: PathLike,
    *,
    fragment_idx: Sequence[int] | None = None,
    orbital_idx: Sequence[int] | None = None,
    cubegen_kwargs: KwargDict | None = None,
) -> None:
    """Write cube files of embedding orbitals from a BE object.

    Parameters
    ----------
    be_object :
        BE object containing the fragments, each of which contains embedding orbitals.
    cube_file_path :
        Directory to write the cube files to. Directory is created,
        if it does not exist yet.
    fragment_idx :
        Index of the fragments to write the cube files for.
        If None, write all fragments.
    orbital_idx :
        Index of the orbitals to write.
        If None, writes all (per) fragment.
    cubegen_kwargs :
        Keyword arguments passed to cubegen.orbital.
    """
    cube_file_path = Path(cube_file_path)
    cube_file_path.mkdir(exist_ok=True, parents=True)
    cubegen_kwargs = cubegen_kwargs if cubegen_kwargs else {}
    if not isinstance(be_object, molbe.BE):
        raise NotImplementedError("Support for Periodic BE not implemented yet.")
    if fragment_idx is None:
        fragment_idx = range(be_object.fobj.n_frag)
    for idx in fragment_idx:
        TA = be_object.Fobjs[idx].TA
        assert TA is not None
        orbital_idx = orbital_idx if orbital_idx else range(TA.shape[1])
        for emb_orb_idx in orbital_idx:
            orbital(
                be_object.fobj.mol,
                cube_file_path / f"frag_{idx}_orb_{emb_orb_idx}.cube",
                TA[:, emb_orb_idx],
                **cubegen_kwargs,
            )
