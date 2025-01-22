from collections.abc import Sequence
from pathlib import Path

from pyscf.tools.cubegen import orbital

from quemb import molbe
from quemb.shared.typing import KwargDict, PathLike


def write_cube(
    be_object: molbe.BE,
    cube_file_path: PathLike,
    fragment_idx: Sequence[int] | None = None,
    cubegen_kwargs: KwargDict | None = None,
) -> None:
    """Write cube files of embedding orbitals from a BE object.

    Parameters
    ----------
    be_object
        BE object containing the fragments, each of which contains embedding orbitals.
    cube_file_path
        Directory to write the cube files to.
    fragment_idx
        Index of the fragments to write the cube files for.
        If None, write all fragments.
    cubegen_kwargs
        Keyword arguments passed to cubegen.orbital.
    """
    cube_file_path = Path(cube_file_path)
    cubegen_kwargs = cubegen_kwargs if cubegen_kwargs else {}
    if not isinstance(be_object, molbe.BE):
        raise NotImplementedError("Support for Periodic BE not implemented yet.")
    if fragment_idx is None:
        fragment_idx = range(be_object.fobj.Nfrag)
    for idx in fragment_idx:
        for emb_orb_idx in range(be_object.Fobjs[idx].TA.shape[1]):
            orbital(
                be_object.fobj.mol,
                cube_file_path / f"frag_{idx}_orb_{emb_orb_idx}.cube",
                be_object.Fobjs[idx].TA[:, emb_orb_idx],
                **cubegen_kwargs,
            )
