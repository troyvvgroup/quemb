from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy
from pyscf.tools.cubegen import orbital

from quemb import molbe
from quemb.shared.typing import KwargDict, Matrix, PathLike


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
        fragment_idx = range(be_object.Nfrag)
    for idx in fragment_idx:
        if be_object.Fobjs[idx].TA is None:
            raise ValueError
        else:
            tmp = cast(Matrix[numpy.float64], be_object.Fobjs[idx].TA)
            for emb_orb_idx in range(tmp.shape[1]):
                orbital(
                    be_object.mol,
                    cube_file_path / f"frag_{idx}_orb_{emb_orb_idx}.cube",
                    tmp[:, emb_orb_idx],
                    **cubegen_kwargs,
                )
