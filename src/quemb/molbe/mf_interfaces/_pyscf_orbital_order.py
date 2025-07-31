import re
from collections.abc import Mapping, Sequence
from functools import total_ordering
from typing import Final, Literal, cast, get_args

from attrs import define
from typing_extensions import Self

M_L_VALS_S = Literal["s"]
M_L_VALS_P = Literal["px", "py", "pz"]
M_L_VALS_D = Literal["dxy", "dyz", "dz^2", "dxz", "dx2-y2"]
M_L_VALS_F = Literal["f-3", "f-2", "f-1", "f+0", "f+1", "f+2", "f+3"]
M_L_VALS_G = Literal["g-4", "g-3", "g-2", "g-1", "g+0", "g+1", "g+2", "g+3", "g+4"]
M_L_VALS_H = Literal[
    "h-5", "h-4", "h-3", "h-2", "h-1", "h+0", "h+1", "h+2", "h+3", "h+4", "h+5"
]

M_L_VALS = M_L_VALS_S | M_L_VALS_P | M_L_VALS_D | M_L_VALS_F | M_L_VALS_G | M_L_VALS_H
M_L_Values: Final[Sequence[M_L_VALS]] = (
    list(get_args(M_L_VALS_S))
    + list(get_args(M_L_VALS_P))
    + list(get_args(M_L_VALS_D))
    + list(get_args(M_L_VALS_F))
    + list(get_args(M_L_VALS_G))
    + list(get_args(M_L_VALS_H))
)

L_VALS = Literal["s", "p", "d", "f", "g", "h"]
L_Values: Final[Sequence[L_VALS]] = list(get_args(L_VALS))


PYSCF_ML: Mapping[L_VALS, list[M_L_VALS]] = {
    "s": ["s"],
    "p": ["px", "py", "pz"],
    "d": ["dxy", "dyz", "dz^2", "dxz", "dx2-y2"],
    "f": ["f-3", "f-2", "f-1", "f+0", "f+1", "f+2", "f+3"],
    "g": ["g-4", "g-3", "g-2", "g-1", "g+0", "g+1", "g+2", "g+3", "g+4"],
    "h": ["h-5", "h-4", "h-3", "h-2", "h-1", "h+0", "h+1", "h+2", "h+3", "h+4", "h+5"],
}


@total_ordering
@define(frozen=True, hash=True)
class Orbital:  # noqa: PLW1641
    """An Orbital class that can be used to represent AO labels
    in a program-independent manner.

    It implements comparison operators such that the order that follows from these
    comparison operators adheres to the pyscf order of orbitals.
    """

    idx_atom: Final[int]
    element_symbol: Final[str]

    n: Final[int]
    l: Final[L_VALS]
    m_l: Final[M_L_VALS]

    def __attrs_post_init__(self) -> None:
        allowed = PYSCF_ML[self.l]
        if self.m_l not in allowed:
            raise ValueError(
                f"Invalid m_l='{self.m_l}' for l='{self.l}'. Expected one of {allowed}."
            )

    @classmethod
    def from_pyscf_label(cls, label: str) -> Self:
        """Parse a pyscf AO label like '0 O 3dx2-y2' into an Orbital."""
        # Clean label and split
        parts = label.strip().split()
        if len(parts) != 3:
            raise ValueError(f"Invalid AO label format: {label!r}")

        idx_atom_str, element_symbol, ao_label = parts

        idx_atom = int(idx_atom_str)

        # Parse principal quantum number n: leading digits in ao_label
        m = re.match(r"(\d+)(([spdfgh]).*)", ao_label)
        if not m:
            raise ValueError(f"Cannot parse AO label orbital part: {ao_label!r}")

        n_str, m_l_str, l_char = m.groups()
        n = int(n_str)
        assert l_char in L_Values
        l = cast(L_VALS, l_char)
        assert m_l_str in M_L_Values, m_l_str

        return cls(
            idx_atom=idx_atom,
            element_symbol=element_symbol,
            n=n,
            l=l,
            m_l=cast(M_L_VALS, m_l_str),
        )

    @classmethod
    def from_orca_label(cls, label: str) -> Self:
        """Parse an ORCA AO label like '0O   1dx2y2' into an Orbital."""

        def infer_l(m_l_char: str) -> L_VALS:
            for l in L_Values:
                if m_l_char.startswith(l):
                    return l
            raise ValueError(f"Unknown orbital shape: {m_l_char!r}")

        def infer_m_l(m_l_char: str) -> M_L_VALS:
            TRANSLATE = {
                "dz2": "dz^2",
                "dx2y2": "dx2-y2",
                "f0": "f+0",
                "g0": "g+0",
                "h0": "h+0",
            }
            m_l_char = TRANSLATE.get(m_l_char, m_l_char)
            assert m_l_char in M_L_Values, (label, m_l_char)
            return cast(M_L_VALS, m_l_char)

        m = re.match(r"(\d+)([A-Z][a-z]?)\s+(\d+)([a-zA-Z0-9+\-]+)", label.strip())
        if not m:
            raise ValueError(f"Cannot parse ORCA label: {label!r}")

        idx_atom_str, element_symbol, n_str, m_l_char = m.groups()

        return cls(
            idx_atom=int(idx_atom_str),
            element_symbol=element_symbol,
            n=int(n_str),
            l=infer_l(m_l_char),
            m_l=infer_m_l(m_l_char),
        )

    def __lt__(self, other: Self) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        elif self.idx_atom != other.idx_atom:
            return self.idx_atom < other.idx_atom
        elif self.l != other.l:
            return L_Values.index(self.l) < L_Values.index(other.l)
        elif self.n != other.n:
            return self.n < other.n
        else:
            self_index = PYSCF_ML[self.l].index(self.m_l)
            other_index = PYSCF_ML[other.l].index(other.m_l)
            return self_index < other_index

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Orbital):
            return NotImplemented
        return (
            self.idx_atom == other.idx_atom
            and self.l == other.l
            and self.n == other.n
            and self.m_l == other.m_l
        )
