"""Remeshing policies for propagation.

In v1: use solver's existing discretize_polylines() refinement knobs; do global rebuild.
In v2+: implement local remeshing (only re-panel near modified tips).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TYPE_CHECKING

from .tip_state import TipState

if TYPE_CHECKING:
    from fracture_utils.Usolver.network import CrackNetworkV4

class RemeshPolicy(Protocol):
    def apply(self, network: "CrackNetworkV4", tip: TipState, *, delta_a: float) -> "CrackNetworkV4":
        ...

@dataclass(frozen=True)
class GlobalRemeshPolicy:
    def apply(self, network: "CrackNetworkV4", tip: TipState, *, delta_a: float) -> "CrackNetworkV4":
        _ = (tip, delta_a)
        return network
