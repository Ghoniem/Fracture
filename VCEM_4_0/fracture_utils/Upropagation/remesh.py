"""Remeshing policies for propagation.

In v1: use solver's existing discretize_polylines() refinement knobs; do global rebuild.
In v2+: implement local remeshing (only re-panel near modified tips).

All policies expose ``remesh_network(network, *, n_crack_elements=None)`` so the propagator
has a single uniform entry point. ``n_crack_elements`` is the solver's half-element budget
along each polyline; policies may ignore it (e.g. the global no-op below).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from fracture_utils.Usolver.network import CrackNetworkV4

class RemeshPolicy(Protocol):
    def remesh_network(
        self,
        network: "CrackNetworkV4",
        *,
        n_crack_elements: Optional[int] = None,
    ) -> "CrackNetworkV4":
        ...

@dataclass(frozen=True)
class GlobalRemeshPolicy:
    """No-op policy: returns the network unchanged.

    Intended as a placeholder when global remeshing is handled elsewhere
    (e.g. by re-running the solver with refreshed discretization kwargs).
    """

    def remesh_network(
        self,
        network: "CrackNetworkV4",
        *,
        n_crack_elements: Optional[int] = None,
    ) -> "CrackNetworkV4":
        _ = n_crack_elements
        return network
