"""Public solver API (facade).

This module preserves the original import surface:
    from utils.solver import Material, AppliedStress, CrackNetworkV4, DCENetworkStaticV4
"""
from .solver_material import Material, Crack, AppliedStress
from .solver_network import VertexV4, EdgeV4, CrackNetworkV4
from .solver_parametrized import DCENetworkStaticV4

__all__ = [
    "Material", "Crack", "AppliedStress",
    "VertexV4", "EdgeV4", "CrackNetworkV4",
    "DCENetworkStaticV4",
]
