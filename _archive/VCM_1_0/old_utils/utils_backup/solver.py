"""Public solver API (facade) for utils_half.

This module preserves the original import surface:
    from utils_half.solver import Material, AppliedStress, CrackNetworkV4, DCENetworkStaticV4

but routes to the *_half implementations where appropriate.
"""
from .solver_material import Material, Crack, AppliedStress
from .solver_network import VertexV4, EdgeV4, CrackNetworkV4

# IMPORTANT:
# - Keep the class name DCENetworkStaticV4 the same to preserve the public API.
# - The implementation should accept an additional option for half cracks
#   (e.g., crack_mode="half") while retaining crack_mode="full".
from .solver_parametrized_half import DCENetworkStaticV4

__all__ = [
    "Material", "Crack", "AppliedStress",
    "VertexV4", "EdgeV4", "CrackNetworkV4",
    "DCENetworkStaticV4",
]
