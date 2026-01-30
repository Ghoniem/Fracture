"""
Crack Network Generator Package (Ugenerator)

A comprehensive toolkit for generating and analyzing crack networks in materials science.

Main Components:
- CrackNetworkGenerator: Main class for generating crack networks
- Geometry utilities: Intersection detection, distance calculations
- Network analysis tools
- Visualization utilities
- Boundary management: Define boundaries, trim cracks, detect intersections
- CAD import/export: Import boundaries from DXF, SVG, STL, text files

Author: Nasr & AI Assistant
Date: 2025
"""

from .generator import CrackNetworkGenerator
from .geometry import GeometryUtils, polyline_arc_network
from .analysis import NetworkAnalyzer
from .visualization import NetworkVisualizer
from .boundary import (
    Boundary, 
    BoundarySegment, 
    LinearSegment, 
    CircularArc, 
    BoundaryManager
)
from .cad_import import (
    CADImporter,
    CADExporter,
    BezierCurve,
    SplineCurve
)

__version__ = "1.1.0"
__all__ = [
    'CrackNetworkGenerator',
    'GeometryUtils',
    'polyline_arc_network',
    'NetworkAnalyzer',
    'NetworkVisualizer',
    'Boundary',
    'BoundarySegment',
    'LinearSegment',
    'CircularArc',
    'BoundaryManager',
    'CADImporter',
    'CADExporter',
    'BezierCurve',
    'SplineCurve',
]
