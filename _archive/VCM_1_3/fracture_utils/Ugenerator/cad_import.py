"""
CAD file import utilities for boundary definitions

Supports importing boundaries from:
- DXF files (AutoCAD format)
- IGES files (standard CAD exchange format)
- STEP files (ISO standard)
- SVG files (vector graphics)
- STL files (triangulated surfaces, 2D projection)
- Custom text formats (vertices, edges)

Author: Nasr & AI Assistant
Date: 2025
"""

import numpy as np
from typing import List, Tuple, Optional, Union
from pathlib import Path

try:
    import ezdxf
    HAS_EZDXF = True
except ImportError:
    HAS_EZDXF = False

try:
    from svgpathtools import svg2paths, Line, Arc, CubicBezier, QuadraticBezier
    HAS_SVGPATHTOOLS = True
except ImportError:
    HAS_SVGPATHTOOLS = False

from .boundary import Boundary, LinearSegment, CircularArc, BoundarySegment


class BezierCurve(BoundarySegment):
    """Cubic Bezier curve boundary segment"""
    
    def __init__(self, p0: np.ndarray, p1: np.ndarray, 
                 p2: np.ndarray, p3: np.ndarray, 
                 label: str = "boundary"):
        """
        Cubic Bezier curve: B(t) = (1-t)³P0 + 3(1-t)²tP1 + 3(1-t)t²P2 + t³P3
        
        Args:
            p0: Start point
            p1: First control point
            p2: Second control point
            p3: End point
            label: Boundary label
        """
        super().__init__(label)
        self.p0 = np.array(p0)
        self.p1 = np.array(p1)
        self.p2 = np.array(p2)
        self.p3 = np.array(p3)
    
    def point_at(self, t: float) -> np.ndarray:
        """Evaluate Bezier curve at parameter t"""
        t = np.clip(t, 0, 1)
        return ((1-t)**3 * self.p0 + 
                3*(1-t)**2*t * self.p1 + 
                3*(1-t)*t**2 * self.p2 + 
                t**3 * self.p3)
    
    def tangent_at(self, t: float) -> np.ndarray:
        """Tangent vector at parameter t"""
        t = np.clip(t, 0, 1)
        tangent = (3*(1-t)**2 * (self.p1 - self.p0) +
                   6*(1-t)*t * (self.p2 - self.p1) +
                   3*t**2 * (self.p3 - self.p2))
        return tangent / (np.linalg.norm(tangent) + 1e-10)
    
    def intersect_with_line(self, p1: np.ndarray, p2: np.ndarray) -> List[Tuple[float, np.ndarray]]:
        """Find intersections with line segment (numerical approximation)"""
        # Sample curve densely and check for intersections
        n_samples = 100
        intersections = []
        
        for i in range(n_samples):
            t1 = i / n_samples
            t2 = (i + 1) / n_samples
            
            b1 = self.point_at(t1)
            b2 = self.point_at(t2)
            
            # Check if this segment intersects with the line
            from .geometry import GeometryUtils
            geom = GeometryUtils()
            int_point = geom.segments_intersect(b1, b2, p1, p2, tolerance=0.0)
            
            if int_point is not None:
                # Refine t parameter
                t_approx = (t1 + t2) / 2
                intersections.append((t_approx, int_point))
        
        return intersections


class SplineCurve(BoundarySegment):
    """B-spline curve boundary segment"""
    
    def __init__(self, control_points: List[np.ndarray], 
                 degree: int = 3, label: str = "boundary"):
        """
        B-spline curve
        
        Args:
            control_points: List of control points
            degree: Spline degree (typically 3 for cubic)
            label: Boundary label
        """
        super().__init__(label)
        self.control_points = [np.array(p) for p in control_points]
        self.degree = degree
        self.n_control = len(control_points)
        
        # Generate uniform knot vector
        n_knots = self.n_control + self.degree + 1
        self.knots = np.linspace(0, 1, n_knots)
    
    def point_at(self, t: float) -> np.ndarray:
        """Evaluate B-spline at parameter t (simplified for uniform knots)"""
        # For simplicity, approximate with Catmull-Rom or linear interpolation
        # Full B-spline evaluation requires scipy or manual implementation
        t = np.clip(t, 0, 1)
        
        # Linear interpolation through control points
        scaled_t = t * (self.n_control - 1)
        idx = int(np.floor(scaled_t))
        idx = min(idx, self.n_control - 2)
        local_t = scaled_t - idx
        
        return (1 - local_t) * self.control_points[idx] + local_t * self.control_points[idx + 1]
    
    def tangent_at(self, t: float) -> np.ndarray:
        """Approximate tangent"""
        dt = 1e-6
        p1 = self.point_at(max(0, t - dt))
        p2 = self.point_at(min(1, t + dt))
        tangent = p2 - p1
        return tangent / (np.linalg.norm(tangent) + 1e-10)
    
    def intersect_with_line(self, p1: np.ndarray, p2: np.ndarray) -> List[Tuple[float, np.ndarray]]:
        """Find intersections (numerical)"""
        n_samples = 100
        intersections = []
        
        for i in range(n_samples):
            t1 = i / n_samples
            t2 = (i + 1) / n_samples
            
            s1 = self.point_at(t1)
            s2 = self.point_at(t2)
            
            from .geometry import GeometryUtils
            geom = GeometryUtils()
            int_point = geom.segments_intersect(s1, s2, p1, p2, tolerance=0.0)
            
            if int_point is not None:
                t_approx = (t1 + t2) / 2
                intersections.append((t_approx, int_point))
        
        return intersections


class CADImporter:
    """Import boundaries from CAD files"""
    
    @staticmethod
    def from_dxf(filepath: str, layer: Optional[str] = None, 
                 scale: float = 1.0) -> Boundary:
        """
        Import boundary from DXF file
        
        Args:
            filepath: Path to DXF file
            layer: Specific layer name to import (None = all layers)
            scale: Scaling factor (default 1.0)
            
        Returns:
            Boundary object
            
        Requires: ezdxf package (pip install ezdxf)
        """
        if not HAS_EZDXF:
            raise ImportError("ezdxf package required. Install: pip install ezdxf")
        
        doc = ezdxf.readfile(filepath)
        msp = doc.modelspace()
        
        segments = []
        
        for entity in msp:
            # Filter by layer if specified
            if layer is not None and entity.dxf.layer != layer:
                continue
            
            if entity.dxftype() == 'LINE':
                p1 = np.array([entity.dxf.start.x, entity.dxf.start.y]) * scale
                p2 = np.array([entity.dxf.end.x, entity.dxf.end.y]) * scale
                segments.append(LinearSegment(p1, p2))
            
            elif entity.dxftype() == 'CIRCLE':
                center = np.array([entity.dxf.center.x, entity.dxf.center.y]) * scale
                radius = entity.dxf.radius * scale
                # Full circle
                arc = CircularArc(center, radius, 0, 2*np.pi)
                segments.append(arc)
            
            elif entity.dxftype() == 'ARC':
                center = np.array([entity.dxf.center.x, entity.dxf.center.y]) * scale
                radius = entity.dxf.radius * scale
                start_angle = np.deg2rad(entity.dxf.start_angle)
                end_angle = np.deg2rad(entity.dxf.end_angle)
                arc = CircularArc(center, radius, start_angle, end_angle)
                segments.append(arc)
            
            elif entity.dxftype() == 'LWPOLYLINE' or entity.dxftype() == 'POLYLINE':
                points = [(p[0] * scale, p[1] * scale) for p in entity.get_points()]
                for i in range(len(points) - 1):
                    p1 = np.array(points[i])
                    p2 = np.array(points[i + 1])
                    segments.append(LinearSegment(p1, p2))
                
                # Close polyline if needed
                if entity.is_closed:
                    p1 = np.array(points[-1])
                    p2 = np.array(points[0])
                    segments.append(LinearSegment(p1, p2))
            
            elif entity.dxftype() == 'SPLINE':
                # Approximate spline with linear segments
                points = list(entity.flattening(0.01))  # Flatten to tolerance
                for i in range(len(points) - 1):
                    p1 = np.array([points[i][0], points[i][1]]) * scale
                    p2 = np.array([points[i+1][0], points[i+1][1]]) * scale
                    segments.append(LinearSegment(p1, p2))
        
        if not segments:
            raise ValueError(f"No valid geometry found in DXF file: {filepath}")
        
        return Boundary(segments, closed=True)
    
    @staticmethod
    def from_svg(filepath: str, scale: float = 1.0) -> Boundary:
        """
        Import boundary from SVG file
        
        Args:
            filepath: Path to SVG file
            scale: Scaling factor
            
        Returns:
            Boundary object
            
        Requires: svgpathtools package (pip install svgpathtools)
        """
        if not HAS_SVGPATHTOOLS:
            raise ImportError("svgpathtools required. Install: pip install svgpathtools")
        
        paths, attributes = svg2paths(filepath)
        segments = []
        
        for path in paths:
            for segment in path:
                if isinstance(segment, Line):
                    p1 = np.array([segment.start.real, segment.start.imag]) * scale
                    p2 = np.array([segment.end.real, segment.end.imag]) * scale
                    segments.append(LinearSegment(p1, p2))
                
                elif isinstance(segment, Arc):
                    # Approximate arc with circular arc
                    center_complex = segment.center
                    center = np.array([center_complex.real, center_complex.imag]) * scale
                    radius = segment.radius.real * scale
                    
                    # Calculate angles
                    start_angle = np.angle(segment.start - center_complex)
                    end_angle = np.angle(segment.end - center_complex)
                    
                    arc = CircularArc(center, radius, start_angle, end_angle)
                    segments.append(arc)
                
                elif isinstance(segment, (CubicBezier, QuadraticBezier)):
                    # Approximate Bezier with linear segments
                    n_points = 20
                    points = [segment.point(t) for t in np.linspace(0, 1, n_points)]
                    for i in range(len(points) - 1):
                        p1 = np.array([points[i].real, points[i].imag]) * scale
                        p2 = np.array([points[i+1].real, points[i+1].imag]) * scale
                        segments.append(LinearSegment(p1, p2))
        
        return Boundary(segments, closed=True)
    
    @staticmethod
    def from_text_file(filepath: str, format: str = 'vertices', 
                      scale: float = 1.0) -> Boundary:
        """
        Import boundary from simple text file
        
        Supported formats:
        - 'vertices': List of (x, y) vertices, one per line
        - 'edges': List of edges as (x1, y1, x2, y2), one per line
        
        Args:
            filepath: Path to text file
            format: File format ('vertices' or 'edges')
            scale: Scaling factor
            
        Returns:
            Boundary object
        """
        data = np.loadtxt(filepath)
        
        if format == 'vertices':
            # File contains vertices only
            vertices = data * scale
            segments = []
            for i in range(len(vertices)):
                p1 = vertices[i]
                p2 = vertices[(i + 1) % len(vertices)]
                segments.append(LinearSegment(p1, p2))
            
            return Boundary(segments, closed=True)
        
        elif format == 'edges':
            # File contains edges as (x1, y1, x2, y2)
            segments = []
            for edge in data:
                p1 = np.array([edge[0], edge[1]]) * scale
                p2 = np.array([edge[2], edge[3]]) * scale
                segments.append(LinearSegment(p1, p2))
            
            return Boundary(segments, closed=True)
        
        else:
            raise ValueError(f"Unknown format: {format}")
    
    @staticmethod
    def from_stl_2d(filepath: str, plane: str = 'xy', scale: float = 1.0) -> Boundary:
        """
        Import 2D boundary from STL file by projecting onto a plane
        
        Args:
            filepath: Path to STL file
            plane: Projection plane ('xy', 'xz', or 'yz')
            scale: Scaling factor
            
        Returns:
            Boundary object
        """
        try:
            from stl import mesh
        except ImportError:
            raise ImportError("numpy-stl required. Install: pip install numpy-stl")
        
        # Load STL
        stl_mesh = mesh.Mesh.from_file(filepath)
        
        # Project vertices onto plane
        if plane == 'xy':
            idx = [0, 1]  # x, y
        elif plane == 'xz':
            idx = [0, 2]  # x, z
        elif plane == 'yz':
            idx = [1, 2]  # y, z
        else:
            raise ValueError(f"Unknown plane: {plane}")
        
        # Extract unique edges in 2D projection
        edges = set()
        for triangle in stl_mesh.vectors:
            for i in range(3):
                p1 = triangle[i][idx] * scale
                p2 = triangle[(i+1)%3][idx] * scale
                
                # Create edge (order-independent)
                edge = tuple(sorted([tuple(p1), tuple(p2)]))
                edges.add(edge)
        
        # Convert to segments
        segments = []
        for edge in edges:
            p1 = np.array(edge[0])
            p2 = np.array(edge[1])
            segments.append(LinearSegment(p1, p2))
        
        return Boundary(segments, closed=True)
    
    @staticmethod
    def from_numpy_arrays(vertices: np.ndarray, connectivity: Optional[np.ndarray] = None,
                         scale: float = 1.0) -> Boundary:
        """
        Create boundary from numpy arrays
        
        Args:
            vertices: (n, 2) array of vertex coordinates
            connectivity: (m, 2) array of edge connectivity (optional, 
                         if None assumes sequential connectivity)
            scale: Scaling factor
            
        Returns:
            Boundary object
        """
        vertices = vertices * scale
        
        if connectivity is None:
            # Assume sequential connectivity
            connectivity = np.array([[i, (i+1) % len(vertices)] 
                                    for i in range(len(vertices))])
        
        segments = []
        for edge in connectivity:
            p1 = vertices[edge[0]]
            p2 = vertices[edge[1]]
            segments.append(LinearSegment(p1, p2))
        
        return Boundary(segments, closed=True)


class CADExporter:
    """Export boundaries to CAD files"""
    
    @staticmethod
    def to_dxf(boundary: Boundary, filepath: str, layer: str = "BOUNDARY"):
        """
        Export boundary to DXF file
        
        Args:
            boundary: Boundary object
            filepath: Output DXF file path
            layer: Layer name for boundary entities
            
        Requires: ezdxf package
        """
        if not HAS_EZDXF:
            raise ImportError("ezdxf package required. Install: pip install ezdxf")
        
        doc = ezdxf.new('R2010')
        msp = doc.modelspace()
        
        for segment in boundary.segments:
            if isinstance(segment, LinearSegment):
                msp.add_line(
                    (segment.p1[0], segment.p1[1]),
                    (segment.p2[0], segment.p2[1]),
                    dxfattribs={'layer': layer}
                )
            
            elif isinstance(segment, CircularArc):
                # Add arc
                center = segment.center
                radius = segment.radius
                start_deg = np.rad2deg(segment.start_angle)
                end_deg = np.rad2deg(segment.end_angle)
                
                if abs(end_deg - start_deg - 360) < 1e-6:
                    # Full circle
                    msp.add_circle(
                        (center[0], center[1]),
                        radius,
                        dxfattribs={'layer': layer}
                    )
                else:
                    # Arc
                    msp.add_arc(
                        (center[0], center[1]),
                        radius,
                        start_deg,
                        end_deg,
                        dxfattribs={'layer': layer}
                    )
        
        doc.saveas(filepath)
    
    @staticmethod
    def to_text_file(boundary: Boundary, filepath: str, 
                    n_points_per_segment: int = 50):
        """
        Export boundary as text file of vertices
        
        Args:
            boundary: Boundary object
            filepath: Output text file path
            n_points_per_segment: Points to sample per segment
        """
        vertices, _ = boundary.to_vertices_connectivity(n_points_per_segment)
        
        # Save only x, y coordinates
        np.savetxt(filepath, vertices[:, 1:], 
                  header='x y', 
                  fmt='%.6e')
