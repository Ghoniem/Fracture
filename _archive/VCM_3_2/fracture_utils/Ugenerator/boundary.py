"""
Boundary management for crack networks

This module provides:
- Boundary definition (linear segments, circular arcs, general curves)
- Crack-boundary intersection detection
- Crack trimming at boundaries
- Interior/exterior point testing
- Boundary integration with crack networks

Author: Nasr & AI Assistant
Date: 2025
"""

import numpy as np
import networkx as nx
from typing import Tuple, List, Optional, Union, Callable
from scipy.optimize import fsolve
from .geometry import GeometryUtils


class BoundarySegment:
    """Base class for boundary segments"""
    
    def __init__(self, label: str = "boundary"):
        self.label = label
    
    def point_at(self, t: float) -> np.ndarray:
        """
        Get point on boundary at parameter t
        
        Args:
            t: Parameter value (typically in [0, 1])
            
        Returns:
            Point [x, y] on boundary
        """
        raise NotImplementedError
    
    def tangent_at(self, t: float) -> np.ndarray:
        """Get tangent vector at parameter t"""
        raise NotImplementedError
    
    def normal_at(self, t: float) -> np.ndarray:
        """Get outward normal vector at parameter t"""
        tangent = self.tangent_at(t)
        # Outward normal (rotate tangent 90° clockwise for CCW curve)
        normal = np.array([tangent[1], -tangent[0]])
        return normal / np.linalg.norm(normal)
    
    def intersect_with_line(self, p1: np.ndarray, p2: np.ndarray) -> List[Tuple[float, np.ndarray]]:
        """
        Find intersections of boundary segment with line segment p1-p2
        
        Args:
            p1, p2: Endpoints of line segment
            
        Returns:
            List of (t, point) tuples where t is parameter on boundary segment
        """
        raise NotImplementedError
    
    def is_point_inside(self, point: np.ndarray) -> bool:
        """
        Check if point is inside the boundary
        
        Args:
            point: Point [x, y] to test
            
        Returns:
            True if point is inside
        """
        raise NotImplementedError


class LinearSegment(BoundarySegment):
    """Linear boundary segment"""
    
    def __init__(self, p1: np.ndarray, p2: np.ndarray, label: str = "boundary"):
        """
        Args:
            p1, p2: Endpoints of line segment
            label: Boundary label
        """
        super().__init__(label)
        self.p1 = np.array(p1)
        self.p2 = np.array(p2)
    
    def point_at(self, t: float) -> np.ndarray:
        """Linear interpolation"""
        return self.p1 + t * (self.p2 - self.p1)
    
    def tangent_at(self, t: float) -> np.ndarray:
        """Constant tangent for linear segment"""
        tangent = self.p2 - self.p1
        return tangent / np.linalg.norm(tangent)
    
    def intersect_with_line(self, p1: np.ndarray, p2: np.ndarray) -> List[Tuple[float, np.ndarray]]:
        """Find intersection of two line segments"""
        geom = GeometryUtils()
        intersection = geom.segments_intersect(self.p1, self.p2, p1, p2, tolerance=0.0)
        
        if intersection is not None:
            # Calculate parameter t on boundary segment
            if np.linalg.norm(self.p2 - self.p1) > 1e-10:
                t = np.linalg.norm(intersection - self.p1) / np.linalg.norm(self.p2 - self.p1)
                return [(t, intersection)]
        return []
    
    def is_point_inside(self, point: np.ndarray) -> bool:
        """For a single segment, cannot determine inside/outside"""
        raise NotImplementedError("Use Boundary.is_point_inside() instead")


class CircularArc(BoundarySegment):
    """Circular arc boundary segment"""
    
    def __init__(self, center: np.ndarray, radius: float, 
                 start_angle: float, end_angle: float, 
                 label: str = "boundary"):
        """
        Args:
            center: Center point [x, y]
            radius: Arc radius
            start_angle: Starting angle in radians
            end_angle: Ending angle in radians
            label: Boundary label
        """
        super().__init__(label)
        self.center = np.array(center)
        self.radius = radius
        self.start_angle = start_angle
        self.end_angle = end_angle
    
    def point_at(self, t: float) -> np.ndarray:
        """Point on arc at parameter t ∈ [0, 1]"""
        angle = self.start_angle + t * (self.end_angle - self.start_angle)
        return self.center + self.radius * np.array([np.cos(angle), np.sin(angle)])
    
    def tangent_at(self, t: float) -> np.ndarray:
        """Tangent vector on arc"""
        angle = self.start_angle + t * (self.end_angle - self.start_angle)
        tangent = np.array([-np.sin(angle), np.cos(angle)])
        return tangent / np.linalg.norm(tangent)
    
    def intersect_with_line(self, p1: np.ndarray, p2: np.ndarray) -> List[Tuple[float, np.ndarray]]:
        """Find intersection of arc with line segment"""
        # Line direction
        d = p2 - p1
        f = p1 - self.center
        
        # Quadratic equation coefficients for line-circle intersection
        a = np.dot(d, d)
        b = 2 * np.dot(f, d)
        c = np.dot(f, f) - self.radius**2
        
        discriminant = b**2 - 4*a*c
        
        if discriminant < 0:
            return []  # No intersection
        
        # Find intersection points on infinite line
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2*a)
        t2 = (-b + sqrt_disc) / (2*a)
        
        intersections = []
        
        for t_line in [t1, t2]:
            # Check if intersection is on line segment
            if 0 <= t_line <= 1:
                point = p1 + t_line * d
                
                # Check if point is on arc (not just circle)
                angle = np.arctan2(point[1] - self.center[1], point[0] - self.center[0])
                
                # Normalize angle to [0, 2π]
                angle = angle % (2 * np.pi)
                start = self.start_angle % (2 * np.pi)
                end = self.end_angle % (2 * np.pi)
                
                # Check if angle is within arc range
                if start <= end:
                    if start <= angle <= end:
                        t_arc = (angle - self.start_angle) / (self.end_angle - self.start_angle)
                        intersections.append((t_arc, point))
                else:  # Arc crosses 0/2π
                    if angle >= start or angle <= end:
                        t_arc = (angle - self.start_angle) / (self.end_angle - self.start_angle)
                        intersections.append((t_arc, point))
        
        return intersections
    
    def is_point_inside(self, point: np.ndarray) -> bool:
        """For a single arc, cannot determine inside/outside"""
        raise NotImplementedError("Use Boundary.is_point_inside() instead")


class Boundary:
    """
    Boundary composed of multiple segments
    
    Represents a closed or open boundary made up of linear segments and arcs
    """
    
    def __init__(self, segments: List[BoundarySegment], closed: bool = True):
        """
        Args:
            segments: List of BoundarySegment objects
            closed: Whether boundary is closed (default True)
        """
        self.segments = segments
        self.closed = closed
        self._validate()
    
    def _validate(self):
        """Check if boundary segments are connected"""
        if not self.closed or len(self.segments) < 2:
            return
        
        tol = 1e-6
        for i in range(len(self.segments)):
            seg1 = self.segments[i]
            seg2 = self.segments[(i+1) % len(self.segments)]
            
            p1_end = seg1.point_at(1.0)
            p2_start = seg2.point_at(0.0)
            
            dist = np.linalg.norm(p1_end - p2_start)
            if dist > tol:
                print(f"Warning: Gap of {dist:.6e} between segments {i} and {i+1}")
    
    def is_point_inside(self, point: np.ndarray) -> bool:
        """
        Check if point is inside closed boundary using ray casting
        
        Args:
            point: Point [x, y] to test
            
        Returns:
            True if point is inside boundary
        """
        if not self.closed:
            raise ValueError("Cannot determine inside/outside for open boundary")
        
        # Ray casting algorithm: cast ray to the right
        ray_start = point
        ray_end = point + np.array([1e6, 0])  # Ray to far right
        
        intersection_count = 0
        
        for segment in self.segments:
            intersections = segment.intersect_with_line(ray_start, ray_end)
            intersection_count += len(intersections)
        
        # Odd number of intersections = inside
        return intersection_count % 2 == 1
    
    def distance_to_point(self, point: np.ndarray) -> float:
        """
        Calculate minimum distance from point to boundary
        
        Args:
            point: Point [x, y]
            
        Returns:
            Minimum distance to boundary
        """
        min_dist = np.inf
        
        for segment in self.segments:
            # Sample segment densely
            n_samples = 100
            for i in range(n_samples + 1):
                t = i / n_samples
                seg_point = segment.point_at(t)
                dist = np.linalg.norm(point - seg_point)
                min_dist = min(min_dist, dist)
        
        return min_dist
    
    def intersect_with_line(self, p1: np.ndarray, p2: np.ndarray) -> List[Tuple[int, float, np.ndarray]]:
        """
        Find all intersections of line segment with boundary
        
        Args:
            p1, p2: Endpoints of line segment
            
        Returns:
            List of (segment_index, t_boundary, point) tuples
        """
        all_intersections = []
        
        for i, segment in enumerate(self.segments):
            intersections = segment.intersect_with_line(p1, p2)
            for t_boundary, point in intersections:
                all_intersections.append((i, t_boundary, point))
        
        return all_intersections
    
    def to_vertices_connectivity(self, n_points_per_segment: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert boundary to discrete vertices and connectivity
        
        Args:
            n_points_per_segment: Number of points to sample per segment
            
        Returns:
            vertices: (n_vertices, 3) array with [id, x, y]
            connectivity: (n_edges, 3) array with [edge_id, v0, v1]
        """
        vertices = []
        connectivity = []
        vertex_id = 0
        edge_id = 0
        
        for segment in self.segments:
            segment_start_id = vertex_id
            
            for i in range(n_points_per_segment):
                t = i / n_points_per_segment
                point = segment.point_at(t)
                vertices.append([vertex_id, point[0], point[1]])
                
                if i > 0:
                    connectivity.append([edge_id, vertex_id - 1, vertex_id])
                    edge_id += 1
                
                vertex_id += 1
            
            # Connect last point of segment to first point of next segment
            if self.closed:
                # Will be handled by connecting last segment to first
                pass
        
        # Close the boundary if closed
        if self.closed and len(vertices) > 0:
            connectivity.append([edge_id, vertex_id - 1, 0])
        
        return np.array(vertices), np.array(connectivity)
    
    @staticmethod
    def create_circle(center: np.ndarray, radius: float, label: str = "boundary") -> 'Boundary':
        """
        Create circular boundary
        
        Args:
            center: Center point [x, y]
            radius: Circle radius
            label: Boundary label
            
        Returns:
            Boundary object representing circle
        """
        arc = CircularArc(center, radius, 0, 2*np.pi, label)
        return Boundary([arc], closed=True)
    
    @staticmethod
    def create_rectangle(xmin: float, xmax: float, ymin: float, ymax: float, 
                        label: str = "boundary") -> 'Boundary':
        """
        Create rectangular boundary
        
        Args:
            xmin, xmax: X-axis bounds
            ymin, ymax: Y-axis bounds
            label: Boundary label
            
        Returns:
            Boundary object representing rectangle
        """
        corners = [
            np.array([xmin, ymin]),
            np.array([xmax, ymin]),
            np.array([xmax, ymax]),
            np.array([xmin, ymax])
        ]
        
        segments = []
        for i in range(4):
            p1 = corners[i]
            p2 = corners[(i+1) % 4]
            segments.append(LinearSegment(p1, p2, label))
        
        return Boundary(segments, closed=True)
    
    @staticmethod
    def create_polygon(vertices: List[np.ndarray], label: str = "boundary") -> 'Boundary':
        """
        Create polygonal boundary
        
        Args:
            vertices: List of vertex positions [x, y]
            label: Boundary label
            
        Returns:
            Boundary object representing polygon
        """
        segments = []
        for i in range(len(vertices)):
            p1 = vertices[i]
            p2 = vertices[(i+1) % len(vertices)]
            segments.append(LinearSegment(p1, p2, label))
        
        return Boundary(segments, closed=True)


class BoundaryManager:
    """
    Manages boundary-crack network integration
    
    Handles:
    - Adding boundary to crack network graph
    - Detecting crack-boundary intersections
    - Trimming cracks that extend outside boundary
    - Preventing crack generation outside boundary
    """
    
    def __init__(self, boundary: Boundary):
        """
        Args:
            boundary: Boundary object
        """
        self.boundary = boundary
        self.geom = GeometryUtils()
    
    def add_boundary_to_graph(self, graph: nx.Graph, n_points_per_segment: int = 50) -> nx.Graph:
        """
        Add boundary edges to crack network graph
        
        Args:
            graph: Crack network graph
            n_points_per_segment: Discretization density
            
        Returns:
            Modified graph with boundary included
        """
        vertices, connectivity = self.boundary.to_vertices_connectivity(n_points_per_segment)
        
        # Get current max node ID
        if len(graph.nodes()) > 0:
            max_id = max(graph.nodes())
        else:
            max_id = -1
        
        # Add boundary vertices
        id_offset = max_id + 1
        for vertex in vertices:
            node_id = int(vertex[0]) + id_offset
            pos = (vertex[1], vertex[2])
            graph.add_node(node_id, pos=pos, type='boundary')
        
        # Add boundary edges
        for edge in connectivity:
            v0 = int(edge[1]) + id_offset
            v1 = int(edge[2]) + id_offset
            edge_vec = np.array(graph.nodes[v1]['pos']) - np.array(graph.nodes[v0]['pos'])
            length = np.linalg.norm(edge_vec)
            graph.add_edge(v0, v1, length=length, type='boundary')
        
        return graph
    
    def trim_cracks_at_boundary(self, graph: nx.Graph, verbose: bool = True) -> nx.Graph:
        """
        Trim crack edges that extend outside boundary
        
        Args:
            graph: Crack network graph
            verbose: Print trimming information
            
        Returns:
            Modified graph with trimmed cracks
        """
        if verbose:
            print("Trimming cracks at boundary...")
        
        edges_to_process = []
        
        # Find all crack edges (not boundary edges)
        for u, v, data in graph.edges(data=True):
            if data.get('type') != 'boundary':
                edges_to_process.append((u, v))
        
        edges_removed = 0
        edges_trimmed = 0
        nodes_added = 0
        
        for u, v in edges_to_process:
            pos_u = np.array(graph.nodes[u]['pos'])
            pos_v = np.array(graph.nodes[v]['pos'])
            
            inside_u = self.boundary.is_point_inside(pos_u)
            inside_v = self.boundary.is_point_inside(pos_v)
            
            if not inside_u and not inside_v:
                # Both endpoints outside - remove edge
                if graph.has_edge(u, v):
                    graph.remove_edge(u, v)
                    edges_removed += 1
            
            elif inside_u and not inside_v:
                # u inside, v outside - trim at boundary
                intersections = self.boundary.intersect_with_line(pos_u, pos_v)
                
                if intersections:
                    # Use first intersection point
                    _, _, int_point = intersections[0]
                    
                    # Create new boundary node
                    new_id = max(graph.nodes()) + 1
                    graph.add_node(new_id, pos=tuple(int_point), type='boundary_intersection')
                    
                    # Replace edge
                    if graph.has_edge(u, v):
                        graph.remove_edge(u, v)
                    
                    new_length = np.linalg.norm(int_point - pos_u)
                    graph.add_edge(u, new_id, length=new_length)
                    
                    edges_trimmed += 1
                    nodes_added += 1
            
            elif not inside_u and inside_v:
                # u outside, v inside - trim at boundary
                intersections = self.boundary.intersect_with_line(pos_u, pos_v)
                
                if intersections:
                    _, _, int_point = intersections[-1]  # Use last intersection
                    
                    new_id = max(graph.nodes()) + 1
                    graph.add_node(new_id, pos=tuple(int_point), type='boundary_intersection')
                    
                    if graph.has_edge(u, v):
                        graph.remove_edge(u, v)
                    
                    new_length = np.linalg.norm(pos_v - int_point)
                    graph.add_edge(new_id, v, length=new_length)
                    
                    edges_trimmed += 1
                    nodes_added += 1
        
        # Remove isolated nodes (nodes with degree 0)
        isolated = [n for n in graph.nodes() if graph.degree(n) == 0 
                   and graph.nodes[n].get('type') != 'boundary']
        graph.remove_nodes_from(isolated)
        
        if verbose:
            print(f"  Removed {edges_removed} edges completely outside")
            print(f"  Trimmed {edges_trimmed} edges at boundary")
            print(f"  Added {nodes_added} boundary intersection nodes")
            print(f"  Removed {len(isolated)} isolated nodes")
        
        return graph
    
    def filter_positions_inside_boundary(self, positions: np.ndarray) -> np.ndarray:
        """
        Filter positions to keep only those inside boundary
        
        Args:
            positions: Array of positions (n, 2)
            
        Returns:
            Filtered positions
        """
        inside_positions = []
        for pos in positions:
            if self.boundary.is_point_inside(pos):
                inside_positions.append(pos)
        
        return np.array(inside_positions) if inside_positions else np.empty((0, 2))
