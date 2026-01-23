"""
Complete Crack Network Generator
=================================

Supports BOTH:
1. Manual loading from vertices/connectivity (load_from_arrays, etc.)
2. Random generation (generate_network)

Usage:
------
# Manual loading:
generator.load_from_arrays(vertices, connectivity, has_node_ids=True)

# Random generation:
generator.generate_network(n_tips=15, edge_length_mean=40*mm, ...)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable, Optional, Union
import networkx as nx
from scipy.spatial import Delaunay


class CrackNetworkGenerator:
    """
    Complete crack network generator supporting both manual specification
    and random generation of crack geometries.
    """
    
    def __init__(self, domain_size=(1.0, 1.0), seed=None):
        """
        Initialize the crack network generator.
        
        Parameters
        ----------
        domain_size : tuple
            (width, height) of the domain
        seed : int, optional
            Random seed for reproducibility
        """
        self.domain_size = domain_size
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
        
        # Network data
        self.vertices = []  # List of (x, y) coordinates
        self.edges = []     # List of (i, j) vertex indices
        self.graph = nx.Graph()
        
        # Metadata
        self.edge_metadata = {}  # Additional info per edge
        self.vertex_metadata = {}  # Additional info per vertex
        
        # Node ID mapping (for manual loading with node IDs)
        self.node_id_to_index = None
        self.index_to_node_id = None
    
    # ========================================================================
    # MANUAL LOADING METHODS
    # ========================================================================
    
    def load_from_arrays(self, vertices: np.ndarray, connectivity: List[List[int]], 
                        vertex_labels: Optional[List[str]] = None,
                        edge_labels: Optional[List[str]] = None,
                        has_node_ids: bool = True):
        """
        Load crack network from vertex array and connectivity list.
        
        Parameters
        ----------
        vertices : np.ndarray
            Array of vertex data. Can be either:
            - shape (n_vertices, 3): [[id, x, y], ...] if has_node_ids=True
            - shape (n_vertices, 2): [[x, y], ...] if has_node_ids=False
        connectivity : list of [int, int]
            List of edges as vertex index pairs
        has_node_ids : bool, default True
            If True, first column of vertices is node ID
        """
        vertices = np.asarray(vertices, dtype=float)
        
        if has_node_ids:
            if vertices.ndim != 2 or vertices.shape[1] != 3:
                raise ValueError("With has_node_ids=True, vertices must be shape (n, 3) with format [id, x, y]")
            
            node_ids = vertices[:, 0].astype(int)
            coords = vertices[:, 1:]
            
            if len(node_ids) != len(set(node_ids)):
                raise ValueError("Duplicate node IDs found in vertices array")
            
            self.node_id_to_index = {node_id: i for i, node_id in enumerate(node_ids)}
            self.index_to_node_id = {i: node_id for i, node_id in enumerate(node_ids)}
            
            n_vertices = len(node_ids)
            
            for i, edge in enumerate(connectivity):
                if len(edge) != 2:
                    raise ValueError(f"Edge {i} must have exactly 2 vertices")
                v1, v2 = edge
                if v1 not in self.node_id_to_index or v2 not in self.node_id_to_index:
                    raise ValueError(f"Edge {i} references unknown node ID: {edge}")
            
            connectivity_internal = []
            for v1_id, v2_id in connectivity:
                idx1 = self.node_id_to_index[v1_id]
                idx2 = self.node_id_to_index[v2_id]
                connectivity_internal.append([idx1, idx2])
        else:
            if vertices.ndim != 2 or vertices.shape[1] != 2:
                raise ValueError("With has_node_ids=False, vertices must be shape (n, 2)")
            
            coords = vertices
            n_vertices = len(coords)
            self.node_id_to_index = None
            self.index_to_node_id = None
            
            for i, edge in enumerate(connectivity):
                if len(edge) != 2:
                    raise ValueError(f"Edge {i} must have exactly 2 vertices")
                v1, v2 = edge
                if v1 < 0 or v1 >= n_vertices or v2 < 0 or v2 >= n_vertices:
                    raise ValueError(f"Edge {i} references invalid vertex index: {edge}")
            
            connectivity_internal = connectivity
        
        self.vertices = [tuple(v) for v in coords]
        self.edges = [tuple(e) for e in connectivity_internal]
        
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(n_vertices))
        self.graph.add_edges_from(connectivity_internal)
        
        if vertex_labels is not None:
            for i, label in enumerate(vertex_labels):
                self.vertex_metadata[i] = {'label': label}
        
        if edge_labels is not None:
            for i, label in enumerate(edge_labels):
                edge = tuple(connectivity_internal[i])
                self.edge_metadata[edge] = {'label': label}
        
        print(f"✓ Loaded network: {n_vertices} vertices, {len(connectivity_internal)} edges")
        if has_node_ids:
            print(f"  Node IDs: {sorted(self.node_id_to_index.keys())}")
        self._classify_vertices()
    
    def load_from_edge_list(self, edges: List[List[float]], tolerance: float = 1e-10):
        """Load from list of edges [x1, y1, x2, y2]"""
        vertex_list = []
        
        def get_or_create_vertex(x, y):
            for idx, (vx, vy) in enumerate(vertex_list):
                if np.sqrt((x - vx)**2 + (y - vy)**2) < tolerance:
                    return idx
            idx = len(vertex_list)
            vertex_list.append((x, y))
            return idx
        
        connectivity = []
        for edge in edges:
            if len(edge) != 4:
                raise ValueError(f"Each edge must have 4 values [x1, y1, x2, y2]")
            x1, y1, x2, y2 = edge
            v1 = get_or_create_vertex(x1, y1)
            v2 = get_or_create_vertex(x2, y2)
            if v1 != v2:
                connectivity.append([v1, v2])
        
        vertices = np.array(vertex_list)
        self.load_from_arrays(vertices, connectivity, has_node_ids=False)
    
    def load_parametric_path(self, path_function: Callable, 
                            t_range: Tuple[float, float],
                            n_segments: int = 20, **kwargs):
        """Create crack from parametric function"""
        t_values = np.linspace(t_range[0], t_range[1], n_segments + 1)
        points = [path_function(t, **kwargs) for t in t_values]
        edges = []
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            edges.append([x1, y1, x2, y2])
        self.load_from_edge_list(edges)
    
    def load_multiple_cracks(self, crack_edge_lists: List[List[List[float]]]):
        """Load multiple separate crack paths"""
        all_edges = []
        for crack_edges in crack_edge_lists:
            all_edges.extend(crack_edges)
        self.load_from_edge_list(all_edges)
    
    # ========================================================================
    # RANDOM GENERATION METHOD
    # ========================================================================
    
    def generate_network(self,
                        n_junctions: int = 5,
                        n_tips: int = 10,
                        edge_length_mean: float = 0.1,
                        edge_length_std: float = 0.02,
                        min_edge_length: Optional[float] = None,
                        max_edge_length: Optional[float] = None,
                        max_junction_order: int = 4,
                        spatial_distribution: str = 'uniform',
                        connection_strategy: str = 'nearest_neighbor',
                        n_crack_paths: Optional[int] = None,
                        segments_per_path_mean: int = 3,
                        segments_per_path_std: int = 1,
                        path_angle_change_mean_deg: float = 0.0,
                        path_angle_change_std_deg: float = 20.0,
                        allow_intersections: bool = True,
                        min_edge_distance: Optional[float] = None):
        """
        Generate random crack network.
        
        Parameters
        ----------
        n_junctions : int
            Number of junction vertices (degree > 2)
        n_tips : int
            Number of tip vertices (degree = 1)
        edge_length_mean : float
            Mean edge length
        edge_length_std : float
            Standard deviation of edge length
        min_edge_length : float, optional
            Minimum edge length
        max_edge_length : float, optional
            Maximum edge length
        max_junction_order : int
            Maximum degree for junction vertices (3-4 typical)
        spatial_distribution : str
            'uniform', 'clustered', or 'grid'
        connection_strategy : str
            'nearest_neighbor', 'delaunay', or 'random'
        n_crack_paths : int, optional
            Generate specified number of meandering crack paths
        segments_per_path_mean : int
            Average segments per path (for path generation)
        segments_per_path_std : int
            Std dev of segments per path
        path_angle_change_mean_deg : float
            Mean angle change between segments (degrees)
        path_angle_change_std_deg : float
            Std dev of angle change (degrees)
        allow_intersections : bool
            Allow cracks to intersect
        min_edge_distance : float, optional
            Minimum distance between non-connected edges
        """
        
        if min_edge_length is None:
            min_edge_length = edge_length_mean * 0.5
        if max_edge_length is None:
            max_edge_length = edge_length_mean * 2.0
        
        # Clear existing network
        self.vertices = []
        self.edges = []
        self.graph = nx.Graph()
        self.vertex_metadata = {}
        self.edge_metadata = {}
        self.node_id_to_index = None
        self.index_to_node_id = None
        
        # Path-based generation
        if n_crack_paths is not None:
            self._generate_crack_paths(
                n_paths=n_crack_paths,
                segments_per_path_mean=segments_per_path_mean,
                segments_per_path_std=segments_per_path_std,
                edge_length_mean=edge_length_mean,
                edge_length_std=edge_length_std,
                min_edge_length=min_edge_length,
                max_edge_length=max_edge_length,
                angle_change_mean=path_angle_change_mean_deg,
                angle_change_std=path_angle_change_std_deg,
                allow_intersections=allow_intersections,
                min_edge_distance=min_edge_distance,
                spatial_distribution=spatial_distribution
            )
        else:
            # Traditional junction/tip based generation
            self._generate_traditional(
                n_junctions=n_junctions,
                n_tips=n_tips,
                edge_length_mean=edge_length_mean,
                edge_length_std=edge_length_std,
                min_edge_length=min_edge_length,
                max_edge_length=max_edge_length,
                max_junction_order=max_junction_order,
                spatial_distribution=spatial_distribution,
                connection_strategy=connection_strategy
            )
        
        self._classify_vertices()
        print(f"✓ Generated network: {len(self.vertices)} vertices, {len(self.edges)} edges")
    
    def _generate_crack_paths(self, n_paths, segments_per_path_mean, segments_per_path_std,
                             edge_length_mean, edge_length_std, min_edge_length, max_edge_length,
                             angle_change_mean, angle_change_std, allow_intersections,
                             min_edge_distance, spatial_distribution):
        """Generate meandering crack paths with intersection detection"""
        
        intersection_tolerance = 1e-10  # Tolerance for detecting intersections
        
        for path_idx in range(n_paths):
            # Determine number of segments for this path
            n_segments = max(1, int(np.random.normal(segments_per_path_mean, segments_per_path_std)))
            
            # Random starting position
            if spatial_distribution == 'uniform':
                start_x = np.random.uniform(0.1 * self.domain_size[0], 0.9 * self.domain_size[0])
                start_y = np.random.uniform(0.1 * self.domain_size[1], 0.9 * self.domain_size[1])
            elif spatial_distribution == 'clustered':
                cx, cy = 0.5 * self.domain_size[0], 0.5 * self.domain_size[1]
                r = 0.3 * min(self.domain_size)
                angle = np.random.uniform(0, 2*np.pi)
                start_x = cx + r * np.cos(angle) * np.random.uniform(0, 1)
                start_y = cy + r * np.sin(angle) * np.random.uniform(0, 1)
            else:
                start_x = np.random.uniform(0, self.domain_size[0])
                start_y = np.random.uniform(0, self.domain_size[1])
            
            # Random initial direction
            current_angle = np.random.uniform(0, 2*np.pi)
            
            # Build path
            path_vertices = [(start_x, start_y)]
            
            for seg_idx in range(n_segments):
                # Generate edge length
                length = np.clip(
                    np.random.normal(edge_length_mean, edge_length_std),
                    min_edge_length, max_edge_length
                )
                
                # Update angle (meandering)
                angle_change = np.radians(np.random.normal(angle_change_mean, angle_change_std))
                current_angle += angle_change
                
                # Compute next vertex
                x_prev, y_prev = path_vertices[-1]
                x_next = x_prev + length * np.cos(current_angle)
                y_next = y_prev + length * np.sin(current_angle)
                
                # Check domain bounds
                if (0 <= x_next <= self.domain_size[0] and 
                    0 <= y_next <= self.domain_size[1]):
                    
                    # Check for intersections with existing edges
                    if allow_intersections and len(self.edges) > 0:
                        intersection_found = False
                        intersection_point = None
                        intersecting_edge_idx = None
                        
                        # Check against all existing edges
                        for edge_idx, (v1_idx, v2_idx) in enumerate(self.edges):
                            x1, y1 = self.vertices[v1_idx]
                            x2, y2 = self.vertices[v2_idx]
                            
                            # Compute intersection
                            int_pt = self._segment_intersection(
                                x_prev, y_prev, x_next, y_next,
                                x1, y1, x2, y2
                            )
                            
                            if int_pt is not None:
                                xi, yi = int_pt
                                # Check if intersection is not at endpoints (avoid merging endpoints)
                                dist_to_new_start = np.sqrt((xi - x_prev)**2 + (yi - y_prev)**2)
                                dist_to_new_end = np.sqrt((xi - x_next)**2 + (yi - y_next)**2)
                                dist_to_old_start = np.sqrt((xi - x1)**2 + (yi - y1)**2)
                                dist_to_old_end = np.sqrt((xi - x2)**2 + (yi - y2)**2)
                                
                                if (dist_to_new_start > intersection_tolerance and 
                                    dist_to_new_end > intersection_tolerance and
                                    dist_to_old_start > intersection_tolerance and
                                    dist_to_old_end > intersection_tolerance):
                                    intersection_found = True
                                    intersection_point = (xi, yi)
                                    intersecting_edge_idx = edge_idx
                                    break
                        
                        if intersection_found:
                            # Add path up to intersection point
                            start_idx = len(self.vertices)
                            for v in path_vertices:
                                self.vertices.append(v)
                            
                            # Add edges for this path segment
                            for i in range(len(path_vertices) - 1):
                                self.edges.append((start_idx + i, start_idx + i + 1))
                            
                            # Create junction vertex at intersection
                            junction_idx = len(self.vertices)
                            self.vertices.append(intersection_point)
                            
                            # Connect last path vertex to junction
                            self.edges.append((start_idx + len(path_vertices) - 1, junction_idx))
                            
                            # Split the intersecting edge
                            old_v1, old_v2 = self.edges[intersecting_edge_idx]
                            # Remove old edge
                            self.edges.pop(intersecting_edge_idx)
                            # Add two new edges through junction
                            self.edges.append((old_v1, junction_idx))
                            self.edges.append((junction_idx, old_v2))
                            
                            # Start new path from junction
                            path_vertices = [intersection_point, (x_next, y_next)]
                        else:
                            # No intersection, just add vertex
                            path_vertices.append((x_next, y_next))
                    else:
                        # Check distance constraint if needed
                        if not allow_intersections and min_edge_distance is not None:
                            if self._check_edge_distance(x_prev, y_prev, x_next, y_next, min_edge_distance):
                                path_vertices.append((x_next, y_next))
                        else:
                            path_vertices.append((x_next, y_next))
            
            # Add remaining path to network
            if len(path_vertices) > 1:
                start_idx = len(self.vertices)
                self.vertices.extend(path_vertices)
                
                for i in range(len(path_vertices) - 1):
                    self.edges.append((start_idx + i, start_idx + i + 1))
    
    def _segment_intersection(self, x1, y1, x2, y2, x3, y3, x4, y4):
        """
        Find intersection point of two line segments.
        
        Returns (x, y) if segments intersect, None otherwise.
        Uses parametric line equations.
        """
        # Direction vectors
        dx1 = x2 - x1
        dy1 = y2 - y1
        dx2 = x4 - x3
        dy2 = y4 - y3
        
        # Determinant
        det = dx1 * dy2 - dy1 * dx2
        
        if abs(det) < 1e-10:
            # Parallel or coincident
            return None
        
        # Parameters for intersection
        t = ((x3 - x1) * dy2 - (y3 - y1) * dx2) / det
        u = ((x3 - x1) * dy1 - (y3 - y1) * dx1) / det
        
        # Check if intersection is within both segments (not at endpoints)
        eps = 1e-6  # Small epsilon to avoid endpoint detection
        if eps < t < 1-eps and eps < u < 1-eps:
            # Compute intersection point
            xi = x1 + t * dx1
            yi = y1 + t * dy1
            return (xi, yi)
        
        return None
    
    def _generate_traditional(self, n_junctions, n_tips, edge_length_mean, edge_length_std,
                             min_edge_length, max_edge_length, max_junction_order,
                             spatial_distribution, connection_strategy):
        """Traditional junction/tip based generation"""
        
        n_vertices = n_junctions + n_tips
        
        # Generate vertex positions
        if spatial_distribution == 'uniform':
            positions = np.random.uniform(
                [0, 0], self.domain_size, size=(n_vertices, 2)
            )
        elif spatial_distribution == 'clustered':
            centers = np.random.uniform([0, 0], self.domain_size, size=(max(1, n_vertices//5), 2))
            cluster_radius = 0.2 * min(self.domain_size)
            positions = []
            for _ in range(n_vertices):
                center = centers[np.random.randint(len(centers))]
                offset = np.random.normal(0, cluster_radius, size=2)
                pos = np.clip(center + offset, [0, 0], self.domain_size)
                positions.append(pos)
            positions = np.array(positions)
        elif spatial_distribution == 'grid':
            nx_grid = int(np.sqrt(n_vertices))
            ny_grid = int(np.ceil(n_vertices / nx_grid))
            x = np.linspace(0, self.domain_size[0], nx_grid)
            y = np.linspace(0, self.domain_size[1], ny_grid)
            xx, yy = np.meshgrid(x, y)
            positions = np.column_stack([xx.ravel(), yy.ravel()])[:n_vertices]
        
        self.vertices = [tuple(p) for p in positions]
        
        # Generate connectivity
        if connection_strategy == 'nearest_neighbor':
            self._connect_nearest_neighbors(edge_length_mean, edge_length_std, 
                                           min_edge_length, max_edge_length)
        elif connection_strategy == 'delaunay':
            self._connect_delaunay()
        elif connection_strategy == 'random':
            self._connect_random(n_tips + n_junctions)
    
    def _check_edge_distance(self, x1, y1, x2, y2, min_dist):
        """Check if new edge maintains minimum distance from existing edges"""
        for v1_idx, v2_idx in self.edges:
            x3, y3 = self.vertices[v1_idx]
            x4, y4 = self.vertices[v2_idx]
            dist = self._segment_distance(x1, y1, x2, y2, x3, y3, x4, y4)
            if dist < min_dist:
                return False
        return True
    
    def _segment_distance(self, x1, y1, x2, y2, x3, y3, x4, y4):
        """Compute minimum distance between two line segments"""
        # Simplified distance check
        points = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        min_dist = float('inf')
        for i, (px, py) in enumerate(points[:2]):
            for (qx, qy) in points[2:]:
                dist = np.sqrt((px - qx)**2 + (py - qy)**2)
                min_dist = min(min_dist, dist)
        return min_dist
    
    def _connect_nearest_neighbors(self, mean_len, std_len, min_len, max_len):
        """Connect vertices using nearest neighbor strategy"""
        n = len(self.vertices)
        connected = set()
        
        for i in range(n):
            if i in connected:
                continue
            
            # Find nearest unconnected neighbor
            xi, yi = self.vertices[i]
            min_dist = float('inf')
            nearest = -1
            
            for j in range(n):
                if i == j or j in connected:
                    continue
                xj, yj = self.vertices[j]
                dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                if min_len <= dist <= max_len and dist < min_dist:
                    min_dist = dist
                    nearest = j
            
            if nearest >= 0:
                self.edges.append((i, nearest))
                connected.add(i)
                connected.add(nearest)
    
    def _connect_delaunay(self):
        """Connect using Delaunay triangulation"""
        if len(self.vertices) < 3:
            return
        points = np.array(self.vertices)
        tri = Delaunay(points)
        edges_set = set()
        for simplex in tri.simplices:
            for i in range(3):
                edge = tuple(sorted([simplex[i], simplex[(i+1)%3]]))
                edges_set.add(edge)
        self.edges = list(edges_set)
    
    def _connect_random(self, n_edges_target):
        """Random connectivity"""
        n = len(self.vertices)
        edges_set = set()
        while len(edges_set) < min(n_edges_target, n*(n-1)//2):
            i, j = np.random.randint(0, n, size=2)
            if i != j:
                edge = tuple(sorted([i, j]))
                edges_set.add(edge)
        self.edges = list(edges_set)
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def _classify_vertices(self):
        """Classify vertices as tips, junctions, or internal nodes"""
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(len(self.vertices)))
        self.graph.add_edges_from(self.edges)
        
        for i in range(len(self.vertices)):
            degree = self.graph.degree(i)
            if i not in self.vertex_metadata:
                self.vertex_metadata[i] = {}
            
            if degree == 1:
                self.vertex_metadata[i]['type'] = 'tip'
            elif degree > 2:
                self.vertex_metadata[i]['type'] = 'junction'
            else:
                self.vertex_metadata[i]['type'] = 'internal'
    
    def get_statistics(self) -> dict:
        """Get network statistics"""
        n_vertices = len(self.vertices)
        n_edges = len(self.edges)
        
        n_tips = sum(1 for v in self.vertex_metadata.values() if v.get('type') == 'tip')
        n_junctions = sum(1 for v in self.vertex_metadata.values() if v.get('type') == 'junction')
        
        edge_lengths = []
        for v1, v2 in self.edges:
            x1, y1 = self.vertices[v1]
            x2, y2 = self.vertices[v2]
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            edge_lengths.append(length)
        
        return {
            'n_vertices': n_vertices,
            'n_edges': n_edges,
            'n_tips': n_tips,
            'n_junctions': n_junctions,
            'n_internal': n_vertices - n_tips - n_junctions,
            'edge_length_mean': np.mean(edge_lengths) if edge_lengths else 0,
            'edge_length_std': np.std(edge_lengths) if edge_lengths else 0,
            'edge_length_min': np.min(edge_lengths) if edge_lengths else 0,
            'edge_length_max': np.max(edge_lengths) if edge_lengths else 0,
            'total_crack_length': np.sum(edge_lengths) if edge_lengths else 0,
        }
    
    def to_arrays(self, include_node_ids=True, return_types=True):
        """
        Export crack network as arrays.
        
        Parameters
        ----------
        include_node_ids : bool, default True
            If True and network has node IDs, return vertices as [id, x, y]
            If False or no node IDs, return vertices as [x, y]
        return_types : bool, default True
            If True, return vertex types as third output
            If False, return only vertices and connectivity
        
        Returns
        -------
        vertices : np.ndarray
            Vertex array, either shape (n, 3) with [id, x, y] or shape (n, 2) with [x, y]
        connectivity : np.ndarray
            Edge connectivity array, shape (n_edges, 2)
        vertex_types : list (only if return_types=True)
            List of vertex types ('tip', 'junction', 'internal')
        
        Examples
        --------
        >>> # Get all three outputs
        >>> vertices, connectivity, types = generator.to_arrays()
        
        >>> # Get only vertices and connectivity
        >>> vertices, connectivity = generator.to_arrays(return_types=False)
        
        >>> # Or use underscore to ignore types
        >>> vertices, connectivity, _ = generator.to_arrays()
        """
        if include_node_ids and self.index_to_node_id is not None:
            # Export with node IDs
            vertices = []
            for i, (x, y) in enumerate(self.vertices):
                node_id = self.index_to_node_id[i]
                vertices.append([node_id, x, y])
            vertices = np.array(vertices)
            
            # Connectivity uses node IDs
            connectivity = []
            for v1_idx, v2_idx in self.edges:
                node_id_1 = self.index_to_node_id[v1_idx]
                node_id_2 = self.index_to_node_id[v2_idx]
                connectivity.append([node_id_1, node_id_2])
            connectivity = np.array(connectivity)
        else:
            # Export without node IDs (just coordinates)
            vertices = np.array(self.vertices)
            connectivity = np.array(self.edges)
        
        if return_types:
            # Get vertex types
            vertex_types = []
            for i in range(len(self.vertices)):
                vtype = self.vertex_metadata.get(i, {}).get('type', 'internal')
                vertex_types.append(vtype)
            return vertices, connectivity, vertex_types
        else:
            # Return only vertices and connectivity
            return vertices, connectivity
    
    def to_dict(self):
        """
        Export crack network as a dictionary.
        
        Returns
        -------
        dict : Dictionary containing:
            - 'vertices': list of [x, y] coordinates
            - 'edges': list of [v1, v2] connectivity
            - 'vertex_types': list of vertex types
            - 'node_ids': dict mapping indices to node IDs (if available)
            - 'statistics': network statistics
        """
        data = {
            'vertices': [list(v) for v in self.vertices],
            'edges': [list(e) for e in self.edges],
            'vertex_types': [self.vertex_metadata.get(i, {}).get('type', 'internal') 
                           for i in range(len(self.vertices))],
            'statistics': self.get_statistics()
        }
        
        if self.index_to_node_id is not None:
            data['node_ids'] = {i: int(node_id) for i, node_id in self.index_to_node_id.items()}
        
        return data
    
    def save(self, filename):
        """
        Save crack network to file.
        
        Parameters
        ----------
        filename : str
            Output filename. Format determined by extension:
            - .npy: NumPy binary format (vertices and connectivity)
            - .npz: NumPy compressed format (all data)
            - .json: JSON format (human-readable)
            - .txt: Text format (simple vertices and edges)
        
        Examples
        --------
        >>> generator.save('network.npz')
        >>> generator.save('network.json')
        """
        import json
        
        ext = filename.split('.')[-1].lower()
        
        if ext == 'npy':
            # Save as simple numpy arrays
            vertices, connectivity, _ = self.to_arrays()
            np.save(filename, {'vertices': vertices, 'connectivity': connectivity})
            
        elif ext == 'npz':
            # Save as compressed numpy format with all data
            vertices, connectivity, vertex_types = self.to_arrays()
            np.savez_compressed(
                filename,
                vertices=vertices,
                connectivity=connectivity,
                vertex_types=vertex_types,
                **self.get_statistics()
            )
            
        elif ext == 'json':
            # Save as JSON
            data = self.to_dict()
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
                
        elif ext == 'txt':
            # Save as simple text format
            with open(filename, 'w') as f:
                f.write("# Crack Network\n")
                f.write(f"# Vertices: {len(self.vertices)}\n")
                f.write(f"# Edges: {len(self.edges)}\n\n")
                
                f.write("# Vertices (x, y)\n")
                for i, (x, y) in enumerate(self.vertices):
                    f.write(f"{x:.6e} {y:.6e}\n")
                
                f.write("\n# Edges (v1, v2)\n")
                for v1, v2 in self.edges:
                    f.write(f"{v1} {v2}\n")
        else:
            raise ValueError(f"Unsupported file format: .{ext}")
        
        print(f"✓ Network saved to: {filename}")
    
    def print_statistics(self):
        """Print network statistics"""
        stats = self.get_statistics()
        print("\n" + "="*60)
        print("CRACK NETWORK STATISTICS")
        print("="*60)
        print(f"Vertices:       {stats['n_vertices']}")
        print(f"  Tips:         {stats['n_tips']}")
        print(f"  Junctions:    {stats['n_junctions']}")
        print(f"  Internal:     {stats['n_internal']}")
        print(f"Edges:          {stats['n_edges']}")
        print(f"Edge lengths:")
        print(f"  Mean:         {stats['edge_length_mean']*1e3:.3f} mm")
        print(f"  Std dev:      {stats['edge_length_std']*1e3:.3f} mm")
        print(f"  Min:          {stats['edge_length_min']*1e3:.3f} mm")
        print(f"  Max:          {stats['edge_length_max']*1e3:.3f} mm")
        print(f"Total length:   {stats['total_crack_length']*1e3:.3f} mm")
        print("="*60)
    
    def plot_network(self, figsize=(10, 10), show_labels=False, 
                    show_vertex_indices=False, show_node_ids=True,
                    show_edge_labels=False, show_statistics=True,
                    node_size=10, edge_width=2, edge_color='blue',
                    tip_color='red', junction_color='green', 
                    internal_color='black', edge_alpha=0.7):
        """
        Plot the crack network.
        
        Parameters
        ----------
        figsize : tuple
            Figure size
        show_labels : bool
            Show vertex and edge labels from metadata if available
        show_vertex_indices : bool
            Show internal vertex array indices (for debugging)
        show_node_ids : bool
            Show node IDs if vertices were loaded with node IDs
        show_edge_labels : bool
            Show edge labels/indices on the plot
        show_statistics : bool
            Display network statistics as text box
        node_size : float
            Size of node markers (default: 10)
        edge_width : float
            Width of edge lines (default: 2)
        edge_color : str
            Color of edges (default: 'blue')
        tip_color : str
            Color of tip nodes (default: 'red')
        junction_color : str
            Color of junction nodes (default: 'green')
        internal_color : str
            Color of internal nodes (default: 'black')
        edge_alpha : float
            Transparency of edges, 0-1 (default: 0.7)
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot edges
        for edge_idx, (v1, v2) in enumerate(self.edges):
            x1, y1 = self.vertices[v1]
            x2, y2 = self.vertices[v2]
            ax.plot([x1, x2], [y1, y2], color=edge_color, 
                   linewidth=edge_width, alpha=edge_alpha)
            
            # Show edge labels if requested
            if show_edge_labels:
                xm = 0.5 * (x1 + x2)
                ym = 0.5 * (y1 + y2)
                
                # Check if there's a custom label in metadata
                edge_tuple = tuple(sorted([v1, v2]))
                if edge_tuple in self.edge_metadata and 'label' in self.edge_metadata[edge_tuple]:
                    label = self.edge_metadata[edge_tuple]['label']
                else:
                    label = str(edge_idx)
                
                ax.text(xm, ym, label, fontsize=8, color='blue', 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                edgecolor='blue', alpha=0.7))
        
        # Plot vertices
        for i, (x, y) in enumerate(self.vertices):
            vtype = self.vertex_metadata.get(i, {}).get('type', 'internal')
            
            if vtype == 'tip':
                ax.plot(x, y, 'o', color=tip_color, markersize=node_size, 
                       label='Tip' if i == 0 else '')
            elif vtype == 'junction':
                ax.plot(x, y, 's', color=junction_color, markersize=node_size, 
                       label='Junction' if i == 0 else '')
            else:
                ax.plot(x, y, '.', color=internal_color, markersize=node_size*0.6)
            
            # Show node IDs if available
            if show_node_ids and self.index_to_node_id is not None:
                node_id = self.index_to_node_id[i]
                ax.text(x, y, f'  {node_id}', fontsize=10, color='red', fontweight='bold')
            elif show_vertex_indices:
                ax.text(x, y, f'  [{i}]', fontsize=8, color='blue')
            elif show_labels and i in self.vertex_metadata and 'label' in self.vertex_metadata[i]:
                # Show custom vertex label
                label = self.vertex_metadata[i]['label']
                ax.text(x, y, f'  {label}', fontsize=8, color='purple')
        
        # Set limits
        if self.vertices:
            xs = [v[0] for v in self.vertices]
            ys = [v[1] for v in self.vertices]
            margin = 0.1 * max(max(xs) - min(xs), max(ys) - min(ys))
            ax.set_xlim(min(xs) - margin, max(xs) + margin)
            ax.set_ylim(min(ys) - margin, max(ys) + margin)
        
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_title('Crack Network', fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Show statistics
        if show_statistics:
            stats = self.get_statistics()
            stats_text = (
                f"Vertices: {stats['n_vertices']}\n"
                f"Edges: {stats['n_edges']}\n"
                f"Tips: {stats['n_tips']}\n"
                f"Junctions: {stats['n_junctions']}\n"
                f"Total length: {stats['total_crack_length']*1e3:.2f} mm"
            )
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', 
                   facecolor='wheat', alpha=0.8), fontsize=10)
        
        plt.tight_layout()
        return fig, ax
    
    def visualize(self, **kwargs):
        """
        Alias for plot_network() for backward compatibility.
        
        Accepts all the same parameters as plot_network():
        - figsize: tuple, default (10, 10)
        - show_labels: bool, default False
        - show_vertex_indices: bool, default False
        - show_node_ids: bool, default True
        - show_statistics: bool, default True
        
        Returns
        -------
        fig, ax : matplotlib figure and axes objects
        """
        return self.plot_network(**kwargs)


# ============================================================================
# EXAMPLE: Random Generation
# ============================================================================

if __name__ == "__main__":
    mm = 1e-3
    
    print("\n" + "="*70)
    print("EXAMPLE: Random Crack Path Generation")
    print("="*70)
    
    generator = CrackNetworkGenerator(
        domain_size=(25*mm, 25*mm),
        seed=42
    )
    
    generator.generate_network(
        n_junctions=0,
        n_tips=0,
        edge_length_mean=5*mm,
        edge_length_std=2*mm,
        min_edge_length=2*mm,
        max_edge_length=10*mm,
        n_crack_paths=5,
        segments_per_path_mean=4,
        segments_per_path_std=2,
        path_angle_change_mean_deg=0.0,
        path_angle_change_std_deg=30.0,
        allow_intersections=False,
        min_edge_distance=1.0*mm,
        spatial_distribution='clustered'
    )
    
    generator.print_statistics()
    
    fig, ax = generator.plot_network(figsize=(10, 10))
    plt.savefig('/mnt/user-data/outputs/crack_network_random.png', dpi=150, bbox_inches='tight')
    print("\n✓ Plot saved: crack_network_random.png")
