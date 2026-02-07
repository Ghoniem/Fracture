"""
Main crack network generator

This is currently using the monolithic implementation.
Future versions will fully integrate with modular components in:
- geometry.py
- topology.py
- analysis.py
- visualization.py
"""

"""
Crack Network Generator - Complete Version
Features:
- Controllable spatial distributions
- Edge length distributions
- Junction order control
- Intersection prevention
- Minimum angle constraints at junctions
- Minimum distance between edges
- Connected crack paths (polyline cracks)

Author: AI Assistant
Date: 2025
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from typing import Tuple, List, Dict, Optional

class CrackNetworkGenerator:
    """
    Generate random crack networks with comprehensive control over topology and geometry
    """
    
    def __init__(self, 
                 domain_size: Tuple[float, float] = (20e-3, 20e-3),
                 seed: int = None):
        """
        Args:
            domain_size: (width, height) in meters
            seed: Random seed for reproducibility
        """
        self.domain_width, self.domain_height = domain_size
        self.rng = np.random.default_rng(seed)
        self.G = nx.Graph()
        
    def generate_network(self,
                        n_junctions: int = 5,
                        n_tips: int = 8,
                        edge_length_mean: float = 5e-3,
                        edge_length_std: float = 2e-3,
                        max_junction_order: int = 4,
                        spatial_distribution: str = 'uniform',
                        connection_strategy: str = 'nearest_neighbor',
                        min_edge_length: float = 1e-3,
                        max_edge_length: float = 10e-3,
                        allow_intersections: bool = False,
                        min_junction_angle_deg: float = 30.0,
                        min_edge_distance: float = 0.0,
                        n_crack_paths: int = 0,
                        segments_per_path_mean: int = 5,
                        segments_per_path_std: int = 2,
                        path_angle_change_mean_deg: float = 0.0,
                        path_angle_change_std_deg: float = 30.0,
                        detect_intersections: bool = False):
        """
        Generate crack network
        
        Args:
            n_junctions: Number of junction vertices (degree >= 3)
            n_tips: Number of tip vertices (degree = 1)
            edge_length_mean: Mean edge length (m)
            edge_length_std: Std dev of edge length (m)
            max_junction_order: Maximum degree of junctions (3-6 typical)
            spatial_distribution: 'uniform', 'clustered', 'grid'
            connection_strategy: 'nearest_neighbor', 'random', 'delaunay'
            min_edge_length: Minimum edge length (m)
            max_edge_length: Maximum edge length (m)
            allow_intersections: If False, prevents edge crossings during generation
            min_junction_angle_deg: Minimum angle between edges at junctions (degrees)
            min_edge_distance: Minimum distance between parallel/nearby edges (m)
            n_crack_paths: Number of connected crack paths (polyline cracks)
            segments_per_path_mean: Average number of segments per path
            segments_per_path_std: Std dev of segments per path
            path_angle_change_mean_deg: Mean direction change between segments (degrees)
            path_angle_change_std_deg: Std dev of direction change (degrees)
            detect_intersections: If True, detect edge intersections after generation and create junctions (default: False)
        """
        
        # Step 1: Generate junction positions
        if n_junctions > 0:
            junction_positions = self._generate_positions(
                n_junctions, spatial_distribution
            )
            
            # Add junctions to graph
            for i, pos in enumerate(junction_positions):
                self.G.add_node(i, pos=pos, type='junction', degree_target=0)
            
            # Step 2: Assign target degrees to junctions
            self._assign_junction_degrees(n_junctions, max_junction_order)
            
            # Step 3: Connect junctions based on strategy
            self._connect_junctions(
                connection_strategy, 
                edge_length_mean, 
                edge_length_std,
                min_edge_length,
                max_edge_length,
                allow_intersections,
                min_junction_angle_deg,
                min_edge_distance
            )
            
            # Step 4: Add tips to satisfy junction degrees
            self._add_tips_to_junctions(
                edge_length_mean,
                edge_length_std,
                min_edge_length,
                max_edge_length,
                allow_intersections,
                min_junction_angle_deg,
                min_edge_distance
            )
        
        # Step 5: Add additional isolated tips if needed
        current_tips = sum(1 for _, d in self.G.degree() if d == 1)
        remaining_tips = max(0, n_tips - current_tips)
        
        if remaining_tips > 0:
            self._add_isolated_tips(
                remaining_tips,
                edge_length_mean,
                edge_length_std,
                min_edge_length,
                max_edge_length,
                spatial_distribution,
                allow_intersections,
                min_edge_distance
            )
        
        # Step 6: Add connected crack paths
        if n_crack_paths > 0:
            self._add_crack_paths(
                n_paths=n_crack_paths,
                segments_per_path_mean=segments_per_path_mean,
                segments_per_path_std=segments_per_path_std,
                edge_length_mean=edge_length_mean,
                edge_length_std=edge_length_std,
                min_edge_length=min_edge_length,
                max_edge_length=max_edge_length,
                spatial_distribution=spatial_distribution,
                mean_angle_change_deg=path_angle_change_mean_deg,
                std_angle_change_deg=path_angle_change_std_deg,
                allow_intersections=allow_intersections,
                min_edge_distance=min_edge_distance
            )
        
        # Step 7: Update vertex types based on actual degrees
        self._update_vertex_types()
        
        # Step 8: Detect and split intersections if requested
        if detect_intersections:
            self.detect_and_split_intersections(verbose=True)
        
        return self
    
    def _check_edge_intersection(self, new_edge: Tuple[int, int]) -> bool:
        """
        Check if new edge intersects with existing edges
        Uses simple geometric algorithm without external dependencies
        
        Args:
            new_edge: Tuple of (node1, node2) to check
            
        Returns:
            True if intersection detected, False otherwise
        """
        def ccw(A, B, C):
            """Check if three points are counterclockwise"""
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
        
        def segments_intersect(A, B, C, D):
            """Check if line segments AB and CD properly intersect"""
            # Check basic intersection
            if ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D):
                # They intersect - check if at endpoints (allowed)
                tol = 1e-10
                for P1 in [A, B]:
                    for P2 in [C, D]:
                        dist = np.sqrt((P1[0]-P2[0])**2 + (P1[1]-P2[1])**2)
                        if dist < tol:
                            return False  # Touching at endpoint is OK
                return True  # True interior intersection
            return False
        
        positions = nx.get_node_attributes(self.G, 'pos')
        pos1 = np.array(positions[new_edge[0]])
        pos2 = np.array(positions[new_edge[1]])
        
        for u, v in self.G.edges():
            if u in new_edge or v in new_edge:
                continue
            
            pos_u = np.array(positions[u])
            pos_v = np.array(positions[v])
            
            if segments_intersect(pos1, pos2, pos_u, pos_v):
                return True
        
        return False
    
    def _check_junction_angle(self, junction_node: int, new_tip_pos: np.ndarray, 
                             min_angle_deg: float = 30.0) -> bool:
        """
        Check if adding a new edge would create too small an angle at junction
        
        Args:
            junction_node: Junction vertex ID
            new_tip_pos: Position of new tip to be added
            min_angle_deg: Minimum allowed angle between edges (degrees)
            
        Returns:
            True if angle is acceptable, False if too small
        """
        positions = nx.get_node_attributes(self.G, 'pos')
        junction_pos = np.array(positions[junction_node])
        
        # Vector to new tip
        vec_new = new_tip_pos - junction_pos
        vec_new_norm = vec_new / np.linalg.norm(vec_new)
        
        # Check angle with all existing edges at this junction
        for neighbor in self.G.neighbors(junction_node):
            neighbor_pos = np.array(positions[neighbor])
            vec_existing = neighbor_pos - junction_pos
            vec_existing_norm = vec_existing / np.linalg.norm(vec_existing)
            
            # Calculate angle between vectors
            cos_angle = np.clip(np.dot(vec_new_norm, vec_existing_norm), -1.0, 1.0)
            angle_rad = np.arccos(cos_angle)
            angle_deg = np.rad2deg(angle_rad)
            
            # Check both the angle and its supplement (180 - angle)
            min_separation = min(angle_deg, 180 - angle_deg)
            
            if min_separation < min_angle_deg:
                return False  # Too small angle
        
        return True  # All angles are acceptable
    
    def _check_edge_proximity(self, new_edge: Tuple[int, int], 
                             min_distance: float) -> bool:
        """
        Check if new edge is too close to existing edges
        
        Args:
            new_edge: Tuple of (node1, node2) for new edge
            min_distance: Minimum allowed distance between edge centers/lines
            
        Returns:
            True if distances are acceptable, False if too close
        """
        if min_distance <= 0:
            return True
        
        positions = nx.get_node_attributes(self.G, 'pos')
        
        # New edge endpoints
        p1 = np.array(positions[new_edge[0]])
        p2 = np.array(positions[new_edge[1]])
        
        # Check against all existing edges
        for u, v in self.G.edges():
            # Skip if shares a vertex (allowed)
            if u in new_edge or v in new_edge:
                continue
            
            # Existing edge endpoints
            p3 = np.array(positions[u])
            p4 = np.array(positions[v])
            
            # Compute minimum distance between line segments
            dist = self._segment_to_segment_distance(p1, p2, p3, p4)
            
            if dist < min_distance:
                return False
        
        return True
    
    def _segment_to_segment_distance(self, p1: np.ndarray, p2: np.ndarray,
                                     p3: np.ndarray, p4: np.ndarray) -> float:
        """
        Calculate minimum distance between two line segments
        
        Args:
            p1, p2: Endpoints of first segment
            p3, p4: Endpoints of second segment
            
        Returns:
            Minimum distance between the segments
        """
        # Vector representations
        d1 = p2 - p1  # Direction of segment 1
        d2 = p4 - p3  # Direction of segment 2
        r = p1 - p3
        
        a = np.dot(d1, d1)
        e = np.dot(d2, d2)
        f = np.dot(d2, r)
        
        # Check if either or both segments degenerate into points
        epsilon = 1e-10
        
        if a <= epsilon and e <= epsilon:
            # Both segments are points
            return np.linalg.norm(p1 - p3)
        
        if a <= epsilon:
            # First segment is a point
            s = 0.0
            t = np.clip(f / e, 0.0, 1.0)
        else:
            c = np.dot(d1, r)
            if e <= epsilon:
                # Second segment is a point
                t = 0.0
                s = np.clip(-c / a, 0.0, 1.0)
            else:
                # General case
                b = np.dot(d1, d2)
                denom = a * e - b * b
                
                # If segments not parallel, compute closest point on infinite lines
                if denom != 0.0:
                    s = np.clip((b * f - c * e) / denom, 0.0, 1.0)
                else:
                    # Parallel segments - pick arbitrary point on first segment
                    s = 0.0
                
                # Compute point on second segment closest to s
                t = (b * s + f) / e
                
                # If t outside [0,1], clamp and recompute s
                if t < 0.0:
                    t = 0.0
                    s = np.clip(-c / a, 0.0, 1.0)
                elif t > 1.0:
                    t = 1.0
                    s = np.clip((b - c) / a, 0.0, 1.0)
        
        # Compute the closest points
        closest_point_1 = p1 + s * d1
        closest_point_2 = p3 + t * d2
        
        return np.linalg.norm(closest_point_1 - closest_point_2)
    
    def _generate_positions(self, n_points: int, distribution: str) -> np.ndarray:
        """Generate spatial positions for vertices"""
        
        if distribution == 'uniform':
            x = self.rng.uniform(-self.domain_width/2, self.domain_width/2, n_points)
            y = self.rng.uniform(-self.domain_height/2, self.domain_height/2, n_points)
            
        elif distribution == 'clustered':
            # Multiple clusters
            n_clusters = max(2, n_points // 3)
            cluster_centers = self.rng.uniform(
                [-self.domain_width/2, -self.domain_height/2],
                [self.domain_width/2, self.domain_height/2],
                (n_clusters, 2)
            )
            
            cluster_std = min(self.domain_width, self.domain_height) / 10
            
            x, y = [], []
            for i in range(n_points):
                cluster = cluster_centers[i % n_clusters]
                x.append(self.rng.normal(cluster[0], cluster_std))
                y.append(self.rng.normal(cluster[1], cluster_std))
            
            x = np.array(x)
            y = np.array(y)
            
        elif distribution == 'grid':
            # Regular grid with jitter
            n_side = int(np.ceil(np.sqrt(n_points)))
            x_grid = np.linspace(-self.domain_width/2, self.domain_width/2, n_side)
            y_grid = np.linspace(-self.domain_height/2, self.domain_height/2, n_side)
            
            X, Y = np.meshgrid(x_grid, y_grid)
            x = X.flatten()[:n_points]
            y = Y.flatten()[:n_points]
            
            # Add jitter
            jitter = min(self.domain_width, self.domain_height) / (2 * n_side)
            x += self.rng.normal(0, jitter, n_points)
            y += self.rng.normal(0, jitter, n_points)
        
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
        
        # Clip to domain
        x = np.clip(x, -self.domain_width/2, self.domain_width/2)
        y = np.clip(y, -self.domain_height/2, self.domain_height/2)
        
        return np.column_stack([x, y])
    
    def _assign_junction_degrees(self, n_junctions: int, max_order: int):
        """Assign target degrees to junction vertices"""
        
        # Junction degrees: 3 to max_order
        # Use weighted random: favor degree 3-4
        weights = {3: 0.5, 4: 0.3, 5: 0.15, 6: 0.05}
        degrees_available = [d for d in range(3, max_order + 1)]
        probs = [weights.get(d, 0.05) for d in degrees_available]
        probs = np.array(probs) / sum(probs)
        
        for i in range(n_junctions):
            degree = self.rng.choice(degrees_available, p=probs)
            self.G.nodes[i]['degree_target'] = degree
    
    def _connect_junctions(self, strategy: str, mean_length: float, 
                          std_length: float, min_length: float, max_length: float,
                          allow_intersections: bool = False,
                          min_junction_angle_deg: float = 30.0,
                          min_edge_distance: float = 0.0):
        """Connect junction vertices"""
        
        n_junctions = sum(1 for n, d in self.G.nodes(data=True) if d.get('type') == 'junction')
        
        if n_junctions < 2:
            return
        
        positions = nx.get_node_attributes(self.G, 'pos')
        junction_ids = [n for n, d in self.G.nodes(data=True) if d.get('type') == 'junction']
        
        if strategy == 'nearest_neighbor':
            # Connect each junction to its nearest neighbors
            from scipy.spatial import distance_matrix
            
            pos_array = np.array([positions[i] for i in junction_ids])
            dist_matrix = distance_matrix(pos_array, pos_array)
            
            # For each junction, connect to nearest neighbors
            for idx, node_id in enumerate(junction_ids):
                current_degree = self.G.degree(node_id)
                target_degree = self.G.nodes[node_id]['degree_target']
                
                # Sort neighbors by distance
                distances = dist_matrix[idx].copy()
                distances[idx] = np.inf  # Exclude self
                
                # Exclude already connected
                for neighbor in self.G.neighbors(node_id):
                    if neighbor in junction_ids:
                        neighbor_idx = junction_ids.index(neighbor)
                        distances[neighbor_idx] = np.inf
                
                # Connect to nearest unconnected junctions
                n_to_connect = min(target_degree - current_degree, 2)
                
                for _ in range(n_to_connect):
                    # Try multiple candidates
                    max_attempts = len(junction_ids)
                    for attempt in range(max_attempts):
                        nearest_idx = np.argmin(distances)
                        if distances[nearest_idx] == np.inf:
                            break
                        
                        nearest_id = junction_ids[nearest_idx]
                        edge_length = distances[nearest_idx]
                        
                        # Check if edge length is reasonable
                        if not (min_length <= edge_length <= max_length):
                            distances[nearest_idx] = np.inf
                            continue
                        
                        # Check angles at both junctions
                        neighbor_pos = positions[nearest_id]
                        
                        if not self._check_junction_angle(node_id, np.array(neighbor_pos), 
                                                         min_junction_angle_deg):
                            distances[nearest_idx] = np.inf
                            continue
                        
                        node_pos = positions[node_id]
                        if not self._check_junction_angle(nearest_id, np.array(node_pos),
                                                         min_junction_angle_deg):
                            distances[nearest_idx] = np.inf
                            continue
                        
                        # Check edge proximity
                        if min_edge_distance > 0:
                            if not self._check_edge_proximity((node_id, nearest_id), 
                                                             min_edge_distance):
                                distances[nearest_idx] = np.inf
                                continue
                        
                        # Check intersection
                        if not allow_intersections:
                            if self._check_edge_intersection((node_id, nearest_id)):
                                distances[nearest_idx] = np.inf
                                continue
                        
                        # Add edge
                        self.G.add_edge(node_id, nearest_id, length=edge_length)
                        distances[nearest_idx] = np.inf
                        break
        
        elif strategy == 'delaunay':
            # Delaunay triangulation, then prune
            from scipy.spatial import Delaunay
            
            pos_array = np.array([positions[i] for i in junction_ids])
            tri = Delaunay(pos_array)
            
            # Add edges from triangulation
            for simplex in tri.simplices:
                for i in range(3):
                    v1 = junction_ids[simplex[i]]
                    v2 = junction_ids[simplex[(i+1)%3]]
                    
                    if self.G.has_edge(v1, v2):
                        continue
                    
                    edge_length = np.linalg.norm(
                        np.array(positions[v1]) - np.array(positions[v2])
                    )
                    
                    if not (min_length <= edge_length <= max_length):
                        continue
                    
                    # Check angles
                    if not self._check_junction_angle(v1, np.array(positions[v2]),
                                                     min_junction_angle_deg):
                        continue
                    if not self._check_junction_angle(v2, np.array(positions[v1]),
                                                     min_junction_angle_deg):
                        continue
                    
                    # Check proximity
                    if min_edge_distance > 0:
                        if not self._check_edge_proximity((v1, v2), min_edge_distance):
                            continue
                    
                    # Check intersection
                    if not allow_intersections:
                        if self._check_edge_intersection((v1, v2)):
                            continue
                    
                    self.G.add_edge(v1, v2, length=edge_length)
            
            # Prune edges to match target degrees
            self._prune_to_target_degrees()
        
        elif strategy == 'random':
            # Random connections respecting target degrees
            available_junctions = junction_ids.copy()
            
            max_attempts = n_junctions * 20
            attempts = 0
            
            while available_junctions and attempts < max_attempts:
                attempts += 1
                
                # Pick random junction
                node_id = self.rng.choice(available_junctions)
                current_degree = self.G.degree(node_id)
                target_degree = self.G.nodes[node_id]['degree_target']
                
                if current_degree >= target_degree:
                    available_junctions.remove(node_id)
                    continue
                
                # Find candidate neighbors
                candidates = [n for n in available_junctions 
                            if n != node_id 
                            and not self.G.has_edge(node_id, n)]
                
                if not candidates:
                    available_junctions.remove(node_id)
                    continue
                
                # Try multiple random candidates
                self.rng.shuffle(candidates)
                
                for neighbor in candidates[:5]:
                    edge_length = np.linalg.norm(
                        np.array(positions[node_id]) - np.array(positions[neighbor])
                    )
                    
                    if not (min_length <= edge_length <= max_length):
                        continue
                    
                    # Check angles
                    if not self._check_junction_angle(node_id, np.array(positions[neighbor]),
                                                     min_junction_angle_deg):
                        continue
                    if not self._check_junction_angle(neighbor, np.array(positions[node_id]),
                                                     min_junction_angle_deg):
                        continue
                    
                    # Check proximity
                    if min_edge_distance > 0:
                        if not self._check_edge_proximity((node_id, neighbor), 
                                                         min_edge_distance):
                            continue
                    
                    # Check intersection
                    if not allow_intersections:
                        if self._check_edge_intersection((node_id, neighbor)):
                            continue
                    
                    self.G.add_edge(node_id, neighbor, length=edge_length)
                    break
    
    def _prune_to_target_degrees(self):
        """Remove edges to match target degrees"""
        
        # Remove edges from over-connected junctions
        edges_to_remove = []
        
        for node in self.G.nodes():
            if self.G.nodes[node].get('type') != 'junction':
                continue
            
            current_degree = self.G.degree(node)
            target_degree = self.G.nodes[node]['degree_target']
            
            if current_degree > target_degree:
                # Remove longest edges
                edges = [(node, neighbor) for neighbor in self.G.neighbors(node)]
                edges_sorted = sorted(edges, 
                                    key=lambda e: self.G[e[0]][e[1]].get('length', 0),
                                    reverse=True)
                
                n_to_remove = current_degree - target_degree
                edges_to_remove.extend(edges_sorted[:n_to_remove])
        
        for edge in edges_to_remove:
            if self.G.has_edge(*edge):
                self.G.remove_edge(*edge)
    
    def _add_tips_to_junctions(self, mean_length: float, std_length: float,
                               min_length: float, max_length: float,
                               allow_intersections: bool = False,
                               min_junction_angle_deg: float = 30.0,
                               min_edge_distance: float = 0.0):
        """Add tip vertices to junctions to satisfy target degrees"""
        
        tip_counter = self.G.number_of_nodes()
        
        for node in list(self.G.nodes()):
            if self.G.nodes[node].get('type') != 'junction':
                continue
            
            current_degree = self.G.degree(node)
            target_degree = self.G.nodes[node]['degree_target']
            
            n_tips_needed = max(0, target_degree - current_degree)
            
            junction_pos = np.array(self.G.nodes[node]['pos'])
            
            for _ in range(n_tips_needed):
                # Sample edge length
                edge_length = self._sample_edge_length(
                    mean_length, std_length, min_length, max_length
                )
                
                # Try random angles until no intersection AND good angle
                max_angle_attempts = 72
                
                for angle_attempt in range(max_angle_attempts):
                    if angle_attempt == max_angle_attempts - 1:
                        # Last attempt - force add
                        angle = self.rng.uniform(0, 2*np.pi)
                    else:
                        # Systematic search with jitter
                        angle = angle_attempt * (2*np.pi / max_angle_attempts)
                        angle += self.rng.uniform(-np.pi/72, np.pi/72)
                    
                    # Tip position
                    tip_pos = junction_pos + edge_length * np.array([
                        np.cos(angle), np.sin(angle)
                    ])
                    
                    # Check junction angle FIRST (cheaper)
                    if not self._check_junction_angle(node, tip_pos, min_junction_angle_deg):
                        continue
                    
                    # Add temporary tip to check other constraints
                    self.G.add_node(tip_counter, pos=tuple(tip_pos), type='tip')
                    
                    # Check edge proximity
                    if min_edge_distance > 0:
                        if not self._check_edge_proximity((node, tip_counter), 
                                                         min_edge_distance):
                            self.G.remove_node(tip_counter)
                            continue
                    
                    # Check intersection
                    if not allow_intersections:
                        if self._check_edge_intersection((node, tip_counter)):
                            self.G.remove_node(tip_counter)
                            continue
                    
                    # All checks passed - add edge and move on
                    self.G.add_edge(node, tip_counter, length=edge_length)
                    tip_counter += 1
                    break
                else:
                    # Failed to find non-intersecting position
                    # Force add in random direction (fallback)
                    angle = self.rng.uniform(0, 2*np.pi)
                    tip_pos = junction_pos + edge_length * np.array([
                        np.cos(angle), np.sin(angle)
                    ])
                    self.G.add_node(tip_counter, pos=tuple(tip_pos), type='tip')
                    self.G.add_edge(node, tip_counter, length=edge_length)
                    tip_counter += 1
    
    def _add_isolated_tips(self, n_tips: int, mean_length: float, 
                          std_length: float, min_length: float, 
                          max_length: float, distribution: str,
                          allow_intersections: bool = False,
                          min_edge_distance: float = 0.0):
        """Add isolated tip-tip cracks"""
        
        tip_counter = self.G.number_of_nodes()
        
        # Generate positions for tip pairs
        n_pairs = n_tips // 2
        
        for _ in range(n_pairs):
            # Sample edge length
            edge_length = self._sample_edge_length(
                mean_length, std_length, min_length, max_length
            )
            
            # Try multiple random positions
            max_attempts = 100
            
            for attempt in range(max_attempts):
                # Random center position
                center = self._generate_positions(1, distribution)[0]
                
                # Random orientation
                angle = self.rng.uniform(0, 2*np.pi)
                direction = np.array([np.cos(angle), np.sin(angle)])
                
                # Two tips
                tip1_pos = center - (edge_length/2) * direction
                tip2_pos = center + (edge_length/2) * direction
                
                # Add temporary nodes
                self.G.add_node(tip_counter, pos=tuple(tip1_pos), type='tip')
                self.G.add_node(tip_counter + 1, pos=tuple(tip2_pos), type='tip')
                
                # Check edge proximity
                if min_edge_distance > 0:
                    if not self._check_edge_proximity((tip_counter, tip_counter + 1),
                                                     min_edge_distance):
                        self.G.remove_node(tip_counter)
                        self.G.remove_node(tip_counter + 1)
                        continue
                
                # Check intersection
                if not allow_intersections:
                    if self._check_edge_intersection((tip_counter, tip_counter + 1)):
                        self.G.remove_node(tip_counter)
                        self.G.remove_node(tip_counter + 1)
                        continue
                
                # No intersection - add edge
                self.G.add_edge(tip_counter, tip_counter + 1, length=edge_length)
                tip_counter += 2
                break
    
    def _create_crack_path(self, start_pos: np.ndarray, n_segments: int,
                          mean_length: float, std_length: float,
                          min_length: float, max_length: float,
                          mean_angle_change_deg: float = 0.0,
                          std_angle_change_deg: float = 30.0,
                          allow_intersections: bool = False,
                          min_edge_distance: float = 0.0) -> List[int]:
        """
        Create a connected path of crack segments (polyline crack)
        
        Args:
            start_pos: Starting position [x, y]
            n_segments: Number of segments in the path
            mean_angle_change_deg: Mean direction change between segments (degrees)
            std_angle_change_deg: Std dev of direction change (degrees)
            
        Returns:
            List of vertex IDs in the path
        """
        vertex_counter = self.G.number_of_nodes()
        path_vertices = []
        
        # Add starting vertex
        self.G.add_node(vertex_counter, pos=tuple(start_pos), type='tip')
        path_vertices.append(vertex_counter)
        vertex_counter += 1
        
        # Initial random direction
        current_angle = self.rng.uniform(0, 2*np.pi)
        current_pos = start_pos.copy()
        
        for seg_idx in range(n_segments):
            # Sample segment length
            seg_length = self._sample_edge_length(
                mean_length, std_length, min_length, max_length
            )
            
            # Try different angle variations
            max_attempts = 36
            added = False
            
            for attempt in range(max_attempts):
                # Change direction for next segment
                if attempt == 0:
                    # Use mean angle change
                    angle_change = self.rng.normal(
                        np.deg2rad(mean_angle_change_deg),
                        np.deg2rad(std_angle_change_deg)
                    )
                else:
                    # Try more random variations
                    angle_change = self.rng.uniform(-np.pi, np.pi)
                
                next_angle = current_angle + angle_change
                
                # Calculate next position
                direction = np.array([np.cos(next_angle), np.sin(next_angle)])
                next_pos = current_pos + seg_length * direction
                
                # Check if within domain
                if (abs(next_pos[0]) > self.domain_width/2 or 
                    abs(next_pos[1]) > self.domain_height/2):
                    continue
                
                # Add temporary vertex
                prev_vertex = path_vertices[-1]
                self.G.add_node(vertex_counter, pos=tuple(next_pos), type='kink')
                
                # Check constraints
                valid = True
                
                # Check edge proximity
                if min_edge_distance > 0:
                    if not self._check_edge_proximity((prev_vertex, vertex_counter),
                                                     min_edge_distance):
                        valid = False
                
                # Check intersection
                if valid and not allow_intersections:
                    if self._check_edge_intersection((prev_vertex, vertex_counter)):
                        valid = False
                
                if valid:
                    # Add edge
                    self.G.add_edge(prev_vertex, vertex_counter, length=seg_length)
                    path_vertices.append(vertex_counter)
                    vertex_counter += 1
                    
                    # Update for next segment
                    current_pos = next_pos
                    current_angle = next_angle
                    added = True
                    break
                else:
                    # Remove temporary vertex
                    self.G.remove_node(vertex_counter)
            
            if not added:
                # Could not add segment - terminate path early
                break
        
        # Mark last vertex as tip
        if path_vertices:
            self.G.nodes[path_vertices[-1]]['type'] = 'tip'
        
        return path_vertices
    
    def _add_crack_paths(self, n_paths: int, segments_per_path_mean: int,
                        segments_per_path_std: int,
                        edge_length_mean: float, edge_length_std: float,
                        min_edge_length: float, max_edge_length: float,
                        spatial_distribution: str,
                        mean_angle_change_deg: float = 0.0,
                        std_angle_change_deg: float = 30.0,
                        allow_intersections: bool = False,
                        min_edge_distance: float = 0.0):
        """Add multiple connected crack paths to network"""
        
        for path_idx in range(n_paths):
            # Sample number of segments for this path
            n_segments = max(1, int(self.rng.normal(
                segments_per_path_mean,
                segments_per_path_std
            )))
            
            # Random starting position
            max_position_attempts = 20
            
            for pos_attempt in range(max_position_attempts):
                start_pos = self._generate_positions(1, spatial_distribution)[0]
                
                # Try to create path
                path = self._create_crack_path(
                    start_pos=start_pos,
                    n_segments=n_segments,
                    mean_length=edge_length_mean,
                    std_length=edge_length_std,
                    min_length=min_edge_length,
                    max_length=max_edge_length,
                    mean_angle_change_deg=mean_angle_change_deg,
                    std_angle_change_deg=std_angle_change_deg,
                    allow_intersections=allow_intersections,
                    min_edge_distance=min_edge_distance
                )
                
                # Accept if path has at least 2 segments
                if len(path) >= 2:
                    break
    
    def _sample_edge_length(self, mean: float, std: float, 
                           min_val: float, max_val: float) -> float:
        """Sample edge length from truncated normal distribution"""
        
        # Truncated normal
        a = (min_val - mean) / std
        b = (max_val - mean) / std
        
        length = truncnorm.rvs(a, b, loc=mean, scale=std, random_state=self.rng)
        return length
    
    def _update_vertex_types(self):
        """Update vertex types based on actual degrees"""
        
        for node in self.G.nodes():
            degree = self.G.degree(node)
            
            if degree == 1:
                self.G.nodes[node]['type'] = 'tip'
            elif degree == 2:
                self.G.nodes[node]['type'] = 'kink'
            else:
                self.G.nodes[node]['type'] = 'junction'
    
    def _find_edge_intersection(self, edge1: Tuple[int, int], 
                                edge2: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        Find the intersection point of two edges if they intersect
        
        Args:
            edge1: Tuple of (node1, node2) for first edge
            edge2: Tuple of (node1, node2) for second edge
            
        Returns:
            Intersection point [x, y] or None if no intersection
        """
        # Get edge endpoints
        positions = nx.get_node_attributes(self.G, 'pos')
        
        p1 = np.array(positions[edge1[0]])
        p2 = np.array(positions[edge1[1]])
        p3 = np.array(positions[edge2[0]])
        p4 = np.array(positions[edge2[1]])
        
        # Check if edges share a vertex (not a true intersection)
        if edge1[0] in edge2 or edge1[1] in edge2:
            return None
        
        # Line segment intersection algorithm
        # Edge 1: p1 + t * (p2 - p1), t in [0, 1]
        # Edge 2: p3 + s * (p4 - p3), s in [0, 1]
        
        d1 = p2 - p1
        d2 = p4 - p3
        d3 = p1 - p3
        
        # Cross product for 2D
        cross_d1_d2 = d1[0] * d2[1] - d1[1] * d2[0]
        
        # Parallel or coincident lines
        if abs(cross_d1_d2) < 1e-10:
            return None
        
        # Calculate parameters
        t = (d3[0] * d2[1] - d3[1] * d2[0]) / cross_d1_d2
        s = (d3[0] * d1[1] - d3[1] * d1[0]) / cross_d1_d2
        
        # Check if intersection is within both segments (excluding endpoints)
        # Use larger tolerance to avoid splitting very close to endpoints
        tol = 0.05  # 5% from endpoints - increased from 1e-6
        if tol < t < 1.0 - tol and tol < s < 1.0 - tol:
            # Calculate intersection point
            intersection = p1 + t * d1
            return intersection
        
        return None
    
    def detect_and_split_intersections(self, verbose: bool = True, 
                                        mode: str = 'single_pass') -> int:
        """
        Detect all edge intersections and create junction nodes at intersection points
        
        This method:
        1. Finds all pairs of edges that intersect
        2. Creates new junction nodes at intersection points
        3. Splits the intersecting edges
        4. Updates the graph topology
        
        Args:
            verbose: If True, print information about intersections found
            mode: 'single_pass' (process all at once) or 'iterative' (re-detect after each split)
            
        Returns:
            Total number of intersections found and processed
        """
        if verbose:
            print("Detecting edge intersections...")
        
        if mode == 'single_pass':
            return self._detect_single_pass(verbose)
        elif mode == 'iterative':
            return self._detect_iterative(verbose)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'single_pass' or 'iterative'")
    
    def _detect_single_pass(self, verbose: bool) -> int:
        """Single pass intersection detection - finds all, processes compatible ones"""
        intersections_found = []
        edges_list = list(self.G.edges())
        
        # Find all intersections
        for i, edge1 in enumerate(edges_list):
            for edge2 in edges_list[i+1:]:
                intersection_point = self._find_edge_intersection(edge1, edge2)
                
                if intersection_point is not None:
                    intersections_found.append({
                        'edge1': edge1,
                        'edge2': edge2,
                        'point': intersection_point
                    })
        
        if verbose:
            print(f"Found {len(intersections_found)} intersection(s)")
        
        if len(intersections_found) == 0:
            self._update_vertex_types()
            return 0
        
        # Process intersections and split edges
        next_node_id = max(self.G.nodes()) + 1
        processed_count = 0
        
        for intersection in intersections_found:
            edge1 = intersection['edge1']
            edge2 = intersection['edge2']
            int_point = intersection['point']
            
            # Check if edges still exist (might have been removed in previous split)
            if not self.G.has_edge(*edge1) or not self.G.has_edge(*edge2):
                continue
            
            # Get edge lengths
            positions = nx.get_node_attributes(self.G, 'pos')
            
            # Calculate distances for new edge lengths
            p1 = np.array(positions[edge1[0]])
            p2 = np.array(positions[edge1[1]])
            p3 = np.array(positions[edge2[0]])
            p4 = np.array(positions[edge2[1]])
            
            len1a = np.linalg.norm(int_point - p1)
            len1b = np.linalg.norm(p2 - int_point)
            len2a = np.linalg.norm(int_point - p3)
            len2b = np.linalg.norm(p4 - int_point)
            
            # Create new junction node at intersection
            self.G.add_node(next_node_id, pos=tuple(int_point), type='junction')
            
            # Remove old edges
            self.G.remove_edge(*edge1)
            self.G.remove_edge(*edge2)
            
            # Add new split edges
            self.G.add_edge(edge1[0], next_node_id, length=len1a)
            self.G.add_edge(next_node_id, edge1[1], length=len1b)
            self.G.add_edge(edge2[0], next_node_id, length=len2a)
            self.G.add_edge(next_node_id, edge2[1], length=len2b)
            
            if verbose:
                print(f"  Created junction node {next_node_id} at ({int_point[0]*1e3:.2f}, {int_point[1]*1e3:.2f}) mm")
                print(f"    Split edge {edge1} into {edge1[0]}-{next_node_id} and {next_node_id}-{edge1[1]}")
                print(f"    Split edge {edge2} into {edge2[0]}-{next_node_id} and {next_node_id}-{edge2[1]}")
            
            next_node_id += 1
            processed_count += 1
        
        # Update vertex types based on new degrees
        self._update_vertex_types()
        
        if verbose:
            print(f"Successfully processed {processed_count} intersection(s)")
        
        return processed_count
    
    def _detect_iterative(self, verbose: bool) -> int:
        """Iterative intersection detection - re-detects after each split"""
        total_intersections = 0
        iteration = 0
        max_iterations = 100  # Prevent infinite loops
        
        while iteration < max_iterations:
            iteration += 1
            
            intersections_found = []
            edges_list = list(self.G.edges())
            
            # Find all intersections in current graph state
            for i, edge1 in enumerate(edges_list):
                for edge2 in edges_list[i+1:]:
                    intersection_point = self._find_edge_intersection(edge1, edge2)
                    
                    if intersection_point is not None:
                        intersections_found.append({
                            'edge1': edge1,
                            'edge2': edge2,
                            'point': intersection_point
                        })
            
            if len(intersections_found) == 0:
                # No more intersections found
                break
            
            if verbose and iteration == 1:
                print(f"Found {len(intersections_found)} intersection(s) initially")
            elif verbose:
                print(f"Iteration {iteration}: Found {len(intersections_found)} more intersection(s)")
            
            # Process ONE intersection at a time (the first one found)
            # Then re-detect to handle cascading changes
            intersection = intersections_found[0]
            edge1 = intersection['edge1']
            edge2 = intersection['edge2']
            int_point = intersection['point']
            
            # Verify edges still exist
            if not self.G.has_edge(*edge1) or not self.G.has_edge(*edge2):
                continue
            
            # Get edge lengths
            positions = nx.get_node_attributes(self.G, 'pos')
            
            # Calculate distances for new edge lengths
            p1 = np.array(positions[edge1[0]])
            p2 = np.array(positions[edge1[1]])
            p3 = np.array(positions[edge2[0]])
            p4 = np.array(positions[edge2[1]])
            
            len1a = np.linalg.norm(int_point - p1)
            len1b = np.linalg.norm(p2 - int_point)
            len2a = np.linalg.norm(int_point - p3)
            len2b = np.linalg.norm(p4 - int_point)
            
            # Create new junction node at intersection
            next_node_id = max(self.G.nodes()) + 1
            self.G.add_node(next_node_id, pos=tuple(int_point), type='junction')
            
            # Remove old edges
            self.G.remove_edge(*edge1)
            self.G.remove_edge(*edge2)
            
            # Add new split edges
            self.G.add_edge(edge1[0], next_node_id, length=len1a)
            self.G.add_edge(next_node_id, edge1[1], length=len1b)
            self.G.add_edge(edge2[0], next_node_id, length=len2a)
            self.G.add_edge(next_node_id, edge2[1], length=len2b)
            
            if verbose:
                print(f"  Created junction node {next_node_id} at ({int_point[0]*1e3:.2f}, {int_point[1]*1e3:.2f}) mm")
                print(f"    Split edge {edge1} into {edge1[0]}-{next_node_id} and {next_node_id}-{edge1[1]}")
                print(f"    Split edge {edge2} into {edge2[0]}-{next_node_id} and {next_node_id}-{edge2[1]}")
            
            total_intersections += 1
        
        # Update vertex types based on new degrees
        self._update_vertex_types()
        
        if verbose:
            print(f"Successfully processed {total_intersections} intersection(s) total in {iteration} iteration(s)")
        
        return total_intersections
    
    def to_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert network to vertex and connectivity arrays
        
        Returns:
            vertices: (n_vertices, 3) array with [id, x, y]
            connectivity: (n_edges, 3) array with [edge_id, v0, v1]
            vertex_types: (n_vertices,) array with vertex types as strings
        """
        
        # Renumber nodes sequentially
        node_mapping = {old_id: new_id for new_id, old_id in enumerate(self.G.nodes())}
        
        # Build vertex array
        vertices = []
        vertex_types = []
        for old_id, new_id in node_mapping.items():
            pos = self.G.nodes[old_id]['pos']
            vtype = self.G.nodes[old_id].get('type', 'unknown')
            vertices.append([new_id, pos[0], pos[1]])
            vertex_types.append(vtype)
        
        vertices = np.array(vertices, dtype=float)
        vertex_types = np.array(vertex_types, dtype=str)
        
        # Build connectivity array
        connectivity = []
        for edge_id, (u, v) in enumerate(self.G.edges()):
            u_new = node_mapping[u]
            v_new = node_mapping[v]
            connectivity.append([edge_id, u_new, v_new])
        
        connectivity = np.array(connectivity, dtype=int)
        
        return vertices, connectivity, vertex_types
    
    def visualize(self, figsize=(12, 10), show_labels=True, 
                 show_edge_labels=True, title=None, 
                 node_size=200, edge_width=1.5, edge_color='black'):
        """
        Visualize crack network with customizable node sizes and edge display
        
        Args:
            figsize: Figure size (width, height)
            show_labels: Show vertex labels with ID
            show_edge_labels: Show edge labels with length
            title: Custom title (auto-generated if None)
            node_size: Size of nodes (default 200)
            edge_width: Width of edge lines (default 1.5)
            edge_color: Color of edges (default 'black')
        """
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get positions
        pos = nx.get_node_attributes(self.G, 'pos')
        
        # Convert to mm for display
        pos_mm = {k: (v[0]*1e3, v[1]*1e3) for k, v in pos.items()}
        
        # Get vertex types
        types = nx.get_node_attributes(self.G, 'type')
        
        # Draw edges FIRST (so they appear behind nodes)
        nx.draw_networkx_edges(
            self.G, pos_mm, 
            width=edge_width, 
            alpha=0.7, 
            edge_color=edge_color,
            ax=ax
        )
        
        # Separate nodes by type for different colors
        tips = [node for node, vtype in types.items() if vtype == 'tip']
        junctions = [node for node, vtype in types.items() if vtype == 'junction']
        kinks = [node for node, vtype in types.items() if vtype == 'kink']
        
        # Draw tips (red)
        if tips:
            nx.draw_networkx_nodes(
                self.G, pos_mm, 
                nodelist=tips,
                node_color='red',
                node_size=node_size, 
                ax=ax, 
                edgecolors='black', 
                linewidths=1.5
            )
        
        # Draw junctions (blue)
        if junctions:
            nx.draw_networkx_nodes(
                self.G, pos_mm, 
                nodelist=junctions,
                node_color='blue',
                node_size=node_size, 
                ax=ax, 
                edgecolors='black', 
                linewidths=1.5
            )
        
        # Draw kinks (green)
        if kinks:
            nx.draw_networkx_nodes(
                self.G, pos_mm, 
                nodelist=kinks,
                node_color='green',
                node_size=node_size, 
                ax=ax, 
                edgecolors='black', 
                linewidths=1.5
            )
        
        # Node labels
        if show_labels:
            # Show just ID numbers (cleaner for large networks)
            labels = {node: str(node) for node in self.G.nodes()}
            
            nx.draw_networkx_labels(
                self.G, pos_mm, labels, 
                font_size=8, 
                font_weight='bold',
                ax=ax
            )
        
        # Edge labels (lengths)
        if show_edge_labels:
            edge_labels = {}
            for u, v, data in self.G.edges(data=True):
                length = data.get('length', 0)
                edge_labels[(u, v)] = f"{length*1e3:.1f}"
            
            nx.draw_networkx_edge_labels(
                self.G, pos_mm, edge_labels,
                font_size=6, 
                font_color='darkred',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                         edgecolor='none', alpha=0.7),
                ax=ax
            )
        
        # Format
        ax.set_xlabel('x [mm]', fontsize=12, weight='bold')
        ax.set_ylabel('y [mm]', fontsize=12, weight='bold')
        
        if title is None:
            n_junctions = len(junctions)
            n_tips = len(tips)
            n_edges = self.G.number_of_edges()
            title = f"Crack Network: {n_junctions} junctions, {n_tips} tips, {n_edges} edges"
        
        ax.set_title(title, fontsize=14, weight='bold')
        ax.grid(True, alpha=0.3)
        
        # Set axis limits to show full domain (in mm)
        x_min, x_max = -self.domain_width/2 * 1e3, self.domain_width/2 * 1e3
        y_min, y_max = -self.domain_height/2 * 1e3, self.domain_height/2 * 1e3
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        # Set equal aspect ratio after setting limits
        ax.set_aspect('equal', adjustable='box')
        
        # Force tick generation and visibility
        import matplotlib.ticker as ticker
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8, integer=False))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=8, integer=False))
        ax.tick_params(axis='both', which='major', labelsize=10, length=6, width=1)
        
        # Make sure labels are not empty
        ax.xaxis.set_tick_params(labelbottom=True)
        ax.yaxis.set_tick_params(labelleft=True)
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', edgecolor='black', label='Tip (degree 1)'),
            Patch(facecolor='green', edgecolor='black', label='Kink (degree 2)'),
            Patch(facecolor='blue', edgecolor='black', label='Junction (degree 3+)')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.tight_layout()
        
        return fig, ax
    
    def visualize_with_paths(self, figsize=(12, 10)):
        """Visualize with different colors for each connected component (path)"""
        import matplotlib.cm as cm
        
        fig, ax = plt.subplots(figsize=figsize)
        
        pos = nx.get_node_attributes(self.G, 'pos')
        pos_mm = {k: (v[0]*1e3, v[1]*1e3) for k, v in pos.items()}
        
        # Find connected components
        components = list(nx.connected_components(self.G))
        
        # Assign colors to components
        colors = cm.tab20(np.linspace(0, 1, min(len(components), 20)))
        if len(components) > 20:
            colors = cm.hsv(np.linspace(0, 1, len(components)))
        
        # Draw each component in different color
        for comp_idx, component in enumerate(components):
            subgraph = self.G.subgraph(component)
            
            nx.draw_networkx_edges(
                subgraph, pos_mm,
                width=2.0,
                edge_color=[colors[comp_idx]],
                alpha=0.8,
                ax=ax
            )
            
            # Draw nodes
            types = nx.get_node_attributes(self.G, 'type')
            tips = [n for n in component if types.get(n) == 'tip']
            others = [n for n in component if types.get(n) != 'tip']
            
            if tips:
                nx.draw_networkx_nodes(
                    self.G, pos_mm, nodelist=tips,
                    node_color=[colors[comp_idx]],
                    node_size=100, edgecolors='black', linewidths=2,
                    ax=ax
                )
            
            if others:
                nx.draw_networkx_nodes(
                    self.G, pos_mm, nodelist=others,
                    node_color=[colors[comp_idx]],
                    node_size=150, edgecolors='black', linewidths=2,
                    ax=ax
                )
        
        ax.set_xlabel('x [mm]', fontsize=12, weight='bold')
        ax.set_ylabel('y [mm]', fontsize=12, weight='bold')
        ax.set_title(f'Crack Network: {len(components)} Connected Paths', 
                    fontsize=14, weight='bold')
        ax.grid(True, alpha=0.3)
        
        # Set axis limits to show full domain (in mm)
        x_min, x_max = -self.domain_width/2 * 1e3, self.domain_width/2 * 1e3
        y_min, y_max = -self.domain_height/2 * 1e3, self.domain_height/2 * 1e3
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        # Set equal aspect ratio after setting limits
        ax.set_aspect('equal', adjustable='box')
        
        # Force tick generation and visibility
        import matplotlib.ticker as ticker
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8, integer=False))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=8, integer=False))
        ax.tick_params(axis='both', which='major', labelsize=10, length=6, width=1)
        
        # Make sure labels are not empty
        ax.xaxis.set_tick_params(labelbottom=True)
        ax.yaxis.set_tick_params(labelleft=True)
        
        plt.tight_layout()
        return fig, ax
    
    def print_statistics(self):
        """Print network statistics"""
        
        types = nx.get_node_attributes(self.G, 'type')
        degrees = dict(self.G.degree())
        
        print("="*60)
        print("CRACK NETWORK STATISTICS")
        print("="*60)
        
        print(f"\nVertices:")
        print(f"  Total: {self.G.number_of_nodes()}")
        print(f"  Tips (degree 1): {sum(1 for t in types.values() if t == 'tip')}")
        print(f"  Kinks (degree 2): {sum(1 for t in types.values() if t == 'kink')}")
        print(f"  Junctions (degree 3+): {sum(1 for t in types.values() if t == 'junction')}")
        
        print(f"\nEdges:")
        print(f"  Total: {self.G.number_of_edges()}")
        
        if self.G.number_of_edges() > 0:
            lengths = [data['length'] for _, _, data in self.G.edges(data=True)]
            print(f"  Length range: [{min(lengths)*1e3:.2f}, {max(lengths)*1e3:.2f}] mm")
            print(f"  Mean length: {np.mean(lengths)*1e3:.2f} mm")
            print(f"  Std dev: {np.std(lengths)*1e3:.2f} mm")
        
        print(f"\nDegree distribution:")
        degree_counts = {}
        for deg in degrees.values():
            degree_counts[deg] = degree_counts.get(deg, 0) + 1
        
        for deg in sorted(degree_counts.keys()):
            print(f"  Degree {deg}: {degree_counts[deg]} vertices")
        
        # Connected components
        components = list(nx.connected_components(self.G))
        print(f"\nConnected components (paths): {len(components)}")
        if len(components) <= 10:
            for i, comp in enumerate(components):
                print(f"  Path {i+1}: {len(comp)} vertices, {self.G.subgraph(comp).number_of_edges()} edges")
        
        print("="*60)


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    mm = 1e-3
    
    print("Example 1: Mixed network with constraints")
    print("-" * 60)
    
    generator = CrackNetworkGenerator(
        domain_size=(20*mm, 20*mm),
        seed=42
    )
    
    generator.generate_network(
        # Junction network
        n_junctions=5,
        n_tips=20,
        max_junction_order=4,
        
        # Edge properties
        edge_length_mean=3*mm,
        edge_length_std=1*mm,
        min_edge_length=1.5*mm,
        max_edge_length=5*mm,
        
        # Constraints
        allow_intersections=False,
        min_junction_angle_deg=45.0,
        min_edge_distance=0.5*mm,
        
        # Connection
        spatial_distribution='uniform',
        connection_strategy='nearest_neighbor'
    )
    
    generator.print_statistics()
    
    fig, ax = generator.visualize(
        figsize=(12, 10),
        show_labels=False,
        show_edge_labels=False,
        node_size=80,
        edge_width=1.5
    )
    plt.savefig('/mnt/user-data/outputs/crack_network_example1.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Export arrays
    vertices, connectivity, vertex_types = generator.to_arrays()
    print("\nExported arrays shape:")
    print(f"  vertices: {vertices.shape}")
    print(f"  connectivity: {connectivity.shape}")
    print(f"  vertex_types: {vertex_types.shape}")
    print(f"  Vertex types: {set(vertex_types)}")
    
    print("\n" + "="*60)
    print("Example 2: Long connected crack paths")
    print("-" * 60)
    
    generator2 = CrackNetworkGenerator(
        domain_size=(25*mm, 25*mm),
        seed=123
    )
    
    generator2.generate_network(
        n_junctions=0,
        n_tips=0,
        
        edge_length_mean=2*mm,
        edge_length_std=0.5*mm,
        min_edge_length=1*mm,
        max_edge_length=3*mm,
        
        # Connected paths
        n_crack_paths=10,
        segments_per_path_mean=15,
        segments_per_path_std=5,
        path_angle_change_mean_deg=0.0,
        path_angle_change_std_deg=20.0,
        
        allow_intersections=False,
        min_edge_distance=0.8*mm,
        
        spatial_distribution='uniform'
    )
    
    generator2.print_statistics()
    
    fig2, ax2 = generator2.visualize_with_paths(figsize=(12, 10))
    plt.savefig('/mnt/user-data/outputs/crack_network_example2_paths.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\nFiles saved to /mnt/user-data/outputs/")
    print("crack_network_generator.py is ready for download!")
