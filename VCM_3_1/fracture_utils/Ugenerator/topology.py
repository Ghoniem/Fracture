"""
Topology operations for crack networks

This module handles:
- Intersection detection and splitting
- Network connectivity analysis
- Junction management
- Path generation
"""

import numpy as np
import networkx as nx
from typing import Tuple, Optional, List, Dict
from .geometry import GeometryUtils


class TopologyManager:
    """Manages network topology operations"""
    
    def __init__(self, graph: nx.Graph):
        """
        Args:
            graph: NetworkX graph representing the crack network
        """
        self.G = graph
        self.geom = GeometryUtils()
    
    def find_edge_intersection(self, edge1: Tuple[int, int], 
                               edge2: Tuple[int, int],
                               tolerance: float = 0.05) -> Optional[np.ndarray]:
        """
        Find the intersection point of two edges if they intersect
        
        Args:
            edge1: Tuple of (node1, node2) for first edge
            edge2: Tuple of (node1, node2) for second edge
            tolerance: Minimum distance from endpoints as fraction
            
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
        
        return self.geom.segments_intersect(p1, p2, p3, p4, tolerance)
    
    def check_edge_intersection_bool(self, new_edge: Tuple[int, int]) -> bool:
        """
        Check if new edge intersects with existing edges (boolean)
        
        Args:
            new_edge: Tuple of (node1, node2) to check
            
        Returns:
            True if intersection detected, False otherwise
        """
        positions = nx.get_node_attributes(self.G, 'pos')
        pos1 = np.array(positions[new_edge[0]])
        pos2 = np.array(positions[new_edge[1]])
        
        for u, v in self.G.edges():
            if u in new_edge or v in new_edge:
                continue
            
            pos_u = np.array(positions[u])
            pos_v = np.array(positions[v])
            
            if self.geom.segments_intersect_bool(pos1, pos2, pos_u, pos_v):
                return True
        
        return False
    
    def check_edge_proximity(self, new_edge: Tuple[int, int], 
                            min_distance: float) -> bool:
        """
        Check if new edge is too close to existing edges
        
        Args:
            new_edge: Tuple of (node1, node2) for new edge
            min_distance: Minimum allowed distance between edges
            
        Returns:
            True if distances are acceptable, False if too close
        """
        if min_distance <= 0:
            return True
        
        positions = nx.get_node_attributes(self.G, 'pos')
        
        p1 = np.array(positions[new_edge[0]])
        p2 = np.array(positions[new_edge[1]])
        
        for u, v in self.G.edges():
            if u in new_edge or v in new_edge:
                continue
            
            p3 = np.array(positions[u])
            p4 = np.array(positions[v])
            
            dist = self.geom.segment_to_segment_distance(p1, p2, p3, p4)
            
            if dist < min_distance:
                return False
        
        return True
    
    def check_junction_angle(self, junction_node: int, new_tip_pos: np.ndarray, 
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
        
        vec_new = new_tip_pos - junction_pos
        vec_new_norm = vec_new / np.linalg.norm(vec_new)
        
        for neighbor in self.G.neighbors(junction_node):
            neighbor_pos = np.array(positions[neighbor])
            vec_existing = neighbor_pos - junction_pos
            vec_existing_norm = vec_existing / np.linalg.norm(vec_existing)
            
            angle_deg = self.geom.angle_between_vectors(vec_new_norm, vec_existing_norm)
            min_separation = min(angle_deg, 180 - angle_deg)
            
            if min_separation < min_angle_deg:
                return False
        
        return True
    
    def detect_and_split_intersections(self, verbose: bool = True, 
                                       mode: str = 'single_pass') -> int:
        """
        Detect all edge intersections and create junction nodes
        
        Args:
            verbose: If True, print information about intersections
            mode: 'single_pass' or 'iterative'
            
        Returns:
            Total number of intersections processed
        """
        if mode == 'single_pass':
            return self._detect_single_pass(verbose)
        elif mode == 'iterative':
            return self._detect_iterative(verbose)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def _detect_single_pass(self, verbose: bool) -> int:
        """Single pass intersection detection"""
        if verbose:
            print("Detecting edge intersections...")
        
        intersections_found = []
        edges_list = list(self.G.edges())
        
        for i, edge1 in enumerate(edges_list):
            for edge2 in edges_list[i+1:]:
                intersection_point = self.find_edge_intersection(edge1, edge2)
                
                if intersection_point is not None:
                    intersections_found.append({
                        'edge1': edge1,
                        'edge2': edge2,
                        'point': intersection_point
                    })
        
        if verbose:
            print(f"Found {len(intersections_found)} intersection(s)")
        
        if len(intersections_found) == 0:
            self.update_vertex_types()
            return 0
        
        next_node_id = max(self.G.nodes()) + 1
        processed_count = 0
        
        for intersection in intersections_found:
            edge1 = intersection['edge1']
            edge2 = intersection['edge2']
            int_point = intersection['point']
            
            if not self.G.has_edge(*edge1) or not self.G.has_edge(*edge2):
                continue
            
            self._split_edges_at_intersection(edge1, edge2, int_point, next_node_id, verbose)
            next_node_id += 1
            processed_count += 1
        
        self.update_vertex_types()
        
        if verbose:
            print(f"Successfully processed {processed_count} intersection(s)")
        
        return processed_count
    
    def _detect_iterative(self, verbose: bool) -> int:
        """Iterative intersection detection - re-detects after each split"""
        if verbose:
            print("Detecting edge intersections...")
        
        total_intersections = 0
        iteration = 0
        max_iterations = 100
        
        while iteration < max_iterations:
            iteration += 1
            
            intersections_found = []
            edges_list = list(self.G.edges())
            
            for i, edge1 in enumerate(edges_list):
                for edge2 in edges_list[i+1:]:
                    intersection_point = self.find_edge_intersection(edge1, edge2)
                    
                    if intersection_point is not None:
                        intersections_found.append({
                            'edge1': edge1,
                            'edge2': edge2,
                            'point': intersection_point
                        })
            
            if len(intersections_found) == 0:
                break
            
            if verbose and iteration == 1:
                print(f"Found {len(intersections_found)} intersection(s) initially")
            elif verbose:
                print(f"Iteration {iteration}: Found {len(intersections_found)} more intersection(s)")
            
            intersection = intersections_found[0]
            edge1 = intersection['edge1']
            edge2 = intersection['edge2']
            int_point = intersection['point']
            
            if not self.G.has_edge(*edge1) or not self.G.has_edge(*edge2):
                continue
            
            next_node_id = max(self.G.nodes()) + 1
            self._split_edges_at_intersection(edge1, edge2, int_point, next_node_id, verbose)
            total_intersections += 1
        
        self.update_vertex_types()
        
        if verbose:
            print(f"Successfully processed {total_intersections} intersection(s) in {iteration} iteration(s)")
        
        return total_intersections
    
    def _split_edges_at_intersection(self, edge1: Tuple[int, int], 
                                     edge2: Tuple[int, int],
                                     int_point: np.ndarray,
                                     new_node_id: int,
                                     verbose: bool):
        """Split two edges at their intersection point"""
        positions = nx.get_node_attributes(self.G, 'pos')
        
        p1 = np.array(positions[edge1[0]])
        p2 = np.array(positions[edge1[1]])
        p3 = np.array(positions[edge2[0]])
        p4 = np.array(positions[edge2[1]])
        
        len1a = np.linalg.norm(int_point - p1)
        len1b = np.linalg.norm(p2 - int_point)
        len2a = np.linalg.norm(int_point - p3)
        len2b = np.linalg.norm(p4 - int_point)
        
        self.G.add_node(new_node_id, pos=tuple(int_point), type='junction')
        
        self.G.remove_edge(*edge1)
        self.G.remove_edge(*edge2)
        
        self.G.add_edge(edge1[0], new_node_id, length=len1a)
        self.G.add_edge(new_node_id, edge1[1], length=len1b)
        self.G.add_edge(edge2[0], new_node_id, length=len2a)
        self.G.add_edge(new_node_id, edge2[1], length=len2b)
        
        if verbose:
            print(f"  Created junction node {new_node_id} at ({int_point[0]*1e3:.2f}, {int_point[1]*1e3:.2f}) mm")
    
    def update_vertex_types(self):
        """Update vertex types based on actual degrees"""
        for node in self.G.nodes():
            degree = self.G.degree(node)
            
            if degree == 1:
                self.G.nodes[node]['type'] = 'tip'
            elif degree == 2:
                self.G.nodes[node]['type'] = 'kink'
            else:
                self.G.nodes[node]['type'] = 'junction'
