"""
Crack Network Simplification Module
====================================

Tools for simplifying evolved crack networks before further growth.

Features:
- Merge colinear/nearly-colinear edges
- Remove/coalesce small edges
- Simplify topology (remove degree-2 nodes)
- Smooth crack paths
- Remove redundant vertices

Author: AI Assistant
Date: 2025
"""

import numpy as np
import networkx as nx
from typing import Tuple, List, Dict, Optional, Set
from dataclasses import dataclass


@dataclass
class SimplificationConfig:
    """Configuration for network simplification"""
    
    # Angle threshold for merging edges (degrees)
    max_angle_deviation: float = 5.0
    
    # Minimum edge length (edges shorter than this are candidates for removal)
    min_edge_length: float = 0.5e-3  # 0.5 mm
    
    # Remove degree-2 internal nodes (simplify polylines)
    remove_degree2_nodes: bool = True
    
    # Merge edges at degree-2 nodes if angle is small
    merge_at_degree2: bool = True
    
    # Minimum distance to merge vertices (m)
    merge_vertex_tolerance: float = 1e-6
    
    # Maximum edge length after merging
    max_merged_edge_length: float = 20e-3  # 20 mm
    
    # Preserve tips (don't remove degree-1 nodes)
    preserve_tips: bool = True
    
    # Preserve junctions (don't modify degree >= 3 nodes)
    preserve_junctions: bool = True


class CrackNetworkSimplifier:
    """
    Simplify crack networks by merging edges, removing small features,
    and cleaning up topology.
    """
    
    def __init__(self, config: Optional[SimplificationConfig] = None):
        """
        Initialize simplifier.
        
        Parameters
        ----------
        config : SimplificationConfig, optional
            Configuration parameters. If None, uses defaults.
        """
        self.config = config if config is not None else SimplificationConfig()
        self.G = None
        self.domain_size = None
    
    def load_from_arrays(self, 
                        vertices: np.ndarray, 
                        connectivity: np.ndarray,
                        domain_size: Optional[Tuple[float, float]] = None):
        """
        Load network from vertex and connectivity arrays.
        
        Parameters
        ----------
        vertices : np.ndarray
            Shape (n, 3) with [id, x, y] or shape (n, 2) with [x, y]
        connectivity : np.ndarray  
            Shape (m, 2) with [v0, v1] or shape (m, 3) with [edge_id, v0, v1]
        domain_size : tuple, optional
            (width, height) of domain
        
        Examples
        --------
        >>> simplifier = CrackNetworkSimplifier()
        >>> simplifier.load_from_arrays(vertices, connectivity)
        """
        self.G = nx.Graph()
        
        # Parse vertices
        if vertices.shape[1] == 3:
            # Format: [id, x, y]
            for row in vertices:
                node_id = int(row[0])
                pos = (float(row[1]), float(row[2]))
                self.G.add_node(node_id, pos=pos)
        elif vertices.shape[1] == 2:
            # Format: [x, y] - create sequential IDs
            for i, row in enumerate(vertices):
                pos = (float(row[0]), float(row[1]))
                self.G.add_node(i, pos=pos)
        else:
            raise ValueError(f"Invalid vertices shape: {vertices.shape}")
        
        # Parse connectivity
        if connectivity.shape[1] == 2:
            # Format: [v0, v1]
            for v0, v1 in connectivity:
                self.G.add_edge(int(v0), int(v1))
        elif connectivity.shape[1] == 3:
            # Format: [edge_id, v0, v1]
            for edge_id, v0, v1 in connectivity:
                self.G.add_edge(int(v0), int(v1), edge_id=int(edge_id))
        else:
            raise ValueError(f"Invalid connectivity shape: {connectivity.shape}")
        
        # Update vertex types
        self._update_vertex_types()
        
        # Set domain size
        if domain_size is not None:
            self.domain_size = domain_size
        else:
            # Infer from network bounds
            positions = np.array([self.G.nodes[n]['pos'] for n in self.G.nodes()])
            self.domain_size = (
                positions[:, 0].max() - positions[:, 0].min(),
                positions[:, 1].max() - positions[:, 1].min()
            )
        
        print(f"✓ Loaded network: {self.G.number_of_nodes()} vertices, "
              f"{self.G.number_of_edges()} edges")
    
    def _update_vertex_types(self):
        """Classify vertices as tips, junctions, or internal nodes"""
        for node in self.G.nodes():
            degree = self.G.degree(node)
            if degree == 1:
                self.G.nodes[node]['type'] = 'tip'
            elif degree == 2:
                self.G.nodes[node]['type'] = 'internal'
            else:
                self.G.nodes[node]['type'] = 'junction'
    
    def simplify(self, verbose: bool = True) -> Dict[str, int]:
        """
        Apply all simplification operations.
        
        Parameters
        ----------
        verbose : bool
            Print progress information
        
        Returns
        -------
        dict
            Statistics about simplification (vertices/edges removed, merged, etc.)
        
        Examples
        --------
        >>> stats = simplifier.simplify()
        >>> print(f"Removed {stats['edges_removed']} edges")
        """
        if self.G is None:
            raise ValueError("No network loaded. Use load_from_arrays() first.")
        
        stats = {
            'initial_vertices': self.G.number_of_nodes(),
            'initial_edges': self.G.number_of_edges(),
            'vertices_merged': 0,
            'edges_merged': 0,
            'edges_removed': 0,
            'degree2_removed': 0
        }
        
        if verbose:
            print("\n" + "="*60)
            print("SIMPLIFYING CRACK NETWORK")
            print("="*60)
            print(f"Initial: {stats['initial_vertices']} vertices, "
                  f"{stats['initial_edges']} edges")
        
        # Step 1: Merge nearby vertices
        if verbose:
            print("\nStep 1: Merging nearby vertices...")
        n_merged = self._merge_close_vertices()
        stats['vertices_merged'] = n_merged
        if verbose:
            print(f"  ✓ Merged {n_merged} vertices")
        
        # Step 2: Remove small edges
        if verbose:
            print("\nStep 2: Removing small edges...")
        n_removed = self._remove_small_edges()
        stats['edges_removed'] = n_removed
        if verbose:
            print(f"  ✓ Removed {n_removed} small edges")
        
        # Step 3: Merge colinear edges at degree-2 nodes
        if self.config.merge_at_degree2:
            if verbose:
                print("\nStep 3: Merging colinear edges...")
            n_merged = self._merge_colinear_edges()
            stats['edges_merged'] = n_merged
            if verbose:
                print(f"  ✓ Merged {n_merged} edge pairs")
        
        # Step 4: Remove degree-2 internal nodes
        if self.config.remove_degree2_nodes:
            if verbose:
                print("\nStep 4: Removing degree-2 internal nodes...")
            n_removed = self._remove_degree2_nodes()
            stats['degree2_removed'] = n_removed
            if verbose:
                print(f"  ✓ Removed {n_removed} internal nodes")
        
        # Update vertex types
        self._update_vertex_types()
        
        stats['final_vertices'] = self.G.number_of_nodes()
        stats['final_edges'] = self.G.number_of_edges()
        
        if verbose:
            print("\n" + "="*60)
            print("SIMPLIFICATION COMPLETE")
            print("="*60)
            print(f"Final: {stats['final_vertices']} vertices, "
                  f"{stats['final_edges']} edges")
            print(f"Reduction: {stats['initial_vertices'] - stats['final_vertices']} vertices, "
                  f"{stats['initial_edges'] - stats['final_edges']} edges")
        
        return stats
    
    def _merge_close_vertices(self) -> int:
        """Merge vertices that are very close together"""
        tol = self.config.merge_vertex_tolerance
        merged_count = 0
        
        nodes_to_merge = []
        nodes = list(self.G.nodes())
        
        for i, n1 in enumerate(nodes):
            if n1 not in self.G.nodes():  # Already merged
                continue
            pos1 = np.array(self.G.nodes[n1]['pos'])
            
            for n2 in nodes[i+1:]:
                if n2 not in self.G.nodes():
                    continue
                pos2 = np.array(self.G.nodes[n2]['pos'])
                
                dist = np.linalg.norm(pos2 - pos1)
                if dist < tol:
                    nodes_to_merge.append((n1, n2))
        
        # Perform merges
        for n1, n2 in nodes_to_merge:
            if n1 in self.G.nodes() and n2 in self.G.nodes():
                # Merge n2 into n1
                for neighbor in list(self.G.neighbors(n2)):
                    if neighbor != n1:
                        self.G.add_edge(n1, neighbor)
                self.G.remove_node(n2)
                merged_count += 1
        
        return merged_count
    
    def _remove_small_edges(self) -> int:
        """Remove edges shorter than minimum length"""
        min_len = self.config.min_edge_length
        removed_count = 0
        
        edges_to_remove = []
        
        for u, v in self.G.edges():
            pos_u = np.array(self.G.nodes[u]['pos'])
            pos_v = np.array(self.G.nodes[v]['pos'])
            length = np.linalg.norm(pos_v - pos_u)
            
            if length < min_len:
                # Check if we can remove this edge
                deg_u = self.G.degree(u)
                deg_v = self.G.degree(v)
                
                # Don't create isolated vertices
                can_remove = True
                if deg_u == 1 and deg_v == 1:
                    can_remove = False  # Would create 2 isolated nodes
                elif deg_u == 1 and self.config.preserve_tips:
                    can_remove = False
                elif deg_v == 1 and self.config.preserve_tips:
                    can_remove = False
                
                if can_remove:
                    edges_to_remove.append((u, v))
        
        # Remove edges and cleanup
        for u, v in edges_to_remove:
            if self.G.has_edge(u, v):
                self.G.remove_edge(u, v)
                removed_count += 1
                
                # Remove isolated vertices
                if self.G.degree(u) == 0:
                    self.G.remove_node(u)
                if v in self.G.nodes() and self.G.degree(v) == 0:
                    self.G.remove_node(v)
        
        return removed_count
    
    def _merge_colinear_edges(self) -> int:
        """Merge edges at degree-2 nodes if they're nearly colinear"""
        max_angle = self.config.max_angle_deviation
        merged_count = 0
        
        # Find all degree-2 nodes
        degree2_nodes = [n for n in self.G.nodes() if self.G.degree(n) == 2]
        
        for node in degree2_nodes:
            if node not in self.G.nodes():  # Already removed
                continue
            
            if self.G.degree(node) != 2:  # Degree changed
                continue
            
            # Get the two neighbors
            neighbors = list(self.G.neighbors(node))
            if len(neighbors) != 2:
                continue
            
            n1, n2 = neighbors
            
            # Check angle deviation
            pos = np.array(self.G.nodes[node]['pos'])
            pos1 = np.array(self.G.nodes[n1]['pos'])
            pos2 = np.array(self.G.nodes[n2]['pos'])
            
            # Vectors from node to neighbors
            v1 = pos1 - pos
            v2 = pos2 - pos
            
            # Angle between vectors
            angle = self._angle_between_vectors(v1, v2)
            deviation = abs(180.0 - angle)  # Deviation from straight line
            
            if deviation <= max_angle:
                # Check merged edge length
                merged_length = np.linalg.norm(pos2 - pos1)
                if merged_length <= self.config.max_merged_edge_length:
                    # Merge: remove node, add direct edge between neighbors
                    self.G.add_edge(n1, n2)
                    self.G.remove_node(node)
                    merged_count += 1
        
        return merged_count
    
    def _remove_degree2_nodes(self) -> int:
        """Remove degree-2 internal nodes (simplify polylines)"""
        removed_count = 0
        
        # Find degree-2 nodes
        degree2_nodes = [n for n in self.G.nodes() 
                        if self.G.degree(n) == 2 
                        and self.G.nodes[n].get('type') == 'internal']
        
        for node in degree2_nodes:
            if node not in self.G.nodes():
                continue
            if self.G.degree(node) != 2:
                continue
            
            # Get neighbors
            neighbors = list(self.G.neighbors(node))
            if len(neighbors) != 2:
                continue
            
            n1, n2 = neighbors
            
            # Create direct edge and remove node
            self.G.add_edge(n1, n2)
            self.G.remove_node(node)
            removed_count += 1
        
        return removed_count
    
    @staticmethod
    def _angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate angle between two vectors in degrees"""
        v1_norm = v1 / (np.linalg.norm(v1) + 1e-10)
        v2_norm = v2 / (np.linalg.norm(v2) + 1e-10)
        cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        return np.degrees(angle_rad)
    
    def to_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Export simplified network to arrays.
        
        Returns
        -------
        vertices : np.ndarray
            Shape (n, 3) with [id, x, y]
        connectivity : np.ndarray
            Shape (m, 2) with [v0, v1]
        
        Examples
        --------
        >>> vertices, connectivity = simplifier.to_arrays()
        """
        if self.G is None:
            raise ValueError("No network loaded")
        
        # Renumber nodes sequentially
        node_mapping = {old_id: new_id for new_id, old_id in enumerate(self.G.nodes())}
        
        # Build vertex array
        vertices = []
        for old_id, new_id in node_mapping.items():
            pos = self.G.nodes[old_id]['pos']
            vertices.append([new_id, pos[0], pos[1]])
        
        # Build connectivity
        connectivity = []
        for u, v in self.G.edges():
            u_new = node_mapping[u]
            v_new = node_mapping[v]
            connectivity.append([u_new, v_new])
        
        return np.array(vertices), np.array(connectivity)
    
    def get_statistics(self) -> Dict:
        """Get network statistics"""
        if self.G is None:
            return {}
        
        degrees = dict(self.G.degree())
        tips = sum(1 for d in degrees.values() if d == 1)
        junctions = sum(1 for d in degrees.values() if d >= 3)
        internal = sum(1 for d in degrees.values() if d == 2)
        
        return {
            'n_vertices': self.G.number_of_nodes(),
            'n_edges': self.G.number_of_edges(),
            'n_tips': tips,
            'n_junctions': junctions,
            'n_internal': internal,
        }
