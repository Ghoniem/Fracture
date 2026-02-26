"""
Network analysis tools for crack networks

This module provides:
- Statistical analysis of network properties
- Connectivity metrics
- Path analysis
- Topological characterization
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple


class NetworkAnalyzer:
    """Analyze crack network properties and statistics"""
    
    def __init__(self, graph: nx.Graph):
        """
        Args:
            graph: NetworkX graph representing the crack network
        """
        self.G = graph
    
    def get_statistics(self) -> Dict:
        """
        Compute comprehensive network statistics
        
        Returns:
            Dictionary with network statistics
        """
        types = nx.get_node_attributes(self.G, 'type')
        degrees = dict(self.G.degree())
        
        stats = {
            'n_vertices': self.G.number_of_nodes(),
            'n_edges': self.G.number_of_edges(),
            'n_tips': sum(1 for t in types.values() if t == 'tip'),
            'n_kinks': sum(1 for t in types.values() if t == 'kink'),
            'n_junctions': sum(1 for t in types.values() if t == 'junction'),
        }
        
        # Degree distribution
        degree_counts = {}
        for deg in degrees.values():
            degree_counts[deg] = degree_counts.get(deg, 0) + 1
        stats['degree_distribution'] = degree_counts
        
        # Edge length statistics
        if self.G.number_of_edges() > 0:
            lengths = [data['length'] for _, _, data in self.G.edges(data=True)]
            stats['edge_length_min'] = min(lengths)
            stats['edge_length_max'] = max(lengths)
            stats['edge_length_mean'] = np.mean(lengths)
            stats['edge_length_std'] = np.std(lengths)
        else:
            stats['edge_length_min'] = 0
            stats['edge_length_max'] = 0
            stats['edge_length_mean'] = 0
            stats['edge_length_std'] = 0
        
        # Connected components
        components = list(nx.connected_components(self.G))
        stats['n_components'] = len(components)
        stats['component_sizes'] = [len(c) for c in components]
        
        return stats
    
    def print_statistics(self):
        """Print formatted network statistics"""
        stats = self.get_statistics()
        
        print("="*60)
        print("CRACK NETWORK STATISTICS")
        print("="*60)
        
        print(f"\nVertices:")
        print(f"  Total: {stats['n_vertices']}")
        print(f"  Tips (degree 1): {stats['n_tips']}")
        print(f"  Kinks (degree 2): {stats['n_kinks']}")
        print(f"  Junctions (degree 3+): {stats['n_junctions']}")
        
        print(f"\nEdges:")
        print(f"  Total: {stats['n_edges']}")
        
        if stats['n_edges'] > 0:
            print(f"  Length range: [{stats['edge_length_min']*1e3:.2f}, {stats['edge_length_max']*1e3:.2f}] mm")
            print(f"  Mean length: {stats['edge_length_mean']*1e3:.2f} mm")
            print(f"  Std dev: {stats['edge_length_std']*1e3:.2f} mm")
        
        print(f"\nDegree distribution:")
        for deg in sorted(stats['degree_distribution'].keys()):
            print(f"  Degree {deg}: {stats['degree_distribution'][deg]} vertices")
        
        print(f"\nConnected components (paths): {stats['n_components']}")
        if stats['n_components'] <= 10:
            for i, size in enumerate(stats['component_sizes']):
                n_edges = self._count_component_edges(i)
                print(f"  Path {i+1}: {size} vertices, {n_edges} edges")
        
        print("="*60)
    
    def _count_component_edges(self, component_idx: int) -> int:
        """Count edges in a specific connected component"""
        components = list(nx.connected_components(self.G))
        if component_idx < len(components):
            component = components[component_idx]
            subgraph = self.G.subgraph(component)
            return subgraph.number_of_edges()
        return 0
    
    def get_crack_density(self, domain_area: float) -> float:
        """
        Calculate crack density (total crack length / domain area)
        
        Args:
            domain_area: Area of the domain in m²
            
        Returns:
            Crack density in m/m² (or 1/m)
        """
        total_length = sum(data['length'] for _, _, data in self.G.edges(data=True))
        return total_length / domain_area if domain_area > 0 else 0
    
    def get_connectivity_index(self) -> float:
        """
        Calculate connectivity index (average junction degree / total vertices)
        
        Returns:
            Connectivity index (0-1, higher = more connected)
        """
        types = nx.get_node_attributes(self.G, 'type')
        junctions = [n for n, t in types.items() if t == 'junction']
        
        if len(junctions) == 0:
            return 0.0
        
        avg_junction_degree = np.mean([self.G.degree(j) for j in junctions])
        return avg_junction_degree / self.G.number_of_nodes() if self.G.number_of_nodes() > 0 else 0.0
    
    def get_path_lengths(self) -> List[float]:
        """
        Get lengths of all connected paths
        
        Returns:
            List of path lengths in meters
        """
        components = list(nx.connected_components(self.G))
        path_lengths = []
        
        for component in components:
            subgraph = self.G.subgraph(component)
            total_length = sum(data['length'] for _, _, data in subgraph.edges(data=True))
            path_lengths.append(total_length)
        
        return path_lengths
    
    def get_junction_angles(self) -> Dict[int, List[float]]:
        """
        Calculate all angles at junctions
        
        Returns:
            Dictionary mapping junction node ID to list of angles (degrees)
        """
        from .geometry import GeometryUtils
        geom = GeometryUtils()
        
        positions = nx.get_node_attributes(self.G, 'pos')
        types = nx.get_node_attributes(self.G, 'type')
        junctions = [n for n, t in types.items() if t == 'junction']
        
        junction_angles = {}
        
        for junction in junctions:
            junction_pos = np.array(positions[junction])
            neighbors = list(self.G.neighbors(junction))
            
            if len(neighbors) < 2:
                continue
            
            angles = []
            for i, n1 in enumerate(neighbors):
                pos1 = np.array(positions[n1])
                vec1 = pos1 - junction_pos
                
                for n2 in neighbors[i+1:]:
                    pos2 = np.array(positions[n2])
                    vec2 = pos2 - junction_pos
                    
                    angle = geom.angle_between_vectors(vec1, vec2)
                    angles.append(min(angle, 180 - angle))
            
            junction_angles[junction] = angles
        
        return junction_angles
    
    def to_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert network to vertex and connectivity arrays
        
        Returns:
            vertices: (n_vertices, 3) array with [id, x, y]
            connectivity: (n_edges, 3) array with [edge_id, v0, v1]
            vertex_types: (n_vertices,) array with vertex types as strings
        """
        node_mapping = {old_id: new_id for new_id, old_id in enumerate(self.G.nodes())}
        
        vertices = []
        vertex_types = []
        for old_id, new_id in node_mapping.items():
            pos = self.G.nodes[old_id]['pos']
            vtype = self.G.nodes[old_id].get('type', 'unknown')
            vertices.append([new_id, pos[0], pos[1]])
            vertex_types.append(vtype)
        
        vertices = np.array(vertices, dtype=float)
        vertex_types = np.array(vertex_types, dtype=str)
        
        connectivity = []
        for edge_id, (u, v) in enumerate(self.G.edges()):
            u_new = node_mapping[u]
            v_new = node_mapping[v]
            connectivity.append([edge_id, u_new, v_new])
        
        connectivity = np.array(connectivity, dtype=int)
        
        return vertices, connectivity, vertex_types
