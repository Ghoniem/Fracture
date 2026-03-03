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
from typing import Dict, List, Tuple, Callable, Iterable, Any, Optional


class NetworkAnalyzer:
    """Analyze crack network properties and statistics"""
    
    def __init__(self, graph: nx.Graph):
        """
        Args:
            graph: NetworkX graph representing the crack network
        """
        self.G = graph

    @staticmethod
    def _is_crack_edge(data: Dict[str, Any]) -> bool:
        """Return True when edge should be treated as a crack edge."""
        # BoundaryManager marks explicit boundary edges with type='boundary'.
        # Missing type defaults to crack.
        return str(data.get('type', 'crack')) != 'boundary'

    def _iter_crack_edges(self):
        for u, v, data in self.G.edges(data=True):
            if self._is_crack_edge(data):
                yield u, v, data

    def _node_in_region(self, node: int, region: Any) -> bool:
        """Generic region test used by spanning checks.

        Supported `region` forms:
        - callable(node_id, node_data) -> bool
        - iterable of node ids
        - dict with optional keys xmin, xmax, ymin, ymax (axis-aligned box)
        """
        node_data = self.G.nodes[node]
        if callable(region):
            return bool(region(node, node_data))
        if isinstance(region, dict):
            if 'pos' not in node_data:
                return False
            x, y = node_data['pos']
            xmin = float(region.get('xmin', -np.inf))
            xmax = float(region.get('xmax', np.inf))
            ymin = float(region.get('ymin', -np.inf))
            ymax = float(region.get('ymax', np.inf))
            return (xmin <= x <= xmax) and (ymin <= y <= ymax)
        try:
            return node in set(region)
        except TypeError:
            return False

    def get_total_crack_length(self) -> float:
        """Total crack length L_tot (sum over non-boundary edges)."""
        return float(sum(float(data.get('length', 0.0)) for _, _, data in self._iter_crack_edges()))

    def get_component_lengths(self) -> List[float]:
        """Lengths L_k of all connected crack components."""
        crack_subgraph = nx.Graph()
        crack_subgraph.add_nodes_from(self.G.nodes(data=True))
        crack_subgraph.add_edges_from((u, v, data) for u, v, data in self._iter_crack_edges())

        lengths = []
        for component in nx.connected_components(crack_subgraph):
            sg = crack_subgraph.subgraph(component)
            Lk = float(sum(float(data.get('length', 0.0)) for _, _, data in sg.edges(data=True)))
            # Keep only components with at least one crack edge.
            if sg.number_of_edges() > 0:
                lengths.append(Lk)
        return lengths

    def get_spanning_components(self, region_1: Any, region_2: Any) -> List[Dict[str, Any]]:
        """Find crack components spanning two disjoint boundary regions.

        Returns list of dicts with:
          - component_nodes
          - length_m
          - touches_region_1
          - touches_region_2
        """
        crack_subgraph = nx.Graph()
        crack_subgraph.add_nodes_from(self.G.nodes(data=True))
        crack_subgraph.add_edges_from((u, v, data) for u, v, data in self._iter_crack_edges())

        out: List[Dict[str, Any]] = []
        for component in nx.connected_components(crack_subgraph):
            sg = crack_subgraph.subgraph(component)
            if sg.number_of_edges() == 0:
                continue

            touches_1 = any(self._node_in_region(n, region_1) for n in component)
            touches_2 = any(self._node_in_region(n, region_2) for n in component)
            if not (touches_1 and touches_2):
                continue

            out.append(
                {
                    'component_nodes': set(component),
                    'length_m': float(sum(float(data.get('length', 0.0)) for _, _, data in sg.edges(data=True))),
                    'touches_region_1': True,
                    'touches_region_2': True,
                }
            )
        return out

    def get_spanning_cluster_fraction(self, region_1: Any, region_2: Any) -> float:
        """P_infty = L_span / L_tot, where L_span is longest spanning component.

        If no spanning component exists, returns 0.
        """
        L_tot = self.get_total_crack_length()
        if L_tot <= 0.0:
            return 0.0
        spans = self.get_spanning_components(region_1, region_2)
        if not spans:
            return 0.0
        L_span = max(float(s['length_m']) for s in spans)
        return float(L_span / L_tot)

    def get_topology_metrics(
        self,
        *,
        domain_area: Optional[float] = None,
        region_1: Optional[Any] = None,
        region_2: Optional[Any] = None,
        loading_axis: Tuple[float, float] = (0.0, 1.0),
        include_alignment_Q: bool = True,
    ) -> Dict[str, Any]:
        """Aggregate metrics aligned with VCM_v3_3_ReadMe topology definitions."""
        L_tot = self.get_total_crack_length()
        component_lengths = self.get_component_lengths()
        metrics: Dict[str, Any] = {
            'L_tot_m': L_tot,
            'n_components': len(component_lengths),
            'component_lengths_m': component_lengths,
        }

        if domain_area is not None and float(domain_area) > 0.0:
            alpha = float(L_tot / float(domain_area))
            metrics['alpha_1_per_m'] = alpha
            metrics['alpha'] = alpha

        if region_1 is not None and region_2 is not None:
            spans = self.get_spanning_components(region_1, region_2)
            L_span = max((float(s['length_m']) for s in spans), default=0.0)
            metrics['L_span_m'] = L_span
            metrics['P_infty'] = float(L_span / L_tot) if L_tot > 0.0 else 0.0
            metrics['n_spanning_components'] = len(spans)

        if include_alignment_Q:
            metrics['Q'] = self.get_alignment_parameter(loading_axis=loading_axis, length_weighted=True)

        return metrics
    
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
        stats['total_crack_length_m'] = self.get_total_crack_length()
        stats['component_lengths_m'] = self.get_component_lengths()
        
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
        total_length = self.get_total_crack_length()
        return total_length / domain_area if domain_area > 0 else 0

    def get_alpha(self, domain_area: float) -> float:
        """Alias for crack density alpha = L_tot / A."""
        return self.get_crack_density(domain_area)

    def get_alignment_parameter(
        self,
        loading_axis: Tuple[float, float] = (0.0, 1.0),
        *,
        length_weighted: bool = True,
    ) -> float:
        """Compute alignment parameter Q = <2 cos^2(theta) - 1>.

        theta is the angle between each crack segment and loading axis.
        """
        axis = np.asarray(loading_axis, float).reshape(2)
        n_axis = float(np.linalg.norm(axis))
        if n_axis <= 0.0:
            raise ValueError("loading_axis must be non-zero.")
        axis = axis / n_axis

        vals: List[float] = []
        weights: List[float] = []
        for u, v, data in self._iter_crack_edges():
            pu = np.asarray(self.G.nodes[u].get('pos', (np.nan, np.nan)), float).reshape(2)
            pv = np.asarray(self.G.nodes[v].get('pos', (np.nan, np.nan)), float).reshape(2)
            t = pv - pu
            L = float(np.linalg.norm(t))
            if L <= 0.0 or not np.isfinite(L):
                continue
            t = t / L

            # Segment orientation is undirected -> use |cos(theta)|.
            c = float(abs(np.dot(t, axis)))
            vals.append(2.0 * c * c - 1.0)
            weights.append(float(data.get('length', L)))

        if not vals:
            return 0.0
        if not length_weighted:
            return float(np.mean(np.asarray(vals, float)))

        w = np.asarray(weights, float)
        ws = float(np.sum(w))
        if ws <= 0.0:
            return float(np.mean(np.asarray(vals, float)))
        return float(np.average(np.asarray(vals, float), weights=w))
    
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
        return self.get_component_lengths()
    
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
