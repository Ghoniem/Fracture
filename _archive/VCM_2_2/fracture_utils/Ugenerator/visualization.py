"""
Visualization tools for crack networks

This module provides:
- Standard network visualization
- Path-colored visualization
- Statistical plots
- Export functions
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch
from typing import Tuple, Optional


class NetworkVisualizer:
    """Visualization utilities for crack networks"""
    
    def __init__(self, graph: nx.Graph, domain_size: Tuple[float, float]):
        """
        Args:
            graph: NetworkX graph representing the crack network
            domain_size: (width, height) of domain in meters
        """
        self.G = graph
        self.domain_width, self.domain_height = domain_size
    
    def visualize(self, figsize: Tuple[float, float] = (12, 10), 
                 show_labels: bool = True,
                 show_edge_labels: bool = True, 
                 title: Optional[str] = None,
                 node_size: int = 200, 
                 edge_width: float = 1.5, 
                 edge_color: str = 'black') -> Tuple[plt.Figure, plt.Axes]:
        """
        Visualize crack network with standard coloring
        
        Args:
            figsize: Figure size (width, height)
            show_labels: Show vertex labels with ID
            show_edge_labels: Show edge labels with length
            title: Custom title (auto-generated if None)
            node_size: Size of nodes
            edge_width: Width of edge lines
            edge_color: Color of edges
            
        Returns:
            Figure and axes objects
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        pos = nx.get_node_attributes(self.G, 'pos')
        pos_mm = {k: (v[0]*1e3, v[1]*1e3) for k, v in pos.items()}
        types = nx.get_node_attributes(self.G, 'type')
        
        # Draw edges first
        nx.draw_networkx_edges(
            self.G, pos_mm, 
            width=edge_width, 
            alpha=0.7, 
            edge_color=edge_color,
            ax=ax
        )
        
        # Separate nodes by type
        tips = [node for node, vtype in types.items() if vtype == 'tip']
        junctions = [node for node, vtype in types.items() if vtype == 'junction']
        kinks = [node for node, vtype in types.items() if vtype == 'kink']
        
        # Draw nodes by type
        if tips:
            nx.draw_networkx_nodes(
                self.G, pos_mm, nodelist=tips,
                node_color='red', node_size=node_size, 
                ax=ax, edgecolors='black', linewidths=1.5
            )
        
        if junctions:
            nx.draw_networkx_nodes(
                self.G, pos_mm, nodelist=junctions,
                node_color='blue', node_size=node_size, 
                ax=ax, edgecolors='black', linewidths=1.5
            )
        
        if kinks:
            nx.draw_networkx_nodes(
                self.G, pos_mm, nodelist=kinks,
                node_color='green', node_size=node_size, 
                ax=ax, edgecolors='black', linewidths=1.5
            )
        
        # Node labels
        if show_labels:
            labels = {node: str(node) for node in self.G.nodes()}
            nx.draw_networkx_labels(
                self.G, pos_mm, labels, 
                font_size=8, font_weight='bold', ax=ax
            )
        
        # Edge labels
        if show_edge_labels:
            edge_labels = {}
            for u, v, data in self.G.edges(data=True):
                length = data.get('length', 0)
                edge_labels[(u, v)] = f"{length*1e3:.1f}"
            
            nx.draw_networkx_edge_labels(
                self.G, pos_mm, edge_labels,
                font_size=6, font_color='darkred',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                         edgecolor='none', alpha=0.7),
                ax=ax
            )
        
        # Format axes
        ax.set_xlabel('x [mm]', fontsize=12, weight='bold')
        ax.set_ylabel('y [mm]', fontsize=12, weight='bold')
        
        if title is None:
            n_junctions = len(junctions)
            n_tips = len(tips)
            n_edges = self.G.number_of_edges()
            title = f"Crack Network: {n_junctions} junctions, {n_tips} tips, {n_edges} edges"
        
        ax.set_title(title, fontsize=14, weight='bold')
        ax.grid(True, alpha=0.3)
        
        # Set axis limits
        x_min, x_max = -self.domain_width/2 * 1e3, self.domain_width/2 * 1e3
        y_min, y_max = -self.domain_height/2 * 1e3, self.domain_height/2 * 1e3
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal', adjustable='box')
        
        # Force tick visibility
        import matplotlib.ticker as ticker
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8, integer=False))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=8, integer=False))
        ax.tick_params(axis='both', which='major', labelsize=10, length=6, width=1)
        ax.xaxis.set_tick_params(labelbottom=True)
        ax.yaxis.set_tick_params(labelleft=True)
        
        # Legend
        legend_elements = [
            Patch(facecolor='red', edgecolor='black', label='Tip (degree 1)'),
            Patch(facecolor='green', edgecolor='black', label='Kink (degree 2)'),
            Patch(facecolor='blue', edgecolor='black', label='Junction (degree 3+)')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.tight_layout()
        return fig, ax
    
    def visualize_with_paths(self, figsize: Tuple[float, float] = (12, 10)) -> Tuple[plt.Figure, plt.Axes]:
        """Visualize with different colors for each connected component (path)"""
        fig, ax = plt.subplots(figsize=figsize)
        
        pos = nx.get_node_attributes(self.G, 'pos')
        pos_mm = {k: (v[0]*1e3, v[1]*1e3) for k, v in pos.items()}
        
        components = list(nx.connected_components(self.G))
        
        # Assign colors
        colors = cm.tab20(np.linspace(0, 1, min(len(components), 20)))
        if len(components) > 20:
            colors = cm.hsv(np.linspace(0, 1, len(components)))
        
        # Draw each component
        for comp_idx, component in enumerate(components):
            subgraph = self.G.subgraph(component)
            
            nx.draw_networkx_edges(
                subgraph, pos_mm, width=2.0,
                edge_color=[colors[comp_idx]],
                alpha=0.8, ax=ax
            )
            
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
        
        # Set axis limits
        x_min, x_max = -self.domain_width/2 * 1e3, self.domain_width/2 * 1e3
        y_min, y_max = -self.domain_height/2 * 1e3, self.domain_height/2 * 1e3
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal', adjustable='box')
        
        # Force tick visibility
        import matplotlib.ticker as ticker
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8, integer=False))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=8, integer=False))
        ax.tick_params(axis='both', which='major', labelsize=10, length=6, width=1)
        ax.xaxis.set_tick_params(labelbottom=True)
        ax.yaxis.set_tick_params(labelleft=True)
        
        plt.tight_layout()
        return fig, ax
