"""
BEM 2D Elasticity - Input/Output and Utilities
==============================================

Helper functions for reading input, writing output, and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional
from pathlib import Path


class BEMInputReader:
    """Read BEM input data from file or dictionary"""
    
    @staticmethod
    def read_from_file(filename: str) -> Dict:
        """
        Read BEM input from text file.
        
        File format (matching FORTRAN input):
        Line 1: Title
        Line 2: N, L, M, NC(1)...NC(M), GE, XNU
        Lines 3+: Element extreme points (x, y)
        Lines N+: Boundary conditions (prescribed values, node, direction, code)
        Lines M+: Internal points (x, y) if any
        
        Parameters
        ----------
        filename : str
            Path to input file
        
        Returns
        -------
        dict
            Dictionary with keys: title, n_elements, n_internal, material,
            node_coords, bc_data
        """
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        idx = 0
        
        # Title
        title = lines[idx].strip()
        idx += 1
        
        # Basic parameters
        params = lines[idx].split()
        n_elements = int(params[0])
        n_internal = int(params[1])  # L
        n_surfaces = int(params[2])  # M
        
        # Last nodes of each surface
        last_nodes = [int(params[3 + i]) for i in range(n_surfaces)]
        
        # Material properties
        G = float(params[3 + n_surfaces])
        nu = float(params[4 + n_surfaces])
        idx += 1
        
        # Read element extreme points
        node_coords = []
        for i in range(n_elements):
            coords = lines[idx].split()
            node_coords.append([float(coords[0]), float(coords[1])])
            idx += 1
        
        node_coords = np.array(node_coords)
        
        # Read boundary conditions
        bc_data = []
        for i in range(2 * n_elements):  # 2 values per node
            bc_line = lines[idx].split()
            value = float(bc_line[0])
            node = int(bc_line[1]) - 1  # Convert to 0-indexed
            direction = int(bc_line[2]) - 1  # Convert to 0-indexed
            bc_type = int(bc_line[3])
            bc_data.append((node, direction, bc_type, value))
            idx += 1
        
        # Read internal points if any
        internal_points = []
        if n_internal > 0:
            for i in range(n_internal):
                coords = lines[idx].split()
                internal_points.append([float(coords[0]), float(coords[1])])
                idx += 1
        
        internal_points = np.array(internal_points) if internal_points else None
        
        return {
            'title': title,
            'n_elements': n_elements,
            'n_internal': n_internal,
            'n_surfaces': n_surfaces,
            'last_nodes': last_nodes,
            'material': {'G': G, 'nu': nu},
            'node_coords': node_coords,
            'bc_data': bc_data,
            'internal_points': internal_points
        }
    
    @staticmethod
    def create_from_dict(data: Dict) -> Dict:
        """Create BEM input from dictionary (for programmatic setup)"""
        required_keys = ['node_coords', 'bc_data', 'material']
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key: {key}")
        
        return data


class BEMOutputWriter:
    """Write BEM results to file and create visualizations"""
    
    @staticmethod
    def write_results(filename: str,
                     node_coords: np.ndarray,
                     displacements: np.ndarray,
                     tractions: np.ndarray,
                     internal_coords: Optional[np.ndarray] = None,
                     internal_disp: Optional[np.ndarray] = None,
                     internal_stress: Optional[np.ndarray] = None):
        """
        Write BEM results to text file.
        
        Parameters
        ----------
        filename : str
            Output filename
        node_coords : np.ndarray
            Boundary node coordinates
        displacements : np.ndarray
            Boundary displacements
        tractions : np.ndarray
            Boundary tractions
        internal_coords : np.ndarray, optional
            Internal point coordinates
        internal_disp : np.ndarray, optional
            Internal point displacements
        internal_stress : np.ndarray, optional
            Internal point stresses
        """
        with open(filename, 'w') as f:
            f.write("=" * 79 + "\n")
            f.write("BEM 2D ELASTICITY RESULTS\n")
            f.write("=" * 79 + "\n\n")
            
            # Boundary nodes
            f.write("BOUNDARY NODES\n")
            f.write("-" * 79 + "\n")
            f.write(f"{'Node':>6} {'X':>12} {'Y':>12} {'u_x':>12} {'u_y':>12} "
                   f"{'p_x':>12} {'p_y':>12}\n")
            f.write("-" * 79 + "\n")
            
            for i in range(len(node_coords)):
                f.write(f"{i+1:6d} "
                       f"{node_coords[i,0]:12.5e} {node_coords[i,1]:12.5e} "
                       f"{displacements[i,0]:12.5e} {displacements[i,1]:12.5e} "
                       f"{tractions[i,0]:12.5e} {tractions[i,1]:12.5e}\n")
            
            # Internal points if provided
            if internal_coords is not None and internal_disp is not None:
                f.write("\n\nINTERNAL POINTS - DISPLACEMENTS\n")
                f.write("-" * 79 + "\n")
                f.write(f"{'Point':>6} {'X':>12} {'Y':>12} {'u_x':>12} {'u_y':>12}\n")
                f.write("-" * 79 + "\n")
                
                for i in range(len(internal_coords)):
                    f.write(f"{i+1:6d} "
                           f"{internal_coords[i,0]:12.5e} {internal_coords[i,1]:12.5e} "
                           f"{internal_disp[i,0]:12.5e} {internal_disp[i,1]:12.5e}\n")
            
            if internal_stress is not None:
                f.write("\n\nINTERNAL POINTS - STRESSES\n")
                f.write("-" * 79 + "\n")
                f.write(f"{'Point':>6} {'X':>12} {'Y':>12} "
                       f"{'σ_xx':>12} {'σ_yy':>12} {'σ_xy':>12}\n")
                f.write("-" * 79 + "\n")
                
                for i in range(len(internal_coords)):
                    f.write(f"{i+1:6d} "
                           f"{internal_coords[i,0]:12.5e} {internal_coords[i,1]:12.5e} "
                           f"{internal_stress[i,0]:12.5e} {internal_stress[i,1]:12.5e} "
                           f"{internal_stress[i,2]:12.5e}\n")
    
    @staticmethod
    def plot_geometry(node_coords: np.ndarray,
                     element_nodes: Optional[np.ndarray] = None,
                     bc_type: Optional[np.ndarray] = None,
                     figsize: Tuple[float, float] = (10, 8)):
        """
        Plot boundary geometry and boundary conditions.
        
        Parameters
        ----------
        node_coords : np.ndarray
            Node coordinates
        element_nodes : np.ndarray, optional
            Element connectivity
        bc_type : np.ndarray, optional
            Boundary condition types (for coloring)
        figsize : tuple
            Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot boundary
        n = len(node_coords)
        for i in range(n):
            i_next = (i + 1) % n
            ax.plot([node_coords[i, 0], node_coords[i_next, 0]],
                   [node_coords[i, 1], node_coords[i_next, 1]],
                   'b-', linewidth=2)
        
        # Plot nodes
        if bc_type is not None:
            # Color by BC type
            disp_nodes = np.where(bc_type[:, 0] == 0)[0]
            trac_nodes = np.where(bc_type[:, 0] == 1)[0]
            
            if len(disp_nodes) > 0:
                ax.scatter(node_coords[disp_nodes, 0], node_coords[disp_nodes, 1],
                          c='red', s=50, label='Displacement BC', zorder=5)
            if len(trac_nodes) > 0:
                ax.scatter(node_coords[trac_nodes, 0], node_coords[trac_nodes, 1],
                          c='green', s=50, label='Traction BC', zorder=5)
            ax.legend()
        else:
            ax.scatter(node_coords[:, 0], node_coords[:, 1],
                      c='blue', s=50, zorder=5)
        
        # Add node numbers
        for i, (x, y) in enumerate(node_coords):
            ax.text(x, y, f'  {i+1}', fontsize=8, ha='left')
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('BEM Boundary Geometry')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        
        return fig, ax
    
    @staticmethod
    def plot_deformed_shape(node_coords: np.ndarray,
                          displacements: np.ndarray,
                          scale: float = 1.0,
                          figsize: Tuple[float, float] = (10, 8)):
        """
        Plot deformed and undeformed shapes.
        
        Parameters
        ----------
        node_coords : np.ndarray
            Original node coordinates
        displacements : np.ndarray
            Node displacements
        scale : float
            Displacement scaling factor for visualization
        figsize : tuple
            Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        n = len(node_coords)
        
        # Undeformed shape
        for i in range(n):
            i_next = (i + 1) % n
            ax.plot([node_coords[i, 0], node_coords[i_next, 0]],
                   [node_coords[i, 1], node_coords[i_next, 1]],
                   'b--', linewidth=1, alpha=0.5, label='Undeformed' if i == 0 else '')
        
        # Deformed shape
        deformed_coords = node_coords + scale * displacements
        for i in range(n):
            i_next = (i + 1) % n
            ax.plot([deformed_coords[i, 0], deformed_coords[i_next, 0]],
                   [deformed_coords[i, 1], deformed_coords[i_next, 1]],
                   'r-', linewidth=2, label='Deformed' if i == 0 else '')
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Deformed Shape (scale = {scale})')
        ax.legend()
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        
        return fig, ax
    
    @staticmethod
    def plot_displacement_field(node_coords: np.ndarray,
                               displacements: np.ndarray,
                               figsize: Tuple[float, float] = (12, 5)):
        """
        Plot displacement components as contours.
        
        Parameters
        ----------
        node_coords : np.ndarray
            Node coordinates
        displacements : np.ndarray
            Displacement vectors
        figsize : tuple
            Figure size
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # u_x displacement
        sc1 = ax1.scatter(node_coords[:, 0], node_coords[:, 1],
                         c=displacements[:, 0], cmap='RdBu_r', s=100)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title('u_x Displacement')
        ax1.axis('equal')
        plt.colorbar(sc1, ax=ax1, label='u_x')
        
        # u_y displacement
        sc2 = ax2.scatter(node_coords[:, 0], node_coords[:, 1],
                         c=displacements[:, 1], cmap='RdBu_r', s=100)
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_title('u_y Displacement')
        ax2.axis('equal')
        plt.colorbar(sc2, ax=ax2, label='u_y')
        
        plt.tight_layout()
        return fig, (ax1, ax2)


def create_circular_cavity_example():
    """
    Create input data for circular cavity under internal pressure example.
    
    This matches Example 4.1 from the text - circular cavity under 
    internal pressure in an infinite medium.
    """
    # Geometry: circular cavity with radius a
    a = 1.0  # Cavity radius
    n_elements = 24  # Number of elements
    
    # Generate nodes around circle
    theta = np.linspace(0, 2*np.pi, n_elements + 1)[:-1]  # Exclude last = first
    node_coords = np.zeros((n_elements, 2))
    node_coords[:, 0] = a * np.cos(theta)
    node_coords[:, 1] = a * np.sin(theta)
    
    # Material properties (from example)
    E = 94500.0  # Young's modulus
    nu = 0.1  # Poisson's ratio
    G = E / (2.0 * (1.0 + nu))
    
    # Boundary conditions: internal pressure p = 1.0
    # All nodes have prescribed traction (pressure normal to surface)
    p_internal = 1.0
    
    bc_data = []
    for i in range(n_elements):
        # Normal vector (pointing inward for cavity)
        normal_x = -np.cos(theta[i])
        normal_y = -np.sin(theta[i])
        
        # Traction = pressure * normal
        tx = p_internal * normal_x
        ty = p_internal * normal_y
        
        # BC type 1 = traction prescribed
        bc_data.append((i, 0, 1, tx))  # x-direction
        bc_data.append((i, 1, 1, ty))  # y-direction
    
    # Internal points for stress evaluation
    r_internal = np.array([0.5, 1.5, 2.0, 3.0, 4.0]) * a
    internal_points = np.column_stack([r_internal, np.zeros_like(r_internal)])
    
    return {
        'title': 'Circular Cavity Under Internal Pressure (24 elements)',
        'n_elements': n_elements,
        'n_internal': len(internal_points),
        'n_surfaces': 1,
        'last_nodes': [n_elements],
        'material': {'E': E, 'nu': nu, 'G': G},
        'node_coords': node_coords,
        'bc_data': bc_data,
        'internal_points': internal_points
    }
