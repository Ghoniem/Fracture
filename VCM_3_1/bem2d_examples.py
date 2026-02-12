"""
Example Jupyter Notebook Cells for BEM 2D Elasticity
====================================================

Copy these cells into your Jupyter notebook to use the BEM solver.
"""

# ============================================================================
# CELL 1: Imports and Setup
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
from bem2d_elasticity import BEM2DElasticity, MaterialProperties
from bem2d_io import BEMInputReader, BEMOutputWriter, create_circular_cavity_example

# Set up matplotlib
%matplotlib inline
plt.rcParams['figure.dpi'] = 100

mm = 1e-3  # Millimeter scale if needed

print("✓ BEM 2D Elasticity modules loaded")

# ============================================================================
# CELL 2: Example 1 - Circular Cavity Under Internal Pressure
# ============================================================================

print("="*70)
print("EXAMPLE: Circular Cavity Under Internal Pressure")
print("="*70)

# Create example problem
data = create_circular_cavity_example()
print(f"\n{data['title']}")
print(f"  Elements: {data['n_elements']}")
print(f"  Material: E = {data['material']['E']:.1f}, ν = {data['material']['nu']}")

# Create material
mat = MaterialProperties(
    E=data['material']['E'],
    nu=data['material']['nu'],
    plane_strain=True
)

# Initialize BEM solver
bem = BEM2DElasticity(material=mat)

# Set up geometry
node_coords = data['node_coords']
n_nodes = len(node_coords)

# Create element connectivity (circular, nodes connect sequentially)
element_nodes = np.array([[i, (i+1) % n_nodes] for i in range(n_nodes)])

bem.setup_geometry(node_coords, element_nodes)

# Set up boundary conditions
bc_type = np.zeros((n_nodes, 2), dtype=int)
bc_values = np.zeros((n_nodes, 2))

# Process BC data
for node, direction, bc_code, value in data['bc_data']:
    bc_type[node, direction] = bc_code
    bc_values[node, direction] = value

bem.apply_boundary_conditions(bc_type, bc_values)

# Visualize geometry
fig, ax = BEMOutputWriter.plot_geometry(node_coords, element_nodes, bc_type)
plt.show()

# ============================================================================
# CELL 3: Solve the BEM System
# ============================================================================

# Compute influence matrices
bem.compute_system_matrices()

# Assemble system
bem.assemble_system()

# Solve
bem.solve()

print("\n" + "="*70)
print("SOLUTION SUMMARY")
print("="*70)
print(f"\nBoundary Solution:")
print(f"  Max displacement: {np.max(np.abs(bem.displacements)):.6e}")
print(f"  Max traction: {np.max(np.abs(bem.tractions)):.6e}")

# ============================================================================
# CELL 4: Visualize Results
# ============================================================================

# Plot deformed shape
fig, ax = BEMOutputWriter.plot_deformed_shape(
    node_coords, bem.displacements, scale=1000.0
)
plt.show()

# Plot displacement field
fig, axes = BEMOutputWriter.plot_displacement_field(
    node_coords, bem.displacements
)
plt.show()

# ============================================================================
# CELL 5: Compute Internal Points (if applicable)
# ============================================================================

if 'internal_points' in data and data['internal_points'] is not None:
    internal_pts = data['internal_points']
    n_internal = len(internal_pts)
    
    print(f"\nComputing solution at {n_internal} internal points...")
    
    internal_disp = np.zeros((n_internal, 2))
    internal_stress = np.zeros((n_internal, 3))
    
    for i, (x, y) in enumerate(internal_pts):
        result = bem.compute_internal_point(x, y)
        internal_disp[i] = result['displacement']
        internal_stress[i] = result['stress']
    
    print("✓ Internal point solution computed")
    
    # Print results
    print("\nInternal Point Results:")
    print("-" * 70)
    print(f"{'Point':>6} {'r':>10} {'u_r':>12} {'σ_rr':>12} {'σ_θθ':>12}")
    print("-" * 70)
    
    for i, (x, y) in enumerate(internal_pts):
        r = np.sqrt(x**2 + y**2)
        u_r = np.sqrt(internal_disp[i, 0]**2 + internal_disp[i, 1]**2)
        print(f"{i+1:6d} {r:10.4f} {u_r:12.6e} "
              f"{internal_stress[i, 0]:12.6e} {internal_stress[i, 1]:12.6e}")

# ============================================================================
# CELL 6: Save Results
# ============================================================================

# Write results to file
BEMOutputWriter.write_results(
    'bem_results.txt',
    node_coords=bem.node_coords,
    displacements=bem.displacements,
    tractions=bem.tractions
)

print("\n✓ Results saved to: bem_results.txt")

# ============================================================================
# CELL 7: Example 2 - Custom Problem (Plate with Hole)
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE: Square Plate with Circular Hole")
print("="*70)

# Geometry parameters
L = 10.0  # Plate half-width
a = 2.0   # Hole radius
n_hole = 20  # Elements on hole
n_edge = 10  # Elements per edge

# Material
E_plate = 200e9  # Steel, Pa
nu_plate = 0.3

mat_plate = MaterialProperties(E=E_plate, nu=nu_plate, plane_strain=False)

# Create nodes (hole + outer boundary)
# Hole (inner boundary)
theta_hole = np.linspace(0, 2*np.pi, n_hole + 1)[:-1]
nodes_hole = np.column_stack([
    a * np.cos(theta_hole),
    a * np.sin(theta_hole)
])

# Outer boundary (square)
nodes_outer = []
# Bottom edge
nodes_outer.extend([[-L + i*2*L/n_edge, -L] for i in range(n_edge)])
# Right edge
nodes_outer.extend([[L, -L + i*2*L/n_edge] for i in range(n_edge)])
# Top edge
nodes_outer.extend([[L - i*2*L/n_edge, L] for i in range(n_edge)])
# Left edge
nodes_outer.extend([[-L, L - i*2*L/n_edge] for i in range(n_edge)])

nodes_outer = np.array(nodes_outer)

# Combine all nodes
all_nodes = np.vstack([nodes_hole, nodes_outer])
n_total = len(all_nodes)

# Initialize BEM
bem_plate = BEM2DElasticity(material=mat_plate)

# Element connectivity
elements = np.array([[i, (i+1) % n_total] for i in range(n_total)])

bem_plate.setup_geometry(all_nodes, elements)

# Boundary conditions
# Hole: traction-free
# Outer edges: tension in x-direction
sigma_applied = 100e6  # 100 MPa tension

bc_type_plate = np.zeros((n_total, 2), dtype=int)
bc_values_plate = np.zeros((n_total, 2))

# Hole: traction-free (already zeros)
for i in range(n_hole):
    bc_type_plate[i, :] = 1  # Traction BC
    bc_values_plate[i, :] = 0.0

# Outer boundary: apply tension
for i in range(n_hole, n_total):
    bc_type_plate[i, :] = 1  # Traction BC
    
    # Normal traction based on position
    node = all_nodes[i]
    
    # Determine which edge
    if abs(node[0] - L) < 1e-6:  # Right edge
        bc_values_plate[i, 0] = sigma_applied  # σ_xx
    elif abs(node[0] + L) < 1e-6:  # Left edge
        bc_values_plate[i, 0] = -sigma_applied
    # Top and bottom: free

bem_plate.apply_boundary_conditions(bc_type_plate, bc_values_plate)

# Visualize
fig, ax = BEMOutputWriter.plot_geometry(all_nodes, elements, bc_type_plate, figsize=(8, 8))
plt.title('Plate with Hole - Geometry and BCs')
plt.show()

print("\n✓ Plate with hole problem set up")
print("  Ready to solve with: bem_plate.compute_system_matrices()")

# ============================================================================
# CELL 8: Analytical Solution Comparison (Circular Cavity)
# ============================================================================

def analytical_circular_cavity(r, a, p, E, nu):
    """
    Analytical solution for circular cavity under internal pressure.
    
    Lamé solution for infinite medium with cavity.
    
    Parameters
    ----------
    r : float or array
        Radial distance from center
    a : float
        Cavity radius
    p : float
        Internal pressure
    E : float
        Young's modulus
    nu : float
        Poisson's ratio
    
    Returns
    -------
    u_r : float or array
        Radial displacement
    sigma_r : float or array
        Radial stress
    sigma_theta : float or array
        Hoop stress
    """
    u_r = p * a**2 * (1 + nu) / (E * r) * (1 - 2*nu + r**2 / a**2)
    sigma_r = p * a**2 / r**2
    sigma_theta = -p * a**2 / r**2
    
    return u_r, sigma_r, sigma_theta

# Compare with BEM solution (for circular cavity example)
if data['title'].startswith('Circular'):
    a_cavity = 1.0
    p_internal = 1.0
    
    # Radial positions
    r_vals = np.linspace(a_cavity, 5*a_cavity, 50)
    
    # Analytical solution
    u_r_analytical, sigma_r_analytical, sigma_theta_analytical = \
        analytical_circular_cavity(r_vals, a_cavity, p_internal, 
                                  mat.E, mat.nu)
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(r_vals / a_cavity, u_r_analytical, 'b-', linewidth=2, 
            label='Analytical')
    ax1.set_xlabel('r / a')
    ax1.set_ylabel('u_r')
    ax1.set_title('Radial Displacement')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(r_vals / a_cavity, sigma_r_analytical, 'r-', linewidth=2, 
            label='σ_r (Analytical)')
    ax2.plot(r_vals / a_cavity, sigma_theta_analytical, 'b-', linewidth=2, 
            label='σ_θ (Analytical)')
    ax2.set_xlabel('r / a')
    ax2.set_ylabel('Stress')
    ax2.set_title('Stresses')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# CELL 9: Parameter Study Helper
# ============================================================================

def run_bem_parameter_study(radius_vals, pressure_vals, n_elements=24):
    """
    Run BEM for different parameter combinations.
    
    Parameters
    ----------
    radius_vals : array-like
        Cavity radii to test
    pressure_vals : array-like
        Internal pressures to test
    n_elements : int
        Number of boundary elements
    
    Returns
    -------
    results : dict
        Dictionary with results for each combination
    """
    results = {}
    
    for a in radius_vals:
        for p in pressure_vals:
            print(f"\nRunning: a={a:.2f}, p={p:.2e}")
            
            # Create problem
            # [Implementation similar to create_circular_cavity_example]
            # ...
            
            # Solve
            # ...
            
            results[(a, p)] = {
                'displacements': None,  # bem.displacements
                'tractions': None,      # bem.tractions
                'max_disp': None,       # np.max(np.abs(bem.displacements))
            }
    
    return results

# Example usage:
# results = run_bem_parameter_study(
#     radius_vals=[0.5, 1.0, 2.0],
#     pressure_vals=[1e5, 5e5, 1e6]
# )

print("\n" + "="*70)
print("BEM 2D Elasticity Examples Complete!")
print("="*70)
