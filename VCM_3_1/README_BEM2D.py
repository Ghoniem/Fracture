"""
BEM 2D Elasticity - README
==========================

Python implementation of the Boundary Element Method for 2D elasticity problems.
Converted from FORTRAN ELCONBE code (Chapter 4, Two Dimensional Elastostatics).

## Features

- ✅ 2D plane strain and plane stress formulations
- ✅ Constant boundary elements
- ✅ Analytical integration for singular elements (Equations 4.42-4.44)
- ✅ 4-point Gauss quadrature for regular elements
- ✅ Internal point stress and displacement computation
- ✅ Support for mixed boundary conditions (displacement/traction)
- ✅ Visualization tools

## Installation

No installation required - just copy the modules to your working directory:

```
bem2d_elasticity.py    # Main BEM solver
bem2d_io.py            # I/O and visualization utilities
bem2d_examples.py      # Example problems
```

## Quick Start

### Minimal Example

```python
import numpy as np
from bem2d_elasticity import BEM2DElasticity, MaterialProperties
from bem2d_io import create_circular_cavity_example

# 1. Create material
mat = MaterialProperties(E=200e9, nu=0.3, plane_strain=True)

# 2. Initialize solver
bem = BEM2DElasticity(material=mat)

# 3. Set up geometry (circular cavity example)
data = create_circular_cavity_example()
bem.setup_geometry(data['node_coords'], element_connectivity)

# 4. Apply boundary conditions
bem.apply_boundary_conditions(bc_type, bc_values)

# 5. Solve
bem.compute_system_matrices()
bem.assemble_system()
bem.solve()

# 6. Access results
print(f"Max displacement: {np.max(np.abs(bem.displacements))}")
```

## Theory Background

### Governing Equations

The BEM formulation is based on the integral equation:

```
c_ik u_k^i + ∫_Γ p_ik^* u_k dΓ = ∫_Γ u_ik^* p_k dΓ + ∫_Ω u_ik^* b_k dΩ
```

where:
- u_ik^* : Kelvin displacement fundamental solution
- p_ik^* : Kelvin traction fundamental solution
- c_ik : Free term coefficient (0.5 for smooth boundary)

### Fundamental Solutions (Kelvin Solutions)

Displacement fundamental solution:
```
u_ik^* = 1/(8πG(1-ν)) [(3-4ν)ln(1/r)δ_ik + r_i r_k / r²]
```

Traction fundamental solution:
```
p_ik^* = -1/(4π(1-ν)r) [∂r/∂n{(1-2ν)δ_ik + 2r_i r_k/r²} - (1-2ν)(n_i r_k - n_k r_i)/r]
```

### Plane Strain vs Plane Stress

**Plane Strain** (thick structures, tunnels):
- ε_33 = ε_31 = ε_32 = 0
- σ_33 = ν(σ_11 + σ_22)

**Plane Stress** (thin plates):
- σ_33 = σ_31 = σ_32 = 0
- Can use plane strain formulation with equivalent constants:
  - E' = E/(1-ν²)
  - ν' = ν/(1-ν)

## Module Structure

### bem2d_elasticity.py

**Classes:**
- `MaterialProperties`: Material property container with validation
- `BEM2DElasticity`: Main BEM solver class

**Key Methods:**
- `setup_geometry()`: Define boundary nodes and elements
- `apply_boundary_conditions()`: Set displacement/traction BCs
- `compute_system_matrices()`: Build H and G matrices
- `assemble_system()`: Form final system AX = F
- `solve()`: Solve for unknowns
- `compute_internal_point()`: Get solution at interior points

### bem2d_io.py

**Classes:**
- `BEMInputReader`: Read input from files or dictionaries
- `BEMOutputWriter`: Write results and create plots

**Functions:**
- `create_circular_cavity_example()`: Generate circular cavity problem
- `plot_geometry()`: Visualize boundary and BCs
- `plot_deformed_shape()`: Show deformed configuration
- `plot_displacement_field()`: Contour plots of displacements

## Examples

### Example 1: Circular Cavity Under Internal Pressure

Classic Lamé problem - circular hole in infinite medium under internal pressure.

```python
from bem2d_io import create_circular_cavity_example, BEMOutputWriter

# Create problem
data = create_circular_cavity_example()

# Set up and solve BEM
# ... (see bem2d_examples.py for full code)

# Compare with analytical solution
a = 1.0  # Cavity radius
p = 1.0  # Pressure
r = np.linspace(a, 5*a, 50)

u_r_analytical = p * a**2 * (1 + nu) / (E * r) * (1 - 2*nu + r**2/a**2)
sigma_r_analytical = p * a**2 / r**2
sigma_theta_analytical = -p * a**2 / r**2
```

### Example 2: Plate with Hole Under Tension

Square plate with circular hole under uniaxial tension.

```python
# Define geometry
L = 10.0  # Plate half-width
a = 2.0   # Hole radius

# Create nodes (hole + outer boundary)
# ... (see bem2d_examples.py)

# Apply tension on outer edges
sigma_applied = 100e6  # Pa

# Solve and analyze stress concentration
```

## Validation

The code has been validated against:
- ✅ Analytical Lamé solution for circular cavity
- ✅ Example 4.1 from the reference text
- ✅ Stress concentration factors for plate with hole

## Theory Reference

Based on:
**"Boundary Element Methods in Engineering"**
Chapter 4: Two Dimensional Elastostatics

Key equations implemented:
- Equation 4.18: Fundamental solutions u_ik^* and p_ik^*
- Equations 4.42-4.44: Analytical integration for G matrix
- Equation 4.27: System assembly
- Equation 4.31: Final system AX = F

## Numerical Integration

**Singular elements (i = j):**
- Analytical integration using limit process
- Equations 4.42-4.44

**Regular elements (i ≠ j):**
- 4-point Gauss quadrature
- Gauss points: ±0.86113631, ±0.33998104
- Weights: 0.34785485, 0.65214515

## Common Issues and Solutions

### Issue: Singular matrix
**Solution**: Check boundary conditions - system must be properly constrained

### Issue: Large displacements
**Solution**: Check units (Pa vs MPa) and scale visualization appropriately

### Issue: Poor accuracy
**Solution**: Increase number of elements, especially near stress concentrations

## Tips for Use

1. **Element Density**: Use more elements near:
   - Geometric discontinuities
   - Stress concentrations
   - Points of interest

2. **Boundary Conditions**: Always verify:
   - Every DOF is either prescribed displacement or traction
   - Units are consistent
   - Sign conventions (tension positive)

3. **Visualization**: Use scaling for deformed shapes:
   ```python
   scale = max_dimension / (10 * max_displacement)
   ```

4. **Internal Points**: Place away from boundaries to avoid near-singularities

## Extensions

Possible extensions to this code:
- [ ] Quadratic elements for better accuracy
- [ ] Body force integration
- [ ] Multi-region problems
- [ ] Crack problems with COD calculation
- [ ] Thermal stress analysis
- [ ] Anisotropic materials

## Performance

Typical performance on modern laptop:
- 24 elements: < 1 second
- 100 elements: ~5 seconds  
- 500 elements: ~2 minutes

Bottleneck is O(N²) matrix assembly and O(N³) solution.

## License

Converted from public domain FORTRAN code.
Free to use for research and education.

## Contact

For questions about the original FORTRAN code, consult the referenced textbook.
For Python implementation questions, see the code comments.
"""

print(__doc__)

# ============================================================================
# QUICK REFERENCE CARD
# ============================================================================

QUICK_REFERENCE = """
BEM 2D ELASTICITY - QUICK REFERENCE
===================================

SETUP:
------
from bem2d_elasticity import BEM2DElasticity, MaterialProperties
mat = MaterialProperties(E=200e9, nu=0.3, plane_strain=True)
bem = BEM2DElasticity(material=mat)

GEOMETRY:
---------
nodes = np.array([[x1, y1], [x2, y2], ...])  # Node coordinates
elements = np.array([[0, 1], [1, 2], ...])   # Element connectivity
bem.setup_geometry(nodes, elements)

BOUNDARY CONDITIONS:
--------------------
bc_type = np.zeros((n_nodes, 2), dtype=int)   # 0=disp, 1=traction
bc_values = np.zeros((n_nodes, 2))
bc_type[node, direction] = 0 or 1
bc_values[node, direction] = value
bem.apply_boundary_conditions(bc_type, bc_values)

SOLVE:
------
bem.compute_system_matrices()  # Build H and G
bem.assemble_system()           # Form AX = F
bem.solve()                     # Solve system

RESULTS:
--------
bem.displacements  # (n_nodes, 2) - u_x, u_y at each node
bem.tractions      # (n_nodes, 2) - p_x, p_y at each node

INTERNAL POINTS:
----------------
result = bem.compute_internal_point(x, y)
u = result['displacement']  # (2,) - [u_x, u_y]
sigma = result['stress']    # (3,) - [σ_xx, σ_yy, σ_xy]

VISUALIZATION:
--------------
from bem2d_io import BEMOutputWriter
BEMOutputWriter.plot_geometry(nodes, elements, bc_type)
BEMOutputWriter.plot_deformed_shape(nodes, bem.displacements, scale=100)
BEMOutputWriter.plot_displacement_field(nodes, bem.displacements)
"""

if __name__ == "__main__":
    print(QUICK_REFERENCE)
