# Crack Network Generator Package

A comprehensive Python package for generating and analyzing crack networks in materials science applications.

## Features

- **Flexible Network Generation**: Create crack networks with controllable topology
- **Geometric Constraints**: Control intersections, angles, and spacing
- **Connected Paths**: Generate meandering crack paths with realistic behavior
- **Intersection Detection**: Automatically detect and create junctions at crossings
- **Network Analysis**: Compute statistical and topological properties
- **Visualization**: Multiple visualization modes for different analysis needs

## Package Structure

```
crack_network/
├── __init__.py          # Package initialization and exports
├── generator.py         # Main CrackNetworkGenerator class
├── geometry.py          # Geometric utilities (intersections, distances)
├── topology.py          # Network topology operations
├── analysis.py          # Statistical analysis tools
└── visualization.py     # Plotting and visualization
```

## Installation

```bash
cd crack_network_package
pip install -e .
```

## Quick Start

```python
from crack_network import CrackNetworkGenerator
import matplotlib.pyplot as plt

# Create generator
mm = 1e-3
generator = CrackNetworkGenerator(
    domain_size=(25*mm, 25*mm),
    seed=42
)

# Generate network
generator.generate_network(
    n_junctions=5,
    n_tips=20,
    edge_length_mean=3*mm,
    edge_length_std=1*mm,
    allow_intersections=False,
    min_junction_angle_deg=45.0
)

# Analyze
generator.analyzer.print_statistics()

# Visualize
fig, ax = generator.visualizer.visualize()
plt.show()

# Export
vertices, connectivity, types = generator.analyzer.to_arrays()
```

## Advanced Usage

### Intersection Detection

```python
generator.generate_network(
    ...
    allow_intersections=True,      # Allow crossings during generation
    detect_intersections=True,      # Create junctions at intersections
)
```

### Meandering Crack Paths

```python
generator.generate_network(
    n_crack_paths=5,
    segments_per_path_mean=10,
    path_angle_change_std_deg=20.0,  # Control meandering
    ...
)
```

## Modules

### `generator.py`
Main class for generating crack networks with comprehensive parameter control.

### `geometry.py`
Geometric algorithms:
- Line segment intersection detection
- Distance calculations
- Angle computations

### `topology.py`
Network topology operations:
- Intersection detection and splitting
- Junction management
- Connectivity analysis

### `analysis.py`
Statistical analysis:
- Network statistics
- Crack density
- Connectivity metrics
- Path analysis

### `visualization.py`
Visualization tools:
- Standard network plots
- Path-colored views
- Customizable styling

## Contributing

This package is under active development. Future additions may include:
- Hierarchical crack systems
- Physically-motivated growth models
- FEM mesh export
- Percolation analysis
- Fractal dimension calculations

## Authors

Nasr & AI Assistant, 2025

## License

MIT License
