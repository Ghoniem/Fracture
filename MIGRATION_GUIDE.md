# Migration Guide

## From Monolithic to Modular Structure

The crack network generator has been refactored into a modular package structure.

### Key Changes

**Old (monolithic):**
```python
from crack_network_generator import CrackNetworkGenerator

generator = CrackNetworkGenerator(...)
generator.generate_network(...)
generator.print_statistics()
fig, ax = generator.visualize()
```

**New (modular):**
```python
from crack_network import CrackNetworkGenerator

generator = CrackNetworkGenerator(...)
generator.generate_network(...)
generator.analyzer.print_statistics()
fig, ax = generator.visualizer.visualize()
```

### Module Separation

| Functionality | Old Location | New Location |
|--------------|--------------|--------------|
| Network generation | CrackNetworkGenerator | crack_network.generator |
| Geometric utils | CrackNetworkGenerator methods | crack_network.geometry |
| Topology operations | CrackNetworkGenerator methods | crack_network.topology |
| Statistics | CrackNetworkGenerator.print_statistics() | crack_network.analysis |
| Visualization | CrackNetworkGenerator.visualize() | crack_network.visualization |

### Backward Compatibility

For now, the complete monolithic version is preserved in `generator_full.py.bak`.

To maintain full backward compatibility, use wrapper methods in generator.py that delegate to the appropriate modules.
