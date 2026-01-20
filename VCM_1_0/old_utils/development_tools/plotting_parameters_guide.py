"""
Complete Plotting Parameters Reference
======================================
All available parameters for plot_network() and visualize()
"""

import numpy as np
import matplotlib.pyplot as plt
from crack_network_complete import CrackNetworkGenerator

mm = 1e-3

# Create a sample network
gen = CrackNetworkGenerator(domain_size=(20*mm, 20*mm))

vertices = np.array([
    [0,  0.0*mm,  0.0*mm],   # Junction
    [1,  5.0*mm,  5.0*mm],   # Tip
    [2, 10.0*mm,  0.0*mm],   # Tip
    [3,  5.0*mm, -5.0*mm],   # Tip
], dtype=float)

connectivity = [[0, 1], [0, 2], [0, 3]]
gen.load_from_arrays(vertices, connectivity, has_node_ids=True)

print("="*70)
print("AVAILABLE PLOTTING PARAMETERS")
print("="*70)

# =============================================================================
# Example 1: Default settings
# =============================================================================
print("\n1. DEFAULT SETTINGS:")
print("-" * 50)
fig, ax = gen.plot_network()
plt.savefig('/mnt/user-data/outputs/plot_default.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Default plot saved")

# =============================================================================
# Example 2: Custom node and edge styling
# =============================================================================
print("\n2. CUSTOM NODE AND EDGE STYLING:")
print("-" * 50)
fig, ax = gen.plot_network(
    node_size=20,           # Larger nodes (default: 10)
    edge_width=4,           # Thicker edges (default: 2)
    edge_color='purple',    # Different edge color (default: 'blue')
    edge_alpha=0.5,         # More transparent edges (default: 0.7)
    tip_color='orange',     # Custom tip color (default: 'red')
    junction_color='cyan',  # Custom junction color (default: 'green')
    internal_color='gray',  # Custom internal node color (default: 'black')
)
plt.savefig('/mnt/user-data/outputs/plot_styled.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Styled plot saved")
print("  Parameters used:")
print("    node_size=20, edge_width=4, edge_color='purple'")
print("    tip_color='orange', junction_color='cyan'")

# =============================================================================
# Example 3: Labels and indices
# =============================================================================
print("\n3. LABELS AND INDICES:")
print("-" * 50)
fig, ax = gen.plot_network(
    show_node_ids=True,        # Show node IDs (default: True)
    show_edge_labels=True,     # Show edge labels (default: False)
    show_vertex_indices=False, # Show internal indices (default: False)
    show_statistics=True,      # Show stats box (default: True)
)
plt.savefig('/mnt/user-data/outputs/plot_labels.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Labels plot saved")
print("  Parameters used:")
print("    show_node_ids=True, show_edge_labels=True")

# =============================================================================
# Example 4: Figure size and layout
# =============================================================================
print("\n4. FIGURE SIZE AND LAYOUT:")
print("-" * 50)
fig, ax = gen.plot_network(
    figsize=(12, 8),        # Custom figure size (default: (10, 10))
    show_statistics=False,  # Hide statistics box
)
plt.savefig('/mnt/user-data/outputs/plot_size.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Custom size plot saved")
print("  Parameters used:")
print("    figsize=(12, 8), show_statistics=False")

# =============================================================================
# Summary table
# =============================================================================
print("\n" + "="*70)
print("COMPLETE PARAMETER REFERENCE")
print("="*70)

params = {
    "Layout & Size": [
        ("figsize", "(10, 10)", "Figure size as (width, height)"),
    ],
    "Node Appearance": [
        ("node_size", "10", "Size of node markers"),
        ("tip_color", "'red'", "Color for tip nodes"),
        ("junction_color", "'green'", "Color for junction nodes"),
        ("internal_color", "'black'", "Color for internal nodes"),
    ],
    "Edge Appearance": [
        ("edge_width", "2", "Width of edge lines"),
        ("edge_color", "'blue'", "Color of edges"),
        ("edge_alpha", "0.7", "Transparency of edges (0-1)"),
    ],
    "Labels & Text": [
        ("show_node_ids", "True", "Show node IDs (if loaded with IDs)"),
        ("show_vertex_indices", "False", "Show internal array indices"),
        ("show_edge_labels", "False", "Show edge labels/indices"),
        ("show_labels", "False", "Show custom metadata labels"),
        ("show_statistics", "True", "Show statistics text box"),
    ],
}

for category, param_list in params.items():
    print(f"\n{category}:")
    print("-" * 70)
    for name, default, description in param_list:
        print(f"  {name:25s} = {default:15s}  # {description}")

print("\n" + "="*70)
print("USAGE EXAMPLES")
print("="*70)

print("\nBasic usage:")
print(">>> generator.plot_network()")
print(">>> generator.visualize()  # Same as plot_network()")

print("\nWith custom styling:")
print(">>> generator.plot_network(")
print("...     node_size=15,")
print("...     edge_width=3,")
print("...     edge_color='darkblue',")
print("...     tip_color='crimson',")
print("...     show_edge_labels=True")
print("... )")

print("\nMinimal plot (no labels, no stats):")
print(">>> generator.plot_network(")
print("...     show_node_ids=False,")
print("...     show_statistics=False")
print("... )")

print("\n" + "="*70)
print("ALL EXAMPLES COMPLETE")
print("="*70)
print("\nGenerated files:")
print("  - plot_default.png      (default settings)")
print("  - plot_styled.png       (custom colors/sizes)")
print("  - plot_labels.png       (with labels)")
print("  - plot_size.png         (custom figure size)")
