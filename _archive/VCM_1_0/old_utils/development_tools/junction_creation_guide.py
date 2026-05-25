"""
Guide: Generating Crack Networks with Junctions at Intersections
=================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from crack_network_complete import CrackNetworkGenerator

mm = 1e-3

print("="*70)
print("GENERATING NETWORKS WITH JUNCTIONS AT INTERSECTIONS")
print("="*70)

# =============================================================================
# KEY INSIGHT: When using n_crack_paths, set allow_intersections=True
# =============================================================================

print("\n⚠️  IMPORTANT: Parameters that affect junction creation")
print("-"*70)
print("When using n_crack_paths:")
print("  • allow_intersections=True  → Detects intersections, creates junctions")
print("  • allow_intersections=False → Keeps cracks separate (no junctions)")
print()
print("Parameters IGNORED when using n_crack_paths:")
print("  • n_junctions (only used in traditional generation)")
print("  • n_tips (only used in traditional generation)")

# =============================================================================
# Example 1: Network WITH junctions (allow_intersections=True)
# =============================================================================

print("\n" + "="*70)
print("EXAMPLE 1: Network WITH Junctions")
print("="*70)

gen1 = CrackNetworkGenerator(domain_size=(25*mm, 25*mm), seed=42)

gen1.generate_network(
    edge_length_mean=5*mm,
    edge_length_std=2*mm,
    min_edge_length=2*mm,
    max_edge_length=10*mm,
    
    n_crack_paths=6,  # More paths = more chance of intersections
    segments_per_path_mean=4,
    segments_per_path_std=2,
    
    path_angle_change_mean_deg=0.0,
    path_angle_change_std_deg=30.0,
    
    allow_intersections=True,  # ✓ Enable junction creation
    
    spatial_distribution='clustered'  # Clustered = more intersections
)

stats1 = gen1.get_statistics()
print(f"\nResults:")
print(f"  Vertices: {stats1['n_vertices']}")
print(f"  Tips: {stats1['n_tips']}")
print(f"  Junctions: {stats1['n_junctions']} ✓")
print(f"  Edges: {stats1['n_edges']}")

fig1, ax1 = gen1.plot_network(figsize=(10, 10))
plt.savefig('/mnt/user-data/outputs/example_with_junctions.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✓ Saved: example_with_junctions.png")

# =============================================================================
# Example 2: Network WITHOUT junctions (allow_intersections=False)
# =============================================================================

print("\n" + "="*70)
print("EXAMPLE 2: Network WITHOUT Junctions")
print("="*70)

gen2 = CrackNetworkGenerator(domain_size=(25*mm, 25*mm), seed=42)

gen2.generate_network(
    edge_length_mean=5*mm,
    edge_length_std=2*mm,
    min_edge_length=2*mm,
    max_edge_length=10*mm,
    
    n_crack_paths=6,
    segments_per_path_mean=4,
    segments_per_path_std=2,
    
    path_angle_change_mean_deg=0.0,
    path_angle_change_std_deg=30.0,
    
    allow_intersections=False,  # ✗ No junctions
    min_edge_distance=1.0*mm,    # Keep cracks separated
    
    spatial_distribution='clustered'
)

stats2 = gen2.get_statistics()
print(f"\nResults:")
print(f"  Vertices: {stats2['n_vertices']}")
print(f"  Tips: {stats2['n_tips']}")
print(f"  Junctions: {stats2['n_junctions']} (separate paths)")
print(f"  Edges: {stats2['n_edges']}")

fig2, ax2 = gen2.plot_network(figsize=(10, 10))
plt.savefig('/mnt/user-data/outputs/example_without_junctions.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✓ Saved: example_without_junctions.png")

# =============================================================================
# Tips for maximizing junctions
# =============================================================================

print("\n" + "="*70)
print("TIPS FOR MAXIMIZING JUNCTIONS")
print("="*70)

tips = [
    ("More paths", "n_crack_paths=8-10", "More paths = more opportunities to intersect"),
    ("Clustered distribution", "spatial_distribution='clustered'", "Paths start closer together"),
    ("Longer paths", "segments_per_path_mean=5-8", "Longer paths cover more area"),
    ("Less meandering", "path_angle_change_std_deg=15-20", "Straighter paths cross more easily"),
    ("Allow intersections", "allow_intersections=True", "Required for junction creation"),
]

print()
for tip, param, explanation in tips:
    print(f"✓ {tip:<25} {param:<40} {explanation}")

# =============================================================================
# Recommended settings
# =============================================================================

print("\n" + "="*70)
print("RECOMMENDED SETTINGS FOR JUNCTION-RICH NETWORKS")
print("="*70)

print("""
generator.generate_network(
    # Edge properties
    edge_length_mean=5*mm,
    edge_length_std=2*mm,
    min_edge_length=2*mm,
    max_edge_length=10*mm,
    
    # Many long paths
    n_crack_paths=8,                    # More paths
    segments_per_path_mean=6,           # Longer paths
    segments_per_path_std=2,
    
    # Moderate meandering
    path_angle_change_mean_deg=0.0,
    path_angle_change_std_deg=20.0,     # Less random turning
    
    # Enable intersections
    allow_intersections=True,           # ✓ Creates junctions!
    
    # Clustered for more intersections
    spatial_distribution='clustered'    # Paths start closer
)
""")

print("="*70)
print("GUIDE COMPLETE")
print("="*70)

print("\nVisualization key:")
print("  🔴 Red circles  = Tips (degree 1)")
print("  🟢 Green squares = Junctions (degree > 2)")
print("  ⚫ Black dots   = Internal nodes (degree 2)")
