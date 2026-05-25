# plot_stress_component.py - COMPLETE VERSION
import numpy as np
import matplotlib.pyplot as plt

def create_stress_contour_plot(filename, stress_component='sigma_yy', output_png=None):
    """Create detailed contour plot of a single stress component"""
    outpath = r'C:/Users/Owner/Documents/Repos/Fracture/MulticracksCpp/output'
    # Load data
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    grid_data = []
    crack_data = []
    mode = 'grid'
    
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        values = [float(x) for x in parts]
        
        if values[0] == -999:
            mode = 'crack'
            continue
        elif values[0] == -998:
            break
        
        if mode == 'grid':
            grid_data.append(values)
        elif mode == 'crack':
            crack_data.append(values)
    
    grid = np.array(grid_data)
    cracks = np.array(crack_data) if crack_data else np.array([])
    
    # Extract data
    x, y = grid[:, 0], grid[:, 1]
    sigma_xx, sigma_yy, sigma_xy = grid[:, 2], grid[:, 3], grid[:, 4]
    
    # Grid dimensions
    unique_x, unique_y = np.unique(x), np.unique(y)
    nx, ny = len(unique_x), len(unique_y)
    
    X, Y = x.reshape(ny, nx), y.reshape(ny, nx)
    
    # Select component
    if stress_component == 'sigma_xx':
        Z = sigma_xx.reshape(ny, nx) / 1e6
        label = r'$\sigma_{xx}$ (MPa)'
    elif stress_component == 'sigma_yy':
        Z = sigma_yy.reshape(ny, nx) / 1e6
        label = r'$\sigma_{yy}$ (MPa)'
    else:
        Z = sigma_xy.reshape(ny, nx) / 1e6
        label = r'$\sigma_{xy}$ (MPa)'
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    vmax = np.percentile(np.abs(Z), 98)
    vmin = -vmax
    levels = np.linspace(vmin, vmax, 40)
    
    contourf = ax.contourf(X*1e6, Y*1e6, Z, levels=levels, cmap='RdBu_r', extend='both')
    contours = ax.contour(X*1e6, Y*1e6, Z, levels=12, colors='black', linewidths=0.5, alpha=0.4)
    ax.clabel(contours, inline=True, fontsize=8, fmt='%1.0f')
    
    cbar = plt.colorbar(contourf, ax=ax, label=label, pad=0.02, shrink=0.9)
    
    # Plot cracks
    if len(cracks) > 0:
        for i, crack in enumerate(cracks):
            cx, cy, length, angle = crack
            dx, dy = length/2 * np.cos(angle), length/2 * np.sin(angle)
            x1, y1 = (cx - dx)*1e6, (cy - dy)*1e6
            x2, y2 = (cx + dx)*1e6, (cy + dy)*1e6
            
            ax.plot([x1, x2], [y1, y2], 'k-', linewidth=5, zorder=100)
            ax.plot([x1, x2], [y1, y2], 'w-', linewidth=2.5, zorder=101)
            ax.text(cx*1e6, cy*1e6, f'{i}', fontsize=10, ha='center',
                   color='black', weight='bold', zorder=102,
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    ax.set_xlabel('X (μm)', fontsize=13, weight='bold')
    ax.set_ylabel('Y (μm)', fontsize=13, weight='bold')
    ax.set_title(label, fontsize=15, weight='bold', pad=15)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.axhline(0, color='k', linewidth=0.5, alpha=0.3)
    ax.axvline(0, color='k', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    
    if output_png is None:
        output_png = f'{outpath}/{stress_component}_{filename.replace(".dat", ".png")}'
    
    plt.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_png}")
    plt.close()

if __name__ == '__main__':
    steps = [0, 2, 4, 6, 8, 10]
    outpath = r'C:/Users/Owner/Documents/Repos/Fracture/MulticracksCpp/output'
    print("="*70)
    print("Generating Stress Visualizations")
    print("="*70)
    
    for step in steps:
        filename = f'field_{step}.dat'
        try:
            print(f"\nStep {step}:")
            create_stress_contour_plot(filename, 'sigma_xx', f'sigma_xx_step{step}.png')
            create_stress_contour_plot(filename, 'sigma_yy', f'sigma_yy_step{step}.png')
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n" + "="*70)
    print("Complete! Check your directory for PNG files.")