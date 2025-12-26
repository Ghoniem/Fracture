# crack_simulation_python.py - Complete Python implementation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow
import os

# Output directory
outpath = r'C:/Users/Owner/Documents/Repos/Fracture/MulticracksCpp/output'

# Create output directory if it doesn't exist
os.makedirs(outpath, exist_ok=True)

class CrackSimulation:
    def __init__(self):
        self.PI = np.pi
        self.nu = 0.3  # Poisson's ratio
        self.G = 80e9  # Shear modulus (Pa)
        self.b = 2.5e-10  # Burgers vector (m)
        
        self.dislocations = []
        self.cracks = []
        self.external_stress = {'sxx': 0, 'syy': 0, 'sxy': 0}
    
    def set_external_stress(self, sxx, syy=0, sxy=0):
        self.external_stress = {'sxx': sxx, 'syy': syy, 'sxy': sxy}
    
    def initialize_cracks(self, num_cracks, min_len, max_len, domain_size, seed=12345):
        np.random.seed(seed)
        
        for i in range(num_cracks):
            crack = {
                'center': np.random.uniform(-domain_size/2, domain_size/2, 2),
                'length': np.random.uniform(min_len, max_len),
                'angle': np.random.uniform(0, 2*self.PI),
                'dipole_indices': []
            }
            crack['resistance'] = 50e6 / np.sqrt(self.PI * crack['length'] / 2.0)
            self.cracks.append(crack)
    
    def create_dipoles(self, n_per_crack):
        disl_idx = 0
        for crack_id, crack in enumerate(self.cracks):
            spacing = crack['length'] / (n_per_crack + 1)
            crack_dir = np.array([np.cos(crack['angle']), np.sin(crack['angle'])])
            normal = np.array([-np.sin(crack['angle']), np.cos(crack['angle'])])
            
            for i in range(n_per_crack):
                s = (i + 1) * spacing - crack['length'] / 2.0
                base_pos = crack['center'] + s * crack_dir
                sep = 0.1 * self.b
                
                # Positive dislocation
                pos_disl = {
                    'pos': base_pos + normal * (sep/2),
                    'b': normal * self.b,
                    'crack_id': crack_id,
                    'is_positive': True
                }
                
                # Negative dislocation
                neg_disl = {
                    'pos': base_pos - normal * (sep/2),
                    'b': -normal * self.b,
                    'crack_id': crack_id,
                    'is_positive': False
                }
                
                crack['dipole_indices'].append(disl_idx)
                self.dislocations.append(pos_disl)
                self.dislocations.append(neg_disl)
                disl_idx += 1
    
    def stress_from_dislocation(self, point, disl):
        rel = point - disl['pos']
        x, y = rel[0], rel[1]
        r2 = x**2 + y**2
        
        if r2 < 0.25 * self.b**2:
            return {'sxx': 0, 'syy': 0, 'sxy': 0}
        
        D = self.G / (2.0 * self.PI * (1.0 - self.nu))
        r4 = r2 * r2
        bx, by = disl['b'][0], disl['b'][1]
        
        sxx = -D * by * y * (3*x**2 + y**2) / r4 + D * bx * x * (x**2 - y**2) / r4
        syy = D * by * y * (x**2 - y**2) / r4 - D * bx * x * (x**2 + 3*y**2) / r4
        sxy = D * by * x * (x**2 - y**2) / r4 - D * bx * y * (x**2 - y**2) / r4
        
        return {'sxx': sxx, 'syy': syy, 'sxy': sxy}
    
    def total_stress_at(self, point):
        total = self.external_stress.copy()
        
        for disl in self.dislocations:
            s = self.stress_from_dislocation(point, disl)
            total['sxx'] += s['sxx']
            total['syy'] += s['syy']
            total['sxy'] += s['sxy']
        
        return total
    
    def peach_koehler_force(self, disl):
        sigma = self.total_stress_at(disl['pos'])
        
        sigma_b_x = sigma['sxx'] * disl['b'][0] + sigma['sxy'] * disl['b'][1]
        sigma_b_y = sigma['sxy'] * disl['b'][0] + sigma['syy'] * disl['b'][1]
        
        return np.array([sigma_b_y, -sigma_b_x])
    
    def solve_equilibrium(self, max_iter=500):
        damping = 0.1
        tol = 1e-6 * self.G * self.b
        
        for iteration in range(max_iter):
            forces = [self.peach_koehler_force(disl) for disl in self.dislocations]
            max_force = max([np.linalg.norm(f) for f in forces])
            
            if max_force < tol:
                print(f"  Converged in {iteration} iterations")
                return True
            
            dt = 0.01 * self.b / (max_force / damping + 1e-10)
            for i, disl in enumerate(self.dislocations):
                disl['pos'] = disl['pos'] + forces[i] * (dt / damping)
        
        print(f"  Warning: max iterations reached")
        return False
    
    def check_propagation(self, crack_id):
        crack = self.cracks[crack_id]
        if not crack['dipole_indices']:
            return False
        
        tip_dipole_idx = crack['dipole_indices'][-1]
        tip_disl = self.dislocations[tip_dipole_idx * 2]
        
        force = self.peach_koehler_force(tip_disl)
        crack_dir = np.array([np.cos(crack['angle']), np.sin(crack['angle'])])
        force_mag = np.dot(force, crack_dir)
        
        return force_mag > crack['resistance']
    
    def propagate_crack(self, crack_id, extension):
        crack = self.cracks[crack_id]
        old_len = crack['length']
        crack['length'] += extension
        crack['resistance'] = 50e6 / np.sqrt(self.PI * crack['length'] / 2.0)
        
        crack_dir = np.array([np.cos(crack['angle']), np.sin(crack['angle'])])
        normal = np.array([-np.sin(crack['angle']), np.cos(crack['angle'])])
        new_pos = crack['center'] + crack_dir * (crack['length'] / 2)
        
        sep = 0.1 * self.b
        
        pos_disl = {
            'pos': new_pos + normal * (sep/2),
            'b': normal * self.b,
            'crack_id': crack_id,
            'is_positive': True
        }
        
        neg_disl = {
            'pos': new_pos - normal * (sep/2),
            'b': -normal * self.b,
            'crack_id': crack_id,
            'is_positive': False
        }
        
        crack['dipole_indices'].append(len(self.dislocations) // 2)
        self.dislocations.append(pos_disl)
        self.dislocations.append(neg_disl)
        
        print(f"  Crack {crack_id} grew: {old_len*1e6:.2f} -> {crack['length']*1e6:.2f} um")
    
    def compute_stress_field(self, xmin, xmax, nx, ymin, ymax, ny):
        x = np.linspace(xmin, xmax, nx)
        y = np.linspace(ymin, ymax, ny)
        X, Y = np.meshgrid(x, y)
        
        sxx = np.zeros_like(X)
        syy = np.zeros_like(X)
        sxy = np.zeros_like(X)
        
        for i in range(ny):
            for j in range(nx):
                point = np.array([X[i, j], Y[i, j]])
                
                # Check if too close to dislocation
                too_close = False
                for disl in self.dislocations:
                    if np.linalg.norm(point - disl['pos']) < 2*self.b:
                        too_close = True
                        break
                
                if not too_close:
                    stress = self.total_stress_at(point)
                    sxx[i, j] = stress['sxx']
                    syy[i, j] = stress['syy']
                    sxy[i, j] = stress['sxy']
        
        return X, Y, sxx, syy, sxy
    
    def plot_stress_field(self, stress_component='sigma_yy', filename='stress_field.png'):
        # Compute stress field
        domain = 25e-6
        X, Y, sxx, syy, sxy = self.compute_stress_field(
            -domain, domain, 100, -domain, domain, 100
        )
        
        # Select component
        if stress_component == 'sigma_xx':
            Z = sxx / 1e6
            label = r'$\sigma_{xx}$ (MPa)'
        elif stress_component == 'sigma_yy':
            Z = syy / 1e6
            label = r'$\sigma_{yy}$ (MPa)'
        else:
            Z = sxy / 1e6
            label = r'$\sigma_{xy}$ (MPa)'
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        vmax = np.percentile(np.abs(Z), 98)
        vmin = -vmax
        levels = np.linspace(vmin, vmax, 40)
        
        contourf = ax.contourf(X*1e6, Y*1e6, Z, levels=levels, cmap='RdBu_r', extend='both')
        contours = ax.contour(X*1e6, Y*1e6, Z, levels=12, colors='black', 
                             linewidths=0.5, alpha=0.4)
        ax.clabel(contours, inline=True, fontsize=8, fmt='%1.0f')
        
        cbar = plt.colorbar(contourf, ax=ax, label=label, pad=0.02, shrink=0.9)
        
        # Plot cracks
        for i, crack in enumerate(self.cracks):
            cx, cy = crack['center']
            length = crack['length']
            angle = crack['angle']
            
            dx = length/2 * np.cos(angle)
            dy = length/2 * np.sin(angle)
            
            x1, y1 = (cx - dx)*1e6, (cy - dy)*1e6
            x2, y2 = (cx + dx)*1e6, (cy + dy)*1e6
            
            ax.plot([x1, x2], [y1, y2], 'k-', linewidth=5, zorder=100)
            ax.plot([x1, x2], [y1, y2], 'w-', linewidth=2.5, zorder=101)
            ax.text(cx*1e6, cy*1e6, f'{i}', fontsize=10, ha='center',
                   color='black', weight='bold', zorder=102,
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        ax.set_xlabel('X (um)', fontsize=13, weight='bold')
        ax.set_ylabel('Y (um)', fontsize=13, weight='bold')
        ax.set_title(label, fontsize=15, weight='bold', pad=15)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.25, linestyle='--')
        ax.axhline(0, color='k', linewidth=0.5, alpha=0.3)
        ax.axvline(0, color='k', linewidth=0.5, alpha=0.3)
        
        plt.tight_layout()
        
        # Save to output directory
        full_path = os.path.join(outpath, filename)
        plt.savefig(full_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {full_path}")
        plt.close()

# Main simulation
if __name__ == '__main__':
    print("="*70)
    print("Crack Growth Simulation - Python Version")
    print(f"Output directory: {outpath}")
    print("="*70)
    
    sim = CrackSimulation()
    
    # Initialize
    sim.initialize_cracks(5, 2e-6, 5e-6, 40e-6, seed=12345)
    sim.create_dipoles(8)
    
    print(f"\nInitialized {len(sim.cracks)} cracks with {len(sim.dislocations)} dislocations\n")
    
    # Run simulation with incremental loading
    stress_levels = [50e6, 100e6, 150e6, 200e6, 250e6, 300e6]
    
    for step, sigma in enumerate(stress_levels):
        print(f"\n=== Step {step}: Applied Stress = {sigma/1e6:.0f} MPa ===")
        
        sim.set_external_stress(sigma)
        sim.solve_equilibrium()
        
        # Check propagation
        n_prop = 0
        for i in range(len(sim.cracks)):
            if sim.check_propagation(i):
                sim.propagate_crack(i, 0.5e-6)
                n_prop += 1
        
        print(f"  Propagations: {n_prop}")
        
        # Generate plots for selected steps
        if step in [0, 2, 4, 5]:
            sim.plot_stress_field('sigma_xx', f'sigma_xx_step{step}.png')
            sim.plot_stress_field('sigma_yy', f'sigma_yy_step{step}.png')
    
    print("\n" + "="*70)
    print(f"Simulation Complete! PNG files saved to: {outpath}")
    print("="*70)