"""
CORRECTED dce_plotting.py
Modified COD and stress field visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class DCEPlotter:
    """Plotting utilities for DCE results"""
    
    def __init__(self, calculator, results):
        self.calc = calculator
        self.results = results
        
    def plot_stress_fields(self, x_range=None, y_range=None, grid_size=200, 
                          output_path=None):
        """Plot stress field contours with open crack visualization"""
        if x_range is None:
            extent = 2 * self.calc.a
            x_range = (self.calc.crack.center[0] - extent, 
                      self.calc.crack.center[0] + extent)
        if y_range is None:
            extent = 1.5 * self.calc.a
            y_range = (self.calc.crack.center[1] - extent, 
                      self.calc.crack.center[1] + extent)
        
        print(f"Calculating stress fields on {grid_size}x{grid_size} grid...")
        X, Y, sigma_xx, sigma_yy, sigma_xy = self.calc.calculate_stress_field(
            self.results['dipole_positions'],
            self.results['b_magnitudes'],
            x_range, y_range, grid_size
        )
        print("  Done!")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Get COD for crack visualization
        positions_global, cod = self.calc.calculate_cod_profile(
            self.results['dipole_positions'],
            self.results['b_magnitudes'],
            n_points=100
        )
        
        # Scale COD for visualization (make it visible)
        # Adjust scale factor based on crack length
        cod_scale = self.calc.a * 0.1 / np.max(np.abs(cod)) if np.max(np.abs(cod)) > 0 else 1.0
        cod_scaled = cod * cod_scale
        
        # Create crack faces (upper and lower) in global coordinates
        crack_direction = self.calc.R[0, :]  # Unit vector along crack
        crack_normal = self.calc.R[1, :]     # Unit vector perpendicular to crack
        
        # Upper and lower crack faces
        upper_face = positions_global + cod_scaled[:, np.newaxis] * crack_normal
        lower_face = positions_global - cod_scaled[:, np.newaxis] * crack_normal
        
        # Determine colormap limits
        max_stress = max(abs(self.calc.stress.sigma_xx), 
                        abs(self.calc.stress.sigma_yy),
                        abs(self.calc.stress.sigma_xy))
        if max_stress < 1e3:
            max_stress = 100e6
        
        vmin = -1.5 * max_stress
        vmax = 2.5 * max_stress
        
        titles = ['$\\sigma_{xx}$', '$\\sigma_{yy}$', '$\\sigma_{xy}$']
        fields = [sigma_xx, sigma_yy, sigma_xy]
        
        for idx, (ax, title, field) in enumerate(zip(axes, titles, fields)):
            if idx == 2:  # Shear stress
                vmin_local = -0.5 * max_stress
                vmax_local = 0.5 * max_stress
            else:
                vmin_local, vmax_local = vmin, vmax
            
            # Contour plot
            contour = ax.contourf(X*1e3, Y*1e3, field/1e6, levels=30, 
                                 cmap='RdBu_r', vmin=vmin_local/1e6, 
                                 vmax=vmax_local/1e6, extend='both')
            
            # Draw open crack (upper and lower faces)
            ax.plot(upper_face[:, 0]*1e3, upper_face[:, 1]*1e3, 
                   'k-', linewidth=2.5, zorder=5)
            ax.plot(lower_face[:, 0]*1e3, lower_face[:, 1]*1e3, 
                   'k-', linewidth=2.5, zorder=5)
            
            # Close crack at tips
            ax.plot([upper_face[0, 0]*1e3, lower_face[0, 0]*1e3],
                   [upper_face[0, 1]*1e3, lower_face[0, 1]*1e3],
                   'k-', linewidth=2.5, zorder=5)
            ax.plot([upper_face[-1, 0]*1e3, lower_face[-1, 0]*1e3],
                   [upper_face[-1, 1]*1e3, lower_face[-1, 1]*1e3],
                   'k-', linewidth=2.5, zorder=5)
            
            # Fill crack interior (optional - makes it more visible)
            crack_x = np.concatenate([upper_face[:, 0], lower_face[::-1, 0]])
            crack_y = np.concatenate([upper_face[:, 1], lower_face[::-1, 1]])
            ax.fill(crack_x*1e3, crack_y*1e3, color='white', zorder=4, edgecolor='none')
            
            ax.set_xlabel('$x$ [mm]', fontsize=12, weight='bold')
            ax.set_ylabel('$y$ [mm]', fontsize=12, weight='bold')
            ax.set_title(f'{title} [MPa]', fontsize=13, weight='bold')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            
            cbar = plt.colorbar(contour, ax=ax)
            cbar.set_label('MPa', fontsize=11)
        
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✓ Saved to {output_path}")
        
        return fig
    
    def plot_cod(self, scale_factor=None, output_path=None):
        """
        Plot crack opening as actual crack shape with scaled COD
        CORRECTED: Calculates proper normal direction
        """
        print("Calculating COD for crack shape visualization...")
        
        # Get numerical COD
        positions_global, cod_numerical = self.calc.calculate_cod_profile(
            self.results['dipole_positions'],
            self.results['b_magnitudes']
        )
        
        # Calculate crack coordinate system directly
        c = np.cos(self.calc.crack.angle)
        s = np.sin(self.calc.crack.angle)
        
        # Direction along crack and perpendicular to it in GLOBAL coordinates
        crack_direction = np.array([c, s])
        crack_normal = np.array([-s, c])
        center = np.array(self.calc.crack.center)
        
        # DEBUG: Verify perpendicularity
        print(f"\n  Crack angle: {np.rad2deg(self.calc.crack.angle):.1f}°")
        print(f"  Crack direction (global): [{crack_direction[0]:.3f}, {crack_direction[1]:.3f}]")
        print(f"  Crack normal (global):    [{crack_normal[0]:.3f}, {crack_normal[1]:.3f}]")
        print(f"  Dot product: {np.dot(crack_direction, crack_normal):.6e} (should be ~0)")
        
        # Check stress
        sigma_app_local = self.calc.get_applied_stress_local()
        sigma_yy_local = sigma_app_local[1, 1]
        
        if sigma_yy_local < 0:
            print(f"  Normal stress is COMPRESSIVE: σ_n = {sigma_yy_local/1e6:.1f} MPa")
        
        # SCALING
        max_cod = np.max(np.abs(cod_numerical))
        crack_length = 2 * self.calc.a
        
        if max_cod > 1e-15:
            if scale_factor is None:
                scale_factor = 0.2 * crack_length / max_cod
            cod_scaled = cod_numerical * scale_factor
        else:
            print(f"  Max COD ≈ 0: Crack is closed")
            scale_factor = 1.0
            cod_scaled = cod_numerical
        
        print(f"  Max COD (actual): {max_cod*1e6:.2f} μm")
        print(f"  Scale factor: {scale_factor:.1f}x")
        print(f"  Max COD (scaled): {np.max(cod_scaled)*1e3:.2f} mm")
        
        # Create upper and lower faces - PERPENDICULAR TO CRACK
        upper_face = positions_global + (cod_scaled/2)[:, np.newaxis] * crack_normal
        lower_face = positions_global - (cod_scaled/2)[:, np.newaxis] * crack_normal
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Draw undeformed crack FIRST (background)
        tip_left_local = np.array([-self.calc.a, 0.0])
        tip_right_local = np.array([self.calc.a, 0.0])
        tip_left_global = self.calc.local_to_global(tip_left_local)
        tip_right_global = self.calc.local_to_global(tip_right_local)
        
        ax.plot([tip_left_global[0]*1e3, tip_right_global[0]*1e3],
            [tip_left_global[1]*1e3, tip_right_global[1]*1e3],
            'k--', linewidth=2, alpha=0.5, label='Undeformed crack', zorder=1)
        
        # Fill crack interior
        crack_x = np.concatenate([upper_face[:, 0], lower_face[::-1, 0]]) * 1e3
        crack_y = np.concatenate([upper_face[:, 1], lower_face[::-1, 1]]) * 1e3
        ax.fill(crack_x, crack_y, color='lightblue', alpha=0.4, edgecolor='none', zorder=2)
        
        # Plot crack faces with different styles
        ax.plot(upper_face[:, 0]*1e3, upper_face[:, 1]*1e3, 
            'b-', linewidth=3, label='Upper crack face (scaled)', zorder=3)
        ax.plot(lower_face[:, 0]*1e3, lower_face[:, 1]*1e3, 
            'r--', linewidth=3, label='Lower crack face (scaled)', zorder=3)
        
        # Close crack at tips
        ax.plot([upper_face[0, 0]*1e3, lower_face[0, 0]*1e3],
            [upper_face[0, 1]*1e3, lower_face[0, 1]*1e3],
            'k-', linewidth=3, zorder=5)
        ax.plot([upper_face[-1, 0]*1e3, lower_face[-1, 0]*1e3],
            [upper_face[-1, 1]*1e3, lower_face[-1, 1]*1e3],
            'k-', linewidth=3, zorder=5)
        
        # Mark crack tips
        ax.scatter([upper_face[0, 0]*1e3, upper_face[-1, 0]*1e3],
                [upper_face[0, 1]*1e3, upper_face[-1, 1]*1e3],
                c='gold', s=200, edgecolors='orange', linewidths=2.5, 
                marker='o', zorder=10, label='Tips')
        
        # Labels
        ax.set_xlabel('x [mm]', fontsize=12, weight='bold')
        ax.set_ylabel('y [mm]', fontsize=12, weight='bold')
        
        if max_cod > 1e-15:
            title_text = f'Crack Opening Shape (COD scaled {scale_factor:.0f}× for visibility)\n'
        else:
            title_text = f'Crack Shape (CLOSED - zero opening)\n'
        
        title_text += f'θ={np.rad2deg(self.calc.crack.angle):.1f}°, '
        title_text += f'σ_n={sigma_yy_local/1e6:.1f} MPa, '
        title_text += f'K_I={self.results["K_I"]/1e6:.2f} MPa√m'
        ax.set_title(title_text, fontsize=13, weight='bold')
        
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Annotation
        if max_cod > 1e-15:
            max_cod_idx = np.argmax(np.abs(cod_numerical))
            max_cod_pos = positions_global[max_cod_idx]
            max_cod_val = cod_numerical[max_cod_idx]
            
            annotation_pos_global = max_cod_pos + (cod_scaled[max_cod_idx]/2 * 1.5) * crack_normal
            
            angle_deg = np.rad2deg(self.calc.crack.angle)
            if 45 < angle_deg < 135 or -135 < angle_deg < -45:
                xytext = (20, 0)
            else:
                xytext = (0, 20)
            
            ax.annotate(f'Max COD = {max_cod_val*1e6:.1f} μm\n(actual, unscaled)\nScaled: {np.max(cod_scaled)*1e3:.2f} mm',
                    xy=(annotation_pos_global[0]*1e3, annotation_pos_global[1]*1e3),
                    xytext=xytext, textcoords='offset points',
                    fontsize=10, bbox=dict(boxstyle='round,pad=0.5', 
                    facecolor='yellow', alpha=0.9, edgecolor='orange', linewidth=2),
                    arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✓ Saved to {output_path}")
        
        return fig
    
    def plot_polar_stress(self, tip='right', output_path=None):
        """Plot polar stress - DEACTIVATED"""
        print(f"Polar stress plotting is currently deactivated.")
        return None
    
    def plot_summary(self, output_path=None):
        """Summary plot with K_I and key results"""
        fig = plt.figure(figsize=(10, 6))
        
        # Text summary
        summary_text = f"""
DCE ANALYSIS SUMMARY
{'='*50}

GEOMETRY:
  Crack length (2a):  {self.calc.crack.length*1e3:.2f} mm
  Crack angle:        {np.rad2deg(self.calc.crack.angle):.1f}°
  Crack center:       ({self.calc.crack.center[0]*1e3:.2f}, {self.calc.crack.center[1]*1e3:.2f}) mm
  Number of dipoles:  {len(self.results['dipole_positions'])}

MATERIAL:
  Shear modulus (μ):  {self.calc.material.mu/1e9:.1f} GPa
  Poisson's ratio:    {self.calc.material.nu:.3f}

APPLIED STRESS (global):
  σ_xx:  {self.calc.stress.sigma_xx/1e6:7.1f} MPa
  σ_yy:  {self.calc.stress.sigma_yy/1e6:7.1f} MPa
  σ_xy:  {self.calc.stress.sigma_xy/1e6:7.1f} MPa

APPLIED STRESS (local crack coords):
  σ_xx:  {self.calc.get_applied_stress_local()[0,0]/1e6:7.1f} MPa (parallel)
  σ_yy:  {self.calc.get_applied_stress_local()[1,1]/1e6:7.1f} MPa (normal)
  σ_xy:  {self.calc.get_applied_stress_local()[0,1]/1e6:7.1f} MPa

RESULTS:
  K_I (DCE):         {self.results['K_I']/1e6:.3f} MPa√m
  K_I (Analytical):  {self.results['K_I_analytical']/1e6:.3f} MPa√m
  Error:             {self.results['error_percent']:.2f}%

{'='*50}
        """
        
        plt.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                verticalalignment='center', transform=fig.transFigure)
        plt.axis('off')
        
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✓ Saved to {output_path}")
        
        return fig