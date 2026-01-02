"""
DCE Plotting Module - Version 1
Visualization for DCE element-based mixed-mode formulation
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class DCEPlotterV1:
    """Plotting utilities for DCE element-based results"""
    
    def __init__(self, calculator, results):
        self.calc = calculator
        self.results = results
        
    def calculate_stress_field(self, x_range, y_range, grid_size=200):
        """
        Calculate total stress field (Mode-I + Mode-II superposition)
        
        Returns: X, Y, sigma_xx, sigma_yy, sigma_xy in GLOBAL coordinates
        """
        X, Y = np.meshgrid(
            np.linspace(x_range[0], x_range[1], grid_size),
            np.linspace(y_range[0], y_range[1], grid_size)
        )
        
        sigma_xx = np.zeros_like(X)
        sigma_yy = np.zeros_like(X)
        sigma_xy = np.zeros_like(X)
        
        sigma_app_local = self.calc.get_applied_stress_local()
        
        print(f"Calculating stress field on {grid_size}x{grid_size} grid...")
        
        for i in range(grid_size):
            if i % 20 == 0:
                print(f"  Progress: {i}/{grid_size}")
            
            for j in range(grid_size):
                point_global = np.array([X[i, j], Y[i, j]])
                point_local = self.calc.global_to_local(point_global)
                
                x1, x2 = point_local
                
                # Skip points on crack face
                if abs(x2) < 1e-10 and abs(x1) < self.calc.a:
                    sigma_xx[i, j] = np.nan
                    sigma_yy[i, j] = np.nan
                    sigma_xy[i, j] = np.nan
                    continue
                
                # Start with applied stress
                sigma_local = sigma_app_local.copy()
                
                # Add contribution from all dislocations in all elements
                for elem in self.results['dce_elements']:
                    for k in range(elem.n_dislocations):
                        x1_k = elem.positions_x1[k]
                        pos_k = np.array([x1_k, 0.0])
                        b_k = elem.get_burgers_vector_local(k)
                        
                        sigma_local += self.calc.stress_from_edge_dislocation(
                            point_local, b_k, pos_k
                        )
                
                # Transform to global coordinates
                sigma_global = self.calc.stress_local_to_global(sigma_local)
                
                sigma_xx[i, j] = sigma_global[0, 0]
                sigma_yy[i, j] = sigma_global[1, 1]
                sigma_xy[i, j] = sigma_global[0, 1]
        
        print("  Done!")
        return X, Y, sigma_xx, sigma_yy, sigma_xy
    
    def calculate_displacement_field(self):
        """
        Calculate crack displacement profile
        
        Now that solver gives correct signs, just accumulate directly
        """
        dce_elements = self.results['dce_elements']
        
        # Separate elements
        negative_elem = None
        positive_elem = None
        
        for elem in dce_elements:
            if elem.polarity.value < 0:
                negative_elem = elem
            else:
                positive_elem = elem
        
        # Build arrays
        positions_local = []
        b_I_values = []
        b_II_values = []
        
        # Left tip
        positions_local.append(-self.calc.a)
        b_I_values.append(0.0)
        b_II_values.append(0.0)
        
        # NEGATIVE element (left) - use directly
        for k in range(negative_elem.n_dislocations):
            x1 = negative_elem.positions_x1[k]
            b_vec = negative_elem.get_burgers_vector_local(k)
            
            positions_local.append(x1)
            b_I_values.append(b_vec[1])   # Should be positive
            b_II_values.append(b_vec[0])
        
        # POSITIVE element (right) - use directly
        for k in range(positive_elem.n_dislocations):
            x1 = positive_elem.positions_x1[k]
            b_vec = positive_elem.get_burgers_vector_local(k)
            
            positions_local.append(x1)
            b_I_values.append(b_vec[1])   # Should be negative
            b_II_values.append(b_vec[0])
        
        # Right tip
        positions_local.append(self.calc.a)
        b_I_values.append(0.0)
        b_II_values.append(0.0)
        
        positions_local = np.array(positions_local)
        b_I_values = np.array(b_I_values)
        b_II_values = np.array(b_II_values)
        
        print(f"\nDEBUG: Displacement calculation")
        print(f"  Left element b range: {negative_elem.b_mode_I.min()*1e9:.1f} to {negative_elem.b_mode_I.max()*1e9:.1f} nm")
        print(f"  Right element b range: {positive_elem.b_mode_I.min()*1e9:.1f} to {positive_elem.b_mode_I.max()*1e9:.1f} nm")
        
        # Cumulative sum
        cod_stepped = np.cumsum(b_I_values)
        csd_stepped = np.cumsum(b_II_values)
        
        print(f"  COD at center index: {cod_stepped[len(negative_elem.positions_x1)]*1e6:.2f} μm")
        print(f"  COD at right tip: {cod_stepped[-1]*1e9:.2f} nm (should be ~0)")
        
        # Smooth
        x1_smooth = []
        cod_smooth = []
        csd_smooth = []
        
        for i in range(len(positions_local) - 1):
            x1_1 = positions_local[i]
            x1_2 = positions_local[i + 1]
            cod1 = cod_stepped[i]
            cod2 = cod_stepped[i + 1]
            csd1 = csd_stepped[i]
            csd2 = csd_stepped[i + 1]
            
            x1_smooth.append(x1_1)
            cod_smooth.append(cod1)
            csd_smooth.append(csd1)
            
            x1_mid = (x1_1 + x1_2) / 2
            cod_mid = (cod1 + cod2) / 2
            csd_mid = (csd1 + csd2) / 2
            x1_smooth.append(x1_mid)
            cod_smooth.append(cod_mid)
            csd_smooth.append(csd_mid)
        
        x1_smooth.append(positions_local[-1])
        cod_smooth.append(cod_stepped[-1])
        csd_smooth.append(csd_stepped[-1])
        
        x1_smooth = np.array(x1_smooth)
        cod_smooth = np.array(cod_smooth)
        csd_smooth = np.array(csd_smooth)
        
        # Force tips to zero
        cod_smooth[0] = 0.0
        cod_smooth[-1] = 0.0
        csd_smooth[0] = 0.0
        csd_smooth[-1] = 0.0
        
        if np.any(cod_smooth < -1e-12):
            print(f"\n⚠ WARNING: Negative COD detected, clipping to zero")
            cod_smooth = np.maximum(cod_smooth, 0.0)
        
        # Transform to global
        positions_global = np.array([
            self.calc.local_to_global(np.array([x1, 0.0])) for x1 in x1_smooth
        ])
        
        return positions_global, cod_smooth, csd_smooth
    
    def plot_stress_fields(self, x_range=None, y_range=None, grid_size=150, 
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
        
        X, Y, sigma_xx, sigma_yy, sigma_xy = self.calculate_stress_field(
            x_range, y_range, grid_size
        )
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Get crack shape
        positions_global, cod, csd = self.calculate_displacement_field()
        
        # Scale for visualization
        max_disp = max(np.max(np.abs(cod)), np.max(np.abs(csd)))
        if max_disp > 0:
            scale_factor = 0.1 * (2 * self.calc.a) / max_disp
        else:
            scale_factor = 1.0
        
        cod_scaled = cod * scale_factor
        csd_scaled = csd * scale_factor
        
        # Create crack faces
        crack_normal = self.calc.crack_normal_global
        crack_direction = self.calc.crack_direction_global
        
        # Total displacement = Mode-I (perpendicular) + Mode-II (parallel)
        upper_face = (positions_global + 
                     (cod_scaled/2)[:, np.newaxis] * crack_normal +
                     (csd_scaled/2)[:, np.newaxis] * crack_direction)
        lower_face = (positions_global - 
                     (cod_scaled/2)[:, np.newaxis] * crack_normal -
                     (csd_scaled/2)[:, np.newaxis] * crack_direction)
        
        # Colormap limits
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
            if idx == 2:  # Shear
                vmin_local = -0.5 * max_stress
                vmax_local = 0.5 * max_stress
            else:
                vmin_local, vmax_local = vmin, vmax
            
            # Contour plot
            contour = ax.contourf(X*1e3, Y*1e3, field/1e6, levels=30, 
                                 cmap='RdBu_r', vmin=vmin_local/1e6, 
                                 vmax=vmax_local/1e6, extend='both')
            
            # Draw crack
            ax.plot(upper_face[:, 0]*1e3, upper_face[:, 1]*1e3, 
                   'b-', linewidth=2.5, zorder=5)
            ax.plot(lower_face[:, 0]*1e3, lower_face[:, 1]*1e3, 
                   'r-', linewidth=2.5, zorder=5)
            
            # Close at tips
            ax.plot([upper_face[0, 0]*1e3, lower_face[0, 0]*1e3],
                   [upper_face[0, 1]*1e3, lower_face[0, 1]*1e3],
                   'k-', linewidth=2.5, zorder=5)
            ax.plot([upper_face[-1, 0]*1e3, lower_face[-1, 0]*1e3],
                   [upper_face[-1, 1]*1e3, lower_face[-1, 1]*1e3],
                   'k-', linewidth=2.5, zorder=5)
            
            # Fill crack
            crack_x = np.concatenate([upper_face[:, 0], lower_face[::-1, 0]]) * 1e3
            crack_y = np.concatenate([upper_face[:, 1], lower_face[::-1, 1]]) * 1e3
            ax.fill(crack_x, crack_y, color='white', zorder=4, edgecolor='none')
            
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
    
    def plot_crack_displacement(self, scale_factor=None, output_path=None):
        """
        Plot crack shape showing both Mode-I (opening) and Mode-II (sliding)
        """
        print("Calculating crack displacement profile...")
        
        positions_global, cod, csd = self.calculate_displacement_field()
        
        # Get coordinate system
        crack_normal = self.calc.crack_normal_global
        crack_direction = self.calc.crack_direction_global
        center = np.array(self.calc.crack.center)
        
        # SCALING
        max_disp = max(np.max(np.abs(cod)), np.max(np.abs(csd)))
        crack_length = 2 * self.calc.a
        
        if max_disp > 1e-15:
            if scale_factor is None:
                scale_factor = 0.2 * crack_length / max_disp
            cod_scaled = cod * scale_factor
            csd_scaled = csd * scale_factor
        else:
            print("  Max displacement ≈ 0: Crack is closed")
            scale_factor = 1.0
            cod_scaled = cod
            csd_scaled = csd
        
        print(f"  Max COD (Mode-I): {np.max(np.abs(cod))*1e6:.2f} μm")
        print(f"  Max CSD (Mode-II): {np.max(np.abs(csd))*1e6:.2f} μm")
        print(f"  Scale factor: {scale_factor:.1f}x")
        
        # Create crack faces with BOTH opening and sliding
        upper_face = (positions_global + 
                     (cod_scaled/2)[:, np.newaxis] * crack_normal +
                     (csd_scaled/2)[:, np.newaxis] * crack_direction)
        lower_face = (positions_global - 
                     (cod_scaled/2)[:, np.newaxis] * crack_normal -
                     (csd_scaled/2)[:, np.newaxis] * crack_direction)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Undeformed crack
        tip_left_local = np.array([-self.calc.a, 0.0])
        tip_right_local = np.array([self.calc.a, 0.0])
        tip_left_global = self.calc.local_to_global(tip_left_local)
        tip_right_global = self.calc.local_to_global(tip_right_local)
        
        ax.plot([tip_left_global[0]*1e3, tip_right_global[0]*1e3],
               [tip_left_global[1]*1e3, tip_right_global[1]*1e3],
               'k--', linewidth=2, alpha=0.5, label='Undeformed crack', zorder=1)
        
        # Mark DCE element boundary (center)
        center_global = self.calc.local_to_global(np.array([0.0, 0.0]))
        ax.axvline(center_global[0]*1e3, color='gray', linestyle=':', 
                  linewidth=1.5, alpha=0.6, label='Element boundary', zorder=1)
        
        # Fill crack
        crack_x = np.concatenate([upper_face[:, 0], lower_face[::-1, 0]]) * 1e3
        crack_y = np.concatenate([upper_face[:, 1], lower_face[::-1, 1]]) * 1e3
        ax.fill(crack_x, crack_y, color='lightblue', alpha=0.4, zorder=2)
        
        # Plot faces
        ax.plot(upper_face[:, 0]*1e3, upper_face[:, 1]*1e3, 
               'b-', linewidth=3, label='Upper crack face', zorder=3)
        ax.plot(lower_face[:, 0]*1e3, lower_face[:, 1]*1e3, 
               'r--', linewidth=3, label='Lower crack face', zorder=3)
        
        # Close at tips
        ax.plot([upper_face[0, 0]*1e3, lower_face[0, 0]*1e3],
               [upper_face[0, 1]*1e3, lower_face[0, 1]*1e3],
               'k-', linewidth=3, zorder=5)
        ax.plot([upper_face[-1, 0]*1e3, lower_face[-1, 0]*1e3],
               [upper_face[-1, 1]*1e3, lower_face[-1, 1]*1e3],
               'k-', linewidth=3, zorder=5)
        
        # Mark tips
        ax.scatter([upper_face[0, 0]*1e3, upper_face[-1, 0]*1e3],
                  [upper_face[0, 1]*1e3, upper_face[-1, 1]*1e3],
                  c='gold', s=200, edgecolors='orange', linewidths=2.5, 
                  marker='o', zorder=10, label='Tips')
        
        # Mark dislocation positions
        for elem_idx, elem in enumerate(self.results['dce_elements']):
            positions_elem = []
            for k in range(elem.n_dislocations):
                x1 = elem.positions_x1[k]
                pos_local = np.array([x1, 0.0])
                pos_global = self.calc.local_to_global(pos_local)
                positions_elem.append(pos_global)
            
            positions_elem = np.array(positions_elem)
            
            color = 'blue' if elem.polarity.value > 0 else 'red'
            label = f'{elem.polarity.name} element' if elem_idx < 2 else None
            
            ax.scatter(positions_elem[:, 0]*1e3, positions_elem[:, 1]*1e3,
                      c=color, s=30, alpha=0.6, marker='x', zorder=6, label=label)
        
        # Labels
        ax.set_xlabel('x [mm]', fontsize=12, weight='bold')
        ax.set_ylabel('y [mm]', fontsize=12, weight='bold')
        
        title_text = f'Mixed-Mode Crack Displacement (scaled {scale_factor:.0f}×)\n'
        title_text += f'θ={np.rad2deg(self.calc.crack.angle):.1f}°, '
        title_text += f'K_I={self.results["K_I"]/1e6:.2f} MPa√m, '
        title_text += f'K_II={self.results["K_II"]/1e6:.2f} MPa√m'
        ax.set_title(title_text, fontsize=13, weight='bold')
        
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Annotations
        max_cod_idx = np.argmax(np.abs(cod))
        max_csd_idx = np.argmax(np.abs(csd))
        
        if np.max(np.abs(cod)) > 1e-15:
            pos_cod = positions_global[max_cod_idx]
            annotation_pos = pos_cod + (cod_scaled[max_cod_idx]/2 * 1.5) * crack_normal
            
            ax.annotate(f'Max COD (Mode-I)\n{np.max(np.abs(cod))*1e6:.1f} μm',
                       xy=(annotation_pos[0]*1e3, annotation_pos[1]*1e3),
                       xytext=(15, 15), textcoords='offset points',
                       fontsize=9, bbox=dict(boxstyle='round,pad=0.5', 
                       facecolor='lightblue', alpha=0.9),
                       arrowprops=dict(arrowstyle='->', lw=1.5))
        
        if np.max(np.abs(csd)) > 1e-15:
            pos_csd = positions_global[max_csd_idx]
            annotation_pos = pos_csd + (csd_scaled[max_csd_idx]/2 * 1.5) * crack_direction
            
            ax.annotate(f'Max CSD (Mode-II)\n{np.max(np.abs(csd))*1e6:.1f} μm',
                       xy=(annotation_pos[0]*1e3, annotation_pos[1]*1e3),
                       xytext=(15, -15), textcoords='offset points',
                       fontsize=9, bbox=dict(boxstyle='round,pad=0.5', 
                       facecolor='lightcoral', alpha=0.9),
                       arrowprops=dict(arrowstyle='->', lw=1.5))
        
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✓ Saved to {output_path}")
        
        return fig
    
    def plot_element_structure(self, output_path=None):
        """
        Visualize DCE element structure showing dislocations
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Draw crack line
        tip_left = self.calc.local_to_global(np.array([-self.calc.a, 0.0]))
        tip_right = self.calc.local_to_global(np.array([self.calc.a, 0.0]))
        center = self.calc.local_to_global(np.array([0.0, 0.0]))
        
        ax.plot([tip_left[0]*1e3, tip_right[0]*1e3],
               [tip_left[1]*1e3, tip_right[1]*1e3],
               'k-', linewidth=3, label='Crack line', zorder=1)
        
        # Mark element boundary
        ax.axvline(center[0]*1e3, color='purple', linestyle='--', 
                  linewidth=2, alpha=0.7, label='Element boundary', zorder=2)
        
        # Plot each element
        colors = ['red', 'blue']
        for elem_idx, elem in enumerate(self.results['dce_elements']):
            color = colors[elem_idx]
            
            # Get all dislocation positions
            positions_global = []
            for k in range(elem.n_dislocations):
                x1 = elem.positions_x1[k]
                pos_local = np.array([x1, 0.0])
                pos_global = self.calc.local_to_global(pos_local)
                positions_global.append(pos_global)
            
            positions_global = np.array(positions_global)
            
            # Plot dislocations
            ax.scatter(positions_global[:, 0]*1e3, positions_global[:, 1]*1e3,
                      s=100, c=color, marker='o', alpha=0.7, edgecolors='black',
                      linewidths=1.5, zorder=5,
                      label=f'{elem.polarity.name} element ({elem.n_dislocations} dislocations)')
            
            # Draw arrows showing element extent
            tip_pos = self.calc.local_to_global(np.array([elem.tip_position, 0.0]))
            base_pos = self.calc.local_to_global(np.array([elem.base_position, 0.0]))
            
            offset_y = 0.5e-3 * (1 if elem_idx == 0 else -1)  # Offset for clarity
            
            ax.annotate('', xy=(base_pos[0]*1e3, (base_pos[1] + offset_y)*1e3),
                       xytext=(tip_pos[0]*1e3, (tip_pos[1] + offset_y)*1e3),
                       arrowprops=dict(arrowstyle='<->', lw=2, color=color, alpha=0.5))
        
        # Mark tips
        ax.scatter([tip_left[0]*1e3, tip_right[0]*1e3],
                  [tip_left[1]*1e3, tip_right[1]*1e3],
                  c='gold', s=300, marker='*', edgecolors='orange',
                  linewidths=2, zorder=10, label='Crack tips')
        
        ax.set_xlabel('x [mm]', fontsize=12, weight='bold')
        ax.set_ylabel('y [mm]', fontsize=12, weight='bold')
        ax.set_title('DCE Element Structure\n(Two elements forming symmetric crack)', 
                    fontsize=13, weight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✓ Saved to {output_path}")
        
        return fig
    
    def plot_summary(self, output_path=None):
        """Summary with mixed-mode results"""
        fig = plt.figure(figsize=(10, 6))
        
        sigma_local = self.calc.get_applied_stress_local()
        
        # Count dislocations
        total_dislocations = sum(elem.n_dislocations for elem in self.results['dce_elements'])
        
        summary_text = f"""
DCE ANALYSIS SUMMARY - VERSION 1
{'='*60}
DCE Element Formulation: Building block approach
- Each element = half crack with N dislocations (Mode-I + Mode-II)
- Symmetric crack = 2 elements (NEGATIVE + POSITIVE)

GEOMETRY:
  Crack length (2a):     {self.calc.crack.length*1e3:.2f} mm
  Crack angle:           {np.rad2deg(self.calc.crack.angle):.1f}°
  Number of elements:    {len(self.results['dce_elements'])}
  Dislocations per mode: {total_dislocations}
  
DCE ELEMENTS:
"""
        
        for i, elem in enumerate(self.results['dce_elements']):
            summary_text += f"  Element {i}: {elem.polarity.name:8s} "
            summary_text += f"(tip={elem.tip_position*1e3:5.2f} mm, "
            summary_text += f"n={elem.n_dislocations})\n"
        
        summary_text += f"""
MATERIAL:
  Shear modulus (μ):     {self.calc.material.mu/1e9:.1f} GPa
  Poisson's ratio (ν):   {self.calc.material.nu:.3f}

APPLIED STRESS (global):
  σ_xx:  {self.calc.stress.sigma_xx/1e6:7.1f} MPa
  σ_yy:  {self.calc.stress.sigma_yy/1e6:7.1f} MPa
  σ_xy:  {self.calc.stress.sigma_xy/1e6:7.1f} MPa

APPLIED STRESS (local crack coords):
  σ_11:  {sigma_local[0,0]/1e6:7.1f} MPa (parallel to crack)
  σ_22:  {sigma_local[1,1]/1e6:7.1f} MPa (normal to crack)
  σ_12:  {sigma_local[0,1]/1e6:7.1f} MPa (shear)

RESULTS - MODE-I:
  K_I (DCE):             {self.results['K_I']/1e6:.3f} MPa√m
  K_I (Analytical):      {self.results['K_I_analytical']/1e6:.3f} MPa√m
  Error:                 {self.results['error_I_percent']:.2f}%

RESULTS - MODE-II:
  K_II (DCE):            {self.results['K_II']/1e6:.3f} MPa√m
  K_II (Analytical):     {self.results['K_II_analytical']/1e6:.3f} MPa√m
  Error:                 {self.results['error_II_percent']:.2f}%

MIXED-MODE:
  K_eff = √(K_I² + K_II²) = {np.sqrt(self.results['K_I']**2 + self.results['K_II']**2)/1e6:.3f} MPa√m
  Mode mixity = atan(K_II/K_I) = {np.rad2deg(np.arctan2(self.results['K_II'], self.results['K_I'])):.1f}°

{'='*60}
        """
        
        plt.text(0.05, 0.5, summary_text, fontsize=9.5, family='monospace',
                verticalalignment='center', transform=fig.transFigure)
        plt.axis('off')
        
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✓ Saved to {output_path}")
        
        return fig