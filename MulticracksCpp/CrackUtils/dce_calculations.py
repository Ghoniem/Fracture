"""
DCE Calculations Module
Computes K_I, stress fields, COD, and polar stress/energy density
"""

import numpy as np
from scipy.linalg import solve
from dataclasses import dataclass
from typing import Tuple, Dict

@dataclass
class Material:
    """Material properties"""
    mu: float  # Shear modulus (Pa)
    nu: float  # Poisson's ratio

@dataclass
class Crack:
    """Crack geometry"""
    length: float  # Total crack length 2a (m)
    angle: float   # Orientation angle from x-axis (radians)
    center: Tuple[float, float] = (0.0, 0.0)  # Crack center position

@dataclass
class AppliedStress:
    """Applied stress tensor"""
    sigma_xx: float  # Normal stress in x (Pa)
    sigma_yy: float  # Normal stress in y (Pa)
    sigma_xy: float  # Shear stress (Pa)

class DCECalculator:
    """DCE method calculations"""
    
    def __init__(self, material: Material, crack: Crack, stress: AppliedStress):
        self.material = material
        self.crack = crack
        self.stress = stress
        self.a = crack.length / 2
        
        # Rotation matrix from global to local coordinates
        c = np.cos(crack.angle)
        s = np.sin(crack.angle)
        
        # R transforms FROM local TO global
        # Local x-axis (along crack) → global direction
        # Local y-axis (perpendicular to crack) → global direction
        self.R = np.array([[c, -s], 
                        [s,  c]])
        self.R_inv = self.R.T  # Inverse rotation
        
        # CORRECTED: Extract crack direction and normal in GLOBAL coordinates
        # These are the columns of R (not rows!)
        # Column 0: where local x-axis [1,0] goes → along crack
        # Column 1: where local y-axis [0,1] goes → perpendicular to crack
        
        self.crack_direction_global = np.array([c, s])      # Along crack in global coords
        self.crack_normal_global = np.array([-s, c])        # Perpendicular to crack in global coords
        
        # Verify: these should be perpendicular
        assert abs(np.dot(self.crack_direction_global, self.crack_normal_global)) < 1e-10
        
    def stress_from_dislocation(self, r, b_vec, disl_pos):
        """Stress from edge dislocation in local crack coordinates"""
        x = r[0] - disl_pos[0]
        y = r[1] - disl_pos[1]
        r2 = x**2 + y**2
        
        if r2 < (1e-10)**2:
            return np.zeros((2, 2))
        
        D = self.material.mu / (2 * np.pi * (1 - self.material.nu))
        r4 = r2 * r2
        bx, by = b_vec[0], b_vec[1]
        
        sig = np.zeros((2, 2))
        
        if abs(bx) > 1e-20:
            sig[0, 0] += -D * bx * y * (3*x**2 + y**2) / r4
            sig[1, 1] += D * bx * y * (x**2 - y**2) / r4
            sig[0, 1] += D * bx * x * (x**2 - y**2) / r4
        
        if abs(by) > 1e-20:
            sig[0, 0] += D * by * x * (x**2 - y**2) / r4
            sig[1, 1] += -D * by * x * (x**2 + 3*y**2) / r4
            sig[0, 1] += -D * by * y * (x**2 - y**2) / r4
        
        sig[1, 0] = sig[0, 1]
        return sig
    
    def stress_from_dipole(self, r, b_magnitude, dipole_x):
        """Stress from dipole in local coordinates"""
        sigma = np.zeros((2, 2))
        sigma += self.stress_from_dislocation(
            r, np.array([0.0, b_magnitude]), np.array([-dipole_x, 0.0])
        )
        sigma += self.stress_from_dislocation(
            r, np.array([0.0, -b_magnitude]), np.array([dipole_x, 0.0])
        )
        return sigma
    
    def global_to_local(self, point_global):
        """Transform point from global to local crack coordinates"""
        point = np.array(point_global) - np.array(self.crack.center)
        return self.R_inv @ point
    
    def local_to_global(self, point_local):
        """Transform point from local to global coordinates"""
        point = self.R @ point_local
        return point + np.array(self.crack.center)
    
    def stress_global_to_local(self, sigma_global):
        """Transform stress tensor from global to local coordinates"""
        return self.R_inv @ sigma_global @ self.R_inv.T
    
    def stress_local_to_global(self, sigma_local):
        """Transform stress tensor from local to global coordinates"""
        return self.R @ sigma_local @ self.R.T
    
    def get_applied_stress_local(self):
        """Applied stress in local crack coordinates"""
        sigma_global = np.array([
            [self.stress.sigma_xx, self.stress.sigma_xy],
            [self.stress.sigma_xy, self.stress.sigma_yy]
        ])
        return self.stress_global_to_local(sigma_global)
    
    # CORRECTED dce_calculations.py
# Replace the solve_burgers_vectors method in DCECalculator class

    def solve_burgers_vectors(self, n_dipoles=30, tip_fraction=0.95):
        """
        Solve for dipole Burgers vectors
        
        Key fix: In local crack coordinates, we enforce equilibrium based on
        the stress that acts NORMAL to the crack (σ_yy in local coords)
        """
        x_tip = self.a * tip_fraction
        x_center = self.a * 0.05
        dipole_positions = np.linspace(x_tip, x_center, n_dipoles)
        
        # Applied stress in LOCAL crack coordinates
        sigma_app_local = self.get_applied_stress_local()
        sigma_yy_local = sigma_app_local[1, 1]  # Normal stress to crack
        
        print(f"\nDEBUG: Applied stress (global):")
        print(f"  σ_xx = {self.stress.sigma_xx/1e6:.2f} MPa")
        print(f"  σ_yy = {self.stress.sigma_yy/1e6:.2f} MPa")
        print(f"  σ_xy = {self.stress.sigma_xy/1e6:.2f} MPa")
        print(f"\nDEBUG: Applied stress (local crack coords):")
        print(f"  σ_xx_local = {sigma_app_local[0,0]/1e6:.2f} MPa (parallel to crack)")
        print(f"  σ_yy_local = {sigma_app_local[1,1]/1e6:.2f} MPa (normal to crack)")
        print(f"  σ_xy_local = {sigma_app_local[0,1]/1e6:.2f} MPa")
        
        if abs(sigma_yy_local) < 1e-6:
            print("\nWARNING: Normal stress to crack is near zero!")
            print("This means no Mode-I loading. Check applied stress orientation.")
        
        n = len(dipole_positions)
        A = np.zeros((n-1, n))
        rhs = np.zeros(n-1)
        
        # Equilibrium equations: σ_yy = 0 at internal dipole positions
        for i in range(1, n):
            x_i = dipole_positions[i]
            eval_pos = np.array([x_i, 0.0])
            rhs[i-1] = -sigma_yy_local
            
            for j in range(n):
                if i == j:
                    continue
                x_j = dipole_positions[j]
                sigma_ij = self.stress_from_dipole(eval_pos, 1.0, x_j)
                A[i-1, j] = sigma_ij[1, 1]
        
        # Additional equation: σ_yy = 0 at crack center
        A_full = np.zeros((n, n))
        A_full[:n-1, :] = A
        rhs_full = np.zeros(n)
        rhs_full[:n-1] = rhs
        
        eval_center = np.array([0.0, 0.0])
        rhs_full[n-1] = -sigma_yy_local
        
        for j in range(n):
            x_j = dipole_positions[j]
            sigma_j = self.stress_from_dipole(eval_center, 1.0, x_j)
            A_full[n-1, j] = sigma_j[1, 1]
        
        b_magnitudes = solve(A_full, rhs_full)
        
        print(f"\nDEBUG: Burgers vectors:")
        print(f"  b_min = {np.min(b_magnitudes)*1e9:.2f} nm")
        print(f"  b_max = {np.max(b_magnitudes)*1e9:.2f} nm")
        print(f"  Sum(b) = {np.sum(b_magnitudes)*1e9:.2f} nm")
        
        return dipole_positions, b_magnitudes


    def calculate_K_I(self, dipole_positions, b_magnitudes):
        """
        Calculate K_I from tip force
        
        Key: Use local normal stress for both DCE and analytical
        """
        sigma_app_local = self.get_applied_stress_local()
        sigma_yy_local = sigma_app_local[1, 1]
        
        tip_idx = 0
        x_tip = dipole_positions[tip_idx]
        b_tip = b_magnitudes[tip_idx]
        
        pos_tip = np.array([x_tip, 0.0])
        b_tip_vec = np.array([0.0, -b_tip])
        
        sigma_app = np.array([[0.0, 0.0], [0.0, sigma_yy_local]])
        sigma_total = sigma_app.copy()
        
        for i in range(len(dipole_positions)):
            if i == tip_idx:
                continue
            x_i = dipole_positions[i]
            b_i = b_magnitudes[i]
            sigma_total += self.stress_from_dipole(pos_tip, b_i, x_i)
        
        sigma_dot_b = sigma_total @ b_tip_vec
        f_x = sigma_dot_b[1]
        K_I = np.sqrt(2 * self.material.mu * abs(f_x) / (1 - self.material.nu))
        
        # Analytical K_I based on normal stress to crack
        K_I_analytical = abs(sigma_yy_local) * np.sqrt(np.pi * self.a)
        
        print(f"\nDEBUG: K_I calculation:")
        print(f"  σ_yy_local (normal to crack) = {sigma_yy_local/1e6:.2f} MPa")
        print(f"  Tip force f_x = {f_x:.3e} N")
        print(f"  K_I (DCE) = {K_I/1e6:.3f} MPa√m")
        print(f"  K_I (analytical) = {K_I_analytical/1e6:.3f} MPa√m")
        
        return K_I, K_I_analytical
    def calculate_stress_field(self, dipole_positions, b_magnitudes, 
                               x_range, y_range, grid_size=200):
        """
        Calculate stress field on a grid in global coordinates
        
        Returns: X, Y, sigma_xx, sigma_yy, sigma_xy (all in global coordinates)
        """
        X, Y = np.meshgrid(
            np.linspace(x_range[0], x_range[1], grid_size),
            np.linspace(y_range[0], y_range[1], grid_size)
        )
        
        sigma_xx = np.zeros_like(X)
        sigma_yy = np.zeros_like(X)
        sigma_xy = np.zeros_like(X)
        
        sigma_app_local = self.get_applied_stress_local()
        
        for i in range(grid_size):
            for j in range(grid_size):
                point_global = np.array([X[i, j], Y[i, j]])
                point_local = self.global_to_local(point_global)
                
                x_local, y_local = point_local
                
                # Skip points on crack face
                if abs(y_local) < 1e-10 and abs(x_local) < self.a:
                    sigma_xx[i, j] = np.nan
                    sigma_yy[i, j] = np.nan
                    sigma_xy[i, j] = np.nan
                    continue
                
                # Calculate stress in local coordinates
                sigma_local = sigma_app_local.copy()
                
                for k in range(len(dipole_positions)):
                    x_k = dipole_positions[k]
                    b_k = b_magnitudes[k]
                    sigma_local += self.stress_from_dipole(point_local, b_k, x_k)
                
                # Transform to global coordinates
                sigma_global = self.stress_local_to_global(sigma_local)
                
                sigma_xx[i, j] = sigma_global[0, 0]
                sigma_yy[i, j] = sigma_global[1, 1]
                sigma_xy[i, j] = sigma_global[0, 1]
        
        return X, Y, sigma_xx, sigma_yy, sigma_xy
    
    def calculate_cod_profile(self, dipole_positions, b_magnitudes, n_points=200):
        """
        Calculate SMOOTH COD profile - DEBUGGED VERSION
        """
        # Build discrete COD at dislocation positions
        positions_local_discrete = np.concatenate([
            [-self.a],
            -dipole_positions,
            dipole_positions[::-1],
            [self.a]
        ])
        
        b_values = np.concatenate([
            [0],
            b_magnitudes,
            -b_magnitudes[::-1],
            [0]
        ])
        
        # Cumulative sum
        cod_stepped = np.cumsum(b_values)
        
        # DEBUG: Check the stepped values
        print(f"\n  DEBUG COD profile:")
        print(f"    Positions (first 3): {positions_local_discrete[:3]*1e3} mm")
        print(f"    Positions (last 3): {positions_local_discrete[-3:]*1e3} mm")
        print(f"    COD (first 3): {cod_stepped[:3]*1e9} nm")
        print(f"    COD (last 3): {cod_stepped[-3:]*1e9} nm")
        
        # Create smooth COD by averaging
        x_smooth = []
        cod_smooth = []
        
        for i in range(len(positions_local_discrete) - 1):
            x1 = positions_local_discrete[i]
            x2 = positions_local_discrete[i + 1]
            cod1 = cod_stepped[i]
            cod2 = cod_stepped[i + 1]
            
            # Add left point
            x_smooth.append(x1)
            cod_smooth.append(cod1)
            
            # Add mid-point
            x_mid = (x1 + x2) / 2
            cod_mid = (cod1 + cod2) / 2
            x_smooth.append(x_mid)
            cod_smooth.append(cod_mid)
        
        # Add final point
        x_smooth.append(positions_local_discrete[-1])
        cod_smooth.append(cod_stepped[-1])
        
        x_smooth = np.array(x_smooth)
        cod_smooth = np.array(cod_smooth)
        
        print(f"    Smooth array length: {len(x_smooth)}")
        print(f"    Smooth COD (first): {cod_smooth[0]*1e9:.3f} nm")
        print(f"    Smooth COD (last): {cod_smooth[-1]*1e9:.3f} nm")
        
        # FORCE tips to exactly zero
        cod_smooth[0] = 0.0
        cod_smooth[-1] = 0.0
        
        # Guard against negative COD
        if np.any(cod_smooth < -1e-12):
            min_cod = np.min(cod_smooth)
            print(f"\n⚠ WARNING: Negative COD detected!")
            print(f"  Min COD = {min_cod*1e6:.2f} μm")
            cod_smooth = np.maximum(cod_smooth, 0.0)
        
        # Transform to global coordinates
        positions_global = np.array([
            self.local_to_global(np.array([x, 0.0])) for x in x_smooth
        ])
        
        return positions_global, cod_smooth
    
    def calculate_polar_stress(self, dipole_positions, b_magnitudes, 
                               tip='right', theta_range=(-180, 180), n_theta=360, 
                               r_range=(0.1, 3.0), n_r=100):
        """
        Calculate stress on polar lines emanating from crack tip
        
        Parameters:
        -----------
        tip: 'right' or 'left'
        theta_range: angle range in degrees (relative to crack line)
        r_range: radial distance range as fraction of crack half-length
        
        Returns: R, THETA (meshgrid), sigma_normal (normal stress on polar plane)
        """
        # Tip position in local coordinates
        if tip == 'right':
            tip_pos_local = np.array([self.a, 0.0])
        else:
            tip_pos_local = np.array([-self.a, 0.0])
        
        # Create polar grid
        theta_deg = np.linspace(theta_range[0], theta_range[1], n_theta)
        theta_rad = np.deg2rad(theta_deg)
        r_values = np.linspace(r_range[0] * self.a, r_range[1] * self.a, n_r)
        
        R, THETA = np.meshgrid(r_values, theta_rad)
        
        sigma_normal = np.zeros_like(R)
        strain_energy_density = np.zeros_like(R)
        
        sigma_app_local = self.get_applied_stress_local()
        
        for i in range(n_theta):
            for j in range(n_r):
                r = R[i, j]
                theta = THETA[i, j]
                
                # Point in local coordinates
                x_local = tip_pos_local[0] + r * np.cos(theta)
                y_local = tip_pos_local[1] + r * np.sin(theta)
                point_local = np.array([x_local, y_local])
                
                # Calculate stress
                sigma_local = sigma_app_local.copy()
                
                for k in range(len(dipole_positions)):
                    x_k = dipole_positions[k]
                    b_k = b_magnitudes[k]
                    sigma_local += self.stress_from_dipole(point_local, b_k, x_k)
                
                # Normal stress on plane perpendicular to radial direction
                # n = [cos(theta), sin(theta)]
                n = np.array([np.cos(theta), np.sin(theta)])
                sigma_n = n @ sigma_local @ n
                sigma_normal[i, j] = sigma_n
                
                # Strain energy density
                # W = (1/2E) * [(σ_xx² + σ_yy²) - 2ν*σ_xx*σ_yy + 2(1+ν)*σ_xy²]
                E = 2 * self.material.mu * (1 + self.material.nu)
                W = (1 / (2 * E)) * (
                    sigma_local[0,0]**2 + sigma_local[1,1]**2 
                    - 2*self.material.nu*sigma_local[0,0]*sigma_local[1,1] 
                    + 2*(1+self.material.nu)*sigma_local[0,1]**2
                )
                strain_energy_density[i, j] = W
        
        return R, THETA, sigma_normal, strain_energy_density, tip_pos_local
    
    # Add this to dce_calculations.py in the solve() method

    def solve(self, n_dipoles=30, tip_fraction=0.95):
        """
        Complete solution with DEBUG output
        
        Returns: results dictionary
        """
        print("\n" + "="*70)
        print("STRESS TRANSFORMATION DEBUG")
        print("="*70)
        
        # Show applied stress in global coordinates
        sigma_global = np.array([
            [self.stress.sigma_xx, self.stress.sigma_xy],
            [self.stress.sigma_xy, self.stress.sigma_yy]
        ])
        
        print("\nApplied stress (GLOBAL coordinates):")
        print(f"  σ_xx = {self.stress.sigma_xx/1e6:8.2f} MPa")
        print(f"  σ_yy = {self.stress.sigma_yy/1e6:8.2f} MPa")
        print(f"  σ_xy = {self.stress.sigma_xy/1e6:8.2f} MPa")
        
        # Transform to local crack coordinates
        sigma_local = self.get_applied_stress_local()
        
        print(f"\nCrack orientation: θ = {np.rad2deg(self.crack.angle):.1f}°")
        print("\nApplied stress (LOCAL crack coordinates):")
        print(f"  σ_xx_local = {sigma_local[0,0]/1e6:8.2f} MPa (parallel to crack)")
        print(f"  σ_yy_local = {sigma_local[1,1]/1e6:8.2f} MPa (normal to crack)")
        print(f"  σ_xy_local = {sigma_local[0,1]/1e6:8.2f} MPa")
        
        # Check if normal stress is tension or compression
        if sigma_local[1, 1] > 0:
            print("\n  → TENSION normal to crack (crack will OPEN)")
        elif sigma_local[1, 1] < 0:
            print("\n  → COMPRESSION normal to crack (crack will CLOSE)")
            print("  ⚠ WARNING: Negative COD expected - crack faces would interpenetrate!")
            print("             This is non-physical for open cracks.")
        else:
            print("\n  → ZERO normal stress (no Mode-I loading)")
        
        print("="*70 + "\n")
        
        # Solve for Burgers vectors
        dipole_positions, b_magnitudes = self.solve_burgers_vectors(n_dipoles, tip_fraction)
        
        # Calculate K_I
        K_I, K_I_analytical = self.calculate_K_I(dipole_positions, b_magnitudes)
        
        results = {
            'dipole_positions': dipole_positions,
            'b_magnitudes': b_magnitudes,
            'K_I': K_I,
            'K_I_analytical': K_I_analytical,
            'error_percent': abs(K_I - K_I_analytical) / K_I_analytical * 100,
            'sigma_local': sigma_local
        }
        
        return results