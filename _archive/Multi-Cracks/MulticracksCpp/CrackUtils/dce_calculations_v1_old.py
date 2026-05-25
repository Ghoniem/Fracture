"""
DCE Calculations Module - Version 1
DCE Element formulation: half-crack building blocks with independent Mode-I and Mode-II
CORRECTED: Opposite unit vectors for symmetric crack closure
"""

import numpy as np
from scipy.linalg import solve
from dataclasses import dataclass
from typing import List, Tuple, Dict
from enum import Enum

class DCEPolarity(Enum):
    """
    DCE element polarity using right-hand rule
    
    For horizontal crack:
    - POSITIVE: tip points to the RIGHT (+x1 direction)
    - NEGATIVE: tip points to the LEFT (-x1 direction)
    """
    POSITIVE = 1
    NEGATIVE = -1

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
    center: Tuple[float, float] = (0.0, 0.0)

@dataclass
class AppliedStress:
    """Applied stress tensor in GLOBAL coordinates"""
    sigma_xx: float
    sigma_yy: float
    sigma_xy: float

class DCEElement:
    """
    DCE Element: Represents HALF of a crack
    
    Contains:
    - N dislocations for Mode-I (perpendicular to crack, x2 direction)
    - N dislocations for Mode-II (parallel to crack, x1 direction)
    """
    
    def __init__(self, polarity: DCEPolarity, n_dislocations: int, 
                 tip_position: float, base_position: float):
        """
        Initialize DCE element
        
        Args:
            polarity: Element polarity (POSITIVE or NEGATIVE)
            n_dislocations: Number of dislocations per mode
            tip_position: Position of crack tip in local x1 coordinate
            base_position: Position of element base (center) in local x1
        """
        self.polarity = polarity
        self.n_dislocations = n_dislocations
        self.tip_position = tip_position
        self.base_position = base_position
        
        # Dislocation positions along crack (x1 direction)
        self.positions_x1 = np.linspace(tip_position, base_position, n_dislocations)
        
        # Burgers vectors (to be solved) - store signed values
        self.b_mode_I = np.zeros(n_dislocations)
        self.b_mode_II = np.zeros(n_dislocations)
        
    def get_burgers_vector_local(self, i: int) -> np.ndarray:
        """
        Get Burgers vector for i-th dislocation in local coordinates
        
        b_mode_I and b_mode_II are signed values from solver
        """
        b1 = self.b_mode_II[i]  # Mode-II (parallel)
        b2 = self.b_mode_I[i]   # Mode-I (perpendicular)
        
        return np.array([b1, b2])
    
    def get_all_dislocation_positions(self) -> np.ndarray:
        """Get all dislocation positions in local x1 coordinate"""
        return self.positions_x1
    
    def __repr__(self):
        return (f"DCEElement(polarity={self.polarity.name}, "
                f"n={self.n_dislocations}, "
                f"tip={self.tip_position:.4f}, "
                f"base={self.base_position:.4f})")

class DCECalculatorV1:
    """
    DCE Calculator - Version 1
    Works with DCE elements as building blocks
    """
    
    def __init__(self, material: Material, crack: Crack, stress: AppliedStress):
        self.material = material
        self.crack = crack
        self.stress = stress
        self.a = crack.length / 2
        
        # Rotation matrix
        c = np.cos(crack.angle)
        s = np.sin(crack.angle)
        self.R = np.array([[c, -s], [s, c]])
        self.R_inv = self.R.T
        
        # Crack direction and normal in global coordinates
        self.crack_direction_global = np.array([c, s])    # x1
        self.crack_normal_global = np.array([-s, c])      # x2
        
    def stress_from_edge_dislocation(self, r, b_vec, disl_pos):
        """
        Stress from single edge dislocation in LOCAL coordinates
        
        Standard edge dislocation stress field equations
        """
        x1 = r[0] - disl_pos[0]
        x2 = r[1] - disl_pos[1]
        r2 = x1**2 + x2**2
        
        if r2 < (1e-10)**2:
            return np.zeros((2, 2))
        
        D = self.material.mu / (2 * np.pi * (1 - self.material.nu))
        r4 = r2 * r2
        b1, b2 = b_vec[0], b_vec[1]
        
        sig = np.zeros((2, 2))
        
        # Mode-II contribution (b1)
        if abs(b1) > 1e-20:
            sig[0, 0] += -D * b1 * x2 * (3*x1**2 + x2**2) / r4
            sig[1, 1] += D * b1 * x2 * (x1**2 - x2**2) / r4
            sig[0, 1] += D * b1 * x1 * (x1**2 - x2**2) / r4
        
        # Mode-I contribution (b2)
        if abs(b2) > 1e-20:
            sig[0, 0] += D * b2 * x1 * (x1**2 - x2**2) / r4
            sig[1, 1] += -D * b2 * x1 * (x1**2 + 3*x2**2) / r4
            sig[0, 1] += -D * b2 * x2 * (x1**2 - x2**2) / r4
        
        sig[1, 0] = sig[0, 1]
        return sig
    
    def global_to_local(self, point_global):
        """Transform point from global to local"""
        point = np.array(point_global) - np.array(self.crack.center)
        return self.R_inv @ point
    
    def local_to_global(self, point_local):
        """Transform point from local to global"""
        point = self.R @ point_local
        return point + np.array(self.crack.center)
    
    def stress_global_to_local(self, sigma_global):
        """Transform stress tensor"""
        return self.R_inv @ sigma_global @ self.R_inv.T
    
    def stress_local_to_global(self, sigma_local):
        """Transform stress tensor"""
        return self.R @ sigma_local @ self.R.T
    
    def get_applied_stress_local(self):
        """Applied stress in local crack coordinates"""
        sigma_global = np.array([
            [self.stress.sigma_xx, self.stress.sigma_xy],
            [self.stress.sigma_xy, self.stress.sigma_yy]
        ])
        return self.stress_global_to_local(sigma_global)
    
    def create_symmetric_crack(self, n_dislocations_per_element, tip_fraction=0.95):
        """
        Create symmetric crack using TWO DCE elements
        
        Returns:
            List containing [negative_element, positive_element]
        """
        tip_position = self.a * tip_fraction
        base_position = self.a * 0.05
        
        # NEGATIVE element: tip at -a*tip_fraction, base at -a*0.05
        elem_negative = DCEElement(
            polarity=DCEPolarity.NEGATIVE,
            n_dislocations=n_dislocations_per_element,
            tip_position=-tip_position,
            base_position=-base_position
        )
        
        # POSITIVE element: tip at +a*tip_fraction, base at +a*0.05
        elem_positive = DCEElement(
            polarity=DCEPolarity.POSITIVE,
            n_dislocations=n_dislocations_per_element,
            tip_position=tip_position,
            base_position=base_position
        )
        
        return [elem_negative, elem_positive]
    
    def solve_mixed_mode(self, dce_elements: List[DCEElement]):
        """
        Solve for Mode-I and Mode-II Burgers vectors
        
        CORRECTED: Use opposite unit vectors for opposite sides
        - Left element (NEGATIVE): b in +x2 (opens crack)
        - Right element (POSITIVE): b in -x2 (closes crack)
        """
        n_total = sum(elem.n_dislocations for elem in dce_elements)
        
        sigma_local = self.get_applied_stress_local()
        sigma_22_applied = sigma_local[1, 1]
        sigma_12_applied = sigma_local[0, 1]
        
        print(f"\nMixed-mode DCE solution:")
        print(f"  Number of DCE elements: {len(dce_elements)}")
        print(f"  Total dislocations: {n_total} (per mode)")
        for i, elem in enumerate(dce_elements):
            print(f"    Element {i}: {elem.polarity.name}, n={elem.n_dislocations}")
        print(f"  Applied stress (local):")
        print(f"    σ_11 = {sigma_local[0,0]/1e6:.2f} MPa")
        print(f"    σ_22 = {sigma_local[1,1]/1e6:.2f} MPa (normal)")
        print(f"    σ_12 = {sigma_local[0,1]/1e6:.2f} MPa (shear)")
        
        # Build global dislocation index mapping
        dislocation_map = []
        for elem_idx, elem in enumerate(dce_elements):
            for local_idx in range(elem.n_dislocations):
                dislocation_map.append((elem_idx, local_idx))
        
        # ==================================================================
        # MODE-I SYSTEM
        # ==================================================================
        A_I = np.zeros((n_total-1, n_total))
        rhs_I = np.zeros(n_total-1)
        
        global_idx_eval = 0
        for i in range(1, n_total):
            elem_idx_i, local_idx_i = dislocation_map[i]
            elem_i = dce_elements[elem_idx_i]
            x1_i = elem_i.positions_x1[local_idx_i]
            eval_pos = np.array([x1_i, 0.0])
            
            rhs_I[global_idx_eval] = -sigma_22_applied
            
            for j in range(n_total):
                if i == j:
                    continue
                
                elem_idx_j, local_idx_j = dislocation_map[j]
                elem_j = dce_elements[elem_idx_j]
                x1_j = elem_j.positions_x1[local_idx_j]
                pos_j = np.array([x1_j, 0.0])
                
                # CRITICAL: Opposite unit vectors for symmetric closure
                if elem_j.polarity.value < 0:
                    # NEGATIVE element (left): +x2 direction (opens)
                    b_unit_I = np.array([0.0, 1.0])
                else:
                    # POSITIVE element (right): -x2 direction (closes)
                    b_unit_I = np.array([0.0, -1.0])
                
                sigma_ij = self.stress_from_edge_dislocation(eval_pos, b_unit_I, pos_j)
                A_I[global_idx_eval, j] = sigma_ij[1, 1]
            
            global_idx_eval += 1
        
        # Center equation
        A_I_full = np.zeros((n_total, n_total))
        A_I_full[:n_total-1, :] = A_I
        rhs_I_full = np.zeros(n_total)
        rhs_I_full[:n_total-1] = rhs_I
        
        eval_center = np.array([0.0, 0.0])
        rhs_I_full[n_total-1] = -sigma_22_applied
        
        for j in range(n_total):
            elem_idx_j, local_idx_j = dislocation_map[j]
            elem_j = dce_elements[elem_idx_j]
            x1_j = elem_j.positions_x1[local_idx_j]
            pos_j = np.array([x1_j, 0.0])
            
            if elem_j.polarity.value < 0:
                b_unit_I = np.array([0.0, 1.0])
            else:
                b_unit_I = np.array([0.0, -1.0])
            
            sigma_j = self.stress_from_edge_dislocation(eval_center, b_unit_I, pos_j)
            A_I_full[n_total-1, j] = sigma_j[1, 1]
        
        b_mode_I_global = solve(A_I_full, rhs_I_full)
        
        # ==================================================================
        # MODE-II SYSTEM
        # ==================================================================
        A_II = np.zeros((n_total-1, n_total))
        rhs_II = np.zeros(n_total-1)
        
        global_idx_eval = 0
        for i in range(1, n_total):
            elem_idx_i, local_idx_i = dislocation_map[i]
            elem_i = dce_elements[elem_idx_i]
            x1_i = elem_i.positions_x1[local_idx_i]
            eval_pos = np.array([x1_i, 0.0])
            
            rhs_II[global_idx_eval] = -sigma_12_applied
            
            for j in range(n_total):
                if i == j:
                    continue
                
                elem_idx_j, local_idx_j = dislocation_map[j]
                elem_j = dce_elements[elem_idx_j]
                x1_j = elem_j.positions_x1[local_idx_j]
                pos_j = np.array([x1_j, 0.0])
                
                # For Mode-II: same approach
                if elem_j.polarity.value < 0:
                    b_unit_II = np.array([1.0, 0.0])
                else:
                    b_unit_II = np.array([-1.0, 0.0])
                
                sigma_ij = self.stress_from_edge_dislocation(eval_pos, b_unit_II, pos_j)
                A_II[global_idx_eval, j] = sigma_ij[0, 1]
            
            global_idx_eval += 1
        
        A_II_full = np.zeros((n_total, n_total))
        A_II_full[:n_total-1, :] = A_II
        rhs_II_full = np.zeros(n_total)
        rhs_II_full[:n_total-1] = rhs_II
        
        rhs_II_full[n_total-1] = -sigma_12_applied
        
        for j in range(n_total):
            elem_idx_j, local_idx_j = dislocation_map[j]
            elem_j = dce_elements[elem_idx_j]
            x1_j = elem_j.positions_x1[local_idx_j]
            pos_j = np.array([x1_j, 0.0])
            
            if elem_j.polarity.value < 0:
                b_unit_II = np.array([1.0, 0.0])
            else:
                b_unit_II = np.array([-1.0, 0.0])
            
            sigma_j = self.stress_from_edge_dislocation(eval_center, b_unit_II, pos_j)
            A_II_full[n_total-1, j] = sigma_j[0, 1]
        
        b_mode_II_global = solve(A_II_full, rhs_II_full)
        
        # Store signed values directly
        for global_idx in range(n_total):
            elem_idx, local_idx = dislocation_map[global_idx]
            dce_elements[elem_idx].b_mode_I[local_idx] = b_mode_I_global[global_idx]
            dce_elements[elem_idx].b_mode_II[local_idx] = b_mode_II_global[global_idx]
        
        # DEBUG output
        print(f"\nDEBUG: Burgers vectors (signed values from solver)")
        for elem_idx, elem in enumerate(dce_elements):
            print(f"Element {elem_idx} ({elem.polarity.name}):")
            print(f"  Mode-I range: [{np.min(elem.b_mode_I)*1e9:.2f}, {np.max(elem.b_mode_I)*1e9:.2f}] nm")
            print(f"  Mode-II range: [{np.min(elem.b_mode_II)*1e9:.2f}, {np.max(elem.b_mode_II)*1e9:.2f}] nm")
        
        return dce_elements
    
    def calculate_K_I_K_II(self, dce_elements: List[DCEElement]):
        """Calculate stress intensity factors"""
        sigma_local = self.get_applied_stress_local()
        
        # Find rightmost tip (positive element tip)
        tip_elem = None
        tip_elem_idx = None
        tip_local_idx = None
        max_x1 = -np.inf
        
        for elem_idx, elem in enumerate(dce_elements):
            for local_idx, x1 in enumerate(elem.positions_x1):
                if x1 > max_x1:
                    max_x1 = x1
                    tip_elem = elem
                    tip_elem_idx = elem_idx
                    tip_local_idx = local_idx
        
        x1_tip = max_x1
        pos_tip = np.array([x1_tip, 0.0])
        
        # Calculate total stress at tip
        sigma_app = np.array([
            [0.0, sigma_local[0,1]],
            [sigma_local[0,1], sigma_local[1,1]]
        ])
        sigma_total = sigma_app.copy()
        
        for elem_idx, elem in enumerate(dce_elements):
            for local_idx in range(elem.n_dislocations):
                if elem_idx == tip_elem_idx and local_idx == tip_local_idx:
                    continue
                
                x1 = elem.positions_x1[local_idx]
                pos = np.array([x1, 0.0])
                b = elem.get_burgers_vector_local(local_idx)
                
                sigma_total += self.stress_from_edge_dislocation(pos_tip, b, pos)
        
        # Peach-Koehler force
        b_tip = tip_elem.get_burgers_vector_local(tip_local_idx)
        sigma_dot_b = sigma_total @ b_tip
        
        f1 = sigma_dot_b[1]
        f2 = -sigma_dot_b[0]
        
        K_I = np.sqrt(2 * self.material.mu * abs(f2) / (1 - self.material.nu))
        K_II = np.sqrt(2 * self.material.mu * abs(f1) / (1 - self.material.nu))
        
        K_I_analytical = abs(sigma_local[1,1]) * np.sqrt(np.pi * self.a)
        K_II_analytical = abs(sigma_local[0,1]) * np.sqrt(np.pi * self.a)
        
        return K_I, K_II, K_I_analytical, K_II_analytical
    
    def solve(self, n_dislocations_per_element=15, tip_fraction=0.95):
        """Complete solution using two DCE elements"""
        print("\n" + "="*70)
        print("DCE ANALYSIS - VERSION 1")
        print("Using DCE elements as building blocks")
        print("="*70)
        
        # Create symmetric crack (2 elements)
        dce_elements = self.create_symmetric_crack(n_dislocations_per_element, tip_fraction)
        
        # Solve
        dce_elements = self.solve_mixed_mode(dce_elements)
        
        # Calculate SIFs
        K_I, K_II, K_I_ana, K_II_ana = self.calculate_K_I_K_II(dce_elements)
        
        error_I = abs(K_I - K_I_ana) / K_I_ana * 100 if K_I_ana > 0 else 0
        error_II = abs(K_II - K_II_ana) / K_II_ana * 100 if K_II_ana > 0 else 0
        
        results = {
            'dce_elements': dce_elements,
            'K_I': K_I,
            'K_II': K_II,
            'K_I_analytical': K_I_ana,
            'K_II_analytical': K_II_ana,
            'error_I_percent': error_I,
            'error_II_percent': error_II
        }
        
        print(f"\n{'='*70}")
        print(f"RESULTS:")
        print(f"  K_I (DCE):         {K_I/1e6:.3f} MPa√m")
        print(f"  K_I (analytical):  {K_I_ana/1e6:.3f} MPa√m")
        print(f"  Error (Mode-I):    {error_I:.2f}%")
        print(f"  K_II (DCE):        {K_II/1e6:.3f} MPa√m")
        print(f"  K_II (analytical): {K_II_ana/1e6:.3f} MPa√m")
        print(f"  Error (Mode-II):   {error_II:.2f}%")
        print(f"{'='*70}\n")
        
        return results