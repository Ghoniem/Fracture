"""
Pure Python BEM Solver - Corrected Sign Conventions
Fixed sign issues in stress calculation
"""
import numpy as np

class BEMSolver2D:
    def __init__(self, E, nu, plane_strain=True):
        self.E = E
        self.nu = nu
        self.G = E / (2.0 * (1.0 + nu))
        if plane_strain:
            self.kappa = 3.0 - 4.0 * nu
        else:
            self.kappa = (3.0 - nu) / (1.0 + nu)
        self.elements = []
        self.u_x = None
        self.u_y = None
        self.t_x = None
        self.t_y = None
    
    def add_element(self, x1, y1, x2, y2, is_traction, bc_x, bc_y):
        elem = {
            'x1': x1, 'y1': y1,
            'x2': x2, 'y2': y2,
            'is_traction': is_traction,
            'bc_x': bc_x, 'bc_y': bc_y
        }
        self.elements.append(elem)
    
    def clear_elements(self):
        self.elements = []
        self.u_x = None
        self.u_y = None
        self.t_x = None
        self.t_y = None
    
    def _kelvin_u(self, xi, yi, x, y):
        """Displacement fundamental solution"""
        dx = x - xi
        dy = y - yi
        r = np.sqrt(dx*dx + dy*dy)
        
        if r < 1e-10:
            return np.zeros((2, 2))
        
        c1 = 1.0 / (8.0 * np.pi * self.G * (1.0 - self.nu))
        log_r = np.log(r)
        
        U = np.zeros((2, 2))
        U[0, 0] = c1 * (-(self.kappa + 1.0) * log_r + dx*dx/(r*r))
        U[0, 1] = c1 * (dx*dy/(r*r))
        U[1, 0] = c1 * (dx*dy/(r*r))
        U[1, 1] = c1 * (-(self.kappa + 1.0) * log_r + dy*dy/(r*r))
        
        return U
    
    def _kelvin_t(self, xi, yi, x, y, nx, ny):
        """Traction fundamental solution"""
        dx = x - xi
        dy = y - yi
        r = np.sqrt(dx*dx + dy*dy)
        
        if r < 1e-10:
            return np.zeros((2, 2))
        
        dr_dn = (dx * nx + dy * ny) / r
        c2 = -1.0 / (4.0 * np.pi * (1.0 - self.nu) * r)
        
        r_i = np.array([dx/r, dy/r])
        
        T = np.zeros((2, 2))
        T[0, 0] = c2 * (dr_dn * ((1.0 - 2.0*self.nu) + 2.0*r_i[0]*r_i[0]))
        T[0, 1] = c2 * (dr_dn * (2.0*r_i[0]*r_i[1]) - (1.0 - 2.0*self.nu) * (r_i[0]*ny - r_i[1]*nx))
        T[1, 0] = c2 * (dr_dn * (2.0*r_i[1]*r_i[0]) - (1.0 - 2.0*self.nu) * (r_i[1]*nx - r_i[0]*ny))
        T[1, 1] = c2 * (dr_dn * ((1.0 - 2.0*self.nu) + 2.0*r_i[1]*r_i[1]))
        
        return T
    
    def _kelvin_s(self, xi, yi, x, y, nx, ny):
        """
        Stress fundamental solution for traction on boundary
        S_ijk: stress at x due to unit traction in direction k at ξ
        
        CORRECTED FORMULATION based on standard BEM references
        """
        dx = x - xi
        dy = y - yi
        r = np.sqrt(dx*dx + dy*dy)
        
        if r < 1e-10:
            return np.zeros((3, 2))
        
        r2 = r * r
        
        # Direction cosines
        dxr = dx / r
        dyr = dy / r
        dr_dn = (dx * nx + dy * ny) / r
        
        # Coefficient
        c = 1.0 / (4.0 * np.pi * (1.0 - self.nu) * r)
        
        # S matrix: rows are [σ_xx, σ_yy, σ_xy], columns are [t_x, t_y]
        S = np.zeros((3, 2))
        
        # For σ_xx from t_x and t_y
        S[0, 0] = c * ((1.0 - 2.0*self.nu) * (ny*dyr - 2.0*dr_dn*dxr) - 
                       2.0*dr_dn*dxr*(1.0 - 2.0*dxr*dxr))
        S[0, 1] = c * (-(1.0 - 2.0*self.nu) * (nx*dyr + 2.0*dr_dn*dyr) - 
                       2.0*dr_dn*dxr*2.0*dxr*dyr)
        
        # For σ_yy from t_x and t_y  
        S[1, 0] = c * ((1.0 - 2.0*self.nu) * (ny*dxr + 2.0*dr_dn*dxr) - 
                       2.0*dr_dn*dyr*2.0*dxr*dyr)
        S[1, 1] = c * (-(1.0 - 2.0*self.nu) * (nx*dxr - 2.0*dr_dn*dyr) - 
                       2.0*dr_dn*dyr*(1.0 - 2.0*dyr*dyr))
        
        # For σ_xy from t_x and t_y
        S[2, 0] = c * ((1.0 - 2.0*self.nu) * (ny*dxr + nx*dyr + 2.0*dr_dn*dyr) - 
                       2.0*dr_dn*(dxr*dyr + dxr*dyr*(1.0 - 2.0*dxr*dxr)))
        S[2, 1] = c * ((1.0 - 2.0*self.nu) * (-2.0*dr_dn*dxr) - 
                       2.0*dr_dn*(dxr*dyr - dyr*dyr*(1.0 - 2.0*dxr*dxr)))
        
        return S
    
    def _kelvin_d(self, xi, yi, x, y):
        """
        Stress from displacement (derivative of U)
        D_ijk: stress at x due to unit displacement in direction k at ξ
        
        CORRECTED: Uses proper stress-displacement relations
        σ_ij = λδ_ij ∂u_k/∂x_k + μ(∂u_i/∂x_j + ∂u_j/∂x_i)
        """
        dx = x - xi
        dy = y - yi
        r = np.sqrt(dx*dx + dy*dy)
        
        if r < 1e-10:
            return np.zeros((3, 2))
        
        r2 = r * r
        r3 = r2 * r
        
        # Material constants for plane strain
        lam = 2.0 * self.G * self.nu / (1.0 - 2.0*self.nu)
        if hasattr(self, 'kappa'):
            # Use kappa formulation
            c = self.G / (2.0 * np.pi * (1.0 - self.nu))
        else:
            c = 1.0 / (8.0 * np.pi * (1.0 - self.nu))
        
        # Direction cosines
        dxr = dx / r
        dyr = dy / r
        
        # D matrix: rows are [σ_xx, σ_yy, σ_xy], columns are [u_x, u_y]
        D = np.zeros((3, 2))
        
        # Derivatives of U (displacement fundamental solution)
        # ∂U_ij/∂x_k at field point x
        
        const = 1.0 / (4.0 * np.pi * (1.0 - self.nu) * r)
        
        # For σ_xx = λ(∂u_x/∂x + ∂u_y/∂y) + 2μ∂u_x/∂x
        # From u_x:
        D[0, 0] = const * ((1.0 - 2.0*self.nu) * (-dxr) + 2.0*dxr*dxr*dxr)
        # From u_y:
        D[0, 1] = const * ((1.0 - 2.0*self.nu) * (-dyr) + 2.0*dxr*dxr*dyr)
        
        # For σ_yy = λ(∂u_x/∂x + ∂u_y/∂y) + 2μ∂u_y/∂y
        # From u_x:
        D[1, 0] = const * ((1.0 - 2.0*self.nu) * (-dxr) + 2.0*dyr*dyr*dxr)
        # From u_y:
        D[1, 1] = const * ((1.0 - 2.0*self.nu) * (-dyr) + 2.0*dyr*dyr*dyr)
        
        # For σ_xy = μ(∂u_x/∂y + ∂u_y/∂x)
        # From u_x:
        D[2, 0] = const * ((1.0 - 2.0*self.nu) * (-dyr) + 2.0*dxr*dyr*dxr)
        # From u_y:
        D[2, 1] = const * ((1.0 - 2.0*self.nu) * (-dxr) + 2.0*dxr*dyr*dyr)
        
        return D
    
    def _integrate_element(self, elem, xi, yi):
        """Numerical integration over element"""
        x1, y1 = elem['x1'], elem['y1']
        x2, y2 = elem['x2'], elem['y2']
        
        # Element properties
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        tx = (x2 - x1) / length  # Tangent
        ty = (y2 - y1) / length
        nx = ty   # Outward normal (rotate tangent 90° CCW)
        ny = -tx
        
        # Midpoint for collocation
        xm = 0.5 * (x1 + x2)
        ym = 0.5 * (y1 + y2)
        
        dist_to_elem = np.sqrt((xi - xm)**2 + (yi - ym)**2)
        is_singular = (dist_to_elem < 0.1 * length)
        
        H = np.zeros((2, 2))
        G_mat = np.zeros((2, 2))
        
        if is_singular:
            # Free term for singular integral
            H[0, 0] = -0.5
            H[1, 1] = -0.5
            # Use more points for weakly singular integral
            n_gauss = 20
        else:
            n_gauss = 8
        
        # Gauss quadrature
        for i in range(n_gauss):
            s = -1.0 + 2.0 * (i + 0.5) / n_gauss
            w = 2.0 / n_gauss
            
            # Map to element
            x = 0.5 * ((1.0 - s) * x1 + (1.0 + s) * x2)
            y = 0.5 * ((1.0 - s) * y1 + (1.0 + s) * y2)
            
            U = self._kelvin_u(xi, yi, x, y)
            T = self._kelvin_t(xi, yi, x, y, nx, ny)
            
            jacobian = length / 2.0
            
            G_mat += U * w * jacobian
            if not is_singular:
                H += T * w * jacobian
        
        return H, G_mat
    
    def solve(self):
        n = len(self.elements)
        if n == 0:
            raise RuntimeError("No boundary elements defined")
        
        # Build system matrix
        A = np.zeros((2*n, 2*n))
        b = np.zeros(2*n)
        
        for i in range(n):
            elem_i = self.elements[i]
            xi = 0.5 * (elem_i['x1'] + elem_i['x2'])
            yi = 0.5 * (elem_i['y1'] + elem_i['y2'])
            
            for j in range(n):
                elem_j = self.elements[j]
                H, G = self._integrate_element(elem_j, xi, yi)
                
                if elem_j['is_traction']:
                    # Known traction, unknown displacement
                    A[2*i:2*i+2, 2*j:2*j+2] = H
                    b[2*i:2*i+2] += G @ np.array([elem_j['bc_x'], elem_j['bc_y']])
                else:
                    # Known displacement, unknown traction
                    A[2*i:2*i+2, 2*j:2*j+2] = -G
                    b[2*i:2*i+2] += H @ np.array([elem_j['bc_x'], elem_j['bc_y']])
        
        # Solve system
        x = np.linalg.solve(A, b)
        
        # Extract solutions and store
        self.u_x = np.zeros(n)
        self.u_y = np.zeros(n)
        self.t_x = np.zeros(n)
        self.t_y = np.zeros(n)
        
        for i in range(n):
            if self.elements[i]['is_traction']:
                self.u_x[i] = x[2*i]
                self.u_y[i] = x[2*i+1]
                self.t_x[i] = self.elements[i]['bc_x']
                self.t_y[i] = self.elements[i]['bc_y']
            else:
                self.t_x[i] = x[2*i]
                self.t_y[i] = x[2*i+1]
                self.u_x[i] = self.elements[i]['bc_x']
                self.u_y[i] = self.elements[i]['bc_y']
        
        return self.u_x, self.u_y, self.t_x, self.t_y
    
    def compute_interior_displacement(self, x, y):
        """Compute displacement at interior point"""
        if self.u_x is None:
            raise RuntimeError("Must call solve() before computing interior fields")
        
        u_x = 0.0
        u_y = 0.0
        
        for idx, elem in enumerate(self.elements):
            x1, y1 = elem['x1'], elem['y1']
            x2, y2 = elem['x2'], elem['y2']
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            tx = (x2 - x1) / length
            ty = (y2 - y1) / length
            nx = ty   # Outward normal
            ny = -tx
            
            # Use solved values
            elem_u_x = self.u_x[idx]
            elem_u_y = self.u_y[idx]
            elem_t_x = self.t_x[idx]
            elem_t_y = self.t_y[idx]
            
            # Integrate over element
            n_gauss = 8
            for i in range(n_gauss):
                s = -1.0 + 2.0 * (i + 0.5) / n_gauss
                w = 2.0 / n_gauss
                
                xs = 0.5 * ((1.0 - s) * x1 + (1.0 + s) * x2)
                ys = 0.5 * ((1.0 - s) * y1 + (1.0 + s) * y2)
                
                U = self._kelvin_u(x, y, xs, ys)
                T = self._kelvin_t(x, y, xs, ys, nx, ny)
                
                jacobian = length / 2.0
                
                u_contrib = U @ np.array([elem_t_x, elem_t_y]) - T @ np.array([elem_u_x, elem_u_y])
                u_x += u_contrib[0] * w * jacobian
                u_y += u_contrib[1] * w * jacobian
        
        return u_x, u_y
    
    def compute_stress_at_point(self, x_field, y_field):
        """
        Compute stress components at an interior point using BEM solution.
        
        CORRECTED BEM stress integral:
        σ_ij(x) = ∫_Γ [D_ijk(x,ξ) t_k(ξ) - S_ijk(x,ξ) u_k(ξ)] dΓ(ξ)
        
        Note the sign: +D·t - S·u (not -D·t + S·u)
        
        Parameters:
        -----------
        x_field, y_field : float
            Coordinates of the field point where stress is evaluated
        
        Returns:
        --------
        sigma_xx, sigma_yy, sigma_xy : float
            Stress components at the field point
        """
        if self.u_x is None:
            raise RuntimeError("Must call solve() before computing stress field")
        
        sigma_xx = 0.0
        sigma_yy = 0.0
        sigma_xy = 0.0
        
        # Loop over all boundary elements
        for idx, elem in enumerate(self.elements):
            # Get element geometry
            x1, y1 = elem['x1'], elem['y1']
            x2, y2 = elem['x2'], elem['y2']
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            
            # Tangent and outward normal
            tx = (x2 - x1) / length
            ty = (y2 - y1) / length
            nx = ty   # Outward normal (90° CCW rotation of tangent)
            ny = -tx
            
            # Get solved boundary values for this element
            elem_u_x = self.u_x[idx]
            elem_u_y = self.u_y[idx]
            elem_t_x = self.t_x[idx]
            elem_t_y = self.t_y[idx]
            
            # Numerical integration over element
            n_gauss = 12  # Higher order for stress calculation
            
            for i in range(n_gauss):
                # Gauss point in parametric space
                s = -1.0 + 2.0 * (i + 0.5) / n_gauss
                w = 2.0 / n_gauss
                
                # Map to physical element
                xs = 0.5 * ((1.0 - s) * x1 + (1.0 + s) * x2)
                ys = 0.5 * ((1.0 - s) * y1 + (1.0 + s) * y2)
                
                # Check if field point is too close to source point
                r = np.sqrt((x_field - xs)**2 + (y_field - ys)**2)
                if r < 1e-10:
                    continue
                
                # Compute fundamental solutions
                D = self._kelvin_d(xs, ys, x_field, y_field)  # Stress from displacement
                S = self._kelvin_s(xs, ys, x_field, y_field, nx, ny)  # Stress from traction
                
                jacobian = length / 2.0
                
                # CORRECTED BEM integral: σ = ∫[D·t - S·u]dΓ
                u_vec = np.array([elem_u_x, elem_u_y])
                t_vec = np.array([elem_t_x, elem_t_y])
                
                stress_contrib = D @ t_vec - S @ u_vec
                
                sigma_xx += stress_contrib[0] * w * jacobian
                sigma_yy += stress_contrib[1] * w * jacobian
                sigma_xy += stress_contrib[2] * w * jacobian
        
        return sigma_xx, sigma_yy, sigma_xy
    
    def compute_interior_stress(self, x, y):
        """Alias for compute_stress_at_point for backward compatibility"""
        return self.compute_stress_at_point(x, y)
