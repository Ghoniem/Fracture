"""
Pure Python BEM Solver - Corrected Stress Kernels
Based on: Aliabadi (2002) "The Boundary Element Method, Vol 2"
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
    
    def _stress_kernels(self, xi, yi, x, y, nx, ny):
        """
        Stress kernels for interior point stress calculation
        
        Based on Aliabadi (2002), equations for σ at interior point
        
        σ(x) = ∫[D(x,ξ)·t(ξ) - S(x,ξ)·u(ξ)]dΓ
        
        where r points from source ξ to field x
        """
        # Vector from source to field
        dx = x - xi
        dy = y - yi
        r = np.sqrt(dx*dx + dy*dy)
        
        if r < 1e-10:
            D = np.zeros((3, 2))
            S = np.zeros((3, 2))
            return D, S
        
        r2 = r * r
        r3 = r2 * r
        
        # Direction cosines
        c_x = dx / r
        c_y = dy / r
        
        # ∂r/∂n
        drdn = (dx * nx + dy * ny) / r
        
        # Material constants
        mu = self.G
        nu = self.nu
        
        # Initialize
        D = np.zeros((3, 2))
        S = np.zeros((3, 2))
        
        # ==========================================
        # D matrix: σ from traction
        # From Kelvin solution derivatives
        # ==========================================
        
        # Coefficient for D (from Kelvin solution)
        # Standard form: σ_ij = C * [derivatives of U_ij * t_k]
        C_D = 1.0 / (2.0 * np.pi * (1.0 - nu) * r)
        
        # σ_xx from [t_x, t_y]
        D[0, 0] = C_D * ((1.0 - 2.0*nu) * c_x - 2.0 * c_x**3)
        D[0, 1] = C_D * ((1.0 - 2.0*nu) * c_y - 2.0 * c_x**2 * c_y)
        
        # σ_yy from [t_x, t_y]
        D[1, 0] = C_D * ((1.0 - 2.0*nu) * c_x - 2.0 * c_y**2 * c_x)
        D[1, 1] = C_D * ((1.0 - 2.0*nu) * c_y - 2.0 * c_y**3)
        
        # σ_xy from [t_x, t_y]
        D[2, 0] = C_D * ((1.0 - 2.0*nu) * c_y - 2.0 * c_x**2 * c_y)
        D[2, 1] = C_D * ((1.0 - 2.0*nu) * c_x - 2.0 * c_x * c_y**2)
        
        # ==========================================
        # S matrix: σ from displacement
        # ==========================================
        
        # Coefficient for S (from traction kernel derivative)
        C_S = mu / (np.pi * (1.0 - nu) * r2)
        
        # Common term
        alpha = 1.0 - 2.0 * nu
        
        # σ_xx from [u_x, u_y]
        S[0, 0] = C_S * (
            drdn * (alpha + 2.0 * c_x**2) +
            alpha * (nx * c_x - ny * c_y)
        )
        S[0, 1] = C_S * (
            drdn * 2.0 * c_x * c_y +
            alpha * (ny * c_x + nx * c_y)
        )
        
        # σ_yy from [u_x, u_y]
        S[1, 0] = C_S * (
            drdn * 2.0 * c_y * c_x +
            alpha * (ny * c_x + nx * c_y)
        )
        S[1, 1] = C_S * (
            drdn * (alpha + 2.0 * c_y**2) +
            alpha * (ny * c_y - nx * c_x)
        )
        
        # σ_xy from [u_x, u_y]
        S[2, 0] = C_S * (
            drdn * 2.0 * c_x * c_y +
            alpha * ny * c_x
        )
        S[2, 1] = C_S * (
            drdn * (alpha - 2.0 * c_x * c_y) +
            alpha * (-nx * c_x)
        )
        
        return D, S
    
    def _integrate_element(self, elem, xi, yi):
        """Numerical integration over element"""
        x1, y1 = elem['x1'], elem['y1']
        x2, y2 = elem['x2'], elem['y2']
        
        # Element properties
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        tx = (x2 - x1) / length
        ty = (y2 - y1) / length
        nx = ty   # Outward normal (90° CCW rotation)
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
        
        # Extract solutions
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
            nx = ty
            ny = -tx
            
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
        Compute stress at interior point
        
        σ(x) = ∫_Γ [D(x,ξ)·t(ξ) - S(x,ξ)·u(ξ)] dΓ(ξ)
        """
        if self.u_x is None:
            raise RuntimeError("Must call solve() before computing stress field")
        
        sigma_xx = 0.0
        sigma_yy = 0.0
        sigma_xy = 0.0
        
        for idx, elem in enumerate(self.elements):
            x1, y1 = elem['x1'], elem['y1']
            x2, y2 = elem['x2'], elem['y2']
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            
            # Tangent and normal
            tx = (x2 - x1) / length
            ty = (y2 - y1) / length
            nx = ty
            ny = -tx
            
            elem_u_x = self.u_x[idx]
            elem_u_y = self.u_y[idx]
            elem_t_x = self.t_x[idx]
            elem_t_y = self.t_y[idx]
            
            # Numerical integration
            n_gauss = 12
            
            for i in range(n_gauss):
                s = -1.0 + 2.0 * (i + 0.5) / n_gauss
                w = 2.0 / n_gauss
                
                xs = 0.5 * ((1.0 - s) * x1 + (1.0 + s) * x2)
                ys = 0.5 * ((1.0 - s) * y1 + (1.0 + s) * y2)
                
                r = np.sqrt((x_field - xs)**2 + (y_field - ys)**2)
                if r < 1e-10:
                    continue
                
                D, S = self._stress_kernels(xs, ys, x_field, y_field, nx, ny)
                
                jacobian = length / 2.0
                
                u_vec = np.array([elem_u_x, elem_u_y])
                t_vec = np.array([elem_t_x, elem_t_y])
                
                stress_contrib = D @ t_vec - S @ u_vec
                
                sigma_xx += stress_contrib[0] * w * jacobian
                sigma_yy += stress_contrib[1] * w * jacobian
                sigma_xy += stress_contrib[2] * w * jacobian
        
        return sigma_xx, sigma_yy, sigma_xy
    
    def compute_interior_stress(self, x, y):
        """Alias for compute_stress_at_point"""
        return self.compute_stress_at_point(x, y)
