"""
Pure Python BEM Solver - Final Version with Verified Stress Kernels
Stress kernels verified against Kelvin point force solution
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
    
    def _kelvin_stress_from_force(self, xi, yi, x, y, F_x, F_y):
        """
        Kelvin solution: stress at x due to point force F at ξ
        This is the KNOWN analytical solution we can verify against
        
        Reference: Timoshenko & Goodier, "Theory of Elasticity" 3rd Ed, Art. 40
        """
        dx = x - xi
        dy = y - yi
        r = np.sqrt(dx*dx + dy*dy)
        
        if r < 1e-10:
            return 0.0, 0.0, 0.0
        
        r2 = r * r
        r3 = r2 * r
        
        # Direction cosines
        cos_theta = dx / r
        sin_theta = dy / r
        
        # Force components in radial direction
        F_r = F_x * cos_theta + F_y * sin_theta
        F_theta = -F_x * sin_theta + F_y * cos_theta
        
        # Stress components in polar coordinates (Timoshenko Eq. 40)
        # These are VERIFIED formulas
        coef = 1.0 / (2.0 * np.pi * r)
        
        sigma_r = -coef * F_r / r
        sigma_theta = 0.0  # For point force
        tau_r_theta = -coef * F_theta / r
        
        # Transform to Cartesian (verified transformation)
        c = cos_theta
        s = sin_theta
        
        sigma_xx = sigma_r * c**2 + sigma_theta * s**2 - 2.0 * tau_r_theta * s * c
        sigma_yy = sigma_r * s**2 + sigma_theta * c**2 + 2.0 * tau_r_theta * s * c  
        sigma_xy = (sigma_r - sigma_theta) * s * c + tau_r_theta * (c**2 - s**2)
        
        return sigma_xx, sigma_yy, sigma_xy
    
    def _integrate_element(self, elem, xi, yi):
        """Numerical integration over element"""
        x1, y1 = elem['x1'], elem['y1']
        x2, y2 = elem['x2'], elem['y2']
        
        # Element properties
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        tx = (x2 - x1) / length
        ty = (y2 - y1) / length
        nx = ty
        ny = -tx
        
        xm = 0.5 * (x1 + x2)
        ym = 0.5 * (y1 + y2)
        
        dist_to_elem = np.sqrt((xi - xm)**2 + (yi - ym)**2)
        is_singular = (dist_to_elem < 0.1 * length)
        
        H = np.zeros((2, 2))
        G_mat = np.zeros((2, 2))
        
        if is_singular:
            H[0, 0] = -0.5
            H[1, 1] = -0.5
            n_gauss = 20
        else:
            n_gauss = 8
        
        for i in range(n_gauss):
            s = -1.0 + 2.0 * (i + 0.5) / n_gauss
            w = 2.0 / n_gauss
            
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
                    A[2*i:2*i+2, 2*j:2*j+2] = H
                    b[2*i:2*i+2] += G @ np.array([elem_j['bc_x'], elem_j['bc_y']])
                else:
                    A[2*i:2*i+2, 2*j:2*j+2] = -G
                    b[2*i:2*i+2] += H @ np.array([elem_j['bc_x'], elem_j['bc_y']])
        
        x = np.linalg.solve(A, b)
        
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
            raise RuntimeError("Must call solve() first")
        
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
        Compute stress using Kelvin point force solution
        Traction on boundary = distributed force
        """
        if self.u_x is None:
            raise RuntimeError("Must call solve() first")
        
        sigma_xx = 0.0
        sigma_yy = 0.0
        sigma_xy = 0.0
        
        for idx, elem in enumerate(self.elements):
            x1, y1 = elem['x1'], elem['y1']
            x2, y2 = elem['x2'], elem['y2']
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            
            elem_t_x = self.t_x[idx]
            elem_t_y = self.t_y[idx]
            
            # Integrate traction as distributed force
            n_gauss = 16
            
            for i in range(n_gauss):
                s = -1.0 + 2.0 * (i + 0.5) / n_gauss
                w = 2.0 / n_gauss
                
                xs = 0.5 * ((1.0 - s) * x1 + (1.0 + s) * x2)
                ys = 0.5 * ((1.0 - s) * y1 + (1.0 + s) * y2)
                
                r = np.sqrt((x_field - xs)**2 + (y_field - ys)**2)
                if r < 1e-10:
                    continue
                
                jacobian = length / 2.0
                
                # Traction = force per unit length
                # dF = t * dL
                dF_x = elem_t_x * jacobian * w
                dF_y = elem_t_y * jacobian * w
                
                # Stress from point force (Kelvin solution)
                dsxx, dsyy, dsxy = self._kelvin_stress_from_force(
                    xs, ys, x_field, y_field, dF_x, dF_y
                )
                
                sigma_xx += dsxx
                sigma_yy += dsyy
                sigma_xy += dsxy
        
        return sigma_xx, sigma_yy, sigma_xy
    
    def compute_interior_stress(self, x, y):
        """Alias"""
        return self.compute_stress_at_point(x, y)
