import numpy as np
import math


def element_nodes(a: float, ne: int, node_distribution: str = "uniform") -> np.ndarray:
    """Element boundary nodes for ne elements on [-a, a]."""
    n = ne + 1
    kind = (node_distribution or "uniform").lower()
    if kind == "uniform":
        return np.linspace(-a, a, n)
    if kind == "tip_dense":
        i = np.arange(n, dtype=float)
        s = np.cos(np.pi * i / (n - 1))  # 1..-1
        return a * s[::-1]               # -a..a
    raise ValueError(f"Unknown node_distribution '{node_distribution}'")


def cheb_clenshaw_T(coeffs: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Evaluate sum_k coeffs[k] T_k(s) at s in [-1,1] using Clenshaw."""
    c = np.asarray(coeffs, float)
    s = np.asarray(s, float)
    if c.size == 0:
        return np.zeros_like(s)
    b_kp1 = np.zeros_like(s)
    b_kp2 = np.zeros_like(s)
    for a_k in c[:0:-1]:  # from c[N] down to c[1]
        b_k = 2.0 * s * b_kp1 - b_kp2 + a_k
        b_kp2, b_kp1 = b_kp1, b_k
    return s * b_kp1 - b_kp2 + c[0]


class DCEResultsV2:
    """Post-processing for DCE solver v0.

    Improvements:
      (2a) Auto fit window based on local tip spacing dx_tip.
      (2b) Two-term asymptotic fit: COD ~ A*sqrt(r) + B*r^(3/2).
      (3) Optional COD_max-based K estimator (robust, cheap cross-check).

    Notes:
      - No analytical LEFM/Williams stress fields are used.
      - COD/CSD reconstruction uses a singularity-aware theta-integration of the regular function
        g(x) = b(x)*sqrt(a^2-x^2).
    """

    def __init__(self, calc, sol: dict):
        self.calc = calc
        self.sol = sol
        self.rep = str(sol.get("representation", "")).lower()
        self.node_distribution = str(sol.get("node_distribution", "uniform")).lower()

    # ---------- evaluation helpers ----------
    def g_regular(self, x: np.ndarray, mode: str = "I") -> np.ndarray:
        """Return g(x)=b(x)*sqrt(a^2-x^2), which is regular at the tips."""
        a = float(self.calc.crack.half_length)
        x = np.asarray(x, float)

        if self.rep == "delta_collocation":
            xs = np.asarray(self.sol["xs"], float)
            key_g = "gI" if mode.upper() == "I" else "gII"
            if key_g in self.sol:
                g_nodes = np.asarray(self.sol[key_g], float)
                return np.interp(x, xs, g_nodes, left=0.0, right=0.0)
            # Backward-compatibility: if only bI/bII provided, build g=b*sqrt(a^2-x^2)
            b_nodes = np.asarray(self.sol["bI" if mode.upper() == "I" else "bII"], float)
            b = np.interp(x, xs, b_nodes, left=0.0, right=0.0)
            return b * np.sqrt(np.maximum(a * a - x * x, 0.0))

        if self.rep == "cheb_quad":
            # We store regular g nodal values qI/qII on xnodes.
            xnodes = np.asarray(self.sol["xnodes"], float)
            q = np.asarray(self.sol["qI" if mode.upper() == "I" else "qII"], float)
            return np.interp(x, xnodes, q, left=q[0], right=q[-1])

        if self.rep == "cheb_spectral":
            s = np.clip(np.asarray(x, float) / a, -1.0, 1.0)
            coeffs = np.asarray(self.sol["coeffs_I" if mode.upper() == "I" else "coeffs_II"], float)
            return cheb_clenshaw_T(coeffs, s)

        raise ValueError(f"Unknown representation '{self.rep}' in results.")

    def b_mode(self, x: np.ndarray, mode: str = "I") -> np.ndarray:
        """Evaluate b(x) for a given mode."""
        a = float(self.calc.crack.half_length)
        x = np.asarray(x, float)
        eps = max(1e-12 * a, 1e-18)
        xx = np.clip(x, -a + eps, a - eps)
        g = self.g_regular(xx, mode=mode)
        return g / np.sqrt(np.maximum(a * a - xx * xx, eps))

    # ---------- COD/CSD reconstruction ----------
    def reconstruct_cod_csd(self, n_theta: int = 8000, enforce_tip_zero: bool = True):
        """Reconstruct COD and CSD using x=a cos(theta) and regular g(theta) integration."""
        a = float(self.calc.crack.half_length)
        eps = max(1e-12 * a, 1e-18)
        theta = np.linspace(eps, math.pi - eps, int(n_theta))
        x = a * np.cos(theta)

        gI = self.g_regular(x, mode="I")
        gII = self.g_regular(x, mode="II")

        dtheta = theta[1] - theta[0]
        # COD(x) = - \int_{theta0}^{theta(x)} gI(theta) dtheta  (up to affine gauge)
        COD = -np.cumsum(gI) * dtheta
        CSD = -np.cumsum(gII) * dtheta

        if enforce_tip_zero:
            # Remove affine gauge so COD(-a)=COD(a)=0 (and similarly for CSD)
            def _fix(y):
                y = y - y[0]
                slope = y[-1] / (x[-1] - x[0])
                return y - slope * (x - x[0])

            COD = _fix(COD)
            CSD = _fix(CSD)

        return x, COD, CSD

    # ---------- SIF extraction helpers ----------
    def _tip_dx(self, tip: str = "right") -> float:
        """Estimate local node spacing near a crack tip, used to auto-select a robust fit window."""
        a = float(self.calc.crack.half_length)
        tip_l = tip.lower()

        if self.rep == "delta_collocation":
            xs = np.asarray(self.sol["xs"], float)
            xs = np.sort(xs)
            if xs.size < 2:
                return 0.05 * a
            if tip_l == "right":
                return float(xs[-1] - xs[-2])
            return float(xs[1] - xs[0])

        if self.rep == "cheb_quad":
            xnodes = np.asarray(self.sol["xnodes"], float)
            xnodes = np.sort(xnodes)
            if xnodes.size < 2:
                return 0.05 * a
            if tip_l == "right":
                return float(xnodes[-1] - xnodes[-2])
            return float(xnodes[1] - xnodes[0])

        if self.rep == "cheb_spectral":
            # No physical nodal spacing; approximate effective tip resolution from N.
            N = int(self.sol.get("N", max(8, 2 * int(self.sol.get("ne_half", 10)))))
            # Chebyshev spacing near ends scales like O(a/N^2)
            return float(max(1e-12 * a, (math.pi * math.pi / (2.0 * N * N)) * a))

        return 0.05 * a

    def _auto_window(self, tip: str, rho_min: float, rho_max: float, c1: float, c2: float):
        a = float(self.calc.crack.half_length)
        dx_tip = self._tip_dx(tip=tip)
        # Keep away from near-tip numerical boundary layer
        r_min = max(float(rho_min) * a, float(c1) * dx_tip)
        # Avoid drifting too far into global regime; cap by geometric mean scale
        r_max = min(float(rho_max) * a, float(c2) * math.sqrt(max(a * dx_tip, 1e-30)))
        # Ensure ordering
        if r_max <= 1.05 * r_min:
            r_max = 2.5 * r_min
        return r_min, r_max, dx_tip

    # ---------- SIF extraction ----------
    def sif_from_cod_fit(
        self,
        tip: str = "right",
        # Window control (either fixed in rho or auto based on dx_tip)
        window: str = "auto",  # "auto" or "fixed"
        rho_min: float = 5e-4,
        rho_max: float = 5e-2,
        c1: float = 10.0,
        c2: float = 0.20,
        # Fit settings
        n_fit: int = 120,
        n_theta: int = 20000,
        two_term: bool = True,
    ):
        """Compute (K_I, K_II) from COD/CSD near-tip fit.

        Asymptotic relations (isotropic, mixed mode):
          COD(r) ≈ ((kappa+1)/mu) * K_I * sqrt(r/(2π))   + O(r^(3/2))
          CSD(r) ≈ ((kappa+1)/mu) * K_II* sqrt(r/(2π))   + O(r^(3/2))

        If two_term=True, fits COD(r) = A*phi + B*phi*rbar where
          phi = sqrt(r/(2π)),  rbar = r/(2π)
        and returns K from A.

        Window:
          - window="fixed": use rho_min, rho_max directly.
          - window="auto":  use dx_tip to keep r_min away from the boundary layer.
        """
        a = float(self.calc.crack.half_length)
        mu = float(self.calc.material.mu)
        kappa = float(self.calc.material.kappa)

        x, COD, CSD = self.reconstruct_cod_csd(n_theta=n_theta, enforce_tip_zero=True)

        tip_l = tip.lower()
        if tip_l == "right":
            r = a - x
        elif tip_l == "left":
            r = a + x
        else:
            raise ValueError("tip must be 'left' or 'right'")

        if str(window).lower() == "auto":
            r_min, r_max, dx_tip = self._auto_window(tip=tip_l, rho_min=rho_min, rho_max=rho_max, c1=c1, c2=c2)
        else:
            r_min = float(rho_min) * a
            r_max = float(rho_max) * a
            dx_tip = self._tip_dx(tip=tip_l)

        mask = (r > r_min) & (r < r_max)
        rr = r[mask]
        CODw = COD[mask]
        CSDw = CSD[mask]

        if rr.size < 8:
            raise ValueError(
                "Fit window too small; increase rho_max or n_theta, or reduce c1 for auto-window."
            )

        # Use the closest n_fit points to the tip within the window (small rr)
        order = np.argsort(rr)
        rr = rr[order]
        CODw = CODw[order]
        CSDw = CSDw[order]

        rr = rr[: int(n_fit)]
        CODw = CODw[: int(n_fit)]
        CSDw = CSDw[: int(n_fit)]

        phi = np.sqrt(rr / (2.0 * np.pi))

        if two_term:
            rbar = rr / (2.0 * np.pi)
            X = np.column_stack([phi, phi * rbar])
            # least squares
            AI, BI = np.linalg.lstsq(X, CODw, rcond=None)[0]
            AII, BII = np.linalg.lstsq(X, CSDw, rcond=None)[0]
            KI = mu / (kappa + 1.0) * float(AI)
            KII = mu / (kappa + 1.0) * float(AII)
        else:
            AI = float(np.dot(phi, CODw) / np.dot(phi, phi))
            AII = float(np.dot(phi, CSDw) / np.dot(phi, phi))
            BI = 0.0
            BII = 0.0
            KI = mu / (kappa + 1.0) * AI
            KII = mu / (kappa + 1.0) * AII

        meta = dict(
            window=str(window).lower(),
            r_min=float(r_min),
            r_max=float(r_max),
            dx_tip=float(dx_tip),
            n_fit=int(len(rr)),
            two_term=bool(two_term),
            A_I=float(AI),
            B_I=float(BI),
            A_II=float(AII),
            B_II=float(BII),
        )
        return KI, KII, meta

    def sif_from_cod_max(self, n_theta: int = 20000):
        """Estimate (K_I, K_II) using max COD and max |CSD| (global method).

        For a center crack in an infinite medium, the analytical COD profile scales as
          COD_max = 4*sigma*(1-ν^2)/E * a  (plane strain)
        and K_I = σ*sqrt(πa). Eliminating σ gives:
          K_I = (E'/2) * COD_max / a
        where E' = E/(1-ν^2) (plane strain) or E (plane stress).

        We use the same scaling for a numerical estimate:
          K_I ≈ (E'/2) * COD_max / a
          K_II ≈ (E'/2) * CSD_peak / a

        This is robust and cheap, and useful as a cross-check.
        """
        a = float(self.calc.crack.half_length)
        mat = self.calc.material
        E = float(mat.E)
        nu = float(mat.nu)
        plane_stress = bool(getattr(mat, "plane_stress", False))
        Eprime = E if plane_stress else E / (1.0 - nu * nu)

        x, COD, CSD = self.reconstruct_cod_csd(n_theta=n_theta, enforce_tip_zero=True)
        COD_max = float(np.max(COD) - np.min(COD))
        # For pure mode I, COD is nonnegative and min ~ 0; above is safe for general sign conventions.
        CSD_peak = float(np.max(np.abs(CSD)))

        KI = 0.5 * Eprime * COD_max / a
        KII = 0.5 * Eprime * CSD_peak / a

        meta = dict(Eprime=Eprime, COD_max=COD_max, CSD_peak=CSD_peak, n_theta=int(n_theta))
        return KI, KII, meta

    # ---------- analytical reference ----------
    @staticmethod
    def KI_tada_infinite(sigma: float, a: float) -> float:
        """Infinite medium, center crack: K_I = sigma*sqrt(pi*a)."""
        return float(sigma) * math.sqrt(math.pi * float(a))
