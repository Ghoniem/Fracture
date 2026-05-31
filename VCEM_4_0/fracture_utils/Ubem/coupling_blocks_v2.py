"""
coupling_blocks_v2.py

Clean block architecture for VCM/BEM coupling:
  - Crack block (existing solver path wrapper)
  - BEM block (external, correction, or both)
  - Mode switch helpers for one-way / full-KKT / occasional full coupling
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from fracture_utils.Ubem.brazilian_disk_bem import (
    BrazilianDiskParams,
    ensure_bem_field,
)
from fracture_utils.Ubem.boundary_conditions import build_boundary, add_boundary_to_solver
from fracture_utils.Ubem.bem_solver import BEMSolver2D
from fracture_utils.Ubem.bem_stress_field import circle_inside
from fracture_utils.Ubem.bem_stress_plotter import (
    ContourOpts,
    eval_and_plot_stress_components_contours_separate,
)
from fracture_utils.Ubem.direct_kkt_coupling import build_disk_traction_augmented_data


def _disk_external_boundary_tractions(mesh, disk: BrazilianDiskParams) -> Tuple[np.ndarray, np.ndarray]:
    theta_deg = np.asarray(mesh.theta_deg, float)
    L = np.asarray(mesh.length, float)
    nx = np.asarray(mesh.nx, float)
    ny = np.asarray(mesh.ny, float)

    arc = float(getattr(disk, "arc_half_angle_deg", 15.0))
    top = (theta_deg >= 90.0 - arc) & (theta_deg <= 90.0 + arc)
    bot = (theta_deg >= -90.0 - arc) & (theta_deg <= -90.0 + arc)

    L_top = float(np.sum(L[top]))
    L_bot = float(np.sum(L[bot]))
    if L_top <= 0.0 or L_bot <= 0.0:
        raise RuntimeError("Top/bottom loaded arc has zero length.")

    h = float(getattr(disk, "h", 1.0))
    p_top = float(disk.P_total) / (L_top * h)
    p_bot = float(disk.P_total) / (L_bot * h)

    tx = np.zeros(mesh.n_seg, float)
    ty = np.zeros(mesh.n_seg, float)
    tx[top] += -p_top * nx[top]
    ty[top] += -p_top * ny[top]
    tx[bot] += -p_bot * nx[bot]
    ty[bot] += -p_bot * ny[bot]
    return tx, ty


@dataclass
class BEMState:
    tx: np.ndarray
    ty: np.ndarray
    xs: np.ndarray
    ys: np.ndarray
    Sxx: np.ndarray
    Syy: np.ndarray
    Sxy: np.ndarray


class BEMBlockV2:
    """
    Independent BEM block with three solve intents:
      - external_only
      - correction_only
      - external_plus_correction
    """

    def __init__(self, disk: BrazilianDiskParams, out_dir: Path):
        self.disk = disk
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        from fracture_utils.Ubem.brazilian_disk_bem import build_disk_boundary
        self.mesh = build_disk_boundary(disk)

        self.tx_ext, self.ty_ext = _disk_external_boundary_tractions(self.mesh, self.disk)
        self.external: Optional[BEMState] = None
        self.correction: Optional[BEMState] = None
        self.total: Optional[BEMState] = None

    def ensure_external_only(self, recompute: bool = False, show: bool = False, save_contours: bool = True) -> BEMState:
        xs, ys, Sxx, Syy, Sxy = ensure_bem_field(
            self.disk,
            self.out_dir,
            recompute=bool(recompute),
            show=bool(show),
            save_contours=bool(save_contours),
        )
        st = BEMState(
            tx=self.tx_ext.copy(),
            ty=self.ty_ext.copy(),
            xs=np.asarray(xs, float).copy(),
            ys=np.asarray(ys, float).copy(),
            Sxx=np.asarray(Sxx, float).copy(),
            Syy=np.asarray(Syy, float).copy(),
            Sxy=np.asarray(Sxy, float).copy(),
        )
        self.external = st
        return st

    def _solve_boundary_tractions(self, tx: np.ndarray, ty: np.ndarray, *, tag: str, show: bool = False) -> BEMState:
        tx = np.asarray(tx, float).reshape(-1)
        ty = np.asarray(ty, float).reshape(-1)
        nb = int(self.mesh.n_seg)
        if tx.size != nb or ty.size != nb:
            raise ValueError("tx/ty size must match boundary element count.")

        solver = BEMSolver2D(
            E=float(self.disk.E),
            nu=float(self.disk.nu),
            h=float(getattr(self.disk, "h", 1.0)),
            plane_strain=True,
        )
        is_traction = np.ones((nb,), bool)
        add_boundary_to_solver(solver, self.mesh, is_traction, tx, ty)
        solver.solve(gauss_n=int(getattr(self.disk, "gauss_n", 4)))

        R = float(self.disk.R)
        bbox = (-R, R, -R, R)
        inside = circle_inside(R, center=(0.0, 0.0), pad=0.03 * R)
        opts = dict(
            dpi=200,
            show=bool(show),
            robust=True,
            robust_pct=97.0,
            n_levels=20,
            n_line_levels=12,
            x_scale=1e3,
            y_scale=1e3,
            x_label="x [mm]",
            y_label="y [mm]",
            value_scale=1e-6,
            cbar_label="Stress [MPa]",
            cmap="jet",
            symmetric=True,
        )
        xs, ys, Sxx, Syy, Sxy, _ = eval_and_plot_stress_components_contours_separate(
            solver,
            bbox=bbox,
            out_dir=self.out_dir,
            basename=f"brazilian_disk_{tag}",
            n=int(getattr(self.disk, "n_grid", 120)),
            inside=inside,
            normalize_by=None,
            opts_xx=ContourOpts(**opts, title="sigma_xx"),
            opts_yy=ContourOpts(**opts, title="sigma_yy"),
            opts_xy=ContourOpts(**opts, title="sigma_xy"),
        )
        return BEMState(
            tx=tx.copy(),
            ty=ty.copy(),
            xs=np.asarray(xs, float).copy(),
            ys=np.asarray(ys, float).copy(),
            Sxx=np.asarray(Sxx, float).copy(),
            Syy=np.asarray(Syy, float).copy(),
            Sxy=np.asarray(Sxy, float).copy(),
        )

    def solve_correction_only(self, tx_corr: np.ndarray, ty_corr: np.ndarray, *, show: bool = False) -> BEMState:
        st = self._solve_boundary_tractions(tx_corr, ty_corr, tag="correction", show=show)
        self.correction = st
        return st

    def solve_external_plus_correction(self, tx_corr: np.ndarray, ty_corr: np.ndarray, *, show: bool = False) -> BEMState:
        tx_tot = self.tx_ext + np.asarray(tx_corr, float).reshape(-1)
        ty_tot = self.ty_ext + np.asarray(ty_corr, float).reshape(-1)
        st = self._solve_boundary_tractions(tx_tot, ty_tot, tag="total", show=show)
        self.total = st
        return st

    def evaluate_field(self, field: str = "external") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        key = str(field).lower().strip()
        if key == "external":
            if self.external is None:
                raise RuntimeError("External field not initialized. Call ensure_external_only().")
            st = self.external
        elif key == "correction":
            if self.correction is None:
                raise RuntimeError("Correction field not available. Call solve_correction_only().")
            st = self.correction
        elif key == "total":
            if self.total is None:
                raise RuntimeError("Total field not available. Call solve_external_plus_correction().")
            st = self.total
        else:
            raise ValueError("field must be one of: external, correction, total")
        return st.xs, st.ys, st.Sxx, st.Syy, st.Sxy

    def build_augmented_payload(self, *, d_mode: str = "correction_zero", gauss_n: int = 4) -> Dict[str, np.ndarray]:
        return build_disk_traction_augmented_data(
            R=float(self.disk.R),
            n_boundary_elements=int(self.disk.n_boundary_elements),
            E=float(self.disk.E),
            nu=float(self.disk.nu),
            P_total=float(self.disk.P_total),
            h=float(self.disk.h),
            arc_half_angle_deg=float(getattr(self.disk, "arc_half_angle_deg", 15.0)),
            plane_strain=True,
            gauss_n=int(gauss_n),
            d_mode=str(d_mode),
            boundary_mesh=str(getattr(self.disk, "boundary_mesh", "uniform")),
            boundary_concentration=float(getattr(self.disk, "boundary_concentration", 8.0)),
            boundary_taper_exponent=float(getattr(self.disk, "boundary_taper_exponent", 4.0)),
        )


class CrackBlockV2:
    """
    Thin helper around solver kwargs for one-way/full coupling switch.
    """

    @staticmethod
    def solver_kwargs_one_way(base: Optional[Dict] = None) -> Dict:
        out = {} if base is None else dict(base)
        out.pop("augmented_coupling", None)
        out.pop("augmented_ridge_q", None)
        out.pop("augmented_ridge_y", None)
        out.pop("augmented_keep_bem_applied", None)
        return out

    @staticmethod
    def solver_kwargs_full_kkt(
        augmented_payload: Dict[str, np.ndarray],
        *,
        base: Optional[Dict] = None,
        augmented_ridge_q: float = 1e-4,
        augmented_ridge_y: float = 1e-10,
        augmented_keep_bem_applied: bool = True,
    ) -> Dict:
        out = {} if base is None else dict(base)
        out["augmented_coupling"] = augmented_payload
        out["augmented_ridge_q"] = float(augmented_ridge_q)
        out["augmented_ridge_y"] = float(augmented_ridge_y)
        out["augmented_keep_bem_applied"] = bool(augmented_keep_bem_applied)
        return out


def compose_solver_kwargs_v2(
    mode: str,
    *,
    base_solver_kwargs: Optional[Dict] = None,
    augmented_payload: Optional[Dict[str, np.ndarray]] = None,
    augmented_ridge_q: float = 1e-4,
    augmented_ridge_y: float = 1e-10,
    augmented_keep_bem_applied: bool = True,
) -> Dict:
    """
    Switch helper:
      - mode='one_way'
      - mode='full_kkt'
      - mode='occasional_full' (caller decides when to pass augmented_payload)
    """
    m = str(mode).lower().strip()
    if m == "one_way":
        return CrackBlockV2.solver_kwargs_one_way(base=base_solver_kwargs)
    if m in ("full_kkt", "occasional_full"):
        if augmented_payload is None:
            # occasional_full can use one-way on cycles without full coupling
            if m == "occasional_full":
                return CrackBlockV2.solver_kwargs_one_way(base=base_solver_kwargs)
            raise ValueError("augmented_payload is required for mode='full_kkt'.")
        return CrackBlockV2.solver_kwargs_full_kkt(
            augmented_payload,
            base=base_solver_kwargs,
            augmented_ridge_q=float(augmented_ridge_q),
            augmented_ridge_y=float(augmented_ridge_y),
            augmented_keep_bem_applied=bool(augmented_keep_bem_applied),
        )
    raise ValueError("mode must be one of: one_way, full_kkt, occasional_full")

