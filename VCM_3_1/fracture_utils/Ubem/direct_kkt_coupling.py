"""
direct_kkt_coupling.py

Utilities for true augmented KKT coupling between VCM crack unknowns and BEM
boundary unknowns on a fixed outer boundary.

Current scope:
- traction-boundary coupling (Gamma_t): enforce t_bc + t_cr(q) = t_ext
- full BEM equation block: H u_bc - G t_bc = 0
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from fracture_utils.Ubem.boundary_conditions import build_boundary
from fracture_utils.Ubem.bem_solver import (
    gauss_legendre,
    kelvin_T,
    kelvin_U,
    map_to_segment,
    Segment,
)


def build_bem_dense_operators(
    *,
    segments: list[Segment],
    E: float,
    nu: float,
    plane_strain: bool = True,
    gauss_n: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build dense (C+H) and G operators for constant-element 2D BEM:

        sum_j ((C+H)_ij u_j - G_ij t_j) = 0

    with row-sum closure for H diagonal and c=1/2 jump term added back.
    """
    N = len(segments)
    if N <= 0:
        raise ValueError("At least one boundary segment is required.")

    xg, wg = gauss_legendre(int(gauss_n))
    H_off = np.zeros((2 * N, 2 * N), float)
    G = np.zeros((2 * N, 2 * N), float)

    for i, si in enumerate(segments):
        xi, yi = float(si.xm), float(si.ym)
        rows = slice(2 * i, 2 * i + 2)

        for j, sj in enumerate(segments):
            cols = slice(2 * j, 2 * j + 2)
            Gij = np.zeros((2, 2), float)
            Hij = np.zeros((2, 2), float)
            rf = 1e-16 * max(float(sj.length), 1e-12)

            for s, w in zip(xg, wg):
                xq, yq, jac = map_to_segment(sj, float(s))
                U = kelvin_U((xi, yi), (xq, yq), E=float(E), nu=float(nu), plane_strain=bool(plane_strain), r_floor=rf)
                Gij += U * (float(w) * float(jac))

                if i != j:
                    T = kelvin_T(
                        field=(xi, yi),
                        source=(xq, yq),
                        n_source=(float(sj.nx), float(sj.ny)),
                        E=float(E),
                        nu=float(nu),
                        plane_strain=bool(plane_strain),
                        r_floor=rf,
                    )
                    Hij += T * (float(w) * float(jac))

            G[rows, cols] = Gij
            if i != j:
                H_off[rows, cols] = Hij

    # Row-sum closure with c=1/2 on smooth boundary
    c = 0.5
    H = H_off.copy()
    for i in range(N):
        rows = slice(2 * i, 2 * i + 2)
        row_sum = np.zeros((2, 2), float)
        for j in range(N):
            if i == j:
                continue
            cols = slice(2 * j, 2 * j + 2)
            row_sum += H_off[rows, cols]
        H[rows, rows] = -c * np.eye(2) - row_sum

    CplusH = H.copy()
    for i in range(N):
        rows = slice(2 * i, 2 * i + 2)
        CplusH[rows, rows] += c * np.eye(2)

    return CplusH, G


def _disk_external_boundary_tractions(mesh, *, P_total: float, arc_half_angle_deg: float, h: float) -> Tuple[np.ndarray, np.ndarray]:
    theta_deg = np.asarray(mesh.theta_deg, float)
    L = np.asarray(mesh.length, float)
    nx = np.asarray(mesh.nx, float)
    ny = np.asarray(mesh.ny, float)

    arc = float(arc_half_angle_deg)
    top = (theta_deg >= 90.0 - arc) & (theta_deg <= 90.0 + arc)
    bot = (theta_deg >= -90.0 - arc) & (theta_deg <= -90.0 + arc)

    L_top = float(np.sum(L[top]))
    L_bot = float(np.sum(L[bot]))
    if L_top <= 0.0 or L_bot <= 0.0:
        raise RuntimeError("Top/bottom loaded arc has zero length.")

    p_top = float(P_total) / (L_top * float(h))
    p_bot = float(P_total) / (L_bot * float(h))

    tx = np.zeros(mesh.n_seg, float)
    ty = np.zeros(mesh.n_seg, float)
    tx[top] += -p_top * nx[top]
    ty[top] += -p_top * ny[top]
    tx[bot] += -p_bot * nx[bot]
    ty[bot] += -p_bot * ny[bot]
    return tx, ty


def build_disk_traction_augmented_data(
    *,
    R: float,
    n_elem: int,
    E: float,
    nu: float,
    P_total: float,
    h: float,
    arc_half_angle_deg: float = 15.0,
    plane_strain: bool = True,
    gauss_n: int = 4,
    d_mode: str = "correction_zero",
) -> Dict[str, np.ndarray]:
    """
    Build fixed augmented-coupling data for a Brazilian disk with traction-only
    boundary conditions.

    Returned dict is suitable for DCENetworkStaticV4.solve(..., augmented_coupling=...).

    d_mode:
      - "correction_zero" (recommended): enforce t_bc + t_cr(q) = 0 on boundary.
        Use this when crack solve already includes applied/BEM field.
      - "external_total": enforce t_bc + t_cr(q) = t_ext.
    """
    mesh = build_boundary({"type": "circle", "R": float(R), "n_boundary": int(n_elem), "center": (0.0, 0.0)})

    segs: list[Segment] = []
    for i in range(mesh.n_seg):
        segs.append(
            Segment(
                x1=float(mesh.x1[i]),
                y1=float(mesh.y1[i]),
                x2=float(mesh.x2[i]),
                y2=float(mesh.y2[i]),
                is_traction=True,
                bc_x=0.0,
                bc_y=0.0,
            )
        )

    CplusH, G = build_bem_dense_operators(
        segments=segs,
        E=float(E),
        nu=float(nu),
        plane_strain=bool(plane_strain),
        gauss_n=int(gauss_n),
    )

    nb = int(mesh.n_seg)
    # y = [u_bc(2nb), t_bc(2nb)]
    C_bem = np.hstack([CplusH, -G])  # (2nb, 4nb)
    C_bc = np.zeros((2 * nb, 4 * nb), float)
    C_bc[:, 2 * nb:] = np.eye(2 * nb)

    tx_ext, ty_ext = _disk_external_boundary_tractions(
        mesh,
        P_total=float(P_total),
        arc_half_angle_deg=float(arc_half_angle_deg),
        h=float(h),
    )
    d_mode = str(d_mode).lower().strip()
    if d_mode == "external_total":
        d = np.empty((2 * nb,), float)
        d[0::2] = tx_ext
        d[1::2] = ty_ext
    elif d_mode == "correction_zero":
        d = np.zeros((2 * nb,), float)
    else:
        raise ValueError("d_mode must be 'correction_zero' or 'external_total'.")

    xy = np.column_stack([np.asarray(mesh.xm, float), np.asarray(mesh.ym, float)])
    nn = np.column_stack([np.asarray(mesh.nx, float), np.asarray(mesh.ny, float)])

    return {
        "type": "bem_traction_only",
        "boundary_xy": xy,
        "boundary_n": nn,
        "boundary_x1": np.asarray(mesh.x1, float).copy(),
        "boundary_y1": np.asarray(mesh.y1, float).copy(),
        "boundary_x2": np.asarray(mesh.x2, float).copy(),
        "boundary_y2": np.asarray(mesh.y2, float).copy(),
        "C_bem": C_bem,
        "C_bc": C_bc,
        "d": d,
        "d_mode": np.array([d_mode], dtype=object),
        "tx_ext": tx_ext,
        "ty_ext": ty_ext,
        "n_boundary": np.array([nb], int),
    }
