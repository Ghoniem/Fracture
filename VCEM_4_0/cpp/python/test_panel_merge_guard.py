"""Verify the small-panel auto-merge guard in discretize_polylines.

Synthesises a single straight crack with aggressive tip clustering so the
first/last panels are 1e-5 of L (far below the 1e-3 * L default threshold).
Without the guard, the panel-length ratio across the crack is ~1e5, which
squares into the KKT system conditioning. With the guard, those panels
should get merged and the ratio dropped below ~1e3.

Run from the vcem_4_0 conda env:
    python test_panel_merge_guard.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "VCEM_4_0"))

from fracture_utils.Usolver.network import CrackNetworkV4, VertexV4, EdgeV4
from fracture_utils.Usolver.build_discretize import discretize_polylines
from fracture_utils.Usolver.build_dipolar_polyline import build_polylines_full


def make_straight_crack(L: float = 1.0e-3) -> CrackNetworkV4:
    """Single straight crack of length L along the x axis."""
    return CrackNetworkV4(
        vertices=[
            VertexV4(id=0, x=0.0, y=0.0),
            VertexV4(id=1, x=L,   y=0.0),
        ],
        edges=[EdgeV4(id=0, v0=0, v1=1)],
    )


def panel_ds(panels):
    """Get the physical panel lengths array (ds) for the first polyline."""
    return np.asarray(panels[0]["ds"], dtype=float)


def main():
    L = 1.0e-3                      # 1 mm crack
    net = make_straight_crack(L=L)
    polylines, _edge_to_polyline = build_polylines_full(net)

    common = dict(
        n_crack_elements=20,
        node_distribution="uniform",
        collocation_mode="midpoint",
        representation="regular",
        nq_stress=4,
        # Aggressive tip clustering -> generates many decades of panel-length spread
        tip_cluster="power", tip_cluster_power=6.0,
        endpoint_min_nodes=12,
        refine_junction_endpoints=True,
        refine_kinks=False,
    )

    # ── Without guard ───────────────────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        try:
            panels_no_guard = discretize_polylines(
                net, polylines, min_panel_length_ratio=0.0, **common)
        except RuntimeWarning as w:
            # singular-tip warning is from a different path and unrelated
            pass
        panels_no_guard = discretize_polylines(
            net, polylines, min_panel_length_ratio=0.0, **common)
    ds_raw = panel_ds(panels_no_guard)
    ratio_raw = float(ds_raw.max() / ds_raw.min())
    print(f"[no guard]   Np = {ds_raw.size}, "
          f"ds min/max = {ds_raw.min():.3e} / {ds_raw.max():.3e}, "
          f"ratio = {ratio_raw:.2e}, min/L = {ds_raw.min()/L:.3e}")

    # ── With guard at 1e-3 * L ──────────────────────────────────────────
    fired = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        panels_guarded = discretize_polylines(
            net, polylines, min_panel_length_ratio=1.0e-3, **common)
        for w in caught:
            if "sub-threshold panels" in str(w.message):
                fired.append(str(w.message))
    ds_guarded = panel_ds(panels_guarded)
    ratio_guarded = float(ds_guarded.max() / ds_guarded.min())
    print(f"[guard 1e-3] Np = {ds_guarded.size}, "
          f"ds min/max = {ds_guarded.min():.3e} / {ds_guarded.max():.3e}, "
          f"ratio = {ratio_guarded:.2e}, min/L = {ds_guarded.min()/L:.3e}")
    print(f"             warnings fired: {len(fired)}")
    for w in fired:
        print(f"               -> {w}")

    # Assertions
    assert ds_raw.min() / L < 1.0e-3, \
        "Test setup didn't produce sub-threshold panels; tune tip_cluster_power."
    assert ds_guarded.min() / L >= 1.0e-3 - 1e-12, \
        f"Guard failed: smallest panel still {ds_guarded.min()/L:.3e} * L < threshold"
    assert len(fired) >= 1, "Expected a RuntimeWarning to be emitted"
    assert ds_guarded.size < ds_raw.size, \
        f"Guard didn't reduce panel count: {ds_raw.size} -> {ds_guarded.size}"

    print(f"\nPASS  conditioning proxy: ratio dropped {ratio_raw:.2e} -> {ratio_guarded:.2e} "
          f"(K^TK condition ~ ratio^2: {ratio_raw**2:.2e} -> {ratio_guarded**2:.2e})")


if __name__ == "__main__":
    main()
