"""Regression test for the kinked-polyline panel-allocation asymmetry bug.

A mirror-symmetric Z-shaped polyline (the typical post-step-1 state of a
propagating central crack) MUST produce identical SIF magnitudes at the two
tips when discretized and solved by the BEM.

Before this fix, two coupled bugs in fracture_utils.Usolver.build_mesh
silently broke mirror symmetry of the panel layout on every multi-kink
polyline:

  (1) identify_refinement_segments stored cluster_target as a flat
      Dict[int,str]; a segment bracketed by kinks on both sides (interior
      segment of a kinked polyline) saw the second kink overwrite the first
      and the segment then clustered toward only one kink.

  (2) build_s_nodes_for_polyline checked cluster_target *before* the
      endpoint-refinement branch, so a tip-adjacent segment whose interior
      side also touched a kink lost its tip-clustering entirely -- the BEM
      couldn't resolve the SIF at that tip.

  (3) allocate_panels_per_segment distributed the remainder R = Np - sum(mins)
      onto the first r segments in iteration order, which biased every
      panel-count remainder toward low-indexed segments and prevented a
      palindromic segL from producing a palindromic Nseg.

After the fix the panel layout is palindromic to machine epsilon and the
SIF magnitudes match across the two tips.
"""

import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "VCEM_4_0"))

from fracture_utils.Usolver.network import CrackNetworkV4, VertexV4, EdgeV4
from fracture_utils.Usolver.material import Material, AppliedStress
from fracture_utils.Usolver.parametrization import DCENetworkStaticV4
from fracture_utils.Uprocessor.results import DCEResultsNetworkV4
from fracture_utils.Upropagation.network_ops import get_netops
from fracture_utils.Upropagation.evaluate import CandidateEvaluator
from fracture_utils.Upropagation.direction import MaximumHoopStressLaw


def build_zshape_network():
    """Mirror-symmetric Z-shape: tip3-(v1)-(v0=O)-(v2)-tip4."""
    mm = 1e-3
    da = 0.635e-3
    v0 = (0.0, 0.0)
    v1 = (+2.0 * mm, +2.0 * mm)
    v2 = (-2.0 * mm, -2.0 * mm)
    th = math.radians(126.494)
    v3 = (v1[0] + da * math.cos(th), v1[1] + da * math.sin(th))
    v4 = (v2[0] - da * math.cos(th), v2[1] - da * math.sin(th))
    net = CrackNetworkV4(
        vertices=[
            VertexV4(id=0, x=v0[0], y=v0[1]),
            VertexV4(id=1, x=v1[0], y=v1[1]),
            VertexV4(id=2, x=v2[0], y=v2[1]),
            VertexV4(id=3, x=v3[0], y=v3[1]),
            VertexV4(id=4, x=v4[0], y=v4[1]),
        ],
        edges=[
            EdgeV4(id=0, v0=0, v1=1),
            EdgeV4(id=1, v0=0, v1=2),
            EdgeV4(id=2, v0=1, v1=3),
            EdgeV4(id=3, v0=2, v1=4),
        ],
    )
    return net


def main():
    net = build_zshape_network()

    # Symmetric biaxial-ish stress (any mirror-symmetric loading works).
    material = Material(E=200e9, nu=0.30, plane_stress=False)
    applied = AppliedStress(sigma_xx=1.0e7, sigma_yy=-3.0e7, sigma_xy=0.0)

    solver_kwargs = dict(
        n_crack_elements=100,
        representation="cheb_quad",
        node_distribution="tip_dense",
        solver_option="parametrized_crack",
        parametrization="polyline",
        nq_col=12,
        nq_stress=16,
        crack_mode="half",
        junction_model="core",
    )

    calc = DCENetworkStaticV4(material=material, network=net, applied=applied)
    sol = calc.solve(**solver_kwargs)
    res = DCEResultsNetworkV4(calc, sol)

    # ---- (1) Panel layout must be palindromic about Ltot/2 ----
    poly_p = sol["parametrized_polylines"][0]
    poly_s = sol["polyline_solutions"][0]
    s_nodes = np.asarray(poly_s["s_nodes"], float)
    ds = np.asarray(poly_s["ds"], float)
    Ltot = float(poly_p["total_length"])

    s_mirror_resid = float(np.max(np.abs((s_nodes + s_nodes[::-1]) - Ltot)))
    ds_mirror_resid = float(np.max(np.abs(ds - ds[::-1])))

    print(f"s_nodes mirror residual: {s_mirror_resid:.3e}  (Ltot={Ltot:.3e})")
    print(f"ds       mirror residual: {ds_mirror_resid:.3e}")
    assert s_mirror_resid < 1.0e-12 * Ltot, (
        f"s_nodes not mirror-symmetric: residual {s_mirror_resid:.3e} >= "
        f"{1.0e-12 * Ltot:.3e}"
    )
    assert ds_mirror_resid < 1.0e-12 * Ltot, (
        f"ds not mirror-symmetric: residual {ds_mirror_resid:.3e} >= "
        f"{1.0e-12 * Ltot:.3e}"
    )

    # ---- (2) SIFs at the two tips must match (KI same, KII opposite) ----
    evaluator = CandidateEvaluator(
        material=material, applied=applied,
        solver_kwargs=solver_kwargs,
        direction_law=MaximumHoopStressLaw(),
        enable_n_crack_elements_escalation=False,
    )
    netops = get_netops()
    deg = netops.degree_map(net)
    polylines = netops.extract_open_polylines(net)
    tips = netops.extract_deg1_tips(net, polylines, deg)

    KI = {}
    KII = {}
    for t in tips:
        tev = evaluator.eval_tip(res, t)
        KI[int(t.v_tip)] = float(tev.KI)
        KII[int(t.v_tip)] = float(tev.KII)

    print(f"vid=3 (top):    KI = {KI[3]: .6e}    KII = {KII[3]: .6e}")
    print(f"vid=4 (bottom): KI = {KI[4]: .6e}    KII = {KII[4]: .6e}")

    rel_KI = abs(abs(KI[3]) - abs(KI[4])) / max(abs(KI[3]), abs(KI[4]), 1e-30)
    rel_KII = abs(abs(KII[3]) - abs(KII[4])) / max(abs(KII[3]), abs(KII[4]), 1e-30)
    print(f"|KI|  match within {rel_KI:.2e}")
    print(f"|KII| match within {rel_KII:.2e}")

    # Tolerance: applied AppliedStress(sigma_*) is uniform & exactly mirror-
    # symmetric, so any residual asymmetry is purely numerical (panel-
    # midpoint accumulation, quadrature, KKT solver tolerance). 1e-3 is a
    # comfortable bound that still catches regressions.
    assert rel_KI < 1.0e-3, f"|KI| asymmetric: rel={rel_KI:.3e}"
    assert rel_KII < 1.0e-3, f"|KII| asymmetric: rel={rel_KII:.3e}"
    assert np.sign(KI[3]) == np.sign(KI[4]), "KI signs must match at both tips"
    assert np.sign(KII[3]) != np.sign(KII[4]), "KII signs must be opposite (mirror shear)"

    print("PASS")


if __name__ == "__main__":
    main()
