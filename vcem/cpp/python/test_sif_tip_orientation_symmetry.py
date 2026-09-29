"""Regression test for the tip-orientation sign bug in SIF extraction.

A symmetric central crack under pure-shear remote stress must produce
KI and KII at the two tips that are equal in magnitude (Brazilian-disk-like
symmetry, but with a planar crack and a single applied-stress tensor).

Before the tip-aware `ex` fix in reconstruct_cod_csd_panel_midpoints, the
tip at v0 of the edge had its (KI, KII) reported in the *inward* tangent
frame while the tip at v1 was reported in the *outward* frame. Because the
MTS formula is NOT invariant under (KI,KII) -> (-KI,-KII), this fed an
opposite-sign KII to the MTS law for one tip and produced a wrong-sign
kink angle there -- the bottom-branch spiral we saw in propagation.

The fix passes tip_xy to the reconstruction so it picks the panel tangent
at the *tip* end (and negates it if needed). After the fix both tips
report SIFs in their own outward frame, so the magnitudes match and the
KI signs match.
"""

import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "vcem"))

from fracture_utils.Usolver.network import CrackNetworkV4, VertexV4, EdgeV4
from fracture_utils.Usolver.material import Material, AppliedStress
from fracture_utils.Usolver.parametrization import DCENetworkStaticV4
from fracture_utils.Uprocessor.results import DCEResultsNetworkV4
from fracture_utils.Uprocessor.SIF_cod import DisplacementSIF


def build_central_crack():
    L = 1.0e-3                            # 1 mm crack
    net = CrackNetworkV4(
        vertices=[VertexV4(id=0, x=-L/2, y=0.0),
                  VertexV4(id=1, x=+L/2, y=0.0)],
        edges=[EdgeV4(id=0, v0=0, v1=1)],
    )
    material = Material(E=70e9, nu=0.3, plane_stress=False)
    # Mixed mode: vertical tension + shear -> both KI and KII non-zero at both tips
    applied = AppliedStress(sigma_xx=0.0, sigma_yy=1.0e6, sigma_xy=5.0e5)
    return net, material, applied


def solve_and_extract(net, material, applied, *, pass_tip_xy: bool):
    calc = DCENetworkStaticV4(material=material, network=net, applied=applied)
    sol = calc.solve(n_crack_elements=40)
    res = DCEResultsNetworkV4(calc, sol)

    L = 1.0e-3
    tip_left  = np.array([-L/2, 0.0])    # this is at v0 of edge 0 -> bug-affected
    tip_right = np.array([+L/2, 0.0])    # this is at v1 of edge 0 -> already correct

    common = dict(
        edge_index=0,
        a_fit=L,
        material=material,
        rotate=False,
        rmax_frac=0.12,
        min_pts=10,
        two_term=False,
    )

    if pass_tip_xy:
        KI_L, KII_L, *_ = DisplacementSIF.euclid_from_edge(res, tip_xy=tip_left,  **common)
        KI_R, KII_R, *_ = DisplacementSIF.euclid_from_edge(res, tip_xy=tip_right, **common)
    else:
        # Simulate the pre-fix behavior by calling the reconstruction directly
        # with tip_xy=None and re-running the fit with that frame for both tips.
        # We just call euclid_from_edge twice with tip_xy=None... but euclid
        # requires tip_xy to compute r. So we mimic the OLD behavior by passing
        # the tip coordinate to compute r, but the *frame* selection in the
        # reconstruction will use the default (t_edge[-1]) because we no longer
        # forward tip_xy further. Trick: monkey-patch around it.
        from fracture_utils.Uprocessor import SIF_cod as _sm
        orig = _sm.euclid_tip_fit_from_edge
        def _no_tip_frame(res, *, edge_index, tip_xy, a_fit, material=None, **kw):
            # call the underlying reconstruction without tip_xy to mimic pre-fix
            x, COD, CSD, extra = res.reconstruct_cod_csd_panel_midpoints(
                edge_index=int(edge_index), enforce_global_tip_zero=True,
            )
            from fracture_utils.Uprocessor.SIF_cod import sif_from_cod_fit_euclid_arrays
            mat = material if material is not None else getattr(res.calc, "material", None)
            mu = float(getattr(mat, "mu", mat.E / (2.0 * (1.0 + mat.nu))))
            kappa = float(getattr(mat, "kappa", (3.0 - 4.0 * mat.nu)))
            xy_mid = extra["xy_mid"]
            KI, KII, _ = sif_from_cod_fit_euclid_arrays(
                xy_mid=np.asarray(xy_mid, float),
                COD=np.asarray(COD, float),
                CSD=np.asarray(CSD, float),
                p_tip=np.asarray(tip_xy, float),
                a_fit=float(a_fit), mu=mu, kappa=kappa,
                rmax_frac=kw.get("rmax_frac", 0.08),
                min_pts=kw.get("min_pts", 10),
                two_term=kw.get("two_term", False),
            )
            return float(KI), float(KII), float(KI), float(KII), {}
        _sm.euclid_tip_fit_from_edge = _no_tip_frame
        try:
            KI_L, KII_L, *_ = DisplacementSIF.euclid_from_edge(res, tip_xy=tip_left,  **common)
            KI_R, KII_R, *_ = DisplacementSIF.euclid_from_edge(res, tip_xy=tip_right, **common)
        finally:
            _sm.euclid_tip_fit_from_edge = orig

    return (KI_L, KII_L), (KI_R, KII_R)


def main():
    net, material, applied = build_central_crack()

    print("=== Pre-fix behavior (frame = polyline-direction last panel) ===")
    (KIL, KIIL), (KIR, KIIR) = solve_and_extract(net, material, applied, pass_tip_xy=False)
    print(f"  left  tip (v0):  KI = {KIL: .4e}   KII = {KIIL: .4e}")
    print(f"  right tip (v1):  KI = {KIR: .4e}   KII = {KIIR: .4e}")
    print(f"  KI sign match : {np.sign(KIL) == np.sign(KIR)}   "
          f"|KI| ratio = {abs(KIL)/abs(KIR):.6f}")
    print(f"  KII sign match: {np.sign(KIIL) == np.sign(KIIR)}   "
          f"|KII| ratio = {abs(KIIL)/abs(KIIR):.6f}")

    print()
    print("=== Post-fix behavior (frame = tip-side panel, tip_xy passed) ===")
    (KIL, KIIL), (KIR, KIIR) = solve_and_extract(net, material, applied, pass_tip_xy=True)
    print(f"  left  tip (v0):  KI = {KIL: .4e}   KII = {KIIL: .4e}")
    print(f"  right tip (v1):  KI = {KIR: .4e}   KII = {KIIR: .4e}")

    # Physical expectations for a symmetric crack under uniform remote stress:
    # KI same sign at both tips (opening), |KI| equal.
    # KII opposite sign at the two tips (shear is mirror-symmetric across the
    # crack center in the outward-tangent frame), |KII| equal.
    rel_KI  = abs(abs(KIL) - abs(KIR)) / max(abs(KIL), abs(KIR), 1e-30)
    rel_KII = abs(abs(KIIL) - abs(KIIR)) / max(abs(KIIL), abs(KIIR), 1e-30)
    print(f"  |KI|  match within {rel_KI:.2e}    (expect <1e-3)")
    print(f"  |KII| match within {rel_KII:.2e}    (expect <1e-3)")
    print(f"  KI signs match (both opening):    {np.sign(KIL) == np.sign(KIR)}")
    print(f"  KII signs OPPOSITE (mirror shear): {np.sign(KIIL) != np.sign(KIIR)}")

    ok = (rel_KI < 1e-3 and rel_KII < 1e-3
          and np.sign(KIL) == np.sign(KIR)
          and np.sign(KIIL) != np.sign(KIIR))
    print()
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
