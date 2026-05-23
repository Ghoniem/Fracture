"""Generate Fig. 2 of the revised manuscript: representative crack-element
discretizations at the coarsest convergence-study resolution N^e_half = 25.

Layout (5 panels, 2 rows):
  Top row:    (a) uniform        (b) tip-dense Chebyshev-Lobatto    (c) singular tip
  Bottom row: (d) inclined crack mesh   (e) branched-crack mesh

Output: crack_discretizations.png next to this script.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# -------- Node distributions ---------------------------------------------------
def nodes_uniform(N: int):
    """N+1 uniform nodes on [-1, 1]."""
    return np.linspace(-1.0, 1.0, N + 1)


def nodes_chebyshev_lobatto(N: int):
    """Chebyshev-Lobatto nodes on [-1, 1] (dense toward tips)."""
    k = np.arange(N + 1)
    return -np.cos(np.pi * k / N)


def singular_weight(xi: np.ndarray) -> np.ndarray:
    """Visual weight 1/sqrt((1-xi)*(1+xi)) clipped near tips, normalized to [0,1]."""
    eps = 1e-3
    w = 1.0 / np.sqrt(np.maximum((1 - xi) * (1 + xi), eps**2))
    return np.clip(w / w.max(), 0.0, 1.0)


# -------- Drawing helpers ------------------------------------------------------
def draw_straight_crack(ax, xi, *, label, show_singular_band=False, tip_pad=0.05):
    """Draw a horizontal crack on [-1, 1] with collocation nodes at xi."""
    a = 1.0
    # crack body
    ax.plot([-a, a], [0, 0], color="0.20", lw=1.6, zorder=2)
    # singular weight shading in (c)
    if show_singular_band:
        xx = np.linspace(-a, a, 600)
        ww = singular_weight(xx)
        for i in range(len(xx) - 1):
            ax.add_patch(Rectangle(
                (xx[i], -0.06), xx[i+1] - xx[i], 0.12,
                color=plt.cm.Reds(0.15 + 0.6 * ww[i]),
                ec="none", zorder=1, alpha=0.55,
            ))
    # collocation nodes (tick marks above the crack)
    for x in xi:
        ax.plot([x, x], [0.02, 0.18], color="tab:blue", lw=1.0, zorder=3)
        ax.plot(x, 0.0, marker="o", ms=2.5, color="tab:blue", zorder=4)
    # tips
    for xt, lab in [(-a, r"tip$_L$"), (a, r"tip$_R$")]:
        ax.plot(xt, 0.0, marker="o", ms=8.0, mfc="white",
                mec="firebrick", mew=2.0, zorder=5)
        ax.text(xt, -0.32, lab, ha="center", va="top", fontsize=16, color="firebrick")
    # axis cosmetics
    ax.set_xlim(-a - tip_pad, a + tip_pad)
    ax.set_ylim(-0.55, 0.55)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(label, fontsize=16)


def draw_inclined_crack(ax, xi_local, beta_deg=30.0, *, label):
    """Draw an inclined crack of half-length 1 at angle beta_deg with nodes at xi_local."""
    beta = np.deg2rad(beta_deg)
    R = np.array([[np.cos(beta), -np.sin(beta)],
                  [np.sin(beta),  np.cos(beta)]])
    a = 1.0
    p0 = R @ np.array([-a, 0.0])
    p1 = R @ np.array([+a, 0.0])
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="0.20", lw=1.6, zorder=2)
    # nodes
    for x in xi_local:
        pn = R @ np.array([x, 0.0])
        # tick perpendicular to crack
        n = R @ np.array([0.0, 0.18])
        ax.plot([pn[0], pn[0] + n[0]], [pn[1], pn[1] + n[1]],
                color="tab:blue", lw=1.0, zorder=3)
        ax.plot(pn[0], pn[1], marker="o", ms=2.5, color="tab:blue", zorder=4)
    # tips
    for pt, lab in [(p0, r"tip$_L$"), (p1, r"tip$_R$")]:
        ax.plot(pt[0], pt[1], marker="o", ms=8.0, mfc="white",
                mec="firebrick", mew=2.0, zorder=5)
    ax.annotate(rf"$\beta = {beta_deg:.0f}^\circ$",
                xy=(0.0, 0.0), xytext=(-0.55, -0.45),
                fontsize=16, color="0.25")
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.0, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(label, fontsize=16)


def draw_branched_crack(ax, xi_local, branch_angle_deg=15.0, *, label):
    """Y-shaped branched crack: main crack along x, single branch from origin."""
    a_main = 1.0
    a_branch = 0.45
    branch = np.deg2rad(branch_angle_deg)

    # main crack (-a_main, 0) -> (0, 0)
    ax.plot([-a_main, 0.0], [0.0, 0.0], color="0.20", lw=1.6, zorder=2)
    # branch (0, 0) -> (a_branch cos, a_branch sin)
    ex, ey = np.cos(branch), np.sin(branch)
    pb = np.array([a_branch * ex, a_branch * ey])
    ax.plot([0.0, pb[0]], [0.0, pb[1]], color="0.20", lw=1.6, zorder=2)

    # nodes on main crack (xi_local maps to physical s along [-a_main, 0])
    for x in xi_local:
        s = 0.5 * (x + 1.0) * a_main - a_main
        ax.plot([s, s], [0.02, 0.18], color="tab:blue", lw=1.0, zorder=3)
        ax.plot(s, 0.0, marker="o", ms=2.5, color="tab:blue", zorder=4)

    # nodes on branch (xi_local maps to physical s along [0, a_branch] tilted)
    n_perp = np.array([-ey, ex])
    for x in xi_local:
        s = 0.5 * (x + 1.0) * a_branch
        pn = s * np.array([ex, ey])
        tip_t = pn + 0.18 * n_perp
        ax.plot([pn[0], tip_t[0]], [pn[1], tip_t[1]],
                color="tab:blue", lw=1.0, zorder=3)
        ax.plot(pn[0], pn[1], marker="o", ms=2.5, color="tab:blue", zorder=4)

    # tips and junction
    ax.plot(-a_main, 0.0, marker="o", ms=8.0, mfc="white",
            mec="firebrick", mew=2.0, zorder=5)
    ax.plot(pb[0], pb[1], marker="o", ms=8.0, mfc="white",
            mec="firebrick", mew=2.0, zorder=5)
    ax.plot(0.0, 0.0, marker="s", ms=8.0, mfc="black",
            mec="black", zorder=5)  # junction
    ax.text(0.0, -0.20, "junction", ha="center", va="top",
            fontsize=16, color="0.20")
    ax.text(pb[0] + 0.05, pb[1] + 0.05, r"branch tip",
            ha="left", va="bottom", fontsize=16, color="firebrick")
    ax.text(-a_main - 0.05, -0.05, r"main tip",
            ha="right", va="top", fontsize=16, color="firebrick")

    ax.set_xlim(-1.35, 0.85)
    ax.set_ylim(-0.6, 0.7)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(label, fontsize=16)


# -------- Figure ---------------------------------------------------------------
def main():
    N = 50  # panels => N+1 = 51 nodes (n_e_half = 25)
    xi_unif = nodes_uniform(N)
    xi_cheb = nodes_chebyshev_lobatto(N)

    fig = plt.figure(figsize=(13, 7.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.2])

    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_b = fig.add_subplot(gs[0, 2:4])
    ax_c = fig.add_subplot(gs[0, 4:6])
    ax_d = fig.add_subplot(gs[1, 0:3])
    ax_e = fig.add_subplot(gs[1, 3:6])

    draw_straight_crack(ax_a, xi_unif,
                        label=r"(a) uniform, $N^{e}_{\mathrm{half}}=25$")
    draw_straight_crack(ax_b, xi_cheb,
                        label=r"(b) tip-dense Chebyshev--Lobatto")
    draw_straight_crack(ax_c, xi_cheb, show_singular_band=True,
                        label=r"(c) singular tip ($1/\sqrt{r}$ embedded)")

    # For the geometry-specific meshes we use the tip-dense Chebyshev nodes
    # (the distribution actually used in Secs. 3.2-3.3).
    # Use a coarser subset for visual clarity on the smaller panels.
    xi_show = nodes_chebyshev_lobatto(30)
    draw_inclined_crack(ax_d, xi_show, beta_deg=30.0,
                        label=r"(d) inclined crack mesh, $\beta=30^{\circ}$")
    draw_branched_crack(ax_e, xi_show, branch_angle_deg=15.0,
                        label=r"(e) branched-crack mesh, $\theta_b=15^{\circ}$")

    out = Path(__file__).resolve().parent / "crack_discretizations.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
