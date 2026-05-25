"""
DCE Plotting Module - Version 1 (CLEAN, INDENTATION-SAFE)
========================================================

Drop-in replacement for CrackUtils/dce_plotting_v1.py.

Public API (used by your notebook):
  - DCEPlotterV1
  - plot_element_structure(...)
  - plot_summary(...)
  - plot_stress_fields(...)
  - plot_crack_displacement(...)

Design goals:
  - No indentation / scoping traps.
  - No post-hoc "smoothing of COD to look good".
  - COD/CSD reconstructed from Burgers jumps, sorted left->right, and ANCHORED
    at both ends to remove integration drift (constant/linear modes).

This plotter is compatible with the clean variational solver file:
  dce_calculations_v1_CORRECTED_DOWNLOAD.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


class DCEPlotterV1:
    def __init__(self, calculator, results: dict):
        self.calc = calculator
        self.results = results
        if "dce_elements" not in results:
            raise KeyError("results must contain 'dce_elements'")
        self.dce_elements = results["dce_elements"]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _gather_nodes_and_burgers(self):
        """Return sorted x1 nodes and Burgers jumps (bI,bII) in that order."""
        x1_list, bI_list, bII_list = [], [], []
        for elem in self.dce_elements:
            for k, x1 in enumerate(elem.positions_x1):
                x1_list.append(float(x1))
                bI_list.append(float(elem.b_mode_I[k]))
                bII_list.append(float(elem.b_mode_II[k]))

        x1 = np.asarray(x1_list, dtype=float)
        bI = np.asarray(bI_list, dtype=float)
        bII = np.asarray(bII_list, dtype=float)

        order = np.argsort(x1)
        return x1[order], bI[order], bII[order]

    def calculate_displacement_field(self):
        """
        Compute crack-line positions (GLOBAL) and COD/CSD (LOCAL reconstruction).

        COD(x)  = cumsum(bI)
        CSD(x)  = cumsum(bII)

        Then anchor both COD and CSD so that endpoints are zero (remove constant/linear drift).
        This is NOT smoothing; it is fixing the reference / integration drift.
        """
        x1_sorted, bI_sorted, bII_sorted = self._gather_nodes_and_burgers()

        cod = np.cumsum(bI_sorted)
        csd = np.cumsum(bII_sorted)

        # Anchor endpoints (removes residual constant/linear drift)
        if cod.size >= 2:
            cod = cod - np.linspace(cod[0], cod[-1], cod.size)
        if csd.size >= 2:
            csd = csd - np.linspace(csd[0], csd[-1], csd.size)

        # Map crack-line points to GLOBAL coordinates
        positions_global = np.array(
            [self.calc.local_to_global(np.array([xi, 0.0], dtype=float)) for xi in x1_sorted],
            dtype=float
        )

        return positions_global, cod, csd

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    def plot_element_structure(self, output_path=None):
        fig, ax = plt.subplots(figsize=(12, 5))

        # crack line
        tip_left = self.calc.local_to_global(np.array([-self.calc.a, 0.0]))
        tip_right = self.calc.local_to_global(np.array([+self.calc.a, 0.0]))
        ax.plot([tip_left[0]*1e3, tip_right[0]*1e3],
                [tip_left[1]*1e3, tip_right[1]*1e3],
                'k-', lw=2, label='Crack line')

        # element boundary at x1=0
        center = self.calc.local_to_global(np.array([0.0, 0.0]))
        ax.axvline(center[0]*1e3, color='gray', ls='--', lw=1.5, label='Element boundary')

        colors = ['red', 'blue', 'green', 'purple']
        for eidx, elem in enumerate(self.dce_elements):
            c = colors[eidx % len(colors)]
            pts = np.array([self.calc.local_to_global(np.array([float(xi), 0.0]))
                            for xi in elem.positions_x1], dtype=float)
            ax.scatter(pts[:, 0]*1e3, pts[:, 1]*1e3, s=60, c=c, edgecolors='k',
                       linewidths=0.6, alpha=0.8, label=f'Element {eidx} ({getattr(elem.polarity,"name","")})')

        # tips
        ax.scatter([tip_left[0]*1e3, tip_right[0]*1e3],
                   [tip_left[1]*1e3, tip_right[1]*1e3],
                   s=200, c='gold', edgecolors='orange', linewidths=2, zorder=10, label='Tips')

        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')
        ax.set_title('DCE Element Structure')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.legend(loc='best', fontsize=9)

        plt.tight_layout()
        if output_path is not None:
            fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        return fig

    def plot_summary(self, output_path=None):
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        ax.axis('off')

        K_I = float(self.results.get("K_I", 0.0))
        K_II = float(self.results.get("K_II", 0.0))
        K_Ia = float(self.results.get("K_I_analytical", 0.0))
        K_IIa = float(self.results.get("K_II_analytical", 0.0))
        errI = float(self.results.get("error_I_percent", 0.0))
        errII = float(self.results.get("error_II_percent", 0.0))

        sig_local = self.calc.get_applied_stress_local()

        text = (
            "DCE SUMMARY (v1)\n"
            "============================\n"
            f"Crack length 2a: {self.calc.crack.length*1e3:.3f} mm\n"
            f"Crack angle:     {np.rad2deg(self.calc.crack.angle):.3f} deg\n\n"
            "Applied stress (local):\n"
            f"  sigma_11 = {sig_local[0,0]/1e6:.3f} MPa\n"
            f"  sigma_22 = {sig_local[1,1]/1e6:.3f} MPa\n"
            f"  sigma_12 = {sig_local[0,1]/1e6:.3f} MPa\n\n"
            "SIFs:\n"
            f"  K_I  = {K_I/1e6:.6f} MPa√m   (ana: {K_Ia/1e6:.6f})  err={errI:.3f}%\n"
            f"  K_II = {K_II/1e6:.6f} MPa√m  (ana: {K_IIa/1e6:.6f}) err={errII:.3f}%\n"
        )

        ax.text(0.03, 0.5, text, va='center', ha='left', family='monospace', fontsize=11)
        if output_path is not None:
            fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        return fig

    def plot_crack_displacement(self, scale_factor=None, output_path=None):
        positions_global, cod, csd = self.calculate_displacement_field()

        # scale for visualization
        crack_length = 2.0 * self.calc.a
        max_disp = max(float(np.max(np.abs(cod))), float(np.max(np.abs(csd))), 1e-30)

        if scale_factor is None:
            scale_factor = 0.25 * crack_length / max_disp

        cod_s = cod * scale_factor
        csd_s = csd * scale_factor

        n = self.calc.crack_normal_global
        t = self.calc.crack_direction_global

        upper = positions_global + 0.5*cod_s[:, None]*n + 0.5*csd_s[:, None]*t
        lower = positions_global - 0.5*cod_s[:, None]*n - 0.5*csd_s[:, None]*t

        # undeformed tips
        tip_left = self.calc.local_to_global(np.array([-self.calc.a, 0.0]))
        tip_right = self.calc.local_to_global(np.array([+self.calc.a, 0.0]))

        fig, ax = plt.subplots(figsize=(10, 4))

        ax.plot([tip_left[0]*1e3, tip_right[0]*1e3],
                [tip_left[1]*1e3, tip_right[1]*1e3],
                'k--', lw=2, alpha=0.5, label='Undeformed crack')

        center = self.calc.local_to_global(np.array([0.0, 0.0]))
        ax.axvline(center[0]*1e3, color='gray', ls=':', lw=1.5, alpha=0.7, label='Element boundary')

        poly_x = np.concatenate([upper[:, 0], lower[::-1, 0]]) * 1e3
        poly_y = np.concatenate([upper[:, 1], lower[::-1, 1]]) * 1e3
        ax.fill(poly_x, poly_y, color='lightblue', alpha=0.4, zorder=1)

        ax.plot(upper[:, 0]*1e3, upper[:, 1]*1e3, 'b-', lw=3, label='Upper crack face')
        ax.plot(lower[:, 0]*1e3, lower[:, 1]*1e3, 'r--', lw=3, label='Lower crack face')

        # Close at true tips to avoid "floating" end caps
        ax.plot([tip_left[0]*1e3, tip_left[0]*1e3],
                [upper[0,1]*1e3, lower[0,1]*1e3],
                'k-', lw=3)
        ax.plot([tip_right[0]*1e3, tip_right[0]*1e3],
                [upper[-1,1]*1e3, lower[-1,1]*1e3],
                'k-', lw=3)

        ax.scatter([tip_left[0]*1e3, tip_right[0]*1e3],
                   [tip_left[1]*1e3, tip_right[1]*1e3],
                   s=200, c='gold', edgecolors='orange', linewidths=2, zorder=10, label='Tips')

        title = (f"Mixed-Mode Crack Displacement (scaled {scale_factor:.0f}×)\n"
                 f"θ={np.rad2deg(self.calc.crack.angle):.1f}°, "
                 f"K_I={self.results.get('K_I',0.0)/1e6:.2f} MPa√m, "
                 f"K_II={self.results.get('K_II',0.0)/1e6:.2f} MPa√m")
        ax.set_title(title)
        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
        ax.set_aspect('equal')

        plt.tight_layout()
        if output_path is not None:
            fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        return fig

    def plot_stress_fields(self, x_range=None, y_range=None, grid_size=150, output_path=None):
        """
        Compute and plot sigma_xx, sigma_yy, sigma_xy (GLOBAL) on a grid.
        This is O(N_grid^2 * N_disl); keep grid_size moderate.
        """
        if x_range is None:
            x_range = (-2.0*self.calc.a, 2.0*self.calc.a)
        if y_range is None:
            y_range = (-1.5*self.calc.a, 1.5*self.calc.a)

        xs = np.linspace(x_range[0], x_range[1], grid_size)
        ys = np.linspace(y_range[0], y_range[1], grid_size)
        X, Y = np.meshgrid(xs, ys)

        sxx = np.zeros_like(X)
        syy = np.zeros_like(X)
        sxy = np.zeros_like(X)

        sigma_app_local = self.calc.get_applied_stress_local()
        # Applied stress in global coords for plotting base field
        sigma_app_global = self.calc.stress_local_to_global(sigma_app_local)

        # Gather dislocation data once
        nodes = []
        burgers = []
        for elem in self.dce_elements:
            for k in range(elem.n_dislocations):
                nodes.append(np.array([float(elem.positions_x1[k]), 0.0], dtype=float))
                burgers.append(elem.get_burgers_vector_local(k))

        nodes = np.array(nodes, dtype=float)
        burgers = np.array(burgers, dtype=float)

        for i in range(grid_size):
            for j in range(grid_size):
                p_global = np.array([X[i, j], Y[i, j]], dtype=float)
                p_local = self.calc.global_to_local(p_global)

                # Mask interior crack line for visualization
                if abs(p_local[1]) < 1e-12 and abs(p_local[0]) < self.calc.a:
                    sxx[i, j] = np.nan
                    syy[i, j] = np.nan
                    sxy[i, j] = np.nan
                    continue

                sigma_local = sigma_app_local.copy()
                for pos, b in zip(nodes, burgers):
                    sigma_local += self.calc.stress_from_edge_dislocation(p_local, b, pos)

                sigma_global = self.calc.stress_local_to_global(sigma_local)
                sxx[i, j] = sigma_global[0, 0]
                syy[i, j] = sigma_global[1, 1]
                sxy[i, j] = sigma_global[0, 1]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fields = [sxx, syy, sxy]
        titles = [r'$\sigma_{xx}$ [MPa]', r'$\sigma_{yy}$ [MPa]', r'$\sigma_{xy}$ [MPa]']

        for ax, F, title in zip(axes, fields, titles):
            cf = ax.contourf(X*1e3, Y*1e3, F/1e6, levels=30, cmap='RdBu_r')
            ax.set_title(title)
            ax.set_xlabel('x [mm]')
            ax.set_ylabel('y [mm]')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            plt.colorbar(cf, ax=ax).set_label('MPa')

        plt.tight_layout()
        if output_path is not None:
            fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        return fig
