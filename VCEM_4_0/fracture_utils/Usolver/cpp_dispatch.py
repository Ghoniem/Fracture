"""Thin Python wrappers around the C++ KKT-pipeline functions exposed by
the `bem_cpp` extension. Each wrapper has the same signature as its Python
counterpart in build_discretize / KKT, plus a leading `engine` kwarg that
selects the implementation:

    engine = "python" (default) -> call the Python reference
    engine = "cpp"               -> call the C++ port

If `engine="cpp"` is requested but `bem_cpp` can't be imported, the
wrapper warns once and silently falls back to Python so notebooks never
crash on a missing extension.

Used by fracture_utils.Usolver.parametrization.solve() when the caller
passes `_ignored["engine"]="cpp"`.
"""
from __future__ import annotations

import warnings
from typing import List, Optional

import numpy as np

# Python references kept under aliased names to avoid recursion when the
# dispatchers fall back to them.
from .KKT import solve_kkt_lsq_eq as _py_solve_kkt_lsq_eq
from .build_discretize import (
    assemble_operator                              as _py_assemble_operator,
    assemble_boundary_traction_operator             as _py_assemble_Mt,
    assemble_boundary_displacement_operator         as _py_assemble_Mu,
    assemble_bem_boundary_to_crack_traction_operator as _py_assemble_Nbc,
)


def _try_import_cpp():
    try:
        import bem_cpp  # noqa: F401
        return bem_cpp
    except Exception as exc:                # noqa: BLE001
        warnings.warn(
            f"cpp_dispatch: bem_cpp extension not importable ({exc!r}); "
            "falling back to Python implementations.",
            RuntimeWarning, stacklevel=3)
        return None


_WARNED_FALLBACK = False
def _maybe_cpp(engine: str):
    """Return the bem_cpp module if engine='cpp' and import succeeds, else None."""
    global _WARNED_FALLBACK
    if str(engine).lower().strip() != "cpp":
        return None
    cpp = _try_import_cpp()
    if cpp is None and not _WARNED_FALLBACK:
        _WARNED_FALLBACK = True
    return cpp


def _sigma_at_collocation(poly_panels, applied) -> np.ndarray:
    """Precompute (sxx, syy, sxy) at each collocation point for C++ assemble_operator."""
    cols = np.vstack([p["x_col"] for p in poly_panels])     # (ncol_tot, 2)
    sig  = np.asarray(applied.tensor_at(cols), dtype=float) # (ncol_tot, 2, 2)
    return np.column_stack([sig[:, 0, 0], sig[:, 1, 1], sig[:, 0, 1]])


# ── Assembler wrappers ────────────────────────────────────────────────────
def assemble_operator(*, material, applied, poly_panels, offsets, nunk, nq_col,
                      engine: str = "python"):
    cpp = _maybe_cpp(engine)
    if cpp is None:
        return _py_assemble_operator(material=material, applied=applied,
                                     poly_panels=poly_panels, offsets=list(offsets),
                                     nunk=int(nunk), nq_col=int(nq_col))
    sigma_at_col = _sigma_at_collocation(poly_panels, applied)
    K, rhs = cpp.assemble_operator(
        list(poly_panels), list(map(int, offsets)), int(nunk),
        float(material.E), float(material.nu),
        bool(getattr(material, "plane_stress", False)),
        sigma_at_col)
    return K, rhs


def assemble_boundary_traction_operator(*, material, poly_panels, offsets, nunk,
                                        boundary_xy, boundary_n,
                                        engine: str = "python"):
    cpp = _maybe_cpp(engine)
    if cpp is None:
        return _py_assemble_Mt(material=material, poly_panels=poly_panels,
                                offsets=list(offsets), nunk=int(nunk),
                                boundary_xy=boundary_xy, boundary_n=boundary_n)
    return cpp.assemble_boundary_traction_operator(
        list(poly_panels), list(map(int, offsets)), int(nunk),
        float(material.E), float(material.nu),
        bool(getattr(material, "plane_stress", False)),
        np.asarray(boundary_xy, float),
        np.asarray(boundary_n, float))


def assemble_boundary_displacement_operator(*, material, poly_panels, offsets, nunk,
                                            boundary_xy,
                                            engine: str = "python"):
    cpp = _maybe_cpp(engine)
    if cpp is None:
        return _py_assemble_Mu(material=material, poly_panels=poly_panels,
                                offsets=list(offsets), nunk=int(nunk),
                                boundary_xy=boundary_xy)
    return cpp.assemble_boundary_displacement_operator(
        list(poly_panels), list(map(int, offsets)), int(nunk),
        float(material.nu),
        bool(getattr(material, "plane_stress", False)),
        np.asarray(boundary_xy, float))


def assemble_bem_boundary_to_crack_traction_operator(*, material, poly_panels,
                                                      boundary_x1, boundary_y1,
                                                      boundary_x2, boundary_y2,
                                                      gauss_n: int = 4,
                                                      engine: str = "python"):
    cpp = _maybe_cpp(engine)
    if cpp is None:
        return _py_assemble_Nbc(material=material, poly_panels=poly_panels,
                                 boundary_x1=boundary_x1, boundary_y1=boundary_y1,
                                 boundary_x2=boundary_x2, boundary_y2=boundary_y2,
                                 gauss_n=int(gauss_n))
    return cpp.assemble_bem_boundary_to_crack_traction_operator(
        list(poly_panels),
        float(material.E), float(material.nu),
        bool(getattr(material, "plane_stress", False)),
        np.asarray(boundary_x1, float).reshape(-1),
        np.asarray(boundary_y1, float).reshape(-1),
        np.asarray(boundary_x2, float).reshape(-1),
        np.asarray(boundary_y2, float).reshape(-1),
        int(gauss_n))


def solve_kkt_lsq_eq(K, rhs, C, d: Optional[np.ndarray] = None,
                     ridge: float = 0.0, ridge_diag=None,
                     engine: str = "python"):
    cpp = _maybe_cpp(engine)
    if cpp is None:
        return _py_solve_kkt_lsq_eq(K=K, rhs=rhs, C=C, d=d,
                                     ridge=float(ridge), ridge_diag=ridge_diag)
    n = int(K.shape[1])
    if C is None or C.size == 0:
        C = np.zeros((0, n), dtype=float)
    if d is None:
        d = np.zeros((C.shape[0],), dtype=float)
    rd = (np.ascontiguousarray(ridge_diag, dtype=float).reshape(-1)
          if ridge_diag is not None
          else np.zeros((0,), dtype=float))
    return cpp.solve_kkt_lsq_eq(
        np.ascontiguousarray(K, dtype=float),
        np.ascontiguousarray(rhs, dtype=float).reshape(-1),
        np.ascontiguousarray(C, dtype=float),
        np.ascontiguousarray(d, dtype=float).reshape(-1),
        float(ridge),
        cpp.KKTBackend.AutoLU,
        1e-12,
        rd)


# Convenience: bound version of solve_kkt_lsq (no d, no ridge_diag).
def solve_kkt_lsq(K, rhs, C, ridge: float = 0.0, engine: str = "python"):
    return solve_kkt_lsq_eq(K, rhs, C, d=None, ridge=ridge,
                            ridge_diag=None, engine=engine)
