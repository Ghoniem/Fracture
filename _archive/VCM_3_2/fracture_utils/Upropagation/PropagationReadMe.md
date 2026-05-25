Below is a **package architecture** for `Upropagation/` that keeps modules small, responsibilities crisp, and aligns with your existing ecosystem (`Usolver`, `Ugenerator`, `Uprocessor.SIF_cod`, etc.). We want **object-oriented orchestration** with a clean “driver” class and pluggable policies (direction, toughness, step-size, remeshing, caching).

---

## `fracture_utils/Upropagation/` package layout

```
fracture_utils/
  Upropagation/
    __init__.py                 # empty (per your requirement)

    config.py                   # dataclasses: tolerances, defaults, run options
    context.py                  # shared objects: solver hooks, caches, logging
    toughness.py                # fracture toughness field interface + implementations
    tip_state.py                # tip identification + local geometry (tangent, arclength)
    direction.py                # propagation direction law (MTS, etc.)
    step_control.py             # adaptive f selection logic (your f,2f,0.5f rules)
    geometry_update.py          # append segment, build candidate geometry objects
    remesh.py                   # local remesh policies (tip neighborhood only)
    evaluate.py                 # “evaluate candidate”: solve + SIF + theta extraction
    propagator.py               # high-level orchestrator (one growth increment)
    results.py                  # result objects: accepted tips, steps, diagnostics
```

Each module stays manageable: one “concept” per file, minimal cross-imports, and clear boundaries.

---

## Core data model (what each class owns)

### 1) `config.py`

**Responsibility:** pure configuration (no solver calls).

* `PropagationConfig`

  * `f0: float = 0.1`
  * `f_min: float = 0.005`
  * `f_max: float = 0.25`
  * `tol_theta_rel: float = 0.10`
  * `tol_theta_abs: float = deg2rad(1.0)` (important when theta≈0)
  * `tol_keff_rel: float = 0.10` (recommended)
  * `prefer_larger_step: bool = True` (your “accept 2f if stable”)
  * `max_trials: int = 6`
  * `simultaneous_tip_growth: bool = True`
  * `local_remesh: bool = True`
  * plus anything you already use (node_distribution, representation, etc.)

### 2) `toughness.py`

**Responsibility:** provide `Kc(x,y)` cleanly, including spatial fields.

* `class ToughnessField(Protocol)` with:

  * `Kc(self, x: float, y: float) -> float`
* Implementations:

  * `ConstantToughness(Kc0)`
  * `CallableToughness(func)` (wrap any `Kc(x,y)` callable)
  * `GridToughness(grid, interp)` later if needed

### 3) `tip_state.py`

**Responsibility:** identify tips and compute local geometric quantities.

* `TipID`: `(vertex_id, which_end, edge_id)` or whatever is stable in your graph.
* `TipState`:

  * `tip_id`
  * `x_tip, y_tip`
  * `tangent_unit` (smoothed from last 1–3 segments)
  * `total_crack_length`
  * optional: `curvature_proxy`, `last_segment_length`, etc.

### 4) `direction.py`

**Responsibility:** direction law from `(KI,KII)` → `theta`.

* `class DirectionLaw(Protocol)`:

  * `theta(self, KI: float, KII: float) -> float`
* Implement:

  * `MaximumHoopStressLaw` (MTS)
  * `MaxEnergyReleaseLaw` (optional)
  * Your own later

### 5) `evaluate.py`

**Responsibility:** given a candidate geometry, compute tip SIFs and angles by calling your existing pipeline.

This is where you plug in:

* solver build/solve (from `Usolver`)
* COD reconstruction (from results)
* `Uprocessor.SIF_cod.sif_from_cod_fit_euclid_arrays`

Key class:

* `class CandidateEvaluator:`

  * `solve(self, geom) -> SolverResults`  (wrap Usolver build+solve)
  * `sif_at_tip(self, res, tip_state) -> (KI, KII, Keff, fit_meta)`
  * `theta_at_tip(self, KI, KII) -> theta` (via `DirectionLaw`)
  * returns a `CandidateEval` object containing per-tip metrics + any diagnostics.

This module is where you can later add **warm-start**, **matrix reuse**, **local updates**, etc., without touching propagation logic.

### 6) `step_control.py`

**Responsibility:** your adaptive `f` logic and acceptance tests.

* `class StepController:`

  * `choose_step(self, base_geom, tip_state, base_eval) -> StepDecision`

Where `StepDecision` includes:

* accepted `f`
* accepted `Δa`
* rejected trial records (for debug/plots)
* reason codes (angle unstable / Keff unstable / hit f_min / etc.)

Implements exactly your flow, but structured:

* try `f`
* try `2f` (if prefer larger)
* compare angles (+ Keff recommended)
* if unstable → halve `f`
* stop when accepted or `f_min` reached

### 7) `geometry_update.py`

**Responsibility:** create new geometry objects by extending a tip.

* `class GeometryUpdater:`

  * `extend_tip(self, geom, tip_state, theta, Δa) -> new_geom`
  * uses `phi = tangent_angle + theta`
  * appends one segment (polyline) at that tip
  * handles indexing/orientation cleanly

### 8) `remesh.py`

**Responsibility:** re-mesh policy (local vs global, tip neighborhood).

* `class RemeshPolicy(Protocol):`

  * `remesh(self, geom, tip_state, Δa) -> geom_remeshed`
* Implement:

  * `GlobalRemeshPolicy` (baseline)
  * `LocalTipRemeshPolicy(neighborhood_factor=4.0)`
    Keeps most panels unchanged; remesh only within arclength `~ neighborhood_factor * Δa`.

### 9) `propagator.py`

**Responsibility:** the “driver” that runs one growth increment (and later loops).

* `class CrackPropagator:`

  * `__init__(evaluator, toughness_field, direction_law, step_controller, updater, remesh_policy, config)`
  * `grow_one_increment(self, geom) -> PropagationResult`

Flow:

1. Evaluate base geometry once.
2. Find active tips (deg-1) and compute `Keff` vs `Kc`.
3. For each active tip:

   * compute `theta0`
   * run `StepController.choose_step(...)` which internally evaluates trial geometries using `CandidateEvaluator`
4. Apply accepted updates:

   * simultaneous or sequential per config
5. Return updated geometry + diagnostics

### 10) `results.py`

**Responsibility:** structured outputs and diagnostics.

* `PropagationResult`:

  * `geom_new`
  * `tip_reports: list[TipPropagationReport]`
  * `base_eval` (optional)
  * `timing`, `solver_call_counts`, etc.

---

## Minimal class interfaces (so implementation doesn’t sprawl)

Here’s the “contract surface” that keeps things clean:

* **Evaluator** is the only object that knows how to call `Usolver` and `Uprocessor`.
* **Updater/RemeshPolicy** only touch geometry.
* **StepController** only decides step sizes (calls evaluator but doesn’t solve itself).
* **Propagator** orchestrates.

That separation is what prevents one giant file.

