# Mesh Budget Allocation Policy

# Mesh Budget Policy for Fixed-Step Crack Growth (VCM / Upropagation)

## Purpose

In fixed-step propagation, the crack polyline accumulates internal degree-2 vertices (“kinks”). If the discretization enforces a fixed **minimum** panel allocation near every kink and endpoint, then the mandatory (fixed) panel count grows with the number of kinks and will eventually exceed the global budget (e.g. $$N_p \sim 2\,\texttt{ne\_half}$$).  

This policy defines an allocation strategy that:

- preserves **tip/endpoint resolution** (propagation-critical),
- provides **optional** kink resolution with graceful degradation,
- avoids hard failures until a true infeasibility is reached,
- supports inexpensive long crack-growth simulations.

---

## Inputs and terminology

For each polyline branch:

- Total panel budget:  
  $$N_p = \max\left(8,\; 2\,\texttt{ne\_half}\right).$$
- Polyline path: vertex IDs $$\{v_0,\ldots,v_m\}$$, with $$n_{\mathrm{seg}} = m$$ segments.
- Segment lengths: $$\{\ell_k\}_{k=0}^{n_{\mathrm{seg}}-1}$$, total length  
  $$L=\sum_k \ell_k.$$
- **Endpoint segments**: segment 0 and/or $$n_{\mathrm{seg}}-1$$ when the corresponding endpoint is a tip (deg=1) or (optionally) a junction endpoint (deg ≥ 3).
- **Kink-adjacent segments**: segment indices adjacent to internal deg=2 vertices along the path.

### User knobs (recommended defaults)

- `endpoint_min_nodes`: minimum nodes clustered near each endpoint (default 6). Implemented as  
  `endpoint_min_panels = endpoint_min_nodes - 1`.
- `other_min_panels`: minimum panels per segment away from special regions (default 4).
- Angle thresholds:
  - $$\theta_{\mathrm{merge}} = 15^\circ$$
  - $$\theta_{\mathrm{strong}} = 30^\circ$$
- Nominal kink padding (per kink-adjacent segment): `kink_min_panels = 4`
- Strong-kink floor: `kink_strong_floor = 2` (per kink-adjacent segment)

---

## Policy overview

The policy has three stages:

1. **Geometry cleanup (merge):** remove deg-2 vertices with turning angle $$< \theta_{\mathrm{merge}}$$ (near-collinear segments). This reduces the growth-induced kink explosion without materially changing crack shape.
2. **Hierarchical minimum allocation:** enforce hard minimums for endpoints and a global backbone; apply kink refinement as a **soft** minimum that may be degraded if necessary.
3. **Uniform remainder distribution:** distribute remaining panels approximately uniformly across non-special segments (or across special ones if all are special).

---

## Angle classification

For an internal vertex $$v_i$$ with neighboring vertices $$v_{i-1}, v_{i+1}$$, define the turning angle

$$
\theta_i = \cos^{-1}\!\left(
\frac{(x_i-x_{i-1})\cdot (x_{i+1}-x_i)}
{\|x_i-x_{i-1}\|\,\|x_{i+1}-x_i\|}
\right)\in[0,\pi],
$$

where $$x_j$$ is the vertex coordinate.

- If $$\theta_i < \theta_{\mathrm{merge}}$$, the vertex is merged away.
- Otherwise classify:
  - **mild kink** if $$\theta_{\mathrm{merge}} \le \theta_i < \theta_{\mathrm{strong}}$$
  - **strong kink** if $$\theta_i \ge \theta_{\mathrm{strong}}$$

For each kink at $$v_i$$, the two adjacent segments $$(i-1)$$ and $$i$$ inherit the same severity label.

---

## Hierarchical budget rules

Let $$\underline{N}_k$$ denote the minimum panels required on segment $$k$$.

- **Base backbone:** initialize  
  $$\underline{N}_k \leftarrow \texttt{other\_min\_panels}\quad \forall k.$$
- **Endpoints (hard):** for refined endpoint segments $$k\in\mathcal{E}$$, set  
  $$\underline{N}_k \leftarrow \max(\underline{N}_k,\; \texttt{endpoint\_min\_panels}).$$
- **Kinks (soft):** for kink-adjacent segments $$k\in\mathcal{K}$$, attempt to enforce  
  $$\underline{N}_k \leftarrow \max(\underline{N}_k,\; \texttt{kink\_min\_panels}),$$  
  but allow graceful degradation below.

### Graceful degradation

If $$\sum_k \underline{N}_k > N_p$$:

1. Decrement kink padding on **mild** kink-adjacent segments, one panel at a time (evenly), down to **zero** padding.
2. If still infeasible, decrement kink padding on **strong** kink-adjacent segments down to a floor of `kink_strong_floor`.
3. If still infeasible, declare infeasible and request escalation:  
   `ne_half ← ne_half + Δ` (default $$\Delta=20$$) and retry.

This guarantees:
- tip/endpoint resolution is preserved,
- mild kink resolution vanishes first,
- strong kinks retain a minimal local resolution.

---

## Pseudocode

```python
def allocate_panels(Np, nseg, endpoint_segs, kink_segs,
                    endpoint_min_panels, other_min_panels,
                    kink_min_panels, kink_severity, kink_strong_floor):
    # Hard minima
    Nseg = [other_min_panels] * nseg
    for k in endpoint_segs:
        Nseg[k] = max(Nseg[k], endpoint_min_panels)

    # Soft kink minima (severity-aware)
    for k in kink_segs:
        if kink_severity[k] == "strong":
            Nseg[k] = max(Nseg[k], max(kink_min_panels, kink_strong_floor))
        else:
            Nseg[k] = max(Nseg[k], kink_min_panels)

    def used():
        return sum(Nseg)

    # Degrade mild kinks first down to zero padding
    while used() > Np and any(kink_padding(k) > 0 for k in mild_kink_segs):
        for k in mild_kink_segs:
            if kink_padding(k) > 0:
                Nseg[k] -= 1

    # Then degrade strong kinks down to floor
    while used() > Np and any(Nseg[k] > base(k) + kink_strong_floor for k in strong_kink_segs):
        for k in strong_kink_segs:
            if Nseg[k] > base(k) + kink_strong_floor:
                Nseg[k] -= 1

    if used() > Np:
        raise Infeasible("increase ne_half by +20")

    # Distribute remainder uniformly to non-special segments
    R = Np - used()
    distribute_uniformly(Nseg, R, prefer="non_special")
    return Nseg
```

---

## Escalation rule

If infeasible after degradation, increase `ne_half` by $$\Delta = 20$$ and retry.  

Frequent escalation indicates either:
- the growth law is producing many **strong** kinks (physics/geometry issue), or
- endpoint/backbone minima are too aggressive for the selected `ne_half`.

## Authors

Nasr Ghoniem, February 2026

## License

MIT License
