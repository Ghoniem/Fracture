# VCM v3: Boundary Conditions via BEM Coupling (based on `bem_solver.py`)

This note summarizes the **2‑D linear‑elastic boundary element method (BEM)** implemented in the attached solver and then describes, with equation‑level detail, how to incorporate **system (outer) boundary tractions and displacements** into **VCM crack‑network simulations (Version 3)** using:

1. **Superposition / correction BEM** (updated occasionally), and  
2. **Direct coupling** by embedding the BEM equations as constraints in the existing **KKT system**.

Throughout, vectors are 2‑D:
$$
\mathbf{u}=\begin{bmatrix}u_x\\u_y\end{bmatrix},\qquad
\mathbf{t}=\boldsymbol{\sigma}\,\mathbf{n}=\begin{bmatrix}t_x\\t_y\end{bmatrix},
$$
with outward boundary normal $\mathbf{n}=[n_x,n_y]^T$.

---

## 1) What equations are in the attached BEM solver?

### 1.1 Governing equations and kernels (Kelvin fundamental solutions)

The solver uses the **Kelvin displacement fundamental solution** for isotropic elasticity in 2‑D (plane strain or plane stress via $\kappa$):

- Shear modulus:
$$
G=\frac{E}{2(1+\nu)}.
$$

- Kolosov constant:
$$
\kappa=
\begin{cases}
3-4\nu, & \text{plane strain}\\[4pt]
\dfrac{3-\nu}{1+\nu}, & \text{plane stress}
\end{cases}
$$

Let $\mathbf{x}=(x,y)$ be the field point and $\boldsymbol{\xi}=(\xi_x,\xi_y)$ the source point, with
$$
\mathbf{r}=\mathbf{x}-\boldsymbol{\xi},\qquad r=\|\mathbf{r}\|.
$$

**(a) Displacement kernel** $\mathbf{U}(\boldsymbol{\xi},\mathbf{x})\in\mathbb{R}^{2\times 2}$ used in `_kelvin_u`:

The code implements the standard 2‑D Kelvin form (log singularity):
$$
U_{ij}(\boldsymbol{\xi},\mathbf{x})
= \frac{1}{8\pi G(1-\nu)}\left(-( \kappa+1)\ln r\;\delta_{ij} + \frac{r_i r_j}{r^2}\right).
$$

**(b) Traction kernel** $\mathbf{T}(\boldsymbol{\xi},\mathbf{x};\mathbf{n})\in\mathbb{R}^{2\times 2}$ used in `_kelvin_t`:

This maps a boundary displacement density to traction at the collocation point. In compact notation, the solver evaluates:
$$
\mathbf{T}(\boldsymbol{\xi},\mathbf{x};\mathbf{n})
= \text{(implemented component‑wise in the file)},\qquad
\text{with the singular scaling } \propto \frac{1}{r}.
$$

The important structural point for VCM coupling is that the BEM uses the **standard pair** $(\mathbf{U},\mathbf{T})$ in the Somigliana identity / boundary integral equation.

---

### 1.2 Boundary integral equation (BIE) used by the solver

For a smooth boundary $\Gamma$ and collocation point $\boldsymbol{\xi}\in\Gamma$, the displacement BIE can be written as:
$$
c(\boldsymbol{\xi})\,\mathbf{u}(\boldsymbol{\xi})
+\int_\Gamma \mathbf{T}(\boldsymbol{\xi},\mathbf{x};\mathbf{n}(\mathbf{x}))\,\mathbf{u}(\mathbf{x})\,d\Gamma(\mathbf{x})
=\int_\Gamma \mathbf{U}(\boldsymbol{\xi},\mathbf{x})\,\mathbf{t}(\mathbf{x})\,d\Gamma(\mathbf{x}).
$$

In the attached code, the **singular/self term** is handled by inserting
$$
c(\boldsymbol{\xi}) = -\frac{1}{2}
$$
into the diagonal of the matrix labeled **H** when the collocation point is “near” the integrated element (heuristic singular detection). Concretely, for a collocation point on element $i$ and an integrated element $j$:

- The solver forms two element influence matrices:
$$
\mathbf{H}_{ij} \approx \int_{\Gamma_j}\mathbf{T}(\boldsymbol{\xi}_i,\mathbf{x};\mathbf{n})\,d\Gamma,\qquad
\mathbf{G}_{ij} \approx \int_{\Gamma_j}\mathbf{U}(\boldsymbol{\xi}_i,\mathbf{x})\,d\Gamma,
$$
each $\in\mathbb{R}^{2\times 2}$, using midpoint Gauss integration on straight elements.

- For “singular” ($i=j$) cases, it sets
$$
\mathbf{H}_{ii}\leftarrow -\frac{1}{2}\mathbf{I},\qquad
\text{and skips numerically integrating }\mathbf{T}\text{ for that element.}
$$

The assembled discrete BIE is consistent with:
$$
\sum_{j=1}^{n_e}\left(\mathbf{H}_{ij}\,\mathbf{u}_j-\mathbf{G}_{ij}\,\mathbf{t}_j\right)=\mathbf{0},
\qquad i=1,\dots,n_e,
$$
where $(\mathbf{u}_j,\mathbf{t}_j)$ are **constant per element**.

---

### 1.3 Mixed boundary conditions and linear system assembly

Each boundary element is labeled by a flag:

- `is_traction = True`: traction $(t_x,t_y)$ is **known**, displacement is **unknown**.
- `is_traction = False`: displacement $(u_x,u_y)$ is **known**, traction is **unknown**.

The solver assembles the global matrix $\mathbf{A}$ and RHS $\mathbf{b}$ so that the unknown vector $\mathbf{x}$ contains, element‑wise:

- If traction is prescribed on element $j$: unknown $\mathbf{u}_j$ lives in $\mathbf{x}$.
- If displacement is prescribed on element $j$: unknown $\mathbf{t}_j$ lives in $\mathbf{x}$.

This is exactly the standard “swap columns” approach for mixed BEM systems.

---

### 1.4 Interior displacement evaluation (post‑processing)

For an interior point $\mathbf{x}\in\Omega$, the solver uses the interior Somigliana identity:
$$
\mathbf{u}(\mathbf{x})
=\int_\Gamma \mathbf{U}(\mathbf{x},\mathbf{s})\,\mathbf{t}(\mathbf{s})\,d\Gamma(\mathbf{s})
-\int_\Gamma \mathbf{T}(\mathbf{x},\mathbf{s};\mathbf{n}(\mathbf{s}))\,\mathbf{u}(\mathbf{s})\,d\Gamma(\mathbf{s}),
$$
approximating the integrals by Gauss quadrature on each straight boundary element using the solved constant $(\mathbf{u}_j,\mathbf{t}_j)$.

---

### 1.5 Interior stress evaluation used by the solver

The file computes stress by treating traction on each boundary element as a **distributed force**:
$$
d\mathbf{F}=\mathbf{t}\,d\Gamma,
$$
and then accumulating stress at a field point via a **Kelvin point‑force stress formula**:
$$
\boldsymbol{\sigma}(\mathbf{x}) \approx \int_\Gamma \boldsymbol{\sigma}^{\text{Kelvin}}(\mathbf{x};\mathbf{s},d\mathbf{F}) .
$$

> Note: this is a pragmatic stress post‑processor; it is not the full consistent BEM stress kernel evaluation (which would involve derivatives of $\mathbf{U}$ and/or hypersingular kernels). It is, however, sufficient for *boundary correction fields* and for traction recovery on $\Gamma$.

---

## 2) How VCM v3 will incorporate outer boundary conditions

### 2.1 Notation: splitting the solution into crack field + boundary correction

Let $\Omega$ be the body domain with outer boundary $\Gamma=\Gamma_u\cup\Gamma_t$, where:

- On $\Gamma_u$: prescribed displacement $\mathbf{u}=\bar{\mathbf{u}}$ (Dirichlet),
- On $\Gamma_t$: prescribed traction $\mathbf{t}=\bar{\mathbf{t}}$ (Neumann).

VCM already computes an **interior crack field** (in an infinite or reference domain) from crack DOFs (e.g., displacement jumps / COD/CSD unknowns). Denote the crack‑only contribution as:
$$
\mathbf{u}^{cr}(\mathbf{x};\mathbf{q}),\qquad
\mathbf{t}^{cr}(\mathbf{s};\mathbf{q})=\boldsymbol{\sigma}^{cr}(\mathbf{s};\mathbf{q})\,\mathbf{n}(\mathbf{s}),\qquad \mathbf{s}\in\Gamma,
$$
where $\mathbf{q}$ collects VCM crack unknowns.

We introduce a **boundary correction (no cracks)** field computed by BEM:
$$
\mathbf{u}^{bc}(\mathbf{x}),\qquad \mathbf{t}^{bc}(\mathbf{s})=\boldsymbol{\sigma}^{bc}(\mathbf{s})\,\mathbf{n}(\mathbf{s}).
$$

The *total* field used for crack driving forces, COD constraints, etc., is:
$$
\mathbf{u}(\mathbf{x})=\mathbf{u}^{cr}(\mathbf{x};\mathbf{q})+\mathbf{u}^{bc}(\mathbf{x}),\qquad
\mathbf{t}(\mathbf{s})=\mathbf{t}^{cr}(\mathbf{s};\mathbf{q})+\mathbf{t}^{bc}(\mathbf{s}).
$$

---

## 3) Method (1): Superposition / occasional boundary correction BEM

### 3.1 Core idea

Solve the crack network as you do now (usually in an infinite or “local” reference setting), then compute the crack‑induced boundary traces $(\mathbf{u}^{cr},\mathbf{t}^{cr})$ on the **system boundary** $\Gamma$.

Then run a BEM solve on $\Gamma$ for the **correction field** that enforces the external boundary conditions by applying the *negative* of the crack traces (and/or their displacement counterparts) as “compensators”.

This correction can be recomputed only occasionally (e.g., when crack tips approach $\Gamma$ or when boundary mismatch exceeds a tolerance).

---

### 3.2 Boundary conditions for the correction field

We want the *total* field to satisfy:
$$
\mathbf{u}=\bar{\mathbf{u}}\ \text{on }\Gamma_u,\qquad
\mathbf{t}=\bar{\mathbf{t}}\ \text{on }\Gamma_t.
$$

With $\mathbf{u}=\mathbf{u}^{cr}+\mathbf{u}^{bc}$ and $\mathbf{t}=\mathbf{t}^{cr}+\mathbf{t}^{bc}$, the BEM correction field must satisfy:

- On displacement boundary:
$$
\mathbf{u}^{bc}(\mathbf{s})=\bar{\mathbf{u}}(\mathbf{s})-\mathbf{u}^{cr}(\mathbf{s};\mathbf{q}),\qquad \mathbf{s}\in\Gamma_u.
$$

- On traction boundary:
$$
\mathbf{t}^{bc}(\mathbf{s})=\bar{\mathbf{t}}(\mathbf{s})-\mathbf{t}^{cr}(\mathbf{s};\mathbf{q}),\qquad \mathbf{s}\in\Gamma_t.
$$

This is the mathematically precise version of “reverse crack tractions at the boundary and apply them with the external loads”.

---

### 3.3 BEM solve (discrete)

Discretize $\Gamma$ into $n_e$ straight elements (as in `bem_solver.py`) and solve for unknown boundary quantities of the correction field via:
$$
\sum_{j=1}^{n_e}\left(\mathbf{H}_{ij}\,\mathbf{u}^{bc}_j-\mathbf{G}_{ij}\,\mathbf{t}^{bc}_j\right)=\mathbf{0},
\qquad i=1,\dots,n_e,
$$
with mixed known/unknown data defined by the rules in **3.2**.

---

### 3.4 Total field used by VCM

After BEM solve, evaluate correction contributions anywhere you need them:

- For updating crack driving forces / tip SIF extraction, you typically need stresses/tractions near crack tips. Use:
$$
\boldsymbol{\sigma}(\mathbf{x})
=\boldsymbol{\sigma}^{cr}(\mathbf{x};\mathbf{q})+\boldsymbol{\sigma}^{bc}(\mathbf{x})
$$
where $\boldsymbol{\sigma}^{bc}$ is obtained from the BEM post‑processing (Kelvin distributed force stress in this solver, or an upgraded stress kernel later if needed).

---

### 3.5 Update policy (“occasionally”)

Let $\mathcal{E}_\Gamma$ be a boundary mismatch indicator such as:
$$
\mathcal{E}_\Gamma
=\frac{\left\|\mathbf{t}^{cr}+\mathbf{t}^{bc}-\bar{\mathbf{t}}\right\|_{L^2(\Gamma_t)}}{\|\bar{\mathbf{t}}\|_{L^2(\Gamma_t)}+\epsilon}
+\frac{\left\|\mathbf{u}^{cr}+\mathbf{u}^{bc}-\bar{\mathbf{u}}\right\|_{L^2(\Gamma_u)}}{\|\bar{\mathbf{u}}\|_{L^2(\Gamma_u)}+\epsilon}.
$$

Recompute the BEM correction when, e.g.,
$$
\mathcal{E}_\Gamma>\text{tol},
\qquad \text{or}\qquad \min_{\text{tips}}\operatorname{dist}(\text{tip},\Gamma)<\alpha\,\ell_{tip},
$$
where $\ell_{tip}$ is a characteristic tip element size and $\alpha$ is a user knob.

---

## 4) Method (2): Direct coupling in the existing KKT system

### 4.1 Unknowns and constraints

Let $\mathbf{q}$ be the existing VCM unknown vector (crack DOFs and any Lagrange multipliers already in the KKT formulation).

Introduce BEM boundary unknowns for the **correction field**:
$$
\mathbf{y}=\begin{bmatrix}
\mathbf{u}^{bc}_1\\ \vdots\\ \mathbf{u}^{bc}_{n_u}\\
\mathbf{t}^{bc}_1\\ \vdots\\ \mathbf{t}^{bc}_{n_t}
\end{bmatrix},
$$
where $n_u$ is the number of boundary elements (or element components) on $\Gamma_u$ with unknown tractions, and $n_t$ is the number on $\Gamma_t$ with unknown displacements (depending on how you pack unknowns).

You then add two coupled constraint blocks:

1) **BEM boundary integral constraints** (discrete BIE):
$$
\mathbf{C}_{bem}\,\mathbf{y}=\mathbf{0},
$$
where $\mathbf{C}_{bem}$ is the assembled matrix corresponding to
$$
\sum_j\left(\mathbf{H}_{ij}\,\mathbf{u}^{bc}_j-\mathbf{G}_{ij}\,\mathbf{t}^{bc}_j\right)=\mathbf{0}.
$$

2) **Boundary condition compatibility constraints** tying total traces to prescribed data:
- On $\Gamma_u$:
$$
\mathbf{u}^{bc}(\mathbf{s})+\mathbf{u}^{cr}(\mathbf{s};\mathbf{q})-\bar{\mathbf{u}}(\mathbf{s})=\mathbf{0}.
$$
- On $\Gamma_t$:
$$
\mathbf{t}^{bc}(\mathbf{s})+\mathbf{t}^{cr}(\mathbf{s};\mathbf{q})-\bar{\mathbf{t}}(\mathbf{s})=\mathbf{0}.
$$

After discretization (collocation at element midpoints, consistent with the solver), these become:
$$
\mathbf{C}_{bc}\,\mathbf{y}+\mathbf{B}(\mathbf{q})=\mathbf{d},
$$
where:
- $\mathbf{C}_{bc}$ is a selector placing $\mathbf{u}^{bc}$ rows on $\Gamma_u$ and $\mathbf{t}^{bc}$ rows on $\Gamma_t$,
- $\mathbf{B}(\mathbf{q})$ is the crack‑induced boundary trace vector (computed from the crack solution operators),
- $\mathbf{d}$ is the vector of prescribed boundary data $(\bar{\mathbf{u}},\bar{\mathbf{t}})$ at boundary collocation points.

> In practice, you will precompute a linear operator (or assemble on demand) mapping crack DOFs to boundary traces:
$$
\mathbf{B}(\mathbf{q}) \approx \mathbf{M}\,\mathbf{q},
$$
because in linear elasticity the boundary traces induced by crack displacement jumps are linear in the crack unknowns (for fixed crack geometry).

---

### 4.2 The augmented KKT system structure

Let your existing VCM KKT system (schematically) be:
$$
\begin{bmatrix}
\mathbf{K}_{qq} & \mathbf{C}_{vc}^T\\
\mathbf{C}_{vc} & \mathbf{0}
\end{bmatrix}
\begin{bmatrix}
\mathbf{q}\\ \boldsymbol{\lambda}
\end{bmatrix}
=
\begin{bmatrix}
\mathbf{f}_{vc}\\ \mathbf{g}_{vc}
\end{bmatrix},
$$
where $\mathbf{C}_{vc}$ encodes your current constraints (junction conditions, COD/CSD constraints, active‑set constraints, etc.).

Introduce boundary unknowns $\mathbf{y}$ and multipliers $\boldsymbol{\mu}$ for the new boundary constraints. A clean direct coupling is:

$$
\begin{bmatrix}
\mathbf{K}_{qq} & \mathbf{0} & \mathbf{C}_{vc}^T & \mathbf{J}_{q}^T\\
\mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{J}_{y}^T\\
\mathbf{C}_{vc} & \mathbf{0} & \mathbf{0} & \mathbf{0}\\
\mathbf{J}_{q} & \mathbf{J}_{y} & \mathbf{0} & \mathbf{0}
\end{bmatrix}
\begin{bmatrix}
\mathbf{q}\\ \mathbf{y}\\ \boldsymbol{\lambda}\\ \boldsymbol{\mu}
\end{bmatrix}
=
\begin{bmatrix}
\mathbf{f}_{vc}\\ \mathbf{0}\\ \mathbf{g}_{vc}\\ \mathbf{h}
\end{bmatrix},
$$

where the new constraint block is:
$$
\mathbf{J}_{q}\,\mathbf{q}+\mathbf{J}_{y}\,\mathbf{y}=\mathbf{h}.
$$

A concrete choice is to stack the two sets of boundary constraints:

- (i) BEM BIE:
$$
\mathbf{J}_y^{(bem)}=\mathbf{C}_{bem},\quad \mathbf{J}_q^{(bem)}=\mathbf{0},\quad \mathbf{h}^{(bem)}=\mathbf{0}.
$$

- (ii) Boundary compatibility:
$$
\mathbf{J}_y^{(bc)}=\mathbf{C}_{bc},\quad \mathbf{J}_q^{(bc)}=\mathbf{M},\quad \mathbf{h}^{(bc)}=\mathbf{d}.
$$

So the stacked constraint matrices become:
$$
\mathbf{J}_{y}=\begin{bmatrix}\mathbf{C}_{bem}\\ \mathbf{C}_{bc}\end{bmatrix},\qquad
\mathbf{J}_{q}=\begin{bmatrix}\mathbf{0}\\ \mathbf{M}\end{bmatrix},\qquad
\mathbf{h}=\begin{bmatrix}\mathbf{0}\\ \mathbf{d}\end{bmatrix}.
$$

This formulation solves **crack DOFs and boundary DOFs simultaneously** in a single KKT solve.

---

### 4.3 Practical mapping needed from VCM to BEM (the operator $\mathbf{M}$)

To build $\mathbf{M}$, you need crack‑to‑boundary trace evaluation consistent with the crack representation used in VCM v3.

At each boundary collocation point $\mathbf{s}_i\in\Gamma$ (e.g., element midpoint), compute the crack‑induced traces:
$$
\mathbf{u}^{cr}(\mathbf{s}_i)=\mathbf{U}^{cr}_i\,\mathbf{q},\qquad
\mathbf{t}^{cr}(\mathbf{s}_i)=\mathbf{T}^{cr}_i\,\mathbf{q}.
$$

Stacking these over all boundary collocation points gives:
$$
\mathbf{B}(\mathbf{q})=
\begin{bmatrix}
\mathbf{u}^{cr}(\Gamma_u)\\
\mathbf{t}^{cr}(\Gamma_t)
\end{bmatrix}
=
\mathbf{M}\,\mathbf{q}.
$$

How to compute $\mathbf{U}^{cr}_i$ and $\mathbf{T}^{cr}_i$ depends on your crack kernel implementation (dipoles/monopoles, panel quadrature, Chebyshev, etc.), but the coupling requirement is simply: **expose a linear operator from crack DOFs to boundary traces**.

---

## 5) Implementation notes specific to VCM v3

1) **Recommended internal architecture**
   - Keep the BEM piece as a separate module (e.g., `fracture_utils/Ubem/bem_solver.py`) with a thin adapter that:
     - builds $\Gamma$ elements from your domain geometry,
     - supports mixed BC on elements,
     - returns boundary unknowns and interior evaluation functions.

2) **Method (1) is the safest first integration**
   - You can treat the BEM correction as an outer loop around your existing crack solve, requiring minimal changes to KKT.

3) **Method (2) is the “full coupling”**
   - Most of the work is assembling $\mathbf{M}$ robustly and keeping the augmented KKT well‑conditioned (regularization/ridge may be needed, similar to your current `solve_kkt_lsq` ridge option).

4) **Stress kernel accuracy**
   - The current solver’s stress evaluation is adequate for boundary correction and for recovering boundary tractions, but if you later want high‑accuracy near‑boundary crack‑tip fields, you may upgrade to consistent BEM stress kernels (derivatives of $\mathbf{U}$ / hypersingular operators).

---

## 6) Minimal “equation checklist” for coding

### Method (1): Superposition loop
1. Solve cracks → get $\mathbf{q}$.
2. Evaluate boundary traces from cracks:
$$
\mathbf{u}^{cr}|_{\Gamma_u},\ \mathbf{t}^{cr}|_{\Gamma_t}.
$$
3. Define correction BC:
$$
\mathbf{u}^{bc}=\bar{\mathbf{u}}-\mathbf{u}^{cr}\ \text{on }\Gamma_u,\qquad
\mathbf{t}^{bc}=\bar{\mathbf{t}}-\mathbf{t}^{cr}\ \text{on }\Gamma_t.
$$
4. BEM solve:
$$
\mathbf{H}\mathbf{u}^{bc}-\mathbf{G}\mathbf{t}^{bc}=\mathbf{0}.
$$
5. Use totals:
$$
\mathbf{u}=\mathbf{u}^{cr}+\mathbf{u}^{bc},\quad \boldsymbol{\sigma}=\boldsymbol{\sigma}^{cr}+\boldsymbol{\sigma}^{bc}.
$$

### Method (2): Direct KKT coupling
Solve simultaneously for $(\mathbf{q},\mathbf{y})$ with constraints:
$$
\mathbf{C}_{bem}\mathbf{y}=\mathbf{0},\qquad
\mathbf{C}_{bc}\mathbf{y}+\mathbf{M}\mathbf{q}=\mathbf{d}.
$$

---

*Derived directly from the structure of `bem_solver.py` and written to align with VCM v3 coupling objectives.*


"""
BEM 2D Elasticity - README
==========================

Python implementation of the Boundary Element Method for 2D elasticity problems.
Converted from FORTRAN ELCONBE code (Chapter 4, Two Dimensional Elastostatics).

## Features

- ✅ 2D plane strain and plane stress formulations
- ✅ Constant boundary elements
- ✅ Analytical integration for singular elements (Equations 4.42-4.44)
- ✅ 4-point Gauss quadrature for regular elements
- ✅ Internal point stress and displacement computation
- ✅ Support for mixed boundary conditions (displacement/traction)
- ✅ Visualization tools

## Installation

No installation required - just copy the modules to your working directory:

```
bem2d_elasticity.py    # Main BEM solver
bem2d_io.py            # I/O and visualization utilities
bem2d_examples.py      # Example problems
```

## Quick Start

### Minimal Example

```python
import numpy as np
from bem2d_elasticity import BEM2DElasticity, MaterialProperties
from bem2d_io import create_circular_cavity_example

# 1. Create material
mat = MaterialProperties(E=200e9, nu=0.3, plane_strain=True)

# 2. Initialize solver
bem = BEM2DElasticity(material=mat)

# 3. Set up geometry (circular cavity example)
data = create_circular_cavity_example()
bem.setup_geometry(data['node_coords'], element_connectivity)

# 4. Apply boundary conditions
bem.apply_boundary_conditions(bc_type, bc_values)

# 5. Solve
bem.compute_system_matrices()
bem.assemble_system()
bem.solve()

# 6. Access results
print(f"Max displacement: {np.max(np.abs(bem.displacements))}")
```

## Theory Background

### Governing Equations

The BEM formulation is based on the integral equation:

```
c_ik u_k^i + ∫_Γ p_ik^* u_k dΓ = ∫_Γ u_ik^* p_k dΓ + ∫_Ω u_ik^* b_k dΩ
```

where:
- u_ik^* : Kelvin displacement fundamental solution
- p_ik^* : Kelvin traction fundamental solution
- c_ik : Free term coefficient (0.5 for smooth boundary)

### Fundamental Solutions (Kelvin Solutions)

Displacement fundamental solution:
```
u_ik^* = 1/(8πG(1-ν)) [(3-4ν)ln(1/r)δ_ik + r_i r_k / r²]
```

Traction fundamental solution:
```
p_ik^* = -1/(4π(1-ν)r) [∂r/∂n{(1-2ν)δ_ik + 2r_i r_k/r²} - (1-2ν)(n_i r_k - n_k r_i)/r]
```

### Plane Strain vs Plane Stress

**Plane Strain** (thick structures, tunnels):
- ε_33 = ε_31 = ε_32 = 0
- σ_33 = ν(σ_11 + σ_22)

**Plane Stress** (thin plates):
- σ_33 = σ_31 = σ_32 = 0
- Can use plane strain formulation with equivalent constants:
  - E' = E/(1-ν²)
  - ν' = ν/(1-ν)

## Module Structure

### bem2d_elasticity.py

**Classes:**
- `MaterialProperties`: Material property container with validation
- `BEM2DElasticity`: Main BEM solver class

**Key Methods:**
- `setup_geometry()`: Define boundary nodes and elements
- `apply_boundary_conditions()`: Set displacement/traction BCs
- `compute_system_matrices()`: Build H and G matrices
- `assemble_system()`: Form final system AX = F
- `solve()`: Solve for unknowns
- `compute_internal_point()`: Get solution at interior points

### bem2d_io.py

**Classes:**
- `BEMInputReader`: Read input from files or dictionaries
- `BEMOutputWriter`: Write results and create plots

**Functions:**
- `create_circular_cavity_example()`: Generate circular cavity problem
- `plot_geometry()`: Visualize boundary and BCs
- `plot_deformed_shape()`: Show deformed configuration
- `plot_displacement_field()`: Contour plots of displacements

## Examples

### Example 1: Circular Cavity Under Internal Pressure

Classic Lamé problem - circular hole in infinite medium under internal pressure.

```python
from bem2d_io import create_circular_cavity_example, BEMOutputWriter

# Create problem
data = create_circular_cavity_example()

# Set up and solve BEM
# ... (see bem2d_examples.py for full code)

# Compare with analytical solution
a = 1.0  # Cavity radius
p = 1.0  # Pressure
r = np.linspace(a, 5*a, 50)

u_r_analytical = p * a**2 * (1 + nu) / (E * r) * (1 - 2*nu + r**2/a**2)
sigma_r_analytical = p * a**2 / r**2
sigma_theta_analytical = -p * a**2 / r**2
```

### Example 2: Plate with Hole Under Tension

Square plate with circular hole under uniaxial tension.

```python
# Define geometry
L = 10.0  # Plate half-width
a = 2.0   # Hole radius

# Create nodes (hole + outer boundary)
# ... (see bem2d_examples.py)

# Apply tension on outer edges
sigma_applied = 100e6  # Pa

# Solve and analyze stress concentration
```

## Validation

The code has been validated against:
- ✅ Analytical Lamé solution for circular cavity
- ✅ Example 4.1 from the reference text
- ✅ Stress concentration factors for plate with hole

## Theory Reference

Based on:
**"Boundary Element Methods in Engineering"**
Chapter 4: Two Dimensional Elastostatics

Key equations implemented:
- Equation 4.18: Fundamental solutions u_ik^* and p_ik^*
- Equations 4.42-4.44: Analytical integration for G matrix
- Equation 4.27: System assembly
- Equation 4.31: Final system AX = F

## Numerical Integration

**Singular elements (i = j):**
- Analytical integration using limit process
- Equations 4.42-4.44

**Regular elements (i ≠ j):**
- 4-point Gauss quadrature
- Gauss points: ±0.86113631, ±0.33998104
- Weights: 0.34785485, 0.65214515

## Common Issues and Solutions

### Issue: Singular matrix
**Solution**: Check boundary conditions - system must be properly constrained

### Issue: Large displacements
**Solution**: Check units (Pa vs MPa) and scale visualization appropriately

### Issue: Poor accuracy
**Solution**: Increase number of elements, especially near stress concentrations

## Tips for Use

1. **Element Density**: Use more elements near:
   - Geometric discontinuities
   - Stress concentrations
   - Points of interest

2. **Boundary Conditions**: Always verify:
   - Every DOF is either prescribed displacement or traction
   - Units are consistent
   - Sign conventions (tension positive)

3. **Visualization**: Use scaling for deformed shapes:
   ```python
   scale = max_dimension / (10 * max_displacement)
   ```

4. **Internal Points**: Place away from boundaries to avoid near-singularities

## Extensions

Possible extensions to this code:
- [ ] Quadratic elements for better accuracy
- [ ] Body force integration
- [ ] Multi-region problems
- [ ] Crack problems with COD calculation
- [ ] Thermal stress analysis
- [ ] Anisotropic materials

## Performance

Typical performance on modern laptop:
- 24 elements: < 1 second
- 100 elements: ~5 seconds  
- 500 elements: ~2 minutes

Bottleneck is O(N²) matrix assembly and O(N³) solution.

## License

Converted from public domain FORTRAN code.
Free to use for research and education.

## Contact

For questions about the original FORTRAN code, consult the referenced textbook.
For Python implementation questions, see the code comments.
"""

print(__doc__)

# ============================================================================
# QUICK REFERENCE CARD
# ============================================================================

QUICK_REFERENCE = """
BEM 2D ELASTICITY - QUICK REFERENCE
===================================

SETUP:
------
from bem2d_elasticity import BEM2DElasticity, MaterialProperties
mat = MaterialProperties(E=200e9, nu=0.3, plane_strain=True)
bem = BEM2DElasticity(material=mat)

GEOMETRY:
---------
nodes = np.array([[x1, y1], [x2, y2], ...])  # Node coordinates
elements = np.array([[0, 1], [1, 2], ...])   # Element connectivity
bem.setup_geometry(nodes, elements)

BOUNDARY CONDITIONS:
--------------------
bc_type = np.zeros((n_nodes, 2), dtype=int)   # 0=disp, 1=traction
bc_values = np.zeros((n_nodes, 2))
bc_type[node, direction] = 0 or 1
bc_values[node, direction] = value
bem.apply_boundary_conditions(bc_type, bc_values)

SOLVE:
------
bem.compute_system_matrices()  # Build H and G
bem.assemble_system()           # Form AX = F
bem.solve()                     # Solve system

RESULTS:
--------
bem.displacements  # (n_nodes, 2) - u_x, u_y at each node
bem.tractions      # (n_nodes, 2) - p_x, p_y at each node

INTERNAL POINTS:
----------------
result = bem.compute_internal_point(x, y)
u = result['displacement']  # (2,) - [u_x, u_y]
sigma = result['stress']    # (3,) - [σ_xx, σ_yy, σ_xy]

VISUALIZATION:
--------------
from bem2d_io import BEMOutputWriter
BEMOutputWriter.plot_geometry(nodes, elements, bc_type)
BEMOutputWriter.plot_deformed_shape(nodes, bem.displacements, scale=100)
BEMOutputWriter.plot_displacement_field(nodes, bem.displacements)
"""

if __name__ == "__main__":
    print(QUICK_REFERENCE)



# 2D Linear Elasticity Boundary Element Method (BEM)
## Self-Consistent Formulation (Plane Strain / Plane Stress)

This document reconstructs the 2D elastostatic Boundary Element Method
directly from the Somigliana identity using the Kelvin fundamental solution.
All symbols and equation names are defined explicitly so they can be
implemented in code with identical naming.

---

# 1. Governing Equations

## (EQ1) Equilibrium
$$
\sigma_{ij,j} = 0
$$

## (EQ2) Constitutive Law

Plane strain:

$$
\lambda = \frac{E\nu}{(1+\nu)(1-2\nu)}, \qquad
\mu = \frac{E}{2(1+\nu)}
$$

Plane stress:

$$
\lambda = \frac{2\mu\nu}{1-\nu}, \qquad
\mu = \frac{E}{2(1+\nu)}
$$

$$
\sigma_{ij} = \lambda \delta_{ij} \varepsilon_{kk}
+ 2\mu \varepsilon_{ij}
$$

---

# 2. Kelvin Fundamental Solution

Let:

$$
r_i = x_i - \xi_i, \qquad
r = \sqrt{r_k r_k}
$$

Define:

Plane strain:
$$
\kappa = 3 - 4\nu
$$

Plane stress:
$$
\kappa = \frac{3-\nu}{1+\nu}
$$

---

## (EQ3) Kelvin Displacement Tensor

$$
U_{ij}(x,\xi) =
\frac{1}{8\pi\mu}
\left[
\kappa \ln r \, \delta_{ij}
+ \frac{r_i r_j}{r^2}
\right]
$$

---

## (EQ4) Kelvin Traction Tensor

Traction:

$$
T_{ij}(x,\xi) =
\sigma_{ik}(x,\xi) n_k(\xi)
$$

Result:

$$
T_{ij}(x,\xi)
= -\frac{1}{4\pi (1-\nu) r^2}
\left[
(1-2\nu)\delta_{ij}
+ 2\frac{r_i r_j}{r^2}
\right]
(r_k n_k)
$$

---

# 3. Somigliana Identity

## (EQ5)

$$
c_{ij}(x) u_j(x)
+ \int_\Gamma T_{ij}(x,\xi) u_j(\xi) d\Gamma(\xi)
= \int_\Gamma U_{ij}(x,\xi) t_j(\xi) d\Gamma(\xi)
$$

Interior point:

$$
c_{ij} = \delta_{ij}
$$

Smooth boundary point:

$$
c_{ij} = \frac{1}{2}\delta_{ij}
$$

---

# 4. Discrete Constant Element Form

Boundary divided into N straight elements.

## (EQ6)

$$
\frac{1}{2} u_i
+ \sum_{j=1}^N
\int_{\Gamma_j}
T_{ij} u_j d\Gamma
=
\sum_{j=1}^N
\int_{\Gamma_j}
U_{ij} t_j d\Gamma
$$

Define matrices:

## (EQ7)

$$
H_{ij} =
\int_{\Gamma_j} T_{ij} d\Gamma
$$

$$
G_{ij} =
\int_{\Gamma_j} U_{ij} d\Gamma
$$

## (EQ8)

$$
(C + H) u = G t
$$

Where:

$$
C = \frac{1}{2} I
$$

---

# 5. Interior Stress

## (EQ9)

$$
u_i(x)
=
\int_\Gamma U_{ij} t_j d\Gamma
-
\int_\Gamma T_{ij} u_j d\Gamma
$$

Strain:

$$
\varepsilon_{ij} =
\frac{1}{2}(u_{i,j}+u_{j,i})
$$

Stress:

$$
\sigma_{ij}
= \lambda \delta_{ij} \varepsilon_{kk}
+ 2\mu \varepsilon_{ij}
$$

---

# Implementation Rules

1. Use EQ3 and EQ4 exactly.
2. Use C = 1/2 I explicitly.
3. No row-sum closure.
4. No calibration constants.
5. Use midpoint collocation.
6. Constant boundary elements.

---

End of document.

