# VCM v3.1: Implementation of Boundary Conditions via BEM Coupling with Crack Evolution (based on `bem_solver.py`)

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

