# 2D Elastostatic BEM — Interior Stress from Double-Layer Potential (Plane Stress/Strain)

This note supplements `BEM_2D_Derived_Equations.md` with a **reference-backed** stress-evaluation pathway for a 2D **collocation, constant-element** BEM that supports **mixed BCs** (both displacement- and traction-prescribed boundary segments) via the **Somigliana identity** with **single- and double-layer** boundary terms.

---

## 1. Material parameters and κ switch

Let the Lamé constants be

\[
\mu = \frac{E}{2(1+\nu)},\qquad
\lambda =
\begin{cases}
\displaystyle \frac{E\nu}{(1+\nu)(1-2\nu)} & \text{plane strain}\\[6pt]
\displaystyle \frac{E\nu}{(1+\nu)(1-\nu)} & \text{plane stress}
\end{cases}
\]

Define the standard **Kolossov constant**

\[
\kappa =
\begin{cases}
3-4\nu & \text{plane strain}\\[4pt]
\displaystyle \frac{3-\nu}{1+\nu} & \text{plane stress}
\end{cases}
\]

These conventions match common BEM texts for 2D isotropic elasticity. citeturn1view0turn1view2

---

## 2. Somigliana identity (interior displacement)

For an interior point \(x\in\Omega\) and boundary \(\Gamma\), the displacement satisfies

\[
u_i(x)=\int_{\Gamma} U_{ij}(x,y)\,t_j(y)\,d\Gamma_y \;-
\;\int_{\Gamma} T_{ij}(x,y)\,u_j(y)\,d\Gamma_y ,
\]

where:
- \(U_{ij}\) is the **Kelvin displacement fundamental solution**,
- \(T_{ij}(x,y)\) is the **traction kernel** obtained by applying the traction operator at \(y\) to \(U_{ij}(x,y)\).

This is the standard starting point for mixed-BC 2D elastostatic BEM. citeturn1view0turn6view1

> In your implementation, the *double-layer* contribution is the second term (involving \(T_{ij} u_j\)).

---

## 3. Stress from interior displacement gradients (what we will do)

You asked whether stress will be computed from **numerical displacement gradients**.

**Answer:** for the production solver we should compute stress from **analytic derivatives of the layer potentials**, not from finite-difference (FD) gradients. FD can be kept as an optional diagnostic.

### 3.1 Kinematics and constitutive law

\[
u_{i,k}(x)=\frac{\partial u_i}{\partial x_k}(x),
\qquad
\varepsilon_{ik}=\frac{1}{2}\left(u_{i,k}+u_{k,i}\right),
\qquad
\sigma_{ik}=\lambda\,\varepsilon_{mm}\,\delta_{ik}+2\mu\,\varepsilon_{ik}.
\]

So the core task is computing \(u_{i,k}(x)\).

---

## 4. Analytic gradient of the Somigliana representation

Differentiate the interior representation (valid for \(x\in\Omega\)):

\[
u_{i,k}(x)=\int_{\Gamma} U_{ij,k}(x,y)\,t_j(y)\,d\Gamma_y
\;-
\int_{\Gamma} T_{ij,k}(x,y)\,u_j(y)\,d\Gamma_y .
\]

Thus the needed kernels are:
- \(U_{ij,k}(x,y)\): derivative (w.r.t. the **field point** \(x\)) of Kelvin displacement,
- \(T_{ij,k}(x,y)\): derivative (w.r.t. field point \(x\)) of the traction kernel.

This “differentiate-then-integrate” approach is standard for interior stress recovery. citeturn6view1turn6view0

---

## 5. Reference for the traction kernel T and higher derivatives

A compact tensor form of the **double-layer traction kernel** appears in boundary integral operator formulations (including elastodynamics); the static limit reduces to the elastostatic traction kernel used in BEM. See, e.g., operator definitions and the double-layer kernel in Calderón-calculus style treatments. citeturn6view0

For elastostatic BEM specifically, the traction kernel and its role in Somigliana’s identity is presented in standard boundary element references (constant elements, collocation). citeturn1view0turn6view1

> Practically: we will implement **closed-form** expressions for \(U\), \(T\), and then compute \(U_{,k}\), \(T_{,k}\) analytically (symbol-derived and then coded), with careful regularization near the boundary.

---

## 6. Double-layer-only option (as requested)

If we choose a **double-layer representation** only (indirect formulation),

\[
u_i(x)=\int_{\Gamma} T_{ij}(x,y)\,\phi_j(y)\,d\Gamma_y,
\]

then

\[
u_{i,k}(x)=\int_{\Gamma} T_{ij,k}(x,y)\,\phi_j(y)\,d\Gamma_y,
\quad\Rightarrow\quad
\sigma_{ik}(x)=\lambda\,\varepsilon_{mm}\,\delta_{ik}+2\mu\,\varepsilon_{ik}.
\]

This is the mathematically complete route to interior stress from a double-layer potential: **no numerical differentiation** is required; everything comes from \(T_{ij,k}\).

---

## 7. Numerical differentiation (optional diagnostic only)

We can optionally compute

\[
u_{i,k}(x)\approx \frac{u_i(x+h e_k)-u_i(x-h e_k)}{2h}
\]

and compare against the analytic-gradient stress. This is useful for debugging but should not be the primary method.

---

## 8. What changes in code next

1. Keep your **direct mixed-BC formulation** (Somigliana boundary equation).
2. Implement **interior stress evaluation** using analytic \(U_{,k}\) and \(T_{,k}\) kernels and then Hooke’s law.
3. Support **plane stress vs plane strain** by switching \(\kappa\) (and \(\lambda\)) consistently everywhere.

---

## 9. References (open / accessible)

- Brebbia & Dominguez, *Boundary Elements: An Introductory Course* (open-access chapters; constant-element collocation context). citeturn1view0turn6view1  
- Calderón-calculus style kernel definitions for the double-layer operator (used here as a kernel reference). citeturn6view0  
- Sivakumar thesis (Kelvin solution / fundamental solutions background). citeturn5search16
