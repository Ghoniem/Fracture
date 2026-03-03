# VCM v3.3: Energetics of Fracture Networks
The objectives of VCM v3.3. are two fold:

1. Improve the performance of crack network evolution via topological network operations (trimming, merging, and boundary detection)

2. Continue development of the python modules crack_energetics.py to calculate and plot network energy components.

Let's define the following metrics to describe the evolution of crack networks.

(1) crack network in a bounded domain $\Omega \subset \mathbb{R}^2$
(e.g., a disk) as a graph $G = (V,E) = \bigcup_{e=1}^{N_e} \Gamma_e$, where $V$ denotes the set of vertices and $E$ the set of crack edges,
each $\Gamma_e$ being a parametric curve segment of length $\ell_e$.

(2) The total crack length is $L_{\mathrm{tot}} = \sum_{e \in E} \ell_e$. 

(3) The crack density is $\alpha=\frac{L_{\mathrm{tot}}}{A}$, where A is the domain area (disk Area).

(4) The graph decomposes into $N_c$ connected components $G = \bigcup_{k=1}^{N_c} G_k$, each $G_k$ being a maximal connected subgraph.
The length of component $G_k$ is $L_k = \sum_{e \in G_k} \ell_e$. 

(5) Let $B = \partial \Omega$ denote the external boundary.
A connected component $G_k$ is said to be \emph{spanning}
if it intersects at least two disjoint boundary regions,
$$
G_k \cap B_1 \neq \varnothing,
\qquad
G_k \cap B_2 \neq \varnothing,
\qquad
B_1 \cap B_2 = \varnothing.
$$
In a diametrically compressed disk, for example,
a spanning cluster connects the upper and lower loading arcs,
thereby forming a system-scale fracture path. 

(6) Let $G_{\mathrm{span}}$ denote the spanning component (if it exists),
with total length 

$L_{\mathrm{span}}=\sum_{e \in G_{\mathrm{span}}} \ell_e$

(7) We define the spanning cluster fraction
$$
P_\infty
=
\frac{L_{\mathrm{span}}}{L_{\mathrm{tot}}}.
$$
If no spanning cluster exists, $P_\infty = 0$.

The quantity $P_\infty$ measures global crack connectivity.
For $P_\infty=0$, cracks remain isolated and damage is distributed.
When $P_\infty>0$, a macroscopic fracture path has formed.

(8) For a linear elastic body subjected to a scalar load parameter $P$
with corresponding generalized displacement $\delta$, define the compliance

$$
C(\alpha) \;=\; \frac{\delta}{P}.
$$

Since crack growth generally increases structural flexibility,
$C(\alpha)$ is an increasing function of $\alpha$, i.e. $dC/da > 0$.

**Force (traction) control.** Under traction or force control (e.g.\ prescribed pressure on loading arcs),
the resultant load $P$ is fixed while the displacement $\delta$ adapts.

(9) The stored elastic energy is

$$
U(\alpha) \;=\; \frac{1}{2} P \delta
\;=\; \frac{1}{2} P^2 C(\alpha).
$$

(7) For dead loads, the external work is
$$
W_{\mathrm{ext}}(\alpha) = P \delta = P^2 C(\alpha),
$$

(10) The total potential energy becomes

$$
\Phi_{\mathrm{force}}(\alpha)
\;=\; U(\alpha) - W_{\mathrm{ext}}(\alpha)
\;=\; -\frac{1}{2} P^2 C(\alpha).
$$

(11) The corresponding energy release rate is

$$
G_{\mathrm{force}}(\alpha)
\;=\; -\frac{d\Phi_{\mathrm{force}}}{dL_{tot}}
\;=\; \frac{1}{2} P^2 \frac{dC}{dL_{tot}}.
$$

Thus, under force control, increasing compliance directly lowers
the potential energy, which can promote unstable crack growth.

**Displacement control.**
Under boundary displacement control, the generalized displacement
$\delta$ is prescribed while the load $P$ adapts according to

$$
P(\alpha) = \frac{\delta}{C(\alpha)}.
$$

(13) The stored elastic energy is now

$$
U(\alpha)
\;=\; \frac{1}{2} P \delta
\;=\; \frac{1}{2} \frac{\delta^2}{C(\alpha)}.
$$

In displacement control, the external work is not an independent
variable; the appropriate energetic functional is simply 

(14) The stored energy,

$$
\Phi_{\mathrm{disp}}(\alpha)
\;=\; U(\alpha)
\;=\; \frac{1}{2} \frac{\delta^2}{C(\alpha)}.
$$

Since $C(\alpha)$ increases with crack length,
$\Phi_{\mathrm{disp}}(\alpha)$ decreases more gradually,
and its curvature differs fundamentally from the force-controlled case.

(15) The corresponding energy release rate is

$$
G_{\mathrm{disp}}(\alpha)
\;=\; -\frac{d\Phi_{\mathrm{disp}}}{dL_{total}}
\;=\; \frac{1}{2} \frac{\delta^2}{C(\alpha)^2}
\frac{dC}{dL_{total}}.
$$

The two expressions differ based on the ensemble in which the system is viewed:
$$
\begin{aligned}
G_{\mathrm{force}}(\alpha)
&=
\frac{1}{2} P^2 \frac{dC}{dL_{total}},
\\
G_{\mathrm{disp}}(\alpha)
&=
\frac{1}{2} \frac{\delta^2}{C^2} \frac{dC}{dL_{total}}.
\end{aligned}
$$



Crack growth incurs a surface energy cost.
If $A(\alpha)$ denotes total crack surface (or length in 2D),
the relevant thermodynamic potential is
$A(\alpha)=L_{total}\times h$, where $h$ is the disk thickness.
$$
\mathcal{F}(\alpha)
=
\Phi(\alpha) + \Gamma A(\alpha),
$$

where $\Gamma$ is the fracture energy.
The evolution condition becomes

$$
\frac{d\mathcal{F}}{d\alpha} = 0,
\qquad
\text{with}
\qquad
G(\alpha) = -\frac{d\Phi}{dA} \ge \Gamma.
$$


(16) The
finite drop in $\Phi$ associated with a discontinuous jump in $\alpha$. Let $k$ index quasi-static growth steps and define the incremental energy drop
%
$$
\Delta \Phi_k = \Phi_{k-1}-\Phi_k,
$$

(17) Crack surface density (damage variable): 

$\alpha = \frac{\sum_i \ell_i}{A}$, 

where $\ell_i$ are crack segment lengths.

(18) Alignment parameter: 

$Q = \left\langle 2\cos^2\theta - 1 \right\rangle$

 where $\theta$ is the angle between crack segments and the loading axis.  $Q=1$ corresponds to perfect alignment,
$Q=0$ to isotropic orientation.
