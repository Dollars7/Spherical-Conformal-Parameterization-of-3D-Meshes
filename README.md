# Spherical Conformal Parameterization of 3D Meshes

A pure-Python reference implementation of spherical conformal parameterization for
closed genus-zero triangular meshes, computed by minimizing the discrete harmonic
energy of a map onto the unit sphere `S²`.

Extended from CSE 570 Project 3.

---

## 1. Problem Statement

Let `M ⊂ ℝ³` be a closed, orientable, genus-zero triangulated surface with vertex set
`V = {v₁,…,v_n}` and triangle set `T`. The uniformization theorem guarantees the
existence of a conformal (angle-preserving) diffeomorphism `f : M → S²`, unique up to
Möbius transformations of the sphere.

Following Gu and Yau, the conformal maps `M → S²` are exactly the critical points of
the **harmonic energy**

```
E(f) = ½ · Σ           k_uv · ‖ f(u) − f(v) ‖²
         [u,v] ∈ E(M)
```

where the edge weight is the classical cotangent weight

```
k_uv = ½ · ( cot α_uv + cot β_uv )
```

and `α_uv`, `β_uv` are the two angles opposite the edge `[u,v]` in the triangles
adjacent to it. Writing the map as a coordinate matrix `F ∈ ℝ^{n×3}` whose `i`-th row is
`f(v_i)`, the energy becomes the quadratic form

```
E(F) = ½ · tr( Fᵀ L F )
```

with `L ∈ ℝ^{n×n}` the discrete Laplace–Beltrami (cotangent Laplacian) operator of the
**source** mesh — not of the sphere, and not of the current iterate.

### 1.1 Constraint Set

The parameterization is the solution of

```
minimize    E(F) = ½ tr(Fᵀ L F)
subject to  ‖F_i‖₂ = 1        for i = 1,…,n        (spherical)
            Σ_i w_i F_i = 0                        (zero mass centre)
```

The first constraint set, `{ F : diag(F Fᵀ) = I_n }`, is the **oblique manifold**
`OB(3, n)` — the product of `n` copies of `S²`, *not* the Stiefel manifold. This
distinction determines which retraction is admissible (§4.4).

The second constraint is **not cosmetic**. `E` is invariant under Möbius
transformations of `S²`, and among those degrees of freedom is the collapse of the
entire map onto a single point, which has `E = 0` and is therefore the *global*
minimizer of `E` over `OB(3, n)`. The conformal map is only a critical point. Fixing
the Möbius gauge by requiring a zero (area-weighted) mass centre excludes the constant
map and makes the parameterization the solution actually being sought. It must be
enforced **at every iteration**, as part of the retraction — enforcing it only on the
converged iterate normalizes an already-collapsed map and yields nothing.

The degeneracy is reproducible: run the driver with `--no-mobius` and the energy
descends to `~1e-12` with 100 % of faces degenerate.

---

## 2. Repository Layout

```
Python_latest/
├── main.py                  Driver / CLI (was test.py)
├── visualize.py             Renders the source/sphere comparison figure
├── mesh.py                  Half-edge structure, OFF/OBJ parsers, vertex normals
├── optimization.py          Laplacian, energy, gradients, solvers, untangling, metrics
├── config.py                Global numerical tolerance (EPSILON = 1e-6)
├── halfedge_mesh_test.py    pytest suite for the half-edge structure
├── requirements.txt         numpy, scipy, pytest
├── setup.py / setup.cfg     Packaging
├── runtests.py              Vendored legacy py.test bootstrap (unused)
├── tests/data/              Input meshes: brain, bunny, teapot, cube (.off/.obj/.ply/.m)
└── output/                  Generated parameterizations (.obj) and viewer dumps (.txt)
```

---

## 3. Data Structures

### 3.1 Half-Edge Mesh (`mesh.py`)

The mesh is stored in a half-edge (doubly connected edge list) representation, giving
`O(1)` access to the one-ring neighbourhood of any vertex.

| Class | Fields |
|---|---|
| `HalfedgeMesh` | `vertices`, `halfedges`, `facets`, `edges: {(u,v) → Halfedge}` |
| `Vertex` | `x, y, z`, `index`, `halfedge`, `normal` |
| `Facet` | `a, b, c` (vertex indices, CCW), `index`, `halfedge` |
| `Halfedge` | `next`, `prev`, `opposite`, `vertex`, `facet`, `index` |

**Conventions**, relied upon throughout and identical on both construction paths:

* `Halfedge.vertex` is the half-edge's **target** vertex; the origin is
  `halfedge.prev.vertex`.
* `Vertex.halfedge` is an **outgoing** half-edge, so the one-ring is walked with
  `he = he.opposite.next`.
* `Facet` indices `(a, b, c)` are counter-clockwise.

**Construction.** OFF files are parsed by `parse_off` → `parse_build_halfedge_off`, a
two-pass algorithm: the first pass allocates one half-edge per directed edge and links
`next` / `prev` around each facet; the second pass pairs `(u,v)` with `(v,u)` to set
`opposite`. OBJ files take a different path (`parse_obj` → `build_halfedge`) that
constructs the same topology from an explicit vertex/facet list. A directed edge
appearing twice (non-manifold or inconsistently oriented input) is reported rather than
silently rebound.

### 3.2 Supported Input Formats

| Format | Parser | Status |
|---|---|---|
| `.off` | `parse_off` | Supported; triangles only (a non-triangular facet raises `ValueError`) |
| `.obj` | `parse_obj` | Supported; `v` / `f` records, `f v/vt/vn` indices tolerated |
| `.ply`, `.m` | — | Present in `tests/data/` but not parsed |

The two paths produce identical results: `brain.obj` and `brain_python.off` describe the
same surface and converge to the same energy to all printed digits.

---

## 4. Algorithm Pipeline

### 4.1 Order of Operations

```
load mesh
  └─ capture source coordinates P
     └─ assemble L from P                    <-- before any map is applied
        └─ build initial map F0 on S²         (separate array; P is never overwritten)
           └─ minimize E(F) subject to the constraints of §1.1
              └─ report energy + distortion, save
```

The Laplacian is assembled from the source geometry and the initial map is held in its
own array. Overwriting the vertex coordinates with the initial map before assembling
`L` would make the cotangent weights encode the metric of the *image* rather than of
the input surface, and `E` would no longer be the harmonic energy of `M`.

### 4.2 Discrete Laplace–Beltrami Operator

`construct_laplacian_matrix(mesh, coords=None, dense=False)` assembles `L` by
accumulating, for each triangle `(i, j, k)`, the weight `½ cot(angle at i)` onto edge
`(j, k)` — the angle **opposite** the edge — and symmetrically for the other two edges,
with the diagonal set to the negative row sum.

The result is symmetric, positive semi-definite, and satisfies `L·1 = 0`. It is
assembled vectorized over facets and returned as a **sparse CSR** matrix; on the brain
mesh that is 17 502 nonzeros (0.14 MB) instead of a dense 2502 × 2502 array (50.1 MB).

### 4.3 Initial Maps

`gauss_map(mesh)` returns the Gauss map, `F⁰_i = N(v_i) ∈ S²`, where `N(v)` is the
normalized sum of incident unit face normals. It is a valid degree-1 map for convex
surfaces; for non-convex genus-zero input it is non-injective, and on both test meshes
it starts with roughly 40–48 % of faces inverted.

`weighted_lb_eigen_projection(mesh)` returns a spectral embedding from the low
eigenvectors of the Laplace–Beltrami operator. It solves the **generalized** problem
`L v = λ M v` with `M` the lumped mass matrix — recast as the symmetric problem
`D^{-1/2} L D^{-1/2} u = λ u` with `v = D^{-1/2}u`, which is well conditioned since `M`
is diagonal and positive. Because eigenvector signs are arbitrary, the resulting
embedding can be mirrored, which inverts *every* triangle; `fix_global_orientation`
detects this and restores the handedness.

### 4.4 Solver A — Projected Gradient (default)

`fast_algorithm(...)`. The Euclidean gradient is `G = L F`; its projection onto the
tangent space of `OB(3, n)` removes the radial component row-wise,
`P_F(G)_i = G_i − ⟨G_i, F_i⟩ F_i`. The step is

```
F_{k+1} = R( F_k − τ_k · P_{F_k}(G_k) )
```

with `R` the retraction of §1.1 (row normalization, then mass-centre projection).

The trial step is a **Barzilai–Borwein** step, alternating BB1 and BB2 from
`S = F_{k+1} − F_k` and `Y = ∇_{k+1} − ∇_k`, then backtracked until the Zhang–Hager
**nonmonotone** Armijo condition

```
E(Y(τ)) ≤ C_k − ρ · τ · ‖∇E(F_k)‖²
```

holds, where the reference value follows `Q_{k+1} = ξ Q_k + 1`,
`C_{k+1} = (ξ Q_k C_k + E(F_{k+1})) / Q_{k+1}`. Termination is `‖∇E‖ ≤ ε`, energy
stagnation, or the iteration cap.

### 4.5 Solver B — Wen–Yin Cayley Curvilinear Search

`orthogonal_constrained_optimization(...)`. From `G = L X`, form the skew-symmetric
`A = G Xᵀ − X Gᵀ` and follow the Cayley curve

```
Y(τ) = ( I + (τ/2) A )⁻¹ ( I − (τ/2) A ) X
```

`A` has rank at most `2p` (`p = 3`), so with `U = [G  X]` and `V = [X  −G]` the
Sherman–Morrison–Woodbury identity gives

```
Y(τ) = X − τ U ( I_{2p} + (τ/2) VᵀU )⁻¹ Vᵀ X
```

a **6 × 6** solve instead of an `n × n` inverse. Measured on the brain mesh
(`n = 2502`): **575.6 ms → 0.17 ms per step, a 3338× speedup**, agreeing with the dense
formula to `2.7e-15`. Over 5 000 iterations that is 48 minutes versus 0.9 seconds.

Two details matter for correctness:

* **Retraction.** The Cayley transform is a retraction for the *Stiefel* manifold
  (`XᵀX = I`); our constraint is the oblique manifold. Left-multiplying by an orthogonal
  matrix preserves `‖X‖_F` but not the individual row norms, so the curve is used only
  to generate the search direction and the iterate is retracted back onto the
  constraint set afterwards.
* **Descent quantity.** The derivative of `E` along the Cayley curve at `τ = 0` is
  `−‖A‖_F²/2`, not `−‖∇E‖²`. The Armijo test uses `‖A‖_F²/2`, evaluated without forming
  `A` via `‖A‖_F² = 2 tr(XᵀX GᵀG) − 2 tr((GᵀX)²)` — only 3 × 3 products.

### 4.6 Untangling Pass (`--untangle`)

Harmonic energy does not see folds, so the flow has no reason to remove the last few
inverted faces. `untangle_local` adds an explicit one-sided penalty on the per-face
orientation `s_t = det[q_a, q_b, q_c]` — the scalar triple product, positive exactly
when the triangle is counter-clockwise seen from outside, with gradients as simple as
`∂s_t/∂q_a = q_b × q_c`:

```
P(F) = Σ_t max(0, δ − s_t)²
```

It grows a patch of a few vertex rings around each fold, pins the surrounding ring, and
minimizes `E(F) + w·P(F)` over the free vertices only, raising `w` geometrically until
no inverted face remains. Because the patch is a few dozen vertices, the conformality
already achieved elsewhere is untouched: on the brain the energy rises 0.09 % and mean
`|μ|` moves from 0.0551 to 0.0552, while inverted faces go 4 → 0. On the bunny the
distortion statistics actually *improve* (p95 `|μ|` 0.993 → 0.711), because the
collapsed faces were dragging them down.

### 4.7 Default Parameters

| Symbol | Flag | Default | Role |
|---|---|---|---|
| `ρ` | `--rho` | `1e-4` | Sufficient-decrease constant |
| `δ` | `--delta` | `0.5` | Backtracking shrink factor |
| `ξ` | `--xi` | `0.85` | Nonmonotone averaging weight |
| `ε` | `--tol` | `1e-6` | Gradient-norm stopping tolerance |
| — | `--max-iter` | `5000` | Iteration cap |
| — | `config.EPSILON` | `1e-6` | Global float comparison tolerance |

---

## 5. Usage

### Requirements

Python ≥ 3.8, `numpy`, `scipy`; `pytest` for the tests.

```bash
pip install -r Python_latest/requirements.txt
```

### Running

```bash
cd Python_latest
python main.py --input tests/data/brain_python.off --solver both --untangle
```

(PowerShell does not accept `&&`; use `;` or separate lines.)

| Flag | Meaning |
|---|---|
| `--input` | Input mesh (`.off` or `.obj`) |
| `--output-dir` | Destination for results (default `output`) |
| `--solver` | `fast` (default), `ortho`, or `both` |
| `--init` | `gauss` (default) or `spectral` |
| `--untangle` | Remove residual inverted faces (§4.6). **Required for a bijective result.** |
| `--untangle-steps` | Penalty-weight escalation steps (default 12). Raise this first if a fold survives. |
| `--untangle-rings` | Vertex rings around a fold that are free to move (default 1) |
| `--max-iter`, `--tol` | Iteration cap and gradient tolerance |
| `--rho`, `--delta`, `--xi` | Line-search parameters |
| `--no-mobius` | Drop the mass-centre constraint (diagnostic; see §1.1) |
| `--quiet` | Suppress per-iteration logging |

### Visualizing a result

```bash
cd Python_latest
python visualize.py --mapping output/spherical_mapping_fast.obj --out output/comparison.png
```

Four panels: a checkerboard defined in the spherical coordinates of the image and
pulled back onto the source by vertex correspondence — the direct visual test of
conformality, since a conformal map carries the cells to cells that change *size* but
not *shape* — and both surfaces coloured by `|μ|`. The `|μ|` colour range is
auto-scaled to the 99th percentile of each map, so **do not compare colours between two
figures**; compare the numbers in the titles.

### Tests

```bash
cd Python_latest
pytest halfedge_mesh_test.py
```

23 tests covering half-edge connectivity invariants (`next`/`prev`/`opposite`
consistency, one-ring traversal, facet normals, dihedral angles) on the cube and bunny
fixtures, plus the vector helpers.

---

## 6. Outputs

| File | Content |
|---|---|
| `normals.obj` | Vertex normals as points, with the input connectivity |
| `gauss_map.obj` | Initial map `F⁰` on `S²` |
| `spherical_mapping_fast.obj` | Result of Solver A (§4.4) |
| `spherical_mapping_ortho.obj` | Result of Solver B (§4.5) |
| `g3dogl_*.txt` | Vertex/normal dumps in G3dOGL viewer format |

All `.obj` outputs preserve the input connectivity, so the source mesh and its
parameterization are in vertex-index correspondence and can be overlaid directly.

---

## 7. Distortion Metrics

The pipeline reports map quality rather than only the energy value.

**Quasi-conformal distortion.** `beltrami_coefficient` flattens each triangle
isometrically in the source and in the image, extracts the Jacobian `J` of the affine
map between them, and forms

```
f_z  = ((J₁₁ + J₂₂) + i (J₂₁ − J₁₂)) / 2
f_z̄  = ((J₁₁ − J₂₂) + i (J₂₁ + J₁₂)) / 2
μ    = f_z̄ / f_z
```

`|μ| = 0` means the triangle is mapped conformally; the quasi-conformal dilatation is
`K = (1 + |μ|)/(1 − |μ|)`. The image frame is oriented by the outward radial direction
at the face centroid, so an inverted triangle produces `|μ| ≥ 1` instead of being
silently indistinguishable from a correctly oriented one.

**Area distortion.** `area_distortion` reports `log` of the ratio of per-face areas,
each normalized by its total, so an area-preserving map gives 0 everywhere.

`K` is summarized by its median: it diverges on near-degenerate faces, so the mean is
dominated by a handful of them and says nothing about the bulk.

**Degree.** `map_degree` sums the signed solid angle of every spherical triangle
(Van Oosterom–Strackee) and divides by `4π`. A valid parameterization must come out at
exactly 1; degree 2 means the image wraps the sphere twice.

**Conformality gap.** On each triangle the piecewise-linear map has Jacobian singular
values `σ₁, σ₂`, giving Dirichlet energy `(σ₁² + σ₂²)/2` and image area `σ₁σ₂` per unit
source area. By AM-GM,

```
E(f) ≥ Area(f(M))
```

with equality **exactly** when every triangle is mapped conformally, so

```
gap = E(f) / Area(f(M)) − 1  ≥ 0
```

is a single scale-free number measuring distance from conformality, independent of mesh
scale and of the choice of distortion statistic. It is printed on every reported map.

This is the sharpest available check on the implementation, because it ties an energy
computed from the cotangent Laplacian to a purely geometric quantity computed from the
image — two code paths that share nothing.

> **The denominator is the image area, not `4π`.** The image triangles are flat chords
> of the sphere, so an inscribed polyhedron always falls short of `4π` — by 3.0 % on the
> 500-face bunny and 0.16 % on the 5000-face brain. Normalizing by `4π` folds that
> discretization deficit into the measure and makes a perfectly valid coarse map appear
> to violate the bound: the untangled, degree-1, fold-free bunny has `E = 12.528` while
> `4π = 12.566`. Against its actual image area of `12.192` the gap is `+0.027`, as it
> must be.

---

## 8. Results

Gauss-map initialization, default parameters, single-threaded on a laptop CPU.
Timings vary by roughly 2× run to run on a loaded machine.

| Mesh | Stage | Time | `E(F)` | deg | gap | mean \|μ\| | median `K` | inverted faces |
|---|---|---|---|---|---|---|---|---|
| bunny (252 v / 500 f) | initial | — | `1.642e+02` | 1.000 | +1.647 | 0.7190 | 12.48 | 205 / 500 (41.0 %) |
| | Solver A | 0.24 s | `1.25258e+01` | 1.000 | +0.0274 | 0.2263 | 1.274 | 22 / 500 (4.4 %) |
| | + untangle | 1.24 s | `1.25260e+01` | 1.000 | +0.0274 | 0.1999 | 1.278 | **0 / 500** |
| | Solver B | 0.41 s | `1.37207e+01` | 1.000 | +0.1095 | 0.3143 | 1.545 | 32 / 500 (6.4 %) |
| | + untangle | 1.09 s | `1.40108e+01` | 1.000 | +0.1419 | 0.2831 | 1.538 | **0 / 500** |
| brain (2502 v / 5000 f) | initial | — | `5.239e+02` | **2.000** | +1.036 | 0.7676 | 23.85 | 2395 / 5000 (47.9 %) |
| | Solver A | 3.19 s | `1.26762e+01` | 1.000 | +0.0103 | 0.0551 | 1.089 | 4 / 5000 (0.08 %) |
| | + untangle | 0.88 s | `1.26936e+01` | 1.000 | +0.0117 | 0.0552 | 1.089 | **0 / 5000** |
| | Solver B | 10.06 s | `1.26996e+01` | 1.000 | +0.0122 | 0.0641 | 1.111 | 4 / 5000 (0.08 %) |
| | + untangle | 0.87 s | `1.27179e+01` | 1.000 | +0.0136 | 0.0640 | 1.111 | **0 / 5000** |

Three independent correctness signals:

1. **The conformality gap lands just above zero.** After untangling it is `+0.012`
   (brain) and `+0.027` (bunny) — within a few percent of a bound that only a conformal
   map attains, and on the correct side of it everywhere. The energy comes from the
   cotangent Laplacian; the bound comes from the image geometry. Nothing forces them
   to agree.

2. **The two solvers agree.** Different search directions, retractions, and
   line-search quantities, converging on the brain to `1.26762e+01` and `1.26996e+01`
   — a 0.18 % spread.

3. **The two parsers agree.** `brain.obj` and `brain_python.off` take entirely separate
   construction paths and converge to the same energy to all printed digits.

One entry above is a diagnostic rather than a result: the brain's Gauss map starts at
**degree 2**, wrapping the sphere twice, which is the origin of its 47.9 % inverted
faces. The flow recovers degree 1 from it.

The gap is a *conformality* measure and does not rank maps by usefulness. Untangling the
bunny under Solver B raises the gap (`+0.109 → +0.142`) while removing all 32 folds: the
result is less conformal and more usable, because a folded map is not a parameterization
at any gap.

---

## 9. Known Limitations

**Bijectivity requires the `--untangle` post-pass.** The flow alone leaves 4 of 5000
faces inverted on the brain and 22 of 500 on the bunny, because harmonic energy is
blind to folding: a folded configuration can have perfectly low energy, so nothing in
the objective drives the last folds out. `--untangle` closes that gap (§4.6) and both
meshes then verify at degree exactly 1 with all faces positively oriented. **Without
the flag the output is not a valid parameterization**, however good its `|μ|`
statistics look.

**The bunny result is substantially worse than the brain result** (mean `|μ|` 0.20 vs
0.055). At 252 vertices the mesh is coarse relative to its curvature, and the Gauss map
starts 41 % inverted.

**The untangling pass is local and greedy.** It pins a ring around each fold and pushes
the enclosed faces positive under a penalty, with the weight raised until no inverted
face remains. That converged on both test meshes, but it is not guaranteed to: a fold
whose removal needs the pinned boundary to move would defeat it. A fold-free
initialization (Tutte embedding of the cut disk, then inverse stereographic projection)
would remove the need for the pass entirely.

Of its two knobs, **`--untangle-steps` is the one that decides success**, not
`--untangle-rings`. Sweeping both on the bunny (22 residual folds) and on deliberately
under-converged runs, every observed failure was a penalty weight that never got large
enough, and every one was fixed by allowing more weight escalation rather than a wider
patch — `rings=0` fails at 6 escalation steps and succeeds at 10. Widening the patch
costs time and lets the penalty perturb more of an already-good map: on the bunny,
`rings=5` raised the energy by 10 % where `rings=1` raised it by 0.002 %. Raise
`--untangle-steps` first; reach for `--untangle-rings` only if that does not help.

**Spectral initialization is worse than the Gauss map here**, despite starting at a much
lower energy (20.7 vs 164 on the bunny). It begins in a heavily folded configuration
(25 % of faces inverted even after the handedness fix) and the flow settles into a
folded local minimum — final mean `|μ|` 0.70 versus 0.23 from the Gauss map. It is kept
behind `--init spectral` as a comparison point, not as a recommended default.

**Area distortion is not optimized.** The Möbius gauge is fixed by the zero mass centre,
which is one admissible choice among many; the remaining Möbius freedom could be spent
minimizing area distortion instead, which is what area-preserving spherical
parameterization methods do.

**Complexity.** The Laplacian is sparse and each Cayley step is `O(n p²)`, so the
per-iteration cost is linear in `n`. `weighted_lb_eigen_projection` falls back to a
dense `eigh` for meshes smaller than the requested eigenvector count, and
`report_distortion` is `O(m)` dense per call.

**Structural.** `runtests.py` is a vendored legacy py.test bootstrap that is not part of
the build and can be removed. `compute_laplacian(vertex)` was dead and buggy and has
been deleted. The `output/spherical_mapping_{1..4}.obj` and `*_ortho_{1,2,3}.obj`
artifacts committed to the repository are stale — they came from a smoothing loop that
no longer exists.

---

## 10. References

1. X. Gu, Y. Wang, T. F. Chan, P. M. Thompson, S.-T. Yau. *Genus Zero Surface Conformal
   Mapping and Its Application to Brain Surface Mapping.* IEEE Transactions on Medical
   Imaging, 23(8), 2004. — Harmonic-energy formulation, and the zero-mass-centre
   Möbius normalization of §1.1.
2. Z. Wen, W. Yin. *A Feasible Method for Optimization with Orthogonality Constraints.*
   Mathematical Programming, 142, 2013. — Cayley curvilinear search, the low-rank
   `A = UVᵀ` factorization, and the BB step.
3. H. Zhang, W. W. Hager. *A Nonmonotone Line Search Technique and Its Application to
   Unconstrained Optimization.* SIAM Journal on Optimization, 14(4), 2004. — The
   `Q_k` / `C_k` reference-value recursion.
4. M. Meyer, M. Desbrun, P. Schröder, A. H. Barr. *Discrete Differential-Geometry
   Operators for Triangulated 2-Manifolds.* Visualization and Mathematics III, 2003. —
   Cotangent Laplacian and lumped-mass weights.
5. P.-A. Absil, R. Mahony, R. Sepulchre. *Optimization Algorithms on Matrix Manifolds.*
   Princeton University Press, 2008. — Retractions on the oblique and Stiefel manifolds.
6. J. Barzilai, J. M. Borwein. *Two-Point Step Size Gradient Methods.* IMA Journal of
   Numerical Analysis, 8(1), 1988.

---

## 11. Attribution and License

The half-edge data structure and its OFF parser derive from
[`halfedge_mesh`](https://github.com/carlosrojas/halfedge_mesh) by Carlos Rojas, used
under the MIT License (see `Python_latest/LICENSE.md`). The parameterization,
Laplacian assembly, energy, optimization, and distortion code in `optimization.py` are
original to this project.
