# optimization.py
"""Spherical conformal parameterization by harmonic energy minimization.

The map from a closed genus-zero mesh M to S^2 is represented by a coordinate
matrix F in R^{n x 3} whose i-th row is the image of vertex i.  The harmonic
energy is the quadratic form

    E(F) = 1/2 tr(F^T L F)

with L the cotangent Laplace-Beltrami operator of the *source* mesh.  The
feasible set is the oblique manifold

    OB(3, n) = { F in R^{n x 3} : ||F_i|| = 1 for all i }

i.e. the product of n copies of S^2 -- not the Stiefel manifold.  See the
retraction notes on `cayley_step` for why that distinction matters.
"""
import config
import functools
import math
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def vertex_coordinates(mesh):
    """Return the mesh vertex positions as an (n, 3) float array."""
    return np.array([[v.x, v.y, v.z] for v in mesh.vertices], dtype=float)


def facet_indices(mesh):
    """Return the triangle vertex indices as an (m, 3) int array."""
    return np.array([[f.a, f.b, f.c] for f in mesh.facets], dtype=np.intp)


def facet_areas(mesh, coords=None):
    """Return the area of every triangle as an (m,) array."""
    P = vertex_coordinates(mesh) if coords is None else np.asarray(coords, dtype=float)
    T = facet_indices(mesh)
    cross = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
    return 0.5 * np.linalg.norm(cross, axis=1)


def vertex_areas(mesh, coords=None):
    """Lumped (barycentric) vertex areas: one third of each incident triangle.

    Computed by accumulation over facets, so it is valid for meshes with
    boundary and does not depend on the half-edge one-ring traversal.
    """
    T = facet_indices(mesh)
    areas = facet_areas(mesh, coords)
    lumped = np.zeros(len(mesh.vertices))
    np.add.at(lumped, T.ravel(), np.repeat(areas / 3.0, 3))
    return lumped


def compute_area_weight_matrix(mesh, coords=None):
    """Lumped mass matrix M (diagonal, sparse) of the mesh."""
    return sp.diags(vertex_areas(mesh, coords), format='csr')


def compute_facet_area(facet):
    """Area of a single triangle, from its half-edge loop."""
    he1 = facet.halfedge
    he2 = he1.next
    he3 = he2.next
    p1 = np.array(he1.vertex.get_vertex())
    p2 = np.array(he2.vertex.get_vertex())
    p3 = np.array(he3.vertex.get_vertex())
    return 0.5 * float(np.linalg.norm(np.cross(p2 - p1, p3 - p1)))


def compute_vertex_area(vertex, mesh=None):
    """Lumped area of one vertex, by walking its one-ring.

    Terminates on boundary (missing `opposite`) instead of dereferencing None.
    Prefer `vertex_areas(mesh)` for anything vectorized.
    """
    area = 0.0
    start = vertex.halfedge
    if start is None:
        return 0.0
    current = start
    while True:
        if current.facet is not None:
            area += compute_facet_area(current.facet) / 3.0
        if current.opposite is None or current.opposite.next is None:
            break                       # boundary: one-ring is not a closed fan
        current = current.opposite.next
        if current is start:
            break
    return area


# ---------------------------------------------------------------------------
# Discrete Laplace-Beltrami operator
# ---------------------------------------------------------------------------

def cotangent_at_vertex(apex, p1, p2):
    """Cotangent of the triangle angle at `apex`, subtended by p1 and p2.

    apex, p1, p2 - Vertex instances (or anything exposing get_vertex()).

    In a triangle (i, j, k) the cotangent weight of edge (j, k) is the
    cotangent of the angle at the *opposite* vertex i, i.e.
    cotangent_at_vertex(v_i, v_j, v_k).
    """
    a = np.asarray(apex.get_vertex(), dtype=float)
    u = np.asarray(p1.get_vertex(), dtype=float) - a
    v = np.asarray(p2.get_vertex(), dtype=float) - a
    sine = np.linalg.norm(np.cross(u, v))
    if sine < config.EPSILON:
        return 0.0                      # degenerate triangle
    return float(np.dot(u, v) / sine)


def construct_laplacian_matrix(mesh, coords=None, dense=False):
    """Assemble the cotangent Laplace-Beltrami operator of `mesh`.

    For every triangle (i, j, k) the weight of edge (j, k) is
    1/2 * cot(angle at i), accumulated over the (one or two) incident
    triangles.  The result is symmetric positive semi-definite with
    L @ ones == 0.

    coords - optional (n, 3) array of vertex positions.  Pass this to build L
             from the *source* geometry when the mesh coordinates have since
             been overwritten (e.g. by an initial map onto the sphere).
    dense  - return a dense ndarray instead of a sparse CSR matrix.

    Returns a scipy.sparse.csr_matrix (or ndarray if dense=True).
    """
    P = vertex_coordinates(mesh) if coords is None else np.asarray(coords, dtype=float)
    T = facet_indices(mesh)
    n = P.shape[0]

    i, j, k = T[:, 0], T[:, 1], T[:, 2]

    def cot(apex, b, c):
        """Cotangent of the angle at `apex`, vectorized over all facets."""
        u = P[b] - P[apex]
        v = P[c] - P[apex]
        sine = np.linalg.norm(np.cross(u, v), axis=1)
        out = np.zeros_like(sine)
        ok = sine > config.EPSILON
        out[ok] = np.einsum('ij,ij->i', u[ok], v[ok]) / sine[ok]
        return out

    # weight of an edge = 1/2 cot(angle at the vertex opposite that edge)
    w_jk = 0.5 * cot(i, j, k)
    w_ki = 0.5 * cot(j, k, i)
    w_ij = 0.5 * cot(k, i, j)

    rows = np.concatenate([j, k, k, i, i, j])
    cols = np.concatenate([k, j, i, k, j, i])
    vals = np.concatenate([-w_jk, -w_jk, -w_ki, -w_ki, -w_ij, -w_ij])

    # diagonal: negative row sum, so that L @ ones == 0
    diag_rows = np.concatenate([j, k, k, i, i, j])
    diag_vals = np.concatenate([w_jk, w_jk, w_ki, w_ki, w_ij, w_ij])

    rows = np.concatenate([rows, diag_rows])
    cols = np.concatenate([cols, diag_rows])
    vals = np.concatenate([vals, diag_vals])

    L = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    L.sum_duplicates()
    return L.toarray() if dense else L


# ---------------------------------------------------------------------------
# Energy, gradient, retraction
# ---------------------------------------------------------------------------

def E(F, L):
    """Harmonic energy E(F) = 1/2 tr(F^T L F).  Works for sparse or dense L."""
    return 0.5 * float(np.sum(F * (L @ F)))


def compute_harmonic_energy(mesh, L=None):
    """Harmonic energy of the mesh's current vertex positions."""
    F = vertex_coordinates(mesh)
    if L is None:
        L = construct_laplacian_matrix(mesh)
    return E(F, L)


def project_gradient_to_sphere(grad, F):
    """Project a Euclidean gradient onto the tangent space of OB(3, n).

    Row-wise: g_i <- g_i - <g_i, F_i> F_i.
    """
    return grad - np.sum(grad * F, axis=1, keepdims=True) * F


def compute_gradient(F, L):
    """Riemannian gradient of E on the oblique manifold at F."""
    return project_gradient_to_sphere(L @ F, F)


def retract(X):
    """Retraction onto OB(3, n): normalize each row to unit length."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, config.EPSILON)
    return X / norms


def normalize_mass_center(F, weights=None, max_iter=100, tol=1e-10):
    """Zero the (area-weighted) mass centre of a map onto S^2.

    This is the Gu-Yau conformal gauge, and it is not cosmetic: the harmonic
    energy is invariant under Mobius transformations of S^2, and among those
    degrees of freedom is a collapse of the whole map onto a single point,
    which has E = 0 and is therefore the *global* minimizer of E over the
    oblique manifold.  The conformal map is only a critical point.  Requiring

        sum_i w_i F_i = 0

    excludes the constant map and makes the conformal parameterization the
    solution actually being sought -- so this must be applied as part of the
    retraction at every iteration, not once at the end.
    """
    F = retract(np.array(F, dtype=float, copy=True))
    w = np.ones(len(F)) if weights is None else np.asarray(weights, dtype=float)
    w = w / w.sum()
    for _ in range(max_iter):
        centre = (w[:, None] * F).sum(axis=0)
        if np.linalg.norm(centre) < tol:
            break
        F = retract(F - centre)
    return F


def make_retraction(weights=None, mobius=True, max_iter=20, tol=1e-12):
    """Build the retraction used by the solvers.

    With mobius=True the retraction is 'normalize rows, then zero the mass
    centre', which keeps the iterates on the constraint set
    { ||F_i|| = 1, sum_i w_i F_i = 0 } and so keeps them away from the
    degenerate constant map.  See `normalize_mass_center`.
    """
    if not mobius:
        return retract

    def retraction(X):
        return normalize_mass_center(X, weights, max_iter=max_iter, tol=tol)

    return retraction


# ---------------------------------------------------------------------------
# Initial maps
# ---------------------------------------------------------------------------

def gauss_map(mesh):
    """Initial map F0: send each vertex to its unit normal on S^2.

    Returns an (n, 3) array.  The mesh's vertex *coordinates* are left intact
    -- the source geometry is needed to build the Laplacian, so the initial map
    must not be written back over it.
    """
    mesh.calculate_vertex_normals()
    F0 = np.array([v.normal for v in mesh.vertices], dtype=float)
    degenerate = np.linalg.norm(F0, axis=1) < config.EPSILON
    if np.any(degenerate):
        print("Warning: %d vertices have a near-zero normal; "
              "falling back to their normalized position."
              % int(degenerate.sum()))
        P = vertex_coordinates(mesh)
        F0[degenerate] = P[degenerate]
    return retract(F0)


def fix_global_orientation(mesh, F):
    """Flip the handedness of a spherical map if most faces face inward.

    A map onto S^2 built from eigenvectors carries an arbitrary sign per
    coordinate, so the embedding can come out mirrored -- which inverts *every*
    triangle and reads as near-total flipping.  Negating one coordinate
    restores the handedness without changing the harmonic energy (E is
    invariant under orthogonal transformations of the image).
    """
    F = np.asarray(F, dtype=float)
    T = facet_indices(mesh)
    a, b, c = F[T[:, 0]], F[T[:, 1]], F[T[:, 2]]
    outward = np.einsum('ij,ij->i', np.cross(b - a, c - a), (a + b + c) / 3.0)
    if np.count_nonzero(outward > 0) < np.count_nonzero(outward < 0):
        F = F.copy()
        F[:, 2] *= -1.0
    return F


def weighted_lb_eigen_projection(mesh, L=None, num_eigenvectors=3, tol=1e-8):
    """Spectral initial map from the low eigenvectors of the LB operator.

    Solves the *generalized* eigenproblem  L v = lambda M v  with M the lumped
    mass matrix, which is the correct discretization of the Laplace-Beltrami
    spectrum (the plain problem L v = lambda v ignores the metric).  Solved as
    the symmetric problem  D^{-1/2} L D^{-1/2} u = lambda u  with v = D^{-1/2}u,
    which is well conditioned because M is diagonal and positive.

    Returns an (n, 3) array on S^2 built from eigenvectors 2..4 (the first is
    the constant kernel vector).
    """
    if L is None:
        L = construct_laplacian_matrix(mesh)
    L = sp.csr_matrix(L)
    m = vertex_areas(mesh)
    m = np.maximum(m, config.EPSILON)
    Dinv = sp.diags(1.0 / np.sqrt(m), format='csr')
    A = Dinv @ L @ Dinv
    A = 0.5 * (A + A.T)                 # enforce exact symmetry

    k = num_eigenvectors + 1
    if k >= A.shape[0] - 1:             # tiny mesh: fall back to dense
        vals, vecs = np.linalg.eigh(A.toarray())
    else:
        vals, vecs = spla.eigsh(A, k=k, sigma=-1e-6, which='LM', tol=tol)
        order = np.argsort(vals)
        vals, vecs = vals[order], vecs[:, order]

    V = Dinv @ vecs[:, 1:num_eigenvectors + 1]
    return fix_global_orientation(mesh, retract(V))


# ---------------------------------------------------------------------------
# Solver A -- projected gradient descent, BB step + nonmonotone line search
# ---------------------------------------------------------------------------

def _bb_step(S, Y, k, lo=1e-10, hi=1e10):
    """Barzilai-Borwein trial step, alternating BB1 / BB2."""
    sy = abs(float(np.sum(S * Y)))
    if sy < 1e-30:
        return 1.0
    if k % 2 == 0:
        tau = float(np.sum(S * S)) / sy
    else:
        tau = sy / float(np.sum(Y * Y))
    if not np.isfinite(tau):
        return 1.0
    return float(np.clip(tau, lo, hi))


def gradient_step(F, tau, L, grad=None, retraction=retract):
    """One projected-gradient step followed by the retraction."""
    g = compute_gradient(F, L) if grad is None else grad
    return retraction(F - tau * g)


# `Y` is the name used in the original code and in the paper's notation.
Y = gradient_step


def fast_algorithm(F0, rho, delta, xi, epsilon, mesh, max_iter=1000, L=None,
                   retraction=None, weights=None, verbose=True, print_every=100):
    """Minimize E on OB(3, n) by projected gradient descent.

    Step length: Barzilai-Borwein trial step, then backtracking until the
    Zhang-Hager *nonmonotone* Armijo condition

        E(Y(tau)) <= C_k - rho * tau * ||grad E(F_k)||^2

    is met, with the reference value C_k updated by

        Q_{k+1} = xi Q_k + 1
        C_{k+1} = (xi Q_k C_k + E(F_{k+1})) / Q_{k+1}

    rho     - sufficient-decrease constant in (0, 1)
    delta   - backtracking shrink factor in (0, 1)
    xi      - nonmonotone averaging weight in [0, 1]
    epsilon - stopping tolerance on ||grad E||

    retraction - map back onto the constraint set.  Defaults to the
        mass-centre-normalizing retraction, which is what keeps the iteration
        from collapsing onto the degenerate constant map (see
        `normalize_mass_center`).  Pass `retract` for row normalization only.

    Returns the optimized (n, 3) map.
    """
    if L is None:
        L = construct_laplacian_matrix(mesh)
    if retraction is None:
        retraction = make_retraction(weights)

    F_k = retraction(np.array(F0, dtype=float, copy=True))
    G_k = compute_gradient(F_k, L)
    Q_k = 1.0
    C_k = E_k = E(F_k, L)
    tau = 1.0 / max(np.linalg.norm(G_k), 1.0)

    for k in range(max_iter):
        gnorm_sq = float(np.sum(G_k * G_k))
        gnorm = math.sqrt(gnorm_sq)
        if gnorm <= epsilon:
            if verbose:
                print("Converged after %d iterations (||grad|| = %.3e)." % (k, gnorm))
            break

        # backtracking on the nonmonotone reference value C_k
        tau_k = tau
        for _ in range(30):
            F_next = gradient_step(F_k, tau_k, L, grad=G_k, retraction=retraction)
            E_next = E(F_next, L)
            if E_next <= C_k - rho * tau_k * gnorm_sq:
                break
            tau_k *= delta
        else:
            if verbose:
                print("Line search stalled at iteration %d (||grad|| = %.3e)."
                      % (k, gnorm))
            break

        G_next = compute_gradient(F_next, L)
        tau = _bb_step(F_next - F_k, G_next - G_k, k)

        Q_next = xi * Q_k + 1.0
        C_k = (xi * Q_k * C_k + E_next) / Q_next
        Q_k = Q_next
        if abs(E_k - E_next) <= 1e-14 * max(1.0, abs(E_k)):
            F_k, G_k = F_next, G_next
            if verbose:
                print("Energy stagnated after %d iterations (E = %.8e)." % (k, E_next))
            break
        F_k, G_k, E_k = F_next, G_next, E_next

        if verbose and k % print_every == 0:
            print("  iter %5d   E = %.8e   ||grad|| = %.3e   tau = %.3e"
                  % (k, E_next, gnorm, tau_k))
    else:
        if verbose:
            print("Reached the iteration cap (%d) with ||grad|| = %.3e."
                  % (max_iter, np.linalg.norm(G_k)))

    return F_k


# ---------------------------------------------------------------------------
# Solver B -- Wen-Yin curvilinear (Cayley) search
# ---------------------------------------------------------------------------

def cayley_step(F, G, tau):
    """Wen-Yin curvilinear step along the Cayley curve.

    With A = G F^T - F G^T (skew-symmetric, n x n), the curve is

        Y(tau) = (I + tau/2 A)^{-1} (I - tau/2 A) F

    Forming A explicitly costs O(n^2) memory and each step an O(n^3) inverse.
    A has rank at most 2p (p = 3 here), A = U V^T with U = [G  F] and
    V = [F  -G], so by Sherman-Morrison-Woodbury

        Y(tau) = F - tau U (I_{2p} + tau/2 V^T U)^{-1} V^T F

    which is O(n p^2): a 6x6 solve instead of an n x n inverse.

    NOTE ON THE RETRACTION.  The Cayley transform is a retraction for the
    *Stiefel* manifold (F^T F = I).  Our constraint is the oblique manifold
    (each row of unit norm), and left-multiplying by an orthogonal matrix
    preserves ||F||_F but not the individual row norms.  The caller must
    therefore renormalize rows afterwards; `orthogonal_constrained_optimization`
    does so via `retract`.
    """
    n, p = F.shape
    U = np.hstack([G, F])                    # n x 2p
    V = np.hstack([F, -G])                   # n x 2p
    VtU = V.T @ U                            # 2p x 2p
    M = np.eye(2 * p) + (tau / 2.0) * VtU
    return F - tau * (U @ np.linalg.solve(M, V.T @ F))


def _cayley_descent(X, G):
    """||A||_F^2 for A = G X^T - X G^T, without forming the n x n matrix A.

    The derivative of E along the Cayley curve at tau = 0 is -||A||_F^2 / 2,
    so this -- not ||grad E||^2 -- is the quantity the Armijo test must use.
    Expanding the trace and using cyclicity leaves only 3x3 products:

        ||A||_F^2 = 2 tr(X^T X G^T G) - 2 tr((G^T X)^2)
    """
    XtX = X.T @ X
    GtG = G.T @ G
    GtX = G.T @ X
    return 2.0 * float(np.trace(XtX @ GtG)) - 2.0 * float(np.trace(GtX @ GtX))


def orthogonal_constrained_optimization(mesh, F0, tolerance, max_iterations=1000,
                                        initial_step_size=0.1, beta=0.9, rho=1e-4,
                                        xi=0.9, L=None, retraction=None,
                                        weights=None, verbose=True,
                                        print_every=100):
    """Minimize E via the Wen-Yin curvilinear search, retracted onto OB(3, n).

    Each iteration follows the Cayley curve of `cayley_step`, retracts back
    onto the constraint set, and accepts the step under the Zhang-Hager
    nonmonotone condition

        E(Y(tau)) <= C_k - rho * tau * ||A||_F^2 / 2

    The step size grows by (1 + beta) on acceptance and shrinks per backtrack
    on rejection.
    """
    if L is None:
        L = construct_laplacian_matrix(mesh)
    if retraction is None:
        retraction = make_retraction(weights)

    X = retraction(np.array(F0, dtype=float, copy=True))
    step_size = initial_step_size
    Q_k = 1.0
    C_k = E_k = E(X, L)

    for k in range(max_iterations):
        G = L @ X
        grad = project_gradient_to_sphere(G, X)
        gnorm = float(np.linalg.norm(grad))
        if gnorm < tolerance:
            if verbose:
                print("Converged after %d iterations (||grad|| = %.3e)." % (k, gnorm))
            break

        # descent rate along the Cayley curve, not along -grad
        descent = max(_cayley_descent(X, G), 0.0) / 2.0
        if descent <= 0.0:
            if verbose:
                print("Cayley direction is degenerate at iteration %d; stopping." % k)
            break

        tau = step_size
        for _ in range(30):
            X_next = retraction(cayley_step(X, G, tau))
            E_next = E(X_next, L)
            if E_next <= C_k - rho * tau * descent:
                break
            tau *= 0.5
        else:
            if verbose:
                print("Line search stalled at iteration %d (||grad|| = %.3e)."
                      % (k, gnorm))
            break

        Q_next = xi * Q_k + 1.0
        C_k = (xi * Q_k * C_k + E_next) / Q_next
        Q_k = Q_next
        X = X_next
        step_size = tau * (1.0 + beta)

        if verbose and k % print_every == 0:
            print("  iter %5d   E = %.8e   ||grad|| = %.3e   tau = %.3e"
                  % (k, E_next, gnorm, tau))

        if abs(E_k - E_next) <= 1e-14 * max(1.0, abs(E_k)):
            if verbose:
                print("Energy stagnated after %d iterations (E = %.8e)." % (k, E_next))
            break
        E_k = E_next
    else:
        if verbose:
            print("Reached the iteration cap (%d)." % max_iterations)

    return X


# ---------------------------------------------------------------------------
# Local untangling
# ---------------------------------------------------------------------------

def signed_orientation(F, T):
    """Per-face orientation measure of a map onto S^2: det[q_a, q_b, q_c].

    For unit vectors this scalar triple product is positive exactly when the
    triangle is counter-clockwise seen from outside the sphere, and it is six
    times the signed volume of the tetrahedron (origin, q_a, q_b, q_c).  It
    vanishes as the triangle degenerates, so it doubles as the quantity a
    fold-prevention penalty should act on.

    Its gradients are simply  d/dq_a = q_b x q_c  (and cyclically).
    """
    a, b, c = F[T[:, 0]], F[T[:, 1]], F[T[:, 2]]
    return np.einsum('ij,ij->i', np.cross(a, b), c)


def _grow_patch(T, seed_faces, rings):
    """Vertices within `rings` of the seed faces (free), the faces they touch,
    and the surrounding vertices that pin that patch in place (fixed)."""
    n_v = int(T.max()) + 1
    free = np.zeros(n_v, bool)
    free[T[seed_faces].ravel()] = True
    for _ in range(max(rings, 0)):
        touched = free[T].any(axis=1)
        free[T[touched].ravel()] = True
    faces = np.where(free[T].any(axis=1))[0]
    fixed = np.zeros(n_v, bool)
    fixed[T[faces].ravel()] = True
    fixed &= ~free
    return np.where(free)[0], faces, np.where(fixed)[0]


def untangle_local(mesh, F, L=None, rings=1, max_outer=12, max_iter=400,
                   weight=1.0, delta_frac=0.5, verbose=True):
    """Remove residual inverted faces by moving a small neighbourhood of them.

    Harmonic energy is blind to folding -- a folded configuration can perfectly
    well have low energy -- so nothing in the main flow drives out the last few
    inverted faces.  This pass adds an explicit one-sided penalty

        P(F) = sum_t  max(0, delta - s_t)^2 ,    s_t = det[q_a, q_b, q_c]

    which is zero once every face is comfortably positively oriented, and
    minimizes  E(F) + w P(F)  over the free vertices only, with the surrounding
    ring pinned.  `w` is raised geometrically until no inverted face remains,
    so the fix stays as gentle as it can be.

    Because the patch is a few dozen vertices out of thousands, the conformality
    already achieved elsewhere is left untouched.

    Of the two knobs, `max_outer` is the one that decides success or failure --
    a fold that survives is nearly always a weight that never got large enough,
    not a patch that was too small.  Widening `rings` instead costs time and
    lets the penalty perturb more of an already-good map, so it is worth raising
    only when raising `max_outer` does not help.

    Returns the corrected (n, 3) map.
    """
    if L is None:
        L = construct_laplacian_matrix(mesh)
    T = facet_indices(mesh)
    F = np.array(F, dtype=float, copy=True)

    seed = np.where(signed_orientation(F, T) <= 0.0)[0]
    if seed.size == 0:
        if verbose:
            print("Untangle: no inverted faces; nothing to do.")
        return F

    free, faces, fixed = _grow_patch(T, seed, rings)
    Tp = T[faces]
    pos = signed_orientation(F, Tp)
    pos = pos[pos > 0]
    delta = delta_frac * (np.median(pos) if pos.size else 1e-3)
    if verbose:
        print("Untangle: %d inverted face(s) -> patch of %d faces, "
              "%d free vertices, %d pinned" % (seed.size, len(faces), len(free), len(fixed)))

    movable = np.zeros(len(F), bool)
    movable[free] = True

    def penalty_and_grad(X):
        s = signed_orientation(X, Tp)
        viol = np.maximum(0.0, delta - s)
        P = float(np.sum(viol ** 2))
        g = np.zeros_like(X)
        if P > 0.0:
            a, b, c = X[Tp[:, 0]], X[Tp[:, 1]], X[Tp[:, 2]]
            coef = (-2.0 * viol)[:, None]
            np.add.at(g, Tp[:, 0], coef * np.cross(b, c))
            np.add.at(g, Tp[:, 1], coef * np.cross(c, a))
            np.add.at(g, Tp[:, 2], coef * np.cross(a, b))
        return P, g

    def step_only_free(X, direction, tau):
        Y = X.copy()
        Y[movable] = X[movable] - tau * direction[movable]
        Y[movable] = retract(Y[movable])
        return Y

    w = weight
    for outer in range(max_outer):
        tau = 1.0
        for _ in range(max_iter):
            P, gP = penalty_and_grad(F)
            if P <= 0.0:
                break
            grad = project_gradient_to_sphere(L @ F + w * gP, F)
            gnorm_sq = float(np.sum(grad[movable] ** 2))
            if gnorm_sq <= 1e-24:
                break
            phi = E(F, L) + w * P
            for _ in range(40):                     # backtracking on E + w P
                F_try = step_only_free(F, grad, tau)
                P_try, _ = penalty_and_grad(F_try)
                if E(F_try, L) + w * P_try <= phi - 1e-4 * tau * gnorm_sq:
                    break
                tau *= 0.5
            else:
                break
            F = F_try
            tau *= 2.0

        remaining = int(np.sum(signed_orientation(F, T) <= 0.0))
        if verbose:
            print("  outer %d: w = %.3g, inverted faces remaining = %d"
                  % (outer, w, remaining))
        if remaining == 0:
            break
        w *= 10.0

    return F


# ---------------------------------------------------------------------------
# Distortion metrics
# ---------------------------------------------------------------------------

def _flatten_triangles(P, T, ref_normal=None):
    """Isometrically flatten every triangle into its own 2D orthonormal frame.

    Returns an (m, 2, 2) array whose [t] entry has columns
    (p2 - p1, p3 - p1) expressed in that frame.

    ref_normal - optional (m, 3) orientation reference.  Without it each frame
        is oriented by the triangle's own normal, which makes an inverted
        triangle indistinguishable from a correctly oriented one.  Passing an
        external reference (for a map onto S^2, the outward radial direction)
        gives a negative determinant on flipped triangles, so they surface as
        |mu| >= 1.
    """
    a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    e1 = b - a
    e2 = c - a
    n = np.cross(e1, e2)
    len1 = np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), config.EPSILON)
    u = e1 / len1
    nn = np.maximum(np.linalg.norm(n, axis=1, keepdims=True), config.EPSILON)
    nhat = n / nn
    if ref_normal is not None:
        sign = np.sign(np.einsum('ij,ij->i', nhat, ref_normal))
        sign[sign == 0] = 1.0
        nhat = nhat * sign[:, None]
    w = np.cross(nhat, u)               # completes the right-handed frame
    B = np.empty((len(T), 2, 2))
    B[:, 0, 0] = np.einsum('ij,ij->i', e1, u)
    B[:, 1, 0] = np.einsum('ij,ij->i', e1, w)
    B[:, 0, 1] = np.einsum('ij,ij->i', e2, u)
    B[:, 1, 1] = np.einsum('ij,ij->i', e2, w)
    return B


def beltrami_coefficient(mesh, F, coords=None):
    """Per-face Beltrami coefficient magnitude |mu| of the map source -> F.

    Each triangle is flattened isometrically in the source and in the image;
    the map between them is affine with Jacobian J, and in complex notation

        f_z  = ((J11 + J22) + i (J21 - J12)) / 2
        f_zb = ((J11 - J22) + i (J21 + J12)) / 2
        mu   = f_zb / f_z

    |mu| = 0 means the triangle is mapped conformally; |mu| -> 1 means it
    degenerates.  The quasi-conformal dilatation is K = (1+|mu|)/(1-|mu|).

    Returns an (m,) array of |mu|.
    """
    P = vertex_coordinates(mesh) if coords is None else np.asarray(coords, dtype=float)
    T = facet_indices(mesh)
    Fa = np.asarray(F, dtype=float)
    # On S^2 the outward radial direction at the face centroid orients the
    # image frame, so inverted triangles are detected rather than hidden.
    centroid = Fa[T].mean(axis=1)
    Bs = _flatten_triangles(P, T)
    Bt = _flatten_triangles(Fa, T, ref_normal=centroid)

    mu = np.ones(len(T))
    det = Bs[:, 0, 0] * Bs[:, 1, 1] - Bs[:, 0, 1] * Bs[:, 1, 0]
    ok = np.abs(det) > config.EPSILON
    if not np.any(ok):
        return mu
    J = Bt[ok] @ np.linalg.inv(Bs[ok])
    fz = ((J[:, 0, 0] + J[:, 1, 1]) + 1j * (J[:, 1, 0] - J[:, 0, 1])) / 2.0
    fzb = ((J[:, 0, 0] - J[:, 1, 1]) + 1j * (J[:, 1, 0] + J[:, 0, 1])) / 2.0
    good = np.abs(fz) > config.EPSILON
    vals = np.ones(ok.sum())
    vals[good] = np.abs(fzb[good] / fz[good])
    mu[ok] = np.minimum(vals, 1.0)
    return mu


def area_distortion(mesh, F, coords=None):
    """Per-face log area-distortion of the map source -> F.

    Areas are normalized by their totals on each side, so a perfectly
    area-preserving map gives 0 everywhere.  Returns an (m,) array.
    """
    src = facet_areas(mesh, coords)
    dst = facet_areas(mesh, F)
    src = src / max(src.sum(), config.EPSILON)
    dst = dst / max(dst.sum(), config.EPSILON)
    ratio = np.maximum(dst, config.EPSILON) / np.maximum(src, config.EPSILON)
    return np.log(ratio)


def map_degree(mesh, F):
    """Topological degree of a map onto S^2, from the total signed solid angle.

    Each spherical triangle's signed solid angle is taken with the
    Van Oosterom-Strackee formula, and the total is 4*pi*deg.  A valid
    parameterization must come out at exactly 1: degree 2 means the image wraps
    the sphere twice, and a non-integer value means the map is not a covering.
    """
    T = facet_indices(mesh)
    X = retract(np.asarray(F, dtype=float))
    a, b, c = X[T[:, 0]], X[T[:, 1]], X[T[:, 2]]
    num = np.einsum('ij,ij->i', np.cross(a, b), c)
    den = (1.0 + np.einsum('ij,ij->i', a, b)
           + np.einsum('ij,ij->i', b, c)
           + np.einsum('ij,ij->i', c, a))
    return float(np.sum(2.0 * np.arctan2(num, den)) / (4.0 * np.pi))


def conformality_gap(mesh, F, L=None):
    """Relative excess of the harmonic energy over its conformal lower bound.

    On each triangle the piecewise-linear map has Jacobian singular values
    sigma_1, sigma_2, giving Dirichlet energy (sigma_1^2 + sigma_2^2)/2 and
    image area sigma_1 * sigma_2 per unit source area.  By AM-GM

        E(f) >= Area(f(M))

    with equality exactly when every triangle is mapped conformally, so

        gap = E(f) / Area(f(M)) - 1  >= 0

    is a scale-free measure of distance from conformality.

    NB: the bound is the area of the *image triangles*, not 4*pi.  The images
    are flat chords of the sphere, so an inscribed polyhedron always falls short
    of 4*pi -- by 3 % on a 500-face mesh and 0.16 % on a 5000-face one.  Using
    4*pi as the denominator therefore mixes a discretization deficit into the
    conformality measure, and can make a perfectly valid coarse map look like it
    violates the bound.
    """
    if L is None:
        L = construct_laplacian_matrix(mesh)
    area = float(facet_areas(mesh, F).sum())
    return E(F, L) / max(area, config.EPSILON) - 1.0


def report_distortion(mesh, F, coords=None, label="", L=None):
    """Print a summary of conformal and area distortion.  Returns a dict."""
    mu = beltrami_coefficient(mesh, F, coords)
    ad = area_distortion(mesh, F, coords)
    K = (1.0 + mu) / np.maximum(1.0 - mu, config.EPSILON)
    degree = map_degree(mesh, F)
    stats = {
        'degree': degree,
        'conformality_gap': conformality_gap(mesh, F, L=L),
        'mu_mean': float(mu.mean()),
        'mu_median': float(np.median(mu)),
        'mu_max': float(mu.max()),
        'mu_p95': float(np.percentile(mu, 95)),
        # median, not mean: K blows up on near-degenerate faces, so the mean is
        # dominated by a handful of them and says nothing about the bulk.
        'dilatation_median': float(np.median(K)),
        'area_log_abs_mean': float(np.abs(ad).mean()),
        'area_log_abs_max': float(np.abs(ad).max()),
        'flipped_faces': int(np.sum(mu >= 1.0 - 1e-12)),
    }
    head = ("Distortion" if not label else "Distortion (%s)" % label)
    print("%s:" % head)
    print("  |mu|  mean %.4f   median %.4f   p95 %.4f   max %.4f   (median K %.4f)"
          % (stats['mu_mean'], stats['mu_median'], stats['mu_p95'], stats['mu_max'],
             stats['dilatation_median']))
    print("  |log area ratio|  mean %.4f   max %.4f"
          % (stats['area_log_abs_mean'], stats['area_log_abs_max']))
    print("  flipped / degenerate faces: %d of %d (%.2f%%)"
          % (stats['flipped_faces'], len(mu),
             100.0 * stats['flipped_faces'] / max(len(mu), 1)))
    gap, deg = stats['conformality_gap'], stats['degree']
    note = "   <-- not a degree-1 cover" if abs(deg - 1.0) > 1e-4 else ""
    print("  degree %+.4f    conformality gap E/Area(f) - 1 = %+.5f%s"
          % (deg, gap, note))
    return stats


# ---------------------------------------------------------------------------
# Small vector helpers (kept here because mesh.py imports them)
# ---------------------------------------------------------------------------

def allclose(v1, v2):
    """Compare if v1 and v2 are close

    v1, v2 - any numerical type or list/tuple of numerical types

    Return bool if vectors are close, up to some epsilon specified in config.py
    """
    v1 = make_iterable(v1)
    v2 = make_iterable(v2)

    elementwise_compare = map(
        (lambda x, y: abs(x - y) < config.EPSILON), v1, v2)
    return functools.reduce((lambda x, y: x and y), elementwise_compare)


def make_iterable(obj):
    """Check if obj is iterable, if not return an iterable with obj inside it.
    Otherwise just return obj.
    obj - any type
    Return an iterable
    """
    try:
        iter(obj)
    except TypeError:
        return [obj]
    else:
        return obj


def dot(v1, v2):
    """Dot product(inner product) of v1 and v2

    v1, v2 - python list

    Return v1 dot v2
    """
    elementwise_multiply = map((lambda x, y: x * y), v1, v2)
    return functools.reduce((lambda x, y: x + y), elementwise_multiply)


def norm(vec):
    """ Return the Euclidean norm of a 3d vector.
    vec - a 3d vector expressed as a list of 3 floats.
    """
    return math.sqrt(functools.reduce((lambda x, y: x + y * y), vec, 0.0))


def normalize(vec):
    """Normalize a vector

    vec - python list

    Return normalized vector
    """
    if norm(vec) < 1e-6:
        return [0 for i in range(len(vec))]
    return list(map(lambda x: x / norm(vec), vec))


def cross_product(v1, v2):
    """ Return the cross product of v1, v2.

    v1, v2 - 3d vector expressed as a list of 3 floats.
    """
    v1 = list(v1)
    v2 = list(v2)
    x3 = v1[1] * v2[2] - v2[1] * v1[2]
    y3 = -(v1[0] * v2[2] - v2[0] * v1[2])
    z3 = v1[0] * v2[1] - v2[0] * v1[1]
    return [x3, y3, z3]


def create_vector(p1, p2):
    """Contruct a vector going from p1 to p2.

    p1, p2 - python list wth coordinates [x,y,z].

    Return a list [x,y,z] for the coordinates of vector
    """
    return list(map((lambda x, y: x - y), p2, p1))
