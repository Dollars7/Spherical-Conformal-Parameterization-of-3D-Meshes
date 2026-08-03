"""Numerical regression tests for the parameterization pipeline.

Kept separate from halfedge_mesh_test.py, which covers half-edge connectivity;
these cover the geometry and optimization in optimization.py.  Each test pins a
property that was actually wrong at some point, so a regression is caught rather
than absorbed into a plausible-looking number.
"""
import numpy as np
import pytest

import optimization as O
from mesh import HalfedgeMesh, Vertex, Facet


def make_mesh(points, triangles):
    """Build a mesh carrying only the vertex/facet data the solvers need."""
    return HalfedgeMesh(
        vertices=[Vertex(float(p[0]), float(p[1]), float(p[2]), i)
                  for i, p in enumerate(points)],
        facets=[Facet(int(a), int(b), int(c), i)
                for i, (a, b, c) in enumerate(triangles)])


SCALENE = np.array([[0., 0., 0.], [3., 0., 0.], [0.7, 2.1, 0.]])

# octahedron inscribed in S^2, every face counter-clockwise seen from outside
OCTA_P = np.array([[1., 0, 0], [-1., 0, 0], [0, 1., 0],
                   [0, -1., 0], [0, 0, 1.], [0, 0, -1.]])
OCTA_T = [[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
          [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]]


@pytest.fixture(scope="module")
def bunny():
    return HalfedgeMesh("tests/data/bunny.off")


@pytest.fixture(scope="module")
def solved(bunny):
    """Bunny run through the default pipeline once, shared by the tests below."""
    source = O.vertex_coordinates(bunny)
    L = O.construct_laplacian_matrix(bunny, coords=source)
    F0 = O.gauss_map(bunny)
    F = O.fast_algorithm(F0, 1e-4, 0.5, 0.85, 1e-6, bunny, max_iter=20000, L=L,
                         retraction=O.make_retraction(O.vertex_areas(bunny, source)),
                         verbose=False)
    return {'mesh': bunny, 'source': source, 'L': L, 'F0': F0, 'F': F}


# ---------------------------------------------------------------------------
# Cotangent Laplacian
# ---------------------------------------------------------------------------

def test_laplacian_matches_reference_cotangent_weights():
    """Each edge must get the cotangent of the angle OPPOSITE it.

    The original code used an angle at one of the edge's own endpoints, which
    still yields a symmetric PSD zero-row-sum matrix -- so it looked like a
    Laplacian and converged, but was not the cotangent Laplacian and the result
    was not conformal.  Only an element-wise comparison catches that.
    """
    L = O.construct_laplacian_matrix(make_mesh(SCALENE, [[0, 1, 2]]), dense=True)

    def cot_at(a, b, c):
        u, v = b - a, c - a
        return np.dot(u, v) / np.linalg.norm(np.cross(u, v))

    cot = [cot_at(SCALENE[0], SCALENE[1], SCALENE[2]),
           cot_at(SCALENE[1], SCALENE[2], SCALENE[0]),
           cot_at(SCALENE[2], SCALENE[0], SCALENE[1])]
    ref = np.zeros((3, 3))
    for i, j, k in [(1, 2, 0), (2, 0, 1), (0, 1, 2)]:   # edge (i,j), opposite k
        w = 0.5 * cot[k]
        ref[i, j] -= w
        ref[j, i] -= w
        ref[i, i] += w
        ref[j, j] += w

    assert np.allclose(L, ref)
    # the three cotangents here are all different, so a mis-assignment shows up
    assert len(set(np.round(cot, 6))) == 3


def test_laplacian_is_symmetric_psd_with_constant_kernel(bunny):
    L = O.construct_laplacian_matrix(bunny)
    n = L.shape[0]
    assert abs(L - L.T).max() < 1e-12
    assert np.abs(L @ np.ones((n, 1))).max() < 1e-10
    assert np.linalg.eigvalsh(L.toarray()).min() > -1e-9
    assert L.nnz < 0.1 * n * n            # must stay sparse


def test_off_and_obj_paths_agree():
    """The two parsers disagreed on the Halfedge.vertex convention."""
    off = HalfedgeMesh("tests/data/brain_python.off")
    obj = HalfedgeMesh("tests/data/brain.obj")
    assert len(off.vertices) == len(obj.vertices)
    e_off = O.E(O.gauss_map(off), O.construct_laplacian_matrix(off))
    e_obj = O.E(O.gauss_map(obj), O.construct_laplacian_matrix(obj))
    assert e_off == pytest.approx(e_obj, rel=1e-12)


# ---------------------------------------------------------------------------
# Cayley step
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tau", [1e-4, 1e-2, 0.5, 3.0])
def test_cayley_woodbury_equals_dense_form(tau):
    """The low-rank form must equal (I + t/2 A)^-1 (I - t/2 A) X exactly."""
    rng = np.random.RandomState(0)
    n = 40
    A0 = rng.randn(n, n)
    L = A0 + A0.T
    L -= np.diag(L.sum(1))
    X = O.retract(rng.randn(n, 3))
    G = L @ X

    A = G @ X.T - X @ G.T
    I = np.eye(n)
    dense = np.linalg.inv(I + (tau / 2) * A) @ (I - (tau / 2) * A) @ X
    assert np.abs(O.cayley_step(X, G, tau) - dense).max() < 1e-11


def test_cayley_preserves_frobenius_but_not_row_norms():
    """Documents why a retraction is still required after the Cayley step: the
    curve is a retraction for the Stiefel manifold, not the oblique one."""
    rng = np.random.RandomState(3)
    n = 30
    A0 = rng.randn(n, n)
    L = A0 + A0.T
    L -= np.diag(L.sum(1))
    X = O.retract(rng.randn(n, 3))
    Y = O.cayley_step(X, L @ X, 1e-2)
    assert np.linalg.norm(Y) == pytest.approx(np.linalg.norm(X), rel=1e-10)
    assert not np.allclose(np.linalg.norm(Y, axis=1), 1.0, atol=1e-6)
    assert np.allclose(np.linalg.norm(O.retract(Y), axis=1), 1.0)


# ---------------------------------------------------------------------------
# Distortion metrics
# ---------------------------------------------------------------------------

def test_beltrami_zero_for_similarity_transforms():
    pts = np.array([[0., 0, 0], [1., 0, 0], [0.3, 0.9, 0], [1.4, 1.1, 0]])
    m = make_mesh(pts, [[0, 1, 2], [1, 3, 2]])
    th = 0.7
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    assert O.beltrami_coefficient(m, pts).max() < 1e-12
    assert O.beltrami_coefficient(m, (2.5 * pts) @ R.T).max() < 1e-9


def test_beltrami_matches_analytic_value_for_anisotropic_scaling():
    """A 2x stretch in one axis has mu = (2-1)/(2+1) = 1/3 exactly."""
    pts = np.array([[0., 0, 0], [1., 0, 0], [0.3, 0.9, 0], [1.4, 1.1, 0]])
    m = make_mesh(pts, [[0, 1, 2], [1, 3, 2]])
    mu = O.beltrami_coefficient(m, pts @ np.diag([2.0, 1.0, 1.0]).T)
    assert mu == pytest.approx(1.0 / 3.0, abs=1e-9)


def test_inverted_faces_are_detected_on_a_sphere():
    """An image frame oriented by the triangle's own normal cannot tell an
    inverted triangle from a correct one; the radial reference can."""
    m = make_mesh(OCTA_P, OCTA_T)
    assert O.beltrami_coefficient(m, OCTA_P).max() < 1e-12

    flipped = OCTA_P.copy()
    flipped[4] = -flipped[4]                       # north pole to the south pole
    mu = O.beltrami_coefficient(m, flipped)
    incident = [t for t, tri in enumerate(OCTA_T) if 4 in tri]
    assert np.all(mu[incident] >= 1.0 - 1e-12)
    assert int(np.sum(mu >= 1.0 - 1e-12)) == len(incident)


def test_degree_of_identity_map_on_octahedron():
    m = make_mesh(OCTA_P, OCTA_T)
    assert O.map_degree(m, OCTA_P) == pytest.approx(1.0, abs=1e-9)
    assert np.all(O.signed_orientation(OCTA_P, O.facet_indices(m)) > 0)


# ---------------------------------------------------------------------------
# The conformality bound
# ---------------------------------------------------------------------------

def test_conformality_gap_is_never_negative(solved):
    """E(f) >= Area(f(M)) by AM-GM on the per-triangle singular values, so the
    gap is non-negative for *every* map, folded or not -- not just the good one.
    """
    rng = np.random.RandomState(7)
    mesh, L, F = solved['mesh'], solved['L'], solved['F']
    for scale in (0.0, 0.05, 0.3, 1.0, 3.0):
        X = O.retract(F + scale * rng.randn(*F.shape))
        assert O.conformality_gap(mesh, X, L=L) >= -1e-12, \
            "gap went negative at perturbation scale %g" % scale


def test_gap_denominator_is_image_area_not_four_pi(solved):
    """Regression guard: 4*pi is the wrong denominator.

    The image triangles are flat chords, so an inscribed polyhedron falls short
    of 4*pi -- 3 % on this 500-face mesh.  Normalizing by 4*pi mixes that
    discretization deficit into the measure and makes a valid coarse map look
    like it violates the bound.
    """
    mesh, L, F = solved['mesh'], solved['L'], solved['F']
    area = O.facet_areas(mesh, F).sum()
    energy = O.E(F, L)

    assert area < 4 * np.pi                                   # inscribed deficit
    assert area / (4 * np.pi) < 0.99
    assert energy >= area                                     # the true bound
    assert energy < 4 * np.pi                                 # but below 4*pi
    assert O.conformality_gap(mesh, F, L=L) > 0


# ---------------------------------------------------------------------------
# Solvers and untangling, end to end
# ---------------------------------------------------------------------------

def test_nonmonotone_acceptance_requires_an_actual_decrease():
    """Pins the sign of the sufficient-decrease test.

    Written with `+` instead of `-`, every trial step is accepted on the first
    try, the line search never backtracks, and the solver quietly degenerates to
    a fixed step size -- while still converging well enough that no end-to-end
    assertion notices.  Only a direct test of the predicate catches it.
    """
    rho, tau, descent, ref = 0.5, 0.2, 4.0, 10.0
    margin = rho * tau * descent                       # 0.4

    assert O.nonmonotone_accepts(ref - margin, ref, rho, tau, descent)
    assert O.nonmonotone_accepts(ref - 2 * margin, ref, rho, tau, descent)
    assert not O.nonmonotone_accepts(ref - margin + 1e-9, ref, rho, tau, descent)

    # an energy INCREASE must never be accepted; under the `+` sign every one
    # of these would pass
    for worse in (ref, ref + 1e-9, ref + margin - 1e-9, ref + margin, ref + 10.0):
        assert not O.nonmonotone_accepts(worse, ref, rho, tau, descent)


def test_cayley_descent_matches_the_frobenius_norm_of_A():
    """||A||_F^2 computed via 3x3 traces must equal the explicit n x n form.

    This is the quantity the Cayley line search tests against; substituting
    ||grad E||^2 stalls Solver B at the first iteration.
    """
    rng = np.random.RandomState(11)
    n = 25
    A0 = rng.randn(n, n)
    L = A0 + A0.T
    L -= np.diag(L.sum(1))
    X = O.retract(rng.randn(n, 3))
    G = L @ X

    A = G @ X.T - X @ G.T
    assert O._cayley_descent(X, G) == pytest.approx(float(np.sum(A * A)), rel=1e-9)
    # and it is a genuinely different quantity from the projected gradient norm
    grad = O.project_gradient_to_sphere(G, X)
    assert O._cayley_descent(X, G) / 2.0 != pytest.approx(float(np.sum(grad ** 2)),
                                                          rel=1e-3)


def test_solver_b_reduces_energy_and_reaches_degree_one(solved):
    """End-to-end guard on the Cayley solver.

    Solver A was covered but Solver B was not, so a wrong descent quantity --
    which makes the line search fail at iteration 0 and return the initial map
    untouched -- went undetected.
    """
    mesh, L, F0 = solved['mesh'], solved['L'], solved['F0']
    F = O.orthogonal_constrained_optimization(
        mesh, F0, 1e-6, max_iterations=20000, L=L,
        retraction=O.make_retraction(O.vertex_areas(mesh, solved['source'])),
        verbose=False)
    assert O.E(F, L) < 0.2 * O.E(F0, L)
    assert np.allclose(np.linalg.norm(F, axis=1), 1.0, atol=1e-6)
    assert O.map_degree(mesh, F) == pytest.approx(1.0, abs=1e-6)
    assert O.conformality_gap(mesh, F, L=L) < 0.5


def test_flow_reduces_energy_and_reaches_degree_one(solved):
    mesh, L, F0, F = solved['mesh'], solved['L'], solved['F0'], solved['F']
    assert O.E(F, L) < 0.1 * O.E(F0, L)
    assert np.allclose(np.linalg.norm(F, axis=1), 1.0, atol=1e-6)
    assert O.map_degree(mesh, F) == pytest.approx(1.0, abs=1e-6)


def test_flow_alone_leaves_folds_that_untangling_removes(solved):
    """Harmonic energy is blind to folding, so the flow does not finish the job."""
    mesh, L, source, F = solved['mesh'], solved['L'], solved['source'], solved['F']
    T = O.facet_indices(mesh)
    assert int(np.sum(O.signed_orientation(F, T) <= 0)) > 0

    Fu = O.untangle_local(mesh, F, L=L, verbose=False)
    assert int(np.sum(O.signed_orientation(Fu, T) <= 0)) == 0
    assert O.map_degree(mesh, Fu) == pytest.approx(1.0, abs=1e-6)
    assert np.allclose(np.linalg.norm(Fu, axis=1), 1.0, atol=1e-6)
    # the fix must be local: conformality elsewhere is not allowed to collapse
    assert O.conformality_gap(mesh, Fu, L=L) < 2.0 * O.conformality_gap(mesh, F, L=L)
    assert O.beltrami_coefficient(mesh, Fu, coords=source).max() < 1.0


def test_weight_escalation_not_patch_width_decides_untangling(solved):
    """The knob that matters is max_outer, not rings: the minimal patch succeeds
    once the penalty weight is allowed to grow far enough."""
    mesh, L, F = solved['mesh'], solved['L'], solved['F']
    T = O.facet_indices(mesh)
    tight = O.untangle_local(mesh, F, L=L, rings=0, max_outer=6, verbose=False)
    patient = O.untangle_local(mesh, F, L=L, rings=0, max_outer=14, verbose=False)
    assert int(np.sum(O.signed_orientation(tight, T) <= 0)) > 0
    assert int(np.sum(O.signed_orientation(patient, T) <= 0)) == 0


def test_untangling_is_a_noop_when_there_is_nothing_to_fix():
    m = make_mesh(OCTA_P, OCTA_T)
    out = O.untangle_local(m, OCTA_P, verbose=False)
    assert np.allclose(out, OCTA_P)


def test_dropping_the_mobius_constraint_collapses_to_a_point(solved):
    """The constant map has E = 0 and is the global minimizer over the oblique
    manifold, so the zero-mass-centre constraint is what makes the problem
    well-posed.  Without it the flow must collapse."""
    mesh, L, F0 = solved['mesh'], solved['L'], solved['F0']
    F = O.fast_algorithm(F0, 1e-4, 0.5, 0.85, 1e-12, mesh, max_iter=20000, L=L,
                         retraction=O.retract, verbose=False)
    assert O.E(F, L) < 1e-6
    spread = np.linalg.norm(F - F.mean(axis=0), axis=1).max()
    assert spread < 1e-3, "expected collapse onto a single point"
