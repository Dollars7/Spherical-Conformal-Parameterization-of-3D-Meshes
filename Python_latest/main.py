"""Driver for spherical conformal parameterization of a genus-zero mesh.

    python main.py --input tests/data/brain_python.off --solver both

Pipeline:
    1. load the mesh into a half-edge structure
    2. assemble the cotangent Laplacian L *from the source geometry*
    3. build an initial map F0 onto S^2 (Gauss map or LB spectral embedding)
    4. minimize E(F) = 1/2 tr(F^T L F) over the oblique manifold
    5. fix the Mobius gauge, then report energy and distortion

Note that step 2 precedes step 3: the cotangent weights must encode the metric
of the input surface, so the initial map is kept in a separate array and never
written back over the mesh coordinates.

original author:
    zhangsihao yang
"""
import argparse
import os
import time

import numpy as np

import optimization
from mesh import HalfedgeMesh


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def save_obj(file_name, coords, facets):
    """Write an OBJ with the given vertex positions and the mesh connectivity."""
    with open(file_name, 'w') as handle:
        for x, y, z in coords:
            handle.write("v {} {} {}\n".format(x, y, z))
        for facet in facets:
            if facet.a is None or facet.b is None or facet.c is None:
                continue
            handle.write("f {} {} {}\n".format(facet.a + 1, facet.b + 1, facet.c + 1))


def save_for_g3dogl(file_name, coords, normals):
    """Write vertices and normals in the G3dOGL viewer format."""
    with open(file_name, 'w') as handle:
        for i, ((vx, vy, vz), (nx, ny, nz)) in enumerate(zip(coords, normals)):
            handle.write(f"Vertex {i + 1} {vx:.6f} {vy:.6f} {vz:.6f}\n")
            handle.write(f"    {{normal=({nx:.6f}, {ny:.6f}, {nz:.6f})\n")
            handle.write(f"     Opos=({vx * 0.7:.6f}, {vy * 0.7:.6f}, {vz * 0.7:.6f})\n")
            handle.write(f"     Onormal=({nx:.6f}, {ny:.6f}, {nz:.6f})}}\n\n")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--input', default='tests/data/brain_python.off',
                   help="input mesh (.off or .obj)")
    p.add_argument('--output-dir', default='output', help="directory for results")
    p.add_argument('--solver', default='fast', choices=['fast', 'ortho', 'both'],
                   help="'fast' = projected gradient, 'ortho' = Cayley curvilinear")
    p.add_argument('--init', default='gauss', choices=['gauss', 'spectral'],
                   help="initial map: Gauss map, or LB spectral embedding")
    p.add_argument('--max-iter', type=int, default=5000)
    p.add_argument('--tol', type=float, default=1e-6,
                   help="stopping tolerance on the Riemannian gradient norm")
    p.add_argument('--rho', type=float, default=1e-4,
                   help="sufficient-decrease constant of the line search")
    p.add_argument('--delta', type=float, default=0.5,
                   help="backtracking shrink factor, in (0, 1)")
    p.add_argument('--xi', type=float, default=0.85,
                   help="nonmonotone averaging weight, in [0, 1]")
    p.add_argument('--untangle', action='store_true',
                   help="post-pass that removes residual inverted faces by "
                        "moving a pinned neighbourhood around each fold")
    p.add_argument('--untangle-rings', type=int, default=1,
                   help="how many vertex rings around a fold are free to move; "
                        "wider patches perturb more of an already-good map")
    p.add_argument('--untangle-steps', type=int, default=12,
                   help="how many times the fold penalty weight may be raised; "
                        "this, not --untangle-rings, is what usually decides "
                        "whether a stubborn fold is removed")
    p.add_argument('--no-mobius', action='store_true',
                   help="drop the zero-mass-centre constraint; the flow then "
                        "collapses to the constant map, which is the true "
                        "global minimizer of E (diagnostic use only)")
    p.add_argument('--quiet', action='store_true')
    return p.parse_args(argv)


def run(args):
    print('-' * 62)
    print("Spherical conformal parameterization")
    print("  input : %s" % args.input)

    mesh = HalfedgeMesh(args.input)
    if not mesh.vertices or not mesh.facets:
        raise SystemExit("Error: mesh failed to load (no vertices or facets).")
    os.makedirs(args.output_dir, exist_ok=True)

    print("  loaded: %d vertices, %d facets, %d halfedges"
          % (len(mesh.vertices), len(mesh.facets), len(mesh.halfedges)))

    chi = len(mesh.vertices) - len(mesh.halfedges) // 2 + len(mesh.facets)
    print("  Euler characteristic chi = %d (genus 0 expects 2)" % chi)
    if chi != 2:
        print("  Warning: the input is not a closed genus-zero surface; the "
              "spherical parameterization is not guaranteed to exist.")
    print('-' * 62)

    # Source geometry, captured before anything touches the mesh.
    source = optimization.vertex_coordinates(mesh)

    t0 = time.time()
    L = optimization.construct_laplacian_matrix(mesh, coords=source)
    print("Cotangent Laplacian: %d x %d, %d nonzeros (%.1f%% dense), %.2fs"
          % (L.shape[0], L.shape[1], L.nnz,
             100.0 * L.nnz / (L.shape[0] ** 2), time.time() - t0))

    areas = optimization.vertex_areas(mesh, source)

    # --- initial map -------------------------------------------------------
    mesh.calculate_vertex_normals()
    normals = np.array([v.normal for v in mesh.vertices])
    save_obj(os.path.join(args.output_dir, 'normals.obj'), normals, mesh.facets)

    if args.init == 'spectral':
        F0 = optimization.weighted_lb_eigen_projection(mesh, L=L)
        print("Initial map: Laplace-Beltrami spectral embedding")
    else:
        F0 = optimization.gauss_map(mesh)
        print("Initial map: Gauss map")
    save_obj(os.path.join(args.output_dir, 'gauss_map.obj'), F0, mesh.facets)

    assert np.allclose(np.linalg.norm(F0, axis=1), 1.0, atol=1e-6), \
        "initial map is not on the unit sphere"
    print("Initial harmonic energy: %.8e" % optimization.E(F0, L))
    optimization.report_distortion(mesh, F0, coords=source, label="initial map", L=L)
    print('-' * 62)

    results = {}
    verbose = not args.quiet

    # The mass-centre constraint is enforced by the retraction at every
    # iteration; applying it only at the end would let the flow collapse onto
    # the degenerate constant map first.
    retraction = optimization.make_retraction(areas, mobius=not args.no_mobius)

    if args.solver in ('fast', 'both'):
        print("Solver A -- projected gradient, BB step + nonmonotone line search")
        t0 = time.time()
        F = optimization.fast_algorithm(
            F0, args.rho, args.delta, args.xi, args.tol, mesh,
            max_iter=args.max_iter, L=L, retraction=retraction, verbose=verbose)
        results['fast'] = (F, time.time() - t0)

    if args.solver in ('ortho', 'both'):
        print("Solver B -- Wen-Yin Cayley curvilinear search")
        t0 = time.time()
        F = optimization.orthogonal_constrained_optimization(
            mesh, F0, args.tol, max_iterations=args.max_iter, L=L,
            rho=args.rho, xi=args.xi, retraction=retraction, verbose=verbose)
        results['ortho'] = (F, time.time() - t0)

    # --- report and save ---------------------------------------------------
    for name, (F, elapsed) in results.items():
        print('-' * 62)
        label = "Solver A (fast)" if name == 'fast' else "Solver B (ortho)"
        assert np.allclose(np.linalg.norm(F, axis=1), 1.0, atol=1e-6), \
            "%s left vertices off the unit sphere" % label
        print("%s: %.2fs, final harmonic energy %.8e"
              % (label, elapsed, optimization.E(F, L)))
        optimization.report_distortion(mesh, F, coords=source, label=label, L=L)

        if args.untangle:
            t0 = time.time()
            F = optimization.untangle_local(mesh, F, L=L, rings=args.untangle_rings,
                                            max_outer=args.untangle_steps,
                                            verbose=verbose)
            results[name] = (F, elapsed)
            print("  untangling took %.2fs, harmonic energy now %.8e"
                  % (time.time() - t0, optimization.E(F, L)))
            optimization.report_distortion(mesh, F, coords=source,
                                           label=label + ", untangled", L=L)

        out = os.path.join(args.output_dir, 'spherical_mapping_%s.obj' % name)
        save_obj(out, F, mesh.facets)
        save_for_g3dogl(os.path.join(args.output_dir, 'g3dogl_%s.txt' % name), F, F)
        print("  written to %s" % out)

    return results


if __name__ == '__main__':
    run(parse_args())
