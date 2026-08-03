"""Render a source mesh and its spherical parameterization side by side.

    python visualize.py --input tests/data/brain_python.off \
                        --mapping output/spherical_mapping_fast.obj

Produces a four-panel figure:

  top row     a checkerboard laid out in the spherical coordinates of the image,
              pulled back onto the source by vertex-index correspondence.  This
              is the direct visual test of conformality: a conformal map carries
              the checker cells to cells that stay locally square, so curvature
              shows up as cells changing *size* but not *shape*.  Sheared or
              collapsed cells mean angle distortion.

  bottom row  the same two surfaces coloured by the per-face Beltrami modulus
              |mu|, so the distortion is localized rather than summarized.
"""
import argparse
import os

import numpy as np

import optimization
from mesh import HalfedgeMesh


def load_obj_vertices(path):
    """Read just the vertex block of an OBJ written by main.py."""
    coords = []
    with open(path) as handle:
        for line in handle:
            if line.startswith('v '):
                parts = line.split()
                coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(coords, dtype=float)


def checker_colors(F, T, n_theta=16, n_phi=8):
    """Two-tone checkerboard indexed by the spherical coordinates of each face."""
    centroid = F[T].mean(axis=1)
    centroid /= np.maximum(np.linalg.norm(centroid, axis=1, keepdims=True), 1e-12)
    theta = np.arctan2(centroid[:, 1], centroid[:, 0])          # -pi .. pi
    phi = np.arccos(np.clip(centroid[:, 2], -1.0, 1.0))         #   0 .. pi
    cell = (np.floor((theta + np.pi) / (2 * np.pi) * n_theta)
            + np.floor(phi / np.pi * n_phi))
    return (cell.astype(int) % 2)


def shade(P, T, base_rgb, light=(0.3, 0.4, 0.85)):
    """Apply simple diffuse shading so the 3D form reads in a flat render."""
    a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    n = np.cross(b - a, c - a)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    lamb = np.abs(n @ (np.array(light) / np.linalg.norm(light)))
    return np.clip(base_rgb * (0.45 + 0.55 * lamb)[:, None], 0, 1)


def add_surface(ax, P, T, facecolors, title, equal_box=True):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    coll = Poly3DCollection(P[T], facecolors=facecolors, linewidths=0)
    coll.set_edgecolor('none')
    ax.add_collection3d(coll)
    lo, hi = P.min(axis=0), P.max(axis=0)
    mid, span = (lo + hi) / 2.0, (hi - lo).max() / 2.0
    for setlim, m in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), mid):
        setlim(m - span, m + span)
    if equal_box:
        ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.set_title(title, fontsize=10)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--input', default='tests/data/brain_python.off')
    p.add_argument('--mapping', default='output/spherical_mapping_fast.obj')
    p.add_argument('--out', default='output/comparison.png')
    p.add_argument('--elev', type=float, default=18.0)
    p.add_argument('--azim', type=float, default=-60.0)
    p.add_argument('--dpi', type=int, default=140)
    args = p.parse_args(argv)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    mesh = HalfedgeMesh(args.input)
    source = optimization.vertex_coordinates(mesh)
    T = optimization.facet_indices(mesh)
    F = load_obj_vertices(args.mapping)
    if len(F) != len(source):
        raise SystemExit("vertex count mismatch: %d in mapping, %d in source"
                         % (len(F), len(source)))

    mu = optimization.beltrami_coefficient(mesh, F, coords=source)
    checker = checker_colors(F, T)

    pale = np.array([[0.92, 0.93, 0.96], [0.20, 0.45, 0.75]])
    checker_rgb = pale[checker]

    # Scale the colour range to the bulk of the distribution.  On a map that is
    # already near-conformal almost every face sits near 0, so a fixed 0..1
    # ramp renders the whole surface flat black and shows nothing.
    inverted = mu >= 1 - 1e-12
    vmax = max(float(np.percentile(mu[~inverted] if np.any(~inverted) else mu, 99)), 1e-3)
    mu_rgb = plt.get_cmap('inferno')(np.clip(mu / vmax, 0, 1))[:, :3]
    mu_rgb[inverted] = [0.10, 0.95, 0.55]          # inverted faces, called out

    fig = plt.figure(figsize=(11, 9))
    panels = [
        (1, source, checker_rgb, "source surface\n(checker pulled back from S$^2$)"),
        (2, F, checker_rgb, "spherical parameterization\n(checker in $(\\theta,\\phi)$)"),
        (3, source, mu_rgb, "source, coloured by $|\\mu|$"),
        (4, F, mu_rgb, "sphere, coloured by $|\\mu|$"),
    ]
    for idx, P, rgb, title in panels:
        ax = fig.add_subplot(2, 2, idx, projection='3d')
        ax.view_init(elev=args.elev, azim=args.azim)
        add_surface(ax, P, T, shade(P, T, rgb), title)

    fig.subplots_adjust(left=0.01, right=0.90, top=0.86, bottom=0.02,
                        wspace=0.0, hspace=0.10)

    sm = plt.cm.ScalarMappable(cmap='inferno', norm=plt.Normalize(0, vmax))
    cbar = fig.colorbar(sm, ax=fig.axes[2:], fraction=0.030, pad=0.04,
                        shrink=0.85)
    cbar.set_label("$|\\mu|$   (0 = conformal; green = inverted face)")

    fig.suptitle("%s  $\\rightarrow$  %s\nmean $|\\mu|$ = %.4f    median = %.4f"
                 "    p95 = %.4f    inverted faces: %d / %d"
                 % (os.path.basename(args.input), os.path.basename(args.mapping),
                    mu.mean(), np.median(mu), np.percentile(mu, 95),
                    int(inverted.sum()), len(mu)),
                 fontsize=11, y=0.97)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi)
    print("written to %s" % args.out)
    return args.out


if __name__ == '__main__':
    main()
