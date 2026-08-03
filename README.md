# Spherical Conformal Parameterization of 3D Meshes

A geometry-processing project that maps a closed genus-zero triangular mesh onto
the unit sphere through harmonic-energy minimization.

The implementation is written in Python and NumPy and includes a half-edge mesh
representation, cotangent-weighted discrete operators, spherical constraints,
and hand-written optimization routines.

## Overview

Spherical parameterization transforms a surface with sphere-like topology into a
mapping

\[
F: \mathcal{M} \rightarrow \mathbb{S}^{2},
\]

where each mesh vertex is assigned a point on the unit sphere.

This project approximates such a mapping by minimizing the discrete harmonic
energy

\[
E(F) = \frac{1}{2}\sum_{d=1}^{3} F_d^\mathsf{T} L F_d,
\]

where \(L\) is a cotangent-weighted discrete Laplace–Beltrami operator and every
mapped vertex is constrained to satisfy

\[
\lVert F_i \rVert_2 = 1.
\]

The project was extended from **CSE570Project3** as an educational and
experimental implementation of spherical mesh parameterization.

## Features

- Half-edge representation for triangular surface meshes
- OFF and OBJ mesh loading
- Vertex-normal and Gauss-map computation
- Cotangent-weighted discrete Laplacian construction
- Harmonic-energy evaluation
- Projected-gradient optimization on the unit sphere
- Orthogonality-constrained optimization using a Cayley-transform-style update
- Unit-sphere constraint validation
- OBJ export for mapped meshes, normals, and intermediate results
- Sample meshes and generated outputs for visualization

## Technical Highlights

The core optimization routines are implemented directly with NumPy—no
`scipy.optimize` solver is used.

The current pipeline:

1. Loads the input mesh into a half-edge data structure.
2. Computes vertex normals and constructs a Gauss-map initialization.
3. Builds a cotangent-weighted discrete Laplacian.
4. Minimizes discrete harmonic energy using hand-written constrained
   optimization routines.
5. Projects or preserves mapped vertices on the unit sphere.
6. Verifies the unit-sphere constraint and reports harmonic energy before and
   after optimization.
7. Exports the resulting spherical mappings as OBJ files.

The project explores the same class of mathematical tools used in geometry
processing, mesh parameterization, and retopology systems—including the
Laplace–Beltrami operator, harmonic maps, and manifold-constrained
optimization—from first principles rather than through an off-the-shelf
optimization package.

## Repository Structure

```text
.
├── README.md
└── Python_latest/
    ├── mesh.py                  # Half-edge mesh representation and parsers
    ├── optimization.py          # Laplacian, energy, and optimization routines
    ├── test.py                  # End-to-end demonstration pipeline
    ├── halfedge_mesh_test.py    # Half-edge regression tests
    ├── config.py                # Numerical tolerance configuration
    ├── tests/
    │   └── data/                # Example OFF and OBJ meshes
    ├── output/                  # Generated mappings and visualization files
    └── LICENSE.md
```

## Requirements

- Python 3.9 or newer is recommended
- NumPy
- Pytest, if running the included tests

SciPy is not required by the optimization pipeline.

## Installation

Clone the repository:

```bash
git clone https://github.com/Dollars7/Spherical-Conformal-Parameterization-of-3D-Meshes.git
cd Spherical-Conformal-Parameterization-of-3D-Meshes/Python_latest
```

Create a virtual environment and install the dependencies.

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install numpy pytest
```

### macOS or Linux

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install numpy pytest
```

## Usage

The current demonstration pipeline is defined in `Python_latest/test.py`.

Select an input mesh by changing `data_path`:

```python
data_path = "tests/data/brain_python.off"
```

Run the pipeline from the `Python_latest` directory:

### Windows

```powershell
.\.venv\Scripts\python test.py
```

### macOS or Linux

```bash
.venv/bin/python test.py
```

Depending on the selected optimization path, generated files may include:

```text
output/normals.obj
output/gauss_map.obj
output/initial_constrained_mesh.obj
output/spherical_mapping_fast.obj
output/spherical_mapping_ortho.obj
```

These OBJ files can be inspected in Blender, MeshLab, or another compatible
mesh viewer.

## Running the Tests

From the `Python_latest` directory:

```bash
python -m pytest halfedge_mesh_test.py
```

The included tests primarily cover the half-edge mesh representation and its
geometry utilities. The project is currently an academic prototype, so the test
suite should be expanded before using the implementation in a production
pipeline.

## Input Assumptions

The spherical parameterization algorithm is intended for meshes that are:

- Connected
- Closed, without boundary
- Orientable
- Triangulated
- Genus zero, with sphere-like topology

Meshes that violate these assumptions may fail to load, produce invalid
half-edge connectivity, or result in unstable mappings.

## Current Limitations

- The implementation uses dense NumPy matrices and is not optimized for
  high-resolution meshes.
- Mesh paths and optimization parameters are currently configured directly in
  `test.py`.
- The present validation checks unit-sphere constraints and harmonic-energy
  values; explicit conformal or angle-distortion metrics have not yet been
  implemented.
- Additional numerical safeguards are needed for degenerate triangles and
  nearly zero-area faces.
- The optimization and mesh-processing test coverage is still limited.

## Future Work

- Add per-face and aggregate angle-distortion measurements
- Use the area mass matrix in a generalized Laplace–Beltrami eigenproblem
- Replace dense matrices with sparse representations
- Add command-line arguments for mesh paths and solver parameters
- Improve convergence diagnostics and failure reporting
- Add automated tests and continuous integration
- Benchmark the solvers on meshes of different resolutions
- Provide rendered before-and-after examples

## Acknowledgements

This project was extended from **CSE570Project3** and includes a Python
half-edge mesh implementation used for educational geometry-processing
experiments.

Please consult the repository history and
[`Python_latest/LICENSE.md`](Python_latest/LICENSE.md) for the attribution and
license information associated with the included half-edge components.

## License

The bundled half-edge mesh component is distributed under the MIT License. See
[`Python_latest/LICENSE.md`](Python_latest/LICENSE.md) for details.

Because this repository contains extended academic and third-party material,
verify that all required attribution is retained when redistributing or
incorporating the code into another project.
