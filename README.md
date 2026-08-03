# Spherical Conformal Parameterization of 3D-Meshes

Extended from CSE570Project3.

Conformal transformation of genus-0 surfaces to a unit sphere, based on harmonic energy minimization.

## Technical Highlights

The optimization solver is hand-written — no `scipy.optimize`. The pipeline:

1. Builds the cotangent-weighted discrete Laplacian for the input mesh
2. Minimizes harmonic energy via projected-gradient descent (hand-rolled, not a
   library call) to map the mesh onto the unit sphere
3. Validates the unit-sphere constraint and reports harmonic energy before and
   after optimization

Same class of math behind mesh parameterization/retopology tools in production 3D
software (Laplace–Beltrami operator, harmonic maps) — implemented from first
principles instead of via an off-the-shelf solver.
