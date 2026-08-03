#! /usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='spherical_conformal',
    version='1.1',
    description="Spherical conformal parameterization of genus-zero triangle meshes",
    # Flat modules, not a package: the old `packages=['halfedge_mesh']` pointed
    # at a directory that does not exist (the module is now mesh.py).
    py_modules=['mesh', 'optimization', 'config', 'main'],
    install_requires=['numpy>=1.20', 'scipy>=1.7'],
    extras_require={'test': ['pytest>=6.0']},
    python_requires='>=3.8',
)
