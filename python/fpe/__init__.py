"""
Implementation of the methods discussed in: 

    G. Acciarini, C. Greco, M. Vasile, "Uncertainty propagation in orbital
    dynamics via Galerkin projection of the Fokker-Planck Equation",
    Advances in Space Research 73 (2024) 53-63,
    https://doi.org/10.1016/j.asr.2023.11.042

with a multithreaded C++ core (B-spline bases, sparse Galerkin assembly,
matrix exponentials) exposed through a NumPy/SciPy-friendly API.

Quick start
-----------
>>> import numpy as np, fpe
>>> basis = fpe.TensorBSplineBasis(domain=[(-6, 2), (-4, 4)], n_basis=36, order=3)
>>> dyn = fpe.dynamics.DampedOscillator(k=1.0, gamma=2.1)
>>> solver = fpe.FokkerPlanckSolver(basis, dyn, diffusion=[[0.0, 0.0], [0.0, 0.08]])
>>> solver.assemble()
>>> a0 = solver.project(fpe.GaussianPDF([-4.0, 0.0], np.diag([0.09, 0.09])))
>>> coeffs = solver.propagate(a0, times=np.linspace(0.0, 2.0, 21))
>>> mean, cov = solver.moments(coeffs[-1])
"""

from . import _core, dynamics, metrics, separable
from ._core import halton
from .basis import TensorBSplineBasis
from .pdf import GaussianPDF
from .solver import FokkerPlanckSolver

__version__ = "0.1.0"

__all__ = [
    "FokkerPlanckSolver",
    "TensorBSplineBasis",
    "GaussianPDF",
    "dynamics",
    "metrics",
    "separable",
    "halton",
    "_core",
    "__version__",
]
