"""Classical transmission baselines.

- Eulerian persistence
- Dense optical flow (Lucas-Kanade on a zenithal chart)
- Frozen-flow cross-correlation
- Lagrangian advection with GOES derived-motion winds
"""

from .lagrangian_winds import LagrangianWindAdvection
from .optical_flow import DenseOpticalFlow
from .persistence import EulerianPersistence
from .segev_crosscorr import FrozenFlowCrossCorr

__all__ = [
    "EulerianPersistence",
    "DenseOpticalFlow",
    "FrozenFlowCrossCorr",
    "LagrangianWindAdvection",
    "classical_baselines",
]


def classical_baselines() -> list:
    """Kinematic baselines, persistence first."""
    return [
        EulerianPersistence(),
        DenseOpticalFlow(),
        FrozenFlowCrossCorr(),
        LagrangianWindAdvection(),
    ]
