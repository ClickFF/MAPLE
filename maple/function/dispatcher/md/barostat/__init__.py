"""
Barostat implementations for NPT simulations.

Provides pressure control methods:
    - Berendsen: Deterministic cell rescaling — fast equilibration, incorrect ensemble
    - C-rescale: Stochastic cell rescaling (Bernetti & Bussi 2020) — correct NPT ensemble
"""

from .berendsen import BerendsenBarostat
from .crescale import CRescaleBarostat

__all__ = ['BerendsenBarostat', 'CRescaleBarostat']
