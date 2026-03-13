"""
Molecular Dynamics (MD) module for MAPLE.

This module provides classical molecular dynamics simulation capabilities
using machine learning potentials.

Supported ensembles:
    - NVE (microcanonical)
    - NVT (canonical) with Langevin or V-rescale thermostat
    - NPT (isothermal-isobaric) with Berendsen or C-rescale barostat

Main components:
    - Integrators: Velocity Verlet (symplectic)
    - Thermostats: Langevin (BAOAB), V-rescale (Bussi 2007)
    - Barostats: Berendsen, C-rescale (Bernetti & Bussi 2020)
"""

__version__ = '0.1.0'
__author__ = 'MAPLE Development Team'

# Main MD dispatcher will be imported here when implemented
# from .md import MDDispatcher

# Utilities
from .utils import (
    calculate_temperature,
    calculate_kinetic_energy,
    initialize_velocities,
    KELVIN_TO_HARTREE,
    AMU_TO_AU,
    FS_TO_AU
)

__all__ = [
    'calculate_temperature',
    'calculate_kinetic_energy',
    'initialize_velocities',
    'KELVIN_TO_HARTREE',
    'AMU_TO_AU',
    'FS_TO_AU',
]
