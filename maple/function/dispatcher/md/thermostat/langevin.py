"""
Langevin thermostat for NVT molecular dynamics.

The Langevin equation adds a friction term and a random force to Newton's
equations of motion, coupling the system to a heat bath at temperature T:

    m * a = F_conservative - γ * m * v + F_random

where:
    γ       = friction coefficient (1/fs)
    F_random ~ N(0, sqrt(2 * γ * m * k_B * T / dt))

This is integrated using the BAOAB splitting scheme (Leimkuhler & Matthews):
    B: half-step velocity update with conservative force
    A: half-step position update
    O: full Ornstein-Uhlenbeck step (thermostat)
    A: half-step position update
    B: half-step velocity update with new conservative force

BAOAB is equivalent to the standard velocity Verlet + thermostat at midstep,
but has better configurational sampling properties.

Reference:
    Leimkuhler & Matthews, Appl. Math. Res. eXpress 2013, 34-56.
"""

import numpy as np
from ase import Atoms
from typing import Optional

from ..utils import AMU_TO_AU, FS_TO_AU, KELVIN_TO_HARTREE


class LangevinThermostat:
    """
    Langevin thermostat using BAOAB splitting.

    Applies stochastic velocity rescaling between the two velocity half-steps
    of Velocity Verlet, providing canonical (NVT) sampling.
    """

    def __init__(
        self,
        atoms: Atoms,
        temperature: float,
        friction: float,
        timestep: float,
        rng: Optional[np.random.Generator] = None,
    ):
        """
        Parameters
        ----------
        atoms : ase.Atoms
            Molecular system
        temperature : float
            Target temperature in Kelvin
        friction : float
            Friction coefficient γ in 1/fs
        timestep : float
            MD timestep in fs
        rng : np.random.Generator, optional
            Random number generator for reproducibility
        """
        self.atoms = atoms
        self.temperature = temperature
        self.friction = friction * FS_TO_AU          # convert 1/fs → 1/a.u.
        self.timestep = timestep * FS_TO_AU          # convert fs → a.u.
        self.masses = atoms.get_masses() * AMU_TO_AU # convert amu → a.u.
        self.rng = rng if rng is not None else np.random.default_rng()

        # Precompute BAOAB O-step coefficients (constant throughout simulation)
        # c1 = exp(-γ * dt),  c2 = sqrt((1 - c1²) * k_B * T / m)
        dt = self.timestep
        gamma = self.friction
        kT = self.temperature * KELVIN_TO_HARTREE

        self._c1 = np.exp(-gamma * dt)                                       # (scalar)
        self._c2 = np.sqrt((1.0 - self._c1 ** 2) * kT / self.masses)        # (N_atoms,)

    def apply(self, velocities: np.ndarray) -> np.ndarray:
        """
        Apply the Ornstein-Uhlenbeck (O) step to velocities.

        v_new = c1 * v + c2 * xi,  xi ~ N(0, 1)

        This is called once per MD step between the two velocity half-steps.

        Parameters
        ----------
        velocities : np.ndarray
            Current velocities in atomic units, shape (N_atoms, 3)

        Returns
        -------
        np.ndarray
            Thermostatted velocities in atomic units
        """
        noise = self.rng.standard_normal(velocities.shape)   # (N_atoms, 3)
        return self._c1 * velocities + self._c2[:, np.newaxis] * noise
