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

Center-of-mass (COM) treatment for isolated systems
----------------------------------------------------
In Langevin dynamics each atom receives an independent random impulse, so
the net force on the COM is non-zero at every step.  For isolated (non-PBC)
systems this excites COM translational motion and thermostatises all 3N
degrees of freedom — including the 3 COM modes — biasing the instantaneous
temperature high by a factor 3N/(3N-3).

To maintain consistency with calculate_temperature() which uses N_dof = 3N-3
for isolated molecules, the COM velocity is removed from the output of each
O-step.  This is the same approach used by OpenMM (CMMotionRemover, default
frequency=1) and GROMACS (comm-mode=Linear, nstcomm=100).

For periodic systems the COM has no physical meaning and is left unchanged.

References
----------
Leimkuhler & Matthews, Appl. Math. Res. eXpress 2013, 34-56.  (BAOAB)
Basconi & Shirts, J. Chem. Theory Comput. 9, 2887 (2013).     (N_dof accounting)
OpenMM CMMotionRemover documentation.                          (COM removal)
GROMACS Reference Manual, comm-mode / nstcomm.                 (COM removal)
"""

import numpy as np
from ase import Atoms
from typing import Optional

from ..utils import AMU_TO_AU, FS_TO_AU, KELVIN_TO_HARTREE


class LangevinThermostat:
    """
    Langevin thermostat using BAOAB splitting.

    Applies the Ornstein-Uhlenbeck (O) step between the two velocity
    half-steps of Velocity Verlet to provide canonical (NVT) sampling.

    For isolated (non-periodic) systems, the COM velocity is removed after
    each O-step so that N_dof = 3N-3 gives the correct temperature,
    consistent with calculate_temperature().  For periodic systems no COM
    correction is applied.
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
            Molecular system.
        temperature : float
            Target temperature in Kelvin.
        friction : float
            Friction coefficient γ in 1/fs.
        timestep : float
            MD timestep in fs.
        rng : np.random.Generator, optional
            Random number generator for reproducibility.
        """
        self.atoms       = atoms
        self.temperature = temperature
        self.friction    = friction * FS_TO_AU           # 1/fs → 1/a.u.
        self.timestep    = timestep * FS_TO_AU           # fs   → a.u.
        self.masses      = atoms.get_masses() * AMU_TO_AU  # amu → a.u.
        self.rng         = rng if rng is not None else np.random.default_rng()

        # COM removal applies only to isolated (non-periodic) systems.
        self._fix_com = not any(atoms.pbc)

        # Precompute BAOAB O-step coefficients (constant throughout run):
        #   c1 = exp(-γ dt)                   (friction decay factor)
        #   c2 = sqrt((1 - c1²) kT / m)       (noise amplitude, per atom)
        kT        = self.temperature * KELVIN_TO_HARTREE
        self._c1  = np.exp(-self.friction * self.timestep)                    # scalar
        self._c2  = np.sqrt((1.0 - self._c1**2) * kT / self.masses)          # (N_atoms,)

    def apply(self, velocities: np.ndarray) -> np.ndarray:
        """
        Apply the Ornstein-Uhlenbeck (O) step to velocities.

            v_new = c1 * v + c2 * ξ,   ξ ~ N(0, 1)

        For isolated systems the COM velocity is subsequently removed so that
        the 3 translational modes do not contribute to the measured temperature.

        Parameters
        ----------
        velocities : np.ndarray
            Current velocities in atomic units, shape (N_atoms, 3).

        Returns
        -------
        np.ndarray
            Thermostatted velocities in atomic units, shape (N_atoms, 3).
        """
        noise = self.rng.standard_normal(velocities.shape)
        v_new = self._c1 * velocities + self._c2[:, np.newaxis] * noise

        if self._fix_com:
            com_vel = (self.masses[:, np.newaxis] * v_new).sum(axis=0) / self.masses.sum()
            v_new  -= com_vel

        return v_new
