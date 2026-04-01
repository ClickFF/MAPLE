"""
C-rescale (stochastic cell rescaling) barostat for NPT molecular dynamics.

C-rescale is the pressure analogue of V-rescale: it corrects the Berendsen
barostat by adding a stochastic term to the volume update, producing the
correct isothermal-isobaric (NPT) ensemble.

Algorithm (Bernetti & Bussi, 2020):
    The volume V is rescaled stochastically each step.  The new volume is
    drawn from the conditional distribution:

        dV = V * β * (dt/τ_P) * (P - P_target)
           + sqrt(2 * k_B * T * V * β * dt / τ_P) * W

    where W ~ N(0, 1) is a Wiener noise term.  This ensures the marginal
    distribution of V follows the correct Gibbs distribution.

    Positions and cell are scaled isotropically by μ = (V_new / V)^(1/3).
    Velocities are left unchanged (Berendsen convention).

Notes:
    - Produces the correct NPT ensemble, unlike plain Berendsen barostat.
    - Isotropic scaling only; anisotropic tensors not yet supported.
    - Pressure is computed from the virial theorem. In this implementation,
      calculator stress is treated as the configurational/virial contribution,
      and the kinetic term is computed explicitly from current velocities.
      If stress is unavailable, the ideal-gas approximation (W=0) is used.

Reference:
    Bernetti & Bussi, J. Chem. Phys. 153, 114107 (2020).
"""

import numpy as np
from ase import Atoms
from typing import Optional

from ..utils import (
    KELVIN_TO_HARTREE,
    EV_PER_ANG3_TO_BAR,
    DEFAULT_COMPRESSIBILITY,
    compute_instantaneous_pressure,
)


class CRescaleBarostat:
    """
    Stochastic cell rescaling barostat (C-rescale).

    Isotropically rescales cell and atomic positions each step to sample
    the correct isothermal-isobaric (NPT) ensemble.
    """

    def __init__(
        self,
        atoms: Atoms,
        pressure: float,
        temperature: float,
        tau_p: float,
        timestep: float,
        compressibility: float = DEFAULT_COMPRESSIBILITY,
        rng: Optional[np.random.Generator] = None,
    ):
        """
        Parameters
        ----------
        atoms : ase.Atoms
            Molecular system (must have a periodic cell)
        pressure : float
            Target pressure in bar
        temperature : float
            Target temperature in Kelvin (needed for the noise term)
        tau_p : float
            Pressure relaxation time in fs.
            Larger τ_P = weaker coupling.  Recommended: 2000–5000 fs for MLP.
        timestep : float
            MD timestep in fs
        compressibility : float
            Isothermal compressibility in 1/bar (default: water ~4.5e-5)
        rng : np.random.Generator, optional
            Random number generator for reproducibility
        """
        self.atoms = atoms
        self.pressure_target = pressure          # bar
        self.temperature = temperature           # K
        self.tau_p = tau_p                       # fs
        self.timestep = timestep                 # fs
        self.compressibility = compressibility   # 1/bar
        self.rng = rng if rng is not None else np.random.default_rng()

        # Warning flag: emit stress-unavailable warning at most once per instance
        self._stress_warned = False

        # Deterministic prefactor: β * dt / τ_P  (dimensionless)
        self._det_prefactor = compressibility * timestep / tau_p

        # Stochastic noise prefactor (dimensionless, multiplied by 1/√V later):
        #   dV/V|_noise = sqrt(2 k_B T β dt / (τ_P V)) * W
        # We precompute sqrt(2 k_B T β dt / τ_P) in units of √Å³:
        #   k_B T in eV = T * KELVIN_TO_HARTREE * HARTREE_TO_EV
        #   β in Å³/eV  = compressibility / EV_PER_ANG3_TO_BAR
        #   → product: [eV * Å³/eV * 1] = Å³  ✓
        HARTREE_TO_EV = 27.211386245988
        kT_ev = temperature * KELVIN_TO_HARTREE * HARTREE_TO_EV      # eV
        beta_ang3_per_ev = compressibility / EV_PER_ANG3_TO_BAR      # Å³/eV
        self._noise_prefactor = np.sqrt(
            2.0 * kT_ev * beta_ang3_per_ev * (timestep / tau_p)
        )   # units: √Å³

    def get_pressure(self, velocities: np.ndarray) -> float:
        """
        Compute instantaneous pressure in bar via the virial theorem.

        P = (2*KE + W) / (3*V)

        Parameters
        ----------
        velocities : np.ndarray
            Current velocities in atomic units, shape (N_atoms, 3)

        Returns
        -------
        float
            Instantaneous pressure in bar
        """
        pressure, self._stress_warned = compute_instantaneous_pressure(
            self.atoms, velocities, self._stress_warned, self.__class__.__name__
        )
        return pressure

    def apply(self, velocities: np.ndarray) -> float:
        """
        Apply one C-rescale barostat step: stochastically rescale cell.

        The volume change has both a deterministic Berendsen-like part and
        a stochastic part that ensures the correct NPT ensemble:

            dV/V = β*(dt/τ_P)*(P - P_target)  +  noise * W / sqrt(V)

        Cell and positions are scaled isotropically by μ = (V_new/V)^(1/3).
        Velocities are not modified.

        Parameters
        ----------
        velocities : np.ndarray
            Current velocities in atomic units

        Returns
        -------
        float
            Instantaneous pressure before rescaling (bar), for logging
        """
        pressure = self.get_pressure(velocities)
        volume = self.atoms.get_volume()   # Å³

        # Deterministic part (Berendsen-like): β*(dt/τ_P)*(P - P_target)
        # so that P < P_target shrinks the cell and P > P_target expands it.
        dv_det = self._det_prefactor * (pressure - self.pressure_target)

        # Stochastic part: _noise_prefactor [√Å³] / sqrt(V [Å³]) * W
        #                = sqrt(2 k_B T β dt / (τ_P V)) * W  (dimensionless)
        w = self.rng.standard_normal()
        dv_stoch = self._noise_prefactor / np.sqrt(volume) * w

        # New volume fraction
        mu3 = 1.0 + dv_det + dv_stoch
        # Clamp: same stability bounds as Berendsen
        mu3 = float(np.clip(mu3, 0.5 ** 3, 2.0 ** 3))
        mu = mu3 ** (1.0 / 3.0)

        # Rescale cell and positions isotropically
        self.atoms.set_cell(self.atoms.get_cell() * mu, scale_atoms=True)

        return pressure
