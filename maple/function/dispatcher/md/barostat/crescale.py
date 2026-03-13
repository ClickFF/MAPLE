"""
C-rescale (stochastic cell rescaling) barostat for NPT molecular dynamics.

C-rescale is the pressure analogue of V-rescale: it corrects the Berendsen
barostat by adding a stochastic term to the volume update, producing the
correct isothermal-isobaric (NPT) ensemble.

Algorithm (Bernetti & Bussi, 2020):
    The volume V is rescaled stochastically each step.  The new volume is
    drawn from the conditional distribution:

        dV = V * β * (dt/τ_P) * (P_target - P)
           + sqrt(2 * k_B * T * V * β * dt / τ_P) * W

    where W ~ N(0, 1) is a Wiener noise term.  This ensures the marginal
    distribution of V follows the correct Gibbs distribution.

    Positions and cell are scaled isotropically by μ = (V_new / V)^(1/3).
    Velocities are left unchanged (Berendsen convention).

Notes:
    - Produces the correct NPT ensemble, unlike plain Berendsen barostat.
    - Isotropic scaling only; anisotropic tensors not yet supported.
    - Pressure is computed from the virial theorem using the stress tensor
      if available; otherwise the ideal-gas approximation (W=0) is used.

Reference:
    Bernetti & Bussi, J. Chem. Phys. 153, 114107 (2020).
"""

import numpy as np
from ase import Atoms
from typing import Optional

from ..utils import (
    AMU_TO_AU, FS_TO_AU, AU_TO_FS, BOHR_TO_ANGSTROM, KELVIN_TO_HARTREE,
    EV_PER_ANG3_TO_BAR, DEFAULT_COMPRESSIBILITY, AMU_ANG2_PER_FS2_TO_EV,
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
        atoms = self.atoms
        volume = atoms.get_volume()   # Å³

        # Kinetic energy in eV
        masses_amu = atoms.get_masses()
        # v in a.u. (Bohr/a.u.time) → convert to Å/fs
        v_ang_per_fs = velocities * BOHR_TO_ANGSTROM / AU_TO_FS
        # KE in eV: 0.5 * m[amu] * v²[Å²/fs²] * (amu·Å²/fs² → eV)
        ke_ev = 0.5 * np.sum(masses_amu[:, np.newaxis] * v_ang_per_fs ** 2) * AMU_ANG2_PER_FS2_TO_EV

        # Virial from stress tensor (eV)
        virial_ev = 0.0
        try:
            stress = atoms.get_stress(voigt=True)   # eV/Å³
            virial_ev = -volume * (stress[0] + stress[1] + stress[2])
        except Exception:
            pass

        pressure_ev_ang3 = (2.0 * ke_ev + virial_ev) / (3.0 * volume)
        return pressure_ev_ang3 * EV_PER_ANG3_TO_BAR

    def apply(self, velocities: np.ndarray) -> float:
        """
        Apply one C-rescale barostat step: stochastically rescale cell.

        The volume change has both a deterministic Berendsen-like part and
        a stochastic part that ensures the correct NPT ensemble:

            dV/V = β*(dt/τ_P)*(P_target - P)  +  noise * W / sqrt(V)

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

        # Deterministic part (Berendsen-like): β*(dt/τ_P)*(P_target - P)
        dv_det = self._det_prefactor * (self.pressure_target - pressure)

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
