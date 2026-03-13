"""
Berendsen barostat for NPT molecular dynamics.

The Berendsen barostat rescales the simulation cell and atomic positions
to drive the instantaneous pressure toward a target value:

    dP/dt = (P_target - P) / τ_P

This is achieved by isotropic scaling of the cell and positions each step:

    μ = [1 - (dt / τ_P) * κ_T * (P_target - P)]^(1/3)

where:
    τ_P   = pressure relaxation time (fs)
    κ_T   = isothermal compressibility (1/bar), approximated as a constant
    P     = instantaneous pressure (bar)
    dt    = timestep (fs)

The instantaneous pressure is computed from the virial theorem:

    P = (2*KE + W) / (3*V)

where W = -dU/dV is the virial, approximated here via the stress tensor
from the calculator (if available), otherwise set to zero (ideal-gas limit).

Note:
    The Berendsen barostat does NOT generate a rigorously correct NPT
    ensemble — it suppresses pressure fluctuations. For production runs
    requiring correct thermodynamic averages, a Parrinello-Rahman or
    Monte Carlo barostat is preferable. Berendsen is well-suited for
    equilibration.

Reference:
    Berendsen et al., J. Chem. Phys. 81, 3684 (1984).
"""

import numpy as np
from ase import Atoms

from ..utils import (
    AMU_TO_AU, FS_TO_AU, AU_TO_FS, BOHR_TO_ANGSTROM,
    EV_PER_ANG3_TO_BAR, DEFAULT_COMPRESSIBILITY, AMU_ANG2_PER_FS2_TO_EV,
)


class BerendsenBarostat:
    """
    Isotropic Berendsen barostat.

    Rescales the simulation cell and atomic positions each step to
    weakly couple the system to a pressure bath.
    """

    def __init__(
        self,
        atoms: Atoms,
        pressure: float,
        tau_p: float,
        timestep: float,
        compressibility: float = DEFAULT_COMPRESSIBILITY,
    ):
        """
        Parameters
        ----------
        atoms : ase.Atoms
            Molecular system (must have a periodic cell)
        pressure : float
            Target pressure in bar
        tau_p : float
            Pressure relaxation time in fs
        timestep : float
            MD timestep in fs
        compressibility : float
            Isothermal compressibility in 1/bar (default: water ~4.5e-5)
        """
        self.atoms = atoms
        self.pressure_target = pressure          # bar
        self.tau_p = tau_p                       # fs
        self.timestep = timestep                 # fs
        self.compressibility = compressibility   # 1/bar

        # Scaling prefactor (constant): β * dt / τ_P
        self._scale_prefactor = compressibility * timestep / tau_p

    def get_pressure(self, velocities: np.ndarray) -> float:
        """
        Compute instantaneous pressure in bar via the virial theorem.

        P = (2*KE + W) / (3*V)

        The virial W is read from the calculator stress tensor if available,
        otherwise the ideal-gas (W=0) approximation is used.

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

        # Kinetic contribution (in eV)
        masses_amu = atoms.get_masses()
        # v in a.u. (Bohr/a.u.time) → convert to Å/fs
        v_ang_per_fs = velocities * BOHR_TO_ANGSTROM / AU_TO_FS
        # KE in eV: 0.5 * m[amu] * v²[Å²/fs²] * (amu·Å²/fs² → eV)
        ke_ev = 0.5 * np.sum(masses_amu[:, np.newaxis] * v_ang_per_fs**2) * AMU_ANG2_PER_FS2_TO_EV

        # Virial contribution from stress tensor (eV)
        virial_ev = 0.0
        try:
            stress = atoms.get_stress(voigt=True)   # eV/Å³, Voigt: xx,yy,zz,yz,xz,xy
            # Hydrostatic virial: W = -V * (σ_xx + σ_yy + σ_zz)
            virial_ev = -volume * (stress[0] + stress[1] + stress[2])
        except Exception:
            pass   # ideal-gas fallback: virial = 0

        # P = (2*KE + W) / (3*V)  in eV/Å³, then convert to bar
        pressure_ev_ang3 = (2.0 * ke_ev + virial_ev) / (3.0 * volume)
        return pressure_ev_ang3 * EV_PER_ANG3_TO_BAR

    def apply(self, velocities: np.ndarray) -> np.ndarray:
        """
        Apply one Berendsen barostat step: rescale cell and positions.

        The scaling factor μ satisfies:
            μ³ = 1 - β * (dt/τ_P) * (P_target - P)

        Positions and cell are scaled by μ; velocities are unchanged
        (volume change does not affect momenta in Berendsen scheme).

        Parameters
        ----------
        velocities : np.ndarray
            Current velocities in atomic units (unchanged, returned as-is)

        Returns
        -------
        float
            Instantaneous pressure before scaling (bar), for logging
        """
        pressure = self.get_pressure(velocities)
        mu3 = 1.0 - self._scale_prefactor * (self.pressure_target - pressure)
        # Clamp to avoid instability
        mu3 = float(np.clip(mu3, 0.5**3, 2.0**3))
        mu = mu3 ** (1.0 / 3.0)

        # Rescale cell and positions isotropically
        self.atoms.set_cell(self.atoms.get_cell() * mu, scale_atoms=True)

        return pressure
