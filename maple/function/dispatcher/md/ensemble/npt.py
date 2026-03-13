"""
NPT (isothermal-isobaric) ensemble implementation.

Supports two combinations:
    thermostat: 'langevin' | 'v-rescale'  (default: v-rescale)
    barostat:   'berendsen' | 'c-rescale' (default: c-rescale)

Recommended combination for production MLP runs:
    thermostat=v-rescale + barostat=c-rescale
    → both produce the correct NPT ensemble.

Berendsen variants are suitable for rapid pre-equilibration but suppress
pressure/temperature fluctuations and do not generate correct ensemble averages.

Integration order each step (Velocity Verlet with midstep thermostat):
    1. B  half-step velocity  (conservative force, cached from previous step)
    2. A  full-step position + PBC wrap
    3. O  thermostat step
    4. B  half-step velocity  (new forces, cached for next step)
    5. Barostat: stochastic/deterministic cell rescaling

Requirements:
    - Atoms object must have a periodic cell (atoms.pbc must be True)
    - Calculator should support stress tensor evaluation for accurate pressure

References:
    Berendsen et al., J. Chem. Phys. 81, 3684 (1984).
    Bussi, Donadio & Parrinello, J. Chem. Phys. 126, 014101 (2007).
    Bernetti & Bussi, J. Chem. Phys. 153, 114107 (2020).
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from ase import Atoms

from ...jobABC import JobABC
from maple.function.timer import timer

from ..thermostat.langevin import LangevinThermostat
from ..thermostat.vrescale import VRescaleThermostat
from ..barostat.berendsen import BerendsenBarostat
from ..barostat.crescale import CRescaleBarostat
from ..utils import (
    calculate_temperature,
    calculate_kinetic_energy,
    initialize_velocities,
    HA_PER_ANG_TO_AU,
    BOHR_TO_ANGSTROM,
    FS_TO_AU,
    AMU_TO_AU,
)
from ..logger import MDLogger


@dataclass
class NPTParams:
    """Parameters for NPT simulation."""
    timestep:        float = 0.5          # fs
    steps:           int   = 10000        # total steps
    temperature:     float = 300.0        # K
    pressure:        float = 1.0          # bar
    thermostat:      str   = 'v-rescale'  # 'langevin' | 'v-rescale'
    barostat:        str   = 'c-rescale'  # 'berendsen' | 'c-rescale'
    # Langevin-specific
    friction:        float = 0.001        # 1/fs
    # V-rescale-specific
    tau_t:           float = 200.0        # fs
    # Barostat common
    tau_p:           float = 2000.0       # fs  (larger for MLP)
    compressibility: float = 4.5e-5       # 1/bar (water, see utils.DEFAULT_COMPRESSIBILITY)
    # Common
    traj_every:      int   = 10
    log_every:       int   = 100
    init_velocities: bool  = True
    remove_com:      bool  = True
    random_seed: Optional[int] = None


class NPT(JobABC):
    """
    NPT (isothermal-isobaric) ensemble simulation.

    Integrates with the MAPLE dispatcher via JobABC.
    """

    _THERMOSTAT_CHOICES = {'langevin', 'v-rescale'}
    _BAROSTAT_CHOICES   = {'berendsen', 'c-rescale'}

    def __init__(self, output: str, atoms: Atoms, paras: Optional[dict] = None):
        super().__init__(output)

        if atoms.calc is None:
            raise ValueError("Atoms object must have a calculator attached")
        if not any(atoms.pbc):
            raise ValueError(
                "NPT ensemble requires a periodic cell (atoms.pbc must be True). "
                "Use NVT or NVE for non-periodic systems."
            )

        self.atoms = atoms
        self.params = self._init_params(NPTParams, paras, ("md", "MD", "npt", "NPT"))

        if self.params.thermostat not in self._THERMOSTAT_CHOICES:
            raise ValueError(
                f"Unknown thermostat '{self.params.thermostat}'. "
                f"Choose from: {self._THERMOSTAT_CHOICES}"
            )
        if self.params.barostat not in self._BAROSTAT_CHOICES:
            raise ValueError(
                f"Unknown barostat '{self.params.barostat}'. "
                f"Choose from: {self._BAROSTAT_CHOICES}"
            )

        # Warn if user set Langevin-specific params but chose v-rescale (or vice versa)
        if self.params.thermostat == 'v-rescale' and paras and 'friction' in (paras or {}):
            self.log_info([
                "\n*** WARNING: 'friction' parameter was specified but thermostat is 'v-rescale'.\n"
                "    The friction parameter is only used by the Langevin thermostat.\n"
                "    If you intended Langevin dynamics, add: thermostat=langevin\n\n"
            ])
        if self.params.thermostat == 'langevin' and paras and 'tau_t' in (paras or {}):
            self.log_info([
                "\n*** WARNING: 'tau_t' parameter was specified but thermostat is 'langevin'.\n"
                "    The tau_t parameter is only used by the V-rescale thermostat.\n\n"
            ])

        self._rng = (np.random.default_rng(self.params.random_seed)
                     if self.params.random_seed is not None
                     else np.random.default_rng())

        if self.params.thermostat == 'langevin':
            self.thermostat = LangevinThermostat(
                atoms,
                temperature=self.params.temperature,
                friction=self.params.friction,
                timestep=self.params.timestep,
                rng=self._rng,
            )
        else:  # v-rescale
            self.thermostat = VRescaleThermostat(
                atoms,
                temperature=self.params.temperature,
                tau_t=self.params.tau_t,
                timestep=self.params.timestep,
                rng=self._rng,
            )

        if self.params.barostat == 'berendsen':
            self.barostat = BerendsenBarostat(
                atoms,
                pressure=self.params.pressure,
                tau_p=self.params.tau_p,
                timestep=self.params.timestep,
                compressibility=self.params.compressibility,
            )
        else:  # c-rescale
            self.barostat = CRescaleBarostat(
                atoms,
                pressure=self.params.pressure,
                temperature=self.params.temperature,
                tau_p=self.params.tau_p,
                timestep=self.params.timestep,
                compressibility=self.params.compressibility,
                rng=self._rng,
            )

        self.logger = MDLogger(
            output_path=output,
            log_every=self.params.log_every,
            traj_every=self.params.traj_every,
        )

    def run(self):
        """Execute NPT simulation."""
        with timer("MD Simulation (NPT)"):
            self._log_parameters()

            if self.params.init_velocities:
                velocities = self._initialize_velocities()
            else:
                if 'velocities' not in self.atoms.arrays:
                    raise ValueError(
                        "init_velocities=False but no velocities found in atoms.arrays"
                    )
                velocities = self.atoms.arrays['velocities']

            final_velocities = self._run_simulation(velocities)
            self.atoms.arrays['velocities'] = final_velocities

    def _log_parameters(self):
        """Log NPT parameters to output."""
        lines = [
            "\n" + "=" * 80 + "\n",
            f"{'NPT MD PARAMETERS':^80}\n",
            "=" * 80 + "\n",
            f"Ensemble:              NPT (isothermal-isobaric)\n",
            f"Thermostat:            {self.params.thermostat}\n",
            f"Barostat:              {self.params.barostat}\n",
            f"Timestep:              {self.params.timestep:.3f} fs\n",
            f"Total steps:           {self.params.steps}\n",
            f"Temperature:           {self.params.temperature:.2f} K\n",
            f"Target pressure:       {self.params.pressure:.2f} bar\n",
        ]
        if self.params.thermostat == 'langevin':
            lines.append(f"Friction (γ):          {self.params.friction:.4f} 1/fs\n")
        else:
            lines.append(f"τ_T:                   {self.params.tau_t:.1f} fs\n")
        lines += [
            f"τ_P:                   {self.params.tau_p:.1f} fs\n",
            f"Compressibility:       {self.params.compressibility:.2e} 1/bar\n",
            f"\nOutput frequencies:\n",
            f"  Log every:           {self.params.log_every} steps\n",
            f"  Traj every:          {self.params.traj_every} steps\n",
            f"\nVelocity init:         {self.params.init_velocities}\n",
            f"Remove COM motion:     {self.params.remove_com}\n",
        ]
        if self.params.random_seed is not None:
            lines.append(f"Random seed:           {self.params.random_seed}\n")
        lines.append("=" * 80 + "\n")
        self.log_info(lines)

    def _initialize_velocities(self) -> np.ndarray:
        """Initialize velocities from Maxwell-Boltzmann distribution."""
        self.log_info([f"\nInitializing velocities at {self.params.temperature:.2f} K...\n"])
        velocities = initialize_velocities(
            atoms=self.atoms,
            temperature=self.params.temperature,
            remove_com=self.params.remove_com,
            rng=self._rng,
        )
        actual_temp = calculate_temperature(self.atoms, velocities)
        self.log_info([f"Initial temperature: {actual_temp:.2f} K\n"])
        return velocities

    def _run_simulation(self, velocities: np.ndarray) -> np.ndarray:
        """
        Run NPT simulation.

        Velocity Verlet with midstep thermostat + barostat step after each
        complete integration cycle.
        """
        self.logger.start_simulation(
            ensemble='npt',
            timestep=self.params.timestep,
            n_steps=self.params.steps,
            temperature=self.params.temperature,
            atoms=self.atoms,
            pressure=self.params.pressure,
        )
        self.logger.log_main([
            f"\nStarting NPT simulation "
            f"({self.params.thermostat} + {self.params.barostat})...\n\n"
        ])

        dt = self.params.timestep * FS_TO_AU
        masses = (self.atoms.get_masses() * AMU_TO_AU)[:, np.newaxis]
        v = velocities.copy()

        # Cache forces at t=0; reused as first B-step forces each cycle.
        forces = self.atoms.get_forces() * HA_PER_ANG_TO_AU  # Ha/Å → a.u.

        for step in range(1, self.params.steps + 1):
            # B: half-step velocity (uses cached forces from end of previous step)
            v += 0.5 * forces / masses * dt

            # A: full-step position
            self.atoms.set_positions(self.atoms.get_positions() + v * dt * BOHR_TO_ANGSTROM)
            if any(self.atoms.pbc):
                self.atoms.wrap()

            # O: thermostat
            v = self.thermostat.apply(v)

            # B: half-step velocity with new forces; cache for next step
            forces = self.atoms.get_forces() * HA_PER_ANG_TO_AU  # Ha/Å → a.u.
            v += 0.5 * forces / masses * dt

            # Barostat: rescale cell, returns instantaneous pressure
            pressure = self.barostat.apply(v)

            current_time     = step * self.params.timestep
            temperature      = calculate_temperature(self.atoms, v)
            kinetic_energy   = calculate_kinetic_energy(self.atoms, v)
            potential_energy = self.atoms.get_potential_energy()  # Ha
            volume           = self.atoms.get_volume()

            self.logger.log_step(
                step=step,
                time=current_time,
                temperature=temperature,
                kinetic_energy=kinetic_energy,
                potential_energy=potential_energy,
                total_energy=kinetic_energy + potential_energy,
                atoms=self.atoms,
                velocities=v,
                pressure=pressure,
                volume=volume,
            )

        self.logger.end_simulation()
        self.logger.log_main(["\nNPT simulation completed successfully.\n"])
        return v
