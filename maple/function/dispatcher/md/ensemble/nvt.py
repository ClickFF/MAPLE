"""
NVT (canonical) ensemble implementation.

Supports two thermostat algorithms:
    - langevin: Langevin dynamics (Leimkuhler & Matthews, AMRX 2013)
                Strong coupling; per-atom stochastic force.
                Good for equilibration or when strong damping is wanted.
    - v-rescale: Stochastic velocity rescaling (Bussi et al., 2007)
                 Correct canonical ensemble; global velocity scaling.
                 Less perturbation to dynamics; preferred for production.

Integration loop (Velocity Verlet with midstep thermostat):
    B: half-step velocity with conservative force  (cached from previous step)
    A: full-step position update + PBC wrap
    O: thermostat step (Langevin O-U step  OR  V-rescale global rescaling)
    B: half-step velocity with newly computed forces (cached for next step)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from ase import Atoms

from ...jobABC import JobABC
from maple.function.timer import timer

from ..thermostat.langevin import LangevinThermostat
from ..thermostat.vrescale import VRescaleThermostat
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
class NVTParams:
    """Parameters for NVT simulation."""
    timestep:        float = 0.5       # fs
    steps:           int   = 10000     # total steps
    temperature:     float = 300.0     # K
    thermostat:      str   = 'v-rescale'  # 'langevin' | 'v-rescale'
    # Langevin-specific
    friction:        float = 0.001     # 1/fs  (γ; GROMACS default ~0.001)
    # V-rescale-specific
    tau_t:           float = 200.0     # fs  (temperature coupling time)
    # Common
    traj_every:      int   = 10
    log_every:       int   = 100
    init_velocities: bool  = True
    remove_com:      bool  = True
    random_seed: Optional[int] = None


class NVT(JobABC):
    """
    NVT (canonical) ensemble simulation.

    Integrates with the MAPLE dispatcher via JobABC.
    """

    _THERMOSTAT_CHOICES = {'langevin', 'v-rescale'}

    def __init__(self, output: str, atoms: Atoms, paras: Optional[dict] = None):
        super().__init__(output)

        if atoms.calc is None:
            raise ValueError("Atoms object must have a calculator attached")

        self.atoms = atoms
        self.params = self._init_params(NVTParams, paras, ("md", "MD", "nvt", "NVT"))

        if self.params.thermostat not in self._THERMOSTAT_CHOICES:
            raise ValueError(
                f"Unknown thermostat '{self.params.thermostat}'. "
                f"Choose from: {self._THERMOSTAT_CHOICES}"
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

        self.logger = MDLogger(
            output_path=output,
            log_every=self.params.log_every,
            traj_every=self.params.traj_every,
        )

    def run(self):
        """Execute NVT simulation."""
        with timer("MD Simulation (NVT)"):
            self._log_parameters()

            if self.params.init_velocities:
                velocities = self._initialize_velocities()
            else:
                if 'velocities' not in self.atoms.arrays:
                    raise ValueError("init_velocities=False but no velocities found in atoms.arrays")
                velocities = self.atoms.arrays['velocities']

            final_velocities = self._run_simulation(velocities)
            self.atoms.arrays['velocities'] = final_velocities

    def _log_parameters(self):
        """Log NVT parameters to output."""
        lines = [
            "\n" + "=" * 80 + "\n",
            f"{'NVT MD PARAMETERS':^80}\n",
            "=" * 80 + "\n",
            f"Ensemble:           NVT (canonical)\n",
            f"Thermostat:         {self.params.thermostat}\n",
            f"Timestep:           {self.params.timestep:.3f} fs\n",
            f"Total steps:        {self.params.steps}\n",
            f"Temperature:        {self.params.temperature:.2f} K\n",
        ]
        if self.params.thermostat == 'langevin':
            lines.append(f"Friction (γ):       {self.params.friction:.4f} 1/fs\n")
        else:
            lines.append(f"τ_T:                {self.params.tau_t:.1f} fs\n")
        lines += [
            f"\nOutput frequencies:\n",
            f"  Log every:        {self.params.log_every} steps\n",
            f"  Traj every:       {self.params.traj_every} steps\n",
            f"\nVelocity init:      {self.params.init_velocities}\n",
            f"Remove COM motion:  {self.params.remove_com}\n",
        ]
        if self.params.random_seed is not None:
            lines.append(f"Random seed:        {self.params.random_seed}\n")
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
        Run NVT simulation with Velocity Verlet + midstep thermostat.

        The O-step is either the Langevin Ornstein-Uhlenbeck step or the
        V-rescale global kinetic energy rescaling, depending on thermostat choice.
        """
        self.logger.start_simulation(
            ensemble='nvt',
            timestep=self.params.timestep,
            n_steps=self.params.steps,
            temperature=self.params.temperature,
            atoms=self.atoms,
        )
        self.logger.log_main([
            f"\nStarting NVT simulation ({self.params.thermostat})...\n\n"
        ])

        dt = self.params.timestep * FS_TO_AU
        masses = (self.atoms.get_masses() * AMU_TO_AU)[:, np.newaxis]
        v = velocities.copy()

        # Cache forces at t=0 to avoid double get_forces() per step.
        # Each step ends with forces at the new position; these are reused
        # as the first B-step forces of the next step (standard BAOAB caching).
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

            current_time     = step * self.params.timestep
            temperature      = calculate_temperature(self.atoms, v)
            kinetic_energy   = calculate_kinetic_energy(self.atoms, v)
            potential_energy = self.atoms.get_potential_energy()  # Ha

            self.logger.log_step(
                step=step,
                time=current_time,
                temperature=temperature,
                kinetic_energy=kinetic_energy,
                potential_energy=potential_energy,
                total_energy=kinetic_energy + potential_energy,
                atoms=self.atoms,
                velocities=v,
            )

        self.logger.end_simulation()
        self.logger.log_main(["\nNVT simulation completed successfully.\n"])
        return v
