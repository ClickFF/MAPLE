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

from ..integrator.velocity_verlet import VelocityVerlet
from ..thermostat.langevin import LangevinThermostat
from ..thermostat.vrescale import VRescaleThermostat
from ..barostat.berendsen import BerendsenBarostat
from ..barostat.crescale import CRescaleBarostat
from ..utils import (
    calculate_temperature,
    calculate_kinetic_energy,
    initialize_velocities,
    HA_PER_ANG_TO_AU,
)
from ..rst_io import get_rng_state_hex, restore_rng_from_hex
from ..logger import MDLogger


@dataclass
class NPTParams:
    """
    Parameters for NPT (isothermal-isobaric) ensemble simulation.

    NPT is used for density equilibration and computing thermodynamic
    properties at constant pressure (e.g., liquid density, compressibility).
    Defaults follow published standards for ML potentials.

    Thermostat default: v-rescale (Bussi et al. 2007 JCP 126, 014101)
      — correct canonical ensemble; less perturbative than Langevin.
    Barostat default: c-rescale (Bernetti & Bussi 2020 JCP 153, 114107)
      — correct isothermal-isobaric ensemble; analogue of v-rescale for pressure.

    Recommended production combination: thermostat=v-rescale + barostat=c-rescale.
    Berendsen variants are suitable for rapid pre-equilibration only.
    """
    # ------------------------------------------------------------------
    # Timestep: 0.1 fs
    # Smaller timestep for ML potentials improves energy conservation.
    # Refs: Zhang et al. (2018) Phys. Rev. Lett. 120, 143001 (DeePMD);
    #       Batatia et al. (2022) NeurIPS 35, 11423 (MACE).
    # ------------------------------------------------------------------
    timestep:        float = 0.1          # fs

    # ------------------------------------------------------------------
    # Total steps: 100000 × 0.1 fs = 10 ps
    # Standard default simulation length for ML-MD runs.
    # Refs: GROMACS Lemkul tutorial; AMBER Tutorial 1.
    # ------------------------------------------------------------------
    steps:           int   = 100000       # steps (= 10 ps at 0.1 fs/step)

    temperature:     float = 300.0        # K
    pressure:        float = 1.0          # bar

    # ------------------------------------------------------------------
    # Thermostat: v-rescale (default for NPT)
    # V-rescale produces correct canonical KE distribution while perturbing
    # dynamics less than Langevin, making it better suited for NPT where
    # both thermostat and barostat act each step.
    # Ref: GROMACS default since v4.5 (Bussi et al. 2007).
    # ------------------------------------------------------------------
    thermostat:      str   = 'v-rescale'  # [GROMACS default; Bussi 2007]

    # ------------------------------------------------------------------
    # Barostat: c-rescale (default for NPT)
    # C-rescale is the correct NPT barostat (Bernetti & Bussi 2020).
    # Unlike Berendsen, it produces the full Gibbs (N,P,T) distribution.
    # ------------------------------------------------------------------
    barostat:        str   = 'c-rescale'  # [Bernetti & Bussi 2020 JCP 153, 114107]

    # Langevin-specific (only used when thermostat='langevin')
    friction:        float = 0.001        # 1/fs = 1 ps⁻¹

    # ------------------------------------------------------------------
    # V-rescale τ_T: 200 fs
    # Slightly larger than NVT default (100 fs) to avoid over-coupling
    # when both thermostat and barostat act each step.
    # Ref: GROMACS NPT tutorial: tau_t = 0.1 ps = 100 fs;
    #      CHARMM-GUI NPT protocol: tau_t = 1 ps (conservative).
    # ------------------------------------------------------------------
    tau_t:           float = 200.0        # fs  [GROMACS NPT tutorial]

    # ------------------------------------------------------------------
    # Barostat τ_P: 2000 fs = 2 ps
    # Larger than classical MD defaults (0.5–1 ps) to account for ML
    # potential noise in instantaneous pressure.  Noisy pressure → large
    # τ_P needed to avoid volume instability.
    # Ref: GROMACS Lemkul NPT tutorial: tau_p = 2.0 ps;
    #      Bernetti & Bussi 2020 §III: τ_P ≥ 1 ps recommended.
    # ------------------------------------------------------------------
    tau_p:           float = 2000.0       # fs  [GROMACS Lemkul; Bernetti 2020]

    # ------------------------------------------------------------------
    # Isothermal compressibility: 4.5e-5 1/bar (liquid water, 300 K, 1 bar)
    # Used by both Berendsen and C-rescale barostats as a scaling prefactor.
    # The barostat dynamics are not very sensitive to this value; using water
    # as a default is standard practice for biomolecular systems.
    # Ref: CRC Handbook of Chemistry and Physics; GROMACS mdp default.
    # ------------------------------------------------------------------
    compressibility: float = 4.5e-5       # 1/bar  [CRC Handbook; GROMACS default]

    # ------------------------------------------------------------------
    # Output frequencies
    #
    # ML potentials are ~1000–3000× slower than classical FFs.
    # A typical ML-NPT run is 10–50 ps; dense output is needed to monitor
    # density convergence and detect volume instabilities early.
    #
    # Target: 100–1000 frames per 10 ps.
    #   traj_every = 100 steps × 0.1 fs/step = 10 fs = 0.01 ps/frame
    #   10 ps → 1000 frames  ✓
    #
    # Refs: Stocker et al. (2022) Mach. Learn.: Sci. Technol. 3, 045010 —
    #         GNN-MD benchmarks, typical run 10–100 ps with dense output.
    #       Kovács et al. (2023) J. Chem. Phys. 159, 044118 — MACE evaluation
    #         with per-step monitoring of thermodynamic convergence.
    # ------------------------------------------------------------------
    traj_every:      int   = 100          # steps (= 10 fs = 0.01 ps at 0.1 fs/step)
    log_every:       int   = 100          # steps (= 10 fs)

    # ------------------------------------------------------------------
    # Trajectory format: xyz (text) or dcd (binary)
    # DCD binary format is ~3-4x smaller than XYZ and faster to read/write.
    # Ref: CHARMM documentation; VMD molfile plugin.
    # ------------------------------------------------------------------
    traj_format:     str   = "xyz"        # "xyz" (text, default) or "dcd" (binary)

    verbose:         int   = 1
    init_velocities: bool  = True
    restart:          bool  = False
    rst_file:         str   = ""           # Path to RST checkpoint file (default: auto-detect)
    rst_every:        int   = 1000
    remove_com:       bool  = True
    remove_rotation:  bool  = False
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
            traj_format=self.params.traj_format,
            verbose=self.params.verbose,
        )

    def run(self):
        """Execute NPT simulation."""
        with timer("MD Simulation (NPT)"):
            self._log_parameters()

            if self.params.restart:
                result = self.logger.restart_simulation(
                    ensemble='npt',
                    timestep=self.params.timestep,
                    n_steps=self.params.steps,
                    temperature=self.params.temperature,
                    atoms=self.atoms,
                    pressure=self.params.pressure,
                    rst_file=self.params.rst_file if self.params.rst_file else None,
                )
                if result is None:   # already completed
                    return
                self.atoms, velocities, step_offset = result
                # Restore RNG state for deterministic continuation
                if self.logger.resumed_rng_state is not None:
                    restore_rng_from_hex(self._rng, self.logger.resumed_rng_state)
                remaining = self.params.steps - step_offset
            else:
                if 'velocities' in self.atoms.arrays and self.params.init_velocities:
                    velocities = self.atoms.arrays['velocities']
                    t_check = calculate_temperature(self.atoms, velocities)
                    self.log_info([
                        f"\nVelocities loaded from input file "
                        f"(T = {t_check:.2f} K); skipping random initialisation.\n"
                    ])
                elif self.params.init_velocities:
                    velocities = self._initialize_velocities()
                else:
                    if 'velocities' not in self.atoms.arrays:
                        raise ValueError(
                            "init_velocities=False, "
                            "but no velocities found in atoms.arrays"
                        )
                    velocities = self.atoms.arrays['velocities']
                step_offset = 0
                remaining   = self.params.steps

            final_velocities = self._run_simulation(velocities,
                                                    step_offset=step_offset,
                                                    n_steps=remaining)
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
            f"Restart mode:          {self.params.restart}\n",
            f"RST every:             {self.params.rst_every} steps\n",
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
            remove_rotation=self.params.remove_rotation,
            rng=self._rng,
        )
        actual_temp = calculate_temperature(self.atoms, velocities)
        self.log_info([f"Initial temperature: {actual_temp:.2f} K\n"])
        return velocities

    def _run_simulation(self, velocities: np.ndarray,
                        step_offset: int = 0, n_steps: int = None) -> np.ndarray:
        """
        Run NPT simulation.

        Velocity Verlet with midstep thermostat + barostat step after each
        complete integration cycle.
        """
        if n_steps is None:
            n_steps = self.params.steps

        self.logger.start_simulation(
            ensemble='npt',
            timestep=self.params.timestep,
            n_steps=n_steps,
            temperature=self.params.temperature,
            atoms=self.atoms,
            pressure=self.params.pressure,
            step_offset=step_offset,
        )
        self.logger.log_main([
            f"\nStarting NPT simulation "
            f"({self.params.thermostat} + {self.params.barostat})...\n\n"
        ])

        integrator = VelocityVerlet(self.atoms, self.params.timestep)
        v = velocities.copy()

        # Cache forces at t=0; complete_split_step() returns fresh forces each step
        # so only one ML force evaluation occurs per BAOAB cycle.
        forces = self.atoms.get_forces() * HA_PER_ANG_TO_AU  # Ha/Å → a.u.

        for step in range(1, n_steps + 1):
            # BAOAB splitting (Leimkuhler & Matthews 2013) + barostat after full step:
            #   B: half-kick  A(dt/2): half-position  O: thermostat
            #   A(dt/2): half-position  B: half-kick  Barostat: cell rescale

            # B-A(half): half-kick + half-position; forces cached from prev step
            v_half = integrator.split_step(v, forces)

            # O: thermostat (V-rescale or Langevin)
            v_therm = self.thermostat.apply(v_half)

            # A(half)-B: half-position + force eval + half-kick; returns cached forces
            v, forces = integrator.complete_split_step(v_therm)

            # Barostat: rescale cell after full BAOAB cycle, returns instantaneous pressure
            pressure = self.barostat.apply(v)

            abs_step         = step_offset + step
            current_time     = abs_step * self.params.timestep
            temperature      = calculate_temperature(self.atoms, v)
            kinetic_energy   = calculate_kinetic_energy(self.atoms, v)
            potential_energy = self.atoms.get_potential_energy()  # Ha
            volume           = self.atoms.get_volume()

            self.logger.log_step(
                step=abs_step,
                time=current_time,
                temperature=temperature,
                kinetic_energy=kinetic_energy,
                potential_energy=potential_energy,
                total_energy=kinetic_energy + potential_energy,
                atoms=self.atoms,
                velocities=v,
                pressure=pressure,
                volume=volume,
                rng_state=get_rng_state_hex(self._rng),
                rst_every=self.params.rst_every,
            )

        self.logger.end_simulation(
            atoms=self.atoms,
            final_velocities=v,
            rng_state=get_rng_state_hex(self._rng)
        )
        self.logger.log_main(["\nNPT simulation completed successfully.\n"])
        return v
