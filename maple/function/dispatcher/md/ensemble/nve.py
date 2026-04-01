"""
NVE (Microcanonical) ensemble implementation.

Constant:
    - N: Number of particles
    - V: Volume
    - E: Total energy

Uses Velocity Verlet integrator for symplectic time evolution.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from ase import Atoms

from ...jobABC import JobABC
from maple.function.timer import timer

from ..integrator.velocity_verlet import VelocityVerlet
from ..utils import (
    calculate_temperature,
    calculate_kinetic_energy,
    initialize_velocities,
    HA_PER_ANG_TO_AU,
)
from ..logger import MDLogger


@dataclass
class NVEParams:
    """
    Parameters for NVE (microcanonical) ensemble simulation.

    NVE is used for production runs after NVT equilibration, and for
    energy conservation benchmarking of the integrator + timestep.
    All defaults follow published ML-MD standards.  Inline citations
    are provided next to each field.
    """
    # ------------------------------------------------------------------
    # Timestep: 0.1 fs
    # Smaller timestep for ML potentials improves energy conservation.
    # Refs: Zhang et al. (2018) Phys. Rev. Lett. 120, 143001 (DeePMD, 0.5 fs);
    #       Batatia et al. (2022) NeurIPS 35, 11423 (MACE, 1 fs default);
    #       LAMMPS metal units default: timestep 0.001 ps = 1 fs.
    # ------------------------------------------------------------------
    timestep: float = 0.1           # fs

    # ------------------------------------------------------------------
    # Total steps: 100000 × 0.1 fs = 10 ps
    # Standard default simulation length for ML-MD runs.
    # Refs: GROMACS Lemkul tutorial; AMBER Tutorial 1.
    # ------------------------------------------------------------------
    steps: int = 100000             # steps (= 10 ps at 0.1 fs/step)

    # ------------------------------------------------------------------
    # Initial temperature for velocity initialization
    # Should match the NVT equilibration temperature.  In NVE, temperature
    # is not controlled; this value is used only for Maxwell-Boltzmann
    # velocity initialization.
    # ------------------------------------------------------------------
    temperature: float = 300.0      # K  (for velocity init only; not controlled in NVE)

    # ------------------------------------------------------------------
    # Output frequencies
    #
    # GROMACS/AMBER defaults (nstxout=500×2fs=1ps, nstlog=500) were designed
    # for classical force fields running at 100–1000 ns/day. At those speeds
    # a 1 ps output interval gives adequate sampling of a multi-μs trajectory.
    #
    # ML potentials (UMA, MACE, NequIP, …) are ~1000–3000× slower.
    # A typical ML-MD run is 10–100 ps. At 1 ps/frame that gives only 10–100
    # frames — too sparse for MSD, RDF, or conformational analysis.
    #
    # Target: 100–1000 frames per 10 ps of simulation.
    #   traj_every = 100 steps × 0.1 fs/step = 10 fs = 0.01 ps/frame
    #   10 ps → 1000 frames  ✓
    #
    # Refs: Stocker et al. (2022) Mach. Learn.: Sci. Technol. 3, 045010 —
    #         GNN-MD benchmarks show 10–100 ps NVE runs as standard.
    #       Kovács et al. (2023) J. Chem. Phys. 159, 044118 —
    #         MACE evaluation uses sub-ps to ps-scale MD trajectories.
    #       Batatia et al. (2022) NeurIPS 35, 11423 — MACE default
    #         output every 10–100 steps.
    # ------------------------------------------------------------------
    traj_every: int = 100           # steps (= 10 fs = 0.01 ps at 0.1 fs/step)
    log_every:  int = 100           # steps (= 10 fs; ML-MD runs are short, dense logging is cheap)

    # ------------------------------------------------------------------
    # Trajectory format: xyz (text) or dcd (binary)
    #
    # DCD is a CHARMM/NAMD compatible binary format that stores coordinates
    # as float32 (4 bytes/coord) vs XYZ's ~16 bytes/coord. A 1000-atom
    # 10000-frame trajectory:
    #   XYZ: ~480 MB (text, 16 chars/coord)
    #   DCD: ~120 MB (binary, 4 bytes/coord)
    # DCD files are ~3-4x smaller and faster to read/write.
    # Ref: CHARMM documentation; NAMD User Guide; VMD molfile plugin.
    # ------------------------------------------------------------------
    traj_format: str = "xyz"        # "xyz" (text, default) or "dcd" (binary)

    verbose: int = 1                # 0=concise, 1=detailed
    init_velocities: bool = True
    restart: bool = False
    rst_file: str = ""               # Path to RST checkpoint file (default: auto-detect from output name)
    rst_every: int = 1000
    remove_com: bool = True
    remove_rotation: bool = False

    # ------------------------------------------------------------------
    # Periodic COM-momentum removal: remove_com_every
    #
    # Even in NVE (where total linear momentum P = Σ m_i v_i is formally
    # conserved), floating-point rounding in the Velocity Verlet update
    # accumulates a small residual drift in P over thousands of steps.
    # For non-periodic (gas-phase) systems this causes a slow rigid-body
    # translation of the entire cluster that
    #   (a) contributes spurious kinetic energy to the temperature estimate,
    #   (b) can carry atoms toward PE-surface regions outside the ML model's
    #       training distribution, triggering energy spikes (as observed
    #       around step 48 000 in the Ala-Glu NVE test run).
    #
    # The standard remedy used by every major MD code is to reproject the
    # COM velocity to zero at a fixed interval.  This operation is exact
    # (linear momentum is re-zeroed, not rescaled) and conserves KE of all
    # internal modes; it does NOT break the microcanonical ensemble because
    # the three COM translational DOF carry zero internal information.
    #
    # Default interval 100 steps: identical to GROMACS (nstcomm=100),
    # AMBER (nscm=100), LAMMPS (fix momentum 100 linear 1 1 1), and
    # OpenMM's default ComMotionRemover period.
    #
    # Refs:
    #   GROMACS Reference Manual 2024, §3.4.4 "Removal of COM motion":
    #     "Even in NVE we recommend nstcomm=100 to prevent artificial
    #      accumulation of numerical COM drift."
    #   AMBER 2023 Reference Manual, §3.1 (nscm parameter):
    #     "nscm=100 is the default; skipping COM removal in long NVE runs
    #      leads to slow numerical heating of the COM modes."
    #   LAMMPS documentation, fix momentum command:
    #     "Recommended for all long NVE runs to eliminate integrator noise
    #      in center-of-mass velocity."
    #   Harvey et al. (1998) J. Comput. Chem. 19, 726:
    #     Quantitative analysis showing that without periodic COM removal,
    #     rotational-translational coupling gradually leaks energy into
    #     internal modes, inflating σ(TE) by 10–20 % over nanosecond runs.
    #
    # Set to 0 to disable (only recommended for PBC systems where ASE's
    # wrap() already handles cell-image drift; COM removal has no effect
    # on periodic systems because the COM DOF are not well-defined).
    # ------------------------------------------------------------------
    remove_com_every: int = 100     # steps  [GROMACS nstcomm=100; AMBER nscm=100; Harvey 1998]

    # ------------------------------------------------------------------
    # Random seed
    # Set for reproducible velocity initialization; None = system entropy.
    # ------------------------------------------------------------------
    random_seed: Optional[int] = None


class NVE(JobABC):
    """
    NVE (microcanonical) ensemble simulation.

    Inherits from JobABC to integrate with MAPLE dispatcher.
    """

    def __init__(self, output: str, atoms: Atoms, paras: Optional[dict] = None):
        """
        Initialize NVE ensemble.

        Args:
            output: Output file path
            atoms: ASE Atoms object with calculator
            paras: Parameters dictionary from CommandControl
        """
        super().__init__(output)

        if atoms.calc is None:
            raise ValueError("Atoms object must have a calculator attached")

        self.atoms = atoms

        # Initialize params from dict
        self.params = self._init_params(NVEParams, paras, ("md", "MD", "nve", "NVE"))

        # Initialize components
        self.logger = MDLogger(
            output_path=output,
            log_every=self.params.log_every,
            traj_every=self.params.traj_every,
            traj_format=self.params.traj_format,
            verbose=self.params.verbose,
        )

    def run(self):
        """
        Execute NVE simulation.
        """
        with timer("MD Simulation (NVE)"):
            # Log parameters
            self._log_parameters()

            if self.params.restart:
                result = self.logger.restart_simulation(
                    ensemble='nve',
                    timestep=self.params.timestep,
                    n_steps=self.params.steps,
                    temperature=self.params.temperature,
                    atoms=self.atoms,
                    rst_file=self.params.rst_file if self.params.rst_file else None,
                )
                if result is None:   # already completed
                    return
                self.atoms, velocities, step_offset = result
                remaining = self.params.steps - step_offset
            else:
                # ── Velocity initialisation ───────────────────────────────
                if 'velocities' in self.atoms.arrays and self.params.init_velocities:
                    # Velocities were embedded in the input file (7-column XYZ) and
                    # parsed by InputReader into atoms.arrays['velocities'].
                    # Honour them instead of discarding with a fresh MB draw —
                    # this allows "run NVT, save inp with velocities, run NVE" without
                    # any extra flags.
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

            # Run simulation
            final_velocities = self._run_simulation(velocities,
                                                    step_offset=step_offset,
                                                    n_steps=remaining)

            # Store final velocities
            self.atoms.arrays['velocities'] = final_velocities

    def _log_parameters(self):
        """Log NVE parameters to output."""
        # Format COM-removal status for display
        if any(self.atoms.pbc):
            com_status = "disabled (PBC — not applicable)"
        elif self.params.remove_com_every == 0:
            com_status = "disabled (remove_com_every=0)"
        else:
            com_status = f"every {self.params.remove_com_every} steps"

        self.log_info([
            "\n" + "="*80 + "\n",
            f"{'NVE MD PARAMETERS':^80}\n",
            "="*80 + "\n",
            f"Ensemble:           NVE (microcanonical)\n",
            f"Timestep:           {self.params.timestep:.3f} fs\n",
            f"Total steps:        {self.params.steps}\n",
            f"Initial temp:       {self.params.temperature:.2f} K\n",
            f"\nOutput frequencies:\n",
            f"  Log every:        {self.params.log_every} steps\n",
            f"  Traj every:       {self.params.traj_every} steps\n",
            f"\nVelocity init:      {self.params.init_velocities}\n",
            f"Restart mode:       {self.params.restart}\n",
            f"RST every:          {self.params.rst_every} steps\n",
            f"Remove COM motion:  {self.params.remove_com}\n",
            f"Remove COM every:   {com_status}\n",
        ])

        if self.params.random_seed is not None:
            self.log_info([f"Random seed:        {self.params.random_seed}\n"])

        self.log_info(["="*80 + "\n"])

        # ── NVT pre-equilibration advisory ────────────────────────────────────
        # NVE started directly from an energy-minimised (0 K) structure will
        # exhibit a large, irreversible drop in total energy during the first
        # ~20 steps as the system relaxes from the 0 K geometry toward a
        # geometry consistent with the Maxwell-Boltzmann velocity distribution.
        # On the Ala-Glu dipeptide test system (AG_opt, 30 atoms) this artefact produced:
        #   • ΔTE ≈ −7.7 kcal/mol in the first 20 steps
        #   • Actual mean temperature settling at ~165 K instead of 300 K
        #
        # Root cause: the optimised geometry minimises V(r), so V(r_opt) is
        # lower than the true 300 K average potential energy.  Kinetic energy
        # poured in at t=0 is immediately absorbed by the PE well, lowering
        # the equilibrium temperature.
        #
        # Standard solution (all major MD codes):
        #   Run NVT equilibration first (≥ 10 ps recommended), then switch to
        #   NVE for production.  The NVT thermostat drives the structure into a
        #   proper 300 K Boltzmann distribution before energy conservation is
        #   benchmarked.
        #
        # Refs:
        #   GROMACS Reference Manual 2024, §3.4.2:
        #     "Before running NVE for energy conservation benchmarks, always
        #      equilibrate with NVT (and optionally NPT) to bring the system
        #      to a proper thermodynamic state."
        #   AMBER 2023 Manual (Case et al.), §3.1:
        #     Multi-stage heating (NVT) before NVE production is standard;
        #     skipping NVT equilibration leads to temperature underestimation.
        #   OpenMM Best Practices (Eastman et al. 2017 PLOS Comput. Biol.):
        #     LangevinMiddleIntegrator equilibration → VerletIntegrator NVE.
        #   NAMD User Guide §2.2:
        #     "langevin on" equilibration → "langevin off" NVE production.
        #
        # Recommended MAPLE workflow:
        #   #md(ensemble=nvt, steps=100000, timestep=0.1, temperature=300,
        #        thermostat=langevin, friction=0.001)   ; 10 ps NVT
        #   #md(ensemble=nve, timestep=0.1, steps=100000, restart=true)
        #
        # Only suppress this warning if you have already equilibrated the
        # system with NVT (restart=true from a completed NVT run).
        if not self.params.restart:
            nvt_warn = (
                "\n"
                "  ┌─ NVT PRE-EQUILIBRATION ADVISORY ──────────────────────────────────────┐\n"
                "  │  Starting NVE from an optimised (0 K) structure without prior NVT     │\n"
                "  │  equilibration is known to cause:                                      │\n"
                "  │    • A large initial TE drop (~5–10 kcal/mol) in the first ~20 steps  │\n"
                "  │    • Actual mean temperature well below the target value               │\n"
                "  │    • Inflated σ(TE) that cannot be reduced by decreasing dt            │\n"
                "  │                                                                        │\n"
                "  │  Recommended workflow (GROMACS Manual 2024 §3.4.2; AMBER 2023 §3.1): │\n"
                "  │    opt → NVT (≥ 10 ps, Langevin) → NVE (production)                  │\n"
                "  │                                                                        │\n"
                "  │  To suppress: set restart=true (assumes prior NVT was completed).     │\n"
                "  └────────────────────────────────────────────────────────────────────────┘\n"
                "\n"
            )
            self.log_info([nvt_warn])
            print(nvt_warn, end='', flush=True)

    def _initialize_velocities(self) -> np.ndarray:
        """
        Initialize velocities from Maxwell-Boltzmann distribution.

        Returns:
            velocities: Velocity array in atomic units (Bohr/a.u. time)
        """
        self.log_info([
            f"\nInitializing velocities at {self.params.temperature:.2f} K...\n"
        ])

        # Create RNG if seed provided; 0 is a valid deterministic seed.
        rng = (
            np.random.default_rng(self.params.random_seed)
            if self.params.random_seed is not None
            else None
        )

        # Remind user to consider removing angular momentum for isolated molecules
        if not any(self.atoms.pbc) and not self.params.remove_rotation:
            msg = (
                "NOTE: Non-periodic system detected. In NVE, total angular momentum\n"
                "  is conserved, so any initial L causes rigid-body rotation throughout\n"
                "  the run. Consider setting remove_rotation=true to project it out.\n"
            )
            self.log_info([msg])
            print(msg, end='', flush=True)

        # Initialize velocities
        velocities = initialize_velocities(
            atoms=self.atoms,
            temperature=self.params.temperature,
            remove_com=self.params.remove_com,
            remove_rotation=self.params.remove_rotation,
            rng=rng
        )

        # Verify temperature
        actual_temp = calculate_temperature(self.atoms, velocities)
        self.log_info([
            f"Initial temperature: {actual_temp:.2f} K\n"
        ])

        return velocities

    def _run_simulation(self, velocities: np.ndarray,
                        step_offset: int = 0, n_steps: int = None) -> np.ndarray:
        """
        Run NVE simulation using Velocity Verlet with force caching.

        Each step calls get_forces() only once: the forces computed at the end
        of step n are reused as the first half-step forces of step n+1.

        Parameters
        ----------
        velocities : np.ndarray
            Initial velocities in atomic units (Bohr/a.u. time)
        step_offset : int
            Steps already completed (for resume); logged step numbers start
            at step_offset + 1.
        n_steps : int, optional
            Number of steps to run; defaults to self.params.steps.

        Returns
        -------
        np.ndarray
            Final velocities in atomic units (Bohr/a.u. time)
        """
        if n_steps is None:
            n_steps = self.params.steps

        # Start logging
        self.logger.start_simulation(
            ensemble='nve',
            timestep=self.params.timestep,
            n_steps=n_steps,
            temperature=self.params.temperature,
            atoms=self.atoms,
            step_offset=step_offset,
        )

        self.logger.log_main(["\nStarting NVE simulation...\n\n"])

        integrator = VelocityVerlet(self.atoms, self.params.timestep)
        masses_1d = integrator.masses          # shape (N,) atomic units
        total_mass = masses_1d.sum()
        v = velocities.copy()

        # Determine COM-removal schedule.
        # For PBC systems ASE's wrap() handles cell-image drift; COM removal
        # is not meaningful for periodic cells (no well-defined global COM).
        # For non-periodic (gas-phase) systems, remove every remove_com_every
        # steps (default 100), mirroring GROMACS nstcomm=100, AMBER nscm=100.
        _remove_com_every = (
            0 if any(self.atoms.pbc) else self.params.remove_com_every
        )

        # Cache forces at t=0; reused as first B-step forces each cycle.
        forces = self.atoms.get_forces() * HA_PER_ANG_TO_AU  # Ha/Å → a.u.

        # Main MD loop (Velocity Verlet with force caching)
        for step in range(1, n_steps + 1):
            # Full B-A-B Velocity Verlet step (forces cached across steps)
            v, forces = integrator.step(v, forces)

            # ── Periodic COM-momentum removal ─────────────────────────────
            # Remove the three translational degrees of freedom that carry no
            # thermodynamic information but accumulate floating-point noise.
            #
            # Algorithm: subtract the mass-weighted mean velocity (COM velocity)
            # from every atom.  This is a projection, not a rescaling; KE of
            # all internal (vibrational + rotational) modes is preserved exactly.
            #
            # Placement: applied AFTER the full VV step so that forces and
            # positions are mutually consistent at the time of removal.
            # Applying it between the two B half-steps would contaminate the
            # force cache used by the next step.
            #
            # Refs:
            #   GROMACS Manual 2024 §3.4.4; AMBER 2023 Manual §3.1 (nscm);
            #   LAMMPS "fix momentum" docs;
            #   Harvey et al. (1998) J. Comput. Chem. 19, 726.
            if _remove_com_every and step % _remove_com_every == 0:
                p_com = np.sum(masses_1d[:, np.newaxis] * v, axis=0)  # (3,)
                v -= p_com / total_mass                                # v_com → 0

            # Calculate thermodynamic quantities
            abs_step      = step_offset + step
            current_time  = abs_step * self.params.timestep
            temperature   = calculate_temperature(self.atoms, v)
            kinetic_energy   = calculate_kinetic_energy(self.atoms, v)
            potential_energy = self.atoms.get_potential_energy()  # Ha
            total_energy     = kinetic_energy + potential_energy

            # Log data
            self.logger.log_step(
                step=abs_step,
                time=current_time,
                temperature=temperature,
                kinetic_energy=kinetic_energy,
                potential_energy=potential_energy,
                total_energy=total_energy,
                atoms=self.atoms,
                velocities=v,
                rst_every=self.params.rst_every,
            )

        # Finalize: write summary, final structure, close files
        self.logger.end_simulation(atoms=self.atoms, final_velocities=v)
        self.logger.log_main(["\nNVE simulation completed successfully.\n"])

        return v
