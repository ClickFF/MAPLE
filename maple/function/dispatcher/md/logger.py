"""
MD simulation logging and output management.

Handles:
    - Thermodynamic data output (.dat file)
    - XYZ/DCD trajectory output
    - Progress logging to main output
    - Final summary statistics
"""

import time as _time
import numpy as np
from pathlib import Path
from typing import Optional, TextIO, Any
from ase import Atoms

from .utils import write_xyz_frame
from .rst_io import read_rst, rotate_rst_checkpoint
from .dcd_writer import DCDWriter


def _backup_file(path: Path) -> Optional[Path]:
    """
    GROMACS-style file backup: if *path* exists, rename it to
    #<name>.<ext>.1# (incrementing until a free slot is found).

    Returns the backup path if a backup was made, else None.
    """
    if not path.exists():
        return None
    n = 1
    while True:
        backup = path.parent / f"#{path.name}.{n}#"
        if not backup.exists():
            path.rename(backup)
            return backup
        n += 1


# ========== Unit Conversion Constants ==========

# 1 Hartree = 2625.4996 kJ/mol  (NIST CODATA 2018)
HARTREE_TO_KJ_PER_MOL = 2625.4996
# 1 Hartree = 627.5095 kcal/mol (1 kJ = 0.239006 kcal)
HARTREE_TO_KCAL_PER_MOL = 627.5095
# 1 fs = 1e-6 ns
FS_TO_NS = 1e-6


# ========== Progress Formatting Helpers ==========

def _fmt_duration(seconds: float) -> str:
    """Format wall-clock duration into human-readable string, e.g. '1h 23m 45s'."""
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _fmt_eta(seconds: float) -> str:
    """Format ETA; show '∞' when remaining time is unknown or very large."""
    if seconds <= 0 or seconds > 1e7:   # > ~4 months
        return "∞"
    return _fmt_duration(seconds)


class MDLogger:
    """
    Manages MD simulation output and logging.

    Attributes:
        output_base: Base path for output files (without extension)
        thermo_file: Thermodynamics data file handle
        traj_file: XYZ trajectory file handle
        main_output: Main output file path
        log_every: Log frequency (steps)
        traj_every: Trajectory write frequency (steps)
    """

    eV2Hartree = 1 / 27.211386245988

    def __init__(
        self,
        output_path: str,
        log_every: int = 100,
        traj_every: int = 10,
        verbose: int = 1,
        traj_format: str = "xyz",
    ):
        """
        Initialize MD logger.

        Args:
            output_path: Main output file path (e.g., "task.out")
            log_every:   Frequency to log thermodynamic data (steps)
            traj_every:  Frequency to write trajectory frames (steps)
            verbose:     Progress verbosity level
                           0 = no progress lines (thermo file still written)
                           1 = GROMACS-style progress line every log_every steps
                           2 = verbose (step-level timing detail)
            traj_format: Trajectory output format: "xyz" (text) or "dcd" (binary)
        """
        self.main_output = output_path
        self.log_every   = log_every
        self.traj_every  = traj_every
        self.verbose      = verbose
        self.traj_format  = traj_format.lower()

        # Generate file paths
        base   = Path(output_path).stem
        parent = Path(output_path).parent

        self.thermo_path   = parent / f"{base}_md_thermo.dat"
        self.traj_path     = parent / f"{base}_md_traj.{self.traj_format}"
        self.summary_path  = parent / f"{base}_md_summary.txt"
        self.final_path    = parent / f"{base}_final.xyz"   # GROMACS confout.gro equivalent
        self.rst_path      = parent / f"{base}_md.rst"
        self.rst_prev_path = parent / f"{base}_md_prev.rst"

        # File handles (opened in start_simulation)
        self.thermo_file: Optional[TextIO] = None
        self.traj_file:   Any = None  # TextIO for XYZ, DCDWriter for DCD

        # Statistics tracking
        self.energies            = []
        self.temperatures        = []
        self.times               = []
        self.pressures           = []   # NPT only
        self.kinetic_energies    = []   # for KE–PE anti-correlation
        self.potential_energies  = []   # for KE–PE anti-correlation
        self._n_atoms            = 0    # set in start_simulation
        self._is_pbc             = False  # set in start_simulation; affects N_dof

        # Performance / progress tracking (set in start_simulation)
        self._n_steps:    int   = 0
        self._timestep:   float = 0.0  # fs
        self._wall_start: float = 0.0  # time.perf_counter() at simulation start
        # Wall time of the last log_step call (for rolling speed estimate)
        self._last_wall:  float = 0.0
        self._last_step:  int   = 0
        self._time_col_w: int   = 14   # width of Time(ps) column header
        self._total_ps:   float = 0.0  # total simulation time in ps

        # RNG state restored from the last trajectory frame on resume (None if not present)
        self.resumed_rng_state: Optional[str] = None

    def start_simulation(self, ensemble: str, timestep: float, n_steps: int,
                        temperature: float, atoms: Atoms,
                        pressure: float = None, step_offset: int = 0):
        """
        Initialize output files and write headers.

        Args:
            ensemble: Ensemble type (nve, nvt, npt)
            timestep: Timestep in fs
            n_steps: Number of steps to run in this segment
            temperature: Target temperature in K
            atoms: ASE Atoms object
            pressure: Target pressure in bar (NPT only)
            step_offset: Step number already completed (for resume)
        """
        self._ensemble    = ensemble.lower()
        self._n_atoms     = len(atoms)
        self._is_pbc      = bool(any(atoms.pbc))
        self._timestep    = timestep
        self._step_offset = step_offset
        # Total steps across the full run (for progress %)
        self._n_steps     = n_steps + step_offset

        # Record wall-clock start; also seed the rolling-speed anchor
        self._wall_start = _time.perf_counter()
        self._last_wall  = self._wall_start
        self._last_step  = step_offset

        if step_offset == 0:
            # Open files (back up any pre-existing files first, GROMACS-style)
            backup_msgs = []
            main_out_path = Path(self.main_output)
            for p in (main_out_path, self.thermo_path, self.traj_path, self.summary_path, self.final_path):
                backup = _backup_file(p)
                if backup is not None:
                    backup_msgs.append(f"  Backed up existing file: {p.name} -> {backup.name}\n")

            self.thermo_file = open(self.thermo_path, 'w')
            # Open trajectory file based on format
            if self.traj_format == 'dcd':
                self.traj_file = DCDWriter(
                    path=self.traj_path,
                    natoms=len(atoms),
                    timestep=timestep,
                    is_periodic=any(atoms.pbc),
                    first_step=step_offset,
                )
            else:  # xyz
                self.traj_file = open(self.traj_path, 'w')
        # else: files already opened in append mode by restart_simulation()

        # Write main output header
        total_steps_display = n_steps + step_offset
        self.log_main([
            "\n" + "="*80 + "\n",
            f"{'MD SIMULATION':^80}\n",
            "="*80 + "\n",
            f"Ensemble:        {ensemble.upper()}\n",
            f"Timestep:        {timestep:.3f} fs\n",
            f"Total steps:     {total_steps_display}\n",
            f"Simulation time: {total_steps_display * timestep:.2f} fs\n",
        ])

        if step_offset > 0:
            self.log_main([
                f"Resuming from:   step {step_offset} "
                f"({step_offset * timestep / 1000.0:.3f} ps)\n",
                f"Steps remaining: {n_steps}\n",
            ])

        if self._ensemble in ('nvt', 'npt'):
            self.log_main([f"Target temp:     {temperature:.2f} K\n"])
        if self._ensemble == 'npt' and pressure is not None:
            self.log_main([f"Target pressure: {pressure:.2f} bar\n"])

        self.log_main([
            f"\nSystem:\n",
            f"  Atoms:         {len(atoms)}\n",
            f"  Formula:       {atoms.get_chemical_formula()}\n",
            f"  Charge:        {atoms.info.get('charge', 0)}\n",
            f"  Multiplicity:  {atoms.info.get('mult', 1)}\n",
            "\n" + "="*80 + "\n",
        ])

        if step_offset == 0 and backup_msgs:
            self.log_main(["\nWARNING: Pre-existing output files were backed up:\n"] + backup_msgs + ["\n"])
            for msg in backup_msgs:
                print(f"WARNING: {msg.strip()}")

        # Write thermodynamics header (fresh run only; resume appends a separator instead)
        if step_offset == 0:
            self.thermo_file.write(f"# MD Simulation - {ensemble.upper()} Ensemble\n")
            self.thermo_file.write(f"# Timestep: {timestep} fs\n")
            if self._ensemble == 'npt':
                self.thermo_file.write(
                    f"# {'Step':>8} {'Time(fs)':>12} {'Temp(K)':>12} "
                    f"{'KE(Ha)':>15} {'PE(Ha)':>15} {'TE(Ha)':>15} "
                    f"{'Press(bar)':>12} {'Vol(A^3)':>12}\n"
                )
            else:
                self.thermo_file.write(
                    f"# {'Step':>8} {'Time(fs)':>12} {'Temp(K)':>12} "
                    f"{'KE(Ha)':>15} {'PE(Ha)':>15} {'TE(Ha)':>15}\n"
                )
            self.thermo_file.flush()

        # ------------------------------------------------------------------
        # Progress header  (verbose >= 1)
        # Printed once; the data row below it is refreshed in-place with \r.
        # ------------------------------------------------------------------
        if self.verbose >= 1:
            is_npt = self._ensemble == 'npt'
            total_ps = self._n_steps * self._timestep / 1000.0
            _time_col_w = max(len(f"0.000/{total_ps:.3f}"), len("Time(ps)")) + 1
            hdr = (
                f"\n"
                f"  {'Step':>9}  {'Time(ps)':>{_time_col_w}}  {'Progress':>8}  "
                f"{'T(K)':>8}  {'E_total(Ha)':>15}  "
                f"{'Speed(ns/day)':>13}  {'ETA':>10}"
                + (f"  {'P(bar)':>10}" if is_npt else "")
            )
            sep = (
                f"  {'-'*9}  {'-'*_time_col_w}  {'-'*8}  "
                f"{'-'*8}  {'-'*15}  "
                f"{'-'*13}  {'-'*10}"
                + (f"  {'-'*10}" if is_npt else "")
            )
            self._time_col_w = _time_col_w
            self._total_ps   = total_ps
            # Write header to file and print to terminal
            self.log_main([hdr + "\n", sep + "\n"])
            print(hdr)
            print(sep)
            # Reserve the data row — cursor stays on this line for \r updates
            print("", end="", flush=True)

    def log_step(self, step: int, time: float, temperature: float,
                 kinetic_energy: float, potential_energy: float,
                 total_energy: float, atoms: Atoms, velocities: np.ndarray,
                 pressure: float = None, volume: float = None,
                 rng_state: Optional[str] = None,
                 rst_every: Optional[int] = None):
        """
        Log data for current step.

        Args:
            step: Current step number
            time: Current simulation time (fs)
            temperature: Current temperature (K)
            kinetic_energy: Kinetic energy (Hartree)
            potential_energy: Potential energy (Hartree)
            total_energy: Total energy (Hartree)
            atoms: Current ASE Atoms object
            velocities: Current velocities (atomic units: Bohr/a.u. time)
            pressure: Instantaneous pressure in bar (NPT only)
            volume: Cell volume in Å³ (NPT only)
            rng_state: Hex-encoded RNG state to embed in trajectory frame (NVT/NPT only)
        """
        # PE and KE are both passed in Hartree (UMACalculator already converts)
        potential_energy_hartree = potential_energy
        kinetic_energy_hartree = kinetic_energy
        total_energy_hartree = kinetic_energy_hartree + potential_energy_hartree

        # Store for statistics
        self.energies.append(total_energy_hartree)
        self.temperatures.append(temperature)
        self.times.append(time)
        self.kinetic_energies.append(kinetic_energy_hartree)
        self.potential_energies.append(potential_energy_hartree)
        if pressure is not None:
            self.pressures.append(pressure)

        # Write thermodynamic data every step
        if self._ensemble == 'npt' and pressure is not None and volume is not None:
            self.thermo_file.write(
                f"{step:>10} {time:>12.3f} {temperature:>12.2f} "
                f"{kinetic_energy_hartree:>15.8f} {potential_energy_hartree:>15.8f} "
                f"{total_energy_hartree:>15.8f} {pressure:>12.3f} {volume:>12.4f}\n"
            )
        else:
            self.thermo_file.write(
                f"{step:>10} {time:>12.3f} {temperature:>12.2f} "
                f"{kinetic_energy_hartree:>15.8f} {potential_energy_hartree:>15.8f} "
                f"{total_energy_hartree:>15.8f}\n"
            )
        self.thermo_file.flush()

        # ------------------------------------------------------------------
        # Progress line  (verbose >= 1, printed every log_every steps)
        # ------------------------------------------------------------------
        # Terminal progress line  (verbose >= 1, every 100 steps)
        # Uses \r to overwrite the same line — no screen scrolling.
        # The .dat file still receives a record at every log_every interval.
        # ------------------------------------------------------------------
        _PRINT_EVERY = 100
        if self.verbose >= 1 and step % _PRINT_EVERY == 0:
            now          = _time.perf_counter()

            # Rolling speed over the last print window
            delta_steps  = step - self._last_step
            delta_wall   = now  - self._last_wall
            if delta_wall > 0:
                ns_per_day = (delta_steps * self._timestep / delta_wall / 1e6 * 86400)
            else:
                ns_per_day = float('nan')
            self._last_wall = now
            self._last_step = step

            # ETA
            steps_remaining = self._n_steps - step
            if ns_per_day > 0 and ns_per_day == ns_per_day:   # not nan
                remaining_s = steps_remaining * self._timestep / 1e6 / ns_per_day * 86400
            else:
                remaining_s = float('inf')

            progress_pct = 100.0 * step / self._n_steps if self._n_steps > 0 else 0.0
            speed_str    = f"{ns_per_day:.3f}" if ns_per_day == ns_per_day else "---"
            eta_str      = _fmt_eta(remaining_s)

            # Build progress row aligned to the header columns:
            #   Step(9)  Time(fs)(10)  Progress(8)  T(K)(8)  E_total(Ha)(15)
            #   Speed(ns/day)(13)  ETA(10)  [P(bar)(10)]
            progress_str = f"{progress_pct:.1f}%"
            current_ps   = time / 1000.0
            time_str     = f"{current_ps:.3f}/{self._total_ps:.3f}"
            line = (
                f"  {step:>9}  {time_str:>{self._time_col_w}}  {progress_str:>8}  "
                f"{temperature:>8.2f}  {total_energy_hartree:>15.6f}  "
                f"{speed_str:>13}  {eta_str:>10}"
            )
            if self._ensemble == 'npt' and pressure is not None:
                line += f"  {pressure:>10.2f}"

            # \r returns cursor to line start; pad with spaces to erase
            # any leftover characters from a longer previous line
            print(f"\r{line:<120}", end='', flush=True)

        # Also write to .dat file at log_every frequency (unchanged)
        if step % self.log_every == 0:
            self.log_main([
                f"  Step {step:>8}  {time:>10.2f} fs  "
                f"T {temperature:>7.2f} K  "
                f"E {total_energy_hartree:>14.6f} Ha\n"
            ])

        # Write trajectory at traj_every frequency.
        # frame_number stores the MD *step* number so that restart_simulation()
        # can recover step_offset directly without needing to know traj_every.
        if step % self.traj_every == 0:
            if self.traj_format == 'dcd':
                # DCD writer handles its own writing
                self.traj_file.write_frame(atoms, step=step)
            else:  # xyz
                write_xyz_frame(
                    self.traj_file,
                    atoms,
                    energy=total_energy_hartree,
                    frame_number=step,          # MD step number, NOT sequential frame index
                    velocity=velocities,
                    rng_state=rng_state,
                )
                self.traj_file.flush()

        # Write restart checkpoint at rst_every frequency
        if rst_every and step % rst_every == 0:
            rotate_rst_checkpoint(
                rst_path=self.rst_path,
                rst_prev_path=self.rst_prev_path,
                atoms=atoms,
                velocities=velocities,
                step=step,
                timestep=self._timestep,
                ensemble=self._ensemble,
                energy=total_energy_hartree,
                rng_state=rng_state,
            )

    def restart_simulation(
        self,
        ensemble: str,
        timestep: float,
        n_steps: int,
        temperature: float,
        atoms: Atoms,
        pressure: float = None,
        rst_file: str = None,
    ):
        """
        Restore state from a .rst checkpoint file, validate against input atoms,
        open output files, and return (atoms, velocities, step_offset).

        Two modes of operation:

        1. **Resume** (rst_file=None):
           Auto-detects ``{base}_md.rst`` / ``{base}_md_prev.rst``.
           Strict validation: ensemble and timestep must match the checkpoint.
           Returns step_offset from the checkpoint so the run continues.

        2. **Load state** (rst_file="/path/to/other.rst"):
           Reads the specified RST file (auto-appends ``.rst`` if missing).
           Relaxed validation: ensemble and timestep may differ (NVT → NVE).
           Always returns step_offset=0 (fresh run with loaded coordinates
           and velocities).

        Returns None if the checkpoint already completed the requested run
        (resume mode only).
        Raises RuntimeError on hard failures (mismatch, missing files, etc.).
        """
        from ase.cell import Cell

        # ------------------------------------------------------------------
        # Determine whether this is a resume or a cross-ensemble load
        # ------------------------------------------------------------------
        is_cross_load = rst_file is not None

        if is_cross_load:
            # Resolve path: auto-append .rst if missing
            rst = Path(rst_file)
            if rst.suffix == '':
                rst = rst.with_suffix('.rst')
            # If relative path, resolve relative to output file's directory
            if not rst.is_absolute():
                rst = Path(self.main_output).parent / rst
            candidates = [rst]
        else:
            candidates = [self.rst_path, self.rst_prev_path]

        state = None
        errors = []
        used_path = None
        for path in candidates:
            if not path.exists():
                errors.append(f"missing: {path}")
                continue
            try:
                state = read_rst(path)
                used_path = path
                break
            except ValueError as exc:
                errors.append(f"{path.name}: {exc}")

        if state is None:
            raise RuntimeError("MD restart failed: " + "; ".join(errors))

        # ------------------------------------------------------------------
        # Validation: atoms must always match
        # ------------------------------------------------------------------
        if state["natoms"] != len(atoms):
            raise RuntimeError(
                f"Atom count mismatch: rst has {state['natoms']}, input has {len(atoms)}"
            )
        ref_symbols = atoms.get_chemical_symbols()
        for idx, (rst_sym, input_sym) in enumerate(zip(state["symbols"], ref_symbols), 1):
            if rst_sym != input_sym:
                raise RuntimeError(
                    f"Element mismatch between rst and input at position {idx}"
                )

        if is_cross_load:
            # Cross-ensemble load: log the transition, skip ensemble/timestep checks
            rst_ens = state["ensemble"].upper()
            new_ens = ensemble.upper()
            if state["ensemble"] != ensemble:
                self.log_main([
                    f"\nLoading state from {used_path.name} "
                    f"({rst_ens} -> {new_ens} transition)\n",
                ])
            step_offset = 0
        else:
            # Resume: strict validation
            if state["ensemble"] != ensemble:
                raise RuntimeError(
                    f"Ensemble mismatch: rst has '{state['ensemble']}', "
                    f"input specifies '{ensemble}'"
                )
            if abs(state["timestep"] - timestep) > 1e-12:
                raise RuntimeError(
                    f"Timestep mismatch: rst has {state['timestep']}, "
                    f"input specifies {timestep}"
                )
            if state["step"] >= n_steps:
                self.log_main([
                    f"\nRestart checkpoint {used_path.name} already completed the "
                    f"requested run ({state['step']}/{n_steps} steps).\n"
                ])
                # Even though the run is complete, export final.xyz from
                # the checkpoint so the user always has the last-frame file.
                if not self.final_path.exists():
                    atoms.set_positions(state["positions"])
                    if state["cell"] is not None:
                        atoms.set_cell(Cell.fromcellpar(state["cell"]))
                    with open(self.final_path, 'w') as f:
                        write_xyz_frame(
                            f,
                            atoms=atoms,
                            energy=state["energy"],
                            frame_number=state["step"],
                            velocity=state["velocities"],
                        )
                    self.log_main([
                        f"  Exported final structure: {self.final_path.name}\n"
                    ])
                return None
            step_offset = state["step"]

        # Restore atoms state
        atoms.set_positions(state["positions"])
        if state["cell"] is not None:
            atoms.set_cell(Cell.fromcellpar(state["cell"]))
        if state["pbc"] is not None:
            atoms.set_pbc(state["pbc"])

        # Store RNG state for ensemble drivers (NVT/NPT) to restore
        self.resumed_rng_state = state.get("rng_state")

        # ------------------------------------------------------------------
        # Open output files
        #
        # Cross-ensemble load (rst_file=...): step_offset=0 → start_simulation()
        #   will open files fresh, so we do NOT open them here.
        # Resume (same ensemble):  step_offset>0 → start_simulation() skips
        #   file opening, so we MUST open in append mode here.
        # ------------------------------------------------------------------
        if not is_cross_load:
            # Resume: append to existing output files
            self.thermo_file = (open(self.thermo_path, "a") if self.thermo_path.exists()
                                else open(self.thermo_path, "w"))
            if self.traj_format == 'dcd':
                if self.traj_path.exists():
                    self.traj_file = DCDWriter.open_for_append(self.traj_path)
                else:
                    self.traj_file = DCDWriter(
                        path=self.traj_path,
                        natoms=len(atoms),
                        timestep=timestep,
                        is_periodic=any(atoms.pbc),
                        first_step=state["step"],
                    )
            else:
                self.traj_file = (open(self.traj_path, "a") if self.traj_path.exists()
                                  else open(self.traj_path, "w"))
            self.thermo_file.write(
                f"\n# --- RESTARTED from {used_path.name} step {state['step']} ---\n"
            )
            self.thermo_file.flush()

        return atoms, state["velocities"], step_offset

    def end_simulation(self, atoms: Atoms = None, final_velocities: np.ndarray = None,
                       rng_state: str = None):
        """
        Finalize simulation, compute publication-quality conservation metrics,
        write summary file, and write final restart checkpoint.

        Parameters
        ----------
        atoms : ase.Atoms, optional
            Final atomic configuration.
        final_velocities : np.ndarray, optional
            Final velocities in atomic units (Bohr/a.u. time).
            Written to the restart checkpoint so that a subsequent
            NVE/NVT/NPT run can restart from the exact end state.
            Analogous to GROMACS confout.gro (coordinates + velocities).
        rng_state : str, optional
            Hex-encoded RNG state (for NVT/NPT). Written to RST checkpoint
            for deterministic continuation. NVE does not need this.
        """
        # End the \r progress line with a newline so the summary starts cleanly
        if self.verbose >= 1:
            print(flush=True)

        energies     = np.array(self.energies)
        temperatures = np.array(self.temperatures)
        times        = np.array(self.times)          # fs
        ke_arr       = np.array(self.kinetic_energies)
        pe_arr       = np.array(self.potential_energies)

        # ------------------------------------------------------------------
        # 1. Basic energy statistics  (full trajectory, no skip)
        # ------------------------------------------------------------------
        energy_mean = np.mean(energies)
        energy_std  = np.std(energies)

        # ------------------------------------------------------------------
        # 2. Linear drift rate via least-squares fit
        #    Front 20% of trajectory skipped as equilibration.
        # ------------------------------------------------------------------
        EQ_FRAC   = 0.20

        eq_cut = max(1, int(len(times) * EQ_FRAC))
        prod_times    = times[eq_cut:]
        prod_energies = energies[eq_cut:]
        prod_time_ns  = (prod_times[-1] - prod_times[0]) * FS_TO_NS if len(prod_times) >= 2 else 0.0
        eq_time_ps    = (times[eq_cut - 1] - times[0]) * 1e-3

        if len(prod_times) >= 2 and prod_time_ns > 0:
            coef = np.polyfit(prod_times, prod_energies, 1)
            drift_rate = (coef[0]
                          * HARTREE_TO_KCAL_PER_MOL
                          / FS_TO_NS)
        else:
            drift_rate = float('nan')

        # ------------------------------------------------------------------
        # 3. Relative energy fluctuation
        # ------------------------------------------------------------------
        rel_fluctuation = energy_std / abs(energy_mean) if energy_mean != 0 else float('nan')

        # ------------------------------------------------------------------
        # 4. KE–PE correlation coefficient
        # ------------------------------------------------------------------
        if len(ke_arr) > 1 and np.std(ke_arr) > 0 and np.std(pe_arr) > 0:
            ke_pe_corr = np.corrcoef(ke_arr, pe_arr)[0, 1]
        else:
            ke_pe_corr = float('nan')

        # ------------------------------------------------------------------
        # 5. Temperature statistics
        # ------------------------------------------------------------------
        temp_mean = np.mean(temperatures)
        temp_std  = np.std(temperatures)

        n_dof = (3 * self._n_atoms if (self._is_pbc or self._n_atoms == 0)
                 else 3 * self._n_atoms - 3)
        if n_dof <= 0:
            n_dof = 1

        observed_ratio = temp_std / temp_mean if temp_mean > 0 else float('nan')

        # ------------------------------------------------------------------
        # 6. Assemble summary lines
        # ------------------------------------------------------------------
        is_nve = self._ensemble == 'nve'
        ens_label = self._ensemble.upper()

        if is_nve:
            energy_section = [
                f"\n{'── [NVE] Energy Conservation ──':^80}\n",
                f"  σ(TE)/|⟨TE⟩|:              {rel_fluctuation:>18.2e}\n",
                f"  Linear drift rate:         {drift_rate:>+18.6f}  kcal/mol/ns\n",
                f"  Production window:         {prod_time_ns*1000:>15.3f}  ps"
                f"  (skipped first {eq_time_ps:.2f} ps as equilibration)\n",
                f"  r(KE,PE):                  {ke_pe_corr:>18.4f}\n",
            ]
            temp_section = [
                f"\n{'── [NVE] Temperature ──':^80}\n",
                f"  ⟨T⟩:                       {temp_mean:>18.2f}  K\n",
                f"  σ(T):                      {temp_std:>18.2f}  K\n",
                f"  σ(T)/<T>:                  {observed_ratio:>18.4f}\n",
                f"  N_dof = {n_dof}  ({'PBC: 3N' if self._is_pbc else 'isolated: 3N-3'})\n",
            ]
        else:
            energy_section = [
                f"\n{'── [' + ens_label + '] Temperature Control ──':^80}\n",
                f"  ⟨T⟩:                       {temp_mean:>18.2f}  K\n",
                f"  σ(T):                      {temp_std:>18.2f}  K\n",
                f"  N_dof = {n_dof}  ({'PBC: 3N' if self._is_pbc else 'isolated: 3N-3'})\n",
                f"  r(KE,PE):                  {ke_pe_corr:>18.4f}\n",
            ]
            temp_section = [
                f"\n{'── [' + ens_label + '] Energy ──':^80}\n",
                f"  Mean TE:                   {energy_mean:>18.8f}  Ha\n",
                f"  σ(TE):                     {energy_std:>18.8f}  Ha\n",
                f"  σ(TE)/|⟨TE⟩|:             {rel_fluctuation:>18.2e}\n",
                f"  Linear drift rate:         {drift_rate:>+18.6f}  kcal/mol/ns\n",
                f"  Production window:         {prod_time_ns*1000:>15.3f}  ps"
                f"  (skipped first {eq_time_ps:.2f} ps as equilibration)\n",
            ]

        summary_lines = [
            "\n" + "="*80 + "\n",
            f"{'MD SIMULATION COMPLETED':^80}\n",
            "="*80 + "\n",

            f"\n{'── Energy Statistics ──':^80}\n",
            f"  Mean total energy:         {energy_mean:>18.8f}  Ha\n",
            f"  Std deviation:             {energy_std:>18.8f}  Ha\n",
        ] + energy_section + temp_section

        # ------------------------------------------------------------------
        # Wall-clock performance summary  (always appended)
        # ------------------------------------------------------------------
        total_wall   = _time.perf_counter() - self._wall_start   # seconds
        sim_time_ns  = (times[-1] - times[0]) * FS_TO_NS         # ns of MD
        if total_wall > 0:
            ns_per_day_avg = sim_time_ns / total_wall * 86400
            s_per_ns       = total_wall / sim_time_ns if sim_time_ns > 0 else float('nan')
        else:
            ns_per_day_avg = float('nan')
            s_per_ns       = float('nan')

        perf_lines = [
            f"\n{'── Performance ──':^80}\n",
            f"  Wall time:                  {_fmt_duration(total_wall):>18}\n",
            f"  Simulation time:            {sim_time_ns*1000:>15.3f}  ps\n",
        ]
        if not (ns_per_day_avg != ns_per_day_avg):   # not nan
            perf_lines += [
                f"  Average speed:              {ns_per_day_avg:>15.4f}  ns/day\n",
                f"  Time per ns:                {s_per_ns:>15.1f}  s/ns\n",
            ]
        summary_lines += perf_lines

        self.log_main(summary_lines, echo=True)

        # ------------------------------------------------------------------
        # Write summary file
        # ------------------------------------------------------------------
        with open(self.summary_path, 'w') as f:
            f.write("MD Simulation Summary\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Ensemble:                   {self._ensemble.upper()}\n")
            f.write(f"Total steps:                {len(self.energies)}\n")
            f.write(f"Total time:                 {self.times[-1]:.2f} fs\n")
            f.write(f"Number of atoms:            {self._n_atoms}\n")
            f.write(f"N_dof:                      {n_dof}  "
                    f"({'PBC: 3N' if self._is_pbc else 'isolated: 3N-3'})\n\n")

            f.write("Energy Statistics:\n")
            f.write(f"  Mean total energy:        {energy_mean:.8f} Ha\n")
            f.write(f"  Std deviation:            {energy_std:.8f} Ha\n\n")

            if is_nve:
                f.write("Energy Conservation Metrics [NVE]:\n")
                f.write(f"  σ(TE)/|⟨TE⟩|:            {rel_fluctuation:.2e}\n")
                f.write(f"  Linear drift rate:        {drift_rate:+.6f} kcal/mol/ns\n")
                f.write(f"  Production window:        {prod_time_ns*1000:.3f} ps"
                        f"  (skipped first {eq_time_ps:.2f} ps as equilibration)\n")
                f.write(f"  r(KE,PE):                 {ke_pe_corr:.4f}\n\n")

                f.write("Temperature Statistics:\n")
                f.write(f"  Mean temperature:         {temp_mean:.2f} K\n")
                f.write(f"  Std deviation:            {temp_std:.2f} K\n")
                f.write(f"  σ(T)/<T>:                 {observed_ratio:.4f}\n")
            else:
                f.write(f"Temperature Control Metrics [{self._ensemble.upper()}]:\n")
                f.write(f"  Mean temperature:         {temp_mean:.2f} K\n")
                f.write(f"  Std deviation:            {temp_std:.2f} K\n")
                f.write(f"  r(KE,PE):                 {ke_pe_corr:.4f}\n\n")

                f.write(f"Energy Metrics [{self._ensemble.upper()}]:\n")
                f.write(f"  σ(TE)/|⟨TE⟩|:            {rel_fluctuation:.2e}\n")
                f.write(f"  Linear drift rate:        {drift_rate:+.6f} kcal/mol/ns\n")
                f.write(f"  Production window:        {prod_time_ns*1000:.3f} ps"
                        f"  (skipped first {eq_time_ps:.2f} ps as equilibration)\n")

            if self.pressures:
                pressures_arr = np.array(self.pressures)
                f.write(f"\nPressure Statistics:\n")
                f.write(f"  Mean pressure:            {np.mean(pressures_arr):.3f} bar\n")
                f.write(f"  Std deviation:            {np.std(pressures_arr):.3f} bar\n")

            f.write(f"\nPerformance:\n")
            f.write(f"  Wall time:                {_fmt_duration(total_wall)}\n")
            if not (ns_per_day_avg != ns_per_day_avg):
                f.write(f"  Average speed:            {ns_per_day_avg:.4f} ns/day\n")
                f.write(f"  Time per ns:              {s_per_ns:.1f} s/ns\n")

        if self.pressures:
            pressures_log = np.array(self.pressures)
            self.log_main([
                f"\n{'── Pressure Statistics ──':^80}\n",
                f"  Mean pressure:              {np.mean(pressures_log):>18.3f}  bar\n",
                f"  Std deviation:              {np.std(pressures_log):>18.3f}  bar\n",
            ], echo=True)

        # ------------------------------------------------------------------
        # Write final restart checkpoint  (GROMACS state.cpt equivalent)
        # Contains final coordinates + velocities + simulation state
        # so that a subsequent NVE/NVT/NPT run can restart from the exact end state.
        # ------------------------------------------------------------------
        # Also write final structure file (GROMACS confout.gro equivalent)
        # Contains final coordinates + velocities in XYZ format for easy inspection
        # and use as input for the next simulation stage.
        # ------------------------------------------------------------------
        final_written = False
        final_xyz_written = False
        if atoms is not None and final_velocities is not None:
            final_energy = self.energies[-1] if self.energies else float("nan")
            final_step = int(round(self.times[-1] / self._timestep)) if self.times and self._timestep > 0 else 0

            # Write RST checkpoint (complete state for restart)
            # For NVT/NPT, include RNG state for deterministic continuation
            rotate_rst_checkpoint(
                rst_path=self.rst_path,
                rst_prev_path=self.rst_prev_path,
                atoms=atoms,
                velocities=final_velocities,
                step=final_step,
                timestep=self._timestep,
                ensemble=self._ensemble,
                energy=final_energy,
                rng_state=rng_state,
            )
            final_written = True

            # Write final structure XYZ (confout.gro equivalent)
            # This file contains:
            #   - Coordinates (can be used as input for next stage)
            #   - Velocities (embedded in XYZ, read by InputReader)
            #   - Cell parameters (if PBC)
            with open(self.final_path, 'w') as f:
                write_xyz_frame(
                    f,
                    atoms=atoms,
                    energy=final_energy,
                    frame_number=final_step,
                    velocity=final_velocities,
                )
            final_xyz_written = True

        self.log_main([
            f"\n{'── Output Files ──':^80}\n",
            f"  Thermodynamics:             {self.thermo_path.name}\n",
            f"  Trajectory:                 {self.traj_path.name}\n",
            f"  Summary:                    {self.summary_path.name}\n",
            *([f"  Final structure:           {self.final_path.name}\n"
               f"    (Use as input for next stage with velocities)\n"]
              if final_xyz_written else []),
            *([f"  Checkpoint:                 {self.rst_path.name}\n"
               f"  Previous checkpoint:        {self.rst_prev_path.name}\n"]
              if final_written else []),
            "="*80 + "\n",
        ], echo=True)

        # Close files
        if self.thermo_file:
            self.thermo_file.close()
        if self.traj_file:
            self.traj_file.close()

    def log_main(self, messages: list, echo: bool = False):
        """
        Write messages to main output file.
        If echo=True and verbose >= 1, also print to stdout (terminal).

        Args:
            messages: List of message strings
            echo:     Mirror output to stdout (used for progress lines)
        """
        with open(self.main_output, 'a') as f:
            for msg in messages:
                f.write(msg)
        if echo and self.verbose >= 1:
            for msg in messages:
                print(msg, end='', flush=True)
