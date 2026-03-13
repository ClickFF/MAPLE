"""
MD simulation logging and output management.

Handles:
    - Thermodynamic data output (.dat file)
    - XYZ trajectory output (.xyz file)
    - Progress logging to main output
    - Final summary statistics
"""

import numpy as np
from pathlib import Path
from typing import Optional, TextIO
from ase import Atoms

from .utils import write_xyz_frame


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
    ):
        """
        Initialize MD logger.

        Args:
            output_path: Main output file path (e.g., "task.out")
            log_every: Frequency to log thermodynamic data (steps)
            traj_every: Frequency to write trajectory frames (steps)
        """
        self.main_output = output_path
        self.log_every = log_every
        self.traj_every = traj_every

        # Generate file paths
        base = Path(output_path).stem
        parent = Path(output_path).parent

        self.thermo_path = parent / f"{base}_md_thermo.dat"
        self.traj_path = parent / f"{base}_md_traj.xyz"
        self.summary_path = parent / f"{base}_md_summary.txt"

        # File handles (opened in start_simulation)
        self.thermo_file: Optional[TextIO] = None
        self.traj_file: Optional[TextIO] = None

        # Statistics tracking
        self.energies = []
        self.temperatures = []
        self.times = []
        self.pressures = []   # NPT only

    def start_simulation(self, ensemble: str, timestep: float, n_steps: int,
                        temperature: float, atoms: Atoms,
                        pressure: float = None):
        """
        Initialize output files and write headers.

        Args:
            ensemble: Ensemble type (nve, nvt, npt)
            timestep: Timestep in fs
            n_steps: Total number of steps
            temperature: Target temperature in K
            atoms: ASE Atoms object
            pressure: Target pressure in bar (NPT only)
        """
        self._ensemble = ensemble.lower()

        # Open files
        self.thermo_file = open(self.thermo_path, 'w')
        self.traj_file = open(self.traj_path, 'w')

        # Write main output header
        self.log_main([
            "\n" + "="*80 + "\n",
            f"{'MD SIMULATION':^80}\n",
            "="*80 + "\n",
            f"Ensemble:        {ensemble.upper()}\n",
            f"Timestep:        {timestep:.3f} fs\n",
            f"Total steps:     {n_steps}\n",
            f"Simulation time: {n_steps * timestep:.2f} fs\n",
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

        # Write thermodynamics header
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

    def log_step(self, step: int, time: float, temperature: float,
                 kinetic_energy: float, potential_energy: float,
                 total_energy: float, atoms: Atoms, velocities: np.ndarray,
                 pressure: float = None, volume: float = None):
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
        """
        # PE and KE are both passed in Hartree (UMACalculator already converts)
        potential_energy_hartree = potential_energy
        kinetic_energy_hartree = kinetic_energy
        total_energy_hartree = kinetic_energy_hartree + potential_energy_hartree

        # Store for statistics
        self.energies.append(total_energy_hartree)
        self.temperatures.append(temperature)
        self.times.append(time)
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

        # Write to main output at log_every frequency
        if step % self.log_every == 0:
            if self._ensemble == 'npt' and pressure is not None:
                self.log_main([
                    f"Step {step:6d} | "
                    f"Time: {time:8.2f} fs | "
                    f"T: {temperature:7.2f} K | "
                    f"E: {total_energy_hartree:12.6f} Ha | "
                    f"P: {pressure:9.2f} bar\n"
                ])
            else:
                self.log_main([
                    f"Step {step:6d} | "
                    f"Time: {time:8.2f} fs | "
                    f"T: {temperature:7.2f} K | "
                    f"E: {total_energy_hartree:12.6f} Ha\n"
                ])

        # Write trajectory at traj_every frequency
        if step % self.traj_every == 0:
            write_xyz_frame(
                self.traj_file,
                atoms,
                energy=total_energy_hartree,
                frame_number=step // self.traj_every,
                velocity=velocities
            )
            self.traj_file.flush()

    def end_simulation(self):
        """
        Finalize simulation and write summary.
        """
        # Calculate statistics
        energies = np.array(self.energies)
        temperatures = np.array(self.temperatures)

        energy_mean = np.mean(energies)
        energy_std = np.std(energies)
        energy_drift = energies[-1] - energies[0]

        temp_mean = np.mean(temperatures)
        temp_std = np.std(temperatures)

        summary_lines = [
            "\n" + "="*80 + "\n",
            f"{'MD SIMULATION COMPLETED':^80}\n",
            "="*80 + "\n",
            f"\nEnergy Statistics:\n",
            f"  Mean total energy:     {energy_mean:15.8f} Hartree\n",
            f"  Std deviation:         {energy_std:15.8f} Hartree\n",
            f"  Energy drift:          {energy_drift:15.8f} Hartree\n",
            f"  Relative drift:        {abs(energy_drift/energy_mean)*100:12.6f} %\n",
            f"\nTemperature Statistics:\n",
            f"  Mean temperature:      {temp_mean:15.2f} K\n",
            f"  Std deviation:         {temp_std:15.2f} K\n",
        ]

        # Write to main output
        self.log_main(summary_lines)

        # Write summary file
        with open(self.summary_path, 'w') as f:
            f.write("MD Simulation Summary\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total steps:             {len(self.energies)}\n")
            f.write(f"Total time:              {self.times[-1]:.2f} fs\n\n")
            f.write(f"Energy Statistics:\n")
            f.write(f"  Mean total energy:     {energy_mean:.8f} Ha\n")
            f.write(f"  Std deviation:         {energy_std:.8f} Ha\n")
            f.write(f"  Energy drift:          {energy_drift:.8f} Ha\n")
            f.write(f"  Relative drift:        {abs(energy_drift/energy_mean)*100:.6f} %\n\n")
            f.write(f"Temperature Statistics:\n")
            f.write(f"  Mean temperature:      {temp_mean:.2f} K\n")
            f.write(f"  Std deviation:         {temp_std:.2f} K\n")

            if self.pressures:
                pressures = np.array(self.pressures)
                p_mean = np.mean(pressures)
                p_std = np.std(pressures)
                self.log_main([
                    f"\nPressure Statistics:\n",
                    f"  Mean pressure:         {p_mean:15.3f} bar\n",
                    f"  Std deviation:         {p_std:15.3f} bar\n",
                ])
                f.write(f"\nPressure Statistics:\n")
                f.write(f"  Mean pressure:         {p_mean:.3f} bar\n")
                f.write(f"  Std deviation:         {p_std:.3f} bar\n")

        self.log_main([
            f"\nOutput Files:\n",
            f"  Thermodynamics:        {self.thermo_path.name}\n",
            f"  Trajectory:            {self.traj_path.name}\n",
            f"  Summary:               {self.summary_path.name}\n",
            "="*80 + "\n",
        ])

        # Close files
        if self.thermo_file:
            self.thermo_file.close()
        if self.traj_file:
            self.traj_file.close()

    def log_main(self, messages: list):
        """
        Write messages to main output file.

        Args:
            messages: List of message strings
        """
        with open(self.main_output, 'a') as f:
            for msg in messages:
                f.write(msg)
