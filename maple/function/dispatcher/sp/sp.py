import numpy as np
from typing import Union, List, Optional
from dataclasses import dataclass

from torch import Tensor
from ase import Atoms

from ..jobABC import JobABC
from maple.function.timer import timer
from maple.function.utility.molecules import Molecules

@dataclass
class SPParams:
    """Parameters for Single Point calculation."""
    verbose: int = 1  # 0=energy only, 1=detailed (default), 2=no coordinates

class SinglePoint(JobABC):

    eV2Hartree = 1 / 27.211386245988

    def __init__(self, output: str, atoms: Union[Atoms, List[Atoms]],
                 paras: Optional[dict] = None):
        super().__init__(output)
        self.atoms = atoms
        self.is_trajectory = isinstance(atoms, list)

        # Initialize params
        self.params = self._init_params(SPParams, paras, ("sp", "SP"))
        self.verbose = self.params.verbose

    def run(self):
        if self.is_trajectory:
            self._run_trajectory()
        else:
            self._run_single()

    def _run_single(self):
        """Original single-point calculation logic."""
        with timer("Single Point Energy Calculation"):
            energy_ev = self.atoms.get_potential_energy()
            energy_hartree = energy_ev * self.eV2Hartree
            self.log_info([f"\nEnergy: {energy_hartree:.10f} Hartree\n"])

    def _run_trajectory(self):
        """Process multiple structures sequentially."""
        with timer("Single Point Energy Calculation (Trajectory)"):
            n_frames = len(self.atoms)
            self.log_info([f"\nProcessing {n_frames} structures from trajectory...\n"])
            self.log_info(["=" * 80 + "\n"])

            energies_hartree = []

            for idx, atoms_frame in enumerate(self.atoms, start=1):
                # Calculate energy
                energy_ev = atoms_frame.get_potential_energy()
                energy_hartree = energy_ev * self.eV2Hartree
                energies_hartree.append(energy_hartree)

                # Output based on verbose level
                if self.verbose == 0:
                    # Minimal: only energy
                    self.log_info([f"Frame {idx:4d}: {energy_hartree:.10f} Hartree\n"])

                elif self.verbose == 1:
                    # Detailed: frame header + charge/mult + energy + coordinates
                    self.log_info([f"\n{('Frame ' + str(idx)):=^80}\n"])

                    charge = atoms_frame.info.get('charge', 0)
                    mult = atoms_frame.info.get('mult', 1)
                    self.log_info([f"Charge: {charge}, Multiplicity: {mult}\n"])

                    self.log_info([f"Energy: {energy_hartree:.10f} Hartree\n\n"])

                    # Coordinates
                    self.log_info(["Coordinates (Angstrom):\n"])
                    symbols = atoms_frame.get_chemical_symbols()
                    positions = atoms_frame.get_positions()
                    for i, (sym, pos) in enumerate(zip(symbols, positions), start=1):
                        self.log_info([
                            f"  {i:<4} {sym:<2} {pos[0]:>15.8f} {pos[1]:>15.8f} {pos[2]:>15.8f}\n"
                        ])
                    self.log_info(["=" * 80 + "\n"])

                elif self.verbose == 2:
                    # Medium: frame header + charge/mult + energy (no coordinates)
                    self.log_info([f"\n{('Frame ' + str(idx)):=^80}\n"])

                    charge = atoms_frame.info.get('charge', 0)
                    mult = atoms_frame.info.get('mult', 1)
                    self.log_info([f"Charge: {charge}, Multiplicity: {mult}\n"])
                    self.log_info([f"Energy: {energy_hartree:.10f} Hartree\n"])
                    self.log_info(["=" * 80 + "\n"])

            # Summary (always shown)
            self.log_info([f"\n{' SUMMARY ':=^80}\n"])
            self.log_info([f"Total frames processed: {n_frames}\n"])
            self.log_info([f"Energy range: {min(energies_hartree):.10f} to {max(energies_hartree):.10f} Hartree\n"])
            energy_span = max(energies_hartree) - min(energies_hartree)
            self.log_info([f"Energy span: {energy_span:.10f} Hartree ({energy_span * 627.509:.4f} kcal/mol)\n"])
            self.log_info(["=" * 80 + "\n"])
