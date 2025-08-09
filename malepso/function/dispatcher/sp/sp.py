import numpy as np
from typing import Tuple

from torch import Tensor
from ase import Atoms

from ..jobABC import JobABC

class SinglePoint(JobABC):

    eV2Hartree = 1 / 27.211386245988

    def __init__(self, output: str, atoms: Atoms):
        super().__init__(output)
        self.atoms = atoms

    def run(self):
        energy = self.atoms.get_potential_energy()
        energy *= self.eV2Hartree
        self.log_info([f"\nEnergy: {energy}"])