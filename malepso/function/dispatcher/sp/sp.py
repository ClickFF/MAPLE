import numpy as np
from typing import Tuple

from torch import Tensor
from ase import Atoms

from ..jobABC import JobABC

class SinglePoint(JobABC):

    def __init__(self, output: str, atoms: Atoms):
        super().__init__(output)
        self.atoms = atoms

    def run(self):
        energy = self.atoms.get_potential_energy()
        self.log_info([f"\nEnergy: {energy}"])