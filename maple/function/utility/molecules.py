from typing import List

from ase import Atoms

class Molecules:
    def __init__(self, atoms_list: List[Atoms], input_groups=None):
        self.multiatoms = atoms_list
        self.input_groups = input_groups if input_groups is not None else [atoms_list]
        self.calc = None

    def get_efh(self):
        return self.calc.get_efh_gpu()

    def get_energies_forces(self):
        return self.calc.get_ef_gpu()
