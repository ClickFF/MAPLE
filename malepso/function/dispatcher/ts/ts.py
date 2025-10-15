import numpy as np
from typing import Tuple

from torch import Tensor
from ase import Atoms

from ..jobABC import JobABC

class TransitionState(JobABC):

    def __init__(self, output: str, atoms: Atoms, params: dict, method:str='newton',criteria:str='default'):
        super().__init__(output)
        self.atoms = atoms
        self.params = params
        self.method=method

    def run(self):
        #from .algorithm import Newton
        #Newton(self.atoms, output=self.output)
        if self.method == 'newton':
            from .algorithm import Newton
            Newton(self.atoms, output=self.output)
        elif self.method == 'prfo':
            from .algorithm import RFO
            RFO(self.atoms, output=self.output)
        elif self.method == 'neb':
            if not isinstance(self.atoms, list):
                raise ValueError('For NEB method, you should provide at least two structures (initial and final states).')
            from .algorithm import NEB
            neb = NEB(
                output=self.output,
                atoms_R=self.atoms[0],
                atoms_P=self.atoms[1],
                paras=self.params
            )
            neb.run()
        elif self.method == 'string':
            if not isinstance(self.atoms, list):
                raise ValueError('For String method, you should provide at least two structures (initial and final states).')
            from .algorithm import String
            string = String(
                output=self.output,
                atoms_R=self.atoms[0],
                atoms_P=self.atoms[1],
                paras=self.params
            )
            string.run()
        else:
            raise ValueError(f'Method {self.method} not recognized. Available methods are: newton, prfo, neb.')


    def get_hessian(self) -> np.ndarray:
        calc = self.atoms.get_calculator()
        hessian_matrix:Tensor = calc.get_hessian(self.atoms)
        return hessian_matrix.numpy()