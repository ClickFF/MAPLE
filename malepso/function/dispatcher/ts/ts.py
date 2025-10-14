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

        if isinstance(atoms, list):
            for atom in self.atoms:
                atom.f_max_th=0.00045*27.211386024367243
                atom.f_rms_th=0.0003*27.211386024367243
                atom.dp_max_th=0.0018   
                atom.dp_rms_th=0.0012
        else:
            self.atoms.f_max_th=0.00045*27.211386024367243
            self.atoms.f_rms_th=0.0003*27.211386024367243
            self.atoms.dp_max_th=0.0018   
            self.atoms.dp_rms_th=0.0012

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
            
        else:
            raise ValueError(f'Method {self.method} not recognized. Available methods are: newton, prfo, neb.')


    def get_hessian(self) -> np.ndarray:
        calc = self.atoms.get_calculator()
        hessian_matrix:Tensor = calc.get_hessian(self.atoms)
        return hessian_matrix.numpy()