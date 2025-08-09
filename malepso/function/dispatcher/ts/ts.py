import numpy as np
from typing import Tuple

from torch import Tensor
from ase import Atoms

from ..jobABC import JobABC

class TransitionState(JobABC):

    def __init__(self, output: str, atoms: Atoms, method:str='newton',criteria:int=1):
        super().__init__(output)
        self.atoms = atoms

        if criteria == 1:
            self.atoms.f_max_th=0.00045*27.211386024367243
            self.atoms.f_rms_th=0.0003*27.211386024367243
            self.atoms.dp_max_th=0.0018   
            self.atoms.dp_rms_th=0.0012

    def run(self):
        #from .algorithm import Newton
        #Newton(self.atoms, output=self.output)
        from .algorithm import RFO
        RFO(self.atoms, output=self.output)

    def get_hessian(self) -> np.ndarray:
        calc = self.atoms.get_calculator()
        hessian_matrix:Tensor = calc.get_hessian(self.atoms)
        return hessian_matrix.numpy()