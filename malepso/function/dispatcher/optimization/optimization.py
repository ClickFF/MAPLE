
from ase import Atoms

from ..jobABC import JobABC

class Optmization(JobABC):
    def __init__(self, output:str, atoms:Atoms, method:str='LBFGS',criteria:int=2):
        super().__init__(output)
        self.atoms = atoms
        self.method = method
        self.output = output

    def run(self):
        if self.method == 'lbfgs':
            from .algorithm import LBFGS
            LBFGS(self.atoms, output=self.output)
        elif self.method == 'rfo':
            from .algorithm import RFO
            RFO(self.atoms, output=self.output)
        elif self.method == 'sd':
            from .algorithm import SD
            SD(self.atoms, output=self.output)
