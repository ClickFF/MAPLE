
from ase import Atoms

from ..jobABC import JobABC

class Optmization(JobABC):
    def __init__(self, params: dict, output:str, atoms:Atoms, method:str='LBFGS'):
        super().__init__(output)
        self.atoms = atoms
        self.method = method
        self.output = output
        self.commandcontrol = params

    def run(self):
        if self.method == 'lbfgs':
            from .algorithm import LBFGS
            opt = LBFGS(self.atoms, output=self.output, paras=self.commandcontrol)
            opt.run()
        elif self.method == 'rfo':
            from .algorithm import RFO
            opt = RFO(self.atoms, output=self.output, paras=self.commandcontrol)
            opt.run()
        elif self.method == 'sd':
            from .algorithm import SD
            SD(self.atoms, output=self.output)
