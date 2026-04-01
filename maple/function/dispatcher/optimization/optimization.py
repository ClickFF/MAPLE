
from ase import Atoms

from ..jobABC import JobABC

from maple.function.timer import timer

class Optmization(JobABC):
    def __init__(self, params: dict, output:str, atoms:Atoms, method:str='LBFGS'):
        super().__init__(output)
        self.atoms = atoms
        self.method = method
        self.output = output
        self.commandcontrol = params
        
    def run(self):
        with timer("Optimization"):
            if self.commandcontrol.get('method', 'lbfgs').lower() == 'lbfgs':
                from .algorithm import LBFGS
                opt = LBFGS(self.atoms, output=self.output, paras=self.commandcontrol)
                opt.run()
            elif self.commandcontrol.get('method').lower() == 'rfo':
                from .algorithm import RFO
                opt = RFO(self.atoms, output=self.output, paras=self.commandcontrol)
                opt.run()
            elif self.commandcontrol.get('method').lower() == 'sd':
                from .algorithm import SD
                SD(self.atoms, output=self.output)
