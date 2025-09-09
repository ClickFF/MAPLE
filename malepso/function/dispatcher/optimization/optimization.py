
from ase import Atoms

from ..jobABC import JobABC

class Optmization(JobABC):
    def __init__(self, output:str, atoms:Atoms, method:str='LBFGS',criteria:int=1):
        super().__init__(output)
        self.atoms = atoms
        self.method = method
        self.output = output

        if criteria == 1:
            self.atoms.f_max_th=0.00045*27.211386024367243
            self.atoms.f_rms_th=0.0003*27.211386024367243
            self.atoms.dp_max_th=0.0018   
            self.atoms.dp_rms_th=0.0012

    def run(self):
        if self.method == 'LBFGS':
            from .algorithm import LBFGS
            LBFGS(self.atoms, output=self.output)
        elif self.method == 'RFO':
            from .algorithm import RFO
            RFO(self.atoms, output=self.output)
        elif self.method == 'DIIS':
            from .algorithm import DIIS
            DIIS(self.atoms, output=self.output)
        elif self.method == 'SD':
            from .algorithm import SD
            SD(self.atoms, output=self.output)
