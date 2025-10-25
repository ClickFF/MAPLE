
from ase import Atoms

from ..jobABC import JobABC

class IRC(JobABC):
    def __init__(self, params: dict, output:str, atoms:Atoms, method:str='gs'):
        super().__init__(output)
        self.atoms = atoms
        self.method = method
        self.output = output
        self.commandcontrol = params

    def run(self):
        if self.method == 'gs':
            from .algorithm import GS
            irc = GS(self.atoms, output=self.output, paras=self.commandcontrol)
            irc.run()
        else:
            raise NotImplementedError(f'IRC method {self.method} not implemented yet.')
