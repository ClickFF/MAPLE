from ase import Atoms
from maple.function.utility import Molecules

from ..jobABC import JobABC

from maple.function.timer import timer

class IRC(JobABC):
    def __init__(self, params: dict, output:str, atoms:Atoms, method:str='gs'):
        super().__init__(output)
        self.atoms = atoms
        self.method = method
        self.output = output
        self.commandcontrol = params

    def run(self):
        with timer("IRC Calculation"):
            if isinstance(self.atoms, (list, Molecules)):
                from ..optimization.algorithm._common import member_outputs
                atoms_list = self.atoms.multiatoms if isinstance(self.atoms, Molecules) else self.atoms
                outputs = member_outputs(self.output, atoms_list, ('_forward', '_backward', '_full'))
                self.results = []
                for i, (atoms, output) in enumerate(zip(atoms_list, outputs), start=1):
                    result = self._run_single(atoms, output)
                    self.results.append(result)
                    converged = all(result[side]['converged'] for side in ('forward', 'backward'))
                    self.log_info([f'Member {i}: IRC {"converged" if converged else "NOT converged"}; '
                                   f'output: {output}\n'])
                return
            self.results = self._run_single(self.atoms, self.output)

    def _run_single(self, atoms, output):
        if self.method == 'gs':
            from .algorithm import GS
            irc = GS(atoms, output=output, paras=self.commandcontrol)
            return irc.run()
        elif self.method == 'hpc':
            from .algorithm import HPC
            irc = HPC(atoms, output=output, paras=self.commandcontrol)
            return irc.run()
        elif self.method == 'eulerpc':
            from .algorithm import EulerPC
            irc = EulerPC(atoms, output=output, paras=self.commandcontrol)
            return irc.run()
        elif self.method == 'lqa':
            from .algorithm import LQA
            irc = LQA(atoms, output=output, paras=self.commandcontrol)
            return irc.run()
        else:
            raise NotImplementedError(f'IRC method {self.method} not implemented yet.')
