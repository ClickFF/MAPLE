
from ase import Atoms

from ..jobABC import JobABC
from maple.function.utility import Molecules

from maple.function.timer import timer


class Optimization(JobABC):
    def __init__(self, params: dict, output: str, atoms: Atoms):
        super().__init__(output)
        self.atoms = atoms
        self.commandcontrol = params

    def run(self):
        with timer("Optimization"):
            method = str(self.commandcontrol.get('method') or 'lbfgs').lower()
            if isinstance(self.atoms, (list, Molecules)):
                return self._run_batch(method)
            if method == 'lbfgs':
                from .algorithm import LBFGS
                job = LBFGS(self.atoms, output=self.output, paras=self.commandcontrol)
            elif method == 'rfo':
                from .algorithm import RFO
                job = RFO(self.atoms, output=self.output, paras=self.commandcontrol)
            elif method in ('sd', 'sdcg', 'cg'):
                from .algorithm import SDCG
                job = SDCG(self.atoms, output=self.output, paras=self.commandcontrol)
            else:
                raise NotImplementedError(f"Unknown opt method: {method!r}. "
                                          "Supported: lbfgs, rfo, sd, sdcg, cg.")
            result = job.run()
            self.results = job.results
            return result

    def _run_batch(self, method):
        from .algorithm.LBFGS import LBFGSParams
        from .algorithm.blbfgs import BatchLBFGS
        mols = self.atoms if isinstance(self.atoms, Molecules) else Molecules(self.atoms)
        if method == 'lbfgs':
            params = self._init_params(LBFGSParams, self.commandcontrol, ('lbfgs', 'opt'))
            calc = mols.multiatoms[0].calc
            job = BatchLBFGS(
                output=self.output, memory=params.memory, curvature=params.curvature,
                maxstep=params.max_step, maxiter=params.max_iter,
                device=getattr(calc, 'device', 'cpu'), verbose=params.verbose,
                batch_size=self.commandcontrol.get('batch_size'),
            )
            result = job.run(mols)
            self.results = job.results
            return result
        from .algorithm._common import member_outputs, batch_results
        outputs = member_outputs(self.output, mols.multiatoms, ('_opt', '_opt_traj'))
        result = []
        self.results = batch_results(mols.multiatoms)
        for i, (atoms, output) in enumerate(zip(mols.multiatoms, outputs)):
            job = Optimization(params=self.commandcontrol, output=output, atoms=atoms)
            result.append(job.run())
            for name in ('energy', 'forces', 'iterations', 'converged'):
                self.results[name][i] = job.results[name]
            self.log_info([f'Member {i + 1}: {method.upper()} '
                           f'{"converged" if job.results["converged"] else "NOT converged"} '
                           f'after {job.results["iterations"]} iterations.\n'])
        return result


# Backward compatibility for the historical misspelling.
Optmization = Optimization
