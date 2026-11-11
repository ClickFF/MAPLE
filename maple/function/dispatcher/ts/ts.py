import numpy as np
from typing import Union, List

from torch import Tensor
from ase import Atoms

from ..jobABC import JobABC

from maple.function.utility import Molecules
from maple.function.timer import timer

class TransitionState(JobABC):

    def __init__(self, output: str, atoms: Union[Atoms, Molecules, List[Atoms]], params: dict, method:str=None):
        super().__init__(output)
        self.atoms = atoms
        self.params = params
        self.method = method

    def run(self):

        if isinstance(self.atoms, (list, Molecules)) and self.method in ('prfo', 'dimer'):
            return self._run_single_ended_batch()
        pairs = self._endpoint_pairs()
        if pairs is not None:
            if len(pairs) > 1:
                return self._run_paired_paths(pairs)
            self.atoms = list(pairs[0])

        with timer("Transition State Optimization"):
            if self.method is None:
                raise ValueError('Method is not provided.')

            elif self.method == 'prfo':
                from .algorithm import PRFO
                prfo = PRFO(
                    atoms=self.atoms,
                    output=self.output,
                    paras=self.params,
                )
                result = prfo.run()
                self.results = prfo.results
                return result
                
            elif self.method == 'neb':
                # NEB now accepts Molecules object
                if isinstance(self.atoms, Molecules):
                    from .algorithm import NEB
                    neb = NEB(
                        output=self.output,
                        atoms_or_molecules=self.atoms,
                        paras=self.params
                    )
                    result = neb.run()
                    self.results = neb.results
                    return result
                elif isinstance(self.atoms, list):
                    # Legacy support for list input
                    if len(self.atoms) < 2:
                        raise ValueError('For NEB method, you should provide at least two structures (initial and final states).')
                    from .algorithm import NEB
                    # Convert list to Molecules for NEB
                    molecules = Molecules(self.atoms)
                    neb = NEB(
                        output=self.output,
                        atoms_or_molecules=molecules,
                        paras=self.params
                    )
                    result = neb.run()
                    self.results = neb.results
                    return result
                else:
                    raise ValueError('For NEB method, you should provide a Molecules object or a list of at least two structures.')
                    
            elif self.method == 'string':
                # TODO: Update String/GSM to accept Molecules object instead of list
                if isinstance(self.atoms, Molecules):
                    self.atoms = self.atoms.multiatoms
                if not isinstance(self.atoms, list):
                    raise ValueError('For String method, you should provide at least two structures (initial and final states).')
                if len(self.atoms) < 2:
                    raise ValueError('For String method, you should provide at least two structures (initial and final states).')
                from .algorithm import GSM
                string = GSM(
                    output=self.output,
                    atoms_R=self.atoms[0],
                    atoms_P=self.atoms[-1],
                    paras=self.params,
                    images=self.atoms if len(self.atoms) > 2 else None,
                )
                result = string.run()
                self.results = string.results
                return result
                
            elif self.method == 'dimer':
                from .algorithm import Dimer
                # Dimer takes a single Atoms as TS guess
                if isinstance(self.atoms, list):
                    atoms_input = self.atoms[0]
                else:
                    atoms_input = self.atoms
                dimer = Dimer(
                    output=self.output,
                    atoms_init=atoms_input,
                    paras=self.params
                )
                result = dimer.run()
                self.results = dimer.results
                return result
                
            elif self.method == 'autoneb':
                # AutoNEB: automated multi-step reaction pathway exploration
                if isinstance(self.atoms, Molecules):
                    from .algorithm import AutoNEB
                    autoneb = AutoNEB(
                        output=self.output,
                        atoms_or_molecules=self.atoms,
                        paras=self.params
                    )
                    result = autoneb.run()
                    self.results = autoneb.results
                    return result
                elif isinstance(self.atoms, list):
                    if len(self.atoms) < 2:
                        raise ValueError('For AutoNEB method, you should provide at least two structures.')
                    from .algorithm import AutoNEB
                    autoneb = AutoNEB(
                        output=self.output,
                        atoms_or_molecules=self.atoms,
                        paras=self.params
                    )
                    result = autoneb.run()
                    self.results = autoneb.results
                    return result
                else:
                    raise ValueError('For AutoNEB method, you should provide a Molecules object or a list of structures.')

            elif self.method == 'dmf':
                try:
                    import cyipopt  # noqa: F401
                except ImportError as exc:
                    raise ImportError(
                        "The Direct MaxFlux needs cyipopt; you should run: "
                        "conda install -c conda-forge cyipopt"
                    ) from exc

                if isinstance(self.atoms, Molecules):
                    molecules = self.atoms
                elif isinstance(self.atoms, list):
                    if len(self.atoms) < 2:
                        raise ValueError('For DMF method, you should provide at least two structures.')
                    molecules = Molecules(self.atoms)
                else:
                    raise ValueError('For DMF method, you should provide a Molecules object or a list of at least two structures.')

                from .algorithm.dmf import DMF
                dmf = DMF(
                    output=self.output,
                    atoms_or_molecules=molecules,
                    paras=self.params,
                )
                result = dmf.run()
                self.results = dmf.results
                return result

            else:
                raise ValueError(f'Method {self.method} not recognized. Available methods are: prfo, neb, string, dimer, autoneb, dmf.')

    def _endpoint_pairs(self):
        if self.method not in ('neb', 'string', 'autoneb', 'dmf'):
            return None
        if not isinstance(self.atoms, Molecules) or len(self.atoms.input_groups) != 2:
            return None
        reactants, products = self.atoms.input_groups
        if len(reactants) != len(products):
            raise ValueError('The two endpoint groups must contain the same number of structures')
        pairs = list(zip(reactants, products))
        for i, (atoms_R, atoms_P) in enumerate(pairs, start=1):
            if not np.array_equal(atoms_R.numbers, atoms_P.numbers):
                raise ValueError(f'Endpoint pair {i}: atom numbers and element order must match')
            if any(atoms_R.info.get(name, default) != atoms_P.info.get(name, default)
                   for name, default in (('charge', 0), ('mult', 1))):
                raise ValueError(f'Endpoint pair {i}: charge and multiplicity must match')
            if not np.array_equal(atoms_R.pbc, atoms_P.pbc):
                raise ValueError(f'Endpoint pair {i}: PBC must match')
            if atoms_R.pbc.any() and not np.allclose(atoms_R.cell, atoms_P.cell):
                raise ValueError(f'Endpoint pair {i}: fixed-cell path endpoints must share a cell')
        return pairs

    def _run_single_ended_batch(self):
        from ..optimization.algorithm._common import member_outputs, batch_results
        atoms_list = self.atoms.multiatoms if isinstance(self.atoms, Molecules) else self.atoms
        if self.method == 'prfo':
            from .algorithm.PRFO import PRFOParams
            from .algorithm.BPRFO import BatchPRFO
            params = self._init_params(PRFOParams, self.params, ('prfo', 'ts'))
            calc = atoms_list[0].calc
            job = BatchPRFO(
                output=self.output, trust_init=params.trust_radius, trust_min=params.trust_min,
                trust_max=params.trust_max, eta_shrink=params.eta_shrink,
                eta_expand=params.eta_expand, max_outer_iter=params.max_iter,
                recalc=params.recalc, hessian_update=params.hessian_update,
                evals_eps=params.evals_eps, mu_margin=params.mu_margin,
                max_bisect_it=params.max_bisect_it,
                device=getattr(calc, 'device', 'cpu'), batch_size=self.params.get('batch_size'),
            )
            result = job.run(Molecules(atoms_list))
            self.results = job.results
            return result
        outputs = member_outputs(self.output, atoms_list, ('_dimer_traj', '_dimer_ts'))
        self.results = batch_results(atoms_list)
        for i, (atoms, output) in enumerate(zip(atoms_list, outputs)):
            job = TransitionState(output=output, atoms=atoms, params=self.params,
                                  method=self.method)
            job.run()
            for name in ('energy', 'forces', 'iterations', 'converged'):
                self.results[name][i] = job.results[name]
            self.log_info([f'Member {i + 1}: Dimer '
                           f'{"converged" if job.results["converged"] else "NOT converged"} '
                           f'after {job.results["iterations"]} iterations.\n'])
        return atoms_list

    def _run_paired_paths(self, pairs):
        from ..optimization.algorithm._common import member_outputs
        suffixes = ('_image_traj', '_mep', '_hei', '_cineb_traj', '_cineb_mep',
                    '_cineb_hei', '_nebts_mep', '_nebts_ts', '_reactant_min',
                    '_product_min', '_gsm_grow_final', '_gsm_mep', '_gsm_hei',
                    '_dmf_mep', '_dmf_tmax', '_dmf_tmax_traj', '_autoneb_global_mep',
                    '_stringts_mep', '_stringts_ts', '_prfo_traj', '_prfo_ts',
                    '_autoneb_intermediates', '_autoneb_ts_list', '_autoneb_tree.json',
                    '_autoneb_path_*_mep',
                    '_dmf_ipopt.out', '_dmf_fbenm_ipopt.out')
        outputs = member_outputs(self.output, [pair[0] for pair in pairs], suffixes)
        self.results = []
        result = []
        for i, (pair, output) in enumerate(zip(pairs, outputs), start=1):
            job = TransitionState(output=output, atoms=list(pair), params=self.params,
                                  method=self.method)
            result.append(job.run())
            self.results.append(job.results)
            self.log_info([f'Member {i}: {self.method.upper()} completed; output: {output}\n'])
        return result
