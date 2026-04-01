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

        with timer("Transition State Optimization"):
            if self.method is None:
                raise ValueError('Method is not provided.')

            elif self.method == 'newton':
                from .algorithm import Newton
                Newton(self.atoms, output=self.output)
                
            elif self.method == 'prfo':
                from .algorithm import PRFO
                prfo = PRFO(
                    atoms=self.atoms,
                    output=self.output,
                    paras=self.params,
                )
                prfo.run()
                
            elif self.method == 'neb':
                # NEB now accepts Molecules object
                if isinstance(self.atoms, Molecules):
                    from .algorithm import NEB
                    neb = NEB(
                        output=self.output,
                        atoms_or_molecules=self.atoms,
                        paras=self.params
                    )
                    neb.run()
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
                    neb.run()
                else:
                    raise ValueError('For NEB method, you should provide a Molecules object or a list of at least two structures.')
                    
            elif self.method == 'string':
                # TODO: Update String/GSM to accept Molecules object instead of list
                if not isinstance(self.atoms, list):
                    raise ValueError('For String method, you should provide at least two structures (initial and final states).')
                if len(self.atoms) < 2:
                    raise ValueError('For String method, you should provide at least two structures (initial and final states).')
                from .algorithm import GSM
                string = GSM(
                    output=self.output,
                    atoms_R=self.atoms[0],
                    atoms_P=self.atoms[1],
                    paras=self.params
                )
                string.run()
                
            elif self.method == 'dimer':
                # TODO: Update Dimer to accept Molecules object instead of list
                if not isinstance(self.atoms, list):
                    raise ValueError('For Dimer method, you should provide a list of structures.')
                from .algorithm import Dimer
                dimer = Dimer(
                    output=self.output,
                    atoms_init=self.atoms,
                    paras=self.params
                )
                dimer.run()
                
            elif self.method == 'afir':
                from .algorithm import DSAFIR
                afir = DSAFIR(
                    output=self.output,
                    atoms_R=self.atoms[0],
                    atoms_P=self.atoms[-1],
                    paras=self.params
                )
                afir.run()
            elif self.method == 'descafir':
                from .algorithm import DESCAFIR
                descafir = DESCAFIR(
                    output=self.output,
                    atoms_R=self.atoms[0],
                    atoms_P=self.atoms[-1],
                    paras=self.params
                )
                descafir.run()

            elif self.method == 'autoneb':
                # AutoNEB: automated multi-step reaction pathway exploration
                if isinstance(self.atoms, Molecules):
                    from .algorithm import AutoNEB
                    autoneb = AutoNEB(
                        output=self.output,
                        atoms_or_molecules=self.atoms,
                        paras=self.params
                    )
                    autoneb.run()
                elif isinstance(self.atoms, list):
                    if len(self.atoms) < 2:
                        raise ValueError('For AutoNEB method, you should provide at least two structures.')
                    from .algorithm import AutoNEB
                    molecules = Molecules(self.atoms)
                    autoneb = AutoNEB(
                        output=self.output,
                        atoms_or_molecules=molecules,
                        paras=self.params
                    )
                    autoneb.run()
                else:
                    raise ValueError('For AutoNEB method, you should provide a Molecules object or a list of structures.')

            else:
                raise ValueError(f'Method {self.method} not recognized. Available methods are: newton, prfo, neb, string, dimer, autoneb.')
