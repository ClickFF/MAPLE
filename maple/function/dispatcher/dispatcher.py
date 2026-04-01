from typing import List, Union

from ase import Atoms
from ..utility import Molecules

from maple.function.timer import timer

class Dispatcher():
    def __init__(self):
        pass

    def __call__(self, commandcontrol: dict, jobtype: int, atoms: Union[Atoms, Molecules, List[Atoms]], output:str, extra:dict=None) -> None:

        """
        Dispatches the job based on the job type.
        Args:
            commandcontrol: CommandControl object
            jobtype: The type of job to be performed.
            atoms: The ASE Atoms object, Molecules object, or a list of Atoms objects.
            output: The path to the output file.
            extra: Extra parameters to be passed to the job.
        """

        self.output = output
        self.commandcontrol = commandcontrol
        self.set_throshould(atoms)
        if jobtype == 'opt':
            from .optimization import Optmization

            if isinstance(atoms, (list, Molecules)):
                raise NotImplementedError('For optimization job, only one Atoms object is allowed.')
            opt = Optmization(output=output, atoms=atoms, method=commandcontrol.params.get('method'), params=commandcontrol.params)
            opt.run()
            
        elif jobtype == 'sp':
            from .sp import SinglePoint

            # Handle trajectory/multiple structures
            if isinstance(atoms, Molecules):
                atoms_input = atoms.multiatoms
                sp = SinglePoint(output=output, atoms=atoms_input, paras=commandcontrol)
                sp.run()
            elif isinstance(atoms, list):
                sp = SinglePoint(output=output, atoms=atoms, paras=commandcontrol)
                sp.run()
            else:
                # Single structure (backward compatibility)
                sp = SinglePoint(output=output, atoms=atoms, paras=commandcontrol)
                sp.run()

        elif jobtype == 'scan':
            from .scan import Scan

            if extra is not None:
                if 'scan' not in extra:
                    raise ValueError('Constraints not provided for scan job')
            else:
                raise ValueError('Constraints not provided for scan job')
            
            if isinstance(atoms, (list, Molecules)):
                raise NotImplementedError('For scan job, only one Atoms object is allowed.')

            scan = Scan(output=output, atoms=atoms, method=commandcontrol.params.get('method'), constraints=extra['scan'], params=commandcontrol.params)
            scan.run()
            
        elif jobtype == 'freq':
            from .frequency import Frequency
            if isinstance(atoms, (list, Molecules)):
                raise NotImplementedError('For frequency job, only one Atoms object is allowed.')
            freq = Frequency(output=output, atoms=atoms, paras=commandcontrol.params)
            freq.run()
            
        elif jobtype == 'ts':
            from .ts import TransitionState
            
            # TS job allows Molecules object for methods like NEB, STRING, DIMER
            if isinstance(atoms, (list, Molecules)):
                method = commandcontrol.params.get('method')
                if method in ['neb', 'string', 'dimer', 'afir', 'descafir', 'autoneb']:
                    # Convert Molecules to its internal list if needed
                    atoms_input = atoms.multiatoms if isinstance(atoms, Molecules) else atoms
                    ts = TransitionState(output=output, atoms=atoms_input, method=method, params=commandcontrol.params)
                    ts.run()
                    return
                elif method in ['prfo', 'newton']:
                    raise NotImplementedError('For transition state search job with PRFO or Newton method, only one Atoms object is allowed.')
                else:
                    raise ValueError(f'Unknown TS method: {method}')

            # Single Atoms object
            ts = TransitionState(output=output, atoms=atoms, method=commandcontrol.params.get('method'), params=commandcontrol.params)
            ts.run()
        
        elif jobtype == 'irc':
            from .irc import IRC

            if isinstance(atoms, (list, Molecules)):
                raise NotImplementedError('For IRC job, only one Atoms object is allowed.')
            irc = IRC(output=output, atoms=atoms, method=commandcontrol.params.get('method'), params=commandcontrol.params)
            irc.run()

        elif jobtype == 'md':
            from .md.ensemble.nve import NVE
            from .md.ensemble.nvt import NVT
            from .md.ensemble.npt import NPT

            if isinstance(atoms, (list, Molecules)):
                raise NotImplementedError('For MD job, only one Atoms object is allowed.')

            if 'charge' not in atoms.info or 'mult' not in atoms.info:
                raise ValueError(
                    "MD requires explicit charge and multiplicity. "
                    "Provide either 'XYZ <charge> <mult> <path>' or an inline 'charge mult' line before coordinates."
                )

            ensemble = commandcontrol.params.get('ensemble', 'nve').lower()
            if ensemble == 'nve':
                md = NVE(output=output, atoms=atoms, paras=commandcontrol.params)
            elif ensemble == 'nvt':
                md = NVT(output=output, atoms=atoms, paras=commandcontrol.params)
            elif ensemble == 'npt':
                md = NPT(output=output, atoms=atoms, paras=commandcontrol.params)
            else:
                raise ValueError(f"Unknown MD ensemble: '{ensemble}'")
            md.run()
            
            
        else:
            try:
                raise NotImplementedError('Job type not implemented')
            except NotImplementedError as e:
                self.log_error(str(e))
                
    def set_throshould(self, atoms) -> None:
        """
        Sets the convergence throshould for the atoms object.

        Args:
            atoms: The ASE Atoms object, Molecules object, or list of Atoms.
        """
        
        # Geometry optimization convergence thresholds (Eh/Å and Å), Gaussian-style
        if self.commandcontrol.params.get('level') == 'extratight':
            self.commandcontrol.params['f_max_th'] = 0.00030
            self.commandcontrol.params['f_rms_th'] = 0.00020
            self.commandcontrol.params['dp_max_th'] = 0.00030
            self.commandcontrol.params['dp_rms_th'] = 0.00020

        elif self.commandcontrol.params.get('level') == 'tight':
            self.commandcontrol.params['f_max_th'] = 0.00085
            self.commandcontrol.params['f_rms_th'] = 0.00055
            self.commandcontrol.params['dp_max_th'] = 0.00110
            self.commandcontrol.params['dp_rms_th'] = 0.00075

        elif self.commandcontrol.params.get('level') == 'medium':
            self.commandcontrol.params['f_max_th'] = 0.00285
            self.commandcontrol.params['f_rms_th'] = 0.00190
            self.commandcontrol.params['dp_max_th'] = 0.00315
            self.commandcontrol.params['dp_rms_th'] = 0.00210

        elif self.commandcontrol.params.get('level') == 'loose':
            self.commandcontrol.params['f_max_th'] = 0.00380
            self.commandcontrol.params['f_rms_th'] = 0.00250
            self.commandcontrol.params['dp_max_th'] = 0.00600
            self.commandcontrol.params['dp_rms_th'] = 0.00400

        elif self.commandcontrol.params.get('level') == 'extraloose':
            self.commandcontrol.params['f_max_th'] = 0.00755
            self.commandcontrol.params['f_rms_th'] = 0.00500
            self.commandcontrol.params['dp_max_th'] = 0.01200
            self.commandcontrol.params['dp_rms_th'] = 0.00800

        elif self.commandcontrol.params.get('level') == 'superloose':
            self.commandcontrol.params['f_max_th'] = 0.08500
            self.commandcontrol.params['f_rms_th'] = 0.05500
            self.commandcontrol.params['dp_max_th'] = 0.14500
            self.commandcontrol.params['dp_rms_th'] = 0.09500

        else:
            # fallback to medium
            self.commandcontrol.params['f_max_th'] = 0.00285
            self.commandcontrol.params['f_rms_th'] = 0.00190
            self.commandcontrol.params['dp_max_th'] = 0.00315
            self.commandcontrol.params['dp_rms_th'] = 0.00210

        # Apply thresholds to atoms
        if isinstance(atoms, Molecules):
            # Apply to all atoms in Molecules object
            for atom in atoms.multiatoms:
                atom.f_max_th = self.commandcontrol.params['f_max_th']
                atom.f_rms_th = self.commandcontrol.params['f_rms_th']
                atom.dp_max_th = self.commandcontrol.params['dp_max_th']   
                atom.dp_rms_th = self.commandcontrol.params['dp_rms_th']
        elif isinstance(atoms, list):
            # Apply to all atoms in list
            for atom in atoms:
                atom.f_max_th = self.commandcontrol.params['f_max_th']
                atom.f_rms_th = self.commandcontrol.params['f_rms_th']
                atom.dp_max_th = self.commandcontrol.params['dp_max_th']   
                atom.dp_rms_th = self.commandcontrol.params['dp_rms_th']
        else:
            # Single Atoms object
            atoms.f_max_th = self.commandcontrol.params['f_max_th']
            atoms.f_rms_th = self.commandcontrol.params['f_rms_th']
            atoms.dp_max_th = self.commandcontrol.params['dp_max_th']   
            atoms.dp_rms_th = self.commandcontrol.params['dp_rms_th']



    def log_error(self, error_message: str) -> None:
        """
        Logs error messages to the output file.

        Args:
            error_message: The error message to log.
        """
        with open(self.output, 'a') as file:
            file.write(f"ERROR: {error_message}\n")

    def log_info(self, info_message: list) -> None:
        """
        Logs info messages to the output file.

        Args:
            info_message: The info message to log.
        """
        with open(self.output, 'a') as file:
            for info in info_message:   
                file.write(f"{info}")