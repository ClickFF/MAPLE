

from typing import List,Union

from ase import Atoms

class Dispatcher():
    def __init__(self):
        pass

    def __call__(self, commandcontrol, jobtype: int, atoms: Union[Atoms, List[Atoms]], output:str, method: str='LBFGS', extra:dict=None, ) -> None:

        """
        Dispatches the job based on the job type.
        Args:
            jobtype: The type of job to be performed.
            atoms: The ASE Atoms object, it can also be a list of Atoms objects.
            output: The path to the output file.
            method: The optimization method to be used. (Default: LBFGS)
            extra: Extra parameters to be passed to the job.
        """

        self.output = output
        self.set_throshould(atoms)
        
        if jobtype == 'opt':
            from .optimization import Optmization

            if isinstance(atoms, list):
                raise NotImplementedError('For optimization job, only one Atoms object is allowed.')

            opt = Optmization(output=output, atoms=atoms, method=method)
            opt.run()
            
        elif jobtype == 'sp':
            from .sp import SinglePoint

            if isinstance(atoms, list):
                raise NotImplementedError('For single point energy job, only one Atoms object is allowed.')
            sp = SinglePoint(output=output, atoms=atoms)
            sp.run()

        elif jobtype == 'scan':
            from .scan import Scan

            if extra is not None:
                if 'scan' not in extra:
                    raise ValueError('Constraints not provided for scan job')
            else:
                raise ValueError('Constraints not provided for scan job')
            
            if isinstance(atoms, list):
                raise NotImplementedError('For scan job, only one Atoms object is allowed.')

            scan = Scan(output=output, atoms=atoms, method=method, constraints=extra['scan'])
            scan.run()
            
        elif jobtype == 'freq':
            from .frequency import Frequency

            if isinstance(atoms, list):
                raise NotImplementedError('For frequency job, only one Atoms object is allowed.')

            freq = Frequency(output=output, atoms=atoms)
            freq.run()
            
        elif jobtype == 'ts':
            from .ts import TransitionState
            if isinstance(atoms, list):
                if commandcontrol.params.get('method') in ['neb', 'string']:
                    ts = TransitionState(output=output, atoms=atoms, method=commandcontrol.params.get('method'), params=commandcontrol.params)
                    ts.run()
                    return
                elif commandcontrol.params.get('method') in ['prfo', 'newton']:
                    raise NotImplementedError('For transition state search job, only one Atoms object is allowed for PRFO or Newton method.')

            ts = TransitionState(output=output, atoms=atoms, params=commandcontrol.params)
            ts.run()
            
            
        else:
            try:
                raise NotImplementedError('Job type not implemented')
            except NotImplementedError as e:
                self.log_error(str(e))
                
    def set_throshould(self, atoms) -> None:
        """
        Sets the convergence throshould for the atoms object.

        Args:
            atoms: The ASE Atoms object.
        """
        
        if self.commandcontrol.params.get('level') == 'high':
            self.commandcontrol.params['f_max_th'] = 0.00015*27.211386024367243
            self.commandcontrol.params['f_rms_th'] = 0.0001*27.211386024367243
            self.commandcontrol.params['dp_max_th'] = 0.0006
            self.commandcontrol.params['dp_rms_th'] = 0.0004
            
        # default level is medium
        elif self.commandcontrol.params.get('level') == 'medium':
            self.commandcontrol.params['f_max_th'] = 0.00045*27.211386024367243
            self.commandcontrol.params['f_rms_th'] = 0.0003*27.211386024367243
            self.commandcontrol.params['dp_max_th'] = 0.0018   
            self.commandcontrol.params['dp_rms_th'] = 0.0012
        
        # low level
        elif self.commandcontrol.params.get('level') == 'low':
            self.commandcontrol.params['f_max_th'] = 0.00075*27.211386024367243
            self.commandcontrol.params['f_rms_th'] = 0.0005*27.211386024367243
            self.commandcontrol.params['dp_max_th'] = 0.003   
            self.commandcontrol.params['dp_rms_th'] = 0.002

        if isinstance(atoms, list):
            for atom in atoms:
                atom.f_max_th=self.commandcontrol.params['f_max_th']
                atom.f_rms_th=self.commandcontrol.params['f_rms_th']
                atom.dp_max_th=self.commandcontrol.params['dp_max_th']   
                atom.dp_rms_th=self.commandcontrol.params['dp_rms_th']
        else:
            atoms.f_max_th=self.commandcontrol.params['f_max_th']
            atoms.f_rms_th=self.commandcontrol.params['f_rms_th']
            atoms.dp_max_th=self.commandcontrol.params['dp_max_th']   
            atoms.dp_rms_th=self.commandcontrol.params['dp_rms_th']

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
