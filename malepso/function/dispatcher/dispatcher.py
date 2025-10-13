

from typing import List,Union

from ase import Atoms

class Dispatcher():
    def __init__(self):
        pass

    def __call__(self, jobtype: int, atoms: Union[Atoms, List[Atoms]], output:str, method: str='LBFGS', extra:dict=None) -> None:

        # jobtype: 1 for optimization, 2 for single point energy, 3 for scan,
        #            4 for frequency, 5 for transition state search

        if jobtype == 1:
            from .optimization import Optmization

            if isinstance(atoms, list):
                raise NotImplementedError('For optimization job, only one Atoms object is allowed.')

            opt = Optmization(output=output, atoms=atoms, method=method)
            opt.run()
        elif jobtype == 2:
            from .sp import SinglePoint

            if isinstance(atoms, list):
                raise NotImplementedError('For single point energy job, only one Atoms object is allowed.')
            sp = SinglePoint(output=output, atoms=atoms)
            sp.run()

        elif jobtype == 3:
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
        
        elif jobtype == 4:
            from .frequency import Frequency

            if isinstance(atoms, list):
                raise NotImplementedError('For frequency job, only one Atoms object is allowed.')

            freq = Frequency(output=output, atoms=atoms)
            freq.run()
        elif jobtype == 5:
            from .ts import TransitionState

            if isinstance(atoms, list):
                raise NotImplementedError('For transition state search job, only one Atoms object is allowed.')
            ts = TransitionState(output=output, atoms=atoms)
            ts.run()
        else:
            try:
                raise NotImplementedError('Job type not implemented')
            except NotImplementedError as e:
                self.log_error(str(e))

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
