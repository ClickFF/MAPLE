

from ase import Atoms

class Dispatcher():
    def __init__(self):
        pass

    def __call__(self, jobtype: int, atoms: Atoms, output:str, method: str='LBFGS') -> None:

        if jobtype == 1:
            from .optimization import Optmization

            opt = Optmization(output=output, atoms=atoms, method=method)
            opt.run()
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
