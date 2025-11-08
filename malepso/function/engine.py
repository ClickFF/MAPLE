from typing import Union, List

from ase import Atoms 
import ase
import torch

from ..function.read import InputReader
from ..function.dispatcher import Dispatcher

class engine():
    def __init__(self):
        
        self.output:str = None
        self.gpuid:int = None
        self.model:int = None
        # 1: ANI-2x
        # 2: ANI-1x
        # 3: ANI-1ccx
        # 4: ANI-1xnr

        self.jobtype:int = None
        # 1: opt
        # 2: sp
        # 3: scan
        # 4: freq
        # 5: ts

        self.calulator = None

        self.d4 = False
        # DFT-D4 dispersion correction

    def __call__(self, input_file_name:str,output_file_name:str=None):
        """
            This is the engine of the program.
            Args:
                input_file_name: The path to the input file.
                output_file_name: The path to the output file. (Default: None)
        """
        
        self._input_reader(input_file_name, output_file_name)
        self._mlp_initiator(self.model, self.device)

        if isinstance(self.atoms, Atoms):       
            self.atoms.calc = self.calulator
        elif isinstance(self.atoms, list):     
            for atom in self.atoms:
                atom.calc = self.calulator
                
        # Self.atoms printing
        self._jobtype_dispatcher(self.commandcontrol, self.jobtype, self.atoms, self.output, extra=self.extra)

    def _input_reader(self, input_file_name:str, output_file_name:str=None) -> Atoms:
        """
            This function reads the input file.

            Args:
                input_file_name: The path to the input file.
                output_file_name: The path to the output file. (Default: Same as input file)
            
            Returns:
                Atoms: ASE Atoms object.
        """
        reader = InputReader()
        self.atoms = reader(input_file_name, output_file_name)
        self.output = reader.output
        self.device = reader.device
        self.model = reader.model
        self.jobtype = reader.jobtype
        self.d4 = reader.d4

        self.extra = {}

        if self.jobtype == 'scan':
            self.extra = {'scan': reader.scan_constraints}
        
        self.commandcontrol = reader.command_control
        
        # Explicit Solvation Treatment
        if self.commandcontrol.get('solv', {}).get('explicit', None) is not None:

            from .read import ExplicitSolv
            self.atoms = ExplicitSolv(self.atoms, params=self.commandcontrol.get('solv'), 
                    device=self.device, output=self.output)


    def _mlp_initiator(self, model:str, device: torch.device):
        """
            This function initializes the model.

            Args:
                model: The model to be used.
                device: torch.device
                    The device to run the model on.
        """
        
        from .calculator import SetClaculator

        implicit_method = self.commandcontrol.get('solv', {}).get('method', None)
        solvent = self.commandcontrol.get('solv', {}).get('implicit', None)

        setcalculator = SetClaculator(device, model, self.output, 
                        d4=self.d4, implicit=implicit_method, solvent=solvent)
        self.calulator = setcalculator.set_calculator()
    
    def _jobtype_dispatcher(self, commandcontrol, jobtype:int, atoms:Union[Atoms,List[Atoms]], output:str, extra:dict=None) -> None:
        """
            This function dispatches the job type.

            Args:
                commandcontrol: CommandControl object
                jobtype(int): The type of job to be performed.
                atoms(Atoms): The ASE Atoms object, it can also be a list of Atoms objects.
                output(str): The path to the output file.
                method(str): The optimization method to be used. (Default: LBFGS)
                extra(dict): Extra parameters to be passed to the job.
        """
        dispatcher = Dispatcher()
        dispatcher(commandcontrol, jobtype, atoms, output, extra)
    





