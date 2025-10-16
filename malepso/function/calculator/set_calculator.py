import torch

import ase

from .ani._ani_calculator import ANICalculator
from .mace._mace_calculator import MACECalculator

IMPLEMENTATION_MODELs = [
            'ani2x',
            'ani1x',
            'ani1ccx',
            'ani1xnr',
            'maceoff23s',
            'maceoff23m',
            'maceoff23l',
            'egret',
            'aimnet2',
            'uma',
        ]

class SetClaculator():

    def __init__(self, device: torch.device, model:str, output: str, d4:bool=False) -> None:
        self.output = output
        self.model = model
        self.d4 = d4
        self.device = device
        self.model = model

    def set_calculator(self) -> ase.calculators.calculator.Calculator:

        if self.model not in IMPLEMENTATION_MODELs:
            error_message = f"\n [ERROR] Unsupported model: {self.model}\n"
            self.log_error(error_message)
            raise ValueError(error_message)

        if self.model in ['ani2x', 'ani1x', 'ani1ccx', 'ani1xnr']:
            calculator = ANICalculator(model=self.model, d4=self.d4, device=self.device)
            return calculator
        else:
            if self.d4 == True : self.log_info([f"\n [WARNING:] D4 is not supported for model {self.model}. D4 will be ignored.\n"])
            if self.model in ['maceoff23s', 'maceoff23m', 'maceoff23l','egret']:
                from .mace._mace_calculator import MACECalculator
                calculator = MACECalculator(model=self.model, device=self.device)
                return calculator   
            elif self.model in ['aimnet2']:
                from .aimnet._aimnet2_calculator import AIMNet2Calculator
                calculator = AIMNet2Calculator(model=self.model, device=self.device)
                return calculator
            else:
                raise ValueError(f"Model '{self.model}' is not implemented yet.")
        

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