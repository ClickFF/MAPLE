#!/usr/bin/env python
"""
Construct Model From NeuroChem Files
====================================

This tutorial illustrates how to manually load model from `NeuroChem files`_.

.. _NeuroChem files:
    https://github.com/isayev/ASE_ANI/tree/master/ani_models

"""

###############################################################################
# To begin with, let's first import the modules we will use:

import numpy as np

from ase import Atoms 
import torchani

from ..function.read import InputReader
from ..function.calculator import Calculator
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
        self._mlp_initiator(self.model, self.gpuid)
        self.atoms.set_calculator(self.calulator)
        self._jobtype_dispatcher(self.jobtype, self.atoms, self.output)

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
        self.gpuid = reader.gpuid
        self.model = reader.model
        self.jobtype = reader.jobtype
        self.d4 = reader.d4

    
    def _mlp_initiator(self, model:int, gpuid:int) -> torchani.ase.Calculator:
        """
            This function initializes the model.

            Args:
                model: int
                    The model to be used for the calculation.
                    1: ANI-2x
                    2: ANI-1x
                    3: ANI-1ccx
                    4: ANI-1xnr
                gpuid: int
                    The GPU ID to be used for calculation. If None, use CPU. Default is None.
        """
        calculator = Calculator(model, gpuid, self.output,d4=self.d4)
        self.calulator = calculator.construct_calculator()

        return self.calulator
    
    def _jobtype_dispatcher(self, jobtype:int, atoms:Atoms, output:str, method:str='RFO') -> None:
        """
            This function dispatches the job type.

            Args:
                jobtype(int): The type of job to be performed.
                atoms(Atoms): The ASE Atoms object to be optimized.
                output(str): The path to the output file.
                method(str): The optimization method to be used. (Default: LBFGS)
        """
        
        dispatcher = Dispatcher()
        dispatcher(jobtype, atoms, output, method)
    





