#########################################################################
# This module is currently decrepted and will be removed in future versions. #
# Please use malepso.function.calculator.ani instead.                     #
#########################################################################




import os
import torch
import torchani
from ._ani_calculator import ANICalculator

import tad_dftd4 as d4

class Calculator():		 
    def __init__(self, model: int = 1, gpuid: int = None, output: str = None, d4:bool=False)-> ANICalculator:
        """
        Args:
            model: int, default=1
                The model to be used for the calculation.
                1: ANI-2x
                2: ANI-1x
                3: ANI-1ccx
                4: ANI-1xnr
            gpuid: int, default=None
                The GPU ID to be used for calculation. If None, use CPU. Default is None.
            output: str, output file path. Default is None.
        """
        self.output = output
        info_message = [f"\nLoading the Machine Learning Potential Model...\n"]
        scripts_path = os.path.dirname(os.path.realpath(__file__))
        if model == 'ani-2x':
            model_path = os.path.join(scripts_path, 'model/ani-2x_8x')
        elif model == 'ani-1x':
            model_path = os.path.join(scripts_path, 'model/ani-1x_8x')
        elif model == 'ani-1ccx':
            model_path = os.path.join(scripts_path, 'model/ani-1ccx_8x')
        elif model == 'ani-1xnr':
            model_path = os.path.join(scripts_path, 'model/ani-1xnr_8x')
        
        try:
            if not os.path.exists(model_path):
                raise ValueError(f'Model path does not exist.')
        except ValueError as e:
            self.log_info(info_message)
            self.log_error(str(e))
            raise

        info_message.append(f'Loading ANI model ({model}) successfully.\n')

        self.model_path = model_path
        self.gpuid = gpuid
        
        self.d4 = d4

        self.log_info(info_message)
    
    def construct_calculator(self) -> ANICalculator:
        """
            Construct the calculator.
        """

        model_path = self.model_path
        gpuid = self.gpuid

        ###print(gpuid)
        info_message = []
        # find the model files
        torch.set_num_threads(2)
        params_file = [file for file in os.listdir(model_path) if file.endswith('.params')]
        linfit_file = [file for file in os.listdir(model_path) if file.endswith('.dat')]
        model_prefix = os.path.join(model_path, 'train')

        # get the full path of the model files
        params_file = os.path.abspath(os.path.join(model_path, params_file[0]))
        linfit_file = os.path.abspath(os.path.join(model_path, linfit_file[0]))

        # build the calculator
        consts = torchani.neurochem.Constants(params_file)
        aev_computer = torchani.AEVComputer(**consts)
        energy_shifter = torchani.neurochem.load_sae(linfit_file)
        ensemble = torchani.neurochem.load_model_ensemble(consts.species, model_prefix, 8)
        nnp1 = torchani.nn.Sequential(aev_computer, ensemble, energy_shifter)

        # set the device
        if gpuid is not None:
            if torch.cuda.is_available():
                try:
                    device = torch.device(f'cuda:{gpuid}')
                    info_message.append(f'Using GPU {gpuid} for calculation.\n')
                except:
                    info_message.append(f'ERROR: GPU {gpuid} is not available.\n')
                    self.log_info(info_message)
                    raise ValueError(f'GPU {gpuid} is not available.\n')
            else:
                info_message.append('ERROR: CUDA is not available.\n')
                self.log_info(info_message)
                raise ValueError('CUDA is not available.\n')
        else:
            device = torch.device('cpu')
            info_message.append('Using CPU for calculation.\n')

        nnp1 = nnp1.to(device)

        if self.d4:
            info_message.append('\nSetting up D4 dispersion correction...\n')


        self.log_info(info_message)
        # set the calculator
        return ANICalculator(consts.species, nnp1, d4=self.d4)
    
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
