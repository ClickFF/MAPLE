import os
import torch
import torchani

class Calculator():		 
    def __init__(self, model: int = 1, gpuid: int = None):
        """
        Args:
            model: int, default=1
                The model to be used for the calculation.
                1: ANI-2x
                2: ANI-1x
                3: ANI-1ccx
            gpuid: int, default=None
                The GPU ID to be used for calculation. If None, use CPU. Default is None.
        """
        super().__init__()

        model_dict = {1: 'ANI-2x', 2: 'ANI-1x', 3: 'ANI-1ccx'}

        scripts_path = os.path.dirname(os.path.realpath(__file__))
        if model == 1:
            model_path = os.path.join(scripts_path, 'model/ani-2x_8x')
        elif model == 2:
            model_path = os.path.join(scripts_path, 'model/ani-1x_8x')
        elif model == 3:
            model_path = os.path.join(scripts_path, 'model/ani-1ccx_8x')

        self.construct_calculator(model_path, gpuid=gpuid)

        print('Loading ANI model',f'({model_dict.get(model)})','successfully.')

    def construct_calculator(self, model_path: str, gpuid: int = None):

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
                except:
                    raise ValueError(f'GPU {gpuid} is not available.')
            else:
                raise ValueError('CUDA is not available.')
        else:
            device = torch.device('cpu')
        
        nnp1 = nnp1.to(device)

        # set the calculator
        self.cal = torchani.ase.Calculator(consts.species, nnp1)
