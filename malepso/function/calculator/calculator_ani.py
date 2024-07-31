import os
import torch
import torchani
###############################################################################
# Now let's read constants from constant file and construct AEV computer.
#try:
   # path = os.path.dirname(os.path.realpath(__file__))
#except NameError:
class Calculator:		 
    def __init__(self):
        path = os.environ['NC_ROOT']
        const_file = os.path.join(path, 'ani_models/ani-2x_8x/rHCNOSFCl-5.1R_16-3.5A_a8-4.params')  # noqa: E501
        print('PATH:',const_file)
        consts = torchani.neurochem.Constants(const_file)
        aev_computer = torchani.AEVComputer(**consts)
        torch.set_num_threads(2)
###############################################################################
# Now let's read self energies and construct energy shifter.
        sae_file = os.path.join(path, 'ani_models/ani-2x_8x/sae_linfit.dat')  # noqa: E501
        energy_shifter = torchani.neurochem.load_sae(sae_file)

###############################################################################
# Now let's read a whole ensemble of models.
        model_prefix = os.path.join(path, 'ani_models/ani-2x_8x/train')  # noqa: E501
        ensemble = torchani.neurochem.load_model_ensemble(consts.species, model_prefix, 8)  # noqa: 
###############################################################################
# You can create the pipeline of computing energies:
        nnp1 = torchani.nn.Sequential(aev_computer, ensemble, energy_shifter)
###############################################################################
# You can also create an ASE calculator using the ensemble:
        self.cal = torchani.ase.Calculator(consts.species, nnp1)
       
