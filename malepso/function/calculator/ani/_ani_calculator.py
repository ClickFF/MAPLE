
import os

import torch

import ase

from ..calculator_base import CalcABC


class ANICalculator(CalcABC):
    implemented_properties = ['energy', 'forces', 'stress', 'free_energy']

    def __init__(self, device: torch.device,
        model:str = 'ani2x',
        overwrite=False,
        d4=False,
        implicit: str = 'none',
        solvent: str = 'none',
        ):
        """
        Initialize the ANICalculator.

        Args:
            device (torch.device): The device to run the model on.
            model (str, optional): The model to use. Defaults to 'ani-2x'.
            overwrite (bool, optional): Whether to overwrite existing models. Defaults to False.
            d4 (bool, optional): Whether to use D4 dispersion correction. Defaults to False.
        """
        super().__init__()
        info_message = [f"\nLoading the Machine Learning Potential Model...\n"]
        
        
        model_dir = os.path.dirname(os.path.realpath(__file__))
        model_dir = os.path.dirname(model_dir)
        model_path = os.path.join(model_dir, 'model', f'{model}.pt')
        
        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()
        
        
        info_message.append(f'Loading model ({model}) successfully.\n')
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.device = device
        self.dtype = torch.float32
        self.overwrite = overwrite
        self.d4 = d4

        # Initialize implicit solvent
        self.implicit_solv_init(implicit=implicit, solvent=solvent)

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=ase.calculators.calculator.all_changes):
        super().calculate(atoms, properties, system_changes)

        coordinates = torch.tensor(atoms.get_positions(), dtype=self.dtype, device=self.device, requires_grad='forces' in properties).unsqueeze(0)
        
        energy = self.get_energy(atoms, coordinates)

        if self.solvent_correction:
            solvent_energy = self.implicit_solv_energy(atoms)
            energy += solvent_energy

        self.results['energy'] = energy.item()
        self.results['free_energy'] = energy.item()

        if 'forces' in properties:
            forces = -torch.autograd.grad(energy, coordinates, retain_graph='stress' in properties)[0]
            if self.solvent_correction:
                solvent_energy, solvent_force = self.implicit_solv_energy_and_force(atoms)
                forces += solvent_force
                
            self.results['forces'] = forces.squeeze(0).cpu().numpy()

    def get_energy(self, atoms, coordinates):
        
        species = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long, device=self.device).unsqueeze(0)

        energy = self.model(species, coordinates)[0]
        if self.d4:
            energy += self.dftd4(species, coordinates)
        
        return energy

    @staticmethod
    def compute_hessian(coords, energy):
    
        num_atoms = coords.shape[1]
        hessian = torch.zeros((3 * num_atoms, 3 * num_atoms), dtype=coords.dtype, device=coords.device)
        grad = torch.autograd.grad(energy, coords, create_graph=True)[0].view(-1)
        for i in range(3 * num_atoms):
            grad2 = torch.autograd.grad(grad[i], coords, retain_graph=True)[0].view(-1)
            hessian[i, :] = grad2
        return hessian

    def get_hessian(self, atoms=None):
        
        coordinates = torch.tensor(atoms.get_positions(), dtype=self.dtype, device=self.device, requires_grad=True).unsqueeze(0)
        
        energy = self.get_energy(atoms, coordinates)

        return self.compute_hessian(coordinates, energy)

    def dftd4(self, species, coordinates):
        import tad_dftd4 as d4
        charge = torch.tensor(0.0)
        param = {
            "s6": coordinates.new_tensor(1.0),
            "s8": coordinates.new_tensor(0.34783580),
            "s9": coordinates.new_tensor(1.0),
            "a1": coordinates.new_tensor(0.57488291),
            "a2": coordinates.new_tensor(6.41921802),
        }
        bohr_coords = coordinates[0] * 1.8897259886
        return torch.sum(d4.dftd4(species[0], bohr_coords, charge, param))


    