
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
        self.hessian: str = 'numerical'  # 'analytic' or 'numerical'

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

    def get_hessian(
        self,
        atoms: ase.Atoms,
        delta: float = 0.002,
    ) -> torch.Tensor:
        """
        Compute the Hessian matrix using either analytic or numerical method.
        
        Method is determined by self.hessian:
        - 'analytic': Use automatic differentiation (faster, exact)
        - 'numerical': Use finite-difference forces (slower, approximate)
        
        Returns a (3N, 3N) torch.Tensor on self.device.
        
        Args:
            atoms: ASE Atoms object
            delta: Step size for numerical differentiation (only used if method='numerical')
        """
        if self.hessian == 'analytic':
            return self._get_hessian_analytic(atoms)
        elif self.hessian == 'numerical':
            return self._get_hessian_numerical(atoms, delta)
        else:
            raise ValueError(f"Unknown hessian method: {self.hessian}. Must be 'analytic' or 'numerical'")


    def _get_hessian_analytic(self, atoms: ase.Atoms) -> torch.Tensor:
        """
        Compute Hessian using automatic differentiation.
        Fast and exact, but requires energy to be differentiable w.r.t. coordinates.
        """
        coordinates = torch.tensor(
            atoms.get_positions(), 
            dtype=self.dtype, 
            device=self.device, 
            requires_grad=True
        ).unsqueeze(0)
        
        energy = self.get_energy(atoms, coordinates)
        
        return self.compute_hessian(coordinates, energy)


    def _get_hessian_numerical(
        self, 
        atoms: ase.Atoms, 
        delta: float = 0.002
    ) -> torch.Tensor:
        """
        Compute Hessian using finite-difference forces.
        Hessian is defined as: H = d²E/dx_i dx_j = -∂F_i/∂x_j
        
        Args:
            atoms: ASE Atoms object
            delta: Step size for finite difference
        """
        import numpy as np
        from ase.constraints import FixAtoms
        from ase.calculators.calculator import all_changes

        device = self.device
        dtype = self.dtype

        # Basic geometry setup
        N = len(atoms)
        pos0 = atoms.get_positions().copy()  # (N, 3) numpy array

        # Identify frozen atoms from FixAtoms constraint
        fixed = {
            i for c in getattr(atoms, "constraints", [])
            if isinstance(c, FixAtoms)
            for i in c.get_indices()
        }
        movable = [i for i in range(N) if i not in fixed]

        # Allocate Hessian on device
        H = torch.zeros((3 * N, 3 * N), dtype=dtype, device=device)

        # If everything is frozen, return zero Hessian
        if len(movable) == 0:
            return H

        # Helper function: evaluate ANI forces at a displaced geometry
        def ani_force_at(pos_numpy: np.ndarray) -> torch.Tensor:
            """
            Evaluate ANI forces at the given coordinates.
            Returns a tensor of shape (N, 3) on the target device.
            """
            at = atoms.copy()
            at.set_positions(pos_numpy)

            # Preserve constraints if present
            if getattr(atoms, "constraints", None):
                at.set_constraint(atoms.constraints)

            # Compute forces with the internal ANI calculator
            self.calculate(at, properties=["forces"], system_changes=all_changes)
            F_np = self.results["forces"]  # numpy (N, 3)
            return torch.tensor(F_np, dtype=dtype, device=device)

        # Finite-difference second derivatives:
        # H_ij = -∂F_i/∂x_j ≈ -(F(+δ) - F(-δ)) / (2δ)
        for a in movable:      # iterate over movable atoms
            for k in range(3):  # iterate over x, y, z directions
                row = 3 * a + k

                # +delta displacement
                pos_p = pos0.copy()
                pos_p[a, k] += delta
                Fp = ani_force_at(pos_p)

                # -delta displacement
                pos_m = pos0.copy()
                pos_m[a, k] -= delta
                Fm = ani_force_at(pos_m)

                # Central difference derivative of force
                # ∂F/∂x ≈ (F(+δ) - F(-δ)) / (2δ)
                dF = (Fp - Fm) / (2.0 * delta)

                # Hessian uses: H = -∂F/∂x
                H[row, :] = (-dF).reshape(-1)

        # Optional: symmetrize to reduce numerical noise
        # H = 0.5 * (H + H.transpose(0, 1))

        return H


    def dftd4(self, species, coordinates):
        import tad_dftd4 as d4
        charge = torch.tensor(0.0, device=self.device)
        param = {
            "s6": coordinates.new_tensor(1.0),
            "s8": coordinates.new_tensor(0.34783580),
            "s9": coordinates.new_tensor(1.0),
            "a1": coordinates.new_tensor(0.57488291),
            "a2": coordinates.new_tensor(6.41921802),
        }
        # Å → Bohr 转换
        bohr_coords = coordinates[0] * 1.8897261245864
        return torch.sum(d4.dftd4(species[0], bohr_coords, charge, param))


    