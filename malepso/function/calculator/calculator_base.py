import torch
import numpy as np

import ase.calculators.calculator

class CalcABC(ase.calculators.calculator.Calculator):
    def __init__(self):
        super().__init__()


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

    def get_hvp(self, atoms, n: np.ndarray):
        """
        Compute Hessian-vector product Hn for the given atoms and direction n using autograd.
        Args:
            atoms (ase.Atoms): system
            n (np.ndarray): direction vector, shape (3N,)
        Returns:
            Hn (torch.Tensor): Hessian-vector product (3N,) on same device/dtype
            forces (torch.Tensor): forces (3N,) on same device/dtype
            energy (torch.Tensor): scalar total energy
        """
        # 1. prepare coordinates with grad enabled
        coords = torch.tensor(
            atoms.get_positions(),
            dtype=self.dtype,
            device=self.device,
            requires_grad=True
        ).unsqueeze(0)

        # 2. atomic numbers
        species = torch.tensor(
            atoms.get_atomic_numbers(),
            dtype=torch.long,
            device=self.device
        ).unsqueeze(0)

        # 3. forward pass → energy
        energy = self.model(species, coords)[0]
        if self.d4:
            energy += self.dftd4(species, coords)

        # 4. compute gradient (forces = -grad V)
        grad = torch.autograd.grad(energy, coords, create_graph=True)[0].squeeze(0)  # shape (N,3)
        grad_vec = grad.view(-1)  # (3N,)

        # 5. Hessian-vector product: grad(grad·n)
        n_tensor = torch.tensor(n, dtype=self.dtype, device=self.device)
        hvp = torch.autograd.grad(
            grad_vec @ n_tensor, coords, retain_graph=True
        )[0].squeeze(0).view(-1)  # (3N,)

        # 6. Forces (already computed, negative gradient)
        forces = -grad_vec

        return hvp, forces, energy
