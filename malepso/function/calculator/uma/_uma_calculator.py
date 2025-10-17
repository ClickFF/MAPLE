import importlib
import torch
import numpy as np
from typing import Literal
from ase import Atoms

try:
    from fairchem.core import pretrained_mlip
    from fairchem.core.calculate.ase_calculator import FAIRChemCalculator
except ImportError:
    raise ImportError("fairchem-core is not installed. Please install it first.")

EV2HARTREE = 1.0 / 27.211386245988


class UMACalculator(FAIRChemCalculator):
    """
    UMA Calculator: extends Meta's FAIRChemCalculator with additional methods for
    total energy and Hessian calculation. Default task is 'omol' (molecular systems).
    """

    def __init__(
        self,
        device: torch.device,
        model: str = "uma",
        overrides: dict | None = None,
    ):
        """
        Initialize UMA Calculator.

        Args:
            device (torch.device): Target device ('cuda' or 'cpu').
            model (str): UMA model name or local checkpoint path.
            overrides (dict, optional): Additional inference configuration overrides.
        """
        UMA_MODELS_MAP = {"uma": "uma-s-1p1"}
        model = UMA_MODELS_MAP.get(model)

        device = str(device)
        device = "cuda" if device.startswith("cuda") else "cpu"

        if not importlib.util.find_spec("fairchem"):
            raise ImportError("fairchem-core is not installed. Please install it first.")

        predictor = pretrained_mlip.get_predict_unit(
            model,
            inference_settings="default",
            overrides=overrides,
            device=device,
        )
        super().__init__(predict_unit=predictor, task_name="omol")
        self.device = torch.device(device)

    def get_energy(self, atoms: Atoms) -> torch.Tensor:
        """
        Compute total energy (Hartree) for a given atomic structure.

        Args:
            atoms (ase.Atoms): Atomic structure.

        Returns:
            torch.Tensor: Total energy in Hartree.
        """
        self.calculate(atoms, properties=["energy"], system_changes=atoms.calc_check())
        energy_value = self.results["energy"]
        return torch.tensor(energy_value, dtype=torch.float32, device=self.device)

    def get_hessian(self, atoms: Atoms) -> torch.Tensor:
        """
        Compute the Hessian matrix using autograd. Complexity is O(N^2).

        Args:
            atoms (ase.Atoms): Atomic structure.

        Returns:
            torch.Tensor: Hessian matrix (3N x 3N) in Hartree/Å².
        """
        coords = torch.tensor(
            atoms.get_positions(),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        ).unsqueeze(0)

        energy = self._energy_from_tensor(atoms, coords)
        grad = torch.autograd.grad(energy, coords, create_graph=True)[0].view(-1)

        num_atoms = coords.shape[1]
        hessian = torch.zeros(
            (3 * num_atoms, 3 * num_atoms), dtype=torch.float32, device=self.device
        )
        for i in range(3 * num_atoms):
            grad2 = torch.autograd.grad(grad[i], coords, retain_graph=True)[0].view(-1)
            hessian[i, :] = grad2

        return hessian

    def _energy_from_tensor(self, atoms: Atoms, coords: torch.Tensor) -> torch.Tensor:
        """
        Internal helper: recompute total energy from a coordinate tensor.

        Args:
            atoms (ase.Atoms): Original atomic structure.
            coords (torch.Tensor): Atomic positions tensor.

        Returns:
            torch.Tensor: Total energy.
        """
        atoms_copy = atoms.copy()
        atoms_copy.set_positions(coords.squeeze(0).detach().cpu().numpy())
        return self.get_energy(atoms_copy)

    def calculate(self, atoms, properties=None, system_changes=None):
        """
        Override base calculate() to convert energy and forces into Hartree.

        Args:
            atoms (ase.Atoms): Atomic structure.
            properties (list, optional): Properties to calculate.
            system_changes (list, optional): System changes to consider.

        Returns:
            dict: Calculation results with energy and forces converted to Hartree.
        """
        super().calculate(atoms, properties, system_changes)

        if "energy" in self.results:
            self.results["energy"] *= EV2HARTREE
        if "free_energy" in self.results:
            self.results["free_energy"] *= EV2HARTREE
        if "forces" in self.results:
            self.results["forces"] *= EV2HARTREE

        return self.results
