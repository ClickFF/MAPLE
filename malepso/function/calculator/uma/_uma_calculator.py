import importlib
import torch
import numpy as np
from typing import Literal
from ase import Atoms
from ase.calculators.calculator import all_changes
from ase.calculators.calculator import Calculator

try:
    from fairchem.core import pretrained_mlip
    from fairchem.core.calculate.ase_calculator import FAIRChemCalculator
    from fairchem.core.datasets import data_list_collater
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
