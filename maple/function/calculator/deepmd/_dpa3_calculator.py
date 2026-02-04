"""
DPA3 / DeepMD-Kit calculator (PyTorch backend).
Uses the unified DP ASE calculator from deepmd.calculator; converts eV -> Hartree.
"""
import os
from typing import Literal

from ase.calculators.calculator import all_changes
from ..calculator_base import CalcABC

EV2HARTREE = 1.0 / 27.211386245988


def _get_dp_calculator():
    """Import DeepMD-Kit DP calculator (works with .pt/.pth DPA3 models)."""
    try:
        from deepmd.calculator import DP
        return DP
    except ImportError:
        try:
            from deepmd.pt.utils.ase_calc import DPCalculator as DP
            return DP
        except ImportError:
            raise ImportError(
                "DeepMD-Kit is required for DPA3. Install with: pip install deepmd-kit"
            ) from None


class DPA3Calculator(CalcABC):
    """ASE calculator wrapping DeepMD-Kit DP (DPA3 and other Deep Potential models)."""

    implemented_properties = ["energy", "forces", "free_energy"]

    def __init__(
        self,
        device,
        model: str = "dpa3",
        overwrite: bool = False,
        implicit: Literal["gbsa", "none"] = "gbsa",
        solvent: str = "none",
    ):
        super().__init__()
        DP = _get_dp_calculator()

        model_dir = os.path.dirname(os.path.realpath(__file__))
        model_dir = os.path.dirname(model_dir)
        model_dir = os.path.dirname(model_dir)
        default_path = os.path.join(model_dir, "model", f"{model}.pt")
        if os.path.isfile(default_path):
            model_path = default_path
        elif os.path.isdir(model):
            model_path = model
        elif os.path.isfile(model):
            model_path = model
        else:
            model_path = default_path

        self._dp = DP(model=model_path)
        self.device = device
        self.overwrite = overwrite
        self.implicit_solv_init(implicit=implicit, solvent=solvent)

    def calculate(
        self, atoms=None, properties=("energy", "forces"), system_changes=all_changes
    ):
        super().calculate(atoms, properties, system_changes)

        self._dp.calculate(atoms, properties=properties, system_changes=system_changes)

        energy_eV = self._dp.results.get("energy", 0.0)
        energy = energy_eV * EV2HARTREE

        if self.solvent_correction:
            energy = energy + self.implicit_solv_energy(atoms).item()

        self.results["energy"] = energy if isinstance(energy, float) else energy.item()
        self.results["free_energy"] = self.results["energy"]

        if "forces" in properties:
            forces = self._dp.results.get("forces")
            if forces is not None:
                forces = forces * EV2HARTREE  # eV/Å -> Hartree/Å
                if self.solvent_correction:
                    _, solvent_force = self.implicit_solv_energy_and_force(atoms)
                    forces = forces + solvent_force.cpu().numpy()
                self.results["forces"] = forces
