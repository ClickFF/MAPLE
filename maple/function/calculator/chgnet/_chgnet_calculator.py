"""
CHGNet calculator wrapper. Uses pretrained CHGNet via chgnet package; converts eV -> Hartree.
"""
from typing import Literal

from ase.calculators.calculator import all_changes
from ..calculator_base import CalcABC

EV2HARTREE = 1.0 / 27.211386245988


def _get_chgnet_calculator(device):
    """Build CHGNet ASE calculator (pretrained model, no local file)."""
    try:
        from chgnet.model import CHGNet
        from chgnet.model import CHGNetCalculator
    except ImportError:
        raise ImportError(
            "CHGNet is required. Install with: pip install chgnet"
        ) from None
    chgnet = CHGNet.load()
    return CHGNetCalculator(potential=chgnet)


class CHGNetCalc(CalcABC):
    """ASE calculator wrapping CHGNet (charge-informed graph neural network potential)."""

    implemented_properties = ["energy", "forces", "free_energy", "stress"]

    def __init__(
        self,
        device,
        model: str = "chgnet",
        overwrite: bool = False,
        implicit: Literal["gbsa", "none"] = "gbsa",
        solvent: str = "none",
    ):
        super().__init__()
        self._chg = _get_chgnet_calculator(device)
        self.device = device
        self.overwrite = overwrite
        self.implicit_solv_init(implicit=implicit, solvent=solvent)

    def calculate(
        self, atoms=None, properties=("energy", "forces"), system_changes=all_changes
    ):
        super().calculate(atoms, properties, system_changes)

        self._chg.calculate(atoms, properties=properties, system_changes=system_changes)

        energy_eV = self._chg.results.get("energy", 0.0)
        energy = energy_eV * EV2HARTREE

        if self.solvent_correction:
            energy = energy + self.implicit_solv_energy(atoms).item()

        self.results["energy"] = energy if isinstance(energy, float) else energy.item()
        self.results["free_energy"] = self.results["energy"]

        if "forces" in properties:
            forces = self._chg.results.get("forces")
            if forces is not None:
                forces = forces * EV2HARTREE  # eV/Å -> Hartree/Å
                if self.solvent_correction:
                    _, solvent_force = self.implicit_solv_energy_and_force(atoms)
                    forces = forces + solvent_force.cpu().numpy()
                self.results["forces"] = forces

        if "stress" in properties:
            stress = self._chg.results.get("stress")
            if stress is not None:
                self.results["stress"] = stress * EV2HARTREE  # eV/Å³ -> Hartree/Å³
