from __future__ import annotations

import os
from typing import Literal

import numpy as np
import torch
from ase.calculators.calculator import all_changes

from ..calculator_base import CalcABC, hessian_via_double_autograd, register_calculator
from ._common import one_hot_node_attrs, radius_graph_no_pbc


def _macepol_atomic_charges(density_coefficients, n_atoms: int) -> np.ndarray:
    density = density_coefficients.detach().cpu().numpy()
    density = np.asarray(density, dtype=float)
    if density.ndim != 2 or density.shape[0] < n_atoms or density.shape[1] < 1:
        raise ValueError(
            "MACE-POLAR density coefficients do not contain one monopole per atom."
        )
    return density[:n_atoms, 0].copy()


# Model name → filename mapping
_MACEPOL_MODEL_FILES = {
    'macepols': 'macepols.pt',
    'macepolm': 'macepolm.pt',
    'macepoll': 'macepoll.pt',
}


def _integer_info(atoms, key: str, default: int) -> int:
    value = atoms.info.get(key, default)
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"MACE-POLAR requires integer atoms.info['{key}']; got {value!r}.") from exc
    if not numeric_value.is_integer():
        raise ValueError(f"MACE-POLAR requires integer atoms.info['{key}']; got {value!r}.")
    return int(numeric_value)


# ------------------------ Calculator ------------------------

@register_calculator
class MACEPolCalculator(CalcABC):
    """ASE calculator for MACE-POLAR models (pure MLIP, f32).

    The traced model accepts flat tensor inputs and returns:
        (total_energy, node_energy, density_coefficients)

    Supports total_charge and total_spin via atoms.info['charge'] and atoms.info['mult'].
    """

    implemented_properties = ['energy', 'forces', 'free_energy', 'hessian', 'charges']

    MODEL_NAMES = ('macepols', 'macepolm', 'macepoll')
    MODEL_ENERGY_UNIT = 'eV'
    SUPPORTED_HESSIAN_MODES = ('analytic', 'numerical')
    SUPPORTS_CHARGE_MULT = True
    SUPPORTS_PBC = False
    CHECKPOINT_FILENAME = None
    REQUIRES_LOCAL_MODEL_FILE = True
    OPTION_KEYS = ()
    MODEL_PATH_OPTION = 'model_path'

    @classmethod
    def build_kwargs_from_options(cls, model, options, *, resolved_model_path=None):
        kwargs = {}
        if resolved_model_path is not None:
            kwargs['model_path'] = resolved_model_path
        return kwargs

    def __init__(self,
        device: torch.device,
        model: str = 'macepols',
        model_path: str = None,
        implicit: Literal['gbsa', 'none'] = 'none',
        solvent: str = 'none',
        ):
        """
        Args:
            device: Torch device.
            model: Model name ('macepols', 'macepolm', 'macepoll').
            model_path: Optional explicit path to .pt file (overrides model name lookup).
            implicit: Implicit solvent model type.
            solvent: Solvent type.
        """
        super().__init__()
        self.model_name = str(model).strip().lower()

        if model_path is None:
            model_dir = os.path.dirname(os.path.realpath(__file__))
            model_dir = os.path.dirname(model_dir)
            filename = _MACEPOL_MODEL_FILES.get(model, f'{model}.pt')
            model_path = os.path.join(model_dir, 'model', filename)

        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad_(False)

        self.device = device
        self.dtype = torch.float32  # MACE-POLAR traced models are f32
        self.r_max = float(self.model.r_max)
        self.atomic_numbers = [int(z) for z in self.model.atomic_numbers]
        self.hessian = 'analytic'

        self.implicit_solv_init(implicit=implicit, solvent=solvent)

    def _build_inputs(self, atoms, requires_grad=False):
        """Build the 12 flat tensor inputs for MACE-POLAR forward pass."""
        device = self.device
        dtype = self.dtype

        positions = torch.tensor(
            atoms.get_positions(), dtype=dtype, device=device,
            requires_grad=requires_grad,
        )
        Z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long, device=device)
        node_attrs = one_hot_node_attrs(Z, self.atomic_numbers, dtype=dtype)
        edge_index, shifts = radius_graph_no_pbc(positions, self.r_max)

        N = positions.size(0)
        unit_shifts = torch.zeros_like(shifts)
        batch = torch.zeros(N, dtype=torch.int64, device=device)
        ptr = torch.tensor([0, N], dtype=torch.int64, device=device)
        cell = torch.zeros(3, 3, dtype=dtype, device=device)

        # Charge and spin from atoms.info (default: 0, singlet)
        charge = float(atoms.info.get('charge', 0))
        mult = _integer_info(atoms, 'mult', 1)
        total_charge = torch.tensor([charge], dtype=dtype, device=device)
        # PolarMACE subtracts one internally; total_spin is the multiplicity.
        total_spin = torch.tensor([float(mult)], dtype=dtype, device=device)

        # No external field for pure MLIP
        external_field = torch.zeros(N, 3, dtype=dtype, device=device)
        local_or_ghost = torch.ones(N, dtype=dtype, device=device)

        return (positions, node_attrs, edge_index, shifts, unit_shifts,
                batch, ptr, cell, total_charge, total_spin,
                external_field, local_or_ghost)

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        """Main ASE calculation entry point."""
        properties = self._normalize_properties(properties)
        atoms = super().calculate(atoms, properties, system_changes)

        # Single forward; positions carry grad only when forces are requested.
        needs_forces = 'forces' in properties
        inputs = self._build_inputs(atoms, requires_grad=needs_forces)
        total_energy, _, density_coefficients = self.model(*inputs)
        energy_eV = total_energy.sum().double()

        forces_np = None
        if needs_forces:
            forces = -torch.autograd.grad(total_energy.sum(), inputs[0])[0]
            forces_np = forces.double().detach().cpu().numpy()

        hessian = None
        if 'hessian' in properties:
            if self.solvent_correction is not None:
                raise NotImplementedError('Hessian calculation with implicit solvent is not implemented yet.')
            hessian = self.get_hessian(atoms)

        self._finalize_results(atoms, energy=energy_eV.item(), forces=forces_np, hessian=hessian)
        if 'charges' in properties:
            self.results['charges'] = _macepol_atomic_charges(
                density_coefficients,
                len(atoms),
            )

    def _analytic_hessian(self, atoms) -> np.ndarray:
        """Analytic Hessian via autograd. Returns (3N, 3N) np.ndarray in Hartree/Å²."""
        from ..calculator_base import EV2HARTREE

        inputs = self._build_inputs(atoms, requires_grad=True)
        positions = inputs[0]

        def energy_fn():
            total_energy, _, _ = self.model(*inputs)
            return total_energy.sum() * EV2HARTREE

        return hessian_via_double_autograd(energy_fn, positions)
