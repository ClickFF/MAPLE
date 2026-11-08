from pathlib import Path
from typing import Literal

import numpy as np
import torch

from .. import _inputs, _torch
from ..calculator_base import CalcABC, register_calculator


def build_inputs_from_atoms(atoms, model, device='cpu', positions=None, dtype=None):
    members = [atoms] if hasattr(atoms, 'positions') else list(atoms)
    native = getattr(model, 'model', model)
    dtype = dtype or next(model.parameters()).dtype
    data = _inputs.prepare_polar(members, native.atomic_numbers.tolist(), float(native.r_max), device=device, dtype=dtype)
    if positions is not None:
        data['positions'] = positions
    return data


@register_calculator
class MACEPolCalculator(CalcABC):
    """ASE calculator for native MACE-POLAR graph outputs and complete-energy derivatives.

    Charge and multiplicity are read from atoms.info; periodic and external-field
    inputs are prepared by the calculator.
    """

    implemented_properties = ['energy', 'forces', 'free_energy', 'hessian', 'charges', 'dipole', 'stress', 'virial']
    MODEL_NAMES = (
        'mace-polar-1s', 'mace-polar-1m', 'mace-polar-1l',
    )
    MODEL_ENERGY_UNIT = 'eV'
    SUPPORTED_HESSIAN_MODES = ('analytic', 'numerical')
    SUPPORTS_CHARGE_MULT = True
    SUPPORTS_PBC = True
    REQUIRES_LOCAL_MODEL_FILE = True
    OPTION_KEYS = ('dtype',)

    @classmethod
    def build_kwargs_from_options(cls, model, options, *, resolved_model_path=None):
        return {'dtype': options.get('dtype')}

    def __init__(self, device: torch.device | str = 'cpu',
                 model: str = 'mace-polar-1s',
                 implicit: Literal['gbsa', 'none'] = 'none',
                 solvent: str = 'none',
                 dtype=None):
        """Load the registered model at the requested precision."""
        super().__init__()
        path = Path(__file__).resolve().parents[1] / 'model' / (model + '.pt')
        native_dtype = _torch.load_model(self, path, device, dtype)
        self.hessian = 'analytic'
        if self.dtype != native_dtype:
            self.model.to(dtype=self.dtype)
        self.model_name = model
        self.implicit_solv_init(implicit=implicit, solvent=solvent)

    def _build_inputs(self, atoms, requires_grad=False):
        data = build_inputs_from_atoms(atoms, self.model, device=self.device, dtype=self.dtype)
        data['positions'].requires_grad_(requires_grad)
        return data

    def _calculate(self, atoms_list, properties, vector=None):
        data = build_inputs_from_atoms(atoms_list, self.model, device=self.device, dtype=self.dtype)
        order = _torch.derivative_order(properties)
        with torch.set_grad_enabled(order > 0), torch.jit.optimized_execution(False):
            positions, live, cells, strain, volume = _torch.prepare_derivatives(
                data['positions'], atoms_list, properties, batch=data['batch'])
            data['positions'] = live
            if cells is not None:
                data['cell'] = cells
                data['shifts'] = _inputs.graph_shifts(data)
            first_order_force = 'forces' in properties and order < 2 and strain is None
            output = self.model(data, training=order > 0 and not first_order_force,
                                compute_force='forces' in properties and not first_order_force,
                                compute_virials=False, compute_stress=False,
                                compute_displacement=False, compute_hessian=False)
            # Use complete-energy strain AD, including reciprocal cell and volume.
            output = {**output, 'stress': None, 'virial': None}
            return _torch.tensor_properties(output['energy'], positions, atoms_list, properties,
                                            outputs=output, source_unit=self.MODEL_ENERGY_UNIT,
                                            strain=strain, volume=volume,
                                            single=vector is not None, vector=vector,
                                            dipole_from_charges=True)

    def _batch_key(self, atoms):
        return tuple(atoms.pbc)

    def _calculate_batch(self, atoms_list, properties, vector=None):
        if not self.model.supports_native_batch:
            return NotImplemented
        if (any(atoms.pbc.any() for atoms in atoms_list)
                and not self.model.supports_native_pbc_batch):
            return NotImplemented
        return self._calculate(atoms_list, properties, vector=vector)

    def _analytic_hessian(self, atoms) -> np.ndarray:
        return self._calculate([atoms], ['hessian'])['hessian'][0]

    def get_hvp(self, atoms, n):
        self._reject_unsupported_pbc(atoms)
        if getattr(self, 'solvent_correction', None) is not None:
            return super().get_hvp(atoms, n)
        return self._calculate([atoms], ['hvp'], vector=n)['hvp']