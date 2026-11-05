from pathlib import Path
from typing import Literal

import numpy as np
import torch
from ase.calculators.calculator import all_changes

from .. import _inputs, _torch
from ..calculator_base import CalcABC, register_calculator


@register_calculator
class NequIPCalculator(CalcABC):
    """ASE calculator for standalone NequIP and Allegro TorchScript models."""

    implemented_properties = ['energy', 'forces', 'free_energy', 'hessian', 'stress', 'virial']
    MODEL_NAMES = (
        'allegro-fm-256', 'allegro-fm-512', 'nequip-t3-cosmo',
        'nequip-t123-cosmo', 'nequip-t3-gas',
    )
    MODEL_ENERGY_UNIT = 'eV'
    SUPPORTED_HESSIAN_MODES = ('analytic', 'numerical')
    SUPPORTS_CHARGE_MULT = False
    SUPPORTS_PBC = True
    REQUIRES_LOCAL_MODEL_FILE = True
    OPTION_KEYS = ('dtype',)

    @classmethod
    def build_kwargs_from_options(cls, model, options, *, resolved_model_path=None):
        return {'dtype': options.get('dtype')}

    def __init__(self, device: torch.device | str = 'cpu',
                 model: str = 'allegro-fm-256',
                 implicit: Literal['gbsa', 'none'] = 'none',
                 solvent: str = 'none',
                 dtype=None):
        super().__init__()
        if dtype not in (None, 'float32', torch.float32):
            raise ValueError('NequIP/Allegro inference inputs require FP32 precision')
        path = Path(__file__).resolve().parents[1] / 'model' / (model + '.pt')
        _torch.load_model(self, path, device)
        # FP32 inputs retain the FM graph's native FP64 radial parameters/buffers.
        self.dtype = torch.float32
        self.hessian = 'analytic'
        self.implicit_solv_init(implicit=implicit, solvent=solvent)

    def _build_inputs(self, atoms_list):
        return _inputs.prepare_nequip(
            atoms_list, _inputs._species(self.model.atomic_numbers), float(self.model.r_max),
            device=self.device, dtype=self.dtype, input_fields=self.model.input_fields,
        )

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        return _torch.calculate_single(self, atoms, properties, system_changes)

    def _calculate(self, atoms_list, properties, vector=None):
        data = self._build_inputs(atoms_list)
        if set(properties) & {'stress', 'virial'}:
            if not all(atoms.pbc.any() for atoms in atoms_list):
                raise ValueError('stress and virial require periodic systems')
            if bool((torch.linalg.det(data['cell']).abs() <= 0).any()):
                raise ValueError('stress requires nonsingular cells')
        order = _torch.derivative_order(properties)
        with torch.set_grad_enabled(order > 0):
            positions = data['pos'].detach().requires_grad_(order > 0)
            data['pos'] = positions
            output = self.model(
                data, training=order > 1,
                compute_force='forces' in properties or order > 1,
                compute_virials='virial' in properties,
                compute_stress='stress' in properties,
            )
            return _torch.tensor_properties(
                output['total_energy'], positions, atoms_list, properties,
                outputs=output, source_unit=self.MODEL_ENERGY_UNIT,
                single=vector is not None, vector=vector,
            )

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