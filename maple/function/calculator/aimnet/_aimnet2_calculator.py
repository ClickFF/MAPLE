from __future__ import annotations

from pathlib import Path
from typing import Dict, Literal, Optional

import numpy as np
import torch

from .. import _inputs, _torch
from ..calculator_base import CalcABC, register_calculator


@register_calculator
class AIMNet2Calculator(CalcABC):
    """ASE calculator for standalone AIMNet2 TorchScript models."""

    implemented_properties = ['energy', 'forces', 'free_energy', 'hessian', 'charges', 'dipole', 'stress', 'virial']
    MODEL_NAMES = (
        'aimnet2', 'aimnet2-nse', 'aimnet2-rxn', 'aimnet2-pd', 'aimnet2-b973c',
        'aimnet2-2025',
    )
    MODEL_ENERGY_UNIT = 'eV'
    SUPPORTED_HESSIAN_MODES = ('analytic', 'numerical')
    SUPPORTS_CHARGE_MULT = True
    SUPPORTS_PBC = True
    REQUIRES_LOCAL_MODEL_FILE = True
    OPTION_KEYS = ('lr', 'lr_cutoff', 'dtype')

    @classmethod
    def build_kwargs_from_options(cls, model, options, *, resolved_model_path=None):
        return {'dtype': options.get('dtype'),
                'lr': options.get('lr', 'auto'), 'lr_cutoff': float(options.get('lr_cutoff', 15.))}

    def __init__(self, device: torch.device | str,
                 model: str = 'aimnet2',
                 lr: str = 'auto',
                 lr_cutoff: float = 15.,
                 implicit: Literal['gbsa', 'none'] = 'none',
                 solvent: str = 'none',
                 dtype=None):
        super().__init__()
        path = Path(__file__).resolve().parents[1] / 'model' / (model + '.pt')
        _torch.load_model(self, path, device, dtype)
        self.hessian = 'analytic'
        self.model_name = model
        self.lr, self.lr_cutoff = str(lr).lower(), float(lr_cutoff)
        _inputs.validate_lr(self.lr, self.lr_cutoff)
        self.cutoff = float(self.model.cutoff)
        self.implicit_solv_init(implicit=implicit, solvent=solvent)

    def _set_lrcoulomb_method(self, method: str, cutoff: float = 15.):
        """Select the LR method; cutoff controls DSF only, in Angstrom."""
        _inputs.validate_lr(method, cutoff)
        if method != 'auto':
            self.model.configure_lr(method, float(cutoff))
        self.lr, self.lr_cutoff = method, float(cutoff)
        self.reset()

    def set(self, **kwargs):
        changed = super().set(**kwargs)
        if hasattr(self, 'lr') and ('lr' in kwargs or 'lr_cutoff' in kwargs):
            self._set_lrcoulomb_method(str(kwargs.get('lr', self.lr)).lower(),
                                      float(kwargs.get('lr_cutoff', self.lr_cutoff)))
        return changed

    def _build_data(self, coord: Optional[torch.Tensor], atoms) -> Dict[str, torch.Tensor]:
        """Prepare native dense or periodic-neighbor inputs for the supplied atoms."""
        members = [atoms] if hasattr(atoms, 'positions') else list(atoms)
        model = self.model
        has_coulomb = bool(getattr(model.long_range, 'has_coulomb', True))
        has_d3 = bool(getattr(model.long_range, 'has_d3', False))
        data = _inputs.prepare_aimnet(members, device=self.device, dtype=self.dtype,
                             supported_species=model.supported_species.tolist(),
                             charge_channels=int(model.channels), supports_charged=bool(model.supports_charged),
                             lr=self.lr if has_coulomb else 'auto', lr_cutoff=self.lr_cutoff,
                             cutoff=self.cutoff,
                             d3_cutoff=float(model.long_range.d3_cutoff) if has_d3 else None)
        if coord is not None:
            if data['coord'].ndim == 3:
                data['coord'] = coord.unsqueeze(0) if coord.ndim == 2 else coord
            else:
                data['coord'] = torch.cat((coord, coord.new_zeros((1, 3))))
        return data

    def _forward(self, data):
        method = self.lr
        if not bool(getattr(self.model.long_range, 'has_coulomb', True)):
            method = 'simple'
        elif method == 'auto':
            method = 'dsf' if 'cell' in data else 'simple'
        self.model.configure_lr(method, self.lr_cutoff)
        return self.model(data)

    def _forward_energy(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return complete model energy in eV with the current LR configuration."""
        with torch.jit.optimized_execution(False):
            return self._forward(data)['energy'].sum()

    def _calculate(self, atoms_list, properties, vector=None):
        data = self._build_data(None, atoms_list)
        # Keep the validated executor for forward and all derivative orders.
        with torch.set_grad_enabled(_torch.derivative_order(properties) > 0), torch.jit.optimized_execution(False):
            positions, live, cells, strain, volume = _torch.prepare_derivatives(
                data['coord'], atoms_list, properties)
            data['coord'] = live
            if cells is not None:
                data['cell'] = cells[0]
            output = self._forward(data)
            return _torch.tensor_properties(output['energy'], positions, atoms_list, properties,
                                            outputs=output, source_unit=self.MODEL_ENERGY_UNIT,
                                            strain=strain, volume=volume,
                                            single=vector is not None, vector=vector,
                                            dipole_from_charges=True)

    def _batch_key(self, atoms):
        return (bool(atoms.pbc.any()),)

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