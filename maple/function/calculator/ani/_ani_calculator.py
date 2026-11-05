from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from ase.calculators.calculator import all_changes

from .. import _inputs, _torch
from ..calculator_base import CalcABC, register_calculator, parse_bool_option


@register_calculator
class ANICalculator(CalcABC):
    """ASE calculator for standalone ANI TorchScript models and optional D4."""

    implemented_properties = ['energy', 'forces', 'free_energy', 'hessian', 'charges', 'dipole', 'stress', 'virial']
    MODEL_NAMES = (
        'ani1x', 'ani1ccx', 'ani2x', 'ani1xnr', 'ani2xr', 'ani2dr', 'snn-ani2xr',
        'ani-mbis',
    )
    MODEL_ENERGY_UNIT = 'hartree'
    SUPPORTED_HESSIAN_MODES = ('analytic', 'numerical')
    SUPPORTS_PBC = True
    SUPPORTS_CHARGE_MULT = False
    REQUIRES_LOCAL_MODEL_FILE = True
    OPTION_KEYS = ('d4', 'dtype')

    @classmethod
    def build_kwargs_from_options(cls, model, options, *, resolved_model_path=None):
        return {'dtype': options.get('dtype'),
                'd4': parse_bool_option(options.get('d4', False), name='d4')}

    def __init__(self, device,
                 model: str = 'ani2x',
                 overwrite=False,
                 d4=False,
                 implicit: str = 'none',
                 solvent: str = 'none',
                 dtype=None):
        super().__init__()
        path = Path(__file__).resolve().parents[1] / 'model' / (model + '.pt')
        _torch.load_model(self, path, device, dtype)
        charges = getattr(self.model, 'supports_atomic_charges', None)
        if not isinstance(charges, bool):
            raise ValueError('ANI artifact requires boolean supports_atomic_charges; install a qualified export')
        self.implemented_properties = list(type(self).implemented_properties)
        if not charges:
            self.implemented_properties = [p for p in self.implemented_properties if p not in ('charges', 'dipole')]
        self.hessian = 'analytic'
        self.model_name, self.overwrite, self.d4 = model, overwrite, d4
        if d4 and getattr(self.model, 'model_name', model) == 'ani2dr':
            raise ValueError('ani2dr already contains native dispersion; extra D4 is unsupported')
        self.implicit_solv_init(implicit=implicit, solvent=solvent)

    def _validate_dispersion(self, atoms_list):
        if self.d4 and any(atoms.pbc.any() for atoms in atoms_list):
            raise ValueError('External D4 has no periodic cell/PBC implementation')

    def _forward(self, data):
        output = self.model((data['numbers'], data['positions']),
                            cell=data['cell'][0] if 'cell' in data else None,
                            pbc=data['pbc'][0] if 'pbc' in data else None)
        result = {'energy': output.energies}
        charges = getattr(output, 'atomic_charges', None)
        if charges is not None:
            result['charges'] = charges
        if self.d4:
            if len(data['positions']) != 1:
                raise NotImplementedError('External D4 uses single-structure composition')
            result['energy'] = result['energy'] + self.dftd4(data['numbers'], data['positions'])
        return result

    def _forward_energy(self, atoms, coordinates):
        self._validate_dispersion([atoms])
        data = _inputs.prepare_ani([atoms], device=self.device, dtype=self.dtype,
                                  supported_elements=self.model.atomic_numbers.tolist())
        data['positions'] = coordinates
        return self._forward(data)['energy']

    def dftd4(self, species, coordinates):
        import tad_dftd4 as d4
        charge = coordinates.new_tensor(0.)
        param = {
            's6': coordinates.new_tensor(1.0),
            's8': coordinates.new_tensor(0.34783580),
            's9': coordinates.new_tensor(1.0),
            'a1': coordinates.new_tensor(0.57488291),
            'a2': coordinates.new_tensor(6.41921802),
        }
        bohr_coords = coordinates[0] * 1.8897261245864
        return torch.sum(d4.dftd4(species[0], bohr_coords, charge, param))

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        return _torch.calculate_single(self, atoms, properties, system_changes)

    def _calculate(self, atoms_list, properties, vector=None):
        self._validate_dispersion(atoms_list)
        data = _inputs.prepare_ani(atoms_list, device=self.device, dtype=self.dtype,
                                  supported_elements=self.model.atomic_numbers.tolist())
        if 'cell' in data and any(
                not np.array_equal(a.cell.array, atoms_list[0].cell.array)
                or not np.array_equal(a.pbc, atoms_list[0].pbc) for a in atoms_list[1:]):
            raise NotImplementedError('ANI periodic native batches require the same cell and PBC mask')
        order = _torch.derivative_order(properties)
        # CUDA fusion can emit unsupported allocations in repeated force/Hessian backward graphs.
        context = (torch.jit.optimized_execution(False)
                   if order and self.device.type == 'cuda' else nullcontext())
        with context, torch.set_grad_enabled(order > 0):
            positions, live, cells, strain, volume = _torch.prepare_derivatives(
                data['positions'], atoms_list, properties)
            data['positions'] = live
            if cells is not None:
                data['cell'] = cells
            output = self._forward(data)
            return _torch.tensor_properties(output['energy'], positions, atoms_list, properties,
                                            outputs=output, source_unit=self.MODEL_ENERGY_UNIT,
                                            strain=strain, volume=volume,
                                            single=vector is not None, vector=vector,
                                            dipole_from_charges=True)

    def _batch_key(self, atoms):
        if atoms.pbc.any():
            return (tuple(atoms.pbc), tuple(atoms.cell.array.flatten()))
        return ()

    def _calculate_batch(self, atoms_list, properties, vector=None):
        if self.d4:
            return NotImplemented
        if not self.model.supports_native_batch:
            return NotImplemented
        if (any(atoms.pbc.any() for atoms in atoms_list)
                and not self.model.supports_native_pbc_batch):
            return NotImplemented
        return self._calculate(atoms_list, properties, vector=vector)

    def _analytic_hessian(self, atoms) -> np.ndarray:
        return self._calculate([atoms], ['hessian'])['hessian'][0]

    def get_hvp(self, atoms, n: np.ndarray):
        self._reject_unsupported_pbc(atoms)
        if getattr(self, 'solvent_correction', None) is not None:
            return super().get_hvp(atoms, n)
        return self._calculate([atoms], ['hvp'], vector=n)['hvp']