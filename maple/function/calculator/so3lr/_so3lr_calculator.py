from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from ase.neighborlist import neighbor_list

from . import _model
from .. import _inputs, _jax
from ..calculator_base import CalcABC, _convert_properties, register_calculator


@register_calculator
class SO3LRCalculator(CalcABC):
    """SO3LR standard JAX export plus its native complete long-range energy."""

    MODEL_NAMES = ('so3lr', 'so3lr-tauri-s', 'so3lr-tauri-m', 'so3lr-tauri-l')
    implemented_properties = ['energy', 'free_energy', 'forces', 'hessian', 'charges', 'dipole', 'stress', 'virial']
    MODEL_ENERGY_UNIT = 'eV'
    SUPPORTED_HESSIAN_MODES = ('analytic', 'numerical')
    SUPPORTS_CHARGE_MULT = True
    SUPPORTS_PBC = True
    LOCAL_MODEL_FILENAMES = {name: name + '.jax' for name in MODEL_NAMES}
    OPTION_KEYS = ('dtype', 'lr')

    @classmethod
    def build_kwargs_from_options(cls, model, model_options, *, resolved_model_path=None):
        return {'dtype': model_options.get('dtype'),
                'lr': model_options.get('lr', 'auto')}

    def __init__(self, device=None, model='so3lr',
                 implicit='none', solvent='none', dtype=None, lr='auto'):
        super().__init__()
        if model not in self.MODEL_NAMES:
            raise ValueError(f'Unknown SO3LR model: {model!r}')
        self.device = str(device or 'cpu')
        self.model_path = str(Path(__file__).resolve().parents[1] / 'model' / self.LOCAL_MODEL_FILENAMES[model])
        self.model, self.jax_device = _jax.load_model(self.model_path, self.device, require_vjp=2)

        template = self.model.in_tree.unflatten(self.model.in_avals)[0][0]
        self.dtype = template['positions'].dtype
        self.metadata = _jax.read_metadata(self.model, family='so3lr', model_id=model)
        self.cutoff = float(self.metadata['cutoff'])
        self.head_index = self.metadata['selected_head_index']
        self.num_theory_heads = self.metadata['num_theory_heads']
        self.atomic_numbers = np.asarray(self.metadata['atomic_numbers'])
        if self.metadata['dtype'] != str(self.dtype):
            raise ValueError('SO3LR metadata dtype conflicts with its input signature')
        if isinstance(self.metadata['cutoff'], bool) or not np.isfinite(self.cutoff) or self.cutoff <= 0:
            raise ValueError('SO3LR cutoff must be finite and positive')
        if (type(self.head_index) is not int or type(self.num_theory_heads) is not int
                or not 0 <= self.head_index < self.num_theory_heads
                or self.num_theory_heads != template['theory_mask'].shape[1]):
            raise ValueError('SO3LR head selection conflicts with theory_mask')
        if self.metadata.get('spin_convention') != 'multiplicity_minus_one':
            raise ValueError('Unsupported SO3LR spin convention')
        if dtype is not None and np.dtype(dtype) != self.dtype:
            raise ValueError(f'Export supports dtype={self.dtype}, requested {dtype}')

        self.lr = str(lr).lower()
        self._validate_lr()
        self._capacity = {}
        self._compiled = {}
        self.hessian = 'analytic'
        self.implicit_solv_init(implicit=implicit, solvent=solvent)

    def _validate_lr(self):
        if self.lr not in ('auto', 'cutoff', 'ewald', 'pme'):
            raise ValueError('SO3LR lr must be auto, cutoff, ewald or pme')

    def set(self, **kwargs):
        if 'lr' in kwargs:
            lr = str(kwargs['lr']).lower()
            if lr not in ('auto', 'cutoff', 'ewald', 'pme'):
                raise ValueError('SO3LR lr must be auto, cutoff, ewald or pme')
            self.lr = lr
        return super().set(**kwargs)

    def _build_inputs(self, atoms_list):
        _inputs._validate_geometry(atoms_list, full_pbc=True)
        periodic = any(atoms.pbc.any() for atoms in atoms_list)
        self._validate_lr()
        if self.lr in ('ewald', 'pme') and not periodic:
            raise ValueError('SO3LR Ewald/PME requires full three-dimensional periodicity')
        arrays = {key: [] for key in ('positions', 'atomic_numbers', 'batch_segments', 'cell',
                                     'idx_i', 'idx_j', 'images', 'idx_i_lr', 'idx_j_lr', 'images_lr')}
        charges, unpaired = [], []
        cutoff = 12. if periodic else 1000.
        offset = 0
        for b, atoms in enumerate(atoms_list):
            charge, mult = _inputs._state(atoms)
            if not np.isin(atoms.numbers, self.atomic_numbers).all():
                raise ValueError(f'SO3LR model supports atomic numbers 1..{self.atomic_numbers[-1]}')
            charges.append(charge)
            unpaired.append(mult - 1.)
            arrays['positions'].extend(atoms.positions)
            arrays['atomic_numbers'].extend(atoms.numbers)
            arrays['batch_segments'].extend([b] * len(atoms))
            arrays['cell'].append(atoms.cell.array if atoms.cell.rank else np.eye(3))
            for radius, suffix in ((self.cutoff, ''), (cutoff, '_lr')):
                i, j, shifts = neighbor_list('ijS', atoms, radius, self_interaction=False)
                arrays['idx_i' + suffix].extend(i + offset)
                arrays['idx_j' + suffix].extend(j + offset)
                arrays['images' + suffix].extend(shifts)
            offset += len(atoms)
        graphs = len(atoms_list)
        sizes = (offset, len(arrays['idx_i']), len(arrays['idx_i_lr']))
        key = graphs, periodic
        capacity = self._capacity.get(key, (0, 0, 0))
        max_atoms, max_edges, max_lr = tuple(int(1.05 * size) + 1 if size >= old else old
                                            for size, old in zip(sizes, capacity))
        self._capacity[key] = max_atoms, max_edges, max_lr
        arrays['positions'].extend([[0., 0., 0.]] * (max_atoms - offset))
        arrays['atomic_numbers'].extend([0] * (max_atoms - offset))
        arrays['batch_segments'].extend([graphs] * (max_atoms - offset))
        arrays['cell'].append(np.eye(3))
        for limit, suffix in ((max_edges, ''), (max_lr, '_lr')):
            padding = limit - len(arrays['idx_i' + suffix])
            arrays['idx_i' + suffix].extend([max_atoms - 1] * padding)
            arrays['idx_j' + suffix].extend([max_atoms - 1] * padding)
            arrays['images' + suffix].extend([[0, 0, 0]] * padding)
        data = {name: np.asarray(value, dtype=self.dtype if name in ('positions', 'cell') else np.int32)
                for name, value in arrays.items()}
        data.update(
            node_mask=np.arange(max_atoms) < offset, graph_mask=np.arange(graphs + 1) < graphs,
            total_charge=np.asarray(charges + [0.], self.dtype),
            num_unpaired_electrons=np.asarray(unpaired + [0.], self.dtype),
            theory_mask=np.tile(np.arange(self.num_theory_heads) == self.head_index, (graphs + 1, 1)),
            lr_cutoff=np.asarray(cutoff, self.dtype), use_kspace=np.asarray(False),
            pbc=atoms_list[0].pbc.copy(),
        )
        return jax.device_put(data, self.jax_device)

    def _functions(self, data, properties=()):
        periodic = bool(np.asarray(data['pbc']).any())
        method = ('pme' if periodic else 'cutoff') if self.lr == 'auto' else self.lr
        shape = () if method == 'cutoff' else _model.grid_shape(np.asarray(data['cell'])[0], method)
        fields = tuple(field for name, field in (('charges', 'partial_charges'), ('dipole', 'dipole_vec'))
                       if name in properties)
        key = method, shape, 12. if periodic else 1000., fields
        if key not in self._compiled:
            complete = partial(_model.forward, self.model, method=method, shape=shape, cutoff=key[2])

            def forward(data):
                output = complete(data)
                return {name: output[name] for name in ('energy', *fields)}

            forces = _jax.compile_forces(forward, position_key='positions', energy_key='energy')
            self._compiled[key] = {
                'energy': jax.jit(lambda data: forward(data)['energy']),
                'output': jax.jit(forward),
                'forces': forces,
                'hvp': _jax.compile_hvp(forces, position_key='positions', force_key='forces'),
                'strain': _jax.compile_strain(forward, position_key='positions', cell_key='cell',
                                              energy_key='energy', batch_key='batch_segments'),
                'strain_forces': _jax.compile_strain(forces, position_key='positions', cell_key='cell',
                                                     energy_key='energy', batch_key='batch_segments'),
            }
        return self._compiled[key]

    def _calculate(self, atoms_list, properties):
        needs_hessian = 'hessian' in properties
        properties = [name for name in properties if name != 'hessian']
        if needs_hessian and not properties:
            return {'hessian': [self.get_hessian(atoms) for atoms in atoms_list]}
        data = self._build_inputs(atoms_list)
        calls = self._functions(data, properties)
        derivative, volume = None, None
        if set(properties) & {'stress', 'virial'}:
            if not all(atoms.pbc.all() for atoms in atoms_list):
                raise ValueError('Stress/virial requires periodic members')
            call = calls['strain_forces' if 'forces' in properties else 'strain']
            (_, output), derivative = call(data, jnp.zeros_like(data['cell']))
            if 'stress' in properties:
                volume = [atoms.get_volume() for atoms in atoms_list]
        elif 'forces' in properties:
            output = calls['forces'](data)
        elif set(properties) & {'charges', 'dipole'}:
            output = calls['output'](data)
        else:
            output = {'energy': calls['energy'](data)}
        result = _jax.array_properties(
            output['energy'], atoms_list, properties,
            forces=output.get('forces'), charges=output.get('partial_charges'),
            dipole=output.get('dipole_vec'), strain_gradient=derivative, volume=volume,
            source_unit=self.MODEL_ENERGY_UNIT,
        )
        if needs_hessian:
            result['hessian'] = [self.get_hessian(atoms) for atoms in atoms_list]
        return result

    def _batch_key(self, atoms):
        periodic = bool(atoms.pbc.any())
        method = ('pme' if periodic else 'cutoff') if self.lr == 'auto' else self.lr
        shape = _model.grid_shape(atoms.cell.array, method) if periodic and method != 'cutoff' else ()
        return periodic, shape

    def _calculate_batch(self, atoms_list, properties):
        return self._calculate(atoms_list, properties)

    def _analytic_hessian(self, atoms):
        data = self._build_inputs([atoms])
        matrix = _jax.hessian(self._functions(data)['hvp'], data, data['positions'], len(atoms), self.jax_device)
        return _convert_properties({'hessian': matrix.astype(float)}, source_unit=self.MODEL_ENERGY_UNIT)['hessian']

    def get_hvp(self, atoms, n):
        if getattr(self, 'solvent_correction', None) is not None:
            return super().get_hvp(atoms, n)
        data = self._build_inputs([atoms])
        vector = n.detach().cpu().numpy() if hasattr(n, 'detach') else np.asarray(n)
        if vector.shape not in ((len(atoms) * 3,), (len(atoms), 3)) or not np.isfinite(vector).all():
            raise ValueError('HVP vector must have shape [N,3] or [3N] and finite values')
        packed = np.zeros(data['positions'].shape, self.dtype)
        packed[:len(atoms)] = vector.reshape(-1, 3)
        product, output = self._functions(data)['hvp'](data, jax.device_put(packed, self.jax_device))
        result = _convert_properties({'hvp': (
            np.asarray(product, dtype=float)[:len(atoms)].reshape(-1),
            np.asarray(output['forces'], dtype=float)[:len(atoms)].reshape(-1),
            float(np.asarray(output['energy'])[0]),
        )}, source_unit=self.MODEL_ENERGY_UNIT)
        import torch
        return tuple(torch.as_tensor(value, dtype=torch.float64, device=self.device) for value in result['hvp'])