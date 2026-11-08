from pathlib import Path

import numpy as np
import tensorflow as tf
from ase.neighborlist import neighbor_list

from .. import _inputs, _tensorflow
from ..calculator_base import CalcABC, _convert_properties, register_calculator


@register_calculator
class GRACECalculator(CalcABC):
    """ASE calculator for the complete, native GRACE-OFF GraphDefs."""

    MODEL_NAMES = (
        'grace-off-1s', 'grace-off-1m', 'grace-off-1l',
        'grace-off-2s', 'grace-off-2m', 'grace-off-2l',
    )
    implemented_properties = ['energy', 'free_energy', 'forces', 'hessian', 'stress', 'virial']
    MODEL_ENERGY_UNIT = 'eV'
    SUPPORTED_HESSIAN_MODES = ('analytic', 'numerical')
    SUPPORTS_PBC = True
    LOCAL_MODEL_FILENAMES = {name: name + '.pb' for name in MODEL_NAMES}
    OPTION_KEYS = ('dtype',)

    @classmethod
    def build_kwargs_from_options(cls, model, model_options, *, resolved_model_path=None):
        return {'dtype': model_options.get('dtype')}

    def __init__(self, device=None, model='grace-off-2m',
                 implicit='none', solvent='none', dtype=None):
        super().__init__()
        if model not in self.MODEL_NAMES:
            raise ValueError(f'Unknown GRACE model: {model!r}')
        self.device = str(device or 'cpu')
        self.model_path = str(Path(__file__).resolve().parents[1] / 'model' / self.LOCAL_MODEL_FILENAMES[model])
        self.model = _tensorflow.load_model(self.model_path, self.device)
        self.dtype = self.model.graph.get_tensor_by_name('bond_vector:0').dtype
        if dtype is not None and tf.as_dtype(dtype) != self.dtype:
            raise ValueError(f'Export supports dtype={self.dtype.name}, requested {dtype}')
        self.atomic_numbers = np.asarray(tf.get_static_value(self.model.graph.get_tensor_by_name('atomic_numbers:0')))
        self.cutoff = float(tf.get_static_value(self.model.graph.get_tensor_by_name('r_max:0')))
        self._capacity = {}
        self.hessian = 'analytic'
        self._load_export()
        self.implicit_solv_init(implicit=implicit, solvent=solvent)

    def _load_export(self):
        # These ports are shared by all six qualified GraphDefs.
        inputs = {name: name + ':0' for name in (
            'atomic_mu_i', 'mu_i', 'mu_j', 'ind_i', 'ind_j', 'bond_vector',
            'batch_tot_nat', 'batch_tot_nat_real', 'atoms_to_structure_map',
            'batch_total_num_structures', 'n_struct_total',
        )}
        energy = _tensorflow.compile_model(self.model, inputs, {'energy': 'Identity_1:0'}, device=self.device)
        joint = _tensorflow.compile_model(self.model, inputs,
                                         {'energy': 'Identity_1:0', 'forces': 'Identity_3:0'}, device=self.device)

        def geometry(forward):
            @tf.function(jit_compile=True, autograph=False)
            def call(data):
                members = tf.gather(data['atoms_to_structure_map'], data['ind_i'])
                shifts = tf.einsum('ei,eij->ej', tf.cast(data['images'], self.dtype),
                                   tf.gather(data['cell'], members))
                vectors = tf.gather(data['positions'], data['ind_j']) - tf.gather(data['positions'], data['ind_i']) + shifts
                real = data['ind_i'] < data['batch_tot_nat_real']
                values = {**data, 'bond_vector': tf.where(real[:, None], vectors, data['bond_vector'])}
                with tf.device(_tensorflow._device(self.device)):
                    return forward({name: values[name] for name in inputs})
            return call

        self._energy_call = geometry(energy)
        self._call = geometry(joint)
        self._hvp = _tensorflow.compile_hvp(self._call, position_key='positions', force_key='forces')
        self._strain_result = _tensorflow.compile_strain(
            self._call, position_key='positions', cell_key='cell',
            energy_key='energy', batch_key='atoms_to_structure_map',
        )
        self._energy_strain_result = _tensorflow.compile_strain(
            self._energy_call, position_key='positions', cell_key='cell',
            energy_key='energy', batch_key='atoms_to_structure_map',
        )

    def _build_inputs(self, atoms_list):
        _inputs._validate_geometry(atoms_list)
        positions, cells, types, senders, receivers, images, members = [], [], [], [], [], [], []
        table = {int(z): i for i, z in enumerate(self.atomic_numbers)}
        offset = 0
        for b, atoms in enumerate(atoms_list):
            charge, mult = _inputs._state(atoms)
            if charge != 0 or mult != 1:
                raise ValueError('GRACE-OFF has no charge/multiplicity conditioning; use charge=0 and mult=1')
            _inputs._elements(atoms, self.atomic_numbers)
            i, j, shifts = neighbor_list('ijS', atoms, self.cutoff, self_interaction=False)
            positions.extend(atoms.positions)
            cells.append(atoms.cell.array)
            types.extend(table[int(z)] for z in atoms.numbers)
            senders.extend(i + offset)
            receivers.extend(j + offset)
            images.extend(shifts)
            members.extend([b] * len(atoms))
            offset += len(atoms)
        n_atoms, n_edges, n_struct = offset, len(senders), len(atoms_list)
        key = (n_struct, any(atoms.pbc.any() for atoms in atoms_list))
        max_atoms, max_edges = self._capacity.get(key, (0, 0))
        if n_atoms >= max_atoms:
            max_atoms = int(1.05 * n_atoms) + 1
        if n_edges >= max_edges:
            max_edges = int(1.05 * n_edges) + 1
        self._capacity[key] = max_atoms, max_edges
        types.extend([types[0]] * (max_atoms - n_atoms))
        positions.extend([np.zeros(3)] * (max_atoms - n_atoms))
        cells.append(np.eye(3))
        members.extend([n_struct] * (max_atoms - n_atoms))
        senders.extend([max_atoms - 1] * (max_edges - n_edges))
        receivers.extend([max_atoms - 1] * (max_edges - n_edges))
        images.extend([np.zeros(3, dtype=int)] * (max_edges - n_edges))
        integers = {'atomic_mu_i': types, 'ind_i': senders, 'ind_j': receivers,
                    'images': np.asarray(images).reshape(-1, 3), 'atoms_to_structure_map': members,
                    'batch_tot_nat': max_atoms, 'batch_tot_nat_real': n_atoms,
                    'batch_total_num_structures': n_struct, 'n_struct_total': n_struct + 1}
        with tf.device(_tensorflow._device(self.device)):
            data = {name: tf.convert_to_tensor(value, tf.int32) for name, value in integers.items()}
            data['positions'] = tf.convert_to_tensor(positions, self.dtype)
            data['cell'] = tf.convert_to_tensor(np.asarray(cells), self.dtype)
            data['mu_i'] = tf.gather(data['atomic_mu_i'], data['ind_i'])
            data['mu_j'] = tf.gather(data['atomic_mu_i'], data['ind_j'])
            data['bond_vector'] = tf.fill([max_edges, 3], tf.cast(52., self.dtype))
        return data

    def _calculate(self, atoms_list, properties):
        needs_hessian = 'hessian' in properties
        properties = [name for name in properties if name != 'hessian']
        if needs_hessian and not properties:
            return {'hessian': [self.get_hessian(atoms) for atoms in atoms_list]}
        data = self._build_inputs(atoms_list)
        derivative, volume = None, None
        if set(properties) & {'stress', 'virial'}:
            if not all(atoms.pbc.any() for atoms in atoms_list):
                raise ValueError('Stress/virial requires periodic members')
            call = self._strain_result if 'forces' in properties else self._energy_strain_result
            (_, output), derivative = call(data, tf.zeros_like(data['cell']))
            if 'stress' in properties:
                volume = [atoms.get_volume() for atoms in atoms_list]
        else:
            output = (self._call if 'forces' in properties else self._energy_call)(data)
        result = _tensorflow.array_properties(
            output['energy'], atoms_list, properties, forces=output.get('forces'),
            strain_gradient=derivative, volume=volume, source_unit=self.MODEL_ENERGY_UNIT,
        )
        if needs_hessian:
            result['hessian'] = [self.get_hessian(atoms) for atoms in atoms_list]
        return result

    def _calculate_batch(self, atoms_list, properties):
        if not bool(tf.get_static_value(self.model.graph.get_tensor_by_name('supports_native_batch:0'))):
            return NotImplemented
        if (any(atoms.pbc.any() for atoms in atoms_list) and not bool(tf.get_static_value(
                self.model.graph.get_tensor_by_name('supports_native_pbc_batch:0')))):
            return NotImplemented
        return self._calculate(atoms_list, properties)

    def _analytic_hessian(self, atoms):
        data = self._build_inputs([atoms])
        matrix = _tensorflow.hessian(self._hvp, data, data['positions'], len(atoms), self.device)
        return _convert_properties({'hessian': matrix}, source_unit=self.MODEL_ENERGY_UNIT)['hessian']

    def get_hvp(self, atoms, n):
        if getattr(self, 'solvent_correction', None) is not None:
            return super().get_hvp(atoms, n)
        data = self._build_inputs([atoms])
        vector = n.detach().cpu().numpy() if hasattr(n, 'detach') else np.asarray(n)
        if vector.shape not in ((len(atoms) * 3,), (len(atoms), 3)) or not np.isfinite(vector).all():
            raise ValueError('HVP vector must have shape [N,3] or [3N] and finite values')
        packed = np.zeros(data['positions'].shape, self.dtype.as_numpy_dtype)
        packed[:len(atoms)] = vector.reshape(-1, 3)
        with tf.device(_tensorflow._device(self.device)):
            product, output = self._hvp(data, tf.convert_to_tensor(packed))
        result = _convert_properties({'hvp': (
            np.asarray(product)[:len(atoms)].reshape(-1),
            np.asarray(output['forces'])[:len(atoms)].reshape(-1),
            float(np.asarray(output['energy'])[0]),
        )}, source_unit=self.MODEL_ENERGY_UNIT)
        import torch
        return tuple(torch.as_tensor(value, dtype=torch.float64, device=self.device) for value in result['hvp'])