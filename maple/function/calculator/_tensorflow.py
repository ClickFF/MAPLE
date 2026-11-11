"""Frozen TensorFlow graphs, compiled derivatives and public property shapes."""

from pathlib import Path

import numpy as np
import tensorflow as tf

from .calculator_base import _convert_properties


def _device(device):
    name = str(device or 'cpu')
    platform, _, index = name.partition(':')
    if platform not in ('cpu', 'cuda'):
        raise ValueError(f'Unsupported TensorFlow device: {device!r}')
    index = int(index or 0)
    devices = tf.config.list_physical_devices('GPU' if platform == 'cuda' else 'CPU')
    if index < 0 or index >= len(devices):
        raise ValueError(f'TensorFlow device is unavailable: {device!r}')
    return f'/GPU:{index}' if platform == 'cuda' else f'/CPU:{index}'


def load_model(model_path, device='cpu'):
    _device(device)
    graph = tf.compat.v1.GraphDef()
    graph.ParseFromString(Path(model_path).read_bytes())
    model = tf.compat.v1.wrap_function(lambda: tf.graph_util.import_graph_def(graph, name=''), [])
    for name, dtype, rank in (('model_name', tf.string, 0), ('atomic_numbers', tf.int32, 1),
                              ('r_max', None, 0), ('supports_native_batch', tf.bool, 0),
                              ('supports_native_pbc_batch', tf.bool, 0)):
        try:
            value = model.graph.get_tensor_by_name(name + ':0')
        except KeyError as error:
            raise ValueError(f'TensorFlow artifact requires field {name!r}: {model_path}') from error
        if (value.op.type != 'Const' or value.shape.rank != rank
                or (dtype is not None and value.dtype != dtype)
                or (name == 'r_max' and not value.dtype.is_floating)):
            raise ValueError(f'TensorFlow artifact has invalid field {name!r}: {model_path}')
    return model


def compile_model(model, inputs, outputs, *, device='cpu'):
    """Bind native fields once; unused outputs stay outside the compiled request."""
    placement = _device(device)
    tensors = [model.graph.get_tensor_by_name(name) for name in inputs.values()]
    pruned = model.prune(tensors, [model.graph.get_tensor_by_name(name) for name in outputs.values()])
    specs = {key: tf.TensorSpec(tensor.shape, tensor.dtype, name=key)
             for key, tensor in zip(inputs, tensors)}

    @tf.function(input_signature=[specs], jit_compile=True, autograph=False)
    def forward(data):
        with tf.device(placement):
            values = pruned(*[data[key] for key in inputs])
            return dict(zip(outputs, values))

    return forward


def prepare_strain(coordinates, cells, strain, *, batch=None):
    transform = tf.eye(3, dtype=strain.dtype) + .5 * (strain + tf.linalg.matrix_transpose(strain))
    if coordinates.shape.rank == 3:
        positions = tf.einsum('bni,bij->bnj', coordinates, transform)
    elif batch is None:
        positions = coordinates @ transform[0]
    else:
        positions = tf.einsum('ni,nij->nj', coordinates, tf.gather(transform, batch))
    return positions, cells @ transform


def compile_hvp(forward, *, position_key, force_key):
    @tf.function(jit_compile=True, autograph=False)
    def product(data, vector):
        positions = data[position_key]
        with tf.GradientTape() as tape:
            tape.watch(positions)
            output = forward(data)
            contraction = -tf.reduce_sum(output[force_key] * vector)
        gradient = tape.gradient(contraction, positions, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        return tf.convert_to_tensor(gradient), output
    return product


def compile_strain(forward, *, position_key, cell_key, energy_key, batch_key=None):
    @tf.function(jit_compile=True, autograph=False)
    def strained_energy(data, strain):
        with tf.GradientTape() as tape:
            tape.watch(strain)
            coordinates, cells = prepare_strain(
                data[position_key], data[cell_key], strain,
                batch=data[batch_key] if batch_key is not None else None,
            )
            output = forward({**data, position_key: coordinates, cell_key: cells})
            energy = tf.reduce_sum(output[energy_key])
        gradient = tape.gradient(energy, strain, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        return (energy, output), tf.convert_to_tensor(gradient)
    return strained_energy


def hessian(hvp, data, coordinates, atom_count, device='cpu'):
    rows = []
    with tf.device(_device(device)):
        for index in range(atom_count * 3):
            vector = np.zeros(coordinates.shape, coordinates.dtype.as_numpy_dtype)
            vector.flat[index] = 1.
            product, _ = hvp(data, tf.convert_to_tensor(vector))
            rows.append(np.asarray(product)[:atom_count].reshape(-1))
    return np.stack(rows)


def array_properties(energy, members, properties, *, forces=None, charges=None, dipole=None,
                     strain_gradient=None, volume=None, source_unit='eV'):
    """Select and split existing native arrays, then convert public units once."""
    unknown = set(properties) - {'energy', 'free_energy', 'forces', 'charges', 'dipole', 'stress', 'virial'}
    if unknown:
        raise NotImplementedError(f'Unsupported array properties: {sorted(unknown)}')
    result = {}
    if 'energy' in properties or 'free_energy' in properties:
        values = np.asarray(energy, dtype=float).reshape(-1)[:len(members)].copy()
        result['energy' if 'energy' in properties else 'free_energy'] = values
    for name, values, dense_rank in (('forces', forces, 3), ('charges', charges, 2)):
        if name in properties:
            if values is None:
                raise NotImplementedError(f'No {name} array supplied')
            array = np.asarray(values, dtype=float)
            if array.ndim == dense_rank:
                result[name] = [array[i, :len(atoms)].copy() for i, atoms in enumerate(members)]
            else:
                offsets = np.cumsum([0, *map(len, members)])
                result[name] = [array[offsets[i]:offsets[i + 1]].copy() for i in range(len(members))]
    if 'dipole' in properties:
        if dipole is None:
            raise NotImplementedError('No dipole array supplied')
        result['dipole'] = np.asarray(dipole, dtype=float)[:len(members)].copy()
    if set(properties) & {'stress', 'virial'}:
        if strain_gradient is None:
            raise NotImplementedError('No strain derivative supplied')
        derivative = np.asarray(strain_gradient, dtype=float)[:len(members)]
        if 'stress' in properties:
            if volume is None:
                raise ValueError('Stress requires cell volumes')
            volumes = np.asarray(volume, dtype=float).reshape(-1)[:len(members)]
            result['stress'] = derivative / volumes[:, None, None]
        if 'virial' in properties:
            result['virial'] = -derivative
    result = _convert_properties(result, source_unit=source_unit)
    if 'energy' in properties and 'free_energy' in properties:
        result['free_energy'] = result['energy']
    return result
