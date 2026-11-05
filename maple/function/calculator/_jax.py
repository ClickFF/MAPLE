"""Standard JAX exports and coordinate derivatives, without native model libraries."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import export, tree_util

from .calculator_base import _convert_properties


@tree_util.register_pytree_node_class
class ArtifactOutputs(dict):
    """Native numerical fields plus a static, serialized model description."""

    def __init__(self, values, metadata):
        super().__init__(values)
        self.metadata_json = (metadata if isinstance(metadata, str) else
                              json.dumps(metadata, sort_keys=True, allow_nan=False))

    def tree_flatten(self):
        return (dict(self),), self.metadata_json

    @classmethod
    def tree_unflatten(cls, metadata_json, children):
        return cls(children[0], metadata_json)


export.register_pytree_node_serialization(
    ArtifactOutputs,
    serialized_name='maple.artifact_outputs.v1',
    serialize_auxdata=lambda value: value.encode('utf-8'),
    deserialize_auxdata=lambda value: value.decode('utf-8'),
)


def load_model(model_path, device='cpu', *, require_vjp=False):
    model = export.deserialize(Path(model_path).read_bytes())
    name = str(device or 'cpu')
    platform = 'cuda' if name.startswith('cuda') else 'cpu'
    if platform not in model.platforms:
        raise ValueError(f'JAX export does not support platform {platform!r}: {model_path}')
    derivative = model
    for order in range(int(require_vjp)):
        if not derivative.has_vjp():
            name = 'first' if order == 0 else 'second'
            raise ValueError(f'JAX export requires a saved {name} VJP: {model_path}')
        if order + 1 < int(require_vjp):
            derivative = derivative.vjp()
    backend = 'gpu' if platform == 'cuda' else 'cpu'
    index = int(name.split(':')[1]) if ':' in name else 0
    return model, jax.devices(backend)[index]


def read_metadata(model, *, family, model_id):
    """Read and validate the description without compiling or calling the model."""
    description = model.out_tree.unflatten(model.out_avals)
    if not isinstance(description, ArtifactOutputs):
        raise ValueError('JAX artifact has no model metadata; install a qualified metadata-v1 export')
    data = json.loads(description.metadata_json)
    if not isinstance(data, dict):
        raise ValueError('JAX artifact metadata must be a model-description object')
    if data.get('schema_version') != 1 or isinstance(data.get('schema_version'), bool):
        raise ValueError('Unsupported JAX artifact metadata schema_version')
    if data.get('family') != family or data.get('model_id') != model_id:
        raise ValueError(f'JAX artifact identity does not match {family}/{model_id}')
    if data.get('dtype') not in ('float32', 'float64'):
        raise ValueError('JAX artifact metadata requires a supported dtype')
    if data.get('units') != {'energy': 'eV', 'positions': 'angstrom'}:
        raise ValueError('JAX artifact metadata requires energy=eV and positions=angstrom')
    species = data.get('atomic_numbers')
    if (not isinstance(species, list) or not species or
            any(type(z) is not int or not 1 <= z <= 118 for z in species) or
            len(set(species)) != len(species)):
        raise ValueError('JAX artifact metadata requires unique supported atomic_numbers')
    return data


def prepare_strain(coordinates, cells, strain, *, batch=None):
    """Apply symmetric strain to dense or flat coordinates and row-vector cells."""
    transform = jnp.eye(3, dtype=strain.dtype) + .5 * (strain + strain.swapaxes(-1, -2))
    if coordinates.ndim == 3:
        positions = jnp.einsum('bni,bij->bnj', coordinates, transform)
    elif batch is None:
        positions = coordinates @ transform[0]
    else:
        positions = jnp.einsum('ni,nij->nj', coordinates, transform[batch])
    return positions, cells @ transform


def compile_forces(forward, *, position_key, energy_key, force_key='forces'):
    """Differentiate complete energy and retain the same call's native outputs."""
    def result(data):
        def energy(coordinates):
            output = forward({**data, position_key: coordinates})
            return output[energy_key].sum(), output
        (_, output), gradient = jax.value_and_grad(energy, has_aux=True)(data[position_key])
        return {**output, force_key: -gradient}
    return jax.jit(result)


def compile_hvp(forward, *, position_key, force_key):
    """Differentiate the native forces using the export's saved first VJP."""
    def product(data, vector):
        def contraction(coordinates):
            output = forward({**data, position_key: coordinates})
            return -jnp.vdot(output[force_key], vector), output
        (_, output), value = jax.value_and_grad(contraction, has_aux=True)(data[position_key])
        return value, output
    return jax.jit(product)


def compile_strain(forward, *, position_key, cell_key, energy_key, batch_key=None):
    """Bind native fields to one energy/strain derivative with native outputs."""
    def strained_energy(data, strain):
        coordinates, cells = prepare_strain(
            data[position_key], data[cell_key], strain,
            batch=data[batch_key] if batch_key is not None else None,
        )
        updated = {**data, position_key: coordinates, cell_key: cells}
        output = forward(updated)
        return output[energy_key].sum(), output
    return jax.jit(jax.value_and_grad(strained_energy, argnums=1, has_aux=True))


def hessian(hvp, data, coordinates, atom_count, device):
    """Assemble real-atom rows from a compiled HVP, excluding padding."""
    rows = []
    for index in range(atom_count * 3):
        vector = np.zeros(coordinates.shape, coordinates.dtype)
        vector.flat[index] = 1.
        product, _ = hvp(data, jax.device_put(vector, device))
        rows.append(np.asarray(product)[:atom_count].flatten())
    return np.stack(rows)


def array_properties(energy, members, properties, *, forces=None, charges=None, dipole=None,
                     strain_gradient=None, volume=None, source_unit='eV'):
    """Split native arrays by member; E/F/virial use Hartree, stress eV/Å³."""
    unknown = set(properties) - {'energy', 'free_energy', 'forces', 'charges', 'dipole', 'stress', 'virial'}
    if unknown:
        raise NotImplementedError(f'Unsupported array properties: {sorted(unknown)}')
    result = {}
    if 'energy' in properties or 'free_energy' in properties:
        values = np.asarray(energy, dtype=float).reshape(-1)[:len(members)].copy()
        for name in ('energy', 'free_energy'):
            if name in properties:
                result[name] = values
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
    return _convert_properties(result, source_unit=source_unit)


def calculate_single(calc, atoms, properties, system_changes):
    """One ASE request using this backend's existing property evaluator."""
    from .calculator_base import CalcABC
    properties = calc._normalize_properties(properties)
    if system_changes is None:
        atoms = CalcABC.calculate(calc, atoms, properties)
    else:
        atoms = CalcABC.calculate(calc, atoms, properties, system_changes)
    requested = ['energy' if name == 'free_energy' else name for name in properties]
    requested = list(dict.fromkeys(['energy', *requested]))
    result = calc._calculate([atoms], requested)
    calc._store_properties(atoms, result)
