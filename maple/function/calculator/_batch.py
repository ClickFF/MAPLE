"""Order-preserving batching behind CalcABC.calculate_batch."""
from collections import OrderedDict
from copy import deepcopy

import numpy as np
from ase import Atoms


def _apply_constraints(atoms, energy, forces=None):
    """Apply the same energy/force adjustments as ASE's public getters."""
    if forces is not None:
        forces = np.array(forces, dtype=np.float64, copy=True)
    for constraint in atoms.constraints:
        if energy is not None and hasattr(constraint, 'adjust_potential_energy'):
            energy += float(constraint.adjust_potential_energy(atoms))
        if forces is not None:
            constraint.adjust_forces(atoms, forces)
    return energy, forces


def energy_forces_single(calc, atoms, force_consistent=True):
    name = 'free_energy' if force_consistent else 'energy'
    calc.calculate(atoms, properties=[name, 'forces'])
    return _apply_constraints(atoms, float(calc.results[name]), calc.results['forces'])


def energy_forces_batch(calc, atoms_list, batch_size=None, force_consistent=True):
    name = 'free_energy' if force_consistent else 'energy'
    result = calculate_batch(calc, atoms_list, [name, 'forces'], batch_size)
    energies, forces = [], []
    for atoms, energy, force in zip(atoms_list, result[name], result['forces']):
        energy, force = _apply_constraints(atoms, float(energy), force)
        energies.append(energy)
        forces.append(force)
    return np.asarray(energies), forces


def energy_forces_images(images, batch_size=None, force_consistent=True):
    """Evaluate ready path images, grouping only those sharing a calculator."""
    groups = OrderedDict()
    for i, atoms in enumerate(images):
        groups.setdefault(id(atoms.calc), []).append(i)
    energies, forces = np.empty(len(images)), [None] * len(images)
    for indices in groups.values():
        members = [images[i] for i in indices]
        E, F = energy_forces_batch(members[0].calc, members, batch_size, force_consistent)
        energies[indices] = E
        for i, force in zip(indices, F):
            forces[i] = force
    return energies, forces


def hvp_single(calc, atoms, n, delta=0.005, batch_size=None):
    """Return Hn, forces and energy at the center; FD uses three structures."""
    native = getattr(calc, 'get_hvp', None)
    # ASE constraints can add energy and project forces outside the model graph.
    if native is not None and not atoms.constraints:
        try:
            values = native(atoms, n)
        except NotImplementedError:
            pass
        else:
            return values
    n = np.asarray(n, dtype=float).reshape(-1, 3)
    displaced = []
    for sign in (1., -1., 0.):
        member = atoms.copy()
        member.set_positions(atoms.positions + sign * delta * n)
        displaced.append(member)
    energies, forces = energy_forces_batch(calc, displaced, batch_size)
    Hn = -(forces[0] - forces[1]).reshape(-1) / (2 * delta)
    return Hn, forces[2].reshape(-1), float(energies[2])


def hessian_single(calc, atoms, batch_size=None):
    """Keep the calculator getter; chunk explicitly requested force differences."""
    if batch_size is not None and getattr(calc, 'hessian', None) == 'numerical':
        return calculate_batch(calc, [atoms], ['hessian'], batch_size)['hessian'][0]
    return calc.get_hessian(atoms)


def _oom(error):
    import sys
    tensorflow = sys.modules.get('tensorflow')
    if tensorflow is not None and isinstance(error, tensorflow.errors.ResourceExhaustedError):
        return any(marker in str(error).lower() for marker in ('out of memory', 'oom when allocating'))
    return isinstance(error, MemoryError) or (isinstance(error, RuntimeError) and any(
        marker in str(error).lower() for marker in (
            'cuda out of memory', 'cuda error: out of memory',
            "defaultcpuallocator: can't allocate memory", 'resource_exhausted: out of memory')))


def calculate_batch(calc, atoms_list, properties=None, batch_size=None):
    from .calculator_base import _property_list, reject_implicit_solvent_derivatives
    members = list(atoms_list)
    if not members or not all(isinstance(atoms, Atoms) and len(atoms) for atoms in members):
        raise ValueError('atoms_list must contain nonempty ASE Atoms objects')
    if batch_size is not None and (isinstance(batch_size, bool) or not isinstance(batch_size, int)
                                   or batch_size < 1):
        raise ValueError('batch_size must be a positive integer')
    requested = list(dict.fromkeys(_property_list(properties)))
    reject_implicit_solvent_derivatives(calc, requested)
    unknown = set(requested) - set(calc.implemented_properties)
    getter_hessian = 'hessian' in unknown and callable(getattr(calc, 'get_hessian', None))
    if getter_hessian:
        unknown.remove('hessian')
    if unknown:
        raise NotImplementedError(f'Unsupported calculator properties: {sorted(unknown)}')
    numeric = 'hessian' in requested and getattr(calc, 'hessian', None) == 'numerical'
    separate_hessian = numeric or getter_hessian
    properties = [name for name in requested if name != 'hessian'] if separate_hessian else requested
    batch_call = getattr(calc, '_calculate_batch', None)
    if getattr(calc, 'solvent_correction', None) is not None:
        batch_call = None
    batch_key = getattr(calc, '_batch_key', None)
    groups = OrderedDict()
    for index, atoms in enumerate(members):
        key = batch_key(atoms) if batch_key is not None else ()
        groups.setdefault(key, []).append(index)
    records = [None] * len(members)
    # Batch requests never replace ASE's last single-structure state, even on error.
    had_state = hasattr(calc, '_input_state')
    saved = (calc.atoms, deepcopy(calc.results), getattr(calc, '_input_state', None))
    try:
        for indices in groups.values():
            width = min(batch_size or len(indices), len(indices))
            offset = 0
            while offset < len(indices):
                chunk = indices[offset:offset + width]
                try:
                    output = NotImplemented
                    if not properties:
                        output = {}
                    elif batch_call is not None:
                        output = batch_call([members[i] for i in chunk], properties)
                    if output is not NotImplemented:
                        if numeric:
                            output['hessian'] = [numerical_hessian(calc, members[i], batch_size=batch_size) for i in chunk]
                        elif getter_hessian:
                            output['hessian'] = [calc.get_hessian(members[i]) for i in chunk]
                except Exception as error:
                    if not _oom(error) or len(chunk) == 1:
                        raise
                    width = max(len(chunk) // 2, 1)
                    continue
                if output is NotImplemented:
                    for index in chunk:
                        calc.calculate(members[index], properties)
                        records[index] = {name: deepcopy(calc.results[name]) for name in properties}
                        if numeric:
                            records[index]['hessian'] = numerical_hessian(calc, members[index], batch_size=batch_size)
                        elif getter_hessian:
                            records[index]['hessian'] = np.asarray(calc.get_hessian(members[index])).copy()
                else:
                    for slot, index in enumerate(chunk):
                        records[index] = {name: output[name][slot] for name in requested}
                offset += len(chunk)
        return {name: ([np.asarray(record[name]).copy() for record in records]
                       if name in {'forces', 'charges', 'hessian', 'atomic_energies'} else
                       np.asarray([record[name] for record in records])) for name in requested}
    finally:
        calc.atoms, calc.results = saved[:2]
        if had_state:
            calc._input_state = saved[2]


def numerical_hessian(calc, atoms, delta=0.002, batch_size=None):
    from ase.constraints import FixAtoms
    reject_pbc = getattr(calc, '_reject_unsupported_pbc', None)
    if reject_pbc is not None:
        reject_pbc(atoms)
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError('Hessian displacement must be finite and positive')
    fixed = {int(i) for constraint in atoms.constraints if isinstance(constraint, FixAtoms)
             for i in constraint.get_indices()}
    movable = [3 * i + k for i in range(len(atoms)) if i not in fixed for k in range(3)]
    matrix = np.zeros((3 * len(atoms), 3 * len(atoms)))
    # Explicit chunks also bound the number of displacement Atoms held at once.
    width = max(1, batch_size // 2) if batch_size else max(1, len(movable))
    for start in range(0, len(movable), width):
        coordinates = movable[start:start + width]
        displaced = []
        for coordinate in coordinates:
            for sign in (1., -1.):
                member = atoms.copy()
                positions = member.positions.copy()
                positions.flat[coordinate] += sign * delta
                member.set_positions(positions)
                displaced.append(member)
        forces = calculate_batch(calc, displaced, ['forces'], batch_size)['forces']
        for slot, coordinate in enumerate(coordinates):
            matrix[:, coordinate] = -(forces[2 * slot] - forces[2 * slot + 1]).flatten() / (2 * delta)
    matrix = .5 * (matrix + matrix.T)
    for atom in fixed:
        matrix[3 * atom:3 * atom + 3] = 0
        matrix[:, 3 * atom:3 * atom + 3] = 0
    return matrix
