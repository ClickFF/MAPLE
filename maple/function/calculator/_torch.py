"""TorchScript loading, derivatives and public property shapes."""

import numpy as np
import torch

from .calculator_base import _convert_properties


def load_model(calc, path, device, dtype=None):
    """Load a script and return its native parameter dtype for family buffers."""
    calc.device = torch.device(device)
    calc.model = torch.jit.load(str(path), map_location=calc.device).eval()
    for name in ('supports_native_batch', 'supports_native_pbc_batch'):
        if not isinstance(getattr(calc.model, name, None), bool):
            raise ValueError(f'TorchScript artifact requires boolean field {name!r}: {path}')
    native = next((p.dtype for p in calc.model.parameters() if p.is_floating_point()), torch.float32)
    requested = native if dtype is None else {
        'float32': torch.float32, 'float64': torch.float64,
        torch.float32: torch.float32, torch.float64: torch.float64,
    }.get(dtype)
    if requested is None or (native == torch.float32 and requested == torch.float64):
        raise ValueError(f'dtype={dtype!r} exceeds artifact precision {native}')
    for parameter in calc.model.parameters():
        parameter.requires_grad_(False)
        if requested != native and parameter.dtype == native:
            parameter.data = parameter.data.to(requested)
    calc.dtype = requested
    return native


def _numpy(tensor):
    # Match ASE's existing float64 result boundary; unit conversion must not
    # round a native FP32 energy a second time in Hartree.
    return tensor.detach().cpu().numpy().astype(np.float64, copy=True)


def _grad(value, variables, *, create_graph=False, retain_graph=False):
    """Constant-energy and locally linear-energy derivatives are exactly zero."""
    if not value.requires_grad:
        return tuple(torch.zeros_like(v) for v in variables)
    gradients = torch.autograd.grad(value, variables, create_graph=create_graph,
                                    retain_graph=retain_graph, allow_unused=True)
    return tuple(torch.zeros_like(v) if g is None else g for g, v in zip(gradients, variables))


def _native_atomic(tensor, positions, members, *, vector=False, as_tensor=False):
    if not isinstance(tensor, torch.Tensor):
        return None
    if not vector and tensor.dim() == positions.dim() and tensor.shape[-1] == 1:
        tensor = tensor.squeeze(-1)
    tail = (3,) if vector else ()
    if positions.dim() == 3:
        expected = tuple(positions.shape[:2]) + tail
        if tuple(tensor.shape) != expected:
            return None
    else:
        count = sum(map(len, members))
        if (tensor.dim() != len(tail) + 1 or tuple(tensor.shape[1:]) != tail
                or tensor.shape[0] not in (count, count + 1)):
            return None
    if not bool(torch.isfinite(tensor).all()):
        return None
    return tensor if as_tensor else _atomic_values(tensor, members, dense=positions.dim() == 3)


def _native_system(tensor, members, shape):
    if not isinstance(tensor, torch.Tensor):
        return None
    if len(members) == 1 and tuple(tensor.shape) == shape:
        tensor = tensor.unsqueeze(0)
    if tuple(tensor.shape) != (len(members),) + shape:
        return None
    if not bool(torch.isfinite(tensor).all()):
        return None
    return _numpy(tensor)


def _atomic_values(tensor, members, *, dense):
    array = _numpy(tensor)
    if dense:
        return [array[i, :len(atoms)].copy() for i, atoms in enumerate(members)]
    values, start = [], 0
    for atoms in members:
        values.append(array[start:start + len(atoms)].copy())
        start += len(atoms)
    return values


def _coordinate_indices(positions, members):
    start = 0
    for i, atoms in enumerate(members):
        if positions.dim() == 3:
            start = i * positions.shape[1] * 3
        yield torch.arange(start, start + len(atoms) * 3, device=positions.device)
        if positions.dim() == 2:
            start += len(atoms) * 3


def _pack_vector(vector, positions, members, single):
    vectors = [vector] if single else list(vector)
    if len(vectors) != len(members):
        raise ValueError("HVP requires one vector per system")
    result = torch.zeros_like(positions)
    start = 0
    for i, (atoms, value) in enumerate(zip(members, vectors)):
        value = torch.as_tensor(value, device=positions.device, dtype=positions.dtype)
        if value.shape == (3 * len(atoms),):
            value = value.reshape(len(atoms), 3)
        if value.shape != (len(atoms), 3) or not bool(torch.isfinite(value).all()):
            raise ValueError("HVP vector must have shape [N,3] or [3N] and finite values")
        if positions.dim() == 3:
            result[i, :len(atoms)] = value
        else:
            result[start:start + len(atoms)] = value
            start += len(atoms)
    return result


def derivative_order(properties):
    if set(properties) & {"hessian", "hvp"}:
        return 2
    return int(bool(set(properties) & {"forces", "stress", "virial"}))


def prepare_derivatives(coordinates, members, properties, *, batch=None):
    """Prepare leaf/live inputs inside the caller's derivative grad context."""
    positions = coordinates.detach().requires_grad_(derivative_order(properties) > 0)
    if not set(properties) & {"stress", "virial"}:
        return positions, positions, None, None, None
    if not all(atoms.pbc.any() for atoms in members):
        raise ValueError("stress and virial require periodic systems")
    cells = torch.as_tensor(np.array([a.cell.array for a in members]),
                            dtype=positions.dtype, device=positions.device)
    volume = torch.linalg.det(cells).abs()
    if bool((volume <= 0).any()):
        raise ValueError("stress requires nonsingular cells")
    strain = torch.zeros_like(cells, requires_grad=True)
    transform = torch.eye(3, dtype=positions.dtype, device=positions.device) + .5 * (
        strain + strain.transpose(-1, -2))
    if positions.dim() == 3:
        live_positions = torch.einsum("bni,bij->bnj", positions, transform)
    elif batch is None:
        live_positions = positions @ transform[0]
    else:
        live_positions = torch.einsum("ni,nij->nj", positions, transform[batch])
    return positions, live_positions, cells @ transform, strain, volume


@torch.enable_grad()
def tensor_properties(energy, positions, members, properties, *, outputs=None,
                      source_unit="eV", strain=None, volume=None, single=False, vector=None,
                      dipole_from_charges=False):
    """Resolve physical properties from an existing forward; E/F/H/Hv use Hartree."""
    outputs = {} if outputs is None else outputs
    second = derivative_order(properties) == 2
    energy = energy.reshape(-1)
    if energy.numel() != len(members) or not bool(torch.isfinite(energy).all()):
        raise ValueError("script energy output must contain one value per system")
    native_force, native_stress, native_virial = None, None, None
    force = outputs.get("forces")
    if "forces" in properties or (second and isinstance(force, torch.Tensor) and force.requires_grad):
        native_force = _native_atomic(force, positions, members, vector=True, as_tensor=True)
    if "stress" in properties:
        native_stress = _native_system(outputs.get("stress"), members, (3, 3))
    if "virial" in properties:
        native_virial = _native_system(outputs.get("virial"), members, (3, 3))
    gradient, strain_gradient = None, None
    # A retained native force graph already supplies the complete energy gradient.
    if (second and native_force is not None and native_force.requires_grad
            and native_force.shape == positions.shape):
        gradient = -native_force
    need_position_gradient = gradient is None and (second or ("forces" in properties and native_force is None))
    need_strain_gradient = (("stress" in properties and native_stress is None)
                            or ("virial" in properties and native_virial is None))
    if need_position_gradient or need_strain_gradient:
        variables = (positions,) if strain is None else (positions, strain)
        gradients = _grad(energy.sum(), variables, create_graph=second, retain_graph=second)
        if gradient is None:
            gradient = gradients[0]
        if strain is not None:
            strain_gradient = gradients[1]
    results = {}
    if "energy" in properties or "free_energy" in properties:
        result = _numpy(energy)
        result = float(result[0]) if single else result
        for name in ("energy", "free_energy"):
            if name in properties:
                results[name] = result
    if "forces" in properties:
        values = _atomic_values(native_force if native_force is not None else -gradient,
                                members, dense=positions.dim() == 3)
        results["forces"] = values[0] if single else values
    if "charges" in properties:
        values = _native_atomic(outputs.get("charges"), positions, members)
        if values is None:
            raise NotImplementedError("this script has no atomic charges output")
        results["charges"] = values[0] if single else values
    if "dipole" in properties:
        values = _native_system(outputs.get("dipole"), members, (3,))
        if values is None:
            if not dipole_from_charges:
                raise NotImplementedError("this script has no dipole output")
            if any(atoms.pbc.any() for atoms in members):
                raise NotImplementedError('Periodic dipole requires a native model output')
            charges = _native_atomic(outputs.get("charges"), positions, members)
            if charges is None:
                raise NotImplementedError("this script has no dipole or atomic charges output")
            values = np.stack([(q[:, None] * atoms.positions).sum(0)
                               for q, atoms in zip(charges, members)])
        results["dipole"] = values[0] if single else values
    if "stress" in properties:
        values = native_stress if native_stress is not None else _numpy(
            strain_gradient / volume[:, None, None])
        results["stress"] = values[0] if single else values
    if "virial" in properties:
        values = native_virial if native_virial is not None else _numpy(-strain_gradient)
        results["virial"] = values[0] if single else values
    if "hessian" in properties:
        indices = list(_coordinate_indices(positions, members))
        entries = [gradient.flatten().index_select(0, index) for index in indices]
        rows = [[] for _ in members]
        # Native members are independent: one reverse pass supplies their matching rows.
        for row in range(max(entry.numel() for entry in entries)):
            active = [i for i, entry in enumerate(entries) if row < entry.numel()]
            value = entries[active[0]][row]
            for i in active[1:]:
                value = value + entries[i][row]
            derivative = _grad(value, (positions,), retain_graph=True)[0].flatten()
            for i in active:
                rows[i].append(derivative.index_select(0, indices[i]))
        matrices = [_numpy(torch.stack(member)) for member in rows]
        results["hessian"] = matrices[0] if single else matrices
    if "hvp" in properties:
        packed_vector = _pack_vector(vector, positions, members, single)
        product = _grad((gradient * packed_vector).sum(), (positions,))[0]
        products, forces = [], []
        for indices in _coordinate_indices(positions, members):
            products.append(product.flatten().index_select(0, indices).detach().clone())
            forces.append(-gradient.flatten().index_select(0, indices).detach().clone())
        # Established dimer contract: flattened Hn/F and Tensor energy,
        # all remaining on the requested device and sharing one forward.
        results["hvp"] = (products[0], forces[0], energy[0].detach().clone()) if single else (
            products, forces, energy.detach().clone())
    return _convert_properties(results, source_unit=source_unit)
