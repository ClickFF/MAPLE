"""Common system validation and native tensor input preparation."""
from __future__ import annotations

from typing import TYPE_CHECKING, Tuple
import numpy as np
from ase.neighborlist import neighbor_list

if TYPE_CHECKING:
    from torch import Tensor

LR_METHODS = {"auto": 0, "simple": 1, "dsf": 2, "ewald": 3, "pme": 4}

def lattice_shifts(cell: Tensor, cutoff: float) -> Tensor:
    import torch
    repeats = torch.ceil(torch.linalg.vector_norm(torch.linalg.inv(cell).mT, dim=-1) * cutoff).to(torch.long)
    x = torch.arange(-int(repeats[0]), int(repeats[0]) + 1, device=cell.device)
    y = torch.arange(-int(repeats[1]), int(repeats[1]) + 1, device=cell.device)
    z = torch.arange(-int(repeats[2]), int(repeats[2]) + 1, device=cell.device)
    return torch.cartesian_prod(x, y, z).to(cell.dtype)

def make_neighbors(coord: Tensor, cell: Tensor, periodic: bool, cutoff: float) -> Tuple[Tensor, Tensor]:
    import torch
    n = coord.shape[0]
    displacement = coord.unsqueeze(0) - coord.unsqueeze(1)
    if periodic:
        # Work with unwrapped input positions; each pair's lattice origin is
        # shifted into the central image before enumerating all replicas.
        center_shift = -torch.round(displacement @ torch.linalg.inv(cell))
        shifts = lattice_shifts(cell, cutoff)
        image_shifts = center_shift.unsqueeze(2) + shifts.reshape(1, 1, -1, 3)
    else:
        image_shifts = coord.new_zeros((n, n, 1, 3))
    r = displacement.unsqueeze(2) + image_shifts @ cell
    # Selection is discrete setup; exclude the zero self replica only.
    squared = r.detach().pow(2).sum(-1)
    valid = (squared > 1.e-14) & (squared <= cutoff * cutoff)
    counts = valid.flatten(1).sum(-1)
    width = max(1, int(counts.max()))
    neighbors = torch.full((n + 1, width), n, dtype=torch.long, device=coord.device)
    packed_shifts = coord.new_zeros((n + 1, width, 3))
    for i in range(n):
        found = torch.nonzero(valid[i])
        amount = found.shape[0]
        neighbors[i, :amount] = found[:, 0]
        packed_shifts[i, :amount] = image_shifts[i, found[:, 0], found[:, 1]]
    return neighbors, packed_shifts

def prepare_graphs(systems, atomic_numbers, cutoff, *, device="cpu", dtype=None,
                   head_index=0, states):
    import torch
    dtype = torch.float64 if dtype is None else dtype
    table = np.asarray(atomic_numbers, dtype=int)
    positions, numbers, attrs, edges, images, batch, cells, pbcs = [], [], [], [], [], [], [], []
    charges, multiplicities, ptr = [], [], [0]
    for member, atoms in enumerate(systems):
        zs = atoms.numbers
        eq = zs[:, None] == table[None, :]
        if not eq.any(axis=1).all():
            raise ValueError(f"unsupported atomic numbers: {zs[~eq.any(axis=1)].tolist()}")
        src, dst, image = neighbor_list("ijS", atoms, float(cutoff), self_interaction=False)
        positions.append(atoms.positions.copy())
        numbers.append(zs.copy())
        attrs.append(eq.astype(float))
        edges.append(np.stack((src, dst)) + ptr[-1])
        images.append(image)
        batch.append(np.full(len(atoms), member))
        cells.append(atoms.cell.array.copy())
        pbcs.append(atoms.pbc.copy())
        charge, multiplicity = states[member]
        charges.append(charge)
        multiplicities.append(multiplicity)
        ptr.append(ptr[-1] + len(atoms))
    def tensor(value, kind=dtype):
        return torch.as_tensor(np.asarray(value), dtype=kind, device=device)
    result = {
        "positions": tensor(np.concatenate(positions)),
        "numbers": tensor(np.concatenate(numbers), torch.long),
        "node_attrs": tensor(np.concatenate(attrs)),
        "edge_index": tensor(np.concatenate(edges, axis=1), torch.long),
        "unit_shifts": tensor(np.concatenate(images)),
        "batch": tensor(np.concatenate(batch), torch.long),
        "ptr": tensor(ptr, torch.long), "cell": tensor(cells),
        "pbc": tensor(pbcs, torch.bool),
        "head": tensor([head_index] * len(systems), torch.long),
        "total_charge": tensor(charges), "total_spin": tensor(multiplicities),
    }
    result["shifts"] = graph_shifts(result)
    return result


def graph_shifts(data):
    """Cartesian image shifts from integer images and the current live cell."""
    import torch
    sender = data["batch"][data["edge_index"][0]]
    return torch.einsum("ei,eij->ej", data["unit_shifts"], data["cell"][sender])


def _species(value):
    entries = value.detach().cpu().tolist() if hasattr(value, 'detach') else value
    return [int(entry.detach().item()) if hasattr(entry, 'detach') else int(entry)
            for entry in entries]

def validate_lr(lr, lr_cutoff):
    if lr not in LR_METHODS:
        raise ValueError("lr must be auto, simple, dsf, ewald or pme")
    if not np.isfinite(lr_cutoff) or float(lr_cutoff) <= 0:
        raise ValueError("lr_cutoff must be finite and positive")

def _validate_geometry(systems, *, full_pbc=False, min_volume=None):
    if not systems or any(len(a) == 0 for a in systems):
        raise ValueError("at least one nonempty structure is required")
    for atoms in systems:
        if not np.isfinite(atoms.positions).all() or not np.isfinite(atoms.cell.array).all():
            raise ValueError("non-finite positions or cell")
        if atoms.pbc.any():
            if atoms.cell.rank != 3:
                raise ValueError("periodic systems require a full-rank cell")
            if full_pbc and not atoms.pbc.all():
                raise ValueError("periodic systems require full three-dimensional periodicity")
            if min_volume is not None and np.linalg.det(atoms.cell.array) < min_volume:
                raise ValueError(f"periodic systems require a right-handed cell with positive cell volume >= {min_volume:g} Å³")


def _charge(atoms):
    charge = float(atoms.info.get("total_charge", atoms.info.get("charge", atoms.get_initial_charges().sum())))
    if not np.isfinite(charge):
        raise ValueError("total charge must be finite")
    return charge


def _state(atoms):
    charge = _charge(atoms)
    spin = float(atoms.info.get("mult", atoms.info.get("multiplicity", 1)))
    if not np.isfinite(spin):
        raise ValueError("electronic state must be finite")
    if spin < 1 or not spin.is_integer():
        raise ValueError("multiplicity must be a positive integer")
    return charge, spin

def _elements(atoms, supported):
    if supported is not None:
        unsupported = set(map(int, atoms.numbers)) - set(map(int, supported))
        if unsupported:
            raise ValueError(f"unsupported atomic numbers: {sorted(unsupported)}")

def prepare_ani(systems, *, device="cpu", dtype=None, supported_elements=None):
    import torch
    dtype = torch.float32 if dtype is None else dtype
    _validate_geometry(systems)
    size, count = max(map(len, systems)), len(systems)
    numbers = torch.full((count, size), -1, dtype=torch.long, device=device)
    positions = torch.zeros((count, size, 3), dtype=dtype, device=device)
    cells = torch.zeros((count, 3, 3), dtype=dtype, device=device)
    pbc = torch.zeros((count, 3), dtype=torch.bool, device=device)
    for i, atoms in enumerate(systems):
        charge, spin = _state(atoms)
        if charge != 0 or any(float(atoms.info[k]) != 0 for k in ("charge", "total_charge") if k in atoms.info):
            raise ValueError("ANI recipes support neutral systems only")
        if spin != 1 or any(float(atoms.info[k]) != 1 for k in ("mult", "multiplicity") if k in atoms.info):
            raise ValueError("ANI has no multiplicity conditioning; multiplicity must be 1")
        _elements(atoms, supported_elements)
        numbers[i, :len(atoms)] = torch.as_tensor(atoms.numbers, dtype=torch.long, device=device)
        positions[i, :len(atoms)] = torch.as_tensor(atoms.positions, dtype=dtype, device=device)
        cells[i] = torch.as_tensor(atoms.cell.array, dtype=dtype, device=device)
        pbc[i] = torch.as_tensor(atoms.pbc, dtype=torch.bool, device=device)
    result = {"numbers": numbers, "positions": positions,
              "total_charge": torch.zeros(count, dtype=dtype, device=device),
              "multiplicity": torch.ones(count, dtype=torch.long, device=device)}
    if bool(pbc.any()):
        result.update(cell=cells, pbc=pbc)
    return result

def prepare_aimnet(systems, *, device="cpu", dtype=None, supported_species=None,
                   charge_channels=1, supports_charged=True, boundary_domain="both",
                   lr="auto", lr_cutoff=15.0, cutoff=5.2, d3_cutoff=None):
    import torch
    dtype = torch.float32 if dtype is None else dtype
    _validate_geometry(systems)
    validate_lr(lr, lr_cutoff)
    charges, mults = [], []
    for atoms in systems:
        periodic = bool(atoms.pbc.any())
        if periodic and not atoms.pbc.all():
            raise ValueError("AIMNet2 periodic systems require full three-dimensional periodicity")
        if boundary_domain == "periodic-only" and not atoms.pbc.all():
            raise ValueError("AIMNet2 Ewald/PME requires full three-dimensional periodicity")
        if boundary_domain == "nonperiodic-only" and periodic:
            raise ValueError("AIMNet2 simple Coulomb requires nonperiodic systems")
        if lr == "simple" and periodic:
            raise ValueError("simple Coulomb requires nonperiodic systems")
        if lr in ("ewald", "pme") and not atoms.pbc.all():
            raise ValueError(f"{lr.upper() if lr == 'pme' else 'Ewald'} requires full three-dimensional periodicity")
        charge, spin = _state(atoms)
        if not supports_charged and charge != 0:
            raise ValueError("AIMNet2 RXN checkpoint supports neutral systems only")
        if charge_channels != 2 and spin != 1:
            raise ValueError("multiplicity conditioning requires an AIMNet2 NSE checkpoint")
        _elements(atoms, supported_species)
        charges.append(charge)
        mults.append(spin)
    periodic = any(a.pbc.any() for a in systems)
    if len(systems) > 1 and periodic:
        raise NotImplementedError("AIMNet2 native batches require nonperiodic systems")
    result = {"charge": torch.tensor(charges, dtype=dtype, device=device),
              "mult": torch.tensor(mults, dtype=dtype, device=device)}
    if not periodic:
        width = max(map(len, systems))
        coord = torch.zeros((len(systems), width, 3), dtype=dtype, device=device)
        numbers = torch.zeros((len(systems), width), dtype=torch.long, device=device)
        for i, atoms in enumerate(systems):
            coord[i, :len(atoms)] = torch.as_tensor(atoms.positions, dtype=dtype, device=device)
            numbers[i, :len(atoms)] = torch.as_tensor(atoms.numbers, dtype=torch.long, device=device)
        result.update(coord=coord, numbers=numbers)
        return result
    # Neighbor membership is setup state. Integer image coefficients remain
    # separate from the live cell used by the native distance expressions.
    atoms = systems[0]
    coord = torch.as_tensor(atoms.positions, dtype=dtype, device=device)
    numbers = torch.as_tensor(atoms.numbers, dtype=torch.long, device=device)
    cell = torch.as_tensor(atoms.cell.array, dtype=dtype, device=device)
    result.update(coord=torch.cat((coord, coord.new_zeros((1, 3)))),
                  numbers=torch.cat((numbers, numbers.new_zeros(1))), cell=cell,
                  mol_idx=torch.zeros(len(atoms) + 1, dtype=torch.long, device=device))
    cutoffs = [("", cutoff), ("_lr", lr_cutoff)]
    if d3_cutoff is not None:
        cutoffs.append(("_dftd3", d3_cutoff))
    for suffix, radius in cutoffs:
        if not np.isfinite(radius) or float(radius) <= 0:
            raise ValueError("AIMNet2 neighbor cutoffs must be finite and positive")
        neighbors, shifts = make_neighbors(coord, cell, True, float(radius))
        result["nbmat" + suffix], result["shifts" + suffix] = neighbors, shifts
    return result

def prepare_nequip(systems, atomic_numbers, cutoff, *, device="cpu", dtype=None, input_fields):
    import torch
    dtype = torch.float32 if dtype is None else dtype
    _validate_geometry(systems)
    states = [_state(atoms) for atoms in systems]
    if any(q != 0 or spin != 1 for q, spin in states):
        raise ValueError("this model has no charge/multiplicity conditioning; require Q=0, M=1")
    graph = prepare_graphs(systems, atomic_numbers, cutoff, device=device, dtype=dtype, states=states)
    atom_types = graph['node_attrs'].argmax(dim=-1)
    data = {'pos': graph['positions'], 'edge_index': graph['edge_index'],
            'batch': graph['batch']}
    if 'num_atoms' in input_fields:
        data.update(atom_types=atom_types, num_atoms=graph['ptr'][1:] - graph['ptr'][:-1])
    else:
        data.update(atom_types=atom_types[:, None], ptr=graph['ptr'])
    if bool(graph['pbc'].any()):
        data.update(cell=graph['cell'], edge_cell_shift=graph['unit_shifts'])
    return data


def prepare_mace(systems, atomic_numbers, cutoff, *, device="cpu", dtype=None,
                 head_index=0, state_names=(), state_offsets=(), state_classes=(), unconditioned=True,
                 full_pbc=False, min_volume=None):
    import torch
    dtype = torch.float32 if dtype is None else dtype
    _validate_geometry(systems, full_pbc=full_pbc, min_volume=min_volume)
    states = [_state(atoms) for atoms in systems]
    if unconditioned and any(q != 0 or spin != 1 for q, spin in states):
        raise ValueError("this model has no charge/multiplicity conditioning; require Q=0, M=1")
    data = prepare_graphs(systems, atomic_numbers, cutoff, device=device, dtype=dtype,
                          head_index=head_index, states=states)
    for name, offset, classes in zip(state_names, state_offsets, state_classes):
        value = data[name]
        if bool(value.ne(value.round()).any()):
            raise ValueError(f"{name} must be an integer for categorical conditioning")
        if bool(((value + offset < 0) | (value + offset >= classes)).any()):
            raise ValueError(f"{name} lies outside the native embedding table")
    return data

def _field_inputs(atoms):
    field = np.asarray(atoms.arrays.get("external_field", atoms.info.get("external_field", np.zeros((len(atoms), 3)))))
    if field.shape == (3,):
        field = np.broadcast_to(field, (len(atoms), 3)).copy()
    if field.shape != (len(atoms), 3):
        raise ValueError("external_field must have shape [3] or [N,3]")
    potential = np.asarray(atoms.arrays.get("external_potential", atoms.info.get("external_potential", np.zeros((len(atoms), 1)))))
    if potential.ndim == 0:
        potential = np.full((len(atoms), 1), potential)
    if potential.shape == (len(atoms),):
        potential = potential[:, None]
    if potential.shape != (len(atoms), 1):
        raise ValueError("external_potential must have shape [N] or [N,1]")
    if not np.isfinite(field).all() or not np.isfinite(potential).all():
        raise ValueError("external field and potential must be finite")
    return field, potential

def prepare_polar(systems, atomic_numbers, cutoff, *, device="cpu", dtype=None):
    import torch
    dtype = torch.float32 if dtype is None else dtype
    data = prepare_mace(systems, atomic_numbers, cutoff, device=device, dtype=dtype,
                        unconditioned=False, min_volume=1e-6)
    if not bool(torch.all(data["pbc"] == data["pbc"][0])):
        raise ValueError("POLAR batches must group structures by the same boundary policy")
    fields, potentials = zip(*(_field_inputs(atoms) for atoms in systems))
    data["external_field"] = torch.as_tensor(np.concatenate(fields), dtype=dtype, device=device)
    data["external_potential"] = torch.as_tensor(np.concatenate(potentials), dtype=dtype, device=device)
    return data
