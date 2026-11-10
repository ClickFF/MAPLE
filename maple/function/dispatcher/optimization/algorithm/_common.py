# -*- coding: utf-8 -*-
"""Shared utilities for OPT algorithms."""
import os
from contextlib import nullcontext
from pathlib import Path
from typing import List, Optional

import numpy as np
from ase import Atoms

from maple.function.read.filereader.pdb_reader import write_pdb_trajectory
from maple.function.calculator._batch import calculate_batch, _apply_constraints
from maple.function.calculator._inputs import _state


def member_outputs(output, atoms_list, suffixes):
    """Prepare stable member filenames once, using MAPLE's existing backups."""
    from maple.function.dispatcher.md.logger import _backup_file
    base, ext = os.path.splitext(output)
    outputs = [f'{base}_{i + 1:04d}{ext}' for i in range(len(atoms_list))]
    for i, (path, atoms) in enumerate(zip(outputs, atoms_list), start=1):
        stem = os.path.splitext(path)[0]
        structure_ext = '.pdb' if atoms.info.get('pdb_template') else '.xyz'
        filenames = [path] + [stem + suffix if '.' in suffix else stem + suffix + structure_ext
                              for suffix in suffixes]
        for filename in filenames:
            filename = Path(filename)
            if '*' in filename.name:
                for existing in filename.parent.glob(filename.name):
                    _backup_file(existing)
            else:
                _backup_file(filename)
        charge, mult = _state(atoms)
        Path(path).write_text(f'Member {i}\nCharge: {charge:.17g}, '
                              f'Multiplicity: {int(mult)}\n')
    return outputs


def batch_tensors(calc, atoms_list, nmax, device, batch_size=None, hessian=False):
    """Adapt calculator results to the batch optimizers' fixed padded space."""
    import torch
    properties = ['free_energy', 'forces']
    if hessian:
        properties.append('hessian')
    result = calculate_batch(calc, atoms_list, properties, batch_size)
    E = np.asarray(result['free_energy'], dtype=np.float64).copy()
    F = np.zeros((len(atoms_list), nmax))
    H = np.zeros((len(atoms_list), nmax, nmax)) if hessian else None
    for b, atoms in enumerate(atoms_list):
        E[b], forces = _apply_constraints(atoms, float(E[b]), result['forces'][b])
        n = 3 * len(atoms)
        F[b, :n] = forces.reshape(-1)
        if hessian:
            H[b, :n, :n] = result['hessian'][b]
    values = (E, F, H) if hessian else (E, F)
    return tuple(torch.as_tensor(value, dtype=torch.float64, device=device) for value in values)


def batch_results(atoms_list):
    return {'atoms': atoms_list, 'energy': np.empty(len(atoms_list)),
            'forces': [None] * len(atoms_list),
            'iterations': np.zeros(len(atoms_list), dtype=int),
            'converged': np.zeros(len(atoms_list), dtype=bool)}


def store_batch_results(results, indices, atoms_list, E, F, iteration, done):
    energies, forces = to_numpy_f64(E), to_numpy_f64(F)
    done = done.detach().cpu().numpy()
    for b, idx_orig in enumerate(indices.detach().cpu().tolist()):
        results['energy'][idx_orig] = energies[b]
        results['forces'][idx_orig] = forces[b, :3 * len(atoms_list[b])].reshape(-1, 3).copy()
        results['iterations'][idx_orig] = iteration
        results['converged'][idx_orig] = bool(done[b])


def write_xyz(filename, atoms_list: List[Atoms],
              energies: Optional[List[float]] = None,
              mode: str = "w",
              start_index: int = 0,
              forces: Optional[List[np.ndarray]] = None,
              comments: Optional[List[str]] = None) -> None:
    """Write supplied XYZ frames to a path or stream without evaluating them."""
    if atoms_list and atoms_list[0].info.get("pdb_template"):
        write_pdb_trajectory(
            filename,
            atoms_list,
            energies=energies,
            mode=mode,
            start_index=start_index,
        )
        return

    blocks = []
    for i, at in enumerate(atoms_list):
        pos = at.get_positions()
        symbols = at.get_chemical_symbols()
        image_index = start_index + i

        lines = [f"{len(symbols)}\n"]
        comment_parts = [comments[i] if comments is not None else f"Image {image_index}"]
        if energies is not None:
            comment_parts.append(f"Energy = {energies[i]:.10f}")
            comment_parts.append('energy_unit=Hartree')
        charge, mult = _state(at)
        comment_parts.append(f"charge={charge:.17g} mult={int(mult)}")
        if at.cell.rank:
            lattice = ' '.join(f'{value:.12g}' for value in at.cell.array.reshape(-1))
            pbc = ' '.join('T' if flag else 'F' for flag in at.pbc)
            comment_parts.append(f'Lattice="{lattice}" pbc="{pbc}"')
        if forces is not None:
            kind = 'constraint_adjusted' if at.constraints else 'raw'
            comment_parts.append('Properties=species:S:1:pos:R:3:forces:R:3')
            comment_parts.append(f'force_unit=Hartree/Angstrom force_kind={kind}')
        lines.append('  '.join(comment_parts) + '\n')
        for j, (s, (x, y, z)) in enumerate(zip(symbols, pos)):
            line = f"{s:2s} {x: .10f} {y: .10f} {z: .10f}"
            if forces is not None:
                fx, fy, fz = forces[i][j]
                line += f" {fx: .10f} {fy: .10f} {fz: .10f}"
            lines.append(line + '\n')
        blocks.append("".join(lines))

    with (nullcontext(filename) if hasattr(filename, 'write') else open(filename, mode)) as f:
        f.writelines(blocks)
        f.flush()


def to_numpy_f64(x):
    """Convert input to float64 numpy array or float."""
    if isinstance(x, np.ndarray):
        return x.astype(np.float64, copy=False)
    try:
        import torch
        if isinstance(x, torch.Tensor):
            arr = x.detach().cpu().numpy()
            return arr.astype(np.float64, copy=False)
    except Exception:
        pass
    if np.isscalar(x):
        return float(x)
    return np.asarray(x, dtype=np.float64)


def vec1d(x, n_expected: Optional[int] = None) -> np.ndarray:
    """Convert to float64 1D vector and optionally check length."""
    v = to_numpy_f64(x).reshape(-1)
    if n_expected is not None and v.size != n_expected:
        raise ValueError(f"Expected size {n_expected}, got {v.size}")
    return v


def compute_metrics(atoms, step_cart, forces) -> None:
    """Set max_dp, rms_dp, max_f, rms_f on atoms from step and force arrays."""
    atoms.max_dp = np.abs(step_cart).max()
    atoms.rms_dp = np.sqrt((step_cart ** 2).sum() / step_cart.size)
    atoms.max_f = np.abs(forces).max()
    atoms.rms_f = np.sqrt((forces ** 2).sum() / forces.size)


def is_converged(atoms) -> bool:
    """Test geometry-optimization convergence against atoms.*_th thresholds."""
    return (
        atoms.max_f <= atoms.f_max_th
        and atoms.rms_f <= atoms.f_rms_th
        and atoms.max_dp <= atoms.dp_max_th
        and atoms.rms_dp <= atoms.dp_rms_th
    )
