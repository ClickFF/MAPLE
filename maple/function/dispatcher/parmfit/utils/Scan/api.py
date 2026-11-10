"""Usage: provide public helpers for running and reading silent scans."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import Atoms

from .engine import SilentScanEngine
from .models import ScanConstraint, SilentScanOptions, SilentScanResult


def _constraint_rows(constraints: list) -> list:
    rows = []
    for constraint in constraints:
        if isinstance(constraint, ScanConstraint):
            rows.append(constraint.as_legacy_row())
        else:
            rows.append(list(constraint))
    return rows


def run_silent_scan(
    *,
    output: str,
    atoms: Atoms,
    constraints: list,
    options: SilentScanOptions | None = None,
    params: dict | None = None,
    method: str = "lbfgs",
    constraint_mode: str = "fixinternals",
) -> SilentScanResult:
    resolved_params = options.to_params() if options is not None else dict(params or {})
    if options is not None:
        method = options.backend
        constraint_mode = options.constraint_mode
    xyz_path = SilentScanEngine(
        output=output,
        atoms=atoms,
        constraints=_constraint_rows(constraints),
        params=resolved_params,
        method=method,
        constraint_mode=constraint_mode,
    ).run()
    return SilentScanResult(output_path=output, xyz_path=xyz_path)


def run_silent_scan_batch(
    *,
    atoms_list: list[Atoms],
    outputs: list[str],
    constraints_list: list[list],
    options: SilentScanOptions | None = None,
    params: dict | None = None,
    method: str = "lbfgs",
    constraint_mode: str = "fixinternals",
) -> list[SilentScanResult]:
    from maple.function.calculator._batch import calculate_batch, _apply_constraints
    from .batch_optimizer import BatchLBFGS, BatchCGWS, BatchCGBS

    if not (len(atoms_list) == len(outputs) == len(constraints_list)):
        raise ValueError("atoms_list, outputs and constraints_list must have the same length")
    if not atoms_list:
        return []
    if len({str(Path(path).resolve().with_suffix("")) for path in outputs}) != len(outputs):
        raise ValueError("Batch scans require distinct output paths")
    resolved_params = options.to_params() if options is not None else dict(params or {})
    if options is not None:
        method, constraint_mode = options.backend, options.constraint_mode
    engines = [SilentScanEngine(
        output=output, atoms=atoms, constraints=_constraint_rows(constraints),
        params=resolved_params, method=method, constraint_mode=constraint_mode,
    ) for atoms, output, constraints in zip(atoms_list, outputs, constraints_list)]
    batch_size = resolved_params.get("batch_size")
    points, current, results = [], [], []
    coords_list = [[] for _ in engines]
    energies = [[] for _ in engines]
    batch = None
    try:
        for engine in engines:
            if engine.constraint_mode == "projected" and engine.mode != "relaxed":
                raise ValueError("projected constraint_mode only supports relaxed scans.")
            scan_values = engine._generate_scan_values()
            xyz_path = engine._prepare_scan(scan_values)
            prepare_node = (engine._apply_constraints
                            if engine.mode != "rigid" and engine.constraint_mode != "projected"
                            else engine._apply_rigid_geometry)
            stream = engine._iter_scan_points(scan_values, prepare_node)
            points.append(stream)
            current.append(next(stream))
            results.append(SilentScanResult(output_path=engine.output, xyz_path=xyz_path))

        if engines[0].mode == "rigid":
            active = list(range(len(engines)))
            while active:
                # Each scan keeps its own predecessor geometry, even for rigid steps.
                groups = {}
                for index in active:
                    groups.setdefault(id(current[index][0].calc), []).append(index)
                values = {}
                for indices in groups.values():
                    atoms = [current[index][0] for index in indices]
                    output = calculate_batch(atoms[0].calc, atoms, ["free_energy"], batch_size)
                    for index, energy in zip(indices, output["free_energy"]):
                        values[index] = _apply_constraints(current[index][0], float(energy))[0]
                finished = active.copy()
                for index in finished:
                    atoms, coord = current[index]
                    engines[index]._record_result(atoms, coord, coords_list[index], energies[index], energy=values[index])
                    try:
                        current[index] = points[index].send(atoms)
                    except StopIteration:
                        active.remove(index)
        else:
            classes = {"lbfgs": BatchLBFGS, "cgws": BatchCGWS, "cgbs": BatchCGBS}
            optimizers = [engine._build_optimizer(
                atoms, use_projection=engine.constraint_mode == "projected",
            ) for engine, (atoms, _) in zip(engines, current)]
            batch = classes[engines[0].backend](optimizers, batch_size=batch_size)
            while batch.active:
                for index in batch.step():
                    atoms = batch.optimizers[index].atoms
                    coord = current[index][1]
                    engines[index]._record_result(atoms, coord, coords_list[index], energies[index], energy=batch.energies[index])
                    try:
                        current[index] = points[index].send(atoms)
                    except StopIteration:
                        continue
                    atoms, _ = current[index]
                    batch.replace(index, engines[index]._build_optimizer(
                        atoms, use_projection=engines[index].constraint_mode == "projected",
                    ))
        for engine, result, values in zip(engines, results, energies):
            engine._finish_scan(result.xyz_path, values)
        return results
    finally:
        if batch is not None:
            batch.close()
        for stream in points:
            stream.close()
        for engine in engines:
            engine._close_scan()


def read_scan_final_atoms(xyz_path: str) -> Atoms:
    lines = Path(xyz_path).read_text(encoding="utf-8").splitlines()
    index = 0
    last_symbols: list[str] = []
    last_positions: list[list[float]] = []
    while index < len(lines):
        if not lines[index].strip():
            index += 1
            continue
        natoms = int(lines[index].strip())
        frame_start = index + 2
        frame_end = frame_start + natoms
        if frame_end > len(lines):
            raise ValueError(f"Incomplete XYZ frame in scan file: {xyz_path}")
        symbols: list[str] = []
        positions: list[list[float]] = []
        for raw in lines[frame_start:frame_end]:
            parts = raw.split()
            if len(parts) < 4:
                raise ValueError(f"Invalid XYZ atom row in scan file {xyz_path}: {raw!r}")
            symbols.append(parts[0])
            positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
        last_symbols = symbols
        last_positions = positions
        index = frame_end
    if not last_symbols:
        raise ValueError(f"Scan XYZ file contains no frames: {xyz_path}")
    return Atoms(symbols=last_symbols, positions=np.asarray(last_positions, dtype=float))
