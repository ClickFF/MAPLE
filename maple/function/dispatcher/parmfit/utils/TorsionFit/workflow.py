"""Usage: run the torsion scan and fitting workflow."""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Callable

import numpy as np
from ase import Atoms

from ..mechanics import build_mm_topology_cache, dihedral_radians, evaluate_mm_energy
from ..readparm import RefinementParameterSet
from ..Scan import run_silent_scan, run_silent_scan_batch
from ..Scan.batch_optimizer import BatchCGBS
from ..Scan.optimizer import CGBS, _get_potential_energy
from ..runtime import copy_thresholds, parmfit_work_prefix, parmfit_workdir
from ..QMInterface.calculator import QMExternalCalculator
from .topology import (
    normalize_torsion_bond,
    representative_dihedral_for_torsion_bond,
    resolve_torsion_bonds,
)
from .fit import run_loss_mode
from .records import TorsionScanData, TorsionScanRuntime, TorsionWorkflowResult
from .ensemble import build_torsion_local_ensemble
from .scanio import HARTREE_TO_KCAL_MOL, read_scan_xyz
from .config import TorsionFitParams
from .report import format_torsion_stage1_lines, format_torsion_stage2_lines


_LOSS_MODE_MM_ORIG_FILTER_KCAL = 50.0


def _scan_output_paths(output: str, torsion_bond: tuple[int, int], workflow: str = "torsionfit") -> tuple[str, str]:
    center = normalize_torsion_bond(torsion_bond)
    prefix = f"{parmfit_work_prefix(output, workflow)}_torsionfit_{center[0]}-{center[1]}"
    return prefix + ".out", prefix + "_scan_final.xyz"


def _qm_scan_output_paths(
    output: str, torsion_bond: tuple[int, int], qm_mode: int, workflow: str = "torsionfit"
) -> tuple[str, str]:
    center = normalize_torsion_bond(torsion_bond)
    tag = "mode3_qm_torsionfit" if int(qm_mode) == 3 else "qm_torsionfit"
    prefix = f"{parmfit_work_prefix(output, workflow)}_{tag}_{center[0]}-{center[1]}"
    return prefix + ".out", prefix + "_scan_final.xyz"


def _expected_scan_angles(
    atoms: Atoms,
    params: TorsionFitParams,
    representative_dihedral: tuple[int, int, int, int],
) -> np.ndarray:
    atom_indices = [int(atom) - 1 for atom in representative_dihedral]
    initial_angle = float(atoms.get_dihedral(*atom_indices))
    step = float(params.torsion_step_deg)
    return np.asarray([initial_angle + step * idx for idx in range(int(params.torsion_steps) + 1)], dtype=float)


def _read_cached_scan_if_valid(
    scan_xyz: str,
    atoms: Atoms,
    params: TorsionFitParams,
    representative_dihedral: tuple[int, int, int, int],
) -> TorsionScanData | None:
    if not os.path.isfile(scan_xyz):
        return None
    try:
        scan_data = read_scan_xyz(scan_xyz)
    except Exception:
        return None

    expected_count = int(params.torsion_steps) + 1
    if len(scan_data.frames) != expected_count:
        return None

    expected_symbols = atoms.get_chemical_symbols()
    for frame in scan_data.frames:
        if len(frame) != len(expected_symbols) or frame.get_chemical_symbols() != expected_symbols:
            return None

    expected_angles = _expected_scan_angles(atoms, params, representative_dihedral)
    cached_angles = np.asarray(scan_data.angles_deg, dtype=float)
    if cached_angles.shape != expected_angles.shape:
        return None
    if not np.allclose(cached_angles, expected_angles, rtol=0.0, atol=1.0e-2):
        return None
    return scan_data


def _run_torsion_bond_scan(
    atoms: Atoms,
    output: str,
    params: TorsionFitParams,
    runtime: TorsionScanRuntime,
    torsion_bond: tuple[int, int],
    representative_dihedral: tuple[int, int, int, int],
    workflow: str = "torsionfit",
) -> str:
    scan_output, _ = _scan_output_paths(output, torsion_bond, workflow)
    result = run_silent_scan(
        output=scan_output,
        atoms=atoms,
        constraints=[[*representative_dihedral, params.torsion_step_deg, params.torsion_steps]],
        params=runtime.to_scan_params(),
        method=runtime.backend,
        constraint_mode=runtime.constraint_mode,
    )
    return result.xyz_path


def _improper_steps_per_side(params: TorsionFitParams) -> int:
    # the +90/-90 protocol is 360/4 per side; require exact divisibility so the
    # scan range named in reports and cache validation is the range actually scanned
    if int(params.torsion_steps) % 4 != 0:
        raise ValueError(
            f"torsion_steps must be divisible by 4 for the improper +/-90 deg protocol, got {params.torsion_steps}."
        )
    return int(params.torsion_steps) // 4


def _expected_improper_scan_angles(
    atoms: Atoms,
    params: TorsionFitParams,
    quartet: tuple[int, int, int, int],
) -> np.ndarray:
    positions = np.asarray(atoms.get_positions(), dtype=float)
    initial_angle = float(np.degrees(dihedral_radians(positions, *quartet)))
    step = float(params.torsion_step_deg)
    steps = _improper_steps_per_side(params)
    return np.asarray(
        [initial_angle - step * idx for idx in range(steps, 0, -1)]
        + [initial_angle]
        + [initial_angle + step * idx for idx in range(1, steps + 1)],
        dtype=float,
    )


def _read_cached_improper_scan_if_valid(
    scan_xyz: str,
    atoms: Atoms,
    params: TorsionFitParams,
    quartet: tuple[int, int, int, int],
) -> TorsionScanData | None:
    if not os.path.isfile(scan_xyz):
        return None
    try:
        scan_data = read_scan_xyz(scan_xyz)
    except Exception:
        return None

    expected_count = 2 * _improper_steps_per_side(params) + 1
    if len(scan_data.frames) != expected_count:
        return None

    expected_symbols = atoms.get_chemical_symbols()
    for frame in scan_data.frames:
        if len(frame) != len(expected_symbols) or frame.get_chemical_symbols() != expected_symbols:
            return None

    expected_angles = _expected_improper_scan_angles(atoms, params, quartet)
    cached_angles = np.asarray(scan_data.angles_deg, dtype=float)
    if cached_angles.shape != expected_angles.shape:
        return None
    # engine headers record unwrapped ramp targets while the parmfit-convention
    # anchor wraps to (-180, 180]; compare modulo 360
    angle_diff = (cached_angles - expected_angles + 180.0) % 360.0 - 180.0
    if not np.allclose(angle_diff, 0.0, rtol=0.0, atol=1.0e-2):
        return None
    return scan_data


def _run_improper_scan(
    atoms: Atoms,
    output: str,
    params: TorsionFitParams,
    runtime: TorsionScanRuntime,
    quartet: tuple[int, int, int, int],
    workflow: str = "torsionfit",
) -> str:
    """Scan one improper +/-90 deg: exact set_dihedral per point, then a projected
    CGBS relaxation. No ASE FixInternals: its adjust_positions diverges on rigid
    ring centers, while the force projection is well defined for any quartet."""
    center = quartet[2]
    _, scan_xyz = _scan_output_paths(output, (center, center), workflow)
    steps = _improper_steps_per_side(params)
    stem = os.path.splitext(scan_xyz)[0]
    frames: list[tuple[float, float, np.ndarray]] = []
    symbols = atoms.get_chemical_symbols()
    points = _iter_improper_points(atoms, params, quartet, steps)
    try:
        scan_atoms, target, direction = next(points)
        while True:
            scan_atoms = _relax_improper_point(
                scan_atoms, quartet, runtime, f"{stem}_{'neg' if direction < 0 else 'pos'}",
            )
            frames.append((target, float(_get_potential_energy(scan_atoms)), scan_atoms.positions.copy()))
            scan_atoms, target, direction = points.send(scan_atoms)
    except StopIteration:
        pass
    finally:
        points.close()

    merged_frames = list(reversed(frames[:steps])) + frames[steps:]
    return _write_improper_scan_xyz(scan_xyz, merged_frames, symbols)


def _iter_improper_points(atoms, params, quartet, steps):
    idx = tuple(atom - 1 for atom in quartet)
    initial = float(atoms.get_dihedral(*idx))
    # Both branches start from the same reference; keep the shared point once.
    for direction in (-1, +1):
        scan_atoms = atoms.copy()
        scan_atoms.calc = atoms.calc
        copy_thresholds(atoms, scan_atoms)
        point_range = range(1, steps + 1) if direction < 0 else range(0, steps + 1)
        for i in point_range:
            target = initial + direction * params.torsion_step_deg * i
            scan_atoms.set_dihedral(*idx, target)
            scan_atoms = yield scan_atoms, target, direction


def _run_improper_scan_batch(atoms, output, params, runtime, quartets, workflow="torsionfit"):
    if not quartets:
        return []
    paths = [_scan_output_paths(output, (q[2], q[2]), workflow)[1] for q in quartets]
    steps = _improper_steps_per_side(params)
    points = [_iter_improper_points(atoms, params, q, steps) for q in quartets]
    current = [next(stream) for stream in points]
    frames = [[] for _ in quartets]

    def build_optimizer(index):
        scan_atoms, _, direction = current[index]
        stem = os.path.splitext(paths[index])[0] + ("_neg" if direction < 0 else "_pos")
        return _build_improper_optimizer(scan_atoms, quartets[index], runtime, stem)

    batch = BatchCGBS([build_optimizer(i) for i in range(len(quartets))], batch_size=runtime.batch_size)
    try:
        while batch.active:
            for index in batch.step():
                scan_atoms = batch.optimizers[index].atoms
                frames[index].append((current[index][1], float(batch.energies[index]), scan_atoms.positions.copy()))
                try:
                    current[index] = points[index].send(scan_atoms)
                except StopIteration:
                    continue
                batch.replace(index, build_optimizer(index))
        symbols = atoms.get_chemical_symbols()
        for path, values in zip(paths, frames):
            _write_improper_scan_xyz(path, list(reversed(values[:steps])) + values[steps:], symbols)
        return paths
    finally:
        batch.close()
        for stream in points:
            stream.close()


def _relax_improper_point(atoms: Atoms, quartet: tuple[int, int, int, int], runtime: TorsionScanRuntime, log_stem: str) -> Atoms:
    return _build_improper_optimizer(atoms, quartet, runtime, log_stem).run()


def _build_improper_optimizer(atoms, quartet, runtime, log_stem):
    paras = {
        "max_iter": int(runtime.max_iter),
        "max_step": float(runtime.max_step),
        "write_traj": False,
        "verbose": 0,
        "torsion_constraints": [tuple(quartet)],
    }
    return CGBS(atoms, output=log_stem + ".out", paras={"opt": paras})


def _write_improper_scan_xyz(path: str, frames, symbols) -> str:
    with open(path, "w", encoding="utf-8") as handle:
        for index, (angle, energy, positions) in enumerate(frames, start=1):
            handle.write(f"{len(symbols)}\n")
            handle.write(f"Scanning combination {index}/{len(frames)}: [{float(angle):.4f}]  Energy = {float(energy):.10f}\n")
            for symbol, (x_coord, y_coord, z_coord) in zip(symbols, positions):
                handle.write(f"{symbol:2s} {x_coord: .10f} {y_coord: .10f} {z_coord: .10f}\n")
    return path


def _scan_has_qm(path: str, qm_mode: int | None = None) -> bool:
    if not os.path.isfile(path):
        return False
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        while True:
            natoms_line = handle.readline()
            if natoms_line == "":
                return False
            if not natoms_line.strip():
                continue
            try:
                natoms = int(natoms_line.strip())
            except ValueError:
                return False
            comment = handle.readline()
            for _ in range(natoms):
                handle.readline()
            if "Reference = QM" not in comment:
                return False
            if qm_mode is None:
                return True
            return f"Reference = QM mode={int(qm_mode)}" in comment


def _write_qm_scan_xyz(
    path: str,
    *,
    angles_deg: np.ndarray,
    energies_hartree: np.ndarray,
    frames: list[Atoms],
    qm_mode: int,
    geometry: str,
) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for index, (angle_deg, energy_hartree, frame) in enumerate(
            zip(angles_deg, energies_hartree, frames, strict=True),
            start=1,
        ):
            handle.write(f"{len(frame)}\n")
            handle.write(
                f"Scanning combination {index}/{len(frames)}: [{float(angle_deg):.4f}]  "
                f"Reference = QM mode={int(qm_mode)}  Geometry = {geometry}  "
                f"Energy = {float(energy_hartree):.10f}\n"
            )
            for symbol, (x_coord, y_coord, z_coord) in zip(
                frame.get_chemical_symbols(),
                np.asarray(frame.get_positions(), dtype=float),
                strict=True,
            ):
                handle.write(f"{symbol:2s} {float(x_coord): .10f} {float(y_coord): .10f} {float(z_coord): .10f}\n")


def _apply_qm_scan(
    *,
    scan_data: TorsionScanData,
    scan_xyz: str,
    output: str,
    torsion_bond: tuple[int, int],
    representative_dihedral: tuple[int, int, int, int],
    qm_runner,
    qm_mode: int,
    use_sp: bool,
    workflow: str = "torsionfit",
) -> TorsionScanData:
    center = normalize_torsion_bond(torsion_bond)
    work_dir = (
        Path(parmfit_workdir(output, workflow))
        / "opt"
        / f"{Path(output).stem}_qm_torsionfit_{center[0]}-{center[1]}"
    )
    qm_frames: list[Atoms] = []
    qm_energies: list[float] = []
    angles = np.asarray(scan_data.angles_deg, dtype=float)
    previous_wfn = getattr(qm_runner, "last_wfn_path", None)
    for point_index, (angle_deg, frame) in enumerate(zip(angles, scan_data.frames, strict=True), start=1):
        prefix = str(work_dir / f"point_{point_index:03d}")
        if qm_mode == 1:
            energy_result = qm_runner.single_point(
                frame,
                prefix,
                wfn_path=previous_wfn,
            )
            qm_frames.append(frame.copy())
            previous_wfn = energy_result.wfn_path
        else:
            result = qm_runner.constrained_torsion_opt(
                frame,
                prefix,
                torsion=representative_dihedral,
                torsion_angle_deg=float(angle_deg),
                wfn_path=previous_wfn,
            )
            energy_result = result
            if use_sp:
                energy_result = qm_runner.single_point(
                    result.atoms,
                    f"{prefix}_sp",
                    wfn_path=result.wfn_path,
                )
            qm_frames.append(result.atoms)
            previous_wfn = result.wfn_path
        qm_energies.append(float(energy_result.energy_hartree))
    _write_qm_scan_xyz(
        scan_xyz,
        angles_deg=angles,
        energies_hartree=np.asarray(qm_energies, dtype=float),
        frames=qm_frames,
        qm_mode=qm_mode,
        geometry="MLIP scan" if int(qm_mode) == 1 else "QM constrained opt",
    )
    qm_kcal = np.asarray(qm_energies, dtype=float) * HARTREE_TO_KCAL_MOL
    ref_idx = int(np.argmin(qm_kcal))
    return TorsionScanData(
        angles_deg=angles.copy(),
        qm_hartree=np.asarray(qm_energies, dtype=float),
        qm_kcal=qm_kcal,
        frames=qm_frames,
        source_path=str(scan_xyz),
        ref_idx=ref_idx,
        qm_rel=qm_kcal - qm_kcal[ref_idx],
    )


def _run_qm_projected_scan(
    atoms: Atoms,
    output: str,
    params: TorsionFitParams,
    runtime: TorsionScanRuntime,
    torsion_bond: tuple[int, int],
    representative_dihedral: tuple[int, int, int, int],
    qm_runner,
    use_sp: bool,
    workflow: str = "torsionfit",
) -> str:
    center = normalize_torsion_bond(torsion_bond)
    scan_output, scan_xyz = _qm_scan_output_paths(output, torsion_bond, 3, workflow)
    scan_atoms = atoms.copy()
    scan_atoms.info = dict(atoms.info)
    copy_thresholds(atoms, scan_atoms)
    qm_work_dir = (
        Path(parmfit_workdir(output, workflow))
        / "opt"
        / f"{Path(output).stem}_mode3_qm_torsionfit_{center[0]}-{center[1]}"
    )
    scan_atoms.calc = QMExternalCalculator(
        qm_runner,
        work_dir=qm_work_dir,
        wfn_path=getattr(qm_runner, "last_wfn_path", None),
    )
    result = run_silent_scan(
        output=scan_output,
        atoms=scan_atoms,
        constraints=[[*representative_dihedral, params.torsion_step_deg, params.torsion_steps]],
        params=runtime.to_scan_params(),
        method=runtime.backend,
        constraint_mode="projected",
    )
    scan_data = read_scan_xyz(result.xyz_path)
    energies = np.asarray(scan_data.qm_hartree, dtype=float)
    previous_wfn = getattr(scan_atoms.calc, "last_wfn_path", getattr(qm_runner, "last_wfn_path", None))
    if use_sp:
        sp_energies = []
        for point_index, frame in enumerate(scan_data.frames, start=1):
            prefix = str(qm_work_dir / f"point_{point_index:03d}_sp")
            result_sp = qm_runner.single_point(frame, prefix, wfn_path=previous_wfn)
            sp_energies.append(float(result_sp.energy_hartree))
            previous_wfn = result_sp.wfn_path
        energies = np.asarray(sp_energies, dtype=float)
    _write_qm_scan_xyz(
        result.xyz_path,
        angles_deg=np.asarray(scan_data.angles_deg, dtype=float),
        energies_hartree=energies,
        frames=list(scan_data.frames),
        qm_mode=3,
        geometry="QM projected scan",
    )
    return scan_xyz


def _filtered_scan_data(scan_data: TorsionScanData, keep_mask: np.ndarray) -> TorsionScanData:
    mask = np.asarray(keep_mask, dtype=bool)
    if mask.ndim != 1 or mask.shape[0] != len(scan_data.frames):
        raise ValueError("Scan-point filter mask must match the scan length.")
    if np.all(mask):
        return scan_data

    qm_hartree = np.asarray(scan_data.qm_hartree, dtype=float)[mask]
    qm_kcal = np.asarray(scan_data.qm_kcal, dtype=float)[mask]
    angles_deg = np.asarray(scan_data.angles_deg, dtype=float)[mask]
    frames = [frame for frame, keep in zip(scan_data.frames, mask) if keep]
    ref_idx = int(np.argmin(qm_kcal))
    return TorsionScanData(
        angles_deg=angles_deg,
        qm_hartree=qm_hartree,
        qm_kcal=qm_kcal,
        frames=frames,
        source_path=scan_data.source_path,
        ref_idx=ref_idx,
        qm_rel=qm_kcal - qm_kcal[ref_idx],
    )


def _filter_loss_mode_scan_points(
    scan_data: TorsionScanData,
    original_paramset: RefinementParameterSet,
    torsion_bond: tuple[int, int],
    *,
    topology_cache=None,
) -> tuple[TorsionScanData, str | None, np.ndarray]:
    mm_orig_total = np.asarray(
        [evaluate_mm_energy(atoms, original_paramset, topology_cache=topology_cache).total for atoms in scan_data.frames],
        dtype=float,
    )
    mm_orig_filter_rel = mm_orig_total - float(np.min(mm_orig_total))
    keep_mask = np.asarray(mm_orig_filter_rel <= _LOSS_MODE_MM_ORIG_FILTER_KCAL, dtype=bool)
    if np.all(keep_mask):
        mm_orig_report_rel = mm_orig_total - mm_orig_total[int(scan_data.ref_idx)]
        return scan_data, None, mm_orig_report_rel

    dropped_angles = np.asarray(scan_data.angles_deg, dtype=float)[~keep_mask]
    filtered = _filtered_scan_data(scan_data, keep_mask)
    filtered_mm_orig_total = mm_orig_total[keep_mask]
    mm_orig_report_rel = filtered_mm_orig_total - filtered_mm_orig_total[int(filtered.ref_idx)]
    warning = (
        f"loss scan filter: torsion bond {normalize_torsion_bond(torsion_bond)} dropped "
        f"{int(np.count_nonzero(~keep_mask))} point(s) with MM_orig_rel > {_LOSS_MODE_MM_ORIG_FILTER_KCAL:.1f} "
        f"at angles [{', '.join(f'{float(angle):.4f}' for angle in dropped_angles)}]"
    )
    return filtered, warning, mm_orig_report_rel


def _run_mlip_scans(*, atoms, output, params, runtime, paramset, scan_targets,
                    topology_cache, workflow, qm_mode):
    """Populate missing MLIP scan files; the existing workflow consumes them in order."""
    proper, improper = [], []
    for center_key, target in scan_targets:
        _, path = _scan_output_paths(output, center_key, workflow)
        if target is not None:
            if qm_mode in {2, 3}:
                raise ValueError(f"QM mode {qm_mode} is not supported for improper torsion scans; use qm_mode=1.")
            cached = _read_cached_improper_scan_if_valid(path, atoms, params, target.atoms)
            if cached is None or _scan_has_qm(path):
                improper.append(target.atoms)
        elif qm_mode != 3:
            representative = representative_dihedral_for_torsion_bond(paramset, center_key, topology_cache=topology_cache)
            cached = _read_cached_scan_if_valid(path, atoms, params, representative.atoms)
            if cached is None or _scan_has_qm(path):
                proper.append((center_key, representative.atoms))
    if len(proper) > 1:
        run_silent_scan_batch(
            atoms_list=[atoms] * len(proper),
            outputs=[_scan_output_paths(output, key, workflow)[0] for key, _ in proper],
            constraints_list=[[[*quartet, params.torsion_step_deg, params.torsion_steps]] for _, quartet in proper],
            params=runtime.to_scan_params(), method=runtime.backend, constraint_mode=runtime.constraint_mode,
        )
    if len(improper) > 1:
        _run_improper_scan_batch(atoms, output, params, runtime, improper, workflow)


def run_torsion_workflow(
    *,
    atoms: Atoms,
    output: str,
    paramset: RefinementParameterSet,
    original_paramset: RefinementParameterSet | None = None,
    params: TorsionFitParams,
    runtime: TorsionScanRuntime,
    torsion_bond_filter: Callable[[tuple[int, int]], bool] | None = None,
    improper_targets: list | None = None,
    radical_centers: tuple[int, ...] = (),
    mobile_atoms=None,
    qm_runner=None,
    log_info: Callable[[list[str]], None] | None = None,
    workflow: str = "torsionfit",
) -> TorsionWorkflowResult:
    improper_targets = list(improper_targets or [])
    improper_map = {(target.atoms[2], target.atoms[2]): target for target in improper_targets}
    base_paramset = deepcopy(paramset)
    original_paramset = deepcopy(original_paramset) if original_paramset is not None else deepcopy(base_paramset)
    if not params.enabled:
        if log_info is not None:
            log_info(format_torsion_stage1_lines(params, []))
            log_info(["\n[Stage 2] Running stage-2 global torsion refinement...\n"])
            log_info(format_torsion_stage2_lines(params, []))
        return TorsionWorkflowResult(
            stage1_paramset=None,
            final_paramset=deepcopy(base_paramset),
        )

    topology_cache = build_mm_topology_cache(base_paramset)
    torsion_bonds, warnings = resolve_torsion_bonds(
        base_paramset,
        params,
        topology_cache=topology_cache,
    )
    if torsion_bond_filter is not None:
        torsion_bonds = [bond for bond in torsion_bonds if torsion_bond_filter(normalize_torsion_bond(bond))]
    if not torsion_bonds and not improper_map:
        empty_result = deepcopy(base_paramset)
        if log_info is not None:
            log_info(format_torsion_stage1_lines(params, warnings, has_torsion_bonds=False))
            log_info(["\n[Stage 2] Running stage-2 global torsion refinement...\n"])
            log_info(format_torsion_stage2_lines(params, [], has_torsion_bonds=False))
        return TorsionWorkflowResult(
            stage1_paramset=empty_result,
            final_paramset=deepcopy(empty_result),
            torsion_bonds=[],
            warnings=warnings,
        )

    scan_data_map = {}
    scan_xyz_map: dict[tuple[int, int], str] = {}
    scan_mm_orig_rel_map: dict[tuple[int, int], np.ndarray] = {}
    original_topology_cache = build_mm_topology_cache(original_paramset)
    qm_config = getattr(qm_runner, "config", None) if qm_runner is not None else None
    qm_mode = int(getattr(qm_config, "qm_mode", 1)) if qm_config is not None else None
    use_sp = bool(
        qm_config is not None
        and str(getattr(qm_config, "sp_level", "")).strip()
        != str(getattr(qm_config, "opt_level", "")).strip()
    )

    scan_targets = [(normalize_torsion_bond(torsion_bond), None) for torsion_bond in torsion_bonds]
    scan_targets += list(improper_map.items())
    _run_mlip_scans(
        atoms=atoms, output=output, params=params, runtime=runtime, paramset=base_paramset,
        scan_targets=scan_targets, topology_cache=topology_cache, workflow=workflow, qm_mode=qm_mode,
    )
    for center_key, improper_target in scan_targets:
        if center_key[0] == center_key[1]:
            if qm_runner is not None and int(qm_mode) in {2, 3}:
                raise ValueError(
                    f"QM mode {int(qm_mode)} is not supported for improper torsion scans; use qm_mode=1."
                )
            _, mlip_scan_xyz = _scan_output_paths(output, center_key, workflow)
            scan_data = _read_cached_improper_scan_if_valid(mlip_scan_xyz, atoms, params, improper_target.atoms)
            if scan_data is not None and _scan_has_qm(mlip_scan_xyz):
                scan_data = None
            if scan_data is not None:
                scan_xyz = mlip_scan_xyz
                if log_info is not None:
                    log_info([f"reuse existing improper scan xyz: {scan_xyz}\n"])
            else:
                scan_xyz = _run_improper_scan(atoms, output, params, runtime, improper_target.atoms, workflow)
                scan_data = read_scan_xyz(scan_xyz)
            if qm_runner is not None and int(qm_mode) == 1:
                _, qm_scan_xyz = _qm_scan_output_paths(output, center_key, 1, workflow)
                qm_scan_data = _read_cached_improper_scan_if_valid(qm_scan_xyz, atoms, params, improper_target.atoms)
                if qm_scan_data is not None and not _scan_has_qm(qm_scan_xyz, 1):
                    qm_scan_data = None
                if qm_scan_data is None:
                    qm_scan_data = _apply_qm_scan(
                        scan_data=scan_data,
                        scan_xyz=qm_scan_xyz,
                        output=output,
                        torsion_bond=center_key,
                        representative_dihedral=improper_target.atoms,
                        qm_runner=qm_runner,
                        qm_mode=1,
                        use_sp=use_sp,
                        workflow=workflow,
                    )
                scan_data = qm_scan_data
                scan_xyz = qm_scan_xyz
            scan_data, warning, mm_orig_rel = _filter_loss_mode_scan_points(
                scan_data,
                original_paramset,
                center_key,
                topology_cache=original_topology_cache,
            )
            if warning is not None:
                warnings.append(warning)
            scan_data_map[center_key] = scan_data
            scan_xyz_map[center_key] = scan_xyz
            scan_mm_orig_rel_map[center_key] = np.asarray(mm_orig_rel, dtype=float)
            continue
        torsion_bond = center_key
        representative = representative_dihedral_for_torsion_bond(
            base_paramset,
            torsion_bond,
            topology_cache=topology_cache,
        )
        _, mlip_scan_xyz = _scan_output_paths(output, torsion_bond, workflow)
        if qm_runner is not None and int(qm_mode) == 3:
            _, scan_xyz = _qm_scan_output_paths(output, torsion_bond, 3, workflow)
            scan_data = _read_cached_scan_if_valid(scan_xyz, atoms, params, representative.atoms)
            if scan_data is not None and not _scan_has_qm(scan_xyz, 3):
                scan_data = None
            if scan_data is not None:
                if log_info is not None:
                    log_info([f"reuse existing torsion scan xyz: {scan_xyz}\n"])
            else:
                scan_xyz = _run_qm_projected_scan(
                    atoms,
                    output,
                    params,
                    runtime,
                    torsion_bond,
                    representative.atoms,
                    qm_runner,
                    use_sp,
                    workflow,
                )
                scan_data = read_scan_xyz(scan_xyz)
        else:
            scan_xyz = mlip_scan_xyz
            scan_data = _read_cached_scan_if_valid(mlip_scan_xyz, atoms, params, representative.atoms)
            if scan_data is not None and _scan_has_qm(mlip_scan_xyz):
                scan_data = None
            if scan_data is not None:
                if log_info is not None:
                    log_info([f"reuse existing torsion scan xyz: {scan_xyz}\n"])
            else:
                scan_xyz = _run_torsion_bond_scan(
                    atoms,
                    output,
                    params,
                    runtime,
                    torsion_bond,
                    representative.atoms,
                    workflow,
                )
                scan_data = read_scan_xyz(scan_xyz)
        if qm_runner is not None and int(qm_mode) in {1, 2}:
            _, qm_scan_xyz = _qm_scan_output_paths(output, torsion_bond, int(qm_mode), workflow)
            qm_scan_data = _read_cached_scan_if_valid(qm_scan_xyz, atoms, params, representative.atoms)
            if qm_scan_data is not None and not _scan_has_qm(qm_scan_xyz, qm_mode):
                qm_scan_data = None
            if qm_scan_data is None:
                qm_scan_data = _apply_qm_scan(
                    scan_data=scan_data,
                    scan_xyz=qm_scan_xyz,
                    output=output,
                    torsion_bond=torsion_bond,
                    representative_dihedral=representative.atoms,
                    qm_runner=qm_runner,
                    qm_mode=int(qm_mode),
                    use_sp=use_sp,
                    workflow=workflow,
                )
            scan_data = qm_scan_data
            scan_xyz = qm_scan_xyz
        scan_data, warning, mm_orig_rel = _filter_loss_mode_scan_points(
            scan_data,
            original_paramset,
            torsion_bond,
            topology_cache=original_topology_cache,
        )
        if warning is not None:
            warnings.append(warning)
        scan_data_map[center_key] = scan_data
        scan_xyz_map[center_key] = scan_xyz
        scan_mm_orig_rel_map[center_key] = np.asarray(mm_orig_rel, dtype=float)

    ensemble_result = None
    if params.torsion_ensemble:
        ensemble_result = build_torsion_local_ensemble(
            atoms=atoms,
            paramset=base_paramset,
            torsion_bonds=torsion_bonds,
            scan_data_map=scan_data_map,
            params=params,
            output=output,
            mobile_atoms=mobile_atoms,
            log_info=log_info,
            workflow=workflow,
            batch_size=runtime.batch_size,
        )
        warnings.extend(ensemble_result.warnings)

    if log_info is not None and warnings:
        log_info(format_torsion_stage1_lines(params, warnings))

    loss_kwargs = {
        "base_paramset": base_paramset,
        "original_paramset": original_paramset,
        "torsion_bonds": torsion_bonds + list(improper_map),
        "scan_data_map": scan_data_map,
        "scan_xyz_map": scan_xyz_map,
        "scan_mm_orig_rel_map": scan_mm_orig_rel_map,
        "params": params,
        "topology_cache": topology_cache,
        "improper_targets": improper_map,
        "radical_centers": tuple(radical_centers),
        "log_info": log_info,
    }
    if ensemble_result is not None:
        loss_kwargs["ensemble_result"] = ensemble_result
    result = run_loss_mode(**loss_kwargs)

    result.warnings = list(warnings)
    result.torsion_bonds = [normalize_torsion_bond(bond) for bond in torsion_bonds] + list(improper_map)
    result.scan_xyz = dict(scan_xyz_map)
    result.ensemble_xyz = dict(ensemble_result.xyz_paths) if ensemble_result is not None else {}
    return result
