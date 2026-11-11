"""Usage: build optional local ensemble targets for Stage2 torsion fitting."""

from __future__ import annotations

from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import math
import os

import numpy as np
from ase import Atoms

from ..mechanics import build_mm_topology_cache, dihedral_radians, evaluate_mm_energy
from ..readparm import RefinementParameterSet
from ..runtime import copy_thresholds, get_potential_energy, parmfit_work_prefix
from .config import TorsionFitParams
from .records import TorsionEnsembleResult, TorsionGlobalProblem, TorsionObjectiveTarget
from .scanio import HARTREE_TO_KCAL_MOL
from .topology import normalize_torsion_bond

_TORSION_STEP_DEG = 15.0
_TORSION_OFFSETS_DEG = tuple(float(angle) for angle in range(-180, 181, int(_TORSION_STEP_DEG)) if angle)
_ENSEMBLE_RANDOM_SEED = 42
_ENSEMBLE_TRIAL_MULTIPLIER = 10
_ENSEMBLE_MIN_TRIALS = 200
_ENSEMBLE_MAX_TRIALS = 1000
_ENSEMBLE_MAX_REL_KCAL = 30.0
_ENSEMBLE_MAX_MLIP_REL_KCAL = 30.0
_ENSEMBLE_MIN_FRAMES = 2
_ENSEMBLE_DUPLICATE_RMSD = 0.05
_ENSEMBLE_FROZEN_RMSD = 0.05
_ENSEMBLE_MIN_NONBONDED_DISTANCE = 0.65

_Candidate = tuple[tuple[int, int], np.ndarray, float, dict[tuple[int, int, int, int], float]]


def _ensemble_output_path(output: str, workflow: str = "torsionfit") -> str:
    prefix = f"{parmfit_work_prefix(output, workflow)}_torsionfit"
    return prefix + "_ensemble.xyz"


def _graph_side(adjacency: dict[int, set[int]], start: int, blocked: int) -> set[int]:
    seen = {int(blocked)}
    side: set[int] = set()
    queue: deque[int] = deque([int(start)])
    while queue:
        atom = queue.popleft()
        if atom in seen:
            continue
        seen.add(atom)
        side.add(atom)
        for neighbor in sorted(adjacency.get(atom, set())):
            if neighbor not in seen:
                queue.append(int(neighbor))
    return side


def _rotation_mask(
    paramset: RefinementParameterSet,
    dihedral_atoms: tuple[int, int, int, int],
    mobile_atoms: set[int] | None,
    atom_count: int,
) -> list[bool]:
    _a, _b, c_atom, d_atom = (int(atom) for atom in dihedral_atoms)
    side_atoms = _graph_side(paramset.mol2.adjacency, d_atom, c_atom)
    if mobile_atoms is not None:
        side_atoms &= mobile_atoms
    return [(index + 1) in side_atoms for index in range(atom_count)]


def _is_hydrogen_atom(paramset: RefinementParameterSet, atom_index: int) -> bool:
    atom = paramset.mol2.atoms[int(atom_index) - 1]
    return atom.name.strip().upper().startswith("H") or atom.atom_type.strip().lower().startswith("h")


def _is_terminal_h_side(paramset: RefinementParameterSet, side_atoms: set[int]) -> bool:
    return len(side_atoms) == 1 and _is_hydrogen_atom(paramset, next(iter(side_atoms)))


def _torsion_bond_rotor(
    paramset: RefinementParameterSet,
    torsion_bond: tuple[int, int],
    mobile_atoms: set[int] | None,
) -> tuple[int, int, int, int] | None:
    left, right = normalize_torsion_bond(torsion_bond)
    adjacency = paramset.mol2.adjacency
    left_side = _graph_side(adjacency, left, right)
    right_side = _graph_side(adjacency, right, left)
    if left_side & right_side:
        return None
    if _is_terminal_h_side(paramset, left_side) or _is_terminal_h_side(paramset, right_side):
        return None

    def valid_rotating_side(side: set[int]) -> bool:
        if not side:
            return False
        if mobile_atoms is not None and not side <= mobile_atoms:
            return False
        return True

    candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    if valid_rotating_side(left_side):
        candidates.append((len(left_side), (left, left, right, left)))
    if valid_rotating_side(right_side):
        candidates.append((len(right_side), (right, right, left, right)))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


def _eligible_rotors(
    paramset: RefinementParameterSet,
    mobile_atoms: set[int] | None,
    atom_count: int,
) -> tuple[tuple[int, int, int, int], ...]:
    del atom_count
    rotor_by_torsion_bond: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for dihedral in paramset.dihedrals:
        bond = normalize_torsion_bond((int(dihedral.atoms[1]), int(dihedral.atoms[2])))
        if bond in rotor_by_torsion_bond:
            continue
        rotor = _torsion_bond_rotor(paramset, bond, mobile_atoms)
        if rotor is None:
            continue
        rotor_by_torsion_bond[bond] = rotor
    return tuple(rotor_by_torsion_bond[bond] for bond in sorted(rotor_by_torsion_bond))


def _rotor_for_center(
    rotors: tuple[tuple[int, int, int, int], ...],
    torsion_bond: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    center = normalize_torsion_bond(torsion_bond)
    for rotor in rotors:
        if normalize_torsion_bond((rotor[1], rotor[2])) == center:
            return rotor
    return None


def _trial_budget(size: int) -> int:
    return min(max(_ENSEMBLE_TRIAL_MULTIPLIER * max(int(size), 1), _ENSEMBLE_MIN_TRIALS), _ENSEMBLE_MAX_TRIALS)


def _effective_ensemble_size(params: TorsionFitParams, torsion_bonds) -> int:
    center_count = len({normalize_torsion_bond(center) for center in torsion_bonds})
    if center_count <= 0:
        return 0
    scan_conf_count = center_count * (int(params.torsion_steps) + 1)
    target = int(math.ceil(float(scan_conf_count) * float(params.torsion_ensemble_ratio)))
    return max(1, target)


def _trial_offset_maps(
    rotors: tuple[tuple[int, int, int, int], ...],
    torsion_bonds,
    *,
    size: int,
) -> list[tuple[tuple[int, int], dict[tuple[int, int, int, int], float]]]:
    if not rotors:
        return []
    rng = np.random.default_rng(_ENSEMBLE_RANDOM_SEED)
    centers = tuple(normalize_torsion_bond(center) for center in torsion_bonds)
    trials: list[tuple[tuple[int, int], dict[tuple[int, int, int, int], float]]] = []
    for _ in range(_trial_budget(size)):
        center = centers[int(rng.integers(0, len(centers)))] if centers else normalize_torsion_bond((rotors[0][1], rotors[0][2]))
        target_rotor = _rotor_for_center(rotors, center)
        active_count = int(rng.integers(1, len(rotors) + 1))
        chosen: list[tuple[int, int, int, int]] = []
        if target_rotor is not None:
            chosen.append(target_rotor)
        remaining = [rotor for rotor in rotors if rotor not in chosen]
        extra_count = min(max(active_count - len(chosen), 0), len(remaining))
        if extra_count:
            picked = rng.choice(len(remaining), size=extra_count, replace=False)
            chosen.extend(remaining[int(index)] for index in np.atleast_1d(picked))
        trial = {
            rotor: float(_TORSION_OFFSETS_DEG[int(rng.integers(0, len(_TORSION_OFFSETS_DEG)))])
            for rotor in chosen
        }
        if trial:
            trials.append((center, trial))
    return trials


def _apply_trial_offsets(
    atoms: Atoms,
    paramset: RefinementParameterSet,
    trial: dict[tuple[int, int, int, int], float],
    mobile_atoms: set[int] | None,
) -> Atoms:
    candidate = atoms.copy()
    positions = np.asarray(candidate.get_positions(), dtype=float).copy()
    for dihedral_atoms, offset in trial.items():
        mask = _rotation_mask(paramset, dihedral_atoms, mobile_atoms, len(candidate))
        if not any(mask):
            continue
        _a, b_atom, c_atom, _d = (int(atom) for atom in dihedral_atoms)
        positions = _rotate_masked_positions(
            positions,
            axis_start=b_atom - 1,
            axis_end=c_atom - 1,
            mask=mask,
            angle_rad=np.deg2rad(float(offset)),
        )
    candidate.set_positions(positions)
    return candidate


def _rotate_masked_positions(
    positions: np.ndarray,
    *,
    axis_start: int,
    axis_end: int,
    mask: list[bool],
    angle_rad: float,
) -> np.ndarray:
    origin = np.asarray(positions[int(axis_start)], dtype=float)
    axis = np.asarray(positions[int(axis_end)], dtype=float) - origin
    norm = float(np.linalg.norm(axis))
    if norm <= 1.0e-12:
        return positions
    axis /= norm
    cos_angle = float(np.cos(angle_rad))
    sin_angle = float(np.sin(angle_rad))
    rotated = np.asarray(positions, dtype=float).copy()
    for index, flag in enumerate(mask):
        if not flag:
            continue
        vector = rotated[index] - origin
        rotated[index] = (
            origin
            + vector * cos_angle
            + np.cross(axis, vector) * sin_angle
            + axis * np.dot(axis, vector) * (1.0 - cos_angle)
        )
    return rotated


def _minimum_nonbonded_distance(atoms: Atoms, paramset: RefinementParameterSet) -> float:
    positions = np.asarray(atoms.get_positions(), dtype=float)
    bonded = {
        normalize_torsion_bond((int(bond.atom1), int(bond.atom2)))
        for bond in paramset.mol2.bonds
    }
    minimum = float("inf")
    for left in range(1, len(atoms) + 1):
        for right in range(left + 1, len(atoms) + 1):
            if normalize_torsion_bond((left, right)) in bonded:
                continue
            distance = float(np.linalg.norm(positions[left - 1] - positions[right - 1]))
            minimum = min(minimum, distance)
    return minimum


def _frozen_rmsd(reference: Atoms, candidate: Atoms, mobile_atoms: set[int] | None) -> float:
    if mobile_atoms is None:
        return 0.0
    frozen = [index for index in range(len(candidate)) if (index + 1) not in mobile_atoms]
    if not frozen:
        return 0.0
    delta = np.asarray(candidate.get_positions(), dtype=float)[frozen] - np.asarray(reference.get_positions(), dtype=float)[frozen]
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def _rmsd_to_accepted(candidate: Atoms, accepted: list[Atoms], mobile_atoms: set[int] | None) -> float:
    if not accepted:
        return float("inf")
    indices = (
        sorted(int(atom) - 1 for atom in mobile_atoms)
        if mobile_atoms is not None
        else list(range(len(candidate)))
    )
    if not indices:
        indices = list(range(len(candidate)))
    candidate_positions = np.asarray(candidate.get_positions(), dtype=float)[indices]
    best = float("inf")
    for previous in accepted:
        delta = candidate_positions - np.asarray(previous.get_positions(), dtype=float)[indices]
        best = min(best, float(np.sqrt(np.mean(np.sum(delta * delta, axis=1)))))
    return best


def _passes_geometry_filters(
    reference: Atoms,
    candidate: Atoms,
    paramset: RefinementParameterSet,
    mobile_atoms: set[int] | None,
    accepted: list[Atoms],
) -> bool:
    if _minimum_nonbonded_distance(candidate, paramset) < _ENSEMBLE_MIN_NONBONDED_DISTANCE:
        return False
    if _frozen_rmsd(reference, candidate, mobile_atoms) > _ENSEMBLE_FROZEN_RMSD:
        return False
    if _rmsd_to_accepted(candidate, accepted, mobile_atoms) < _ENSEMBLE_DUPLICATE_RMSD:
        return False
    return True


def _passes_hard_geometry_filters(
    reference: Atoms,
    candidate: Atoms,
    paramset: RefinementParameterSet,
    mobile_atoms: set[int] | None,
) -> bool:
    if _minimum_nonbonded_distance(candidate, paramset) < _ENSEMBLE_MIN_NONBONDED_DISTANCE:
        return False
    if _frozen_rmsd(reference, candidate, mobile_atoms) > _ENSEMBLE_FROZEN_RMSD:
        return False
    return True


def _atoms_from_positions(symbols: list[str], positions: np.ndarray) -> Atoms:
    return Atoms(symbols=symbols, positions=np.asarray(positions, dtype=float).copy())


def _score_trial_with_mm(job) -> _Candidate | None:
    symbols, positions, paramset, center, trial, mobile_atoms = job
    reference = _atoms_from_positions(list(symbols), np.asarray(positions, dtype=float))
    candidate = _apply_trial_offsets(reference, paramset, trial, mobile_atoms)
    if not _passes_hard_geometry_filters(reference, candidate, paramset, mobile_atoms):
        return None
    try:
        energy = float(evaluate_mm_energy(candidate, paramset).total)
    except Exception:
        return None
    if not np.isfinite(energy):
        return None
    return normalize_torsion_bond(center), np.asarray(candidate.get_positions(), dtype=float).copy(), energy, dict(trial)


def _ensemble_worker_count() -> int:
    return min(max((os.cpu_count() or 1) // 2, 1), 8)


def _screen_trials_with_mm(
    *,
    atoms: Atoms,
    paramset: RefinementParameterSet,
    trials: list[tuple[tuple[int, int], dict[tuple[int, int, int, int], float]]],
    mobile_atoms: set[int] | None,
) -> list[_Candidate]:
    if not trials:
        return []
    symbols = atoms.get_chemical_symbols()
    positions = np.asarray(atoms.get_positions(), dtype=float)
    jobs = [(symbols, positions, paramset, center, trial, mobile_atoms) for center, trial in trials]
    max_workers = _ensemble_worker_count()
    if max_workers <= 1:
        scored = [_score_trial_with_mm(job) for job in jobs]
    else:
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                scored = list(pool.map(_score_trial_with_mm, jobs))
        except Exception:
            scored = [_score_trial_with_mm(job) for job in jobs]
    return [candidate for candidate in scored if candidate is not None]


def _select_low_mm_energy_diverse(
    candidates: list[_Candidate],
    *,
    atoms: Atoms,
    paramset: RefinementParameterSet,
    mobile_atoms: set[int] | None,
    total_budget: int,
) -> list[_Candidate]:
    if total_budget <= 0 or not candidates:
        return []
    symbols = atoms.get_chemical_symbols()
    by_center: dict[tuple[int, int], list[_Candidate]] = {}
    for candidate in candidates:
        by_center.setdefault(candidate[0], []).append(candidate)

    filtered: list[tuple[float, _Candidate]] = []
    for center_candidates in by_center.values():
        mm_min = min(float(candidate[2]) for candidate in center_candidates)
        for candidate in center_candidates:
            mm_rel = float(candidate[2]) - mm_min
            if mm_rel <= _ENSEMBLE_MAX_REL_KCAL:
                filtered.append((mm_rel, candidate))
    filtered.sort(key=lambda item: item[0])

    selected: list[_Candidate] = []
    accepted_frames: dict[tuple[int, int], list[Atoms]] = {}
    for _mm_rel, candidate in filtered:
        center, positions, _mm_energy, _trial = candidate
        frame = _atoms_from_positions(symbols, positions)
        if not _passes_geometry_filters(atoms, frame, paramset, mobile_atoms, accepted_frames.setdefault(center, [])):
            continue
        selected.append(candidate)
        accepted_frames[center].append(frame)
        if len(selected) >= int(total_budget):
            break
    return selected


def _write_ensemble_xyz(
    path: str,
    torsion_bond: tuple[int, int],
    frames: list[Atoms],
    mm_screen_rel: np.ndarray,
    mlip_stage_rel: np.ndarray,
    trials: list[dict[tuple[int, int, int, int], float]],
    *,
    append: bool = False,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    center = normalize_torsion_bond(torsion_bond)
    with open(path, "a" if append else "w", encoding="utf-8") as handle:
        for index, atoms in enumerate(frames):
            handle.write(f"{len(atoms)}\n")
            sampled = ",".join(
                f"{dihedral[1]}-{dihedral[2]}:{offset:+.0f}"
                for dihedral, offset in sorted(trials[index].items())
            )
            handle.write(
                f"center={center[0]}-{center[1]} target={center[0]}-{center[1]} accepted={index + 1} "
                f"mm_screen_rel={float(mm_screen_rel[index]):.6f} "
                f"mlip_stage_rel={float(mlip_stage_rel[index]):.6f} sampled={sampled}\n"
            )
            positions = np.asarray(atoms.get_positions(), dtype=float)
            for symbol, xyz in zip(atoms.get_chemical_symbols(), positions):
                handle.write(f"{symbol:2s} {xyz[0]: .8f} {xyz[1]: .8f} {xyz[2]: .8f}\n")


def _scan_ref_mlip_kcal(scan_data_map, torsion_bond: tuple[int, int]) -> float:
    center = normalize_torsion_bond(torsion_bond)
    normalized_scan_map = {normalize_torsion_bond(key): value for key, value in dict(scan_data_map).items()}
    scan_data = normalized_scan_map.get(center)
    if scan_data is None:
        raise ValueError(f"Cannot build torsion ensemble target for {center}: missing scan reference data.")
    return float(np.asarray(scan_data.qm_kcal, dtype=float)[int(scan_data.ref_idx)])


def _mlip_energies(frames, batch_size):
    if not frames:
        return []
    from maple.function.calculator._batch import calculate_batch, _apply_constraints

    try:
        output = calculate_batch(frames[0].calc, frames, ["free_energy"], batch_size)
        return [_apply_constraints(frame, float(energy))[0]
                for frame, energy in zip(frames, output["free_energy"])]
    except Exception:
        # Ensemble screening already discards individual failed candidates.
        energies = []
        for frame in frames:
            try:
                energies.append(float(get_potential_energy(frame)))
            except Exception:
                energies.append(float("nan"))
        return energies


def build_torsion_local_ensemble(
    *,
    atoms: Atoms,
    paramset: RefinementParameterSet,
    torsion_bonds,
    scan_data_map,
    params: TorsionFitParams,
    output: str,
    mobile_atoms=None,
    log_info=None,
    workflow: str = "torsionfit",
    batch_size: int | None = None,
) -> TorsionEnsembleResult:
    if not params.torsion_ensemble:
        return TorsionEnsembleResult()

    mobile_set = None if mobile_atoms is None else {int(atom) for atom in mobile_atoms}
    frames_by_center: dict[tuple[int, int], tuple[Atoms, ...]] = {}
    energies_by_center: dict[tuple[int, int], np.ndarray] = {}
    xyz_paths: dict[tuple[int, int], str] = {}
    warnings: list[str] = []

    centers = tuple(normalize_torsion_bond(center) for center in torsion_bonds)
    if len(set(centers)) < 2:
        warnings.append("torsion ensemble skipped: only one fitted torsion bond.")
        if log_info is not None:
            log_info([f"  [TorsionFit] {warnings[-1]}\n"])
        return TorsionEnsembleResult(warnings=tuple(warnings))

    rotors = _eligible_rotors(paramset, mobile_set, len(atoms))
    if not rotors:
        warnings.append("torsion ensemble skipped: no eligible rotatable torsions.")
    else:
        effective_size = _effective_ensemble_size(params, centers)
        trials = _trial_offset_maps(
            rotors,
            centers,
            size=effective_size,
        )
        candidates = _screen_trials_with_mm(
            atoms=atoms,
            paramset=paramset,
            trials=trials,
            mobile_atoms=mobile_set,
        )
        selected = _select_low_mm_energy_diverse(
            candidates,
            atoms=atoms,
            paramset=paramset,
            mobile_atoms=mobile_set,
            total_budget=effective_size,
        )
        if not selected:
            warnings.append("torsion ensemble skipped: no accepted MM-screened frames.")

        selected_counts: dict[tuple[int, int], int] = {}
        for center, _positions, _mm_energy, _trial in selected:
            selected_counts[center] = selected_counts.get(center, 0) + 1
        selected = [candidate for candidate in selected if selected_counts.get(candidate[0], 0) >= _ENSEMBLE_MIN_FRAMES]
        for center, count in sorted(selected_counts.items()):
            if 0 < count < _ENSEMBLE_MIN_FRAMES:
                warnings.append(f"torsion ensemble skipped for {center}: fewer than {_ENSEMBLE_MIN_FRAMES} accepted frames.")

        selected_by_center: dict[tuple[int, int], list[tuple[Atoms, float, float, dict[tuple[int, int, int, int], float]]]] = {}
        symbols = atoms.get_chemical_symbols()
        frames = []
        for center, positions, mm_energy, trial in selected:
            frame = _atoms_from_positions(symbols, positions)
            frame.calc = atoms.calc
            copy_thresholds(atoms, frame)
            frames.append(frame)
        mlip_energies = _mlip_energies(frames, batch_size)
        for (center, _, mm_energy, trial), frame, energy in zip(selected, frames, mlip_energies):
            mlip_energy = float(energy) * HARTREE_TO_KCAL_MOL
            if not np.isfinite(mlip_energy):
                continue
            selected_by_center.setdefault(center, []).append((frame, float(mm_energy), mlip_energy, dict(trial)))

        xyz_path = _ensemble_output_path(output, workflow)
        wrote_xyz = False
        for center in centers:
            records = selected_by_center.get(center, [])
            if not records:
                continue
            mm_values = np.asarray([record[1] for record in records], dtype=float)
            mlip_values = np.asarray([record[2] for record in records], dtype=float)
            mlip_rel = mlip_values - float(np.min(mlip_values))
            keep_mask = mlip_rel <= _ENSEMBLE_MAX_MLIP_REL_KCAL
            dropped = int(len(records) - int(np.count_nonzero(keep_mask)))
            if dropped:
                warnings.append(f"torsion ensemble filtered {dropped} high-MLIP frames for {center}.")
            records = [record for record, keep in zip(records, keep_mask) if bool(keep)]
            if len(records) < _ENSEMBLE_MIN_FRAMES:
                warnings.append(f"torsion ensemble skipped for {center}: fewer than {_ENSEMBLE_MIN_FRAMES} accepted frames.")
                continue
            frames = [record[0].copy() for record in records]
            mm_values = np.asarray([record[1] for record in records], dtype=float)
            mlip_values = np.asarray([record[2] for record in records], dtype=float)
            trials_for_center = [record[3] for record in records]
            mm_screen_rel = mm_values - float(np.min(mm_values))
            mlip_stage_rel = mlip_values - _scan_ref_mlip_kcal(scan_data_map, center)
            _write_ensemble_xyz(xyz_path, center, frames, mm_screen_rel, mlip_stage_rel, trials_for_center, append=wrote_xyz)
            wrote_xyz = True
            xyz_paths[center] = xyz_path
            frames_by_center[center] = tuple(frame.copy() for frame in frames)
            energies_by_center[center] = np.asarray(mlip_values, dtype=float)

        stage2_frame_count = sum(len(frames) for frames in frames_by_center.values())
        if stage2_frame_count == 0 and selected:
            warnings.append("torsion ensemble skipped: no MLIP-screened frames entered Stage2.")

    if log_info is not None and warnings:
        for warning in warnings:
            log_info([f"  [TorsionFit] {warning}\n"])

    return TorsionEnsembleResult(
        frames_by_center=frames_by_center,
        energies_by_center=energies_by_center,
        xyz_paths=xyz_paths,
        warnings=tuple(warnings),
    )


def _absolute_basis_for_frames(
    problem: TorsionGlobalProblem,
    frames: tuple[Atoms, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_terms = len(problem.k_orig)
    abs_const = np.zeros((len(frames), n_terms), dtype=float)
    abs_cos = np.zeros((len(frames), n_terms), dtype=float)
    abs_sin = np.zeros((len(frames), n_terms), dtype=float)
    reference = problem.reference_paramset if problem.reference_paramset is not None else problem.stage0_paramset

    if problem.grouped:
        for frame_index, atoms in enumerate(frames):
            positions = np.asarray(atoms.get_positions(), dtype=float)
            phi_cache: dict[int, float] = {}
            for torsion_bond in problem.torsion_bonds:
                for group in problem.shared_groups_map.get(torsion_bond, ()):
                    for dihedral_index in group.dihedral_indices:
                        if dihedral_index not in phi_cache:
                            phi_cache[dihedral_index] = dihedral_radians(
                                positions,
                                *reference.dihedrals[int(dihedral_index)].atoms,
                            )
                    for slot_index, slot_period in zip(group.slot_indices, group.slot_periods):
                        abs_const[frame_index, slot_index] = float(len(group.dihedral_indices))
                        abs_cos[frame_index, slot_index] = sum(
                            np.cos(float(slot_period) * phi_cache[int(dihedral_index)])
                            for dihedral_index in group.dihedral_indices
                        )
                        abs_sin[frame_index, slot_index] = sum(
                            np.sin(float(slot_period) * phi_cache[int(dihedral_index)])
                            for dihedral_index in group.dihedral_indices
                        )
        return abs_const, abs_cos, abs_sin

    for frame_index, atoms in enumerate(frames):
        positions = np.asarray(atoms.get_positions(), dtype=float)
        phi_cache: dict[int, float] = {}
        for global_index, (dihedral_index, term_index) in enumerate(problem.term_paths):
            dihedral = reference.dihedrals[int(dihedral_index)]
            if dihedral_index not in phi_cache:
                phi_cache[int(dihedral_index)] = dihedral_radians(positions, *dihedral.atoms)
            term = dihedral.terms[int(term_index)]
            abs_const[frame_index, global_index] = 1.0
            abs_cos[frame_index, global_index] = np.cos(float(term.period) * phi_cache[int(dihedral_index)])
            abs_sin[frame_index, global_index] = np.sin(float(term.period) * phi_cache[int(dihedral_index)])
    return abs_const, abs_cos, abs_sin


def build_stage2_extra_targets(
    problem: TorsionGlobalProblem,
    ensemble: TorsionEnsembleResult | None,
    params: TorsionFitParams,
) -> tuple[TorsionObjectiveTarget, ...]:
    if ensemble is None or not params.torsion_ensemble or params.torsion_ensemble_weight <= 0.0:
        return ()
    reference = problem.reference_paramset if problem.reference_paramset is not None else problem.stage0_paramset
    topology_cache = build_mm_topology_cache(reference)
    targets: list[TorsionObjectiveTarget] = []
    for torsion_bond in problem.torsion_bonds:
        center = normalize_torsion_bond(torsion_bond)
        frames = tuple(ensemble.frames_by_center.get(center, ()))
        if len(frames) < _ENSEMBLE_MIN_FRAMES:
            continue
        energies = np.asarray(ensemble.energies_by_center[center], dtype=float)
        scan_data = problem.scan_map[center]
        scan_ref_idx = int(scan_data.ref_idx)
        scan_ref_frame = scan_data.frames[scan_ref_idx]
        scan_ref_mlip = float(np.asarray(scan_data.qm_kcal, dtype=float)[scan_ref_idx])
        qm_rel = energies - scan_ref_mlip
        abs_const, abs_cos, abs_sin = _absolute_basis_for_frames(problem, frames)
        ref_const, ref_cos, ref_sin = _absolute_basis_for_frames(problem, (scan_ref_frame,))
        torsion_total = np.sum(
            np.asarray(problem.k_orig, dtype=float)[np.newaxis, :]
            * (
                abs_const
                + abs_cos * np.cos(np.asarray(problem.phase_orig, dtype=float))[np.newaxis, :]
                + abs_sin * np.sin(np.asarray(problem.phase_orig, dtype=float))[np.newaxis, :]
            ),
            axis=1,
        )
        full_total = np.asarray(
            [evaluate_mm_energy(frame, reference, topology_cache=topology_cache).total for frame in frames],
            dtype=float,
        )
        ref_torsion_total = float(
            np.sum(
                np.asarray(problem.k_orig, dtype=float)
                * (
                    ref_const[0]
                    + ref_cos[0] * np.cos(np.asarray(problem.phase_orig, dtype=float))
                    + ref_sin[0] * np.sin(np.asarray(problem.phase_orig, dtype=float))
                )
            )
        )
        ref_full_total = float(evaluate_mm_energy(scan_ref_frame, reference, topology_cache=topology_cache).total)
        ref_constant_total = ref_full_total - ref_torsion_total
        constant_total = full_total - torsion_total
        targets.append(
            TorsionObjectiveTarget(
                label=f"ensemble {center[0]}-{center[1]}",
                torsion_bond=center,
                qm_rel=qm_rel,
                constant_rel=constant_total - ref_constant_total,
                cos_basis=abs_cos - ref_cos[0],
                sin_basis=abs_sin - ref_sin[0],
                weight=float(params.torsion_ensemble_weight),
                source_path=ensemble.xyz_paths.get(center, ""),
            )
        )
    return tuple(targets)


def attach_stage2_extra_targets(
    problem: TorsionGlobalProblem,
    ensemble: TorsionEnsembleResult | None,
    params: TorsionFitParams,
) -> TorsionGlobalProblem:
    targets = build_stage2_extra_targets(problem, ensemble, params)
    return problem if not targets else replace(problem, extra_targets=targets)
