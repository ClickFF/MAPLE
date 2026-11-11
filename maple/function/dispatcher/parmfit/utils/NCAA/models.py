"""Usage: build NCAA residue models, conformers, graph helpers, and torsion filters."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import os

import numpy as np

from ..capping import build_ace_cap, build_nme_cap
from ..context import find_residue_by_key
from ..model import flatten_model_atoms, infer_bond_pairs, model_to_atoms, update_model_from_atoms
from ..runtime import (
    copy_thresholds,
    get_potential_energy,
    optimize_atoms_geometry,
    optimize_model_geometry,
    run_silent_lbfgs_batch,
    silent_lbfgs_params,
)
from ..structure import copy_residue, covalent_cutoff, get_atom_xyz, get_resid_key, get_resid_label, max_serial, search_atom, _measure_dihedral


@dataclass(frozen=True)
class NCAAIdentity:
    residue_key: tuple[str, int, str]
    resname: str
    chirality: str
    sidechain_anchor: str


@dataclass(frozen=True)
class NCAAConformer:
    label: str
    phi_deg: float
    psi_deg: float
    energy: float
    model: dict


def identity_ncaa(target_residue: dict) -> NCAAIdentity:
    sidechain_atom = _find_sidechain_anchor(target_residue)
    chirality = detect_ncaa_chirality(target_residue, sidechain_atom)
    return NCAAIdentity(
        residue_key=get_resid_key(target_residue),
        resname=target_residue["resname"],
        chirality=chirality,
        sidechain_anchor=sidechain_atom["name"],
    )


def _find_sidechain_anchor(residue: dict) -> dict:
    ca_atom = search_atom(residue, "CA")
    n_atom = search_atom(residue, "N")
    c_atom = search_atom(residue, "C")
    if ca_atom is None or n_atom is None or c_atom is None:
        raise ValueError("NCAA chirality detection requires residue backbone atoms N, CA, and C.")

    sidechain_candidates: list[tuple[float, dict]] = []
    for atom in residue["atoms"]:
        if atom.get("role", atom["name"]) in {"N", "CA", "C", "O", "OXT"} or atom["element"] == "H":
            continue
        delta = get_atom_xyz(atom) - get_atom_xyz(ca_atom)
        distance = float(np.linalg.norm(delta))
        if distance <= covalent_cutoff(ca_atom, atom):
            sidechain_candidates.append((distance, atom))
    if not sidechain_candidates:
        raise ValueError(
            f"NCAA target residue {get_resid_label(residue)} has no sidechain heavy atom attached to CA."
        )
    sidechain_candidates.sort(key=lambda item: (item[0], item[1]["serial"]))
    return sidechain_candidates[0][1]


def detect_ncaa_chirality(residue: dict, sidechain_atom: dict | None = None) -> str:
    ca_atom = search_atom(residue, "CA")
    n_atom = search_atom(residue, "N")
    c_atom = search_atom(residue, "C")
    sidechain_atom = _find_sidechain_anchor(residue) if sidechain_atom is None else sidechain_atom

    ca_xyz = get_atom_xyz(ca_atom)
    n_vec = get_atom_xyz(n_atom) - ca_xyz
    c_vec = get_atom_xyz(c_atom) - ca_xyz
    sidechain_vec = get_atom_xyz(sidechain_atom) - ca_xyz
    signed_volume = float(np.linalg.det(np.stack([n_vec, c_vec, sidechain_vec], axis=1)))
    return "L" if signed_volume > 0.0 else "D"


def conformer_targets(chirality: str) -> list[tuple[str, float, float]]:
    if chirality == "L":
        return [("alpha", -60.0, -40.0), ("beta", -120.0, 140.0)]
    else:  #chirality == "D"
        return [("alpha", 60.0, 40.0), ("beta", 120.0, -140.0)]


def _backbone_n_hydrogens(residue: dict) -> list[dict]:
    nitrogen = search_atom(residue, "N")
    return sorted(
        (
            atom
            for atom in residue["atoms"]
            if atom["element"] == "H"
            and float(np.linalg.norm(get_atom_xyz(atom) - get_atom_xyz(nitrogen))) <= covalent_cutoff(nitrogen, atom)
        ),
        key=lambda atom: atom["serial"],
    )


def build_capped_ncaa_model(
    target_residue: dict,
    rn: str,
    *,
    prev_residue: dict | None = None,
    next_residue: dict | None = None,
) -> dict:
    target_copy = copy_residue(target_residue, resname=rn)
    next_serial = max_serial([target_residue]) + 1
    ace_residue, next_serial = build_ace_cap(target_residue, next_serial, prev_residue=prev_residue)
    nme_residue, next_serial = build_nme_cap(target_residue, next_serial, next_residue=next_residue)
    residues = [ace_residue, target_copy, nme_residue]
    return {
        "name": "ncaa_capped_model",
        "target_key": get_resid_key(target_copy),
        "residues": residues,
        "segment_sizes": {
            "ace": len(ace_residue["atoms"]),
            "residue": len(target_copy["atoms"]),
            "nme": len(nme_residue["atoms"]),
        },
    }


def infer_terminal_omit_names(resid: dict) -> list[str]:
    bonds = infer_bond_pairs({"residues": [resid]})
    atom_names = [atom["name"] for atom in sorted(resid["atoms"], key=lambda atom: atom["serial"])]
    adjacency: dict[str, set[str]] = defaultdict(set)
    for left, right in bonds:
        left_name = atom_names[left - 1]
        right_name = atom_names[right - 1]
        adjacency[left_name].add(right_name)
        adjacency[right_name].add(left_name)

    omit_names: set[str] = set()
    n_hydrogens = sorted(name for name in adjacency.get("N", set()) if search_atom(resid, name)["element"] == "H")
    if len(n_hydrogens) > 1:
        keep_name = next((name for name in ("H", "HN", "H1", "HN1") if name in n_hydrogens), n_hydrogens[0])
        for atom_name in n_hydrogens:
            if atom_name != keep_name:
                omit_names.add(atom_name)

    c_oxygen_names = sorted(name for name in adjacency.get("C", set()) if search_atom(resid, name)["element"] == "O")
    if len(c_oxygen_names) > 1:
        keep_name = "O" if "O" in c_oxygen_names else c_oxygen_names[0]
        for atom_name in c_oxygen_names:
            if atom_name == keep_name:
                continue
            omit_names.add(atom_name)
            for hydrogen_name in sorted(
                neighbor for neighbor in adjacency.get(atom_name, set()) if search_atom(resid, neighbor)["element"] == "H"
            ):
                omit_names.add(hydrogen_name)

    return sorted(omit_names)


def optimize_capped_confs(
    model: dict,
    *,
    source_atoms,
    output: str,
    max_iter: int = 256,
    max_step: float = 0.2,
    frozen_indices: tuple[int, ...] | None = None,
) -> NCAAConformer:
    if frozen_indices:
        atoms = model_to_atoms(model, charge=model.get("charge"), mult=model.get("mult"))
        copy_thresholds(source_atoms, atoms)
        atoms.calc = source_atoms.calc
        from ase.constraints import FixAtoms

        atoms.set_constraint(FixAtoms(indices=list(frozen_indices)))
        try:
            optimize_atoms_geometry(
                atoms,
                output=output,
                max_iter=max_iter,
                max_step=max_step,
                failure_message="NCAA representative minimization.",
            )
        finally:
            atoms.set_constraint(None)
        minimized_model = update_model_from_atoms(model, atoms)
        energy = float(get_potential_energy(atoms))
    else:
        minimized_model = optimize_model_geometry(
            model,
            output=output,
            source_atoms=source_atoms,
            max_iter=max_iter,
            max_step=max_step,
            failure_message="NCAA representative minimization.",
        )
        optimized_atoms = model_to_atoms(
            minimized_model,
            charge=minimized_model.get("charge"),
            mult=minimized_model.get("mult"),
        )
        optimized_atoms.calc = source_atoms.calc
        energy = float(get_potential_energy(optimized_atoms))
    return NCAAConformer(
        label="ref",
        phi_deg=0.0,
        psi_deg=0.0,
        energy=energy,
        model=minimized_model,
    )


def _wrap_degrees(delta: float) -> float:
    return float(((delta + 180.0) % 360.0) - 180.0)


def _rotate_cap_to_dihedral(atoms, quartet: tuple[int, int, int, int], target_deg: float, mask: list[bool]) -> None:
    delta = _wrap_degrees(target_deg - _measure_dihedral(atoms, quartet))
    if abs(delta) < 1.0e-8:
        return
    start = np.asarray(atoms.get_positions(), dtype=float)
    origin = start[quartet[1]]
    axis = start[quartet[2]] - origin
    norm = float(np.linalg.norm(axis))
    if norm < 1.0e-12:
        raise ValueError("Cannot rotate NCAA cap around a degenerate backbone axis.")
    axis /= norm

    best_error = float("inf")
    best_positions = start
    for signed_delta in (delta, -delta):
        angle = np.radians(signed_delta)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        trial = start.copy()
        for index, move_atom in enumerate(mask):
            if not move_atom:
                continue
            vector = start[index] - origin
            trial[index] = (
                origin
                + vector * cos_a
                + np.cross(axis, vector) * sin_a
                + axis * np.dot(axis, vector) * (1.0 - cos_a)
            )
        atoms.set_positions(trial)
        error = abs(_wrap_degrees(target_deg - _measure_dihedral(atoms, quartet)))
        if error < best_error:
            best_error = error
            best_positions = trial
    atoms.set_positions(best_positions)


def _prepare_conformer(
    model: dict, *, label: str, phi_deg: float, psi_deg: float,
    source_atoms, output: str,
):
    try:
        from ase.constraints import FixInternals
    except ModuleNotFoundError:
        class FixInternals:  # pragma: no cover - test fallback only
            def __init__(self, *, bonds=None, dihedrals_deg=None, **kwargs):
                del kwargs
                self.bonds = bonds
                self.dihedrals = dihedrals_deg or []

    atoms = model_to_atoms(model, charge=model.get("charge"), mult=model.get("mult"))
    copy_thresholds(source_atoms, atoms)
    atoms.calc = source_atoms.calc

    index_map = _build_backbone_rotation_map(model)
    current_phi = _measure_dihedral(atoms, index_map["phi"])
    current_psi = _measure_dihedral(atoms, index_map["psi"])
    phi_delta = _wrap_degrees(phi_deg - current_phi)
    psi_delta = _wrap_degrees(psi_deg - current_psi)
    resolved_phi = current_phi + phi_delta
    resolved_psi = current_psi + psi_delta

    minimized_atoms = atoms.copy()
    ace_count = int(model["segment_sizes"]["ace"])
    target_count = int(model["segment_sizes"]["residue"])
    nme_start = ace_count + target_count
    target_residue = model["residues"][1]
    index_by_serial = {
        atom["serial"]: index
        for index, (_residue, atom) in enumerate(flatten_model_atoms(model))
    }
    phi_extra_indices = {
        index_by_serial[atom["serial"]]
        for atom in _backbone_n_hydrogens(target_residue)
    }
    psi_extra_indices = {
        index_by_serial[atom["serial"]]
        for atom in target_residue["atoms"]
        if atom["name"] in {"O", "OXT", "OT1", "OT2"}
    }
    _rotate_cap_to_dihedral(
        minimized_atoms,
        index_map["phi"],
        resolved_phi,
        [index < ace_count or index in phi_extra_indices for index in range(len(atoms))],
    )
    _rotate_cap_to_dihedral(
        minimized_atoms,
        index_map["psi"],
        resolved_psi,
        [index >= nme_start or index in psi_extra_indices for index in range(len(atoms))],
    )
    guess_xyz = os.path.splitext(output)[0] + "_guess.xyz"
    with open(guess_xyz, "w", encoding="utf-8") as handle:
        symbols = minimized_atoms.get_chemical_symbols()
        handle.write(f"{len(symbols)}\n")
        handle.write(f"NCAA cap-only conformer guess: phi={resolved_phi:.4f} psi={resolved_psi:.4f}\n")
        for symbol, (x, y, z) in zip(symbols, minimized_atoms.get_positions()):
            handle.write(f"{symbol:2s} {x: .10f} {y: .10f} {z: .10f}\n")
    copy_thresholds(source_atoms, minimized_atoms)
    minimized_atoms.calc = source_atoms.calc
    nme_residue = model["residues"][2]
    nnm_idx,hnm_idx = index_by_serial[search_atom(nme_residue, "NNM")["serial"]], index_by_serial[search_atom(nme_residue, "HNM")["serial"]]
    positions = np.asarray(minimized_atoms.get_positions(), dtype=float)
    nh_pairs = [
        (
            index_by_serial[search_atom(target_residue, "N")["serial"]],
            index_by_serial[hydrogen["serial"]],
        )
        for hydrogen in _backbone_n_hydrogens(target_residue)
    ]
    nh_pairs.append((nnm_idx, hnm_idx))
    constraint = FixInternals(
        bonds=[
            [float(np.linalg.norm(positions[n_idx] - positions[h_idx])), [n_idx, h_idx]]
            for n_idx, h_idx in nh_pairs
        ],
        dihedrals_deg=[
            [resolved_phi, list(index_map["phi"])],
            [resolved_psi, list(index_map["psi"])],
        ]
    )
    if hasattr(minimized_atoms, "set_constraint"):
        minimized_atoms.set_constraint(constraint)
    else:  # pragma: no cover - test stub fallback
        minimized_atoms.constraints = [constraint]
    return minimized_atoms


def minimize_conformer(
    model: dict,
    *,
    label: str,
    phi_deg: float,
    psi_deg: float,
    source_atoms,
    output: str,
    max_iter: int = 256,
    max_step: float = 0.2,
) -> NCAAConformer:
    minimized_atoms = _prepare_conformer(
        model, label=label, phi_deg=phi_deg, psi_deg=psi_deg,
        source_atoms=source_atoms, output=output,
    )
    try:
        optimize_atoms_geometry(
            minimized_atoms,
            output=f"{os.path.splitext(output)[0]}_opt.out",
            max_iter=max_iter,
            max_step=max_step,
            failure_message=f"NCAA {label} conformer optimization did not converge.",
        )
    finally:
        if hasattr(minimized_atoms, "set_constraint"):
            minimized_atoms.set_constraint(None)
        else:  # pragma: no cover - test stub fallback
            minimized_atoms.constraints = []
    minimized = update_model_from_atoms(model, minimized_atoms)
    return NCAAConformer(
        label=label,
        phi_deg=phi_deg,
        psi_deg=psi_deg,
        energy=float(get_potential_energy(minimized_atoms)),
        model=minimized,
    )


def build_charge_conformers(
    reference_model: dict,
    *,
    chirality: str,
    source_atoms,
    output_base: str,
    max_iter: int = 256,
    max_step: float = 0.2,
    batch_size: int | None = None,
) -> list[NCAAConformer]:
    targets = list(conformer_targets(chirality))
    atoms_list, outputs = [], []
    try:
        for label, phi_deg, psi_deg in targets:
            output = f"{output_base}_{label}.out"
            atoms_list.append(_prepare_conformer(
                reference_model, label=label, phi_deg=phi_deg, psi_deg=psi_deg,
                source_atoms=source_atoms, output=output,
            ))
            outputs.append(f"{os.path.splitext(output)[0]}_opt.out")
        params = silent_lbfgs_params()
        params.max_iter = int(max_iter)
        params.max_step = float(max_step)
        optimizer = run_silent_lbfgs_batch(
            atoms_list, outputs=outputs, params=params, batch_size=batch_size,
        )
        for index, (label, _, _) in enumerate(targets):
            if not optimizer.converged[index]:
                raise RuntimeError(f"NCAA {label} conformer optimization did not converge.")
    finally:
        for atoms in atoms_list:
            atoms.set_constraint(None)
    return [NCAAConformer(
        label=label, phi_deg=phi_deg, psi_deg=psi_deg,
        energy=float(optimizer.energies[index]),
        model=update_model_from_atoms(reference_model, atoms),
    ) for index, ((label, phi_deg, psi_deg), atoms) in enumerate(zip(targets, atoms_list))]


def _build_backbone_rotation_map(model: dict) -> dict[str, tuple[int, int, int, int]]:
    index_by_serial = {atom["serial"]: index for index, (_residue, atom) in enumerate(flatten_model_atoms(model))}

    ace_resid, target_resid, nme_resid = model["residues"]
    return {
        "omega_pre": (
            index_by_serial[search_atom(ace_resid, "OAC")["serial"]],
            index_by_serial[search_atom(ace_resid, "CAC")["serial"]],
            index_by_serial[search_atom(target_resid, "N")["serial"]],
            index_by_serial[search_atom(target_resid, "CA")["serial"]],
        ),
        "phi": (
            index_by_serial[search_atom(ace_resid, "CAC")["serial"]],
            index_by_serial[search_atom(target_resid, "N")["serial"]],
            index_by_serial[search_atom(target_resid, "CA")["serial"]],
            index_by_serial[search_atom(target_resid, "C")["serial"]],
        ),
        "psi": (
            index_by_serial[search_atom(target_resid, "N")["serial"]],
            index_by_serial[search_atom(target_resid, "CA")["serial"]],
            index_by_serial[search_atom(target_resid, "C")["serial"]],
            index_by_serial[search_atom(nme_resid, "NNM")["serial"]],
        ),
        "omega_post": (
            index_by_serial[search_atom(target_resid, "CA")["serial"]],
            index_by_serial[search_atom(target_resid, "C")["serial"]],
            index_by_serial[search_atom(nme_resid, "NNM")["serial"]],
            index_by_serial[search_atom(nme_resid, "CNM")["serial"]],
        ),
    }


def build_residue_local_adjacency(residue: dict) -> tuple[list[dict], dict[str, int], dict[int, list[int]]]:
    atoms = sorted(residue["atoms"], key=lambda atom: atom["serial"])
    local_index_by_name = {atom["name"]: index + 1 for index, atom in enumerate(atoms)}
    adjacency = {index: [] for index in range(1, len(atoms) + 1)}
    for left, right in infer_bond_pairs({"residues": [residue]}):
        adjacency[left].append(right)
        adjacency[right].append(left)
    for index in adjacency:
        adjacency[index].sort(key=lambda neighbor: atoms[neighbor - 1]["serial"])
    return atoms, local_index_by_name, adjacency


def infer_mainchain_names(residue: dict) -> list[str]:
    atoms, local_index_by_name, adjacency = build_residue_local_adjacency(residue)
    start = local_index_by_name["N"]
    goal = local_index_by_name["C"]

    parent = {start: None}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        if current == goal:
            break
        for neighbor in adjacency[current]:
            if atoms[neighbor - 1]["element"] == "H":
                continue
            if neighbor in parent:
                continue
            parent[neighbor] = current
            queue.append(neighbor)
    if goal not in parent:
        raise ValueError("Could not infer NCAA mainchain path from target residue N to C.")

    path: list[int] = []
    node = goal
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()
    return [atoms[index - 1]["name"] for index in path[1:-1]]


def _ncaa_residue_r_group_indices(representative_model: dict) -> tuple[set[int], set[int], set[int], set[int]]:
    target_residue = find_residue_by_key(
        representative_model,
        representative_model["target_key"],
        label="Representative NCAA target residue",
    )
    residue_atoms, _local_index_by_name, local_adjacency = build_residue_local_adjacency(target_residue)
    residue_start = int(representative_model["segment_sizes"]["ace"]) + 1
    residue_indices = {residue_start + offset for offset in range(len(residue_atoms))}
    backbone_indices = {
        residue_start + offset
        for offset, atom in enumerate(residue_atoms)
        if atom.get("role", atom["name"]) in {"N", "CA", "C", "O", "OXT"}
    }

    ca_local_index = next(
        index
        for index, atom in enumerate(residue_atoms, start=1)
        if atom.get("role", atom["name"]) == "CA"
    )
    sidechain_anchors = [
        neighbor
        for neighbor in sorted(local_adjacency[ca_local_index])
        if residue_atoms[neighbor - 1]["element"] != "H"
        and residue_atoms[neighbor - 1].get("role", residue_atoms[neighbor - 1]["name"])
        not in {"N", "C", "O", "OXT"}
    ]
    if not sidechain_anchors:
        raise ValueError(
            f"NCAA target residue {get_resid_label(target_residue)} has no sidechain heavy atoms attached to CA."
        )

    r_group_indices: set[int] = set()
    seen = set(sidechain_anchors)
    stack = sidechain_anchors[:]
    while stack:
        local_index = stack.pop()
        atom = residue_atoms[local_index - 1]
        if atom["element"] == "H" or atom.get("role", atom["name"]) in {"N", "CA", "C", "O", "OXT"}:
            continue
        r_group_indices.add(residue_start + local_index - 1)
        for neighbor in sorted(local_adjacency[local_index]):
            if neighbor in seen:
                continue
            seen.add(neighbor)
            stack.append(neighbor)

    sidechain_relax_indices = set(r_group_indices)
    heavy_atoms = [
        (local_index, atom)
        for local_index, atom in enumerate(residue_atoms, start=1)
        if atom["element"] != "H"
    ]
    for local_index, atom in enumerate(residue_atoms, start=1):
        if atom["element"] != "H":
            continue
        nearest = min(
            heavy_atoms,
            key=lambda item: float(np.linalg.norm(get_atom_xyz(atom) - get_atom_xyz(item[1]))),
        )
        if (residue_start + nearest[0] - 1) in r_group_indices:
            sidechain_relax_indices.add(residue_start + local_index - 1)
    return residue_indices, backbone_indices, r_group_indices, sidechain_relax_indices


def build_ncaa_sidechain_relax_indices(representative_model: dict) -> tuple[int, ...]:
    _residue_indices, _backbone_indices, _r_group_indices, sidechain_relax_indices = _ncaa_residue_r_group_indices(representative_model)
    return tuple(sorted(sidechain_relax_indices))


def warn_capped_proton_transfer(conformers: list[NCAAConformer]) -> None:
    for conformer in conformers:
        nme_residue = next((residue for residue in conformer.model["residues"] if residue["resname"].upper() == "NME"), None)
        if nme_residue is None:
            continue
        nnm_atom = search_atom(nme_residue, "NNM")
        hnm_atom = search_atom(nme_residue, "HNM")
        if nnm_atom is None or hnm_atom is None:
            continue
        hnm_xyz = get_atom_xyz(hnm_atom)
        nnm_distance = float(np.linalg.norm(hnm_xyz - get_atom_xyz(nnm_atom)))
        nearest = None
        for _residue, atom in flatten_model_atoms(conformer.model):
            if atom is hnm_atom or atom["element"] == "H":
                continue
            distance = float(np.linalg.norm(hnm_xyz - get_atom_xyz(atom)))
            if nearest is None or distance < nearest[0]:
                nearest = (distance, atom)
        if nnm_distance > 1.35 or nearest is not None and nearest[1] is not nnm_atom:
            nearest_name = "unknown" if nearest is None else nearest[1]["name"]
            print(
                f"  [WARNING] NCAA capped model {conformer.label}: possible cap proton transfer; "
                f"HNM-NNM={nnm_distance:.3f} A, nearest heavy atom is {nearest_name}.",
            )


def build_ncaa_torsion_bond_filter(representative_model: dict):
    residue_indices, backbone_indices, r_group_indices, _mobile_indices = _ncaa_residue_r_group_indices(representative_model)

    def keep(torsion_bond: tuple[int, int]) -> bool:
        return (
            torsion_bond[0] in residue_indices
            and torsion_bond[1] in residue_indices
            and not (torsion_bond[0] in backbone_indices and torsion_bond[1] in backbone_indices)
            and (torsion_bond[0] in r_group_indices or torsion_bond[1] in r_group_indices)
        )

    return keep
