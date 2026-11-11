"""Usage: execute constrained silent scan grids with ASE optimizers."""

from __future__ import annotations

import os
from copy import deepcopy
from typing import Callable, Optional

from ase import Atoms
from ase.constraints import FixInternals
from ase.neighborlist import NeighborList, natural_cutoffs

from .optimizer import CGBS, CGWS, LBFGS

class SilentScanEngine:
    def __init__(
        self,
        *,
        output: str,
        atoms: Atoms,
        constraints: list,
        params: Optional[dict] = None,
        method: str = "lbfgs",
        constraint_mode: str = "fixinternals",
    ):
        self.output = output
        self.atoms = atoms
        self.raw_constraints = deepcopy(constraints)
        self.params = deepcopy(params or {})
        self.backend = self.params.get("backend", method).strip().lower()
        self.constraint_mode = self.params.get("constraint_mode", constraint_mode).strip().lower()
        self.mode = str(self.params.get("mode", "relaxed")).strip().lower()

        opt = dict(self.params.get("opt", {}))
        opt["write_traj"] = False
        opt["verbose"] = 0
        self.params["opt"] = opt
        self.params["backend"] = self.backend
        self.params["constraint_mode"] = self.constraint_mode

        self.constraints = self._convert_constraints(self.raw_constraints)
        self.initial_calc = atoms.calc
        self._threshold_attrs = ["f_max_th", "f_rms_th", "dp_max_th", "dp_rms_th"]
        self._initial_thresholds = {attr: getattr(self.atoms, attr, 1.0e10) for attr in self._threshold_attrs}
        self._adj = self._build_connectivity(self.atoms)
        self.xyz_file = None

    def _log_info(self, info) -> None:
        if isinstance(info, str):
            lines = [info]
        else:
            lines = list(info)
        with open(self.output, "a", encoding="utf-8") as handle:
            for line in lines:
                handle.write(str(line))

    def _convert_constraints(self, original_constraints: list) -> list:
        converted = []
        for item in original_constraints:
            if len(item) == 4:
                a1, a2, step, steps = item
                converted.append({"type": "distance", "atoms": [a1, a2], "step": step, "steps": steps})
            elif len(item) == 5:
                a1, a2, a3, step, steps = item
                converted.append({"type": "angle", "atoms": [a1, a2, a3], "step": step, "steps": steps})
            elif len(item) == 6:
                a1, a2, a3, a4, step, steps = item
                converted.append({"type": "dihedral", "atoms": [a1, a2, a3, a4], "step": step, "steps": steps})
            else:
                raise ValueError(f"Unsupported constraint format: {item}")
        return converted

    def _generate_scan_values(self) -> list[list[float]]:
        scan_values: list[list[float]] = []
        for constraint in self.constraints:
            atoms_idx = [atom - 1 for atom in constraint["atoms"]]
            if constraint["type"] == "distance":
                initial = self.atoms.get_distance(*atoms_idx)
            elif constraint["type"] == "angle":
                initial = self.atoms.get_angle(*atoms_idx)
            elif constraint["type"] == "dihedral":
                initial = self.atoms.get_dihedral(*atoms_idx)
            else:
                raise ValueError(f"Unknown constraint type: {constraint['type']}")
            scan_values.append([initial + i * constraint["step"] for i in range(constraint["steps"] + 1)])
        return scan_values

    def _build_fix_internals(self, current_values: list[float]) -> FixInternals:
        bonds, angles, dihedrals = [], [], []
        for idx, constraint in enumerate(self.constraints):
            atoms_idx = [atom - 1 for atom in constraint["atoms"]]
            value = current_values[idx]
            if constraint["type"] == "distance":
                bonds.append([value, atoms_idx])
            elif constraint["type"] == "angle":
                angles.append([value, atoms_idx])
            elif constraint["type"] == "dihedral":
                dihedrals.append([value, atoms_idx])
        return FixInternals(
            bonds=bonds if bonds else None,
            angles_deg=angles if angles else None,
            dihedrals_deg=dihedrals if dihedrals else None,
        )

    def _safe_copy(self, atoms: Atoms) -> Atoms:
        copied = atoms.copy()
        copied.info = dict(atoms.info)
        copied.calc = atoms.calc if atoms.calc is not None else self.initial_calc
        for attr, value in self._initial_thresholds.items():
            setattr(copied, attr, value)
        return copied

    def _print_progress(self, idx: int, total: int, coord: list[float]) -> None:
        coord_str = "[" + ", ".join(f"{value:.2f}" for value in coord) + "]"
        self._log_info("\n")
        self._log_info("-" * 70)
        self._log_info(f"\n            Scanning combination {idx}/{total}: {coord_str}\n")

    def _set_constraint(self, atoms: Atoms, constraint) -> None:
        if hasattr(atoms, "set_constraint"):
            atoms.set_constraint(constraint)
            return
        setattr(atoms, "constraint", constraint)

    def _apply_constraints(self, atoms: Atoms, coord: list[float]) -> Atoms:
        self._set_constraint(atoms, self._build_fix_internals(coord))
        if atoms.calc is None:
            atoms.calc = self.initial_calc
        return atoms

    def _apply_rigid_geometry(self, atoms: Atoms, coord: list[float]) -> Atoms:
        self._set_constraint(atoms, None)
        for idx, constraint in enumerate(self.constraints):
            value = coord[idx]
            if constraint["type"] == "distance":
                mask, a0, a1 = self._get_rigid_mask(constraint)
                atoms.set_distance(a0, a1, value, fix=0, mask=mask)
            elif constraint["type"] == "angle":
                mask, a1, a2, a3 = self._get_rigid_mask(constraint)
                atoms.set_angle(a1, a2, a3, value, mask=mask)
            elif constraint["type"] == "dihedral":
                mask, a1, a2, a3, a4 = self._get_rigid_mask(constraint)
                atoms.set_dihedral(a1, a2, a3, a4, value, mask=mask)
            else:
                raise ValueError(f"Unknown constraint type: {constraint['type']}")
        if atoms.calc is None:
            atoms.calc = self.initial_calc
        return atoms

    def _optimizer_constraint_options(self) -> dict[str, list[tuple[int, ...]]]:
        options: dict[str, list[tuple[int, ...]]] = {
            "bond_constraints": [],
            "angle_constraints": [],
            "torsion_constraints": [],
        }
        for constraint in self.constraints:
            atoms = tuple(int(atom) for atom in constraint["atoms"])
            if constraint["type"] == "distance":
                options["bond_constraints"].append(atoms)
            elif constraint["type"] == "angle":
                options["angle_constraints"].append(atoms)
            elif constraint["type"] == "dihedral":
                options["torsion_constraints"].append(atoms)
            else:
                raise ValueError(f"Unknown constraint type: {constraint['type']}")
        return options

    def _build_optimizer_params(self, *, use_projection: bool = False) -> dict:
        params = deepcopy(self.params)
        opt = dict(params.get("opt", {}))
        opt["write_traj"] = False
        opt["verbose"] = 0
        if self.backend == "lbfgs":
            opt["use_projection"] = bool(use_projection)
            opt.setdefault("use_line_search", False)
        if use_projection:
            opt.update(self._optimizer_constraint_options())
            if self.backend == "lbfgs":
                opt["use_line_search"] = False
        params["opt"] = opt
        return params

    def _build_optimizer(self, atoms: Atoms, *, use_projection=False):
        params = self._build_optimizer_params(use_projection=use_projection)
        if self.backend == "lbfgs":
            return LBFGS(atoms, output=self.output, paras=params)
        if self.backend == "cgws":
            return CGWS(atoms, output=self.output, paras=params)
        return CGBS(atoms, output=self.output, paras=params)

    def _run_fixinternals_optimizer(self, atoms: Atoms) -> Atoms:
        if self.mode == "rigid":
            return atoms
        return self._build_optimizer(atoms).run()

    def _run_projected_optimizer(self, atoms: Atoms) -> Atoms:
        return self._build_optimizer(atoms, use_projection=True).run()

    def _record_result(self, atoms: Atoms, coord: list[float], coords_list: list, energies: list, *, energy=None) -> None:
        if energy is None:
            energy = float(atoms.get_potential_energy(force_consistent=True))
        positions = atoms.get_positions()
        symbols = atoms.get_chemical_symbols()

        self.xyz_file.write(f"{len(symbols)}\n")
        coord_str = "[" + ", ".join(f"{value:.4f}" for value in coord) + "]"
        self.xyz_file.write(
            f"Scanning combination {self._current_index}/{self._total_combinations}: {coord_str}  Energy = {energy:.10f}\n"
        )
        for symbol, (x, y, z) in zip(symbols, positions):
            self.xyz_file.write(f"{symbol:2s} {x: .10f} {y: .10f} {z: .10f}\n")
        self.xyz_file.flush()
        coords_list.append(coord[:])
        energies.append(energy)

        if self.mode == "rigid":
            info = [f'\n{"Coordinates".center(70)}\n', "-" * 70 + "\n"]
            for atom_index, atom in enumerate(atoms):
                x, y, z = atom.position
                info.append(f"{atom_index:<4} {atom.symbol:<2} {x:>20.4f} {y:>20.4f} {z:>20.4f}\n")
            info.append(f"\n\nEnergy:                {energy:>12.6f}\n")
            self._log_info(info)

    def _iter_scan_points(self, scan_values, prepare_node):
        if len(scan_values) == 1:
            x_values = scan_values[0]
            atoms_current = self._safe_copy(self.atoms)
            for value in x_values:
                coord = [value]
                self._current_index += 1
                self._print_progress(self._current_index, self._total_combinations, coord)
                atoms_current = prepare_node(atoms_current, coord)
                atoms_current = yield atoms_current, coord
            return
        elif len(scan_values) == 2:
            x_values, y_values = scan_values[0], scan_values[1]
            grid_xy = {}

            atoms_current = self._safe_copy(self.atoms)
            for ix, xv in enumerate(x_values):
                coord = [xv, y_values[0]]
                self._current_index += 1
                self._print_progress(self._current_index, self._total_combinations, coord)
                atoms_current = prepare_node(atoms_current, coord)
                atoms_current = yield atoms_current, coord
                grid_xy[(ix, 0)] = self._safe_copy(atoms_current)

            for ix, xv in enumerate(x_values):
                atoms_current = self._safe_copy(grid_xy[(ix, 0)])
                for iy in range(1, len(y_values)):
                    coord = [xv, y_values[iy]]
                    self._current_index += 1
                    self._print_progress(self._current_index, self._total_combinations, coord)
                    atoms_current = prepare_node(atoms_current, coord)
                    atoms_current = yield atoms_current, coord
            return
        elif len(scan_values) == 3:
            x_values, y_values, z_values = scan_values[0], scan_values[1], scan_values[2]
            grid_xy = {}

            atoms_current = self._safe_copy(self.atoms)
            for ix, xv in enumerate(x_values):
                coord = [xv, y_values[0], z_values[0]]
                self._current_index += 1
                self._print_progress(self._current_index, self._total_combinations, coord)
                atoms_current = prepare_node(atoms_current, coord)
                atoms_current = yield atoms_current, coord
                grid_xy[(ix, 0)] = self._safe_copy(atoms_current)

            for ix, xv in enumerate(x_values):
                atoms_current = self._safe_copy(grid_xy[(ix, 0)])
                for iy in range(1, len(y_values)):
                    coord = [xv, y_values[iy], z_values[0]]
                    self._current_index += 1
                    self._print_progress(self._current_index, self._total_combinations, coord)
                    atoms_current = prepare_node(atoms_current, coord)
                    atoms_current = yield atoms_current, coord
                    grid_xy[(ix, iy)] = self._safe_copy(atoms_current)

            for ix, xv in enumerate(x_values):
                for iy, yv in enumerate(y_values):
                    atoms_current = self._safe_copy(grid_xy[(ix, iy)])
                    for iz, zv in enumerate(z_values):
                        if iz == 0:
                            continue
                        coord = [xv, yv, zv]
                        self._current_index += 1
                        self._print_progress(self._current_index, self._total_combinations, coord)
                        atoms_current = prepare_node(atoms_current, coord)
                        atoms_current = yield atoms_current, coord
                    del grid_xy[(ix, iy)]
            return
        else:
            raise ValueError(f"Only 1D, 2D, 3D scans are supported, got {len(scan_values)}D")

    def _run_scan_points(self, scan_values, prepare_node, relax_node):
        coords_list, energies = [], []
        points = self._iter_scan_points(scan_values, prepare_node)
        try:
            atoms_current, coord = next(points)
            while True:
                atoms_current = relax_node(atoms_current)
                self._record_result(atoms_current, coord, coords_list, energies)
                atoms_current, coord = points.send(atoms_current)
        except StopIteration:
            return coords_list, energies
        finally:
            points.close()

    def _scan_1d(
        self,
        scan_values: list[list[float]],
        prepare_node: Callable[[Atoms, list[float]], Atoms],
        relax_node: Callable[[Atoms], Atoms],
    ) -> tuple[list, list]:
        return self._run_scan_points(scan_values, prepare_node, relax_node)

    def _scan_2d(
        self,
        scan_values: list[list[float]],
        prepare_node: Callable[[Atoms, list[float]], Atoms],
        relax_node: Callable[[Atoms], Atoms],
    ) -> tuple[list, list]:
        return self._run_scan_points(scan_values, prepare_node, relax_node)

    def _scan_3d(
        self,
        scan_values: list[list[float]],
        prepare_node: Callable[[Atoms, list[float]], Atoms],
        relax_node: Callable[[Atoms], Atoms],
    ) -> tuple[list, list]:
        return self._run_scan_points(scan_values, prepare_node, relax_node)

    def _identity_relax(self, atoms: Atoms) -> Atoms:
        return atoms

    def _prepare_scan(self, scan_values):
        self._total_combinations = 1
        for values in scan_values:
            self._total_combinations *= len(values)
        self._current_index = 0
        base, _ = os.path.splitext(self.output)
        xyz_filename = base + "_scan_final.xyz"
        with open(self.output, "w", encoding="utf-8"):
            pass
        self.xyz_file = open(xyz_filename, "w", encoding="utf-8")
        return xyz_filename

    def _finish_scan(self, xyz_filename, energies):
        self._log_info("\n")
        self._log_info("=" * 70)
        self._log_info(f"\nScan completed! Total points: {len(energies)}")
        self._log_info(f"Results saved to: {xyz_filename}\n")
        self._log_info(f"Energy range: {min(energies):.6f} to {max(energies):.6f} eV\n")

    def _close_scan(self):
        if self.xyz_file is not None:
            self.xyz_file.close()
            self.xyz_file = None
        self._cleanup_opt_files(self.output)

    def _run_internal_scan(
        self,
        scan_values: list[list[float]],
        *,
        prepare_node: Callable[[Atoms, list[float]], Atoms],
        relax_node: Callable[[Atoms], Atoms],
    ) -> str:
        dim = len(scan_values)
        xyz_filename = self._prepare_scan(scan_values)
        try:
            if dim == 1:
                _, energies = self._scan_1d(scan_values, prepare_node, relax_node)
            elif dim == 2:
                _, energies = self._scan_2d(scan_values, prepare_node, relax_node)
            elif dim == 3:
                _, energies = self._scan_3d(scan_values, prepare_node, relax_node)
            else:
                raise ValueError(f"Only 1D, 2D, 3D scans are supported, got {dim}D")
            self._finish_scan(xyz_filename, energies)
        finally:
            self._close_scan()
        return xyz_filename

    def _run_fixinternals_scan(self) -> str:
        scan_values = self._generate_scan_values()
        prepare_node = self._apply_rigid_geometry if self.mode == "rigid" else self._apply_constraints
        relax_node = self._identity_relax if self.mode == "rigid" else self._run_fixinternals_optimizer
        return self._run_internal_scan(
            scan_values,
            prepare_node=prepare_node,
            relax_node=relax_node,
        )

    def _run_projected_scan(self) -> str:
        if self.mode != "relaxed":
            raise ValueError("projected constraint_mode only supports relaxed scans.")
        scan_values = self._generate_scan_values()
        return self._run_internal_scan(
            scan_values,
            prepare_node=self._apply_rigid_geometry,
            relax_node=self._run_projected_optimizer,
        )

    def run(self) -> str:
        if self.constraint_mode == "projected":
            return self._run_projected_scan()
        return self._run_fixinternals_scan()

    @staticmethod
    def _cleanup_opt_files(output_path):
        base, _ = os.path.splitext(str(output_path))
        for path in (base + "_opt.xyz", base + "_traj.xyz", base + "_opt_traj.xyz"):
            if os.path.exists(path):
                os.unlink(path)

    def _build_connectivity(self, atoms: Atoms):
        cutoffs = natural_cutoffs(atoms)
        nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
        nl.update(atoms)
        adjacency = [set() for _ in range(len(atoms))]
        for i in range(len(atoms)):
            neigh, _ = nl.get_neighbors(i)
            for j in neigh:
                j = int(j)
                adjacency[i].add(j)
                adjacency[j].add(i)
        return adjacency

    def _fragment(self, start: int, blocked_edge=None):
        u, v = blocked_edge if blocked_edge else (None, None)
        stack = [start]
        seen = {start}
        while stack:
            i = stack.pop()
            for j in self._adj[i]:
                if blocked_edge is not None and ((i == u and j == v) or (i == v and j == u)):
                    continue
                if j not in seen:
                    seen.add(j)
                    stack.append(j)
        return seen

    @staticmethod
    def _mask_from_set(n: int, idx_set):
        mask = [False] * n
        for idx in idx_set:
            mask[idx] = True
        return mask

    def _get_rigid_mask(self, constraint):
        n = len(self.atoms)
        atoms_idx = [atom - 1 for atom in constraint["atoms"]]
        if constraint["type"] == "distance":
            a0, a1 = atoms_idx
            blocked = (a0, a1) if a1 in self._adj[a0] else None
            frag = self._fragment(start=a1, blocked_edge=blocked)
            return self._mask_from_set(n, frag), a0, a1
        if constraint["type"] == "angle":
            a1, a2, a3 = atoms_idx
            blocked = (a2, a3) if a3 in self._adj[a2] else None
            frag = self._fragment(start=a3, blocked_edge=blocked)
            return self._mask_from_set(n, frag), a1, a2, a3
        if constraint["type"] == "dihedral":
            a1, a2, a3, a4 = atoms_idx
            blocked = (a2, a3) if a3 in self._adj[a2] else None
            frag = self._fragment(start=a4, blocked_edge=blocked)
            return self._mask_from_set(n, frag), a1, a2, a3, a4
        raise ValueError(constraint["type"])
