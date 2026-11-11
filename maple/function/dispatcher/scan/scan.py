import os
from contextlib import nullcontext
from typing import List, Optional

from ase import Atoms
from maple.function.calculator._inputs import _state
from ase.constraints import FixInternals
from ase.neighborlist import NeighborList, natural_cutoffs

from ..jobABC import JobABC

from maple.function.timer import timer
from maple.function.read.filereader.pdb_reader import write_pdb_model, write_pdb_trajectory


def write_xyz(filename, atoms_list: List[Atoms],
              energies: Optional[List[float]] = None,
              mode: str = "w",
              start_index: int = 0,
              forces=None,
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


class Scan(JobABC):
    """
    N-dimensional relaxed/rigid scan (supports 1D, 2D, 3D).
    Uses hierarchical continuous scanning strategy with streaming output.
    """

    def __init__(self, output: str, atoms: Atoms, method: str = "lbfgs", 
                 constraints: Optional[list] = None, params: Optional[dict] = None):
        super().__init__(output)
        self.atoms = atoms
        self.output = output
        self.params = params if params is not None else {}
        self.initial_calc = atoms.calc
        self.method = str(method or self.params.get("method") or "lbfgs").lower()
        self.mode = str(self.params.get("mode", "relaxed")).lower()
        if self.mode not in ("relaxed", "rigid"):
            raise ValueError(f"mode must be 'relaxed' or 'rigid', got: {self.mode}")
        self._adj = self._build_connectivity(self.atoms)
        if constraints is None:
            raise ValueError("Constraints must be provided for scan.")
        self.constraints = self._convert_constraints(constraints)
        
        # Save initial custom threshold attributes
        self._threshold_attrs = ["f_max_th", "f_rms_th", "dp_max_th", "dp_rms_th"]
        self._initial_thresholds = {}
        for attr in self._threshold_attrs:
            self._initial_thresholds[attr] = getattr(self.atoms, attr, 1e10)
        
        # Initialize XYZ file handle
        self.xyz_file = None
        self._pdb_model_index = 0
        self.results = dict(coords_list=[], energies=[], iterations=[], converged=[], trajectory=None)

    def _convert_constraints(self, original_constraints: list) -> list:
        """Normalize constraint definitions."""
        converted = []
        for c in original_constraints:
            if len(c) == 4:     # distance: [a1, a2, step, steps]
                a1, a2, step, steps = c
                converted.append({
                    "type": "distance",
                    "atoms": [a1, a2],
                    "step": step,
                    "steps": steps
                })
            elif len(c) == 5:   # angle: [a1, a2, a3, step, steps]
                a1, a2, a3, step, steps = c
                converted.append({
                    "type": "angle",
                    "atoms": [a1, a2, a3],
                    "step": step,
                    "steps": steps
                })
            elif len(c) == 6:   # dihedral: [a1, a2, a3, a4, step, steps]
                a1, a2, a3, a4, step, steps = c
                converted.append({
                    "type": "dihedral",
                    "atoms": [a1, a2, a3, a4],
                    "step": step,
                    "steps": steps
                })
            else:
                raise ValueError(f"Unsupported constraint format: {c}")
        return converted

    def _generate_scan_values(self) -> List[List[float]]:
        """Generate scan grid values for each dimension."""
        scan_values = []
        for con in self.constraints:
            ctype = con["type"]
            step = con["step"]
            steps = con["steps"]
            # Convert atom indices to 0-based
            atoms_idx = [a - 1 for a in con["atoms"]]

            if ctype == "distance":
                initial = self.atoms.get_distance(*atoms_idx)
            elif ctype == "angle":
                initial = self.atoms.get_angle(*atoms_idx)
            elif ctype == "dihedral":
                initial = self.atoms.get_dihedral(*atoms_idx)
            else:
                raise ValueError(f"Unknown constraint type: {ctype}")

            values = [initial + i * step for i in range(steps + 1)]
            scan_values.append(values)
        return scan_values

    def _build_fix_internals(self, current_values: List[float]) -> FixInternals:
        """Build FixInternals constraint for given values."""
        bonds, angles, dihedrals = [], [], []

        for idx, con in enumerate(self.constraints):
            ctype = con["type"]
            atoms_idx = [a - 1 for a in con["atoms"]]  # convert to 0-based
            val = current_values[idx]

            if ctype == "distance":
                bonds.append([val, atoms_idx])
            elif ctype == "angle":
                angles.append([val, atoms_idx])
            elif ctype == "dihedral":
                dihedrals.append([val, atoms_idx])

        return FixInternals(
            bonds=bonds if bonds else None,
            angles_deg=angles if angles else None,
            dihedrals_deg=dihedrals if dihedrals else None
        )

    def _safe_copy(self, atoms: Atoms) -> Atoms:
        """
        Create a deep copy of Atoms with calculator and custom attributes restored.
        """
        new_atoms = atoms.copy()
        new_atoms.info = dict(atoms.info)
        
        # Restore calculator
        new_atoms.calc = atoms.calc if atoms.calc is not None else self.initial_calc
        
        # Restore custom threshold attributes
        for attr, val in self._initial_thresholds.items():
            setattr(new_atoms, attr, val)

        return new_atoms

    def _print_progress(self, idx: int, total: int, coord: List[float]):
        """Print progress header before calling optimizer."""
        coord_str = "[" + ", ".join(f"{v:.2f}" for v in coord) + "]"
        self.log_info(["\n"])
        self.log_info(["-" * 70])
        self.log_info([f"\n            Scanning combination {idx}/{total}: {coord_str}\n"])

    def _apply_constraints(self, atoms: Atoms, coord: List[float]) -> Atoms:
        """
        Apply FixInternals constraint to atoms (in-place modification).
        Returns the same atoms object with constraint applied.
        """
        constraint = self._build_fix_internals(coord)
        atoms.set_constraint(constraint)  # in-place
        
        # Ensure calculator remains valid
        if atoms.calc is None:
            atoms.calc = self.initial_calc
        
        return atoms

    def _apply_rigid_geometry(self, atoms: Atoms, coord: List[float]) -> Atoms:
        """ 
        Moving rigid body subgroup according to constraints
        Recongnize the moving fragment automatically.
        """
        atoms.set_constraint(None)
    
        for idx, con in enumerate(self.constraints):
            val = coord[idx]
            ctype = con["type"]
            if ctype == "distance":
                mask, a0, a1 = self._get_rigid_mask(con)
                atoms.set_distance(a0, a1, val, fix=0, mask=mask)  # fix a0, move a1 fragment
            elif ctype == "angle":
                mask, a1, a2, a3 = self._get_rigid_mask(con)
                atoms.set_angle(a1, a2, a3, val, mask=mask)
            elif ctype == "dihedral":
                mask, a1, a2, a3, a4 = self._get_rigid_mask(con)
                atoms.set_dihedral(a1, a2, a3, a4, val, mask=mask)
            else:
                raise ValueError(f"Unknown constraint type: {ctype}")
    
        if atoms.calc is None:
            atoms.calc = self.initial_calc
        return atoms

    def _run_optimizer(self, atoms: Atoms) -> Atoms:
        """Run geometry optimization."""
        if self.mode == "rigid":
            return atoms
        from maple.function.dispatcher.optimization import Optimization

        params = dict(self.params)
        params["method"] = self.method
        params["verbose"] = 0  # suppress optimizer output inside each scan point
        params["log_final_paths"] = False  # scan removes per-point optimizer temp files
        job = Optimization(params=params, output=self.output, atoms=atoms)
        atoms = job.run()
        self._opt_results = job.results
        return atoms

    def _record_result(self, atoms: Atoms, coord: List[float],
                       coords_list: list, energies: list):
        """Record a scan point result by streaming to file."""
        # The optimizer already evaluated this accepted geometry, including constraints.
        if self.mode == "relaxed":
            e = float(self._opt_results['energy'])
            forces = self._opt_results['forces']
            iterations = self._opt_results['iterations']
            converged = self._opt_results['converged']
        else:
            e = float(atoms.get_potential_energy(force_consistent=True))
            forces, iterations, converged = None, 0, None
        coord_str = "[" + ", ".join(f"{v:.4f}" for v in coord) + "]"
        comment = (
            f"Scanning combination {self._current_index}/{self._total_combinations}: "
            f"{coord_str}  Energy = {e:.10f}\n"
        )
        output_pdb = atoms.info.get("pdb_template")
        if output_pdb:
            self._pdb_model_index += 1
            write_pdb_model(
                self.xyz_file,
                atoms,
                output_pdb,
                model_index=self._pdb_model_index,
                remark=comment.strip(),
            )
        else:
            write_xyz(self.xyz_file, [atoms], energies=[e],
                      forces=[forces] if forces is not None else None,
                      comments=[f"Scanning combination {self._current_index}/{self._total_combinations}: {coord_str}"])
        self.xyz_file.flush()  # Ensure data is written
        
        # Store lightweight data
        coords_list.append(coord[:])
        energies.append(e)
        self.results['iterations'].append(iterations)
        self.results['converged'].append(converged)
        if self.mode == "relaxed" and not converged:
            self.log_info([f"Scan point {self._current_index}: NOT converged after {iterations} iterations.\n"])

        # For rigid scan, log coordinates to output
        if self.mode == "rigid":
            info = []
            info.append(f'\n{"Coordinates".center(70)}\n')
            info.append('-' * 70 + '\n')
            for atom_index, atom in enumerate(atoms):
                x, y, z = atom.position
                info.append(f"{atom_index:<4} {atom.symbol:<2} {x:>20.4f} {y:>20.4f} {z:>20.4f}\n")
            info.append(f"\n\nEnergy:                {e:>12.6f}\n")
            self.log_info(info)

    def _scan_1d(self, scan_values: List[List[float]]):
        """Execute 1D scan."""
        x_values = scan_values[0]
        coords_list, energies = [], []
        
        atoms_current = self._safe_copy(self.atoms)

        for xv in x_values:
            coord = [xv]
            self._current_index += 1
            self._print_progress(self._current_index, self._total_combinations, coord)
            if self.mode == "rigid":
                atoms_current = self._apply_rigid_geometry(atoms_current, coord)
            else:
                atoms_current = self._apply_constraints(atoms_current, coord)
            atoms_current = self._run_optimizer(atoms_current)
            self._record_result(atoms_current, coord, coords_list, energies)

        return coords_list, energies

    def _scan_2d(self, scan_values: List[List[float]]):
        """Execute 2D scan using hierarchical strategy."""
        x_values, y_values = scan_values[0], scan_values[1]
        coords_list, energies = [], []
        grid_xy = {}

        # Step 1: scan along X (y = y0) - keep this initial line
        atoms_current = self._safe_copy(self.atoms)
        for ix, xv in enumerate(x_values):
            coord = [xv, y_values[0]]
            self._current_index += 1
            self._print_progress(self._current_index, self._total_combinations, coord)

            if self.mode == "rigid":
                atoms_current = self._apply_rigid_geometry(atoms_current, coord)
            else:
                atoms_current = self._apply_constraints(atoms_current, coord)
            atoms_current = self._run_optimizer(atoms_current)
            
            grid_xy[(ix, 0)] = self._safe_copy(atoms_current)  # Keep initial line
            self._record_result(atoms_current, coord, coords_list, energies)

        # Step 2: for each X, scan along Y (no need to keep these)
        for ix, xv in enumerate(x_values):
            atoms_current = self._safe_copy(grid_xy[(ix, 0)])
            
            for iy in range(1, len(y_values)):
                coord = [xv, y_values[iy]]
                self._current_index += 1
                self._print_progress(self._current_index, self._total_combinations, coord)
                if self.mode == "rigid":
                    atoms_current = self._apply_rigid_geometry(atoms_current, coord)
                else:
                    atoms_current = self._apply_constraints(atoms_current, coord)
                atoms_current = self._run_optimizer(atoms_current)
                self._record_result(atoms_current, coord, coords_list, energies)

        return coords_list, energies

    def _scan_3d(self, scan_values: List[List[float]]):
        """Execute 3D scan using hierarchical strategy."""
        x_values, y_values, z_values = scan_values[0], scan_values[1], scan_values[2]
        coords_list, energies = [], []
        grid_xy = {}

        # Step 1: scan along X (y=y0, z=z0)
        atoms_current = self._safe_copy(self.atoms)
        for ix, xv in enumerate(x_values):
            coord = [xv, y_values[0], z_values[0]]
            self._current_index += 1
            self._print_progress(self._current_index, self._total_combinations, coord)

            if self.mode == "rigid":
                atoms_current = self._apply_rigid_geometry(atoms_current, coord)
            else:
                atoms_current = self._apply_constraints(atoms_current, coord)
            atoms_current = self._run_optimizer(atoms_current)
            
            grid_xy[(ix, 0)] = self._safe_copy(atoms_current)
            self._record_result(atoms_current, coord, coords_list, energies)

        # Step 2: scan along Y (z=z0) for each X - build the initial plane
        for ix, xv in enumerate(x_values):
            atoms_current = self._safe_copy(grid_xy[(ix, 0)])
            
            for iy in range(1, len(y_values)):
                coord = [xv, y_values[iy], z_values[0]]
                self._current_index += 1
                self._print_progress(self._current_index, self._total_combinations, coord)
                if self.mode == "rigid":
                    atoms_current = self._apply_rigid_geometry(atoms_current, coord)
                else:
                    atoms_current = self._apply_constraints(atoms_current, coord)
                atoms_current = self._run_optimizer(atoms_current)
                
                grid_xy[(ix, iy)] = self._safe_copy(atoms_current)  # Keep initial plane
                self._record_result(atoms_current, coord, coords_list, energies)
            
            # Keep y=y0 as a starting point for its Z branch below.

        # Step 3: scan along Z for each (X, Y)
        for ix, xv in enumerate(x_values):
            for iy, yv in enumerate(y_values):
                atoms_current = self._safe_copy(grid_xy[(ix, iy)])
                
                for iz, zv in enumerate(z_values):
                    # Skip z=z0 (already computed)
                    if iz == 0:
                        continue
                    
                    coord = [xv, yv, zv]
                    self._current_index += 1
                    self._print_progress(self._current_index, self._total_combinations, coord)
                    if self.mode == "rigid":
                        atoms_current = self._apply_rigid_geometry(atoms_current, coord)
                    else:
                        atoms_current = self._apply_constraints(atoms_current, coord)
                    atoms_current = self._run_optimizer(atoms_current)
                    self._record_result(atoms_current, coord, coords_list, energies)
                
                # Delete this (x,y) plane point after finishing its z-scan
                del grid_xy[(ix, iy)]

        return coords_list, energies

    def run_scan(self):
        """Main scan entry point."""
        scan_values = self._generate_scan_values()
        self.results = dict(coords_list=[], energies=[], iterations=[], converged=[], trajectory=None)
        dim = len(scan_values)

        total = 1
        for values in scan_values:
            total *= len(values)
        self._total_combinations = total
        self._current_index = 0

        # Open output structure file for streaming
        base, _ = os.path.splitext(self.output)
        output_pdb = self.atoms.info.get("pdb_template")
        xyz_filename = base + ("_scan_final.pdb" if output_pdb else "_scan_final.xyz")
        self.results['trajectory'] = xyz_filename
        self._pdb_model_index = 0
        
        try:
            self.xyz_file = open(xyz_filename, "w")
            
            if dim == 1:
                coords_list, energies = self._scan_1d(scan_values)
            elif dim == 2:
                coords_list, energies = self._scan_2d(scan_values)
            elif dim == 3:
                coords_list, energies = self._scan_3d(scan_values)
            else:
                raise ValueError(f"Only 1D, 2D, 3D scans are supported, got {dim}D")

            self.results.update(coords_list=coords_list, energies=energies)
            
            self.log_info(["\n"])
            self.log_info(["=" * 70])
            self.log_info([f"\nScan completed! Total points: {len(energies)}"])
            self.log_info([f"Results saved to: {xyz_filename}\n"])
            self.log_info([f"Energy range: {min(energies):.6f} to {max(energies):.6f} Hartree\n"])
            
        finally:
            if self.xyz_file is not None:
                if output_pdb:
                    self.xyz_file.write("END\n")
                self.xyz_file.close()

    def run(self):
        """JobABC interface."""
        with timer("Scan"):
            self.run_scan()
            self._cleanup_opt_files(self.output)  # cleanup opt temp files

  
    @staticmethod
    def _cleanup_opt_files(output_path):
        from pathlib import Path
        base, _ = os.path.splitext(str(output_path))
        for f in (
            base + "_opt.xyz",
            base + "_opt_traj.xyz",
            base + "_opt.pdb",
            base + "_opt_traj.pdb",
        ):
            Path(f).unlink(missing_ok=True)

    def _build_connectivity(self, atoms: Atoms):
        """Build adjacency list(bond graph) based on neighbor list."""
        cutoffs = natural_cutoffs(atoms)
        nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
        nl.update(atoms)
    
        n = len(atoms)
        adj = [set() for _ in range(n)]
        for i in range(n):
            neigh, _ = nl.get_neighbors(i)
            for j in neigh:
                j = int(j)
                adj[i].add(j)
                adj[j].add(i)
        return adj
    
    def _fragment(self, start: int, blocked_edge=None):
        """Find connected fragment from start, optionally blocking an edge."""
        # blocked_edge: tuple(u, v) meaning forbid traversing u<->v
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
    
    def _mask_from_set(self, n, idx_set):
        mask = [False] * n
        for i in idx_set:
            mask[i] = True
        return mask
    
    def _get_rigid_mask(self, con):
        n = len(self.atoms)
        atoms_idx = [a - 1 for a in con["atoms"]]
        ctype = con["type"]
    
        if ctype == "distance":
            a0, a1 = atoms_idx
            blocked = (a0, a1) if a1 in self._adj[a0] else None
            frag = self._fragment(start=a1, blocked_edge=blocked)
            return self._mask_from_set(n, frag), a0, a1
    
        if ctype == "angle":
            a1, a2, a3 = atoms_idx
            blocked = (a2, a3) if a3 in self._adj[a2] else None
            frag = self._fragment(start=a3, blocked_edge=blocked)
            return self._mask_from_set(n, frag), a1, a2, a3
    
        if ctype == "dihedral":
            a1, a2, a3, a4 = atoms_idx
            blocked = (a2, a3) if a3 in self._adj[a2] else None
            frag = self._fragment(start=a4, blocked_edge=blocked)
            return self._mask_from_set(n, frag), a1, a2, a3, a4
    
        raise ValueError(ctype)
    
