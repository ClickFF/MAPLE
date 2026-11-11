"""Usage: implement projected optimizers for constrained silent scans."""

from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np
from ase import Atoms

from ....jobABC import JobABC


# ============================================================================
# ### Projected helpers
# ============================================================================

def log_info(info, output: str) -> None:
    lines = [info] if isinstance(info, str) else list(info)
    with open(output, "a", encoding="utf-8") as handle:
        for line in lines:
            handle.write(str(line))

@dataclass(frozen=True)
class ConstraintSet:
    bonds: tuple[tuple[int, int], ...] = ()
    angles: tuple[tuple[int, int, int], ...] = ()
    torsions: tuple[tuple[int, int, int, int], ...] = ()

    @property
    def has_any(self) -> bool:
        return bool(self.bonds or self.angles or self.torsions)


def write_xyz(filename: str, atoms_list: Sequence[Atoms], energies: Sequence[float] | None = None) -> None:
    with open(filename, "w", encoding="utf-8") as handle:
        for index, atoms in enumerate(atoms_list):
            positions = np.asarray(atoms.get_positions(), dtype=float)
            symbols = atoms.get_chemical_symbols()
            handle.write(f"{len(symbols)}\n")
            if energies is not None:
                handle.write(f"Image {index}  Energy = {float(energies[index]):.10f}\n")
            else:
                handle.write(f"Image {index}\n")
            for symbol, (x, y, z) in zip(symbols, positions):
                handle.write(f"{symbol:2s} {x: .10f} {y: .10f} {z: .10f}\n")


def normalize_constraints(
    bond_constraints: Iterable[Sequence[int]] | None = None,
    angle_constraints: Iterable[Sequence[int]] | None = None,
    torsion_constraints: Iterable[Sequence[int]] | None = None,
) -> ConstraintSet:
    return ConstraintSet(
        bonds=_normalize_constraint_block(bond_constraints, 2),
        angles=_normalize_constraint_block(angle_constraints, 3),
        torsions=_normalize_constraint_block(torsion_constraints, 4),
    )


def clip_step(step_cart: np.ndarray, max_step: float) -> np.ndarray:
    clipped = np.asarray(step_cart, dtype=float).copy()
    if max_step <= 0.0:
        return clipped
    max_disp = float(np.max(np.abs(clipped)))
    if max_disp > max_step:
        clipped *= max_step / max_disp
    return clipped


def clip_step_by_atom_norm(step_cart: np.ndarray, max_step: float) -> np.ndarray:
    clipped = np.asarray(step_cart, dtype=float).copy()
    if max_step <= 0.0 or clipped.size == 0:
        return clipped
    atom_steps = clipped.reshape((-1, 3))
    max_atom_step = float(np.linalg.norm(atom_steps, axis=1).max())
    if max_atom_step > max_step:
        clipped *= max_step / max_atom_step
    return clipped


def angle_local_components(
    positions: np.ndarray,
    forces: np.ndarray,
    angle: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = np.asarray(positions, dtype=float)
    vecs = np.asarray(forces, dtype=float)
    i, j, k = _ensure_tuple(angle, 3)
    frame_i = _axis_frame(coords[i] - coords[j])
    frame_k = _axis_frame(coords[k] - coords[j])
    return frame_i @ vecs[i], vecs[j].copy(), frame_k @ vecs[k]


def torsion_local_components(
    positions: np.ndarray,
    forces: np.ndarray,
    torsion: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coords = np.asarray(positions, dtype=float)
    vecs = np.asarray(forces, dtype=float)
    a, b, c, d = _ensure_tuple(torsion, 4)
    base_frame, rel_local = _torsion_base_frame(coords, (a, b, c, d))
    force_local = vecs @ base_frame.T
    rot_a = _rotation_x_to_xy(rel_local[a])
    rot_d = _rotation_x_to_xy(rel_local[d])
    return (
        rot_a @ force_local[a],
        force_local[b].copy(),
        force_local[c].copy(),
        rot_d @ force_local[d],
    )


def project_forces(
    positions: np.ndarray,
    forces: np.ndarray,
    constraints: ConstraintSet,
) -> np.ndarray:
    coords = np.asarray(positions, dtype=float)
    projected = np.asarray(forces, dtype=float).copy()

    bonded_atoms: set[int] = set()
    for i, j in constraints.bonds:
        bonded_atoms.add(i)
        bonded_atoms.add(j)
        projected[i] = 0.0
        projected[j] = 0.0

    for angle in constraints.angles:
        projected = _project_angle_forces(coords, projected, angle)

    has_multiple_torsions = len(constraints.torsions) > 1
    for torsion in constraints.torsions:
        projected = _project_torsion_forces(coords, projected, torsion)
        if has_multiple_torsions:
            projected[torsion[2]] = 0.0

    for index in bonded_atoms:
        projected[index] = 0.0

    return projected


def project_force(
    *args,
    force: np.ndarray | None = None,
    coordinates: np.ndarray | None = None,
    bond_constraints: Iterable[Sequence[int]] | None = None,
    angle_constraints: Iterable[Sequence[int]] | None = None,
    torsion_constraints: Iterable[Sequence[int]] | None = None,
) -> np.ndarray:
    if force is not None and coordinates is not None:
        raw_force = np.asarray(force, dtype=float)
        raw_coords = np.asarray(coordinates, dtype=float)
    elif len(args) == 2:
        raw_force = np.asarray(args[0], dtype=float)
        raw_coords = np.asarray(args[1], dtype=float)
    elif len(args) == 5:
        bond_constraints, angle_constraints, torsion_constraints, raw_force, raw_coords = args
        raw_force = np.asarray(raw_force, dtype=float)
        raw_coords = np.asarray(raw_coords, dtype=float)
    else:
        raise TypeError("Unsupported project_force call signature.")

    squeezed_force = np.asarray(raw_force, dtype=float).squeeze()
    squeezed_coords = np.asarray(raw_coords, dtype=float).squeeze()
    constraints = normalize_constraints(bond_constraints, angle_constraints, torsion_constraints)
    projected = project_forces(squeezed_coords, squeezed_force, constraints)
    return projected.reshape(raw_force.shape)


force_projection = project_force
projected_force = project_force
apply_force_projection = project_force


def get_energy_and_forces(
    atoms: Atoms,
    constraints: ConstraintSet,
    *,
    use_projection: bool,
) -> tuple[float, np.ndarray]:
    energy = _get_potential_energy(atoms)
    forces = _get_forces(atoms)
    if use_projection and constraints.has_any:
        forces = project_forces(np.asarray(atoms.get_positions(), dtype=float), forces, constraints)
    return energy, forces


def _run_steps(steps):
    """Drive the same numerical iteration with scalar calculator getters."""
    try:
        atoms, constraints, use_projection = next(steps)
        while True:
            values = get_energy_and_forces(atoms, constraints, use_projection=use_projection)
            atoms, constraints, use_projection = steps.send(values)
    except StopIteration as done:
        return done.value
    finally:
        steps.close()


def _get_potential_energy(atoms: Atoms) -> float:
    if hasattr(atoms, "get_potential_energy"):
        try:
            return float(atoms.get_potential_energy(force_consistent=True))
        except AttributeError:
            pass
    calc = getattr(atoms, "calc", None)
    if calc is not None and hasattr(calc, "get_potential_energy"):
        return float(calc.get_potential_energy(atoms, force_consistent=True))
    raise ValueError("Scan optimizer requires atoms.calc with potential energy evaluation.")


def _get_forces(atoms: Atoms) -> np.ndarray:
    if hasattr(atoms, "get_forces"):
        try:
            return np.asarray(atoms.get_forces(), dtype=float)
        except AttributeError:
            pass
    calc = getattr(atoms, "calc", None)
    if calc is not None and hasattr(calc, "get_forces"):
        return np.asarray(calc.get_forces(atoms), dtype=float)
    raise ValueError("Scan optimizer requires atoms.calc with force evaluation.")


def wolfe_line_search(
    atoms: Atoms,
    direction: np.ndarray,
    energy0: float,
    forces0: np.ndarray,
    constraints: ConstraintSet,
    *,
    use_projection: bool,
    max_step: float,
    alpha0: float = 0.5,
    c1: float = 1.0e-4,
    c2: float = 0.49,
    stpmax: float = 10.0,
    max_iter: int = 24,
) -> tuple[float | None, float, np.ndarray]:
    return _TorchStyleLineSearch(xtol=1.0e-14).run(
        atoms,
        direction,
        energy0,
        forces0,
        constraints,
        use_projection=use_projection,
        max_step=max_step,
        alpha0=alpha0,
        c1=c1,
        c2=c2,
        stpmax=stpmax,
        max_iter=max_iter,
    )


def _iter_wolfe_line_search(
    batch, atoms, direction, energy0, forces0, constraints, *, use_projection,
    max_step, alpha0=0.5, c1=1.0e-4, c2=0.49, stpmax=10.0, max_iter=24,
):
    if not batch:
        return strong_wolfe_line_search(
            atoms, direction, energy0, forces0, constraints,
            use_projection=use_projection, max_step=max_step, alpha0=alpha0,
            c1=c1, c2=c2, stpmax=stpmax, max_iter=max_iter,
        )
    return (yield from _TorchStyleLineSearch(xtol=1.0e-14)._iterate(
        atoms, direction, energy0, forces0, constraints,
        use_projection=use_projection, max_step=max_step, alpha0=alpha0,
        c1=c1, c2=c2, stpmax=stpmax, max_iter=max_iter,
    ))


strong_wolfe_line_search = wolfe_line_search


def _normalize_constraint_block(
    constraints: Iterable[Sequence[int]] | None,
    size: int,
) -> tuple[tuple[int, ...], ...]:
    if constraints is None:
        return ()
    normalized = []
    for item in constraints:
        values = tuple(int(value) for value in item)
        if len(values) != size:
            raise ValueError(f"Constraint {values!r} must contain {size} atoms.")
        if any(value <= 0 for value in values):
            raise ValueError(f"Constraint {values!r} must use 1-based positive atom indices.")
        normalized.append(tuple(value - 1 for value in values))
    return tuple(normalized)


def _ensure_tuple(values: Sequence[int], size: int) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if len(result) != size:
        raise ValueError(f"Expected {size} indices, got {values!r}")
    return result


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-14:
        raise ValueError("Degenerate geometry produced a zero-length vector during projection.")
    return np.asarray(vector, dtype=float) / norm


def _axis_frame(axis: np.ndarray) -> np.ndarray:
    ex = _unit(axis)
    helper = np.asarray((0.0, 0.0, 1.0), dtype=float)
    if abs(float(np.dot(ex, helper))) > 0.95:
        helper = np.asarray((0.0, 1.0, 0.0), dtype=float)
    ey = helper - np.dot(helper, ex) * ex
    ey = _unit(ey)
    ez = _unit(np.cross(ex, ey))
    return np.vstack((ex, ey, ez))


def _rotation_x_to_xy(vector: np.ndarray) -> np.ndarray:
    _, y, z = np.asarray(vector, dtype=float)
    yz_norm = float(np.hypot(y, z))
    if yz_norm <= 1.0e-14:
        return np.eye(3, dtype=float)
    return np.asarray(
        [
            (1.0, 0.0, 0.0),
            (0.0, y / yz_norm, z / yz_norm),
            (0.0, -z / yz_norm, y / yz_norm),
        ],
        dtype=float,
    )


def _project_angle_forces(
    positions: np.ndarray,
    forces: np.ndarray,
    angle: Sequence[int],
) -> np.ndarray:
    projected = np.asarray(forces, dtype=float).copy()
    i, j, k = _ensure_tuple(angle, 3)
    frame_i = _axis_frame(np.asarray(positions[i], dtype=float) - np.asarray(positions[j], dtype=float))
    local_i = frame_i @ projected[i]
    local_i[1:] = 0.0
    projected[i] = frame_i.T @ local_i
    projected[j] = 0.0
    frame_k = _axis_frame(np.asarray(positions[k], dtype=float) - np.asarray(positions[j], dtype=float))
    local_k = frame_k @ projected[k]
    local_k[1:] = 0.0
    projected[k] = frame_k.T @ local_k
    return projected


def _torsion_base_frame(
    positions: np.ndarray,
    torsion: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    a, b, c, d = _ensure_tuple(torsion, 4)
    coords = np.asarray(positions, dtype=float)
    rel = coords - coords[c]
    ex = _unit(coords[b] - coords[c])
    d_vec = coords[d] - coords[c]
    d_orth = d_vec - np.dot(d_vec, ex) * ex
    ey = _unit(d_orth)
    ez = _unit(np.cross(ex, ey))
    frame = np.vstack((ex, ey, ez))
    return frame, rel @ frame.T


def _project_torsion_forces(
    positions: np.ndarray,
    forces: np.ndarray,
    torsion: Sequence[int],
) -> np.ndarray:
    projected = np.asarray(forces, dtype=float).copy()
    a, b, c, d = _ensure_tuple(torsion, 4)
    frame, rel_local = _torsion_base_frame(positions, (a, b, c, d))
    force_local = projected @ frame.T

    force_local[b, 1:] = 0.0
    force_local[c, 1:] = 0.0
    rot_a = _rotation_x_to_xy(rel_local[a])
    local_a = rot_a @ force_local[a]
    local_a[2] = 0.0
    force_local[a] = rot_a.T @ local_a
    rot_d = _rotation_x_to_xy(rel_local[d])
    local_d = rot_d @ force_local[d]
    local_d[2] = 0.0
    force_local[d] = rot_d.T @ local_d

    return force_local @ frame


class _TorchStyleLineSearch:
    def __init__(self, xtol: float = 1.0e-14) -> None:
        self.xtol = float(xtol)
        self.task = "START"
        self.isave = np.zeros((2,), np.intc)
        self.dsave = np.zeros((13,), dtype=float)
        self.fc = 0
        self.gc = 0
        self.case = 0
        self.old_stp = 0.0
        self.no_update = False
        self.bracket = False
        self.stpmin = 1.0e-8
        self.stpmax = 0.0
        self.xtrapl = 1.1
        self.xtrapu = 4.0
        self.maxstep = 0.0
        self.pk = np.zeros((0,), dtype=float)

    def run(
        self,
        atoms: Atoms,
        direction: np.ndarray,
        energy0: float,
        forces0: np.ndarray,
        constraints: ConstraintSet,
        *,
        use_projection: bool,
        max_step: float,
        alpha0: float,
        c1: float,
        c2: float,
        stpmax: float,
        max_iter: int,
    ) -> tuple[float | None, float, np.ndarray]:
        return _run_steps(self._iterate(
            atoms, direction, energy0, forces0, constraints,
            use_projection=use_projection, max_step=max_step, alpha0=alpha0,
            c1=c1, c2=c2, stpmax=stpmax, max_iter=max_iter,
        ))

    def _iterate(
        self,
        atoms: Atoms,
        direction: np.ndarray,
        energy0: float,
        forces0: np.ndarray,
        constraints: ConstraintSet,
        *,
        use_projection: bool,
        max_step: float,
        alpha0: float,
        c1: float,
        c2: float,
        stpmax: float,
        max_iter: int,
    ):
        origin = np.asarray(atoms.get_positions(), dtype=float).copy()
        self.task = "START"
        self.fc = 0
        self.gc = 0
        self.case = 0
        self.old_stp = 0.0
        self.no_update = False
        self.bracket = False
        self.isave.fill(0)
        self.dsave.fill(0.0)
        self.pk = np.asarray(direction, dtype=float).reshape(-1)
        self.maxstep = float(max_step)
        self.stpmax = float(stpmax)
        last_phi = float(energy0)
        last_force = np.asarray(forces0, dtype=float).copy()

        if self.pk.size == 0 or self.maxstep <= 0.0:
            return None, last_phi, last_force

        phi = float(energy0)
        derphi = -float(np.dot(last_force.reshape(-1), self.pk))
        alpha = float(alpha0)
        steps = 0

        try:
            while steps < max_iter:
                stp = self.step(alpha, phi, derphi, c1, c2)
                if self.task.startswith("FG"):
                    alpha = float(stp)
                    atoms.set_positions(origin + alpha * direction)
                    phi, force = yield atoms, constraints, use_projection
                    last_phi = phi
                    last_force = force
                    derphi = -float(np.dot(force.reshape(-1), self.pk))
                    self.fc += 1
                    self.gc += 1
                    self.old_stp = alpha
                    steps += 1
                    if self.no_update:
                        break
                    continue
                if self.task.startswith("CONVERGENCE"):
                    return float(stp), last_phi, last_force
                break
        finally:
            atoms.set_positions(origin)

        if self.task.startswith("ERROR") or self.task.startswith("WARNING"):
            return None, last_phi, last_force
        return None, last_phi, last_force

    def step(self, stp: float, f: float, g: float, c1: float, c2: float) -> float:
        if self.task.startswith("START"):
            if stp < self.stpmin:
                self.task = "ERROR: STP .LT. minstep"
            if stp > self.stpmax:
                self.task = "ERROR: STP .GT. maxstep"
            if g >= 0.0:
                self.task = "ERROR: INITIAL G >= 0"
            if c1 < 0.0:
                self.task = "ERROR: c1 .LT. 0"
            if c2 < 0.0:
                self.task = "ERROR: c2 .LT. 0"
            if self.xtol < 0.0:
                self.task = "ERROR: XTOL .LT. 0"
            if self.stpmin < 0.0:
                self.task = "ERROR: minstep .LT. 0"
            if self.stpmax < self.stpmin:
                self.task = "ERROR: maxstep .LT. minstep"
            if self.task.startswith("ERROR"):
                return float(stp)

            self.bracket = False
            stage = 1
            finit = float(f)
            ginit = float(g)
            gtest = c1 * ginit
            width = self.stpmax - self.stpmin
            width1 = width / 0.5
            stx = 0.0
            fx = finit
            gx = ginit
            sty = 0.0
            fy = finit
            gy = ginit
            stmin = 0.0
            stmax = stp + self.xtrapu * stp
            self.task = "FG"
            self.save((stage, ginit, gtest, gx, gy, finit, fx, fy, stx, sty, stmin, stmax, width, width1))
            return self.determine_step(float(stp))

        self.bracket = bool(self.isave[0] == 1)
        stage = int(self.isave[1])
        ginit, gtest, gx, gy, finit, fx, fy, stx, sty, stmin, stmax, width, width1 = self.dsave

        ftest = finit + stp * gtest
        if stage == 1 and f < ftest and g >= 0.0:
            stage = 2

        if self.bracket and (stp <= stmin or stp >= stmax):
            self.task = "WARNING: ROUNDING ERRORS PREVENT PROGRESS"
        if self.bracket and stmax - stmin <= self.xtol * stmax:
            self.task = "WARNING: XTOL TEST SATISFIED"
        if stp == self.stpmax and f <= ftest and g <= gtest:
            self.task = "WARNING: STP = maxstep"
        if stp == self.stpmin and (f > ftest or g >= gtest):
            self.task = "WARNING: STP = minstep"
        if f <= ftest and abs(g) <= c2 * (-ginit):
            self.task = "CONVERGENCE"
        if self.task.startswith("WARN") or self.task.startswith("CONV"):
            self.save((stage, ginit, gtest, gx, gy, finit, fx, fy, stx, sty, stmin, stmax, width, width1))
            return float(stp)

        stx, sty, stp, gx, fx, gy, fy = self.update(stx, fx, gx, sty, fy, gy, stp, f, g, stmin, stmax)
        if self.bracket:
            if abs(sty - stx) >= 0.66 * width1:
                stp = stx + 0.5 * (sty - stx)
            width1 = width
            width = abs(sty - stx)

        if self.bracket:
            stmin = min(stx, sty)
            stmax = max(stx, sty)
        else:
            stmin = stp + self.xtrapl * (stp - stx)
            stmax = stp + self.xtrapu * (stp - stx)

        stp = max(float(stp), self.stpmin)
        stp = min(stp, self.stpmax)
        if stx == stp and stp == self.stpmax and stmin > self.stpmax:
            self.no_update = True
        if (self.bracket and stp < stmin or stp >= stmax) or (
            self.bracket and stmax - stmin < self.xtol * stmax
        ):
            stp = stx

        self.task = "FG"
        self.save((stage, ginit, gtest, gx, gy, finit, fx, fy, stx, sty, stmin, stmax, width, width1))
        return float(stp)

    def update(
        self,
        stx: float,
        fx: float,
        gx: float,
        sty: float,
        fy: float,
        gy: float,
        stp: float,
        fp: float,
        gp: float,
        stpmin: float,
        stpmax: float,
    ) -> tuple[float, float, float, float, float, float, float]:
        sign = gp * np.sign(gx)

        if fp > fx:
            self.case = 1
            theta = 3.0 * (fx - fp) / (stp - stx) + gx + gp
            s = max(abs(theta), abs(gx), abs(gp))
            gamma = s * np.sqrt(max(0.0, (theta / s) ** 2 - (gx / s) * (gp / s)))
            if stp < stx:
                gamma = -gamma
            p = (gamma - gx) + theta
            q = ((gamma - gx) + gamma) + gp
            r = p / q
            stpc = stx + r * (stp - stx)
            stpq = stx + (gx / (((fx - fp) / (stp - stx)) + gx) / 2.0) * (stp - stx)
            if abs(stpc - stx) < abs(stpq - stx):
                stpf = stpc
            else:
                stpf = stpc + (stpq - stpc) / 2.0
            self.bracket = True
        elif sign < 0:
            self.case = 2
            theta = 3.0 * (fx - fp) / (stp - stx) + gx + gp
            s = max(abs(theta), abs(gx), abs(gp))
            gamma = s * np.sqrt(max(0.0, (theta / s) ** 2 - (gx / s) * (gp / s)))
            if stp > stx:
                gamma = -gamma
            p = (gamma - gp) + theta
            q = ((gamma - gp) + gamma) + gx
            r = p / q
            stpc = stp + r * (stx - stp)
            stpq = stp + (gp / (gp - gx)) * (stx - stp)
            if abs(stpc - stp) > abs(stpq - stp):
                stpf = stpc
            else:
                stpf = stpq
            self.bracket = True
        elif abs(gp) < abs(gx):
            self.case = 3
            theta = 3.0 * (fx - fp) / (stp - stx) + gx + gp
            s = max(abs(theta), abs(gx), abs(gp))
            gamma = s * np.sqrt(max(0.0, (theta / s) ** 2 - (gx / s) * (gp / s)))
            if stp > stx:
                gamma = -gamma
            p = (gamma - gp) + theta
            q = (gamma + (gx - gp)) + gamma
            r = p / q
            if r < 0.0 and gamma != 0.0:
                stpc = stp + r * (stx - stp)
            elif stp > stx:
                stpc = stpmax
            else:
                stpc = stpmin
            stpq = stp + (gp / (gp - gx)) * (stx - stp)
            if self.bracket:
                if abs(stpc - stp) < abs(stpq - stp):
                    stpf = stpc
                else:
                    stpf = stpq
                if stp > stx:
                    stpf = min(stp + 0.66 * (sty - stp), stpf)
                else:
                    stpf = max(stp + 0.66 * (sty - stp), stpf)
            else:
                if abs(stpc - stp) > abs(stpq - stp):
                    stpf = stpc
                else:
                    stpf = stpq
                stpf = min(stpmax, stpf)
                stpf = max(stpmin, stpf)
        else:
            self.case = 4
            if self.bracket:
                theta = 3.0 * (fp - fy) / (sty - stp) + gy + gp
                s = max(abs(theta), abs(gy), abs(gp))
                gamma = s * np.sqrt(max(0.0, (theta / s) ** 2 - (gy / s) * (gp / s)))
                if stp > sty:
                    gamma = -gamma
                p = (gamma - gp) + theta
                q = ((gamma - gp) + gamma) + gy
                r = p / q
                stpf = stp + r * (sty - stp)
            elif stp > stx:
                stpf = stpmax
            else:
                stpf = stpmin

        if fp > fx:
            sty = stp
            fy = fp
            gy = gp
        else:
            if sign < 0:
                sty = stx
                fy = fx
                gy = gx
            stx = stp
            fx = fp
            gx = gp

        stp = self.determine_step(stpf)
        return stx, sty, stp, gx, fx, gy, fy

    def determine_step(self, stp: float) -> float:
        dr = float(stp) - self.old_stp
        if self.maxstep <= 0.0 or self.pk.size == 0:
            return float(stp)
        step_vectors = dr * np.reshape(self.pk, (-1, 3))
        step_lengths = np.sqrt((step_vectors**2).sum(axis=1))
        max_step_length = float(np.max(step_lengths))
        if max_step_length >= self.maxstep and max_step_length > 0.0:
            dr *= self.maxstep / max_step_length
        return self.old_stp + dr

    def save(self, data: tuple[float, ...]) -> None:
        self.isave[0] = 1 if self.bracket else 0
        self.isave[1] = int(data[0])
        self.dsave[:] = data[1:]


# ============================================================================
# ### Projected LBFGS
# ============================================================================

@dataclass
class LBFGSParams:
    memory: int = 5
    curvature: float = 70.0
    max_step: float = 0.2
    max_iter: int = 256
    write_traj: bool = False
    traj_every: int = 1
    verbose: int = 1
    use_projection: bool = False
    use_line_search: bool = False
    bond_constraints: tuple[tuple[int, int], ...] | list[tuple[int, int]] = ()
    angle_constraints: tuple[tuple[int, int, int], ...] | list[tuple[int, int, int]] = ()
    torsion_constraints: tuple[tuple[int, int, int, int], ...] | list[tuple[int, int, int, int]] = ()


class LBFGS(JobABC):
    """
    L-BFGS optimizer for parmfit experiments.

    When `use_projection=True`, the optimizer follows the approximation for
    constrained motion by projecting both the current force field and the
    proposed search direction into the allowed subspace.
    """

    def __init__(
        self,
        atoms: Atoms,
        output: str,
        paras: Optional[dict] = None,
        params: LBFGSParams | None = None,
    ):
        super().__init__(output)
        self.atoms = atoms
        self.params = (
            deepcopy(params)
            if params is not None
            else self._init_params(LBFGSParams, paras, ("lbfgs", "LBFGS", "opt"))
        )
        self.constraints = normalize_constraints(
            bond_constraints=self.params.bond_constraints,
            angle_constraints=self.params.angle_constraints,
            torsion_constraints=self.params.torsion_constraints,
        )
        self.S: List[np.ndarray] = []
        self.Y: List[np.ndarray] = []
        self.rhos: List[float] = []
        self._last_iter_info: list[str] | None = None
        self.converged = False
        self._log_params()

    def _log_params(self) -> None:
        if self.params.verbose != 1:
            return
        param_info = [
            "\n" + "=" * 70 + "\n",
            "LBFGS Parameters\n",
            "=" * 70 + "\n",
            f"memory:          {self.params.memory}\n",
            f"curvature:       {self.params.curvature}\n",
            f"max_step:        {self.params.max_step}\n",
            f"max_iter:        {self.params.max_iter}\n",
            f"write_traj:      {self.params.write_traj}\n",
            f"traj_every:      {self.params.traj_every}\n",
            f"verbose:         {self.params.verbose}\n",
            f"use_projection:  {self.params.use_projection}\n",
            f"use_line_search: {self.params.use_line_search}\n",
            f"bond_constraints:{len(self.constraints.bonds):>12d}\n",
            f"angle_constraints:{len(self.constraints.angles):>11d}\n",
            f"torsion_constraints:{len(self.constraints.torsions):>9d}\n",
            "=" * 70 + "\n\n",
        ]
        log_info(param_info, self.output)

    def _two_loop(self, grad_flat: np.ndarray) -> np.ndarray:
        q = np.asarray(grad_flat, dtype=float).copy()
        alpha_list = []

        for s_vec, y_vec, rho in reversed(list(zip(self.S, self.Y, self.rhos))):
            alpha = rho * np.dot(s_vec, q)
            alpha_list.append(alpha)
            q -= alpha * y_vec

        if self.Y:
            gamma = np.dot(self.Y[-1], self.S[-1]) / (np.dot(self.Y[-1], self.Y[-1]) + 1.0e-20)
        else:
            gamma = 1.0 / self.params.curvature

        z = gamma * q
        for (s_vec, y_vec, rho), alpha in zip(zip(self.S, self.Y, self.rhos), reversed(alpha_list)):
            beta = rho * np.dot(y_vec, z)
            z += s_vec * (alpha - beta)

        return -z

    def _update_history(self, s_vec: np.ndarray, y_vec: np.ndarray) -> None:
        rho = 1.0 / (np.dot(y_vec, s_vec) + 1.0e-20)
        if np.isfinite(rho):
            self.S.append(np.asarray(s_vec, dtype=float).copy())
            self.Y.append(np.asarray(y_vec, dtype=float).copy())
            self.rhos.append(float(rho))
        if len(self.S) > self.params.memory:
            self.S.pop(0)
            self.Y.pop(0)
            self.rhos.pop(0)

    def _apply_direction_projection(self, positions: np.ndarray, direction: np.ndarray) -> np.ndarray:
        if not (self.params.use_projection and self.constraints.has_any):
            return np.asarray(direction, dtype=float)
        return project_forces(positions, np.asarray(direction, dtype=float), self.constraints)

    def _build_iter_message(self, iteration: int, energy: float, step_cart: np.ndarray, forces: np.ndarray) -> list[str]:
        atoms = self.atoms
        atoms.max_dp = np.abs(step_cart).max()
        atoms.rms_dp = np.sqrt((step_cart ** 2).sum() / step_cart.size)
        atoms.max_f = np.abs(forces).max()
        atoms.rms_f = np.sqrt((forces ** 2).sum() / step_cart.size)

        if self.params.verbose != 1:
            return []

        title = f"Iteration: {iteration}"
        info = ["\n" + "-" * 70 + "\n", f"{title.center(70)}\n\n"]
        info.append(f'\n{"Coordinates".center(70)}\n')
        info.append("-" * 70 + "\n")
        for atom_index, atom in enumerate(atoms):
            x, y, z = atom.position
            info.append(f"{atom_index:<4} {atom.symbol:<2} {x:>20.4f} {y:>20.4f} {z:>20.4f}\n")

        info.append(f"\n\nEnergy:                {energy:>12.6f} Convergence criteria  Is converged \n")
        info.append(
            f"Maximum Force:         {atoms.max_f:>12.6f} {atoms.f_max_th:>12.6f}  "
            f"{'Yes' if atoms.max_f <= atoms.f_max_th else 'No'}\n"
        )
        info.append(
            f"RMS Force:             {atoms.rms_f:>12.6f} {atoms.f_rms_th:>12.6f}  "
            f"{'Yes' if atoms.rms_f <= atoms.f_rms_th else 'No'}\n"
        )
        info.append(
            f"Maximum Displacement:  {atoms.max_dp:>12.6f} {atoms.dp_max_th:>12.6f}  "
            f"{'Yes' if atoms.max_dp <= atoms.dp_max_th else 'No'}\n"
        )
        info.append(
            f"RMS Displacement:      {atoms.rms_dp:>12.6f} {atoms.dp_rms_th:>12.6f}  "
            f"{'Yes' if atoms.rms_dp <= atoms.dp_rms_th else 'No'}\n"
        )
        return info

    def _check_convergence(self) -> bool:
        atoms = self.atoms
        return (
            atoms.max_f <= atoms.f_max_th
            and atoms.rms_f <= atoms.f_rms_th
            and atoms.max_dp <= atoms.dp_max_th
            and atoms.rms_dp <= atoms.dp_rms_th
        )

    def run(self) -> Atoms:
        return _run_steps(self._iterate())[0]

    def _iterate(self, *, batch=False):
        """Yield E/F requests while retaining the original numerical loop."""
        base, _ = os.path.splitext(self.output)
        traj_file = base + "_opt_traj.xyz"
        final_traj_file = base + "_traj.xyz"

        atoms = self.atoms
        positions = np.asarray(atoms.get_positions(), dtype=float)
        energy, forces = yield atoms, self.constraints, self.params.use_projection

        traj_atoms_list: list[Atoms] = [atoms.copy()]
        traj_energies_list: list[float] = [energy]

        if self.params.verbose == 1 and self.params.write_traj:
            write_xyz(traj_file, [atoms.copy()], energies=[energy])

        iteration = 0
        while iteration < self.params.max_iter:
            direction = self._two_loop(-forces.reshape(-1)).reshape(forces.shape)
            direction = self._apply_direction_projection(positions, direction)

            if self.params.use_line_search:
                alpha, _, _ = yield from _iter_wolfe_line_search(
                    batch,
                    atoms,
                    direction,
                    energy,
                    forces,
                    self.constraints,
                    use_projection=self.params.use_projection,
                    max_step=self.params.max_step,
                    alpha0=0.5,
                    c1=0.23,
                    c2=0.46,
                    stpmax=50.0,
                )
                if alpha is None:
                    break
            else:
                alpha = 1.0

            raw_step = alpha * direction
            if self.params.use_projection:
                step = clip_step_by_atom_norm(raw_step, self.params.max_step)
            else:
                step = clip_step(raw_step, self.params.max_step)
            positions_old = positions.copy()
            forces_old = forces.copy()

            atoms.set_positions(positions + step)
            positions = np.asarray(atoms.get_positions(), dtype=float)
            energy, forces = yield atoms, self.constraints, self.params.use_projection
            self._update_history((positions - positions_old).reshape(-1), (-forces - (-forces_old)).reshape(-1))
            iteration += 1

            traj_atoms_list.append(atoms.copy())
            traj_energies_list.append(energy)
            self._last_iter_info = self._build_iter_message(iteration, energy, step, forces)

            if self.params.verbose == 1:
                log_info(self._last_iter_info, self.output)
            if self.params.verbose == 1 and self.params.write_traj and iteration % self.params.traj_every == 0:
                write_xyz(traj_file, [atoms.copy()], energies=[energy])

            if self._check_convergence():
                self.converged = True
                self._write_finish_message(
                    converged=True,
                    iteration=iteration,
                    atoms=atoms,
                    energy=energy,
                    traj_atoms_list=traj_atoms_list,
                    traj_energies_list=traj_energies_list,
                    final_traj_file=final_traj_file,
                    opt_file=base + "_opt.xyz",
                )
                return atoms, energy, forces, iteration, True

        self.converged = False
        self._write_finish_message(
            converged=False,
            iteration=iteration,
            atoms=atoms,
            energy=energy,
            traj_atoms_list=traj_atoms_list,
            traj_energies_list=traj_energies_list,
            final_traj_file=final_traj_file,
            opt_file=base + "_opt.xyz",
        )
        return atoms, energy, forces, iteration, False

    def _write_finish_message(
        self,
        *,
        converged: bool,
        iteration: int,
        atoms: Atoms,
        energy: float,
        traj_atoms_list: list[Atoms],
        traj_energies_list: list[float],
        final_traj_file: str,
        opt_file: str,
    ) -> None:
        if self.params.write_traj:
            write_xyz(final_traj_file, traj_atoms_list, energies=traj_energies_list)
            write_xyz(opt_file, [atoms], energies=[energy])
        if self.params.verbose != 1:
            return
        log_info(self._last_iter_info or [], self.output)
        if converged:
            lines = [f"\nLBFGS converged at iteration {iteration}.\n"]
        else:
            lines = [f"\nLBFGS did NOT converge after {self.params.max_iter} iterations.\n"]
        if self.params.write_traj:
            lines.extend(
                [
                    f"Final frame written to {opt_file}\n",
                    f"Complete trajectory written to {final_traj_file}\n",
                ]
            )
        log_info(lines, self.output)

# ============================================================================
# ### CG_WS
# ============================================================================

@dataclass
class CGWSParams:
    max_step: float = 0.2
    max_iter: int = 256
    write_traj: bool = False
    traj_every: int = 1
    verbose: int = 1
    c1: float = 1.0e-4
    c2: float = 0.49
    stpmax: float = 10.0
    bond_constraints: tuple[tuple[int, int], ...] | list[tuple[int, int]] = ()
    angle_constraints: tuple[tuple[int, int, int], ...] | list[tuple[int, int, int]] = ()
    torsion_constraints: tuple[tuple[int, int, int, int], ...] | list[tuple[int, int, int, int]] = ()


class CGWS(JobABC):
    """Projected conjugate-gradient optimizer with Wolfe line search."""

    def __init__(self, atoms: Atoms, output: str, paras: Optional[dict] = None):
        super().__init__(output)
        self.atoms = atoms
        self.params = self._init_params(CGWSParams, paras, ("cgws", "CGWS", "cg_ws", "CG_WS", "opt"))
        self.constraints = normalize_constraints(
            bond_constraints=self.params.bond_constraints,
            angle_constraints=self.params.angle_constraints,
            torsion_constraints=self.params.torsion_constraints,
        )
        self._last_iter_info: list[str] | None = None
        self._log_params()

    def _log_params(self) -> None:
        param_info = [
            "\n" + "=" * 70 + "\n",
            "CG-WS Parameters\n",
            "=" * 70 + "\n",
            f"max_step:          {self.params.max_step}\n",
            f"max_iter:          {self.params.max_iter}\n",
            f"write_traj:        {self.params.write_traj}\n",
            f"traj_every:        {self.params.traj_every}\n",
            f"verbose:           {self.params.verbose}\n",
            f"c1:                {self.params.c1}\n",
            f"c2:                {self.params.c2}\n",
            f"stpmax:            {self.params.stpmax}\n",
            f"bond_constraints:  {len(self.constraints.bonds)}\n",
            f"angle_constraints: {len(self.constraints.angles)}\n",
            f"torsion_constraints:{len(self.constraints.torsions)}\n",
            "=" * 70 + "\n\n",
        ]
        log_info(param_info, self.output)

    def _build_iter_message(self, iteration: int, energy: float, step: np.ndarray, forces: np.ndarray) -> list[str]:
        atoms = self.atoms
        atoms.max_dp = np.abs(step).max()
        atoms.rms_dp = np.sqrt((step ** 2).sum() / step.size)
        atoms.max_f = np.abs(forces).max()
        atoms.rms_f = np.sqrt((forces ** 2).sum() / step.size)

        if self.params.verbose == 1:
            title = f"Iteration: {iteration}"
            info = ["\n" + "-" * 70 + "\n", f"{title.center(70)}\n\n"]
        else:
            info = []

        info.append(f'\n{"Coordinates".center(70)}\n')
        info.append("-" * 70 + "\n")
        for atom_index, atom in enumerate(atoms):
            x, y, z = atom.position
            info.append(f"{atom_index:<4} {atom.symbol:<2} {x:>20.4f} {y:>20.4f} {z:>20.4f}\n")

        info.append(f"\n\nEnergy:                {energy:>12.6f} Convergence criteria  Is converged \n")
        info.append(
            f"Maximum Force:         {atoms.max_f:>12.6f} {atoms.f_max_th:>12.6f}  "
            f"{'Yes' if atoms.max_f <= atoms.f_max_th else 'No'}\n"
        )
        info.append(
            f"RMS Force:             {atoms.rms_f:>12.6f} {atoms.f_rms_th:>12.6f}  "
            f"{'Yes' if atoms.rms_f <= atoms.f_rms_th else 'No'}\n"
        )
        info.append(
            f"Maximum Displacement:  {atoms.max_dp:>12.6f} {atoms.dp_max_th:>12.6f}  "
            f"{'Yes' if atoms.max_dp <= atoms.dp_max_th else 'No'}\n"
        )
        info.append(
            f"RMS Displacement:      {atoms.rms_dp:>12.6f} {atoms.dp_rms_th:>12.6f}  "
            f"{'Yes' if atoms.rms_dp <= atoms.dp_rms_th else 'No'}\n"
        )
        return info

    def _check_convergence(self) -> bool:
        atoms = self.atoms
        return (
            atoms.max_f <= atoms.f_max_th
            and atoms.rms_f <= atoms.f_rms_th
            and atoms.max_dp <= atoms.dp_max_th
            and atoms.rms_dp <= atoms.dp_rms_th
        )

    def run(self) -> Atoms:
        return _run_steps(self._iterate())[0]

    def _iterate(self, *, batch=False):
        """Yield E/F requests while retaining the original numerical loop."""
        base, _ = os.path.splitext(self.output)
        traj_file = base + "_opt_traj.xyz"
        final_traj_file = base + "_traj.xyz"

        atoms = self.atoms
        positions = np.asarray(atoms.get_positions(), dtype=float)
        energy, forces = yield atoms, self.constraints, True

        traj_atoms_list: list[Atoms] = [atoms.copy()]
        traj_energies_list: list[float] = [energy]
        if self.params.verbose == 1 and self.params.write_traj:
            write_xyz(traj_file, [atoms.copy()], energies=[energy])

        iteration = 0
        search_direction = forces.copy()

        while iteration < self.params.max_iter:
            alpha, _, _ = yield from _iter_wolfe_line_search(
                    batch,
                atoms,
                search_direction,
                energy,
                forces,
                self.constraints,
                use_projection=True,
                max_step=self.params.max_step,
                alpha0=0.5,
                c1=self.params.c1,
                c2=self.params.c2,
                stpmax=self.params.stpmax,
            )
            if alpha is None:
                break

            step = clip_step_by_atom_norm(alpha * search_direction, self.params.max_step)
            previous_forces = forces.copy()
            atoms.set_positions(positions + step)
            positions = np.asarray(atoms.get_positions(), dtype=float)
            energy, forces = yield atoms, self.constraints, True

            iteration += 1
            traj_atoms_list.append(atoms.copy())
            traj_energies_list.append(energy)
            self._last_iter_info = self._build_iter_message(iteration, energy, step, forces)

            if self.params.verbose == 1:
                log_info(self._last_iter_info, self.output)
            if self.params.verbose == 1 and self.params.write_traj and iteration % self.params.traj_every == 0:
                write_xyz(traj_file, [atoms.copy()], energies=[energy])

            if self._check_convergence():
                write_xyz(final_traj_file, traj_atoms_list, energies=traj_energies_list)
                opt_file = base + "_opt.xyz"
                write_xyz(opt_file, [atoms], energies=[energy])
                log_info(self._last_iter_info or [], self.output)
                log_info(
                    [
                        f"\nCG-WS converged at iteration {iteration}.\n"
                        f"Final frame written to {opt_file}\n"
                        f"Complete trajectory written to {final_traj_file}\n"
                    ],
                    self.output,
                )
                return atoms, energy, forces, iteration, True

            diff = forces - previous_forces
            denom = float(np.dot(previous_forces.reshape(-1), previous_forces.reshape(-1)))
            if denom > 1.0e-20:
                gamma = max(float(np.dot(diff.reshape(-1), forces.reshape(-1))) / denom, 0.0)
            else:
                gamma = 0.0
            search_direction = forces + gamma * search_direction

        write_xyz(final_traj_file, traj_atoms_list, energies=traj_energies_list)
        opt_file = base + "_opt.xyz"
        write_xyz(opt_file, [atoms], energies=[energy])
        log_info(self._last_iter_info or [], self.output)
        log_info(
            [
                f"\nCG-WS did NOT converge after {self.params.max_iter} iterations.\n"
                f"Final frame written to {opt_file}\n"
                f"Complete trajectory written to {final_traj_file}\n"
            ],
            self.output,
        )
        return atoms, energy, forces, iteration, False

# ============================================================================
# ### CG_BS
# ============================================================================

@dataclass
class CGBSParams:
    max_step: float = 0.2
    max_iter: int = 256
    write_traj: bool = False
    traj_every: int = 1
    verbose: int = 1
    alpha0: float = 0.2
    alpha_max: float = 0.8
    armijo_c1: float = 1.0e-3
    beta_shrink: float = 0.5
    alpha_grow: float = 1.01
    beta_min: float = 0.02
    max_backtracks: int = 32
    bond_constraints: tuple[tuple[int, int], ...] | list[tuple[int, int]] = ()
    angle_constraints: tuple[tuple[int, int, int], ...] | list[tuple[int, int, int]] = ()
    torsion_constraints: tuple[tuple[int, int, int, int], ...] | list[tuple[int, int, int, int]] = ()


class CGBS(JobABC):
    """Projected conjugate-gradient optimizer with Armijo backtracking."""

    def __init__(self, atoms: Atoms, output: str, paras: Optional[dict] = None):
        super().__init__(output)
        self.atoms = atoms
        self.params = self._init_params(CGBSParams, paras, ("cgbs", "CGBS", "cg_bs", "CG_BS", "opt"))
        self.constraints = normalize_constraints(
            bond_constraints=self.params.bond_constraints,
            angle_constraints=self.params.angle_constraints,
            torsion_constraints=self.params.torsion_constraints,
        )
        self._last_iter_info: list[str] | None = None
        self._log_params()

    def _log_params(self) -> None:
        param_info = [
            "\n" + "=" * 70 + "\n",
            "CG-BS Parameters\n",
            "=" * 70 + "\n",
            f"max_step:          {self.params.max_step}\n",
            f"max_iter:          {self.params.max_iter}\n",
            f"write_traj:        {self.params.write_traj}\n",
            f"traj_every:        {self.params.traj_every}\n",
            f"verbose:           {self.params.verbose}\n",
            f"alpha0:            {self.params.alpha0}\n",
            f"alpha_max:         {self.params.alpha_max}\n",
            f"armijo_c1:         {self.params.armijo_c1}\n",
            f"beta_shrink:       {self.params.beta_shrink}\n",
            f"alpha_grow:        {self.params.alpha_grow}\n",
            f"beta_min:          {self.params.beta_min}\n",
            f"max_backtracks:    {self.params.max_backtracks}\n",
            f"bond_constraints:  {len(self.constraints.bonds)}\n",
            f"angle_constraints: {len(self.constraints.angles)}\n",
            f"torsion_constraints:{len(self.constraints.torsions)}\n",
            "=" * 70 + "\n\n",
        ]
        log_info(param_info, self.output)

    def _project_direction(self, positions: np.ndarray, direction: np.ndarray) -> np.ndarray:
        direction = np.asarray(direction, dtype=float)
        if not self.constraints.has_any:
            return direction
        return project_forces(positions, direction, self.constraints)

    def _build_iter_message(self, iteration: int, energy: float, step: np.ndarray, forces: np.ndarray) -> list[str]:
        atoms = self.atoms
        atoms.max_dp = np.abs(step).max()
        atoms.rms_dp = np.sqrt((step ** 2).sum() / step.size)
        atoms.max_f = np.abs(forces).max()
        atoms.rms_f = np.sqrt((forces ** 2).sum() / step.size)

        if self.params.verbose == 1:
            title = f"Iteration: {iteration}"
            info = ["\n" + "-" * 70 + "\n", f"{title.center(70)}\n\n"]
        else:
            info = []

        info.append(f'\n{"Coordinates".center(70)}\n')
        info.append("-" * 70 + "\n")
        for atom_index, atom in enumerate(atoms):
            x, y, z = atom.position
            info.append(f"{atom_index:<4} {atom.symbol:<2} {x:>20.4f} {y:>20.4f} {z:>20.4f}\n")

        info.append(f"\n\nEnergy:                {energy:>12.6f} Convergence criteria  Is converged \n")
        info.append(
            f"Maximum Force:         {atoms.max_f:>12.6f} {atoms.f_max_th:>12.6f}  "
            f"{'Yes' if atoms.max_f <= atoms.f_max_th else 'No'}\n"
        )
        info.append(
            f"RMS Force:             {atoms.rms_f:>12.6f} {atoms.f_rms_th:>12.6f}  "
            f"{'Yes' if atoms.rms_f <= atoms.f_rms_th else 'No'}\n"
        )
        info.append(
            f"Maximum Displacement:  {atoms.max_dp:>12.6f} {atoms.dp_max_th:>12.6f}  "
            f"{'Yes' if atoms.max_dp <= atoms.dp_max_th else 'No'}\n"
        )
        info.append(
            f"RMS Displacement:      {atoms.rms_dp:>12.6f} {atoms.dp_rms_th:>12.6f}  "
            f"{'Yes' if atoms.rms_dp <= atoms.dp_rms_th else 'No'}\n"
        )
        return info

    def _check_convergence(self) -> bool:
        atoms = self.atoms
        return (
            atoms.max_f <= atoms.f_max_th
            and atoms.rms_f <= atoms.f_rms_th
            and atoms.max_dp <= atoms.dp_max_th
            and atoms.rms_dp <= atoms.dp_rms_th
        )

    def _backtracking_step(
        self,
        positions: np.ndarray,
        energy: float,
        forces: np.ndarray,
        direction: np.ndarray,
        alpha: float,
    ) -> tuple[np.ndarray | None, float, np.ndarray, float, bool]:
        return _run_steps(self._iterate_backtracking(
            positions, energy, forces, direction, alpha,
        ))

    def _iterate_backtracking(
        self,
        positions: np.ndarray,
        energy: float,
        forces: np.ndarray,
        direction: np.ndarray,
        alpha: float,
    ):
        base_step = clip_step_by_atom_norm(alpha * direction, self.params.max_step)
        if float(np.max(np.abs(base_step))) <= 0.0:
            return None, energy, forces.copy(), 1.0, False

        directional_gain = float(np.dot(forces.reshape(-1), base_step.reshape(-1)))
        if directional_gain <= 0.0:
            return None, energy, forces.copy(), 1.0, False

        trial_scale = 1.0
        was_shrunk = False
        attempts = 0
        while True:
            step = trial_scale * base_step

            self.atoms.set_positions(positions + step)
            trial_energy, trial_forces = yield self.atoms, self.constraints, True
            armijo_limit = energy - self.params.armijo_c1 * trial_scale * directional_gain
            if (
                trial_energy <= armijo_limit
                or trial_scale <= self.params.beta_min
                or attempts >= self.params.max_backtracks
            ):
                return step, trial_energy, trial_forces, trial_scale, was_shrunk

            self.atoms.set_positions(positions)
            trial_scale *= self.params.beta_shrink
            was_shrunk = True
            attempts += 1

    def run(self) -> Atoms:
        return _run_steps(self._iterate())[0]

    def _iterate(self, *, batch=False):
        """Yield E/F requests while retaining the original numerical loop."""
        base, _ = os.path.splitext(self.output)
        traj_file = base + "_opt_traj.xyz"
        final_traj_file = base + "_traj.xyz"

        atoms = self.atoms
        positions = np.asarray(atoms.get_positions(), dtype=float)
        energy, forces = yield atoms, self.constraints, True

        best_energy = energy
        best_positions = positions.copy()
        best_atoms = atoms.copy()

        traj_atoms_list: list[Atoms] = [atoms.copy()]
        traj_energies_list: list[float] = [energy]
        if self.params.verbose == 1 and self.params.write_traj:
            write_xyz(traj_file, [atoms.copy()], energies=[energy])

        alpha = min(max(self.params.alpha0, 0.0), self.params.alpha_max)
        iteration = 0
        search_direction = self._project_direction(positions, forces.copy())
        line_search_failed = False

        while iteration < self.params.max_iter:
            if float(np.dot(forces.reshape(-1), search_direction.reshape(-1))) <= 0.0:
                search_direction = self._project_direction(positions, forces.copy())

            if batch:
                step, trial_energy, trial_forces, trial_scale, was_shrunk = yield from self._iterate_backtracking(
                    positions,
                    energy,
                    forces,
                    search_direction,
                    alpha,
                )
            else:
                step, trial_energy, trial_forces, trial_scale, was_shrunk = self._backtracking_step(
                    positions,
                    energy,
                    forces,
                    search_direction,
                    alpha,
                )
            if step is None:
                line_search_failed = True
                alpha = max(alpha * max(trial_scale, self.params.beta_shrink), self.params.beta_min)
                break

            previous_forces = forces.copy()
            positions = np.asarray(atoms.get_positions(), dtype=float)
            energy = trial_energy
            forces = trial_forces
            iteration += 1

            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
                best_atoms = atoms.copy()

            alpha = min(alpha * self.params.alpha_grow, self.params.alpha_max)

            traj_atoms_list.append(atoms.copy())
            traj_energies_list.append(energy)
            self._last_iter_info = self._build_iter_message(iteration, energy, step, forces)

            if self.params.verbose == 1:
                log_info(self._last_iter_info, self.output)
            if self.params.verbose == 1 and self.params.write_traj and iteration % self.params.traj_every == 0:
                write_xyz(traj_file, [atoms.copy()], energies=[energy])

            if self._check_convergence():
                write_xyz(final_traj_file, traj_atoms_list, energies=traj_energies_list)
                opt_file = base + "_opt.xyz"
                write_xyz(opt_file, [atoms], energies=[energy])
                log_info(self._last_iter_info or [], self.output)
                log_info(
                    [
                        f"\nCG-BS converged at iteration {iteration}.\n"
                        f"Final frame written to {opt_file}\n"
                        f"Complete trajectory written to {final_traj_file}\n"
                    ],
                    self.output,
                )
                return atoms, energy, forces, iteration, True

            if was_shrunk:
                gamma = 0.0
            else:
                diff = forces - previous_forces
                denom = float(np.dot(previous_forces.reshape(-1), previous_forces.reshape(-1)))
                if denom > 1.0e-20:
                    gamma = max(float(np.dot(diff.reshape(-1), forces.reshape(-1))) / denom, 0.0)
                else:
                    gamma = 0.0
            search_direction = self._project_direction(positions, forces + gamma * search_direction)

        atoms.set_positions(best_positions)
        energy = best_energy
        forces = (yield (atoms, self.constraints, True))[1]

        write_xyz(final_traj_file, traj_atoms_list, energies=traj_energies_list)
        opt_file = base + "_opt.xyz"
        write_xyz(opt_file, [best_atoms], energies=[energy])
        log_info(self._last_iter_info or [], self.output)
        if line_search_failed:
            log_info(
                [
                    "\nCG-BS line search failed to find an Armijo-acceptable step.\n"
                    f"Best-so-far frame written to {opt_file}\n"
                    f"Complete trajectory written to {final_traj_file}\n"
                ],
                self.output,
            )
        else:
            log_info(
                [
                    f"\nCG-BS did NOT converge after {self.params.max_iter} iterations.\n"
                    f"Final frame written to {opt_file}\n"
                    f"Complete trajectory written to {final_traj_file}\n"
                ],
                self.output,
            )
        return atoms, energy, forces, iteration, False

