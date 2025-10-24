# -*- coding: utf-8 -*-
import os
from dataclasses import dataclass, fields
from typing import List, Optional

import numpy as np
from ase import Atoms
from .logger import log_info


# ==============================================
# Utility: write XYZ trajectory
# ==============================================
def write_xyz(filename: str, atoms_list: List[Atoms], energies: List[float] | None = None):
    """Write one or more structures in XYZ format."""
    with open(filename, "w") as f:
        for i, at in enumerate(atoms_list):
            pos = at.get_positions()
            symbols = at.get_chemical_symbols()
            f.write(f"{len(symbols)}\n")
            if energies is not None:
                f.write(f"Image {i}  Energy = {energies[i]:.10f}\n")
            else:
                f.write(f"Image {i}\n")
            for s, (x, y, z) in zip(symbols, pos):
                f.write(f"{s:2s} {x: .10f} {y: .10f} {z: .10f}\n")


# ==============================================
# Dict helpers (same as NEB)
# ==============================================
def _lower_keys(d):
    """Return a copy of dict with all string keys lowercased."""
    if not isinstance(d, dict):
        return {}
    return {(k.lower() if isinstance(k, str) else k): v for k, v in d.items()}

def _select_subdict(paras: dict, name_aliases: tuple[str, ...]) -> dict:
    """Extract a sub-dict using aliases (e.g., 'lbfgs', 'LBFGS')."""
    if not isinstance(paras, dict):
        return {}
    low = _lower_keys(paras)
    for alias in name_aliases:
        key = alias.lower()
        if key in low and isinstance(low[key], dict):
            return low[key]
    return low

def _update_dataclass_from_dict(dc_obj, d: dict):
    """Update a dataclass instance from a dict (case-insensitive keys)."""
    if not isinstance(d, dict):
        return dc_obj
    low = _lower_keys(d)
    field_map = {f.name.lower(): f.name for f in fields(dc_obj)}
    for k_low, v in low.items():
        if k_low in field_map:
            setattr(dc_obj, field_map[k_low], v)
    return dc_obj


# ==============================================
# LBFGS Parameters
# ==============================================
@dataclass
class LBFGSParams:
    memory: int = 5
    curvature: float = 70.0
    maxstep: float = 0.2
    maxiteration: int = 128
    write_traj: bool = False
    traj_every: int = 1


# ==============================================
# LBFGS Optimizer (NEB-style init)
# ==============================================
class LBFGS:
    """
    Classic L-BFGS optimizer with NEB-style init:
    - All inputs defined in __init__
    - run() requires no arguments
    - Two-loop recursion, dynamic H0, step clipping
    """

    def __init__(self,
                 atoms: Atoms,
                 output: str,
                 params: Optional[LBFGSParams] = None,
                 paras: Optional[dict] = None):
        self.atoms = atoms
        self.output = output

        # 1) built-in defaults
        self.params = params if params is not None else LBFGSParams()

        # 2) override from external dict if provided
        if isinstance(paras, dict):
            lbfgs_dict = _select_subdict(paras, ("lbfgs", "LBFGS"))
            _update_dataclass_from_dict(self.params, lbfgs_dict)

        self.S: List[np.ndarray] = []
        self.Y: List[np.ndarray] = []
        self.rhos: List[float] = []

    # ----------------------------------------------------------
    # two-loop recursion
    # ----------------------------------------------------------
    def _two_loop(self, grad_flat: np.ndarray) -> np.ndarray:
        q = grad_flat.copy()
        alpha_list = []

        for s, y, rho in reversed(list(zip(self.S, self.Y, self.rhos))):
            a = rho * np.dot(s, q)
            alpha_list.append(a)
            q -= a * y

        if self.Y:
            gamma = np.dot(self.Y[-1], self.S[-1]) / (np.dot(self.Y[-1], self.Y[-1]) + 1e-20)
        else:
            gamma = 1.0 / self.params.curvature

        z = gamma * q

        for (s, y, rho), a in zip(zip(self.S, self.Y, self.rhos), reversed(alpha_list)):
            b = rho * np.dot(y, z)
            z += s * (a - b)

        return -z

    def _clip_step(self, step_cart: np.ndarray) -> np.ndarray:
        max_disp = float(np.max(np.abs(step_cart)))
        if max_disp > self.params.maxstep:
            step_cart *= self.params.maxstep / max_disp
        return step_cart

    def _update_history(self, s_vec: np.ndarray, y_vec: np.ndarray):
        rho_val = 1.0 / (np.dot(y_vec, s_vec) + 1e-20)
        if np.isfinite(rho_val):
            self.S.append(s_vec.copy()); self.Y.append(y_vec.copy()); self.rhos.append(rho_val)
        if len(self.S) > self.params.memory:
            self.S.pop(0); self.Y.pop(0); self.rhos.pop(0)

    def _log_iter(self, iteration: int, e: float, step_cart: np.ndarray, f: np.ndarray):
        atoms = self.atoms
        atoms.max_dp = np.abs(step_cart).max()
        atoms.rms_dp = np.sqrt((step_cart ** 2).sum() / step_cart.size * 3)
        atoms.max_f = np.abs(f).max()
        atoms.rms_f = np.sqrt((f ** 2).sum() / step_cart.size * 3)

        iter_title = f"Iteration: {iteration}"
        info_message = ['\n' + '-' * 70 + '\n', f'{iter_title.center(70)}\n\n']
        info_message.append(f'\n{"Coordinates".center(70)}\n')
        info_message.append('-' * 70 + '\n')

        for atom_index, atom in enumerate(atoms):
            element_type = atom.symbol
            x, y, z = atom.position
            info_message.append(f"{atom_index:<4} {element_type:<2} {x:>20.4f} {y:>20.4f} {z:>20.4f}\n")

        info_message.append(f"\n\nEnergy:                {e:>12.6f} Convergence criteria  Is converged \n")
        info_message.append(f"Maximum Force:         {atoms.max_f:>12.6f} {atoms.f_max_th:>12.6f}                {'Yes' if atoms.max_f <= atoms.f_max_th else 'No'}\n")
        info_message.append(f"RMS Force:             {atoms.rms_f:>12.6f} {atoms.f_rms_th:>12.6f}                {'Yes' if atoms.rms_f <= atoms.f_rms_th else 'No'}\n")
        info_message.append(f"Maximum Displacement:  {atoms.max_dp:>12.6f} {atoms.dp_max_th:>12.6f}                {'Yes' if atoms.max_dp <= atoms.dp_max_th else 'No'}\n")
        info_message.append(f"RMS Displacement:      {atoms.rms_dp:>12.6f} {atoms.dp_rms_th:>12.6f}                {'Yes' if atoms.rms_dp <= atoms.dp_rms_th else 'No'}\n")

        log_info(info_message, self.output)

    # ----------------------------------------------------------
    # run()
    # ----------------------------------------------------------
    def run(self) -> int:
        base, _ = os.path.splitext(self.output)
        traj_file = base + "_opt_traj.xyz"

        atoms = self.atoms
        r = atoms.get_positions()
        e = float(atoms.get_potential_energy(force_consistent=True))
        f = atoms.get_forces()

        iteration = 0
        if self.params.write_traj and iteration % self.params.traj_every == 0:
            write_xyz(traj_file, [atoms.copy()], energies=[e])

        while iteration < self.params.maxiteration:
            grad = f.reshape(-1)
            step_flat = self._two_loop(grad)
            step = self._clip_step(step_flat.reshape(f.shape))

            r_old = r.copy()
            f_old = f.copy()
            atoms.set_positions(r + step)

            r = atoms.get_positions()
            f = atoms.get_forces()
            e = float(atoms.get_potential_energy(force_consistent=True))

            s_vec = (r - r_old).reshape(-1)
            y_vec = (f - f_old).reshape(-1)
            self._update_history(s_vec, y_vec)

            iteration += 1
            self._log_iter(iteration, e, step_cart=step, f=f)

            if self.params.write_traj and iteration % self.params.traj_every == 0:
                write_xyz(traj_file, [atoms.copy()], energies=[e])

            if (
                atoms.max_f <= atoms.f_max_th
                and atoms.rms_f <= atoms.f_rms_th
                and atoms.max_dp <= atoms.dp_max_th
                and atoms.rms_dp <= atoms.dp_rms_th
            ):
                opt_file = base + "_opt.xyz"
                e_final = float(atoms.get_potential_energy(force_consistent=True))
                write_xyz(opt_file, [atoms], energies=[e_final])
                log_info([
                    f"\nLBFGS optimization converged at iteration {iteration}.\n",
                    f"\nFinal optimized structure written to: {opt_file}\n"
                ], self.output)
                return iteration

        opt_file = base + "_opt.xyz"
        e_final = float(atoms.get_potential_energy(force_consistent=True))
        write_xyz(opt_file, [atoms], energies=[e_final])
        log_info([
            f"\nLBFGS optimization reached max iterations ({self.params.maxiteration}).\n",
            f"\nLast optimized structure written to: {opt_file}\n"
        ], self.output)
        return iteration
