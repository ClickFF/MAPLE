# -*- coding: utf-8 -*-
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from ase import Atoms
from maple.function.calculator._batch import energy_forces_single

from ._common import compute_metrics, is_converged, write_xyz
from ...jobABC import JobABC


# ==============================================
# LBFGS Parameters
# ==============================================
@dataclass
class LBFGSParams:
    memory: int = 5
    curvature: float = 70.0
    max_step: float = 0.2
    max_iter: int = 256
    verbose: int = 1
    log_final_paths: bool = True


# ==============================================
# LBFGS Optimizer
# ==============================================
class LBFGS(JobABC):
    """Classic L-BFGS optimizer."""

    def __init__(self,
                 atoms: Atoms,
                 output: str,
                 paras: Optional[dict] = None):
        super().__init__(output)
        self.atoms = atoms
        self.params = self._init_params(LBFGSParams, paras, ("lbfgs", "LBFGS", "opt"))
        
        # Log the actual parameters being used
        param_info = [
            "\n" + "=" * 70 + "\n",
            "LBFGS Parameters\n",
            "=" * 70 + "\n",
            f"memory:     {self.params.memory}\n",
            f"curvature:  {self.params.curvature}\n",
            f"max_step:   {self.params.max_step}\n",
            f"max_iter:   {self.params.max_iter}\n",
            f"verbose:    {self.params.verbose}\n",
            "=" * 70 + "\n\n",
        ]
        self.log_info(param_info)

        self.S: List[np.ndarray] = []
        self.Y: List[np.ndarray] = []
        self.rhos: List[float] = []

        self._last_iter_info = None

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

        direction = -z
        if not np.all(np.isfinite(direction)) or np.dot(direction, grad_flat) >= 0.0:
            direction = -(1.0 / self.params.curvature) * grad_flat
        return direction

    def _clip_step(self, step_cart: np.ndarray) -> np.ndarray:
        max_disp = float(np.max(np.abs(step_cart)))
        if max_disp > self.params.max_step:
            step_cart *= self.params.max_step / max_disp
        return step_cart

    def _update_history(self, s_vec: np.ndarray, y_vec: np.ndarray):
        curvature = np.dot(y_vec, s_vec)
        if not np.isfinite(curvature) or curvature <= 1e-12:
            return

        rho_val = 1.0 / curvature
        self.S.append(s_vec.copy())
        self.Y.append(y_vec.copy())
        self.rhos.append(rho_val)
        if len(self.S) > self.params.memory:
            self.S.pop(0); self.Y.pop(0); self.rhos.pop(0)

    # ----------------------------------------------------------
    def _build_iter_message(self, iteration, e, step_cart, f):
        """Build per-iteration info message (store even if verbose=0)."""
        atoms = self.atoms
        compute_metrics(atoms, step_cart, f)

        if self.params.verbose == 1:
            title = f"Iteration: {iteration}"
            info = ['\n' + '-' * 70 + '\n', f'{title.center(70)}\n\n']
        else:
            info = []
            
        info.append(f'\n{"Coordinates".center(70)}\n')
        info.append('-' * 70 + '\n')

        for atom_index, atom in enumerate(atoms):
            x, y, z = atom.position
            info.append(f"{atom_index:<4} {atom.symbol:<2} {x:>20.4f} {y:>20.4f} {z:>20.4f}\n")

        info.append(f"\n\nEnergy:                {e:>12.6f} Convergence criteria  Is converged \n")
        info.append(f"Maximum Force:         {atoms.max_f:>12.6f} {atoms.f_max_th:>12.6f}  "
                    f"{'Yes' if atoms.max_f <= atoms.f_max_th else 'No'}\n")
        info.append(f"RMS Force:             {atoms.rms_f:>12.6f} {atoms.f_rms_th:>12.6f}  "
                    f"{'Yes' if atoms.rms_f <= atoms.f_rms_th else 'No'}\n")
        info.append(f"Maximum Displacement:  {atoms.max_dp:>12.6f} {atoms.dp_max_th:>12.6f}  "
                    f"{'Yes' if atoms.max_dp <= atoms.dp_max_th else 'No'}\n")
        info.append(f"RMS Displacement:      {atoms.rms_dp:>12.6f} {atoms.dp_rms_th:>12.6f}  "
                    f"{'Yes' if atoms.rms_dp <= atoms.dp_rms_th else 'No'}\n")

        return info

    # ----------------------------------------------------------
    def _log_iter(self, info_message):
        """Print iteration info only if verbose=1."""
        if self.params.verbose == 1:
            self.log_info(info_message)

    def _finalize_run(self, e: float, summary: str, opt_traj_file: str, forces=None) -> None:
        """Write final optimized structure and log the closing summary."""
        base, _ = os.path.splitext(self.output)
        output_pdb = self.atoms.info.get("pdb_template")
        opt_file = base + ("_opt.pdb" if output_pdb else "_opt.xyz")
        write_xyz(opt_file, [self.atoms], energies=[e],
                  forces=[forces] if forces is not None else None)
        if self.params.verbose != 1 and self._last_iter_info is not None:
            self.log_info(self._last_iter_info)
        info = [f"\n{summary}\n"]
        if self.params.log_final_paths:
            info.extend([
                f"Final frame written to {opt_file}\n",
                f"Optimization trajectory written to {opt_traj_file}\n",
            ])
        self.log_info(info)

    # ----------------------------------------------------------
    def run(self) -> Atoms:
        base, _ = os.path.splitext(self.output)
        output_pdb = self.atoms.info.get("pdb_template")
        opt_traj_file = base + ("_opt_traj.pdb" if output_pdb else "_opt_traj.xyz")

        atoms = self.atoms
        iteration = 0
        r = atoms.get_positions()
        e, f = energy_forces_single(atoms.calc, atoms)
        write_xyz(opt_traj_file, [atoms.copy()], energies=[e], forces=[f])

        self.results = {'atoms': atoms, 'energy': e, 'forces': f,
                        'iterations': 0, 'converged': False}
        while iteration < self.params.max_iter:
            grad = (-f).reshape(-1)
            step_flat = self._two_loop(grad)
            step = self._clip_step(step_flat.reshape(f.shape))

            r_old = r.copy()
            grad_old = grad.copy()
            atoms.set_positions(r + step)

            r = atoms.get_positions()
            try:
                e, f = energy_forces_single(atoms.calc, atoms)
            except Exception:
                atoms.set_positions(r_old)
                raise
            step = r - r_old

            s_vec = (r - r_old).reshape(-1)
            grad = (-f).reshape(-1)
            y_vec = grad - grad_old
            self._update_history(s_vec, y_vec)

            iteration += 1

            # build & store last iteration info
            last_info = self._build_iter_message(iteration, e, step_cart=step, f=f)
            self._last_iter_info = last_info

            # per-iteration log only when verbose=1
            self._log_iter(last_info)

            converged = is_converged(atoms)
            self.results.update(energy=e, forces=f, iterations=iteration, converged=converged)
            write_xyz(
                opt_traj_file,
                [atoms.copy()],
                energies=[e],
                mode="a",
                start_index=iteration,
                forces=[f],
            )

            if converged:
                self._finalize_run(
                    e,
                    f"LBFGS converged at iteration {iteration}.",
                    opt_traj_file, forces=f,
                )
                return atoms

        self._finalize_run(
            e,
            f"LBFGS did NOT converge after {self.params.max_iter} iterations.",
            opt_traj_file, forces=f,
        )
        return atoms
