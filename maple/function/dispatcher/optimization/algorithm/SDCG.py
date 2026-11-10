# -*- coding: utf-8 -*-
"""
SDCG (Steepest Descent + Conjugate Gradient) Fusion Optimizer

Implements a phased geometry optimization strategy:
  Phase 1 (SD): Steepest descent for robust initial approach
  Phase 2 (CG): PRP+ conjugate gradient for faster convergence

Optional GDIIS (Geometry DIIS) acceleration operates in both phases.

CG variant: PRP+ (Polak-Ribiere-Polyak Plus) with Powell restart.
  - PRP+ beta = max(f_k . (f_k - f_{k-1}) / ||f_{k-1}||^2, 0)
  - Powell restart: if |f_k . f_{k-1}| >= 0.2 * ||f_k||^2, reset beta=0
  - Self-correcting: automatically reverts to SD direction when beta <= 0

References:
  Polak & Ribiere, Rev. Francaise Informat. Rech. Operat. 3, 35-43 (1969)
  Powell, Math. Programming 12, 241-254 (1977)
  Csaszar & Pulay, J. Mol. Struct. 114, 31-34 (1984) [GDIIS]
  Farkas & Schlegel, PCCP 4, 11-15 (2002) [GDIIS improvements]
"""
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from ase import Atoms
from maple.function.calculator._batch import energy_forces_single

from ._common import compute_metrics, is_converged, write_xyz
from .DIIS import DIISAccelerator, DIISParams
from ...jobABC import JobABC


@dataclass
class SDCGParams:
    """SDCG fusion optimizer parameters."""
    # General
    max_step: float = 0.2           # Maximum step size (Angstrom)
    max_iter: int = 256             # Maximum number of iterations
    verbose: int = 1                # Verbosity level (0=silent, 1=detailed)

    # Phase control
    sd_enabled: bool = True         # Enable SD phase
    cg_enabled: bool = True         # Enable CG phase
    sd_max_iter: int = 50           # Max SD iterations before forced CG switch
    cg_switch_fmax: float = 0.0     # Switch to CG when max_f < threshold (0.0 = auto)

    # CG parameters
    cg_restart_threshold: float = 0.2  # Powell restart threshold
    cg_beta_method: str = "prp+"       # CG variant

    # GDIIS parameters
    diis_enabled: bool = True       # Enable GDIIS acceleration
    diis_store_every: int = 5       # Store a snapshot every N steps
    diis_min_snapshots: int = 3     # Minimum snapshots before GDIIS attempt
    diis_memory: int = 6            # Maximum GDIIS history vectors
    log_final_paths: bool = True    # Log standalone optimizer artifact paths


class SDCG(JobABC):
    """
    SDCG fusion optimizer: Steepest Descent + Conjugate Gradient with GDIIS.

    Operates in two phases:
      SD phase: follows negative gradient direction (robust, handles large forces)
      CG phase: PRP+ conjugate gradient (faster convergence near minimum)

    Phase transition occurs when:
      - max_f drops below cg_switch_fmax threshold, OR
      - SD has run for sd_max_iter iterations
    whichever comes first.

    GDIIS (Geometry DIIS) acceleration is available in both phases,
    storing snapshots at intervals and attempting extrapolation.

    User control:
      method=sd    -> SD only (cg_enabled=False)
      method=cg    -> CG only (sd_enabled=False)
      method=sdcg  -> SD+CG fusion (default)
      diis_enabled=false -> disable GDIIS in any mode
    """

    def __init__(self,
                 atoms: Atoms,
                 output: str,
                 paras: Optional[dict] = None):
        super().__init__(output)
        self.atoms = atoms
        self.params = self._init_params(
            SDCGParams, paras, ("sdcg", "SDCG", "sd", "SD", "cg", "CG", "opt")
        )

        # Apply method-based phase control
        self._apply_method_overrides(paras)

        # Initialize GDIIS accelerator
        self.diis: Optional[DIISAccelerator] = None
        if self.params.diis_enabled:
            diis_params = DIISParams(memory=self.params.diis_memory)
            self.diis = DIISAccelerator(diis_params)

        # Phase state
        self._phase = "sd" if self.params.sd_enabled else "cg"
        self._sd_iter_count = 0      # SD iterations in current run
        self._sd_step_counter = 0    # counts steps since last GDIIS store

        # Barzilai-Borwein history
        self._prev_positions: Optional[np.ndarray] = None
        self._prev_forces: Optional[np.ndarray] = None

        # CG state
        self._cg_prev_forces: Optional[np.ndarray] = None
        self._cg_prev_direction: Optional[np.ndarray] = None

        self._last_iter_info: Optional[List[str]] = None

        self._log_params()

    def _apply_method_overrides(self, paras: Optional[dict]) -> None:
        """Apply phase control based on the method keyword."""
        if not isinstance(paras, dict):
            return
        method = paras.get("method", "").lower()
        if method == "sd":
            self.params.sd_enabled = True
            self.params.cg_enabled = False
        elif method == "cg":
            self.params.sd_enabled = False
            self.params.cg_enabled = True
        # sdcg or empty: use defaults (both enabled)

    def _log_params(self) -> None:
        """Log optimizer parameters."""
        mode = "SD+CG"
        if not self.params.cg_enabled:
            mode = "SD only"
        elif not self.params.sd_enabled:
            mode = "CG only"

        param_info = [
            "\n" + "=" * 70 + "\n",
            "SDCG Parameters\n",
            "=" * 70 + "\n",
            f"mode:              {mode}\n",
            f"max_step:          {self.params.max_step}\n",
            f"max_iter:          {self.params.max_iter}\n",
            f"sd_enabled:        {self.params.sd_enabled}\n",
            f"cg_enabled:        {self.params.cg_enabled}\n",
            f"sd_max_iter:       {self.params.sd_max_iter}\n",
            f"cg_switch_fmax:    {self.params.cg_switch_fmax}\n",
            f"cg_restart_thresh: {self.params.cg_restart_threshold}\n",
            f"cg_beta_method:    {self.params.cg_beta_method}\n",
            f"diis_enabled:      {self.params.diis_enabled}\n",
            f"diis_store_every:  {self.params.diis_store_every}\n",
            f"diis_memory:       {self.params.diis_memory}\n",
            f"verbose:           {self.params.verbose}\n",
            "=" * 70 + "\n\n",
        ]
        self.log_info(param_info)

    # ----------------------------------------------------------
    # Step computation
    # ----------------------------------------------------------

    def _clip_step(self, step: np.ndarray) -> np.ndarray:
        """Clip step so max atomic displacement <= max_step."""
        max_disp = float(np.max(np.abs(step)))
        if max_disp > self.params.max_step:
            step = step * (self.params.max_step / max_disp)
        return step

    def _sd_step(self, forces: np.ndarray) -> np.ndarray:
        """Compute a steepest descent step."""
        step = self.params.max_step * forces
        return self._clip_step(step)

    def _cg_step(self, forces: np.ndarray) -> np.ndarray:
        """
        Compute a PRP+ conjugate gradient step.

        PRP+ beta = max(f_k . (f_k - f_{k-1}) / ||f_{k-1}||^2, 0)
        Powell restart: if |f_k . f_{k-1}| >= 0.2 * ||f_k||^2, reset to SD

        Step size uses the same force-proportional scaling as SD:
          step = max_step * direction, then clip to max_step
        This ensures steps shrink naturally near the minimum (direction ~ forces).
        The CG conjugate direction is normalized to the force scale before storage
        to prevent unbounded growth across iterations.
        """
        if self._cg_prev_forces is None or self._cg_prev_direction is None:
            # First CG step = steepest descent direction
            direction = forces.copy()
            self._cg_beta = 0.0
        else:
            f_curr = forces.ravel()
            f_prev = self._cg_prev_forces.ravel()

            # PRP+ beta
            df = f_curr - f_prev
            denom = np.dot(f_prev, f_prev)
            if denom < 1e-20:
                beta = 0.0
            else:
                beta = np.dot(f_curr, df) / denom
            beta = max(beta, 0.0)  # PRP+ clamp: ensures non-negative

            # Powell restart condition
            if abs(np.dot(f_curr, f_prev)) >= self.params.cg_restart_threshold * np.dot(f_curr, f_curr):
                beta = 0.0

            direction = forces + beta * self._cg_prev_direction

            # Ensure descent direction: if d . f <= 0, reset to SD
            if np.dot(direction.ravel(), f_curr) <= 0:
                direction = forces.copy()
                beta = 0.0

            self._cg_beta = beta

        # Force-proportional step (same formula as SD)
        # This ensures step size shrinks naturally as forces decrease
        step = self.params.max_step * direction
        step = self._clip_step(step)

        # Store normalized direction for next iteration:
        # Scale prev_direction to force magnitude to prevent accumulation
        f_max = np.abs(forces).max()
        d_max = np.abs(direction).max()
        if d_max > 0 and f_max > 0:
            self._cg_prev_direction = direction * (f_max / d_max)
        else:
            self._cg_prev_direction = direction.copy()
        self._cg_prev_forces = forces.copy()

        return step

    def _reset_cg(self) -> None:
        """Reset CG conjugate direction (restart to SD direction next step)."""
        self._cg_prev_forces = None
        self._cg_prev_direction = None

    # ----------------------------------------------------------
    # Phase transition
    # ----------------------------------------------------------

    def _switch_to_cg(self, iteration: int) -> None:
        """Switch from SD phase to CG phase."""
        self._phase = "cg"
        self._cg_prev_forces = None
        self._cg_prev_direction = None

        # Reset GDIIS for fresh start in CG phase
        if self.diis is not None:
            self.diis.reset()
            self._sd_step_counter = 0

        if self.params.verbose == 1:
            self.log_info([
                f"\n{'=' * 70}\n",
                f"Phase transition: SD -> CG at iteration {iteration}\n",
                f"  SD iterations completed: {self._sd_iter_count}\n",
                f"{'=' * 70}\n\n",
            ])

    # ----------------------------------------------------------
    # GDIIS acceleration
    # ----------------------------------------------------------

    def _estimate_step_scale(self, forces: np.ndarray) -> float:
        """
        Estimate step_scale (approximate H^{-1}) using Barzilai-Borwein method.
        BB1: alpha = (dx . df) / (df . df)
        """
        if self._prev_positions is None or self._prev_forces is None:
            return self.params.max_step

        dx = (self.atoms.get_positions() - self._prev_positions).ravel()
        df = (forces - self._prev_forces).ravel()

        df_dot_df = np.dot(df, df)
        if df_dot_df < 1e-20:
            return self.params.max_step

        alpha = np.dot(dx, df) / df_dot_df
        alpha = max(0.01, min(abs(alpha), 2.0))
        return alpha

    def _try_diis_acceleration(self, forces: np.ndarray,
                               iteration: int,
                               current_energy: float
                               ) -> Optional[np.ndarray]:
        """
        Try GDIIS acceleration. Stores a snapshot every diis_store_every
        steps, and attempts GDIIS when enough snapshots are collected.

        Returns:
            New positions from GDIIS, or None if not applicable / rejected
        """
        if self.diis is None:
            return None

        self._sd_step_counter += 1

        if self._sd_step_counter < self.params.diis_store_every:
            return None

        # Time to store a snapshot
        self._sd_step_counter = 0
        self.diis.store(self.atoms.get_positions(), forces)

        if not self.diis.can_extrapolate():
            return None

        step_scale = self._estimate_step_scale(forces)

        result = self.diis.extrapolate(step_scale=step_scale)
        if result is None:
            if self.params.verbose == 1:
                reason = getattr(self.diis, '_last_reject_reason', 'unknown')
                self.log_info([
                    f"  GDIIS extrapolation failed (iter {iteration}, "
                    f"nvec={len(self.diis.error_vectors)}, "
                    f"step_scale={step_scale:.4f}, reason={reason}). Skipping.\n"
                ])
            return None

        new_pos, coeffs = result

        # Validate: reject excessively large steps
        step = new_pos - self.atoms.get_positions()
        max_disp = np.abs(step).max()
        if max_disp > self.params.max_step * 5.0:
            if self.params.verbose == 1:
                self.log_info([
                    f"  GDIIS step too large (max_disp={max_disp:.4f}, "
                    f"limit={self.params.max_step * 5.0:.4f}). Dropping oldest.\n"
                ])
            self.diis.drop_oldest()
            return None

        return new_pos

    # ----------------------------------------------------------
    # Iteration logging
    # ----------------------------------------------------------

    def _build_iter_message(self, iteration: int, energy: float,
                            step: np.ndarray, forces: np.ndarray,
                            diis_step: bool = False) -> List[str]:
        """Build per-iteration info message."""
        atoms = self.atoms

        compute_metrics(atoms, step, forces)

        if self.params.verbose == 1:
            title = f"Iteration: {iteration} [{self._phase.upper()}]"
            if diis_step:
                title += " (GDIIS)"
            info = ['\n' + '-' * 70 + '\n', f'{title.center(70)}\n\n']
        else:
            info = []

        info.append(f'\n{"Coordinates".center(70)}\n')
        info.append('-' * 70 + '\n')

        for atom_index, atom in enumerate(atoms):
            x, y, z = atom.position
            info.append(
                f"{atom_index:<4} {atom.symbol:<2} "
                f"{x:>20.4f} {y:>20.4f} {z:>20.4f}\n"
            )

        info.append(
            f"\n\nEnergy:                {energy:>12.6f} "
            f"Convergence criteria  Is converged \n"
        )
        info.append(
            f"Maximum Force:         {atoms.max_f:>12.6f} "
            f"{atoms.f_max_th:>12.6f}  "
            f"{'Yes' if atoms.max_f <= atoms.f_max_th else 'No'}\n"
        )
        info.append(
            f"RMS Force:             {atoms.rms_f:>12.6f} "
            f"{atoms.f_rms_th:>12.6f}  "
            f"{'Yes' if atoms.rms_f <= atoms.f_rms_th else 'No'}\n"
        )
        info.append(
            f"Maximum Displacement:  {atoms.max_dp:>12.6f} "
            f"{atoms.dp_max_th:>12.6f}  "
            f"{'Yes' if atoms.max_dp <= atoms.dp_max_th else 'No'}\n"
        )
        info.append(
            f"RMS Displacement:      {atoms.rms_dp:>12.6f} "
            f"{atoms.dp_rms_th:>12.6f}  "
            f"{'Yes' if atoms.rms_dp <= atoms.dp_rms_th else 'No'}\n"
        )

        return info

    def _log_iter(self, info_message: List[str]) -> None:
        """Print iteration info only if verbose=1."""
        if self.params.verbose == 1:
            self.log_info(info_message)

    def _finalize_run(self, energy: float, summary: str, opt_traj_file: str, forces=None) -> None:
        """Write final optimized structure and log the closing summary."""
        base, _ = os.path.splitext(self.output)
        output_pdb = self.atoms.info.get("pdb_template")
        opt_file = base + ("_opt.pdb" if output_pdb else "_opt.xyz")
        write_xyz(opt_file, [self.atoms], energies=[energy], forces=[forces])
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
    # Main optimization loop
    # ----------------------------------------------------------

    def run(self) -> Atoms:
        """
        Run SDCG optimization.

        Returns:
            Optimized Atoms object
        """
        base, _ = os.path.splitext(self.output)
        output_pdb = self.atoms.info.get("pdb_template")
        opt_traj_file = base + ("_opt_traj.pdb" if output_pdb else "_opt_traj.xyz")

        atoms = self.atoms

        # Get initial state
        energy, forces = energy_forces_single(atoms.calc, atoms)
        write_xyz(opt_traj_file, [atoms.copy()], energies=[energy], forces=[forces])

        # Auto cg_switch_fmax: 0.5 * initial max force
        initial_max_f = np.abs(forces).max()
        cg_switch_threshold = self.params.cg_switch_fmax
        if cg_switch_threshold <= 0.0 and self.params.sd_enabled and self.params.cg_enabled:
            cg_switch_threshold = 0.5 * initial_max_f
            if self.params.verbose == 1:
                self.log_info([
                    f"  Auto cg_switch_fmax = {cg_switch_threshold:.6f} "
                    f"(0.5 * initial max_f = {initial_max_f:.6f})\n"
                ])

        iteration = 0

        self.results = {'atoms': atoms, 'energy': energy, 'forces': forces,
                        'iterations': 0, 'converged': False}
        while iteration < self.params.max_iter:

            # Check SD -> CG phase transition
            if self._phase == "sd" and self.params.cg_enabled:
                max_f = np.abs(forces).max()
                should_switch = False
                if self._sd_iter_count >= self.params.sd_max_iter:
                    should_switch = True
                if cg_switch_threshold > 0 and max_f < cg_switch_threshold:
                    should_switch = True
                if should_switch:
                    self._switch_to_cg(iteration)

            # Save state for BB estimation and GDIIS validation
            saved_positions = atoms.get_positions().copy()
            saved_forces = forces.copy()
            saved_energy = energy

            # Try GDIIS acceleration
            diis_step = False
            diis_pos = self._try_diis_acceleration(forces, iteration, energy)

            if diis_pos is not None:
                # GDIIS step: apply and validate
                step = diis_pos - saved_positions
                step = self._clip_step(step)
                atoms.set_positions(saved_positions + step)

                try:
                    new_energy, new_forces = energy_forces_single(atoms.calc, atoms)
                except Exception:
                    atoms.set_positions(saved_positions)
                    raise
                new_max_f = np.abs(new_forces).max()
                old_max_f = np.abs(forces).max()

                # Validate: reject if energy rises significantly AND force increases
                if new_energy > saved_energy + 0.05 and new_max_f > old_max_f * 1.5:
                    atoms.set_positions(saved_positions)
                    self.diis.drop_oldest()
                    if self.params.verbose == 1:
                        self.log_info([
                            f"  GDIIS step rejected (dE={new_energy - saved_energy:+.6f}, "
                            f"fmax ratio={new_max_f / old_max_f:.2f}). Falling back to {self._phase.upper()}.\n"
                        ])
                else:
                    energy = new_energy
                    forces = new_forces
                    diis_step = True

            if not diis_step:
                # Phase-specific step
                if self._phase == "sd":
                    step = self._sd_step(forces)
                else:
                    step = self._cg_step(forces)

                atoms.set_positions(atoms.get_positions() + step)

                try:
                    energy, forces = energy_forces_single(atoms.calc, atoms)
                except Exception:
                    atoms.set_positions(saved_positions)
                    raise

            step = atoms.positions - saved_positions

            # Update BB history
            self._prev_positions = saved_positions
            self._prev_forces = saved_forces

            # Track SD iterations for phase transition
            if self._phase == "sd":
                self._sd_iter_count += 1

            iteration += 1

            # Build and log iteration info
            last_info = self._build_iter_message(
                iteration, energy, step, forces, diis_step
            )
            self._last_iter_info = last_info
            self._log_iter(last_info)

            converged = is_converged(atoms)
            self.results.update(energy=energy, forces=forces, iterations=iteration, converged=converged)
            write_xyz(
                opt_traj_file,
                [atoms.copy()],
                energies=[energy],
                mode="a",
                start_index=iteration,
                forces=[forces],
            )

            if converged:
                self._finalize_run(
                    energy,
                    f"SDCG converged at iteration {iteration} "
                    f"(phase: {self._phase.upper()}).",
                    opt_traj_file, forces=forces,
                )
                return atoms

        self._finalize_run(
            energy,
            f"SDCG did NOT converge after {self.params.max_iter} iterations "
            f"(final phase: {self._phase.upper()}).",
            opt_traj_file, forces=forces,
        )
        return atoms
