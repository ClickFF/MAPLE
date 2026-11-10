# -*- coding: utf-8 -*-
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from ase import Atoms
from maple.function.calculator._batch import energy_forces_single

from ._common import (
    compute_metrics,
    is_converged,
    to_numpy_f64,
    vec1d,
    write_xyz,
)
from ...jobABC import JobABC


# ==============================================
# RFO parameters (minimization-only, RS trust region)
# ==============================================
@dataclass
class RFOParams:
    # iterations / trust region
    max_iter: int = 256
    trust_radius_init: float = 0.2
    trust_radius_min: float = 1e-3
    trust_radius_max: float = 1.0

    # trust-region acceptance thresholds (like PRFO)
    eta_shrink: float = 0.75   # if rho < eta_shrink and not at min radius -> reject & shrink
    eta_expand: float = 1.75   # if rho > eta_expand and on boundary -> expand

    # eigen / numerical details
    evals_eps: float = 1e-10   # tiny eigenvalue regularization
    mu_margin: float = 1e-8    # margin below min(w) for bisection upper bound
    max_bisect_it: int = 60

    # output verbosity
    verbose: int = 1
    log_final_paths: bool = True


# ==============================================
# RFO Minimizer
# ==============================================
class RFO(JobABC):
    """
    Rational Function Optimization (RFO) for local minimization with
    trust region logic, mass-weighting, and rho-based acceptance/rejection.
    """

    def __init__(self,
                 atoms: Atoms,
                 output: str,
                 paras: Optional[dict] = None):
        super().__init__(output)
        self.atoms = atoms
        self.params = self._init_params(RFOParams, paras, ("rfo", "RFO", "opt"))
        self.trust_radius = float(self.params.trust_radius_init)
        self._last_iter_info = None

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------
    def run(self) -> Atoms:
        """
        Run RFO-based minimization using trust region in mass-weighted coordinates.
        Writes <base>_opt_traj.xyz during accepted steps and <base>_opt.xyz on finish.
        Returns the optimized Atoms object.
        """
        atoms = self.atoms
        base, _ = os.path.splitext(self.output)
        output_pdb = atoms.info.get("pdb_template")
        opt_traj_file = base + ("_opt_traj.pdb" if output_pdb else "_opt_traj.xyz")
        iteration = 0

        # initial energy/forces
        E, F = energy_forces_single(atoms.calc, atoms)
        write_xyz(opt_traj_file, [atoms.copy()], energies=[float(E)], forces=[F])

        self.results = {'atoms': atoms, 'energy': E, 'forces': F,
                        'iterations': 0, 'converged': False}

        # main loop
        while iteration < self.params.max_iter:
            X = atoms.get_positions().reshape(-1, 3)
            E_old = float(E)
            F_cart = to_numpy_f64(F)
            g_cart = vec1d(-F_cart)  # gradient = -forces (flattened)

            # Hessian (Cartesian) -> to numpy
            H_cart = self._calculate_hessian(atoms)
            n3 = H_cart.shape[0]
            if g_cart.size != n3:
                raise ValueError(f"Gradient size {g_cart.size} != Hessian dim {n3}")

            # Mass weighting
            D, g_mw, H_mw = self._mass_weight(H_cart, g_cart, atoms)

            # Build single-shift RFO step in MW coords with trust region
            s_mw, on_boundary, model_change = self._rfo_step_mw(H_mw, g_mw, self.trust_radius)

            # Convert step back to Cartesian
            s_cart = vec1d(D * s_mw, n3)

            # Trial geometry
            X_new = (X + s_cart.reshape(-1, 3))
            atoms.set_positions(X_new)

            # Evaluate actual energy and forces at trial
            try:
                E_new, F_new = energy_forces_single(atoms.calc, atoms)
            except Exception:
                atoms.set_positions(X)
                raise
            s_cart = (atoms.positions - X).reshape(-1)

            actual_change = float(E_new - E_old)
            rho = None
            if abs(model_change) > 1e-16 and np.isfinite(model_change):
                rho = actual_change / model_change

            # Accept / reject based on rho (PRFO-style)
            accepted = self._accept_or_reject(rho, on_boundary)

            if accepted:
                # accept: update reference state, possibly expand radius
                if (rho is not None) and (rho > self.params.eta_expand) and on_boundary:
                    self.trust_radius = min(self.params.trust_radius_max, 2.0 * self.trust_radius)

                # logging & convergence metrics
                compute_metrics(atoms, s_cart, F_new)

                step_norm_mw = float(np.linalg.norm(s_mw))
                actual_change = float(E_new - E_old)

                self._log_iteration(
                    iteration=iteration + 1,
                    energy=float(E_new),
                    s_cart=s_cart,
                    forces=F_new,
                    rho=rho,
                    model_change=model_change,
                    actual_change=actual_change,
                    step_norm_mw=step_norm_mw,
                    on_boundary=on_boundary,
                )


                # trajectory
                converged = is_converged(atoms)
                self.results.update(energy=E_new, forces=F_new, iterations=iteration + 1,
                                    converged=converged)
                write_xyz(
                    opt_traj_file,
                    [atoms.copy()],
                    energies=[float(E_new)],
                    mode="a",
                    start_index=iteration + 1,
                    forces=[F_new],
                )

                # convergence check
                if converged:
                    self._finalize_run(
                        float(E_new),
                        f"RFO optimization converged at iteration {iteration + 1}.",
                        opt_traj_file,
                        final_path_label="Final optimized structure written to:", forces=F_new,
                    )
                    return atoms

                # advance
                E = E_new
                F = F_new
                iteration += 1

            else:
                # reject: rollback geometry, shrink radius, DO NOT increment iteration
                atoms.set_positions(X)  # rollback
                self.trust_radius = max(self.params.trust_radius_min, 0.5 * self.trust_radius)
                # re-evaluate current E/F (at the old geometry)
                E, F = E_old, F_cart
                # log the rejection event
                # self._log_rejection(iteration + 1, rho)

        # max iterations reached
        e_final = float(E)
        self._finalize_run(
            e_final,
            f"RFO optimization reached max iterations ({self.params.max_iter}).",
            opt_traj_file,
            final_path_label="Last optimized structure written to:", forces=F,
        )
        return atoms

    # ----------------------------------------------------------
    # Core math
    # ----------------------------------------------------------
    def _calculate_hessian(self, atoms: Atoms) -> np.ndarray:
        """Fetch Hessian from calculator and ensure square (3N x 3N) float64."""
        H = atoms.calc.get_hessian(atoms)
        H = to_numpy_f64(H)
        if H.ndim == 3 and H.shape[0] == 1:
            H = H[0]
        if H.ndim != 2 or H.shape[0] != H.shape[1]:
            raise ValueError(f"Hessian must be square, got shape={H.shape}")
        return H

    def _mass_weight(self, H_cart: np.ndarray, g_cart: np.ndarray, atoms: Atoms) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Mass weight gradient and Hessian. Returns (D, g_mw, H_mw) with D=1/sqrt(m)."""
        n3 = H_cart.shape[0]
        masses = to_numpy_f64(atoms.get_masses())
        masses = np.where(masses > 0.0, masses, 1.0)
        D = vec1d(1.0 / np.sqrt(np.repeat(masses, 3)), n3)  # (3N,)
        g_mw = vec1d(D * g_cart, n3)
        H_mw = (D[:, None] * H_cart) * D[None, :]
        return D, g_mw, H_mw

    def _rfo_step_mw(self, H_mw: np.ndarray, g_mw: np.ndarray, trust_radius: float) -> Tuple[np.ndarray, bool, float]:
        """
        Single-shift RFO step in mass-weighted coordinates with trust region.
        Returns:
            s_mw        : step in MW coords
            on_boundary : True if ||s_mw|| == trust_radius (within tolerance)
            model_change: quadratic model energy change (g·s + 0.5 s^T H s) in MW coords
        """
        p = self.params

        # Eigendecomposition in MW coords
        w, V = np.linalg.eigh(H_mw)
        g_proj = V.T @ g_mw

        # Regularize tiny eigenvalues to avoid dividing by ~0
        tiny = (np.abs(w) < p.evals_eps)
        w = np.where(tiny & (w == 0.0), p.evals_eps, w)
        w = np.where(tiny & (w != 0.0), np.sign(w) * p.evals_eps, w)

        # Unconstrained RFO (single-shift with mu=0) equals steepest Newton-like:
        # s_unc = -g_proj / w  (in eigen basis), then rotate back
        denom0 = np.where(np.abs(w) < p.evals_eps, np.sign(w) * p.evals_eps, w)
        s_unc_eig = - g_proj / denom0
        s_unc = V @ s_unc_eig
        norm_unc2 = float(np.dot(s_unc, s_unc))
        R2 = trust_radius * trust_radius

        if norm_unc2 <= R2:
            # inside trust radius: accept unconstrained step
            s = s_unc
            # model change (in MW coords)
            Hs = H_mw @ s
            model_change = float(g_mw.dot(s) + 0.5 * s.dot(Hs))
            return s, False, model_change

        # Otherwise, solve for shift μ < min(w) such that ||s(μ)||^2 = R^2
        # s_eig(μ) = - g_proj / (w - μ)
        # F(μ) = sum_i (g_proj_i / (w_i - μ))^2 - R^2 = 0, μ < min(w)
        w_min = float(np.min(w))
        b = w_min - p.mu_margin

        def F(mu):
            denom = w - mu
            denom = np.where(np.abs(denom) < p.evals_eps, np.sign(denom) * p.evals_eps, denom)
            return float(np.sum((g_proj / denom)**2) - R2)

        # Find a < b with F(a) < 0 (norm below R) and F(b) > 0 (norm above R)
        # Start from b-1.0 and expand downwards if needed.
        a = b - 1.0
        Fa = F(a)
        it = 0
        while Fa > 0.0 and it < p.max_bisect_it:
            a -= max(1.0, abs(a) * 0.5)
            Fa = F(a)
            it += 1

        lo, hi = a, b
        for _ in range(p.max_bisect_it):
            mid = 0.5 * (lo + hi)
            Fm = F(mid)
            if Fm > 0.0:
                hi = mid
            else:
                lo = mid
            if abs(hi - lo) < 1e-12:
                break
        mu_star = 0.5 * (lo + hi)

        denom = w - mu_star
        denom = np.where(np.abs(denom) < p.evals_eps, np.sign(denom) * p.evals_eps, denom)
        s_eig = - g_proj / denom
        s = V @ s_eig
        # Clamp numerically to the boundary (make sure ||s||≈R)
        nrm = float(np.linalg.norm(s))
        if nrm > 0.0:
            s *= (trust_radius / nrm)

        # model change (quadratic approximation) in MW coords
        Hs = H_mw @ s
        model_change = float(g_mw.dot(s) + 0.5 * s.dot(Hs))
        return s, True, model_change

    def _accept_or_reject(self, rho: Optional[float], on_boundary: bool) -> bool:
        """
        Decide accept/reject based on rho and adapt trust radius if rejecting.
        """
        p = self.params
        # reject if rho bad and we can still shrink
        bad = (rho is None) or (not np.isfinite(rho)) or (rho < p.eta_shrink)
        if bad and (self.trust_radius > p.trust_radius_min * (1.0 + 1e-12)):
            return False
        return True

    # ----------------------------------------------------------
    # Logging
    # ----------------------------------------------------------
    def _finalize_run(
        self,
        energy: float,
        summary: str,
        opt_traj_file: str,
        final_path_label: str,
        forces=None,
    ) -> None:
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
                f"\n{final_path_label} {opt_file}\n",
                f"Optimization trajectory written to: {opt_traj_file}\n",
            ])
        self.log_info(info)

    def _log_iteration(self, iteration: int, energy: float, s_cart: np.ndarray,
                    forces: np.ndarray, rho: Optional[float],
                    model_change: float, actual_change: float,
                    step_norm_mw: float, on_boundary: bool):
        atoms = self.atoms

        if self.params.verbose == 1:
            iter_str = f"Iteration: {iteration}"
            info_message = ['\n' + '-' * 70 + '\n', f'{iter_str.center(70)}\n\n']
        else:
            info_message = []

        info_message.append(f'\n{"Coordinates".center(70)}\n')
        info_message.append('-' * 70)
        info_message.append('\n')

        for atom_index, atom in enumerate(atoms):
            element_type = atom.symbol
            coord = atom.position
            info_message.append(f"{atom_index:<4} {element_type:<2} {coord[0]:>20.4f} {coord[1]:>20.4f} {coord[2]:>20.4f}\n")

        info_message.append(f"\n\nEnergy:                {energy:>12.6f} Convergence criteria  Is converged \n")

        # ---- convergence metrics ----
        info_message.append(f"Maximum Force:         {atoms.max_f:>12.6f} {atoms.f_max_th:>12.6f}                {'Yes' if atoms.max_f <= atoms.f_max_th else 'No'}\n")
        info_message.append(f"RMS Force:             {atoms.rms_f:>12.6f} {atoms.f_rms_th:>12.6f}                {'Yes' if atoms.rms_f <= atoms.f_rms_th else 'No'}\n")
        info_message.append(f"Maximum Displacement:  {atoms.max_dp:>12.6f} {atoms.dp_max_th:>12.6f}                {'Yes' if atoms.max_dp <= atoms.dp_max_th else 'No'}\n")
        info_message.append(f"RMS Displacement:      {atoms.rms_dp:>12.6f} {atoms.dp_rms_th:>12.6f}                {'Yes' if atoms.rms_dp <= atoms.dp_rms_th else 'No'}\n")

        if self.params.verbose == 1:
            # ---- model vs actual ----
            info_message.append(
                f"\nModel change: {model_change: .6e}  "
                f"Actual change: {actual_change: .6e}  "
                f"rho: {rho if rho is not None else float('nan'): .3f}\n"
            )

            # ---- trust region info ----
            info_message.append(
                f"Trust radius (MW): {self.trust_radius: .6f}  "
                f"Step norm (MW): {step_norm_mw: .6f}  On boundary: {on_boundary}\n"
            )

        self._last_iter_info = info_message

        if self.params.verbose == 1:
            self.log_info(info_message)

    def _log_rejection(self, iteration: int, rho: Optional[float]):
        """Log a rejection event and the trust radius shrink."""
        info_message = [
            '\n' + '-' * 70 + '\n',
            f'{"Step Rejected".center(70)}\n\n',
            f"Iteration: {iteration}\n",
            f"rho: {rho if rho is not None else float('nan'): .3f}\n",
            f"New trust radius: {self.trust_radius: .6f}\n"
        ]
        self.log_info(info_message)
