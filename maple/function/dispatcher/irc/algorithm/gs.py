# -*- coding: utf-8 -*-
"""
Intrinsic Reaction Coordinate (IRC) integrator using Gonzalez–Schlegel (GS) scheme:

- Mass-weighted coordinates and Hessian
- Pivot step + constrained optimization on a hypersphere
- Micro-cycles using a (quasi-)Newton step with a Lagrange multiplier λ
  to enforce |p|^2 = (step_length/2)^2 in mass-weighted space
- Hessian reuse with BFGS updates (optional periodic full recalculation)
- Forward and backward paths from the TS geometry, starting along the
  lowest negative eigenmode of the mass-weighted Hessian
- Path merge with the lower-energy endpoint set as dE=0 and the TS
  marked at the maximum energy point

Units:
- Cartesian coordinates: Å
- Forces: Eh/Å
- Energies: Eh
- Mass-weighted coordinates: sqrt(amu) * Å (via masses_D)
"""

import os
from dataclasses import dataclass, fields
from typing import List, Optional, Dict, Tuple

import numpy as np
from ase import Atoms

from .logger import log_info, log_error

# =============================== Utilities ===============================
BOHR_TO_ANG = 0.529177210903
KCAL_PER_EH = 627.509474


def to_f64(x):
    """Convert input to float64 numpy array or scalar."""
    if isinstance(x, np.ndarray):
        return x.astype(np.float64, copy=False)
    try:
        import torch
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
            return x.astype(np.float64, copy=False)
    except Exception:
        pass
    if np.isscalar(x):
        return float(x)
    return np.asarray(x, dtype=np.float64)


def v1(x, n: Optional[int] = None) -> np.ndarray:
    """Flatten to 1D float64 array, optionally enforcing size."""
    v = to_f64(x).reshape(-1)
    if n is not None and v.size != n:
        raise ValueError(f"Expected size {n}, got {v.size}")
    return v


def masses_D(atoms: Atoms) -> np.ndarray:
    """
    Return diagonal scaling vector D (length 3N) for mass-weighted quantities.

    Convention here (consistent with earlier code):

        q_mw = q_cart / D       where D = 1/sqrt(m_i)

    so that q_mw has units sqrt(amu) * Å, and the mass-weighted Hessian is

        H_mw = D * H_cart * D

    Gradients in MW coordinates follow

        g_mw = D * g_cart
    """
    m = to_f64(atoms.get_masses())
    m = np.where(m > 0.0, m, 1.0)
    return v1(1.0 / np.sqrt(np.repeat(m, 3)))


def write_xyz(path: str, atoms_list: List[Atoms], energies: Optional[List[float]] = None):
    """Write a list of structures to an XYZ file."""
    with open(path, "w") as f:
        for i, at in enumerate(atoms_list):
            pos = at.get_positions()
            symbols = at.get_chemical_symbols()
            f.write(f"{len(symbols)}\n")
            if energies is not None and i < len(energies):
                f.write(f"Image {i}  Energy = {energies[i]:.10f}\n")
            else:
                f.write(f"Image {i}\n")
            for s, (x, y, z) in zip(symbols, pos):
                f.write(f"{s:2s} {x: .10f} {y: .10f} {z: .10f}\n")


def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def _unit(v: np.ndarray, eps: float = 1e-16) -> np.ndarray:
    n = _norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


# =============================== Parameters ===============================
@dataclass
class GSParams:
    # Which negative eigenmode (1 = most negative) to use for initial direction
    target_mode: int = 1

    # GS macro step length in mass-weighted coordinates (same units as q_mw)
    # User-facing name kept in "Bohr" for compatibility with old inputs.
    step_length_bohr: float = 0.30

    # Number of macro steps per direction
    max_steps: int = 50

    # Micro-cycle controls
    max_micro_cycles: int = 20
    micro_step_thresh: float = 1e-3

    # Recalculate Hessian every N micro-steps (None = never, only BFGS updates)
    hessian_recalc: Optional[int] = None

    # Convergence on forces in Cartesian space (Eh/Å)
    f_max_th: float = 2e-3
    f_rms_th: float = 5e-4

    # Output controls
    print_each: bool = True
    write_traj: bool = True


# ================================== GS ===================================
class GS:
    """
    Gonzalez–Schlegel IRC integrator:

    - Works in mass-weighted coordinates.
    - Each macro step:
        pivot half-step along negative gradient,
        then constrained optimization on a hypersphere via micro-cycles.
    - Uses BFGS updates of the mass-weighted Hessian, with optional
      periodic full recalculation.
    - Integrates forward and backward from the TS along the lowest
      negative eigenmode of H_mw, then merges paths.
    """

    def __init__(self, atoms: Atoms, output: str,
                 params: Optional[GSParams] = None,
                 paras: Optional[dict] = None):
        self.atoms = atoms
        self.output = output
        self.p = params if params is not None else GSParams()

        # Parse optional dict overrides, supporting both {"gs": {...}}
        # and flat dict style. Keep backward compatibility aliases.
        if isinstance(paras, dict):
            low = {k.lower(): v for k, v in paras.items()}
            sub = None
            for key in ("gs", "irc"):
                if key in low and isinstance(low[key], dict):
                    sub = low[key]
                    break
            if sub is None:
                sub = low
            sub_low = {k.lower(): v for k, v in sub.items()}

            # Aliases / compatibility mapping
            aliases = {
                "sd_len_bohr": "step_length_bohr",
                "steplength_bohr": "step_length_bohr",
                "max_points": "max_steps",
                "max_micro_cycles": "max_micro_cycles",
                "micro_step_thresh": "micro_step_thresh",
                "hessian_recalc": "hessian_recalc",
                "target_mode": "target_mode",
                "f_max_th": "f_max_th",
                "f_rms_th": "f_rms_th",
                "tol_maxf": "f_max_th",       # backward compatibility
                "tol_rmsf": "f_rms_th",       # backward compatibility
                "print_each": "print_each",
                "write_traj": "write_traj",
            }

            for k, v in sub_low.items():
                if k in aliases:
                    setattr(self.p, aliases[k], v)
                elif hasattr(self.p, k):
                    setattr(self.p, k, v)

        # Internal state for GS integration
        self._D: Optional[np.ndarray] = None  # mass-weight scaling vector
        self._step_len_mw: float = float(self.p.step_length_bohr)

        self.mw_coords: Optional[np.ndarray] = None
        self.mw_hessian: Optional[np.ndarray] = None
        self.prev_coords: Optional[np.ndarray] = None
        self.prev_grad: Optional[np.ndarray] = None
        self.displacement: Optional[np.ndarray] = None

        self.pivot_coords: List[np.ndarray] = []
        self.micro_coords: List[np.ndarray] = []
        self.micro_counter: int = 0

    # ------------------------------ Public API ------------------------------
    def run(self) -> Dict[str, any]:
        """
        Compute forward and backward GS IRC paths and produce a merged summary.

        Returns:
            {
                "forward": {...},
                "backward": {...},
                "summary": {...}
            }
        """
        # Prepare mass weights once at TS geometry
        self._D = masses_D(self.atoms)
        self._step_len_mw = float(self.p.step_length_bohr)

        # Diagonalize mass-weighted Hessian at TS to get negative mode
        H_cart_ts = self._get_hessian_cart()
        H_mw_ts = (self._D[:, None] * H_cart_ts) * self._D[None, :]
        w, V = np.linalg.eigh(H_mw_ts)

        neg_idx = np.where(w < 0.0)[0]
        if len(neg_idx) == 0:
            log_error(
                ["[ERROR] GS-IRC: No negative eigenvalues found — "
                 "starting geometry is not a saddle point.\n"],
                self.output,
            )
            raise RuntimeError("GS-IRC: no negative eigenvalues at TS.")

        if len(neg_idx) < self.p.target_mode:
            log_error(
                [f"[ERROR] GS-IRC: Requested mode {self.p.target_mode}, "
                 f"but only {len(neg_idx)} negative modes found.\n"],
                self.output,
            )
            raise RuntimeError("GS-IRC: requested negative mode does not exist.")

        sorted_neg = neg_idx[np.argsort(w[neg_idx])]  # most negative first
        idx = sorted_neg[self.p.target_mode - 1]
        eigval = w[idx]
        v_neg_mw = V[:, idx]

        log_info(
            [
                "\n[INFO] GS-IRC: Selected negative eigenmode "
                f"#{self.p.target_mode} with λ = {eigval:.6e} (MW basis)\n"
            ],
            self.output,
        )

        # Reference TS energy
        E_ts = float(self.atoms.get_potential_energy(force_consistent=True))

        # Store original TS Cartesian positions, reused for both directions
        R_ts_cart = self.atoms.get_positions().copy().reshape(-1)

        # Forward and backward GS-IRC
        forward_log = self._one_side(
            forward=True,
            sign=+1.0,
            q_ts_cart=R_ts_cart,
            v_neg_mw=v_neg_mw,
            E_ts=E_ts,
        )
        backward_log = self._one_side(
            forward=False,
            sign=-1.0,
            q_ts_cart=R_ts_cart,
            v_neg_mw=v_neg_mw,
            E_ts=E_ts,
        )

        merged = self._merge_and_mark_ts(forward_log, backward_log)

        if self.p.write_traj:
            self._write_trajs(forward_log, backward_log)

        return {"forward": forward_log, "backward": backward_log, "summary": merged}

    # --------------------------- Low-level helpers --------------------------
    def _cart_from_mw(self, q_mw: np.ndarray) -> np.ndarray:
        """Convert mass-weighted coordinates back to Cartesian (Å)."""
        return (q_mw * self._D).reshape(-1)

    def _mw_from_cart(self, q_cart: np.ndarray) -> np.ndarray:
        """Convert Cartesian coordinates (Å) to mass-weighted coordinates."""
        return (q_cart / self._D).reshape(-1)

    def _get_hessian_cart(self) -> np.ndarray:
        """
        Get Cartesian Hessian (3N x 3N) from the calculator.

        Assumes atoms.calc implements get_hessian(atoms) and returns 3N x 3N
        or shape (1, 3N, 3N).
        """
        H = self.atoms.calc.get_hessian(self.atoms)
        H = to_f64(H)
        if H.ndim == 3 and H.shape[0] == 1:
            H = H[0]
        if H.ndim != 2 or H.shape[0] != H.shape[1]:
            raise ValueError(f"Hessian must be square 2D, got shape {H.shape}")
        return H

    def _energy_forces_from_mw(self, q_mw: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Set positions from MW coordinates, then return (E, F_cart).

        E: Eh
        F_cart: (3N,) Eh/Å
        """
        q_cart = self._cart_from_mw(q_mw)
        self.atoms.set_positions(q_cart.reshape(-1, 3))
        E = float(self.atoms.get_potential_energy(force_consistent=True))
        F_cart = to_f64(self.atoms.get_forces()).reshape(-1)
        return E, F_cart

    def _gradient_mw_from_forces(self, F_cart: np.ndarray) -> np.ndarray:
        """
        Convert Cartesian forces (Eh/Å) to MW gradient:

            g_cart = dE/dR = -F_cart
            g_mw = D * g_cart
        """
        g_cart = -F_cart.reshape(-1)
        return self._D * g_cart

    @staticmethod
    def _perp_component(vec: np.ndarray, perp_to: np.ndarray) -> np.ndarray:
        """Return component of vec perpendicular to perp_to."""
        denom = float(np.dot(perp_to, perp_to))
        if denom <= 0.0:
            return np.zeros_like(vec)
        return vec - np.dot(perp_to, vec) * perp_to / denom

    @staticmethod
    def _bfgs_update(H: np.ndarray,
                     s: np.ndarray,
                     y: np.ndarray) -> np.ndarray:
        """
        Symmetric BFGS update:

            H_{k+1} = H_k + (y y^T)/(y·s) - (H s s^T H)/(s^T H s)

        Safeguards if denominators are too small or curvature condition fails.
        """
        ys = float(np.dot(y, s))
        if ys <= 1e-12:
            return H

        Hy = H.dot(s)
        sTHs = float(np.dot(s, Hy))
        if sTHs <= 1e-12:
            return H

        term1 = np.outer(y, y) / ys
        term2 = np.outer(Hy, Hy) / sTHs
        return H + term1 - term2

    @staticmethod
    def _newton_1d(on_sphere,
                   lambda_0: float,
                   maxiter: int = 50,
                   tol: float = 1e-10) -> float:
        """
        Simple 1D Newton root finder with finite-difference derivative.
        Used instead of scipy.optimize.newton to avoid extra dependency.
        """
        lam = float(lambda_0)
        for _ in range(maxiter):
            f = float(on_sphere(lam))
            if abs(f) < tol:
                break
            h = 1e-4 * max(1.0, abs(lam))
            f1 = float(on_sphere(lam + h))
            df = (f1 - f) / h
            if abs(df) < 1e-16:
                lam *= 0.5
                continue
            lam -= f / df
        return lam

    # --------------------------- Micro-step (GS) ----------------------------
    def _micro_step(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform one GS micro-step (constrained optimization on hypersphere).

        Updates:
            self.mw_coords, self.mw_hessian, self.prev_coords,
            self.prev_grad, self.displacement, self.micro_counter

        Returns:
            dx: MW displacement for this micro-step
            g_tan: gradient tangent to the hypersphere (MW)
        """
        # Gradient at current coordinates (MW)
        _, F_cart = self._energy_forces_from_mw(self.mw_coords)
        gradient = self._gradient_mw_from_forces(F_cart)

        # BFGS update (or optional full Hessian recalculation)
        gradient_diff = gradient - self.prev_grad
        coords_diff = self.mw_coords - self.prev_coords

        # Update "previous" values
        self.prev_coords = self.mw_coords.copy()
        self.prev_grad = gradient.copy()

        recalc = (
            self.p.hessian_recalc is not None
            and self.p.hessian_recalc > 0
            and (self.micro_counter % self.p.hessian_recalc == 0)
        )
        if recalc and self.micro_counter > 0:
            H_cart = self._get_hessian_cart()
            self.mw_hessian = (self._D[:, None] * H_cart) * self._D[None, :]
        else:
            self.mw_hessian = self._bfgs_update(self.mw_hessian, coords_diff, gradient_diff)

        eigvals, eigvecs = np.linalg.eigh(self.mw_hessian)

        # Constraint radius in MW space (half of the macro step length)
        constraint = (0.5 * self._step_len_mw) ** 2

        mask = np.abs(eigvals) > 1e-8
        big_eigvals = eigvals[mask]
        big_eigvecs = eigvecs[:, mask]

        if big_eigvals.size == 0:
            # Fallback: no "big" eigenvalues; skip constrained step
            dx = np.zeros_like(self.mw_coords)
            g_tan = np.zeros_like(self.mw_coords)
            return dx, g_tan

        grad_star = big_eigvecs.T.dot(gradient)
        displ_star = big_eigvecs.T.dot(self.displacement)

        def get_dx(lambda_):
            """Return dx in eigenbasis for a given λ."""
            return -(grad_star - lambda_ * displ_star) / (big_eigvals - lambda_)

        def on_sphere(lambda_):
            p = displ_star + get_dx(lambda_)
            return p.dot(p) - constraint

        # Initial guess for λ: scaled smallest eigenvalue
        lambda_0 = float(big_eigvals[0])
        lambda_0 *= 1.5 if (lambda_0 < 0.0) else 0.5

        lambda_opt = self._newton_1d(on_sphere, lambda_0, maxiter=50, tol=1e-10)

        dx_star = get_dx(lambda_opt)
        dx = big_eigvecs.dot(dx_star)

        # Update MW displacement and coordinates
        self.displacement += dx
        self.mw_coords += dx

        # Gradient tangent to the sphere (MW)
        g_tan = self._perp_component(gradient, self.displacement)

        self.micro_counter += 1

        return dx, g_tan

    # --------------------------- One direction ------------------------------
    def _one_side(self,
                  forward: bool,
                  sign: float,
                  q_ts_cart: np.ndarray,
                  v_neg_mw: np.ndarray,
                  E_ts: float) -> Dict[str, any]:
        """
        Single-sided GS IRC integration.

        Args:
            forward: True for forward path, False for backward.
            sign: +1.0 or -1.0 to choose direction along the negative mode.
            q_ts_cart: TS Cartesian coordinates (flattened, Å).
            v_neg_mw: selected negative eigenmode in MW basis (3N, normalized later).
            E_ts: TS reference energy (Eh).
        """
        p = self.p
        title = "FORWARD GS-IRC" if forward else "BACKWARD GS-IRC"

        self._print_header(title)
        self._print_conv_thresholds(p.f_max_th, p.f_rms_th)

        # Initial MW coordinates at TS
        q_ts_mw = self._mw_from_cart(q_ts_cart)

        # Initial displacement along negative mode in MW
        v_dir = _unit(v_neg_mw) * sign
        q0_mw = q_ts_mw + 0.5 * self._step_len_mw * v_dir

        # Initial energy / forces / gradient at starting point
        E0, F0_cart = self._energy_forces_from_mw(q0_mw)
        g0_mw = self._gradient_mw_from_forces(F0_cart)

        maxF0 = float(np.max(np.abs(F0_cart)))
        rmsF0 = float(np.sqrt(np.mean(F0_cart ** 2)))

        # Initial Hessian at starting point (MW)
        H0_cart = self._get_hessian_cart()
        H0_mw = (self._D[:, None] * H0_cart) * self._D[None, :]

        # Initialize GS state
        self.mw_coords = q0_mw.copy()
        self.mw_hessian = H0_mw.copy()
        self.prev_coords = q0_mw.copy()
        self.prev_grad = g0_mw.copy()
        self.displacement = np.zeros_like(q0_mw)
        self.micro_counter = 0
        self.pivot_coords = []
        self.micro_coords = []

        # Iteration 0 logging
        if p.print_each:
            self._print_iter_line(0, E0, (E0 - E_ts) * KCAL_PER_EH, maxF0, rmsF0)

        records: List[Dict[str, any]] = []
        records.append(
            {
                "E": E0,
                "maxG": maxF0,
                "rmsG": rmsF0,
                "x": self.atoms.get_positions().copy(),
            }
        )

        # Macro steps
        for it in range(1, p.max_steps + 1):
            # Anchor gradient at current MW coordinates
            E_anchor, F_anchor = self._energy_forces_from_mw(self.mw_coords)
            g_anchor_mw = self._gradient_mw_from_forces(F_anchor)
            g_norm = _norm(g_anchor_mw)

            if g_norm < 1e-12:
                log_info(
                    [f"[INFO] {title}: gradient norm ~ 0 at step {it}, stopping.\n"],
                    self.output,
                )
                break

            # For BFGS in first micro-cycle
            self.prev_coords = self.mw_coords.copy()
            self.prev_grad = g_anchor_mw.copy()

            # Pivot half-step in MW space: move downhill along -gradient
            pivot_step = -0.5 * self._step_len_mw * g_anchor_mw / g_norm
            pivot_coords = self.mw_coords + pivot_step
            self.pivot_coords.append(pivot_coords.copy())

            # Initial guess for new point: another half-step from the pivot
            self.mw_coords = pivot_coords + pivot_step
            self.displacement = pivot_step.copy()

            # Micro-cycles on hypersphere
            micro_coords_side: List[np.ndarray] = []
            for i_micro in range(p.max_micro_cycles):
                dx, _ = self._micro_step()
                micro_coords_side.append(self.mw_coords.copy())
                if _norm(dx) <= p.micro_step_thresh:
                    break
            else:
                log_info(
                    [f"[WARNING] {title}: max micro cycles exceeded at macro step {it}.\n"],
                    self.output,
                )

            self.micro_coords.append(np.array(micro_coords_side))

            # Final energy / forces at new point
            E_new, F_new = self._energy_forces_from_mw(self.mw_coords)
            maxF = float(np.max(np.abs(F_new)))
            rmsF = float(np.sqrt(np.mean(F_new ** 2)))

            if p.print_each:
                self._print_iter_line(it, E_new, (E_new - E_ts) * KCAL_PER_EH, maxF, rmsF)

            records.append(
                {
                    "E": E_new,
                    "maxG": maxF,
                    "rmsG": rmsF,
                    "x": self.atoms.get_positions().copy(),
                }
            )

            # Convergence in terms of Cartesian forces
            if (maxF <= p.f_max_th) and (rmsF <= p.f_rms_th):
                self._print_hurray()
                break

        return {"title": title, "records": records, "E_ts": E_ts}

    # --------------------------- Merge & summary ----------------------------
    def _merge_and_mark_ts(self, f: Dict, b: Dict) -> Dict:
        fR, bR = f["records"], b["records"]
        if not fR or not bR:
            log_error(["[ERROR] GS-IRC: One path side is empty.\n"], self.output)
            raise RuntimeError("GS-IRC: one path side is empty.")

        Ef_end, Eb_end = fR[-1]["E"], bR[-1]["E"]
        if Ef_end <= Eb_end:
            first, second = fR, bR
        else:
            first, second = bR, fR

        merged = []
        # Reverse first (to go from minimum to TS), then append second
        for k in range(len(first) - 1, -1, -1):
            merged.append(first[k])
        for k in range(0, len(second)):
            merged.append(second[k])

        # Find TS index as maximum energy point
        E_list = [rec["E"] for rec in merged]
        ts_idx = int(np.argmax(E_list))
        E_ref = float(min(Ef_end, Eb_end))

        log_info(
            [
                "\n---------------------------------------------------------------\n",
                "                       GS-IRC PATH SUMMARY           \n",
                "---------------------------------------------------------------\n",
                "All forces are in Eh/Å.\n\n",
                "Step        E(Eh)      dE(kcal/mol)  max(|G|)   RMS(G) \n",
            ],
            self.output,
        )

        rows = []
        for i, rec in enumerate(merged, start=1):
            dE_kcal = (rec["E"] - E_ref) * KCAL_PER_EH
            line = (
                f"{i:4d}  {rec['E']:14.6f}  {dE_kcal:12.6f}    "
                f"{rec['maxG']:8.6f}  {rec['rmsG']:8.6f}"
            )
            if i - 1 == ts_idx:
                line += " <= TS"
            rows.append(line + "\n")
        log_info(rows, self.output)

        return {
            "E_ref": E_ref,
            "ts_index": ts_idx + 1,  # 1-based
            "merged_rows": rows,
        }

    # --------------------------- Trajectories -----------------------------
    def _write_trajs(self, f: Dict, b: Dict):
        """Write full, forward, and backward trajectories as XYZ files."""
        base, _ = os.path.splitext(self.output)
        full_path = base + "_full.xyz"
        fwd_path = base + "_forward.xyz"
        bwd_path = base + "_backward.xyz"

        # Forward
        f_atoms, f_E = [], []
        for rec in f["records"]:
            a = self.atoms.copy()
            a.set_positions(rec["x"])
            f_atoms.append(a)
            f_E.append(rec["E"])
        write_xyz(fwd_path, f_atoms, f_E)

        # Backward
        b_atoms, b_E = [], []
        for rec in b["records"]:
            a = self.atoms.copy()
            a.set_positions(rec["x"])
            b_atoms.append(a)
            b_E.append(rec["E"])
        write_xyz(bwd_path, b_atoms, b_E)

        # Full (concatenate forward then backward)
        full_atoms = f_atoms + b_atoms
        full_E = f_E + b_E
        write_xyz(full_path, full_atoms, full_E)

        log_info(
            [
                f"\n[INFO] GS-IRC forward trajectory written to: {fwd_path}\n",
                f"[INFO] GS-IRC backward trajectory written to: {bwd_path}\n",
                f"[INFO] GS-IRC full trajectory written to: {full_path}\n",
            ],
            self.output,
        )

    # ----------------------------- Printing ------------------------------
    def _print_header(self, title: str):
        log_info(
            [
                f"\n         {'*' * 61}\n",
                f"         *{title.center(59)}*\n",
                f"         {'*' * 61}\n\n",
            ],
            self.output,
        )

    def _print_conv_thresholds(self, f_max_th: float, f_rms_th: float):
        log_info(
            [
                "Iteration    E(Eh)      dE(kcal/mol)  max(|G|)   RMS(G) \n",
                f"Convergence thresholds                {f_max_th:0.6f}  {f_rms_th:0.6f}\n",
            ],
            self.output,
        )

    def _print_iter_line(self,
                         i: int,
                         E: float,
                         dE_kcal: float,
                         maxF: float,
                         rmsF: float):
        log_info(
            [f"{i:5d}  {E:14.6f}  {dE_kcal:12.6f}    {maxF:8.6f}  {rmsF:8.6f}\n"],
            self.output,
        )

    def _print_hurray(self):
        log_info(
            [
                "\n                      *************************************************\n",
                "                      ***          THE GS-IRC HAS CONVERGED        ***\n",
                "                      *************************************************\n\n",
            ],
            self.output,
        )
