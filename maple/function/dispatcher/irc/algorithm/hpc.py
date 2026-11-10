# -*- coding: utf-8 -*-
"""
Intrinsic Reaction Coordinate (IRC) integrator using Hessian-based Predictor-Corrector(HPC):

- Mass-weighted coordinates and Hessian
- Predictor step via local quadratic approximation
- DWI surface fitting and mBS (modified Bulirsch–Stoer) corrector integration
- Hessian updated by BFGS/Bofill (optional periodic full recalculation)
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
from contextlib import nullcontext
from collections import deque
from dataclasses import dataclass, fields
from typing import List, Optional, Dict, Tuple

import numpy as np
from ase import Atoms
from maple.function.read.filereader.pdb_reader import write_pdb_trajectory
from maple.function.calculator._inputs import _state

from .logger import log_info, log_error
from maple.function.calculator._batch import energy_forces_single, hessian_single

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


def write_xyz(path, atoms_list: List[Atoms],
              energies: Optional[List[float]] = None,
              mode: str = "w",
              start_index: int = 0,
              forces: Optional[List[np.ndarray]] = None,
              comments: Optional[List[str]] = None) -> None:
    """Write supplied XYZ frames to a path or stream without evaluating them."""
    if atoms_list and atoms_list[0].info.get("pdb_template"):
        write_pdb_trajectory(
            path,
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

    with (nullcontext(path) if hasattr(path, 'write') else open(path, mode)) as f:
        f.writelines(blocks)
        f.flush()

def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def _unit(v: np.ndarray, eps: float = 1e-16) -> np.ndarray:
    n = _norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


def _taylor_energy(energy: float, gradient: np.ndarray, hessian: np.ndarray, step: np.ndarray) -> float:
    """Second-order Taylor energy expansion."""
    return float(energy + step @ gradient + 0.5 * step @ hessian @ step)


def _taylor_grad(gradient: np.ndarray, hessian: np.ndarray, step: np.ndarray) -> np.ndarray:
    """Gradient of the second-order Taylor expansion."""
    return gradient + hessian @ step


class DWI:
    """
    Distance-weighted interpolation (DWI) of a local quadratic PES.
    Keeps the most recent two points (coords, energy, gradient, hessian).
    """

    def __init__(self, n: int = 4, maxlen: int = 2):
        self.n = int(n)
        if self.n <= 0 or (self.n % 2) != 0:
            raise ValueError("DWI n must be a positive even integer.")
        self.maxlen = int(maxlen)
        if self.maxlen != 2:
            raise ValueError("DWI currently supports maxlen=2 only.")

        self.coords = deque(maxlen=self.maxlen)
        self.energies = deque(maxlen=self.maxlen)
        self.gradients = deque(maxlen=self.maxlen)
        self.hessians = deque(maxlen=self.maxlen)

    def update(self, coords: np.ndarray, energy: float, gradient: np.ndarray, hessian: np.ndarray) -> None:
        self.coords.append(coords)
        self.energies.append(float(energy))
        self.gradients.append(gradient)
        self.hessians.append(hessian)

        if not (len(self.coords) == len(self.energies) == len(self.gradients) == len(self.hessians)):
            raise RuntimeError("DWI internal buffers are inconsistent.")

    def interpolate(self, at_coords: np.ndarray, gradient: bool = False):
        """Distance-weighted interpolation of energy (and optionally gradient)."""
        if len(self.coords) < 2:
            raise RuntimeError("DWI requires two points before interpolation.")

        c1, c2 = self.coords
        dx1 = at_coords - c1
        dx2 = at_coords - c2

        dx1_norm = _norm(dx1)
        dx2_norm = _norm(dx2)
        dx1_norm_n = dx1_norm ** self.n
        dx2_norm_n = dx2_norm ** self.n

        denom = dx1_norm_n + dx2_norm_n
        if denom <= 0.0:
            raise RuntimeError("DWI interpolation denominator is zero.")

        w1 = dx2_norm_n / denom
        w2 = dx1_norm_n / denom

        e1, e2 = self.energies
        g1, g2 = self.gradients
        h1, h2 = self.hessians

        t1 = _taylor_energy(e1, g1, h1, dx1)
        t2 = _taylor_energy(e2, g2, h2, dx2)
        e_dwi = w1 * t1 + w2 * t2

        if not gradient:
            return e_dwi

        t1_grad = _taylor_grad(g1, h1, dx1)
        t2_grad = _taylor_grad(g2, h2, dx2)

        n_2 = self.n // 2
        dx1_norm_n_grad = 2 * n_2 * (dx1_norm ** (2 * n_2 - 2)) * dx1
        dx2_norm_n_grad = 2 * n_2 * (dx2_norm ** (2 * n_2 - 2)) * dx2
        w1_grad = (dx2_norm_n_grad * dx1_norm_n - dx1_norm_n_grad * dx2_norm_n) / (denom ** 2)
        w2_grad = -w1_grad

        g_dwi = w1_grad * t1 + w1 * t1_grad + w2_grad * t2 + w2 * t2_grad
        return e_dwi, g_dwi


# =============================== Parameters ===============================
@dataclass
class HPCParams:
    # Which negative eigenmode (1 = most negative) to use for initial direction
    target_mode: int = 1

    # HPC step length in Bohr
    step_length_bohr: float = 0.10

    # Number of macro steps per direction
    max_steps: int = 50

    # LQA predictor integration sub-steps
    euler_n: int = 5000

    # Recalculate Hessian every N micro-steps (None = never)
    hessian_recalc: Optional[int] = None

    # Hessian update method: "bfgs" or "bofill"
    hessian_update: str = "bofill"

    # DWI / mBS corrector controls
    dwi_n: int = 4
    mbs_max_k: int = 15
    mbs_points: int = 20
    mbs_tol: float = 1e-5

    # Convergence on forces in Cartesian space (Eh/Å)
    f_max_th: float = 2e-3
    f_rms_th: float = 5e-4

    # Output controls
    print_each: bool = True
    write_traj: bool = True
    batch_size: Optional[int] = None


# ================================== HPC ===================================
class HPC:
    """
    Hessian-based Predictor-Corrector(HPC):

    - Works in mass-weighted coordinates.
    - Each macro step uses LQA predictor propagation followed by mBS
      corrector integration on a DWI-fitted surface.
    - Uses BFGS/Bofill updates of the mass-weighted Hessian, with optional
      periodic full recalculation.
    - Integrates forward and backward from the TS along the lowest
      negative eigenmode of H_mw, then merges paths.
    """

    def __init__(self, atoms: Atoms, output: str,
                 params: Optional[HPCParams] = None,
                 paras: Optional[dict] = None):
        self.atoms = atoms
        self.output = output
        self.p = params if params is not None else HPCParams()

        # Parse optional dict overrides, supporting both {"hpc": {...}}
        # and flat dict style. Keep backward compatibility aliases.
        if isinstance(paras, dict):
            low = {k.lower(): v for k, v in paras.items()}
            sub = None
            for key in ("hpc", "irc"):
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
                "euler_n": "euler_n",
                "hessian_update": "hessian_update",
                "dwi_n": "dwi_n",
                "mbs_max_k": "mbs_max_k",
                "mbs_points": "mbs_points",
                "mbs_tol": "mbs_tol",
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

        # Internal state for HPC integration
        self._D: Optional[np.ndarray] = None  # mass-weight scaling vector
        self._step_len_mw: float = float(self.p.step_length_bohr * BOHR_TO_ANG)
        self._step_len_umw: float = float(self.p.step_length_bohr * BOHR_TO_ANG)
        self.mw_coords: Optional[np.ndarray] = None
        self.mw_hessian: Optional[np.ndarray] = None
        self.prev_coords: Optional[np.ndarray] = None
        self.prev_grad: Optional[np.ndarray] = None
        self.micro_counter: int = 0
        self._dwi: Optional[DWI] = None

    # ------------------------------ Public API ------------------------------
    def run(self) -> Dict[str, any]:
        """
        Compute forward and backward HPC IRC paths and produce a merged summary.

        Returns:
            {
                "forward": {...},
                "backward": {...},
                "summary": {...}
            }
        """
        # Prepare mass weights once at TS geometry
        self._D = masses_D(self.atoms)
        self._step_len_mw = float(self.p.step_length_bohr * BOHR_TO_ANG)
        self._step_len_umw = float(self.p.step_length_bohr * BOHR_TO_ANG)

        # Diagonalize mass-weighted Hessian at TS to get negative mode
        H_cart_ts = self._get_hessian_cart()
        H_mw_ts = (self._D[:, None] * H_cart_ts) * self._D[None, :]
        w, V = np.linalg.eigh(H_mw_ts)

        neg_idx = np.where(w < 0.0)[0]
        if len(neg_idx) == 0:
            log_error(
                ["[ERROR] HPC-IRC: No negative eigenvalues found — "
                 "starting geometry is not a saddle point.\n"],
                self.output,
            )
            raise RuntimeError("HPC-IRC: no negative eigenvalues at TS.")

        if len(neg_idx) < self.p.target_mode:
            log_error(
                [f"[ERROR] HPC-IRC: Requested mode {self.p.target_mode}, "
                 f"but only {len(neg_idx)} negative modes found.\n"],
                self.output,
            )
            raise RuntimeError("HPC-IRC: requested negative mode does not exist.")

        sorted_neg = neg_idx[np.argsort(w[neg_idx])]  # most negative first
        idx = sorted_neg[self.p.target_mode - 1]
        eigval = w[idx]
        v_neg_mw = V[:, idx]

        log_info(
            [
                "\n[INFO] HPC-IRC: Selected negative eigenmode "
                f"#{self.p.target_mode} with λ = {eigval:.6e} (MW basis)\n"
            ],
            self.output,
        )

        # Reference TS energy
        E_ts, F_ts_cart = energy_forces_single(self.atoms.calc, self.atoms)

        # Store original TS Cartesian positions, reused for both directions
        R_ts_cart = self.atoms.get_positions().copy().reshape(-1)

        # Forward and backward HPC-IRC
        forward_log = self._one_side(
            forward=True,
            sign=+1.0,
            q_ts_cart=R_ts_cart,
            v_neg_mw=v_neg_mw,
            E_ts=E_ts,
            H_cart_ts=H_cart_ts,
            F_ts_cart=F_ts_cart,
        )
        backward_log = self._one_side(
            forward=False,
            sign=-1.0,
            q_ts_cart=R_ts_cart,
            v_neg_mw=v_neg_mw,
            E_ts=E_ts,
            H_cart_ts=H_cart_ts,
            F_ts_cart=F_ts_cart,
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

    def _unweight_len(self, dq_mw: np.ndarray) -> float:
        """Return unweighted length of a MW displacement."""
        return _norm(dq_mw * self._D)

    def _scale_mw_step(self, direction_mw: np.ndarray, step_umw: float) -> np.ndarray:
        """Scale a MW direction to a target unweighted step length."""
        denom = self._unweight_len(direction_mw)
        if denom <= 1e-16:
            return np.zeros_like(direction_mw)
        return direction_mw * (step_umw / denom)

    def _unweight_grad(self, g_mw: np.ndarray) -> np.ndarray:
        """Convert MW gradient to Cartesian gradient."""
        return g_mw / self._D

    def _get_conv_fact(self, g_mw: np.ndarray, min_fact: float = 2.0) -> float:
        """Estimate MW-to-unweighted conversion factor for step length."""
        norm_mw = _norm(g_mw)
        if norm_mw <= 1e-16:
            return min_fact
        norm_cart = _norm(self._unweight_grad(g_mw))
        return max(min_fact, norm_cart / norm_mw)

    def _get_hessian_cart(self) -> np.ndarray:
        """
        Get Cartesian Hessian (3N x 3N) from the calculator.

        Assumes atoms.calc implements get_hessian(atoms) and returns 3N x 3N
        or shape (1, 3N, 3N).
        """
        H = hessian_single(self.atoms.calc, self.atoms, self.p.batch_size)
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
        E, F_cart = energy_forces_single(self.atoms.calc, self.atoms)
        return E, to_f64(F_cart).reshape(-1)

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
    def _bofill_update(H: np.ndarray,
                       s: np.ndarray,
                       y: np.ndarray) -> np.ndarray:
        """
        Bofill update:
        Combines Murtagh-Sargent and Powell-symmetric-Broyden updates using the Bofill mixing factor.
        """
        dx, dg = s, y

        z = dg - H.dot(dx)
        # MS
        ms = np.outer(z, z) / z.dot(dx)
        # PSB
        dx2 = dx.dot(dx)
        psb = ((np.outer(dx, z) + np.outer(z, dx)) / dx2) - (z.dot(dx)) * np.outer(dx, dx) / (dx2 * dx2)
        # Bofill mixing
        mix = (z.dot(dx) ** 2) / (z.dot(z) * dx2)

        dH = mix * ms + (1.0 - mix) * psb
        return H + dH

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

    # --------------------------- Micro-step (HPC) ----------------------------
    def _micro_step(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform one HPC step using LQA predictor + mBS corrector.

        Updates:
            self.mw_coords, self.mw_hessian, self.prev_coords,
            self.prev_grad, self.micro_counter

        Returns:
            dx: MW displacement for this step
            g_pred: predicted MW gradient (for diagnostics)
        """
        init_mw = self.mw_coords.copy()

        # Current energy / gradient
        e_curr, F_cart = self._energy_forces_from_mw(init_mw)
        g_curr = self._gradient_mw_from_forces(F_cart)
        if _norm(g_curr) < 1e-12:
            return np.zeros_like(g_curr), g_curr

        # Update Hessian (or optionally recalc)
        recalc = (
            self.p.hessian_recalc is not None
            and self.p.hessian_recalc > 0
            and (self.micro_counter % self.p.hessian_recalc == 0)
        )
        if recalc and self.micro_counter > 0:
            H_cart = self._get_hessian_cart()
            self.mw_hessian = (self._D[:, None] * H_cart) * self._D[None, :]
        elif (self.prev_coords is not None) and (self.prev_grad is not None):
            gradient_diff = g_curr - self.prev_grad
            coords_diff = init_mw - self.prev_coords
            if str(self.p.hessian_update).lower() == "bofill":
                self.mw_hessian = self._bofill_update(self.mw_hessian, coords_diff, gradient_diff)
            else:
                self.mw_hessian = self._bfgs_update(self.mw_hessian, coords_diff, gradient_diff)

        # Store for next update
        self.prev_coords = init_mw
        self.prev_grad = g_curr

        # Update DWI with current point (skip the very first step)
        if self._dwi is not None and self.micro_counter > 0:
            self._dwi.update(init_mw.copy(), e_curr, g_curr, self.mw_hessian.copy())

        # Predictor: LQA propagation using current Hessian
        eigvals, eigvecs = np.linalg.eigh(self.mw_hessian)
        mask = np.abs(eigvals) > 1e-8
        if not np.any(mask):
            return np.zeros_like(g_curr), g_curr

        eigvals = eigvals[mask]
        eigvecs = eigvecs[:, mask]

        g_star = eigvecs.T.dot(g_curr)
        g_norm = _norm(g_curr)
        if g_norm < 1e-12:
            return np.zeros_like(g_curr), g_curr

        dt = self._step_len_mw / (float(self.p.euler_n) * g_norm)
        t = dt
        cur_length = 0.0
        for _ in range(int(self.p.euler_n)):
            dsdt = np.sqrt(np.sum((g_star ** 2) * np.exp(-2.0 * eigvals * t)))
            cur_length += dsdt * dt
            if cur_length >= self._step_len_mw:
                break
            t += dt

        alphas = (np.exp(-eigvals * t) - 1.0) / eigvals
        dx_pred = eigvecs.dot(alphas * g_star)
        pred_mw = init_mw + dx_pred

        # Evaluate predicted point and update Hessian
        E_pred, F_pred = self._energy_forces_from_mw(pred_mw)
        g_pred = self._gradient_mw_from_forces(F_pred)

        dg_pred = g_pred - g_curr
        if str(self.p.hessian_update).lower() == "bofill":
            self.mw_hessian = self._bofill_update(self.mw_hessian, dx_pred, dg_pred)
        else:
            self.mw_hessian = self._bfgs_update(self.mw_hessian, dx_pred, dg_pred)

        # Update DWI with predicted point
        if self._dwi is not None:
            self._dwi.update(pred_mw.copy(), E_pred, g_pred, self.mw_hessian.copy())

        # Corrector: mBS integration on DWI surface
        if self._dwi is not None:
            corr_mw = self._corrector_step(init_mw, self._step_len_umw, self._dwi)
        else:
            corr_mw = pred_mw

        self.mw_coords = corr_mw
        self.micro_counter += 1

        dx = self.mw_coords - init_mw
        return dx, g_pred

    def _corrector_step(self, init_mw: np.ndarray, step_length: float, dwi: DWI) -> np.ndarray:
        """mBS (modified Bulirsch–Stoer) corrector integration on DWI PES."""
        errors = []
        richardson: Dict[Tuple[int, int], np.ndarray] = {}

        for k in range(int(self.p.mbs_max_k)):
            points = int(self.p.mbs_points) * (2 ** k)
            corr_step = step_length / (points - 1)
            cur_coords = init_mw.copy()
            k_coords: List[np.ndarray] = []
            cur_length = 0.0

            while True:
                k_coords.append(cur_coords.copy())
                if abs(step_length - cur_length) < 0.5 * corr_step:
                    break

                _, gradient = dwi.interpolate(cur_coords, gradient=True)
                grad_norm = _norm(gradient)
                if grad_norm < 1e-12:
                    break
                cur_coords = cur_coords + corr_step * (-gradient / grad_norm)
                cur_length = self._unweight_len(cur_coords - init_mw)

                # Oscillation check
                if len(k_coords) > 1:
                    prev_coords = k_coords[-2]
                    if _norm(cur_coords - prev_coords) <= corr_step:
                        return prev_coords

            richardson[(k, 0)] = cur_coords

            for j in range(1, k + 1):
                richardson[(k, j)] = (
                    (2 ** j) * richardson[(k, j - 1)] - richardson[(k - 1, j - 1)]
                ) / (2 ** j - 1)

            if k > 0:
                error = float(np.sqrt(np.mean((richardson[(k, k)] - richardson[(k, k - 1)]) ** 2)))
                errors.append(error)
                if error <= float(self.p.mbs_tol):
                    break
        else:
            raise RuntimeError("mBS Richardson extrapolation did not converge.")

        return richardson[(k, k)]

    # --------------------------- One direction ------------------------------
    def _one_side(self,
                  forward: bool,
                  sign: float,
                  q_ts_cart: np.ndarray,
                  v_neg_mw: np.ndarray,
                  E_ts: float,
                  H_cart_ts: np.ndarray,
                  F_ts_cart: np.ndarray) -> Dict[str, any]:
        """
        Single-sided HPC IRC integration.

        Args:
            forward: True for forward path, False for backward.
            sign: +1.0 or -1.0 to choose direction along the negative mode.
            q_ts_cart: TS Cartesian coordinates (flattened, Å).
            v_neg_mw: selected negative eigenmode in MW basis (3N, normalized later).
            E_ts: TS reference energy (Eh).
        """
        p = self.p
        title = "FORWARD HPC-IRC" if forward else "BACKWARD HPC-IRC"

        self._print_header(title)
        self._print_conv_thresholds(p.f_max_th, p.f_rms_th)

        # Initial MW coordinates at TS
        q_ts_mw = self._mw_from_cart(q_ts_cart)

        # Initial displacement along negative mode in MW
        v_dir = _unit(v_neg_mw) * sign
        q0_mw = q_ts_mw + self._scale_mw_step(v_dir, 0.5 * self._step_len_mw)

        # Gradient at TS (for initial Hessian update)
        g_ts_mw = self._gradient_mw_from_forces(F_ts_cart)

        # Initial Hessian at TS (MW)
        H_ts_cart = H_cart_ts.copy()
        H0_mw = (self._D[:, None] * H_ts_cart) * self._D[None, :]

        # Initial energy / forces / gradient at displaced point
        E0, F0_cart = self._energy_forces_from_mw(q0_mw)
        g0_mw = self._gradient_mw_from_forces(F0_cart)

        maxF0 = float(np.max(np.abs(F0_cart)))
        rmsF0 = float(np.sqrt(np.mean(F0_cart ** 2)))

        # Hessian update using TS -> displaced point
        if str(self.p.hessian_update).lower() == "bofill":
            H0_mw = self._bofill_update(H0_mw, q0_mw - q_ts_mw, g0_mw - g_ts_mw)
        else:
            H0_mw = self._bfgs_update(H0_mw, q0_mw - q_ts_mw, g0_mw - g_ts_mw)

        # Initialize HPC state
        self.mw_coords = q0_mw.copy()
        self.mw_hessian = H0_mw.copy()
        self.prev_coords = None
        self.prev_grad = None
        self.micro_counter = 0
        self._dwi = DWI(n=self.p.dwi_n, maxlen=2)
        self._dwi.update(q0_mw.copy(), E0, g0_mw, self.mw_hessian.copy())

        # Iteration 0 logging
        if p.print_each:
            self._print_iter_line(0, E0, (E0 - E_ts) * KCAL_PER_EH, maxF0, rmsF0)

        records: List[Dict[str, any]] = []
        records.append(
            {
                "E": E0,
                "forces": F0_cart.reshape(-1, 3).copy(),
                "maxG": maxF0,
                "rmsG": rmsF0,
                "x": self.atoms.get_positions().copy(),
            }
        )

        converged = False

        # Macro steps
        for it in range(1, p.max_steps + 1):
            dx, _ = self._micro_step()
            if _norm(dx) <= 1e-12:
                log_info(
                    [f"[INFO] {title}: step too small at step {it}, stopping.\n"],
                    self.output,
                )
                break

            # Energy / forces at new point
            E_new, F_new = self._energy_forces_from_mw(self.mw_coords)
            maxF = float(np.max(np.abs(F_new)))
            rmsF = float(np.sqrt(np.mean(F_new ** 2)))

            if p.print_each:
                self._print_iter_line(it, E_new, (E_new - E_ts) * KCAL_PER_EH, maxF, rmsF)

            records.append(
                {
                    "E": E_new,
                    "forces": F_new.reshape(-1, 3).copy(),
                    "maxG": maxF,
                    "rmsG": rmsF,
                    "x": self.atoms.get_positions().copy(),
                }
            )

            # Convergence in terms of Cartesian forces
            if (maxF <= p.f_max_th) and (rmsF <= p.f_rms_th):
                converged = True
                self._print_hurray()
                break

        return {"title": title, "records": records, "E_ts": E_ts,
                "iterations": len(records) - 1, "converged": converged}

    # --------------------------- Merge & summary ----------------------------
    def _merge_and_mark_ts(self, f: Dict, b: Dict) -> Dict:
        fR, bR = f["records"], b["records"]
        if not fR or not bR:
            log_error(["[ERROR] HPC-IRC: One path side is empty.\n"], self.output)
            raise RuntimeError("HPC-IRC: one path side is empty.")

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
                "                       HPC-IRC PATH SUMMARY           \n",
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
        ext = ".pdb" if self.atoms.info.get("pdb_template") else ".xyz"
        full_path = base + "_full" + ext
        fwd_path = base + "_forward" + ext
        bwd_path = base + "_backward" + ext

        # Forward
        f_atoms, f_E, f_F = [], [], []
        for rec in f["records"]:
            a = self.atoms.copy()
            a.set_positions(rec["x"])
            f_atoms.append(a)
            f_E.append(rec["E"])
            f_F.append(rec["forces"])
        write_xyz(fwd_path, f_atoms, f_E, forces=f_F)

        # Backward
        b_atoms, b_E, b_F = [], [], []
        for rec in b["records"]:
            a = self.atoms.copy()
            a.set_positions(rec["x"])
            b_atoms.append(a)
            b_E.append(rec["E"])
            b_F.append(rec["forces"])
        write_xyz(bwd_path, b_atoms, b_E, forces=b_F)

        # Full (concatenate forward then backward)
        full_atoms = f_atoms + b_atoms
        full_E = f_E + b_E
        write_xyz(full_path, full_atoms, full_E, forces=f_F + b_F)

        log_info(
            [
                f"\n[INFO] HPC-IRC forward trajectory written to: {fwd_path}\n",
                f"[INFO] HPC-IRC backward trajectory written to: {bwd_path}\n",
                f"[INFO] HPC-IRC full trajectory written to: {full_path}\n",
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
                "                      ***          THE HPC-IRC HAS CONVERGED        ***\n",
                "                      *************************************************\n\n",
            ],
            self.output,
        )
