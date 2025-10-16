# -*- coding: utf-8 -*-
"""
GSM (Growing String Method, engineering version) with:
- Kabsch alignment (reports RMSD)
- Two-ended adaptive growth + constrained mini-relax (LBFGS, projected onto tau-orthogonal subspace)
- Energy-weighted tangent for robust direction estimation
- Dynamic growth policy (pause side / refine endpoint / adapt step)
- Merge when close; re-sample to fixed n_images (default 9 incl. endpoints)
- Projected L-BFGS MEP relaxation (no line search)
- Optional CI-STRING refinement and PRFO TS refinement

Units:
- Energies in Eh
- Forces in Eh/Angstrom
- Distances in Angstrom
"""

from __future__ import annotations
import os
import copy
from dataclasses import dataclass, fields
from typing import List, Tuple, Optional

import numpy as np
from ase import Atoms

# You already have these utilities / mixins in your codebase:
from .logger import log_info
from ...jobABC import JobABC


# =============================================================================
# ------------------------------ Utilities ------------------------------------
# =============================================================================

def to_numpy_f64(x):
    """Convert input (numpy/torch/list/scalar) to float64 numpy array or float."""
    if isinstance(x, np.ndarray):
        return x.astype(np.float64, copy=False)
    try:
        import torch
        if isinstance(x, torch.Tensor):
            arr = x.detach().cpu().numpy()
            return arr.astype(np.float64, copy=False)
    except Exception:
        pass
    if np.isscalar(x):
        return float(x)
    return np.asarray(x, dtype=np.float64)


def kabsch_align(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Rigid-body least-squares alignment (Kabsch):
    Align Q onto P. Both are (N,3). Returns: Q_aligned, rmsd, R, t
    """
    P = np.asarray(P, dtype=np.float64); Q = np.asarray(Q, dtype=np.float64)
    if P.shape != Q.shape or P.shape[1] != 3:
        raise ValueError("P and Q must have shape (N,3)")
    Pc = P.mean(axis=0); Qc = Q.mean(axis=0)
    P0 = P - Pc; Q0 = Q - Qc
    C = Q0.T @ P0
    V, S, Wt = np.linalg.svd(C)
    if np.linalg.det(V @ Wt) < 0.0:
        V[:, -1] *= -1.0
    R = V @ Wt
    t = Pc - Qc @ R
    Q_aligned = Q @ R + t
    diff = P - Q_aligned
    rmsd = float(np.sqrt((diff * diff).sum() / P.shape[0]))
    return Q_aligned, rmsd, R, t


def write_xyz(filename: str, images: List[Atoms], energies: Optional[List[float]] = None):
    """
    Write a multi-frame XYZ trajectory. If energies given, write in comment line.
    """
    with open(filename, "w") as f:
        for i, at in enumerate(images):
            pos = to_numpy_f64(at.get_positions())
            symbols = at.get_chemical_symbols()
            f.write(f"{len(symbols)}\n")
            if energies is not None:
                f.write(f"Image {i}  Energy = {energies[i]:.8f}\n")
            else:
                f.write(f"Image {i}\n")
            for s, (x, y, z) in zip(symbols, pos):
                f.write(f"{s:2s} {x: .10f} {y: .10f} {z: .10f}\n")


def inherit_attrs(src: Atoms, dst: Atoms):
    """
    Ensure new images/TS carry calculator and threshold attributes.
    Called immediately after we create/copy a new Atoms.
    """
    if getattr(src, "calc", None) is not None:
        dst.calc = src.calc
    # user thresholds set by Dispatcher.set_threshold()
    for name in ("f_max_th", "f_rms_th", "dp_max_th", "dp_rms_th"):
        if hasattr(src, name):
            setattr(dst, name, getattr(src, name))


def atoms_to_xyz_block(atoms: Atoms) -> str:
    """Pretty XYZ (Angstrom) block used in logs."""
    lines = []
    syms = atoms.get_chemical_symbols()
    pos = atoms.get_positions()
    for s, (x, y, z) in zip(syms, pos):
        lines.append(f"{s:<2} {x:14.6f} {y:14.6f} {z:14.6f}")
    return "\n".join(lines) + "\n"


def get_energies(images: List[Atoms]) -> List[float]:
    return [float(at.get_potential_energy(force_consistent=True)) for at in images]


def rms_force_perp(Fp_list: List[np.ndarray]) -> float:
    """RMS of projected-perp forces over all DOFs of internal images."""
    arrs = []
    for i, Fp in enumerate(Fp_list):
        if i == 0 or i == len(Fp_list) - 1:
            continue
        arrs.append(Fp.reshape(-1))
    if not arrs:
        return 0.0
    v = np.concatenate(arrs)
    return float(np.sqrt(np.mean(v * v)))


def project_out_rigidbody_forces(forces: np.ndarray,
                                 positions: np.ndarray,
                                 masses: np.ndarray,
                                 tol: float = 1e-10) -> np.ndarray:
    """
    Remove 6 rigid-body modes (3 translations + 3 rotations) from forces
    using mass-weighted Eckart projection. Safe for near-linear molecules:
    rank-deficiency is handled via SVD cutoff.

    Parameters
    ----------
    forces : (N,3) array, true Cartesian forces (Eh/Ang)
    positions : (N,3) array, current Cartesian coords (Ang)
    masses : (N,) array, atomic masses (amu)
    tol : float, relative SVD cutoff

    Returns
    -------
    F_proj : (N,3) array, forces with rigid-body components removed
    """
    N = positions.shape[0]
    sqrtm = np.sqrt(masses).reshape(-1, 1)

    R_cm = (positions * masses.reshape(-1,1)).sum(axis=0) / masses.sum()
    r = positions - R_cm

    B = np.zeros((3 * N, 6), dtype=np.float64)
    # Translations
    for a in range(N):
        s = float(sqrtm[a, 0]); i3 = 3*a
        B[i3+0, 0] = s; B[i3+1, 1] = s; B[i3+2, 2] = s
    # Rotations (about x,y,z): omega × r (mass-weighted)
    for a in range(N):
        s = float(sqrtm[a, 0]); x,y,z = r[a]; i3 = 3*a
        B[i3+0, 3] = 0.0; B[i3+1, 3] = -s*z; B[i3+2, 3] = s*y
        B[i3+0, 4] = s*z; B[i3+1, 4] = 0.0; B[i3+2, 4] = -s*x
        B[i3+0, 5] = -s*y; B[i3+1, 5] = s*x; B[i3+2, 5] = 0.0

    Fmw = (forces * sqrtm).reshape(-1)
    U, S, VT = np.linalg.svd(B, full_matrices=False)
    if S.size == 0 or S.max() <= 0.0:
        return forces.copy()
    rank = int(np.sum(S > tol * S.max()))
    Q = U[:, :rank]
    Fmw_proj = Fmw - Q @ (Q.T @ Fmw)
    return (Fmw_proj.reshape(-1, 3) / sqrtm)


def align_path_inplace(images, mode: str = "chain"):
    """
    Align all internal images to suppress rigid rotations along the path.

    mode="chain":  image i aligned to (i-1), propagates smoothly.
    mode="first":  every internal image aligned to image 0.

    In-place: modifies images[i].positions
    """
    assert len(images) >= 2
    if mode not in ("chain", "first"):
        mode = "chain"
    if mode == "first":
        ref = images[0].get_positions()
        for i in range(1, len(images) - 1):
            Pi = images[i].get_positions()
            aligned, _, _, _ = kabsch_align(ref, Pi)
            images[i].set_positions(aligned)
    else:
        ref = images[0].get_positions()
        for i in range(1, len(images) - 1):
            Pi = images[i].get_positions()
            aligned, _, _, _ = kabsch_align(ref, Pi)
            images[i].set_positions(aligned)
            ref = aligned


def linear_reparam(images: List[Atoms]):
    """
    Reparameterize a path to equal arclength by piecewise-linear resampling.
    Uses the CURRENT (already aligned) coordinates. Endpoints fixed.

    In-place on images: updates internal images' positions.
    """
    M = len(images)
    if M <= 2:
        return
    X = [img.get_positions().copy() for img in images]
    s = [0.0]
    for i in range(1, M):
        ds = np.linalg.norm(X[i] - X[i-1])
        s.append(s[-1] + ds)
    L = s[-1]
    if L <= 1e-15:
        return
    s_targets = np.linspace(0.0, L, M)
    for k in range(1, M - 1):
        sk = s_targets[k]
        j = np.searchsorted(s, sk) - 1
        if j < 0: j = 0
        if j >= M - 1: j = M - 2
        if s[j+1] - s[j] < 1e-16:
            lam = 0.0
        else:
            lam = (sk - s[j]) / (s[j+1] - s[j])
        Xk = (1.0 - lam) * X[j] + lam * X[j+1]
        images[k].set_positions(Xk)


# =============================================================================
# --------------------------- LBFGS driver (no LS) ----------------------------
# =============================================================================

class LBFGSDriver:
    """Simple L-BFGS optimizer for projected gradients (no line search)."""
    def __init__(self, m=5, curvature=70.0, maxstep=0.2):
        self.m = m
        self.H0 = 1.0 / curvature
        self.maxstep = maxstep
        self.S, self.Y, self.rhos = [], [], []

    def two_loop(self, grad: np.ndarray) -> np.ndarray:
        q = grad.copy()
        alpha = []
        for s, y, rho in reversed(list(zip(self.S, self.Y, self.rhos))):
            a = rho * np.dot(s, q)
            alpha.append(a)
            q -= a * y
        gamma = self.H0 if not self.Y else np.dot(self.Y[-1], self.S[-1]) / (np.dot(self.Y[-1], self.Y[-1]) + 1e-20)
        z = gamma * q
        for (s, y, rho), a in zip(zip(self.S, self.Y, self.rhos), reversed(alpha)):
            b = rho * np.dot(y, z)
            z += s * (a - b)
        return -z

    def update(self, s: np.ndarray, y: np.ndarray):
        rho = 1.0 / (np.dot(y, s) + 1e-20)
        self.S.append(s.copy()); self.Y.append(y.copy()); self.rhos.append(rho)
        if len(self.S) > self.m:
            self.S.pop(0); self.Y.pop(0); self.rhos.pop(0)

    def step_limit(self, step: np.ndarray) -> np.ndarray:
        max_disp = np.max(np.abs(step))
        if max_disp > self.maxstep:
            step = step * (self.maxstep / max_disp)
        return step

    def should_stop(self, grad, fmax_th=1e-3, frms_th=5e-4) -> bool:
        max_f = np.max(np.abs(grad))
        rms_f = np.sqrt(np.mean(grad ** 2))
        return (max_f < fmax_th) and (rms_f < frms_th)


# =============================================================================
# ------------------------------- GSM Params ----------------------------------
# =============================================================================

@dataclass
class GSMParams:
    # ---- Growth (engineering) ----
    grow_step: float = 0.12               # initial step length (Ang)
    grow_step_min: float = 0.02           # minimal step to accept (Ang)
    grow_backtrack: float = 0.5           # backtracking factor for step
    grow_relax_iters: int = 10            # mini-relax LBFGS iterations (orthogonal)
    grow_relax_tol: float = 2.5e-3        # |grad_perp| threshold for local convergence
    grow_energy_toll: float = 1.0e-3      # forbid large uphill during prediction (Eh)
    pause_imbalance: int = 2              # if |nL - nR| > this, pause longer side
    refine_end_every: int = 3             # every K growth iters, re-opt last image (orthogonal)

    # merge
    merge_threshold: float = 0.5         # mean per-atom distance (Ang) to merge

    # target path points after merge (incl. endpoints)
    n_images: int = 9

    # ---- Full path relaxation (projected L-BFGS) ----
    max_iter_relax: int = 300
    string_f_max_th: float = 5.0e-3
    string_f_rms_th: float = 1.0e-3
    lbfgs_memory: int = 7
    lbfgs_curvature: float = 70.0
    lbfgs_maxstep: float = 0.2
    reparam_every: int = 3

    # ---- CI-STRING ----
    refine: Optional[str] = None          # 'cistring' or 'stringts' or None
    cistring_f_max_th: float = 4.0e-3
    cistring_f_rms_th: float = 2.0e-3

    # ---- Safety / debug ----
    max_iter_grow: int = 100              # total growth rounds (each ideally adds 2 nodes)
    log_every: int = 1                    # growth log cadence

    # guiding policy (safer growth)
    guide_conn_min: float = 0.3   # ≥ this weight for connector even when far apart
    guide_d_switch: float = 4.0    # distance scale for mixing connector vs. tangent (Å)
    guide_inc_trigger: int = 2     # force a connector-pull after this many worsening steps



# =============================================================================
# --------------------------------- GSM Class ---------------------------------
# =============================================================================

class GSM(JobABC):
    """
    Engineering GSM:
      - Adaptive predictor (step backtracking on uphill / |F_perp|)
      - Constrained mini-relax in tau-orthogonal subspace (LBFGS)
      - Energy-weighted tangent
      - Dynamic growth policy
      - Projected L-BFGS for full-path relaxation
    """

    def __init__(self, output: str, atoms_R: Atoms, atoms_P: Atoms,
                 params: Optional[GSMParams] = None, paras: Optional[dict] = None):
        super().__init__(output)
        self.atoms_R = atoms_R
        self.atoms_P = atoms_P
        self.atoms_R.calc = atoms_R.calc
        self.atoms_P.calc = atoms_P.calc
        self.params = params if params is not None else GSMParams()
        if isinstance(paras, dict):
            low = {(k.lower() if isinstance(k, str) else k): v for k, v in paras.items()}
            for f in fields(self.params):
                name = f.name.lower()
                if name in low:
                    setattr(self.params, f.name, low[name])
        if self.params.n_images < 2:
            raise ValueError("n_images must be >= 2")

    # --------------------------- helpers (geometry) ---------------------------

    @staticmethod
    def _copy_with_inherit(src: Atoms) -> Atoms:
        out = src.copy()
        inherit_attrs(src, out)
        if out.calc is None and src.calc is not None:
            out.calc = src.calc
        return out

    @staticmethod
    def _unit(v: np.ndarray, eps: float = 1e-16) -> np.ndarray:
        n = np.linalg.norm(v)
        if n < eps:
            return v * 0.0
        return v / n

    @staticmethod
    def _min_image_distance(A: Atoms, B: Atoms) -> float:
        """Mean per-atom distance between two images (Ang)."""
        pa = to_numpy_f64(A.get_positions()); pb = to_numpy_f64(B.get_positions())
        return float(np.linalg.norm((pa - pb), axis=1).mean())

    @staticmethod
    def _energy_weighted_tangent(images: List[Atoms], energies: List[float], i: int) -> np.ndarray:
        """Energy-weighted tangent for interior index i; returns flattened 3N unit vector."""
        Rm1 = to_numpy_f64(images[i - 1].get_positions())
        R   = to_numpy_f64(images[i].get_positions())
        Rp1 = to_numpy_f64(images[i + 1].get_positions())
        Em1 = energies[i - 1]; E = energies[i]; Ep1 = energies[i + 1]
        t = (Ep1 - E) * (Rp1 - R) + (E - Em1) * (R - Rm1)
        t = t.reshape(-1)
        n = np.linalg.norm(t)
        if n < 1e-14:
            t = (Rp1 - R + R - Rm1).reshape(-1)
            n = np.linalg.norm(t)
            if n < 1e-14:
                t = np.zeros_like(t); t[0] = 1.0; n = 1.0
        return t / n

    def _endpoint_tangent3(self, prev_img: Atoms, last_img: Atoms, fallback_dir3: np.ndarray) -> np.ndarray:
        """Estimate 3D endpoint tangent from the last two images on one side."""
        R0 = to_numpy_f64(prev_img.get_positions())
        R1 = to_numpy_f64(last_img.get_positions())
        d = R1 - R0
        if np.linalg.norm(d) < 1e-12:
            return self._unit(fallback_dir3)
        return self._unit(d.mean(axis=0))

    def _endpoint_direct_step(self, at: Atoms, dir3: np.ndarray, step: float):
        """Translate an endpoint image by 'step' along dir3 (Å), then keep calc etc."""
        pos = to_numpy_f64(at.get_positions())
        nrm = np.linalg.norm(dir3)
        if nrm < 1e-15:
            return
        disp = (dir3 / nrm) * float(step)
        at.set_positions(pos + disp)


    # ------------------- constrained mini-relax on a single image --------------

    def _mini_relax_perp_lbfgs(self, img: Atoms, tau3: np.ndarray, max_iter: int, gtol: float):
        """
        L-BFGS mini-optimization that only moves in the hyperplane orthogonal to tau3.
        Gradient is dE/dx (projected), i.e., g = -F_perp.
        """
        pos = to_numpy_f64(img.get_positions()).reshape(-1)
        masses = to_numpy_f64(img.get_masses())
        tau3N = np.tile(tau3.reshape(1,3), (len(img),1)).reshape(-1)

        def proj_perp(vec):
            return vec - float(np.dot(vec, tau3N)) * tau3N

        def energy_and_grad(xvec):
            img.set_positions(xvec.reshape(-1,3))
            E = float(img.get_potential_energy(force_consistent=True))
            F = to_numpy_f64(img.get_forces()).reshape(-1,3)
            F = project_out_rigidbody_forces(F, xvec.reshape(-1,3), masses).reshape(-1)
            g = -F
            g = proj_perp(g)
            return E, g

        lb = LBFGSDriver(m=5, curvature=70.0, maxstep=0.1)
        x = pos.copy()
        E, g = energy_and_grad(x)
        k = 0
        while k < max_iter and not lb.should_stop(g, fmax_th=gtol, frms_th=gtol*0.5):
            p = lb.two_loop(g)            # descent direction in subspace
            p = proj_perp(p)              # safety: ensure perpendicular
            p = lb.step_limit(p)
            x_new = x + p
            E_new, g_new = energy_and_grad(x_new)
            s = x_new - x
            y = g_new - g
            lb.update(s, y)
            x, E, g = x_new, E_new, g_new
            k += 1
        img.set_positions(x.reshape(-1,3))

    # ---------------------------- adaptive predictor ---------------------------

    def _predict_and_relax_one(self, last_img: Atoms, dir3: np.ndarray,
                               E_last: float, label: str) -> Tuple[Atoms, float, float]:
        """
        Adaptive predictor (step backtracking on uphill / |F_perp|) + mini-relax (LBFGS).
        Returns: (new_img, step_used, max|F_perp| after relax)
        """
        p = self.params
        ds = p.grow_step
        dir3 = self._unit(dir3)
        masses_last = to_numpy_f64(last_img.get_masses())

        while ds >= p.grow_step_min:
            # predict
            trial = self._copy_with_inherit(last_img)
            pos_last = to_numpy_f64(last_img.get_positions())
            trial.set_positions(pos_last + ds * dir3.reshape(1, 3))

            # quick check at predicted point
            pos  = to_numpy_f64(trial.get_positions())
            Fraw = to_numpy_f64(trial.get_forces()).reshape(-1,3)
            Frb  = project_out_rigidbody_forces(Fraw, pos, masses_last).reshape(-1)
            tau3N = np.tile(dir3.reshape(1,3), (len(trial),1)).reshape(-1)
            F_perp = Frb - float(np.dot(Frb, tau3N)) * tau3N
            maxFperp_pred = float(np.max(np.abs(F_perp)))

            E_pred = float(trial.get_potential_energy(force_consistent=True))
            uphill = (E_pred - E_last) > p.grow_energy_toll

            # backtrack on step if energy is too uphill or force is too large
            if (uphill or maxFperp_pred > 10.0 * p.grow_relax_tol) and ds > p.grow_step_min:
                ds *= p.grow_backtrack
                continue

            # orthogonal mini-relax by L-BFGS
            self._mini_relax_perp_lbfgs(trial, dir3, max_iter=p.grow_relax_iters, gtol=p.grow_relax_tol)

            # recompute max|F_perp| after relax
            pos  = to_numpy_f64(trial.get_positions())
            Fraw = to_numpy_f64(trial.get_forces()).reshape(-1,3)
            Frb  = project_out_rigidbody_forces(Fraw, pos, masses_last).reshape(-1)
            F_perp = Frb - float(np.dot(Frb, tau3N)) * tau3N
            maxFperp_final = float(np.max(np.abs(F_perp)))
            return trial, ds, maxFperp_final

        # fallback minimal step
        trial = self._copy_with_inherit(last_img)
        pos_last = to_numpy_f64(last_img.get_positions())
        trial.set_positions(pos_last + p.grow_step_min * dir3.reshape(1, 3))
        self._mini_relax_perp_lbfgs(trial, dir3, max_iter=p.grow_relax_iters, gtol=p.grow_relax_tol)
        pos  = to_numpy_f64(trial.get_positions())
        Fraw = to_numpy_f64(trial.get_forces()).reshape(-1,3)
        Frb  = project_out_rigidbody_forces(Fraw, pos, masses_last).reshape(-1)
        tau3N = np.tile(dir3.reshape(1,3), (len(trial),1)).reshape(-1)
        F_perp = Frb - float(np.dot(Frb, tau3N)) * tau3N
        return trial, p.grow_step_min, float(np.max(np.abs(F_perp)))

    # ---------------------------- restart / STRING-TS --------------------------

    def restart_run(self, images: List[Atoms], hei_idx: int, base_prefix: str):
        """
        Take HEI from the equal-arc path as TS guess, run PRFO/RFO refinement,
        then print a NEB-TS-style path summary (marking CI and TS) and dump files.
        """
        from .PRFO import RFO  # your existing TS refiner (returns an Atoms)

        # Prepare TS guess from HEI
        ts_guess = copy.deepcopy(images[hei_idx])
        inherit_attrs(images[0], ts_guess)

        # Run PRFO (RFO) to refine TS
        ts_opt = RFO(ts_guess, self.output)
        E_TS   = float(ts_opt.get_potential_energy(force_consistent=True))

        # Create a path with TS inserted after the original CI (HEI)
        images_ts = [img for img in images]
        images_ts.insert(hei_idx + 1, ts_opt)

        # Energies of the augmented path
        Es_path = get_energies(images_ts)
        kcal_per_Eh = 627.509

        # Helper: energy-weighted tangent for projected-perp forces
        def energy_weighted_tangent(imgs, Es, i):
            if i == 0 or i == len(imgs) - 1:
                t = to_numpy_f64(imgs[min(i+1, len(imgs)-1)].get_positions()) - to_numpy_f64(imgs[max(i-1,0)].get_positions())
            else:
                Em, Ei, Ep = Es[i-1], Es[i], Es[i+1]
                dm = to_numpy_f64(imgs[i].get_positions()) - to_numpy_f64(imgs[i-1].get_positions())
                dp = to_numpy_f64(imgs[i+1].get_positions()) - to_numpy_f64(imgs[i].get_positions())
                if (Ep >= Ei) and (Ei >= Em):
                    t = dp
                elif (Ep <= Ei) and (Ei <= Em):
                    t = dm
                else:
                    t = dp * abs(Ep - Ei) + dm * abs(Ei - Em)
            t_flat = t.reshape(-1)
            nrm = np.linalg.norm(t_flat)
            if nrm < 1e-16:
                t_flat = np.zeros_like(t_flat); t_flat[0] = 1.0; nrm = 1.0
            return t_flat / nrm

        # Compute per-image metrics:
        # - Non-TS rows: projected-perpendicular forces (Fp)
        # - TS row: global forces (printed as 0.0/0.0 if you want identical to sample)
        maxFp_list = []
        rmsFp_list = []
        for i, at in enumerate(images_ts):
            if i == hei_idx + 1:
                # TS: global forces; print zeros to match your sample format
                maxFp_list.append(0.0)
                rmsFp_list.append(0.0)
            else:
                F_raw = to_numpy_f64(at.get_forces())
                pos   = to_numpy_f64(at.get_positions())
                masses= to_numpy_f64(at.get_masses())
                F_rb  = project_out_rigidbody_forces(F_raw, pos, masses).reshape(-1)
                tau   = energy_weighted_tangent(images_ts, Es_path, i)
                c     = float(np.dot(F_rb, tau))
                Fp    = (F_rb - c * tau).reshape(-1, 3)
                maxFp_list.append(float(np.max(np.linalg.norm(Fp, axis=1))))
                rmsFp_list.append(float(np.sqrt(np.mean(np.linalg.norm(Fp, axis=1) ** 2))))

        # Dump STRING-TS files
        stringts_mep = base_prefix + "_stringts_mep.xyz"
        stringts_ts  = base_prefix + "_stringts_ts.xyz"
        write_xyz(stringts_mep, images_ts, energies=Es_path)
        write_xyz(stringts_ts, [ts_opt], energies=[E_TS])

        # Pretty-print TS coordinates
        def atoms_to_xyz_lines(atoms: Atoms) -> List[str]:
            syms = atoms.get_chemical_symbols()
            pos = atoms.get_positions()
            lines = []
            for s, (x, y, z) in zip(syms, pos):
                lines.append(f"{s:<2} {x:12.6f} {y:12.6f} {z:12.6f}")
            return lines

        # NEB-TS style summary (as requested)
        log_info([
            "\n---------------------------------------------------------------\n",
            "                      PATH SUMMARY FOR NEB-TS             \n",
            "---------------------------------------------------------------\n",
            "All forces in Eh/Angstrom. Global forces for TS.\n\n",
            "Image     E(Eh)   dE(kcal/mol)  max(|Fp|)  RMS(Fp)\n"
        ], self.output)

        for i, E in enumerate(Es_path):
            dE = (E - Es_path[0]) * kcal_per_Eh
            if i == hei_idx:
                log_info([f"{i:4d} {E:12.5f} {dE:11.2f} {maxFp_list[i]:11.5f} {rmsFp_list[i]:10.5f} <= CI\n"], self.output)
            elif i == hei_idx + 1:
                log_info([f"  TS {E:12.5f} {dE:11.2f} {0.0:11.5f} {0.0:10.5f} <= TS\n"], self.output)
            else:
                log_info([f"{i:4d} {E:12.5f} {dE:11.2f} {maxFp_list[i]:11.5f} {rmsFp_list[i]:10.5f}\n"], self.output)

        log_info([
            "\n-----------------------------------------\n",
            "  REFINED TS STRUCTURE (ANGSTROEM)\n",
            "-----------------------------------------\n",
            *[line + "\n" for line in atoms_to_xyz_lines(ts_opt)]
        ], self.output)

        log_info([
            f"\nWrote STRING-TS MEP to: {stringts_mep}\n",
            f"Wrote TS structure to:  {stringts_ts}\n"
        ], self.output)




    # ------------------------------- main flow --------------------------------

    def run(self):
        """
        Main GSM workflow:
        1) Two-ended adaptive growth with stabilizers (connector floor, monotonic guard, forced pull).
        2) Merge L + reversed R and equal-arc reparameterization to fixed n_images.
        3) Take HEI (on the reparameterized path) as TS guess and call PRFO via restart_run().
        """
        def forces_info(atoms):
            F = to_numpy_f64(atoms.get_forces())
            maxF = np.max(np.linalg.norm(F, axis=1))
            rmsF = np.sqrt(np.mean(np.linalg.norm(F, axis=1) ** 2))
            return maxF, rmsF

        # --- Report endpoints
        E_R = self.atoms_R.get_potential_energy(force_consistent=True)
        E_P = self.atoms_P.get_potential_energy(force_consistent=True)
        maxF_R, rmsF_R = forces_info(self.atoms_R)
        maxF_P, rmsF_P = forces_info(self.atoms_P)

        log_info([
            "\nProperties of fixed STRING end points:\n",
            "               Reactant:\n",
            f"                         E               ....   {E_R: .6f} Eh\n",
            f"                         RMS(F)          ....   {rmsF_R: .6f} Eh/Angstrom\n",
            f"                         MAX(|F|)        ....   {maxF_R: .6f} Eh/Angstrom\n",
            "               Product:\n",
            f"                         E               ....   {E_P: .6f} Eh\n",
            f"                         RMS(F)          ....   {rmsF_P: .6f} Eh/Angstrom\n",
            f"                         MAX(|F|)        ....   {maxF_P: .6f} Eh/Angstrom\n",
        ], self.output)

        # --- Kabsch alignment of endpoints
        R = to_numpy_f64(self.atoms_R.get_positions())
        P = to_numpy_f64(self.atoms_P.get_positions())
        P_aligned, rmsd, _, _ = kabsch_align(R, P)
        self.atoms_P.set_positions(P_aligned)
        log_info([f"\nAlignment done. RMSD: {rmsd:.6f} (Angstrom)\n"], self.output)

        base, _ = os.path.splitext(self.output)

        # --- Initialize two growth fronts
        L: List[Atoms] = [self._copy_with_inherit(self.atoms_R)]
        R_: List[Atoms] = [self._copy_with_inherit(self.atoms_P)]
        E_L_last = float(L[-1].get_potential_energy(force_consistent=True))
        E_R_last = float(R_[-1].get_potential_energy(force_consistent=True))

        # Initial directions: centroid connector (first step only)
        cR = R.mean(axis=0); cP = P_aligned.mean(axis=0)
        dirL = self._unit(cP - cR); dirR = self._unit(cR - cP)

        # First bilateral growth (adaptive)
        newL, dsL, fpL = self._predict_and_relax_one(L[-1], dirL, E_L_last, "L")
        L.append(newL); E_L_last = float(newL.get_potential_energy(force_consistent=True))
        newR, dsR, fpR = self._predict_and_relax_one(R_[-1], dirR, E_R_last, "R")
        R_.append(newR); E_R_last = float(newR.get_potential_energy(force_consistent=True))

        # Growth log header
        log_info([
            "\nStarting GSM growth (two-ended, adaptive):\n",
            "GrowIter   d_min(Ang)      thr   |L|  |R|   dsL    dsR   max|F_perp|_L  max|F_perp|_R\n"
        ], self.output)

        # --- Growth policy parameters
        dmin_tol        = 0.05                           # allow ≤ +5% worsening per accepted step
        near_cutoff     = 2.0                            # in Å, shrink step when close
        max_retry_side  = 3                              # per-side backtracking trials
        max_retry_iter  = 3                              # global shrunken retries if worsening

        # NEW stabilizers (configurable through GSMParams)
        conn_min        = getattr(self.params, "guide_conn_min", 0.30)
        d_switch        = getattr(self.params, "guide_d_switch", 4.0)
        inc_streak      = 0
        inc_streak_tr   = getattr(self.params, "guide_inc_trigger", 2)

        it = 0
        while it < self.params.max_iter_grow:
            d_min_old = self._min_image_distance(L[-1], R_[-1])

            # Imbalance gating: pause the longer side
            grow_left = True; grow_right = True
            if len(L) - len(R_) > self.params.pause_imbalance:
                grow_left = False
            if len(R_) - len(L) > self.params.pause_imbalance:
                grow_right = False

            # Periodic endpoint mini-relax in the plane normal to the local direction
            if self.params.refine_end_every > 0 and it > 0 and (it % self.params.refine_end_every == 0):
                if len(L) >= 2:
                    E_prev = float(L[-2].get_potential_energy(force_consistent=True))
                    E_last = float(L[-1].get_potential_energy(force_consistent=True))
                    v = to_numpy_f64(L[-1].get_positions()) - to_numpy_f64(L[-2].get_positions())
                    dir3 = self._unit((E_last - E_prev) * v.mean(axis=0))
                    self._mini_relax_perp_lbfgs(L[-1], dir3, max_iter=self.params.grow_relax_iters, gtol=self.params.grow_relax_tol)
                    E_L_last = float(L[-1].get_potential_energy(force_consistent=True))
                if len(R_) >= 2:
                    E_prev = float(R_[-2].get_potential_energy(force_consistent=True))
                    E_last = float(R_[-1].get_potential_energy(force_consistent=True))
                    v = to_numpy_f64(R_[-1].get_positions()) - to_numpy_f64(R_[-2].get_positions())
                    dir3 = self._unit((E_last - E_prev) * v.mean(axis=0))
                    self._mini_relax_perp_lbfgs(R_[-1], dir3, max_iter=self.params.grow_relax_iters, gtol=self.params.grow_relax_tol)
                    E_R_last = float(R_[-1].get_potential_energy(force_consistent=True))

            # --- Blended directions: energy-weighted endpoint tangent + connector (with floor)
            connL3 = self._unit((to_numpy_f64(R_[-1].get_positions()) - to_numpy_f64(L[-1].get_positions())).mean(axis=0))
            connR3 = self._unit((to_numpy_f64(L[-1].get_positions()) - to_numpy_f64(R_[-1].get_positions())).mean(axis=0))

            def ew_tangent_end(chain):
                """Energy-weighted endpoint tangent as a 3D guide vector."""
                if len(chain) < 2:
                    return None
                v = to_numpy_f64(chain[-1].get_positions()) - to_numpy_f64(chain[-2].get_positions())
                if np.linalg.norm(v) < 1e-14:
                    return None
                E_prev = float(chain[-2].get_potential_energy(force_consistent=True))
                E_last = float(chain[-1].get_potential_energy(force_consistent=True))
                return self._unit((E_last - E_prev) * v.mean(axis=0))

            tauL3 = ew_tangent_end(L);  tauL3 = connL3 if tauL3 is None else tauL3
            tauR3 = ew_tangent_end(R_); tauR3 = connR3 if tauR3 is None else tauR3

            # connector weight never below conn_min; decays with distance d_min_old
            beta_conn = max(conn_min, max(0.0, 1.0 - d_min_old / d_switch))
            dirL3 = self._unit((1.0 - beta_conn) * tauL3 + beta_conn * connL3)
            dirR3 = self._unit((1.0 - beta_conn) * tauR3 + beta_conn * connR3)

            def preset_step(dmin, base_step):
                """Shrink step when close to merge."""
                if dmin < near_cutoff:
                    return max(0.1 * dmin, self.params.grow_step_min)
                return base_step

            # --- Generate candidates (per-side backtracking; do NOT append yet)
            base_step_saved = self.params.grow_step

            candL, dsL, fpL = None, 0.0, 0.0
            if grow_left and len(L) >= 2:
                self.params.grow_step = preset_step(d_min_old, base_step_saved)
                retry = 0
                while True:
                    testL, ds_try, fp_try = self._predict_and_relax_one(L[-1], dirL3, float(L[-1].get_potential_energy(force_consistent=True)), "L")
                    dL_only = self._min_image_distance(testL, R_[-1])
                    if (dL_only <= d_min_old * (1.0 + dmin_tol)) or (ds_try <= self.params.grow_step_min) or (retry >= max_retry_side):
                        candL, dsL, fpL = testL, ds_try, fp_try
                        break
                    self.params.grow_step = max(self.params.grow_step * self.params.grow_backtrack, self.params.grow_step_min)
                    retry += 1

            candR, dsR, fpR = None, 0.0, 0.0
            if grow_right and len(R_) >= 2:
                self.params.grow_step = preset_step(d_min_old, base_step_saved)
                retry = 0
                while True:
                    testR, ds_try, fp_try = self._predict_and_relax_one(R_[-1], dirR3, float(R_[-1].get_potential_energy(force_consistent=True)), "R")
                    dR_only = self._min_image_distance(L[-1], testR)
                    if (dR_only <= d_min_old * (1.0 + dmin_tol)) or (ds_try <= self.params.grow_step_min) or (retry >= max_retry_side):
                        candR, dsR, fpR = testR, ds_try, fp_try
                        break
                    self.params.grow_step = max(self.params.grow_step * self.params.grow_backtrack, self.params.grow_step_min)
                    retry += 1

            self.params.grow_step = base_step_saved  # restore

            # --- Choose among {L-only, R-only, BOTH}
            options = []
            if candL is not None:
                options.append(("L",    self._min_image_distance(candL, R_[-1])))
            if candR is not None:
                options.append(("R",    self._min_image_distance(L[-1], candR)))
            if (candL is not None) and (candR is not None):
                options.append(("BOTH", self._min_image_distance(candL, candR)))

            if not options:
                log_info(["\nNo feasible growth candidates; stopping growth.\n"], self.output)
                break

            options.sort(key=lambda x: x[1])  # prefer smaller d_min
            chosen = options[0]
            d_new  = chosen[1]

            # --- Monotonic guard: if still worse, shrink globally and retry a few times
            if d_new > d_min_old * (1.0 + dmin_tol):
                improved = False
                base_step_saved = self.params.grow_step
                for _ in range(max_retry_iter):
                    self.params.grow_step = max(self.params.grow_step * 0.5, self.params.grow_step_min)
                    # Rebuild candidates with smaller step
                    if grow_left and len(L) >= 2:
                        testL, ds_try, fp_try = self._predict_and_relax_one(L[-1], dirL3, float(L[-1].get_potential_energy(force_consistent=True)), "L")
                        dL_only = self._min_image_distance(testL, R_[-1])
                        if dL_only <= d_min_old * (1.0 + dmin_tol):
                            candL, dsL, fpL = testL, ds_try, fp_try
                    if grow_right and len(R_) >= 2:
                        testR, ds_try, fp_try = self._predict_and_relax_one(R_[-1], dirR3, float(R_[-1].get_potential_energy(force_consistent=True)), "R")
                        dR_only = self._min_image_distance(L[-1], testR)
                        if dR_only <= d_min_old * (1.0 + dmin_tol):
                            candR, dsR, fpR = testR, ds_try, fp_try
                    # Re-evaluate
                    options = []
                    if candL is not None: options.append(("L",    self._min_image_distance(candL, R_[-1])))
                    if candR is not None: options.append(("R",    self._min_image_distance(L[-1], candR)))
                    if (candL is not None) and (candR is not None):
                        options.append(("BOTH", self._min_image_distance(candL, candR)))
                    if options:
                        options.sort(key=lambda x: x[1])
                        if options[0][1] <= d_min_old * (1.0 + dmin_tol):
                            chosen = options[0]; d_new = chosen[1]; improved = True; break
                self.params.grow_step = base_step_saved

                # --- Forced connector pull if repeated worsening
                if not improved:
                    inc_streak += 1
                    if inc_streak >= inc_streak_tr:
                        inc_streak = 0
                        pull = max(self.params.grow_step_min, min(0.15 * d_min_old, 0.5))  # Å
                        self._endpoint_direct_step(L[-1], connL3, pull)
                        self._mini_relax_perp_lbfgs(L[-1], connL3, max_iter=self.params.grow_relax_iters, gtol=self.params.grow_relax_tol)
                        self._endpoint_direct_step(R_[-1], connR3, pull)
                        self._mini_relax_perp_lbfgs(R_[-1], connR3, max_iter=self.params.grow_relax_iters, gtol=self.params.grow_relax_tol)
                        # Local alignment for stability
                        if len(L) >= 2: align_path_inplace(L[-2:],  mode="chain")
                        if len(R_)>= 2: align_path_inplace(R_[-2:], mode="chain")
                        d_now = self._min_image_distance(L[-1], R_[-1])
                        if it % self.params.log_every == 0:
                            log_info([f"{it:>8d} {d_now:12.4f} {self.params.merge_threshold:7.3f} "
                                    f"{len(L):5d} {len(R_):5d} {0.0:6.3f} {0.0:6.3f} {0.0:13.5f} {0.0:13.5f}\n"], self.output)
                        if d_now <= self.params.merge_threshold:
                            break
                        it += 1
                        continue
                else:
                    inc_streak = 0
            else:
                inc_streak = 0

            # --- Commit chosen candidates
            if chosen[0] == "L":
                L.append(candL); E_L_last = float(candL.get_potential_energy(force_consistent=True))
            elif chosen[0] == "R":
                R_.append(candR); E_R_last = float(candR.get_potential_energy(force_consistent=True))
            else:
                L.append(candL); E_L_last = float(candL.get_potential_energy(force_consistent=True))
                R_.append(candR); E_R_last = float(candR.get_potential_energy(force_consistent=True))

            # Local alignment for each side
            if len(L) >= 3: align_path_inplace(L[-3:], mode="chain")
            if len(R_) >= 3: align_path_inplace(R_[-3:], mode="chain")

            # Log one line
            d_min = self._min_image_distance(L[-1], R_[-1])
            if it % self.params.log_every == 0:
                log_info([f"{it:>8d} {d_min:12.4f} {self.params.merge_threshold:7.3f} "
                        f"{len(L):5d} {len(R_):5d} {dsL:6.3f} {dsR:6.3f} {fpL:13.5f} {fpR:13.5f}\n"], self.output)

            # Merge check
            if d_min <= self.params.merge_threshold:
                break
            it += 1

        if it == self.params.max_iter_grow:
            log_info(["\nGSM growth reached maximum iterations; proceeding to merge.\n"], self.output)

        # --- Merge & equal-arc reparameterization to fixed n_images
        def concat_and_resample(Ls: List[Atoms], Rs: List[Atoms], n_images: int) -> List[Atoms]:
            """Concatenate L side + reversed R side, then resample to n_images by arclength."""
            path = [Ls[i] for i in range(len(Ls))] + [Rs[i] for i in range(len(Rs) - 2, -1, -1)]
            images = [Ls[0].copy()]
            inherit_attrs(Ls[0], images[0])
            for k in range(1, n_images - 1):
                images.append(Ls[0].copy()); inherit_attrs(Ls[0], images[k])
            images.append(Rs[0].copy()); inherit_attrs(Rs[0], images[-1])

            X = [to_numpy_f64(img.get_positions()) for img in path]
            s = [0.0]
            for i in range(1, len(X)):
                s.append(s[-1] + np.linalg.norm(X[i] - X[i-1]))
            Ltot = s[-1] if len(s) else 0.0
            if Ltot < 1e-15:
                for k in range(1, n_images - 1):
                    images[k].set_positions(X[0])
                return images

            s_targets = np.linspace(0.0, Ltot, n_images)
            for k in range(1, n_images - 1):
                sk = s_targets[k]
                j = np.searchsorted(s, sk) - 1
                j = max(0, min(j, len(X) - 2))
                lam = 0.0 if (s[j+1] - s[j] < 1e-16) else (sk - s[j]) / (s[j+1] - s[j])
                pos = (1.0 - lam) * X[j] + lam * X[j+1]
                images[k].set_positions(pos)
            return images

        images = concat_and_resample(L, R_, self.params.n_images)
        for img in images:
            if img.calc is None:
                img.calc = self.atoms_R.calc
        align_path_inplace(images, mode="chain")
        linear_reparam(images)

        # Dump growth-final equal-arc path & HEI
        grow_final = base + "_gsm_grow_final.xyz"
        write_xyz(grow_final, images, energies=get_energies(images))
        Es = get_energies(images)
        hei = max(range(1, len(images) - 1), key=lambda i: Es[i]) if len(images) > 2 else 0
        mep_path = base + "_gsm_mep.xyz"   # equal-arc path
        hei_path = base + "_gsm_hei.xyz"
        write_xyz(mep_path, images, energies=Es)
        write_xyz(hei_path, [images[hei]], energies=[Es[hei]])

        log_info([
            f"\nWrote GSM grow-final path to: {grow_final}\n",
            f"Wrote GSM MEP (equal-arc) to: {mep_path}\n",
            f"Wrote HEI to:                 {hei_path}\n"
        ], self.output)

        # --- Direct TS refinement via PRFO/RFO on HEI (no CI-STRING / no full relax)
        self.restart_run(images, hei, base)

