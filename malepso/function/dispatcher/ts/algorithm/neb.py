# -*- coding: utf-8 -*-
"""
NEB implementation with:
- Kabsch alignment (reports RMSD)
- IDPP interpolation for robust initial path
- Improved tangent NEB forces (true_perp + spring_parallel)
- L-BFGS optimizer on projected NEB forces
"""
from __future__ import annotations
from lib2to3.pgen2 import driver
import os
import sys
import math
import copy
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from ase import Atoms

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

def vec1d(x, n_expected=None):
    """Convert to float64 1D vector and optionally check length."""
    v = to_numpy_f64(x).reshape(-1)
    if n_expected is not None and v.size != n_expected:
        raise ValueError(f"Expected size {n_expected}, got {v.size}")
    return v

def kabsch_align(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Rigid-body least-squares alignment (Kabsch):
    Align Q onto P. Both are (N,3). Returns: Q_aligned, rmsd, R, t
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    if P.shape != Q.shape or P.shape[1] != 3:
        raise ValueError("P and Q must have shape (N,3)")

    Pc = P.mean(axis=0)
    Qc = Q.mean(axis=0)
    P0 = P - Pc
    Q0 = Q - Qc

    C = Q0.T @ P0
    V, S, Wt = np.linalg.svd(C)
    d = (np.linalg.det(V @ Wt) < 0.0)
    if d:
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

def write_all_images_xyz(filename: str, images: List[Atoms], energies: Optional[List[float]] = None, iteration: int = 0):
    """
    Append all current images to an xyz trajectory file (for NEB debug).
    Each block corresponds to one iteration of NEB optimization.
    """
    with open(filename, "a") as f:
        for i, at in enumerate(images):
            pos = to_numpy_f64(at.get_positions())
            symbols = at.get_chemical_symbols()
            E = None if energies is None else energies[i]
            f.write(f"{len(symbols)}\n")
            if E is not None:
                f.write(f"Iter {iteration} Image {i}  Energy = {E:.10f}\n")
            else:
                f.write(f"Iter {iteration} Image {i}\n")
            for s, (x, y, z) in zip(symbols, pos):
                f.write(f"{s:2s} {x: .10f} {y: .10f} {z: .10f}\n")



# --- add to your neb.py (or the module where NEB lives) ---

from dataclasses import asdict, fields

def _lower_keys(d):
    """Return a shallow copy with all string keys lowered (ignore non-str keys)."""
    if not isinstance(d, dict):
        return {}
    return { (k.lower() if isinstance(k, str) else k): v for k, v in d.items() }

def _select_subdict(paras: dict, name_aliases: tuple[str, ...]) -> dict:
    """
    Extract a sub-dict by aliases. E.g., name_aliases=("neb","NEB").
    Fallback to {} if not present.
    """
    if not isinstance(paras, dict):
        return {}
    # exact (case-insensitive) sub-dict
    low = _lower_keys(paras)
    for alias in name_aliases:
        key = alias.lower()
        if key in low and isinstance(low[key], dict):
            return low[key]
    # also support flat (top-level) style
    return low

def _update_dataclass_from_dict(dc_obj, d: dict, *, log_prefix: str, output:str,logger=None):
    """
    Update dataclass fields from dict keys that match exactly (case-insensitive).
    Unknown keys are ignored (optionally logged).
    """
    if not isinstance(d, dict):
        return dc_obj
    low = _lower_keys(d)
    fld_names = {f.name.lower(): f.name for f in fields(dc_obj)}
    for k_low, v in low.items():
        if k_low in fld_names:
            setattr(dc_obj, fld_names[k_low], v)
        else:
            if logger is not None:
                logger([f"[{log_prefix}] Ignore unknown key: {k_low}\n"], output)
    return dc_obj



# =============================================================================
# ------------------------------- IDPP ----------------------------------------
# =============================================================================

@dataclass
class IDPPParams:
    max_steps: int = 300
    step_size: float = 0.01       # Cartesian step size for IDPP relaxation
    r_cut: float = 12.0           # cutoff for pair list (Å)
    eps: float = 1e-8             # to avoid 1/0
    w_power: float = 2.0          # weight ~ (1 / d_target^w_power)^2 (classical choice: 2-4)

def _pair_indices(n_atoms: int) -> np.ndarray:
    """Return upper-triangular (i<j) index pairs as (M,2)."""
    idx = []
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            idx.append((i, j))
    return np.asarray(idx, dtype=np.int32)

def _idpp_targets(P0: np.ndarray, P1: np.ndarray, n_inner: int) -> List[np.ndarray]:
    """
    Build target 1/dist arrays for each internal image by linear interpolation
    between endpoints pair distances.
    Returns list of arrays (M,), M = Npair.
    """
    n_atoms = P0.shape[0]
    pairs = _pair_indices(n_atoms)
    Rij0 = np.linalg.norm(P0[pairs[:, 0]] - P0[pairs[:, 1]], axis=1)
    Rij1 = np.linalg.norm(P1[pairs[:, 0]] - P1[pairs[:, 1]], axis=1)
    inv0 = 1.0 / np.maximum(Rij0, 1e-8)
    inv1 = 1.0 / np.maximum(Rij1, 1e-8)

    targets = []
    for k in range(1, n_inner + 1):
        lam = k / (n_inner + 1)
        inv_tgt = (1.0 - lam) * inv0 + lam * inv1
        targets.append(inv_tgt)
    return targets

import numpy as np
from ase import Atoms

# =============================================================================
# ------------------------------- NEB Core ------------------------------------
# =============================================================================

@dataclass
class NEBParams:
    n_images: int = 9                     # total images including endpoints
    k_spring: float = 0.2                 # spring "stiffness" (same units as force * length^-1)
    max_iter: int = 200
    lbfgs_m: int = 5                      # memory size for L-BFGS
    step0: float = 0.2                    # initial step length on search direction
    step_min: float = 5e-4
    step_max: float = 1.0
    # ORCA-like convergence on projected forces
    f_max_th: float = 1.0e-3             # max(|Fp|) threshold
    f_rms_th: float = 5.0e-4             # RMS(Fp) threshold

def improved_tangent(Rm1, R, Rp1, Em1, E, Ep1):
    """
    Energy-weighted (improved) tangent vector as in standard NEB literature.
    Returns a normalized vector of shape (3N,).
    """
    dm = vec1d(R - Rm1)
    dp = vec1d(Rp1 - R)
    Em, Ep = Em1, Ep1

    # determine tangent per Henkelman-Jonsson scheme
    if Ep > E and E > Em:
        t = dp
    elif Ep < E and E < Em:
        t = dm
    else:
        dE_plus = max(Ep - E, 0.0)
        dE_minus = max(Em - E, 0.0)
        t = dE_plus * dp + dE_minus * dm

    norm = float(np.linalg.norm(t))
    if norm < 1e-16:
        # fall back to arithmetic average direction
        t = dp + dm
        norm = float(np.linalg.norm(t))
        if norm < 1e-16:
            # last resort
            t = np.zeros_like(dp)
            t[0] = 1.0
            return t
    return t / norm

def neb_forces(images: List[Atoms], energies: List[float], k_spring: float) -> Tuple[List[np.ndarray], float, int]:
    """
    Compute NEB projected forces for internal images:
    F_NEB = F_true_perp + F_spring_parallel  (per image)
    Returns: list of projected forces (cartesian shape (N,3)), max|Fp|, HEI_index
    """
    n_img = len(images)
    forces_proj = [None] * n_img
    max_fp = 0.0
    hei_idx = 1

    # get raw forces and flatten
    raw_forces = [to_numpy_f64(at.get_forces()) for at in images]
    coords = [to_numpy_f64(at.get_positions()) for at in images]
    Es = [float(e) for e in energies]

    # determine HEI
    inner_indices = list(range(1, n_img - 1))
    hei_idx = max(inner_indices, key=lambda i: Es[i])

    # loop over internal images
    for i in inner_indices:
        Rm1 = coords[i - 1]; R = coords[i]; Rp1 = coords[i + 1]
        Em1 = Es[i - 1];     E = Es[i];     Ep1 = Es[i + 1]

        # tangent
        tau = improved_tangent(Rm1, R, Rp1, Em1, E, Ep1)  # (3N,)
        tau_resh = tau.reshape(-1, 3)

        # true force (calculator gives forces = -grad V); NEB uses gradient convention:
        # F_true = -∇V, but our "forces" are already that.
        F_true = vec1d(raw_forces[i])

        # remove tangent component: F_true_perp = F_true - (F_true·tau) tau
        c = float(np.dot(F_true, tau))
        F_true_perp = F_true - c * tau
        F_true_perp = F_true_perp.reshape(-1, 3)

        # spring force along tangent: ks * (|R_{i+1}-R_i| - |R_i - R_{i-1}|) * tau
        d_next = float(np.linalg.norm(Rp1 - R))
        d_prev = float(np.linalg.norm(R - Rm1))
        F_spring_par = k_spring * (d_next - d_prev) * tau_resh

        # total projected
        Fp = F_true_perp + F_spring_par
        forces_proj[i] = Fp

        max_fp = max(max_fp, float(np.abs(Fp).max()))

    # endpoints fixed: zero projected forces
    forces_proj[0] = np.zeros_like(coords[0])
    forces_proj[-1] = np.zeros_like(coords[-1])

    return forces_proj, max_fp, hei_idx

def rms_force(forces_list: List[np.ndarray]) -> float:
    """RMS of projected forces over all DOFs of internal images."""
    arrs = []
    for i, F in enumerate(forces_list):
        if i == 0 or i == len(forces_list) - 1:
            continue
        arrs.append(F.reshape(-1))
    if not arrs:
        return 0.0
    v = np.concatenate(arrs)
    return float(np.sqrt(np.mean(v * v)))

# =============================================================================
# ------------------------------- L-BFGS --------------------------------------
# =============================================================================

class LBFGSDriver:
    """
    Smarter L-BFGS driver for NEB projected forces with dual-buffer Hessian and adaptive step.
    """
    def __init__(self, m: int, step0: float, step_min: float, step_max: float,
                    armijo_c1: float = 1e-4, nonmonotone_M: int = 5,
                    switch_period: int = 50, prebuild: int = 20):
            self.m = m
            self.step0 = step0
            self.step_min = step_min
            self.step_max = step_max
            self.armijo_c1 = armijo_c1
            self.nonmonotone_M = nonmonotone_M

            # active history
            self.S, self.Y, self.rhos = [], [], []

            # shadow history (to be built in advance)
            self.S2, self.Y2, self.rhos2 = [], [], []
            self.shadow_on = False

            # nonmonotone buffer for J
            self.J_hist = []

            # periodic schedule
            self.switch_period = int(switch_period)  # e.g. 50
            self.prebuild = int(prebuild)            # e.g. 20  -> start at 30, switch at 50

    def begin_shadow(self):
        """Start building a fresh shadow history from this iteration onward."""
        self.S2, self.Y2, self.rhos2 = [], [], []
        self.shadow_on = True

    def switch_to_shadow(self):
        """Switch active history to the freshly built shadow history."""
        if self.S2:  # only switch if shadow has some content
            self.S, self.Y, self.rhos = self.S2, self.Y2, self.rhos2
        # clear shadow and turn off
        self.S2, self.Y2, self.rhos2 = [], [], []
        self.shadow_on = False

    def schedule(self, iteration: int):
        """
        Periodic two-buffer policy:
          - Every 'switch_period' iterations (50, 100, 150, ...), switch to shadow.
          - 'prebuild' iterations before each switch (30, 80, 130, ...), begin shadow.
        Call this ONCE per outer iteration, BEFORE computing the search direction.
        """
        if iteration > 0 and (iteration % self.switch_period) == 0:
            # e.g. 50, 100, 150, ... -> switch to shadow
            self.switch_to_shadow()
        # e.g. 30, 80, 130, ... -> begin shadow
        if (iteration % self.switch_period) == (self.switch_period - self.prebuild):
            self.begin_shadow()
            
    def two_loop(self, g: np.ndarray, iteration: int) -> np.ndarray:
        
        S, Y, RHO = self.S, self.Y, self.rhos

        q = g.copy()
        alphas = []
        for s, y, rho in reversed(list(zip(S, Y, RHO))):
            alpha = rho * np.dot(s, q)
            alphas.append(alpha)
            q -= alpha * y

        if Y:
            gamma = np.dot(Y[-1], S[-1]) / (np.dot(Y[-1], Y[-1]) + 1e-20)
        else:
            gamma = 1.0
        z = gamma * q

        for (s, y, rho), alpha in zip(zip(S, Y, RHO), reversed(alphas)):
            beta = rho * np.dot(y, z)
            z += s * (alpha - beta)

        return -z


    def update(self, s: np.ndarray, y: np.ndarray):
            """Push new (s,y) into active; if building shadow, also push into shadow."""
            dot = np.dot(y, s)
            rho = 1.0 / (dot + 1e-20)

            # active
            self.S.append(s.copy()); self.Y.append(y.copy()); self.rhos.append(rho)
            if len(self.S) > self.m:
                self.S.pop(0); self.Y.pop(0); self.rhos.pop(0)

            # shadow
            if self.shadow_on:
                self.S2.append(s.copy()); self.Y2.append(y.copy()); self.rhos2.append(rho)
                if len(self.S2) > self.m:
                    self.S2.pop(0); self.Y2.pop(0); self.rhos2.pop(0)


    def line_search_backtrack(self, x, p, set_x, eval_grad, iteration):
        alpha = self.step0
        pmax = float(np.max(np.abs(p)))
        if pmax > 0:
            alpha = min(alpha, self.step_max / pmax)

        g0 = eval_grad(x)
        g0_norm = float(np.linalg.norm(g0))
        accept_ratio = 0.98 if g0_norm < 1e-2 else 0.9

        while alpha >= self.step_min:
            x_trial = x + alpha * p
            set_x(x_trial)
            g_trial = eval_grad(x_trial)
            if float(np.linalg.norm(g_trial)) <= accept_ratio * g0_norm:
                return x_trial, g_trial, alpha
            alpha *= 0.5

        x_trial = x + self.step_min * p
        set_x(x_trial)
        g_trial = eval_grad(x_trial)
        return x_trial, g_trial, self.step_min

    def should_stop(self, grad):
        max_force = np.max(np.abs(grad))
        rms_force = np.sqrt(np.mean(grad**2))
        return (max_force < 0.005) and (rms_force < 0.001)



# =============================================================================
# ------------------------------- NEB Class -----------------------------------
# =============================================================================

class NEB(JobABC):
    def __init__(self,
                 output: str,
                 atoms_R: Atoms,
                 atoms_P: Atoms,
                 params: Optional[NEBParams] = None,
                 idpp: Optional[IDPPParams] = None,
                 paras: Optional[dict] = None):   # <-- NEW
        super().__init__(output)
        
        self.atoms_R = atoms_R.copy()
        self.atoms_P = atoms_P.copy()
        self.atoms_R.calc = atoms_R.calc
        self.atoms_P.calc = atoms_P.calc

        # 1) start from built-in defaults
        self.params = params if params is not None else NEBParams()
        self.idpp_params = idpp if idpp is not None else IDPPParams()

        # 2) apply overrides from paras (if provided)
        if isinstance(paras, dict):
            # support "neb"/"NEB"/top-level keys for NEBParams
            neb_dict = _select_subdict(paras, ("neb", "NEB"))
            # if keys are flat at top-level (e.g., {"n_images": 11}), _select_subdict returns lowered top-level dict
            # so we only take the fields that exist in NEBParams
            _update_dataclass_from_dict(self.params, neb_dict, log_prefix="NEB", output=self.output, logger=log_info)

        # Safety: minimal guard
        if self.params.n_images < 2:
            raise ValueError("n_images must be >= 2 (including endpoints)")


    # -------------------------- helpers to map x <-> images -------------------

    def _pack_internal(self, images: List[Atoms]) -> np.ndarray:
        """Flatten internal images (1..N-2) Cartesian into a single vector."""
        arrs = []
        for i in range(1, len(images) - 1):
            arrs.append(to_numpy_f64(images[i].get_positions()).reshape(-1))
        return np.concatenate(arrs) if arrs else np.zeros(0, dtype=np.float64)

    def _unpack_internal(self, x: np.ndarray, images: List[Atoms]):
        """Write back flattened x into internal images (1..N-2)."""
        offset = 0
        for i in range(1, len(images) - 1):
            n = images[i].get_number_of_atoms() * 3
            Xi = x[offset:offset + n].reshape(-1, 3)
            images[i].set_positions(Xi)
            offset += n

    # ------------------------------- main flow --------------------------------

    def run(self):
        t0 = time.time()
        info = []

        # 0) Alignment (Kabsch): align product to reactant, report RMSD
        R = to_numpy_f64(self.atoms_R.get_positions())
        P = to_numpy_f64(self.atoms_P.get_positions())
        P_aligned, rmsd, Rmat, tvec = kabsch_align(R, P)
        self.atoms_P.set_positions(P_aligned)

        info += ["\n" + "-" * 62 + "\n",
                 " Reactant/Product rigid alignment (Kabsch) \n",
                 f" RMSD(after align): {rmsd:.6f} (in your coordinate units)\n",
                 "-" * 62 + "\n\n"]
        log_info(info, self.output); info = []

        # 1) Build initial images (linear) then IDPP relax
        n_img = self.params.n_images
        if n_img < 2:
            raise ValueError("n_images must be >= 2")

        images: List[Atoms] = []
        images.append(self.atoms_R.copy())
        for k in range(1, n_img - 1):
            lam = k / (n_img - 1)
            A = self.atoms_R.copy()
            A.set_positions((1.0 - lam) * R + lam * P_aligned)
            images.append(A)
        images.append(self.atoms_P.copy())

        # set calculators for images
        for img in images:
            if img.calc is None:
                img.calc = self.atoms_R.calc

        # 2) NEB loop with L-BFGS
        info += ["\n", "Starting NEB iterations:\n",
                 "Optim.  Iteration  HEI  E(HEI)-E(0)  max(|Fp|)   RMS(Fp)    dS\n",
                 f"Convergence thresholds         {self.params.f_max_th: .6f}   {self.params.f_rms_th: .6f} \n"]
        log_info(info, self.output); info = []

        # helper closures for LBFGS
        driver = LBFGSDriver(
            m=self.params.lbfgs_m,
            step0=self.params.step0,
            step_min=self.params.step_min,
            step_max=self.params.step_max,
            switch_period=50,   
            prebuild=20)

        def get_energies(imgs: List[Atoms]) -> List[float]:
            energies = [float(at.get_potential_energy(force_consistent=True)) for at in imgs]
            return energies

        def eval_grad(x_flat: np.ndarray) -> np.ndarray:
            # write x to images, compute projected forces, then return g = -flatten(Fp_internal)
            self._unpack_internal(x_flat, images)
            Es = get_energies(images)
            Fp_list, _, _ = neb_forces(images, Es, self.params.k_spring)
            # gather internal only
            grads = []
            for i in range(1, len(images) - 1):
                # projected "gradient" = -Fp
                grads.append((-Fp_list[i]).reshape(-1))
            if grads:
                return np.concatenate(grads)
            return np.zeros_like(x_flat)
        
        def eval_grad_and_J(x_flat: np.ndarray):
            self._unpack_internal(x_flat, images)
            Es = get_energies(images)
            Fp_list, _, _ = neb_forces(images, Es, self.params.k_spring)
            grads = []
            J = 0.0
            for i in range(1, len(images)-1):
                Fi = Fp_list[i].reshape(-1)
                J += 0.5 * float(np.dot(Fi, Fi))
                grads.append(-Fi)  # g = -Fp
            gcat = np.concatenate(grads) if grads else np.zeros_like(x_flat)
            return gcat, J


        def set_x(x_new: np.ndarray):
            self._unpack_internal(x_new, images)

        # initial pack
        x = self._pack_internal(images)
        Es = get_energies(images)
        Fp_list, maxfp, hei = neb_forces(images, Es, self.params.k_spring)
        g = eval_grad(x)   # -Fp_internal
        rmsfp = rms_force(Fp_list)

        iteration = 0
        converged = (maxfp <= self.params.f_max_th) and (rmsfp <= self.params.f_rms_th)
        
        # trajectory of all images 
        traj_file = os.path.splitext(self.output)[0] + "_image_traj.xyz"
        
        while (iteration < self.params.max_iter) and (not converged):
            driver.schedule(iteration)
            # L-BFGS direction
            p = driver.two_loop(g, iteration)
            # backtracking (heuristic)
            x_new, g_new, step_len = driver.line_search_backtrack(x, p, set_x, eval_grad, iteration)

            driver.update(x_new - x, g_new - g)

            # update for next iteration
            x = x_new
            g = g_new
            Es = get_energies(images)
            Fp_list, maxfp, hei = neb_forces(images, Es, self.params.k_spring)
            rmsfp = rms_force(Fp_list)

            # write all images to traj file
            write_all_images_xyz(traj_file, images, energies=Es, iteration=iteration)

            # print status
            dE_hei = Es[hei] - Es[0]
            info = [f"   LBFGS {iteration:>4d} {hei:>6d} {dE_hei:>10.6f} {maxfp:>11.6f} {rmsfp:>10.6f} {step_len:>7.4f}\n"]
            log_info(info, self.output)

            # check convergence
            converged = (maxfp <= self.params.f_max_th and rmsfp <= self.params.f_rms_th) \
                        or driver.should_stop(g)
            iteration += 1

        # final convergence banner
        info = [
            ".--------------------. \n",
            " ----------------------| NEB convergence |------------------------- \n",
            " Item           value           Tolerance        Converged \n",
            " --------------------------------------------------------------------- \n",
            f" RMS(Fp)   {rmsfp: .10f}   {self.params.f_rms_th: .10f}   {'YES' if rmsfp <= self.params.f_rms_th else ' NO'}\n",
            f" MAX(|Fp|) {maxfp: .10f}   {self.params.f_max_th: .10f}   {'YES' if maxfp <= self.params.f_max_th else ' NO'}\n",
            " --------------------------------------------------------------------- \n"
        ]
        if converged:
            info += ["The elastic band has converged successfully!\n",
                     "*********************H U R R A Y*********************\n",
                     "*** THE NEB OPTIMIZATION HAS CONVERGED ***\n",
                     "*****************************************************\n"]
        else:
            info += ["NEB did NOT converge within max_iter.\n"]
        log_info(info, self.output)

        # 3) PATH SUMMARY (ORCA-like)
        # Distances along the path
        coords = [to_numpy_f64(at.get_positions()) for at in images]
        dists = [0.0]
        for i in range(1, len(images)):
            d = float(np.linalg.norm(coords[i] - coords[i - 1]))
            dists.append(dists[-1] + d)

        # convert energy deltas to kcal/mol (only for display, not used anywhere else)
        # If your calculator already in Eh, factor = 627.509; if eV then factor=23.0605.
        # We infer by magnitude: assume Eh if |E|~1e2..1e3; otherwise eV. You can pin it if desired.
        E0 = Es[0]
        abE0 = abs(E0)
        if 10.0 <= abE0 <= 1.0e4:
            kcal_factor = 627.509474  # Eh -> kcal/mol
            unit_energy = "Eh"
        else:
            kcal_factor = 23.060548    # eV -> kcal/mol
            unit_energy = "eV"

        info = []
        info += ["\n---------------------------------------------------------------\n",
                 " PATH SUMMARY \n",
                 "---------------------------------------------------------------\n",
                 "All forces in the same units as your calculator (Fp is projected).\n",
                 f"Image   Dist(Ang.)     E({unit_energy})     dE(kcal/mol)   max(|Fp|)    RMS(Fp)\n"]
        for i in range(len(images)):
            fp_max_i = float(np.abs(Fp_list[i]).max()) if Fp_list[i] is not None else 0.0
            fp_rms_i = float(np.sqrt(np.mean((Fp_list[i].reshape(-1))**2))) if Fp_list[i] is not None else 0.0
            info.append(f"{i:2d} {dists[i]:>10.3f} {Es[i]:>14.5f} {((Es[i]-E0)*kcal_factor):>12.2f} {fp_max_i:>12.6f} {fp_rms_i:>10.6f}")
            if i == hei:
                info.append("   <= HEI")
            info.append("\n")
        # straight-line distances
        info.append("Straight line distance between images along the path:\n")
        for i in range(len(images) - 1):
            seg = float(np.linalg.norm(coords[i + 1] - coords[i]))
            info.append(f"D({i:2d}-{i+1:2d}) = {seg:.4f} Ang.\n")
        log_info(info, self.output)

        # 4) Print HEI block (geometry, forces, unit tangent), ORCA-like
        # unit tangent at HEI
        i = hei
        tau = improved_tangent(coords[i - 1], coords[i], coords[i + 1], Es[i - 1], Es[i], Es[i + 1]).reshape(-1, 3)
        rawF = to_numpy_f64(images[i].get_forces())
        info = []
        info += ["\n---------------------------------------------------------------\n",
                 " INFORMATION ABOUT HIGHEST ENERGY IMAGE \n",
                 "---------------------------------------------------------------\n",
                 f"Highest energy image .... {hei}\n",
                 f"Energy .... {Es[hei]: .10f} {unit_energy}\n",
                 f"Max. abs. projected force .... {float(np.abs(Fp_list[hei]).max()): .6e} (same force units)\n",
                 "----------------------------------------- HIGHEST ENERGY IMAGE (ANGSTROEM) -----------------------------------------\n"]
        symb = images[i].get_chemical_symbols()
        pos = images[i].get_positions()
        for s, (x, y, z) in zip(symb, pos):
            info.append(f"{s:2s} {x: .6f} {y: .6f} {z: .6f}\n")
        info.append("----------------------------------------- FORCES (calculator units) -----------------------------------------\n")
        for s, (fx, fy, fz) in zip(symb, rawF):
            info.append(f"{s:2s} {fx: .6e} {fy: .6e} {fz: .6e}\n")
        info.append("----------------------------------------- UNIT TANGENT -----------------------------------------\n")
        for s, (tx, ty, tz) in zip(symb, tau):
            info.append(f"{s:2s} {tx: .6e} {ty: .6e} {tz: .6e}\n")
        info.append("=> Unit tangent is an approximation to the TS mode at the saddle point\n")
        log_info(info, self.output)

        # 5) Timings
        t1 = time.time()
        info = []
        info += ["\n---------- TIMINGS ----------\n",
                 f"Total ... {t1 - t0: .3f} sec\n",
                 f"NEB   ... {t1 - t0: .3f} sec 100.0%\n"]
        log_info(info, self.output)

        # 6) Write files
        base, ext = os.path.splitext(self.output)
        mep_path = base + "_mep.xyz"
        hip_path = base + "_hip.xyz"   # per your request: "hip"
        write_xyz(mep_path, images, energies=Es)
        # HEI single-frame
        write_xyz(hip_path, [images[hei]], energies=[Es[hei]])

        # final one-liner so that upstream runners can detect files
        log_info([f"\nWrote MEP to: {mep_path}\nWrote HEI to: {hip_path}\n"], self.output)
