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
    f_max_th: float = 1.0e-2             # max(|Fp|) threshold
    f_rms_th: float = 5.0e-3             # RMS(Fp) threshold
    initial_opt: bool = False               # do initial relaxation of endpoints

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
    """Simple L-BFGS optimizer for NEB projected forces (no DIIS / no line search)."""
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
            step *= self.maxstep / max_disp
        return step

    def should_stop(self, grad, fmax_th=1e-3, frms_th=5e-4) -> bool:
        max_f = np.max(np.abs(grad))
        rms_f = np.sqrt(np.mean(grad ** 2))
        return (max_f < fmax_th) and (rms_f < frms_th)


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


    def optimize_endpoints(self, atoms_R: Atoms, atoms_P: Atoms, 
                        f_max_th=1.0e-3, f_rms_th=5.0e-4, max_iter=200):
        """
        Optimize the reactant and product endpoints before NEB if initial_opt=True.
        After optimization, write the minimized structures to XYZ files.
        """
        def single_point_optimize(atoms: Atoms):
            driver = LBFGSDriver(m=self.params.lbfgs_m, curvature=70.0, maxstep=self.params.step0)
            iteration = 0
            g = -to_numpy_f64(atoms.get_forces()).reshape(-1)
            x = to_numpy_f64(atoms.get_positions()).reshape(-1)

            while iteration < max_iter and not driver.should_stop(g, f_max_th, f_rms_th):
                p = driver.two_loop(g)
                p = driver.step_limit(p)
                x_new = x + p

                # update positions and forces
                atoms.set_positions(x_new.reshape(-1, 3))
                g_new = -to_numpy_f64(atoms.get_forces()).reshape(-1)
                driver.update(x_new - x, g_new - g)

                x = x_new
                g = g_new
                iteration += 1

            return atoms

        log_info(["\nInitial endpoint optimization started...\n"], self.output)
        atoms_R = single_point_optimize(atoms_R)
        atoms_P = single_point_optimize(atoms_P)
        log_info(["Initial endpoint optimization completed.\n"], self.output)

        # --- write optimized endpoints to XYZ ---
        base, ext = os.path.splitext(self.output)
        reactant_path = base + "_reactant_min.xyz"
        product_path  = base + "_product_min.xyz"
        write_xyz(reactant_path, [atoms_R], energies=[atoms_R.get_potential_energy(force_consistent=True)])
        write_xyz(product_path,  [atoms_P], energies=[atoms_P.get_potential_energy(force_consistent=True)])

        log_info([f"Optimized reactant written to: {reactant_path}\n"], self.output)
        log_info([f"Optimized product written to:  {product_path}\n"], self.output)

        return atoms_R, atoms_P



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

        # 0) Optional: Optimize endpoints before NEB
        if self.params.initial_opt:
            self.atoms_R, self.atoms_P = self.optimize_endpoints(
                self.atoms_R, self.atoms_P,
                f_max_th=self.params.f_max_th,
                f_rms_th=self.params.f_rms_th,
                max_iter=self.params.max_iter
            )

        # 1) Alignment (Kabsch)
        R = to_numpy_f64(self.atoms_R.get_positions())
        P = to_numpy_f64(self.atoms_P.get_positions())
        P_aligned, rmsd, _, _ = kabsch_align(R, P)
        self.atoms_P.set_positions(P_aligned)
        log_info([f"Alignment done. RMSD: {rmsd:.6f}\n"], self.output, ' (Angstrom)')

        # 1) Build linear path
        n_img = self.params.n_images
        images = [self.atoms_R.copy()]
        for k in range(1, n_img - 1):
            lam = k / (n_img - 1)
            A = self.atoms_R.copy()
            A.set_positions((1.0 - lam) * R + lam * P_aligned)
            images.append(A)
        images.append(self.atoms_P.copy())

        for img in images:
            if img.calc is None:
                img.calc = self.atoms_R.calc

        # prepare trajectory file
        traj_file = os.path.splitext(self.output)[0] + "_image_traj.xyz"

        driver = LBFGSDriver(
            m=self.params.lbfgs_m,
            curvature=70.0,
            maxstep=self.params.step0
        )

        def get_energies(imgs): return [float(at.get_potential_energy(force_consistent=True)) for at in imgs]

        def eval_grad(x_flat):
            self._unpack_internal(x_flat, images)
            Es = get_energies(images)
            Fp_list, _, _ = neb_forces(images, Es, self.params.k_spring)
            grads = [(-Fp_list[i]).reshape(-1) for i in range(1, len(images) - 1)]
            return np.concatenate(grads) if grads else np.zeros_like(x_flat)

        x = self._pack_internal(images)
        g = eval_grad(x)
        iteration = 0

        log_info([
            "\nStarting NEB iterations:\n",
            "Optim.  Iteration  HEI  E(HEI)-E(0)  max(|Fp|)   RMS(Fp)\n",
            f"Convergence thresholds         {self.params.f_max_th: .6f}   {self.params.f_rms_th: .6f}\n"
        ], self.output)

        while iteration < self.params.max_iter and not driver.should_stop(g, self.params.f_max_th, self.params.f_rms_th):
            p = driver.two_loop(g)
            p = driver.step_limit(p)
            x_new = x + p
            g_new = eval_grad(x_new)

            driver.update(x_new - x, g_new - g)

            x = x_new
            g = g_new

            Es = get_energies(images)
            Fp_list, maxfp, hei = neb_forces(images, Es, self.params.k_spring)
            rmsfp = rms_force(Fp_list)
            dE_hei = Es[hei] - Es[0]

            write_all_images_xyz(traj_file, images, energies=Es, iteration=iteration)

            log_info([f"   LBFGS {iteration:>4d} {hei:>6d} {dE_hei:>10.6f} {maxfp:>11.6f} {rmsfp:>10.6f}\n"], self.output)

            iteration += 1

        base, _ = os.path.splitext(self.output)
        mep_path = base + "_mep.xyz"
        hip_path = base + "_hip.xyz"
        write_xyz(mep_path, images, energies=Es)
        write_xyz(hip_path, [images[hei]], energies=[Es[hei]])
        log_info([f"\nWrote MEP to: {mep_path}\nWrote HEI to: {hip_path}\n"], self.output)

