# -*- coding: utf-8 -*-
"""
NEB implementation with:
- Kabsch alignment (reports RMSD)
- IDPP interpolation for robust initial path
- Improved tangent NEB forces (true_perp + spring_parallel)
- L-BFGS optimizer on projected NEB forces
- ORCA-style dynamic spring constants
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

from maple.function.utility import Molecules

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
    if iteration == 0 and os.path.exists(filename):
        os.remove(filename)
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
    n_images: int = 10                     # total images excluding endpoints
    k_min: float = 0.03                    # minimum spring constant
    k_max: float = 0.3                     # maximum spring constant
    use_dynamic_k: bool = True             # enable ORCA-style dynamic spring constants
    k_decay: float = 0.5                   # decay factor for dynamic k (ORCA default: 0.5)
    max_iter: int = 500 #256
    lbfgs_m: int = 20                      # memory size for L-BFGS
    step0: float = 2e-2                    # initial step length on search direction
    # IDPP control
    ifidpp: int = 1                     # 1: use IDPP for initial path, 0: linear interp.
    # Convergence on projected forces
    neb_f_max_th: float = 9.5e-3           # max(|Fp|) threshold
    neb_f_rms_th: float = 5e-3             # RMS(Fp) threshold
    initial_opt: bool = False              # do initial relaxation of endpoints
    refine: Optional[str] = None           # 'cineb' or 'nebts' or None
    # CINEB-specific
    cineb_f_max_th: float = 1e-2 #3e-03          # max(|Fp|) threshold for CINEB
    cineb_f_rms_th: float = 1e-2 #2e-03          # RMS(Fp) threshold for CINEB
    cilbfgs_m: int = 20                    # memory size for L-BFGS in CINEB
    cistep0: float = 5e-3                  # initial step length for CINEB


def compute_dynamic_k(energies: List[float], k_min: float, k_max: float, k_decay: float = 0.5) -> List[float]:
    """
    Compute ORCA-style dynamic spring constants for each internal image.
    
    The spring constant for image i is computed as:
    k_i = k_max - (k_max - k_min) * exp(-decay * ΔE_i / ΔE_max)
    
    where:
    - ΔE_i = max(E_i - E_ref, 0) with E_ref = max(E_{i-1}, E_{i+1})
    - ΔE_max = max energy barrier along the path
    
    This ensures:
    - Images near the barrier get weaker springs (closer to k_min)
    - Images far from the barrier get stronger springs (closer to k_max)
    
    Parameters
    ----------
    energies : List[float]
        Energies of all images (including endpoints)
    k_min : float
        Minimum spring constant (for barrier region)
    k_max : float
        Maximum spring constant (for flat regions)
    k_decay : float
        Decay factor controlling transition sharpness (ORCA default: 0.5)
    
    Returns
    -------
    List[float]
        Spring constants for each image (length = len(energies))
        Endpoints get k_max by convention (not used in force calculation)
    """
    n_img = len(energies)
    k_springs = [k_max] * n_img  # initialize with k_max
    
    if n_img < 3:
        return k_springs
    
    # Find global energy reference (typically reactant energy)
    E_ref_global = min(energies[0], energies[-1])
    
    # Compute energy barriers for each internal image
    delta_E = []
    for i in range(1, n_img - 1):
        # Local reference: max of neighboring energies
        E_ref_local = max(energies[i-1], energies[i+1])
        # Energy rise above local reference
        dE = max(energies[i] - E_ref_local, 0.0)
        delta_E.append(dE)
    
    # Find maximum barrier
    delta_E_max = max(delta_E) if delta_E else 1.0
    
    # Avoid division by zero
    if delta_E_max < 1e-10:
        # Essentially flat path, use k_max everywhere
        return k_springs
    
    # Compute dynamic k for each internal image
    for idx, dE in enumerate(delta_E):
        i = idx + 1  # actual image index
        # Normalized energy barrier
        dE_norm = dE / delta_E_max
        # ORCA formula: k decreases exponentially as we approach the barrier
        k_i = k_max - (k_max - k_min) * np.exp(-k_decay * dE_norm)
        k_springs[i] = k_i
    
    return k_springs


def improved_tangent(Rm1, R, Rp1, Em1, E, Ep1):
    """
    Energy-weighted (Henkelman-Jonsson) improved tangent.
    Always returns a normalized (3N,) vector.
    """
    dm = vec1d(R - Rm1)
    dp = vec1d(Rp1 - R)
    Em, Ep = Em1, Ep1

    # decide direction
    if Ep > E and E > Em:
        t = dp
    elif Ep < E and E < Em:
        t = dm
    else:
        dE_plus  = max(Ep - E, 0.0)
        dE_minus = max(Em - E, 0.0)
        t = dE_plus * dp + dE_minus * dm

    # normalize
    norm = np.linalg.norm(t)

    if norm < 1e-16:
        # fallback 1
        t = dp + dm
        norm = np.linalg.norm(t)
        if norm < 1e-16:
            # fallback 2: return any unit vector
            unit = np.zeros_like(t)
            unit[0] = 1.0
            return unit
    
    return t / norm

def cineb_tangent(Rm1, R, Rp1, Em1, E, Ep1):
    """
    Upwinding tangent for CINEB (Henkelman-Jonsson style).
    Returns a normalized (3N,) vector.

    This function is separate from `improved_tangent` so that:
    - NEB keeps using the original `improved_tangent`.
    - CINEB can use a robust upwinding scheme without touching NEB behavior.
    """
    dm = vec1d(R - Rm1)
    dp = vec1d(Rp1 - R)

    prev_energy = Em1
    ith_energy  = E
    next_energy = Ep1

    # Energy differences relative to image i
    dE_plus  = abs(next_energy - ith_energy)
    dE_minus = abs(prev_energy - ith_energy)

    # Strict uphill: E_{i-1} < E_i < E_{i+1} -> use forward tangent
    if (next_energy > ith_energy) and (ith_energy > prev_energy):
        t = dp

    # Strict downhill: E_{i-1} > E_i > E_{i+1} -> use backward tangent
    elif (next_energy < ith_energy) and (ith_energy < prev_energy):
        t = dm

    else:
        # Around maxima or minima: use energy-weighted combination
        delta_max = max(dE_plus, dE_minus)
        delta_min = min(dE_plus, dE_minus)

        if next_energy >= prev_energy:
            # Next image has higher energy
            t = delta_max * dp + delta_min * dm
        else:
            # Previous image has higher energy
            t = delta_min * dp + delta_max * dm

    norm = np.linalg.norm(t)
    if norm < 1e-16:
        # Degenerate case, fall back to simple sum
        t = dp + dm
        norm = np.linalg.norm(t)
        if norm < 1e-16:
            # Extreme fallback: choose any unit vector
            unit = np.zeros_like(t)
            unit[0] = 1.0
            return unit

    return t / norm


def neb_forces(images: List[Atoms], energies: List[float], 
               k_spring: float = None, k_springs: List[float] = None,
               use_dynamic_k: bool = False, k_min: float = 0.03, 
               k_max: float = 0.3, k_decay: float = 0.5) -> Tuple[List[np.ndarray], float, int]:
    """
    Compute NEB projected forces for internal images:
    F_NEB = F_true_perp + F_spring_parallel  (per image)
    
    Parameters
    ----------
    images : List[Atoms]
        All images including endpoints
    energies : List[float]
        Energies of all images
    k_spring : float, optional
        Single spring constant (used if k_springs is None and use_dynamic_k=False)
    k_springs : List[float], optional
        Pre-computed spring constants for each image
    use_dynamic_k : bool
        If True and k_springs is None, compute dynamic spring constants
    k_min, k_max, k_decay : float
        Parameters for dynamic spring constant calculation
    
    Returns
    -------
    forces_proj : List[np.ndarray]
        Projected forces (cartesian shape (N,3)) for each image
    max_fp : float
        Maximum force component among all internal images
    hei_idx : int
        Index of highest energy image
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

    # Determine spring constants
    if k_springs is None:
        if use_dynamic_k:
            k_springs = compute_dynamic_k(Es, k_min, k_max, k_decay)
        else:
            # Use single spring constant for all images
            k_springs = [k_spring if k_spring is not None else k_max] * n_img

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

        # spring force along tangent with dynamic k
        k_i = k_springs[i]
        d_next = float(np.linalg.norm(Rp1 - R))
        d_prev = float(np.linalg.norm(R - Rm1))
        F_spring_par = k_i * (d_next - d_prev) * tau_resh

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

    def ci_should_stop(self, regular_Fp, ci_F):
        """
        Dual convergence check for CINEB, ORCA-style.

        Parameters
        ----------
        regular_Fp : np.ndarray
            1D array of projected NEB forces (F_perp + spring_parallel) for
            all *non-climbing* internal images, flattened.
        ci_F : np.ndarray
            1D array of *true* forces for the climbing image (no projection).

        Uses thresholds stored on the driver:
        - self.fmax_reg, self.frms_reg : regular images (Fp)
        - self.fmax_ci,  self.frms_ci  : climbing image (F)
        """
        if regular_Fp.size == 0:
            maxFp = 0.0
            rmsFp = 0.0
        else:
            maxFp = float(np.max(np.abs(regular_Fp)))
            rmsFp = float(np.sqrt(np.mean(regular_Fp ** 2)))

        if ci_F.size == 0:
            maxF = 0.0
            rmsF = 0.0
        else:
            maxF = float(np.max(np.abs(ci_F)))
            rmsF = float(np.sqrt(np.mean(ci_F ** 2)))

        return (
            (maxFp < self.fmax_reg) and
            (rmsFp < self.frms_reg) and
            (maxF  < self.fmax_ci)  and
            (rmsF  < self.frms_ci)
        )


# =============================================================================
# ------------------------------- NEB Class -----------------------------------
# =============================================================================

class NEB(JobABC):
    def __init__(self,
                output: str,
                atoms_or_molecules,
                paras: Optional[dict] = None):
        super().__init__(output)

        # Handle Molecules input
        if isinstance(atoms_or_molecules, Molecules):
            self.input_images = atoms_or_molecules.multiatoms
            self.atoms_R = None
            self.atoms_P = None
        else:
            raise ValueError("Please provide Molecules object containing all images")

        # Initialize params from paras dict
        self.params = self._init_params(NEBParams, paras, ("neb", "NEB", "ts"))

        # Safety: minimal guard
        if self.params.n_images < 1:
            raise ValueError("n_images must be >= 1")

        self.params.n_images = int(self.params.n_images) + 2  # total images including endpoints


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

    def atoms_to_xyz(self, atoms: Atoms) -> str:
        lines = []
        syms = atoms.get_chemical_symbols()
        pos = atoms.get_positions()
        for s, (x, y, z) in zip(syms, pos):
            lines.append(f"{s:<2} {x:14.6f} {y:14.6f} {z:14.6f}")
        return "\n".join(lines) + "\n"

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
            n = len(images[i]) * 3
            Xi = x[offset:offset + n].reshape(-1, 3)
            images[i].set_positions(Xi)
            offset += n
            
    def _align_path(self, images: List[Atoms], ref_mode: str = "reactant"):
        """
        Rigid alignment for all internal images each iteration.
        ref_mode = "reactant"  → use images[0] as reference
                 = "centroid"  → use centroid of all images as reference
        Returns a list of RMSDs for internal images.
        """
        # choose reference coordinates
        if ref_mode == "reactant":
            ref = to_numpy_f64(images[0].get_positions())
        elif ref_mode == "centroid":
            coords = [to_numpy_f64(img.get_positions()) for img in images]
            ref = np.mean(coords, axis=0)
        else:
            ref = to_numpy_f64(images[0].get_positions())

        rmsds = []
        # align internal images only (1..N-2)
        for i in range(1, len(images) - 1):
            Q = to_numpy_f64(images[i].get_positions())
            Q_aligned, rmsd, _, _ = kabsch_align(ref, Q)
            images[i].set_positions(Q_aligned)
            rmsds.append(rmsd)
        return rmsds


    def get_energies(self, imgs): 
        return [float(at.get_potential_energy(force_consistent=True)) for at in imgs]
    
    def _compute_distances(self, images: List[Atoms]) -> List[float]:
        """
        Compute straight-line distances between consecutive images.
        Returns list of distances with length = len(images) - 1
        """
        distances = []
        for i in range(len(images) - 1):
            pos1 = to_numpy_f64(images[i].get_positions())
            pos2 = to_numpy_f64(images[i+1].get_positions())
            dist = np.linalg.norm(pos2 - pos1)
            distances.append(dist)
        return distances

    def _determine_insertion_plan(self, distances: List[float], n_to_insert: int) -> List[Tuple[int, int]]:
        """
        Determine where to insert new images based on distances.
        
        Parameters
        ----------
        distances : List[float]
            Distances between consecutive images
        n_to_insert : int
            Total number of images to insert
        
        Returns
        -------
        List[Tuple[int, int]]
            List of (segment_index, count) indicating how many images to insert after segment_index
            
        Example: if distances = [4.0, 2.0, 1.0] and n_to_insert = 3
                largest gaps are at index 0 (4.0) and 1 (2.0)
                return [(0, 2), (1, 1)] means insert 2 after segment 0, 1 after segment 1
        """
        # Create list of (index, distance) and sort by distance (descending)
        indexed_distances = [(i, d) for i, d in enumerate(distances)]
        indexed_distances.sort(key=lambda x: x[1], reverse=True)
        
        # Distribute insertions proportionally to gap size
        insertion_counts = [0] * len(distances)
        
        for k in range(n_to_insert):
            # Insert in the largest remaining gap
            # Find segment with largest distance/insertions ratio
            max_ratio = -1
            max_idx = 0
            for i, d in enumerate(distances):
                ratio = d / (insertion_counts[i] + 1)
                if ratio > max_ratio:
                    max_ratio = ratio
                    max_idx = i
            insertion_counts[max_idx] += 1
        
        # Convert to list of (index, count) tuples
        plan = [(i, count) for i, count in enumerate(insertion_counts) if count > 0]
        return plan

    def _insert_images_by_plan(self, images: List[Atoms], plan: List[Tuple[int, int]]) -> List[Atoms]:
        """
        Insert interpolated images according to the insertion plan.
        
        Parameters
        ----------
        images : List[Atoms]
            Current list of images
        plan : List[Tuple[int, int]]
            Insertion plan from _determine_insertion_plan
        
        Returns
        -------
        List[Atoms]
            New list with interpolated images inserted
        """
        # Sort plan by index (descending) to insert from back to front
        plan_sorted = sorted(plan, key=lambda x: x[0], reverse=True)
        
        new_images = images.copy()
        
        for seg_idx, count in plan_sorted:
            # Insert 'count' images between new_images[seg_idx] and new_images[seg_idx+1]
            pos1 = to_numpy_f64(new_images[seg_idx].get_positions())
            pos2 = to_numpy_f64(new_images[seg_idx + 1].get_positions())
            
            inserted = []
            for k in range(1, count + 1):
                # Linear interpolation
                lam = k / (count + 1)
                new_pos = (1.0 - lam) * pos1 + lam * pos2
                
                # Copy atoms object and set new positions
                new_atom = new_images[seg_idx].copy()
                new_atom.set_positions(new_pos)
                new_atom.calc = new_images[seg_idx].calc  # Inherit calculator
                inserted.append(new_atom)
            
            # Insert all new images after seg_idx
            for idx, img in enumerate(inserted):
                new_images.insert(seg_idx + 1 + idx, img)
        
        return new_images
    
    # ---------------- Climbing Image NEB (CINEB) -------------------------------
    def cineb_forces(self, images: List[Atoms], energies: List[float], k_spring: float) -> Tuple[List[np.ndarray], float, int]:
            """
            Climbing Image NEB projected forces:
            - normal NEB force for all non-endpoints except HEI
            - for HEI: remove spring force and reverse parallel component of true force

            The climbing image index can be "frozen" via self._cineb_fixed_hei:
            - if set and valid, use it
            - otherwise use the current highest-energy internal image
            """
            n_img = len(images)
            forces_proj = [None] * n_img
            max_fp = 0.0

            raw_forces = [to_numpy_f64(at.get_forces()) for at in images]
            coords = [to_numpy_f64(at.get_positions()) for at in images]
            Es = [float(e) for e in energies]

            inner_indices = list(range(1, n_img - 1))

            # --- choose HEI index: fixed one if present, else dynamic ---
            fixed_hei = getattr(self, "_cineb_fixed_hei", None)
            if (fixed_hei is not None) and (fixed_hei in inner_indices):
                hei_idx = fixed_hei
            else:
                hei_idx = max(inner_indices, key=lambda i: Es[i])

            for i in inner_indices:
                Rm1, R, Rp1 = coords[i - 1], coords[i], coords[i + 1]
                Em1, E, Ep1 = Es[i - 1], Es[i], Es[i + 1]

                tau = improved_tangent(Rm1, R, Rp1, Em1, E, Ep1)
                tau_resh = tau.reshape(-1, 3)
                F_true = vec1d(raw_forces[i])

                if i == hei_idx:
                    # ----------------------------------------------------------
                    # Climbing image force (ONLY true force, no projection)
                    # F_CI = F_true - 2 (F_true·tau) tau
                    # ----------------------------------------------------------
                    f_true = F_true                      # 1D (3N,)
                    tau_ci = tau / np.linalg.norm(tau)   # ensure normalized

                    ft = np.dot(f_true, tau_ci)
                    f_ci = f_true - 2.0 * ft * tau_ci

                    forces_proj[i] = f_ci.reshape(-1, 3)
                    continue

                else:
                    # Normal NEB projected force
                    F_true_perp = F_true - np.dot(F_true, tau) * tau
                    d_next = float(np.linalg.norm(Rp1 - R))
                    d_prev = float(np.linalg.norm(R - Rm1))
                    F_spring_par = k_spring * (d_next - d_prev) * tau_resh
                    Fp = F_true_perp.reshape(-1, 3) + F_spring_par

                forces_proj[i] = Fp
                max_fp = max(max_fp, float(np.abs(Fp).max()))

            # endpoints fixed
            forces_proj[0] = np.zeros_like(coords[0])
            forces_proj[-1] = np.zeros_like(coords[-1])

            return forces_proj, max_fp, hei_idx


    def restart_run(self, images: List[Atoms], energies: Optional[List[float]] = None):
        """
        Continue from a converged NEB path and perform:
        - CINEB refinement (always)
        - Optional TS refinement with RFO (if params.refine == 'nebts')

        Logic:
        - regular images: use projected NEB forces Fp for convergence
        - climbing image: use *true* forces F for convergence
        - climbing image index is frozen for the whole CINEB run
        """
        if energies is None:
            energies = [float(at.get_potential_energy(force_consistent=True)) for at in images]

        # reset any previous fixed HEI
        self._cineb_fixed_hei = None

        traj_file = os.path.splitext(self.output)[0] + "_cineb_traj.xyz"
        driver = LBFGSDriver(m=self.params.lbfgs_m, curvature=70.0, maxstep=self.params.cistep0)
        # set convergence thresholds
        driver.fmax_reg = self.params.neb_f_max_th
        driver.frms_reg = self.params.neb_f_rms_th
        driver.fmax_ci  = self.params.cineb_f_max_th
        driver.frms_ci  = self.params.cineb_f_rms_th

        def eval_grad(x_flat: np.ndarray) -> np.ndarray:
            """Return gradient dE/dx (flattened) for all internal images using CINEB forces."""
            self._unpack_internal(x_flat, images)
            Es_local = [float(at.get_potential_energy(force_consistent=True)) for at in images]
            Fp_list_local, _, _ = self.cineb_forces(images, Es_local, self.params.k_max)
            grads = [(-Fp_list_local[i]).reshape(-1) for i in range(1, len(images) - 1)]
            return np.concatenate(grads) if grads else np.zeros_like(x_flat)

        # initial packing / gradient
        x = self._pack_internal(images)
        g = eval_grad(x)
        iteration = 0

        # initial energies / forces for logging & to freeze HEI
        Es = [float(at.get_potential_energy(force_consistent=True)) for at in images]
        Fp_list, maxfp, hei = self.cineb_forces(images, Es, self.params.k_max)

        # freeze HEI index for the whole CINEB run
        self._cineb_fixed_hei = hei


        F_CI_vec = to_numpy_f64(images[hei].get_forces())
        maxF_CI = float(np.max(np.linalg.norm(F_CI_vec, axis=1)))
        rmsF_CI = float(np.sqrt(np.mean(np.linalg.norm(F_CI_vec, axis=1) ** 2)))

        log_info([
            "\nStarting CINEB refinement:\n",
            "Optim.  Iteration  CI   E(CI)-E(0)   max(|Fp|)   RMS(Fp)   max(|FCI|)   RMS(|FCI|)\n",
            f"Convergence thresholds regular: {self.params.neb_f_max_th: .6f}/{self.params.neb_f_rms_th: .6f},  "
            f"CI: {self.params.cineb_f_max_th: .6f}/{self.params.cineb_f_rms_th: .6f}\n"
        ], self.output)

        while iteration < self.params.max_iter:
            # ===== convergence check at current geometry =====
            # regular Fp: all internal non-CI images
            inner_idx = list(range(1, len(images) - 1))
            regular_flat = []
            for i in inner_idx:
                if i == hei:
                    continue
                regular_flat.append(Fp_list[i].reshape(-1))
            Fp_all = np.concatenate(regular_flat) if regular_flat else np.zeros(0, dtype=np.float64)

            # true forces on CI (no projection)
            F_CI = to_numpy_f64(images[hei].get_forces()).reshape(-1)

            if driver.ci_should_stop(Fp_all, F_CI):
                log_info([f"\nCINEB converged after {iteration} iterations.\n"], self.output)
                break

            # ===== L-BFGS step using CINEB gradient =====
            p = driver.two_loop(g)
            p = driver.step_limit(p)

            # take full step (no line search)
            x_new = x + p

            # evaluate new gradient
            g_new = eval_grad(x_new)


            driver.update(x_new - x, g_new - g)

            x = x_new
            g = g_new

            # ===== recompute energies / forces for next iteration & logging =====
            Es = [float(at.get_potential_energy(force_consistent=True)) for at in images]
            Fp_list, maxfp, _ = self.cineb_forces(images, Es, self.params.k_max)

            F_CI_vec = to_numpy_f64(images[hei].get_forces())
            maxF_CI = float(np.max(np.linalg.norm(F_CI_vec, axis=1)))
            rmsF_CI = float(np.sqrt(np.mean(np.linalg.norm(F_CI_vec, axis=1) ** 2)))
            rmsfp = rms_force(Fp_list)
            dE_CI = Es[hei] - Es[0]

            write_all_images_xyz(traj_file, images, energies=Es, iteration=iteration)
            log_info([f"   LBFGS {iteration:>4d} {hei:>6d} {dE_CI:>10.6f} "
                      f"{maxfp:>11.6f} {rmsfp:>10.6f} {maxF_CI:>10.6f} {rmsF_CI:>10.6f}\n"], self.output)

            iteration += 1
        else:
            log_info(["\nCINEB refinement reached maximum iterations.\n"], self.output)

        # Clean up fixed HEI
        self._cineb_fixed_hei = None

        # --- Stage 1 summary: CI part ---
        base, _ = os.path.splitext(self.output)
        cineb_mep = base + "_cineb_mep.xyz"
        cineb_hei = base + "_cineb_hei.xyz"
        write_xyz(cineb_mep, images, energies=Es)
        write_xyz(cineb_hei, [images[hei]], energies=[Es[hei]])

        log_info([
            "\n---------------------------------------------------------------\n",
            "               INFORMATION ABOUT SADDLE POINT     \n",
            "---------------------------------------------------------------\n",
            f"Climbing image                            ....  {hei}\n",
            f"Energy                                    ....  {Es[hei]: .8f} Eh\n",
            f"Max. abs. force                           ....  {maxF_CI: .4e} Eh/Angstrom\n",
            "\n-----------------------------------------\n",
            "  SADDLE POINT (ANGSTROEM)\n",
            "-----------------------------------------\n",
            self.atoms_to_xyz(images[hei])
        ], self.output)

        # --- Stage 2: Optional PRFO refinement ---
        if self.params.refine == 'nebts':
            from .PRFO import RFO

            # Use CI geometry as TS guess
            ts_guess = images[hei].copy()
            ts_guess.calc = self.atoms_R.calc
            ts_guess.f_max_th = images[0].f_max_th
            ts_guess.f_rms_th = images[0].f_rms_th
            ts_guess.dp_max_th = images[0].dp_max_th
            ts_guess.dp_rms_th = images[0].dp_rms_th

            ts_opt = RFO(ts_guess, self.output)

            E_TS = ts_opt.get_potential_energy(force_consistent=True)
            maxF_TS = np.max(np.linalg.norm(ts_opt.get_forces(), axis=1))
            rmsF_TS = np.sqrt(np.mean(np.linalg.norm(ts_opt.get_forces(), axis=1) ** 2))

            # Insert TS right after CI
            images.insert(hei + 1, ts_opt)
            Es.insert(hei + 1, E_TS)

            nebts_mep = base + "_nebts_mep.xyz"
            nebts_ts = base + "_nebts_ts.xyz"
            write_xyz(nebts_mep, images, energies=Es)
            write_xyz(nebts_ts, [ts_opt], energies=[E_TS])

            log_info([
                "\n---------------------------------------------------------------\n",
                "                      PATH SUMMARY FOR NEB-TS             \n",
                "---------------------------------------------------------------\n",
                "All forces in Eh/Angstrom. Global forces for TS.\n\n",
                "Image     E(Eh)   dE(kcal/mol)  max(|Fp|)  RMS(Fp)\n"
            ], self.output)

            kcal_per_Eh = 627.509
            for i, E in enumerate(Es):
                dE = (E - Es[0]) * kcal_per_Eh
                label = " TS" if i == hei + 1 else f"{i:3d}"
                marker = " <= TS" if i == hei + 1 else (" <= CI" if i == hei else "")
                maxF = np.max(np.linalg.norm(images[i].get_forces(), axis=1))
                rmsF = np.sqrt(np.mean(np.linalg.norm(images[i].get_forces(), axis=1) ** 2))
                log_info([
                    f"{label:>4s} {E:12.5f} {dE:11.2f} {maxF:11.5f} {rmsF:10.5f}{marker}\n"
                ], self.output)

            log_info([
                "\n-----------------------------------------\n",
                "  REFINED TS STRUCTURE (ANGSTROEM)\n",
                "-----------------------------------------\n",
                self.atoms_to_xyz(ts_opt)
            ], self.output)

            log_info([
                f"\nWrote NEB-TS MEP to: {nebts_mep}\n",
                f"Wrote TS structure to: {nebts_ts}\n"
            ], self.output)



    # ==============================================================
    # IDPP smoothing for initial path
    # ==============================================================
    def _run_idpp_smoothing(self, images, idpp_params=IDPPParams()):
        """
        Strong IDPP smoothing (BNEB-style):
        - LBFGS optimization instead of GD
        - damping + curvature correction
        - step clipping
        - projected force formulation identical to pysisyphus/BNEB

        This function preserves your image list + ASE Atoms interface.
        """

        # ----- extract coords -----
        coords = [to_numpy_f64(img.get_positions()) for img in images]
        R0 = coords[0]
        R1 = coords[-1]

        n_img = len(images)
        n_inner = n_img - 2
        n_atoms = R0.shape[0]

        # pair list (upper-triangle)
        pairs = _pair_indices(n_atoms)
        n_pair = pairs.shape[0]

        # ----- IDPP target 1/dist -----
        targets = _idpp_targets(R0, R1, n_inner)  # list length = n_inner

        # ----- flatten internal images -----
        x = np.concatenate([coords[i].reshape(-1) for i in range(1, n_img - 1)])

        # ----- LBFGS state -----
        m = 7
        S = []
        Y = []
        rho = []
        maxstep = 0.1
        curvature = 1.0

        def lbfgs_direction(g):
            """Two-loop recursion."""
            q = g.copy()
            alpha = []

            for s, y, r in reversed(list(zip(S, Y, rho))):
                a = r * np.dot(s, q)
                alpha.append(a)
                q -= a * y

            if len(Y) > 0:
                gamma = np.dot(S[-1], Y[-1]) / (np.dot(Y[-1], Y[-1]) + 1e-20)
            else:
                gamma = 1.0 / curvature

            z = gamma * q

            for (s, y, r), a in zip(zip(S, Y, rho), reversed(alpha)):
                b = r * np.dot(y, z)
                z += s * (a - b)

            return -z

        def compute_grad(x):
            """Compute IDPP gradient for all internal images."""
            grad = np.zeros_like(x)

            offset = 0
            for k in range(n_inner):
                Xi = x[offset:offset + n_atoms * 3].reshape(n_atoms, 3)

                # pair distances
                Rij = Xi[pairs[:, 0]] - Xi[pairs[:, 1]]      # (M,3)
                dij = np.linalg.norm(Rij, axis=1) + 1e-12
                inv = 1.0 / dij

                diff = inv - targets[k]                     # (M,)
                dE_dd = -2.0 * diff / (dij**2)              # derivative wrt dij

                g_pair = (dE_dd[:, None] * (Rij / dij[:, None]))  # (M,3)

                # accumulate forces
                g_atoms = np.zeros_like(Xi)
                for p in range(n_pair):
                    i, j = pairs[p]
                    g_atoms[i] +=  g_pair[p]
                    g_atoms[j] += -g_pair[p]

                grad[offset:offset + n_atoms * 3] = g_atoms.reshape(-1)
                offset += n_atoms * 3

            return grad

        # ----- optimization loop -----
        grad = compute_grad(x)

        for _ in range(idpp_params.max_steps):
            g = grad.copy()
            step = lbfgs_direction(g)

            # step clipping
            max_disp = np.max(np.abs(step))
            if max_disp > maxstep:
                step *= maxstep / max_disp

            x_new = x + step
            grad_new = compute_grad(x_new)

            # LBFGS update
            s = x_new - x
            y = grad_new - grad

            sy = np.dot(s, y)
            if sy > 1e-12:
                if len(S) == m:
                    S.pop(0); Y.pop(0); rho.pop(0)
                S.append(s); Y.append(y); rho.append(1.0 / sy)

            x = x_new
            grad = grad_new

            # stopping based on RMS gradient
            if np.sqrt(np.mean(grad * grad)) < idpp_params.eps:
                break

        # ----- write back optimized coords -----
        offset = 0
        for i in range(1, n_img - 1):
            Xi = x[offset:offset + n_atoms * 3].reshape(n_atoms, 3)
            images[i].set_positions(Xi)
            offset += n_atoms * 3

        return images



    # ------------------------------- main flow --------------------------------

    def run(self):
        # ===================================================================
        # Step 0: Process input images and determine if interpolation needed
        # ===================================================================
        n_input = len(self.input_images)
        n_required = self.params.n_images  # This is already n_images + 2
        
        log_info([
            f"\n{'='*70}\n",
            f"NEB Initialization\n",
            f"{'='*70}\n",
            f"Input images:    {n_input}\n",
            f"Required images: {n_required} (including endpoints)\n"
        ], self.output)
        
        # Check input validity
        if n_input < 2:
            raise ValueError(f"Need at least 2 images (reactant + product), got {n_input}")
        
        if n_input > n_required:
            raise ValueError(
                f"Too many input images: got {n_input}, but n_images={self.params.n_images-2} "
                f"requires exactly {n_required} images (including endpoints). "
                f"Please reduce input images or increase n_images parameter."
            )
        
        # Assign endpoints
        self.atoms_R = self.input_images[0]
        self.atoms_P = self.input_images[-1]
        
        # ===================================================================
        # CRITICAL: Align all input images BEFORE any interpolation/processing
        # This ensures interpolation happens in the correct coordinate system
        # ===================================================================
        log_info(["\nAligning input images to reactant reference...\n"], self.output)
        
        # Align product endpoint to reactant
        R_ref = to_numpy_f64(self.atoms_R.get_positions())
        P = to_numpy_f64(self.atoms_P.get_positions())
        P_aligned, rmsd_product, _, _ = kabsch_align(R_ref, P)
        self.atoms_P.set_positions(P_aligned)
        self.input_images[-1] = self.atoms_P  # Update in input list
        
        log_info([f"Product aligned to reactant. RMSD: {rmsd_product:.6f} Angstrom\n"], self.output)
        
        # Align all intermediate input images (if any exist)
        if n_input > 2:
            log_info([f"Aligning {n_input - 2} intermediate input image(s)...\n"], self.output)
            for i in range(1, n_input - 1):
                Q = to_numpy_f64(self.input_images[i].get_positions())
                Q_aligned, rmsd_i, _, _ = kabsch_align(R_ref, Q)
                self.input_images[i].set_positions(Q_aligned)
                log_info([f"  Image {i} aligned. RMSD: {rmsd_i:.6f} Angstrom\n"], self.output)
        
        log_info(["Input alignment completed.\n"], self.output)
        
        # ===================================================================
        # Determine if interpolation is needed
        # ===================================================================
        
        # Case 1: Exact number of images - use directly (already aligned)
        if n_input == n_required:
            log_info([f"\nExact number of images provided. Using input directly.\n"], self.output)
            images = self.input_images
            
        # Case 2: Need to insert additional images via interpolation
        else:
            n_to_insert = n_required - n_input
            log_info([
                f"\nNeed to insert {n_to_insert} additional image(s).\n",
                f"Analyzing distances to determine optimal insertion points...\n"
            ], self.output)
            
            # Compute straight-line distances between consecutive input images
            distances = self._compute_distances(self.input_images)
            
            log_info(["\nDistances between consecutive input images:\n"], self.output)
            for i, d in enumerate(distances):
                log_info([f"  Image {i} -> {i+1}: {d:.4f} Angstrom\n"], self.output)
            
            # Determine optimal insertion plan (prioritize largest gaps)
            insertion_plan = self._determine_insertion_plan(distances, n_to_insert)
            
            log_info(["\nInsertion plan:\n"], self.output)
            for seg_idx, count in sorted(insertion_plan, key=lambda x: x[0]):
                log_info([f"  Insert {count} image(s) between image {seg_idx} and {seg_idx+1}\n"], self.output)
            
            # Perform linear interpolation according to plan
            images = self._insert_images_by_plan(self.input_images, insertion_plan)
            
            log_info([f"\nTotal images after insertion: {len(images)}\n"], self.output)
            
            # Apply IDPP smoothing to refine interpolated path (optional)
            if self.params.ifidpp == 1:
                log_info(["\nApplying IDPP smoothing to interpolated images...\n"], self.output)
                images = self._run_idpp_smoothing(images)
                log_info(["IDPP smoothing completed.\n"], self.output)
            else:
                log_info(["\nIDPP smoothing disabled (ifidpp=0). Using linear interpolation only.\n"], self.output)
        
        # ===================================================================
        # CRITICAL: Re-align entire path after interpolation/IDPP
        # This ensures the complete path is properly aligned before NEB starts
        # ===================================================================
        log_info(["\nPerforming final Kabsch alignment of all images to reactant...\n"], self.output)
        self._align_path(images, ref_mode="reactant")
        log_info(["Final alignment completed.\n"], self.output)
        

        # ===================================================================
        # Step 1: Optional endpoint optimization
        # ===================================================================
        if self.params.initial_opt:
            self.atoms_R, self.atoms_P = self.optimize_endpoints(
                images[0], images[-1],
                f_max_th=self.params.neb_f_max_th,
                f_rms_th=self.params.neb_f_rms_th,
                max_iter=self.params.max_iter
            )
            images[0] = self.atoms_R
            images[-1] = self.atoms_P
        
        # ===================================================================
        # Step 2: Endpoint properties and alignment
        # ===================================================================
        def forces_info(atoms):
            F = atoms.get_forces()
            maxF = np.max(np.linalg.norm(F, axis=1))            
            rmsF = np.sqrt(np.mean(np.linalg.norm(F, axis=1) ** 2))
            return maxF, rmsF

        E_R = images[0].get_potential_energy(force_consistent=True)
        E_P = images[-1].get_potential_energy(force_consistent=True)
        maxF_R, rmsF_R = forces_info(images[0])
        maxF_P, rmsF_P = forces_info(images[-1])

        log_info([
            "\nProperties of fixed NEB end points:\n",
            "               Reactant:\n",
            f"                         E               ....   {E_R: .6f} Eh\n",
            f"                         RMS(F)          ....   {rmsF_R: .6f} Eh/Angstrom\n",
            f"                         MAX(|F|)        ....   {maxF_R: .6f} Eh/Angstrom\n",
            "               Product:\n",
            f"                         E               ....   {E_P: .6f} Eh\n",
            f"                         RMS(F)          ....   {rmsF_P: .6f} Eh/Angstrom\n",
            f"                         MAX(|F|)        ....   {maxF_P: .6f} Eh/Angstrom\n",
            "\nReactant XYZ (Angstrom):\n",
            self.atoms_to_xyz(images[0]),
            "\nProduct XYZ (Angstrom):\n",
            self.atoms_to_xyz(images[-1]),
            "\n"
        ], self.output)

        # Alignment
        log_info(["\nPerforming Kabsch alignment of all images...\n"], self.output)
        self._align_path(images, ref_mode="reactant")
        log_info(["Alignment completed.\n"], self.output)
        
        # Ensure all images have calculators
        for img in images:
            if img.calc is None:
                img.calc = self.atoms_R.calc

        # ===================================================================
        # Step 3: Normal NEB optimization loop
        # ===================================================================
        
        traj_file = os.path.splitext(self.output)[0] + "_image_traj.xyz"
        
        driver = LBFGSDriver(
            m=self.params.lbfgs_m,
            curvature=70.0,
            maxstep=self.params.step0
        )
        
        self._k_springs_history = []
        
        def eval_grad(x_flat):
            self._unpack_internal(x_flat, images)
            Es = self.get_energies(images)
            
            Fp_list, _, _ = neb_forces(
                images, Es,
                k_spring=None,
                k_springs=None,
                use_dynamic_k=self.params.use_dynamic_k,
                k_min=self.params.k_min,
                k_max=self.params.k_max,
                k_decay=self.params.k_decay
            )
            
            grads = [(-Fp_list[i]).reshape(-1) for i in range(1, len(images) - 1)]
            return np.concatenate(grads) if grads else np.zeros_like(x_flat)
        
        x = self._pack_internal(images)
        g = eval_grad(x)
        iteration = 0
        
        Es = self.get_energies(images)
        
        if self.params.use_dynamic_k:
            k_springs = compute_dynamic_k(Es, self.params.k_min, self.params.k_max, self.params.k_decay)
            self._k_springs_history.append(k_springs.copy())
        else:
            k_springs = [self.params.k_max] * len(images)
        
        Fp_list, maxfp, hei = neb_forces(
            images, Es,
            k_spring=None,
            k_springs=k_springs,
            use_dynamic_k=False
        )
        rmsfp = rms_force(Fp_list)
        dE_hei = Es[hei] - Es[0]
        
        # Log header with initial state
        if self.params.use_dynamic_k:
            k_hei = k_springs[hei]
            log_info([
                "\nStarting NEB iterations with ORCA-style dynamic spring constants:\n",
                f"k_min = {self.params.k_min:.4f}, k_max = {self.params.k_max:.4f}, k_decay = {self.params.k_decay:.4f}\n",
                "Optim.  Iteration  HEI  E(HEI)-E(0)  max(|Fp|)   RMS(Fp)   k(HEI)\n",
                f"Convergence thresholds         {self.params.neb_f_max_th: .6f}   {self.params.neb_f_rms_th: .6f}\n",
                f"Initial {iteration:>4d} {hei:>6d} {dE_hei:>10.6f} {maxfp:>11.6f} {rmsfp:>10.6f}   {k_hei:.4f}\n"
            ], self.output)
        else:
            log_info([
                "\nStarting NEB iterations with fixed spring constant:\n",
                f"k_spring = {self.params.k_max:.4f}\n",
                "Optim.  Iteration  HEI  E(HEI)-E(0)  max(|Fp|)   RMS(Fp)\n",
                f"Convergence thresholds         {self.params.neb_f_max_th: .6f}   {self.params.neb_f_rms_th: .6f}\n",
                f"Initial {iteration:>4d} {hei:>6d} {dE_hei:>10.6f} {maxfp:>11.6f} {rmsfp:>10.6f}\n"
            ], self.output)
        
        # Write initial trajectory
        write_all_images_xyz(traj_file, images, energies=Es, iteration=iteration)
        
        # ===== Main optimization loop =====
        while iteration < self.params.max_iter and not driver.should_stop(g, self.params.neb_f_max_th, self.params.neb_f_rms_th):
            iteration += 1  # Increment at start to avoid confusion
            
            # L-BFGS step
            p = driver.two_loop(g)
            p = driver.step_limit(p)
            
            x_new = x + p
            g_new = eval_grad(x_new)
            
            driver.update(x_new - x, g_new - g)
            
            x = x_new
            g = g_new
            
            # Compute forces and energies for logging
            Es = self.get_energies(images)
            
            if self.params.use_dynamic_k:
                k_springs = compute_dynamic_k(Es, self.params.k_min, self.params.k_max, self.params.k_decay)
                self._k_springs_history.append(k_springs.copy())
            else:
                k_springs = [self.params.k_max] * len(images)
            
            Fp_list, maxfp, hei = neb_forces(
                images, Es,
                k_spring=None,
                k_springs=k_springs,
                use_dynamic_k=False
            )
            
            rmsfp = rms_force(Fp_list)
            dE_hei = Es[hei] - Es[0]
            
            write_all_images_xyz(traj_file, images, energies=Es, iteration=iteration)
            
            if self.params.use_dynamic_k:
                k_hei = k_springs[hei]
                log_info([f"   LBFGS {iteration:>4d} {hei:>6d} {dE_hei:>10.6f} {maxfp:>11.6f} {rmsfp:>10.6f}   {k_hei:.4f}\n"], self.output)
            else:
                log_info([f"   LBFGS {iteration:>4d} {hei:>6d} {dE_hei:>10.6f} {maxfp:>11.6f} {rmsfp:>10.6f}\n"], self.output)
        
        if iteration == 0:
            log_info(["\nNEB already converged at initial geometry (iteration 0).\n"], self.output)
        elif iteration == self.params.max_iter:
            log_info(["\nNEB optimization reached maximum iterations.\n"], self.output)
        else:
            log_info([f"\nNEB optimization converged after {iteration} iterations.\n"], self.output)


        # ---------------------------------------------------------------
        #  Final path summary
        # ---------------------------------------------------------------
        base, _ = os.path.splitext(self.output)
        mep_path = base + "_mep.xyz"
        hip_path = base + "_hei.xyz"
        write_xyz(mep_path, images, energies=Es)
        write_xyz(hip_path, [images[hei]], energies=[Es[hei]])

        # Compute straight-line distances
        distances = [np.linalg.norm(images[i+1].get_positions() - images[i].get_positions()) for i in range(len(images)-1)]

        kcal_per_Eh = 627.509
        summary = [
            "\n---------------------------------------------------------------\n",
            "                         PATH SUMMARY              \n",
            "---------------------------------------------------------------\n",
            "All forces in Eh/Angstrom.\n\n",
            "Image Dist.(Ang.)    E(Eh)   dE(kcal/mol)  max(|Fp|)  RMS(Fp)\n"
        ]

        for i, (E, Fp) in enumerate(zip(Es, Fp_list)):
            dE = (E - Es[0]) * kcal_per_Eh
            dist = 0.0 if i == 0 else distances[i-1]
            maxF = np.max(np.linalg.norm(Fp, axis=1))
            rmsF = np.sqrt(np.mean(np.linalg.norm(Fp, axis=1) ** 2))
            marker = " <= HEI" if i == hei else ""
            summary.append(f"{i:3d} {dist:9.3f} {E:13.5f} {dE:11.2f} {maxF:11.5f} {rmsF:10.5f}{marker}\n")

        summary.append("\nStraight line distance between images along the path:\n")
        for i, d in enumerate(distances):
            summary.append(f"        D({i:2d}-{i+1:2d}) = {d:7.4f} Ang.\n")

        summary.append(
            "\n---------------------------------------------------------------\n"
            "           INFORMATION ABOUT HIGHEST ENERGY IMAGE\n"
            "---------------------------------------------------------------\n"
            f"Highest energy image                      ....  {hei}\n"
            f"Energy                                    ....  {Es[hei]: .8f} Eh\n"
            f"Max. abs. force                           ....  {np.max(np.linalg.norm(Fp_list[hei], axis=1)) : .4e} Eh/Angstrom\n"
            "\n-----------------------------------------\n"
            "  HIGHEST ENERGY IMAGE (ANGSTROEM)\n"
            "-----------------------------------------\n"
        )
        summary.append(self.atoms_to_xyz(images[hei]))
        log_info(summary, self.output)

        log_info([
            f"\nWrote MEP to: {mep_path}\n",
            f"Wrote HEI to: {hip_path}\n",
            f"Wrote trajectory to: {traj_file}\n"
        ], self.output)
        
        
        # 3) Optional: CINEB refinement
        if self.params.refine == 'cineb' or self.params.refine == 'nebts':
            if iteration == self.params.max_iter:
                log_info([
                    "\nNEB did not converge. Skipping CINEB refinement.\n",
                    "You may try to increase max_iter or check the initial path.\n"
                ], self.output)
            else:
                self.restart_run(images, energies=Es)
