# -*- coding: utf-8 -*-
"""
STRING implementation with:
- Kabsch alignment (reports RMSD)
- Linear interpolation for initial path
- Projected FIRE optimizer on perpendicular forces
- Linear arclength reparametrization (equal arc bins)
- Optional CI-STRING refinement and PRFO TS refinement (integrated)

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
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
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


def append_all_images_xyz(filename: str, images: List[Atoms], energies: Optional[List[float]] = None, iteration: int = 0):
    """
    Append all current images to an xyz trajectory file (for debug).
    Each block corresponds to one iteration.
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


def inherit_attrs(src: Atoms, dst: Atoms):
    """
    Ensure new images/TS carry calculator and threshold attributes.
    Called immediately after we create/copy a new Atoms.
    """
    if getattr(src, "calc", None) is not None:
        dst.calc = src.calc
    # user thresholds set by Dispatcher.set_throshould()
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


# =============================================================================
# ------------------------- String Core Components ----------------------------
# =============================================================================

@dataclass
class STRINGParams:
    n_images: int = 9                 # total images including endpoints
    max_iter: int = 300
    # Force thresholds for projected forces (same meaning as NEB)
    string_f_max_th: float = 5.0e-3          # max(|F_perp|) threshold
    string_f_rms_th: float = 1.0e-3          # RMS(F_perp) threshold
    cistring_f_max_th: float = 4.00e-03           # max(|Fp|) threshold for CINEB
    cistring_f_rms_th: float = 2.00e-03           # RMS(Fp) threshold for CINEB
    # FIRE parameters (projected forces drive motion)
    fire_dt: float = 0.2              # initial "time step" (acts like step length)
    fire_dt_max: float = 0.6
    fire_finc: float = 1.1
    fire_fdec: float = 0.4
    fire_alpha0: float = 0.2
    fire_falpha: float = 0.99
    fire_Nmin: int = 5
    # Reparametrization cadence
    reparam_every: int = 3            # how often (iterations) to reparametrise along arclength
    # Refinement control
    refine: Optional[str] = None      # 'cistring' or 'stringts' or None


def tangent(images: List[Atoms], idx: int) -> np.ndarray:
    """
    Compute unit tangent at image idx using neighbors.
    Simple central difference; endpoints are never asked here.
    """
    Rm1 = to_numpy_f64(images[idx - 1].get_positions())
    R   = to_numpy_f64(images[idx    ].get_positions())
    Rp1 = to_numpy_f64(images[idx + 1].get_positions())
    t = (Rp1 - R) + (R - Rm1)
    t = t.reshape(-1)
    nrm = np.linalg.norm(t)
    if nrm < 1e-16:
        # fallback to forward difference
        t = (Rp1 - R).reshape(-1)
        nrm = np.linalg.norm(t)
        if nrm < 1e-16:
            # fallback to a dummy axis; extremely unlikely
            t = np.zeros_like(t)
            t[0] = 1.0
            nrm = 1.0
    return t / nrm


def project_perp(F: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """
    Project the force onto the hyperplane orthogonal to tau (both flattened).
    """
    c = float(np.dot(F, tau))
    Fp = F - c * tau
    return Fp


def linear_reparam(images: List[Atoms]):
    """
    Reparametrize a path to equal arclength by piecewise-linear resampling.
    Uses the CURRENT (already aligned) coordinates. Endpoints fixed.

    In-place on images: updates internal images' positions.
    """
    M = len(images)
    if M <= 2:
        return

    # gather current coords
    X = [img.get_positions().copy() for img in images]

    # cumulative arclength on aligned coordinates
    s = [0.0]
    for i in range(1, M):
        ds = np.linalg.norm(X[i] - X[i-1])
        s.append(s[-1] + ds)
    L = s[-1]
    if L <= 1e-15:
        return

    # target arc positions for internal nodes
    s_targets = np.linspace(0.0, L, M)
    # endpoints unchanged
    for k in range(1, M - 1):
        sk = s_targets[k]
        # find segment [j, j+1] such that s[j] <= sk <= s[j+1]
        j = np.searchsorted(s, sk) - 1
        if j < 0: j = 0
        if j >= M - 1: j = M - 2

        if s[j+1] - s[j] < 1e-16:
            lam = 0.0
        else:
            lam = (sk - s[j]) / (s[j+1] - s[j])

        Xk = (1.0 - lam) * X[j] + lam * X[j+1]
        images[k].set_positions(Xk)


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

def project_forces_perpendicular(F_list, images):
    """
    Project atomic forces onto the subspace perpendicular to the local tangent vector of the string.
    Used in string method and NEB to remove tangential components that do not contribute to MEP convergence.

    Parameters
    ----------
    F_list : list of (N_atoms x 3) np.ndarray
        Raw forces on each image (已做刚体投影的)
    images : list of ase.Atoms
        Path images, including endpoints

    Returns
    -------
    Fp_list : list of (N_atoms x 3) np.ndarray
        Forces with tangent components removed (perpendicular to path)
    """
    n_img = len(images)
    Fp_list = []

    for i in range(n_img):
        if i == 0 or i == n_img - 1:
            # endpoints are fixed: zero force
            Fp_list.append(np.zeros_like(F_list[i]))
            continue

        # --- compute local tangent ---
        R_prev = images[i - 1].get_positions()
        R_next = images[i + 1].get_positions()
        tangent = R_next - R_prev
        tangent /= np.linalg.norm(tangent)

        # --- project out parallel component ---
        F = F_list[i]
        F_para = np.sum(F * tangent, axis=1)[:, None] * tangent
        F_perp = F - F_para
        Fp_list.append(F_perp)

    return Fp_list


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
    m = masses.reshape(-1, 1)
    sqrtm = np.sqrt(masses).reshape(-1, 1)

    # mass-weighted coordinates relative to COM
    R_cm = (positions * m).sum(axis=0) / m.sum()
    r = positions - R_cm  # (N,3)

    # Build 6 rigid-body mode columns in mass-weighted space
    # Translations (Tx, Ty, Tz): same direction for all atoms, scaled by sqrt(m)
    B = np.zeros((3 * N, 6), dtype=np.float64)

    # T_x, T_y, T_z
    for a in range(N):
        s = float(sqrtm[a, 0])
        i3 = 3 * a
        B[i3 + 0, 0] = s
        B[i3 + 1, 1] = s
        B[i3 + 2, 2] = s

    # Rotations (about x,y,z): omega × r  (mass-weighted)
    # e_Rx -> [0, -z,  y], e_Ry -> [ z, 0, -x], e_Rz -> [-y, x, 0]
    for a in range(N):
        s = float(sqrtm[a, 0])
        x, y, z = r[a]
        i3 = 3 * a
        # R_x
        B[i3 + 0, 3] = 0.0
        B[i3 + 1, 3] = -s * z
        B[i3 + 2, 3] =  s * y
        # R_y
        B[i3 + 0, 4] =  s * z
        B[i3 + 1, 4] = 0.0
        B[i3 + 2, 4] = -s * x
        # R_z
        B[i3 + 0, 5] = -s * y
        B[i3 + 1, 5] =  s * x
        B[i3 + 2, 5] = 0.0

    # Mass-weighted forces (vectorized 3N)
    Fmw = (forces * sqrtm).reshape(-1)  # (3N,)

    # Robust projection via SVD (handles rank-deficiency)
    U, S, VT = np.linalg.svd(B, full_matrices=False)
    if S.size == 0 or S.max() <= 0.0:
        # Degenerate pathological case; nothing to project
        return forces.copy()

    rank = int(np.sum(S > tol * S.max()))
    Q = U[:, :rank]  # orthonormal basis of the subspace

    # Project out: Fmw' = (I - Q Q^T) Fmw
    Fmw_proj = Fmw - Q @ (Q.T @ Fmw)

    # Back to non-mass-weighted
    F_proj = (Fmw_proj.reshape(-1, 3) / sqrtm)
    return F_proj

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
            # next reference is the aligned current (smooth propagation)
            ref = aligned

# =============================================================================
# ----------------------------- Projected FIRE --------------------------------
# =============================================================================

class ProjectedFIRE:
    """
    FIRE optimizer working on the projected-perpendicular forces of all internal images.
    It does NOT keep long history (so reparametrization won't hurt convergence).
    """

    def __init__(self, dt=0.2, dt_max=1.0, finc=1.1, fdec=0.5, alpha0=0.1, falpha=0.99, Nmin=5):
        self.dt = dt
        self.dt_max = dt_max
        self.finc = finc
        self.fdec = fdec
        self.alpha0 = alpha0
        self.falpha = falpha
        self.Nmin = Nmin
        self.reset()

    def reset(self):
        self.v = None
        self.alpha = self.alpha0
        self.npos = 0  # number of positive P steps

    def step(self, X: np.ndarray, F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Single FIRE step for a vectorized system:
        X, F are flattened coordinates/forces of all internal images concatenated.
        Returns new (X, v).
        """
        if self.v is None or self.v.shape != X.shape:
            self.v = np.zeros_like(X)

        # Velocity-Verlet-like update
        self.v += self.dt * F

        # Power
        P = float(np.dot(self.v, F))

        # Adjust velocities
        v_norm = np.linalg.norm(self.v); F_norm = np.linalg.norm(F)
        if v_norm > 0 and F_norm > 0:
            self.v = (1 - self.alpha) * self.v + self.alpha * (self.v / v_norm) * F_norm

        # Position update
        X_new = X + self.dt * self.v

        # Adapt step sizes
        if P > 0:
            self.npos += 1
            if self.npos > self.Nmin:
                self.dt = min(self.dt * self.finc, self.dt_max)
                self.alpha *= self.falpha
        else:
            self.npos = 0
            self.dt *= self.fdec
            self.alpha = self.alpha0
            # Rebound
            self.v[:] = 0.0

        return X_new, self.v


# =============================================================================
# ------------------------------- STRING Class --------------------------------
# =============================================================================

class String(JobABC):
    def __init__(self,
                 output: str,
                 atoms_R: Atoms,
                 atoms_P: Atoms,
                 params: Optional[STRINGParams] = None,
                 paras: Optional[dict] = None):
        super().__init__(output)

        # IMPORTANT: don't copy unless we create new images
        self.atoms_R = atoms_R
        self.atoms_P = atoms_P
        self.atoms_R.calc = atoms_R.calc
        self.atoms_P.calc = atoms_P.calc

        self.params = params if params is not None else STRINGParams()
        if isinstance(paras, dict):
            low = { (k.lower() if isinstance(k, str) else k): v for k, v in paras.items() }
            for f in fields(self.params):
                name = f.name.lower()
                if name in low:
                    setattr(self.params, f.name, low[name])

        if self.params.n_images < 2:
            raise ValueError("n_images must be >= 2 (including endpoints)")

    # -------------------------- pack/unpack internal images -------------------

    def _pack_internal(self, images: List[Atoms]) -> np.ndarray:
        arrs = []
        for i in range(1, len(images) - 1):
            arrs.append(to_numpy_f64(images[i].get_positions()).reshape(-1))
        return np.concatenate(arrs) if arrs else np.zeros(0, dtype=np.float64)

    def _unpack_internal(self, x: np.ndarray, images: List[Atoms]):
        offset = 0
        for i in range(1, len(images) - 1):
            n = len(images[i]) * 3
            Xi = x[offset:offset + n].reshape(-1, 3)
            images[i].set_positions(Xi)
            offset += n

    # -------------------------- projected forces (string) ---------------------

    def string_forces(self, images: List[Atoms]) -> Tuple[List[np.ndarray], float, int]:
        """
        Compute perpendicular forces for internal images:
        F_perp = F_true - (F_true · tau) tau
        Returns: list of F_perp (shape (N,3)), max|F_perp|, HEI index by energy
        """
        n_img = len(images)
        forces_perp = [None] * n_img

        # energies and raw forces
        Es = get_energies(images)
        raw_forces = [to_numpy_f64(at.get_forces()).reshape(-1) for at in images]

        # HEI detection among inner images
        inner = list(range(1, n_img - 1))
        hei = max(inner, key=lambda i: Es[i])

        max_fp = 0.0
        for i in inner:
            tau = tangent(images, i)  # flattened
            F = raw_forces[i]
            Fp = project_perp(F, tau).reshape(-1, 3)
            forces_perp[i] = Fp
            max_fp = max(max_fp, float(np.abs(Fp).max()))

        # endpoints: zero
        forces_perp[0] = np.zeros_like(to_numpy_f64(images[0].get_positions()))
        forces_perp[-1] = np.zeros_like(to_numpy_f64(images[-1].get_positions()))

        return forces_perp, max_fp, hei

    # ------------------------------- restart (CI-STRING + TS) -----------------

    def restart_run(self, images: List[Atoms], energies: Optional[List[float]] = None):
        """
        Continue from a converged STRING path and perform:
        - CI-STRING refinement (climbing image: reverse parallel component, no spring here)
        - Optional TS refinement by PRFO if params.refine == 'stringts'
        """
        # --- initialize ---
        base, _ = os.path.splitext(self.output)
        traj_file = base + "_cistring_traj.xyz"

        # inherit attrs for safety on existing images (if re-created elsewhere)
        for i in range(len(images)):
            inherit_attrs(images[0], images[i])

        # simple FIRE on CI-STRING (perp for others; CI gets parallel reversed)
        fire = ProjectedFIRE(dt=self.params.fire_dt, dt_max=self.params.fire_dt_max,
                             finc=self.params.fire_finc, fdec=self.params.fire_fdec,
                             alpha0=self.params.fire_alpha0, falpha=self.params.fire_falpha,
                             Nmin=self.params.fire_Nmin)

        def eval_grad_cistring(x_flat):
            self._unpack_internal(x_flat, images)
            Es = get_energies(images)
            # find climbing image among inner images
            inner = list(range(1, len(images) - 1))
            hei = max(inner, key=lambda i: Es[i])

            F_list = []
            for i in inner:
                pos = to_numpy_f64(images[i].get_positions()).reshape(-1, 3)
                F = to_numpy_f64(images[i].get_forces()).reshape(-1)
                tau = tangent(images, i)  # flattened
                if i == hei:
                    # reverse parallel component: F_ci = F - 2 (F·tau) tau
                    Fp = (F - 2.0 * np.dot(F, tau) * tau).reshape(-1, 3)
                else:
                    # standard perpendicular projection
                    Fp = (F - np.dot(F, tau) * tau).reshape(-1, 3)
                F_list.append(Fp.reshape(-1))
            return np.concatenate(F_list) if F_list else np.zeros_like(x_flat), Es, hei

        # pack internal images
        x = self._pack_internal(images)
        v_dummy = None  # handled by FIRE

        # logging header
        log_info([
            "\nStarting CI-STRING refinement:\n",
            "Optim.  Iteration  CI   E(CI)-E(0)   max(|Fp|)   RMS(Fp)   max(|FCI|)   RMS(FCI)\n",
            f"Convergence thresholds         {self.params.cistring_f_max_th: .6f}   {self.params.cistring_f_rms_th: .6f}       0.002000    0.001000\n"
        ], self.output)

        # iterate
        iteration = 0
        while iteration < self.params.max_iter:
            g, Es, hei = eval_grad_cistring(x)
            # grad here is +F_proj; FIRE expects F, we pass g as force.
            # flatten Fp for all internal images:
            # compute RMS, max
            # rebuild list Fp per image (for RMS computation)
            Fp_list = [np.zeros_like(to_numpy_f64(images[0].get_positions()))]
            offset = 0
            maxfp = 0.0
            for i in range(1, len(images) - 1):
                n = len(images[i]) * 3
                Fi = g[offset:offset + n].reshape(-1, 3)
                maxfp = max(maxfp, float(np.abs(Fi).max()))
                Fp_list.append(Fi)
                offset += n
            Fp_list.append(np.zeros_like(to_numpy_f64(images[-1].get_positions())))
            rmsfp = rms_force_perp(Fp_list)

            dE_ci = Es[hei] - Es[0]
            append_all_images_xyz(traj_file, images, energies=Es, iteration=iteration)
            log_info([f"   FIRE  {iteration:>4d} {hei:>6d} {dE_ci:>10.6f} {maxfp:>11.6f} {rmsfp:>10.6f}        "], self.output)

            # CI global force
            F_CI = to_numpy_f64(images[hei].get_forces())
            maxF_CI = np.max(np.linalg.norm(F_CI, axis=1))
            rmsF_CI = np.sqrt(np.mean(np.linalg.norm(F_CI, axis=1) ** 2))
            log_info([f"{maxF_CI:>10.6f}   {rmsF_CI:>10.6f}\n"], self.output)

            # stopping
            if (maxfp < self.params.cistring_f_max_th) and (rmsfp < self.params.cistring_f_rms_th):
                break

            # FIRE step on internal images (x is concatenation of internal)
            x_new, v_dummy = fire.step(x, g)
            self._unpack_internal(x_new, images)

            # optional reparam every few steps
            if (iteration + 1) % self.params.reparam_every == 0:
                linear_reparam(images)
                # important: keep endpoints exactly as they are (unchanged)
                inherit_attrs(images[0], images[0]); inherit_attrs(images[0], images[-1])

            x = self._pack_internal(images)
            iteration += 1

        if iteration == self.params.max_iter:
            log_info(["\nCI-STRING reached maximum iterations.\n"], self.output)
        else:
            log_info([f"\nCI-STRING refinement converged after {iteration} iterations.\n"], self.output)

        # write CI-STRING artifacts
        Es = get_energies(images)
        hei = max(range(1, len(images) - 1), key=lambda i: Es[i])
        base, _ = os.path.splitext(self.output)
        cistr_mep = base + "_cistring_mep.xyz"
        cistr_hei = base + "_cistring_hei.xyz"
        write_xyz(cistr_mep, images, energies=Es)
        write_xyz(cistr_hei, [images[hei]], energies=[Es[hei]])

        log_info([
            "\n---------------------------------------------------------------\n",
            "               INFORMATION ABOUT SADDLE POINT     \n",
            "---------------------------------------------------------------\n",
            f"Climbing image                            ....  {hei}\n",
            f"Energy                                    ....  {Es[hei]: .8f} Eh\n",
            f"Max. abs. force                           ....  {np.max(np.linalg.norm(to_numpy_f64(images[hei].get_forces()), axis=1)) : .4e} Eh/Angstrom\n",
            "\n-----------------------------------------\n",
            "  SADDLE POINT (ANGSTROEM)\n",
            "-----------------------------------------\n",
            atoms_to_xyz_block(images[hei])
        ], self.output)

        # --- Optional PRFO TS refinement ---
        if self.params.refine == 'stringts':
            from .PRFO import RFO  # your existing TS refiner

            ts_guess = copy.deepcopy(images[hei])
            inherit_attrs(images[0], ts_guess)  # calc & thresholds

            ts_opt = RFO(ts_guess, self.output)  # you said: don't change this function
            E_TS = ts_opt.get_potential_energy(force_consistent=True)

            # insert TS near CI (after CI)
            images.insert(hei + 1, ts_opt)
            Es.insert(hei + 1, E_TS)

            # write NEB-TS-like artifacts for STRINGTS
            stringts_mep = base + "_stringts_mep.xyz"
            stringts_ts = base + "_stringts_ts.xyz"
            write_xyz(stringts_mep, images, energies=Es)
            write_xyz(stringts_ts, [ts_opt], energies=[E_TS])

            # print summary
            log_info([
                "\n---------------------------------------------------------------\n",
                "                      PATH SUMMARY FOR STRING-TS          \n",
                "---------------------------------------------------------------\n",
                "All forces in Eh/Angstrom. Global forces for TS.\n\n",
                "Image     E(Eh)   dE(kcal/mol)  max(|F|)  RMS(F)\n"
            ], self.output)

            kcal_per_Eh = 627.509
            for i, E in enumerate(Es):
                dE = (E - Es[0]) * kcal_per_Eh
                label = " TS" if i == hei + 1 else f"{i:3d}"
                marker = " <= TS" if i == hei + 1 else (" <= CI" if i == hei else "")
                maxF = np.max(np.linalg.norm(to_numpy_f64(images[i].get_forces()), axis=1))
                rmsF = np.sqrt(np.mean(np.linalg.norm(to_numpy_f64(images[i].get_forces()) ** 2, axis=1)))
                log_info([f"{label:>4s} {E:12.5f} {dE:11.2f} {maxF:11.5f} {rmsF:10.5f}{marker}\n"], self.output)

            log_info([
                "\n-----------------------------------------\n",
                "  REFINED TS STRUCTURE (ANGSTROEM)\n",
                "-----------------------------------------\n",
                atoms_to_xyz_block(ts_opt)
            ], self.output)

            log_info([
                f"\nWrote STRING-TS MEP to: {stringts_mep}\n",
                f"Wrote TS structure to:  {stringts_ts}\n"
            ], self.output)

    # -------------------------------- main flow --------------------------------

    def run(self):

        # --- 0) Report endpoints and XYZ ---
        def forces_info(atoms):
            F = to_numpy_f64(atoms.get_forces())
            maxF = np.max(np.linalg.norm(F, axis=1))
            rmsF = np.sqrt(np.mean(np.linalg.norm(F, axis=1) ** 2))
            return maxF, rmsF

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
            "\nReactant XYZ (Angstrom):\n",
            atoms_to_xyz_block(self.atoms_R),
            "\nProduct XYZ (Angstrom):\n",
            atoms_to_xyz_block(self.atoms_P),
            "\n"
        ], self.output)

        # --- 1) Alignment (Kabsch) + initial linear path ---
        R = to_numpy_f64(self.atoms_R.get_positions())
        P = to_numpy_f64(self.atoms_P.get_positions())
        P_aligned, rmsd, _, _ = kabsch_align(R, P)
        self.atoms_P.set_positions(P_aligned)
        log_info([f"Alignment done. RMSD: {rmsd:.6f} (Angstrom)\n"], self.output)

        n_img = self.params.n_images
        if n_img < 2:
            raise ValueError("n_images must be >= 2")

        # create images only once; immediately inherit attrs
        images: List[Atoms] = [self.atoms_R]
        for k in range(1, n_img - 1):
            lam = k / (n_img - 1)
            A = self.atoms_R.copy()
            inherit_attrs(self.atoms_R, A)  # calc & thresholds!
            A.set_positions((1.0 - lam) * R + lam * P_aligned)
            images.append(A)
        images.append(self.atoms_P)

        for img in images:
            if img.calc is None:
                img.calc = self.atoms_R.calc

        base, _ = os.path.splitext(self.output)
        traj_file = base + "_string_traj.xyz"

        # --- 2) FIRE optimization on projected-perp forces + periodic reparam ---
        fire = ProjectedFIRE(dt=self.params.fire_dt, dt_max=self.params.fire_dt_max,
                             finc=self.params.fire_finc, fdec=self.params.fire_fdec,
                             alpha0=self.params.fire_alpha0, falpha=self.params.fire_falpha,
                             Nmin=self.params.fire_Nmin)

        # pack internal images
        x = self._pack_internal(images)

        # header
        log_info([
            "\nStarting STRING optimization:\n",
            "Optim.  Iteration  HEI  E(HEI)-E(0)  max(|Fp|)   RMS(Fp)\n",
            f"Convergence thresholds         {self.params.string_f_max_th: .6f}   {self.params.string_f_rms_th: .6f}\n"
        ], self.output)

        iteration = 0
        while iteration < self.params.max_iter:
            # --- ① compute forces (with rigid-body projection) ---
            Fp_list = []
            Es = []
            for img in images:
                E = img.get_potential_energy(force_consistent=True)
                Es.append(E)
            hei = int(np.argmax(Es[1:-1]) + 1)

            for img in images:
                F_raw = to_numpy_f64(img.get_forces())
                masses = to_numpy_f64(img.get_masses())
                pos = to_numpy_f64(img.get_positions())
                # remove rigid-body modes from forces
                F_proj = project_out_rigidbody_forces(F_raw, pos, masses)
                Fp_list.append(F_proj)

            # --- ② project forces perpendicular to tangent ---
            Fp_list = project_forces_perpendicular(Fp_list, images)  

            maxfp = np.max([np.linalg.norm(Fp_list[i], axis=1).max() for i in range(1, len(images) - 1)])
            rmsfp = rms_force_perp(Fp_list)
            dE_hei = Es[hei] - Es[0]

            append_all_images_xyz(traj_file, images, energies=Es, iteration=iteration)
            log_info([f"   FIRE  {iteration:>4d} {hei:>6d} {dE_hei:>10.6f} {maxfp:>11.6f} {rmsfp:>10.6f}\n"], self.output)

            if (maxfp < self.params.string_f_max_th) and (rmsfp < self.params.string_f_rms_th):
                break

            # --- ③ FIRE step ---
            g = np.concatenate([Fp_list[i].reshape(-1) for i in range(1, len(images) - 1)]) if len(images) > 2 else np.zeros_like(x)
            x_new, _ = fire.step(x, g)
            self._unpack_internal(x_new, images)

            # --- ④ Alignment ---
            if (iteration + 1) % self.params.reparam_every == 0:
                # Align images along the path
                align_path_inplace(images, mode="chain")
                # Linear reparameterization
                linear_reparam(images)
                # Align images along the path
                align_path_inplace(images, mode="chain")
                inherit_attrs(images[0], images[0])
                inherit_attrs(images[0], images[-1])

            x = self._pack_internal(images)
            iteration += 1

        if iteration == self.params.max_iter:
            log_info(["\nSTRING reached maximum iterations.\n"], self.output)
        else:
            log_info([f"\nSTRING optimization converged after {iteration} iterations.\n"], self.output)

        # --- 3) Final path summary and dumps ---
        Es = get_energies(images)
        hei = max(range(1, len(images) - 1), key=lambda i: Es[i])

        mep_path = base + "_string_mep.xyz"
        hei_path = base + "_string_hei.xyz"
        write_xyz(mep_path, images, energies=Es)
        write_xyz(hei_path, [images[hei]], energies=[Es[hei]])

        # arc distances between neighboring images
        distances = [np.linalg.norm(
            to_numpy_f64(images[i+1].get_positions()) - to_numpy_f64(images[i].get_positions())
        ) for i in range(len(images)-1)]

        kcal_per_Eh = 627.509
        summary = [
            "\n---------------------------------------------------------------\n",
            "                         PATH SUMMARY (STRING)         \n",
            "---------------------------------------------------------------\n",
            "All forces in Eh/Angstrom.\n\n",
            "Image Dist.(Ang.)    E(Eh)   dE(kcal/mol)  max(|Fp|)  RMS(Fp)\n"
        ]
        for i, (E, Fp) in enumerate(zip(Es, [np.zeros_like(images[0].get_positions())] + [Fp_list[j] for j in range(1, len(images)-1)] + [np.zeros_like(images[-1].get_positions())])):
            dE = (E - Es[0]) * kcal_per_Eh
            dist = 0.0 if i == 0 else distances[i-1]
            maxF = 0.0 if i in (0, len(images)-1) else float(np.max(np.linalg.norm(Fp, axis=1)))
            rmsF = 0.0 if i in (0, len(images)-1) else float(np.sqrt(np.mean(np.linalg.norm(Fp, axis=1) ** 2)))
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
            f"Max. abs. force                           ....  {np.max(np.linalg.norm(to_numpy_f64(images[hei].get_forces()), axis=1)) : .4e} Eh/Angstrom\n"
            "\n-----------------------------------------\n"
            "  HIGHEST ENERGY IMAGE (ANGSTROEM)\n"
            "-----------------------------------------\n"
        )
        summary.append(atoms_to_xyz_block(images[hei]))
        log_info(summary, self.output)

        log_info([
            f"\nWrote STRING MEP to: {mep_path}\n",
            f"Wrote HEI to:        {hei_path}\n",
            f"Wrote trajectory to: {traj_file}\n"
        ], self.output)


        # 4) Optional: CI-String / STRINGTS
        if self.params.refine in ['cistring', 'stringts']:
            if iteration == self.params.max_iter:
                log_info([
                    "\nSince STRING did not converge, skipping CI-STRING refinement.\n",
                    "You may try increasing max_iter or relaxing convergence thresholds.\n"
                ], self.output)
            else:
                self.restart_run(images, energies=Es)
