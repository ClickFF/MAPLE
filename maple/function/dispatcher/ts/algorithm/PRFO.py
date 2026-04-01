# -*- coding: utf-8 -*-
"""
PRFO (Partitioned Rational Function Optimization) implementation for TS search.
Features:
- Dual-shift PRFO with trust region (RS-PRFO)
- Mass-weighted coordinates for step computation
- Trust radius adaptation based on model agreement
- Mode-following for transition state optimization
"""
from __future__ import annotations
import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple, List

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

def write_xyz(filename: str, atoms: Atoms, energy: Optional[float] = None, 
              iteration: Optional[int] = None):
    """
    Write a single geometry to XYZ file.
    
    Parameters
    ----------
    filename : str
        Output file path
    atoms : Atoms
        ASE Atoms object
    energy : float, optional
        Energy value to include in comment line
    iteration : int, optional
        Iteration number to include in comment line
    """
    pos = to_numpy_f64(atoms.get_positions())
    symbols = atoms.get_chemical_symbols()
    
    with open(filename, "w") as f:
        f.write(f"{len(symbols)}\n")
        
        # Build comment line
        comment_parts = []
        if iteration is not None:
            comment_parts.append(f"Iteration {iteration}")
        if energy is not None:
            comment_parts.append(f"Energy = {energy:.10f}")
        
        if comment_parts:
            f.write("  ".join(comment_parts) + "\n")
        else:
            f.write("TS optimization\n")
        
        for s, (x, y, z) in zip(symbols, pos):
            f.write(f"{s:2s} {x: .10f} {y: .10f} {z: .10f}\n")

def append_xyz_trajectory(filename: str, atoms: Atoms, energy: Optional[float] = None,
                         iteration: int = 0):
    """
    Append a geometry to XYZ trajectory file.
    
    Parameters
    ----------
    filename : str
        Output trajectory file path
    atoms : Atoms
        ASE Atoms object
    energy : float, optional
        Energy value to include in comment line
    iteration : int
        Iteration number
    """
    # Remove file if first iteration
    if iteration == 0 and os.path.exists(filename):
        os.remove(filename)
    
    pos = to_numpy_f64(atoms.get_positions())
    symbols = atoms.get_chemical_symbols()
    
    with open(filename, "a") as f:
        f.write(f"{len(symbols)}\n")
        
        if energy is not None:
            f.write(f"Iteration {iteration}  Energy = {energy:.10f}\n")
        else:
            f.write(f"Iteration {iteration}\n")
        
        for s, (x, y, z) in zip(symbols, pos):
            f.write(f"{s:2s} {x: .10f} {y: .10f} {z: .10f}\n")

# =============================================================================
# ------------------------------- PRFO Core -----------------------------------
# =============================================================================

def prfo_step(H, g, is_ts=False, target_mode=None, trust_radius=0.2,
              evals_eps=1e-10, mu_margin=1e-8, max_bisect_it=60,
              pre_eig=None):
    """
    Dual-shift PRFO trust-region step in the SAME coordinates as H,g (e.g., MW coords).

    Parameters
    ----------
    H : (n,n) array-like
        Hessian matrix
    g : (n,) array-like
        Gradient vector (NOT forces)
    is_ts : bool
        TS mode if True (enables partition into uphill/downhill subspaces)
    target_mode : int or None
        Uphill mode index in TS mode (defaults to most negative)
    trust_radius : float
        Trust radius bound measured in coordinates of H,g
    evals_eps : float
        Eigenvalue regularization threshold
    mu_margin : float
        Safety margin for bisection boundary
    max_bisect_it : int
        Maximum bisection iterations
    pre_eig : optional tuple
        Pre-computed eigendecomposition (w, V, g_proj) to reuse

    Returns
    -------
    s : (n,) array
        Step in the SAME coordinates as H,g
    """
    H = to_numpy_f64(H)
    if H.ndim == 3 and H.shape[0] == 1:
        H = H[0]
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError(f"H must be square 2D, got shape={H.shape}")
    n = H.shape[0]

    g = vec1d(g, n)

    # Eigendecomposition (or reuse)
    if pre_eig is not None:
        w, V, gp = pre_eig
        w = vec1d(w, n)
        V = to_numpy_f64(V)
        gp = vec1d(gp, n)
        if V.shape != (n, n):
            raise ValueError("pre_eig V has wrong shape.")
    else:
        w, V = np.linalg.eigh(H)
        gp = V.T @ g

    # Gentle regularization of tiny eigenvalues (keep sign if nonzero)
    tiny = (np.abs(w) < evals_eps)
    w = np.where(tiny & (w == 0.0), evals_eps, w)
    w = np.where(tiny & (w != 0.0), np.sign(w) * evals_eps, w)

    # Partition into uphill (minus set) and downhill (plus set)
    if is_ts:
        if target_mode is None:
            neg_idx = int(np.argmin(w))
            if w[neg_idx] < -1e-6:
                j = neg_idx
            else:
                j = int(np.argmax(np.abs(gp)))
        else:
            j = int(target_mode)
        minus_idx = np.array([j], dtype=int)
        plus_mask = np.ones(n, dtype=bool)
        plus_mask[minus_idx] = False
        plus_idx = np.where(plus_mask)[0]
    else:
        minus_idx = np.array([], dtype=int)
        plus_idx = np.arange(n, dtype=int)

    # Helper to compute unconstrained step & norm^2 for a subspace with sigma flip
    def unconstrained_component(idx, sigma_sign):
        """
        Compute unconstrained step components (μ=0) in eigen-basis.
        
        Parameters
        ----------
        idx : array of int
            Indices into eigen-basis for this subspace
        sigma_sign : float
            +1 for downhill, -1 for uphill (signature flip)
        
        Returns
        -------
        s_unc : array
            Unconstrained step components in eigen-basis on idx
        norm2_unc : float
            Squared norm of unconstrained step on idx
        w_tilde : array
            Flipped curvatures for bisection use
        num : array
            Flipped gradient components for bisection use
        """
        if idx.size == 0:
            return np.zeros(0, dtype=np.float64), 0.0, np.zeros(0), np.zeros(0)
        w_sub = w[idx]
        gp_sub = gp[idx]
        w_tilde = sigma_sign * w_sub
        num = sigma_sign * gp_sub
        denom0 = np.where(np.abs(w_tilde) < evals_eps, 
                         np.sign(w_tilde) * evals_eps, w_tilde)
        s_unc = -num / denom0
        return s_unc, float(np.dot(s_unc, s_unc)), w_tilde, num

    # Uphill (minus) uses sigma = -1; Downhill (plus) uses sigma = +1
    s_unc_minus, norm2_unc_minus, wtil_minus, num_minus = \
        unconstrained_component(minus_idx, -1.0)
    s_unc_plus, norm2_unc_plus, wtil_plus, num_plus = \
        unconstrained_component(plus_idx, +1.0)

    # Total unconstrained norm^2 after partition
    total_unc = norm2_unc_minus + norm2_unc_plus
    R2 = trust_radius * trust_radius

    # Edge case: if total_unc already within R^2
    if total_unc <= R2:
        s_p = np.zeros(n, dtype=np.float64)
        if minus_idx.size:
            s_p[minus_idx] = s_unc_minus
        if plus_idx.size:
            s_p[plus_idx] = s_unc_plus
        return V @ s_p

    # Allocate trust radius between two subspaces
    if total_unc > 0.0:
        alpha = norm2_unc_minus / total_unc
    else:
        alpha = 0.5
    alpha_min, alpha_max = 0.05, 0.95
    alpha = float(np.clip(alpha, alpha_min, alpha_max))

    R2_minus = alpha * R2
    R2_plus = (1.0 - alpha) * R2

    # Solve two scalar bisections independently
    def solve_mu_for_norm2(target_norm2, w_tilde, num):
        """
        Find μ such that sum_i (num_i/(w_tilde_i - μ))^2 = target_norm2.
        Domain: μ < min(w_tilde). Monotonically increasing in this interval.
        """
        if num.size == 0:
            return 0.0, np.zeros(0, dtype=np.float64)

        # Unconstrained norm^2 (μ=0)
        denom0 = np.where(np.abs(w_tilde) < evals_eps,
                         np.sign(w_tilde) * evals_eps, w_tilde)
        s_unc = -num / denom0
        norm2_unc = float(np.dot(s_unc, s_unc))
        if norm2_unc <= target_norm2:
            return 0.0, s_unc

        # Bisection for μ
        def F(mu):
            denom = w_tilde - mu
            denom = np.where(np.abs(denom) < evals_eps,
                           np.sign(denom) * evals_eps, denom)
            return np.sum((num / denom) ** 2)

        wt_min = float(np.min(w_tilde))
        b = wt_min - mu_margin
        Fb = F(b)
        if (not np.isfinite(Fb)) or (Fb > 1e300):
            b = wt_min - 1e-4
            Fb = F(b)

        # Find a < b with F(a) < target_norm2
        a = b - 1.0
        Fa = F(a)
        it = 0
        while Fa > target_norm2 and it < 60:
            a -= max(1.0, abs(a) * 0.5)
            Fa = F(a)
            it += 1

        lo, hi = a, b
        for _ in range(max_bisect_it):
            mid = 0.5 * (lo + hi)
            Fm = F(mid)
            if Fm > target_norm2:
                hi = mid
            else:
                lo = mid
            if abs(Fm - target_norm2) <= 1e-12 * max(1.0, target_norm2) or \
               abs(hi - lo) < 1e-12:
                break
        mu_star = 0.5 * (lo + hi)

        denom = w_tilde - mu_star
        denom = np.where(np.abs(denom) < evals_eps,
                        np.sign(denom) * evals_eps, denom)
        s_part = -num / denom
        return mu_star, s_part

    # Solve for uphill and downhill parts
    mu_minus, s_part_minus = solve_mu_for_norm2(R2_minus, wtil_minus, num_minus)
    mu_plus, s_part_plus = solve_mu_for_norm2(R2_plus, wtil_plus, num_plus)

    # Assemble full eigen-basis step and rotate back
    s_p = np.zeros(n, dtype=np.float64)
    if minus_idx.size:
        s_p[minus_idx] = s_part_minus
    if plus_idx.size:
        s_p[plus_idx] = s_part_plus

    return V @ s_p

def calculate_Hessian(atoms: Atoms):
    """
    Calculate Hessian matrix from calculator.
    
    Parameters
    ----------
    atoms : Atoms
        ASE Atoms object with calculator attached
    
    Returns
    -------
    H : array
        Hessian matrix in float64
    """
    calc = atoms.calc
    H = calc.get_hessian(atoms)
    return to_numpy_f64(H)

# =============================================================================
# ------------------------------- PRFO Parameters -----------------------------
# =============================================================================

@dataclass
class PRFOParams:
    """Parameters for PRFO transition state search."""
    
    # Optimization control
    max_iter: int = 256                    # Maximum optimization iterations
    
    # Trust region parameters
    trust_radius: float = 0.2              # Initial trust radius (in MW coords)
    trust_min: float = 1e-3                # Minimum trust radius
    trust_max: float = 1.0                 # Maximum trust radius
    
    # Trust region adaptation thresholds
    eta_shrink: float = 0.75               # If rho < eta_shrink -> reject & shrink
    eta_expand: float = 1.75               # If rho > eta_expand and on boundary -> expand
    
    # PRFO step parameters
    evals_eps: float = 1e-10               # Eigenvalue regularization threshold
    mu_margin: float = 1e-8                # Safety margin for bisection
    max_bisect_it: int = 60                # Maximum bisection iterations
    
    # Convergence thresholds (should be set from atoms object)
    f_max_th: float = 9.5e-3               # Maximum force threshold (Eh/Angstrom)
    f_rms_th: float = 5e-3                 # RMS force threshold (Eh/Angstrom)
    dp_max_th: float = 1.8e-3              # Maximum displacement threshold (Angstrom)
    dp_rms_th: float = 1.2e-3              # RMS displacement threshold (Angstrom)

# =============================================================================
# ------------------------------- PRFO Class ----------------------------------
# =============================================================================

class PRFO(JobABC):
    """
    Transition state search using Dual-Shift PRFO with trust region adaptation.

    The optimization is performed in mass-weighted coordinates for the step
    computation and trust-region enforcement, while geometry updates are done
    in Cartesian coordinates.
    """

    def __init__(self,
                 output: str,
                 atoms: Atoms,
                 paras: Optional[dict] = None):
        super().__init__(output)
        self.atoms = atoms

        # Initialize params from paras dict
        self.params = self._init_params(PRFOParams, paras, ("prfo", "PRFO", "ts"))

        # Override convergence thresholds from atoms if available
        for attr in ('f_max_th', 'f_rms_th', 'dp_max_th', 'dp_rms_th'):
            if hasattr(atoms, attr):
                setattr(self.params, attr, getattr(atoms, attr))

        # Mode tracking
        self.tracked_mode_vec_mw = None
        self.tracked_mode_idx = None
    
    def atoms_to_xyz(self, atoms: Atoms) -> str:
        """Convert Atoms object to XYZ format string."""
        lines = []
        syms = atoms.get_chemical_symbols()
        pos = atoms.get_positions()
        for s, (x, y, z) in zip(syms, pos):
            lines.append(f"{s:<2} {x:14.6f} {y:14.6f} {z:14.6f}")
        return "\n".join(lines) + "\n"
    
    def check_convergence(self, atoms: Atoms) -> bool:
        """
        Check if optimization has converged based on forces and displacements.
        
        Parameters
        ----------
        atoms : Atoms
            Current geometry with convergence metrics attached
        
        Returns
        -------
        bool
            True if all convergence criteria are met
        """
        return (atoms.max_f <= self.params.f_max_th and
                atoms.rms_f <= self.params.f_rms_th and
                atoms.max_dp <= self.params.dp_max_th and
                atoms.rms_dp <= self.params.dp_rms_th)
    
    def log_iteration(self, iteration: int, atoms: Atoms, E: float,
                     model_change: float, actual_change: float,
                     rho: Optional[float], trust_radius: float,
                     norm_mw: float, on_boundary: bool):
        """
        Log detailed information for current iteration.
        
        Parameters
        ----------
        iteration : int
            Current iteration number
        atoms : Atoms
            Current geometry
        E : float
            Current energy
        model_change : float
            Predicted energy change by model
        actual_change : float
            Actual energy change
        rho : float or None
            Model agreement ratio
        trust_radius : float
            Current trust radius
        norm_mw : float
            Step norm in MW coordinates
        on_boundary : bool
            Whether step is on trust radius boundary
        """
        iter_str = f"Iteration: {iteration}"
        info_message = ['\n' + '-' * 70 + '\n', f'{iter_str.center(70)}\n\n']
        
        # Coordinates section
        info_message.append(f'\n{"Coordinates".center(70)}\n')
        info_message.append('-' * 70 + '\n')
        for atom_index, atom in enumerate(atoms):
            element_type = atom.symbol
            coord = atom.position
            info_message.append(
                f"{atom_index:<4} {element_type:<2} "
                f"{coord[0]:>20.4f} {coord[1]:>20.4f} {coord[2]:>20.4f}\n"
            )
        
        # Energy and convergence
        info_message.append(
            f"\n\nEnergy:                {E:>12.6f} "
            "Convergence criteria  Is converged \n"
        )
        
        # Force convergence
        if atoms.max_f > self.params.f_max_th:
            info_message.append(
                f"Maximum Force:         {atoms.max_f:>12.6f} "
                f"{self.params.f_max_th:>12.6f}                No\n"
            )
        else:
            info_message.append(
                f"Maximum Force:         {atoms.max_f:>12.6f} "
                f"{self.params.f_max_th:>12.6f}                Yes\n"
            )
        
        if atoms.rms_f > self.params.f_rms_th:
            info_message.append(
                f"RMS Force:             {atoms.rms_f:>12.6f} "
                f"{self.params.f_rms_th:>12.6f}                No\n"
            )
        else:
            info_message.append(
                f"RMS Force:             {atoms.rms_f:>12.6f} "
                f"{self.params.f_rms_th:>12.6f}                Yes\n"
            )
        
        # Displacement convergence
        if atoms.max_dp > self.params.dp_max_th:
            info_message.append(
                f"Maximum Displacement:  {atoms.max_dp:>12.6f} "
                f"{self.params.dp_max_th:>12.6f}                No\n"
            )
        else:
            info_message.append(
                f"Maximum Displacement:  {atoms.max_dp:>12.6f} "
                f"{self.params.dp_max_th:>12.6f}                Yes\n"
            )
        
        if atoms.rms_dp > self.params.dp_rms_th:
            info_message.append(
                f"RMS Displacement:      {atoms.rms_dp:>12.6f} "
                f"{self.params.dp_rms_th:>12.6f}                No\n"
            )
        else:
            info_message.append(
                f"RMS Displacement:      {atoms.rms_dp:>12.6f} "
                f"{self.params.dp_rms_th:>12.6f}                Yes\n"
            )
        
        # Trust region info
        rho_str = f"{rho:.3f}" if rho is not None else "nan"
        info_message.append(
            f"\nModel change: {model_change: .6e}  "
            f"Actual change: {actual_change: .6e}  "
            f"rho: {rho_str}\n"
        )
        info_message.append(
            f"Trust radius (MW): {trust_radius: .6f}  "
            f"Step norm (MW): {norm_mw: .6f}  "
            f"On boundary: {on_boundary}\n"
        )
        
        log_info(info_message, self.output)
    
    def update_mode_tracking(self, w_mw: np.ndarray, V_mw: np.ndarray,
                           gp_mw: np.ndarray) -> int:
        """
        Update mode-following for transition state search.
        
        Parameters
        ----------
        w_mw : array
            Eigenvalues in mass-weighted coordinates
        V_mw : array
            Eigenvectors in mass-weighted coordinates
        gp_mw : array
            Projected gradient in eigen-basis
        
        Returns
        -------
        int
            Index of tracked mode
        """
        if self.tracked_mode_vec_mw is None:
            # Initialize: track most negative mode or largest gradient
            neg_idx = int(np.argmin(w_mw))
            if w_mw[neg_idx] < -1e-6:
                self.tracked_mode_idx = neg_idx
            else:
                self.tracked_mode_idx = int(np.argmax(np.abs(gp_mw)))
            self.tracked_mode_vec_mw = V_mw[:, self.tracked_mode_idx].copy()
        else:
            # Follow mode with maximum overlap
            overlaps = np.abs(V_mw.T @ self.tracked_mode_vec_mw)
            self.tracked_mode_idx = int(np.argmax(overlaps))
            
            # Keep consistent sign to avoid flips
            sign_align = np.sign(np.dot(V_mw[:, self.tracked_mode_idx],
                                       self.tracked_mode_vec_mw))
            if sign_align == 0.0:
                sign_align = 1.0
            self.tracked_mode_vec_mw = \
                V_mw[:, self.tracked_mode_idx] * sign_align
        
        return self.tracked_mode_idx
    
    def run(self) -> Atoms:
        """
        Run PRFO transition state optimization.
        
        Returns
        -------
        Atoms
            Optimized (or final) geometry
        """
        sys.setrecursionlimit(1000)
        
        atoms = self.atoms
        trust_radius = self.params.trust_radius
        
        converged = False
        iteration = 0
        
        # Setup trajectory file
        base, _ = os.path.splitext(self.output)
        traj_file = base + "_prfo_traj.xyz"
        ts_file = base + "_prfo_ts.xyz"
        
        # Log header
        info_message = [
            f"\nStarting Transition State Search (TS) with RS-PRFO...\n",
            f"Trust radius adaptation: eta_shrink={self.params.eta_shrink}, "
            f"eta_expand={self.params.eta_expand}\n",
            f"Convergence thresholds: "
            f"f_max={self.params.f_max_th:.6f}, "
            f"f_rms={self.params.f_rms_th:.6f}, "
            f"dp_max={self.params.dp_max_th:.6f}, "
            f"dp_rms={self.params.dp_rms_th:.6f}\n"
        ]
        log_info(info_message, self.output)
        
        # Main optimization loop
        while iteration < self.params.max_iter:
            # Get current geometry and energy/forces
            X = atoms.get_positions().reshape(-1, 3)
            E_old = to_numpy_f64(
                atoms.get_potential_energy(force_consistent=True)
            )
            
            F_cart = to_numpy_f64(atoms.get_forces())
            g_cart = vec1d(-F_cart)
            
            # Get Hessian in Cartesian
            H_cart = to_numpy_f64(calculate_Hessian(atoms))
            if H_cart.ndim == 3 and H_cart.shape[0] == 1:
                H_cart = H_cart[0]
            if H_cart.ndim != 2 or H_cart.shape[0] != H_cart.shape[1]:
                raise ValueError(f"Hessian must be square, got {H_cart.shape}")
            
            n3 = H_cart.shape[0]
            if g_cart.size != n3:
                raise ValueError(
                    f"Gradient size {g_cart.size} != Hessian dim {n3}"
                )
            
            # Eigenvalues for logging
            eigvals_log, _ = np.linalg.eigh(H_cart)
            eigvals_log = np.real(eigvals_log).astype(np.float64).squeeze()
            
            # Mass-weighting
            masses = to_numpy_f64(atoms.get_masses())
            masses = np.where(masses > 0.0, masses, 1.0)
            D = vec1d(1.0 / np.sqrt(np.repeat(masses, 3)), n3)
            
            g_mw = vec1d(D * g_cart, n3)
            H_mw = (D[:, None] * H_cart) * D[None, :]
            
            # Eigendecomposition in MW coords
            w_mw, V_mw = np.linalg.eigh(H_mw)
            gp_mw = vec1d(V_mw.T @ g_mw, n3)
            
            # Regularize tiny eigenvalues
            tiny = (np.abs(w_mw) < 1e-10)
            w_mw = np.where(tiny & (w_mw == 0.0), 1e-10, w_mw)
            w_mw = np.where(tiny & (w_mw != 0.0),
                          np.sign(w_mw) * 1e-10, w_mw)
            
            # Update mode tracking
            tracked_mode_idx = self.update_mode_tracking(w_mw, V_mw, gp_mw)
            
            # RS loop: try step with current trust_radius, accept/reject by rho
            accepted = False
            max_attempts = 8
            attempts = 0
            
            while not accepted and attempts < max_attempts:
                attempts += 1
                
                # Compute PRFO step in MW coords
                s_mw = prfo_step(
                    H=H_mw,
                    g=g_mw,
                    is_ts=True,
                    target_mode=tracked_mode_idx,
                    trust_radius=trust_radius,
                    evals_eps=self.params.evals_eps,
                    mu_margin=self.params.mu_margin,
                    max_bisect_it=self.params.max_bisect_it,
                    pre_eig=(w_mw, V_mw, gp_mw)
                )
                norm_mw = float(np.linalg.norm(s_mw))
                on_boundary = (abs(norm_mw - trust_radius) <=
                             1e-6 * max(1.0, trust_radius))
                
                # Back to Cartesian
                s_cart = vec1d(D * s_mw, n3)
                
                # Model agreement for trust-radius adaptation
                Hs = H_cart @ s_cart
                model_change = float(g_cart.dot(s_cart) + 0.5 * s_cart.dot(Hs))
                
                # Trial geometry
                X_new = X.reshape(-1, 3) + s_cart.reshape(-1, 3)
                atoms.set_positions(X_new)
                
                # New energy
                E_new = to_numpy_f64(
                    atoms.get_potential_energy(force_consistent=True)
                )
                
                actual_change = float(E_new - E_old)
                rho = None
                if abs(model_change) > 1e-16:
                    rho = actual_change / model_change
                
                # Accept/reject decision
                bad_model = (rho is None or rho < self.params.eta_shrink or
                           not np.isfinite(rho))
                
                if bad_model and trust_radius > self.params.trust_min * (1.0 + 1e-12):
                    # Reject: rollback geometry, shrink radius, retry
                    atoms.set_positions(X)
                    trust_radius = max(self.params.trust_min,
                                     0.5 * trust_radius)
                    continue
                else:
                    # Accept the step
                    accepted = True
                    
                    # Radius adaptation after acceptance
                    if (rho is not None and rho > self.params.eta_expand and
                        on_boundary):
                        trust_radius = min(self.params.trust_max,
                                         2.0 * trust_radius)
                    
                    # Get new forces for convergence check
                    F_new = to_numpy_f64(atoms.get_forces())
                    
                    # Compute convergence metrics (per DOF RMS)
                    dof = s_cart.size
                    atoms.max_dp = abs(s_cart).max()
                    atoms.rms_dp = np.sqrt((s_cart**2).sum() / dof)
                    atoms.max_f = abs(F_new).max()
                    atoms.rms_f = np.sqrt((F_new**2).sum() / dof)
                    
                    # Log iteration
                    self.log_iteration(iteration + 1, atoms, E_new,
                                     model_change, actual_change, rho,
                                     trust_radius, norm_mw, on_boundary)
                    
                    # Write to trajectory
                    append_xyz_trajectory(traj_file, atoms, energy=E_new,
                                        iteration=iteration + 1)
                    
                    # Check convergence
                    if self.check_convergence(atoms):
                        converged = True
                        info_message = [
                            '\n\n' + '-' * 70 + '\n',
                            f'{"Normal Termination".center(70)}\n\n'
                        ]
                        log_info(info_message, self.output)
                        
                        # Write final TS structure
                        write_xyz(ts_file, atoms, energy=E_new,
                                iteration=iteration + 1)
                        
                        return atoms
            
            iteration += 1
        
        # Maximum iterations reached
        log_info([f'\n\n{"Maximum Iterations Reached".center(70)}\n\n'],
                self.output)
        
        # Write final structure even if not converged
        E_final = atoms.get_potential_energy(force_consistent=True)
        write_xyz(ts_file, atoms, energy=E_final, iteration=iteration)
        
        log_info([
            f"\nWrote trajectory to: {traj_file}\n",
            f"Wrote final TS structure to: {ts_file}\n"
        ], self.output)
        
        return atoms