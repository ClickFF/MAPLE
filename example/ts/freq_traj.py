# -*- coding: utf-8 -*-
import numpy as np
import sys
from ase import Atoms
from scipy.optimize import brentq

from .logger import *

# =========================================
# ============  Type Utilities  ===========
# =========================================
def to_numpy_f64(x):
    """Convert input (numpy/torch/list/scalar) to float64 numpy array or float."""
    if isinstance(x, np.ndarray):
        return x.astype(np.float64, copy=False)
    # Torch tensor?
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

# =========================================
# ============   PRFO Step   ==============
# =========================================
def prfo_step(H, g, is_ts=False, target_mode=None, trust_radius=0.2,
              evals_eps=1e-10, mu_margin=1e-8, max_bisect_it=60,
              pre_eig=None):
    """
    Dual-shift PRFO trust-region step in the SAME coordinates as H,g (e.g., MW coords).

    H : (n,n) array-like
    g : (n,)   gradient (NOT forces)
    is_ts : bool, TS mode if True (enables partition into uphill/downhill subspaces)
    target_mode : int or None, uphill mode index in TS mode (defaults to most negative)
    trust_radius : float, ||s|| bound measured in coordinates of H,g
    pre_eig : optional (w, V, g_proj) to reuse eigendecomp in same coords

    Returns:
        s : (n,) step in the SAME coordinates as H,g
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

    # ---------- Partition into uphill (minus set) and downhill (plus set) ----------
    # TS: uphill subspace is the target mode (usually the unique negative eigenmode).
    # Non-TS: everything is downhill; we fall back to single-shift on the plus set.
    if is_ts:
        if target_mode is None:
            neg_idx = int(np.argmin(w))
            if w[neg_idx] < -1e-6:
                j = neg_idx
            else:
                j = int(np.argmax(np.abs(gp)))
        else:
            j = int(target_mode)
        minus_idx = np.array([j], dtype=int)               # uphill subspace (dim 1 typical)
        plus_mask = np.ones(n, dtype=bool)
        plus_mask[minus_idx] = False
        plus_idx = np.where(plus_mask)[0]                   # downhill subspace
    else:
        minus_idx = np.array([], dtype=int)
        plus_idx  = np.arange(n, dtype=int)

    # Helper to compute unconstrained step & norm^2 for a subspace with sigma flip
    def unconstrained_component(idx, sigma_sign):
        """
        idx: integer indices into eigen-basis for this subspace
        sigma_sign: +1 for downhill, -1 for uphill (signature flip)
        Returns:
            s_p_unc   : unconstrained step components (μ=0) in eigen-basis on idx
            norm2_unc : squared norm of unconstrained step on idx
            w_tilde   : flipped curvatures for bisection use
            num       : flipped gradient components for bisection use
        """
        if idx.size == 0:
            return np.zeros(0, dtype=np.float64), 0.0, np.zeros(0), np.zeros(0)
        w_sub  = w[idx]
        gp_sub = gp[idx]
        w_tilde = sigma_sign * w_sub
        num     = sigma_sign * gp_sub
        # μ = 0 unconstrained on this subspace:
        denom0 = np.where(np.abs(w_tilde) < evals_eps, np.sign(w_tilde) * evals_eps, w_tilde)
        s_unc = - num / denom0
        return s_unc, float(np.dot(s_unc, s_unc)), w_tilde, num

    # Uphill (minus) uses sigma = -1 ; Downhill (plus) uses sigma = +1
    s_unc_minus, norm2_unc_minus, wtil_minus, num_minus = unconstrained_component(minus_idx, -1.0)
    s_unc_plus,  norm2_unc_plus,  wtil_plus,  num_plus  = unconstrained_component(plus_idx,  +1.0)

    # Total unconstrained norm^2 after partition (for α weighting)
    total_unc = norm2_unc_minus + norm2_unc_plus
    R2 = trust_radius * trust_radius

    # Edge cases: if one subspace empty or total_unc already within R^2
    if total_unc <= R2:
        # No need to shift (μ_up=μ_down=0); just assemble step and return.
        s_p = np.zeros(n, dtype=np.float64)
        if minus_idx.size:
            s_p[minus_idx] = s_unc_minus
        if plus_idx.size:
            s_p[plus_idx]  = s_unc_plus
        return V @ s_p

    # ---------- Allocate trust radius between two subspaces ----------
    # Heuristic yet robust: allocate by unconstrained contributions (like "who wants more gets more"),
    # with caps to avoid starving either side.
    if total_unc > 0.0:
        alpha = norm2_unc_minus / total_unc   # portion of R^2 to uphill subspace
    else:
        alpha = 0.5
    alpha_min, alpha_max = 0.05, 0.95
    alpha = float(np.clip(alpha, alpha_min, alpha_max))

    R2_minus = alpha * R2
    R2_plus  = (1.0 - alpha) * R2

    # ---------- Solve two scalar bisections independently ----------
    def solve_mu_for_norm2(target_norm2, w_tilde, num):
        """
        Find μ such that sum_i (num_i/(w_tilde_i - μ))^2 = target_norm2
        Domain: μ < min(w_tilde). Monotonically increasing in this interval.
        If unconstrained norm^2 <= target_norm2, return μ=0 (no shift needed).
        """
        if num.size == 0:
            return 0.0, np.zeros(0, dtype=np.float64)

        # unconstrained norm^2 (μ=0)
        denom0 = np.where(np.abs(w_tilde) < evals_eps, np.sign(w_tilde) * evals_eps, w_tilde)
        s_unc = - num / denom0
        norm2_unc = float(np.dot(s_unc, s_unc))
        if norm2_unc <= target_norm2:
            # No need to shift; directly return the unconstrained component
            return 0.0, s_unc

        # Otherwise bisection for μ
        def F(mu):
            denom = w_tilde - mu
            denom = np.where(np.abs(denom) < evals_eps, np.sign(denom) * evals_eps, denom)
            return np.sum((num / denom) ** 2)

        wt_min = float(np.min(w_tilde))
        b = wt_min - mu_margin
        Fb = F(b)
        if (not np.isfinite(Fb)) or (Fb > 1e300):
            b = wt_min - 1e-4
            Fb = F(b)

        # find a < b with F(a) < target_norm2
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
            if abs(Fm - target_norm2) <= 1e-12 * max(1.0, target_norm2) or abs(hi - lo) < 1e-12:
                break
        mu_star = 0.5 * (lo + hi)

        denom = w_tilde - mu_star
        denom = np.where(np.abs(denom) < evals_eps, np.sign(denom) * evals_eps, denom)
        s_part = - num / denom
        return mu_star, s_part

    # Solve for uphill and downhill parts
    mu_minus, s_part_minus = solve_mu_for_norm2(R2_minus, wtil_minus, num_minus)
    mu_plus,  s_part_plus  = solve_mu_for_norm2(R2_plus,  wtil_plus,  num_plus)

    # Assemble full eigen-basis step and rotate back
    s_p = np.zeros(n, dtype=np.float64)
    if minus_idx.size:
        s_p[minus_idx] = s_part_minus
    if plus_idx.size:
        s_p[plus_idx]  = s_part_plus

    return V @ s_p

# =========================================
# ============   TS Optimizer   ===========
# =========================================
def RFO(atoms: Atoms, output):
    """
    TS search using Dual-Shift PRFO + trust region (RS-PRFO) + mode-following (mass-weighted).
    - Geometry update in Cartesian; step computed & trust-region enforced in MW coords.
    - RS rules: compute rho, accept/reject step; adapt trust radius accordingly.

    Returns: (iteration, converged: bool)
    """
    sys.setrecursionlimit(1000)

    max_iter     = 256
    trust_radius = 0.2
    trust_min    = 1e-3
    trust_max    = 1.0

    # Trust-region acceptance thresholds (classical)
    eta_shrink = 0.25   # if rho < eta_shrink -> reject & shrink
    eta_expand = 0.75   # if rho > eta_expand and on boundary -> expand

    tracked_mode_vec_mw = None   # eigenvector in MW coords
    tracked_mode_idx    = None

    converged = False
    iteration = 0  # count only accepted steps

    while iteration < max_iter:
        iter_str = f"Iteration: {iteration+1}"

        # ---- geometry & energy/forces (to numpy) ----
        X = atoms.get_positions().reshape(-1, 3)
        E_old = to_numpy_f64(atoms.get_potential_energy(force_consistent=True))

        F_cart = to_numpy_f64(atoms.get_forces())      # (N,3) or similar
        g_cart = vec1d(-F_cart)                        # flatten (3N,)

        # ---- Hessian in Cartesian ----
        H_cart = to_numpy_f64(calculate_Hessian(atoms))
        if H_cart.ndim == 3 and H_cart.shape[0] == 1:
            H_cart = H_cart[0]
        if H_cart.ndim != 2 or H_cart.shape[0] != H_cart.shape[1]:
            raise ValueError(f"Hessian must be square, got {H_cart.shape}")

        n3 = H_cart.shape[0]
        if g_cart.size != n3:
            raise ValueError(f"Gradient size {g_cart.size} != Hessian dim {n3}")

        # ---- eigen diag for logging (Cartesian) ----
        eigvals_for_log, _ = np.linalg.eigh(H_cart)
        eigvals_for_log = np.real(eigvals_for_log).astype(np.float64).squeeze()
        lowest = float(np.min(eigvals_for_log))
        second_lowest = float(np.sort(eigvals_for_log)[1]) if eigvals_for_log.size >= 2 else np.nan

        # -------- mass-weighting --------
        masses = to_numpy_f64(atoms.get_masses())
        masses = np.where(masses > 0.0, masses, 1.0)
        D = vec1d(1.0 / np.sqrt(np.repeat(masses, 3)), n3)  # (3N,)

        g_mw = vec1d(D * g_cart, n3)
        H_mw = (D[:, None] * H_cart) * D[None, :]

        # ---- eigendecomposition in MW coords ----
        w_mw, V_mw = np.linalg.eigh(H_mw)
        gp_mw = vec1d(V_mw.T @ g_mw, n3)

        # regularize tiny evals
        tiny = (np.abs(w_mw) < 1e-10)
        w_mw = np.where(tiny & (w_mw == 0.0), 1e-10, w_mw)
        w_mw = np.where(tiny & (w_mw != 0.0), np.sign(w_mw) * 1e-10, w_mw)

        # ---- mode-following ----
        if tracked_mode_vec_mw is None:
            neg_idx = int(np.argmin(w_mw))
            if w_mw[neg_idx] < -1e-6:
                tracked_mode_idx = neg_idx
            else:
                tracked_mode_idx = int(np.argmax(np.abs(gp_mw)))
            tracked_mode_vec_mw = V_mw[:, tracked_mode_idx].copy()
        else:
            overlaps = np.abs(V_mw.T @ tracked_mode_vec_mw)
            tracked_mode_idx = int(np.argmax(overlaps))
            # keep a consistent sign to avoid flips
            sign_align = np.sign(np.dot(V_mw[:, tracked_mode_idx], tracked_mode_vec_mw))
            if sign_align == 0.0:
                sign_align = 1.0
            tracked_mode_vec_mw = V_mw[:, tracked_mode_idx] * sign_align

        # ========== RS loop: try step with current trust_radius, accept/reject by rho ==========
        accepted = False
        max_attempts = 8  # avoid infinite retries in pathological cases
        attempts = 0

        while not accepted and attempts < max_attempts:
            attempts += 1

            # ---- Dual-Shift PRFO step in MW coords (trust enforced INSIDE) ----
            s_mw = prfo_step(
                H=H_mw,
                g=g_mw,
                is_ts=True,
                target_mode=tracked_mode_idx,
                trust_radius=trust_radius,
                evals_eps=1e-10,
                mu_margin=1e-8,
                max_bisect_it=60,
                pre_eig=(w_mw, V_mw, gp_mw)
            )
            norm_mw = float(np.linalg.norm(s_mw))
            on_boundary = (abs(norm_mw - trust_radius) <= 1e-6 * max(1.0, trust_radius))

            # ---- back to Cartesian ----
            s_cart = vec1d(D * s_mw, n3)

            # ---- model agreement for trust-radius adaptation ----
            Hs = H_cart @ s_cart
            model_change = float(g_cart.dot(s_cart) + 0.5 * s_cart.dot(Hs))

            # ---- trial geometry ----
            X_new = (X.reshape(-1, 3) + s_cart.reshape(-1, 3))
            atoms.set_positions(X_new)

            # energies/forces at new geometry
            E_new = to_numpy_f64(atoms.get_potential_energy(force_consistent=True))

            actual_change = float(E_new - E_old)
            rho = None
            if abs(model_change) > 1e-16:
                rho = actual_change / model_change

            # ---- accept / reject decision ----
            bad_model = (rho is None) or (rho < eta_shrink) or (not np.isfinite(rho))
            if bad_model and trust_radius > trust_min * (1.0 + 1e-12):
                # Reject: rollback geometry, shrink radius, and retry
                atoms.set_positions(X)  # rollback
                trust_radius = max(trust_min, 0.5 * trust_radius)
                continue  # recompute step at same geometry with smaller radius
            else:
                # Accept the step (even if bad_model but already at trust_min)
                accepted = True

                # radius adaptation after acceptance
                if (rho is not None) and (rho > eta_expand) and on_boundary:
                    trust_radius = min(trust_max, 2.0 * trust_radius)

                # pull new forces for logging & convergence only AFTER accept
                F_new = to_numpy_f64(atoms.get_forces())

                # ====== logging (保持你的格式) ======
                info_message = ['\n' + '-' * 70 + '\n', f'{iter_str.center(70)}\n\n']
                info_message.append(f"lambda1: {'N/A(PRFO)'}, lambda2: {'N/A(PRFO)'}\n")
                info_message.append(f"lowest eigenvalue: {lowest}, second lowest eigenvalue: {second_lowest}\n")

                info_message.append(f'\n{"Coordinates".center(70)}\n')
                info_message.append('-' * 70)
                info_message.append('\n')

                for atom_index, atom in enumerate(atoms):
                    element_type = atom.symbol
                    coord = atom.position
                    info_message.append(f"{atom_index:<4} {element_type:<2} {coord[0]:>20.4f} {coord[1]:>20.4f} {coord[2]:>20.4f}\n")

                info_message.append(f"\n\nEnergy:                {E_new:>12.6f} Convergence criteria  Is converged \n")

                # convergence metrics（每自由度 RMS）
                dof = s_cart.size  # 3N
                atoms.max_dp = abs(s_cart).max()
                atoms.rms_dp = np.sqrt((s_cart**2).sum() / dof)
                atoms.max_f  = abs(F_new).max()
                atoms.rms_f  = np.sqrt((F_new**2).sum() / dof)

                # 力/位移与各自阈值比较（单位保持一致）
                if atoms.max_f > atoms.f_max_th:
                    info_message.append(f"Maximum Force:         {atoms.max_f:>12.6f} {atoms.f_max_th:>12.6f}                No\n")
                else:
                    info_message.append(f"Maximum Force:         {atoms.max_f:>12.6f} {atoms.f_max_th:>12.6f}                Yes\n")

                if atoms.rms_f > atoms.f_rms_th:
                    info_message.append(f"RMS Force:             {atoms.rms_f:>12.6f} {atoms.f_rms_th:>12.6f}                No\n")
                else:
                    info_message.append(f"RMS Force:             {atoms.rms_f:>12.6f} {atoms.f_rms_th:>12.6f}                Yes\n")

                if atoms.max_dp > atoms.dp_max_th:
                    info_message.append(f"Maximum Displacement:  {atoms.max_dp:>12.6f} {atoms.dp_max_th:>12.6f}                No\n")
                else:
                    info_message.append(f"Maximum Displacement:  {atoms.max_dp:>12.6f} {atoms.dp_max_th:>12.6f}                Yes\n")

                if atoms.rms_dp > atoms.dp_rms_th:
                    info_message.append(f"RMS Displacement:      {atoms.rms_dp:>12.6f} {atoms.dp_rms_th:>12.6f}                No\n")
                else:
                    info_message.append(f"RMS Displacement:      {atoms.rms_dp:>12.6f} {atoms.dp_rms_th:>12.6f}                Yes\n")

                # 记录模型一致性与信赖域信息
                info_message.append(f"\nModel change: {model_change: .6e}  Actual change: {actual_change: .6e}  rho: {rho if rho is not None else float('nan'): .3f}\n")
                info_message.append(f"Trust radius (MW): {trust_radius: .6f}  Step norm (MW): {norm_mw: .6f}  On boundary: {on_boundary}\n")

                log_info(info_message, output)

                # convergence check
                if (atoms.max_f <= atoms.f_max_th and atoms.rms_f <= atoms.f_rms_th and
                    atoms.max_dp <= atoms.dp_max_th and atoms.rms_dp <= atoms.dp_rms_th):
                    converged = True
                    info_message = ['\n\n' + '-' * 70 + '\n', f'{"Normal Termination".center(70)}\n\n']
                    log_info(info_message, output)
                    return iteration + 1, converged

        # 如果连续多次拒绝直到 trust_min 仍然 bad_model，我们已经“接受最小半径步”并继续
        iteration += 1

    print("未能在最大迭代次数内收敛")
    return iteration, False

# =========================================
# ============  Helpers (kept)  ===========
# =========================================
def calculate_Hessian(atoms: Atoms):
    calc = atoms.get_calculator()
    H = calc.get_hessian(atoms)
    return to_numpy_f64(H)

def find_nth_smallest_index(eigvals, n):
    sorted_indices = np.argsort(eigvals)
    return sorted_indices[n-1]

# 以下两个在当前实现中未使用，保留接口不变
def calculate_lambda1(g1, h1, epsilon=1e-6, max_iter=256, alpha=0.2):
    def f1(lambda1):
        return g1**2 / (lambda1 - h1) - lambda1
    def f1_prime(lambda1):
        return -g1**2 / (lambda1 - h1)**2 - 1
    lambda1 = h1 / 2 + 1e-6
    for _ in range(max_iter):
        f_val = f1(lambda1)
        f_prime_val = f1_prime(lambda1)
        if abs(f_prime_val) < epsilon:
            break
        lambda1_new = lambda1 - alpha * (f_val / f_prime_val)
        if abs(lambda1_new - lambda1) < epsilon:
            return lambda1_new
        lambda1 = lambda1_new
    return lambda1

def calculate_lambda2(g_vals, h_vals, h2, epsilon=1e-6):
    g_vals = to_numpy_f64(g_vals)
    h_vals = to_numpy_f64(h_vals)
    h2     = float(h2)
    def f2(lambda2):
        numerator = np.sum(g_vals**2 / (lambda2 - h_vals))
        return numerator - lambda2
    upper = h2 - epsilon
    lower = h2 - 100.0
    lambda2 = brentq(f2, lower, upper)
    return lambda2

def scale_down(step, max_step_size):
    # kept for backward compatibility; not used in RS-PRFO outer loop now
    v = vec1d(step)
    nrm = np.linalg.norm(v)
    if nrm == 0.0:
        return v
    return v * (max_step_size / nrm)
