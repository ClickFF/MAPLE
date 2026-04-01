# -*- coding: utf-8 -*-
"""
DS-AFIR (Double-Sphere Artificial Force Induced Reaction) implementation:
- Finds reaction path between given reactant and product structures
- Automatically handles multi-step reactions (multiple TSs and intermediates)
- Uses LQA (Local Quadratic Approximation) for path integration
- Integrates with PRFO for TS refinement

Reference:
    Maeda et al. J. Comput. Chem. 2018, 39, 233-251.
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from ase import Atoms

from .logger import log_info
from ...jobABC import JobABC


def to_numpy_f64(x):
    if isinstance(x, np.ndarray):
        return x.astype(np.float64, copy=False)
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().astype(np.float64, copy=False)
    except:
        pass
    if np.isscalar(x):
        return float(x)
    return np.asarray(x, dtype=np.float64)


def vec1d(x, n_expected=None):
    v = to_numpy_f64(x).reshape(-1)
    if n_expected is not None and v.size != n_expected:
        raise ValueError(f"Expected size {n_expected}, got {v.size}")
    return v


def kabsch_align(P, Q):
    P, Q = np.asarray(P, dtype=np.float64), np.asarray(Q, dtype=np.float64)
    Pc, Qc = P.mean(axis=0), Q.mean(axis=0)
    P0, Q0 = P - Pc, Q - Qc
    C = Q0.T @ P0
    V, S, Wt = np.linalg.svd(C)
    if np.linalg.det(V @ Wt) < 0.0:
        V[:, -1] *= -1.0
    R = V @ Wt
    t = Pc - Qc @ R
    Q_aligned = Q @ R + t
    rmsd = float(np.sqrt(((P - Q_aligned)**2).sum() / P.shape[0]))
    return Q_aligned, rmsd, R, t


def write_multi_xyz(filename, atoms_list, energies=None, comments=None, mode="w"):
    with open(filename, mode) as f:
        for i, atoms in enumerate(atoms_list):
            pos = to_numpy_f64(atoms.get_positions())
            symbols = atoms.get_chemical_symbols()
            f.write(f"{len(symbols)}\n")
            if comments and i < len(comments):
                f.write(f"{comments[i]}\n")
            elif energies and i < len(energies):
                f.write(f"Image {i}  Energy = {energies[i]:.10f}\n")
            else:
                f.write(f"Image {i}\n")
            for s, (x, y, z) in zip(symbols, pos):
                f.write(f"{s:2s} {x: .10f} {y: .10f} {z: .10f}\n")


@dataclass
class DSAFIRParams:
    """Parameters for DS-AFIR calculation."""
    delta: float = 100.0
    convergence_threshold: float = 0.12
    max_steps: int = 2000
    step_size: float = 0.01
    use_lup: bool = True
    lup_iterations_coarse: int = 5
    lup_iterations_fine: int = 3
    lup_interval_coarse: float = 1.0
    lup_interval_fine: float = 0.5
    lup_step_size: float = 0.01
    refine_ts: bool = True
    ts_max_iter: int = 200
    do_irc: bool = False
    irc_step_size: float = 0.1
    irc_max_steps: int = 100
    f_max_th: float = 4.5e-4
    f_rms_th: float = 3.0e-4
    dp_max_th: float = 1.8e-3
    dp_rms_th: float = 1.2e-3


def compute_Y(q_i, p_j, q_0, is_initial=False):
    """Compute weight parameter Y for DS-AFIR."""
    if is_initial:
        return 1.0
    q_i, p_j, q_0 = vec1d(q_i), vec1d(p_j), vec1d(q_0)
    dist_from_ref = np.linalg.norm(q_i - q_0)
    if dist_from_ref < 1e-10:
        return 1.0
    dist_to_other = np.linalg.norm(q_i - p_j)
    if dist_to_other < 1e-10:
        return 0.5
    vec_to_other = q_i - p_j
    vec_from_ref = q_i - q_0
    term1 = dist_from_ref / dist_to_other
    cos_angle = np.dot(vec_to_other, vec_from_ref) / (dist_to_other * dist_from_ref)
    Z = term1 + cos_angle
    return Z / (1.0 + Z) if Z > 0 else 0.0


def compute_u(q_i, p_j, q_0, Y, is_initial=False):
    """Compute direction vector u for DS-AFIR force."""
    q_i, p_j, q_0 = vec1d(q_i), vec1d(p_j), vec1d(q_0)
    vec_to_other = q_i - p_j
    dist_to_other = np.linalg.norm(vec_to_other)
    if dist_to_other < 1e-10:
        return np.zeros_like(q_i)
    unit_to_other = vec_to_other / dist_to_other
    if is_initial or np.linalg.norm(q_i - q_0) < 1e-10:
        return Y * unit_to_other
    vec_from_ref = q_i - q_0
    dist_from_ref = np.linalg.norm(vec_from_ref)
    unit_from_ref = vec_from_ref / dist_from_ref
    return Y * unit_to_other - (1.0 - Y) * unit_from_ref


def compute_X(gradient, u, delta):
    """Compute force strength parameter X."""
    gradient, u = vec1d(gradient), vec1d(u)
    u_norm = np.linalg.norm(u)
    if u_norm < 1e-10:
        return 0.0
    g_dot_u = np.dot(gradient, u)
    return delta / u_norm - g_dot_u / (u_norm ** 2)


def ds_afir_gradient(coords, other_coords, ref_coords, pes_gradient, delta, is_initial=False):
    """Compute DS-AFIR modified gradient."""
    q_i, p_j, q_0 = vec1d(coords), vec1d(other_coords), vec1d(ref_coords)
    g = vec1d(pes_gradient)
    Y = compute_Y(q_i, p_j, q_0, is_initial)
    u = compute_u(q_i, p_j, q_0, Y, is_initial)
    X = compute_X(g, u, delta)
    ds_afir_grad = g + X * u
    return ds_afir_grad.reshape(-1, 3), Y, X


def lqa_step(coords, gradient, step_size):
    """LQA step for path integration."""
    g = vec1d(gradient)
    g_norm = np.linalg.norm(g)
    if g_norm < 1e-12:
        return coords.copy()
    delta_q = -g / g_norm * step_size
    return (vec1d(coords) + delta_q).reshape(-1, 3)


def find_path_extrema(path_energies, threshold=1e-6):
    """Find local minima and maxima along path."""
    n = len(path_energies)
    minima, maxima = [], []
    for i in range(1, n - 1):
        E_prev, E_curr, E_next = path_energies[i-1], path_energies[i], path_energies[i+1]
        if E_curr < E_prev - threshold and E_curr < E_next - threshold:
            minima.append(i)
        elif E_curr > E_prev + threshold and E_curr > E_next + threshold:
            maxima.append(i)
    return minima, maxima


def redistribute_path_points(path_coords, interval):
    """Redistribute path points with equal spacing."""
    if len(path_coords) < 2:
        return path_coords, [0.0]
    arc_lengths = [0.0]
    for i in range(1, len(path_coords)):
        dist = np.linalg.norm(vec1d(path_coords[i]) - vec1d(path_coords[i-1]))
        arc_lengths.append(arc_lengths[-1] + dist)
    total_length = arc_lengths[-1]
    if total_length < 1e-10:
        return path_coords, arc_lengths
    n_new = max(2, int(np.ceil(total_length / interval)) + 1)
    new_arc_lengths = np.linspace(0, total_length, n_new)
    new_path = []
    for s_target in new_arc_lengths:
        for i in range(len(arc_lengths) - 1):
            if arc_lengths[i] <= s_target <= arc_lengths[i + 1]:
                seg_length = arc_lengths[i + 1] - arc_lengths[i]
                lam = 0.0 if seg_length < 1e-10 else (s_target - arc_lengths[i]) / seg_length
                new_pos = (1.0 - lam) * path_coords[i] + lam * path_coords[i + 1]
                new_path.append(new_pos.copy())
                break
    return new_path, list(new_arc_lengths)


def lup_optimization(path_coords, atoms_template, n_iterations, interval, step_size, output):
    """Locally Updated Planes path optimization."""
    path = [p.copy() for p in path_coords]
    for iteration in range(n_iterations):
        path, _ = redistribute_path_points(path, interval)
        n_points = len(path)
        if n_points < 3:
            break
        for i in range(1, n_points - 1):
            tangent = vec1d(path[i + 1]) - vec1d(path[i - 1])
            tangent_norm = np.linalg.norm(tangent)
            if tangent_norm < 1e-10:
                continue
            tangent = tangent / tangent_norm
            atoms_template.set_positions(path[i])
            forces = to_numpy_f64(atoms_template.get_forces())
            grad = -vec1d(forces)
            grad_parallel = np.dot(grad, tangent) * tangent
            grad_perp = grad - grad_parallel
            grad_perp_norm = np.linalg.norm(grad_perp)
            if grad_perp_norm > 1e-10:
                step = -step_size * grad_perp / grad_perp_norm
                path[i] = (vec1d(path[i]) + step).reshape(-1, 3)
    energies = []
    for coords in path:
        atoms_template.set_positions(coords)
        energies.append(float(atoms_template.get_potential_energy(force_consistent=True)))
    return path, energies


class DSAFIR(JobABC):
    """DS-AFIR calculator for finding reaction paths."""

    def __init__(self, output, atoms_R, atoms_P, paras=None):
        super().__init__(output)
        self.atoms_R = atoms_R.copy()
        self.atoms_P = atoms_P.copy()
        if atoms_R.calc is not None:
            self.atoms_R.calc = atoms_R.calc
            self.atoms_P.calc = atoms_R.calc

        # Initialize params from paras dict
        self.params = self._init_params(DSAFIRParams, paras,
                                        ("dsafir", "ds_afir", "afir", "DSAFIR", "DS_AFIR", "AFIR", "ts"))

        # Override convergence thresholds from atoms if available
        for attr in ('f_max_th', 'f_rms_th', 'dp_max_th', 'dp_rms_th'):
            if hasattr(atoms_R, attr):
                setattr(self.params, attr, getattr(atoms_R, attr))

        self.afir_path, self.afir_energies = [], []
        self.lup_path, self.lup_energies = [], []
        self.eq_list, self.ts_list = [], []
    
    def atoms_to_xyz(self, atoms):
        lines = []
        for s, (x, y, z) in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
            lines.append(f"{s:<2} {x:14.6f} {y:14.6f} {z:14.6f}")
        return "\n".join(lines) + "\n"
    
    def _get_energy_and_gradient(self, coords):
        self.atoms_R.set_positions(coords)
        energy = float(self.atoms_R.get_potential_energy(force_consistent=True))
        gradient = -to_numpy_f64(self.atoms_R.get_forces())
        return energy, gradient
    
    def _make_atoms_copy(self, coords):
        atoms = self.atoms_R.copy()
        atoms.set_positions(coords)
        atoms.calc = self.atoms_R.calc
        for attr in ['f_max_th', 'f_rms_th', 'dp_max_th', 'dp_rms_th']:
            if hasattr(self.atoms_R, attr):
                setattr(atoms, attr, getattr(self.atoms_R, attr))
        return atoms
    
    def _integrate_afir_path(self):
        q_i = to_numpy_f64(self.atoms_R.get_positions())
        p_j = to_numpy_f64(self.atoms_P.get_positions())
        p_j, rmsd, _, _ = kabsch_align(q_i, p_j)
        
        q_0, p_0 = q_i.copy(), p_j.copy()
        path_q, path_p = [q_i.copy()], [p_j.copy()]
        
        E_q, g_q = self._get_energy_and_gradient(q_i)
        E_p, g_p = self._get_energy_and_gradient(p_j)
        energies_q, energies_p = [E_q], [E_p]
        
        # 用于跟踪是否在下降
        prev_E_q, prev_E_p = E_q, E_p
        
        step = 0
        while step < self.params.max_steps:
            distance = np.linalg.norm(vec1d(q_i) - vec1d(p_j))
            if distance < self.params.convergence_threshold:
                break
            
            E_q, g_q = self._get_energy_and_gradient(q_i)
            E_p, g_p = self._get_energy_and_gradient(p_j)
            
            if E_q <= E_p:
                is_initial = (len(path_q) == 1)
                ds_grad, Y, X = ds_afir_gradient(q_i, p_j, q_0, g_q, self.params.delta, is_initial)
                q_i_new = lqa_step(q_i, ds_grad, self.params.step_size)
                E_q_new, _ = self._get_energy_and_gradient(q_i_new)
                
                # **关键修改：检测局部极小点**
                if E_q_new > E_q and prev_E_q > E_q:  # 从下降转为上升
                    q_0 = q_i.copy()  # 更新参考点为当前极小点
                    log_info([f"  → New reference point found at step {step}, E={E_q:.8f}\n"], self.output)
                
                prev_E_q = E_q
                q_i = q_i_new
                path_q.append(q_i.copy())
                energies_q.append(E_q_new)
            else:
                # p_j侧的对称逻辑
                is_initial = (len(path_p) == 1)
                ds_grad, Y, X = ds_afir_gradient(p_j, q_i, p_0, g_p, self.params.delta, is_initial)
                p_j_new = lqa_step(p_j, ds_grad, self.params.step_size)
                E_p_new, _ = self._get_energy_and_gradient(p_j_new)
                
                if E_p_new > E_p and prev_E_p > E_p:
                    p_0 = p_j.copy()
                    log_info([f"  → New reference point found at step {step}, E={E_p:.8f}\n"], self.output)
                
                prev_E_p = E_p
                p_j = p_j_new
                path_p.append(p_j.copy())
                energies_p.append(E_p_new)
            
            step += 1
        
        full_path = path_q + path_p[::-1]
        full_energies = energies_q + energies_p[::-1]
        return full_path, full_energies
    
    def _refine_ts(self, ts_guess, ts_idx):
        from .PRFO import PRFO, PRFOParams
        ts_atoms = self._make_atoms_copy(ts_guess)
        base, _ = os.path.splitext(self.output)
        prfo_output = base + f"_ts{ts_idx}_prfo.out"
        prfo_params = PRFOParams(
            max_iter=self.params.ts_max_iter, f_max_th=self.params.f_max_th,
            f_rms_th=self.params.f_rms_th, dp_max_th=self.params.dp_max_th, dp_rms_th=self.params.dp_rms_th
        )
        try:
            prfo = PRFO(prfo_output, ts_atoms, params=prfo_params)
            ts_optimized = prfo.run()
            ts_energy = float(ts_optimized.get_potential_energy(force_consistent=True))
            forces = to_numpy_f64(ts_optimized.get_forces())
            converged = float(np.abs(forces).max()) < self.params.f_max_th
            return ts_optimized, ts_energy, converged
        except Exception as e:
            log_info([f"    PRFO failed: {e}\n"], self.output)
            ts_atoms.set_positions(ts_guess)
            return ts_atoms, float(ts_atoms.get_potential_energy(force_consistent=True)), False
    
    def run(self):
        log_info([
            f"\n{'#'*70}\n", f"{'DS-AFIR Path Search':^70}\n", f"{'#'*70}\n\n",
            f"Ref: Maeda et al. J. Comput. Chem. 2018, 39, 233-251.\n\n"
        ], self.output)
        E_R = float(self.atoms_R.get_potential_energy(force_consistent=True))
        E_P = float(self.atoms_P.get_potential_energy(force_consistent=True))
        kcal = 627.509
        log_info([
            f"Reactant: {E_R:.8f} Eh\nProduct:  {E_P:.8f} Eh\ndE: {(E_P-E_R)*kcal:.2f} kcal/mol\n\n",
            f"Thresholds: f_max={self.params.f_max_th:.6f}, f_rms={self.params.f_rms_th:.6f}\n\n",
            f"Reactant:\n{self.atoms_to_xyz(self.atoms_R)}\nProduct:\n{self.atoms_to_xyz(self.atoms_P)}"
        ], self.output)
        base, _ = os.path.splitext(self.output)
        
        # Step 1: AFIR path
        afir_path, afir_energies = self._integrate_afir_path()
        self.afir_path, self.afir_energies = afir_path, afir_energies
        afir_atoms_list = [self._make_atoms_copy(c) for c in afir_path]
        afir_path_file = base + "_afir_path.xyz"
        write_multi_xyz(afir_path_file, afir_atoms_list, afir_energies)
        log_info([f"AFIR path: {afir_path_file}\n"], self.output)
        
        # Step 2: LUP optimization
        if self.params.use_lup:
            log_info([f"\n{'='*70}\n", f"{'LUP Optimization':^70}\n", f"{'='*70}\n\n"], self.output)
            lup_path, lup_energies = lup_optimization(
                afir_path, self.atoms_R, self.params.lup_iterations_coarse,
                self.params.lup_interval_coarse, self.params.lup_step_size, self.output
            )
            lup_path, lup_energies = lup_optimization(
                lup_path, self.atoms_R, self.params.lup_iterations_fine,
                self.params.lup_interval_fine, self.params.lup_step_size, self.output
            )
            self.lup_path, self.lup_energies = lup_path, lup_energies
            lup_atoms_list = [self._make_atoms_copy(c) for c in lup_path]
            lup_path_file = base + "_lup_path.xyz"
            write_multi_xyz(lup_path_file, lup_atoms_list, lup_energies)
            log_info([f"LUP path: {lup_path_file}\n"], self.output)
            final_path, final_energies = lup_path, lup_energies
        else:
            final_path, final_energies = afir_path, afir_energies
        
        # Step 3: Find extrema
        log_info([f"\n{'='*70}\n", f"{'TS/EQ Identification':^70}\n", f"{'='*70}\n\n"], self.output)
        minima_idx, maxima_idx = find_path_extrema(final_energies)
        log_info([f"Found {len(maxima_idx)} TS, {len(minima_idx)} intermediates\n\n"], self.output)
        
        # Step 4: Build EQ list
        eq_list = [(self._make_atoms_copy(final_path[0]), final_energies[0], "Reactant")]
        for i, idx in enumerate(minima_idx):
            eq_list.append((self._make_atoms_copy(final_path[idx]), final_energies[idx], f"Int_{i+1}"))
        eq_list.append((self._make_atoms_copy(final_path[-1]), final_energies[-1], "Product"))
        self.eq_list = eq_list
        
        # Step 5: Optimize TSs
        ts_list = []
        for i, ts_idx in enumerate(maxima_idx):
            log_info([f"TS {i} (point {ts_idx}): E = {final_energies[ts_idx]:.8f} Eh\n"], self.output)
            ts_guess = final_path[ts_idx]
            if self.params.refine_ts:
                ts_atoms, ts_energy, converged = self._refine_ts(ts_guess, i)
                log_info([f"  PRFO {'converged' if converged else 'not converged'}: {ts_energy:.8f} Eh\n"], self.output)
            else:
                ts_atoms = self._make_atoms_copy(ts_guess)
                ts_energy = final_energies[ts_idx]
            ts_list.append((ts_atoms, ts_energy, (i, i+1)))
        self.ts_list = ts_list
        
        # Step 6: Output
        log_info([f"\n{'='*70}\n", f"{'Results':^70}\n", f"{'='*70}\n\n"], self.output)
        eq_file = base + "_eq_list.xyz"
        eq_comments = [f"EQ{i} {eq[2]} E={eq[1]:.10f}" for i, eq in enumerate(eq_list)]
        write_multi_xyz(eq_file, [eq[0] for eq in eq_list], comments=eq_comments)
        if ts_list:
            ts_file = base + "_ts_list.xyz"
            ts_comments = [f"TS{i} EQ{ts[2][0]}-EQ{ts[2][1]} E={ts[1]:.10f}" for i, ts in enumerate(ts_list)]
            write_multi_xyz(ts_file, [ts[0] for ts in ts_list], comments=ts_comments)
        
        E_ref = eq_list[0][1]
        log_info([f"EQ List:\n{'-'*60}\n{'Idx':>4s} {'Label':>12s} {'E(Eh)':>14s} {'dE(kcal)':>12s}\n{'-'*60}\n"], self.output)
        for i, (_, E, label) in enumerate(eq_list):
            log_info([f"{i:>4d} {label:>12s} {E:14.8f} {(E-E_ref)*kcal:12.2f}\n"], self.output)
        if ts_list:
            log_info([f"\nTS List:\n{'-'*70}\n{'Idx':>4s} {'Connects':>12s} {'E(Eh)':>14s} {'dE(kcal)':>12s} {'Barrier':>12s}\n{'-'*70}\n"], self.output)
            for i, (_, E, conn) in enumerate(ts_list):
                E_lower = min(eq_list[conn[0]][1], eq_list[conn[1]][1])
                log_info([f"{i:>4d} {'EQ'+str(conn[0])+'-EQ'+str(conn[1]):>12s} {E:14.8f} {(E-E_ref)*kcal:12.2f} {(E-E_lower)*kcal:12.2f}\n"], self.output)
        
        log_info([f"\nOutput: {eq_file}\n"], self.output)
        if ts_list:
            log_info([f"        {ts_file}\n"], self.output)
        log_info([f"\n{'='*70}\n", f"{'DS-AFIR Completed':^70}\n", f"{'='*70}\n"], self.output)
        return eq_list, ts_list