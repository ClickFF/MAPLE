# -*- coding: utf-8 -*-
"""
Batch L-BFGS optimizer with fixed padded dimension nmax.
Highly parallelized optimization using GPU tensors.
Individual batches exit when converged (dynamic batch shrinking).
"""

from typing import List, Optional
import os
import numpy as np
import torch
from ase import Atoms
from .LBFGS import LBFGS
from ._common import (batch_tensors, batch_results, store_batch_results,
                      member_outputs, write_xyz)

DTYPE = torch.float64



class BatchLBFGS:
    """
    Batched L-BFGS optimizer with dynamic batch shrinking.
    Each batch maintains its own L-BFGS history (S, Y, rho).
    """

    def __init__(self,
                 output: str,
                 memory: int = 5,
                 curvature: float = 70.0,
                 maxstep: float = 0.2,
                 maxiter: int = 256,
                 device: str = "cuda",
                 write_traj: bool = False,
                 traj_every: int = 1,
                 verbose: int = 1,
                 batch_size: Optional[int] = None):
        
        self.memory = memory
        self.curvature = curvature
        self.maxstep = maxstep
        self.maxiter = maxiter
        self.device = torch.device(device)
        self.write_traj = write_traj
        self.traj_every = traj_every
        self.verbose = verbose
        self.batch_size = batch_size
        
        self.output = os.path.abspath(output)
        self.out_dir = os.path.dirname(self.output) or "."
        os.makedirs(self.out_dir, exist_ok=True)

        self.log_fp = None
        self.xyz_paths = []
        self.frame_counts = []

        self._L_vec = None
        self._nmax = 0
        self._B = 0
        self._arange_n = None
        self._real_mask = None
        self._orig_index = None

        # Per-batch L-BFGS history
        self.S_history = []  # List of (B, nmax) tensors per history step
        self.Y_history = []  # List of (B, nmax) tensors per history step
        self.rho_history = []  # List of (B,) tensors per history step
        self.history_valid = None  # (B, memory) mask indicating valid history slots

    # ===================================================
    # PUBLIC RUN
    # ===================================================
    def run(self, mols) -> List[Atoms]:
        device = self.device
        all_atoms = list(mols.multiatoms)
        atoms_list = all_atoms.copy()
        calc = atoms_list[0].calc
        B0 = len(atoms_list)
        self.results = batch_results(all_atoms)
        self._orig_index = torch.arange(B0, dtype=torch.long, device=device)
        self._nmax = 3 * max(len(at) for at in atoms_list)
        self._arange_n = torch.arange(self._nmax, device=device)
        self._rebuild_topology(atoms_list)
        self.history_valid = torch.zeros((B0, self.memory), dtype=torch.bool, device=device)
        self.S_history, self.Y_history, self.rho_history = [], [], []
        self._init_xyz_paths(all_atoms)
        self._open_log()
        try:
            self._w("# Batch L-BFGS optimization start\n")
            E_old, F_old = self._get_ef(calc, atoms_list)
            store_batch_results(self.results, self._orig_index, atoms_list,
                                E_old, F_old, 0, torch.zeros(B0, dtype=torch.bool, device=device))
            if self.write_traj:
                self._dump_xyz_all(atoms_list, E_old, F_old)
            iteration = 0
            while iteration < self.maxiter and len(atoms_list) > 0:
                iteration += 1
                real_mask = self._real_mask
                g_cart = -F_old * real_mask.to(DTYPE)
                search_dir = self._two_loop_batched(g_cart)
                step_cart = self._clip_step_batched(search_dir)

                r_old = [at.get_positions() for at in atoms_list]
                step_np = step_cart.detach().cpu().numpy()
                for b, atoms in enumerate(atoms_list):
                    step = step_np[b, :3 * len(atoms)].reshape(-1, 3)
                    atoms.set_positions(r_old[b] + step)

                try:
                    E_new, F_new = self._get_ef(calc, atoms_list)
                except Exception:
                    for atoms, r in zip(atoms_list, r_old):
                        atoms.set_positions(r)
                    raise
                g_new = -F_new * real_mask.to(DTYPE)
                s_vec = np.zeros(step_np.shape)
                for b, atoms in enumerate(atoms_list):
                    s_vec[b, :3 * len(atoms)] = (atoms.positions - r_old[b]).reshape(-1)
                s_vec = torch.as_tensor(s_vec, dtype=DTYPE, device=device)
                y_vec = g_new - g_cart
                self._update_history_batched(s_vec, y_vec)
                step_cart = s_vec

                self._log_iteration(iteration, E_new, step_cart, F_new, real_mask)
                done = self._check_convergence(iteration, E_new, atoms_list,
                                               step_cart, F_new, real_mask)
                self._log_members(iteration, E_new, F_new, step_cart)
                if self.write_traj and iteration % self.traj_every == 0:
                    self._dump_xyz_all(atoms_list, E_new, F_new)
                store_batch_results(self.results, self._orig_index, atoms_list,
                                    E_new, F_new, iteration, done)

                survive_local = (~done).nonzero(as_tuple=False).flatten()
                if survive_local.numel() < len(atoms_list):
                    atoms_list = [atoms_list[i] for i in survive_local.cpu().tolist()]
                    self._orig_index = self._orig_index[survive_local]
                    self._shrink_history(survive_local)
                    self._rebuild_topology(atoms_list)
                E_old, F_old = E_new[survive_local], F_new[survive_local]

            for idx_orig, logger in enumerate(self._loggers):
                converged = self.results['converged'][idx_orig]
                iterations = self.results['iterations'][idx_orig]
                summary = (f"LBFGS converged at iteration {iterations}." if converged else
                           f"LBFGS did NOT converge after {iterations} iterations.")
                logger._finalize_run(self.results['energy'][idx_orig], summary,
                                     self.xyz_paths[idx_orig], self.results['forces'][idx_orig])
                self._w(f"Member {idx_orig + 1}: {summary} "
                        f"Energy = {self.results['energy'][idx_orig]:.10f} Hartree\n")
            return all_atoms
        finally:
            self._close_log()

    def _get_ef(self, calc, atoms_list):
        return batch_tensors(calc, atoms_list, self._nmax, self.device, self.batch_size)

    # ===================================================
    # L-BFGS TWO-LOOP RECURSION (BATCHED)
    # ===================================================
    def _two_loop_batched(self, grad: torch.Tensor) -> torch.Tensor:
        """
        Batched L-BFGS two-loop recursion.
        
        Args:
            grad: (B, nmax) gradient tensor
            
        Returns:
            search_dir: (B, nmax) search direction tensor
        """
        device = self.device
        B, nmax = grad.shape
        
        q = grad.clone()
        alpha_list = []

        # Backward pass through history (most recent first)
        num_history = len(self.S_history)
        for t in range(num_history - 1, -1, -1):
            s_t = self.S_history[t]  # (B, nmax)
            y_t = self.Y_history[t]  # (B, nmax)
            rho_t = self.rho_history[t]  # (B,)
            valid_t = self.history_valid[:, t]  # (B,)

            # alpha = rho * (s^T q)
            alpha = rho_t * (s_t * q).sum(dim=-1)  # (B,)
            alpha = torch.where(valid_t, alpha, torch.zeros_like(alpha))
            alpha_list.append(alpha)

            # q = q - alpha * y
            q = q - alpha.unsqueeze(-1) * y_t

        # Reverse alpha_list for forward pass
        alpha_list = list(reversed(alpha_list))

        # Use each member's most recent valid pair, including partial histories.
        if num_history > 0:
            counts = self.history_valid[:, :num_history].sum(dim=1)
            latest = counts.clamp(min=1) - 1
            members = torch.arange(B, device=device)
            s_last = torch.stack(self.S_history, dim=1)[members, latest]
            y_last = torch.stack(self.Y_history, dim=1)[members, latest]
            gamma = (y_last * s_last).sum(-1) / ((y_last * y_last).sum(-1) + 1e-20)
            gamma = torch.where(counts > 0, gamma, torch.full_like(gamma, 1. / self.curvature))
        else:
            gamma = torch.full((B,), 1. / self.curvature, dtype=DTYPE, device=device)

        z = gamma.unsqueeze(-1) * q

        # Forward pass through history (oldest first)
        for t in range(num_history):
            s_t = self.S_history[t]
            y_t = self.Y_history[t]
            rho_t = self.rho_history[t]
            valid_t = self.history_valid[:, t]
            alpha_t = alpha_list[t]

            # beta = rho * (y^T z)
            beta = rho_t * (y_t * z).sum(dim=-1)
            beta = torch.where(valid_t, beta, torch.zeros_like(beta))

            # z = z + s * (alpha - beta)
            z = z + s_t * (alpha_t - beta).unsqueeze(-1)

        direction = -z
        bad = ~torch.isfinite(direction).all(dim=1) | ((direction * grad).sum(-1) >= 0)
        return torch.where(bad[:, None], -grad / self.curvature, direction)

    # ===================================================
    # STEP CLIPPING
    # ===================================================
    def _clip_step_batched(self, step: torch.Tensor) -> torch.Tensor:
        """
        Clip step size per batch to maxstep.
        
        Args:
            step: (B, nmax) step tensor
            
        Returns:
            clipped_step: (B, nmax) clipped step tensor
        """
        max_disp = step.abs().amax(dim=-1)  # (B,)
        scale = torch.clamp(self.maxstep / (max_disp + 1e-20), max=1.0)
        return step * scale.unsqueeze(-1)

    # ===================================================
    # HISTORY UPDATE
    # ===================================================
    def _update_history_batched(self, s_vec: torch.Tensor, y_vec: torch.Tensor):
        ys = (y_vec * s_vec).sum(-1)
        valid = torch.isfinite(ys) & (ys > 1e-12)
        rho = 1. / torch.where(valid, ys, torch.ones_like(ys))
        counts = self.history_valid.sum(dim=1)
        size = len(self.S_history)
        if size < self.memory and (valid & (counts == size)).any():
            self.S_history.append(torch.zeros_like(s_vec))
            self.Y_history.append(torch.zeros_like(y_vec))
            self.rho_history.append(torch.zeros_like(rho))
            size += 1
        # Invalid curvature never consumes another member's history slot.
        shift = valid & (counts == self.memory)
        for t in range(size - 1):
            self.S_history[t] = torch.where(shift[:, None], self.S_history[t + 1], self.S_history[t])
            self.Y_history[t] = torch.where(shift[:, None], self.Y_history[t + 1], self.Y_history[t])
            self.rho_history[t] = torch.where(shift, self.rho_history[t + 1], self.rho_history[t])
        slots = counts.clamp(max=self.memory - 1)
        for t in range(size):
            write = valid & (slots == t)
            self.S_history[t] = torch.where(write[:, None], s_vec, self.S_history[t])
            self.Y_history[t] = torch.where(write[:, None], y_vec, self.Y_history[t])
            self.rho_history[t] = torch.where(write, rho, self.rho_history[t])
            self.history_valid[:, t] |= write

    # ===================================================
    # HISTORY SHRINKING
    # ===================================================
    def _shrink_history(self, survive_idx: torch.Tensor):
        """Shrink history when batches are removed."""
        for i in range(len(self.S_history)):
            self.S_history[i] = self.S_history[i][survive_idx]
            self.Y_history[i] = self.Y_history[i][survive_idx]
            self.rho_history[i] = self.rho_history[i][survive_idx]
        self.history_valid = self.history_valid[survive_idx]

    # ===================================================
    # CONVERGENCE CHECK
    # ===================================================
    def _check_convergence(self, it, E, atoms_list, step_cart, F, real_mask):
        """
        Check convergence criteria for each batch.
        
        Returns:
            done: (B,) boolean tensor indicating converged batches
        """
        device = self.device
        B = len(atoms_list)

        # Get convergence thresholds
        f_max_th = torch.tensor(
            [getattr(at, "f_max_th", 2e-3) for at in atoms_list],
            dtype=DTYPE, device=device)
        f_rms_th = torch.tensor(
            [getattr(at, "f_rms_th", 1e-3) for at in atoms_list],
            dtype=DTYPE, device=device)
        dp_max_th = torch.tensor(
            [getattr(at, "dp_max_th", 1e-3) for at in atoms_list],
            dtype=DTYPE, device=device)
        dp_rms_th = torch.tensor(
            [getattr(at, "dp_rms_th", 5e-4) for at in atoms_list],
            dtype=DTYPE, device=device)

        L_eff = self._L_vec.clamp(min=1).to(DTYPE)

        # Compute metrics
        max_f = F.abs().amax(dim=-1)
        rms_f = torch.sqrt((F ** 2).sum(-1) / L_eff)
        max_dp = step_cart.abs().amax(dim=-1)
        rms_dp = torch.sqrt((step_cart ** 2).sum(-1) / L_eff)

        # Check convergence
        done = (
            (max_f <= f_max_th)
            & (rms_f <= f_rms_th)
            & (max_dp <= dp_max_th)
            & (rms_dp <= dp_rms_th)
        )

        # Log convergence table
        self._w(self._fmt_convergence_table(
            it=it,
            E=E,
            max_f=max_f, rms_f=rms_f,
            max_dp=max_dp, rms_dp=rms_dp,
            f_max_th=f_max_th, f_rms_th=f_rms_th,
            dp_max_th=dp_max_th, dp_rms_th=dp_rms_th,
            done=done
        ))

        return done

    # ===================================================
    # TOPOLOGY
    # ===================================================
    def _rebuild_topology(self, atoms_list):
        device = self.device
        B = len(atoms_list)
        self._B = B

        L_list = [3 * len(at) for at in atoms_list]
        self._L_vec = torch.tensor(L_list, dtype=torch.int64, device=device)

        # real_mask padded to fixed nmax
        self._real_mask = (self._arange_n[None, :] < self._L_vec[:, None])

    # ===================================================
    # LOGGING
    # ===================================================
    def _open_log(self):
        self.log_fp = open(self.output, "a", encoding="utf-8")

    def _close_log(self):
        if self.log_fp:
            self.log_fp.close()
            self.log_fp = None

    def _w(self, s):
        if self.log_fp:
            self.log_fp.write(s)
            self.log_fp.flush()

    def _log_iteration(self, it, E, step_cart, F, real_mask):
        """Log iteration information."""
        if self.verbose == 0:
            return

        L_eff = self._L_vec.clamp(min=1).to(DTYPE)
        max_f = F.abs().amax(dim=-1)
        rms_f = torch.sqrt((F ** 2).sum(-1) / L_eff)
        max_dp = step_cart.abs().amax(dim=-1)
        rms_dp = torch.sqrt((step_cart ** 2).sum(-1) / L_eff)

        msg = f"\nIter {it}: "
        msg += f"E_mean={E.mean().item():.6f} "
        msg += f"max|F|_mean={max_f.mean().item():.6f} "
        msg += f"rms|F|_mean={rms_f.mean().item():.6f} "
        msg += f"max|dX|_mean={max_dp.mean().item():.6f} "
        msg += f"rms|dX|_mean={rms_dp.mean().item():.6f}\n"
        
        self._w(msg)

    def _fmt_convergence_table(self, it, E, max_f, rms_f, max_dp, rms_dp,
                               f_max_th, f_rms_th, dp_max_th, dp_rms_th, done):
        """Format convergence table."""
        lines = []
        lines.append("-" * 70 + "\n")
        lines.append(f"{('Batch L-BFGS Iteration ' + str(it)).center(70)}\n")
        lines.append("-" * 70 + "\n")
        lines.append(
            "Batch   Energy(Ha)   Max|F|(H/A)  RMS|F|  Max|dX|(A)  RMS|dX|  Conv\n"
        )
        lines.append("-" * 70 + "\n")

        E, max_f, rms_f, max_dp, rms_dp = torch.stack(
            (E, max_f, rms_f, max_dp, rms_dp)).detach().cpu().numpy()
        done = done.detach().cpu().numpy()
        indices = self._orig_index.detach().cpu().tolist()
        B = E.shape[0]
        for b in range(B):
            idx_orig = indices[b]
            lines.append(
                f"[{idx_orig:2d}] "
                f"{E[b].item():14.6f} "
                f"{max_f[b].item():12.6f} "
                f"{rms_f[b].item():10.6f} "
                f"{max_dp[b].item():12.6f} "
                f"{rms_dp[b].item():10.6f} "
                f"{('YES' if bool(done[b]) else 'NO')}\n"
            )

        lines.append("\n")
        return "".join(lines)

    # ===================================================
    # XYZ OUTPUT
    # ===================================================
    def _init_xyz_paths(self, atoms_list):
        outputs = member_outputs(self.output, atoms_list, ('_opt', '_opt_traj'))
        self.xyz_paths = []
        self._loggers = []
        for atoms, output in zip(atoms_list, outputs):
            ext = '.pdb' if atoms.info.get('pdb_template') else '.xyz'
            self.xyz_paths.append(os.path.splitext(output)[0] + '_opt_traj' + ext)
            self._loggers.append(LBFGS(atoms, output=output, paras={
                'memory': self.memory, 'curvature': self.curvature,
                'max_step': self.maxstep, 'max_iter': self.maxiter,
                'verbose': self.verbose, 'log_final_paths': self.write_traj,
            }))
        self.frame_counts = [0] * len(atoms_list)

    def _dump_xyz_all(self, atoms_list, E, F):
        energies, forces = E.detach().cpu().numpy(), F.detach().cpu().numpy()
        indices = self._orig_index.detach().cpu().tolist()
        for b, atoms in enumerate(atoms_list):
            idx_orig = indices[b]
            write_xyz(self.xyz_paths[idx_orig], [atoms], energies=[energies[b]],
                      forces=[forces[b, :3 * len(atoms)].reshape(-1, 3)],
                      mode='a', start_index=self.frame_counts[idx_orig])
            self.frame_counts[idx_orig] += 1

    def _log_members(self, iteration, E, F, step_cart):
        energies = E.detach().cpu().numpy()
        forces, steps = F.detach().cpu().numpy(), step_cart.detach().cpu().numpy()
        for b, idx_orig in enumerate(self._orig_index.detach().cpu().tolist()):
            logger = self._loggers[idx_orig]
            n = len(logger.atoms)
            info = logger._build_iter_message(iteration, energies[b],
                steps[b, :3 * n].reshape(n, 3), forces[b, :3 * n].reshape(n, 3))
            logger._last_iter_info = info
            logger._log_iter(info)
