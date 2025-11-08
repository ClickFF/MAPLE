# -*- coding: utf-8 -*-
from typing import List, Optional
import os
import numpy as np
import torch
from ase import Atoms

BIG = 1e8
DTYPE = torch.float64


def _stack_positions(at_list: List[Atoms], nmax: int, device):
    B = len(at_list)
    X = torch.zeros((B, nmax // 3, 3), dtype=DTYPE, device=device)
    L = []
    for b, at in enumerate(at_list):
        n = len(at)
        X[b, :n, :] = torch.from_numpy(at.get_positions()).to(DTYPE).to(device)
        L.append(3 * n)
    return X, torch.tensor(L, dtype=torch.int64, device=device)


def _masses_flat(at_list: List[Atoms], nmax: int, device):
    B = len(at_list)
    mass = torch.ones((B, nmax), dtype=DTYPE, device=device)
    for b, at in enumerate(at_list):
        m = np.asarray(at.get_masses(), dtype=np.float64)
        mass[b, :3 * len(at)] = torch.from_numpy(np.repeat(m, 3)).to(DTYPE).to(device)
    return mass


class BatchPRFO:
    """
    Batched RS-PRFO TS search
    Units: E in Ha, F in Ha/Å, H in Ha/Å^2
    """

    def __init__(self,
                 output: str,
                 trust_init: float = 0.20,
                 trust_min: float = 1e-3,
                 trust_max: float = 1.00,
                 eta_shrink: float = 0.75,
                 eta_expand: float = 1.75,
                 max_inner_attempts: int = 8,
                 max_outer_iter: int = 256,
                 device: str = "cuda"):

        self.trust_init = trust_init
        self.trust_min = trust_min
        self.trust_max = trust_max
        self.eta_shrink = eta_shrink
        self.eta_expand = eta_expand
        self.max_inner_attempts = max_inner_attempts
        self.max_outer_iter = max_outer_iter
        self.device = torch.device(device)

        self.output = os.path.abspath(output)
        self.out_dir = os.path.dirname(self.output) if os.path.dirname(self.output) else "."
        os.makedirs(self.out_dir, exist_ok=True)
        self.log_fp = None
        self.xyz_paths: List[str] = []
        self.frame_counts: List[int] = []

        self.tracked_mode_vec_mw: Optional[torch.Tensor] = None  # (B, n)
        self.tracked_mode_idx: Optional[torch.Tensor] = None     # (B,)

    # ====================== public ======================

    def run(self, mols) -> List[Atoms]:
        device = self.device
        B = len(mols.multiatoms)

        self._open_log()
        self._w("# RS-PRFO batched TS search start\n")

        # warm-up to get n
        E0, F0, H0, _ = mols.get_efh()
        F0 = torch.as_tensor(F0, dtype=DTYPE, device=device)
        n = F0.shape[1]

        self._init_xyz_paths(B)
        self._dump_xyz_all(mols.multiatoms, tag="init")

        mass = _masses_flat(mols.multiatoms, n, device)
        D = 1.0 / torch.sqrt(torch.clamp(mass, min=1e-12))
        trust_r = torch.full((B,), self.trust_init, dtype=DTYPE, device=device)
        active = torch.ones(B, dtype=torch.bool, device=device)

        self.tracked_mode_vec_mw = None
        self.tracked_mode_idx = None

        L_vec = torch.tensor([3 * len(at) for at in mols.multiatoms],
                             dtype=torch.int64, device=device)
        last_step = torch.zeros((B, n), dtype=DTYPE, device=device)

        arange_n = torch.arange(n, device=device)

        for it in range(1, self.max_outer_iter + 1):
            X_backup, _ = _stack_positions(mols.multiatoms, n, device)

            # pull EFH
            E_old, F_raw, H_raw, _ = mols.get_efh()
            E_old = torch.as_tensor(E_old, dtype=DTYPE, device=device)            # (B,)
            F_raw = torch.as_tensor(F_raw, dtype=DTYPE, device=device)            # (B,n)
            H_raw = torch.as_tensor(H_raw, dtype=DTYPE, device=device)            # (B,n,n)
            H_raw = 0.5 * (H_raw + H_raw.transpose(-1, -2))

            # masks in Cartesian domain
            real_mask = (arange_n[None, :] < L_vec[:, None])                      # (B,n)
            mask_ij = (real_mask.unsqueeze(-1) & real_mask.unsqueeze(-2))
            H = H_raw * mask_ij.to(DTYPE)
            g_cart = -F_raw * real_mask.to(DTYPE)                                 # (B,n)

            # mass-weight
            g_mw = D * g_cart
            H_mw = D.unsqueeze(-1) * H * D.unsqueeze(-2)

            # isolate padding dimensions from eigenproblem
            pad_mask = ~real_mask
            if pad_mask.any():
                H_mw = H_mw.clone()
                H_mw[..., arange_n, arange_n] = H_mw[..., arange_n, arange_n] + pad_mask.to(DTYPE) * BIG
                g_mw = g_mw * (~pad_mask).to(DTYPE)

            # eigendecomp in MW space
            w, V = torch.linalg.eigh(H_mw)                                        # (B,n), (B,n,n)
            gp = (V.transpose(-1, -2) @ g_mw.unsqueeze(-1)).squeeze(-1)           # (B,n)

            tiny = w.abs() < 1e-10
            w = torch.where(tiny & (w == 0), torch.full_like(w, 1e-10), w)
            w = torch.where(tiny & (w != 0), torch.sign(w) * 1e-10, w)

            # mode following (eigen domain only)
            if self.tracked_mode_vec_mw is None:
                neg_idx = torch.argmin(w, dim=1)
                has_neg = w.gather(1, neg_idx[:, None]).squeeze(1) < -1e-6
                alt_idx = torch.argmax(gp.abs(), dim=1)
                tracked_idx = torch.where(has_neg, neg_idx, alt_idx)
                self.tracked_mode_idx = tracked_idx.clone()
                self.tracked_mode_vec_mw = torch.stack(
                    [V[b, :, tracked_idx[b]] for b in range(B)], dim=0)
            else:
                overlap = torch.matmul(V.transpose(-1, -2),
                                       self.tracked_mode_vec_mw.unsqueeze(-1)).squeeze(-1)
                idx = torch.argmax(overlap.abs(), dim=-1)
                signs = torch.sign(overlap.gather(1, idx[:, None]).squeeze(1))
                signs = torch.where(signs == 0, torch.ones_like(signs), signs)
                self.tracked_mode_idx = idx
                self.tracked_mode_vec_mw = torch.stack(
                    [V[b, :, idx[b]] * signs[b] for b in range(B)], dim=0
                )

            minus_mask = torch.nn.functional.one_hot(self.tracked_mode_idx, num_classes=n).to(torch.bool)
            plus_mask = ~minus_mask

            accepted = torch.zeros(B, dtype=torch.bool, device=device)
            last_rho = torch.full((B,), float("nan"), dtype=DTYPE, device=device)

            # ===== RS inner attempts =====
            for _try in range(self.max_inner_attempts):
                pend = active & (~accepted)
                if not pend.any():
                    break

                # unconstrained μ=0 on subspaces
                s_unc_minus = torch.zeros_like(gp)
                s_unc_plus = torch.zeros_like(gp)

                denom_m0 = -w.masked_select(minus_mask)
                denom_m0 = torch.where(denom_m0.abs() < 1e-10, torch.sign(denom_m0) * 1e-10, denom_m0)
                s_unc_minus[minus_mask] = -(-gp[minus_mask]) / denom_m0

                denom_p0 = w.masked_select(plus_mask)
                denom_p0 = torch.where(denom_p0.abs() < 1e-10, torch.sign(denom_p0) * 1e-10, denom_p0)
                s_unc_plus[plus_mask] = -(gp[plus_mask]) / denom_p0

                norm2_minus = (s_unc_minus ** 2).sum(-1)
                norm2_plus = (s_unc_plus ** 2).sum(-1)
                total_unc = norm2_minus + norm2_plus
                alpha = torch.where(total_unc > 0, norm2_minus / total_unc,
                                    torch.full_like(total_unc, 0.5)).clamp(0.05, 0.95)
                R2 = trust_r ** 2
                R2_minus = alpha * R2
                R2_plus = (1.0 - alpha) * R2

                mu_minus, s_part_minus = self._solve_mu_batched(w, gp, minus_mask, R2_minus, sigma=-1, only=pend)
                mu_plus, s_part_plus = self._solve_mu_batched(w, gp, plus_mask, R2_plus, sigma=+1, only=pend)

                s_p = s_part_minus + s_part_plus                                    # (B,n) eigen-domain
                norm_mw = torch.linalg.norm(s_p, dim=-1)                             # (B,)

                s_mw = (V @ s_p.unsqueeze(-1)).squeeze(-1) * real_mask.to(DTYPE)     # (B,n)
                s_cart = D * s_mw                                                    # (B,n)

                # apply tentative steps
                with torch.no_grad():
                    for b, at in enumerate(mols.multiatoms):
                        if not pend[b]:
                            continue
                        nb = len(at)
                        step_b = s_cart[b, :3 * nb].reshape(nb, 3).cpu().numpy()
                        x_old = X_backup[b, :nb, :].cpu().numpy()
                        at.set_positions(x_old + step_b)

                # new energies and forces
                E_new, F_new = mols.get_energies_forces()
                E_new = torch.as_tensor(E_new, dtype=DTYPE, device=device)
                F_new = torch.as_tensor(F_new, dtype=DTYPE, device=device)

                # model agreement
                Hs = torch.einsum('bij,bj->bi', H, s_cart)
                model_change = (g_cart * s_cart).sum(-1) + 0.5 * (s_cart * Hs).sum(-1)
                actual_change = (E_new - E_old)

                rho = torch.full_like(model_change, float('nan'))
                ok = model_change.abs() > 1e-16
                rho[ok] = actual_change[ok] / model_change[ok]
                last_rho = torch.where(pend, rho, last_rho)

                on_boundary = (norm_mw - trust_r).abs() <= (1e-6 * torch.clamp(trust_r, min=1.0))
                bad = pend & ((~torch.isfinite(rho)) | (rho < self.eta_shrink))
                force_accept = trust_r <= (self.trust_min * (1.0 + 1e-12))
                acc = pend & (~bad | force_accept)
                rej = pend & (~acc)

                # rollback rejects + shrink radius + clear last_step for those rejects
                with torch.no_grad():
                    for b, at in enumerate(mols.multiatoms):
                        if rej[b]:
                            nb = len(at)
                            at.set_positions(X_backup[b, :nb, :].cpu().numpy())
                trust_r = torch.where(rej, torch.clamp(0.5 * trust_r, min=self.trust_min), trust_r)
                last_step[rej] = 0.0

                # acceptances
                if acc.any():
                    grow = (rho > self.eta_expand) & on_boundary & acc
                    trust_r = torch.where(grow,
                                          torch.clamp(2.0 * trust_r, max=self.trust_max),
                                          trust_r)
                    last_step[acc] = s_cart[acc]
                    self._dump_xyz_subset(mols.multiatoms, acc, it)

                # cycle head line (accepted/rejected/rho/R/E heads)
                self._w(self._fmt_iter_head(it, acc, rej, rho, trust_r, E_new))
                accepted |= acc
                X_backup, _ = _stack_positions(mols.multiatoms, n, device)

            # ===== convergence check (four metrics) =====
            E_final, F_final = mols.get_energies_forces()
            E_final = torch.as_tensor(E_final, dtype=DTYPE, device=device)
            F_final = torch.as_tensor(F_final, dtype=DTYPE, device=device)
            g_last = -F_final * real_mask.to(DTYPE)

            # thresholds per Atoms
            f_max_th = torch.tensor([getattr(at, 'f_max_th', 2e-3) for at in mols.multiatoms], dtype=DTYPE, device=device)
            f_rms_th = torch.tensor([getattr(at, 'f_rms_th', 1e-3) for at in mols.multiatoms], dtype=DTYPE, device=device)
            dp_max_th = torch.tensor([getattr(at, 'dp_max_th', 1e-3) for at in mols.multiatoms], dtype=DTYPE, device=device)
            dp_rms_th = torch.tensor([getattr(at, 'dp_rms_th', 5e-4) for at in mols.multiatoms], dtype=DTYPE, device=device)

            L_eff = (L_vec.clamp(min=1)).to(DTYPE)
            max_f = g_last.abs().amax(dim=-1)
            rms_f = torch.sqrt((g_last ** 2).sum(-1) / L_eff)
            max_dp = last_step.abs().amax(dim=-1)
            rms_dp = torch.sqrt((last_step ** 2).sum(-1) / L_eff)

            done = (max_f <= f_max_th) & (rms_f <= f_rms_th) & (max_dp <= dp_max_th) & (rms_dp <= dp_rms_th)

            # ORCA-style per-batch table for this cycle
            self._w(self._fmt_orca_cycle_table(
                it=it,
                E=E_final,
                rho=last_rho,
                R=trust_r,
                max_f=max_f, rms_f=rms_f,
                max_dp=max_dp, rms_dp=rms_dp,
                f_max_th=f_max_th, f_rms_th=f_rms_th,
                dp_max_th=dp_max_th, dp_rms_th=dp_rms_th,
                done=done
            ))

            newly_done = done & active
            for idx in newly_done.nonzero(as_tuple=False).flatten().cpu().tolist():
                self._w(f">>> Batch {idx} converged at cycle {it}\n")
            active = active & (~done)
            if not active.any():
                self._w("# Normal termination.\n")
                break
        else:
            self._w("# Maximum iterations reached.\n")

        self._close_log()
        return mols.multiatoms

    # ====================== eigen μ solve ======================

    @staticmethod
    @torch.no_grad()
    def _solve_mu_batched(w, gp, mask, R2, sigma, only: torch.Tensor):
        device = w.device
        B, n = w.shape
        lam_all = (sigma * w)
        num_all = (sigma * gp)
        mu_out = torch.zeros(B, dtype=DTYPE, device=device)
        s_part = torch.zeros((B, n), dtype=DTYPE, device=device)

        for b in range(B):
            if not only[b]:
                continue
            m_b = mask[b]
            lam_b = lam_all[b][m_b]
            num_b = num_all[b][m_b]
            if lam_b.numel() == 0:
                mu_out[b] = 0.0
                continue

            R2_b = float(R2[b].item())
            denom0 = torch.where(lam_b.abs() < 1e-10, torch.sign(lam_b) * 1e-10, lam_b)
            s_unc = - num_b / denom0
            norm2_unc = float((s_unc * s_unc).sum().item())
            if norm2_unc <= R2_b:
                mu_out[b] = 0.0
                s_tmp = s_unc
            else:
                def F(mu_val: float) -> float:
                    mu_t = torch.tensor(mu_val, dtype=DTYPE, device=device)
                    denom = lam_b - mu_t
                    denom = torch.where(denom.abs() < 1e-12, torch.sign(denom) * 1e-12, denom)
                    return float(((num_b / denom) ** 2).sum().item())

                wt_min = float(lam_b.min().item())
                hi = wt_min - 1e-6
                # ensure finite
                if not np.isfinite(F(hi)):
                    hi = wt_min - 1e-4

                lo = hi - 1.0
                Fa = F(lo)
                it_ex = 0
                while Fa > R2_b and it_ex < 60:
                    step = max(1.0, abs(lo) * 0.5)
                    lo -= step
                    Fa = F(lo)
                    it_ex += 1

                for _ in range(60):
                    mid = 0.5 * (lo + hi)
                    Fm = F(mid)
                    if abs(Fm - R2_b) <= 1e-12 * max(1.0, R2_b) or abs(hi - lo) < 1e-12:
                        lo, hi = mid, mid
                        break
                    if Fm > R2_b:
                        hi = mid
                    else:
                        lo = mid

                mu_star = 0.5 * (lo + hi)
                mu_out[b] = mu_star
                mu_t = torch.tensor(mu_star, dtype=DTYPE, device=device)
                denom = lam_b - mu_t
                denom = torch.where(denom.abs() < 1e-12, torch.sign(denom) * 1e-12, denom)
                s_tmp = - num_b / denom

            s_full = torch.zeros(n, dtype=DTYPE, device=device)
            s_full[m_b] = s_tmp
            s_part[b] = s_full
        return mu_out, s_part

    # ====================== logging helpers ======================

    def _open_log(self):
        self.log_fp = open(self.output, "w", encoding="utf-8")

    def _close_log(self):
        if self.log_fp:
            self.log_fp.close()
            self.log_fp = None

    def _w(self, s: str):
        self.log_fp.write(s)
        self.log_fp.flush()

    def _init_xyz_paths(self, B: int):
        self.xyz_paths = [os.path.join(self.out_dir, f"ts_batch{i+1}.xyz") for i in range(B)]
        self.frame_counts = [0 for _ in range(B)]
        for p in self.xyz_paths:
            open(p, "w").close()

    def _dump_xyz_all(self, atoms_list: List[Atoms], tag: str = ""):
        for i, at in enumerate(atoms_list):
            self._append_xyz(i, at, comment=tag)

    def _dump_xyz_subset(self, atoms_list: List[Atoms], accept_mask: torch.Tensor, it: int):
        for i, at in enumerate(atoms_list):
            if bool(accept_mask[i].item()):
                self._append_xyz(i, at, comment=f"iter={it}")

    def _append_xyz(self, idx: int, at: Atoms, comment: str = ""):
        path = self.xyz_paths[idx]
        n = len(at)
        pos = at.get_positions()
        sym = at.get_chemical_symbols()
        with open(path, "a") as f:
            f.write(f"{n}\n")
            f.write(f"{comment}\n")
            for k in range(n):
                x, y, z = pos[k]
                f.write(f"{sym[k]:<2s} {x:20.10f} {y:20.10f} {z:20.10f}\n")
        self.frame_counts[idx] += 1

    @staticmethod
    def _fmt_iter_head(it: int,
                       acc: torch.Tensor,
                       rej: torch.Tensor,
                       rho: torch.Tensor,
                       trust_r: torch.Tensor,
                       E_new: torch.Tensor) -> str:
        acc_list = acc.nonzero(as_tuple=False).flatten().cpu().tolist()
        rej_list = rej.nonzero(as_tuple=False).flatten().cpu().tolist()
        rhos = rho.detach().cpu().numpy()
        Rs = trust_r.detach().cpu().numpy()
        En = E_new.detach().cpu().numpy()

        def head(arr, k=3, fmt="{:.3f}"):
            out = []
            for v in arr[:k]:
                try:
                    out.append(fmt.format(float(v)))
                except Exception:
                    out.append("nan")
            return "[" + ", ".join(out) + "]"

        line = (f"Iter {it}: accepted={acc_list} rejected={rej_list} "
                f"rho_head={head(rhos, 3, '{:.3f}')} "
                f"R_head={head(Rs, 3, '{:.3f}')} "
                f"E_head={head(En, 3, '{:.6f}')}\n")
        return line

    @staticmethod
    def _fmt_orca_cycle_table(it: int,
                              E: torch.Tensor,
                              rho: torch.Tensor,
                              R: torch.Tensor,
                              max_f: torch.Tensor, rms_f: torch.Tensor,
                              max_dp: torch.Tensor, rms_dp: torch.Tensor,
                              f_max_th: torch.Tensor, f_rms_th: torch.Tensor,
                              dp_max_th: torch.Tensor, dp_rms_th: torch.Tensor,
                              done: torch.Tensor) -> str:
        # Header
        lines = []
        lines.append("-" * 70 + "\n")
        lines.append(f"{('RS-PRFO Cycle ' + str(it)).center(70)}\n")
        lines.append("-" * 70 + "\n")
        lines.append("Batch   Energy(Ha)      rho      R(MW)   Max|F|(H/A)  RMS|F|      Max|dX|(A)  RMS|dX|    Conv\n")
        lines.append("-" * 70 + "\n")

        def fmt(v, wid=12, p=6, allow_nan=True):
            x = float(v)
            if not np.isfinite(x) and not allow_nan:
                return f"{'nan':>{wid}}"
            return f"{x:>{wid}.{p}f}"

        B = E.shape[0]
        for b in range(B):
            lines.append(
                f"[{b:>2}] "
                f"{fmt(E[b], wid=14, p=6)} "
                f"{fmt(rho[b], wid=8, p=3)} "
                f"{fmt(R[b],   wid=8, p=3)} "
                f"{fmt(max_f[b], wid=12, p=6)} "
                f"{fmt(rms_f[b], wid=10, p=6)} "
                f"{fmt(max_dp[b], wid=12, p=6)} "
                f"{fmt(rms_dp[b], wid=10, p=6)} "
                f"{('YES' if bool(done[b].item()) else 'NO')}\n"
            )

        lines.append("\n")
        # Also echo thresholds once per cycle for reference (ORCA-like footer)
        lines.append(" Convergence criteria (H/A, Å):\n")
        lines.append(f"   Max |F| ≤ {float(f_max_th.max().item()):.6f}   RMS |F| ≤ {float(f_rms_th.max().item()):.6f}   "
                     f"Max |dX| ≤ {float(dp_max_th.max().item()):.6f}   RMS |dX| ≤ {float(dp_rms_th.max().item()):.6f}\n")
        return "".join(lines)
