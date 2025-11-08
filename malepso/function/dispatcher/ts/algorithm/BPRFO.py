# -*- coding: utf-8 -*-
from typing import List, Optional
import os
import numpy as np
import torch
from ase import Atoms

DTYPE = torch.float64
BIG = 1e8


def _ptr_from_atoms(at_list, device):
    ptr = [0]
    for at in at_list:
        ptr.append(ptr[-1] + len(at))
    return torch.tensor(ptr, dtype=torch.long, device=device)


def _masses_flat(at_list, nmax, device):
    B = len(at_list)
    mass = torch.ones((B, nmax), dtype=DTYPE, device=device)
    for b, at in enumerate(at_list):
        m = np.asarray(at.get_masses(), dtype=np.float64)
        mass[b, :3 * len(at)] = torch.from_numpy(np.repeat(m, 3)).to(device=device, dtype=DTYPE)
    return mass


def _symbols_flat(at_list):
    return [at.get_chemical_symbols() for at in at_list]


def _get_coord_gpu(calc) -> torch.Tensor:
    # 统一取 GPU 上的坐标缓冲，dtype=DTYPE
    if hasattr(calc, "coord"):
        return calc.coord
    elif hasattr(calc, "coord32"):
        return calc.coord32
    else:
        raise AttributeError("calculator has neither 'coord' nor 'coord32' coordinate buffer")


class BatchPRFO:
    """
    RS-PRFO（老版本算法完全保持），全 64 位：
      - EFH 每外迭都计算
      - 内层尝试用 EF 评估 rho
      - 信任域与模式跟踪逻辑不变
    仅优化数据流：全在 GPU；通过 calc.step_cart_ 加步；写盘时再转 CPU。
    依赖 calc 接口：
      prepare(atoms), backup_coords(), restore_coords(), step_cart_(B,3*Nmax),
      get_ef_gpu()->(E,F), get_efh_gpu()->(E,F,H,P), 坐标缓冲属性 coord/coord32
    单位：E Ha, F Ha/Å, H Ha/Å²
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

        # 模式跟踪
        self.tracked_mode_vec_mw: Optional[torch.Tensor] = None  # (B,n)
        self.tracked_mode_idx: Optional[torch.Tensor] = None     # (B,)

        # 缓存拓扑
        self._ptr = None
        self._L_vec = None
        self._nmax = 0
        self._B = 0
        self._symbols_per_batch = None
        self._arange_n = None
        self._real_mask = None
        self._D = None

    # ====================== 主流程 ======================

    def run(self, mols) -> None:
        device = self.device
        atoms_list: List[Atoms] = mols.multiatoms
        calc = mols.calc
        B = len(atoms_list)
        self._B = B

        # 一次性准备（ASE -> GPU，calc 内部应为 float64）
        calc.prepare(atoms_list)

        # 固定拓扑与尺寸
        self._ptr = _ptr_from_atoms(atoms_list, device)  # (B+1,)
        L_list = [3 * len(at) for at in atoms_list]
        self._L_vec = torch.tensor(L_list, dtype=torch.int64, device=device)
        self._nmax = int(max(L_list) if L_list else 0)
        self._symbols_per_batch = _symbols_flat(atoms_list)
        self._arange_n = torch.arange(self._nmax, device=device)

        # 打开日志与 XYZ
        self._open_log()
        self._w("# RS-PRFO batched TS search start\n")
        self._init_xyz_paths(B)
        self._dump_xyz_all(calc)

        # 质量与信任域
        mass = _masses_flat(atoms_list, self._nmax, device)                 # (B,n)
        self._D = 1.0 / torch.sqrt(torch.clamp(mass, min=1e-12))            # (B,n)
        trust_r = torch.full((B,), self.trust_init, dtype=DTYPE, device=device)
        active = torch.ones(B, dtype=torch.bool, device=device)

        # 预热获取 n
        _, F0 = calc.get_ef_gpu()
        n = F0.shape[1]
        assert n == self._nmax
        self._real_mask = (self._arange_n[None, :] < self._L_vec[:, None])  # (B,n)
        arange_n = self._arange_n

        self.tracked_mode_vec_mw = None
        self.tracked_mode_idx = None

        last_step = torch.zeros((B, self._nmax), dtype=DTYPE, device=device)

        for it in range(1, self.max_outer_iter + 1):
            # 备份坐标
            calc.backup_coords()

            # EFH（每轮都算）
            E_old, F_raw, H_raw, _ = calc.get_efh_gpu()
            # 保证 dtype=DTYPE（calc 已 64，以下转型为幂等）
            E_old = E_old.to(dtype=DTYPE)
            F_raw = F_raw.to(dtype=DTYPE)
            H_raw = H_raw.to(dtype=DTYPE)
            H_raw = 0.5 * (H_raw + H_raw.transpose(-1, -2))

            real_mask = self._real_mask
            mask_ij = (real_mask.unsqueeze(-1) & real_mask.unsqueeze(-2)).to(DTYPE)
            H = H_raw * mask_ij
            g_cart = -F_raw * real_mask.to(DTYPE)                              # (B,n)

            # 质量加权
            g_mw = self._D * g_cart
            H_mw = self._D.unsqueeze(-1) * H * self._D.unsqueeze(-2)

            # 用 BIG 隔离 padding 维
            pad_mask = ~real_mask
            if pad_mask.any():
                H_mw = H_mw.clone()
                diag = H_mw[..., arange_n, arange_n]
                H_mw[..., arange_n, arange_n] = diag + pad_mask.to(DTYPE) * BIG
                g_mw = g_mw * (~pad_mask).to(DTYPE)

            # 本征分解
            w, V = torch.linalg.eigh(H_mw)                                    # (B,n),(B,n,n)
            gp = (V.transpose(-1, -2) @ g_mw.unsqueeze(-1)).squeeze(-1)       # (B,n)

            # 极小截断，保持数值稳定
            tiny = w.abs() < 1e-10
            w = torch.where(tiny & (w == 0), torch.full_like(w, 1e-10), w)
            w = torch.where(tiny & (w != 0), torch.sign(w) * 1e-10, w)

            # 模式跟踪
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

            # ===== RS 内层尝试 =====
            # ===== RS 内层尝试 =====
            for _try in range(self.max_inner_attempts):
                pend = active & (~accepted)
                if not pend.any():
                    break

                # μ=0 无约束步
                s_unc_minus = torch.zeros_like(gp)
                s_unc_plus  = torch.zeros_like(gp)

                denom_m0 = -w.masked_select(minus_mask)
                denom_m0 = torch.where(denom_m0.abs() < 1e-10, torch.sign(denom_m0) * 1e-10, denom_m0)
                s_unc_minus[minus_mask] = -(-gp[minus_mask]) / denom_m0

                denom_p0 = w.masked_select(plus_mask)
                denom_p0 = torch.where(denom_p0.abs() < 1e-10, torch.sign(denom_p0) * 1e-10, denom_p0)
                s_unc_plus[plus_mask] = -(gp[plus_mask]) / denom_p0

                norm2_minus = (s_unc_minus ** 2).sum(-1)
                norm2_plus  = (s_unc_plus  ** 2).sum(-1)
                total_unc   = norm2_minus + norm2_plus
                alpha = torch.where(total_unc > 0, norm2_minus / total_unc,
                                    torch.full_like(total_unc, 0.5)).clamp(0.05, 0.95)
                R2 = trust_r ** 2
                R2_minus = alpha * R2
                R2_plus  = (1.0 - alpha) * R2

                mu_minus, s_part_minus = self._solve_mu_batched(w, gp, minus_mask, R2_minus, sigma=-1, only=pend)
                mu_plus,  s_part_plus  = self._solve_mu_batched(w, gp, plus_mask,  R2_plus,  sigma=+1, only=pend)

                s_p    = s_part_minus + s_part_plus                         # (B,n) eigen
                norm_mw = torch.linalg.norm(s_p, dim=-1)
                s_mw   = (V @ s_p.unsqueeze(-1)).squeeze(-1) * real_mask    # (B,n)
                s_cart = self._D * s_mw                                     # (B,n)

                # === 评估 rho：只对 pend 临时加步，评估后还原到当前基线 ===
                s_try = torch.zeros_like(s_cart)
                s_try[pend] = s_cart[pend]

                calc.backup_coords()             # 临时备份当前“基线”（已包含前次接受步）
                calc.step_cart_(s_try)           # 加试探步
                E_new, _ = calc.get_ef_gpu()     # 计算新能量
                E_new = E_new.to(dtype=DTYPE)
                calc.restore_coords()            # 回到基线

                # 模型一致性
                Hs = torch.einsum('bij,bj->bi', H, s_try)
                model_change  = (g_cart * s_try).sum(-1) + 0.5 * (s_try * Hs).sum(-1)
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

                # === 提交：仅对 acc 永久加步；不再 restore，从而基线前移 ===
                if acc.any():
                    s_commit = torch.zeros_like(s_cart)
                    s_commit[acc] = s_cart[acc]
                    calc.step_cart_(s_commit)                     # 基线更新
                    grow = (rho > self.eta_expand) & on_boundary & acc
                    trust_r = torch.where(grow,
                                        torch.clamp(2.0 * trust_r, max=self.trust_max),
                                        trust_r)
                    last_step[acc] = s_commit[acc]
                    self._dump_xyz_subset(calc, acc, it)

                # 对 rej 仅收缩半径；无需回滚，因为我们没把 rej 步写入基线
                trust_r = torch.where(rej, torch.clamp(0.5 * trust_r, min=self.trust_min), trust_r)

                self._w(self._fmt_iter_head(it, acc, rej, rho, trust_r, E_new))
                accepted |= acc


            # ===== 收敛判据 =====
            E_final, F_final = calc.get_ef_gpu()
            F_final = F_final.to(dtype=DTYPE)
            g_last = -F_final * real_mask.to(DTYPE)

            f_max_th = torch.tensor([getattr(at, 'f_max_th', 2e-3) for at in atoms_list], dtype=DTYPE, device=device)
            f_rms_th = torch.tensor([getattr(at, 'f_rms_th', 1e-3) for at in atoms_list], dtype=DTYPE, device=device)
            dp_max_th = torch.tensor([getattr(at, 'dp_max_th', 1e-3) for at in atoms_list], dtype=DTYPE, device=device)
            dp_rms_th = torch.tensor([getattr(at, 'dp_rms_th', 5e-4) for at in atoms_list], dtype=DTYPE, device=device)

            L_eff = (self._L_vec.clamp(min=1)).to(DTYPE)
            max_f = g_last.abs().amax(dim=-1)
            rms_f = torch.sqrt((g_last ** 2).sum(-1) / L_eff)
            max_dp = last_step.abs().amax(dim=-1)
            rms_dp = torch.sqrt((last_step ** 2).sum(-1) / L_eff)

            done = (max_f <= f_max_th) & (rms_f <= f_rms_th) & (max_dp <= dp_max_th) & (rms_dp <= dp_rms_th)

            self._w(self._fmt_orca_cycle_table(
                it=it,
                E=E_final.to(dtype=DTYPE),
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
        # 最终坐标在 calc 的坐标缓冲中

    # ====================== μ 求解 ======================

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

    # ====================== 日志与 XYZ ======================

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

    def _dump_xyz_all(self, calc, tag: str = "init"):
        with torch.no_grad():
            pos = _get_coord_gpu(calc).detach().to('cpu').numpy()
        ptr = self._ptr.detach().cpu().numpy()
        for i in range(self._B):
            s, t = ptr[i], ptr[i+1]
            self._append_xyz(i, self._symbols_per_batch[i], pos[s:t], comment=tag)

    def _dump_xyz_subset(self, calc, accept_mask: torch.Tensor, it: int):
        with torch.no_grad():
            pos = _get_coord_gpu(calc).detach().to('cpu').numpy()
        ptr = self._ptr.detach().cpu().numpy()
        for i in range(self._B):
            if bool(accept_mask[i].item()):
                s, t = ptr[i], ptr[i+1]
                self._append_xyz(i, self._symbols_per_batch[i], pos[s:t], comment=f"iter={it}")

    def _append_xyz(self, idx: int, symbols: List[str], pos_np: np.ndarray, comment: str = ""):
        path = self.xyz_paths[idx]
        n = pos_np.shape[0]
        with open(path, "a") as f:
            f.write(f"{n}\n")
            f.write(f"{comment}\n")
            for k in range(n):
                x, y, z = pos_np[k]
                f.write(f"{symbols[k]:<2s} {x:20.10f} {y:20.10f} {z:20.10f}\n")
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
        lines.append(" Convergence criteria (H/A, Å):\n")
        lines.append(f"   Max |F| ≤ {float(f_max_th.max().item()):.6f}   RMS |F| ≤ {float(f_rms_th.max().item()):.6f}   "
                     f"Max |dX| ≤ {float(dp_max_th.max().item()):.6f}   RMS |dX| ≤ {float(dp_rms_th.max().item()):.6f}\n")
        return "".join(lines)
