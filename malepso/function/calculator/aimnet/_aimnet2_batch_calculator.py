# -*- coding: utf-8 -*-
import torch
import numpy as np
from ase import Atoms
from typing import List, Dict, Any

from ..calculator_base import CalcBatchABC

# 单位换算：1 Hartree = 27.211386245988 eV
EH2EV = 27.211386245988

# ---------------- helpers ----------------
def pad_dim0(a: torch.Tensor, value=0) -> torch.Tensor:
    pad_shape = list(a.shape)
    pad_shape[0] = 1
    pad_row = torch.full(pad_shape, value, dtype=a.dtype, device=a.device)
    return torch.cat([a, pad_row], dim=0)

def nblist_dense_padded_multi(coord: torch.Tensor, mol_idx: torch.Tensor, cutoff: float) -> torch.Tensor:
    """邻居表 (N+1, M)。仅同一分子内建边；最后一行 sentinel = N。"""
    device = coord.device
    N = coord.shape[0]
    if N == 0:
        return torch.full((1, 1), 0, dtype=torch.int64, device=device)

    diff  = coord[:, None, :] - coord[None, :, :]
    dist2 = (diff * diff).sum(dim=-1)
    same  = (mol_idx[:, None] == mol_idx[None, :])
    eye   = torch.eye(N, dtype=torch.bool, device=device)
    mask  = (dist2 <= cutoff * cutoff) & same & (~eye)

    deg = mask.sum(dim=1)
    M   = int(max(int(deg.max().item()), 1))

    nbmat = torch.full((N + 1, M), N, dtype=torch.int64, device=device)  # sentinel = N
    for i in range(N):
        nei = torch.nonzero(mask[i], as_tuple=False).flatten()
        k = min(nei.numel(), M)
        if k > 0:
            nbmat[i, :k] = nei[:k].to(torch.int64)
    return nbmat

def _ptr_from_atoms(atoms_list: List[Atoms], device) -> torch.Tensor:
    ptr = [0]
    for at in atoms_list:
        ptr.append(ptr[-1] + len(at))
    return torch.tensor(ptr, dtype=torch.long, device=device)

def _build_flat_inputs(atoms_list: List[Atoms], device) -> Dict[str, torch.Tensor]:
    coords_list, numbers_list, molidx_list = [], [], []
    for i, atoms in enumerate(atoms_list):
        # 前向用 float32；后续对外统一转 float64
        pos = torch.tensor(atoms.get_positions(), dtype=torch.float32, device=device)
        Z   = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.int64,  device=device)
        n   = pos.shape[0]
        coords_list.append(pos)
        numbers_list.append(Z)
        molidx_list.append(torch.full((n,), i, dtype=torch.int64, device=device))
    if len(coords_list) == 0:
        coord   = torch.zeros((0, 3), dtype=torch.float32, device=device)
        numbers = torch.zeros((0,),    dtype=torch.int64,  device=device)
        mol_idx = torch.zeros((0,),    dtype=torch.int64,  device=device)
    else:
        coord   = torch.cat(coords_list, dim=0)
        numbers = torch.cat(numbers_list, dim=0)
        mol_idx = torch.cat(molidx_list, dim=0)
    return {"coord": coord, "numbers": numbers, "mol_idx": mol_idx}

# ---------------- batch calculator ----------------
class AIMNet2BatchCalc(CalcBatchABC):
    """
    返回（单位统一为 Hartree / Å）：
      get_forces_and_energy -> (E[B], F[B, 3*Nmax])
      get_efh               -> (E[B], F[B, 3*Nmax], H[B, 3*Nmax, 3*Nmax], P[B])
        其中 H 为 Hartree/Å²，P 为“按原子数的 padding”：P[i] = Nmax_atoms - n_atoms(i)
    """

    def __init__(self, model_path: str, device="cpu", cutoff: float = 5.0):
        super().__init__()
        self.device = torch.device(device)
        self.model  = torch.jit.load(model_path, map_location=self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.cutoff = float(cutoff)

    def build_batch_data(self, atoms_list: List[Atoms]) -> Dict[str, torch.Tensor]:
        flat = _build_flat_inputs(atoms_list, self.device)
        coord, numbers, mol_idx = flat["coord"], flat["numbers"], flat["mol_idx"]
        ptr = _ptr_from_atoms(atoms_list, self.device)
        return {"coord": coord, "numbers": numbers, "mol_idx": mol_idx, "ptr": ptr}

    # -------- E + F --------
    def get_forces_and_energy(self, atoms_list):
        B = len(atoms_list)
        if B == 0:
            E = torch.zeros((0,),   dtype=torch.float64, device=self.device)
            F = torch.zeros((0, 0), dtype=torch.float64, device=self.device)
            return E, F

        data_all = self.build_batch_data(atoms_list)
        coord32 = data_all["coord"].clone().detach().requires_grad_(True)  # float32 for model
        numbers = data_all["numbers"]
        mol_idx = data_all["mol_idx"]
        ptr     = data_all["ptr"]

        N = coord32.shape[0]
        nbmat = nblist_dense_padded_multi(coord32, mol_idx, self.cutoff)
        sentinel_mol = (mol_idx.max().item() + 1) if N > 0 else 0
        charge = torch.zeros(B + 1, dtype=torch.float32, device=self.device)  # 每分子 + 哨兵

        data = {
            "coord":    pad_dim0(coord32, 0.0),
            "numbers":  pad_dim0(numbers, 0).to(torch.int64),
            "charge":   charge,
            "mol_idx":  pad_dim0(mol_idx, sentinel_mol).to(torch.int64),
            "nbmat":    nbmat,
            "nbmat_lr": nbmat,
        }

        with torch.jit.optimized_execution(False):
            out = self.model(data)

        # 能量（模型输出 eV/分子或 eV/原子）。统一聚合到 eV/分子。
        e_vec = out["energy"].to(torch.float32).reshape(-1)
        if e_vec.numel() == N + 1 or e_vec.numel() == B + 1:
            e_vec = e_vec[:-1]
        elif e_vec.numel() not in (N, B):
            raise RuntimeError(f"Unexpected energy shape: {tuple(e_vec.shape)}")

        if e_vec.numel() != B:
            # per-atom -> per-molecule
            E_eV = torch.empty((B,), dtype=torch.float32, device=self.device)
            ptr_cpu = ptr.detach().cpu().numpy()
            for i in range(B):
                s, t = ptr_cpu[i], ptr_cpu[i + 1]
                E_eV[i] = e_vec[s:t].sum()
        else:
            E_eV = e_vec  # (B,)

        # 力：eV/Å
        grad32 = torch.autograd.grad(E_eV.sum(), coord32, create_graph=False, retain_graph=False)[0]
        F_all_eV = -grad32  # (N,3)

        # padding 到每批 3*Nmax
        Nmax_atoms = int(max(len(at) for at in atoms_list)) if B > 0 else 0
        nmax = 3 * Nmax_atoms
        F_eV = torch.zeros((B, nmax), dtype=torch.float32, device=self.device)
        ptr_cpu = ptr.detach().cpu().numpy()
        for i in range(B):
            s, t = ptr_cpu[i], ptr_cpu[i + 1]
            Fi = F_all_eV[s:t, :].reshape(-1)
            F_eV[i, :Fi.numel()] = Fi

        # 转 Ha、Ha/Å（升为 float64）
        E = (E_eV / EH2EV).to(torch.float64)
        F = (F_eV / EH2EV).to(torch.float64)
        return E, F

    # -------- E + F + H + P --------
    def get_efh(self, atoms_list):
        B = len(atoms_list)
        if B == 0:
            E = torch.zeros((0,),   dtype=torch.float64, device=self.device)
            F = torch.zeros((0, 0), dtype=torch.float64, device=self.device)
            H = torch.zeros((0, 0, 0), dtype=torch.float64, device=self.device)
            P = torch.zeros((0,),   dtype=torch.int64,   device=self.device)
            return E, F, H, P

        data_all = self.build_batch_data(atoms_list)
        coord32 = data_all["coord"].clone().detach().requires_grad_(True)  # float32
        numbers = data_all["numbers"]
        mol_idx = data_all["mol_idx"]
        ptr     = data_all["ptr"]

        N = coord32.shape[0]
        nbmat = nblist_dense_padded_multi(coord32, mol_idx, self.cutoff)
        sentinel_mol = (mol_idx.max().item() + 1) if N > 0 else 0
        charge = torch.zeros(B + 1, dtype=torch.float32, device=self.device)

        Nmax_atoms = int(max(len(at) for at in atoms_list)) if B > 0 else 0
        nmax = 3 * Nmax_atoms
        P_atoms = torch.empty((B,), dtype=torch.int64, device=self.device)  # 按原子数的 padding

        data = {
            "coord":    pad_dim0(coord32, 0.0),
            "numbers":  pad_dim0(numbers, 0).to(torch.int64),
            "charge":   charge,
            "mol_idx":  pad_dim0(mol_idx, sentinel_mol).to(torch.int64),
            "nbmat":    nbmat,
            "nbmat_lr": nbmat,
        }

        with torch.jit.optimized_execution(False):
            out = self.model(data)

        # 能量：eV
        e_vec = out["energy"].to(torch.float32).reshape(-1)
        if e_vec.numel() == N + 1 or e_vec.numel() == B + 1:
            e_vec = e_vec[:-1]
        elif e_vec.numel() not in (N, B):
            raise RuntimeError(f"Unexpected energy shape {tuple(e_vec.shape)}")

        if e_vec.numel() != B:
            E_eV = torch.empty((B,), dtype=torch.float32, device=self.device)
            ptr_cpu = ptr.detach().cpu().numpy()
            for i in range(B):
                s, t = ptr_cpu[i], ptr_cpu[i + 1]
                E_eV[i] = e_vec[s:t].sum()
        else:
            E_eV = e_vec

        # 力：eV/Å（保留计算图以便二阶）
        grad32 = torch.autograd.grad(E_eV.sum(), coord32, create_graph=True, retain_graph=True)[0]
        F_all_eV = -grad32  # (N,3)

        # Hessian：eV/Å²，按全局打平坐标求二阶，再拆分为分子块
        f_flat = F_all_eV.reshape(-1)    # 长度 3N
        cols = []
        for k in range(f_flat.numel()):
            g2 = torch.autograd.grad(f_flat[k], coord32, retain_graph=True, create_graph=False)[0]
            cols.append(g2.reshape(-1))
        H_global_eV = -torch.stack(cols, dim=1)  # (3N, 3N)
        # 数值对称化
        H_global_eV = 0.5 * (H_global_eV + H_global_eV.transpose(0, 1))

        # 拆块 + padding；P 为按原子数的 padding
        F_eV = torch.zeros((B, nmax), dtype=torch.float32, device=self.device)
        H_eV = torch.zeros((B, nmax, nmax), dtype=torch.float32, device=self.device)
        ptr_cpu = ptr.detach().cpu().numpy()
        for i in range(B):
            s, t = ptr_cpu[i], ptr_cpu[i + 1]
            ni = t - s                 # 原子数
            dof = 3 * ni
            P_atoms[i] = Nmax_atoms - ni

            F_eV[i, :dof] = F_all_eV[s:t, :].reshape(-1)
            H_eV[i, :dof, :dof] = H_global_eV[3*s:3*t, 3*s:3*t]

        # 统一转 Ha、Ha/Å、Ha/Å²（升为 float64）
        E = (E_eV / EH2EV).to(torch.float64)       # Hartree
        F = (F_eV / EH2EV).to(torch.float64)       # Hartree/Å
        H = (H_eV / EH2EV).to(torch.float64)       # Hartree/Å²
        return E, F, H, P_atoms
