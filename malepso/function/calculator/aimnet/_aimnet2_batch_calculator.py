# -*- coding: utf-8 -*-
import torch
from typing import List
from ase import Atoms
import numpy as np

# 单位换算常量
EH2EV = 27.211386245988  # 1 Ha = 27.211386... eV


# ---------------- helpers ----------------
# 正确的 pad_dim0（保持 64 精度，由 a.dtype 决定）
def pad_dim0(a: torch.Tensor, value=0) -> torch.Tensor:
    pad_shape = list(a.shape); pad_shape[0] = 1
    pad_row = torch.full(pad_shape, value, dtype=a.dtype, device=a.device)
    return torch.cat([a, pad_row], dim=0)


def nblist_dense_padded_multi(coord: torch.Tensor, mol_idx: torch.Tensor, cutoff: float) -> torch.Tensor:
    """
    稳定邻居表 (N+1, M)。仅同一分子内建边；最后一行 sentinel = N。
    邻居按距离排序；M=每原子最大度。
    """
    device = coord.device
    dtype  = coord.dtype
    N = coord.shape[0]
    if N == 0:
        return torch.full((1, 1), 0, dtype=torch.int64, device=device)

    diff  = coord[:, None, :] - coord[None, :, :]
    dist2 = (diff * diff).sum(dim=-1)                  # (N,N) dtype
    same  = (mol_idx[:, None] == mol_idx[None, :])     # (N,N) bool
    eye   = torch.eye(N, dtype=torch.bool, device=device)
    mask  = (dist2 <= cutoff * cutoff) & same & (~eye) # (N,N)

    deg = mask.sum(dim=1)                              # (N,)
    M   = int(max(int(deg.max().item()), 1))

    # 距离排序，非邻居设置为大数
    big = torch.finfo(dtype).max / 4.0
    sort_key = torch.where(mask, dist2, dist2.new_full(dist2.shape, big))
    order = torch.argsort(sort_key, dim=1, stable=True)  # (N,N)

    nbmat = torch.full((N + 1, M), N, dtype=torch.int64, device=device)
    for i in range(N):
        ki = int(deg[i].item())
        if ki > 0:
            k = min(ki, M)
            nbmat[i, :k] = order[i, :k]
    return nbmat

def _ptr_from_atoms(atoms_list: List[Atoms], device) -> torch.Tensor:
    ptr = [0]
    for at in atoms_list:
        ptr.append(ptr[-1] + len(at))
    return torch.tensor(ptr, dtype=torch.long, device=device)


# ---------------- main calculator ----------------
class AIMNet2BatchCalc:
    """
    AIMNet2 批量计算器（GPU 常驻，dtype 可控，默认 float64）
    - prepare() 仅一次从 ASE 读拓扑与首帧坐标，缓存到 GPU
    - 坐标缓冲 self.coord: (N,3) [dtype] on GPU，Å
    - 接口：
        set_coords_(coord) / step_cart_(s_cart) / backup_coords / restore_coords
        get_ef_gpu() / get_efh_gpu()
        ef_from_coords(coord) / efh_from_coords(coord)
    - 返回单位：E Ha，F Ha/Å，H Ha/Å^2，P 每分子 padding 原子数
    """

    def __init__(self, model_path: str, device: str = "cuda", cutoff: float = 5.0, dtype: torch.dtype = torch.float64):
        self.device = torch.device(device)
        self.dtype  = dtype               # 全局数据类型控制点
        self.model  = torch.jit.load(model_path, map_location=self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.cutoff = float(cutoff)

        # 缓存（prepare 后有效）
        self._prepared    = False
        self._atoms_B     = 0
        self._ptr         = None      # (B+1,)
        self.numbers      = None      # (N,) int64
        self.mol_idx      = None      # (N,) int64
        self.coord        = None      # (N,3) dtype on GPU
        self.N_atoms      = 0
        self.Nmax_atoms   = 0
        self.nmax_dof     = 0
        self.sentinel_mol = 0
        self.charge       = None      # (B+1,) dtype

        self._coord_backup = None

    # ---------- 一次性准备：仅首帧需要 ----------
    def prepare(self, atoms_list: List[Atoms]):
        device, dtype = self.device, self.dtype
        self._atoms_B = len(atoms_list)
        self._ptr     = _ptr_from_atoms(atoms_list, device)  # (B+1,)

        # 常量拓扑
        nums, mids = [], []
        for i, at in enumerate(atoms_list):
            Z = torch.tensor(at.get_atomic_numbers(), dtype=torch.int64, device=device)
            n = Z.shape[0]
            nums.append(Z)
            mids.append(torch.full((n,), i, dtype=torch.int64, device=device))
        self.numbers = torch.cat(nums, dim=0) if nums else torch.zeros((0,), dtype=torch.int64, device=device)
        self.mol_idx = torch.cat(mids, dim=0) if mids else torch.zeros((0,), dtype=torch.int64, device=device)

        # 尺寸
        self.N_atoms     = int(self.numbers.numel())
        self.Nmax_atoms  = int(max((len(at) for at in atoms_list), default=0))
        self.nmax_dof    = 3 * self.Nmax_atoms

        # 坐标缓冲（GPU，统一 dtype）
        if self.N_atoms > 0:
            pos_list = [torch.tensor(at.get_positions(), dtype=dtype) for at in atoms_list]
            coord0   = torch.cat(pos_list, dim=0)  # (N,3)
        else:
            coord0   = torch.zeros((0, 3), dtype=dtype)
        self.coord = coord0.to(device, non_blocking=True).contiguous()

        # 其它常量
        self.sentinel_mol = (int(self.mol_idx.max().item()) + 1) if self.N_atoms > 0 else 0
        self.charge       = torch.zeros(self._atoms_B + 1, dtype=dtype, device=device)

        self._coord_backup = None
        self._prepared     = True

    # ---------- 坐标管理，全在 GPU ----------
    @torch.no_grad()
    def set_coords_(self, coord: torch.Tensor):
        """覆盖内部坐标。coord: (N,3)，任意设备/精度，将复制到 GPU 并转为全局 dtype。"""
        assert self._prepared, "call prepare() first"
        assert coord.shape == (self.N_atoms, 3)
        self.coord.copy_(coord.to(self.device, dtype=self.dtype))

    @torch.no_grad()
    def step_cart_(self, s_cart: torch.Tensor):
        """按批量位移更新坐标。s_cart: (B, 3*Nmax) Å。"""
        assert self._prepared, "call prepare() first"
        B = self._atoms_B
        assert s_cart.shape == (B, self.nmax_dof)
        s_cart = s_cart.to(self.device, dtype=self.dtype)
        s = self._ptr[:-1]
        t = self._ptr[1:]
        for i in range(B):
            ni = int((t[i] - s[i]).item())
            if ni > 0:
                self.coord[s[i]:t[i], :].add_(s_cart[i, :3*ni].reshape(ni, 3))

    @torch.no_grad()
    def backup_coords(self):
        if self._prepared:
            self._coord_backup = self.coord.clone()

    @torch.no_grad()
    def restore_coords(self):
        if self._coord_backup is not None:
            self.coord.copy_(self._coord_backup)
            self._coord_backup = None

    # ---------- 内部公共前向 ----------
    def _forward_energy_forces_(self, c: torch.Tensor, need_graph: bool):
        """
        c: (N,3) [dtype, GPU]
        返回:
          E_eV:      (B,)  eV/分子（保持对 c 的梯度）
          F_all_eV:  (N,3) eV/Å
          coord_leaf:      作为叶子变量的坐标（供二阶继续用）
        """
        assert self._prepared, "call prepare() first"
        device, dtype = self.device, self.dtype
        B = self._atoms_B
        N = self.N_atoms

        # 叶子变量
        coord_leaf = c.detach().to(device=device, dtype=dtype).requires_grad_(True)

        # 邻表（同 dtype）
        nbmat = nblist_dense_padded_multi(coord_leaf, self.mol_idx, self.cutoff)

        # 打包模型输入：
        # 注意：这里严格使用全局 dtype，把 charge、coord、nb 等统一 dtype/int。
        data = {
            "coord":    pad_dim0(coord_leaf, 0.0),                     # (N+1,3) float64
            "numbers":  pad_dim0(self.numbers, 0).to(torch.int64),     # (N+1,)
            "charge":   self.charge,                                    # (B+1,)
            "mol_idx":  pad_dim0(self.mol_idx, self.sentinel_mol).to(torch.int64),
            "nbmat":    nbmat,
            "nbmat_lr": nbmat,
        }

        # 前向
        with torch.jit.optimized_execution(False):
            out = self.model(data)

        # 能量向量，转全局 dtype
        e_vec = out["energy"].to(dtype).reshape(-1)
        if e_vec.numel() == N + 1 or e_vec.numel() == B + 1:
            e_vec = e_vec[:-1]

        if e_vec.numel() == B:
            E_eV = e_vec
        elif e_vec.numel() == N:
            E_eV = torch.bincount(self.mol_idx, weights=e_vec, minlength=B).to(dtype)
        else:
            raise RuntimeError(f"Unexpected energy shape {tuple(e_vec.shape)}")

        # 一阶力
        grad = torch.autograd.grad(E_eV.sum(), coord_leaf,
                                   create_graph=need_graph, retain_graph=need_graph)[0]
        F_all_eV = -grad  # (N,3), dtype

        return E_eV, F_all_eV, coord_leaf

    # ---------- 计算接口：使用内部坐标 ----------
    def get_ef_gpu(self):
        """
        返回：
          E: (B,)        Ha
          F: (B,3*Nmax)  Ha/Å
        """
        B = self._atoms_B
        device, dtype = self.device, self.dtype
        if B == 0:
            E = torch.zeros((0,), dtype=dtype, device=device)
            F = torch.zeros((0, 0), dtype=dtype, device=device)
            return E, F

        E_eV, F_all_eV, _ = self._forward_energy_forces_(self.coord, need_graph=False)

        # 装配到 (B, 3*Nmax)
        nmax = self.nmax_dof
        F_eV = torch.zeros((B, nmax), dtype=dtype, device=device)
        s = self._ptr[:-1]
        t = self._ptr[1:]
        for i in range(B):
            ni = int((t[i] - s[i]).item())
            if ni > 0:
                F_eV[i, :3*ni] = F_all_eV[s[i]:t[i], :].reshape(-1)

        # eV -> Ha
        E = (E_eV / EH2EV)
        F = (F_eV / EH2EV)
        return E, F

    def get_efh_gpu(self):
        """
        返回：
          E: (B,)                 Ha
          F: (B, 3*Nmax)         Ha/Å
          H: (B, 3*Nmax,3*Nmax)  Ha/Å²
          P: (B,)                每分子 padding 原子数
        """
        B = self._atoms_B
        device, dtype = self.device, self.dtype
        if B == 0:
            E = torch.zeros((0,),   dtype=dtype, device=device)
            F = torch.zeros((0, 0), dtype=dtype, device=device)
            H = torch.zeros((0, 0, 0), dtype=dtype, device=device)
            P = torch.zeros((0,),   dtype=torch.int64, device=device)
            return E, F, H, P

        # 一阶
        E_eV, F_all_eV, coord_leaf = self._forward_energy_forces_(self.coord, need_graph=True)

        # 二阶：全局 Hessian（eV/Å²）
        f_flat = F_all_eV.reshape(-1)
        cols = []
        for k in range(f_flat.numel()):
            g2 = torch.autograd.grad(f_flat[k], coord_leaf, retain_graph=True, create_graph=False)[0]
            cols.append(g2.reshape(-1))
        H_global_eV = -torch.stack(cols, dim=1)  # (3N,3N)
        H_global_eV = 0.5 * (H_global_eV + H_global_eV.transpose(0, 1))

        # 分块 + padding + P
        nmax = self.nmax_dof
        F_eV = torch.zeros((B, nmax), dtype=dtype, device=device)
        H_eV = torch.zeros((B, nmax, nmax), dtype=dtype, device=device)
        P    = torch.empty((B,), dtype=torch.int64, device=device)

        s = self._ptr[:-1]
        t = self._ptr[1:]
        for i in range(B):
            ni  = int((t[i] - s[i]).item())
            dof = 3 * ni
            P[i] = self.Nmax_atoms - ni
            if dof > 0:
                F_eV[i, :dof]       = F_all_eV[s[i]:t[i], :].reshape(-1)
                H_eV[i, :dof, :dof] = H_global_eV[3*s[i]:3*t[i], 3*s[i]:3*t[i]]

        # eV -> Ha
        E = (E_eV / EH2EV)
        F = (F_eV / EH2EV)
        H = (H_eV / EH2EV)
        return E, F, H, P

    # ---------- 计算接口：传入坐标但不改内部状态 ----------
    def ef_from_coords(self, coord: torch.Tensor):
        """用外部坐标计算 E/F，不写回内部缓存。"""
        assert self._prepared, "call prepare() first"
        device, dtype = self.device, self.dtype
        coord = coord.to(device, dtype=dtype)

        E_eV, F_all_eV, _ = self._forward_energy_forces_(coord, need_graph=False)

        B, nmax = self._atoms_B, self.nmax_dof
        F_eV = torch.zeros((B, nmax), dtype=dtype, device=device)
        s = self._ptr[:-1]
        t = self._ptr[1:]
        for i in range(B):
            ni = int((t[i] - s[i]).item())
            if ni > 0:
                F_eV[i, :3*ni] = F_all_eV[s[i]:t[i], :].reshape(-1)

        E = (E_eV / dtype(EH2EV))
        F = (F_eV / dtype(EH2EV))
        return E, F

    def efh_from_coords(self, coord: torch.Tensor):
        """用外部坐标计算 E/F/H/P，不写回内部缓存。"""
        assert self._prepared, "call prepare() first"
        device, dtype = self.device, self.dtype
        coord = coord.to(device, dtype=dtype)

        E_eV, F_all_eV, coord_leaf = self._forward_energy_forces_(coord, need_graph=True)

        f_flat = F_all_eV.reshape(-1)
        cols = []
        for k in range(f_flat.numel()):
            g2 = torch.autograd.grad(f_flat[k], coord_leaf, retain_graph=True, create_graph=False)[0]
            cols.append(g2.reshape(-1))
        H_global_eV = -torch.stack(cols, dim=1)
        H_global_eV = 0.5 * (H_global_eV + H_global_eV.transpose(0, 1))

        B, nmax = self._atoms_B, self.nmax_dof
        F_eV = torch.zeros((B, nmax), dtype=dtype, device=device)
        H_eV = torch.zeros((B, nmax, nmax), dtype=dtype, device=device)
        P    = torch.empty((B,), dtype=torch.int64, device=device)

        s = self._ptr[:-1]
        t = self._ptr[1:]
        for i in range(B):
            ni  = int((t[i] - s[i]).item())
            dof = 3 * ni
            P[i] = self.Nmax_atoms - ni
            if dof > 0:
                F_eV[i, :dof]       = F_all_eV[s[i]:t[i], :].reshape(-1)
                H_eV[i, :dof, :dof] = H_global_eV[3*s[i]:3*t[i], 3*s[i]:3*t[i]]

        E = (E_eV / EH2EV)
        F = (F_eV / EH2EV)
        H = (H_eV / EH2EV)
        return E, F, H, P
