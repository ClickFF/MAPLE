# -*- coding: utf-8 -*-
"""
DE-SC-AFIR (Double-Ended informed Single-Component AFIR)
双端知情的单组分AFIR：结合DS-AFIR的目标导向和SC-AFIR的系统性

核心思想：
1. 分析R→P的结构变化（键长、原子对距离）
2. 只对"发生变化"的原子对生成片段并施加人工力
3. 广度优先搜索，找到连接R和P的第一条完整路径即停止
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List, Tuple, Set, Optional
import numpy as np
from ase import Atoms

from .logger import log_info
from ...jobABC import JobABC
from .afir import (
    to_numpy_f64, vec1d, kabsch_align, write_multi_xyz,
    ds_afir_gradient, lqa_step, find_path_extrema,
    redistribute_path_points, lup_optimization
)


@dataclass
class DESCAFIRParams:
    """Parameters for DE-SC-AFIR calculation."""
    # AFIR基本参数
    delta: float = 200.0
    convergence_threshold: float = 0.12
    max_steps_per_path: int = 2000
    step_size: float = 0.05
    
    # 变化检测参数
    distance_threshold: float = 0.5  # Å，超过此值认为原子对"发生变化"
    bond_threshold: float = 0.3      # Å，键长变化超过此值认为"键断裂/形成"
    
    # 搜索控制参数
    max_total_paths: int = 100       # 最多尝试100条AFIR路径
    max_eq_to_explore: int = 20      # 最多探索20个新EQ
    
    # LUP和TS优化参数
    use_lup: bool = True
    lup_iterations_coarse: int = 5
    lup_iterations_fine: int = 3
    lup_interval_coarse: float = 1.0
    lup_interval_fine: float = 0.5
    lup_step_size: float = 0.02
    refine_ts: bool = True
    ts_max_iter: int = 200
    
    # 收敛阈值
    f_max_th: float = 4.5e-4
    f_rms_th: float = 3.0e-4
    dp_max_th: float = 1.8e-3
    dp_rms_th: float = 1.2e-3


def analyze_structural_changes(atoms_R: Atoms, atoms_P: Atoms, 
                               params: DESCAFIRParams) -> List[Tuple[int, int]]:
    """
    分析R→P的结构变化，识别"重要"的原子对
    
    返回：[(k1, l1), (k2, l2), ...]，需要探索的原子对列表
    """
    pos_R = to_numpy_f64(atoms_R.get_positions())
    pos_P = to_numpy_f64(atoms_P.get_positions())
    
    # Kabsch对齐（消除整体旋转/平移）
    pos_P_aligned, rmsd, _, _ = kabsch_align(pos_R, pos_P)
    
    N = len(atoms_R)
    symbols = atoms_R.get_chemical_symbols()
    
    # 获取共价半径
    from ase.data import covalent_radii
    radii = np.array([covalent_radii[atoms_R.numbers[i]] for i in range(N)])
    
    important_pairs = []
    
    log_info([
        f"\n{'='*70}\n",
        f"{'Structural Change Analysis':^70}\n",
        f"{'='*70}\n\n",
        f"RMSD after alignment: {rmsd:.4f} Å\n",
        f"Distance threshold: {params.distance_threshold:.3f} Å\n",
        f"Bond threshold: {params.bond_threshold:.3f} Å\n\n",
        f"{'Atom Pair':>12s} {'Dist(R)':>10s} {'Dist(P)':>10s} {'ΔDist':>10s} {'Status':>15s}\n",
        f"{'-'*70}\n"
    ], "output")
    
    for i in range(N):
        for j in range(i+1, N):
            # 计算R和P中的距离
            r_ij_R = np.linalg.norm(pos_R[i] - pos_R[j])
            r_ij_P = np.linalg.norm(pos_P_aligned[i] - pos_P_aligned[j])
            
            delta_r = abs(r_ij_P - r_ij_R)
            
            # 判断是否成键（在R或P中）
            bond_cutoff = 1.3 * (radii[i] + radii[j])
            is_bonded_R = r_ij_R < bond_cutoff
            is_bonded_P = r_ij_P < bond_cutoff
            
            # 分类变化
            status = ""
            is_important = False
            
            if delta_r > params.distance_threshold:
                # 距离显著变化
                if is_bonded_R and not is_bonded_P:
                    status = "Bond Breaking"
                    is_important = True
                elif not is_bonded_R and is_bonded_P:
                    status = "Bond Forming"
                    is_important = True
                elif is_bonded_R and is_bonded_P and delta_r > params.bond_threshold:
                    status = "Bond Stretch"
                    is_important = True
                else:
                    status = "Distance Change"
                    is_important = True
            
            if is_important:
                important_pairs.append((i, j))
                log_info([
                    f"{symbols[i]}{i}-{symbols[j]}{j:>2d}  "
                    f"{r_ij_R:10.4f}  {r_ij_P:10.4f}  {delta_r:10.4f}  {status:>15s}\n"
                ], "output")
    
    log_info([
        f"\n{'='*70}\n",
        f"Found {len(important_pairs)} important atom pairs\n\n"
    ], "output")
    
    return important_pairs


def generate_fragment_around_pair(atoms: Atoms, k: int, l: int) -> Tuple[List[int], List[int]]:
    """
    基于原子对(k, l)生成片段A和B
    
    算法（论文Scheme 1）：
    1. 收缩k-l距离
    2. 基于连接性分割片段
    """
    pos = to_numpy_f64(atoms.get_positions())
    N = len(atoms)
    symbols = atoms.get_chemical_symbols()
    
    # 获取共价半径
    from ase.data import covalent_radii
    radii = np.array([covalent_radii[atoms.numbers[i]] for i in range(N)])
    
    # 步骤1：创建扰动结构（收缩k-l距离20%）
    pos_perturbed = pos.copy()
    vec_kl = pos[l] - pos[k]
    dist_kl = np.linalg.norm(vec_kl)
    
    # 收缩到80%距离
    target_dist = dist_kl * 0.8
    pos_perturbed[l] = pos[k] + vec_kl / dist_kl * target_dist
    
    # 步骤2：构建连接矩阵（基于扰动结构）
    bond_matrix = np.zeros((N, N), dtype=bool)
    for i in range(N):
        for j in range(i+1, N):
            r_ij = np.linalg.norm(pos_perturbed[i] - pos_perturbed[j])
            if r_ij < 1.5 * (radii[i] + radii[j]):
                bond_matrix[i, j] = True
                bond_matrix[j, i] = True
    
    # 步骤3：从k和l出发，扩展片段（2层）
    fragment_A = {k}
    fragment_B = {l}
    
    # 第1层
    for i in range(N):
        if bond_matrix[k, i]:
            fragment_A.add(i)
        if bond_matrix[l, i]:
            fragment_B.add(i)
    
    # 第2层
    layer1_A = fragment_A.copy()
    layer1_B = fragment_B.copy()
    
    for atom in layer1_A:
        for i in range(N):
            if bond_matrix[atom, i]:
                fragment_A.add(i)
    
    for atom in layer1_B:
        for i in range(N):
            if bond_matrix[atom, i]:
                fragment_B.add(i)
    
    # 步骤4：移除重叠原子（在k和l之间的原子）
    r_kl = np.linalg.norm(pos[l] - pos[k])
    overlap = fragment_A & fragment_B
    for atom in overlap:
        r_ak = np.linalg.norm(pos[atom] - pos[k])
        r_al = np.linalg.norm(pos[atom] - pos[l])
        if r_ak < r_kl and r_al < r_kl:
            # 原子在k和l之间
            fragment_A.discard(atom)
            fragment_B.discard(atom)
    
    return sorted(list(fragment_A)), sorted(list(fragment_B))


class DESCAFIR(JobABC):
    """Double-Ended informed SC-AFIR calculator"""

    def __init__(self, output, atoms_R, atoms_P, paras=None):
        super().__init__(output)
        self.atoms_R = atoms_R.copy()
        self.atoms_P = atoms_P.copy()

        if atoms_R.calc is not None:
            self.atoms_R.calc = atoms_R.calc
            self.atoms_P.calc = atoms_R.calc

        # Initialize params from paras dict
        self.params = self._init_params(DESCAFIRParams, paras,
                                        ("descafir", "de_sc_afir", "DESCAFIR", "ts"))

        # Override convergence thresholds from atoms if available
        for attr in ('f_max_th', 'f_rms_th', 'dp_max_th', 'dp_rms_th'):
            if hasattr(atoms_R, attr):
                setattr(self.params, attr, getattr(atoms_R, attr))

        self.eq_list = []
        self.ts_list = []
        self.important_pairs = []
        
    def _get_energy_and_gradient(self, coords):
        """计算能量和梯度"""
        self.atoms_R.set_positions(coords)
        energy = float(self.atoms_R.get_potential_energy(force_consistent=True))
        gradient = -to_numpy_f64(self.atoms_R.get_forces())
        return energy, gradient
    
    def _make_atoms_copy(self, coords):
        """创建原子副本"""
        atoms = self.atoms_R.copy()
        atoms.set_positions(coords)
        atoms.calc = self.atoms_R.calc
        for attr in ['f_max_th', 'f_rms_th', 'dp_max_th', 'dp_rms_th']:
            if hasattr(self.atoms_R, attr):
                setattr(atoms, attr, getattr(self.atoms_R, attr))
        return atoms
    
    def _structure_match(self, coords1, coords2, threshold=0.1):
        """判断两个结构是否相同（RMSD < threshold）"""
        coords1, coords2 = to_numpy_f64(coords1), to_numpy_f64(coords2)
        _, rmsd, _, _ = kabsch_align(coords1, coords2)
        return rmsd < threshold
    
    def _integrate_afir_path_single(self, eq_coords, fragment_A, fragment_B):
        """
        从一个EQ出发，对特定片段对积分AFIR路径
        返回：路径终点坐标（未优化的）
        """
        q = eq_coords.copy()  # 确保是1D数组
        q = vec1d(q)  # ← 添加这行，转换为1D
        
        path = [q.copy()]
        energies = []
        
        E, g = self._get_energy_and_gradient(q.reshape(-1, 3))  # ← 传3D给计算函数
        energies.append(E)
        
        # 创建片段掩码
        N = len(self.atoms_R)
        mask_A = np.zeros(N, dtype=bool)
        mask_B = np.zeros(N, dtype=bool)
        mask_A[fragment_A] = True
        mask_B[fragment_B] = True
        
        for step in range(self.params.max_steps_per_path):
            E, g = self._get_energy_and_gradient(q.reshape(-1, 3))  # ← 传3D
            g = vec1d(g)  # ← 确保梯度是1D
            
            # 计算AFIR梯度（单组分版本）
            pos = q.reshape(-1, 3)  # ← 临时转3D用于几何计算
            
            # 计算片段质心
            if len(fragment_A) > 0 and len(fragment_B) > 0:
                center_A = pos[fragment_A].mean(axis=0)
                center_B = pos[fragment_B].mean(axis=0)
                
                dist_AB = np.linalg.norm(center_B - center_A)
                if dist_AB < 0.1:  # 片段已经重叠
                    break
                
                direction = (center_B - center_A) / dist_AB
                
                # 施加人工力（简化版）
                force_magnitude = self.params.delta / 100.0  # 转换单位
                artificial_gradient = np.zeros_like(pos)
                
                for i in fragment_A:
                    artificial_gradient[i] = direction * force_magnitude
                for j in fragment_B:
                    artificial_gradient[j] = -direction * force_magnitude
                
                # 组合梯度 - 确保都是1D
                artificial_gradient = vec1d(artificial_gradient)  # ← 转1D
                combined_grad = g + artificial_gradient  # ← 现在形状匹配
            else:
                combined_grad = g
            
            # LQA步进
            grad_norm = np.linalg.norm(combined_grad)
            if grad_norm < 1e-12:
                break
            
            q = q - (combined_grad / grad_norm) * self.params.step_size  # ← 现在都是1D
            path.append(q.copy())
            energies.append(E)
            
            # 检查是否到达局部极小（能量梯度很小）
            if grad_norm < 1e-3 and len(path) > 10:
                break
        
        return q.reshape(-1, 3)  # ← 返回3D坐标
    
    def _optimize_to_minimum(self, coords):
        """优化到最近的局部极小点"""
        from scipy.optimize import minimize
        
        def objective(x):
            E, g = self._get_energy_and_gradient(x)
            return E, g
        
        result = minimize(
            objective,
            coords,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 100, 'ftol': 1e-6}
        )
        
        return result.x
    
    def _breadth_first_search(self):
        """
        广度优先搜索：从R出发，探索所有相关片段，直到找到P
        """
        # 初始化
        eq_queue = [self.atoms_R.get_positions().copy()]  # 待探索的EQ
        visited_structures = [eq_queue[0]]  # 已访问的结构
        
        self.eq_list = [(self._make_atoms_copy(eq_queue[0]), 
                        float(self.atoms_R.get_potential_energy(force_consistent=True)),
                        "Reactant")]
        
        target_P = self.atoms_P.get_positions().copy()
        
        path_count = 0
        
        log_info([
            f"\n{'='*70}\n",
            f"{'Breadth-First Search':^70}\n",
            f"{'='*70}\n\n",
            f"Starting from Reactant...\n",
            f"Target: Product\n",
            f"Important pairs to explore: {len(self.important_pairs)}\n\n"
        ], self.output)
        
        while eq_queue and len(self.eq_list) < self.params.max_eq_to_explore:
            current_eq = eq_queue.pop(0)
            
            log_info([
                f"\n--- Exploring EQ{len(self.eq_list)-1} ---\n"
            ], self.output)
            
            # 对每个重要的原子对尝试AFIR
            for k, l in self.important_pairs:
                if path_count >= self.params.max_total_paths:
                    log_info([f"\nReached max path limit ({self.params.max_total_paths})\n"], 
                            self.output)
                    return False
                
                # 生成片段
                fragment_A, fragment_B = generate_fragment_around_pair(
                    self._make_atoms_copy(current_eq), k, l
                )
                
                log_info([
                    f"  Path {path_count}: atom pair ({k},{l}), "
                    f"fragments: {len(fragment_A)}+{len(fragment_B)} atoms\n"
                ], self.output)
                
                # 积分AFIR路径
                endpoint = self._integrate_afir_path_single(current_eq, fragment_A, fragment_B)
                
                # 优化到局部极小
                try:
                    optimized = self._optimize_to_minimum(endpoint)
                except:
                    log_info([f"    Optimization failed, skipping\n"], self.output)
                    path_count += 1
                    continue
                
                # 检查是否是新结构
                is_new = True
                for visited in visited_structures:
                    if self._structure_match(optimized, visited):
                        is_new = False
                        break
                
                if is_new:
                    visited_structures.append(optimized)
                    eq_queue.append(optimized)
                    
                    E_new = self._get_energy_and_gradient(optimized)[0]
                    self.eq_list.append((
                        self._make_atoms_copy(optimized),
                        E_new,
                        f"EQ{len(self.eq_list)}"
                    ))
                    
                    log_info([f"    → New EQ found! E={E_new:.8f} Eh\n"], self.output)
                    
                    # 检查是否到达目标
                    if self._structure_match(optimized, target_P):
                        log_info([
                            f"\n{'='*70}\n",
                            f"{'SUCCESS: Found path to Product!':^70}\n",
                            f"{'='*70}\n\n"
                        ], self.output)
                        return True
                else:
                    log_info([f"    → Known structure\n"], self.output)
                
                path_count += 1
        
        log_info([
            f"\n{'='*70}\n",
            f"{'Search terminated without finding Product':^70}\n",
            f"{'='*70}\n\n"
        ], self.output)
        return False
    
    def run(self):
        """主运行函数"""
        log_info([
            f"\n{'#'*70}\n",
            f"{'DE-SC-AFIR (Double-Ended informed SC-AFIR)':^70}\n",
            f"{'#'*70}\n\n",
            f"Combining DS-AFIR's goal-directedness with SC-AFIR's systematicity\n\n"
        ], self.output)
        
        base, _ = os.path.splitext(self.output)
        
        # 步骤1：分析结构变化
        self.important_pairs = analyze_structural_changes(
            self.atoms_R, self.atoms_P, self.params
        )
        
        if len(self.important_pairs) == 0:
            log_info([
                f"WARNING: No significant structural changes detected!\n",
                f"R and P may be too similar. Consider:\n",
                f"  1. Decreasing distance_threshold (current: {self.params.distance_threshold})\n",
                f"  2. Checking if R and P are correct\n\n"
            ], self.output)
            return [], []
        
        # 步骤2：广度优先搜索
        found_product = self._breadth_first_search()
        
        # 步骤3：输出结果
        E_R = self.eq_list[0][1]
        kcal = 627.509
        
        log_info([
            f"\n{'='*70}\n",
            f"{'Results':^70}\n",
            f"{'='*70}\n\n",
            f"EQ List:\n{'-'*60}\n",
            f"{'Idx':>4s} {'Label':>12s} {'E(Eh)':>14s} {'dE(kcal)':>12s}\n",
            f"{'-'*60}\n"
        ], self.output)
        
        for i, (atoms, E, label) in enumerate(self.eq_list):
            log_info([
                f"{i:>4d} {label:>12s} {E:14.8f} {(E-E_R)*kcal:12.2f}\n"
            ], self.output)
        
        # 保存结构
        eq_file = base + "_eq_list.xyz"
        eq_comments = [f"EQ{i} {eq[2]} E={eq[1]:.10f}" for i, eq in enumerate(self.eq_list)]
        write_multi_xyz(eq_file, [eq[0] for eq in self.eq_list], comments=eq_comments)
        
        log_info([
            f"\nOutput: {eq_file}\n",
            f"\n{'='*70}\n",
            f"{'DE-SC-AFIR Completed':^70}\n",
            f"{'='*70}\n"
        ], self.output)
        
        return self.eq_list, []