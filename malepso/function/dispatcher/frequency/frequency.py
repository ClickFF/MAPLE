import numpy as np
from typing import Tuple

import torch
from torch import Tensor
from ase import Atoms

from ..jobABC import JobABC

class Frequency(JobABC):

    def __init__(self, output: str, atoms: Atoms):
        super().__init__(output)
        self.atoms = atoms

    def run(self): 
        hessian_matrix = self.get_hessian()
        weighted_hessian = self.MWeightHessian(hessian_matrix)
        frequencies = self.get_frequencies(weighted_hessian)

    def get_hessian(self) -> np.ndarray:
        calc = self.atoms.get_calculator()
        hessian_matrix: Tensor = calc.get_hessian(self.atoms)  # 获取 Hessian 矩阵 
        masses = np.array([atom.mass for atom in self.atoms])  # 获取原子质量数组

        # 转换为 PyTorch tensor，并调整形状
        masses = torch.tensor(masses, dtype=torch.float32).unsqueeze(0)  # 形状 (1, 22)
        
        return hessian_matrix.numpy()

    def MWeightHessian(self, hessian_matrix: np.ndarray) -> np.ndarray:
        # 获取每个原子的质量并转换为 PyTorch tensor
        masses = np.array([atom.mass for atom in self.atoms])
        
        # 将质量转换为 PyTorch tensor，并对质量取平方根的倒数
        inv_sqrt_masses = torch.tensor(1 / np.sqrt(masses), dtype=torch.float32)
        
        # 使用 repeat_interleave 重复质量 3 次，适应每个原子的 x, y, z 方向自由度
        inv_sqrt_masses = inv_sqrt_masses.repeat_interleave(3)  # 形状变为 (3 * num_atoms,)
        
        # 将 Hessian 矩阵转换为 PyTorch tensor
        hessian_tensor = torch.tensor(hessian_matrix, dtype=torch.float32)

        # 对 Hessian 矩阵逐元素进行质权化操作
        mass_weighted_hessian = hessian_tensor * inv_sqrt_masses.unsqueeze(0) * inv_sqrt_masses.unsqueeze(1)

        # 将结果转换回 NumPy 数组并返回
        return mass_weighted_hessian.numpy()

    def get_frequencies(self, hessian_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # 获取每个原子的质量并计算质量的平方根倒数
        masses = np.array([atom.mass for atom in self.atoms])
        inv_sqrt_masses = 1 / np.sqrt(masses)  # (num_atoms,)
        inv_sqrt_masses = np.repeat(inv_sqrt_masses, 3)  # 重复质量 3 次，适应每个自由度

        # 计算 Hessian 矩阵的特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eigh(hessian_matrix)
        
        # 频率计算，单位转换因子
        conversion_factor = 17091.7006789297
        frequencies = np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues)) * conversion_factor
        frequencies = frequencies.flatten() / (2 * np.pi)

        # 如果 eigenvectors 是三维的 (1, 66, 66)，则移除多余的维度
        if eigenvectors.ndim == 3 and eigenvectors.shape[0] == 1:
            eigenvectors = eigenvectors.squeeze(0)  # 将 (1, 66, 66) 变为 (66, 66)

        # 计算 MWN (Mass Weighted Normalized) 模式
        mw_normalized = eigenvectors.T  # 转置以获取每一列作为模式向量

        # 去质量权重，得到 MDU (Mass Deweighted Unnormalized) 模式
        md_unnormalized = mw_normalized * inv_sqrt_masses

        # 归一化处理，得到 MDN (Mass Deweighted Normalized) 模式
        norm_factors = 1 / np.linalg.norm(md_unnormalized, axis=1)
        md_normalized = md_unnormalized * norm_factors[:, np.newaxis]  # 归一化模式

        # 获取原子数量
        num_atoms = self.atoms.get_number_of_atoms()

        # 准备输出信息
        info_message = ['\n' + '-' * 70 + '\n', f'{"Frequency".center(70)}\n\n']

        for i, freq in enumerate(frequencies):
            freq_value = float(freq)
            info_message.append(f"Frequency {i+1}: {freq_value:>12.6f} cm^-1\n")
            info_message.append("Eigenvectors       X          Y          Z\n")

            for j in range(num_atoms):
                # 提取原子 j 的 X, Y, Z 分量
                x, y, z = md_normalized[i, 3*j], md_normalized[i, 3*j+1], md_normalized[i, 3*j+2]
                info_message.append(f"Atom {j+1:>3}: {x:>10.6f} {y:>10.6f} {z:>10.6f}\n")

            info_message.append('\n')

        # 记录信息
        self.log_info(info_message)

        return frequencies, md_normalized
