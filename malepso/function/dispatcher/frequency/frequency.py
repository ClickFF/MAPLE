import numpy as np
from typing import Tuple

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
        hessian_matrix:Tensor = calc.get_hessian(self.atoms)
        return hessian_matrix.numpy()

    def MWeightHessian(self, hessian_matrix: np.ndarray) -> np.ndarray:

        masses = np.array([atom.mass for atom in self.atoms])
        masses = np.repeat(masses, 3)
        sqrt_masses = np.sqrt(masses)
        mass_weight_matrix = np.diag(sqrt_masses)
        weighted_hessian = mass_weight_matrix @ hessian_matrix @ mass_weight_matrix
        
        return weighted_hessian

    def get_frequencies(self, hessian_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # 计算 Hessian 矩阵的特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eigh(hessian_matrix)
        
        # 频率计算，单位转换因子
        conversion_factor = 5140.48  
        frequencies = np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues)) * conversion_factor
        frequencies = frequencies.flatten()

        # 如果 eigenvectors 是三维的 (1, 66, 66)，则移除多余的维度
        if eigenvectors.ndim == 3 and eigenvectors.shape[0] == 1:
            eigenvectors = eigenvectors.squeeze(0)  # 将 (1, 66, 66) 变为 (66, 66)
        
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
                x, y, z = eigenvectors[i, 3*j], eigenvectors[i, 3*j+1], eigenvectors[i, 3*j+2]
                info_message.append(f"Atom {j+1:>3}: {x:>10.6f} {y:>10.6f} {z:>10.6f}\n")

            info_message.append('\n')

        # 记录信息
        self.log_info(info_message)

        return frequencies, eigenvectors