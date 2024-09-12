import numpy as np
from typing import Tuple

from .logger import *

from ase import Atoms

def Newton(atoms: Atoms, output: str):
    
    step_size = 0.1
    g_au = 27.211386024367243
    info_message = ['\n' + '-' * 70 + '\n', f'{"Newton Optimization".center(70)}\n\n']

    iteration = 0
    maxiteration = 256
    converged = False

    trust_max = 0.5
    trust0 = 0.05
    trust_radius = trust0
    
    while not converged and iteration < maxiteration:
        iteration += 1
        current_force = atoms.get_forces()
        current_crd = atoms.get_positions()
        current_force_flat = current_force.flatten()
        calc = atoms.get_calculator()
        hessian = calc.get_hessian(atoms).numpy()

        # 计算Hessian的特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eigh(hessian)
        
        # 找到最小的特征值及其对应的特征向量
        min_eigenvalue_index = np.argmin(eigenvalues)
        min_eigenvalue = eigenvalues[min_eigenvalue_index]
        min_eigenvector = eigenvectors[:, min_eigenvalue_index]

        if min_eigenvalue < 0:
            # 过渡态优化时沿着负特征值方向移动
            step = step_size * -min_eigenvector
        else:

            step = step_size * (-np.linalg.solve(hessian, current_force_flat))

        step = step.reshape(-1, 3)

        # 如果步长超出信赖域半径，则缩放步长
        step_norm = np.linalg.norm(step)
        if step_norm > trust_radius:
            step = (trust_radius / step_norm) * step

        old_energy = atoms.get_potential_energy(force_consistent=True)

        # 更新原子位置
        atoms.set_positions(current_crd + step)
        
        # 预测减少量
        pred_energy = -0.5 * np.dot(current_force_flat, np.dot(hessian, step.flatten()))
        
        # 实际的减少量
        
        energy = atoms.get_potential_energy(force_consistent=True)
        energy_change = old_energy - energy

        # 计算rho值
        rho = energy_change / pred_energy
        
        # 调整信赖域半径
        min_trust_radius = 1e-3
        if rho < 0.25:
            trust_radius = max(0.25 * trust_radius, min_trust_radius)
        elif rho > 0.75 and step_norm == trust_radius:
            trust_radius = min(2.0 * trust_radius, trust_max)
        
        # 判断是否收敛
        force = atoms.get_forces()
        
        # Convergence criteria:   
        atoms.max_dp = abs(step).max()
        atoms.rms_dp = np.sqrt((step**2).sum()/step.size*3)
        atoms.max_f = abs(force).max()
        atoms.rms_f = np.sqrt((force**2).sum()/step.size*3)
        
        # Log the information:
        iter = f"Iteration: {iteration}"
        info_message = ['\n' + '-' * 70 + '\n',f'{iter.center(70)}\n\n']

        info_message.append(f'\n{"Coordinates".center(70)}\n')
        info_message.append('-' * 70)
        info_message.append('\n')

        for atom_index, atom in enumerate(atoms):
            element_type = atom.symbol 
            coord = atom.position 

            info_message.append(f"{atom_index:<4} {element_type:<2} {coord[0]:>20.4f} {coord[1]:>20.4f} {coord[2]:>20.4f}\n")
        
        
        info_message.append(f"\n\nEnergy:                {energy/g_au:>12.6f} Convergence criteria  Is converged \n")

        if atoms.max_f > atoms.f_max_th:
            info_message.append(f"Maximum Force:         {atoms.max_f/g_au:>12.6f} {atoms.f_max_th/g_au:>12.6f}                No\n")
        else:
            info_message.append(f"Maximum Force:         {atoms.max_f/g_au:>12.6f} {atoms.f_max_th/g_au:>12.6f}                Yes\n")

        if atoms.rms_f > atoms.f_rms_th:
            info_message.append(f"RMS Force:             {atoms.rms_f/g_au:>12.6f} {atoms.f_rms_th/g_au:>12.6f}                No\n")
        else:
            info_message.append(f"RMS Force:             {atoms.rms_f/g_au:>12.6f} {atoms.f_rms_th/g_au:>12.6f}                Yes\n")

        if atoms.max_dp > atoms.dp_max_th:
            info_message.append(f"Maximum Displacement:  {atoms.max_dp:>12.6f} {atoms.dp_max_th:>12.6f}                No\n")
        else:
            info_message.append(f"Maximum Displacement:  {atoms.max_dp:>12.6f} {atoms.dp_max_th:>12.6f}                Yes\n")

        if atoms.rms_dp > atoms.dp_rms_th:
            info_message.append(f"RMS Displacement:      {atoms.rms_dp:>12.6f} {atoms.dp_rms_th:>12.6f}                No\n")
        else:
            info_message.append(f"RMS Displacement:      {atoms.rms_dp:>12.6f} {atoms.dp_rms_th:>12.6f}                Yes\n")

        log_info(info_message,output)
        
        if atoms.max_f<=atoms.f_max_th and atoms.rms_f<=atoms.f_rms_th and atoms.max_dp <=atoms.dp_max_th and atoms.rms_dp<=atoms.dp_rms_th:
            converged = True
            info_message = ['\n\n' + '-' * 70 + '\n', f'{"Normal Termination".center(70)}\n\n']
            log_info(info_message,output)
            return iteration 
        
    return  iteration



