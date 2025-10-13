# 输入: 
# X0: 初始猜测的几何结构
# max_iter: 最大迭代次数
# tol: 收敛容忍度
# Hessian: 初始Hessian矩阵（可以通过有限差分或其它方法估算）
# gradient: 初始梯度向量
import numpy as np
import sys
import torch
from ase import Atoms
from scipy.optimize import brentq

from .logger import *

def RFO(atoms: Atoms, output):
	sys.setrecursionlimit(1000)
	g_au = 27.211386024367243
	max_iter = 256  # 最大迭代次数
	max_step_size = 0.2  # 最大步长
	using_step_size = 0.1
	iteration = 0  # 初始化迭代计数
	
	while iteration < max_iter:

		energy_list = []
		iteration += 1
		iter = f"Iteration: {iteration}"
		info_message = ['\n' + '-' * 70 + '\n',f'{iter.center(70)}\n\n']

		# Step 1: 初始化
		X = atoms.get_positions().flatten()  # 初始几何结构
		Hessian = calculate_Hessian(atoms)  # 计算Hessian矩阵
		gradient = -atoms.get_forces().flatten()  # 计算梯度向量
		
		# Step 2: 计算Hessian矩阵的特征值和特征向量
		eigvals, eigvecs = np.linalg.eig(Hessian)  # 特征值和特征向量分解
		
		# Step 3: 计算合理函数近似步长 x

		# 计算shitf参数 λ
		lambda_val = calculate_lambda(eigvals, gradient.flatten())
		
		eigvals = np.real(eigvals).astype(np.float32)
		# 去除 Hessian 的批次维度
		Hessian_squeezed = Hessian.squeeze(0)  # 变为 (66, 66)
		# 将梯度转换为二维列向量 (66, 1)
		gradient_reshaped = gradient.reshape(-1, 1)  # 使用 numpy 的 reshape
		# 创建单位矩阵
		dim = gradient.shape[0]
		I = torch.eye(dim)
		# 计算 H - λI
		shifted_hessian = Hessian_squeezed - lambda_val * I
		# 使用 numpy 进行线性求解
		# 先将 Torch 张量转换为 numpy 数组
		shifted_hessian_np = shifted_hessian.numpy()
		gradient_np = gradient_reshaped

		# 解决线性方程 (H - λI) * x = -gradient
		x = -np.linalg.solve(shifted_hessian_np, gradient_np)
		x = x.flatten()
		# Step 4: 检查步长大小是否超过允许范围，必要时进行缩放
		if np.linalg.norm(x) > using_step_size:
			x = scale_down(x, using_step_size)
		
		#detalE = predict_energy_change(x, lambda_val, eigvals, gradient)
		#deltaE = detalE/g_au

		#info_message.append(f"Predicted energy change: {detalE}\n")
		info_message.append(f"lambda: {lambda_val}\n")
		info_message.append(f"lowest eigenvalue: {np.min(eigvals)}\n")

		# Step 5: 更新几何结构
		X_new = X.flatten() + x
		atoms.set_positions(X_new.reshape(-1, 3))


		# Convergence criteria:
		energy = atoms.get_potential_energy(force_consistent=True)
		energy_list.append(energy)
		force = atoms.get_forces()  
		atoms.max_dp = abs(x).max()
		atoms.rms_dp = np.sqrt((x**2).sum()/x.size*3)
		atoms.max_f = abs(force).max()
		atoms.rms_f = np.sqrt((force**2).sum()/x.size*3)

		if iteration%10 == 0:
			if check_energy_trend(energy_list):
				using_step_size *= 0.8


		# Log the information:
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
		
	print("未能在最大迭代次数内收敛")
	return X  # 返回最后的几何结构作为近似过渡态

# 辅助函数:
# 计算Hessian矩阵
def calculate_Hessian(atoms: Atoms):
	calc = atoms.get_calculator()
	Hessian = calc.get_hessian(atoms)
	return Hessian


def f(lambda_val, eigvals, gradient):
    """
    定义方程的求解函数 f(λ) = sum(g_i^2 / (λ - h_i)) - λ
    该函数将用于 Bracketing 方法(Brent's Method)中寻找根.
    
    参数:
    - lambda_val: λ 的当前值
    - eigvals: Hessian 矩阵的特征值 (numpy 数组)
    - gradient: 梯度向量 (numpy 数组)
    
    返回:
    - 目标函数值 f(λ)
    """
    squared_g = gradient ** 2
    return np.sum(squared_g / (lambda_val - eigvals)) - lambda_val

def calculate_lambda(eigvals, gradient, tol=1e-5, max_iter=64):
    """
    使用 Brent's Method 计算 RFO 算法中的拉姆达参数 λ。
    
    参数:
    - eigvals: Hessian 矩阵的特征值 (torch tensor)
    - gradient: 梯度向量 (torch tensor)
    - tol: 收敛容忍度 (默认值为1e-5)
    
    返回:
    - λ: 计算得到的最小拉姆达值
    """
    # 转换为 numpy 数组进行求解
    eigvals_np = eigvals
    gradient_np = gradient
    
    # 选择区间的上下界, 保证 λ < h_min (即特征值中的最小值)
    h_min = np.min(eigvals_np)
    a = h_min - 1e2  # 区间左边界，取 h_min 之前的一个大负数
    b = h_min - 1e-6  # 区间右边界，略小于 h_min，避免奇异点
    
    # 检查区间两端 f(a) 和 f(b) 的符号是否相反，保证函数有根
    if f(a, eigvals_np, gradient_np) * f(b, eigvals_np, gradient_np) >= 0:
        raise ValueError("无法找到有效的区间 (a, b)，请调整区间或输入数据。")
    
    # 使用 Brent's Method 找到方程的根
    lambda_root = brentq(f, a, b, args=(eigvals_np, gradient_np), xtol=tol, maxiter=max_iter)
    
    return lambda_root



# 缩放步长以符合最大步长限制
def scale_down(step, max_step_size):
	return  step * (max_step_size / np.linalg.norm(step))

def predict_energy_change(x, lambda0, eigvals, gradient):
	z = 1 + np.dot(x, x)
	z_inv = 1 / z
	g = gradient.flatten()
	sum = 0
	for i in range(len(eigvals)):
		sum += (g[i]**2) * (lambda0 - eigvals[i]/2)/((lambda0-eigvals[i])**2)
	results = z_inv * sum
	deltaE = 0
	for i in range(len(results)):
		deltaE += results[i]
	return deltaE
	
def check_energy_trend(energy_list):
    if len(energy_list) < 10:
        return None  # 数据不足，无法判断

    last_10_energies = energy_list[-10:]

    # 计算相邻能量差
    diffs = [last_10_energies[i+1] - last_10_energies[i] for i in range(9)]

    # 检查总体趋势是否下降
    if all(d <= 0 for d in diffs):
        return False  # 总体趋势下降

    # 检查是否出现震荡
    sign_changes = sum(1 for i in range(len(diffs)-1) if diffs[i]*diffs[i+1] < 0)
    if sign_changes > 0:
        return True  # 出现震荡

    # 如果既不完全下降也没有震荡，默认返回False
    return False