# 输入: 
# X0: 初始猜测的几何结构
# max_iter: 最大迭代次数
# tol: 收敛容忍度
# Hessian: 初始Hessian矩阵（可以通过有限差分或其它方法估算）
# gradient: 初始梯度向量
import numpy as np
import sys
from ase import Atoms
from scipy.optimize import brentq,newton

from .logger import *

def RFO(atoms: Atoms, output):
	sys.setrecursionlimit(1000)
	g_au = 27.211386024367243
	max_iter = 256  # 最大迭代次数
	max_step_size = 0.2  # 最大步长
	iteration = 0  # 初始化迭代计数
	
	while iteration < max_iter:

		iteration += 1
		iter = f"Iteration: {iteration}"
		info_message = ['\n' + '-' * 70 + '\n',f'{iter.center(70)}\n\n']

		# Step 1: 初始化
		X = atoms.get_positions().flatten()  # 初始几何结构
		Hessian = calculate_Hessian(atoms)  # 计算Hessian矩阵
		gradient = -atoms.get_forces().flatten()  # 计算梯度向量
		
		# Step 2: 计算Hessian矩阵的特征值和特征向量
		eigvals, eigvecs = np.linalg.eigh(Hessian)  # 特征值和特征向量分解
		eigvals = np.real(eigvals).astype(np.float32).squeeze()
		eigvecs = eigvecs.astype(np.float32).squeeze()
		g_prime = eigvecs.T @ gradient
		# Step 3: 计算合理函数近似步长 x

		# 计算shitf参数 λ
		index_1 = find_nth_smallest_index(eigvals, 1)  # 找到最小的特征值的索引
		index_2 = find_nth_smallest_index(eigvals, 2)  # 找到第二小的特征值的索引

		g1 = g_prime[index_1]
		h1 = eigvals[index_1]
		h2 = eigvals[index_2]

		mask = np.arange(len(eigvals)) != index_1
		g_i = g_prime[mask]
		h_i = eigvals[mask]

		lambda1 = calculate_lambda1(g1, h1)
		lambda2 = calculate_lambda2(g_i, h_i,h2=h2)
	
		h_minus_lambda = np.zeros_like(eigvals)
		h_minus_lambda[index_1] = eigvals[index_1] - lambda1
		h_minus_lambda[mask] = eigvals[mask] - lambda2

		# 检查分母是否为零
		if np.any(h_minus_lambda == 0):
			raise ValueError("分母 h_i - lambda_i 不能为零。")

		delta_x_prime = -g_prime / h_minus_lambda

		# 7. 转换回原始坐标系 Δx = U * Δx'
		x = eigvecs @ delta_x_prime
		
		#detalE = predict_energy_change(x, lambda_val, eigvals, gradient)
		#deltaE = detalE/g_au

		# Step 4: 检查步长大小是否超过允许范围，必要时进行缩放
		if np.linalg.norm(x) > max_step_size:
			x = scale_down(x, max_step_size)

		#info_message.append(f"Predicted energy change: {detalE}\n")
		info_message.append(f"lambda1: {lambda1}, lambda2: {lambda2}\n")
		info_message.append(f"lowest eigenvalue: {np.min(eigvals)}, second lowest eigenvalue: {np.sort(eigvals)[1]}\n")

		# Step 5: 更新几何结构
		X_new = X.flatten() + x.flatten()
		atoms.set_positions(X_new.reshape(-1, 3))

		# Convergence criteria:
		energy = atoms.get_potential_energy(force_consistent=True)
		force = atoms.get_forces()  
		atoms.max_dp = abs(x).max()
		atoms.rms_dp = np.sqrt((x**2).sum()/x.size*3)
		atoms.max_f = abs(force).max()
		atoms.rms_f = np.sqrt((force**2).sum()/x.size*3)

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

def find_nth_smallest_index(eigvals, n):
	# 对特征值进行排序，并获取排序后的索引
	sorted_indices = np.argsort(eigvals)
	# 返回第n小的特征值的索引
	return sorted_indices[n-1]

def calculate_lambda1(g1, h1, epsilon=1e-6, max_iter=256, alpha=0.2):
    def f1(lambda1):
        return g1**2 / (lambda1 - h1) - lambda1

    def f1_prime(lambda1):
        return -g1**2 / (lambda1 - h1)**2 - 1

    # 初始猜测值
    lambda1 = h1 / 2 + 1e-6

    for _ in range(max_iter):
        f_val = f1(lambda1)
        f_prime_val = f1_prime(lambda1)
        
        if abs(f_prime_val) < epsilon:
            break
        
        lambda1_new = lambda1 - alpha * (f_val / f_prime_val)
        
        if abs(lambda1_new - lambda1) < epsilon:
            return lambda1_new
        
        lambda1 = lambda1_new

    return lambda1

def calculate_lambda2(g_vals, h_vals, h2, epsilon=1e-6):
	def f2(lambda2):
		numerator = np.sum(g_vals**2 / (lambda2 - h_vals))
		return numerator - lambda2
	# lambda2 - h1/2 > 0
	upper = h2 - epsilon
	lower = h2 - 100
	print(f2(lower), f2(upper))
	lambda2 = brentq(f2, upper, lower)
	return lambda2

# 缩放步长以符合最大步长限制
def scale_down(step, max_step_size):
	return  step * (max_step_size / np.linalg.norm(step))


	