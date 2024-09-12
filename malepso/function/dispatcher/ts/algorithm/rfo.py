# 输入: 
# X0: 初始猜测的几何结构
# max_iter: 最大迭代次数
# tol: 收敛容忍度
# Hessian: 初始Hessian矩阵（可以通过有限差分或其它方法估算）
# gradient: 初始梯度向量
import numpy as np

from ase import Atoms

from .logger import *

def RFO(atoms: Atoms, output):
	g_au = 27.211386024367243
	max_iter = 256  # 最大迭代次数
	max_step_size = 0.5  # 最大步长
	# Step 1: 初始化
	X = atoms.get_positions().flatten()  # 初始几何结构
	Hessian = calculate_Hessian(atoms)  # 计算Hessian矩阵
	gradient = -atoms.get_forces().flatten()  # 计算梯度向量
	iteration = 0  # 初始化迭代计数
	
	while iteration < max_iter:
		
		# Step 2: 计算Hessian矩阵的特征值和特征向量
		eigvals, eigvecs = np.linalg.eig(Hessian)  # 特征值和特征向量分解
		
		# Step 3: 计算合理函数近似步长 h
		lambda_val = calculate_lambda(eigvals, eigvecs, gradient.flatten())
		
		
		# 通过合理函数近似的公式计算步长 h
		h = np.zeros_like(gradient.flatten())  # 初始化步长
		for i in range(len(eigvals)):
			Fi = np.dot(eigvecs[:, i], gradient.flatten())  # 计算每个方向上的梯度分量 Fi
			h += -(Fi / (eigvals[i] - lambda_val)) * eigvecs[:, i]  # 累加步长
		
		# Step 4: 检查步长大小是否超过允许范围，必要时进行缩放
		if np.linalg.norm(h) > max_step_size:
			h = scale_down(h, max_step_size)
		
		# Step 5: 更新几何结构
		X_new = X.flatten() + h
		
		atoms.set_positions(X_new.reshape(-1, 3))
		# Step 6: 计算新的梯度和Hessian矩阵
		gradient_new = -atoms.get_forces()
		Hessian_new = calculate_Hessian(atoms)

		# Convergence criteria:
		energy = atoms.get_potential_energy(force_consistent=True)
		force = -gradient_new   
		atoms.max_dp = abs(h).max()
		atoms.rms_dp = np.sqrt((h**2).sum()/h.size*3)
		atoms.max_f = abs(force).max()
		atoms.rms_f = np.sqrt((force**2).sum()/h.size*3)
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
			break
			return iteration 
		
		# 更新当前状态
		X = X_new
		gradient = gradient_new
		Hessian = Hessian_new
		iteration += 1
	
	print("未能在最大迭代次数内收敛")
	return X  # 返回最后的几何结构作为近似过渡态

# 辅助函数:
# 计算Hessian矩阵
def calculate_Hessian(atoms: Atoms):
	calc = atoms.get_calculator()
	Hessian = calc.get_hessian(atoms)
	return Hessian


def calculate_lambda(eigvals, eigvecs, gradient, max_iter=100, tol=1e-6, epsilon=1e-8):
	"""
	计算 RFO 算法中的拉姆达参数 λ。
	
	参数:
	- eigvals: Hessian 矩阵的特征值 (numpy array)
	- eigvecs: Hessian 矩阵的特征向量 (numpy array, 每一列是一个特征向量)
	- gradient: 梯度向量 (numpy array)
	- max_iter: 最大迭代次数 (默认值为100)
	- tol: 收敛容忍度 (默认值为1e-6)
	
	返回:
	- λ: 计算得到的拉姆达值
	"""
	# Step 1: 计算梯度在每个特征向量方向上的分量 F_i
	F = np.dot(eigvecs.T, gradient)  # F_i = V_i^T * g
	
	# Step 2: 初始化 λ，选择最小的正特征值作为初值
	for value in eigvals:
		if value > 0:
			lambda_val = value
			break

	# Step 3: 迭代求解 λ，直到收敛
	for iteration in range(max_iter):
		# 计算 sum(F_i^2 / (λ - b_i))，添加 epsilon 以避免除以零
		sum_val = np.sum(F**2 / ((lambda_val - eigvals) + epsilon))
		
		# 检查收敛
		if abs(sum_val - 1) < tol:
			break

		# 计算 f'(λ)，即 denominator，添加 epsilon 以避免除以零
		denominator = np.sum(F**2 / ((lambda_val - eigvals)**2 + epsilon))
		
		# 使用牛顿法更新 λ
		lambda_val -= (sum_val - 1) / (denominator + epsilon)

	return lambda_val

# 缩放步长以符合最大步长限制
def scale_down(h, max_step_size):
	return h * (max_step_size / np.linalg.norm(h))

