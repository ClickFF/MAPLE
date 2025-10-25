import numpy as np
import matplotlib.pyplot as plt

# 读取文件并提取能量值
energy_values = []
with open('exo.out', 'r') as file:
    for line in file:
        if 'Energy:' in line:
            parts = line.split()
            energy_ev = float(parts[1])
            energy_kjmol = energy_ev * 627.5
            energy_values.append(energy_kjmol)

# 计算平均值并进行归一化
average_energy = np.min(energy_values)
normalized_energies = [value - average_energy for value in energy_values]

# 确保数据完整
grid_size = 51
expected_size = grid_size * grid_size
actual_size = len(normalized_energies)

if actual_size < expected_size:
    print(f"Warning: Expected {expected_size} data points, but got {actual_size}.")
    normalized_energies += [np.nan] * (expected_size - actual_size)
elif actual_size > expected_size:
    print(f"Warning: Expected {expected_size} data points, but got {actual_size}. Truncating extra values.")
    normalized_energies = normalized_energies[:expected_size]

# 重新调整数据的排列方式
energy_grid = np.zeros((grid_size, grid_size))
for j in range(grid_size):  # 遍历 y 轴
    for i in range(grid_size):  # 遍历 x 轴
        energy_grid[grid_size - 1 - j, i] = normalized_energies[j * grid_size + i]

# 设置 x 轴和 y 轴的范围
x = np.linspace(1.5, 4.0, grid_size)
y = np.linspace(1.5, 4.0, grid_size)
X, Y = np.meshgrid(x, y)

# 绘制等高线图
plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, energy_grid, levels=20, cmap='viridis')
plt.colorbar(contour)
plt.title('Energy Contour Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.savefig('exo.png')
plt.show()
