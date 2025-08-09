import numpy as np

def parse_coordinates(lines):
    """从文件中解析原子坐标，同时记录原子类型."""
    coordinates = []
    elements = []
    for line in lines:
        if line.strip() == "":
            break
        parts = line.split()
        elements.append(parts[1])
        coordinates.append([float(parts[2]), float(parts[3]), float(parts[4])])
    return np.array(coordinates), elements

def parse_frequencies(lines, freq_num, atom_num):
    """解析特定频率的特征向量."""
    start_idx = 0
    for i, line in enumerate(lines):
        if f"Frequency {freq_num}:" in line:
            start_idx = i + 2  # 调整以正确获取频率下的特征向量
            break
    
    eigenvectors = []
    for i in range(start_idx, start_idx + atom_num):  # 根据原子数量读取
        parts = lines[i].split()
        if len(parts) >= 5:  # 确保该行有足够的元素
            eigenvectors.append([float(parts[2]), float(parts[3]), float(parts[4])])
        else:
            break
    
    if len(eigenvectors) != atom_num:
        raise ValueError(f"Expected {atom_num} eigenvectors but got {len(eigenvectors)}")

    return np.array(eigenvectors)

def interpolate_coordinates(start_coords, end_coords, num_interpolations):
    """计算插值坐标."""
    return [start_coords * (1 - t) + end_coords * t for t in np.linspace(0, 1, num_interpolations)]

def write_xyz(filename, trajectory, elements):
    """将轨迹写入xyz文件."""
    num_atoms = len(elements)
    with open(filename, 'w') as f:
        for i, coords in enumerate(trajectory):
            f.write(f"{num_atoms}\n")
            f.write(f"Frame {i+1}: {i+1}\n")
            for j, atom in enumerate(coords):
                f.write(f"{elements[j]}   {atom[0]:>8.4f}   {atom[1]:>8.4f}   {atom[2]:>8.4f}\n")

def main():
    # 读取文件内容
    with open('./MaLePSO/freq.out', 'r') as file:
        lines = file.readlines()

    # 解析坐标部分
    coordinates_start = None
    for i, line in enumerate(lines):
        if "Coordinates" in line:
            coordinates_start = i + 2
            break
    
    coordinates, elements = parse_coordinates(lines[coordinates_start:])
    atom_num = len(elements)  # 获取原子数量

    # 获取特征向量
    freq_num = 1  # 假设我们需要Frequency 1
    eigenvectors = parse_frequencies(lines, freq_num, atom_num)
    
    # 计算末态坐标
    final_coordinates = coordinates + eigenvectors

    # 插值并生成轨迹
    interpolations = 25
    trajectory = []
    trajectory.extend(interpolate_coordinates(coordinates, final_coordinates, interpolations))
    trajectory.append(final_coordinates)  # 加入末态
    trajectory.extend(interpolate_coordinates(final_coordinates, coordinates, interpolations))
    
    # 写入xyz文件
    write_xyz('freq.xyz', trajectory, elements)

if __name__ == "__main__":
    main()
