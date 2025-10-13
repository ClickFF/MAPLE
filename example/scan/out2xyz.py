import re

def process_coordinates(file_path, output_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    atom_count = 0
    frame_num = 1
    coordinates = []
    recording = False
    frame_data = ""  # 初始化帧数据

    for line in lines:
        # 检查是否遇到“Coordinates”标识符
        if "Coordinates" in line:
            if recording:  # 如果正在记录，则保存上一帧
                coordinates.append(f"{atom_count}\nFrame {frame_num}: {frame_num}\n" + frame_data)
                frame_num += 1
            recording = True
            frame_data = ""  # 初始化新的帧数据
        elif recording:
            match = re.match(r'\s*\d+\s+(\w+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)', line)
            if match:
                if frame_num == 1:
                    atom_count += 1  # 只在第一帧计算原子数
                element = match.group(1)
                x = match.group(2)
                y = match.group(3)
                z = match.group(4)
                frame_data += f"{element}   {x}   {y}   {z}\n"

    # 保存最后一帧
    if recording and frame_data:
        coordinates.append(f"{atom_count}\nFrame {frame_num}: {frame_num}\n" + frame_data)

    # 将结果写入输出文件
    with open(output_path, 'w') as out_file:
        out_file.writelines(coordinates)

# 使用示例
process_coordinates('endo.out', 'endo.xyz')
