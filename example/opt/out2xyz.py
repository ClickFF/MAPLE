import re
import sys

def process_coordinates(file_path, output_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    atom_count = 0
    frame_num = 1
    coordinates = []
    recording = False
    frame_data = ""

    for line in lines:
        if "Coordinates" in line:
            if recording:
                coordinates.append(f"{atom_count}\nFrame {frame_num}: {frame_num}\n" + frame_data)
                frame_num += 1
            recording = True
            frame_data = ""
        elif recording:
            match = re.match(r'\s*\d+\s+(\w+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)', line)
            if match:
                if frame_num == 1:
                    atom_count += 1
                element = match.group(1)
                x, y, z = match.group(2), match.group(3), match.group(4)
                frame_data += f"{element}   {x}   {y}   {z}\n"

    if recording and frame_data:
        coordinates.append(f"{atom_count}\nFrame {frame_num}: {frame_num}\n" + frame_data)

    with open(output_path, 'w') as out_file:
        out_file.writelines(coordinates)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py input.out output.xyz")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    process_coordinates(input_file, output_file)
