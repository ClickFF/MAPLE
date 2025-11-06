#!/usr/bin/env python3
import re
import sys

# 能匹配元素符号 + 3 个坐标，坐标支持科学计数法
LINE_RE = re.compile(
    r"\s*\d+\s+([A-Za-z][A-Za-z0-9]*)\s+"
    r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s+"
    r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s+"
    r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
)

def write_frame(out, frame_idx, atoms):
    """按 XYZ 写一帧"""
    if not atoms:
        return
    out.write(f"{len(atoms)}\n")
    out.write(f"Frame {frame_idx}\n")
    for el, x, y, z in atoms:
        out.write(f"{el}   {x}   {y}   {z}\n")

def process_coordinates(file_path, output_path):
    frame_idx = 0
    in_block = False
    atoms_this_frame = []

    with open(file_path, "r") as fin, open(output_path, "w") as fout:
        for line in fin:
            if "Coordinates" in line:
                # 进入新帧前把上一帧写出去
                if in_block:
                    frame_idx += 1
                    write_frame(fout, frame_idx, atoms_this_frame)
                    atoms_this_frame = []
                in_block = True
                continue

            if in_block:
                m = LINE_RE.match(line)
                if m:
                    el, x, y, z = m.groups()
                    atoms_this_frame.append((el, x, y, z))
                # 若遇到非匹配行且已在 block，可选择忽略或判断 block 结束
                # 这里选择忽略，直到下一次遇到 "Coordinates"

        # 文件结束，收尾写最后一帧
        if in_block and atoms_this_frame:
            frame_idx += 1
            write_frame(fout, frame_idx, atoms_this_frame)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py input.out output.xyz")
        sys.exit(1)
    process_coordinates(sys.argv[1], sys.argv[2])
