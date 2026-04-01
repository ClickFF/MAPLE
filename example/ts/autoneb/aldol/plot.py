import re
import matplotlib.pyplot as plt

# Hartree to kcal/mol
HARTREE_TO_KCAL = 627.5094740631

def parse_xyz_trajectory(filename):
    """解析xyz轨迹文件，提取image编号和能量"""
    images = []
    energies = []
    
    with open(filename, 'r') as f:
        content = f.read()
    
    # 匹配 "Image X  Energy = Y" 格式
    pattern = r'Image\s+(\d+)\s+Energy\s*=\s*([-\d.]+)'
    matches = re.findall(pattern, content)
    
    for match in matches:
        images.append(int(match[0]))
        energies.append(float(match[1]))
    
    return images, energies

def plot_neb_energy(filename, output='neb_energy.png'):
    """绘制NEB能量曲线"""
    images, energies_hartree = parse_xyz_trajectory(filename)
    
    # 转换为kcal/mol，最低点设为0
    e_min = min(energies_hartree)
    energies_kcal = [(e - e_min) * HARTREE_TO_KCAL for e in energies_hartree]
    
    # 绘图
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(images, energies_kcal, 'o-', color='#2563eb', lw=2, markersize=8)
    ax.fill_between(images, energies_kcal, alpha=0.2, color='#2563eb')
    
    ax.set_xlabel('Image', fontsize=12)
    ax.set_ylabel('Relative Energy (kcal/mol)', fontsize=12)
    ax.set_title('NEB Energy Profile', fontsize=14)
    ax.set_xlim(min(images), max(images))
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    
    # 标注最高点
    e_max = max(energies_kcal)
    idx_max = energies_kcal.index(e_max)
    ax.annotate(f'TS: {e_max:.1f}', xy=(images[idx_max], e_max),
                xytext=(images[idx_max]+0.5, e_max+1),
                fontsize=10, ha='left')
    
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.show()
    
    # 打印数据
    print(f"{'Image':<8}{'E (Hartree)':<18}{'E (kcal/mol)':<12}")
    print('-' * 38)
    for i, eh, ek in zip(images, energies_hartree, energies_kcal):
        print(f"{i:<8}{eh:<18.8f}{ek:<12.2f}")
    print(f"\nBarrier: {e_max:.2f} kcal/mol")

if __name__ == '__main__':
    plot_neb_energy('inp1_autoneb_global_mep.xyz',output='inp1.png')