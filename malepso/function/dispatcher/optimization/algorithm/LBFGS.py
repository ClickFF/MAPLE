import numpy as np
from ase import Atoms
from .logger import *
from .DIIS import OptimizationStorage, DIIS

g_au = 27.211386024367243


def LBFGS(
    atoms: Atoms,
    output: str,
    use_line_search: bool = False,
    memory: int = 5,              # ✅ 改成默认 5，与 Driver 思路一致
    curvature: float = 70.0,
    maxstep: float = 0.2,
    maxiteration: int = 128,
    use_diis: bool = True,        # ✅ DIIS 开关
) -> int:
    """
    Textbook-style L-BFGS optimizer (two-loop recursion, dynamic H0),
    keeping original ASE interface, DIIS logic, and logging output.
    """

    S, Y, rhos = [], [], []   # ✅ 标准 L-BFGS 历史
    diis_storage = OptimizationStorage()
    diis_counter = 0

    r = atoms.get_positions()
    e = atoms.get_potential_energy(force_consistent=True)
    f = atoms.get_forces()

    iteration = 0
    convergence = False

    while not convergence and iteration < maxiteration:
        # ================================
        # ✅ DIIS 加速（可选，不改变主算法）
        # ================================
        if use_diis:
            diis_storage.diis_x_vectors.append(r.copy())
            diis_storage.diis_error_vectors.append(f.copy())
            diis_storage.iteration = iteration
            diis_counter += 1

            if diis_counter >= 10:
                DIIS(atoms, output, max_step_size=maxstep, maxiterations=maxiteration, storage=diis_storage)
                diis_counter = 0
                diis_storage.reset()

                # ⚠️ 重要：DIIS 改变了坐标和力场，重取当前位置/力
                r = atoms.get_positions()
                f = atoms.get_forces()

                # 👉 可选策略：清空历史，避免方向污染
                # S.clear(); Y.clear(); rhos.clear()

        # ================================
        # ✅ 两步递推求搜索方向 (two-loop)
        # ================================
        grad = f.reshape(-1)  # 扁平化方便向量操作
        q = grad.copy()
        alpha_list = []

        for s, y, rho in reversed(list(zip(S, Y, rhos))):
            a = rho * np.dot(s, q)
            alpha_list.append(a)
            q -= a * y

        # ✅ 动态 H0 (gamma) 替代固定 1/curvature
        if Y:
            gamma = np.dot(Y[-1], S[-1]) / (np.dot(Y[-1], Y[-1]) + 1e-20)
        else:
            gamma = 1.0 / curvature

        z = gamma * q

        for (s, y, rho), a in zip(zip(S, Y, rhos), reversed(alpha_list)):
            b = rho * np.dot(y, z)
            z += s * (a - b)

        step = -z.reshape(f.shape)  # 还原原子维度

        # ================================
        # ✅ 步长限制
        # ================================
        max_disp = np.max(np.abs(step))
        if max_disp > maxstep:
            step *= maxstep / max_disp

        # ================================
        # ✅ 更新坐标 / 能量 / 力
        # ================================
        r_old = r.copy()
        f_old = f.copy()
        atoms.set_positions(r + step)

        r = atoms.get_positions()
        f = atoms.get_forces()
        e = atoms.get_potential_energy(force_consistent=True)

        s_vec = (r - r_old).reshape(-1)
        y_vec = (f - f_old).reshape(-1)
        rho_val = 1.0 / (np.dot(y_vec, s_vec) + 1e-20)

        if np.isfinite(rho_val):  # 避免除 0
            S.append(s_vec.copy()); Y.append(y_vec.copy()); rhos.append(rho_val)
        if len(S) > memory:
            S.pop(0); Y.pop(0); rhos.pop(0)

        iteration += 1

        # ================================
        # ✅ 收敛判据 & 日志输出（原样保留）
        # ================================
        atoms.max_dp = np.abs(step).max()
        atoms.rms_dp = np.sqrt((step ** 2).sum() / step.size * 3)
        atoms.max_f = np.abs(f).max()
        atoms.rms_f = np.sqrt((f ** 2).sum() / step.size * 3)

        iter = f"Iteration: {iteration}"
        info_message = ['\n' + '-' * 70 + '\n', f'{iter.center(70)}\n\n']
        info_message.append(f'\n{"Coordinates".center(70)}\n')
        info_message.append('-' * 70)
        info_message.append('\n')

        for atom_index, atom in enumerate(atoms):
            element_type = atom.symbol
            coord = atom.position
            info_message.append(f"{atom_index:<4} {element_type:<2} {coord[0]:>20.4f} {coord[1]:>20.4f} {coord[2]:>20.4f}\n")

        info_message.append(f"\n\nEnergy:                {e:>12.6f} Convergence criteria  Is converged \n")

        if atoms.max_f > atoms.f_max_th:
            info_message.append(f"Maximum Force:         {atoms.max_f:>12.6f} {atoms.f_max_th:>12.6f}                No\n")
        else:
            info_message.append(f"Maximum Force:         {atoms.max_f:>12.6f} {atoms.f_max_th:>12.6f}                Yes\n")

        if atoms.rms_f > atoms.f_rms_th:
            info_message.append(f"RMS Force:             {atoms.rms_f:>12.6f} {atoms.f_rms_th:>12.6f}                No\n")
        else:
            info_message.append(f"RMS Force:             {atoms.rms_f:>12.6f} {atoms.f_rms_th:>12.6f}                Yes\n")

        if atoms.max_dp > atoms.dp_max_th:
            info_message.append(f"Maximum Displacement:  {atoms.max_dp:>12.6f} {atoms.dp_max_th:>12.6f}                No\n")
        else:
            info_message.append(f"Maximum Displacement:  {atoms.max_dp:>12.6f} {atoms.dp_max_th:>12.6f}                Yes\n")

        if atoms.rms_dp > atoms.dp_rms_th:
            info_message.append(f"RMS Displacement:      {atoms.rms_dp:>12.6f} {atoms.dp_rms_th:>12.6f}                No\n")
        else:
            info_message.append(f"RMS Displacement:      {atoms.rms_dp:>12.6f} {atoms.dp_rms_th:>12.6f}                Yes\n")

        log_info(info_message, output)

        if (
            atoms.max_f <= atoms.f_max_th
            and atoms.rms_f <= atoms.f_rms_th
            and atoms.max_dp <= atoms.dp_max_th
            and atoms.rms_dp <= atoms.dp_rms_th
        ):
            return iteration

    return iteration
