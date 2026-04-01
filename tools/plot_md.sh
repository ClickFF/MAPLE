#!/bin/bash
# 用法: ./plot_md.sh <*_md_thermo.dat>
#
# 生成 MD 模拟科学验证图（每项一个独立窗口），用于论文发表前的结果校验。
# 自动识别 ensemble（NVE / NVT / NPT），使用对应的判据。
#
# 参考文献：
#   [1] Páll et al., J. Chem. Theory Comput. 2020 — 能量漂移率标准
#   [2] Case et al., AMBER 2022 Manual §3   — NVE 能量涨落标准
#   [3] Shirts & Chodera, JCP 129, 124105 (2008) — KE-PE 相关系数
#   [4] Allen & Tildesley, Computer Simulation of Liquids 2nd ed. 2017 §2.4, §6.1
#   [5] Frenkel & Smit, Understanding Molecular Simulation 2nd ed. 2002, Ch. 6.1
#   [6] Basconi & Shirts, JCTC 9, 2887 (2013) — NVT thermostat benchmarking
#   [7] Bussi et al., JCP 126, 014101 (2007) — V-rescale thermostat

if [ $# -ne 1 ]; then
    echo "用法: $0 <*_md_thermo.dat>"
    exit 1
fi

input_file="$1"

if [ ! -f "$input_file" ]; then
    echo "错误：文件 '$input_file' 不存在。"
    exit 1
fi

python3 - "$input_file" <<'EOF'
import re, sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from pathlib import Path
from scipy.stats import norm, gamma, kstest, chi2

filename = sys.argv[1]

# ─── 解析 thermo.dat ──────────────────────────────────────────────────────
col_names, rows = [], []
ensemble = 'NVE'   # 保守默认：用最严格标准

with open(filename) as fh:
    for line in fh:
        s = line.strip()
        if not s:
            continue
        if s.startswith('#'):
            m = re.search(r'MD Simulation - (\w+) Ensemble', s)
            if m:
                ensemble = m.group(1).upper()
            cand = s.lstrip('#').split()
            if cand and re.match(r'(?i)step|time', cand[0]):
                col_names = cand
            continue
        try:
            rows.append([float(v) for v in s.split()])
        except ValueError:
            continue

if not rows:
    print("错误：未能解析数据。")
    sys.exit(1)

data = np.array(rows)
n = data.shape[1]
col_names = (col_names + [f"Col{i+1}" for i in range(len(col_names), n)])[:n]

def col(pat):
    for i, c in enumerate(col_names):
        if re.search(pat, c, re.I):
            return data[:, i]
    return None

time  = col(r'time\(fs\)|^time$')
step  = col(r'^step$')
T     = col(r'temp')
KE    = col(r'^ke\b|^ke\(')
PE    = col(r'^pe\b|^pe\(')
TE    = col(r'^te\b|^te\(')
press = col(r'press')
vol   = col(r'vol')

if TE is None or T is None:
    print("错误：文件中缺少必要列（Temp / TE）。")
    sys.exit(1)

xdata  = time if time is not None else step
xlabel = 'Time (fs)' if time is not None else 'Step'

# ─── 尝试从同目录 traj.xyz 读取原子数 + PBC 标志 ────────────────────────
# 注释行含 Lattice= → PBC；否则为孤立分子（N_dof = 3N-3）
# 与 utils.py:calculate_temperature() 保持一致
n_atoms = None
is_pbc  = False   # 默认孤立分子（保守）
traj_candidate = Path(filename).with_name(
    Path(filename).stem.replace('_md_thermo', '_md_traj') + '.xyz'
)
if traj_candidate.exists():
    try:
        with open(traj_candidate) as _tf:
            n_atoms  = int(_tf.readline().strip())
            comment  = _tf.readline()
            is_pbc   = 'Lattice=' in comment or 'lattice=' in comment.lower()
    except Exception:
        pass

# ─── 布尔标志 ─────────────────────────────────────────────────────────────
is_nve = ensemble == 'NVE'
is_nvt = ensemble == 'NVT'
is_npt = ensemble == 'NPT'
# 若 press/vol 列存在但 ensemble 未识别为 NPT，也按 NPT 逻辑显示
has_pv = press is not None and vol is not None
if has_pv and not is_npt:
    is_npt = True

# ─── 统计量 ───────────────────────────────────────────────────────────────
Ha_to_kJ = 2625.4996
fs_to_ns  = 1e-6
kB        = 8.617333e-5   # eV/K，仅作单位参考；计算用 Ha 内部一致

coef        = np.polyfit(xdata, TE, 1)
te_fit      = np.polyval(coef, xdata)
te_fluct    = TE.std() / abs(TE.mean())
# 漂移率：每原子 kJ/mol/ns（Páll 2020）
if n_atoms:
    drift_kJ_ns_atom = coef[0] * Ha_to_kJ / fs_to_ns / n_atoms
    drift_label      = "kJ/mol/ns/atom"
else:
    drift_kJ_ns_atom = coef[0] * Ha_to_kJ / fs_to_ns
    drift_label      = "kJ/mol/ns (N unknown)"
residual    = (TE - te_fit) * Ha_to_kJ * 1000   # J/mol
rms_resid   = np.sqrt(np.mean(residual**2))

r_kepe  = float(np.corrcoef(KE, PE)[0, 1]) if KE is not None and PE is not None else None
T_mean  = T.mean()
T_std   = T.std()

# ─── NVT/NPT 专属：canonical 温度涨落预测（Frenkel & Smit §6.1）─────────
# σ_canonical(T) = T_target * √(2 / N_dof)
# N_dof 与 utils.py:calculate_temperature() 保持一致：
#   PBC 系统: N_dof = 3N（无整体平动约束）
#   孤立分子: N_dof = 3N - 3（移除质心平动）
n_dof = (3 * n_atoms if is_pbc else 3 * n_atoms - 3) if n_atoms else None
if n_dof:
    T_sigma_canonical = T_mean * np.sqrt(2.0 / n_dof)   # 正则系综精确预测
    T_sigma_ratio     = T_std / T_sigma_canonical         # 应接近 1.0
    # NVE 近似（Allen & Tildesley §2.4）——保留供对比
    T_sigma_nve_approx = T_mean / np.sqrt(n_dof)
else:
    T_sigma_canonical = None
    T_sigma_ratio     = None
    T_sigma_nve_approx = None

# ─── NVT 专属：KE chi-squared / Gamma 检验（Shirts & Chodera 2008）────────
# 总 KE ~ Gamma(N_dof/2, k_B*T)，其中 k_B*T 用 Ha 单位
# 用 K-S 检验验证 KE 分布
ks_pval = None
ke_gamma_ok = None
if KE is not None and n_dof and (is_nvt or is_npt):
    # Gamma 参数：shape = N_dof/2，scale = k_B*T 转换为 Ha
    # k_B = 3.166811e-6 Ha/K
    kB_Ha     = 3.166811e-6
    shape_th  = n_dof / 2.0
    scale_th  = kB_Ha * T_mean
    # K-S 检验：H0: KE ~ Gamma(shape_th, scale_th)
    # 判据用统计量 D 而非 p 值：大样本下 p 值过度灵敏（几乎必然拒绝）
    # D < 0.05 → 分布高度吻合；D > 0.1 → 有实质偏差
    # Ref: Razali & Wah, J. Stat. Modeling & Analytics 2(1), 2011
    ks_stat, ks_pval = kstest(KE, 'gamma', args=(shape_th, 0, scale_th))
    ke_gamma_ok = ks_stat < 0.05   # D < 0.05 → 分布吻合

# ─── 判据标志 ─────────────────────────────────────────────────────────────
# 能量守恒（NVE 核心；NVT/NPT 仅作参考）
ok_fluct = te_fluct < 1e-4   # AMBER §3

# KE-PE 相关判据随 ensemble 切换（Shirts & Chodera 2008）
if r_kepe is not None:
    if is_nve:
        ok_r  = r_kepe < -0.95     # NVE: r ≈ -1（能量守恒迫使反相关）
        r_target_str = "≈ −1 (NVE)"
    else:
        ok_r  = abs(r_kepe) < 0.5  # NVT/NPT: r ≈ 0（恒温器解耦 KE 与 PE）
        r_target_str = "≈ 0 (NVT/NPT)"
else:
    ok_r = None
    r_target_str = ''

# 温度偏差（NVT/NPT 主要判据）
T_bias_pct = abs(T_mean - T_mean) / T_mean * 100   # 无 T_target，用自均值

# ─── 打印摘要 ─────────────────────────────────────────────────────────────
SEP = "=" * 64
print(SEP)
print(f" MD Validation Report  —  {ensemble} Ensemble")
print(SEP)
print(f" File   : {filename}")
print(f" Steps  : {len(xdata)}   Time: {xdata[-1]:.1f} fs")
if n_atoms:
    print(f" Atoms  : {n_atoms}   N_dof = {n_dof}   PBC = {'yes' if is_pbc else 'no (3N-3)'}")
print()

if is_nve:
    print(" [NVE] Energy Conservation (primary criterion):")
    print(f"   σ(TE)/|⟨TE⟩|    = {te_fluct:.2e}   [AMBER: < 1e-4]  {'✓' if ok_fluct else '✗'}")
    print(f"   Linear drift    = {drift_kJ_ns_atom:.4g} {drift_label}")
    if r_kepe is not None:
        print(f"   r(KE,PE)        = {r_kepe:.4f}   [target {r_target_str}]  {'✓' if ok_r else '✗'}")
else:
    print(f" [{'NVT' if is_nvt else 'NPT'}] Temperature Control (primary criterion):")
    print(f"   ⟨T⟩             = {T_mean:.2f} K")
    print(f"   σ(T) observed   = {T_std:.2f} K")
    if T_sigma_canonical is not None:
        ok_sigma = 0.5 < T_sigma_ratio < 2.0
        print(f"   σ(T) canonical  = {T_sigma_canonical:.2f} K  [Frenkel & Smit 2002 §6.1: √(2/N_dof)·T]")
        print(f"   σ_obs/σ_canon   = {T_sigma_ratio:.3f}   [target: 0.5–2.0]  {'✓' if ok_sigma else '✗'}")
    if r_kepe is not None:
        print(f"   r(KE,PE)        = {r_kepe:.4f}   [target {r_target_str}]  {'✓' if ok_r else '✗'}")
    if ks_stat is not None:
        print(f"   KE Gamma K-S D  = {ks_stat:.4f}   [Razali&Wah 2011: D < 0.05]  {'✓' if ke_gamma_ok else '✗'}")
    print()
    print(" [Reference] Energy Drift (secondary, not conservation criterion):")
    print(f"   σ(TE)/|⟨TE⟩|   = {te_fluct:.2e}   (NVT TE fluctuates by design)")
    print(f"   Linear drift   = {drift_kJ_ns_atom:.4g} {drift_label}   (gross instability only)")

print(SEP)
print()

# ─── 通用样式 ─────────────────────────────────────────────────────────────
BLUE   = '#1f77b4'
RED    = '#d62728'
GREEN  = '#2ca02c'
ORANGE = '#ff7f0e'
GRAY   = '#7f7f7f'
title_base = Path(filename).stem.replace('_md_thermo', '')
sup = f"{title_base}  |  {ensemble} MD"

def new_fig(subtitle):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    fig.suptitle(sup, fontsize=11, fontweight='bold', y=1.0)
    ax.set_title(subtitle, fontsize=9, loc='left', pad=5)
    return fig, ax

def finish(ax, ylabel, xl=None):
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlabel(xl or xlabel, fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.ticklabel_format(useOffset=False, style='plain', axis='y')
    ax.figure.tight_layout()

# ══════════════════════════════════════════════════════════════════════════
# 图 1 — Total Energy + linear drift
# NVE: 主要判据（σ/|E| < 1e-4）
# NVT/NPT: 参考图（TE 有意波动，仅检查系统性漂移）
# ══════════════════════════════════════════════════════════════════════════
if is_nve:
    title1 = (
        f'Total Energy  |  σ/|⟨E⟩| = {te_fluct:.2e}'
        + ('  ✓' if ok_fluct else '  ✗  (>1×10⁻⁴, AMBER §3)')
    )
else:
    title1 = (
        f'Total Energy  |  NVT/NPT: TE fluctuates by design  '
        f'drift = {drift_kJ_ns_atom:.3g} {drift_label}'
    )

_drift_unit_short = "kJ/mol/ns/atom" if n_atoms else "kJ/mol/ns"
fig1, ax1 = new_fig(title1)
ax1.plot(xdata, TE, color=BLUE, lw=0.9,
         marker='o', markersize=1.5, markevery=1, label='TE')
ax1.plot(xdata, te_fit, color=RED, lw=1.5, ls='--',
         label=f'Linear fit  ({drift_kJ_ns_atom:.3g} {_drift_unit_short})')
ax1.legend(fontsize=8, loc='upper right')
finish(ax1, 'E_total (Ha)')

# ══════════════════════════════════════════════════════════════════════════
# 图 2 — TE residual（去除线性漂移后的积分器噪声）
# ══════════════════════════════════════════════════════════════════════════
fig2, ax2 = new_fig(
    f'TE residual (drift removed)  RMS = {rms_resid:.2g} J/mol'
    '  —  integrator noise'
)
ax2.plot(xdata, residual, color=GREEN, lw=0.9,
         marker='o', markersize=1.5, markevery=1)
ax2.axhline(0, color=GRAY, lw=0.8, ls='--')
finish(ax2, 'ΔE (J/mol)')

# ══════════════════════════════════════════════════════════════════════════
# 图 3 — Temperature time series
# NVE:     参考带 ±σ_NVE_approx = T/√N_dof（Allen & Tildesley §2.4）
# NVT/NPT: 参考带 ±σ_canonical = T·√(2/N_dof)（Frenkel & Smit §6.1）——更精确
# ══════════════════════════════════════════════════════════════════════════
fig3, ax3 = new_fig(
    f'Temperature  |  ⟨T⟩={T_mean:.1f} K  σ={T_std:.1f} K'
)
ax3.plot(xdata, T, color=RED, lw=0.9,
         marker='o', markersize=1.5, markevery=1)
ax3.axhline(T_mean, color=GRAY, lw=1.2, ls='--',
            label=f'⟨T⟩ = {T_mean:.1f} K')

if is_nve and T_sigma_nve_approx is not None:
    # NVE 近似带（Allen & Tildesley §2.4）
    ax3.fill_between(xdata,
                     T_mean - T_sigma_nve_approx,
                     T_mean + T_sigma_nve_approx,
                     alpha=0.18, color=ORANGE,
                     label=f'±σ_NVE = ±{T_sigma_nve_approx:.1f} K  [A&T §2.4]')
elif T_sigma_canonical is not None:
    # NVT/NPT canonical 带（Frenkel & Smit §6.1）——主要参考线
    ok_sigma = 0.5 < T_sigma_ratio < 2.0
    ax3.fill_between(xdata,
                     T_mean - T_sigma_canonical,
                     T_mean + T_sigma_canonical,
                     alpha=0.18, color=GREEN,
                     label=(
                         f'±σ_canonical = ±{T_sigma_canonical:.1f} K  '
                         f'[F&S §6.1]  ratio={T_sigma_ratio:.2f}'
                         + ('  ✓' if ok_sigma else '  ✗')
                     ))
    # 也画观测带（灰色虚线），方便对比
    ax3.fill_between(xdata,
                     T_mean - T_std,
                     T_mean + T_std,
                     alpha=0.10, color=RED,
                     label=f'±σ_obs = ±{T_std:.1f} K')
else:
    ax3.fill_between(xdata, T_mean - T_std, T_mean + T_std,
                     alpha=0.15, color=RED, label=f'±σ = ±{T_std:.1f} K')

ax3.legend(fontsize=8, loc='upper right')
finish(ax3, 'T (K)')

# ══════════════════════════════════════════════════════════════════════════
# 图 4 — KE vs PE time series（双 Y 轴）
# NVE:     r ≈ -1（能量守恒 → 完全反相关）
# NVT/NPT: r ≈  0（恒温器每步重置 KE → 解耦）
# 判据和标注随 ensemble 自动切换（Shirts & Chodera 2008）
# ══════════════════════════════════════════════════════════════════════════
if KE is not None and PE is not None:
    r_str = f'r = {r_kepe:.4f}' if r_kepe is not None else ''
    if is_nve:
        corr_verdict = ('  ✓' if ok_r else '  ✗') if ok_r is not None else ''
        title4 = f'KE – PE  [NVE: target r ≈ −1]  {r_str}{corr_verdict}'
    else:
        corr_verdict = ('  ✓' if ok_r else '  ✗') if ok_r is not None else ''
        title4 = f'KE – PE  [NVT/NPT: target r ≈ 0]  {r_str}{corr_verdict}'

    fig4, ax4 = new_fig(title4 + '  [Shirts & Chodera, JCP 2008]')
    ax4r = ax4.twinx()
    l1, = ax4.plot(xdata, KE, color=BLUE, lw=0.9,
                   marker='o', markersize=1.5, markevery=1, label='KE')
    l2, = ax4r.plot(xdata, PE, color=RED, lw=0.9,
                    marker='o', markersize=1.5, markevery=1, alpha=0.7, label='PE')
    ax4.set_ylabel('KE (Ha)', color=BLUE, fontsize=9)
    ax4r.set_ylabel('PE (Ha)', color=RED, fontsize=9)
    ax4.tick_params(axis='y', labelcolor=BLUE)
    ax4r.tick_params(axis='y', labelcolor=RED)
    ax4.ticklabel_format(useOffset=False, style='plain', axis='y')
    ax4r.ticklabel_format(useOffset=False, style='plain', axis='y')
    ax4.set_xlabel(xlabel, fontsize=9)
    ax4.grid(True, linestyle='--', alpha=0.35)
    ax4.xaxis.set_minor_locator(AutoMinorLocator())
    ax4.legend(handles=[l1, l2], fontsize=8, loc='upper right')
    fig4.tight_layout()

# ══════════════════════════════════════════════════════════════════════════
# 图 5 — NPT: Pressure  /  NVE: KE-PE scatter  /  NVT: KE 分布检验
# ══════════════════════════════════════════════════════════════════════════
if is_npt and press is not None:
    # NPT: 压力时间序列
    fig5, ax5 = new_fig(
        f'Pressure  |  ⟨P⟩={press.mean():.1f} bar  σ={press.std():.1f} bar'
    )
    ax5.plot(xdata, press, color=BLUE, lw=0.9,
             marker='o', markersize=1.5, markevery=1)
    ax5.axhline(press.mean(), color=GRAY, lw=1.2, ls='--',
                label=f'⟨P⟩ = {press.mean():.1f} bar')
    ax5.legend(fontsize=8)
    finish(ax5, 'P (bar)')

elif is_nvt and KE is not None and n_dof:
    # NVT: KE 分布 vs Gamma 理论曲线（Shirts & Chodera 2008）
    kB_Ha    = 3.166811e-6
    shape_th = n_dof / 2.0
    scale_th = kB_Ha * T_mean
    ke_mean  = KE.mean()
    ke_std   = KE.std()

    pval_str = f'  K-S D = {ks_stat:.4f}' if ks_stat is not None else ''
    ok_str   = ('  ✓' if ke_gamma_ok else '  ✗') if ks_pval is not None else ''
    fig5, ax5 = new_fig(
        f'KE distribution  [NVT: Gamma({shape_th:.0f}/2, k_BT)]'
        f'{pval_str}{ok_str}  [Shirts & Chodera, JCP 2008]'
    )
    ax5.hist(KE, bins=40, color=BLUE, alpha=0.6,
             edgecolor='white', lw=0.4, density=True, label='Simulation')
    kx = np.linspace(KE.min(), KE.max(), 400)
    ax5.plot(kx, gamma.pdf(kx, shape_th, scale=scale_th),
             color=RED, lw=1.8,
             label=f'Gamma theory  (T={T_mean:.1f} K)')
    ax5.legend(fontsize=8)
    finish(ax5, 'Probability density', 'KE (Ha)')

elif is_nve and KE is not None and PE is not None:
    # NVE: KE-PE scatter（已有 r 值）
    fig5, ax5 = new_fig(
        'KE vs PE scatter  [NVE, Shirts & Chodera, JCP 129, 124105 (2008)]'
    )
    ax5.scatter(KE, PE, s=5, alpha=0.45, color=BLUE, edgecolors='none')
    m_s, b_s = np.polyfit(KE, PE, 1)
    kx = np.linspace(KE.min(), KE.max(), 200)
    ax5.plot(kx, m_s * kx + b_s, color=RED, lw=1.5,
             label=f'slope = {m_s:.2f}   r = {r_kepe:.4f}')
    ax5.legend(fontsize=8)
    finish(ax5, 'PE (Ha)', 'KE (Ha)')

# ══════════════════════════════════════════════════════════════════════════
# 图 6 — NPT: Volume  /  NVT/NVE: Temperature distribution
# NVE:     Gaussian 拟合（A&T §2.4，大 N 极限近似）
# NVT/NPT: 正则系综 Gaussian 拟合 + canonical σ 参考线（F&S §6.1）
# ══════════════════════════════════════════════════════════════════════════
if is_npt and vol is not None:
    fig6, ax6 = new_fig(
        f'Volume  |  ⟨V⟩={vol.mean():.2f} Å³  σ={vol.std():.2f} Å³'
    )
    ax6.plot(xdata, vol, color=GREEN, lw=0.9,
             marker='o', markersize=1.5, markevery=1)
    ax6.axhline(vol.mean(), color=GRAY, lw=1.2, ls='--',
                label=f'⟨V⟩ = {vol.mean():.2f} Å³')
    ax6.legend(fontsize=8)
    finish(ax6, 'V (Å³)')

else:
    if is_nve:
        ref_str = '[Allen & Tildesley 2017 §2.4]'
        extra_legend = None
    else:
        ref_str = '[Frenkel & Smit 2002 §6.1 — canonical σ = T·√(2/N_dof)]'
        extra_legend = None

    fig6, ax6 = new_fig(f'Temperature distribution  {ref_str}')
    ax6.hist(T, bins=40, color=BLUE, alpha=0.6,
             edgecolor='white', lw=0.4, density=True, label='Simulation')

    tx = np.linspace(T.min(), T.max(), 400)
    # 拟合曲线（模拟实际 σ）
    ax6.plot(tx, norm.pdf(tx, T_mean, T_std),
             color=RED, lw=1.8,
             label=f'Gaussian fit  σ_obs={T_std:.1f} K')

    # NVT/NPT: 叠加 canonical 理论曲线（用 canonical σ，中心为 T_mean）
    if (is_nvt or is_npt) and T_sigma_canonical is not None:
        ax6.plot(tx, norm.pdf(tx, T_mean, T_sigma_canonical),
                 color=GREEN, lw=1.5, ls='--',
                 label=f'Canonical σ={T_sigma_canonical:.1f} K  [F&S §6.1]')

    ax6.legend(fontsize=8)
    finish(ax6, 'Probability density', 'T (K)')

# ─── 显示所有窗口 ─────────────────────────────────────────────────────────
try:
    plt.show()
except Exception as e:
    stem = Path(filename).stem
    for i, fig in enumerate(map(plt.figure, plt.get_fignums()), 1):
        out = f"{stem}_validation_{i}.png"
        fig.savefig(out, bbox_inches='tight', dpi=150)
        print(f"  已保存: {out}")
EOF
