"""
Generate weekly progress PPT for MAPLE MD module work.
Style: white background, black text, minimal colour.
"""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import io

# ── palette (minimal) ────────────────────────────────────────────────────────
BLACK  = RGBColor(0x1A, 0x1A, 0x1A)
GREY   = RGBColor(0x88, 0x88, 0x88)
BLUE   = RGBColor(0x1F, 0x77, 0xB4)   # matplotlib default blue
RED    = RGBColor(0xD6, 0x27, 0x28)
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
LGGREY = RGBColor(0xF2, 0xF2, 0xF2)

SLIDE_W = Inches(13.33)
SLIDE_H = Inches(7.5)

# ── helpers ───────────────────────────────────────────────────────────────────

def set_bg_white(slide):
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = WHITE

def tb(slide, text, left, top, width, height,
       size=14, bold=False, color=BLACK, align=PP_ALIGN.LEFT, italic=False):
    txb = slide.shapes.add_textbox(left, top, width, height)
    tf  = txb.text_frame
    tf.word_wrap = True
    p   = tf.paragraphs[0]
    p.alignment = align
    r   = p.add_run()
    r.text = text
    r.font.size   = Pt(size)
    r.font.bold   = bold
    r.font.italic = italic
    r.font.color.rgb = color
    return txb

def hline(slide, top, left=Inches(0.4), width=None):
    """Thin horizontal rule."""
    w = width or (SLIDE_W - Inches(0.8))
    line = slide.shapes.add_shape(1, left, top, w, Inches(0.02))
    line.fill.solid()
    line.fill.fore_color.rgb = RGBColor(0xCC, 0xCC, 0xCC)
    line.line.fill.background()

def slide_header(slide, title, subtitle=None):
    hline(slide, Inches(0.1))
    tb(slide, title, Inches(0.4), Inches(0.15), Inches(12.5), Inches(0.55),
       size=26, bold=True, color=BLACK)
    if subtitle:
        tb(slide, subtitle, Inches(0.4), Inches(0.72), Inches(12.5), Inches(0.35),
           size=12, color=GREY, italic=True)
    hline(slide, Inches(1.1))

def fig_to_buf(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor='white')
    buf.seek(0)
    plt.close(fig)
    return buf

def add_fig(slide, fig, left, top, width, height):
    buf = fig_to_buf(fig)
    slide.shapes.add_picture(buf, left, top, width, height)

# ── matplotlib style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'axes.edgecolor':   '#333333',
    'axes.linewidth':   0.8,
    'axes.grid':        True,
    'grid.color':       '#DDDDDD',
    'grid.linewidth':   0.5,
    'xtick.color':      '#333333',
    'ytick.color':      '#333333',
    'text.color':       '#1A1A1A',
    'font.family':      'sans-serif',
    'font.size':        10,
})

# ── figures ───────────────────────────────────────────────────────────────────

def fig_commit_timeline():
    commits = [
        ('3105de1', 'feat: add NVT/NPT ensembles'),
        ('e655188', 'Add --version flag'),
        ('7be3601', 'bump version pbc-md'),
        ('6205ea7', 'fix: inline coord parsing'),
        ('838af77', 'refactor: SetCalculator'),
        ('efc22c8', 'feat: enhance MD defaults ★'),
        ('7190033', 'chore: NVE examples & tools ★'),
    ]
    n = len(commits)
    fig, ax = plt.subplots(figsize=(11, 2.2))
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-1, 1)
    ax.axis('off')

    ax.plot(range(n), [0]*n, color='#AAAAAA', lw=1.5, zorder=1)
    for i, (sha, msg) in enumerate(commits):
        this_week = msg.endswith('★')
        col = '#1F77B4' if this_week else '#888888'
        ax.plot(i, 0, 'o', color=col, markersize=10, zorder=3)
        y = 0.45 if i % 2 == 0 else -0.45
        ax.text(i, y, msg.replace(' ★', ''), ha='center',
                va='bottom' if y > 0 else 'top',
                fontsize=8, color='#1A1A1A',
                fontstyle='normal')
        ax.plot([i, i], [0.05 if y > 0 else -0.05, y * 0.85],
                color='#CCCCCC', lw=0.8)

    # legend
    ax.plot([], [], 'o', color='#1F77B4', label='This week')
    ax.plot([], [], 'o', color='#888888', label='Prior')
    ax.legend(loc='lower right', fontsize=8, frameon=False)
    ax.set_title('Commits ahead of upstream/feat/pbc-md  (★ = this week)',
                 fontsize=10, pad=4)
    fig.tight_layout(pad=0.3)
    return fig


def fig_files_changed():
    data = [
        ('logger.py',          920),
        ('nve.py',             429),
        ('utils.py',           277),
        ('set_calculator.py',  295),
        ('nvt.py',             249),
        ('npt.py',             230),
        ('command_control.py', 180),
        ('langevin.py',         89),
        ('velocity_verlet.py',  50),
        ('input_reader.py',     99),
    ]
    data.sort(key=lambda x: x[1])
    names  = [d[0] for d in data]
    values = [d[1] for d in data]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(names, values, color='#1F77B4', height=0.55)
    for bar, v in zip(bars, values):
        ax.text(v + 8, bar.get_y() + bar.get_height()/2,
                str(v), va='center', fontsize=8)
    ax.set_xlabel('Lines inserted')
    ax.set_title('Source code changes vs upstream/feat/pbc-md')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout(pad=0.5)
    return fig


def fig_arch():
    """Simple text-box architecture diagram."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis('off')

    def box(x, y, w, h, label, fill='#F2F2F2', edge='#333333', fs=9):
        from matplotlib.patches import FancyBboxPatch
        rect = FancyBboxPatch((x, y), w, h,
                              boxstyle='round,pad=0.06',
                              facecolor=fill, edgecolor=edge, linewidth=1)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                fontsize=fs, multialignment='center')

    def arr(x1, y, x2):
        ax.annotate('', xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle='->', color='#555555', lw=1.2))

    # top row
    box(0.1, 3.2, 1.8, 0.65, '#md(...)\nparser', fill='#DDEEFF')
    arr(1.9, 3.52, 2.7)
    box(2.7, 3.2, 2.0, 0.65, 'engine.py\ndispatcher.py', fill='#DDEEFF')
    arr(4.7, 3.52, 5.5)
    box(5.5, 3.2, 4.3, 0.65, 'NVE / NVT / NPT  ensemble', fill='#DDEEFF', edge='#1F77B4')

    # middle: integrator + utils
    box(0.1, 1.85, 2.2, 0.85, 'Velocity Verlet\nintegrator')
    box(2.5, 1.85, 2.2, 0.85, 'utils.py\nunit conv, MB init')
    box(4.9, 1.85, 2.2, 0.85, 'logger.py\nthermo / traj / stats')

    # thermostats
    box(0.1, 0.5, 2.2, 0.9, 'Langevin\n(BAOAB)')
    box(2.5, 0.5, 2.2, 0.9, 'V-rescale\n(Bussi 2007)')

    # barostats
    box(4.9, 0.5, 2.2, 0.9, 'Berendsen\nbarostat')
    box(7.3, 0.5, 2.4, 0.9, 'C-rescale\n(Bernetti 2020)')

    # connectors (vertical)
    for x in [1.2, 3.6, 6.0]:
        ax.plot([x, x], [3.2, 2.7], color='#AAAAAA', lw=0.9)
    for x in [1.2, 3.6]:
        ax.plot([x, x], [1.85, 1.4], color='#AAAAAA', lw=0.9)
    for x in [6.0, 8.5]:
        ax.plot([x, x], [1.85, 1.4], color='#AAAAAA', lw=0.9)

    # layer labels (left margin)
    for y, lbl in [(3.45, 'Input / Dispatch'), (2.2, 'MD Core'),
                   (0.85, 'Thermostat / Barostat')]:
        ax.text(-0.05, y, lbl, ha='right', va='center',
                fontsize=7.5, color='#666666', style='italic')

    ax.set_title('MD module architecture  (dispatcher/md/)', fontsize=10)
    fig.tight_layout(pad=0.3)
    return fig


def fig_dt_sweep():
    dts    = ['0.125 fs', '0.25 fs', '0.50 fs']
    sigma  = [0.00211816 * 627.509,   # Ha → kcal/mol
              0.00172788 * 627.509,
              0.00189509 * 627.509]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    colors = ['#888888', '#1F77B4', '#888888']
    bars   = ax.bar(dts, sigma, color=colors, width=0.45, edgecolor='white')
    ax.set_ylabel('σ(E_total)  [kcal/mol]')
    ax.set_title('NVE energy fluctuation vs timestep\nAla-Glu dipeptide, 30 atoms, UMA-S, 5000 steps')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for bar, v in zip(bars, sigma):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f'{v:.3f}', ha='center', fontsize=9)
    ax.annotate('minimum → new default',
                xy=(1, sigma[1]), xytext=(1.6, sigma[1] + 0.04),
                arrowprops=dict(arrowstyle='->', color='#333333'),
                fontsize=8)
    fig.tight_layout(pad=0.4)
    return fig


def fig_nve_trace():
    steps = np.arange(100, 2300, 100)
    E = np.array([
        -779.089295, -779.086143, -779.087270, -779.087555, -779.091029,
        -779.086190, -779.088210, -779.095526, -779.093179, -779.097000,
        -779.095335, -779.090245, -779.092642, -779.091045, -779.086509,
        -779.090589, -779.090390, -779.090525, -779.091568, -779.090847,
        -779.086711, -779.085709])
    T = np.array([
        320.66, 352.46, 341.23, 338.18, 303.15,
        352.21, 331.56, 257.60, 281.57, 242.86,
        259.59, 311.28, 286.94, 302.95, 348.90,
        307.61, 309.60, 308.32, 297.61, 305.09,
        347.04, 356.89])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4), sharex=True)
    ax1.plot(steps, E, color='#1F77B4', lw=1.2)
    ax1.axhline(E.mean(), color='#D62728', lw=0.8, ls='--', label=f'mean = {E.mean():.4f} Ha')
    ax1.set_ylabel('E_total (Ha)')
    ax1.legend(fontsize=8, frameon=False)
    ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)

    ax2.plot(steps, T, color='#333333', lw=1.2)
    ax2.axhline(300, color='#D62728', lw=0.8, ls='--', label='T_init = 300 K')
    ax2.set_ylabel('T (K)')
    ax2.set_xlabel('Step')
    ax2.legend(fontsize=8, frameon=False)
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

    fig.suptitle('NVE run — Ala-Glu dipeptide, dt = 0.25 fs', fontsize=10)
    fig.tight_layout(pad=0.4)
    return fig


def fig_workflow():
    fig, ax = plt.subplots(figsize=(10, 2.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 2); ax.axis('off')

    steps = [
        (0.3,  'Structure\n(.xyz)'),
        (2.5,  'Geometry\nOptimisation'),
        (5.0,  'NVT\nEquilibration\n100 ps, 300 K'),
        (7.5,  'NVE\nProduction\n1 ns'),
    ]
    labels = ['', '#opt', '#md(ensemble=nvt)', '#md(ensemble=nve,\ninit_from=...)']

    for i, (x, lbl) in enumerate(steps):
        fc = '#DDEEFF' if i > 0 else '#F2F2F2'
        from matplotlib.patches import FancyBboxPatch
        rect = FancyBboxPatch((x, 0.4), 1.8, 1.1,
                              boxstyle='round,pad=0.06',
                              facecolor=fc, edgecolor='#333333', lw=1)
        ax.add_patch(rect)
        ax.text(x + 0.9, 0.95, lbl, ha='center', va='center',
                fontsize=8.5, multialignment='center')

    for x in [2.1, 4.3, 6.8]:
        ax.annotate('', xy=(x + 0.4, 0.95), xytext=(x, 0.95),
                    arrowprops=dict(arrowstyle='->', color='#555555', lw=1.2))

    # init_from annotation
    ax.annotate('init_from=*_md_final.xyz', xy=(7.5, 0.4), xytext=(5.5, 0.1),
                arrowprops=dict(arrowstyle='->', color='#888888', lw=0.9, ls='dashed'),
                fontsize=7.5, color='#555555')

    ax.set_title('Simulation workflow — NVT pre-equilibration → NVE production', fontsize=10)
    fig.tight_layout(pad=0.3)
    return fig


# ══════════════════════════════════════════════════════════════════════════════

def build():
    prs = Presentation()
    prs.slide_width  = SLIDE_W
    prs.slide_height = SLIDE_H
    blank = prs.slide_layouts[6]

    # ── 1. Title ──────────────────────────────────────────────────────────────
    s = prs.slides.add_slide(blank)
    set_bg_white(s)
    hline(s, Inches(2.5))
    tb(s, 'MAPLE MD Module', Inches(1.0), Inches(1.0), Inches(11), Inches(1.2),
       size=40, bold=True, align=PP_ALIGN.CENTER)
    tb(s, 'Weekly Progress Report — 2026-03-19',
       Inches(1.0), Inches(2.6), Inches(11), Inches(0.5),
       size=18, color=GREY, align=PP_ALIGN.CENTER)
    hline(s, Inches(3.2))
    tb(s, 'Branch: feat/pbc-md   ·   7 commits ahead of upstream   ·   +4 512 lines',
       Inches(1.0), Inches(3.3), Inches(11), Inches(0.4),
       size=13, color=GREY, align=PP_ALIGN.CENTER, italic=True)

    for i, (val, lbl) in enumerate([
        ('7', 'Commits'), ('+4512', 'Lines added'), ('23', 'Files changed'), ('2', 'New this week')
    ]):
        cx = Inches(1.2 + i * 2.8)
        tb(s, val, cx, Inches(4.3), Inches(2.4), Inches(0.7),
           size=34, bold=True, color=BLUE, align=PP_ALIGN.CENTER)
        tb(s, lbl, cx, Inches(5.0), Inches(2.4), Inches(0.4),
           size=12, color=GREY, align=PP_ALIGN.CENTER)

    # ── 2. Commit timeline ────────────────────────────────────────────────────
    s = prs.slides.add_slide(blank)
    set_bg_white(s)
    slide_header(s, 'Commit Timeline')
    add_fig(s, fig_commit_timeline(), Inches(0.4), Inches(1.2), Inches(12.5), Inches(2.8))

    rows = [
        ('3105de1', 'feat: add NVT/NPT MD ensembles — Langevin, V-rescale, Berendsen, C-rescale'),
        ('e655188', 'Add --version flag; bump version to 0.1.1+enhance3.13'),
        ('7be3601', 'chore: bump version to 0.1.1+pbc-md'),
        ('6205ea7', 'fix: inline coordinate parsing and log method bugs'),
        ('838af77', 'refactor: unify model file management in SetCalculator'),
        ('efc22c8', '[THIS WEEK]  feat: enhance MD defaults — literature-grounded params, restart, angular momentum'),
        ('7190033', '[THIS WEEK]  chore: add NVE examples (Ala-Glu dipeptide) and analysis tools'),
    ]
    for i, (sha, msg) in enumerate(rows):
        col = BLUE if '[THIS WEEK]' in msg else BLACK
        bold = '[THIS WEEK]' in msg
        tb(s, f'{sha}  {msg}', Inches(0.5), Inches(4.15 + i*0.44),
           Inches(12.3), Inches(0.4), size=10, color=col, bold=bold)

    # ── 3. Files changed ──────────────────────────────────────────────────────
    s = prs.slides.add_slide(blank)
    set_bg_white(s)
    slide_header(s, 'Files Changed', 'vs upstream/feat/pbc-md')
    add_fig(s, fig_files_changed(), Inches(0.4), Inches(1.2), Inches(7.2), Inches(5.6))

    tb(s, 'New files', Inches(7.8), Inches(1.2), Inches(5.2), Inches(0.4),
       size=13, bold=True)
    new_files = [
        'md/__init__.py',
        'md/ensemble/nve.py',
        'md/ensemble/nvt.py',
        'md/ensemble/npt.py',
        'md/integrator/velocity_verlet.py',
        'md/thermostat/langevin.py',
        'md/thermostat/vrescale.py',
        'md/barostat/berendsen.py',
        'md/barostat/crescale.py',
        'md/utils.py',
        'md/logger.py',
        'tools/align_traj.py',
        'tools/plot_md.sh',
    ]
    for i, f in enumerate(new_files):
        tb(s, f, Inches(7.8), Inches(1.65 + i*0.43),
           Inches(5.2), Inches(0.38), size=10, color=GREY)

    # ── 4. Architecture ───────────────────────────────────────────────────────
    s = prs.slides.add_slide(blank)
    set_bg_white(s)
    slide_header(s, 'MD Module Architecture', 'New dispatcher/md/ subpackage')
    add_fig(s, fig_arch(), Inches(0.3), Inches(1.2), Inches(12.7), Inches(5.6))

    # ── 5. dt sweep ───────────────────────────────────────────────────────────
    s = prs.slides.add_slide(blank)
    set_bg_white(s)
    slide_header(s, 'Timestep Benchmark — dt Sweep',
                 'Ala-Glu dipeptide (30 atoms, gas phase, UMA-S, NVE)')
    add_fig(s, fig_dt_sweep(), Inches(0.4), Inches(1.2), Inches(5.8), Inches(5.0))

    lines = [
        ('Finding', True, 13, BLACK),
        ('σ(TE) is nearly flat across dt = 0.125–0.5 fs:', False, 11, BLACK),
        ('  1.30 / 1.06 / 1.18 kcal/mol', False, 11, GREY),
        ('', False, 10, GREY),
        ('Energy fluctuation is set by MLFF prediction', False, 11, BLACK),
        ('noise, not by the integrator (which would scale', False, 11, BLACK),
        ('as dt²). dt = 0.25 fs achieves the minimum.', False, 11, BLACK),
        ('', False, 10, GREY),
        ('→  New NVE default: timestep = 0.25 fs', True, 12, BLUE),
        ('→  New NVE default: steps = 400 000 (= 100 ps)', True, 12, BLUE),
        ('', False, 10, GREY),
        ('References:', False, 10, GREY),
        ('Fu et al. (2023) JCTC 19, 1863', False, 9, GREY),
        ('Kovács et al. (2023) JPCL 14, 8725', False, 9, GREY),
        ('Zhang et al. (2023) J. Chem. Phys. 159, 054801', False, 9, GREY),
    ]
    for i, (txt, bold, size, col) in enumerate(lines):
        tb(s, txt, Inches(6.5), Inches(1.2 + i*0.39),
           Inches(6.5), Inches(0.37), size=size, bold=bold, color=col)

    # ── 6. NVE production run ─────────────────────────────────────────────────
    s = prs.slides.add_slide(blank)
    set_bg_white(s)
    slide_header(s, 'NVE Production Run',
                 'Ala-Glu dipeptide · NVT 10 ps pre-equilibration → NVE')
    add_fig(s, fig_nve_trace(), Inches(0.4), Inches(1.2), Inches(6.5), Inches(5.5))

    metrics = [
        ('Metric',               'Value',         'Pass'),
        ('σ(E)/|⟨E⟩|',          '2.22 × 10⁻⁶',   '< 1×10⁻⁴  ✓'),
        ('KE–PE correlation r',  '−1.0000',        '≈ −1  ✓'),
        ('σ(T)/σ_equip',         '0.96',           '0.5–2.0  ✓'),
        ('NVT ⟨T⟩',              '299.8 K',        '300 ± 5 K  ✓'),
        ('NVT σ(T)/σ_canonical', '0.998',          '0.5–2.0  ✓'),
        ('r(KE,PE) in NVT',      '−0.015',         '≈ 0  ✓'),
    ]
    col_xs = [Inches(7.0), Inches(9.5), Inches(11.3)]
    tb(s, 'Energy conservation metrics', Inches(7.0), Inches(1.2),
       Inches(5.9), Inches(0.4), size=12, bold=True)
    hline(s, Inches(1.65), left=Inches(7.0), width=Inches(5.9))
    for ri, row in enumerate(metrics):
        y = Inches(1.7 + ri * 0.52)
        for ci, (val, cx) in enumerate(zip(row, col_xs)):
            bold = ri == 0
            col  = GREY if ri == 0 else (BLUE if ci == 2 else BLACK)
            tb(s, val, cx, y, Inches(1.8), Inches(0.45),
               size=10, bold=bold, color=col)

    # ── 7. Workflow ───────────────────────────────────────────────────────────
    s = prs.slides.add_slide(blank)
    set_bg_white(s)
    slide_header(s, 'NVT → NVE Restart Workflow',
                 'New init_from parameter — load positions & velocities from prior run')
    add_fig(s, fig_workflow(), Inches(0.4), Inches(1.2), Inches(12.5), Inches(2.8))

    code = [
        '# Step 1: optimise geometry',
        '#model=uma',
        '#opt(method=lbfgs)',
        'C8H15N3O4   (inline coords)',
        '',
        '# Step 2: NVT equilibration — writes *_md_final.xyz',
        '#model=uma',
        '#md(ensemble=nvt, steps=100000, temperature=300, random_seed=42)',
        'XYZ 0 1 AG_opt.xyz',
        '',
        '# Step 3: NVE production — loads coords + velocities from NVT output',
        '#model=uma',
        '#md(ensemble=nve, steps=4000000, init_from=AG_opt_md_final.xyz)',
    ]
    tb(s, 'Example input sequence', Inches(0.4), Inches(4.1),
       Inches(12.5), Inches(0.35), size=11, bold=True)
    for i, line in enumerate(code):
        col = GREY if line.startswith('#') else BLACK
        tb(s, line, Inches(0.5), Inches(4.5 + i * 0.29),
           Inches(12.3), Inches(0.27), size=9, color=col,
           italic=line.startswith('#'))

    # ── 8. Parameter changes ──────────────────────────────────────────────────
    s = prs.slides.add_slide(blank)
    set_bg_white(s)
    slide_header(s, 'Parameter Defaults — Updated',
                 'All defaults now carry inline literature citations in docstrings')

    headers = ['Parameter', 'Before', 'After', 'Rationale']
    rows = [
        ['NVE timestep',     '0.5 fs',   '0.25 fs',   'dt-sweep: min σ(TE)'],
        ['NVE steps',        '1 000',    '400 000',    '100 ps @ 0.25 fs'],
        ['NVT timestep',     '0.5 fs',   '1.0 fs',     'Standard ML-FF (Zhang 2018)'],
        ['NVT steps',        '10 000',   '100 000',    '100 ps @ 1 fs'],
        ['traj_every',       '10',       '100',        'ML-MD 10× slower than classical FF'],
        ['init_from',        '—',        'path/file',  'NVT → NVE restart'],
        ['remove_rotation',  '—',        'True/False', 'Angular momentum removal'],
    ]
    col_xs = [Inches(0.4), Inches(3.5), Inches(5.5), Inches(7.5)]
    col_ws = [Inches(3.0), Inches(1.9), Inches(1.9), Inches(5.3)]

    for ci, (hdr, cx) in enumerate(zip(headers, col_xs)):
        tb(s, hdr, cx, Inches(1.3), col_ws[ci], Inches(0.38),
           size=11, bold=True, color=GREY)
    hline(s, Inches(1.72))

    for ri, row in enumerate(rows):
        y = Inches(1.78 + ri * 0.55)
        for ci, (val, cx) in enumerate(zip(row, col_xs)):
            col = RED if ci == 1 else (BLUE if ci == 2 else BLACK)
            tb(s, val, cx, y, col_ws[ci], Inches(0.48),
               size=11, color=col)

    tb(s, 'red = old value   blue = new value',
       Inches(0.4), Inches(6.4), Inches(5), Inches(0.35),
       size=9, color=GREY, italic=True)

    # ── 9. SetCalculator ──────────────────────────────────────────────────────
    s = prs.slides.add_slide(blank)
    set_bg_white(s)
    slide_header(s, 'UMA Calculator & SetCalculator Refactor',
                 'Unified model management; auto task detection; multi-model support')

    left = [
        ('Before', True, 12, BLACK),
        ('Hardcoded model path', False, 11, BLACK),
        ('Single model: uma-s-1p1 only', False, 11, BLACK),
        ('Manual task= selection required', False, 11, BLACK),
        ('Separate download logic per calculator', False, 11, BLACK),
    ]
    right = [
        ('After', True, 12, BLUE),
        ('Unified _MODEL_FILE / _HF_REPO constants', False, 11, BLACK),
        ('Multi-model: uma-s-1p1, uma-m-1p1, …', False, 11, BLACK),
        ('Auto task: PBC → omat, non-PBC → omol', False, 11, BLACK),
        ('Local checkpoint from model/ directory', False, 11, BLACK),
    ]
    for items, x0 in [(left, Inches(0.4)), (right, Inches(6.8))]:
        for i, (txt, bold, size, col) in enumerate(items):
            tb(s, txt, x0, Inches(1.3 + i * 0.55),
               Inches(6.0), Inches(0.48), size=size, bold=bold, color=col)

    hline(s, Inches(4.3))
    tb(s, 'Input syntax', Inches(0.4), Inches(4.4), Inches(12), Inches(0.35),
       size=11, bold=True)
    for i, ln in enumerate([
        '#model=uma                   → default: uma-s-1p1, task auto-detected from PBC',
        '#model=uma(task=oc20)        → explicit task selection',
        '#model=uma-m-1p1             → larger model  [NEW]',
        '#pbc(a, b, c, α, β, γ)      → flexible cell specification  [NEW]',
    ]):
        tb(s, ln, Inches(0.5), Inches(4.8 + i * 0.42),
           Inches(12.3), Inches(0.38), size=10, color=GREY, italic=True)

    # ── 10. Summary ───────────────────────────────────────────────────────────
    s = prs.slides.add_slide(blank)
    set_bg_white(s)
    slide_header(s, 'Summary')

    sections = [
        ('This week', [
            'Timestep default updated to 0.25 fs (dt-sweep benchmark, Ala-Glu dipeptide)',
            'init_from parameter — load NVT output as NVE starting point',
            'remove_rotation — angular momentum zeroing for isolated molecules',
            'logger rewritten: energy drift, σ(E)/|⟨E⟩|, KE–PE correlation, T fluctuation',
            'GROMACS-style file backup (#name.ext.N#), wall-clock speed reporting',
            'NVE examples (Ala-Glu), analysis tools (align_traj.py, plot_md.sh)',
        ], BLUE),
        ('Prior commits on branch', [
            'NVT + NPT ensembles: Langevin, V-rescale, Berendsen, C-rescale',
            'NVE ensemble with Velocity Verlet + PBC wrapping',
            'SetCalculator / UMA: multi-model support, auto task detection',
            'Inline coordinate / charge / spin parsing fixes',
            '--version CLI flag',
        ], GREY),
    ]

    y = Inches(1.2)
    for title, items, col in sections:
        tb(s, title, Inches(0.4), y, Inches(12.5), Inches(0.42),
           size=13, bold=True, color=col)
        y += Inches(0.45)
        for item in items:
            tb(s, '  • ' + item, Inches(0.4), y, Inches(12.5), Inches(0.38),
               size=11, color=BLACK)
            y += Inches(0.40)
        y += Inches(0.15)

    out = '/home/axie/MAPLE/MAPLE-fork/weekly_report_MD.pptx'
    prs.save(out)
    print(f'Saved: {out}')

if __name__ == '__main__':
    build()
