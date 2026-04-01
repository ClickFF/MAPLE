#!/usr/bin/env python3
"""
align_traj.py — Remove overall rotation/translation from a MAPLE MD trajectory.

Uses the Kabsch algorithm (mass-weighted RMSD minimization) to superpose each
frame onto a reference frame (default: first frame).

Usage:
    python align_traj.py <traj.xyz> [options]

Options:
    --ref N       Reference frame index (0-based, default: 0)
    --out FILE    Output file (default: <stem>_aligned.xyz)
    --no-mass     Use uniform weights instead of mass-weighted alignment
"""

import argparse
import re
import sys
import numpy as np
from pathlib import Path


# ── Element masses (amu) ───────────────────────────────────────────────────────
_MASSES = {
    'H':1.008,'He':4.003,'Li':6.941,'Be':9.012,'B':10.811,'C':12.011,
    'N':14.007,'O':15.999,'F':18.998,'Ne':20.180,'Na':22.990,'Mg':24.305,
    'Al':26.982,'Si':28.086,'P':30.974,'S':32.065,'Cl':35.453,'Ar':39.948,
    'K':39.098,'Ca':40.078,'Sc':44.956,'Ti':47.867,'V':50.942,'Cr':51.996,
    'Mn':54.938,'Fe':55.845,'Co':58.933,'Ni':58.693,'Cu':63.546,'Zn':65.38,
    'Ga':69.723,'Ge':72.630,'As':74.922,'Se':78.971,'Br':79.904,'Kr':83.798,
    'Rb':85.468,'Sr':87.620,'Y':88.906,'Zr':91.224,'Nb':92.906,'Mo':95.960,
    'Ag':107.868,'I':126.904,'Pt':195.084,'Au':196.967,'Pb':207.200,
}

def _mass(sym):
    return _MASSES.get(sym, 12.0)


# ── MAPLE XYZ reader ───────────────────────────────────────────────────────────

_COMMENT_RE = re.compile(
    r"Frame\s+(\d+)\s+Energy\s*=\s*([\d.\-eE+]+)\s+Hartree"
    r"(?:\s+Cell\s*=\s*([\d.\s]+))?"
)

def read_maple_traj(path):
    """
    Parse a MAPLE _md_traj.xyz file.

    Returns list of dicts:
        frame_num, energy, symbols, positions (Å), velocities (a.u.) or None
    """
    lines = Path(path).read_text().splitlines()
    frames = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue
        try:
            n_atoms = int(stripped)
        except ValueError:
            i += 1
            continue

        if i + 1 + n_atoms >= len(lines):
            break

        comment = lines[i + 1]
        m = _COMMENT_RE.search(comment)
        frame_num = int(m.group(1)) if m else len(frames)
        energy    = float(m.group(2)) if m else float('nan')

        symbols, positions, velocities = [], [], []
        ok = True
        for j in range(n_atoms):
            parts = lines[i + 2 + j].split()
            if len(parts) < 4:
                ok = False
                break
            symbols.append(parts[0])
            positions.append([float(x) for x in parts[1:4]])
            if len(parts) >= 7:
                velocities.append([float(x) for x in parts[4:7]])

        if ok:
            frames.append({
                'frame_num':  frame_num,
                'energy':     energy,
                'symbols':    symbols,
                'positions':  np.array(positions),
                'velocities': np.array(velocities) if velocities else None,
            })
        i += 2 + n_atoms

    return frames


# ── Kabsch alignment ───────────────────────────────────────────────────────────

def kabsch(P, Q, weights=None):
    """
    Find rotation matrix R that minimises weighted RMSD between P and Q
    after centering.  Both (N,3).  Returns R (3,3).

    Kabsch (1976) Acta Cryst. A32, 922.
    """
    if weights is None:
        weights = np.ones(len(P))
    w = weights / weights.sum()

    # Weighted centroids
    cp = (w[:, None] * P).sum(axis=0)
    cq = (w[:, None] * Q).sum(axis=0)
    Pc = P - cp
    Qc = Q - cq

    # Covariance matrix
    H = (w[:, None] * Pc).T @ Qc   # (3,3)

    U, S, Vt = np.linalg.svd(H)

    # Ensure proper rotation (det = +1)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])

    R = Vt.T @ D @ U.T
    return R, cp, cq


def align_frame(pos, ref_pos, weights=None):
    """Return pos rotated+translated onto ref_pos."""
    R, cp, cq = kabsch(pos, ref_pos, weights)
    return (pos - cp) @ R.T + cq


# ── XYZ writer ────────────────────────────────────────────────────────────────

def write_frame(fh, frame, positions_aligned):
    n = len(frame['symbols'])
    fh.write(f"{n}\n")
    fh.write(f"Frame {frame['frame_num']}  Energy = {frame['energy']:.10f} Hartree\n")
    if frame['velocities'] is not None:
        for sym, (x, y, z), (vx, vy, vz) in zip(
                frame['symbols'], positions_aligned, frame['velocities']):
            fh.write(f"{sym:2s} {x:15.8f} {y:15.8f} {z:15.8f}"
                     f"  {vx:12.6f} {vy:12.6f} {vz:12.6f}\n")
    else:
        for sym, (x, y, z) in zip(frame['symbols'], positions_aligned):
            fh.write(f"{sym:2s} {x:15.8f} {y:15.8f} {z:15.8f}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('traj', help='Input trajectory (.xyz)')
    ap.add_argument('--ref', type=int, default=0, metavar='N',
                    help='Reference frame index (default: 0)')
    ap.add_argument('--out', default=None, metavar='FILE',
                    help='Output file (default: <stem>_aligned.xyz)')
    ap.add_argument('--no-mass', action='store_true',
                    help='Uniform weights instead of mass-weighted alignment')
    args = ap.parse_args()

    traj_path = Path(args.traj)
    if not traj_path.exists():
        sys.exit(f"ERROR: file not found: {traj_path}")

    out_path = Path(args.out) if args.out else traj_path.with_name(
        traj_path.stem + '_aligned.xyz')

    print(f"Reading {traj_path} ...", flush=True)
    frames = read_maple_traj(traj_path)
    if not frames:
        sys.exit("ERROR: no frames parsed")
    print(f"  {len(frames)} frames, {len(frames[0]['symbols'])} atoms")

    ref_idx = args.ref
    if ref_idx >= len(frames):
        sys.exit(f"ERROR: --ref {ref_idx} out of range (0–{len(frames)-1})")

    ref_pos = frames[ref_idx]['positions']
    symbols = frames[ref_idx]['symbols']

    if args.no_mass:
        weights = None
        print("  Alignment: uniform weights")
    else:
        weights = np.array([_mass(s) for s in symbols])
        print("  Alignment: mass-weighted Kabsch")

    print(f"  Reference frame: {ref_idx} (frame_num={frames[ref_idx]['frame_num']})")
    print(f"Writing {out_path} ...", flush=True)

    with open(out_path, 'w') as fh:
        for i, frame in enumerate(frames):
            aligned = align_frame(frame['positions'], ref_pos, weights)
            write_frame(fh, frame, aligned)
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(frames)} frames", flush=True)

    print(f"Done. Aligned trajectory written to {out_path}")


if __name__ == '__main__':
    main()
