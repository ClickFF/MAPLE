# -*- coding: utf-8 -*-
"""
Intrinsic Reaction Coordinate (IRC) integrator using:
- A steepest-descent (SD) predictor in Cartesian coordinates
- A perpendicular correction step in the subspace orthogonal to SD
- Optional parabolic (quadratic) fit on SD or correction segment if uphill is detected
- Adaptive update of SD length
- Forward and backward paths from the transition-state (TS) geometry
- Path merge with the lower-energy endpoint set as dE=0 and the TS marked at the maximum energy point

Units:
- Coordinates: Å
- Forces: Eh/Å
- Energies: Eh
- dE: kcal/mol (for reporting)
"""

import os
from dataclasses import dataclass, fields
from typing import List, Optional, Dict, Tuple
import numpy as np
from ase import Atoms

from .logger import log_info, log_error


# =============================== Utilities ===============================
BOHR_TO_ANG = 0.529177210903
KCAL_PER_EH = 627.509474

def to_f64(x):
    if isinstance(x, np.ndarray):
        return x.astype(np.float64, copy=False)
    try:
        import torch
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
            return x.astype(np.float64, copy=False)
    except Exception:
        pass
    if np.isscalar(x):
        return float(x)
    return np.asarray(x, dtype=np.float64)

def v1(x, n=None):
    v = to_f64(x).reshape(-1)
    if n is not None and v.size != n:
        raise ValueError(f"Expected size {n}, got {v.size}")
    return v

def masses_D(atoms: Atoms) -> np.ndarray:
    """Return diagonal mass-weight scaling vector D with length 3N: q_mw = D * q_cart."""
    m = to_f64(atoms.get_masses())
    m = np.where(m > 0.0, m, 1.0)
    return v1(1.0 / np.sqrt(np.repeat(m, 3)))

def norm_cart(v):
    return float(np.linalg.norm(v1(v)))

def unit_cart(v, eps=1e-16):
    v = v1(v)
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n

def write_xyz(path: str, atoms_list: List[Atoms], energies: Optional[List[float]] = None):
    """Write a list of structures to an XYZ file."""
    with open(path, "w") as f:
        for i, at in enumerate(atoms_list):
            pos = at.get_positions()
            symbols = at.get_chemical_symbols()
            f.write(f"{len(symbols)}\n")
            if energies is not None and i < len(energies):
                f.write(f"Image {i}  Energy = {energies[i]:.10f}\n")
            else:
                f.write(f"Image {i}\n")
            for s, (x, y, z) in zip(symbols, pos):
                f.write(f"{s:2s} {x: .10f} {y: .10f} {z: .10f}\n")


# =============================== Parameters ===============================
@dataclass
class GSParams:
    # Negative mode selection (1 = most negative)
    target_mode: int = 1

    # Initial displacement using quadratic ΔE target (mEh → Eh)
    init_delta_E_mEh: float = 2.0
    use_quadratic_init: bool = True

    # SD step configuration (given in bohr; converted to Å)
    sd_len_bohr: float = 0.150
    sd_len_min_bohr: float = 0.050
    sd_len_max_bohr: float = 0.300

    # Correction length relative to SD length
    corr_scale: float = 1.0 / 3.0

    # Adaptive update factors
    grow: float = 1.20
    shrink: float = 0.80

    # Parabolic fit on uphill (interpolation only)
    do_parabolic_fit: bool = True

    # Convergence on forces (Eh/Å)
    tol_maxf: float = 2e-3
    tol_rmsf: float = 5e-4

    # Maximum macro steps per side
    max_points: int = 100

    # Output controls
    print_each: bool = True
    write_traj: bool = True


# ================================== GS ===================================
class GS:
    """
    IRC integrator with SD predictor, perpendicular correction, parabolic interpolation,
    adaptive step length, two-sided integration, and merged summary.
    """

    def __init__(self, atoms: Atoms, output: str, params: Optional[GSParams] = None, paras: Optional[dict] = None):
        self.atoms = atoms
        self.output = output
        self.p = params if params is not None else GSParams()

        # Optional dict override: accept {"gs": {...}} or flat dict of param names.
        if isinstance(paras, dict):
            low = {k.lower(): v for k, v in paras.items()}
            sub = None
            for key in ("gs", "irc"):
                if key in low and isinstance(low[key], dict):
                    sub = low[key]
                    break
            if sub is None:
                sub = low
            fmap = {f.name.lower(): f.name for f in fields(self.p)}
            for k, v in {k.lower(): v for k, v in sub.items()}.items():
                if k in fmap:
                    setattr(self.p, fmap[k], v)

    # ------------------------------ Public API ------------------------------
    def run(self) -> Dict[str, any]:
        """Compute both directions and produce a merged summary."""
        try:
            neg_mode_cart = self._get_negative_mode_cart()
        except Exception as e:
            log_error([f"[ERROR] Failed to obtain negative mode from Hessian: {e}\n"], self.output)
            raise

        forward_log = self._one_side(True,  +1.0, neg_mode_cart)
        backward_log = self._one_side(False, -1.0, neg_mode_cart)
        merged = self._merge_and_mark_ts(forward_log, backward_log)

        if self.p.write_traj:
            self._write_trajs(forward_log, backward_log)

        return {"forward": forward_log, "backward": backward_log, "summary": merged}

    # --------------------------- Hessian & mode -----------------------------
    def _get_negative_mode_cart(self) -> np.ndarray:
        """Diagonalize the mass-weighted Hessian and return the chosen negative mode in Cartesian coordinates."""
        H = self._get_hessian()
        D = masses_D(self.atoms)
        H_mw = (D[:, None] * H) * D[None, :]
        w, V = np.linalg.eigh(H_mw)

        neg_idx = np.where(w < 0.0)[0]
        if len(neg_idx) == 0:
            log_error(["[ERROR] No negative eigenvalues found — the geometry is not a saddle point.\n"], self.output)
            raise RuntimeError("No negative eigenvalues found.")
        if len(neg_idx) < self.p.target_mode:
            log_error([f"[ERROR] Requested mode {self.p.target_mode}, but only {len(neg_idx)} negative modes exist.\n"], self.output)
            raise RuntimeError("Requested negative mode does not exist.")

        sorted_neg = neg_idx[np.argsort(w[neg_idx])]   # ascending (most negative first)
        idx = sorted_neg[self.p.target_mode - 1]
        eigval = w[idx]
        log_info([f"\nSelected eigenmode #{self.p.target_mode} (λ = {eigval:.6e} in MW basis)\n"], self.output)

        v_mw = V[:, idx]
        v_cart = v_mw / D  # map back to Cartesian direction
        return v1(v_cart)

    def _get_hessian(self) -> np.ndarray:
        H = self.atoms.calc.get_hessian(self.atoms)
        H = to_f64(H)
        if H.ndim == 3 and H.shape[0] == 1:
            H = H[0]
        if H.ndim != 2 or H.shape[0] != H.shape[1]:
            raise ValueError(f"Hessian must be square 2D, got shape {H.shape}")
        return H

    # --------------------------- Parabolic fit ------------------------------
    @staticmethod
    def _parabolic_fit_interpolate(s0, E0, s1, E1, s2, E2) -> Tuple[bool, float]:
        """
        Fit E(s)=a s^2 + b s + c for three samples and return (ok, s_min) where s_min is strictly inside [s0,s2].
        Only interpolation is allowed; extrapolation is rejected.
        """
        try:
            coeff = np.polyfit([s0, s1, s2], [E0, E1, E2], 2)  # returns a, b, c
            a, b, c = coeff
            if abs(a) < 1e-20:
                return False, 0.0
            s_min = -b / (2.0 * a)
            s_lo, s_hi = (s0, s2) if s0 <= s2 else (s2, s0)
            if s_min <= s_lo or s_min >= s_hi:
                return False, 0.0
            return True, float(s_min)
        except Exception:
            return False, 0.0

    # --------------------------- One direction ------------------------------
    def _one_side(self, forward: bool, sign: float, neg_mode_cart: np.ndarray) -> Dict[str, any]:
        """
        Single-sided path integration:
        - Initial push along the chosen negative mode using a quadratic energy target.
        - Per step: SD predictor -> optional parabolic interpolation if uphill -> perpendicular correction
          -> optional parabolic interpolation if uphill -> adaptive SD length -> log and convergence check.
        """
        p = self.p
        title = "FORWARD IRC" if forward else "BACKWARD IRC"

        self._print_header(title)
        self._print_conv_thresholds(p.tol_maxf, p.tol_rmsf)

        # Reference energy at TS
        E_ts = float(self.atoms.get_potential_energy(force_consistent=True))

        # SD length bounds (Å)
        sd_len = float(np.clip(p.sd_len_bohr, p.sd_len_min_bohr, p.sd_len_max_bohr)) * BOHR_TO_ANG
        sd_min = p.sd_len_min_bohr * BOHR_TO_ANG
        sd_max = p.sd_len_max_bohr * BOHR_TO_ANG

        # Initial displacement using quadratic ΔE target on the negative mode
        D = masses_D(self.atoms)
        if p.use_quadratic_init:
            H = self._get_hessian()
            H_mw = (D[:, None] * H) * D[None, :]
            w_mw, _ = np.linalg.eigh(H_mw)
            lam_min = float(np.min(w_mw))
            if lam_min >= 0.0:
                log_error([f"[ERROR] {title}: no negative eigenvalue at the starting geometry.\n"], self.output)
                raise RuntimeError("No negative eigenvalue at start.")
            dE = p.init_delta_E_mEh * 1e-3  # Eh
            dq_mw = np.sqrt(2.0 * abs(dE) / abs(lam_min))
            u0 = unit_cart(neg_mode_cart * sign)
            alpha = dq_mw / (np.linalg.norm(D * u0) + 1e-16)  # ensures MW arc-length = dq_mw
            dx0 = alpha * u0
        else:
            u0 = unit_cart(neg_mode_cart * sign)
            dx0 = (sd_len) * u0

        R0 = v1(self.atoms.get_positions())
        self.atoms.set_positions((R0 + dx0).reshape(-1, 3))
        E0 = float(self.atoms.get_potential_energy(force_consistent=True))
        if E0 > E_ts + 1e-12:
            log_info([f"[WARNING]: {title} initial push increased energy by {(E0 - E_ts):.6e} Eh. Continuing.\n"], self.output)

        # Iteration 0 logging
        F0 = to_f64(self.atoms.get_forces()).reshape(-1)
        maxF0 = float(np.max(np.abs(F0)))
        rmsF0 = float(np.sqrt(np.mean(F0 ** 2)))
        self._print_iter_line(0, E0, (E0 - E_ts) * KCAL_PER_EH, maxF0, rmsF0)

        records: List[Dict[str, any]] = []
        records.append({"E": E0, "maxG": maxF0, "rmsG": rmsF0, "x": self.atoms.get_positions().copy()})

        F_prev = F0  # for adaptive rule reference
        uphill_count = 0

        for it in range(1, p.max_points + 1):
            # -------- SD predictor (Cartesian) --------
            R_anchor = v1(self.atoms.get_positions())
            E_anchor = float(self.atoms.get_potential_energy(force_consistent=True))
            F = to_f64(self.atoms.get_forces()).reshape(-1)  # Eh/Å
            t = unit_cart(F)  # SD direction = +F
            if np.allclose(t, 0.0):
                log_error([f"[ERROR] {title}: zero force encountered.\n"], self.output)
                raise RuntimeError("Zero force encountered.")

            used_fit_on_sd = False
            dx_sd = sd_len * t
            R_sd = R_anchor + dx_sd
            self.atoms.set_positions(R_sd.reshape(-1, 3))
            E_sd = float(self.atoms.get_potential_energy(force_consistent=True))

            if p.do_parabolic_fit and (E_sd > E_anchor + 1e-12):
                # Parabolic interpolation on [0, sd_len] along t
                R_mid = R_anchor + 0.5 * dx_sd
                self.atoms.set_positions(R_mid.reshape(-1, 3))
                E_mid = float(self.atoms.get_potential_energy(force_consistent=True))
                ok, s_min = self._parabolic_fit_interpolate(0.0, E_anchor, 0.5 * sd_len, E_mid, sd_len, E_sd)
                if ok:
                    R_best = R_anchor + (s_min / sd_len) * dx_sd
                    self.atoms.set_positions(R_best.reshape(-1, 3))
                    E_sd = float(self.atoms.get_potential_energy(force_consistent=True))
                    used_fit_on_sd = True
                else:
                    # Go back to SD end even if uphill; continue with warning
                    self.atoms.set_positions(R_sd.reshape(-1, 3))
                    log_info([f"[WARNING]: {title} step {it} SD uphill; parabolic fit failed to interpolate. Continuing.\n"], self.output)

            # -------- Perpendicular correction --------
            # g = ∇E = -F; g_perp = g - (g·t)t; correction along -g_perp
            F_now = to_f64(self.atoms.get_forces()).reshape(-1)
            g_now = -F_now
            g_perp = g_now - np.dot(g_now, t) * t
            used_fit_on_corr = False

            if norm_cart(g_perp) > 0.0:
                u_corr = unit_cart(-g_perp)
                corr_len = self.p.corr_scale * sd_len
                dx_corr = corr_len * u_corr

                R_sd_end = v1(self.atoms.get_positions())
                R_corr_end = R_sd_end + dx_corr
                self.atoms.set_positions(R_corr_end.reshape(-1, 3))
                E_corr_end = float(self.atoms.get_potential_energy(force_consistent=True))

                if p.do_parabolic_fit and (E_corr_end > E_sd + 1e-12):
                    # Interpolate along correction segment [0, corr_len]
                    R_midc = R_sd_end + 0.5 * dx_corr
                    self.atoms.set_positions(R_midc.reshape(-1, 3))
                    E_midc = float(self.atoms.get_potential_energy(force_consistent=True))
                    okc, s_min_c = self._parabolic_fit_interpolate(0.0, E_sd, 0.5 * corr_len, E_midc, corr_len, E_corr_end)
                    if okc:
                        R_bestc = R_sd_end + (s_min_c / corr_len) * dx_corr
                        self.atoms.set_positions(R_bestc.reshape(-1, 3))
                        E_corr_end = float(self.atoms.get_potential_energy(force_consistent=True))
                        used_fit_on_corr = True
                    else:
                        # Keep correction end; continue with warning
                        self.atoms.set_positions(R_corr_end.reshape(-1, 3))
                        log_info([f"[WARNING]: {title} step {it} correction uphill; parabolic fit failed to interpolate. Continuing.\n"], self.output)
            # else: no correction step if g_perp ~ 0

            # -------- Accept & log --------
            E_new = float(self.atoms.get_potential_energy(force_consistent=True))
            F_new = to_f64(self.atoms.get_forces()).reshape(-1)
            maxF = float(np.max(np.abs(F_new)))
            rmsF = float(np.sqrt(np.mean(F_new ** 2)))

            if E_new > records[-1]["E"] + 1e-12:
                log_info([f"[WARNING]: {title} step {it} increased energy by {(E_new - records[-1]['E']):.6e} Eh. Continuing.\n"], self.output)
                uphill_count += 1
            else:
                uphill_count = 0

            if p.print_each:
                self._print_iter_line(it, E_new, (E_new - E_ts) * KCAL_PER_EH, maxF, rmsF)

            records.append({"E": E_new, "maxG": maxF, "rmsG": rmsF, "x": self.atoms.get_positions().copy()})

            # -------- Convergence check --------
            if (maxF <= p.tol_maxf) and (rmsF <= p.tol_rmsf):
                self._print_hurray()
                break

            # -------- Adaptive update of SD length --------
            if used_fit_on_sd or used_fit_on_corr or uphill_count > 0:
                sd_len = max(sd_min, sd_len * p.shrink)
            else:
                rmsF_prev = float(np.sqrt(np.mean(F_prev ** 2)))
                # simple heuristic: if rmsF decreased sufficiently, grow a bit
                if rmsF < 0.9 * rmsF_prev:
                    sd_len = min(sd_max, sd_len * p.grow)

            F_prev = F_new

        return {"title": title, "records": records, "E_ts": E_ts}

    # --------------------------- Merge & summary ----------------------------
    def _merge_and_mark_ts(self, f: Dict, b: Dict) -> Dict:
        fR, bR = f["records"], b["records"]
        if not fR or not bR:
            log_error(["[ERROR] One IRC side is empty.\n"], self.output)
            raise RuntimeError("One IRC side is empty.")

        Ef_end, Eb_end = fR[-1]["E"], bR[-1]["E"]
        if Ef_end <= Eb_end:
            first, second, chosen = fR, bR, "forward"
        else:
            first, second, chosen = bR, fR, "backward"

        merged = []
        for k in range(len(first) - 1, -1, -1):
            merged.append(first[k])
        for k in range(0, len(second)):
            merged.append(second[k])

        # Mark TS at the maximum energy in the merged path
        E_list = [rec["E"] for rec in merged]
        ts_idx = int(np.argmax(E_list))
        E_ref = float(min(Ef_end, Eb_end))

        log_info([
            "\n---------------------------------------------------------------\n",
            "                       IRC PATH SUMMARY              \n",
            "---------------------------------------------------------------\n",
            "All forces are in Eh/Å.\n\n",
            "Step        E(Eh)      dE(kcal/mol)  max(|G|)   RMS(G) \n"
        ], self.output)

        rows = []
        for i, rec in enumerate(merged, start=1):
            dE_kcal = (rec["E"] - E_ref) * KCAL_PER_EH
            line = f"{i:4d}  {rec['E']:14.6f}  {dE_kcal:12.6f}    {rec['maxG']:8.6f}  {rec['rmsG']:8.6f}"
            if i - 1 == ts_idx:
                line += " <= TS"
            rows.append(line + "\n")
        log_info(rows, self.output)

        return {
            "chosen_forward": chosen,
            "E_ref": E_ref,
            "ts_index": ts_idx + 1,  # 1-based
            "merged_rows": rows
        }

    # --------------------------- Trajectories -----------------------------
    # --------------------------- Trajectories -----------------------------
    def _write_trajs(self, f: Dict, b: Dict):
        """Write full, forward, and backward trajectories as XYZ files with names derived from self.output."""
        base, _ = os.path.splitext(self.output)
        full_path = base + "_full.xyz"
        fwd_path = base + "_forward.xyz"
        bwd_path = base + "_backward.xyz"

        # Forward
        f_atoms, f_E = [], []
        for rec in f["records"]:
            a = self.atoms.copy()
            a.set_positions(rec["x"])
            f_atoms.append(a)
            f_E.append(rec["E"])
        write_xyz(fwd_path, f_atoms, f_E)

        # Backward
        b_atoms, b_E = [], []
        for rec in b["records"]:
            a = self.atoms.copy()
            a.set_positions(rec["x"])
            b_atoms.append(a)
            b_E.append(rec["E"])
        write_xyz(bwd_path, b_atoms, b_E)

        # Full (concatenate forward then backward)
        full_atoms = f_atoms + b_atoms
        full_E = f_E + b_E
        write_xyz(full_path, full_atoms, full_E)

        log_info([
            f"\n[INFO] IRC forward trajectory written to: {fwd_path}\n",
            f"[INFO] IRC backward trajectory written to: {bwd_path}\n",
            f"[INFO] Full IRC trajectory written to: {full_path}\n"
        ], self.output)


    # ----------------------------- Printing ------------------------------
    def _print_header(self, title: str):
        log_info([f"\n         {'*' * 61}\n",
                  f"         *{title.center(59)}*\n",
                  f"         {'*' * 61}\n\n"], self.output)

    def _print_conv_thresholds(self, tol_maxf: float, tol_rmsf: float):
        log_info(["Iteration    E(Eh)      dE(kcal/mol)  max(|G|)   RMS(G) \n",
                  f"Convergence thresholds                {tol_maxf:0.6f}  {tol_rmsf:0.6f}\n"], self.output)

    def _print_iter_line(self, i: int, E: float, dE_kcal: float, maxF: float, rmsF: float):
        log_info([f"{i:5d}  {E:14.6f}  {dE_kcal:12.6f}    {maxF:8.6f}  {rmsF:8.6f}\n"], self.output)

    def _print_hurray(self):
        log_info([
            "\n                      ***********************HURRAY********************\n",
            "                      ***            THE IRC HAS CONVERGED          ***\n",
            "                      *************************************************\n\n"
        ], self.output)
