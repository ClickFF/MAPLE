# -*- coding: utf-8 -*-
"""
Dimer implementation with:
- Minimum-mode following (rotation: minimize kappa = n^T H n)
- Translation with parallel flip once kappa < 0
- Optional HVP (autograd) callback to replace finite-difference (no Δ tuning)
- Trust-radius / max-step control
- Detailed human-readable logging and XYZ outputs per iteration
"""
from __future__ import annotations

import os
from contextlib import nullcontext
import math
import torch
from dataclasses import dataclass
from typing import Optional, Callable, List, Tuple

import numpy as np
from ase import Atoms
from maple.function.calculator._inputs import _state

from .logger import log_info
from ...jobABC import JobABC
from maple.function.read.filereader.pdb_reader import write_pdb_model, write_pdb_trajectory

# =============================================================================
# ------------------------------ Utilities ------------------------------------
# =============================================================================

def to_numpy_f64(x):
    """Convert input (numpy/torch/list/scalar) to float64 numpy array or float."""
    if isinstance(x, np.ndarray):
        return x.astype(np.float64, copy=False)
    try:
        import torch
        if isinstance(x, torch.Tensor):
            arr = x.detach().cpu().numpy()
            return arr.astype(np.float64, copy=False)
    except Exception:
        pass
    if np.isscalar(x):
        return float(x)
    return np.asarray(x, dtype=np.float64)

def vec1d(x, n_expected=None):
    """Convert to float64 1D vector and optionally check length."""
    v = to_numpy_f64(x).reshape(-1)
    if n_expected is not None and v.size != n_expected:
        raise ValueError(f"Expected size {n_expected}, got {v.size}")
    return v

def write_xyz(filename: str, images: List[Atoms], energies: Optional[List[float]] = None,
              forces=None):
    """Write supplied trajectory frames without evaluating them."""
    if images and images[0].info.get("pdb_template"):
        write_pdb_trajectory(filename, images, energies=energies)
        return

    blocks = []
    for i, at in enumerate(images):
        pos = at.get_positions()
        symbols = at.get_chemical_symbols()

        lines = [f"{len(symbols)}\n"]
        comment_parts = [f"Image {i}"]
        if energies is not None:
            comment_parts.append(f"Energy = {energies[i]:.10f}")
            comment_parts.append('energy_unit=Hartree')
        charge, mult = _state(at)
        comment_parts.append(f"charge={charge:.17g} mult={int(mult)}")
        if at.cell.rank:
            lattice = ' '.join(f'{value:.12g}' for value in at.cell.array.reshape(-1))
            pbc = ' '.join('T' if flag else 'F' for flag in at.pbc)
            comment_parts.append(f'Lattice="{lattice}" pbc="{pbc}"')
        if forces is not None:
            kind = 'constraint_adjusted' if at.constraints else 'raw'
            comment_parts.append('Properties=species:S:1:pos:R:3:forces:R:3')
            comment_parts.append(f'force_unit=Hartree/Angstrom force_kind={kind}')
        lines.append('  '.join(comment_parts) + '\n')
        for j, (s, (x, y, z)) in enumerate(zip(symbols, pos)):
            line = f"{s:2s} {x: .10f} {y: .10f} {z: .10f}"
            if forces is not None:
                fx, fy, fz = forces[i][j]
                line += f" {fx: .10f} {fy: .10f} {fz: .10f}"
            lines.append(line + '\n')
        blocks.append("".join(lines))

    with (nullcontext(filename) if hasattr(filename, 'write') else open(filename, "w")) as f:
        f.writelines(blocks)
        f.flush()


def write_all_images_xyz(filename: str, atoms: Atoms, energy: Optional[float] = None,
                         iteration: int = 0, forces=None):
    """Append the current geometry and its retained results."""
    if atoms.info.get("pdb_template"):
        remark = f"Iter {iteration}"
        if energy is not None:
            remark += f"  Energy = {energy:.10f}"
        with open(filename, 'w' if iteration <= 1 else 'a', encoding='utf-8') as f:
            write_pdb_model(f, atoms, atoms.info['pdb_template'],
                            model_index=iteration + 1, remark=remark)
        return

    pos = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()

    lines = [f"{len(symbols)}\n"]
    comment_parts = [f"Iter {iteration}"]
    if energy is not None:
        comment_parts.append(f"Energy = {energy:.10f}")
        comment_parts.append('energy_unit=Hartree')
    charge, mult = _state(atoms)
    comment_parts.append(f"charge={charge:.17g} mult={int(mult)}")
    if atoms.cell.rank:
        lattice = ' '.join(f'{value:.12g}' for value in atoms.cell.array.reshape(-1))
        pbc = ' '.join('T' if flag else 'F' for flag in atoms.pbc)
        comment_parts.append(f'Lattice="{lattice}" pbc="{pbc}"')
    if forces is not None:
        kind = 'constraint_adjusted' if atoms.constraints else 'raw'
        comment_parts.append('Properties=species:S:1:pos:R:3:forces:R:3')
        comment_parts.append(f'force_unit=Hartree/Angstrom force_kind={kind}')
    lines.append('  '.join(comment_parts) + '\n')
    for j, (s, (x, y, z)) in enumerate(zip(symbols, pos)):
        line = f"{s:2s} {x: .10f} {y: .10f} {z: .10f}"
        if forces is not None:
            fx, fy, fz = forces[j]
            line += f" {fx: .10f} {fy: .10f} {fz: .10f}"
        lines.append(line + '\n')

    with (nullcontext(filename) if hasattr(filename, 'write') else open(filename, "w" if iteration <= 1 else "a")) as f:
        f.writelines(lines)
        f.flush()

# =============================================================================
# ------------------------------ Dimer Params ---------------------------------
# =============================================================================

@dataclass
class DimerParams:
    # Rotation / curvature estimation
    use_hvp: bool = False               # if True, call hvp_fn(atoms, n_flat) for Hn
    delta: float = 0.005                 # Angstrom; used only if not use_hvp
    rot_max_iter: int = 5               # rotation inner iterations per outer step
    rot_alpha: float = 0.5              # rotation step factor on F_rot (unitless); small ~ (0.1~1)
    rot_f_max_th: float = 1.0e-3        # convergence threshold on max|F_rot| (Eh/Ang)
    rot_f_rms_th: float = 5.0e-4        # convergence threshold on RMS(F_rot)

    # Translation / trust region
    step0: float = 0.2                  # initial step scaling on search direction
    step_max: float = 0.15              # absolute max Cartesian displacement (Ang)
    trust_radius: float = 0.15          # same role as max_step; kept both for clarity

    # Convergence (translation forces)
    f_max_th: float = 5.0e-3            # max(|F_trans|) Eh/Ang
    f_rms_th: float = 1.0e-3            # RMS(F_trans) Eh/Ang
    kappa_to_flip: float = 0.0          # if kappa < this value, flip parallel component

    # Iterations
    batch_size: Optional[int] = None
    max_iter: int = 200

    # Metric / projections
    use_mass_weight: bool = False       # if True, use M-metric for dot/proj (mass weighted)
    remove_rigid: bool = True           # remove global translation/rotation from n (molecular)

    # Initialization of n
    n_init: str = "random"              # "random" | "force" | "given"
    n_given: Optional[np.ndarray] = None

    # Outputs
    save_traj: bool = True
    save_metrics: bool = True

# =============================================================================
# ------------------------------ Metric helpers -------------------------------
# =============================================================================

def _get_metric(atoms: Atoms, use_mass_weight: bool):
    if not use_mass_weight:
        return None
    m = to_numpy_f64(atoms.get_masses()).reshape(-1, 1)  # (N,1)
    M = np.repeat(m, 3, axis=1).reshape(-1)              # (3N,)
    return M                                             # diagonal metric entries

def _dot(v, w, M=None):
    """Metric dot: if M is None -> Euclidean; else mass-weighted sum(M * v * w)."""
    if M is None:
        return float(np.dot(v, w))
    return float(np.dot(M * v, w))

def _norm(v, M=None):
    val = _dot(v, v, M=M)
    return math.sqrt(max(val, 0.0))

def _proj_parallel(F, n, M=None):
    """Return parallel component of F along n (unit under metric)."""
    # assume n is normalized in metric (n^T M n = 1) if M given
    c = _dot(F, n, M=M)
    return c * n

def _proj_perp(F, n, M=None):
    return F - _proj_parallel(F, n, M=M)

def _normalize(n, M=None, eps=1e-20):
    dn = _norm(n, M=M)
    if dn < eps:
        raise ValueError("Zero-length direction encountered during normalization.")
    return n / dn

def _remove_rigid_body_components(n, atoms: Atoms, M=None):
    """
    Remove global translation & rotation components from direction n.
    Only meaningful for molecules (non-periodic). For simplicity:
    - Remove translation: subtract mean per axis (mass-weighted average if M given)
    - Remove rotation: project out 3 rotational modes around COM using cross r x axis
      (simple approximate projector; robust enough for search direction).
    """
    # translation
    N = len(atoms)
    X = to_numpy_f64(atoms.get_positions()).reshape(-1, 3)
    v = n.reshape(-1, 3).copy()
    if M is None:
        t = v.mean(axis=0, keepdims=True)
    else:
        mw = to_numpy_f64(atoms.get_masses()).reshape(-1, 1)
        t = (mw * v).sum(axis=0, keepdims=True) / (mw.sum() + 1e-20)
    v -= t

    # rotation (approx): project out components proportional to r x omega, omega = basis unit vectors
    # center positions
    rc = X.mean(axis=0)
    r = X - rc
    # three axes basis
    axes = np.eye(3)
    for k in range(3):
        rot_mode = np.cross(r, axes[k])  # (N,3)
        # project each component of v onto rot_mode
        a = (v * rot_mode).sum() / ((rot_mode * rot_mode).sum() + 1e-20)
        v -= a * rot_mode
    return v.reshape(-1)

# =============================================================================
# ------------------------------ Dimer class ----------------------------------
# =============================================================================

class Dimer(JobABC):
    """
    Dimer saddle search with optional HVP (autograd) backend.

    hvp_fn: Optional[Callable[[Atoms, np.ndarray], np.ndarray]]
        If provided and DimerParams.use_hvp=True, returns H @ n (shape (3N,))
        Else: we use finite-difference via two force calls at R ± Δ n.
    """
    def __init__(self,
                 output: str,
                 atoms_init: Atoms,
                 paras: Optional[dict] = None,
                 hvp_fn: Optional[Callable[[Atoms, np.ndarray], np.ndarray]] = None):
        super().__init__(output)

        self.atoms = atoms_init
        if self.atoms.calc is None:
            raise ValueError("atoms_init must have a working calculator set (atoms.calc).")

        # Initialize params from paras dict
        self.params = self._init_params(DimerParams, paras, ("dimer", "DIMER", "ts"))

        self.hvp_fn = hvp_fn if self.params.use_hvp else None

        # metric
        self.M = _get_metric(self.atoms, self.params.use_mass_weight)

        # init direction n
        self.n = self._init_direction()

        # trust region bookkeeping
        self.alpha = float(self.params.step0)

    # ------------------------------ init n -----------------------------------

    def _init_direction(self) -> np.ndarray:
        N = len(self.atoms)
        D = 3 * N
        p = self.params

        if p.n_init.lower() == "given" and (p.n_given is not None):
            n = vec1d(p.n_given, D)
        elif p.n_init.lower() == "force":
            F = -vec1d(self.atoms.get_forces(), D)  # gradient = -F; here use force itself
            n = F
        else:  # random
            rng = np.random.default_rng()
            n = rng.normal(size=D)

        if p.remove_rigid:
            n = _remove_rigid_body_components(n, self.atoms, M=self.M)
        n = _normalize(n, M=self.M)
        return n

    # ----------------------- curvature & rotation helpers ---------------------

    def _finite_diff_Hn_and_F(self, x_flat: np.ndarray, n: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (Hn, F_avg, F_diff) with central difference using forces at R ± Δ n.
        """
        p = self.params
        Δ = float(p.delta)

        # save & perturb positions
        X = x_flat.reshape(-1, 3)
        # +Δ
        self.atoms.set_positions((X + Δ * n.reshape(-1, 3)))
        F_plus = to_numpy_f64(self.atoms.get_forces()).reshape(-1)
        # -Δ
        self.atoms.set_positions((X - Δ * n.reshape(-1, 3)))
        F_minus = to_numpy_f64(self.atoms.get_forces()).reshape(-1)
        # restore
        self.atoms.set_positions(X)

        F_avg = 0.5 * (F_plus + F_minus)
        F_diff = 0.5 * (F_plus - F_minus)
        Hn = -(F_plus - F_minus) / (2.0 * Δ)  # central difference for Hn
        return Hn, F_avg, F_diff

    def _hvp_Hn_and_F(self, x_flat: np.ndarray, n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (Hn, F) using autograd HVP for Hn, and single force at R for F.
        """
        # force at current R
        F = to_numpy_f64(self.atoms.get_forces()).reshape(-1)
        # Hn via callback
        if self.hvp_fn is None:
            raise RuntimeError("use_hvp=True but hvp_fn is not provided.")
        Hn = vec1d(self.hvp_fn(self.atoms, n), n.size)
        return Hn, F

    def _rotate_minimize_kappa(self, x_flat: np.ndarray, n: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Do up to rot_max_iter steps of rotation to minimize kappa = n^T H n.
        Returns (n_new, max|F_rot|, rms(F_rot)).
        """
        p = self.params
        n_cur = n.copy()

        for _ in range(p.rot_max_iter):
            if self.hvp_fn is not None and p.use_hvp:
                Hn, F = self._hvp_Hn_and_F(x_flat, n_cur)
                # we only need Hn for rotation; F used later for translation
            else:
                Hn, F_avg, _ = self._finite_diff_Hn_and_F(x_flat, n_cur)
                F = F_avg  # keep F_avg for later translation when using FD

            # rotation force: (I - nn^T) Hn
            F_par = _proj_parallel(Hn, n_cur, M=self.M)
            F_rot = Hn - F_par

            max_frot = float(np.max(np.abs(F_rot)))
            rms_frot = float(math.sqrt(np.mean(F_rot * F_rot)))

            # convergence of rotation
            if (max_frot < p.rot_f_max_th) and (rms_frot < p.rot_f_rms_th):
                return n_cur, max_frot, rms_frot

            # gradient descent on kappa: n <- n - α * F_rot (and renormalize)
            n_next = n_cur - p.rot_alpha * F_rot
            if self.params.remove_rigid:
                n_next = _remove_rigid_body_components(n_next, self.atoms, M=self.M)
            n_next = _normalize(n_next, M=self.M)
            n_cur = n_next

        # after max rot steps return last
        return n_cur, max_frot, rms_frot

    # ------------------------------- main flow --------------------------------

    def atoms_to_xyz_block(self, atoms: Atoms) -> str:
        lines = []
        syms = atoms.get_chemical_symbols()
        pos = atoms.get_positions()
        for idx, (s, (x, y, z)) in enumerate(zip(syms, pos)):
            lines.append(f"{idx:<4d}{s:>2s}{x:18.4f}{y:18.4f}{z:18.4f}")
        return "\n".join(lines) + "\n"

    def _get_hvp(self, n):
        from maple.function.calculator._batch import hvp_single
        Hn, forces, energy = hvp_single(self.atoms.calc, self.atoms, n,
                                    self.params.delta, self.params.batch_size)
        device = getattr(self.atoms.calc, 'device', 'cpu')
        return tuple(torch.as_tensor(value, dtype=torch.float64, device=device).reshape(-1)
                     for value in (Hn, forces, energy))

    def run(self):
        """
        Main Dimer optimization loop with autograd-based Hessian-vector product (Hn).
        - Rotation uses atoms.calc.get_hvp(atoms, n) to compute Hn (no finite difference)
        - Translation also uses get_hvp to obtain (Hn, forces, energy) in one pass
        - No duplicate energy/force evaluations
        - Logs curvature, rotational force, mode flip, 4 PRFO-style criteria, and trust region
        """
        import torch

        p = self.params
        base, _ = os.path.splitext(self.output)
        ext = ".pdb" if self.atoms.info.get("pdb_template") else ".xyz"
        traj_file = base + "_dimer_traj" + ext
        ts_file = base + "_dimer_ts" + ext
        kcal_per_Eh = 627.509

        # ------------------ init direction & step size ------------------
        n = self.n.copy()        # initial dimer orientation (assumed normalized & rigid-body removed if requested)
        alpha = float(self.alpha)

        # ------------------ initial eval via autograd HVP ------------------
        # get forces & energy once (Hn unused for initial report)
        _, forces_t, energy_t = self._get_hvp(n)
        forces_np = forces_t.detach().cpu().numpy()
        E0 = float(energy_t.detach().cpu().item())
        maxF0 = float(np.max(np.linalg.norm(forces_np.reshape(-1, 3), axis=1)))
        rmsF0  = float(np.sqrt(np.mean(np.linalg.norm(forces_np.reshape(-1, 3), axis=1) ** 2)))

        log_info([
            "\n---------------------------------------------------------------\n",
            "                       DIMER INITIAL STATE\n",
            "---------------------------------------------------------------\n",
            f"Energy (initial)                        ....  {E0: .8f} Eh\n",
            f"RMS(|F|)                                ....  {rmsF0: .6f} Eh/Angstrom\n",
            f"MAX(|F|)                                ....  {maxF0: .6f} Eh/Angstrom\n",
            "\nINITIAL COORDINATES (ANGSTROEM):\n",
            self.atoms_to_xyz_block(self.atoms),
        ], self.output)

        # ------------------ ensure PRFO-style thresholds exist ------------------
        if not hasattr(self.atoms, "f_max_th"):  self.atoms.f_max_th  = p.f_max_th
        if not hasattr(self.atoms, "f_rms_th"):  self.atoms.f_rms_th  = p.f_rms_th
        if not hasattr(self.atoms, "dp_max_th"): self.atoms.dp_max_th = 1.8e-3
        if not hasattr(self.atoms, "dp_rms_th"): self.atoms.dp_rms_th = 1.2e-3

        log_info([
            "\n----------------------------------------------------------------------\n",
            "                           Dimer Iterations                           \n",
            "----------------------------------------------------------------------\n"
        ], self.output)

        # ------------------ rotation helper (HVP-based) ------------------
        def rotate_minimize_kappa(n_vec: np.ndarray):
            """
            Up to rot_max_iter inner steps to reduce kappa, using autograd Hn.
            Returns: n_new (np.ndarray), max|F_rot| (float), RMS(F_rot) (float)
            """
            n_curr = n_vec.copy()
            max_frot = np.inf
            rms_frot = np.inf

            for _ in range(p.rot_max_iter):
                # Hn from autograd; discard forces/energy here
                Hn_t, _, _ = self._get_hvp(n_curr)
                dev, dty = Hn_t.device, Hn_t.dtype
                n_th = torch.tensor(n_curr, device=dev, dtype=dty)

                # rotational force: (I - n n^T) Hn
                Hn_par = torch.dot(Hn_t, n_th) * n_th
                F_rot_t = Hn_t - Hn_par

                # metrics
                max_frot = float(torch.max(torch.abs(F_rot_t)).detach().cpu().item())
                rms_frot = float(torch.sqrt(torch.mean(F_rot_t * F_rot_t)).detach().cpu().item())

                # convergence of rotation
                if (max_frot <= p.rot_f_max_th) and (rms_frot <= p.rot_f_rms_th):
                    break

                # gradient descent on kappa in orientation space, then re-normalize (and rigid-body remove if requested)
                n_next = (n_th - p.rot_alpha * F_rot_t).detach().cpu().numpy()
                if p.remove_rigid:
                    n_next = _remove_rigid_body_components(n_next, self.atoms, M=self.M)
                n_next = _normalize(n_next, M=self.M)
                n_curr = n_next

            return n_curr, max_frot, rms_frot

        # ======================= main iteration loop =======================
        converged = False
        it = 0
        for it in range(1, p.max_iter + 1):

            # (1) rotation step using autograd HVP
            n, max_frot, rms_frot = rotate_minimize_kappa(n)

            # (2) translation-side evaluation in one pass
            Hn_t, forces_t, energy_t = self._get_hvp(n)
            forces_np = forces_t.detach().cpu().numpy()
            E = float(energy_t.detach().cpu().item())

            # curvature kappa = n^T H n
            dev, dty = Hn_t.device, Hn_t.dtype
            n_th = torch.tensor(n, device=dev, dtype=dty)
            kappa = float(torch.dot(n_th, Hn_t).detach().cpu().item())

            # project forces parallel / perpendicular to n (torch tensors)
            Fpar_t  = torch.dot(forces_t, n_th) * n_th
            Fperp_t = forces_t - Fpar_t
            mode_flip = (kappa < p.kappa_to_flip)
            Ftrans_t = (Fperp_t - Fpar_t) if mode_flip else Fperp_t

            # (3) trust-region step (convert only the step to numpy)
            step_vec = (alpha * Ftrans_t).detach().cpu().numpy()
            step_norm_inf = float(np.max(np.abs(step_vec)))
            on_boundary = False
            max_allow = min(p.trust_radius, p.step_max)
            if step_norm_inf > max_allow:
                step_vec *= (max_allow / (step_norm_inf + 1e-20))
                on_boundary = True

            new_positions = self.atoms.get_positions() + step_vec.reshape(-1, 3)
            trial = self.atoms.copy()
            trial.set_positions(new_positions)
            new_positions = trial.positions
            step_vec = (new_positions - self.atoms.positions).reshape(-1)
            alpha = (max(0.5 * alpha, 0.1 * p.step0) if on_boundary
                    else min(1.2 * alpha, p.step_max))

            # (4) PRFO-style metrics
            max_dp = float(np.max(np.linalg.norm(step_vec.reshape(-1, 3), axis=1)))
            rms_dp = float(np.sqrt(np.mean(np.linalg.norm(step_vec.reshape(-1, 3), axis=1) ** 2)))
            max_f  = float(np.max(np.linalg.norm(forces_np.reshape(-1, 3), axis=1)))
            rms_f  = float(np.sqrt(np.mean(np.linalg.norm(forces_np.reshape(-1, 3), axis=1) ** 2)))

            self.atoms.max_f  = max_f
            self.atoms.rms_f  = rms_f
            self.atoms.max_dp = max_dp
            self.atoms.rms_dp = rms_dp

            # (5) report one iteration block
            coords_block = self.atoms_to_xyz_block(self.atoms)
            info = [
                "\n----------------------------------------------------------------------\n",
                f"                             Iteration: {it:<3d}                              \n\n",
                "                             Coordinates                               \n",
                "----------------------------------------------------------------------\n",
                coords_block,
                "\n",
                f"Energy:                  {E: .6f} Convergence criteria  Is converged \n",
                f"Maximum Force:         {max_f:>12.6f} {self.atoms.f_max_th:>12.6f}                {'Yes' if max_f <= self.atoms.f_max_th else 'No'}\n",
                f"RMS Force:             {rms_f:>12.6f} {self.atoms.f_rms_th:>12.6f}                {'Yes' if rms_f <= self.atoms.f_rms_th else 'No'}\n",
                f"Maximum Displacement:  {max_dp:>12.6f} {self.atoms.dp_max_th:>12.6f}                {'Yes' if max_dp <= self.atoms.dp_max_th else 'No'}\n",
                f"RMS Displacement:      {rms_dp:>12.6f} {self.atoms.dp_rms_th:>12.6f}                {'Yes' if rms_dp <= self.atoms.dp_rms_th else 'No'}\n",
                f"\nTrust radius (MW): {max_allow: .6f}  Step norm (MW): {step_norm_inf: .6f}  On boundary: {on_boundary}\n"
                f"Curvature (kappa):      {kappa:>12.6f} Eh/Å²\n",
                f"Max Rotational Force:   {max_frot:>12.6f} Eh/Angstrom\n",
                f"RMS Rotational Force:   {rms_frot:>12.6f} Eh/Angstrom\n",
                f"Mode Flip:              {'Yes' if mode_flip else 'No'}\n\n",
            ]
            log_info(info, self.output)

            if p.save_traj:
                # Your local helper supports single Atoms; keeping your current call signature
                write_all_images_xyz(traj_file, self.atoms, energy=E, iteration=it,
                                     forces=forces_np.reshape(-1, 3))

            self.atoms.set_positions(new_positions)

            # (6) convergence: all 4 PRFO-style criteria
            if (max_f <= self.atoms.f_max_th and
                rms_f <= self.atoms.f_rms_th and
                max_dp <= self.atoms.dp_max_th and
                rms_dp <= self.atoms.dp_rms_th):
                converged = True
                log_info([f"\nDimer optimization converged at iteration {it}.\n"], self.output)
                break

        # ------------------ final write & brief summary ------------------
        _, F_final_t, E_final_t = self._get_hvp(n)
        E_final = float(E_final_t.detach().cpu().item())
        F_final = to_numpy_f64(F_final_t).reshape(-1, 3)
        self.results = {'atoms': self.atoms, 'energy': E_final, 'forces': F_final,
                        'iterations': it, 'converged': converged}
        write_xyz(ts_file, [self.atoms], energies=[E_final], forces=[F_final])
        if p.save_traj:
            write_all_images_xyz(traj_file, self.atoms, energy=E_final,
                                 iteration=it + 1, forces=F_final)

        log_info([
            "\n---------------------------------------------------------------\n",
            "                 INFORMATION ABOUT SADDLE POINT                \n",
            "---------------------------------------------------------------\n",
            f"Energy (final)                           ....  {E_final: .8f} Eh\n",
            "\n-----------------------------------------\n",
            "  SADDLE GUESS (ANGSTROEM)\n",
            "-----------------------------------------\n",
            self.atoms_to_xyz_block(self.atoms),
            f"\nWrote Dimer trajectory to: {traj_file}\n",
            f"Wrote TS guess to:         {ts_file}\n"
        ], self.output)


        return self.atoms
