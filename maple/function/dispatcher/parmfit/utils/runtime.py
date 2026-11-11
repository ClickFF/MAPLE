"""Usage: manage parmfit workdirs, geometry optimization, and Hessian helpers."""

from __future__ import annotations

import os
from copy import deepcopy

import numpy as np
from ase import Atoms

from .Scan.optimizer import LBFGS, LBFGSParams


MEDIUM_THRESHOLDS = {
    "f_max_th": 0.00285,
    "f_rms_th": 0.00190,
    "dp_max_th": 0.00315,
    "dp_rms_th": 0.00210,
}



def parmfit_workdir(output: str, workflow: str) -> str:
    path = os.path.join(parmfit_output_dir(output), workflow)
    os.makedirs(path, exist_ok=True)
    return path



def parmfit_output_dir(output: str) -> str:
    base = os.path.splitext(os.path.abspath(output))[0]
    path = f"{base}_work"
    os.makedirs(path, exist_ok=True)
    return path



def parmfit_work_prefix(output: str, workflow: str) -> str:
    base_name = os.path.splitext(os.path.basename(output))[0]
    return os.path.join(parmfit_workdir(output, workflow), base_name)



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



def get_forces(atoms: Atoms) -> np.ndarray:
    if hasattr(atoms, "get_forces"):
        try:
            return np.asarray(atoms.get_forces(), dtype=float)
        except AttributeError:
            pass
    if getattr(atoms, "calc", None) is not None and hasattr(atoms.calc, "get_forces"):
        return np.asarray(atoms.calc.get_forces(atoms), dtype=float)
    raise ValueError("Silent runtime requires force evaluation from atoms or atoms.calc.")



def get_potential_energy(atoms: Atoms, force_consistent: bool = True) -> float:
    if hasattr(atoms, "get_potential_energy"):
        try:
            return float(atoms.get_potential_energy(force_consistent=force_consistent))
        except AttributeError:
            pass
    if getattr(atoms, "calc", None) is not None and hasattr(atoms.calc, "get_potential_energy"):
        return float(atoms.calc.get_potential_energy(atoms, force_consistent=force_consistent))
    raise ValueError("Silent runtime requires energy evaluation from atoms or atoms.calc.")



def get_cartesian_hessian(atoms: Atoms) -> np.ndarray:
    hessian = to_f64(atoms.calc.get_hessian(atoms))
    if hessian.ndim == 3 and hessian.shape[0] == 1:
        hessian = hessian[0]
    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
        raise ValueError(f"Hessian must be square 2D, got shape {hessian.shape}")
    return hessian



def copy_thresholds(source_atoms, target_atoms) -> None:
    for attr, default in MEDIUM_THRESHOLDS.items():
        setattr(target_atoms, attr, float(getattr(source_atoms, attr, default)))



def silent_lbfgs_params(params: LBFGSParams | None = None) -> LBFGSParams:
    resolved = deepcopy(params) if params is not None else LBFGSParams()
    resolved.write_traj = False
    resolved.verbose = 0
    resolved.use_projection = False
    resolved.use_line_search = False
    return resolved



SilentLBFGS = LBFGS



def _require_lbfgs_thresholds(atoms: Atoms) -> None:
    missing = [
        attr
        for attr in ("f_max_th", "f_rms_th", "dp_max_th", "dp_rms_th")
        if not hasattr(atoms, attr)
    ]
    if missing:
        raise ValueError(f"LBFGS thresholds are missing on atoms: {', '.join(missing)}.")



def run_silent_lbfgs(
    atoms: Atoms,
    *,
    output: str,
    params: LBFGSParams | None = None,
) -> LBFGS:
    _require_lbfgs_thresholds(atoms)
    optimizer = LBFGS(atoms=atoms, output=output, params=silent_lbfgs_params(params))
    optimizer.run()
    return optimizer


def run_silent_lbfgs_batch(atoms_list, *, outputs, params=None, batch_size=None):
    from .Scan.batch_optimizer import BatchLBFGS

    if len(atoms_list) != len(outputs):
        raise ValueError("atoms_list and outputs must have the same length")
    optimizers = []
    for atoms, output in zip(atoms_list, outputs):
        _require_lbfgs_thresholds(atoms)
        optimizers.append(LBFGS(atoms, output=output, params=silent_lbfgs_params(params)))
    optimizer = BatchLBFGS(optimizers, batch_size=batch_size)
    optimizer.run()
    return optimizer



def optimize_atoms_geometry(
    atoms: Atoms,
    *,
    output: str,
    max_iter: int = 256,
    max_step: float = 0.2,
    failure_message: str | None = None,
) -> Atoms:
    params = silent_lbfgs_params()
    params.max_iter = int(max_iter)
    params.max_step = float(max_step)
    optimizer = run_silent_lbfgs(atoms, output=output, params=params)
    if not optimizer.converged:
        raise RuntimeError(failure_message or "Geometry optimization did not converge.")
    return atoms



def optimize_model_geometry(
    model: dict,
    *,
    output: str,
    source_atoms: Atoms,
    calculator=None,
    max_iter: int = 256,
    max_step: float = 0.2,
    failure_message: str | None = None,
) -> dict:
    from .model import model_to_atoms, update_model_from_atoms

    atoms = model_to_atoms(model, charge=model.get("charge"), mult=model.get("mult"))
    copy_thresholds(source_atoms, atoms)
    atoms.calc = source_atoms.calc if calculator is None else calculator
    optimize_atoms_geometry(
        atoms,
        output=output,
        max_iter=max_iter,
        max_step=max_step,
        failure_message=failure_message or f"Geometry optimization did not converge for {model.get('name', 'model')}.",
    )
    return update_model_from_atoms(model, atoms)

