"""Usage: parse user-facing refinement parameters."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

from ....jobABC import JobABC
from ...runconfig import as_tracked, read_batch_size
from ..chgfit import ChargeFitConfig, build_charge_fit_config
from ..QMInterface import QMReferenceConfig, build_qm_reference_config
from ..Scan.optimizer import LBFGSParams
from ..TorsionFit import TorsionFitParams, build_torsion_fit_params


SUPPORTED_BONDED_METHODS = ("mseminario", "seminario", "none")

@dataclass
class RefinementConfig:
    mol2: str = ""
    bonded: str = "mseminario"
    vib_scale: float = 1.0
    qm: QMReferenceConfig = field(default_factory=QMReferenceConfig)
    charge_fit: ChargeFitConfig = field(default_factory=ChargeFitConfig)
    torsion: TorsionFitParams = field(default_factory=TorsionFitParams)
    lbfgs: LBFGSParams = field(default_factory=LBFGSParams)
    scan_opt: LBFGSParams = field(default_factory=LBFGSParams)
    batch_size: int | None = None


def build_refinement_config(raw_params: Optional[dict]) -> RefinementConfig:
    params = raw_params if isinstance(raw_params, dict) else {}
    nested = next(
        (params[alias] for alias in ("parmfit", "refinement") if alias in params and isinstance(params[alias], dict)),
        None,
    )
    raw = as_tracked(nested if nested is not None else params)
    raw.set_group("Refinement run", "Site composition, charge source and general refinement settings")
    config = RefinementConfig()
    config.batch_size = read_batch_size(raw)
    config.mol2 = raw.get("mol2", config.mol2)
    config.bonded = str(raw.get("bonded", config.bonded)).strip().lower()
    if config.bonded not in SUPPORTED_BONDED_METHODS:
        raise ValueError(
            f"Unsupported bonded method {config.bonded!r}; expected one of {', '.join(SUPPORTED_BONDED_METHODS)}."
        )
    config.vib_scale = float(raw.get("vib_scale", config.vib_scale))

    config.qm = build_qm_reference_config(raw)
    raw.set_group("charge fitting", "Charge fitting method, level and RESP backend")
    config.charge_fit = build_charge_fit_config(
        raw,
        default_method="none",
        default_level="HF/6-31G(d)",
        default_nproc=config.qm.qm_nproc,
        default_mem=config.qm.qm_mem,
    )
    config.torsion = build_torsion_fit_params(raw)

    raw.set_group("LBFGS optimizer", "Geometry optimizer controls")
    lbfgs_nested = next(
        (params[alias] for alias in ("lbfgs", "LBFGS", "opt") if alias in params and isinstance(params[alias], dict)),
        None,
    )
    if lbfgs_nested is not None:
        lbfgs = JobABC._update_dataclass_from_dict(LBFGSParams(), lbfgs_nested)
    else:
        lbfgs = LBFGSParams()
        lbfgs.memory = raw.get("memory", lbfgs.memory)
        lbfgs.curvature = raw.get("curvature", lbfgs.curvature)
        lbfgs.max_step = raw.get("max_step", lbfgs.max_step)
        lbfgs.max_iter = raw.get("max_iter", lbfgs.max_iter)
        lbfgs.write_traj = raw.get("write_traj", lbfgs.write_traj)
        lbfgs.traj_every = raw.get("traj_every", lbfgs.traj_every)
        lbfgs.verbose = raw.get("verbose", lbfgs.verbose)
        lbfgs.use_projection = raw.get("use_projection", lbfgs.use_projection)
        lbfgs.use_line_search = raw.get("use_line_search", lbfgs.use_line_search)
        lbfgs.bond_constraints = raw.get("bond_constraints", lbfgs.bond_constraints)
        lbfgs.angle_constraints = raw.get("angle_constraints", lbfgs.angle_constraints)
        lbfgs.torsion_constraints = raw.get("torsion_constraints", lbfgs.torsion_constraints)
    config.lbfgs = lbfgs
    config.scan_opt = deepcopy(lbfgs)
    config.scan_opt.write_traj = False
    config.scan_opt.verbose = 0
    return config
