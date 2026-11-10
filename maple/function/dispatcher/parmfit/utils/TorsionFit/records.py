"""Usage: define torsion fitting data models and result records."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..readparm import RefinementParameterSet, Dihedral, FourierTerm


@dataclass(frozen=True)
class TorsionScanRuntime:
    max_iter: int
    memory: int
    curvature: float
    max_step: float
    mode: str = "relaxed"
    backend: str = "lbfgs"   #lbfgs/cgws/cgbs
    constraint_mode: str = "fixinternals"  #fixinternals/projected 
    batch_size: int | None = None

    @property
    def method(self) -> str:
        return self.backend

    def to_scan_params(self) -> dict[str, object]:
        return {
            "batch_size": self.batch_size,
            "mode": self.mode,
            "backend": self.backend,
            "constraint_mode": self.constraint_mode,
            "opt": {
                "max_iter": int(self.max_iter),
                "memory": int(self.memory),
                "curvature": float(self.curvature),
                "max_step": float(self.max_step),
                "write_traj": False,
                "verbose": 0,
            },
        }


@dataclass(frozen=True)
class TorsionScanData:
    angles_deg: np.ndarray
    qm_hartree: np.ndarray
    qm_kcal: np.ndarray
    frames: list[Any]
    source_path: str
    ref_idx: int
    qm_rel: np.ndarray


@dataclass(frozen=True)
class TorsionSharedGroupSpec:
    label: str
    atom_types: tuple[str, str, str, str]
    improper: bool
    dihedral_indices: tuple[int, ...]
    instances: tuple[tuple[int, int, int, int], ...]
    slot_indices: tuple[int, ...]
    slot_periods: tuple[int, ...]
    slot_phases: tuple[float, ...]
    existing_slot_mask: tuple[bool, ...]
    slot_sources: tuple[str, ...] = ()
    slot_coherences: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        slot_count = len(self.slot_indices)
        if not self.slot_sources:
            object.__setattr__(
                self,
                "slot_sources",
                tuple("existing" if existing else "candidate" for existing in self.existing_slot_mask),
            )
        if not self.slot_coherences:
            object.__setattr__(self, "slot_coherences", tuple(1.0 for _ in range(slot_count)))
        if len(self.slot_sources) != slot_count or len(self.slot_coherences) != slot_count:
            raise ValueError("TorsionSharedGroupSpec slot metadata must match slot_indices.")


@dataclass(frozen=True)
class TorsionLocalProblem:
    torsion_bond: tuple[int, int]
    scan_data: TorsionScanData
    target_instances: list[Dihedral]
    representative_dihedral: tuple[int, int, int, int]
    basis: np.ndarray
    qm_rel: np.ndarray
    orig_mm_rel: np.ndarray
    mm_zeroed_rel: np.ndarray
    residual: np.ndarray
    k_orig: np.ndarray
    scales: np.ndarray
    active_mask: np.ndarray
    prior_weights: np.ndarray
    shared_groups: tuple[TorsionSharedGroupSpec, ...] = ()
    phase_orig: np.ndarray | None = None
    cos_basis: np.ndarray | None = None
    sin_basis: np.ndarray | None = None

    @property
    def mm_base_rel(self) -> np.ndarray:
        return self.mm_zeroed_rel

    @property
    def fit_target_rel(self) -> np.ndarray:
        return self.residual


@dataclass(frozen=True)
class TorsionObjectiveEvaluation:
    total_loss: float
    data_loss: float
    prior_loss: float
    global_rmse: float
    per_scan_rmse: dict[tuple[int, int], float]
    per_scan_data_loss: dict[tuple[int, int], float] = field(default_factory=dict)
    objective_kind: str = "direct_k_phase"
    scan_data_loss: float = 0.0
    ensemble_data_loss: float = 0.0


@dataclass(frozen=True)
class TorsionObjectiveTarget:
    label: str
    torsion_bond: tuple[int, int]
    qm_rel: np.ndarray
    constant_rel: np.ndarray
    cos_basis: np.ndarray
    sin_basis: np.ndarray
    weight: float
    source_path: str


@dataclass(frozen=True)
class TorsionGlobalProblem:
    stage0_paramset: RefinementParameterSet
    torsion_bonds: tuple[tuple[int, int], ...]
    scan_map: dict[tuple[int, int], TorsionScanData]
    term_paths: tuple[tuple[int, int], ...]
    block_slices: dict[tuple[int, int], tuple[int, int]]
    k_orig: np.ndarray
    phase_orig: np.ndarray
    period_orig: np.ndarray
    scales: np.ndarray
    qm_rel_map: dict[tuple[int, int], np.ndarray]
    centered_basis_map: dict[tuple[int, int], np.ndarray]
    centered_cos_basis_map: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    centered_sin_basis_map: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    constant_rel_map: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    grouped: bool = False
    reference_paramset: RefinementParameterSet | None = None
    prior_weights: np.ndarray | None = None
    shared_groups_map: dict[tuple[int, int], tuple[TorsionSharedGroupSpec, ...]] = field(default_factory=dict)
    prior_weight: float = 1.0
    global_max_iter: int = 50
    extra_targets: tuple[TorsionObjectiveTarget, ...] = ()


@dataclass(frozen=True)
class TorsionEnsembleResult:
    frames_by_center: dict[tuple[int, int], tuple[Any, ...]] = field(default_factory=dict)
    energies_by_center: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    xyz_paths: dict[tuple[int, int], str] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()


@dataclass
class TorsionSharedGroupReport:
    label: str
    atom_types: tuple[str, str, str, str]
    improper: bool
    instances: list[tuple[int, int, int, int]]
    original_terms: list[FourierTerm]
    fitted_terms: list[FourierTerm]
    active_slots: tuple[str, ...] = ()
    frozen_non_template_slots: tuple[str, ...] = ()
    effective_rank: int = 0
    dropped_singular_directions: int = 0
    diagnostic_flags: tuple[str, ...] = ()


@dataclass
class TorsionFitTerms:
    original_terms: list[list[FourierTerm]]
    fitted_terms: list[list[FourierTerm]]
    delta_kphi: np.ndarray
    shared_groups: list[TorsionSharedGroupReport] = field(default_factory=list)


@dataclass
class TorsionFitCurves:
    angles_deg: np.ndarray
    qm_rel: np.ndarray | None = None
    mm_orig_rel: np.ndarray | None = None
    mm_stage0_rel: np.ndarray | None = None
    mm_stage1_rel: np.ndarray | None = None
    mm_stage2_rel: np.ndarray | None = None
    mm_zeroed_rel: np.ndarray | None = None


@dataclass
class TorsionFitMetrics:
    residual_before: np.ndarray
    residual_after: np.ndarray | None
    rmse: float
    mae: float
    max_abs_error: float


@dataclass
class TorsionFitAbsoluteTable:
    qm_kcal: np.ndarray
    orig_mm_total: np.ndarray
    torsion_fit: np.ndarray
    mm_refit_total: np.ndarray
    offset_k: float
    residual_after_abs: np.ndarray


def _clone_fourier_blocks(terms: list[list[FourierTerm]]) -> list[list[FourierTerm]]:
    return [
        [FourierTerm(term.kPhi, term.period, term.phase) for term in term_block]
        for term_block in terms
    ]


def _copy_array(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float).copy()


def _copy_optional_array(values: np.ndarray | None) -> np.ndarray | None:
    return None if values is None else _copy_array(values)


def _copy_curves(curves: TorsionFitCurves, *, mm_stage2_rel: np.ndarray | None = None) -> TorsionFitCurves:
    return TorsionFitCurves(
        angles_deg=_copy_array(curves.angles_deg),
        qm_rel=_copy_optional_array(curves.qm_rel),
        mm_orig_rel=_copy_optional_array(curves.mm_orig_rel),
        mm_stage0_rel=_copy_optional_array(curves.mm_stage0_rel),
        mm_stage1_rel=_copy_optional_array(curves.mm_stage1_rel),
        mm_stage2_rel=_copy_optional_array(curves.mm_stage2_rel if mm_stage2_rel is None else mm_stage2_rel),
        mm_zeroed_rel=_copy_optional_array(curves.mm_zeroed_rel),
    )


def _aligned_delta_kphi(
    original_terms: list[list[FourierTerm]],
    fitted_terms: list[list[FourierTerm]],
) -> np.ndarray:
    delta_values: list[float] = []
    for original_block, fitted_block in zip(original_terms, fitted_terms):
        original_by_slot = {
            (float(term.period), float(term.phase)): float(term.kPhi)
            for term in original_block
        }
        for fitted_term in fitted_block:
            slot = (float(fitted_term.period), float(fitted_term.phase))
            baseline = original_by_slot.get(slot, 0.0)
            delta_values.append(float(fitted_term.kPhi) - baseline)
    return np.asarray(delta_values, dtype=float)


@dataclass
class TorsionFitReport:
    torsion_bond: tuple[int, int]
    target_instances: list[Dihedral]
    representative_dihedral: tuple[int, int, int, int]
    scan_source_path: str
    terms: TorsionFitTerms
    curves: TorsionFitCurves
    metrics: TorsionFitMetrics
    absolute: TorsionFitAbsoluteTable | None = None

    def with_stage2_result(
        self,
        fitted_terms: list[list[FourierTerm]],
        mm_stage2_rel: np.ndarray,
    ) -> "TorsionFitReport":
        fitted_blocks = _clone_fourier_blocks(fitted_terms)
        stage2_rel = np.asarray(mm_stage2_rel, dtype=float).copy()
        qm_rel = np.asarray(self.curves.qm_rel, dtype=float)
        residual_after = qm_rel - stage2_rel
        rmse = float(np.sqrt(np.mean(residual_after**2))) if residual_after.size else 0.0
        curves = _copy_curves(self.curves, mm_stage2_rel=stage2_rel)
        terms = TorsionFitTerms(
            original_terms=_clone_fourier_blocks(self.terms.original_terms),
            fitted_terms=fitted_blocks,
            delta_kphi=_aligned_delta_kphi(self.terms.original_terms, fitted_blocks),
            shared_groups=deepcopy(self.terms.shared_groups),
        )
        metrics = TorsionFitMetrics(
            residual_before=np.asarray(self.metrics.residual_before, dtype=float).copy(),
            residual_after=residual_after.copy(),
            rmse=rmse,
            mae=float(np.mean(np.abs(residual_after))) if residual_after.size else 0.0,
            max_abs_error=float(np.max(np.abs(residual_after))) if residual_after.size else 0.0,
        )
        return TorsionFitReport(
            torsion_bond=self.torsion_bond,
            target_instances=deepcopy(self.target_instances),
            representative_dihedral=self.representative_dihedral,
            scan_source_path=self.scan_source_path,
            terms=terms,
            curves=curves,
            metrics=metrics,
            absolute=deepcopy(self.absolute),
        )


@dataclass
class TorsionRefineCycle:
    cycle: int
    total_loss_before: float
    total_loss_after: float
    global_rmse_before: float
    global_rmse_after: float
    accepted_blocks: int
    rejected_blocks: int
    per_scan_rmse_before: dict[tuple[int, int], float]
    per_scan_rmse_after: dict[tuple[int, int], float]
    data_loss_before: float = 0.0
    data_loss_after: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class TorsionWorkflowResult:
    stage1_paramset: RefinementParameterSet | None
    final_paramset: RefinementParameterSet
    refine_cycles: list[TorsionRefineCycle] = field(default_factory=list)
    scan_xyz: dict[tuple[int, int], str] = field(default_factory=dict)
    torsion_bonds: list[tuple[int, int]] = field(default_factory=list)
    fit_reports: list[TorsionFitReport] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stage1_diagnostics: dict[str, Any] = field(default_factory=dict)
    stage2_diagnostics: dict[str, Any] = field(default_factory=dict)
    ensemble_xyz: dict[tuple[int, int], str] = field(default_factory=dict)
