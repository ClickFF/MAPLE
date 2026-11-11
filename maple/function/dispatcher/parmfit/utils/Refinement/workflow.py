"""Usage: run the parmfit refinement workflow."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from time import perf_counter
from typing import Callable

from ase import Atoms

from ..Seminario import apply_seminario
from ..mSeminario import apply_mseminario
from ..chgfit import apply_atomic_charges, fit_molecule_charges
from ..QMInterface import build_qm_reference_runner
from ..mlip_tools import release_charge_calculator_cache
from ..readparm import RefinementParameterSet, Improper
from ..runtime import get_cartesian_hessian, parmfit_work_prefix, run_silent_lbfgs
from ..Scan.optimizer import LBFGS
from ..TorsionFit import TorsionScanRuntime, TorsionWorkflowResult, run_torsion_workflow
from .artifacts import RefinementWorkflowResult, export_amber, export_gromacs
from .config import RefinementConfig
from .parameters import build_init_parmset
from .report import refinement_result_lines, has_parameter_changes, parameter_change_lines, stage_lines, summary_lines


@contextmanager
def _timed_stage(name: str, timings: list[tuple[str, float]]):
    start = perf_counter()
    try:
        yield
    finally:
        timings.append((name, perf_counter() - start))


def run_geometry_optimization(atoms: Atoms, output: str, config: RefinementConfig) -> LBFGS:
    return run_silent_lbfgs(atoms, output=output, params=config.lbfgs)


def _bonded_refinement_label(method: str) -> str:
    return "Seminario" if method == "seminario" else "mSeminario"


def run_bonded_refinement(
    atoms: Atoms,
    parmset: RefinementParameterSet,
    config: RefinementConfig,
    *,
    output: str,
    qm_runner=None,
) -> RefinementParameterSet:
    stage0_parmset = deepcopy(parmset)
    if qm_runner is not None:
        freq = qm_runner.opt_frequency(atoms, f"{parmfit_work_prefix(output, 'qm')}_ref")
        if freq.hessian is None:
            raise RuntimeError("QM opt-frequency job did not provide a Cartesian Hessian.")
        atoms.set_positions(freq.atoms.get_positions())
        hessian = freq.hessian
    else:
        hessian = get_cartesian_hessian(atoms)
    apply_bonded = apply_seminario if config.bonded == "seminario" else apply_mseminario
    apply_bonded(
        atoms,
        hessian,
        stage0_parmset.bonds,
        stage0_parmset.angles,
        config.vib_scale,
    )
    return stage0_parmset


def build_torsion_runtime(config: RefinementConfig) -> TorsionScanRuntime:
    return TorsionScanRuntime(
        batch_size=config.batch_size,
        max_iter=int(config.scan_opt.max_iter),
        memory=int(config.scan_opt.memory),
        curvature=float(config.scan_opt.curvature),
        max_step=float(config.scan_opt.max_step),
        backend=config.torsion.backend,
        constraint_mode=config.torsion.constraint_mode,
    )


def _needs_refine(config: RefinementConfig) -> bool:
    return config.bonded != "none" or bool(config.torsion.enabled)


def run_refinement_workflow(
    *,
    output: str,
    atoms: Atoms,
    config: RefinementConfig,
    log_info: Callable[[list[str]], None],
    torsion_workflow_fn=run_torsion_workflow,
) -> RefinementWorkflowResult:
    stage_timings: list[tuple[str, float]] = []
    log_info(stage_lines("\n[Refinement] initial parameter assignment ..."))
    with _timed_stage("initial parameter assignment", stage_timings):
        init_parmset, init_frcmod_path = build_init_parmset(output, atoms, config)
    log_info(summary_lines(config, init_parmset, init_frcmod_path))

    torsion_enabled = bool(config.torsion.enabled)
    refine_enabled = _needs_refine(config)
    qm_runner = build_qm_reference_runner(config.qm)
    qm_compare_enabled = bool(qm_runner is not None and getattr(config.qm, "qm_compare", False) and refine_enabled)
    route_name = "qm_route" if qm_runner is not None else "mlip_route"
    mlip_atoms = None

    if refine_enabled:
        log_info(stage_lines("\n[Refinement] MLIP geometry optimization ..."))
        with _timed_stage("geometry optimization", stage_timings):
            optimizer = run_geometry_optimization(atoms, output, config)
        if not optimizer.converged:
            raise RuntimeError(f"LBFGS did not converge within {optimizer.params.max_iter} iterations.")
        mlip_atoms = atoms.copy()
        mlip_atoms.calc = atoms.calc
        if qm_runner is not None and config.bonded == "none" and torsion_enabled:
            log_info(stage_lines("[Refinement] QM reference optimization ..."))
            with _timed_stage("QM reference optimization", stage_timings):
                qm_opt = qm_runner.optimize(atoms, f"{parmfit_work_prefix(output, 'qm')}_ref")
            atoms.set_positions(qm_opt.atoms.get_positions())

    mlip_stage0_parmset = None
    if config.bonded == "none":
        stage0_parmset = deepcopy(init_parmset)
        if qm_compare_enabled:
            mlip_stage0_parmset = deepcopy(init_parmset)
        log_info(stage_lines("[Refinement] bond/angle refinement skipped ..."))
    else:
        bonded_label = _bonded_refinement_label(config.bonded)
        if qm_compare_enabled:
            log_info(stage_lines(f"[Refinement] MLIP Hessian + {bonded_label} (comparison) ..."))
            with _timed_stage(f"MLIP Hessian + {bonded_label}", stage_timings):
                mlip_ref_atoms = (mlip_atoms or atoms).copy()
                mlip_ref_atoms.calc = (mlip_atoms or atoms).calc
                mlip_stage0_parmset = run_bonded_refinement(
                    mlip_ref_atoms,
                    init_parmset,
                    config,
                    output=output,
                    qm_runner=None,
                )
            log_info(
                parameter_change_lines(
                    f"{bonded_label} bond/angle changes (MLIP)",
                    init_parmset,
                    mlip_stage0_parmset,
                    sections=("bonds", "angles"),
                )
            )
        if qm_runner is not None:
            log_info(stage_lines("[Refinement] QM reference optimization ..."))
            log_info(stage_lines(f"[Refinement] QM Hessian + {bonded_label} ..."))
        else:
            log_info(stage_lines(f"[Refinement] MLIP Hessian + {bonded_label} ..."))
        with _timed_stage(f"Hessian + {bonded_label}", stage_timings):
            stage0_parmset = run_bonded_refinement(
                atoms,
                init_parmset,
                config,
                output=output,
                qm_runner=qm_runner,
            )
        log_info(
            parameter_change_lines(
                f"{bonded_label} bond/angle changes" + (" (QM)" if qm_compare_enabled else ""),
                init_parmset,
                stage0_parmset,
                sections=("bonds", "angles"),
            )
        )

    calculator_cache: dict = {}
    charge_fit_configured = config.charge_fit.method != "none"
    try:
        if charge_fit_configured:
            log_info(stage_lines("\n[Refinement] atomic charge fitting ..."))
        with _timed_stage("charge fitting", stage_timings):
            charge_result = fit_molecule_charges(
                output=output,
                atoms=atoms,
                source_mol2=config.mol2,
                config=config.charge_fit,
                route=route_name,
                calculator_cache=calculator_cache,
            )
        charge_timing = stage_timings[-1][1]
        route_baseline_parmset = apply_atomic_charges(
            init_parmset,
            charge_result.charges,
        )
        stage0_parmset = apply_atomic_charges(
            stage0_parmset,
            charge_result.charges,
        )

        mlip_charge_result = None
        mlip_charge_timing = None
        mlip_baseline_parmset = None
        if qm_compare_enabled:
            mlip_charge_atoms = mlip_atoms or atoms
            if charge_fit_configured:
                log_info(stage_lines("[Refinement] atomic charge fitting (MLIP comparison) ..."))
            with _timed_stage("charge fitting (MLIP comparison)", stage_timings):
                mlip_charge_result = fit_molecule_charges(
                    output=output,
                    atoms=mlip_charge_atoms,
                    source_mol2=config.mol2,
                    config=config.charge_fit,
                    route="mlip_route",
                    calculator_cache=calculator_cache,
                )
            mlip_charge_timing = stage_timings[-1][1]
            mlip_baseline_parmset = apply_atomic_charges(
                init_parmset,
                mlip_charge_result.charges,
            )
            mlip_stage0_parmset = apply_atomic_charges(
                mlip_stage0_parmset or deepcopy(init_parmset),
                mlip_charge_result.charges,
            )
    finally:
        release_charge_calculator_cache(
            calculator_cache,
            active_calculators=(
                getattr(atoms, "calc", None),
                getattr(mlip_atoms, "calc", None) if mlip_atoms is not None else None,
            ),
        )

    improper_targets = [improper for improper in stage0_parmset.impropers if improper.refit]
    improper_lines = [f"  improper refit: {imp.atom_types} {imp.atoms}" for imp in improper_targets]
    for center in config.torsion.radical_center or ():
        neighbors = stage0_parmset.mol2.adjacency.get(center, set())
        if len(neighbors) != 3:
            improper_lines.append(f"  radical center {center} ignored (coordination != 3)")
            continue
        matched = [improper for improper in stage0_parmset.impropers if improper.atoms[2] == center]
        if matched:
            matched[0].refit = True
            if matched[0] not in improper_targets:
                improper_targets.append(matched[0])
                improper_lines.append(f"  improper refit (radical): {matched[0].atom_types} {matched[0].atoms}")
            continue
        trio = sorted(neighbors)
        quartet = (trio[0], trio[1], center, trio[2])
        types = tuple(stage0_parmset.mol2.atoms[atom - 1].atom_type for atom in quartet)
        improper_targets.append(Improper(atoms=quartet, atom_types=types, terms=[], refit=True))
        improper_lines.append(f"  improper refit (radical): {types} {quartet}")
    if int(atoms.info.get("mult", 1)) > 1 and not config.torsion.radical_center:
        improper_lines.append(
            "  open-shell system detected; radical_center is not set, radical-specific improper refinement is not enabled."
        )
    if improper_lines:
        log_info(stage_lines("\n[Refinement] TorsionFit improper target selection ..."))
        selection_lines = [f"improper refit targets: {len(improper_targets)}\n"]
        if not torsion_enabled:
            selection_lines.append("  -> skipped because torsionfit=False (parmchk2 estimates kept)\n")
        log_info(selection_lines + improper_lines)

    mlip_torsion = None
    mlip_final_parmset = None
    if torsion_enabled:
        if qm_compare_enabled:
            log_info(stage_lines("\n[Refinement] TorsionFit (MLIP comparison) ..."))
            with _timed_stage("TorsionFit (MLIP comparison)", stage_timings):
                mlip_torsion_atoms = (mlip_atoms or atoms).copy()
                mlip_torsion_atoms.calc = (mlip_atoms or atoms).calc
                mlip_torsion = torsion_workflow_fn(
                    atoms=mlip_torsion_atoms,
                    output=output,
                    paramset=mlip_stage0_parmset or deepcopy(init_parmset),
                    original_paramset=mlip_baseline_parmset or init_parmset,
                    params=config.torsion,
                    runtime=build_torsion_runtime(config),
                    log_info=None,
                )
            mlip_final_parmset = deepcopy(mlip_torsion.final_paramset)
            log_info(
                parameter_change_lines(
                    "TorsionFit dihedral changes (MLIP)",
                    mlip_stage0_parmset or init_parmset,
                    mlip_final_parmset,
                    sections=("dihedrals",),
                )
            )
        log_info(stage_lines("\n[Refinement] TorsionFit ..."))
        if qm_runner is not None:
            qm_mode = int(getattr(config.qm, "qm_mode", 1))
            log_info(stage_lines(f"[Refinement] Using QM reference data for TorsionFit (mode={qm_mode}) ..."))
        with _timed_stage("TorsionFit", stage_timings):
            torsion_kwargs = {
                "atoms": atoms,
                "output": output,
                "paramset": stage0_parmset,
                "original_paramset": route_baseline_parmset,
                "params": config.torsion,
                "runtime": build_torsion_runtime(config),
                "log_info": None,
            }
            if improper_targets:
                torsion_kwargs["improper_targets"] = improper_targets
                torsion_kwargs["radical_centers"] = config.torsion.radical_center or ()
            if qm_runner is not None:
                torsion_kwargs["qm_runner"] = qm_runner
            torsion = torsion_workflow_fn(**torsion_kwargs)
        final_parmset = deepcopy(torsion.final_paramset)
        log_info(
            parameter_change_lines(
                "TorsionFit torsion changes" + (" (QM)" if qm_compare_enabled else ""),
                stage0_parmset,
                final_parmset,
                sections=("dihedrals", "impropers"),
            )
        )
    else:
        torsion = TorsionWorkflowResult(
            stage1_paramset=None,
            final_paramset=stage0_parmset,
        )
        final_parmset = stage0_parmset
        if qm_compare_enabled:
            mlip_torsion = TorsionWorkflowResult(
                stage1_paramset=None,
                final_paramset=mlip_stage0_parmset or deepcopy(init_parmset),
            )
            mlip_final_parmset = mlip_torsion.final_paramset

    if has_parameter_changes(
        route_baseline_parmset,
        final_parmset,
        sections=("impropers", "nonbonds"),
    ):
        log_info(
            parameter_change_lines(
                "Other final changes",
                route_baseline_parmset,
                final_parmset,
                sections=("impropers", "nonbonds"),
            )
        )

    log_info(stage_lines("\n[Refinement] export GROMACS ..."))
    with _timed_stage("export GROMACS", stage_timings):
        gromacs = export_gromacs(output, atoms, final_parmset)

    log_info(stage_lines("[Refinement] export Amber ..."))
    export_config = deepcopy(config)
    export_config.mol2 = charge_result.work_mol2
    with _timed_stage("export Amber", stage_timings):
        amber = export_amber(
            output,
            atoms,
            export_config,
            final_parmset,
            use_refined_parameters=refine_enabled,
            source_frcmod=init_frcmod_path,
        )

    mlip_gromacs = None
    mlip_amber = None
    if qm_compare_enabled and mlip_final_parmset is not None:
        log_info(stage_lines("[Refinement] export GROMACS (MLIP comparison) ..."))
        with _timed_stage("export GROMACS (MLIP comparison)", stage_timings):
            mlip_gromacs = export_gromacs(
                output,
                mlip_atoms or atoms,
                mlip_final_parmset,
                output_suffix="_mlip",
            )
        log_info(stage_lines("[Refinement] export Amber (MLIP comparison) ..."))
        mlip_export_config = deepcopy(config)
        if mlip_charge_result is not None:
            mlip_export_config.mol2 = mlip_charge_result.work_mol2
        with _timed_stage("export Amber (MLIP comparison)", stage_timings):
            mlip_amber = export_amber(
                output,
                mlip_atoms or atoms,
                mlip_export_config,
                mlip_final_parmset,
                use_refined_parameters=refine_enabled,
                source_frcmod=init_frcmod_path,
                output_suffix="_mlip",
            )

    result = RefinementWorkflowResult(
        init_parmset=init_parmset,
        stage0_parmset=stage0_parmset,
        final_parmset=final_parmset,
        torsion=torsion,
        gromacs=gromacs,
        amber=amber,
        init_frcmod=init_frcmod_path,
        stage_timings=stage_timings,
        charge_result=charge_result,
        charge_timing=charge_timing,
        mlip_stage0_parmset=mlip_stage0_parmset,
        mlip_final_parmset=mlip_final_parmset,
        mlip_torsion=mlip_torsion,
        mlip_gromacs=mlip_gromacs,
        mlip_amber=mlip_amber,
        mlip_charge_result=mlip_charge_result,
        mlip_charge_timing=mlip_charge_timing,
    )
    log_info(refinement_result_lines(config, result))
    return result
