"""Usage: run the NCAA abinitio parameterization workflow."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import os
from time import perf_counter
from typing import Callable

from .. import interface
from ..chgfit import (
    ChargeFitResult,
    apply_model_charges,
    fit_multiconformer_charges,
)
from ..context import find_prev_next_peptide_residues, find_residue_by_key
from ..Seminario import apply_seminario
from ..mSeminario import apply_mseminario
from ..model import infer_bond_pairs, model_to_atoms, update_model_from_atoms
from ..QMInterface import build_qm_reference_runner
from ..readparm import RefinementParameterSet, Improper, build_refinement_paramset
from ..runtime import (
    copy_thresholds,
    get_cartesian_hessian,
    parmfit_work_prefix,
)
from ..structure import copy_residue
from dataclasses import replace as _dc_replace

from ..TorsionFit import TorsionScanRuntime, TorsionWorkflowResult, run_torsion_workflow
from .artifacts import NCAAArtifacts, build_ncaa_amber_artifacts, build_ncaa_export_bundle
from .config import NCAAAbinitioConfig
from .models import NCAAConformer, NCAAIdentity, build_capped_ncaa_model, build_charge_conformers, build_ncaa_torsion_bond_filter, build_ncaa_sidechain_relax_indices, identity_ncaa, optimize_capped_confs, warn_capped_proton_transfer
from .report import format_ncaa_final_lines, format_ncaa_start_lines


@contextmanager
def _timed_stage(stage_timings: list[tuple[str, float]], label: str):
    start = perf_counter()
    try:
        yield
    finally:
        stage_timings.append((label, perf_counter() - start))


@dataclass(frozen=True)
class NCAAWorkflowResult:
    identity: NCAAIdentity
    representative: NCAAConformer
    conformers: list[NCAAConformer]
    paramset: RefinementParameterSet
    torsion: TorsionWorkflowResult
    artifacts: NCAAArtifacts
    charge_result: ChargeFitResult
    representative_model: dict
    residue_model: dict
    stage_timings: list[tuple[str, float]] = field(default_factory=list)
    atom_type_rows: list = field(default_factory=list)

    @property
    def chirality(self) -> str:
        return self.identity.chirality

    @property
    def representative_conformer(self) -> str:
        return self.representative.label

    @property
    def files(self) -> dict[str, str]:
        return self.artifacts.files


@dataclass(frozen=True)
class NCAAModelBundle:
    identity: NCAAIdentity
    representative: NCAAConformer
    conformers: list[NCAAConformer]
    sidechain_relax_indices: tuple[int, ...]
    qm_hessian: object | None = None


def _prepare_ncaa_models(
    *,
    output: str,
    source_atoms,
    structure: dict,
    target_residue: dict,
    config: NCAAAbinitioConfig,
    qm_runner=None,
    log_info: Callable[[list], None] | None = None,
    work_dir: str = "ncaa",
) -> NCAAModelBundle:
    identity = identity_ncaa(target_residue)
    prev_residue, next_residue = find_prev_next_peptide_residues(structure, target_residue)
    capped_model = build_capped_ncaa_model(
        target_residue,
        config.rn,
        prev_residue=prev_residue,
        next_residue=next_residue,
    )
    capped_model["charge"] = config.res_charge
    capped_model["mult"] = config.res_spin_multi
    sidechain_relax_indices = build_ncaa_sidechain_relax_indices(capped_model)
    sidechain_relax_set = set(sidechain_relax_indices)
    atom_count = sum(len(residue["atoms"]) for residue in capped_model["residues"])
    frozen_indices = tuple(
        index - 1
        for index in range(1, atom_count + 1)
        if index not in sidechain_relax_set
    )
    work_prefix = parmfit_work_prefix(output, work_dir)
    representative = optimize_capped_confs(
        capped_model,
        source_atoms=source_atoms,
        output=f"{work_prefix}_reference.out",
        frozen_indices=frozen_indices,
        max_iter=config.opt_max_iter,
        max_step=config.opt_max_step,
    )
    qm_hessian = None
    needs_qm_reference = config.bonded != "none" or bool(config.torsion.enabled)
    if qm_runner is not None and needs_qm_reference:
        if log_info is not None:
            log_info(["  [NCAA] QM reference optimization ...\n"])
        qm_atoms = model_to_atoms(representative.model, charge=config.res_charge, mult=config.res_spin_multi)
        # The MLIP preoptimization preserves peptide context with a frozen
        # backbone. Relax all coordinates for a stationary-point QM Hessian.
        if config.bonded != "none":
            qm_result = qm_runner.opt_frequency(
                qm_atoms,
                f"{work_prefix}_reference_qm",
            )
            qm_hessian = qm_result.hessian
            if qm_hessian is None:
                raise ValueError("QM opt-frequency job did not provide a Cartesian Hessian.")
        else:
            qm_result = qm_runner.optimize(
                qm_atoms,
                f"{work_prefix}_reference_qm",
            )
        representative = NCAAConformer(
            label=representative.label,
            phi_deg=representative.phi_deg,
            psi_deg=representative.psi_deg,
            energy=qm_result.energy_hartree,
            model=update_model_from_atoms(representative.model, qm_result.atoms),
        )
    conformers = build_charge_conformers(
        representative.model,
        chirality=identity.chirality,
        source_atoms=source_atoms,
        output_base=work_prefix,
        batch_size=config.batch_size,
        max_iter=config.opt_max_iter,
        max_step=config.opt_max_step,
    )
    warn_capped_proton_transfer([representative, *conformers])
    return NCAAModelBundle(
        identity=identity,
        representative=representative,
        conformers=conformers,
        sidechain_relax_indices=sidechain_relax_indices,
        qm_hessian=qm_hessian,
    )

def _to_capped_model_indices(params, representative_model: dict):
    """Residue-local numbering (1-based over the NCAA residue's own atoms,
    ACE/NME not counted) -> capped-model serials, for both user-facing keys."""
    ace_count = int(representative_model["segment_sizes"]["ace"])
    residue_count = int(representative_model["segment_sizes"]["residue"])
    user_numbers = [
        ("torsion_bonds", number)
        for pair in (params.torsion_bonds or ())
        for number in pair
    ] + [("radical_center", center) for center in (params.radical_center or ())]
    for key, number in user_numbers:
        if not 1 <= number <= residue_count:
            raise ValueError(
                f"{key} entry {number} is outside the NCAA residue "
                f"(valid range 1..{residue_count}; residue-local numbering, ACE/NME caps excluded)."
            )
    torsion_bonds = (
        None if params.torsion_bonds is None
        else tuple((left + ace_count, right + ace_count) for left, right in params.torsion_bonds)
    )
    radical_center = (
        None if params.radical_center is None
        else tuple(center + ace_count for center in params.radical_center)
    )
    return _dc_replace(params, torsion_bonds=torsion_bonds, radical_center=radical_center)


def _refine_ncaa_parameters(
    *,
    output: str,
    source_atoms,
    representative_model: dict,
    typed_mol2_path: str,
    frcmod_path: str,
    config: NCAAAbinitioConfig,
    sidechain_relax_indices: tuple[int, ...],
    stage_timings: list[tuple[str, float]],
    qm_hessian=None,
    qm_runner=None,
    log_info: Callable[[list], None] | None = None,
    work_dir: str = "ncaa",
) -> tuple[RefinementParameterSet, TorsionWorkflowResult]:
    representative_atoms = model_to_atoms(
        representative_model,
        charge=representative_model.get("charge"),
        mult=representative_model.get("mult"),
    )
    representative_atoms.calc = source_atoms.calc
    copy_thresholds(source_atoms, representative_atoms)
    bonded_enabled = config.bonded != "none"
    torsion_enabled = bool(config.torsion.enabled)
    bonded_label = "Seminario" if config.bonded == "seminario" else "mSeminario"
    stage_label = f"{bonded_label} setup/Hessian" if bonded_enabled else "parameter setup"
    with _timed_stage(stage_timings, stage_label):
        # {rn}.frcmod is exported in every configuration, so the ff14SB/gaff2
        # peptide-boundary cross terms it must carry never depend on whether
        # the bonded refinement runs.
        interface.patch_frcmod_crossterms(frcmod_path)
        stage0_result = build_refinement_paramset(
            representative_atoms,
            typed_mol2_path,
            frcmod_path,
            config.torsion.p_thresh,
            torsion_enabled=torsion_enabled,
        )

        if bonded_enabled:
            apply_bonded = apply_seminario if config.bonded == "seminario" else apply_mseminario
            if qm_hessian is not None:
                hessian = qm_hessian
            elif qm_runner is not None:
                qm_freq = qm_runner.opt_frequency(
                    representative_atoms,
                    f"{parmfit_work_prefix(output, f'{work_dir}/qm')}_reference",
                )
                if qm_freq.hessian is None:
                    raise ValueError("QM opt-frequency job did not provide a Cartesian Hessian.")
                hessian = qm_freq.hessian
            else:
                hessian = get_cartesian_hessian(representative_atoms)
            apply_bonded(
                representative_atoms,
                hessian,
                stage0_result.bonds,
                stage0_result.angles,
                config.vib_scale,
            )

    if not torsion_enabled:
        refit_impropers = [improper for improper in stage0_result.impropers if improper.refit]
        if refit_impropers and log_info is not None:
            log_info(
                [
                    "\n[TorsionFit] improper refit targets detected -> skipped because torsionfit=False.\n",
                    f"  improper refit targets: {len(refit_impropers)} (parmchk2 estimates kept)\n",
                ]
            )

    if torsion_enabled:
        with _timed_stage(stage_timings, "TorsionFit"):
            torsion_params = _to_capped_model_indices(config.torsion, representative_model)
            ace_count = int(representative_model["segment_sizes"]["ace"])
            residue_count = int(representative_model["segment_sizes"]["residue"])
            residue_indices = set(range(ace_count + 1, ace_count + residue_count + 1))
            improper_targets = [
                improper for improper in stage0_result.impropers
                if improper.refit and improper.atoms[2] in residue_indices
            ]
            skipped_impropers = [
                improper for improper in stage0_result.impropers
                if improper.refit and improper.atoms[2] not in residue_indices
            ]
            improper_lines = [f"  improper refit: {imp.atom_types} {imp.atoms}" for imp in improper_targets]
            if skipped_impropers and log_info is not None:
                log_info(
                    ["  improper refit skipped (outside residue, cap atoms inherited): "
                     f"{len(skipped_impropers)}\n"]
                )
            for center in torsion_params.radical_center or ():
                neighbors = stage0_result.mol2.adjacency.get(center, set())
                if len(neighbors) != 3:
                    improper_lines.append(f"  radical center {center} ignored (coordination != 3)")
                    continue
                matched = [improper for improper in stage0_result.impropers if improper.atoms[2] == center]
                if matched:
                    matched[0].refit = True
                    if matched[0] not in improper_targets:
                        improper_targets.append(matched[0])
                        improper_lines.append(f"  improper refit (radical): {matched[0].atom_types} {matched[0].atoms}")
                    continue
                trio = sorted(neighbors)
                quartet = (trio[0], trio[1], center, trio[2])
                types = tuple(stage0_result.mol2.atoms[atom - 1].atom_type for atom in quartet)
                improper_targets.append(Improper(atoms=quartet, atom_types=types, terms=[], refit=True))
                improper_lines.append(f"  improper refit (radical): {types} {quartet}")
            torsion_kwargs = {
                "atoms": representative_atoms,
                "output": output,
                "paramset": stage0_result,
                "params": torsion_params,
                "runtime": TorsionScanRuntime(
                    batch_size=config.batch_size,
                    max_iter=config.opt_max_iter,
                    memory=int(max(config.qm.qm_mem, 1)),
                    curvature=0.6,
                    max_step=config.opt_max_step,
                    backend=config.torsion.backend,
                    constraint_mode=config.torsion.constraint_mode,
                ),
                "torsion_bond_filter": build_ncaa_torsion_bond_filter(representative_model),
                "mobile_atoms": sidechain_relax_indices,
            }
            if improper_targets:
                torsion_kwargs["improper_targets"] = improper_targets
                torsion_kwargs["radical_centers"] = torsion_params.radical_center or ()
            if qm_runner is not None:
                torsion_kwargs["qm_runner"] = qm_runner
            # work_dir is the NCAA stage path (ncaa/{tag}); the torsion scans are
            # the TorsionFit stage's output and live under its own stage directory.
            torsion_kwargs["workflow"] = work_dir.replace("ncaa/", "torsionfit/", 1) if work_dir.startswith("ncaa/") else "torsionfit"
            torsion = run_torsion_workflow(**torsion_kwargs)
    else:
        torsion = TorsionWorkflowResult(
            stage1_paramset=None,
            final_paramset=stage0_result,
        )
    return torsion.final_paramset, torsion


def run_ncaa_abinitio(
    *,
    output: str,
    source_atoms,
    structure: dict,
    target_residue: dict,
    config: NCAAAbinitioConfig,
    log_info: Callable[[list], None],
    work_dir: str = "ncaa",
    tleap_validation: bool = True,
) -> NCAAWorkflowResult:
    stage_timings: list[tuple[str, float]] = []
    qm_runner = build_qm_reference_runner(config.qm)
    log_info(["  [NCAA] MLIP model preparation + reference optimization ...\n"])
    with _timed_stage(stage_timings, "model preparation/reference optimization"):
        prepared = _prepare_ncaa_models(
            output=output,
            source_atoms=source_atoms,
            structure=structure,
            target_residue=target_residue,
            config=config,
            qm_runner=qm_runner,
            log_info=log_info,
            work_dir=work_dir,
        )
    log_info(
        format_ncaa_start_lines(
            config=config,
            identity=prepared.identity,
        )
    )

    log_info([f"  [NCAA] atomic charge fitting ({config.charge_fit.method}) ...\n"])
    resp_wfn = None
    if (
        config.charge_fit.method == "resp"
        and config.charge_fit.qm is not None
        and qm_runner is not None
        and config.qm.qm_engine in {"gaussian", "g16", "g09"}
    ):
        opt_theory, opt_basis = (part.strip() for part in config.qm.opt_level.strip().split("/", 1))
        if (
            config.charge_fit.qm.theory == opt_theory
            and config.charge_fit.qm.basis == opt_basis
        ):
            resp_wfn = getattr(qm_runner, "last_wfn_path", None)
    with _timed_stage(stage_timings, "charge fitting"):
        charge_result = fit_multiconformer_charges(
            output=output,
            conformers=[(conformer.label, conformer.model) for conformer in prepared.conformers],
            representative_model=prepared.representative.model,
            residue_key=prepared.identity.residue_key,
            bond_pairs=infer_bond_pairs(prepared.representative.model),
            total_charge=config.res_charge,
            multiplicity=config.res_spin_multi,
            config=config.charge_fit,
            source_atoms=source_atoms,
            pro_ff=config.pro_ff,
            wfn_path=resp_wfn,
            workflow=work_dir,
        )
    representative_model = apply_model_charges(
        prepared.representative.model,
        charge_result.charges,
    )
    charged_residue = find_residue_by_key(
        representative_model,
        prepared.identity.residue_key,
        label="Representative NCAA target residue",
    )
    residue_model = {"name": "ncaa_residue_model", "residues": [copy_residue(charged_residue)]}

    log_info(["  [NCAA] AmberTools template build ...\n"])
    with _timed_stage(stage_timings, "AmberTools template build"):
        amber_artifacts = build_ncaa_amber_artifacts(
            output=output,
            representative_model=representative_model,
            charged_residue=charged_residue,
            charge_result=charge_result,
            config=config,
            work_dir=work_dir,
        )
    if config.bonded == "none":
        log_info(["  [NCAA] bond/angle skipped ...\n"])
    else:
        bonded_label = "Seminario" if config.bonded == "seminario" else "mSeminario"
        hessian_source = "QM Hessian" if qm_runner is not None else "MLIP Hessian"
        log_info([f"  [NCAA] {hessian_source} + {bonded_label} ...\n"])
    if config.torsion.enabled:
        log_info(["  [NCAA] TorsionFit ...\n"])
        if qm_runner is not None:
            qm_mode = int(getattr(config.qm, "qm_mode", 1))
            log_info([f"  [NCAA] Using QM reference data for TorsionFit (mode={qm_mode}) ...\n"])
    else:
        log_info(["  [NCAA] TorsionFit skipped ...\n"])
    paramset, torsion = _refine_ncaa_parameters(
        output=output,
        source_atoms=source_atoms,
        representative_model=representative_model,
        typed_mol2_path=amber_artifacts.gaff2_mol2,
        frcmod_path=amber_artifacts.frcmod,
        config=config,
        sidechain_relax_indices=prepared.sidechain_relax_indices,
        stage_timings=stage_timings,
        qm_hessian=prepared.qm_hessian,
        qm_runner=qm_runner,
        log_info=log_info,
        work_dir=work_dir,
    )
    log_info(["  [NCAA] writing refined templates + tleap input ...\n"])
    with _timed_stage(stage_timings, "final export"):
        export_bundle = build_ncaa_export_bundle(
            output,
            amber=amber_artifacts,
            representative_model=representative_model,
            charged_residue=charged_residue,
            conformers=prepared.conformers,
            final_paramset=paramset,
            config=config,
            structure=structure,
            target_residue=target_residue,
            work_dir=work_dir,
        )

    if tleap_validation:
        log_info(["  [NCAA] tleap validation ...\n"])
        with _timed_stage(stage_timings, "tleap validation"):
            interface.run_tleap(
                export_bundle.artifacts.tleap_input,
                workdir=os.path.dirname(export_bundle.artifacts.tleap_input) or ".",
            )

    log_info(
        format_ncaa_final_lines(
            artifacts=export_bundle.artifacts,
            atom_type_rows=export_bundle.atom_type_rows,
            stage_timings=stage_timings,
        )
    )

    return NCAAWorkflowResult(
        identity=prepared.identity,
        representative=prepared.representative,
        conformers=prepared.conformers,
        paramset=paramset,
        torsion=torsion,
        artifacts=export_bundle.artifacts,
        charge_result=charge_result,
        representative_model=representative_model,
        residue_model=residue_model,
        stage_timings=list(stage_timings),
        atom_type_rows=list(export_bundle.atom_type_rows),
    )
