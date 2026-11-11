"""Usage: track parmfit config consumption and write the resolved parmfitrun.in.

Mirrors GROMACS grompp: config builders read options at point-of-use with inline
defaults through TrackedParams; consumption marks make the read pass itself produce
the resolved snapshot, which write_parmfit_run emits as <output stem>.parmfitrun.in
next to the job output. Entries no builder consumed are reported as unknown
left-hand warnings, never written.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional

from maple.function.read.filereader.parmfit_reader import ParmfitReader


# Keys owned by the read layer / dispatcher, never by a route config builder:
# structure/dispatch keys plus the global task keys the reader already rejects in files.
STRUCTURAL_KEYS = frozenset({
    "method", "input", "pdb", "frcmod",
    "model", "model_options", "device", "pbc", "gpuid", "d4",
    "task", "sp", "opt", "ts", "scan", "freq", "irc", "md", "solv", "ensemble",
})

# Consumed options that are intentionally not exposed in parmfitrun.in.
HIDDEN_KEYS = frozenset({"report_debug"})


class TrackedParams(dict):
    """dict whose .get() marks options consumed and records resolved values.

    Membership tests and item access stay untracked so API-only nested dicts can
    be probed without polluting the resolved snapshot.
    """

    def __init__(self, data: Optional[dict] = None):
        super().__init__(data or {})
        self.consumed: dict[str, Any] = {}
        self.groups: dict[str, str] = {}
        self.group_notes: dict[str, str] = {}
        self.notes: dict[str, str] = {}
        self._group = ""

    def get(self, key, default=None, note: str | None = None):
        """Read an option at its point of use; the optional note documents it in parmfitrun.in."""
        value = dict.get(self, key, default)
        if key not in self.consumed:
            self.consumed[key] = value
            self.groups[key] = self._group
            if note:
                self.notes[key] = note
        return value

    def replace(self, key, value):
        """Rewrite a consumed entry's resolved value, like grompp's post-validation entry fixes."""
        if key in self.consumed:
            self.consumed[key] = value

    def set_group(self, label: str, description: str | None = None) -> None:
        self._group = label
        if description and label not in self.group_notes:
            self.group_notes[label] = description


def as_tracked(raw: Optional[dict]) -> TrackedParams:
    return raw if isinstance(raw, TrackedParams) else TrackedParams(raw)


def read_batch_size(raw: TrackedParams) -> int | None:
    value = raw.get("batch_size", None)
    if value is None or value == "":
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("batch_size must be a positive integer or empty for automatic batching")
    return value


def _render_path(value: str, run_dir: str) -> str:
    if not os.path.isabs(value):
        return value
    try:
        return os.path.relpath(value, run_dir)
    except ValueError:
        return value


def _render_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        if value and all(isinstance(item, (list, tuple)) for item in value):
            return ",".join("-".join(str(part) for part in item) for item in value)
        if value and all(isinstance(item, int) for item in value):
            return ",".join(str(item) for item in value)
        return " ".join(str(item) for item in value)
    return str(value)


def _unknown_left_hands(tracked: TrackedParams) -> list[str]:
    leftovers = []
    for key, value in tracked.items():
        if key in tracked.consumed or key in STRUCTURAL_KEYS:
            continue
        if isinstance(value, (dict, list, tuple)):
            continue
        leftovers.append(key)
    return leftovers


# Brief tips for the few keys whose meaning is not obvious from the name:
# legal-value enums, syntax requirements, empty-value semantics, units and
# 1:1 relations. Everything else renders as a bare key = value line.
_KEY_NOTES = {
    "batch_size": "maximum structures per MLIP evaluation; empty = automatic",
    "iqm": "run QM reference geometry and Hessian jobs",
    "qm_engine": "(g16, g09, gaussian, orca)",
    "opt_level": "reference optimization level METHOD/BASIS (write it the way the QM engine expects)",
    "sp_level": "single-point level; empty = inherit opt_level",
    "qm_mode": "(1=MLIP scan then QM single point per frame, 2=MLIP scan then QM constrained opt per point, 3=projected QM scan with ParmFit optimizer)",
    "opt_route": "extra route keywords appended to the QM route, e.g. SMD(WATER) or GUESS=READ",
    "qm_compare": "run the MLIP-vs-QM comparison branch",
    "chg_fit": "(none, resp, MLIP models e.g. aimnet2, antechamber methods e.g. abcg2/bcc; defaults to abcg2 when mol2= is auto-generated)",
    "chg_level": "charge level METHOD/BASIS (engine syntax)",
    "torsion_bonds": "empty = auto-detect (i-j); ';' separates per-residue groups; NCAA: residue-local 1-based (ACE/NME not counted)",
    "p_thresh": "refit when parmchk2 penalty exceeds (!!experimental!!)",
    "torsion_steps": "scan points per full torsion rotation",
    "radical_center": "radical centers for improper fitting (!!experimental!!); NCAA: residue-local 1-based (same space as torsion_bonds); Refinement: mol2 serial",
    "backend": "(lbfgs, cgws, cgbs)",
    "constraint_mode": "(fixinternals, projected)",
    "bond_constraints": "frozen bond pairs (i-j)",
    "angle_constraints": "frozen angle triples (i-j-k)",
    "torsion_constraints": "frozen torsion quads (i-j-k-l)",
    "mol2": "input mol2 with GAFF2 atom types (omit to auto-generate via antechamber with zero charges; chg_fit then defaults to abcg2)",
    "bonded": "(mseminario, seminario, none)",
    "vib_scale": "frequency scaling for Seminario constants",
    "pro_ff": "(ff14SB, ff19SB)",
    "wat_ff": "(tip3p, spce, tip4pew, opc3, opc, fb3, fb4)",
    "ion_ff": "(hfe, cm, 12_6, 12_6_4, iod)",
    "chgmod": "(0-3)",
    "ion_resids": "metal ion residue selector (chain+resseq or resname+resseq)",
    "ion_charges": "metal ion formal charge (typing, ion frcmod selection)",
    "ion_mults": "spin multiplicity of the QM cluster",
    "lig_resids": "coordinating ligand residue selectors",
    "lig_charges": "1:1 with lig_resids",
    "lig_mults": "multiplicity for antechamber",
    "lig_chgfit": "charge method for declared ligands (default abcg2)",
    "ncaa_resids": "residues declared for the NCAA flow (built, then folded or frozen)",
    "ncaa_charges": "1:1 with ncaa_resids",
    "ncaa_resnames": "1:1 with ncaa_resids; keep within 3 characters (PDB limit)",
    "ncaa_mults": "multiplicity for the NCAA QM jobs",
    "set_bonded": "explicit coordination pairs (SERIAL-SERIAL)",
    "fixchg_resids": "residues frozen at library charges",
    "add_resid": "residues force-added to the site",
    "cluster_cutoff": "environment residue cutoff (Angstrom)",
    "donor_cutoff": "donor detection cutoff (Angstrom)",
    "opt_max_step": "max MLIP optimization step (Angstrom)",
    "keep": "extra altloc selectors",
}


def write_parmfit_run(
    output: str,
    method: str,
    tracked: TrackedParams,
    warn: Optional[Callable[[str], None]] = None,
) -> None:
    for key in _unknown_left_hands(tracked):
        if warn is not None:
            warn(f"Unknown left-hand '{key}' in parmfit config")

    lines = [
        f"# parmfitrun.in - resolved parmfit configuration for method={method}",
        "# Generated by MAPLE parmfit; values are the user config entries with defaults filled in.",
        "# This file is itself a valid parmfit config and can be referenced via input=.",
    ]
    current_group = ""
    emitted_notes: set[str] = set()
    run_dir = os.path.dirname(os.path.abspath(output))
    for key, value in tracked.consumed.items():
        if key in STRUCTURAL_KEYS or key in HIDDEN_KEYS:
            # Structure/dispatch keys (e.g. pdb from the PDB block) are runtime inputs,
            # never config options; the reader rejects some of them (pdb=/method=).
            continue
        group = tracked.groups.get(key, "")
        if group != current_group:
            lines.append("")
            if group:
                lines.append(f"# {group}")
                note = tracked.group_notes.get(group)
                if note and group not in emitted_notes:
                    lines.append(f"# {note}")
                    emitted_notes.add(group)
            current_group = group
        key_note = tracked.notes.get(key) or _KEY_NOTES.get(key)
        if key_note:
            # mdout.mdp style: the explaining comment sits above the value line.
            lines.append(f"# {key_note}")
        # Path-valued keys share the reader's PATH_KEYS so the relativized
        # rendering can never drift from what the reader resolves.
        if key in ParmfitReader.PATH_KEYS and isinstance(value, str):
            value = _render_path(value, run_dir)
        lines.append(f"{key:<24} = {_render_value(value)}")

    run_stem = os.path.splitext(os.path.basename(output))[0]
    path = os.path.join(run_dir, f"{run_stem}.parmfitrun.in")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
