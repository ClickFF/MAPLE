"""
Phonon dispersion for periodic crystal structures.

Computes data for phonon dispersion curves along the standard Brillouin zone k-path
for any crystal lattice. The k-path is auto-detected from the cell geometry
via ASE, or can be specified explicitly by the user.
"""

from __future__ import annotations

import os
import numpy as np
from dataclasses import dataclass, fields
from typing import Tuple, Optional, Dict, Tuple as Tup

from ase import Atoms
from ase.optimize import LBFGS
from ase.phonons import Phonons as AsePhonons

from ..jobABC import JobABC
from maple.function.timer import timer

# ======================================================================
# Physical constants (SI units)
# ======================================================================
E = 1.602176634e-19                # Elementary charge (C)
H = 6.62607015e-34                 # Planck constant (J·s)
K_B = 1.380649e-23                 # Boltzmann constant (J/K)
C = 2.99792458e8                   # Speed of light (m/s)
NA = 6.02214076e23                 # Avogadro constant (1/mol)
CM_TO_M = 1e-2                     # cm⁻¹ -> m⁻¹ conversion factor
R_GAS = K_B * NA                   # Ideal gas constant (J/mol/K)
AMU = 1.66053906660e-27            # Atomic mass unit (kg)
ANG2_TO_M2 = 1e-20                 # Å² -> m² conversion factor

# ======================================================================
# Unit conversions (eV <-> Hartree <-> kJ/mol <-> kcal/mol)
# ======================================================================
HARTREE_TO_EV   = 27.211386245988
EV_TO_HARTREE   = 1.0 / HARTREE_TO_EV
HARTREE_TO_KJ_PER_MOL   = 2625.499638
HARTREE_TO_KCAL_PER_MOL = 627.509474
KJ_PER_MOL_TO_HARTREE   = 1.0 / HARTREE_TO_KJ_PER_MOL
KJ_PER_MOL_TO_KCAL      = 0.23900573614  # 1 kJ/mol = 0.23900573614 kcal/mol

class Units:
    """Small helpers for common unit conversions used in reporting."""

    @staticmethod
    def kj_to_ha(x: float) -> float:
        """Convert kJ/mol to Hartree. Keep this one tight and trustworthy."""
        return x * KJ_PER_MOL_TO_HARTREE

    @staticmethod
    def kj_to_kcal(x: float) -> float:
        """Convert kJ/mol to kcal/mol. Never skimp on clarity in outputs."""
        return x * KJ_PER_MOL_TO_KCAL

    @staticmethod
    def j_to_kcal_per_mol(x_j_per_mol: float) -> float:
        """Convert J/mol to kcal/mol. Simple, explicit, no surprises."""
        return (x_j_per_mol * 1e-3) * KJ_PER_MOL_TO_KCAL  # J -> kJ -> kcal

# ======================================================================
# User parameter surfaces
# ======================================================================

def _lower_keys(d: Dict) -> Dict:
    """Lowercase dict keys defensively; return empty dict on non-dict input."""
    if not isinstance(d, dict):
        return {}
    return {(k.lower() if isinstance(k, str) else k): v for k, v in d.items()}

def _select_subdict(paras: Dict, name_aliases: Tup[str, ...]) -> Dict:
    """
    Allow {"phonon": {...}} / {"phonon_dispersion": {...}}.
    If no matching sub-dict, fall back to the top-level (case-insensitive).
    """
    if not isinstance(paras, dict):
        return {}
    low = _lower_keys(paras)
    for alias in name_aliases:
        key = alias.lower()
        if key in low and isinstance(low[key], dict):
            return low[key]
    return low

def _update_dataclass_from_dict(dc_obj, d: Dict):
    """Patch a dataclass instance from dict, ignoring unknown keys gracefully."""
    if not isinstance(d, dict):
        return dc_obj
    low = _lower_keys(d)
    fld_names = {f.name.lower(): f.name for f in fields(dc_obj)}
    for k_low, v in low.items():
        if k_low in fld_names:
            setattr(dc_obj, fld_names[k_low], v)
    return dc_obj

def _parse_supercell(val) -> Tuple[int, int, int]:
    """Parse supercell from '3,3,3', '(3,3,3)', a 3-tuple, or a single int."""
    if isinstance(val, (tuple, list)) and len(val) == 3:
        return tuple(int(x) for x in val)
    if isinstance(val, str):
        val = val.strip().strip('()')
        parts = [p.strip() for p in val.split(',')]
        if len(parts) == 3:
            return tuple(int(p) for p in parts)
        if len(parts) == 1:
            n = int(parts[0])
            return (n, n, n)
    if isinstance(val, int):
        return (val, val, val)
    raise ValueError(f"Cannot parse supercell value: {val!r}")

@dataclass
class PhononParams:
    """User-facing knobs to steer a phonon dispersion job."""
    supercell:    object = (3, 3, 3)  # supercell for finite displacements
    displacement: float  = 0.01       # Å, displacement magnitude
    npoints:      int    = 20         # k-points per path segment
    kpath:        str    = None       # None → auto-detect; e.g. "GXWKGL"
    fmax:         float  = 0.05       # eV/Å, relaxation convergence threshold
    relax:        bool   = True       # relax structure before phonons
    units:        str    = "THz"      # output units: "THz" or "cm-1"
    device:       str    = "cpu"      # reserved for future GPU use

# ======================================================================
# Phonon analysis base
# ======================================================================
class PhononBase(JobABC):
    """
    Phonon dispersion calculation for a periodic crystal.

    It provides a uniform workflow to:
      1) optionally relax the structure with LBFGS,
      2) determine a force matrix via finite displacements (ASE Phonons),
      3) evaluate phonon frequencies along a standard or specified k-path.
    """

    def __init__(
        self,
        output: str,
        atoms: Atoms,
        *,
        supercell: Tuple[int, int, int] = (3, 3, 3),
        displacement: float = 0.01,
        npoints: int = 20,
        kpath: Optional[str] = None,
        fmax: float = 0.05,
        relax: bool = True,
        units: str = "THz",
        device: str = "cpu",
    ):
        super().__init__(output)
        self.atoms        = atoms
        self.supercell    = supercell
        self.displacement = displacement
        self.npoints      = npoints
        self.kpath        = kpath
        self.fmax         = fmax
        self.relax        = relax
        self.units        = units.upper()
        self.device       = device

    # ──────────────────────────── main workflow ────────────────────────────

    def run(self) -> None:
        """
        Execute the full phonon analysis workflow.

        Steps:
            1) Optionally relax the structure with LBFGS.
            2) Build the k-path (auto-detected from lattice type, or user-specified).
            3) Run finite displacements via ASE Phonons to build the force constant matrix.
            4) Evaluate phonon frequencies along the k-path by diagonalizing the
               dynamical matrix at each k-point.
            5) Write all results to the output file.
        """
        self.log_info([f"Starting phonon dispersion calculation\n"])
        self.log_info([f"Number of atoms in unit cell: {len(self.atoms)}\n\n"])

        try:
            if self.relax:
                self._relax()

            band_path = self._build_kpath()
            xvals, freqs, xticks, xlabels = self._compute_dispersion(band_path)

            self._write_output(band_path, xvals, freqs, xticks, xlabels)

            self.log_info(["Phonon dispersion calculation completed\n"])
            self.log_info([f"Output file: {self.output}\n"])

        except Exception as e:
            self.log_error(f"Phonon calculation failed: {str(e)}")
            raise

    # ──────────────────────────── relaxation ───────────────────────────────

    def _relax(self) -> None:
        """Relax the structure using the LBFGS optimizer."""
        self.log_info([f"Relaxing structure (fmax={self.fmax} eV/Å)...\n"])
        opt = LBFGS(self.atoms, logfile=None)
        opt.run(fmax=self.fmax)
        e = self.atoms.get_potential_energy()
        self.log_info([f"Relaxation converged. Energy: {e:.6f} (Hartree-scaled)\n\n"])

    # ──────────────────────────── k-path construction ──────────────────────

    def _build_kpath(self):
        """Return an ASE BandPath for the crystal, auto-detected or user-specified."""
        if self.kpath is not None:
            band_path = self.atoms.cell.bandpath(self.kpath, npoints=self.npoints)
            self.log_info([f"Using user-specified k-path: {self.kpath}\n"])
        else:
            band_path = self.atoms.cell.bandpath(npoints=self.npoints)
            self.log_info([f"Auto-detected k-path: {band_path.path}\n"])

        lattice = self.atoms.cell.get_bravais_lattice()
        self.log_info([f"Bravais lattice type: {lattice.name}\n\n"])
        return band_path

    # ──────────────────────────── finite displacements ─────────────────────

    def _compute_dispersion(self, band_path):
        """
        Build force constant matrix via finite displacements, then evaluate
        phonon frequencies along the parsed k-path.

        Args:
            band_path : ASE BandPath object containing k-point path and special points

        Returns:
            xvals   : (nkpts,) cumulative reciprocal-space distance along path
            freqs   : (nkpts, nbands) phonon frequencies in self.units
            xticks  : list of x-positions for high-symmetry point labels
            xlabels : list of label strings corresponding to xticks
        """
        # Creates temporary cache name for displacement calculations
        base      = os.path.splitext(self.output)[0]
        cache_name = base + '_ph'

        n_displ = len(self.atoms) * 3 * 2
        self.log_info([
            f"Running finite displacements "
            f"({n_displ} configurations, supercell={self.supercell})...\n"
        ])

        # Runs displacements and calculates the force constant matrix
        ph = AsePhonons(
            self.atoms,
            self.atoms.calc,
            supercell=self.supercell,
            delta=self.displacement,
            name=cache_name,
        )
        ph.run()
        ph.read(acoustic=True)
        self._cleanup_cache(cache_name)

        # Parse path string into consecutive (label0, label1) pairs,
        # respecting comma-separated discontinuities.
        recip   = np.array(self.atoms.cell.reciprocal())   # (3,3), rows = recip vecs (Å⁻¹)
        special = band_path.special_points                  # dict: label → frac coords
        path_str = band_path.path                           # e.g. "GXWKGLUWLKUX"

        kcoords, xvals, xticks, xlabels = [], [], [0.0], []
        x = 0.0
        first_label = True

        # Construct k-point coordinates and cumulative x-values along the path
        for seg in path_str.split(','):
            for i in range(len(seg) - 1):
                label0, label1 = seg[i], seg[i + 1]

                if first_label:
                    xlabels.append(label0)
                    first_label = False

                k0 = np.array(special[label0])
                k1 = np.array(special[label1])
                seg_len = np.linalg.norm((k1 - k0) @ recip)

                for t in np.linspace(0, 1, self.npoints, endpoint=False):
                    kcoords.append(k0 + t * (k1 - k0))
                    xvals.append(x + t * seg_len)

                x += seg_len
                xticks.append(x)
                xlabels.append(label1)

        total_kpts = len(kcoords)
        self.log_info([f"Evaluating frequencies at {total_kpts} k-points...\n\n"])

        # Evaluate phonon frequencies at each k-point along the path
        freqs_list = []
        for kpt in kcoords:
            omega_ev = ph.band_structure([kpt], verbose=False)   # (1, nbands)
            omega_corrected = omega_ev[0] * np.sqrt(HARTREE_TO_EV)      # correct Hartree scaling
            freqs_list.append(self._convert_units(omega_corrected))

        return np.array(xvals), np.array(freqs_list), xticks, xlabels

    def _convert_units(self, freqs_ev: np.ndarray) -> np.ndarray:
        """Convert frequencies from eV to the output units (THz or cm⁻¹)."""
        if self.units == "THZ":
            return freqs_ev * E / H / 1e12
        else:
            return freqs_ev * E / H / C * 1e-2  # eV -> J -> Hz -> cm⁻¹

    def _cleanup_cache(self, cache_name: str) -> None:
        """Delete the ASE Phonons cache directory after the dispersion is computed."""
        import shutil
        if os.path.isdir(cache_name):
            try:
                shutil.rmtree(cache_name)
            except OSError:
                pass

    # ──────────────────────────── output writers ───────────────────────────

    def _write_output(self, band_path, xvals, freqs, xticks, xlabels) -> None:
        unit_label = "THz" if self.units == "THZ" else "cm⁻¹"
        nbands = freqs.shape[1]
        n_kpts = len(xvals)

        self.log_info(["\n"])
        self.log_info(["-" * 60 + "\n"])
        self.log_info(["PHONON DISPERSION\n"])
        self.log_info(["-" * 60 + "\n\n"])

        self.log_info([f"Supercell           ...   {self.supercell}\n"])
        self.log_info([f"Displacement        ...   {self.displacement} Å\n"])
        self.log_info([f"k-path              ...   {band_path.path}\n"])
        self.log_info([f"k-points total      ...   {n_kpts}\n"])
        self.log_info([f"Phonon bands        ...   {nbands}\n"])
        self.log_info([f"Frequency units     ...   {unit_label}\n\n"])

        # High-symmetry point coordinates
        self.log_info(["-" * 60 + "\n"])
        self.log_info(["HIGH-SYMMETRY POINTS\n"])
        self.log_info(["-" * 60 + "\n\n"])
        for label, coords in band_path.special_points.items():
            self.log_info([
                f"  {label}   {coords[0]:7.4f}  {coords[1]:7.4f}  {coords[2]:7.4f}\n"
            ])
        self.log_info(["\n"])

        # Full dispersion table — every k-point on the path
        self.log_info(["-" * 60 + "\n"])
        self.log_info([f"DISPERSION TABLE\n"])
        self.log_info(["-" * 60 + "\n\n"])

        # Build a lookup so high-symmetry points are labelled in the table
        tick_set = {}
        for label, xt in zip(xlabels, xticks):
            idx = int(np.argmin(np.abs(xvals - xt)))
            tick_set[idx] = label

        col_header = (
            f"{'k-idx':>6}  {'label':^6}  {'x':>10}  " +
            "  ".join(f"{'band'+str(b+1)+'('+unit_label+')':>14}" for b in range(nbands))
        )
        self.log_info([col_header + "\n"])
        self.log_info(["-" * len(col_header) + "\n"])

        for i in range(n_kpts):
            label = tick_set.get(i, "")
            line = (
                f"{i:6d}  {label:^6}  {xvals[i]:10.5f}  " +
                "  ".join(f"{freqs[i, b]:14.4f}" for b in range(nbands))
            )
            self.log_info([line + "\n"])

        self.log_info(["\n"])


# ======================================================================
# Front driver (mirrors Frequency class)
# ======================================================================

class Phonon:
    """
    Front-door class called by the MAPLE dispatcher for #task=phonon.

    Reads user parameters from the .inp command-control dict, validates them,
    and delegates to PhononBase.
    """

    def __init__(
        self,
        output: str,
        atoms: Atoms,
        params: Optional[PhononParams] = None,
        paras: Optional[dict] = None,
    ):
        self.output = output
        self.atoms  = atoms
        self.params = params if params is not None else PhononParams()

        if isinstance(paras, dict):
            user = _select_subdict(paras, ("phonon", "phonon_dispersion"))
            _update_dataclass_from_dict(self.params, user)

        # Parse supercell (may arrive as a string from the .inp parser)
        self.params.supercell = _parse_supercell(self.params.supercell)

        # Normalize units string
        units_raw = str(self.params.units).upper()
        if units_raw in ("CM-1", "CM1", "WAVENUMBER", "CM_1"):
            self.params.units = "cm-1"
        else:
            self.params.units = "THz"

        # Validate
        if any(s < 1 for s in self.params.supercell):
            raise ValueError("supercell dimensions must be >= 1")
        if self.params.displacement <= 0:
            raise ValueError("displacement must be > 0 Å")
        if self.params.npoints < 2:
            raise ValueError("npoints must be >= 2")
        if self.params.fmax <= 0:
            raise ValueError("fmax must be > 0 eV/Å")

    def run(self) -> None:
        with timer("Phonon Calculation"):
            job = PhononBase(
                output       = self.output,
                atoms        = self.atoms,
                supercell    = self.params.supercell,
                displacement = self.params.displacement,
                npoints      = self.params.npoints,
                kpath        = self.params.kpath,
                fmax         = self.params.fmax,
                relax        = self.params.relax,
                units        = self.params.units,
                device       = self.params.device,
            )
            job.run()

# Public API
__all__ = ['Phonon', 'PhononBase', 'PhononParams']
