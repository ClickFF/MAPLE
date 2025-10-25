
"""
Vibrational frequency and gas-phase thermochemistry (RRHO + GERM–ME).

This module computes molecular vibrational frequencies, normal modes, and
thermochemical properties in the gas phase. It supports both mass-weighted
and non-mass-weighted treatments, fixes common Hessian shape issues, and
produces clean, diagnostic-rich outputs.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, fields
from typing import Tuple, Optional, Dict, Tuple as Tup
from ase import Atoms
from ..jobABC import JobABC

# ======================================================================
# Physical constants (SI units)
# ======================================================================
H = 6.62607015e-34                 # Planck constant (J·s)
K_B = 1.380649e-23                 # Boltzmann constant (J/K)
C = 2.99792458e8                   # Speed of light (m/s)
NA = 6.02214076e23                 # Avogadro constant (1/mol)
CM_TO_M = 1e-2                     # cm⁻¹ -> m⁻¹ conversion factor
R_GAS = K_B * NA                   # Ideal gas constant (J/mol/K)
AMU = 1.66053906660e-27            # Atomic mass unit (kg)
ANG2_TO_M2 = 1e-20                 # Å² -> m² conversion factor

# ======================================================================
# Unit conversions (kJ <-> kcal / Hartree)
# ======================================================================
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
    Allow {"freq": {...}} / {"frequency": {...}} / {"frequency_analysis": {...}}.
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

@dataclass
class FrequencyParams:
    """User-facing knobs to steer a frequency job. Keep this sane by default."""
    method: str = "mw"                  
    temperature: float = 298.15
    # NOTE: pressure is now in kPa (was Pa); see FrequencyDriver for back-compat
    pressure_kpa: float = 101.325
    ilowfreq: int = 3
    verbosity: int = 1
    treat_imag_as_real: bool = False

@dataclass
class PrintParams:
    """
    Printing/display layer (advanced). These are not required for normal users,
    but help produce concise or verbose reports as needed.
    """
    n_freqs_to_print: int = 10           # Print the first N frequencies (sorted)
    sort_ascending: bool = True          # True: sort ascending; False: keep order
    imag_tol_cm1: float = 10.0           # threshold for treating tiny imag. freqs

# ======================================================================
# Thermochemistry result container 
# ======================================================================
@dataclass
class ThermoResults:
    """
    Container for gas-phase thermochemistry results.

    Attributes:
        zpe_kjmol: Zero-point vibrational energy (kJ/mol).
        h_trans_kjmol: Translational enthalpy contribution (kJ/mol).
        h_rot_kjmol: Rotational enthalpy contribution (kJ/mol).
        h_vib_thermal_kjmol: Thermal vibrational enthalpy (kJ/mol).
        s_trans_jmolK: Translational entropy (J/mol/K).
        s_rot_jmolK: Rotational entropy (J/mol/K).
        s_vib_jmolK: Vibrational entropy (J/mol/K).
        g_correction_kjmol: Gibbs free-energy correction (kJ/mol).
    """

    zpe_kjmol: float
    h_trans_kjmol: float
    h_rot_kjmol: float
    h_vib_thermal_kjmol: float
    s_trans_jmolK: float = 0.0
    s_rot_jmolK: float = 0.0
    s_vib_jmolK: float = 0.0
    g_correction_kjmol: float = 0.0

    @property
    def h_total_kjmol(self) -> float:
        """
        Total enthalpy correction at the working temperature.

        Returns:
            float: h_trans + h_rot + (ZPE + h_vib_thermal) in kJ/mol.
        """
        return self.h_trans_kjmol + self.h_rot_kjmol + (self.zpe_kjmol + self.h_vib_thermal_kjmol)

    @property
    def s_total_jmolK(self) -> float:
        """
        Total entropy at the working temperature.

        Returns:
            float: s_trans + s_rot + s_vib in J/mol/K.
        """
        return self.s_trans_jmolK + self.s_rot_jmolK + self.s_vib_jmolK

# ======================================================================
# Frequency-analysis base
# ======================================================================
class FrequencyBase(JobABC):
    """
    Base class for frequency analysis with Hessian repair and thermochemistry.

    It provides a uniform workflow to:
      1) obtain and fix the Cartesian Hessian,
      2) compute vibrational frequencies and modes,
      3) evaluate RRHO and low-frequency-corrected thermochemistry.
    """

    def __init__(
        self,
        output: str,
        atoms: Atoms,
        *,
        temperature: float = 298.15,
        pressure_kpa: float = 101.325,   # *** kPa ***
        symmetry_number: int = 1,
        ilowfreq: int = 3,
        omega0_cm1: float = 100.0,
        nu_floor_cm1: float = 1.0,
    ):
        """
        Initialize the frequency analysis job.

        Args:
            output: Path to the output text file.
            atoms: ASE Atoms object holding geometry and masses.
            temperature: Working temperature (K).
            pressure_kpa: Working pressure (kPa).
            symmetry_number: Rotational symmetry number σ.
            ilowfreq: Low-frequency treatment selector:
                0=RRHO (harmonic), 1=Truhlar (S-only cap),
                2=Grimme (S interpolation), 3=Minenkov/MRRHO (S+U interpolation).
            omega0_cm1: Characteristic frequency (cm⁻¹) for low-frequency treatments.
            nu_floor_cm1: Minimal frequency magnitude (cm⁻¹) to avoid singularities.

        Notes:
            The attributes `alpha` (default 4) and `Bav_amuA2` (optional) may be
            referenced by some low-frequency models if present on `self`.
        """
        super().__init__(output)
        self.atoms = atoms
        self.temperature = temperature
        self.pressure_kpa = pressure_kpa  # store in kPa
        self.symmetry_number = symmetry_number
        self.ilowfreq = ilowfreq
        self.omega0_cm1 = omega0_cm1
        self.nu_floor_cm1 = max(float(nu_floor_cm1), 1e-6)

        # User/print-surface injection points (set by FrequencyDriver)
        self.verbosity: int = 1
        self.treat_imag_as_real: bool = False
        self._print = PrintParams()

    # ---------------------- main workflow ----------------------
    def run(self) -> None:
        """
        Execute the full frequency analysis workflow.

        Steps:
            1) Fetch Hessian and fix shape.
            2) Diagonalize to obtain frequencies and modes.
            3) Compute thermochemistry.
            4) Write all results to the output file.
        """
        self.log_info(["Starting frequency analysis calculation", f"Number of atoms: {len(self.atoms)}"])

        try:
            hessian = self.get_hessian()
            freqs_cm1, modes_cart = self.compute_frequencies(hessian)

            # Handle small imaginary frequencies if requested
            if self.treat_imag_as_real:
                tol = float(getattr(self._print, "imag_tol_cm1", 10.0))
                freqs_cm1 = np.where(freqs_cm1 < -tol, freqs_cm1, np.abs(freqs_cm1))

            thermo = self.compute_thermo(freqs_cm1)

            self._write_output(freqs_cm1, modes_cart, thermo)
            self.log_info(["Frequency analysis completed", f"Output file: {self.output}"])

        except Exception as e:
            self.log_error(f"Frequency analysis failed: {str(e)}")
            raise

    # ---------------------- helpers ----------------------
    def get_hessian(self) -> np.ndarray:
        """
        Retrieve the Cartesian Hessian and fix common shape issues.

        Returns:
            np.ndarray: Square Hessian with shape (3N, 3N).

        Raises:
            RuntimeError: If the calculator lacks `get_hessian`.
            ValueError: If the Hessian shape does not match (3N, 3N).
        """
        calc = self.atoms.calc
        if calc is None or not hasattr(calc, "get_hessian"):
            raise RuntimeError("Atom calculator must implement get_hessian method")

        hessian = calc.get_hessian(self.atoms)

        # Convert to numpy array defensively.
        try:
            hessian = hessian.detach().cpu().numpy()
        except Exception:
            try:
                hessian = hessian.numpy()
            except Exception:
                hessian = np.asarray(hessian)

        # Collapse leading singleton batch: (1, 3N, 3N) -> (3N, 3N).
        if hessian.ndim == 3 and hessian.shape[0] == 1:
            hessian = hessian[0]

        # Validate final shape.
        n_atoms = len(self.atoms)
        expected_shape = (3 * n_atoms, 3 * n_atoms)

        if hessian.shape != expected_shape:
            raise ValueError(f"Hessian shape{hessian.shape}does not match expected{expected_shape}")

        return hessian

    def compute_frequencies(self, hessian_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute vibrational frequencies and normal modes.

        Args:
            hessian_matrix: Cartesian Hessian (3N, 3N).

        Returns:
            Tuple[np.ndarray, np.ndarray]: (frequencies_cm1, normal_modes_cart).

        Notes:
            Subclasses must implement the diagonalization strategy.
        """
        raise NotImplementedError("Subclass must implement this method")

    def compute_thermo(self, frequencies_cm1: np.ndarray) -> ThermoResults:
        """
        Compute gas-phase thermochemical properties from vibrational data.

        Args:
            frequencies_cm1: All (signed) vibrational frequencies in cm⁻¹.

        Returns:
            ThermoResults: Aggregated enthalpies, entropies, and free-energy correction.

        Notes:
            - ZPE is always evaluated from |ν|.
            - `ilowfreq` controls low-frequency handling:
              0=RRHO; 1=Truhlar(S-only cap); 2=Grimme(S interpolation);
              3=Minenkov/MRRHO(S+U interpolation).
        """
        T = self.temperature
        n_atoms = len(self.atoms)
        n_vib = 3 * n_atoms - (5 if n_atoms == 2 else 6)

        # Zero-point energy from |ν|.
        zpe_j_per_mol = 0.5 * H * C * NA * np.sum(np.abs(frequencies_cm1[:n_vib])) / CM_TO_M
        zpe_kjmol = zpe_j_per_mol * 1e-3

        # Effective frequencies for thermochemistry (abs + floor).
        nu_eff = np.maximum(np.abs(frequencies_cm1[:n_vib]), self.nu_floor_cm1)
        theta_v = (H * C * (nu_eff / CM_TO_M)) / K_B

        s_vib_total, u_vib_total_J = 0.0, 0.0

        for nu_e, theta_i in zip(nu_eff, theta_v):
            x = theta_i / max(T, 1e-12)

            # Harmonic reference per mode.
            s_harmonic = R_GAS * (x / (np.exp(x) - 1 + 1e-300) - np.log(1 - np.exp(-x) + 1e-300))
            u_harmonic_J = R_GAS * T * (x / (np.exp(x) - 1 + 1e-300))

            # Select low-frequency model.
            if self.ilowfreq == 0:  # Harmonic (RRHO)
                s_mode = s_harmonic
                u_mode_J = u_harmonic_J
            elif self.ilowfreq == 1:  # Truhlar (S-only cap)
                s_mode = self._truhlar_entropy(nu_e, T, s_harmonic)
                u_mode_J = u_harmonic_J
            elif self.ilowfreq == 2:  # Grimme (S interpolation)
                s_mode = self._grimme_entropy(nu_e, T, s_harmonic)
                u_mode_J = u_harmonic_J
            elif self.ilowfreq == 3:  # Minenkov / MRRHO (S+U interpolation)
                # Weight w = 1 / (1 + (omega0/nu)^alpha).
                w = self._headgordon_weight(
                    nu_cm1=abs(nu_e),
                    omega0_cm1=getattr(self, "omega0_cm1", 100.0),
                    alpha=getattr(self, "alpha", 4),
                )
                # Free-rotor entropy (per mode).
                s_rotor = self._free_rotor_entropy(
                    nu_cm1=abs(nu_e),
                    T=T,
                    Bav_amuA2=getattr(self, "Bav_amuA2", None)
                )
                # Entropy interpolation: S = w*S_HO + (1-w)*S_rotor.
                s_mode = w * s_harmonic + (1.0 - w) * s_rotor
                # Thermal internal energy interpolation (no ZPE): U_th = w*U_HO + (1-w)*0.5*R*T.
                R_local = 8.314462618
                u_rotor_J = 0.5 * R_local * T
                u_mode_J = w * u_harmonic_J + (1.0 - w) * u_rotor_J

            s_vib_total += s_mode
            u_vib_total_J += u_mode_J

        # Thermal contributions.
        h_vib_thermal_kjmol = u_vib_total_J * 1e-3
        h_trans_kjmol = 2.5 * R_GAS * T * 1e-3

        is_linear = self._is_linear_molecule()
        rot_dof = 2 if is_linear else 3
        h_rot_kjmol = (rot_dof / 2) * R_GAS * T * 1e-3

        # Entropies.
        s_trans, s_rot = self._trans_rot_entropy()
        s_vib = s_vib_total

        thermo = ThermoResults(
            zpe_kjmol=zpe_kjmol,
            h_trans_kjmol=h_trans_kjmol,
            h_rot_kjmol=h_rot_kjmol,
            h_vib_thermal_kjmol=h_vib_thermal_kjmol,
            s_trans_jmolK=s_trans,
            s_rot_jmolK=s_rot,
            s_vib_jmolK=s_vib,
        )

        # Gibbs correction (H - T*S).
        s_total = s_trans + s_rot + s_vib
        thermo.g_correction_kjmol = thermo.h_total_kjmol - T * s_total * 1e-3

        return thermo

    def _truhlar_entropy(self, nu_cm1: float, T: float, s_harmonic: float) -> float:
        """
        Entropy with low-frequency capping in the harmonic formula.

        Args:
            nu_cm1: Mode frequency (cm⁻¹).
            T: Temperature (K).
            s_harmonic: Harmonic entropy of the mode (J/mol/K).

        Returns:
            float: Entropy of the mode after low-frequency capping (J/mol/K).
        """
        threshold = self.omega0_cm1
        nu_effective = max(nu_cm1, threshold)
        theta_eff = (H * C * (nu_effective / CM_TO_M)) / K_B
        x_eff = theta_eff / max(T, 1e-12)
        return R_GAS * (x_eff / (np.exp(x_eff) - 1 + 1e-300) - np.log(1 - np.exp(-x_eff) + 1e-300))

    def _grimme_entropy(self, nu_cm1: float, T: float, s_harmonic: float) -> float:
        """
        Entropy interpolation between harmonic oscillator and free-rotor.

        Args:
            nu_cm1: Mode frequency (cm⁻¹).
            T: Temperature (K).
            s_harmonic: Harmonic entropy of the mode (J/mol/K).

        Returns:
            float: Interpolated entropy (J/mol/K).
        """
        w = self._weight_w(nu_cm1)
        s_rotor = self._rotor_entropy(T, nu_cm1)
        return w * s_harmonic + (1.0 - w) * s_rotor

    def _weight_w(self, nu_cm1: float) -> float:
        """
        Damping weight for entropy interpolation (alpha=4 by construction here).

        Args:
            nu_cm1: Mode frequency (cm⁻¹).

        Returns:
            float: Weight w in [0, 1].
        """
        nu = max(float(nu_cm1), 1e-12)
        w0 = max(self.omega0_cm1, 1e-12)
        return 1.0 / (1.0 + (w0 / nu) ** 4)

    def _rotor_entropy(self, T: float, nu_cm1: float) -> float:
        """
        Reference rotor-based entropy used in interpolation.

        Args:
            T: Temperature (K).
            nu_cm1: Mode frequency (cm⁻¹).

        Returns:
            float: A smoothed rotor-reference entropy (J/mol/K).
        """
        nu = max(float(nu_cm1), 1e-12)
        theta_eff = (H * C * (nu / CM_TO_M)) / K_B
        x = theta_eff / max(T, 1e-12)
        s_harm = R_GAS * (x / (np.exp(x) - 1 + 1e-300) - np.log(1 - np.exp(-x) + 1e-300))
        s_ref = R_GAS * np.log(1.0 + (8.0 * np.pi**2 * K_B * T * 1e-44) / (H**2))
        return max(0.0, s_harm + s_ref)

    # =========================
    # New helper methods (used by ilowfreq==3)
    # =========================
    def _headgordon_weight(self, nu_cm1, omega0_cm1=100.0, alpha=4):
        """
        Smooth damping weight: w = 1 / (1 + (omega0/nu)^alpha).

        Args:
            nu_cm1: Scalar or array-like of frequencies (cm⁻¹).
            omega0_cm1: Characteristic frequency (cm⁻¹).
            alpha: Damping exponent (dimensionless).

        Returns:
            np.ndarray or float: Weight(s) w in [0, 1] matching the input shape.
        """
        nu = np.asarray(nu_cm1, dtype=float)
        nu = np.where(nu <= 0.0, 1e-12, nu)
        return 1.0 / (1.0 + (omega0_cm1 / nu) ** float(alpha))

    def _free_rotor_entropy(self, nu_cm1, T, Bav_amuA2=None):
        """
        Free-rotor entropy per mode for interpolation (J/mol/K).

        Args:
            nu_cm1: Scalar or array-like frequencies (cm⁻¹).
            T: Temperature (K).
            Bav_amuA2: Optional average moment of inertia in amu·Å² for an
                effective reduced inertia; if None, uses a direct mapping.

        Returns:
            np.ndarray or float: Free-rotor entropy value(s) (J/mol/K).
        """
        R = 8.314462618
        h = 6.62607015e-34
        kB = 1.380649e-23
        c = 2.99792458e10  # cm/s

        nu_hz = np.asarray(nu_cm1, dtype=float) * c
        pi = np.pi
        mu = h / (8.0 * pi * pi * np.where(nu_hz <= 0.0, 1e-12, nu_hz))

        if Bav_amuA2 is not None:
            Bav = Bav_amuA2 * 1.66053906660e-27 * 1e-20
            mu_eff = (mu * Bav) / (mu + Bav)
        else:
            mu_eff = mu

        term = ((8.0 * (pi**3) * mu_eff * kB * T) / (h * h)) ** 0.5
        return R * (0.5 + np.log(term))

    def _trans_rot_entropy(self) -> Tuple[float, float]:
        """
        Translational and rotational entropy for a nonlinear/linear molecule.

        Returns:
            Tuple[float, float]: (S_trans, S_rot) in J/mol/K.
        """
        T, P_kpa = self.temperature, self.pressure_kpa
        P = P_kpa * 1000.0  # convert kPa -> Pa for ideal-gas formulas

        I_SI, m_tot = self._principal_moments()
        sigma = max(self.symmetry_number, 1)

        # Translation (ideal gas).
        q_trans = ((2 * np.pi * m_tot * K_B * T) / (H**2)) ** 1.5 * (K_B * T / P)
        s_trans = R_GAS * (np.log(q_trans) + 2.5)

        # Rotation (linear vs nonlinear).
        is_linear = self._is_linear_molecule()
        if is_linear:
            I = np.max(I_SI)
            q_rot = (8 * np.pi**2 * I * K_B * T) / (sigma * H**2)
            s_rot = R_GAS * (np.log(q_rot) + 1)
        else:
            q_rot = (np.sqrt(np.pi) / sigma) * ((8 * np.pi**2 * K_B * T) / (H**2)) ** 1.5 * np.sqrt(np.prod(I_SI))
            s_rot = R_GAS * (np.log(q_rot) + 1.5)

        return s_trans, s_rot

    def _principal_moments(self) -> Tuple[np.ndarray, float]:
        """
        Principal moments of inertia (SI) and total mass (kg).

        Returns:
            Tuple[np.ndarray, float]: (I_x,I_y,I_z) in kg·m² and total mass (kg).
        """
        I_amuA2 = np.asarray(self.atoms.get_moments_of_inertia(vectors=False))
        I_SI = I_amuA2 * AMU * ANG2_TO_M2
        m_tot = np.sum(self.atoms.get_masses()) * AMU
        return I_SI, m_tot

    def _is_linear_molecule(self, tol: float = 1e-2) -> bool:
        """
        Heuristic linearity check from principal moments.

        Args:
            tol: Threshold ratio I_min / I_max below which the molecule is treated as linear.

        Returns:
            bool: True if linear, False otherwise.
        """
        try:
            I = self.atoms.get_moments_of_inertia(vectors=False)
            I = np.sort(np.asarray(I))
            return I[0] / max(I[-1], 1e-12) < tol
        except Exception:
            return len(self.atoms) == 2

    # ---------------------- writers ----------------------
    def _write_output(self, freqs: np.ndarray, modes: np.ndarray, thermo: ThermoResults) -> None:
        """
        Write the full analysis report to the output file.

        Args:
            freqs: Vibrational frequencies (cm⁻¹).
            modes: Normal modes in Cartesian components.
            thermo: ThermoResults instance with aggregated properties.
        """
        with open(self.output, 'w', encoding='utf-8') as f:
            self._write_header(f)
            self._write_thermochemistry(f, thermo)
            self._write_frequencies(f, freqs)
            self._write_normal_modes(f, freqs, modes)

    def _write_header(self, f) -> None:
        """
        Print the job header and coordinates section.

        Args:
            f: An opened text IO handle.
        """
        f.write("Preprocessing the settings ...\n")
        f.write("Model: ANI-1ccx.\n")
        f.write("Job type: freq.\n")
        f.write("GPU ID is not set. Using CPU to calculate.\n\n")
        f.write("                             Coordinates                              \n")
        f.write("**********************************************************************\n")

        # Atomic coordinates.
        symbols = self.atoms.get_chemical_symbols()
        positions = self.atoms.get_positions()
        for i, (symbol, pos) in enumerate(zip(symbols, positions), 1):
            f.write(f"{i:2d}    {symbol:2s}               {pos[0]:8.4f}              {pos[1]:8.4f}               {pos[2]:8.4f}\n")

        f.write("\nLoading the Machine Learning Potential Model...\n")
        f.write("Loading ANI model (ANI-1ccx) successfully.\n")
        f.write("Using CPU for calculation.\n\n")

    def _write_thermochemistry(self, f, thermo: ThermoResults) -> None:
        """
        Print thermochemistry summary block.
        —— Main unit: kcal/mol (or /K); when verbosity>=2, also print Hartree. ——

        Args:
            f: An opened text IO handle.
            thermo: ThermoResults with enthalpies, entropies, and free-energy.
        """
        T = self.temperature

        # Convert to kcal/mol and Hartree (conditionally).
        zpe_kcal = Units.kj_to_kcal(thermo.zpe_kjmol)
        h_trans_kcal = Units.kj_to_kcal(thermo.h_trans_kjmol)
        h_rot_kcal = Units.kj_to_kcal(thermo.h_rot_kjmol)
        h_vib_kcal = Units.kj_to_kcal(thermo.h_vib_thermal_kjmol)
        h_total_kcal = Units.kj_to_kcal(thermo.h_total_kjmol)
        g_corr_kcal = Units.kj_to_kcal(thermo.g_correction_kjmol)

        # Entropy to kcal/mol·K
        s_trans_kcalK = thermo.s_trans_jmolK * 1e-3 * KJ_PER_MOL_TO_KCAL
        s_rot_kcalK   = thermo.s_rot_jmolK   * 1e-3 * KJ_PER_MOL_TO_KCAL
        s_vib_kcalK   = thermo.s_vib_jmolK   * 1e-3 * KJ_PER_MOL_TO_KCAL
        s_total_kcalK = s_trans_kcalK + s_rot_kcalK + s_vib_kcalK

        # Provide Hartree as a secondary unit when desired.
        if self.verbosity >= 2:
            zpe_ha = Units.kj_to_ha(thermo.zpe_kjmol)
            h_trans_ha = Units.kj_to_ha(thermo.h_trans_kjmol)
            h_rot_ha = Units.kj_to_ha(thermo.h_rot_kjmol)
            h_vib_ha = Units.kj_to_ha(thermo.h_vib_thermal_kjmol)
            h_total_ha = Units.kj_to_ha(thermo.h_total_kjmol)
            g_corr_ha = Units.kj_to_ha(thermo.g_correction_kjmol)

        f.write("Thermochemistry Summary\n")
        f.write("="*60 + "\n")
        f.write(f"Working pressure: {self.pressure_kpa:.3f} kPa\n")

        # Main outputs in kcal/mol
        f.write(f"Zero-point vibrational energy:        {zpe_kcal:12.6f} kcal/mol\n")
        f.write(f"Translational enthalpy at {T:.0f} K:    {h_trans_kcal:12.6f} kcal/mol\n")
        f.write(f"Rotational enthalpy at {T:.0f} K:       {h_rot_kcal:12.6f} kcal/mol\n")
        f.write(f"Vibrational thermal enthalpy at {T:.0f} K: {h_vib_kcal:12.6f} kcal/mol\n")
        f.write(f"Total enthalpy correction at {T:.0f} K:  {h_total_kcal:12.6f} kcal/mol\n")
        f.write("\n")
        f.write(f"Gibbs free energy correction at {T:.0f} K: {g_corr_kcal:12.6f} kcal/mol\n")
        f.write("\n")

        # Secondary outputs in Hartree when verbosity>=2
        if self.verbosity >= 2:
            f.write(" (Hartree)\n")
            f.write(f"   ZPE:       {zpe_ha:12.8f} Ha   |  H_trans: {h_trans_ha:12.8f}  H_rot: {h_rot_ha:12.8f}  H_vib: {h_vib_ha:12.8f}  H_total: {h_total_ha:12.8f}\n")
            f.write(f"   G_corr:    {g_corr_ha:12.8f} Ha\n\n")

        f.write("Entropy components (kcal/mol·K):\n")
        f.write(f"  S_trans: {s_trans_kcalK:10.6f}\n")
        f.write(f"  S_rot:   {s_rot_kcalK:10.6f}\n")
        f.write(f"  S_vib:   {s_vib_kcalK:10.6f}\n")
        f.write(f"  S_total: {s_total_kcalK:10.6f}\n")
        f.write("="*60 + "\n\n")

    def _write_frequencies(self, f, freqs: np.ndarray) -> None:
        """
        Print an aligned table of vibrational frequencies.

        Rules:
            verbosity = 0/1 -> print first 10 (sorted ascending)
            verbosity >= 2  -> print first 100 (sorted ascending; cap by available)
        """
        f.write("Vibrational Frequencies (cm⁻¹)\n")
        f.write("-" * 60 + "\n")

        n_freqs = len(freqs)
        n_imag = int(np.sum(freqs < 0))
        n_real = n_freqs - n_imag
        f.write(f"Imaginary frequencies: {n_imag:3d}\n")
        f.write(f"Real frequencies:     {n_real:3d}\n")
        f.write("-" * 60 + "\n")

        # Sort ascending (negative first for diagnostics)
        arr = np.sort(freqs.copy())

        v = int(getattr(self, "verbosity", 1))
        limit = 100 if v >= 2 else 10
        n_to_print = min(limit, len(arr))
        arr = arr[:n_to_print]

        # 6 columns; imaginary marked with 'i'
        for i in range(0, n_to_print, 6):
            line = ""
            for j in range(6):
                idx = i + j
                if idx < n_to_print:
                    vval = arr[idx]
                    tag = "i" if vval < 0 else " "
                    line += f"{vval:10.2f}{tag}  "
            f.write(line + "\n")
        f.write("\n")
        
    def _write_normal_modes(self, f, freqs: np.ndarray, modes: np.ndarray) -> None:
        """
        Print a compact listing of normal modes (Cartesian components).

        Rules:
            verbosity == 0: print nothing
            verbosity == 1: print 10 modes (ascending by frequency)
            verbosity >= 2: print 100 modes (ascending; cap by available)
        """
        # 1) Coarse verbosity gating
        v = int(getattr(self, "verbosity", 1))
        if v <= 0:
            return

        # 2) How many to print?
        total_modes = int(modes.shape[0])
        n_to_print = min(100 if v >= 2 else 10, total_modes)
        if n_to_print <= 0:
            return

        # 3) Pick modes by ascending frequency (imag first helps debugging)
        order = np.argsort(freqs[:total_modes])
        sel_idx = order[:n_to_print]

        # 4) Print
        f.write("Normal Modes (Cartesian displacements)\n")
        f.write("=" * 60 + "\n")

        n_atoms = len(self.atoms)
        for rank, mode_idx in enumerate(sel_idx, start=1):
            f.write(f"Frequency {mode_idx + 1:3d}: {freqs[mode_idx]:10.2f} cm⁻¹\n")
            f.write("Eigenvectors       X          Y          Z\n")

            mode_vector = modes[mode_idx].reshape(n_atoms, 3)
            for atom_idx in range(n_atoms):
                dx, dy, dz = mode_vector[atom_idx]
                f.write(f"Atom {atom_idx + 1:3d}:   {dx:10.6f}  {dy:10.6f}  {dz:10.6f}\n")

            f.write("\n")

# ======================================================================
# Concrete implementations
# ======================================================================
class MWFrequency(FrequencyBase):
    """
    Mass-weighted frequency analysis.

    Diagonalizes the mass-weighted Hessian to obtain frequencies and modes.
    """

    def compute_frequencies(self, hessian_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Diagonalize the mass-weighted Hessian to get frequencies and modes.

        Args:
            hessian_matrix: Cartesian Hessian (3N, 3N).

        Returns:
            Tuple[np.ndarray, np.ndarray]: (frequencies_cm1, modes_in_cartesian).
        """
        masses = np.asarray(self.atoms.get_masses())
        inv_sqrt_m = np.repeat(1.0 / np.sqrt(masses), 3)

        h_m = hessian_matrix * inv_sqrt_m[None, :] * inv_sqrt_m[:, None]
        evals, evecs_mw = np.linalg.eigh(h_m)

        # Convert eigenvalues to signed frequencies in cm⁻¹.
        conversion = 5140.484
        freqs_cm1 = np.sign(evals) * np.sqrt(np.abs(evals)) * conversion

        # Transform modes back to Cartesian coordinates.
        modes_cart = evecs_mw.T * inv_sqrt_m[None, :]

        return freqs_cm1, modes_cart

class NonMWFrequency(FrequencyBase):
    """
    Non-mass-weighted frequency analysis.

    Directly diagonalizes the Cartesian Hessian.
    """

    def compute_frequencies(self, hessian_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Diagonalize the Cartesian Hessian to get frequencies and modes.

        Args:
            hessian_matrix: Cartesian Hessian (3N, 3N).

        Returns:
            Tuple[np.ndarray, np.ndarray]: (frequencies_cm1, modes_in_cartesian).
        """
        evals, evecs = np.linalg.eigh(hessian_matrix)

        conversion = 5140.484
        freqs_cm1 = np.sign(evals) * np.sqrt(np.abs(evals)) * conversion

        modes_cart = evecs.T
        return freqs_cm1, modes_cart

class BothFrequency(FrequencyBase):
    """
    Dual-mode frequency analysis.

    Produces both mass-weighted and non-mass-weighted results in one run.
    """

    def run(self) -> None:
        """
        Execute both MW and non-MW frequency analyses and write combined output.
        """
        hessian = self.get_hessian()

        # Mass-weighted branch.
        mw_freqs, mw_modes = self._compute_mw(hessian)
        if self.treat_imag_as_real:
            tol = float(getattr(self._print, "imag_tol_cm1", 10.0))
            mw_freqs = np.where(mw_freqs < -tol, mw_freqs, np.abs(mw_freqs))
        mw_thermo = self.compute_thermo(mw_freqs)

        # Non-mass-weighted branch.
        nonmw_freqs, nonmw_modes = self._compute_nonmw(hessian)
        if self.treat_imag_as_real:
            tol = float(getattr(self._print, "imag_tol_cm1", 10.0))
            nonmw_freqs = np.where(nonmw_freqs < -tol, nonmw_freqs, np.abs(nonmw_freqs))
        nonmw_thermo = self.compute_thermo(nonmw_freqs)

        # Write combined report.
        self._write_combined_output(mw_freqs, mw_modes, mw_thermo,
                                    nonmw_freqs, nonmw_modes, nonmw_thermo)

    def _compute_mw(self, hessian: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build and diagonalize the mass-weighted Hessian: H_m = M^{-1/2} H M^{-1/2}.

        Args:
            hessian: Cartesian Hessian (3N, 3N).

        Returns:
            Tuple[np.ndarray, np.ndarray]: (frequencies_cm1, modes_in_cartesian).
        """
        masses = np.asarray(self.atoms.get_masses())
        inv_sqrt_m = np.repeat(1.0 / np.sqrt(masses), 3)
        h_m = hessian * inv_sqrt_m[None, :] * inv_sqrt_m[:, None]
        evals, evecs_mw = np.linalg.eigh(h_m)

        conversion = 5140.484
        freqs_cm1 = np.sign(evals) * np.sqrt(np.abs(evals)) * conversion
        modes_cart = evecs_mw.T * inv_sqrt_m[None, :]

        return freqs_cm1, modes_cart

    def _compute_nonmw(self, hessian: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Diagonalize the Cartesian Hessian without mass-weighting.

        Args:
            hessian: Cartesian Hessian (3N, 3N).

        Returns:
            Tuple[np.ndarray, np.ndarray]: (frequencies_cm1, modes_in_cartesian).
        """
        evals, evecs = np.linalg.eigh(hessian)

        conversion = 5140.484
        freqs_cm1 = np.sign(evals) * np.sqrt(np.abs(evals)) * conversion
        modes_cart = evecs.T

        return freqs_cm1, modes_cart

    def _write_combined_output(self, mw_freqs: np.ndarray, mw_modes: np.ndarray,
                               mw_thermo: ThermoResults, nonmw_freqs: np.ndarray,
                               nonmw_modes: np.ndarray, nonmw_thermo: ThermoResults) -> None:
        """
        Write a combined report for MW and non-MW analyses.

        Args:
            mw_freqs: Mass-weighted frequencies (cm⁻¹).
            mw_modes: Mass-weighted normal modes (Cartesian).
            mw_thermo: Thermochemistry from MW frequencies.
            nonmw_freqs: Non-mass-weighted frequencies (cm⁻¹).
            nonmw_modes: Non-mass-weighted normal modes (Cartesian).
            nonmw_thermo: Thermochemistry from non-MW frequencies.
        """
        with open(self.output, 'w', encoding='utf-8') as f:
            # Common header
            self._write_header(f)

            # MW section
            f.write("\n" + "="*60 + "\n")
            f.write("MASS-WEIGHTED ANALYSIS\n")
            f.write("="*60 + "\n")
            self._write_thermochemistry(f, mw_thermo)
            self._write_frequencies(f, mw_freqs)
            self._write_normal_modes(f, mw_freqs, mw_modes)

            # Non-MW section
            f.write("\n" + "="*60 + "\n")
            f.write("NON-MASS-WEIGHTED ANALYSIS\n")
            f.write("="*60 + "\n")
            self._write_thermochemistry(f, nonmw_thermo)
            self._write_frequencies(f, nonmw_freqs)
            self._write_normal_modes(f, nonmw_freqs, nonmw_modes)

# ======================================================================
# Front driver 
# ======================================================================
class FrequencyDriver:
    """
    Public entry point that wires together a frequency and thermochemistry run.
    - method: "mw" | "nonmw" | "both"
    - treat_imag_as_real / verbosity / print truncations are injected here.
    """

    def __init__(self,
                 output: str,
                 atoms: Atoms,
                 params: Optional[FrequencyParams] = None,
                 paras: Optional[dict] = None):
        self.output = output
        self.atoms = atoms

        # 1) Merge user-level params (dataclass is the single source of truth)
        self.params = params if params is not None else FrequencyParams()
        if isinstance(paras, dict):
            user = _select_subdict(paras, ("freq", "frequency", "frequency_analysis"))
            _update_dataclass_from_dict(self.params, user)

        # Back-compat: map legacy 'mode' -> 'method' if present
        if hasattr(self.params, "method"):
            if isinstance(paras, dict):
                user = _select_subdict(paras, ("freq", "frequency", "frequency_analysis"))
                if isinstance(user, dict) and ("mode" in user) and ("method" not in user):
                    self.params.method = str(user["mode"]).lower()

        # ---- Backward compatibility: pressure_pa -> pressure_kpa ----
        # If legacy scripts provided pressure in Pa, accept and convert to kPa.
        if isinstance(paras, dict):
            low = _lower_keys(_select_subdict(paras, ("freq", "frequency", "frequency_analysis")))
            if "pressure_pa" in low and "pressure_kpa" not in low:
                try:
                    pa_val = float(low["pressure_pa"])
                    self.params.pressure_kpa = pa_val / 1000.0
                except Exception:
                    pass  # leave for later validation

        # 2) Merge print-layer (advanced) parameters
        self.print_params = PrintParams()
        if isinstance(paras, dict):
            user = _select_subdict(paras, ("freq", "frequency", "frequency_analysis"))
            _update_dataclass_from_dict(self.print_params, user)

        # Basic validation
        if self.params.temperature <= 0:
            raise ValueError("temperature must be > 0 K")
        if self.params.pressure_kpa <= 0:
            raise ValueError("pressure_kpa must be > 0 kPa")
        if self.params.method.lower() not in ("mw", "nonmw", "both"):
            raise ValueError('method must be one of: "mw", "nonmw", "both"')

    def run(self) -> None:
        """Instantiate the concrete job and execute it end-to-end."""
        method = self.params.method.lower()

        # Shared constructor kwargs
        common_kwargs = dict(
            output=self.output,
            atoms=self.atoms,
            temperature=self.params.temperature,
            pressure_kpa=self.params.pressure_kpa,
            ilowfreq=self.params.ilowfreq
        )

        if method == "both":
            job = BothFrequency(**common_kwargs)
        elif method == "nonmw":
            job = NonMWFrequency(**common_kwargs)
        else:  # "mw"
            job = MWFrequency(**common_kwargs)

        # Inject user toggles
        job.verbosity = int(self.params.verbosity)
        job.treat_imag_as_real = bool(self.params.treat_imag_as_real)

        # Inject print-layer params
        job._print = self.print_params

        # Execute
        job.run()

# Public API
__all__ = [
    'FrequencyBase', 'MWFrequency', 'NonMWFrequency', 'BothFrequency', 'ThermoResults',
    'FrequencyParams', 'PrintParams', 'FrequencyDriver'
]
