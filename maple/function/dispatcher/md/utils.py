"""
Utility functions for molecular dynamics simulations.

This module provides essential calculations for MD:
- Temperature from velocities
- Kinetic energy calculations
- Velocity initialization from Maxwell-Boltzmann distribution
- Instantaneous pressure calculation
- XYZ trajectory writing utilities
"""

import warnings

import numpy as np
from ase import Atoms
from ase.calculators.calculator import PropertyNotImplementedError
from typing import Optional, Tuple


# ========== Physical Constants and Unit Conversions ==========

# Temperature conversions
KELVIN_TO_HARTREE = 3.1668114e-6  # k_B in Hartree/K
HARTREE_TO_KELVIN = 1.0 / KELVIN_TO_HARTREE

# Mass conversions
AMU_TO_AU = 1822.888486209  # atomic mass unit to atomic units

# Time conversions
FS_TO_AU = 41.341374575751  # femtoseconds to atomic units
AU_TO_FS = 1.0 / FS_TO_AU

# Length conversions
# NIST CODATA 2018: 1 Bohr = 0.529177210903 Å (exact to 12 sig. fig.)
BOHR_TO_ANGSTROM = 0.529177210903
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM

# Force conversions
# MAPLE calculators (AIMNet2, MACE, UMA) return forces in Ha/Å.
# The MD integrator needs Ha/Bohr (atomic units).
# Ha/Å → Ha/Bohr: multiply by Å/Bohr = ANGSTROM_TO_BOHR ≈ 1.8897
HA_PER_ANG_TO_AU = ANGSTROM_TO_BOHR  # Ha/Å → Ha/Bohr

# Legacy alias kept for backward compatibility (was used when forces were assumed eV/Å)
EV_PER_ANG_TO_AU = 1.0 / (27.211386245988 * BOHR_TO_ANGSTROM)  # ≈ 0.019447

# Pressure unit conversions
# Derivation: 1 eV = 1.6021766208e-19 J, 1 Å³ = 1e-30 m³ → 1 eV/Å³ = 1.6021766208e11 Pa = 1.6021766208e6 bar
EV_PER_ANG3_TO_BAR = 1.6021766208e-19 / 1e-30 * 1e-5   # eV/Å³ → bar
BAR_TO_EV_PER_ANG3 = 1.0 / EV_PER_ANG3_TO_BAR          # bar → eV/Å³

# Kinetic energy conversion for pressure calculation
# 1 amu·Å²/fs² = 1.66054e-27 kg × 1e-20 m² / 1e-30 s² = 1.66054e-17 J
# → 1.66054e-17 / 1.60218e-19 eV ≈ 103.6427 eV
AMU_ANG2_PER_FS2_TO_EV = 1.03642695e+2

# Default isothermal compressibility (liquid water at 300 K, 1 bar)
# Reference: CRC Handbook of Chemistry and Physics
DEFAULT_COMPRESSIBILITY = 4.5e-5   # 1/bar


# ========== Core MD Calculations ==========

def calculate_temperature(atoms: Atoms, velocities: np.ndarray) -> float:
    """
    Calculate instantaneous temperature from velocities.

    Uses the equipartition theorem:
        T = 2 * KE / (N_dof * k_B)

    where N_dof = 3N - 3 for isolated molecules (removing COM translation),
          or 3N for periodic systems (no overall translation constraint)

    Parameters
    ----------
    atoms : ase.Atoms
        Atomic system
    velocities : np.ndarray
        Atomic velocities in atomic units (Bohr/a.u. time)
        Shape: (N_atoms, 3)

    Returns
    -------
    float
        Temperature in Kelvin
    """
    masses = atoms.get_masses() * AMU_TO_AU  # Convert to atomic units
    kinetic = 0.5 * np.sum(masses[:, np.newaxis] * velocities**2)

    n_atoms = len(atoms)
    # Periodic systems have no overall translation; isolated molecules lose 3 COM DOF.
    # Allen & Tildesley, Computer Simulation of Liquids, 2nd ed., §3.3
    if any(atoms.pbc):
        n_dof = 3 * n_atoms
    else:
        n_dof = 3 * n_atoms - 3

    if n_dof <= 0:
        return 0.0

    temperature = 2.0 * kinetic / (n_dof * KELVIN_TO_HARTREE)
    return temperature


def calculate_kinetic_energy(atoms: Atoms, velocities: np.ndarray) -> float:
    """
    Calculate total kinetic energy.

    KE = 0.5 * sum(m_i * v_i^2)

    Parameters
    ----------
    atoms : ase.Atoms
        Atomic system
    velocities : np.ndarray
        Atomic velocities in atomic units
        Shape: (N_atoms, 3)

    Returns
    -------
    float
        Kinetic energy in Hartree
    """
    masses = atoms.get_masses() * AMU_TO_AU
    kinetic = 0.5 * np.sum(masses[:, np.newaxis] * velocities**2)
    return kinetic


def compute_instantaneous_pressure(
    atoms: Atoms,
    velocities: np.ndarray,
    stress_warned: bool = False,
    class_name: str = "Barostat",
) -> Tuple[float, bool]:
    """
    Compute instantaneous pressure from the virial theorem.

    P = (2*KE + W) / (3*V)

    where W = -V * (σ_xx + σ_yy + σ_zz) is the virial from the stress tensor.

    Parameters
    ----------
    atoms : ase.Atoms
        Atomic system with attached calculator.
    velocities : np.ndarray
        Current velocities in atomic units (Bohr/a.u. time), shape (N_atoms, 3).
    stress_warned : bool, default=False
        Flag indicating whether stress-unavailable warning has already been issued.
        If False and stress is unavailable, a warning is emitted and the flag is
        set to True in the return value.
    class_name : str, default="Barostat"
        Class name to include in warning message.

    Returns
    -------
    tuple of (float, bool)
        (pressure_in_bar, new_stress_warned_flag)
        pressure_in_bar: Instantaneous pressure in bar.
        new_stress_warned_flag: Updated warning flag (True if warning was issued).

    Notes
    -----
    If the calculator does not support stress tensor, the ideal-gas approximation
    (W=0) is used, which underestimates pressure for dense systems.

    References
    ----------
    Allen & Tildesley, Computer Simulation of Liquids, 2nd ed. (2017), §3.3.
    """
    volume = atoms.get_volume()   # Å³

    # Kinetic contribution (in eV)
    masses_amu = atoms.get_masses()
    # v in a.u. (Bohr/a.u.time) → convert to Å/fs
    v_ang_per_fs = velocities * BOHR_TO_ANGSTROM / AU_TO_FS
    # KE in eV: 0.5 * m[amu] * v²[Å²/fs²] * (amu·Å²/fs² → eV)
    ke_ev = 0.5 * np.sum(masses_amu[:, np.newaxis] * v_ang_per_fs**2) * AMU_ANG2_PER_FS2_TO_EV

    # Virial contribution from stress tensor (eV)
    virial_ev = 0.0
    new_stress_warned = stress_warned
    try:
        stress = atoms.get_stress(voigt=True)   # eV/Å³, Voigt: xx,yy,zz,yz,xz,xy
        # Hydrostatic virial: W = -V * (σ_xx + σ_yy + σ_zz)
        virial_ev = -volume * (stress[0] + stress[1] + stress[2])
    except (PropertyNotImplementedError, RuntimeError):
        # Calculator does not support stress; fall back to ideal-gas pressure (virial = 0).
        # Warn once per barostat instance so the user is aware.
        if not stress_warned:
            warnings.warn(
                f"{class_name}: calculator does not provide a stress tensor; "
                "pressure estimated from kinetic term only (ideal-gas approximation). "
                "For accurate NPT simulations, use a calculator that supports stress.",
                UserWarning, stacklevel=2
            )
            new_stress_warned = True

    # P = (2*KE + W) / (3*V)  in eV/Å³, then convert to bar
    pressure_ev_ang3 = (2.0 * ke_ev + virial_ev) / (3.0 * volume)
    return pressure_ev_ang3 * EV_PER_ANG3_TO_BAR, new_stress_warned


def initialize_velocities(
    atoms: Atoms,
    temperature: float,
    remove_com: bool = True,
    remove_rotation: bool = False,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Initialize velocities from Maxwell-Boltzmann distribution.

    For each atom i with mass m_i at temperature T:
        v_i ~ N(0, sqrt(k_B * T / m_i))

    Parameters
    ----------
    atoms : ase.Atoms
        Atomic system
    temperature : float
        Target temperature in Kelvin
    remove_com : bool, default=True
        Remove center of mass translational motion (3 DOF).
        Ref: Allen & Tildesley, Computer Simulation of Liquids,
             2nd ed. (2017), §3.2
    remove_rotation : bool, default=False
        Remove overall rigid-body rotational motion (up to 3 DOF).
        Only meaningful for non-periodic isolated molecules.
        For periodic systems this parameter is ignored.

        Scientific rationale
        --------------------
        A Maxwell-Boltzmann draw generically yields a non-zero net angular
        momentum L = Σ_i r_i × (m_i v_i).  For an isolated molecule in NVE,
        L is a conserved quantity, so any initial L causes the entire molecule
        to rotate as a rigid body throughout the simulation, obscuring internal
        dynamics in visualisation.

        The standard remedy is to project out the three rotational DOF from the
        velocities immediately after COM removal.  The projection is exact for a
        rigid body and removes only the infinitesimal rigid-rotation component
        from the velocity field; internal (vibrational) DOF are unaffected.

        After projection, velocities are rescaled to restore the target
        temperature, accounting for the reduced DOF count (3N − 6 for a
        non-linear molecule, analogous to the COM correction).

        This procedure is the default in several major MD codes:
          • GROMACS: `comm-mode = Angular` removes both translation and
            rotation; recommended for isolated molecules in vacuum.
            Ref: GROMACS Reference Manual 2024, §3.4.1 "Removing COM motion"
          • AMBER: `nscm` option; rotation removal is standard for gas-phase
            peptide simulations.
            Ref: Case et al. (2023) AMBER 2023 Reference Manual, §3.1
          • NAMD: `zeroMomentum yes` + angular momentum zeroing described in
            the User's Guide §2.6 for vacuum simulations.
          • LAMMPS: `fix momentum ... angular` explicitly zeroes angular
            momentum.
            Ref: LAMMPS documentation, fix momentum command.

        Theoretical basis: the projection is equivalent to constraining the
        three rigid-rotation modes of the molecule, reducing the effective DOF
        from 3N−3 to 3N−6 (non-linear) or 3N−5 (linear).  The equipartition
        theorem still holds for the remaining DOF after rescaling.
        Ref: Shirts (2013) J. Chem. Theory Comput. 9, 909, §2 "Removing rigid
             body motion"; Eastman & Pande (2010) J. Chem. Theory Comput. 6,
             434, §2.

        Limitation: for periodic (PBC) systems there is no well-defined
        rigid-body rotation of the whole cell, so this flag is silently ignored
        when any(atoms.pbc) is True.
    rng : np.random.Generator, optional
        Random number generator (for reproducibility)

    Returns
    -------
    np.ndarray
        Velocities in atomic units
        Shape: (N_atoms, 3)
    """
    if rng is None:
        rng = np.random.default_rng()

    kT = temperature * KELVIN_TO_HARTREE
    masses = atoms.get_masses() * AMU_TO_AU
    n_atoms = len(atoms)

    # Generate random velocities from standard normal distribution
    velocities = rng.standard_normal(size=(n_atoms, 3))

    # Scale each atom's velocity by sqrt(kT/m)
    for i, mass in enumerate(masses):
        sigma = np.sqrt(kT / mass)
        velocities[i] *= sigma

    # Remove center of mass motion, then rescale to restore target temperature.
    # COM removal reduces the number of active DOF by 3, which lowers the
    # instantaneous kinetic energy below the target; rescaling corrects this.
    # Ref: Allen & Tildesley, Computer Simulation of Liquids, 2nd ed. (2017), §3.2
    if remove_com:
        total_momentum = np.sum(masses[:, np.newaxis] * velocities, axis=0)
        total_mass = np.sum(masses)
        velocities -= total_momentum / total_mass

    # Remove overall rigid-body rotation (non-PBC only).
    #
    # Method (Shirts 2013, §2):
    #   1. Compute angular momentum L = Σ r_i × (m_i v_i)  in the COM frame.
    #   2. Compute the inertia tensor I = Σ m_i (|r_i|² E − r_i ⊗ r_i).
    #   3. Solve ω = I⁻¹ L  for the rigid-body angular velocity.
    #   4. Subtract the rigid-rotation contribution: v_i -= ω × r_i.
    #
    # This is a linear projection onto the subspace orthogonal to the three
    # infinitesimal rotation generators; internal DOF are exactly preserved.
    if remove_rotation and not any(atoms.pbc):
        positions_au = atoms.get_positions() * ANGSTROM_TO_BOHR  # Å → Bohr

        # Step 1 — COM frame positions
        total_mass = np.sum(masses)
        com = np.sum(masses[:, np.newaxis] * positions_au, axis=0) / total_mass
        r = positions_au - com  # (N, 3)

        # Step 2 — Angular momentum
        L = np.sum(
            masses[:, np.newaxis] * np.cross(r, velocities),
            axis=0
        )  # (3,)

        # Step 3 — Inertia tensor
        I = np.zeros((3, 3))
        for mi, ri in zip(masses, r):
            I += mi * (np.dot(ri, ri) * np.eye(3) - np.outer(ri, ri))

        # Step 4 — Solve for ω; use pseudoinverse to handle near-singular I
        # (e.g. linear molecules where one principal moment is ~0)
        try:
            omega = np.linalg.solve(I, L)
        except np.linalg.LinAlgError:
            omega = np.linalg.lstsq(I, L, rcond=None)[0]

        # Step 5 — Subtract rigid rotation from each atom
        velocities -= np.cross(omega, r)  # v_i -= ω × r_i

    # Rescale to exact target temperature via velocity scaling.
    #
    # The DOF count used here must match calculate_temperature(), which uses
    # 3N − 3 for non-PBC systems (COM translation removed) and 3N for PBC.
    # Rotational constraints are NOT subtracted here: the equipartition
    # temperature estimator calculate_temperature() is unaware of them, so
    # keeping consistent DOF counts ensures the reported initial temperature
    # matches the target.
    # Ref: Allen & Tildesley (2017) §3.2 (temperature from kinetic energy, DOF counting).
    is_pbc = any(atoms.pbc)
    n_dof = 3 * n_atoms - (0 if is_pbc else 3)
    current_ke2 = np.sum(masses[:, np.newaxis] * velocities**2)  # 2*KE
    if n_dof > 0 and current_ke2 > 0:
        actual_temp = current_ke2 / (n_dof * KELVIN_TO_HARTREE)
        velocities *= np.sqrt(temperature / actual_temp)

    return velocities


# ========== Trajectory I/O ==========

def write_xyz_frame(
    file_handle,
    atoms: Atoms,
    energy: float,
    frame_number: int,
    velocity: Optional[np.ndarray] = None,
    rng_state: Optional[str] = None,
):
    """
    Write a single frame to XYZ file.

    Format:
        N_atoms
        Frame <number>  Energy = <energy> Hartree[  Cell = ...][ RNG = <hex>]
        Symbol  x  y  z  [vx  vy  vz]

    Parameters
    ----------
    file_handle : file object
        Opened file handle
    atoms : ase.Atoms
        Atomic system
    energy : float
        Total energy in Hartree
    frame_number : int
        Frame index
    velocity : np.ndarray, optional
        Velocities to write (in atomic units)
        Shape: (N_atoms, 3)
    rng_state : str, optional
        Hex-encoded RNG state from get_rng_state_hex(). When provided, appended
        to the comment line as ``  RNG = <hex>`` for deterministic resume.
    """
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()

    # Header lines
    file_handle.write(f"{len(symbols)}\n")
    cell_str = ""
    if any(atoms.pbc):
        cp = atoms.cell.cellpar()  # [a, b, c, alpha, beta, gamma]
        cell_str = (f"  Cell = {cp[0]:.6f} {cp[1]:.6f} {cp[2]:.6f}"
                    f" {cp[3]:.6f} {cp[4]:.6f} {cp[5]:.6f}")
    rng_str = f"  RNG = {rng_state}" if rng_state is not None else ""
    # frame_number stores the MD step number (not sequential frame index) so that
    # resume_simulation() can recover the exact step offset without knowing traj_every.
    file_handle.write(f"Frame {frame_number}  Energy = {energy:.10f} Hartree{cell_str}{rng_str}\n")
    # NOTE: frame_number is the MD *step* number (passed as `step` from the ensemble loop).
    # The regex _TRAJ_COMMENT_RE parses this as frame_num; resume_simulation uses it
    # directly as step_offset (no multiplication by traj_every needed).

    # Atomic coordinates (and optionally velocities)
    if velocity is not None:
        for symbol, (x, y, z), (vx, vy, vz) in zip(symbols, positions, velocity):
            file_handle.write(
                f"{symbol:2s} {x:15.8f} {y:15.8f} {z:15.8f}  "
                f"{vx:12.6f} {vy:12.6f} {vz:12.6f}\n"
            )
    else:
        for symbol, (x, y, z) in zip(symbols, positions):
            file_handle.write(f"{symbol:2s} {x:15.8f} {y:15.8f} {z:15.8f}\n")


def write_xyz_trajectory(
    filename: str,
    atoms_list: list,
    energies: list,
    append: bool = False
):
    """
    Write multiple frames to XYZ file.

    Parameters
    ----------
    filename : str
        Output file path
    atoms_list : list of ase.Atoms
        List of atomic configurations
    energies : list of float
        Corresponding energies in Hartree
    append : bool, default=False
        Append to existing file or overwrite
    """
    mode = 'a' if append else 'w'

    with open(filename, mode) as f:
        for i, (atoms, energy) in enumerate(zip(atoms_list, energies)):
            write_xyz_frame(f, atoms, energy, frame_number=i)


# ========== Velocity Utilities ==========

def remove_center_of_mass_motion(atoms: Atoms, velocities: np.ndarray) -> np.ndarray:
    """
    Remove center of mass translational motion.

    Parameters
    ----------
    atoms : ase.Atoms
        Atomic system
    velocities : np.ndarray
        Atomic velocities
        Shape: (N_atoms, 3)

    Returns
    -------
    np.ndarray
        Velocities with COM motion removed
    """
    masses = atoms.get_masses() * AMU_TO_AU
    total_momentum = np.sum(masses[:, np.newaxis] * velocities, axis=0)
    total_mass = np.sum(masses)

    velocities_corrected = velocities - total_momentum / total_mass
    return velocities_corrected


def scale_velocities_to_temperature(
    atoms: Atoms,
    velocities: np.ndarray,
    target_temperature: float
) -> np.ndarray:
    """
    Scale velocities to match target temperature.

    Useful for initialization or re-thermalization.

    Parameters
    ----------
    atoms : ase.Atoms
        Atomic system
    velocities : np.ndarray
        Current velocities
    target_temperature : float
        Target temperature in Kelvin

    Returns
    -------
    np.ndarray
        Scaled velocities
    """
    current_temp = calculate_temperature(atoms, velocities)

    if current_temp < 1e-10:  # Avoid division by zero
        return velocities

    scale_factor = np.sqrt(target_temperature / current_temp)
    return velocities * scale_factor


# ========== Statistics ==========

def calculate_momentum(atoms: Atoms, velocities: np.ndarray) -> np.ndarray:
    """
    Calculate total momentum.

    Parameters
    ----------
    atoms : ase.Atoms
        Atomic system
    velocities : np.ndarray
        Atomic velocities

    Returns
    -------
    np.ndarray
        Total momentum vector (3,)
    """
    masses = atoms.get_masses() * AMU_TO_AU
    total_momentum = np.sum(masses[:, np.newaxis] * velocities, axis=0)
    return total_momentum


def calculate_angular_momentum(
    atoms: Atoms,
    velocities: np.ndarray,
    origin: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Calculate total angular momentum in atomic units.

    L = sum_i (r_i - origin) × (m_i * v_i)

    All quantities are converted to atomic units before computation:
    positions Å → Bohr, masses amu → a.u., velocities already in a.u.

    Parameters
    ----------
    atoms : ase.Atoms
        Atomic system
    velocities : np.ndarray
        Atomic velocities in atomic units (Bohr/a.u. time)
    origin : np.ndarray, optional
        Reference point in Å (default: center of mass).
        Converted to Bohr internally.

    Returns
    -------
    np.ndarray
        Angular momentum vector in atomic units (3,)
    """
    masses = atoms.get_masses() * AMU_TO_AU
    positions = atoms.get_positions() * ANGSTROM_TO_BOHR  # Å → Bohr

    if origin is None:
        # Center of mass in Bohr
        origin = np.sum(masses[:, np.newaxis] * positions, axis=0) / np.sum(masses)
    else:
        origin = np.asarray(origin) * ANGSTROM_TO_BOHR

    angular_momentum = np.zeros(3)
    for mass, pos, vel in zip(masses, positions, velocities):
        r = pos - origin
        p = mass * vel
        angular_momentum += np.cross(r, p)

    return angular_momentum
