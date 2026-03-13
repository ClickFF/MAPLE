"""
Utility functions for molecular dynamics simulations.

This module provides essential calculations for MD:
- Temperature from velocities
- Kinetic energy calculations
- Velocity initialization from Maxwell-Boltzmann distribution
- XYZ trajectory writing utilities
"""

import numpy as np
from ase import Atoms
from typing import Optional


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
BOHR_TO_ANGSTROM = 0.529177249
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


def initialize_velocities(
    atoms: Atoms,
    temperature: float,
    remove_com: bool = True,
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
        Remove center of mass motion
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

    # Remove center of mass motion
    if remove_com:
        total_momentum = np.sum(masses[:, np.newaxis] * velocities, axis=0)
        total_mass = np.sum(masses)
        velocities -= total_momentum / total_mass

    return velocities


# ========== Trajectory I/O ==========

def write_xyz_frame(
    file_handle,
    atoms: Atoms,
    energy: float,
    frame_number: int,
    velocity: Optional[np.ndarray] = None
):
    """
    Write a single frame to XYZ file.

    Format:
        N_atoms
        Frame <number>  Energy = <energy> Hartree
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
    """
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()

    # Header lines
    file_handle.write(f"{len(symbols)}\n")
    file_handle.write(f"Frame {frame_number}  Energy = {energy:.10f} Hartree\n")

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
