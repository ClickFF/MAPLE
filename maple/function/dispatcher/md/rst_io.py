"""
Restart (RST) checkpoint file I/O for MD simulations.

RST files store the complete simulation state for restart/resume:
    - Atomic positions and velocities
    - Cell parameters (for periodic systems)
    - Simulation metadata (step, time, ensemble, timestep, energy)
    - RNG state (for NVT/NPT deterministic continuation)

File format
-----------
::

    MAPLE_RST_V1
    natoms = N
    step = S
    time = T  (fs)
    ensemble = nve|nvt|npt
    timestep = dt  (fs)
    energy = E  (Hartree)
    [rng_state = hex_string]  (NVT/NPT only)
    [cell = a b c alpha beta gamma]  (PBC only)
    [pbc = T/F T/F T/F]  (PBC only)
    Symbol  x  y  z  vx  vy  vz
    ...
    END_RST

Velocities are stored in atomic units (Bohr/a.u. time).

Compatible with GROMACS checkpoint concept; enables exact continuation
of MD trajectories with identical thermodynamic evolution.
"""

import json
from pathlib import Path

import numpy as np
from ase.cell import Cell


RST_HEADER = "MAPLE_RST_V1"



def get_rng_state_hex(rng: np.random.Generator) -> str:
    """
    Serialize RNG state to hex string for checkpoint storage.

    Parameters
    ----------
    rng : np.random.Generator
        NumPy random generator instance.

    Returns
    -------
    str
        Hex-encoded JSON representation of the RNG state.
        Can be restored via ``restore_rng_from_hex()``.
    """
    state_json = json.dumps(rng.bit_generator.state, sort_keys=True)
    return state_json.encode().hex()


def restore_rng_from_hex(rng: np.random.Generator, hex_str: str) -> None:
    """
    Restore RNG state from hex string.

    Parameters
    ----------
    rng : np.random.Generator
        NumPy random generator instance to restore into.
    hex_str : str
        Hex-encoded RNG state from ``get_rng_state_hex()``.
    """
    state = json.loads(bytes.fromhex(hex_str).decode())
    rng.bit_generator.state = state


def write_rst(path, atoms, velocities, step, timestep, ensemble, energy, rng_state=None):
    """
    Write MD restart checkpoint file.

    Parameters
    ----------
    path : str or Path
        Output file path.
    atoms : ase.Atoms
        Atomic system.
    velocities : np.ndarray
        Velocities in atomic units (Bohr/a.u. time), shape (N, 3).
    step : int
        Current MD step number.
    timestep : float
        Timestep in fs.
    ensemble : str
        Ensemble type ('nve', 'nvt', 'npt').
    energy : float
        Total energy in Hartree.
    rng_state : str, optional
        Hex-encoded RNG state (for NVT/NPT deterministic continuation).

    Raises
    ------
    ValueError
        If velocities shape mismatch with atoms count.
    """
    path = Path(path)
    expected_shape = (len(atoms), 3)
    if np.shape(velocities) != expected_shape:
        raise ValueError(
            "Velocities must have shape "
            f"{expected_shape}, got {np.shape(velocities)}"
        )
    time_fs = step * timestep
    cell_line = ""
    pbc_line = ""
    if any(atoms.pbc):
        cell = atoms.cell.cellpar()
        cell_line = (
            f"cell = {cell[0]:.10f} {cell[1]:.10f} {cell[2]:.10f} "
            f"{cell[3]:.10f} {cell[4]:.10f} {cell[5]:.10f}\n"
        )
        pbc_flags = ["T" if flag else "F" for flag in atoms.pbc]
        pbc_line = f"pbc = {' '.join(pbc_flags)}\n"

    lines = [
        f"{RST_HEADER}\n",
        f"natoms = {len(atoms)}\n",
        f"step = {step}\n",
        f"time = {time_fs:.10f}\n",
        f"ensemble = {ensemble}\n",
        f"timestep = {timestep:.10f}\n",
        f"energy = {energy:.10f}\n",
    ]
    if rng_state is not None:
        lines.append(f"rng_state = {rng_state}\n")
    if cell_line:
        lines.append(cell_line)
    if pbc_line:
        lines.append(pbc_line)

    for symbol, pos, vel in zip(atoms.get_chemical_symbols(), atoms.get_positions(), velocities):
        lines.append(
            f"{symbol:<2s} {pos[0]: .10f} {pos[1]: .10f} {pos[2]: .10f}"
            f" {vel[0]: .10e} {vel[1]: .10e} {vel[2]: .10e}\n"
        )
    lines.append("END_RST\n")
    path.write_text("".join(lines))



def read_rst(path):
    """
    Read MD restart checkpoint file.

    Parameters
    ----------
    path : str or Path
        RST file path.

    Returns
    -------
    dict
        Dictionary containing:
        - ``natoms`` : int — Number of atoms
        - ``step`` : int — MD step number
        - ``time`` : float — Simulation time in fs
        - ``ensemble`` : str — Ensemble type
        - ``timestep`` : float — Timestep in fs
        - ``energy`` : float — Total energy in Hartree
        - ``rng_state`` : str or None — Hex-encoded RNG state
        - ``symbols`` : list[str] — Element symbols
        - ``positions`` : np.ndarray — Positions in Angstrom, shape (N, 3)
        - ``velocities`` : np.ndarray — Velocities in a.u., shape (N, 3)
        - ``cell`` : list or None — Cell parameters [a,b,c,alpha,beta,gamma]
        - ``pbc`` : list or None — Periodic boundary flags

    Raises
    ------
    ValueError
        If file is not a valid RST file or has missing/invalid fields.
    """
    path = Path(path)
    lines = path.read_text().splitlines()
    if not lines or lines[0].strip() != RST_HEADER:
        raise ValueError(f"Not a valid MAPLE RST file: {path}")
    if not lines or lines[-1].strip() != "END_RST":
        raise ValueError(f"Missing END_RST in {path}")

    header = {}
    atom_lines = []
    for line in lines[1:-1]:
        if "=" in line:
            key, value = line.split("=", 1)
            header[key.strip()] = value.strip()
        elif line.strip():
            atom_lines.append(line)

    required_fields = ["natoms", "step", "time", "ensemble", "timestep", "energy"]
    missing_fields = [field for field in required_fields if field not in header]
    if missing_fields:
        missing_list = ", ".join(missing_fields)
        raise ValueError(f"Missing required RST header fields: {missing_list}")

    natoms = int(header["natoms"])
    if len(atom_lines) != natoms:
        raise ValueError(
            f"Atom count mismatch inside RST file: expected {natoms}, found {len(atom_lines)}"
        )

    symbols = []
    positions = []
    velocities = []
    for idx, line in enumerate(atom_lines, 1):
        parts = line.split()
        if len(parts) != 7:
            raise ValueError(f"Invalid atom line {idx} in {path}: {line!r}")
        symbol = parts[0]
        xyz = [float(x) for x in parts[1:4]]
        vel = [float(x) for x in parts[4:7]]
        symbols.append(symbol)
        positions.append(xyz)
        velocities.append(vel)

    cell = None
    if "cell" in header:
        cell = [float(x) for x in header["cell"].split()]
        if len(cell) != 6:
            raise ValueError(f"Invalid cell line in {path}")

    pbc = None
    if "pbc" in header:
        pbc = [flag == "T" for flag in header["pbc"].split()]
        if len(pbc) != 3:
            raise ValueError(f"Invalid pbc line in {path}")

    return {
        "natoms": natoms,
        "step": int(header["step"]),
        "time": float(header["time"]),
        "ensemble": header["ensemble"],
        "timestep": float(header["timestep"]),
        "energy": float(header["energy"]),
        "rng_state": header.get("rng_state"),
        "symbols": symbols,
        "positions": np.array(positions),
        "velocities": np.array(velocities),
        "cell": cell,
        "pbc": pbc,
    }


def rotate_rst_checkpoint(
    rst_path,
    rst_prev_path,
    atoms,
    velocities,
    step,
    timestep,
    ensemble,
    energy,
    rng_state=None,
):
    """
    Rotate checkpoint files and write new checkpoint.

    Implements GROMACS-style checkpoint rotation:
        1. If ``rst_path`` exists, move it to ``rst_prev_path``
        2. Write new checkpoint to ``rst_path``

    This ensures at least two recent checkpoints are always available,
    protecting against corruption during write.

    Parameters
    ----------
    rst_path : str or Path
        Current checkpoint file path.
    rst_prev_path : str or Path
        Previous checkpoint file path (backup).
    atoms : ase.Atoms
        Atomic system.
    velocities : np.ndarray
        Velocities in atomic units, shape (N, 3).
    step : int
        Current MD step number.
    timestep : float
        Timestep in fs.
    ensemble : str
        Ensemble type ('nve', 'nvt', 'npt').
    energy : float
        Total energy in Hartree.
    rng_state : str, optional
        Hex-encoded RNG state (for NVT/NPT).
    """
    rst_path = Path(rst_path)
    rst_prev_path = Path(rst_prev_path)

    if rst_path.exists() and rst_path.stat().st_size > 0:
        if rst_prev_path.exists():
            rst_prev_path.unlink()
        rst_path.replace(rst_prev_path)

    write_rst(
        rst_path,
        atoms=atoms,
        velocities=velocities,
        step=step,
        timestep=timestep,
        ensemble=ensemble,
        energy=energy,
        rng_state=rng_state,
    )
