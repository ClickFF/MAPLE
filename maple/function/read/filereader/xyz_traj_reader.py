# xyz_traj_reader.py

import os
from typing import List, Optional, Tuple
from ase import Atoms
import numpy as np
import re

from maple.function.utility import Molecules

_COORD_RE = re.compile(
    r'^\s*([A-Za-z][a-z]?)\s+'
    r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s+'
    r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s+'
    r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)'
)

def _case_insensitive_lookup(path: str) -> str:
    """
    Try to resolve a path in a case-insensitive manner within its directory.
    If a case-insensitive match is found, return the resolved absolute path.
    Otherwise, return the original path.
    """
    d, fname = os.path.split(path)
    if not d:
        return path
    if not os.path.isdir(d):
        return path
    # Try exact first
    exact = os.path.join(d, fname)
    if os.path.exists(exact):
        return exact
    # Case-insensitive search
    target_lower = fname.lower()
    for cand in os.listdir(d):
        if cand.lower() == target_lower:
            return os.path.join(d, cand)
    return path


def _parse_charge_mult_traj(input_str: str) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    """
    Parse input string to extract file path, charge, and multiplicity for trajectory.
    
    Supported formats:
        - /path/to/file.xyz
        - /path/to/file.xyz -14 2
        - XYZTRAJ -14 2 /path/to/file.xyz
        - XYZTRAJ /path/to/file.xyz -14 2
    
    Returns:
        Tuple of (file_path, charge, multiplicity)
    """
    parts = input_str.strip().split()
    
    if not parts:
        raise ValueError("Empty input string")
    
    # Remove 'XYZTRAJ' keyword if present (case-insensitive)
    if parts[0].upper() == 'XYZTRAJ':
        parts = parts[1:]
    
    if not parts:
        raise ValueError("No file path provided")
    
    file_path = None
    charge = None
    mult = None
    numbers = []
    
    # Separate path and numbers
    for part in parts:
        # Check if it's a number (int, could be negative)
        try:
            num = int(part)
            numbers.append(num)
        except ValueError:
            # Not a number, assume it's the file path
            if file_path is None:
                file_path = part
            else:
                raise ValueError(f"Multiple file paths found: {file_path} and {part}")
    
    if file_path is None:
        raise ValueError("No file path found in input")
    
    # Parse numbers as charge and multiplicity
    if len(numbers) == 2:
        charge, mult = numbers
    elif len(numbers) == 1:
        raise ValueError("Both charge and multiplicity must be provided together")
    elif len(numbers) > 2:
        raise ValueError(f"Too many numeric arguments: {numbers}")
    
    return file_path, charge, mult


class XYZTrajReader:
    """
    Robust XYZ trajectory file reader with support for charge and multiplicity.

    Reads multi-frame XYZ files where each frame has:
      - First line: integer atom count (N)
      - Second line: comment (can contain energy info, ignored)
      - Next N lines: atomic coordinates

    Input format:
      - Simple: '/path/to/trajectory.xyz' or 'traj.xyz' or './traj.xyz'
      - With charge/mult: 'XYZTRAJ -14 2 /path/to/trajectory.xyz' or '/path/to/trajectory.xyz -14 2'

    Path resolution:
      - Absolute paths are used directly.
      - Relative paths are resolved relative to base_dir (defaults to current working directory).

    The charge and multiplicity will be applied to ALL frames and stored in
    atoms.info['charge'] and atoms.info['mult'] for each frame.

    Returns:
        Molecules: Molecules object containing all frames from the trajectory.
    """

    def __new__(cls, file_path: str, charge: Optional[int] = None, mult: Optional[int] = None, base_dir: Optional[str] = None) -> Molecules:
        # Parse input string if charge and mult not explicitly provided
        if charge is None and mult is None:
            parsed_path, parsed_charge, parsed_mult = _parse_charge_mult_traj(file_path)
            file_path = parsed_path
            charge = parsed_charge
            mult = parsed_mult

        # Resolve path: absolute paths used directly, relative paths resolved against base_dir
        if not os.path.isabs(file_path):
            if base_dir is not None:
                file_path = os.path.join(base_dir, file_path)
            else:
                file_path = os.path.abspath(file_path)

        # Try exact path; if missing, attempt case-insensitive lookup
        resolved = _case_insensitive_lookup(file_path)
        if not os.path.exists(resolved):
            raise FileNotFoundError(f"XYZ trajectory file not found: {file_path}")
        
        # Read all frames from the trajectory file
        atoms_list = cls._read_all_frames(resolved, charge, mult)
        
        if not atoms_list:
            raise ValueError(f"No valid frames found in trajectory file: {resolved}")
        
        # Return Molecules object
        return Molecules(atoms_list)
    
    @staticmethod
    def _read_all_frames(file_path: str, charge: Optional[int], mult: Optional[int]) -> List[Atoms]:
        """
        Read all frames from an XYZ trajectory file.
        
        Args:
            file_path: Path to the XYZ trajectory file
            charge: Charge to apply to all frames
            mult: Multiplicity to apply to all frames
        
        Returns:
            List of ASE Atoms objects, one per frame
        """
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            raw_lines = f.readlines()
        
        lines = [ln.rstrip('\r\n') for ln in raw_lines]
        
        frames = []
        idx = 0
        
        while idx < len(lines):
            # Skip leading blank lines
            while idx < len(lines) and not lines[idx].strip():
                idx += 1
            
            if idx >= len(lines):
                break
            
            # Parse atom count for this frame
            try:
                natoms = int(lines[idx].strip())
                idx += 1
            except (ValueError, IndexError):
                # Not a valid frame header, skip this line
                idx += 1
                continue
            
            # Skip comment line
            if idx >= len(lines):
                break
            idx += 1  # Skip comment line
            
            # Read coordinate lines for this frame
            elements = []
            coords = []
            lines_read = 0
            
            while idx < len(lines) and lines_read < natoms:
                ln = lines[idx]
                idx += 1
                
                if not ln.strip():
                    continue
                
                m = _COORD_RE.match(ln)
                if m:
                    elem = m.group(1)
                    x = float(m.group(2))
                    y = float(m.group(3))
                    z = float(m.group(4))
                    elements.append(elem)
                    coords.append([x, y, z])
                    lines_read += 1
            
            # Only add frame if we got the expected number of atoms
            if lines_read == natoms and natoms > 0:
                atoms = Atoms(symbols=elements, positions=np.array(coords, dtype=np.float64))
                
                # Apply charge and multiplicity to this frame
                if charge is not None:
                    atoms.info['charge'] = charge
                if mult is not None:
                    atoms.info['mult'] = mult
                    atoms.info['spin'] = (mult - 1) / 2
                
                frames.append(atoms)
        
        return frames