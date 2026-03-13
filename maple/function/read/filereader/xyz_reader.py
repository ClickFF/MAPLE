# xyz_reader.py

import os
from typing import List, Optional, Tuple
from ase import Atoms
import numpy as np
import re

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


def _parse_charge_mult(input_str: str) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    """
    Parse input string to extract file path, charge, and multiplicity.
    
    Supported formats:
        - /path/to/file.xyz
        - /path/to/file.xyz -14 2
        - XYZ -14 2 /path/to/file.xyz
        - XYZ /path/to/file.xyz -14 2
    
    Returns:
        Tuple of (file_path, charge, multiplicity)
    """
    parts = input_str.strip().split()
    
    if not parts:
        raise ValueError("Empty input string")
    
    # Remove 'XYZ' keyword if present (case-insensitive)
    if parts[0].upper() == 'XYZ':
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


class XYZReader:
    """
    Robust XYZ file reader with support for charge and multiplicity.

    Format assumptions (lenient):
      - First non-empty line: integer atom count (N).
      - Second line: comment (ignored, can be empty).
      - Next lines: at least N coordinate-like lines ('Elem x y z [...]').
        * If the actual number of coordinate-like lines != N, we prefer the actual count.
        * Extra columns after z are allowed and ignored.

    Input format:
      - Simple: '/path/to/file.xyz'
      - With charge/mult: 'XYZ -14 2 /path/to/file.xyz' or '/path/to/file.xyz -14 2'
      
    The charge and multiplicity will be stored in atoms.info['charge'] and atoms.info['mult']
    for UMA to use.

    Behavior:
      - The file path must be absolute. If not found, the reader tries case-insensitive lookup in the same directory.
      - Returns an ASE Atoms with float64 positions.
      - Raises ValueError only when no valid coordinate lines can be parsed.

    Args:
        file_path (str): Path to XYZ file, optionally with charge and multiplicity.

    Returns:
        Atoms: ASE Atoms object parsed from the file, with charge/mult in .info dict.
    """

    def __new__(cls, file_path: str, charge: Optional[int] = None, mult: Optional[int] = None) -> Atoms:
        # Parse input string if charge and mult not explicitly provided
        if charge is None and mult is None:
            parsed_path, parsed_charge, parsed_mult = _parse_charge_mult(file_path)
            file_path = parsed_path
            charge = parsed_charge
            mult = parsed_mult
        
        if not os.path.isabs(file_path):
            raise ValueError(f"XYZ file path must be absolute: {file_path}")

        # Try exact path; if missing, attempt case-insensitive lookup
        resolved = _case_insensitive_lookup(file_path)
        if not os.path.exists(resolved):
            raise FileNotFoundError(f"XYZ file not found: {file_path}")

        with open(resolved, 'r', encoding='utf-8', errors='ignore') as f:
            raw_lines = f.readlines()

        # Strip only trailing newlines; keep possible internal spaces
        lines = [ln.rstrip('\r\n') for ln in raw_lines]

        # Skip leading blank lines
        idx = 0
        while idx < len(lines) and not lines[idx].strip():
            idx += 1

        if idx >= len(lines):
            raise ValueError(f"Empty XYZ file: {resolved}")

        # Parse header natoms if possible (lenient)
        natoms_declared = None
        first = lines[idx].strip()
        try:
            natoms_declared = int(first)
            idx += 1
        except Exception:
            # No integer at first non-empty line; treat as malformed header
            natoms_declared = None  # fall back to auto-detect later

        # Skip the comment line if present (standard XYZ has one)
        if idx < len(lines):
            # comment line may be empty; still advance by one if we had a natoms line
            if natoms_declared is not None:
                idx += 1

        # Collect coordinate-like lines from the remainder
        coord_lines: List[str] = []
        for ln in lines[idx:]:
            if not ln.strip():
                continue
            # Accept lines that start with Elem x y z (ignore extra trailing columns)
            m = _COORD_RE.match(ln)
            if m:
                coord_lines.append(ln)

        if not coord_lines:
            raise ValueError(f"No valid XYZ coordinate lines found in: {resolved}")

        # If natoms declared and mismatched, prefer actual detected lines
        # but do not fail—be lenient and trust the body.
        # If there are more lines than declared, we will use all detected lines.
        # If fewer than declared, we still use what we have.
        elements: List[str] = []
        coords: List[List[float]] = []
        for ln in coord_lines:
            m = _COORD_RE.match(ln)
            # m must exist because we filtered above
            elem = m.group(1)
            x = float(m.group(2))
            y = float(m.group(3))
            z = float(m.group(4))
            elements.append(elem)
            coords.append([x, y, z])

        atoms = Atoms(symbols=elements, positions=np.array(coords, dtype=np.float64))

        # Parse extXYZ comment line for Lattice and PBC info
        # The comment line was at index (natoms_line_idx + 1); we need to re-find it.
        # Re-scan: after the natoms line, the very next non-empty line is the comment.
        comment_line = None
        _scan = 0
        while _scan < len(lines) and not lines[_scan].strip():
            _scan += 1
        # skip natoms line
        if _scan < len(lines):
            try:
                int(lines[_scan].strip())
                _scan += 1
            except ValueError:
                pass
        # comment line
        if _scan < len(lines):
            comment_line = lines[_scan]

        if comment_line:
            # Try to parse Lattice="a b c d e f g h i"
            lattice_match = re.search(r'[Ll]attice\s*=\s*"([^"]+)"', comment_line)
            pbc_match = re.search(r'[Pp][Bb][Cc]\s*=\s*"([^"]+)"', comment_line)
            if lattice_match:
                try:
                    vals = [float(v) for v in lattice_match.group(1).split()]
                    if len(vals) == 9:
                        cell = np.array(vals).reshape(3, 3)
                        atoms.set_cell(cell)
                        if pbc_match:
                            pbc_str = pbc_match.group(1).upper().split()
                            pbc = [s in ('T', 'TRUE', '1') for s in pbc_str]
                            if len(pbc) == 3:
                                atoms.set_pbc(pbc)
                        else:
                            atoms.set_pbc(True)
                except (ValueError, IndexError):
                    pass

        # Store charge and multiplicity in atoms.info for UMA
        if charge is not None:
            atoms.info['charge'] = charge
        if mult is not None:
            atoms.info['mult'] = mult
        if mult is not None:
            atoms.info['spin'] = (mult -1)/2

        return atoms