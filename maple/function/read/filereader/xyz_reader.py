# xyz_reader.py

import os
from typing import List, Optional, Tuple, Union
from ase import Atoms
from ase.io.extxyz import parse_properties
import numpy as np
import re

_COORD_RE = re.compile(
    r'^\s*([A-Za-z][a-z]?)\s+'
    r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s+'
    r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s+'
    r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)'
)


def _parse_pbc_tokens(pbc_str: str) -> List[bool]:
    """Parse PBC metadata tokens into a 3-element boolean list."""
    pbc_tokens = pbc_str.upper().split()
    if len(pbc_tokens) != 3 or any(
        token not in ('T', 'TRUE', '1', 'F', 'FALSE', '0') for token in pbc_tokens
    ):
        raise ValueError('PBC must contain three boolean values')
    return [token in ('T', 'TRUE', '1') for token in pbc_tokens]



def _parse_cell_and_pbc(comment_line: str) -> Tuple[Optional[np.ndarray], Optional[List[bool]]]:
    """Extract cell and PBC metadata from extXYZ or MAPLE XYZ comment lines."""
    lattice_match = re.search(r'\b[Ll]attice\s*=\s*"([^"]*)"', comment_line)
    pbc_match = re.search(
        r'\b[Pp][Bb][Cc]\s*=\s*(?:"([^"]*)"|(\S+\s+\S+\s+\S+))', comment_line
    )
    explicit_pbc = None
    if pbc_match:
        explicit_pbc = _parse_pbc_tokens(
            pbc_match.group(1) if pbc_match.group(1) is not None else pbc_match.group(2)
        )
    elif re.search(r'\b[Pp][Bb][Cc]\s*=', comment_line):
        raise ValueError('Invalid PBC field')
    if lattice_match:
        vals = [float(v) for v in lattice_match.group(1).split()]
        if len(vals) != 9 or not np.isfinite(vals).all():
            raise ValueError('Lattice must contain nine finite values')
        cell = np.array(vals, dtype=np.float64).reshape(3, 3)
        return cell, explicit_pbc or [True, True, True]
    if re.search(r'\b[Ll]attice\s*=', comment_line):
        raise ValueError('Lattice must be a quoted list of nine values')

    cell_match = re.search(
        r'\bCell\s*=\s*'
        r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s+'
        r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s+'
        r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s+'
        r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s+'
        r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s+'
        r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)',
        comment_line,
    )
    if cell_match:
        cellpar = [float(cell_match.group(i)) for i in range(1, 7)]
        if any(length <= 0 for length in cellpar[:3]) or any(
            angle <= 0 or angle >= 180 for angle in cellpar[3:]
        ):
            raise ValueError('Cell must contain positive lengths and angles between 0 and 180 degrees')
        try:
            cell = Atoms(cell=cellpar, pbc=True).cell.array.copy()
        except AssertionError as e:
            raise ValueError('Cell lengths and angles do not define a valid cell') from e
        if not np.isfinite(cell).all():
            raise ValueError('Cell must contain valid lengths and angles')
        return cell, explicit_pbc or [True, True, True]
    if re.search(r'\bCell\s*=', comment_line):
        raise ValueError('Cell must contain three lengths and three angles')

    return None, explicit_pbc


def _parse_charge_and_mult(comment_line: str) -> Tuple[float, int]:
    """Read per-structure electronic state from the XYZ comment line."""
    charge_match = re.search(
        r'\b(?:charge|total_charge)\s*=\s*(?:"([^"]*)"|(\S*))', comment_line, re.IGNORECASE
    )
    mult_match = re.search(
        r'\b(?:mult|multiplicity)\s*=\s*(?:"([^"]*)"|(\S*))', comment_line, re.IGNORECASE
    )
    charge = 0.0
    mult = 1.0
    if charge_match:
        charge = float(charge_match.group(1) if charge_match.group(1) is not None else charge_match.group(2))
    if mult_match:
        mult = float(mult_match.group(1) if mult_match.group(1) is not None else mult_match.group(2))
    if not np.isfinite(charge):
        raise ValueError('Charge must be finite')
    if not np.isfinite(mult) or mult < 1 or not mult.is_integer():
        raise ValueError('Multiplicity must be a positive integer')
    return charge, int(mult)


def _parse_properties(comment_line: str):
    """Read an optional EXTXYZ column declaration using ASE's converters."""
    properties_match = re.search(
        r'\bProperties\s*=\s*(?:"([^"]*)"|(\S*))', comment_line
    )
    if not properties_match:
        return None
    fields = (properties_match.group(1) if properties_match.group(1) is not None
              else properties_match.group(2)).split(':')
    if len(fields) % 3 or not fields[0]:
        raise ValueError('Properties requires name:type:column-count entries')
    names = fields[::3]
    if len(set(names)) != len(names) or any(int(width) < 1 for width in fields[2::3]):
        raise ValueError('Properties requires unique names and positive column counts')
    properties, names, dtype, converters = parse_properties(':'.join(fields))
    if properties.get('pos') != ('positions', 3):
        raise ValueError('Properties must declare pos with three columns')
    if properties.get('species') != ('symbols', 1) and properties.get('Z') != ('numbers', 1):
        raise ValueError('Properties must declare species or Z with one column')
    return properties, names, dtype, converters


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
    Read one or more complete XYZ/EXTXYZ structures in file order.

    Each structure has an atom count, a comment line (possibly empty), and
    the declared number of atom records. Comments may specify electronic
    state, cell/PBC, and Properties. Unlabelled extra columns are ignored.

    Input format:
      - Simple: '/path/to/file.xyz' or 'ag.xyz' or './ag.xyz'
      - With charge/mult: 'XYZ -14 2 /path/to/file.xyz' or '/path/to/file.xyz -14 2'

    Explicit charge/multiplicity override all structures in the file;
    otherwise each structure uses its comment values, defaulting to 0/1.

    Path resolution:
      - Absolute paths are used directly.
      - Relative paths are resolved relative to base_dir (defaults to current working directory).

    Returns:
        Atoms for a single structure, List[Atoms] for multiple structures.
        Extra Properties columns remain input arrays, without a calculator.
    """

    def __new__(cls, file_path: str, charge: Optional[int] = None, mult: Optional[int] = None, base_dir: Optional[str] = None) -> Union[Atoms, List[Atoms]]:
        # Parse input string if charge and mult not explicitly provided
        if charge is None and mult is None:
            parsed_path, parsed_charge, parsed_mult = _parse_charge_mult(file_path)
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
            raise FileNotFoundError(f"XYZ file not found: {file_path}")

        with open(resolved, 'r', encoding='utf-8', errors='ignore') as f:
            raw_lines = f.readlines()

        # Strip only trailing newlines; keep possible internal spaces
        lines = [ln.rstrip('\r\n') for ln in raw_lines]

        atoms_list: List[Atoms] = []
        idx = 0
        while idx < len(lines):
            while idx < len(lines) and not lines[idx].strip():
                idx += 1
            if idx >= len(lines):
                break

            try:
                first = lines[idx].strip()
                natoms_declared = int(first)
                idx += 1
                if natoms_declared < 1 or idx >= len(lines):
                    raise ValueError('Expected a positive atom count and a comment line')
                comment_line = lines[idx]
                idx += 1

                frame_charge, frame_mult = _parse_charge_and_mult(comment_line)
                cell, pbc = _parse_cell_and_pbc(comment_line)
                properties = _parse_properties(comment_line)
                elements: List[str] = []
                coords: List[List[float]] = []
                coord_lines = []
                lines_read = 0

                while idx < len(lines) and lines_read < natoms_declared:
                    ln = lines[idx]
                    idx += 1
                    if not ln.strip():
                        continue
                    if properties is None:
                        m = _COORD_RE.match(ln)
                        if not m:
                            raise ValueError(f'Invalid atom record: {ln}')
                        elements.append(m.group(1))
                        coords.append([float(m.group(i)) for i in range(2, 5)])
                    else:
                        _, _, _, converters = properties
                        values = ln.split()
                        if len(values) != len(converters):
                            raise ValueError('Atom record does not match Properties column count')
                        values = tuple(convert(value) for convert, value in zip(converters, values))
                        if any(value is None for value in values):
                            raise ValueError('Invalid logical Properties value')
                        coord_lines.append(values)
                    lines_read += 1

                if lines_read != natoms_declared:
                    raise ValueError(f'Expected {natoms_declared} atoms, found {lines_read}')
                if properties is None:
                    atoms = Atoms(symbols=elements, positions=np.array(coords, dtype=np.float64))
                else:
                    properties, names, dtype, _ = properties
                    data = np.array(coord_lines, dtype=dtype)
                    arrays = {}
                    for name in names:
                        ase_name, width = properties[name]
                        arrays[ase_name] = (data[name] if width == 1 else
                                            np.column_stack([data[name + str(i)] for i in range(width)]))
                    elements = arrays.pop('symbols', None)
                    numbers = arrays.pop('numbers', None)
                    coords = arrays.pop('positions')
                    atoms = (Atoms(numbers=numbers, positions=coords) if numbers is not None else
                             Atoms(symbols=elements, positions=coords))
                    for name, values in arrays.items():
                        atoms.new_array(name, values)

                if not np.isfinite(atoms.positions).all():
                    raise ValueError('Coordinates must be finite')
                if cell is not None:
                    atoms.set_cell(cell)
                if pbc is not None:
                    atoms.set_pbc(pbc)

                frame_charge = float(frame_charge if charge is None else charge)
                frame_mult = float(frame_mult if mult is None else mult)
                if not np.isfinite(frame_charge):
                    raise ValueError('Charge must be finite')
                if not np.isfinite(frame_mult) or frame_mult < 1 or not frame_mult.is_integer():
                    raise ValueError('Multiplicity must be a positive integer')
                atoms.info['charge'] = frame_charge
                atoms.info['mult'] = int(frame_mult)
                atoms.info['spin'] = (frame_mult - 1) / 2
                atoms_list.append(atoms)
            except (ValueError, TypeError, KeyError) as e:
                raise ValueError(f'XYZ structure {len(atoms_list) + 1} in {resolved}: {e}') from e

        if not atoms_list:
            raise ValueError(f"Empty XYZ file: {resolved}")
        return atoms_list[0] if len(atoms_list) == 1 else atoms_list
