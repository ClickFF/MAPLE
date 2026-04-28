import os
import re
from pathlib import Path
from typing import Any, List, Union

from ase import Atoms
import numpy as np
import torch

from .filereader import XYZReader
from .filereader import PostReader
from .filereader import XYZTrajReader
from .command_control import CommandControl

from .header.header import print_banner

from maple.function.utility import Molecules
from maple.function.timer import timer
from maple.function.dispatcher.md.logger import _backup_file

class InputReader():
    def __init__(self):
        self.input:str = None
        self.output:str = None
        self.error:bool = False
        self.device:torch.device = None

        self.model:int = None
        # 1: ANI-2x
        # 2: ANI-1x
        # 3: ANI-1ccx
        # 4: ANI-1xnr

        self.jobtype:int = None
        # 1: opt
        # 2: sp
        # 3: scan
        # 4: freq
        # 5: ts

        self.d4:bool = False

        self.scan = False

    def __call__(self, input_file_name: str, output_file_name: str = None) -> Union[Atoms, Molecules]:
        """
        Read the input file, parse settings, molecular coordinates, and post-processing commands.
        Support multiple coordinate groups separated by a blank line or '&'.
        Allow arbitrary blank lines between sections without breaking parsing.
        Support POST file references for loading post-processing commands from external files.

        Args:
            input_file_name (str): Path to the input file.
            output_file_name (str, optional): Path to the output file. Defaults to None.

        Returns:
            Atoms or Molecules: ASE Atoms object if a single structure is present,
                            or Molecules object if multiple structures are present.
        """

        try:
            # Resolve absolute paths for input and output
            if isinstance(input_file_name, str):
                self.input = os.path.abspath(input_file_name)
            else:
                raise TypeError("The input_file_name should be a string.")

            if output_file_name is not None:
                if isinstance(output_file_name, str):
                    self.output = os.path.abspath(output_file_name)
                else:
                    raise TypeError("The output_file_name should be a string.")
            else:
                self.output = os.path.splitext(self.input)[0] + ".out"
                self.output = os.path.abspath(self.output)

            # Back up existing output file using GROMACS-style numbering
            output_path = Path(self.output)
            _backup_file(output_path)

            print_banner(self.output)

            # ------------------------------------------------------------------
            # Robust three-section split:
            #   1) SETTINGS  : consecutive lines starting with '#' at the top
            #                  (blank lines allowed; they are not part of settings)
            #   2) MOLECULES : lines that are either blank, '&',
            #                  'XYZ /abs/path', or atomic lines 'Elem x y z'
            #                  (supports scientific notation). Arbitrary blank
            #                  lines INSIDE this section are allowed.
            #                  The section ends at the first non-matching, non-blank line.
            #   3) POSTPROC  : everything after MOLECULES (blank lines ignored).
            #                  Now also supports 'POST /abs/path/to/file.out' references.
            # ------------------------------------------------------------------

            with open(self.input, 'r') as f:
                raw_lines = f.readlines()

            # Regex used to detect coordinate-like lines while splitting sections.
            # It matches standard inline structure lines "Elem x y z" and also the
            # legacy 7-column shape "Elem x y z vx vy vz" so the inline parser can
            # raise a clear error instead of silently truncating velocity columns.
            atom_line_re = re.compile(
                r'^\s*([A-Za-z][a-z]?)\s+'
                r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s+'
                r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s+'
                r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)'
                r'(?:\s+[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?){0,3}'
                r'\s*$'
            )

            def is_settings_line(s: str) -> bool:
                return s.lstrip().startswith('#')

            # Regex for charge/multiplicity line: two integers (e.g. "0 1", "-1 2")
            charge_mult_re = re.compile(r'^\s*[+-]?\d+\s+\d+\s*$')

            def is_xyz_ref(s: str) -> bool:
                upper = s.upper()
                return upper.startswith('XYZ ') or upper.startswith('XYZTRAJ ')

            def is_coord_like(s: str) -> bool:
                if s == '' or s == '&':
                    return True
                if is_xyz_ref(s):
                    return True
                if charge_mult_re.match(s):
                    return True
                return atom_line_re.match(s) is not None

            # === 1) SETTINGS ===
            settings = []
            i = 0
            n = len(raw_lines)

            while i < n:
                line = raw_lines[i].rstrip('\n')
                if line.strip() == '':
                    i += 1
                    continue
                if is_settings_line(line):
                    settings.append(line)
                    i += 1
                    continue
                break  # First non-settings line

            # Skip any blank lines before molecule section
            while i < n and raw_lines[i].strip() == '':
                i += 1

            # === 2) MOLECULES ===
            molecules = []
            while i < n:
                line = raw_lines[i].rstrip('\n')
                s = line.strip()
                if s == '':
                    molecules.append(line)
                    i += 1
                    continue
                if is_coord_like(s):
                    molecules.append(s)
                    i += 1
                    continue
                # First non-coordinate-like line marks end of molecule block
                break

            if len(molecules) == 0:
                raise ValueError("Cannot find the coordinate block.")

            # === 3) POST-PROCESSING ===
            post_processing = []
            while i < n:
                line = raw_lines[i].rstrip('\n')
                s = line.strip()
                if s != '':
                    post_processing.append(s)
                i += 1

            # Basic validation (actual content validated later)
            if not self.input:
                raise ValueError("Unrecognized input file.")
            if not self.output:
                raise ValueError("Unrecognized output file.")
            if len(settings) == 0:
                raise ValueError("Cannot find the settings block.")
            if len(molecules) == 0:
                raise ValueError("Cannot find the coordinate block.")
            # post_processing can be empty → optional

        except (AssertionError, TypeError, ValueError) as e:
            self.log_error(str(e))
            raise

        # === Step 1: Parse settings ===
        with timer("Settings Parsing"):
            self.settings_command(settings)

        # === Step 2: Parse coordinate section ===
        with timer("Coordinate Section Parsing"):
            atoms_or_list = self.element_and_coordinates(molecules)
        
        # === Step 3: Expand post-processing commands (handle POST references) ===
        with timer("Post-Processing Expansion"):
            if post_processing:
                expanded_post_processing = self.expand_post_processing(post_processing)
                
                if isinstance(atoms_or_list, list):
                    processed_list = []
                    for idx, atoms in enumerate(atoms_or_list, start=1):
                        self.log_info([f"\nApplying post-processing to group {idx}...\n"])
                        processed_list.append(self.post_processing_command(expanded_post_processing, atoms))
                    atoms_or_list = processed_list
                else:
                    atoms_or_list = self.post_processing_command(expanded_post_processing, atoms_or_list)

        # Return Atoms if single structure, Molecules if multiple
        if isinstance(atoms_or_list, list):
            return Molecules(atoms_or_list)
        else:
            return atoms_or_list

    def expand_post_processing(self, post_processing: List[str]) -> List[str]:
        """
        Expand post-processing commands by replacing POST file references with their contents.
        
        POST file references have the format:
            POST /absolute/path/to/file.out
        
        Args:
            post_processing (List[str]): List of post-processing commands and POST references.
        
        Returns:
            List[str]: Expanded list of post-processing commands with POST references resolved.
        """
        expanded = []
        info_message = []
        
        for line in post_processing:
            stripped = line.strip()
            if not stripped:
                continue
            
            # Check if this is a POST file reference
            if stripped.upper().startswith('POST '):
                parts = stripped.split(maxsplit=1)
                if len(parts) != 2:
                    raise ValueError(f"Invalid POST reference line: '{line}'")
                
                file_path = parts[1]
                
                # Log that we're loading from file
                info_message.append(f"\nLoading post-processing commands from: {file_path}\n")
                info_message.append('-' * 70 + '\n')
                
                try:
                    # Use PostReader to load commands from file
                    file_commands = PostReader(file_path)
                    expanded.extend(file_commands)
                    info_message.append(f"Loaded {len(file_commands)} commands from {os.path.basename(file_path)}\n")
                except Exception as e:
                    raise ValueError(f"Failed to read POST file '{file_path}': {e}")
            else:
                # Regular command, add directly
                expanded.append(stripped)
        
        if info_message:
            self.log_info(info_message)
        
        return expanded

    def log_error(self, error_message: str) -> None:
        """Logs error messages to the output file."""
        with open(self.output, 'a') as file:
            file.write(f"ERROR: {error_message}\n")

    def log_info(self, info_message: list) -> None:
        """Logs info messages to the output file."""
        with open(self.output, 'a') as file:
            for info in info_message:
                file.write(f"{info}")

    def settings_command(self, settings: list):
        """
        Parse all # commands using CommandControl and store them in self.
        """
        try:
            cc = CommandControl.from_settings(settings, output_path=self.output)
            self.command_control = cc
            params = cc.as_dict()

            # Normalize model/options into the merged `model + model_options` contract.
            model_val = params.get("model")
            model_options = dict(params.get("model_options", {}))
            if isinstance(model_val, dict):
                self.model = model_val.get("name", "").lower()
                model_options.update({k: v for k, v in model_val.items() if k != "name"})
            else:
                self.model = model_val.lower() if model_val else None
            # Port top-level #external_field directive into model_options
            # so MACEPolCalculator(external_field=...) constructor receives it.
            if "external_field" in params:
                model_options["external_field"] = params["external_field"]

            self.model_options = model_options
            self.model_params = model_options

            dev_str: str = params.get("device", "cpu").lower()

            # Automatically handle device selection with availability checks
            if dev_str.startswith("gpu") or dev_str.startswith("cuda"):
                idx = ''.join([c for c in dev_str if c.isdigit()])
                cuda_idx = idx if idx != '' else '0'
                if torch.cuda.is_available():
                    self.device = torch.device(f'cuda:{cuda_idx}')
                else:
                    self.log_info(["\nWARNING: CUDA is not available. Falling back to CPU.\n"])
                    self.device = torch.device('cpu')
            elif dev_str == "mps":
                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    self.device = torch.device('mps')
                else:
                    self.log_info(["\nWARNING: MPS is not available. Falling back to CPU.\n"])
                    self.device = torch.device('cpu')
            else:
                try:
                    self.device = torch.device(dev_str)
                except:
                    self.log_info(["\nWARNING: Unrecognized device. Falling back to CPU.\n"])
                    self.device = torch.device('cpu')


            self.d4 = params.get("d4", False)
            self.jobtype = params.get("task")  # ← now replaces jobtype
            self.pbc = params.get("pbc", None)  # PBC cell dimensions [X, Y, Z]


            self.log_info([cc.summary()])

        except ValueError as e:
            self.log_error(str(e))
            raise




    def element_and_coordinates(self, molecules: List[str]) -> Union[Atoms, List[Atoms]]:
        """
        Parse the coordinate block and support multiple groups.
        Groups are separated by a blank line or by a single '&' line.

        Each group can be:
          - Inline atomic coordinates (Elem x y z), or
          - External file reference(s): 'XYZ /absolute/path/to/file.xyz'
            If a group contains multiple 'XYZ ...' lines, each line is treated as a separate structure.

        Examples:

            # Inline + file, separated by '&'
            H 0 0 0
            O 0 0 1
            &
            XYZ /abs/path/mol.xyz

            # Two files in one group (no blank lines) -> two structures
            XYZ /abs/path/react.xyz
            XYZ /abs/path/prod.xyz

        Returns:
            Atoms: if only one structure is present
            List[Atoms]: if multiple structures are present (will be converted to Molecules in __call__)
        """

        # Regex for inline atomic line: element + 3 position floats, with optional
        # legacy extra columns only so we can detect and reject inline velocity input
        # explicitly. Formal inline/XYZ structure input is coordinates-only; velocity
        # restoration must go through RST state loading.
        atom_pattern = re.compile(
            r'^\s*([A-Za-z][a-z]?)\s+'
            r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s+'
            r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s+'
            r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)'
            r'(?:\s+[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?){0,3}'
            r'\s*$'
        )

        # Regex for charge and multiplicity line: two integers (charge can be negative)
        charge_mult_pattern = re.compile(r'^\s*([+-]?\d+)\s+(\d+)\s*$')

        info_message = [f'\n{"Coordinates".center(70)}\n', '*' * 70 + '\n']

        blocks: List[List[str]] = []
        current_block: List[str] = []

        def flush_block() -> None:
            """Finalize current block if it contains anything."""
            nonlocal current_block
            if current_block:
                blocks.append(current_block)
                current_block = []

        try:
            # Split input lines into blocks by blank line or '&'
            for raw in molecules:
                line = raw.strip()
                if line == '' or line == '&':
                    flush_block()
                    continue
                current_block.append(line)
            flush_block()

            if not blocks:
                raise ValueError("No coordinate groups found in the input.")

            atoms_list: List[Atoms] = []
            group_counter = 0

            # Get input file directory for resolving relative paths
            input_dir = os.path.dirname(self.input)

            for block in blocks:
                # Normalize tokens for this block
                tokens = [b.strip() for b in block if b.strip()]
                if not tokens:
                    continue

                # Case 1: the block contains only XYZ/XYZTRAJ file references
                all_xyz = all(t.upper().startswith("XYZ ") or t.upper().startswith("XYZTRAJ ") for t in tokens)
                any_xyz = any(t.upper().startswith("XYZ ") or t.upper().startswith("XYZTRAJ ") for t in tokens)

                if all_xyz:
                    for xyz_line in tokens:
                        parts = xyz_line.split(maxsplit=1)
                        if len(parts) != 2:
                            raise ValueError(f"Invalid XYZ reference line: '{xyz_line}'")
                        
                        # Check if this is XYZTRAJ or XYZ
                        keyword = parts[0].upper()
                        file_path = parts[1]
                        
                        if keyword == 'XYZTRAJ':
                            # Read trajectory file, returns Molecules object
                            molecules_obj = XYZTrajReader(file_path, base_dir=input_dir)

                            # Apply PBC to all frames if specified
                            if self.pbc is not None:
                                from ase.cell import Cell
                                for traj_atoms in molecules_obj.multiatoms:
                                    traj_atoms.set_pbc([True, True, True])
                                    # PBC format: [a, b, c, alpha, beta, gamma]
                                    traj_atoms.set_cell(Cell.fromcellpar(self.pbc))

                            # Add all frames from the trajectory to atoms_list
                            atoms_list.extend(molecules_obj.multiatoms)

                            group_counter += len(molecules_obj.multiatoms)
                            info_message.append(f"\nLoaded {len(molecules_obj.multiatoms)} frames from trajectory: {file_path}\n")
                            if self.pbc is not None:
                                info_message.append(f"PBC applied to trajectory: cell = [{self.pbc[0]:.3f}, {self.pbc[1]:.3f}, {self.pbc[2]:.3f}] Angstrom\n")
                            info_message.append('-' * 20 + '\n')
                        elif keyword == 'XYZ':
                            # Regular XYZ file
                            atoms = XYZReader(file_path, base_dir=input_dir)

                            # Apply PBC if specified
                            if self.pbc is not None:
                                from ase.cell import Cell
                                atoms.set_pbc([True, True, True])
                                # PBC format: [a, b, c, alpha, beta, gamma]
                                atoms.set_cell(Cell.fromcellpar(self.pbc))

                            atoms_list.append(atoms)

                            group_counter += 1
                            info_message.append(f"\nGroup {group_counter} (from file: {file_path})\n")
                            info_message.append('-' * 20 + '\n')
                            syms = atoms.get_chemical_symbols()
                            poss = atoms.get_positions()
                            for i, (e, (x, y, z)) in enumerate(zip(syms, poss), start=1):
                                info_message.append(f"{i:<4} {e:<2} {x:>20.6f} {y:>20.6f} {z:>20.6f}\n")
                    continue

                # Case 2: mixed XYZ + inline in the same block -> force user to split
                if any_xyz and not all_xyz:
                    raise ValueError(
                        "Mixed inline coordinates and 'XYZ <path>' in the same group. "
                        "Please separate them with a blank line or '&'."
                    )

                # Case 3: inline coordinates
                elements: List[str] = []
                coords: List[tuple] = []
                charge = None
                mult = None

                # Check if first line contains charge and multiplicity
                if tokens:
                    first_line_match = charge_mult_pattern.match(tokens[0])
                    if first_line_match:
                        charge = int(first_line_match.group(1))
                        mult = int(first_line_match.group(2))
                        tokens = tokens[1:]  # Remove charge/mult line from processing

                        # Validate multiplicity
                        if mult < 1:
                            raise ValueError(f"Invalid multiplicity: {mult}. Must be >= 1")

                # Parse inline atomic coordinates. Formal inline structure input is
                # still position-based, but we also detect the legacy 7-column form
                # "Elem x y z vx vy vz" so we can warn clearly that those velocity
                # columns are ignored for runtime state initialization. Velocity/state
                # restoration must use RST via load_state/restart + rst_file.
                inline_velocity_detected = False
                for line in tokens:
                    m = atom_pattern.match(line)
                    if not m:
                        raise ValueError(f"Invalid element or coordinate line: '{line}'")

                    parts = line.split()
                    if len(parts) not in (4, 7):
                        raise ValueError(
                            "Inline atomic lines must be either 'Elem x y z' or "
                            "'Elem x y z vx vy vz'. "
                            f"Offending line: '{line}'"
                        )
                    if len(parts) == 7:
                        inline_velocity_detected = True

                    elem = m.group(1)
                    x = float(m.group(2))
                    y = float(m.group(3))
                    z = float(m.group(4))
                    elements.append(elem)
                    coords.append((x, y, z))

                # Create Atoms object
                atoms = Atoms(symbols=elements, positions=np.array(coords, dtype=np.float64))

                # Store charge and multiplicity if provided
                if charge is not None:
                    atoms.info['charge'] = charge
                if mult is not None:
                    atoms.info['mult'] = mult
                    atoms.info['spin'] = (mult - 1) / 2

                # Apply PBC if specified
                if self.pbc is not None:
                    from ase.cell import Cell
                    atoms.set_pbc([True, True, True])
                    # PBC format: [a, b, c, alpha, beta, gamma]
                    atoms.set_cell(Cell.fromcellpar(self.pbc))

                atoms_list.append(atoms)

                group_counter += 1
                info_message.append(f"\nGroup {group_counter} (inline)\n")
                if charge is not None and mult is not None:
                    info_message.append(f"Charge: {charge}, Multiplicity: {mult}\n")
                if inline_velocity_detected:
                    info_message.append(
                        "WARNING: Detected inline/XYZ velocity columns (Elem x y z vx vy vz). "
                        "These velocities are ignored and will not be used as formal runtime state input.\n"
                    )
                    info_message.append(
                        "WARNING: MD will regenerate velocities at runtime. "
                        "To restore velocities/state, use RST via load_state/restart + rst_file.\n"
                    )
                info_message.append('-' * 20 + '\n')
                for i, (e, (x, y, z)) in enumerate(zip(elements, coords), start=1):
                    info_message.append(f"{i:<4} {e:<2} {x:>20.6f} {y:>20.6f} {z:>20.6f}\n")

            self.log_info(info_message)
            return atoms_list[0] if len(atoms_list) == 1 else atoms_list

        except (ValueError, TypeError) as e:
            self.log_info(info_message)
            self.log_error(str(e))
            raise



    def post_processing_command(self, post_processing: list, atoms: Union[Atoms, List[Atoms]]) -> Union[Atoms, List[Atoms]]:
        """
        Parse and apply post-processing commands (constraints and scans) to one or multiple Atoms objects.
        Supported commands:
            C i           -> Fix atom i
            B i j         -> Fix bond between atoms i and j
            A i j k       -> Fix angle between atoms i, j, k
            D i j k l     -> Fix dihedral between atoms i, j, k, l
            S ...         -> Scan command (only valid when jobtype == 'scan')

        Args:
            post_processing (list): List of post-processing commands from the input file.
            atoms (Atoms or List[Atoms]): ASE Atoms object(s) to which constraints will be applied.

        Returns:
            Atoms or List[Atoms]: Processed structure(s) with applied constraints.
        """

        # If multiple structures are provided, process each independently
        if isinstance(atoms, list):
            processed_list = []
            for idx, at in enumerate(atoms, start=1):
                self.log_info([f"\nProcessing post-processing commands for group {idx}...\n"])
                processed_list.append(self.post_processing_command(post_processing, at))
            return processed_list

        # Import ASE constraints here to avoid import errors if ASE is not installed globally
        from ase.constraints import FixAtoms, FixInternals

        # Initialize constraint counters
        constraint_counts = {
            'fixed_atoms': 0,
            'fixed_bonds': 0,
            'fixed_angles': 0,
            'fixed_dihedrals': 0,
            'scans': 0
        }

        info_message = []
        constraints = []

        try:
            for line in post_processing:
                tokens = line.strip().split()
                if not tokens:
                    continue

                cmd = tokens[0].upper()

                if cmd not in ['C', 'B', 'A', 'D', 'S']:
                    raise ValueError(f"Invalid post-processing command: {line.strip()}")

                # ---- Fix atom ----
                if cmd == 'C':
                    if len(tokens) != 2:
                        raise ValueError(f"C command requires 1 index: {line.strip()}")
                    index = int(tokens[1])
                    if index < 1 or index > len(atoms):
                        raise ValueError(f"Atom index {index} out of range for C command.")
                    constraints.append(FixAtoms(indices=[index - 1]))
                    constraint_counts['fixed_atoms'] += 1

                # ---- Fix bond ----
                elif cmd == 'B':
                    if len(tokens) != 3:
                        raise ValueError(f"B command requires 2 indices: {line.strip()}")
                    i1, i2 = int(tokens[1]), int(tokens[2])
                    if i1 < 1 or i2 < 1 or i1 > len(atoms) or i2 > len(atoms):
                        raise ValueError(f"Atom index out of range for B command: {line.strip()}")
                    distance = atoms.get_distance(i1 - 1, i2 - 1)
                    constraints.append(FixInternals(bonds=[[distance, [i1 - 1, i2 - 1]]]))
                    constraint_counts['fixed_bonds'] += 1

                # ---- Fix angle ----
                elif cmd == 'A':
                    if len(tokens) != 4:
                        raise ValueError(f"A command requires 3 indices: {line.strip()}")
                    i1, i2, i3 = int(tokens[1]), int(tokens[2]), int(tokens[3])
                    for i in [i1, i2, i3]:
                        if i < 1 or i > len(atoms):
                            raise ValueError(f"Atom index {i} out of range for A command.")
                    angle = atoms.get_angle(i1 - 1, i2 - 1, i3 - 1)
                    constraints.append(FixInternals(angles_deg=[[angle, [i1 - 1, i2 - 1, i3 - 1]]]))
                    constraint_counts['fixed_angles'] += 1

                # ---- Fix dihedral ----
                elif cmd == 'D':
                    if len(tokens) != 5:
                        raise ValueError(f"D command requires 4 indices: {line.strip()}")
                    i1, i2, i3, i4 = map(int, tokens[1:])
                    for i in [i1, i2, i3, i4]:
                        if i < 1 or i > len(atoms):
                            raise ValueError(f"Atom index {i} out of range for D command.")
                    dihedral = atoms.get_dihedral(i1 - 1, i2 - 1, i3 - 1, i4 - 1)
                    constraints.append(
                        FixInternals(dihedrals_deg=[[dihedral, [i1 - 1, i2 - 1, i3 - 1, i4 - 1]]])
                    )
                    constraint_counts['fixed_dihedrals'] += 1

                # ---- Scan command ----
                elif cmd == 'S':
                    if self.jobtype != 'scan':
                        raise ValueError("Scan command is only available for jobtype='scan'.")

                    # Parse numeric parameters, last two are step size and steps
                    try:
                        params = [int(x) if idx != len(tokens) - 2 else float(x) for idx, x in enumerate(tokens[1:], 1)]
                    except ValueError:
                        raise ValueError(f"Invalid numeric parameter in scan command: {line.strip()}")

                    # Validate positive indices
                    for idx, val in enumerate(params[:-2]):
                        if val <= 0:
                            raise ValueError(f"Atom indices must be positive in scan command: {line.strip()}")

                    # Store parsed scan constraints
                    if not hasattr(self, 'scan_constraints'):
                        self.scan_constraints = []
                    self.scan_constraints.append(params)
                    constraint_counts['scans'] += 1

            # ---- Print summary ----
            info_message.append('\n' + '='*70 + '\n')
            info_message.append('Constraints and Restraints Summary'.center(70) + '\n')
            info_message.append('='*70 + '\n')

            if constraint_counts['fixed_atoms'] > 0:
                info_message.append(f"Fixed atoms:      {constraint_counts['fixed_atoms']}\n")
            if constraint_counts['fixed_bonds'] > 0:
                info_message.append(f"Fixed bonds:      {constraint_counts['fixed_bonds']}\n")
            if constraint_counts['fixed_angles'] > 0:
                info_message.append(f"Fixed angles:     {constraint_counts['fixed_angles']}\n")
            if constraint_counts['fixed_dihedrals'] > 0:
                info_message.append(f"Fixed dihedrals:  {constraint_counts['fixed_dihedrals']}\n")
            if constraint_counts['scans'] > 0:
                info_message.append(f"Scan coordinates: {constraint_counts['scans']}\n")

            if sum(constraint_counts.values()) == 0:
                info_message.append("No constraints applied.\n")

            info_message.append('='*70 + '\n')

            # ---- Apply all constraints at once (important!) ----
            if constraints:
                atoms.set_constraint(constraints)

            self.log_info(info_message)
            return atoms

        except ValueError as e:
            self.log_info(info_message)
            self.log_error(str(e))
            raise

        except Exception as e:
            self.log_error(f"Unexpected error during post-processing: {str(e)}")
            raise
