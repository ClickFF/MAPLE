import os
import re
from typing import Any, List, Union

from ase import Atoms
import numpy as np

from .filereader import XYZReader

class InputReader():
    def __init__(self):
        self.input:str = None
        self.output:str = None
        self.error:bool = False
        self.gpuid:int = None

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

    def __call__(self, input_file_name: str, output_file_name: str = None) -> Union[Atoms, List[Atoms]]:
        """
        Read the input file, parse settings, molecular coordinates, and post-processing commands.
        Support multiple coordinate groups separated by a blank line or '&'.
        Allow arbitrary blank lines between sections without breaking parsing.

        Args:
            input_file_name (str): Path to the input file.
            output_file_name (str, optional): Path to the output file. Defaults to None.

        Returns:
            Atoms or List[Atoms]: ASE Atoms object if a single group is present,
                                  or a list of ASE Atoms objects if multiple groups are present.
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

            # Remove existing output file if present
            if os.path.exists(self.output):
                os.remove(self.output)

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
            # ------------------------------------------------------------------

            with open(self.input, 'r') as f:
                raw_lines = f.readlines()

            # Regex used to detect coordinate-like lines
            atom_line_re = re.compile(
                r'^\s*([A-Za-z][a-z]?)\s+'
                r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s+'
                r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s+'
                r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$'
            )

            def is_settings_line(s: str) -> bool:
                return s.lstrip().startswith('#')

            def is_xyz_ref(s: str) -> bool:
                return s.upper().startswith('XYZ ') and len(s.split(maxsplit=1)) == 2

            def is_coord_like(s: str) -> bool:
                if s == '' or s == '&':
                    return True
                if is_xyz_ref(s):
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
        self.settings_command(settings)

        # === Step 2: Parse coordinate section ===
        atoms_or_list = self.element_and_coordinates(molecules)

        # === Step 3: Apply post-processing if present ===
        if post_processing:
            if isinstance(atoms_or_list, list):
                processed_list = []
                for idx, atoms in enumerate(atoms_or_list, start=1):
                    self.log_info([f"\nApplying post-processing to group {idx}...\n"])
                    processed_list.append(self.post_processing_command(post_processing, atoms))
                atoms_or_list = processed_list
            else:
                atoms_or_list = self.post_processing_command(post_processing, atoms_or_list)

        return atoms_or_list


    def log_error(self, error_message: str) -> None:
        """
        Logs error messages to the output file.

        Args:
            error_message: The error message to log.
        """
        with open(self.output, 'a') as file:
            file.write(f"ERROR: {error_message}\n")

    def log_info(self, info_message: list) -> None:
        """
        Logs info messages to the output file.

        Args:
            info_message: The info message to log.
        """
        with open(self.output, 'a') as file:
            for info in info_message:   
                file.write(f"{info}")

    def settings_command(self, settings: list):
        """
        This function is used to preprocess the command.

        Args:
            settings: The command to be preprocessed.
        """

        keywords = ['model', 'gpuid', 'jobtype', 'd4']
        model_dict = {'ANI-2x':1, 'ANI-1x':2, 'ANI-1ccx':3, 'ANI-1xnr':4}
        jobtype_dict = {'opt':1, 'sp':2, 'scan':3, 'freq':4, 'ts':5}

        info_message = ['Preprocessing the settings ...\n']

        try:
            for line in settings:
                # Remove the comment indicator and split by '='
                parts = line[1:].split('=')
                if len(parts) != 2:
                    raise ValueError(f'Invalid format for setting: {line.strip()}')

                key = parts[0].strip()
                value = parts[1].strip()

                # Debug output
                # print(f"Processing setting: {key} = {value}")

                if key in keywords:

                    # Set the model
                    if key == 'model':
                        if self.model is not None:
                            raise ValueError('You set model multiple times.')
                        else:
                            if value in model_dict.keys():
                                self.model = model_dict[value]
                                info_message.append(f'Model: {value}.\n')
                            else:
                                raise ValueError(f'The model \'{value}\' is not recognized.')

                    # Set the gpuid
                    elif key == 'gpuid':
                        try:
                            self.gpuid = int(value)
                            info_message.append(f'GPU ID: {self.gpuid}.\n')
                        except ValueError:
                            raise ValueError(f'Invalid GPU ID: {value}')

                    # Set the jobtype
                    elif key == 'jobtype':
                        if self.jobtype is not None:
                            raise ValueError('You set jobtype multiple times.')
                        else:
                            if value in jobtype_dict.keys():
                                self.jobtype = jobtype_dict[value]
                                info_message.append(f'Job type: {value}.\n')
                            else:
                                raise ValueError(f'The job type \'{value}\' is not recognized.')
                    
                    # Set the DFT-D4 dispersion correction
                    elif key == 'd4':
                        if value.lower() == 'true':
                            self.d4 = True
                            info_message.append('DFT-D4 dispersion correction is enabled.\n')
                        elif value.lower() == 'false':
                            self.d4 = False
                            info_message.append('DFT-D4 dispersion correction is disabled.\n')
                        else:
                            raise ValueError(f'Invalid D4 setting: {value}')

                else:
                    raise ValueError(f'The setting: \'{line.strip()}\' is not recognized.')

            # Final checks after processing all settings
            if self.gpuid is None:
                info_message.append('GPU ID is not set. Using CPU to calculate.\n')

            if self.jobtype is None:
                raise ValueError('The jobtype is not set.')
        
        except ValueError as e:
            self.log_info(info_message)
            self.log_error(str(e))
            raise

        # Log final successful configuration
        self.log_info(info_message)


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
            List[Atoms]: if multiple structures are present
        """

        # Regex for atomic line: element + 3 floats (supports scientific notation)
        atom_pattern = re.compile(
            r'^\s*([A-Za-z][a-z]?)\s+'
            r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s+'
            r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s+'
            r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$'
        )

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

            for block in blocks:
                # Normalize tokens for this block
                tokens = [b.strip() for b in block if b.strip()]
                if not tokens:
                    continue

                # Case 1: the block contains only XYZ file references
                all_xyz = all(t.upper().startswith("XYZ ") for t in tokens)
                any_xyz = any(t.upper().startswith("XYZ ") for t in tokens)

                if all_xyz:
                    for xyz_line in tokens:
                        parts = xyz_line.split(maxsplit=1)
                        if len(parts) != 2:
                            raise ValueError(f"Invalid XYZ reference line: '{xyz_line}'")
                        file_path = parts[1]
                        atoms = XYZReader(file_path)  # robust reader
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
                for line in tokens:
                    m = atom_pattern.match(line)
                    if not m:
                        raise ValueError(f"Invalid element or coordinate line: '{line}'")
                    elem = m.group(1)
                    x = float(m.group(2))
                    y = float(m.group(3))
                    z = float(m.group(4))
                    elements.append(elem)
                    coords.append((x, y, z))

                atoms = Atoms(symbols=elements, positions=np.array(coords, dtype=np.float64))
                atoms_list.append(atoms)

                group_counter += 1
                info_message.append(f"\nGroup {group_counter} (inline)\n")
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
            S ...         -> Scan command (only valid when jobtype == 3)

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

        info_message = ['\nApplying constraints and restraints ...\n']
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
                    info_message.append(f"Fixing atom {index}.\n")

                # ---- Fix bond ----
                elif cmd == 'B':
                    if len(tokens) != 3:
                        raise ValueError(f"B command requires 2 indices: {line.strip()}")
                    i1, i2 = int(tokens[1]), int(tokens[2])
                    if i1 < 1 or i2 < 1 or i1 > len(atoms) or i2 > len(atoms):
                        raise ValueError(f"Atom index out of range for B command: {line.strip()}")
                    distance = atoms.get_distance(i1 - 1, i2 - 1)
                    constraints.append(FixInternals(bonds=[[distance, [i1 - 1, i2 - 1]]]))
                    info_message.append(f"Fixing bond between atoms {i1} and {i2} (distance = {distance:.4f}).\n")

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
                    info_message.append(
                        f"Fixing angle between atoms {i1}-{i2}-{i3} (angle = {angle:.4f} deg).\n"
                    )

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
                    info_message.append(
                        f"Fixing dihedral between atoms {i1}-{i2}-{i3}-{i4} (dihedral = {dihedral:.4f} deg).\n"
                    )

                # ---- Scan command ----
                elif cmd == 'S':
                    if self.jobtype != 3:
                        raise ValueError("Scan command is only available for jobtype=3 (scan).")

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

                    scan_type = {4: "dihedral", 3: "angle", 2: "bond"}.get(len(params) - 2, "unknown")
                    steps = int(params[-1])
                    atom_indices = params[:-2]
                    info_message.append(
                        f"Defined scan of type {scan_type} on atoms {atom_indices} for {steps} steps.\n"
                    )

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

