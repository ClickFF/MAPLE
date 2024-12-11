import os
import re
from typing import Any, List

from ase import Atoms
import numpy as np

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

    def __call__(self, input_file_name: str, output_file_name: str = None) -> Atoms:
        """
        The class is used to read the input file.

        Args:
            input_file_name: The path to the input file.
            output_file_name: The path to the output file. (Default: None)
        """

        try:
            # Determine the input and output file names.
            if isinstance(input_file_name, str):
                self.input = os.path.abspath(input_file_name)
            else:
                raise TypeError('The input_file_name should be a string.')
            
            if output_file_name is not None:
                if isinstance(output_file_name, str):
                    self.output = os.path.abspath(output_file_name)
                else:
                    raise TypeError('The output_file_name should be a string.')
            else:
                self.output = os.path.splitext(self.input)[0] + '.out'
                self.output = os.path.abspath(self.output)

            if os.path.exists(self.output):
                os.remove(self.output)

            # Separate the input into three parts: settings, molecules, and post-processing.
            settings = []
            settings_flag = False
            molecules = []
            molecules_flag = False
            post_processing = []
            post_processing_flag = False

            with open(self.input, 'r') as f:
                lines = f.readlines()

            for line in lines:
                if line.startswith('#') and not settings_flag:
                    settings.append(line)
                elif not settings_flag and line.strip() == '':
                    settings_flag = True  # All settings are read.

                elif not molecules_flag and settings_flag:
                    if line.strip() == '':
                        molecules_flag = True  # All molecules are read.
                    else:
                        molecules.append(line)
                
                elif not post_processing_flag and settings_flag and molecules_flag:
                    if line.strip() == '':
                        post_processing_flag = True  # All post-processing commands are read.
                    else:
                        post_processing.append(line.strip())



            if not self.input:
                raise ValueError('Unrecognized input file.')

            if not self.output:
                raise ValueError('Unrecognized output file.')

            if not settings_flag:
                raise ValueError('There is something wrong with file reading. Cannot find the settings.')
            
            if not molecules_flag:
                raise ValueError('There is something wrong with file reading. Cannot find the coordinates or elements.')
            
            if not post_processing_flag:
                raise ValueError('There is something wrong with file reading. Cannot find the post-processing commands\n \
                    or you should leave at least two empty lines at the end of the file.')
        
        except (AssertionError, TypeError, ValueError) as e:
            # Handle exceptions, log them, then raise
            self.log_error(str(e))
            raise

        # Process settings and write to the output file
        self.settings_command(settings)

        # Process molecules and write to the output file
        atoms = self.element_and_coordinates(molecules)

        if post_processing:
            # Process post-processing commands and write to the output file
            self.post_processing_command(post_processing,atoms)

        return atoms

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




    def element_and_coordinates(self, molecules: List[str]) -> Atoms:
        """
        This function is used to extract the elements and the coordinates.
        
        Args:
            molecules: The list of text lines to be processed.

        Returns:
            The ASE Atoms object containing the elements and the coordinates.
        """

        # Regular expression to match the lines like "H -11.2 0.9 0.008"
        atom_pattern = r'^\s*([A-Za-z]+)\s+([-+]?\d*\.\d+|\d+)\s+([-+]?\d*\.\d+|\d+)\s+([-+]?\d*\.\d+|\d+)\s*$'

        # Initializing log message
        info_message = [f'\n{"Coordinates".center(70)}\n', '*' * 70 + '\n']
        elements = []
        coordinates = []

        try:
            atom_index = 1
            for line in molecules:
                match = re.match(atom_pattern, line)
                if match:
                    # Append element symbol
                    elements.append(match.group(1))

                    # Append coordinates as a tuple of float32
                    coord = (
                        np.float32(match.group(2)), 
                        np.float32(match.group(3)), 
                        np.float32(match.group(4))
                    )
                    coordinates.append(coord)

                    # Add formatted line info to log
                    info_message.append(f"{atom_index:<4} {match.group(1):<2} {coord[0]:>20.4f} {coord[1]:>20.4f} {coord[2]:>20.4f}\n")
                else:
                    raise ValueError(f'Invalid Element or Coordination: {line.strip()}')
                atom_index += 1

            # Create ASE Atoms object
            atoms = Atoms(symbols=elements, positions=coordinates)
        except (ValueError, TypeError) as e:
            self.log_info(info_message)
            self.log_error(str(e))
            raise

        self.log_info(info_message)

        return atoms
    
    def post_processing_command(self, post_processing: list, atoms:Atoms) -> None:
        """
        This function is used to preprocess the post-processing commands.

        Args:
            post_processing: The list of post-processing commands to be preprocessed.
        """

        post_processing_keywords = ['C','B','A','D','S']
        # C: Coordinates fix
        # B: Bond fix
        # A: Angle fix
        # D: Dihedral fix
        # S: Scan

        constraints = []
        info_message = ['\nPerforming the constraints and restraints ...\n']
        try:
            from ase.constraints import FixAtoms
            from ase.constraints import FixInternals

            for line in post_processing:
                if line.strip().split()[0] not in post_processing_keywords:
                    raise ValueError(f'Invalid post-processing command: {line.strip()}')
                
                elif line.strip().split()[0] == 'C':
                    if len(line.strip().split()) != 2:
                        raise ValueError(f'There should be only one atomic index for the command: {line.strip()}')
                    
                    index = int(line.strip().split()[1])

                    constraints.append(FixAtoms(indices=[index-1]))

                elif line.strip().split()[0] == 'B':
                    if len(line.strip().split()) != 3:
                        raise ValueError(f'There should be only two atomic indices for the command: {line.strip()}')

                    index1 = int(line.strip().split()[1])
                    index2 = int(line.strip().split()[2])

                    # Atomic indices in ASE start from 0
                    distance = atoms.get_distance(index1-1, index2-1)    
                    constraints.append(FixInternals(bonds=[[distance,[index1-1, index2-1]]]))
                    info_message.append(f'Fixing bond between atoms {index1} and {index2} with distance of {distance}.\n')
                
                elif line.strip().split()[0] == 'A':
                    if len(line.strip().split()) != 4:
                        raise ValueError(f'There should be only three atomic indices for the command: {line.strip()}')

                    index1 = int(line.strip().split()[1])
                    index2 = int(line.strip().split()[2])
                    index3 = int(line.strip().split()[3])

                    degree = atoms.get_angle(index1-1,index2-1,index3-1)
                    angle1 = [degree, [index1-1, index2-1, index3-1]]
                    constraints.append(FixInternals(angles_deg=[angle1]))

                    info_message.append(f'Fixing angle between atoms {index1}, {index2}, and {index3} with angle of {degree}.\n')

                elif line.strip().split()[0] == 'D':
                    if len(line.strip().split()) != 5:
                        raise ValueError(f'There should be only four atomic indices for the command: {line.strip()}')

                    index1 = int(line.strip().split()[1])
                    index2 = int(line.strip().split()[2])
                    index3 = int(line.strip().split()[3])
                    index4 = int(line.strip().split()[4])

                    dihedral_indices = [index1-1, index2-1, index3-1, index4-1]

                    dihedral = atoms.get_dihedral(*dihedral_indices)
                    constraints.append(FixInternals(dihedrals_deg=[[dihedral, dihedral_indices]]))
                    
                    info_message.append(f'Fixing dihedral between atoms {index1}, {index2}, {index3}, and {index4} with dihedral of {dihedral}.\n')

                elif line.strip().split()[0] == 'S':
                    raise NotImplementedError('The scan command is not implemented yet.')
                

        except ValueError as e:
            self.log_info(info_message)
            self.log_error(str(e))
            raise

        try:
            for constraint in constraints:
                atoms.set_constraint(constraint)
        except Exception as e:
            self.log_error(str(e))
            raise

        self.log_info(info_message)
        return atoms
