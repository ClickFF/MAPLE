
import itertools
import copy

from ase import Atoms
from ase.constraints import FixInternals

from ..jobABC import JobABC

class Scan(JobABC):
	def __init__(self, output:str, atoms:Atoms, method:str='RFO',criteria:int=2, constraints:list=None):
		super().__init__(output)
		self.atoms = atoms
		self.method = method
		self.output = output
		self.constraints = self.convert_constraints(constraints)

		if criteria == 1:
			self.atoms.f_max_th=0.00045*27.211386024367243
			self.atoms.f_rms_th=0.0003*27.211386024367243
			self.atoms.dp_max_th=0.0018   
			self.atoms.dp_rms_th=0.0012
		elif criteria == 2: # loose criteria
			self.atoms.f_max_th=0.01*27.211386024367243
			self.atoms.f_rms_th=0.006*27.211386024367243
			self.atoms.dp_max_th=0.04   
			self.atoms.dp_rms_th=0.025

	def convert_constraints(self, original_constraints):
		"""
		Convert the original constraints to a list of dictionaries.

		:param original_constraints: The original constraints list, each constraint is a list of atoms and steps, for example:
									[[atom1, atom2, step, steps], 
									[atom1, atom2, atom3, step, steps],
									[atom1, atom2, atom3, atom4, step, steps]]
		:return: The converted constraints list, each constraint is a dictionary, for example:
				[
					{
						'type': 'distance',
						'atoms': [0, 1],
						'step': -0.1,
						'steps': 10
					},
					{
						'type': 'angle',
						'atoms': [1, 2, 3],
						'step': 5,
						'steps': 20
					},
					{
						'type': 'dihedral',
						'atoms': [1, 2, 3, 4],
						'step': 2,
						'steps': 15
					}
				]
		"""
		converted = []
		for constraint in original_constraints:
			length = len(constraint)
			if length == 4:
				# Distance constraint: [atom1, atom2, step, steps]
				atom1, atom2, step, steps = constraint
				converted.append({
					'type': 'distance',
					'atoms': [atom1, atom2],
					'step': step,
					'steps': steps
				})
			elif length == 5:
				# Angle constraint: [atom1, atom2, atom3, step, steps]
				atom1, atom2, atom3, step, steps = constraint
				converted.append({
					'type': 'angle',
					'atoms': [atom1, atom2, atom3],
					'step': step,
					'steps': steps
				})
			elif length == 6:
				# Dihedral constraint: [atom1, atom2, atom3, atom4, step, steps]
				atom1, atom2, atom3, atom4, step, steps = constraint
				converted.append({
					'type': 'dihedral',
					'atoms': [atom1, atom2, atom3, atom4],
					'step': step,
					'steps': steps
				})
			else:
				raise ValueError(f"Unsupported constraint format with length {length}: {constraint}")
		return converted

	def generate_scan_values(self):
		"""
		Generate the scan values for each constraint.

		Args:
			original_constraints: The original constraints list, each constraint is a list of atoms and steps, for example:
								[[atom1, atom2, step, steps], 
								[atom1, atom2, atom3, step, steps],
								[atom1, atom2, atom3, atom4, step, steps]]
		Returns:
			The scan values for each constraint, for example:
			[
				[1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
				[100.0, 105.0, 110.0, 115.0, 120.0],
				[-180.0, -178.0, -176.0, -174.0, -172.0]
			]
		"""
		scan_values = []
		for constraint in self.constraints:
			c_type = constraint['type']
			step = constraint['step']
			steps = constraint['steps']
			if c_type == 'distance':
				atom1, atom2 = constraint['atoms']
				initial_distance = self.atoms.get_distance(atom1-1, atom2-1)
				values = [initial_distance + step * i for i in range(steps + 1)]
			elif c_type == 'angle':
				atom1, atom2, atom3 = constraint['atoms']
				initial_angle = self.atoms.get_angle(atom1-1, atom2-1, atom3-1)
				values = [initial_angle + step * i for i in range(steps + 1)]
			elif c_type == 'dihedral':
				atom1, atom2, atom3, atom4 = constraint['atoms']
				initial_dihedral = self.atoms.get_dihedral(atom1-1, atom2-1, atom3-1, atom4-1)
				values = [initial_dihedral + step * i for i in range(steps + 1)]
			else:
				raise ValueError(f"Unsupported constraint type: {c_type}")
			scan_values.append(values)
		return scan_values

	def set_constraints(self, current_values):
		"""
		Set the constraints for the current scan values.

		Args:
			current_values: The current values for each constraint, for example:
							[1.0, 100.0, -180.0]
		Returns:
			The FixInternals object with the current constraints set.
		"""
		bonds = []
		angles = []
		dihedrals = []

		for idx, constraint in enumerate(self.constraints):
			c_type = constraint['type']
			atoms_involved = constraint['atoms']
			value = current_values[idx]

			atoms_involved = [atom - 1 for atom in atoms_involved]  # Convert to 0-based index

			if c_type == 'distance':
				# Distance constraint
				bonds.append([value, atoms_involved])
			elif c_type == 'angle':
				# Angle constraint
				angles.append([value, atoms_involved])
			elif c_type == 'dihedral':
				# Dihedral constraint
				dihedrals.append([value, atoms_involved])
			else:
				raise ValueError(f"Unsupported constraint type: {c_type}")

		# Create the FixInternals object
		fix_internals = FixInternals(
			bonds=bonds if bonds else None,
			angles_deg=angles if angles else None,
			dihedrals_deg=dihedrals if dihedrals else None
		)
		return fix_internals

	def run_scan(self):
		# Generate all scan values
		scan_values = self.generate_scan_values()
		# Generate all combinations
		all_combinations = list(itertools.product(*scan_values))
		total = len(all_combinations)
		info_message =[f"\nTotal combinations to scan: {total}\n"]
		self.log_info(info_message)

		atoms_copy = copy.deepcopy(self.atoms)

		for i, combo in enumerate(all_combinations):
			info_message = ['\n']
			# Deep copy the atoms object
			self.atoms = atoms_copy

			# Set the constraints for the current scan values
			constrain = self.set_constraints(combo)
			self.atoms.set_constraint(constrain)

			combo = [f"{val:.2f}" for val in combo]

			info_message.append('-' * 70)
			info_message.append(f'\n{f"Scanning combination {i+1}/{total}: {combo}".center(70)}')
			info_message.append(f'\n{"Coordinates".center(70)}\n')
			info_message.append('-' * 70)
			info_message.append('\n')
			self.log_info(info_message)

			self.method = 'LBFGS'
			# Run the optimization
			if self.method.upper() == 'LBFGS':
				from .algorithm import LBFGS
				LBFGS(self.atoms, output=self.output)
			elif self.method.upper() == 'RFO':
				from .algorithm import RFO
				RFO(self.atoms, output=self.output)
			else:
				raise ValueError(f"Unsupported optimization method: {self.method}")
			
		info_message = ['\n\n' + '-' * 70 + '\n', f'{"Normal Termination".center(70)}\n\n']
		self.log_info(info_message)


	def run(self):
		self.run_scan()


