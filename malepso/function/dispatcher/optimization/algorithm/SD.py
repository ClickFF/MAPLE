import numpy as np
from ase import Atoms
from .logger import *
from .DIIS import OptimizationStorage

def SD(atoms: Atoms, output: str, max_step_size= 0.2, maxiterations=128) -> int:
    """
    This function is used to perform the Steepest Descent optimization.

    Args:
        atoms: The ASE Atoms object to be optimized.
        output: The path to the output file.
        max_step_size: The maximum step size.
        maxiterations: The maximum number of iterations.
    """

    info_message = ['Running the Steepest Descent ...\n']
    log_info(info_message, output)

    diis_storage = OptimizationStorage()
    diis_flag = False
    

    iteration = 0
    while iteration < maxiterations or atoms.max_f > atoms.f_max_th:
        forces = atoms.get_forces()
        new_positions = atoms.get_positions() + max_step_size * forces
        atoms.set_positions(new_positions)
        diis_vectors.append(new_positions)
        diis_error_vectors.append(atoms.get_forces())
        max_force_component = np.sqrt((atoms.get_forces()**2).max())
        if len(diis_vectors) == 10:
            diis_vectors = []
            diis_error_vectors = []
            diis_flag = True
        if max_force_component < atoms.f_max_th:
            print("-" * 40)
            print("Convergence criteria met. Optimization finished.")
            return iteration
        iteration += 1
    return iteration
