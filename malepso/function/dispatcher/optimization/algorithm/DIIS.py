import numpy as np
from ase import Atoms
from .logger import *

def DIIS(atoms: Atoms, output: str, memory = 100, max_step_size= 0.2) -> int:
    """
    This function is used to perform the DIIS optimization.

    Args:
        atoms: The ASE Atoms object to be optimized.
        output: The path to the output file.
        memory: The number of previous error vectors to remember. (Default: 10)
        maxiteration: The maximum number of iterations. (Default: 128)

    Returns:
        An integer indicating the optimization status (e.g., 0 for success).
    """
    info_message = ['Running the DIIS ...\n']
    diis_error_vectors = []
    diis_x_vectors = []
    diis_x_vectors.append(atoms.get_positions())
    forces = atoms.get_forces()
    diis_error_vectors.append(forces)

    ## Steepest Descent for matrix entry 2
    max_force_component = np.sqrt((forces**2).max())
    if max_force_component < atoms.f_max_th:
        print("-" * 40)
        print("Convergence criteria met. Optimization finished.")
        return
    
    new_positions = atoms.get_positions() + max_step_size * forces
    atoms.set_positions(new_positions)
    diis_x_vectors.append(new_positions.copy())
    print(diis_x_vectors)

    while len(diis_x_vectors) < memory and max_force_component > atoms.f_max_th:
        m = len(diis_error_vectors)

        # 1. Build the B matrix.
        # This is a square matrix where B[i, j] is the dot product of
        # error vector i and error vector j.
        B = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                # Flatten the vectors to ensure they are 1D for the dot product.
                B[i, j] = np.dot(diis_error_vectors[i].flat, diis_error_vectors[j].flat)

        # 2. Build the A matrix.
        A = np.zeros((m + 1, m + 1))
        A[:m, :m] = B  # Top-left part is the B matrix
        A[m, :m] = -1.0
        A[:m, m] = -1.0

        b = np.zeros(m + 1)
        b[m] = -1.0

        # 3. Solve the linear system A * x = b.
        x = np.linalg.solve(A, b)
        coeff = x[:m]

        # 4. Update the positions.
        new_position = np.dot(coeff, diis_x_vectors)
        atoms.set_positions(new_position)
        diis_x_vectors.append(new_position.copy())

        # 5. Update the error vectors.
        new_forces = atoms.get_forces()
        diis_error_vectors.append(new_forces)

        max_force_component = np.sqrt((new_forces**2).max())

    print(diis_x_vectors)




