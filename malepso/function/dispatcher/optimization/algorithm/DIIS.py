import numpy as np
from ase import Atoms
from .logger import *

g_au = 27.211386024367243

def DIIS(atoms: Atoms, output: str, memory = 100, max_step_size= 0.2, maxiterations=128) -> int:
    """
    This function is used to perform the DIIS optimization. I still need to add proper logging, test and optimize 

    Args:
        atoms: The ASE Atoms object to be optimized.
        output: The path to the output file.
        memory: The number of previous error vectors to remember. (Default: 10)
        maxiteration: The maximum number of iterations. (Default: 128)

    Returns:
        An integer indicating the optimization status (e.g., 0 for success).
    """
    info_message = ['Running the DIIS ...\n']
    if max_step_size > 1.0:
        info_message.append(f'You are using a much too large value for \
        the maximum step size: {max_step_size} Angstrom')
    log_info(info_message, output)
    
    #Initial setup
    energy = atoms.get_potential_energy()
    iteration = 0
    diis_error_vectors = []
    diis_x_vectors = []
    diis_x_vectors.append(atoms.get_positions())
    forces = atoms.get_forces()
    diis_error_vectors.append(forces)

    # check initial convergence
    max_force_component = np.sqrt((forces**2).max())
    if max_force_component < atoms.f_max_th:
        print("-" * 40)
        print("Convergence criteria met. Optimization finished.")
        return iteration
    
    # Steepest Descent for matrix entry 2, check convergence 2
    new_positions = atoms.get_positions() + max_step_size * forces
    atoms.set_positions(new_positions)
    diis_x_vectors.append(new_positions.copy())
    new_forces = atoms.get_forces()
    diis_error_vectors.append(new_forces)
    max_force_component = np.sqrt((new_forces**2).max())
    iteration += 1
    if max_force_component < atoms.f_max_th:
        print("-" * 40)
        print("Convergence criteria met. Optimization finished.")
        return iteration
    
    print(diis_x_vectors)

    #perform DIIS for first two geometries to find third and so on until it converges -optimized geometry- or exceeds maxiterations
    while iteration < maxiterations and max_force_component > atoms.f_max_th:
        m = len(diis_error_vectors)

        # 1. Build the B matrix.
        # This is a square matrix where B[i, j] is the dot product of
        # error vector i and error vector j.
        B = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                # Flatten the vectors
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

        # 4. Update the positions
        new_position = np.zeros_like(diis_x_vectors[0]) 
        for i, c in enumerate(coeff):
            new_position += c * diis_x_vectors[i]  # Element-wise scaling and addition that AI gave me when I got a dimension  error

        # Calculate actual step
        step = new_position - atoms.get_positions()
        step_length = np.sqrt((step**2).sum())

        # Limit step size
        if step_length > max_step_size:
            step = step * (max_step_size / step_length)
            new_position = atoms.get_positions() + step

        atoms.set_positions(new_position)
        
        diis_x_vectors.append(new_position.copy())

        # 5. Update the error vectors.
        new_forces = atoms.get_forces()
        diis_error_vectors.append(new_forces)

        if len(diis_x_vectors) > memory:
            diis_x_vectors.pop(0)
            diis_error_vectors.pop(0)
        
        # 6. Update the energy for logging
        energy = atoms.get_potential_energy(force_consistent=True)

        # 6. Update the forces for convergence check
        max_force_component = np.sqrt((new_forces**2).max())
        
        # DEBUG: Check what's happening
        print(f"Iter {iteration+1}: Force = {max_force_component:.8f}, Threshold = {atoms.f_max_th:.8f}")

        if max_force_component <= atoms.f_max_th:
            print("-" * 40)
            print(f"DIIS converged after {iteration+1} iterations!")
            print(f"Final max force: {max_force_component:.6f} < {atoms.f_max_th:.6f}")
            log_info(info_message, output)
            return iteration+1  
            break


        # ===== LOG ITERATION ( LIKE LBFGS) =====
        iter_header = f"Iteration: {iteration}"
        info_message = ['\n' + '-' * 70 + '\n', f'{iter_header.center(70)}\n\n']

        info_message.append(f'\n{"Coordinates".center(70)}\n')
        info_message.append('-' * 70)
        info_message.append('\n')

        for atom_index, atom in enumerate(atoms):
            element_type = atom.symbol 
            coord = atom.position 
            info_message.append(f"{atom_index:<4} {element_type:<2} {coord[0]:>20.4f} {coord[1]:>20.4f} {coord[2]:>20.4f}\n")
        
        ##info_message.append(f"\n\nEnergy:                {energy/g_au:>12.6f} Convergence criteria  Is converged \n")
       
        # # Force convergence check
        # if atoms.max_f > atoms.f_max_th:
        #     info_message.append(f"Maximum Force:         {atoms.max_f/g_au:>12.6f} {atoms.f_max_th/g_au:>12.6f}                No\n")
        # else:
        #     info_message.append(f"Maximum Force:         {atoms.max_f/g_au:>12.6f} {atoms.f_max_th/g_au:>12.6f}                Yes\n")
        #
        # if atoms.rms_f > atoms.f_rms_th:
        #     info_message.append(f"RMS Force:             {atoms.rms_f/g_au:>12.6f} {atoms.f_rms_th/g_au:>12.6f}                No\n")
        # else:
        #     info_message.append(f"RMS Force:             {atoms.rms_f/g_au:>12.6f} {atoms.f_rms_th/g_au:>12.6f}                Yes\n")
        #
        # if atoms.max_dp > atoms.dp_max_th:
        #     info_message.append(f"Maximum Displacement:  {atoms.max_dp:>12.6f} {atoms.dp_max_th:>12.6f}                No\n")
        # else:
        #     info_message.append(f"Maximum Displacement:  {atoms.max_dp:>12.6f} {atoms.dp_max_th:>12.6f}                Yes\n")
        #
        # if atoms.rms_dp > atoms.dp_rms_th:
        #     info_message.append(f"RMS Displacement:      {atoms.rms_dp:>12.6f} {atoms.dp_rms_th:>12.6f}                No\n")
        # else:
        #     info_message.append(f"RMS Displacement:      {atoms.rms_dp:>12.6f} {atoms.dp_rms_th:>12.6f}                Yes\n")
        #
        # # Write to output file
        log_info(info_message, output)
        iteration += 1

    return iteration
 

   




