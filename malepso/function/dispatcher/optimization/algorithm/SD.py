import numpy as np
from ase import Atoms
from .logger import *
from .DIIS import OptimizationStorage
#from .DIIS import DIIS

g_au = 27.211386024367243

def SD(atoms: Atoms, output: str, max_step_size=0.2, maxiterations=128) -> int:
    """
    This function is used to perform the Steepest Descent optimization.

    Args:
        atoms: The ASE Atoms object to be optimized.
        output: The path to the output file.
        max_step_size: The maximum step size.
        maxiterations: The maximum number of iterations.
    """

    info_message = ['Running the Steepest Descent ...\n']
    if max_step_size > 1.0:
        info_message.append(f'You are using a much too large value for \
        the maximum step size: {max_step_size} Angstrom')
    log_info(info_message, output)
   

    diis_storage = OptimizationStorage()
    iteration = 0
    diis_counter = 0

    while iteration < maxiterations:
        forces = atoms.get_forces()
        
        # Calculate convergence criteria
        atoms.max_f = abs(forces).max()
        atoms.rms_f = np.sqrt((forces**2).sum() / forces.size)
        
        # Check convergence
        if atoms.max_f <= atoms.f_max_th and atoms.rms_f <= atoms.f_rms_th:
            return iteration
            
        # Store for DIIS
        diis_storage.diis_x_vectors.append(atoms.get_positions().copy())
        diis_storage.diis_error_vectors.append(forces.copy())
        diis_storage.iteration = iteration
        diis_counter += 1
        
        # Call DIIS every 10 iterations
        if diis_counter >= 10:
            from .DIIS import DIIS
            DIIS(atoms, output, max_step_size=max_step_size, maxiterations=maxiterations, storage=diis_storage)
            diis_counter = 0  # Reset counter to continue SD
            diis_storage.reset()
        
    
        #Steepest Descent 
        step = max_step_size * forces
        step_length = np.sqrt((step**2).sum())

        if step_length >= max_step_size:
            step *= max_step_size / step_length

        new_positions = atoms.get_positions() + step
        atoms.set_positions(new_positions)
       
        
        # Calculate displacements
        atoms.max_dp = abs(step).max()
        atoms.rms_dp = np.sqrt((step**2).sum() / step.size)
        
        # Update energy
        energy = atoms.get_potential_energy(force_consistent=True)
        
        iteration += 1
        
        # Log iteration (same format as LBFGS)
        iter_header = f"Iteration: {iteration}"
        info_message = ['\n' + '-' * 70 + '\n', f'{iter_header.center(70)}\n\n']
        
        info_message.append(f'\n{"Coordinates".center(70)}\n')
        info_message.append('-' * 70)
        info_message.append('\n')
        
        for atom_index, atom in enumerate(atoms):
            element_type = atom.symbol 
            coord = atom.position 
            info_message.append(f"{atom_index:<4} {element_type:<2} {coord[0]:>20.4f} {coord[1]:>20.4f} {coord[2]:>20.4f}\n")
        
        info_message.append(f"\n\nEnergy:                {energy/g_au:>12.6f} Convergence criteria  Is converged \n")
        
        if atoms.max_f > atoms.f_max_th:
            info_message.append(f"Maximum Force:         {atoms.max_f/g_au:>12.6f} {atoms.f_max_th/g_au:>12.6f}                No\n")
        else:
            info_message.append(f"Maximum Force:         {atoms.max_f/g_au:>12.6f} {atoms.f_max_th/g_au:>12.6f}                Yes\n")
        
        if atoms.rms_f > atoms.f_rms_th:
            info_message.append(f"RMS Force:             {atoms.rms_f/g_au:>12.6f} {atoms.f_rms_th/g_au:>12.6f}                No\n")
        else:
            info_message.append(f"RMS Force:             {atoms.rms_f/g_au:>12.6f} {atoms.f_rms_th/g_au:>12.6f}                Yes\n")
        
        if atoms.max_dp > atoms.dp_max_th:
            info_message.append(f"Maximum Displacement:  {atoms.max_dp:>12.6f} {atoms.dp_max_th:>12.6f}                No\n")
        else:
            info_message.append(f"Maximum Displacement:  {atoms.max_dp:>12.6f} {atoms.dp_max_th:>12.6f}                Yes\n")
        
        if atoms.rms_dp > atoms.dp_rms_th:
            info_message.append(f"RMS Displacement:      {atoms.rms_dp:>12.6f} {atoms.dp_rms_th:>12.6f}                No\n")
        else:
            info_message.append(f"RMS Displacement:      {atoms.rms_dp:>12.6f} {atoms.dp_rms_th:>12.6f}                Yes\n")
        
        log_info(info_message, output)
        
    return iteration

