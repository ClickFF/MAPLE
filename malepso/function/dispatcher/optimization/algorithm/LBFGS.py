import numpy as np

from ase import Atoms

from .logger import *

###############################
g_au=27.211386024367243

def LBFGS(atoms:Atoms, output:str, use_line_search=False, memory=100, curvature=70.0,
	maxstep=0.2, maxiteration=128,) -> int:
	"""
	This function is used to perform the L-BFGS optimization.

	Args:
		mol: The ASE Atoms object to be optimized.
		use_line_search: Whether to use line search or not. (Default: False)
		memory: The number of previous steps to remember. (Default: 100)
		curvature: The initial approximation of the inverse Hessian. (Default: 70.0)
		maxstep: The maximum step size. (Default: 0.2)
		maxiteration: The maximum number of iterations. (Default: 10)
	
	Returns:
	
	"""
	info_message = []

	if maxstep > 1.0:
			info_message.append(f'You are using a much too large value for \
			the maximum step size: {maxstep} Angstrom')

	
	"""
	Variables:
	H0: Initial approximation of inverse Hessian 1./70. is to emulate the behaviour of BFGS.
	alph: The step size.
	p: The search direction of LBFGS.
	iteration: The number of iterations.
	s: The list to store r-r0.
	y: The list to store g-g0. (g: gradient)
	rho: The list to store 1/(r-r0)(g-g0).
	r0: The initial positions of the atoms.
	r: The current positions of the atoms.
	dr: The difference between the current and the previous positions.
	e: The potential energy of the atoms.
	f: The forces acting on the atoms.
	f0: The forces acting on the atoms.
	q: The search direction of LBFGS. (q = -f)

	"""
	
	# Initial approximation of inverse Hessian 1./70. is to emulate the behaviour of BFGS. 

	H0 = 1. / curvature
	alph=1.0
	p = None  
	iteration = 0
	s = []
	y = []       
	rho = []
	r0 = atoms.get_positions()
	r=r0*1.0
	dr = atoms.get_positions()-r0
	e  = atoms.get_potential_energy(force_consistent=True)
	f  = atoms.get_forces()

	f0 = f*1.0
	q  = -f*1.0

	convergence = False
	a = np.empty((memory,), dtype=np.float64)



	while not convergence and iteration < maxiteration:
		   
		if iteration > 0:
			s0 = alph*dr
			s.append(s0)
			y0 = f0 - f
			y.append(y0)

			rho0 = 1.0 / (y0*s0).sum()
			rho.append(rho0)

		if iteration > memory:
			s.pop(0)
			y.pop(0)
			rho.pop(0)
		   
		loopmax = np.min([memory, iteration])

		# LBFGS algorithm:
		q=-f*1.0
	   		
		for i in range(loopmax - 1, -1, -1):
			a[i] = rho[i] * (s[i]* q).sum()
			q -= a[i] * y[i]
		
		z = H0 * q

		for i in range(loopmax):
			b = rho[i] * (y[i]*z).sum()
			z += s[i] * (a[i] - b)
		
		p = -1.0*z
		dr=p*1.0
		longest_step =np.max((dr**2).sum(1)**0.5)
		if use_line_search==True:
			raise NotImplementedError('Line search not implemented')
		   
		elif longest_step >= maxstep:
			dr *= maxstep / longest_step	 		
		
		r0 = r*1.0
		e0 = e*1.0
		f0 = f*1.0
		iteration += 1
		atoms.set_positions(r0+alph*dr) 
		r = atoms.get_positions()
		f = atoms.get_forces()
		e = atoms.get_potential_energy(force_consistent=True)

		# Convergence criteria:               
		atoms.max_dp = alph * abs(dr).max()
		atoms.rms_dp = alph *np.sqrt((dr**2).sum()/dr.size*3)
		atoms.max_f = abs(f).max()
		atoms.rms_f = np.sqrt((f**2).sum()/dr.size*3)

		
		# Log the information:
		iter = f"Iteration: {iteration}"
		info_message = ['\n' + '-' * 70 + '\n',f'{iter.center(70)}\n\n']

		info_message.append(f'\n{"Coordinates".center(70)}\n')
		info_message.append('-' * 70)
		info_message.append('\n')

		for atom_index, atom in enumerate(atoms):
			element_type = atom.symbol 
			coord = atom.position 

			info_message.append(f"{atom_index:<4} {element_type:<2} {coord[0]:>20.4f} {coord[1]:>20.4f} {coord[2]:>20.4f}\n")
		
		
		info_message.append(f"\n\nEnergy:                {e/g_au:>12.6f} Convergence criteria  Is converged \n")

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

		log_info(info_message,output)

		#con_flag = convergence_fun(mol)
		if atoms.max_f<=atoms.f_max_th and atoms.rms_f<=atoms.f_rms_th and atoms.max_dp <=atoms.dp_max_th and atoms.rms_dp<=atoms.dp_rms_th:
			return iteration 
	return  iteration          	    
#################################################################################


	 

