import numpy as np
###############################
g_au=27.211386024367243

def func(self, x):
        """Objective function for use of the optimizers"""
        self.atoms.set_positions(x.reshape(-1, 3))
        self.function_calls += 1
        return self.atoms.get_potential_energy(
            force_consistent=self.force_consistent)
def LBFGS(mol,  use_line_search=False,memory=100,  curvature=70.0,  maxstep=0.2, maxiteration=10):
        """Parameters:

        """
 
        if maxstep > 1.0:
            raise ValueError('You are using a much too large value for ' +
                             'the maximum step size: %.1f Angstrom' %
                             maxstep)

        
        # Initial approximation of inverse Hessian 1./70. is to emulate the behaviour of BFGS. 
        H0 = 1. / curvature
        
        alph=1.0
       
        p = None  #search direction of LBFGS
 
        iteration = 0
# Store r-r0, g-g0, and 1/(r-r0)(g-g0) into s[], y[], rho[] list respectively. 
        s = []
        y = []       
        rho = []
        r0 = mol.get_positions()
        r=r0*1.0
        dr = mol.get_positions()-r0
        e  = mol.get_potential_energy(force_consistent=True)
        f  = mol.get_forces()
        f0 = f*1.0
        q  = -f*1.0
        log=1
        r_size = r0.size/3.0
###########convergence flag##################
        con_flag = 'NO'
        a = np.empty((memory,), dtype=np.float64)
###########iteration#########################
        while con_flag == 'NO' and iteration < maxiteration:
               
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
        	# ## The algorithm itself:
                q=-f*1.0
                g=-f*1.0
       	        for i in range(loopmax - 1, -1, -1):
            	    a[i] = rho[i] * (s[i]* q).sum()
            	    q -= a[i] * y[i]
                z = H0 * q

                for i in range(loopmax):
            	    b = rho[i] * (y[i]*z).sum()
            	    z += s[i] * (a[i] - b)
                #print('i:',loopmax)
       	        p = -1.0*z
                dr=p*1.0
                longest_step =np.max((dr**2).sum(1)**0.5)
                if use_line_search==True:
                   ls = LineSearch()
                   alph, e, e0, no_update = \
                     ls._line_search(mol, r, p, -f, e, e0, maxstep, c1=.0001,c2=.49, stpmax=50.)
                   if alph is None:
                      print('LBFGS_WS failed!, try LBFGS,CG_WS or CG_BS')
                      break
                elif longest_step >= maxstep:
                   dr *= maxstep / longest_step	 		
                
                r0 = r*1.0
                e0 = e*1.0
                f0 = f*1.0
                iteration += 1
                mol.set_positions(r0+alph*dr) 
                r = mol.get_positions()
                f = mol.get_forces()
                e = mol.get_potential_energy(force_consistent=True)
###########convergence_FLAG##############################
               
                mol.max_dp = alph * abs(dr).max()
                mol.rms_dp = alph *np.sqrt((dr**2).sum()/dr.size*3)
                mol.max_f = abs(f).max()
                mol.rms_f = np.sqrt((f**2).sum()/dr.size*3)
#########test print, could be removed ###################################
               # print('mol.t_to',mol.t_tor)
                #a=mol.get_dihedrals([mol.t_tor[:4]-1,mol.t_tor[4:8]-1])
               # bbb=mol.get_angles([mol.t_ang-1])
               # ccc=mol.get_distance(mol.t_bond[0]-1,mol.t_bond[1]-1)
               # print(iteration,aaa,bbb,ccc)
                print(iteration,e/g_au, mol.max_f/g_au, mol.max_dp, alph)
############################################################################################
                #con_flag = convergence_fun(mol)
                if mol.max_f<=mol.f_max_th and mol.rms_f<=mol.f_rms_th and mol.max_dp <=mol.dp_max_th and mol.rms_dp<=mol.dp_rms_th:
                    return iteration 
        return  iteration          	    
#################################################################################


     

