#!/usr/bin/env python
"""
Construct Model From NeuroChem Files
====================================

This tutorial illustrates how to manually load model from `NeuroChem files`_.

.. _NeuroChem files:
    https://github.com/isayev/ASE_ANI/tree/master/ani_models

"""

###############################################################################
# To begin with, let's first import the modules we will use:
import os
import ase
import numpy as np
import sys
import getopt
from ase.constraints import FixAtoms, FixBondLengths, FixInternals
from restrain_ani import Position_restraints, Fix_dihedrals,Fix_Phi_Psi,Fix_Angle
from ase import Atoms 
#from ase_interface import ANIENS
#from ase_interface import aniensloader
import time
#from ase.optimize import BFGS   , LBFGS
from read_ani import read
from con_opt import LBFGS
from software.ANIOptimizer.malepso.calculator.calculator_ani import Calculator
g_au=27.211386024367243
###############################################################################
# Now let's read constants from constant file and construct AEV computer.
#try:
   # path = os.path.dirname(os.path.realpath(__file__))
#except NameError:
if __name__ == '__main__':

###############################################input##########
    
     input_file_name=''
     output_file_name=''
     output_log_name=''
     energy_force_log =''
     gpuid=''
     Method=''
     
     try:
         myopts, args = getopt.getopt(sys.argv[1:],"i:o:m:g:l:")
     except getopt.GetoptError as err:
         print ('Error: ',"%s"%(str(err)))
         print ('Usage: _*.py  -i input.ani -o output.log -g 0 or 1 or other gpu id and no parameter for cpu')
         sys.exit(2)
     for o, a in myopts:
         if o == "-i":
             input_file_name = a
         elif o == "-o":
             output_file_name = a
         elif o == "-g":
             gpuid = a
             print ('Usage:',gpuid)
         elif o == "-l":
             output_log_name = a
             energy_force_log = open(output_log_name,'w')
         elif o == "-m":
             Method = a
             
         else:
             print ('Usage: _*.py  -i input.ani -o output.log -g 0 or 1 or other gpu id and no parameter for cpu -m 1-CG-BS,2-CG-WS,3-LBFGS,4-LBFGS-WS,5-BFGS,0-single point')
             sys.exit(0)
###############################################################################
# You can also create an ASE calculator using the ensemble or single model:
if gpuid=='':
     calculator = Calculator().cal
else:
     from ase_interface import ANIENS
     from ase_interface import aniensloader
     path = os.environ['NC_ROOT']
     path_file = os.path.join(path, 'ani_models/ani-2x_8x.info')
     calculator=ANIENS(aniensloader(path_file,gpuid=0))
mol=read(input_file_name)
###############################################################################
# Now let's define a moltoase molecule
species = mol.species
moltoase = ase.Atoms(mol.t_elem, positions=mol.coord*1.0)
moltoase.k_flag=mol.k_flag
moltoase.r_flag=mol.r_flag
moltoase.t_bond=mol.t_bond
moltoase.t_ang=mol.t_ang
moltoase.t_tor=mol.t_tor
moltoase.f_max_th=0.00045*27.211386024367243
moltoase.f_rms_th=0.0003*27.211386024367243
moltoase.dp_max_th=0.0018   
moltoase.dp_rms_th=0.0012
print('torsion angle constrain index:',mol.t_tor)
print('angle constrain index:',mol.t_ang)
print('bond constrain index:',mol.t_bond)
#set constrains and restraints##############################
r0=moltoase.get_positions()*1.0
constraint=[]
tor0=[]
ang0=[]
d_bond0=[]
tor=[]
ang=[]
d_bond=[]
moltoase.f_max_flag = 'NO'
moltoase.f_rms_flag = 'NO'
moltoase.dp_max_flag = 'NO'
moltoase.dp_rms_flag ='NO'
constraint1 = Position_restraints(mol.r_flag,mol.k_flag,r0)
constraint.append(constraint1)
#constraint1=FixAtoms(indices=[0,1])
#moltoase.set_constraint(constraint1)
#constraint2 = FixBondLengths([(5,6)],1.33)
if mol.t_tor.size>3:
    constraint2=Fix_dihedrals(mol.t_tor)
    constraint.append(constraint2)
    tor0=moltoase.get_dihedrals(mol.t_tor.reshape(-1,4)-1)
if mol.t_ang.size>2:
    constraint3=Fix_Angle(mol.t_ang)
    constraint.append(constraint3)
    ang0=moltoase.get_angles(mol.t_ang.reshape(-1,3)-1)
if mol.t_bond.size>1:
    
    d_bond0=moltoase.get_distance(mol.t_bond[0]-1,mol.t_bond[1]-1)
    constraint4=FixBondLengths(mol.t_bond.reshape(-1,2)-1,d_bond0)
    constraint.append(constraint4)
    
moltoase.set_constraint(constraint)

###############################################################################
# And using the ASE interface of the ensemble:
moltoase.set_calculator(calculator)
e0=moltoase.get_potential_energy(force_consistent=True)/g_au
print('Energy:', e0)
print('Force:', moltoase.get_forces()/ ase.units.Hartree)
i=0
s=time.time()
iteration = LBFGS(moltoase,maxiteration=10)#mol.max_iteration
if mol.t_tor.size>3:
    tor=moltoase.get_dihedrals(mol.t_tor.reshape(-1,4)-1)
if mol.t_ang.size>2:
   ang=moltoase.get_angles(mol.t_ang.reshape(-1,3)-1)
if mol.t_bond.size>1:
   ang=moltoase.get_angles(mol.t_ang.reshape(-1,3)-1)
   d_bond=moltoase.get_distance(mol.t_bond[0]-1,mol.t_bond[1]-1)
if moltoase.max_f<=moltoase.f_max_th:
   moltoase.f_max_flag = 'TRUE'
if moltoase.rms_f<=moltoase.f_rms_th:
   moltoase.f_rms_flag = 'TRUE'
if moltoase.max_dp<=moltoase.dp_max_th:
   moltoase.dp_max_flag = 'TRUE'
if moltoase.rms_dp<=moltoase.dp_rms_th:
   moltoase.dp_rms_flag ='TRUE'
#dyn.run(fmax=0.01215)
#print('molspec',moltoase.get_positions()[0,0])
print('Energy:', moltoase.get_potential_energy(force_consistent=True)/g_au)
outfile=open(output_file_name,'w')
print('        Item             Value        Threshold      Converged',file=outfile)
     
print('Maximum Force         ', "%10.6f   %10.6f       %s"%(moltoase.max_f/g_au,moltoase.f_max_th/g_au,moltoase.f_max_flag),file=outfile)
print('RMS     Force         ', "%10.6f   %10.6f       %s"%(moltoase.rms_f/g_au,moltoase.f_rms_th/g_au,moltoase.f_rms_flag),file=outfile)
print('Maximum Displacement  ', "%10.6f   %10.6f       %s"%(moltoase.max_dp,moltoase.dp_max_th,moltoase.dp_max_flag),file=outfile)
print('RMS     Displacement  ', "%10.6f   %10.6f       %s"%(moltoase.rms_dp,moltoase.dp_rms_th,moltoase.dp_rms_flag),file=outfile)
print('Energy:',e0,moltoase.get_potential_energy(force_consistent=True)/g_au,file=outfile)
print('bond length:',d_bond0, d_bond,file=outfile)
print('bond angle:',ang0, ang,file=outfile)
print('Toristion angle:',tor0, tor,file=outfile)
print('Step:',iteration,file=outfile)
print('Time:',time.time()-s,file=outfile)
for i in range(0,int(mol.species.size()[1])):
            print ("%-6s%5d %-4s%4s%6d    %16.10f%16.10f%16.10f%6.2lf%6.2lf%12s"%('ATOM',i+1,mol.t_elem[i]+str(i),'MOL',1, moltoase.get_positions()[i,0],moltoase.get_positions()[i,1],moltoase.get_positions()[i,2], mol.r0_flag[0,i],mol.k_flag[0,i],mol.t_elem[i]), file=outfile)
outfile.close()   
   






