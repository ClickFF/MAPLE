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

import numpy as np

from ase import Atoms 
import torchani

from function.read import InputReader
from function.calculator import Calculator

g_au=27.211386024367243

class engine():
    def __init__(self):
        
        self.output:str = None
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

        self.calulator = None

    def __call__(self, input_file_name:str,output_file_name:str=None):
        """
            This is the engine of the program.
            Args:
                input_file_name: The path to the input file.
                output_file_name: The path to the output file. (Default: None)
        """
        self._input_reader(input_file_name, output_file_name)
        self._mlp_initiator(self.model, self.gpuid)
        self.atoms.set_calculator(self.calulator)
        self._jobtype_dispatcher(self.jobtype, self.atoms, self.output)

    def _input_reader(self, input_file_name:str, output_file_name:str=None) -> Atoms:
        """
            This function reads the input file.

            Args:
                input_file_name: The path to the input file.
                output_file_name: The path to the output file. (Default: Same as input file)
            
            Returns:
                Atoms: ASE Atoms object.
        """
        reader = InputReader()
        self.atoms = reader(input_file_name, output_file_name)
        self.output = reader.output
        self.gpuid = reader.gpuid
        self.model = reader.model
        self.jobtype = reader.jobtype

    
    def _mlp_initiator(self, model:int, gpuid:int) -> torchani.ase.Calculator:
        """
            This function initializes the model.

            Args:
                model: int
                    The model to be used for the calculation.
                    1: ANI-2x
                    2: ANI-1x
                    3: ANI-1ccx
                    4: ANI-1xnr
                gpuid: int
                    The GPU ID to be used for calculation. If None, use CPU. Default is None.
        """
        calculator = Calculator(model, gpuid, self.output)
        self.calulator = calculator.construct_calculator()

        return self.calulator
    
    def _jobtype_dispatcher(self, jobtype:int, atoms:Atoms, output:str, method:str='LBFGS') -> None:
        """
            This function dispatches the job type.

            Args:
                jobtype(int): The type of job to be performed.
                atoms(Atoms): The ASE Atoms object to be optimized.
                output(str): The path to the output file.
                method(str): The optimization method to be used. (Default: LBFGS)
        """
        from function.dispatcher import Dispatcher
        
        dispatcher = Dispatcher()
        dispatcher(jobtype, atoms, output, method)
        

    def _constrints_restaints_maker(self, mol:Atoms):
        pass
"""
    def _run_optimization(self, input_file_name:str, output_file_name:str, gpuid:int):

            This function runs the optimization.
        

        
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

        print('Energy:', moltoase.get_potential_energy(force_consistent=True)/g_au)

        # Open the output file in write mode
        outfile = open(output_file_name, 'w')

        # Use a list to store each line of output content
        contents = []

        # Add the header line
        contents.append('        Item             Value        Threshold      Converged\n')

        # Add each item with f-string formatting
        contents.append(f'Maximum Force         {moltoase.max_f / g_au:10.6f}   {moltoase.f_max_th / g_au:10.6f}       {moltoase.f_max_flag}\n')
        contents.append(f'RMS     Force         {moltoase.rms_f / g_au:10.6f}   {moltoase.f_rms_th / g_au:10.6f}       {moltoase.f_rms_flag}\n')
        contents.append(f'Maximum Displacement  {moltoase.max_dp:10.6f}   {moltoase.dp_max_th:10.6f}       {moltoase.dp_max_flag}\n')
        contents.append(f'RMS     Displacement  {moltoase.rms_dp:10.6f}   {moltoase.dp_rms_th:10.6f}       {moltoase.dp_rms_flag}\n')
        contents.append(f'Energy: {e0} {moltoase.get_potential_energy(force_consistent=True) / g_au}\n')
        contents.append(f'bond length: {d_bond0} {d_bond}\n')
        contents.append(f'bond angle: {ang0} {ang}\n')
        contents.append(f'Torsion angle: {tor0} {tor}\n')
        contents.append(f'Step: {iteration}\n')
        contents.append(f'Time: {time.time() - s}\n')

        # Add information for each atom
        for i in range(int(mol.species.size()[1])):
            contents.append(
                f"ATOM  {i+1:5d} {mol.t_elem[i]+str(i):<4} MOL  {1:4d} "
                f"{moltoase.get_positions()[i,0]:16.10f}{moltoase.get_positions()[i,1]:16.10f}"
                f"{moltoase.get_positions()[i,2]:16.10f}{mol.r0_flag[0,i]:6.2f}{mol.k_flag[0,i]:6.2f}"
                f"{mol.t_elem[i]:<12}\n"
            )

        # Write all content to the file at once
        outfile.writelines(contents)
        # Close the file
        outfile.close()
"""






