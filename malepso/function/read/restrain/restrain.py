from math import sqrt
from warnings import warn

import numpy as np
from scipy.linalg import expm, logm
from ase.calculators.calculator import PropertyNotImplementedError
from ase.geometry import (find_mic, wrap_positions, get_distances_derivatives,
                          get_angles_derivatives, get_dihedrals_derivatives,
                          conditional_find_mic, get_angles, get_dihedrals)
from ase.utils.parsemath import eval_expression
from ase.stress import (full_3x3_to_voigt_6_stress,
                        voigt_6_to_full_3x3_stress)
def Rotation_BCD_xy(coord,i,j,k):
   
    bc = coord[i] - coord[j]
    dc = coord[k] - coord[j]
    t_n1 = np.cross(bc,dc)
    Vac = t_n1 / np.linalg.norm(t_n1,ord=2)
    a,b,c = Vac[0],Vac[1],Vac[2]
    Rby = np.array([[1.0,0,0],
                    [0,1.0,0],
                    [0,0,1.0]])
    if a**2+c**2>0.0:
       Rby = np.array([[c/np.sqrt(a**2+c**2),0,-a/np.sqrt(a**2+c**2)],
                        [0,1,0],
                        [a/np.sqrt(a**2+c**2),0,c/np.sqrt(a**2+c**2)]],dtype=np.float64) 
    Rbx = np.array([[1,0,0],
                    [0,np.sqrt(a**2+c**2),-b],
                    [0,b,np.sqrt(a**2+c**2)],
                    ],dtype=np.float64)  
    Rbcd = np.matmul(Rbx,Rby)
    return Rbcd
def Translation_0(coord,k0):
    
    trans=np.ones(coord.shape)
    trans[:,0]=coord[k0,0]*trans[:,0]
    trans[:,1]=coord[k0,1]*trans[:,1]
    trans[:,2]=coord[k0,2]*trans[:,2]    
    return coord-trans
def Rotation_x(coord,j):
    bc=coord[j]
    Vbc=bc/np.linalg.norm(bc,ord=2)
    a,b,c = Vbc[0],Vbc[1],Vbc[2]
    Rby = np.array([[1,0,0],
                           [0,1,0],
                           [0,0,1]],dtype=np.float64)
    if a**2+c**2>0.0:
       Rby = np.array([[a/np.sqrt(a**2+c**2),0,c/np.sqrt(a**2+c**2)],
                        [0,1,0],
                        [-c/np.sqrt(a**2+c**2),0,a/np.sqrt(a**2+c**2)]],dtype=np.float64) 

    Rbz = np.array([[np.sqrt(a**2+c**2),b,0],
                        [-b,np.sqrt(a**2+c**2),0],
                        [0,0,1]],dtype=np.float64)  
 
    Rb=np.matmul(Rbz,Rby)
    
    return Rb   
def Rotation_xy(Rb,coord,i):
    coord0=np.matmul(Rb,coord.T)
    ac=coord0.T[i]
    Vac=ac/np.linalg.norm(ac,ord=2)
    a,b,c = Vac[0],Vac[1],Vac[2]
    Rax = np.array([[1,0,0],
                        [0,1,0],
                    [0,0,1]],dtype=np.float64)  
    if b**2+c**2>0.0000000000000001:
              Rax = np.array([[1,0,0],
                        [0,b/np.sqrt(b**2+c**2),c/np.sqrt(b**2+c**2)],
                    [0,-c/np.sqrt(b**2+c**2),b/np.sqrt(b**2+c**2)]],dtype=np.float64)
    
    return Rax
def Angle_Force(Ra,Rc,force,i,j,k):  
    Rc_f=np.matmul(Rc,force.T)
    Ra_f=np.matmul(Ra,Rc_f)
    Ra_f.T[i,1:3]=0.0
    Rc_f.T[j,0:3]=0.0
    Rc_f.T[k,1:3]=0.0
    Ra_f=np.matmul(np.linalg.inv(Ra),Ra_f)
    Ra_f=np.matmul(np.linalg.inv(Rc),Ra_f)
    Rc_f=np.matmul(np.linalg.inv(Rc),Rc_f)
    force[i]=Ra_f.T[i]
    force[j]=Rc_f.T[j]
    force[k]=Rc_f.T[k]
    return force    
def Torision_Force_BCD(Rbcd,Ra,Rb,Rd,force,i,j,k,l):  
    Rbcd_f=np.matmul(Rbcd,force.T)
    Rb_f=np.matmul(Rb,Rbcd_f)
    Ra_f=np.matmul(Ra,Rb_f)
    Rd_f=np.matmul(Rd,Rb_f)
    Ra_f.T[i,2]=0.0
    Rb_f.T[j,1:3]=0.0
    Rb_f.T[k,1:3]=0.0
    Rd_f.T[l,2]=0.0
    Ra_f=np.matmul(np.linalg.inv(Ra),Ra_f)
    Ra_f=np.matmul(np.linalg.inv(Rb),Ra_f)
    Ra_f=np.matmul(np.linalg.inv(Rbcd),Ra_f)
    Rb_f=np.matmul(np.linalg.inv(Rb),Rb_f)
    Rb_f=np.matmul(np.linalg.inv(Rbcd),Rb_f)
    Rd_f=np.matmul(np.linalg.inv(Rd),Rd_f)
    Rd_f=np.matmul(np.linalg.inv(Rb),Rd_f)
    Rd_f=np.matmul(np.linalg.inv(Rbcd),Rd_f)
    force[i]=Ra_f.T[i]
    force[j]=Rb_f.T[j]
    force[k]=Rb_f.T[k]
    force[l]=Rd_f.T[l]
    return force  
class FixConstraint:
    """Base class for classes that fix one or more atoms in some way."""

    def index_shuffle(self, atoms, ind):
        """Change the indices.

        When the ordering of the atoms in the Atoms object changes,
        this method can be called to shuffle the indices of the
        constraints.

        ind -- List or tuple of indices.

        """
        raise NotImplementedError

                        
class Position_restraints(FixConstraint):
    """"""

    def __init__(self, r_flag, k_flag, r0, rt=None):
        """Forces two atoms to stay close together by applying no force if
        they are below a threshold length, rt, and applying a Hookean
       
        """
        self.spring = 0.000796812
        self.k_flag=k_flag
        self.r_flag=r_flag
        self.r0=r0

    def adjust_positions(self, atoms, newpositions):
        pass

    def adjust_momenta(self, atoms, momenta):
        pass

    def adjust_forces(self, atoms, forces):
        positions = atoms.positions
        forces[:]=forces-2*self.spring*self.k_flag.T*(positions -self.r0)/27.211386024367243
        forces[:] = self.r_flag.T*forces
        #print('forces:',forces)
        
    def adjust_potential_energy(self, atoms):
        """Returns the difference to the potential energy due to an active
        constraint. (That is, the quantity returned is to be added to the
        potential energy.)"""
        positions = atoms.positions
        e=self.spring * self.k_flag.T * (positions - self.r0)**2/27.211386024367243
        return e.sum()
      
class Fix_dihedrals(FixConstraint):
    """"""
    
    def __init__(self, tor_index, rt=None):
        """Forces two atoms to stay close together by applying no force if
        they are below a threshold length, rt, and applying a Hookean
       
        """
        self.tor = tor_index

    def adjust_positions(self, atoms, newpositions):
        pass

    def adjust_momenta(self, atoms, momenta):
        pass

    def adjust_forces(self, atoms, forces):
        coord = atoms.positions*1.0
        flag=0
        if self.tor.size>=4:           
            for i in range(0,self.tor.size,4):               
                if  i<=self.tor.size-4:
                    Rbcd=Rotation_BCD_xy(coord,self.tor[i+1]-1,self.tor[i+2]-1,self.tor[i+3]-1)
                BCD_coord=np.matmul(Rbcd, coord.T)
                coord0=BCD_coord.T  
                trc_coord=Translation_0(coord0,self.tor[i+2]-1)
                #print('trc_coord:',trc_coord,self.tor[i+2]-1)
                Rb_mat=Rotation_x(trc_coord,self.tor[i+1]-1)
                #print('Rb_mat:',Rb_mat)
                Rb_coord=np.matmul(Rb_mat,trc_coord.T)
                #print('Rb_coord:',Rb_coord.T)
                Ra_mat=Rotation_xy(Rb_mat,trc_coord,self.tor[i]-1)
                Ra_coord=np.matmul(Ra_mat,trc_coord.T)
                #print('Ra_coord:',Ra_coord.T)
                Rd_mat=Rotation_xy(Rb_mat,trc_coord,self.tor[i+3]-1)
                Rd_coord=np.matmul(Rd_mat,trc_coord.T)
                #print('Rd_coord:',Rd_coord.T)
                forces[:]=Torision_Force_BCD(Rbcd,Ra_mat,Rb_mat,Rd_mat,forces,self.tor[i]-1,self.tor[i+1]-1,self.tor[i+2]-1,self.tor[i+3]-1)
               
                
    def adjust_potential_energy(self, atoms):
        """Returns the difference to the potential energy due to an active
        constraint. (That is, the quantity returned is to be added to the
        potential energy.)"""
        return 0
class Fix_Phi_Psi(FixConstraint):
    """"""
    
    def __init__(self, tor_index, rt=None):
        """Forces two atoms to stay close together by applying no force if
        they are below a threshold length, rt, and applying a Hookean
       
        """
        self.tor = tor_index

    def adjust_positions(self, atoms, newpositions):
        pass

    def adjust_momenta(self, atoms, momenta):
        pass

    def adjust_forces(self, atoms, forces):
        coord = atoms.positions*1.0
        flag=0
        if self.tor.size>=4:
            
            for i in range(0,self.tor.size,4):               
                if  i<self.tor.size-4:
                    Rbcd=Rotation_BCD_xy(coord,self.tor[i+1]-1,self.tor[i+2]-1,self.tor[i+3]-1)
                BCD_coord=np.matmul(Rbcd, coord.T)
                coord0=BCD_coord.T  
                trc_coord=Translation_0(coord0,self.tor[i+2]-1)
                #print('trc_coord:',trc_coord,self.tor[i+2]-1)
                Rb_mat=Rotation_x(trc_coord,self.tor[i+1]-1)
                #print('Rb_mat:',Rb_mat)
                Rb_coord=np.matmul(Rb_mat,trc_coord.T)
                #print('Rb_coord:',Rb_coord.T)
                Ra_mat=Rotation_xy(Rb_mat,trc_coord,self.tor[i]-1)
                Ra_coord=np.matmul(Ra_mat,trc_coord.T)
                #print('Ra_coord:',Ra_coord.T)
                Rd_mat=Rotation_xy(Rb_mat,trc_coord,self.tor[i+3]-1)
                Rd_coord=np.matmul(Rd_mat,trc_coord.T)
                #print('Rd_coord:',Rd_coord.T)
                forces[:]=Torision_Force_BCD(Rbcd,Ra_mat,Rb_mat,Rd_mat,forces,self.tor[i]-1,self.tor[i+1]-1,self.tor[i+2]-1,self.tor[i+3]-1)
                forces[self.tor[i+2]-1]=0      
    def adjust_potential_energy(self, atoms):
        """Returns the difference to the potential energy due to an active
        constraint. (That is, the quantity returned is to be added to the
        potential energy.)"""
        return 0 
class Fix_Angle(FixConstraint):
    """"""
    
    def __init__(self, ang_index, rt=None):
        """Forces two atoms to stay close together by applying no force if
        they are below a threshold length, rt, and applying a Hookean
       
        """
        self.ang = ang_index

    def adjust_positions(self, atoms, newpositions):
        pass

    def adjust_momenta(self, atoms, momenta):
        pass

    def adjust_forces(self, atoms, forces):
        coord = atoms.positions*1.0
        if self.ang.size>=3: 
            for i in range(0,self.ang.size,3):
                trc_coord=Translation_0(coord,self.ang[i+1]-1)
                Rc_mat=Rotation_x(trc_coord,self.ang[i+2]-1)
                coord_rc=np.matmul(Rc_mat,trc_coord.T)
                coord_rc=coord_rc.T    
                Ra_mat=Rotation_x(coord_rc,self.ang[i]-1)
                forces[:]=Angle_Force(Ra_mat,Rc_mat,forces,self.ang[i]-1,self.ang[i+1]-1,self.ang[i+2]-1)
    def adjust_potential_energy(self, atoms):
        """Returns the difference to the potential energy due to an active
        constraint. (That is, the quantity returned is to be added to the
        potential energy.)"""
        return 0         
