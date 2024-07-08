import numpy as np
import torch
max_iteration_default=10000
f_max_th=0.00045
f_rms_th=0.0003
dp_max_th=0.0018   
dp_rms_th=0.0012
Converge_kind=np.array([1.0,1.2,1.4,1.6,1.8,2,2.2,2.4,2.6,2.8])
def extract(text,target):
    linenum = 0
    found = False
    for line in text:
       if not found:
         if (line.find(target)) > -1:
           found=True
         else:
           linenum += 1
    return linenum 
class read:		# 
    def __init__(self,filename,device='cpu'): #
        
        try:
          
          input_file  = open(filename, 'r') # File to be processed
          text = input_file.readlines()
          input_file.close()  
         
        except IOError as err:
          print ('Error: ',"%s"%(str(err)))
          sys.exit(1)
        flag=0
        t_spec = [] 
        t_coord = []
        t_bond = []
        t_ang = []
        t_tor = []
        t_elem = []
        t_dis = []
        max_iteration = text[extract(text,'#MAXSTEP')+1:]
        max_iteration = max_iteration[:extract(max_iteration,'!')]
        self.device=device
        #########max_iteration###########
        max_iteration = np.array(max_iteration[0].split()).astype(int)[0]
        self.max_iteration=max_iteration
   
        if self.max_iteration<0:
           self.max_iteration=max_iteration_default
        print ('max_iteration:',self.max_iteration)
        ######converge level#################
        converge = text[extract(text,'#CONVERGE')+1:]
        converge = converge[:extract(converge,'!')]
        converge = np.array(converge[0].split()).astype(int)[0]
        self.converge=torch.tensor([converge], device=device)
        f_max_th=0.00045*Converge_kind[converge-1]
        f_rms_th=0.0003*Converge_kind[converge-1]
        dp_max_th=0.0018*Converge_kind[converge-1]
        dp_rms_th=0.0012*Converge_kind[converge-1]
        self.f_max_th=torch.tensor([f_max_th], device=device)
        self.f_rms_th=torch.tensor([f_rms_th], device=device)
        self.dp_max_th=torch.tensor([dp_max_th], device=device)
        self.dp_rms_th=torch.tensor([dp_rms_th], device=device)
        ######read elem#############
        elem = text[extract(text,'#ELEM')+1:]
        elem = elem[:extract(elem,'#ATOMIC_NUM')]       	       
        for line in elem:
             column=line.split()
             t_elem.extend(column)
        self.t_elem=t_elem
        ######ATOMIC_NUM#############
        a_num = text[extract(text,'#ATOMIC_NUM')+1:]
        a_num = a_num[:extract(a_num,'#COORD')]
        for line in a_num:
             column=line.split()
             t_spec.extend(column)
        t_spec=np.array(t_spec)
        self.species = torch.tensor([t_spec.astype(int)], device=device)
        ###read coordinates########
        coord= text[extract(text,'#COORD')+1:]
        coord= coord[:extract(coord,'#CONS_BOND')]
        for line in coord:
             column=line.split()
             t_coord.extend(column)
        t_coord=np.array(t_coord)
        t0_coord= t_coord.reshape(int(t_coord.size/6),6)
        t1_coord= t0_coord[:,0:3]
        self.coord=torch.tensor([t1_coord.astype(float)],requires_grad=True,dtype=torch.float32, device=device)
        ################bond constarin atom###################
        cons_bond = text[extract(text,'#CONS_BOND')+1:]
        cons_bond = cons_bond[:extract(cons_bond,'#CONS_ANGLE')]
        for line in cons_bond:
             column=line.split()
             t_bond.extend(column)
        t_bond=np.array(t_bond).astype(int)
        self.t_bond=torch.tensor(t_bond, device=device)
     ################angle constrain################# 
        cons_ang =text[extract(text,'#CONS_ANGLE')+1:]
        cons_ang =cons_ang[:extract(cons_ang,'#CONS_TOR')]
        for line in cons_ang:
             column=line.split()
             t_ang.extend(column)
        t_ang=np.array(t_ang).astype(int)
        self.t_ang=torch.tensor(t_ang, device=device)
     ################torsion constrain#################
        cons_tor =text[extract(text,'#CONS_TOR')+1:]
        cons_tor =cons_tor[:extract(cons_tor,'#END')]      
        for line in cons_tor:
             column=line.split()
             t_tor.extend(column)
        t_tor=np.array(t_tor).astype(int)
        self.t_tor=torch.tensor(t_tor, device=device)
        #dis_res =text[extract(text,'#END')+1:]
        #dis_res =dis_res[:extract(dis_res,'#END')]
        #text = text[:extract(text,'--Link1--')-3]
        ##############restain force constant############################
        k_flag=t0_coord[:,5]
        self.k_flag=torch.tensor([k_flag.astype(float)],requires_grad=False,dtype=torch.float32, device=device)
        ###########restain flag##############################
        r_flag=t0_coord[:,4]
        r_flag=torch.tensor([r_flag.astype(float)],requires_grad=False,dtype=torch.float32, device=device)
        one=torch.ones_like(r_flag)
        zero=torch.zeros_like(r_flag)
        self.r0_flag= r_flag*1.0
        r_flag= torch.where(r_flag!=-1,one,r_flag)

        self.r_flag= torch.where(r_flag==-1,zero,r_flag)
        ##############energy&force#######################
        #self.energy = model((self.species, self.coord)).energies
        #self.force = -torch.autograd.grad(self.energy.sum(), self.coord)[0]
        ##########intinal converge value###############################
        self.coord_size=self.species.size()[1]*3.0
        #dp=self.coord-self.coord*1.0
        self.max_dp=0
        self.rms_dp=0
        self.max_f=0
        self.rms_f =0
        self.k_flag=np.array([self.k_flag.squeeze().detach().numpy()])
        self.r_flag=np.array([self.r_flag.squeeze().detach().numpy()])
        self.t_bond=np.array([self.t_bond.squeeze().detach().numpy()]).flatten()
        self.t_ang=np.array([self.t_ang.squeeze().detach().numpy()]).flatten()
        self.t_tor=np.array([self.t_tor.squeeze().detach().numpy()]).flatten()
        self.coord=self.coord.squeeze().detach().numpy()
        
