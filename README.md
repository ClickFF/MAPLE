# ASE Install##########
conda install -c conda-forge ase
########ANI-ASE Install#########
Download from https://github.com/isayev/ASE_ANI
unzip ASE_ANI
cd ASE_ANI
csh env.sh
Then put our python srcipt in the ASE_ANI/lib except wangani.py
Following is the commond(need to place wangani.py on the working directory)
python wangani.py -i input2.ani -o ani.out -g 0(gpuid, for cpu version it should not set it)
or 
chmod a+rwx wangani.py
and then put it on any path(to amber/bin)
and the same to MMPBSA.py
wangani.py -i input2.ani -o ani.out

