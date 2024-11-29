# from time import clock
import numbers
from operator import inv
from scipy.io import mmwrite
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import uniform
from scipy.misc import derivative

from scipy import misc
from qutip import *
from qutip.piqs import *
from numpy import *


def main():
    cumulative_sum = cumulative_sum = [0] * (51)
    for _ in range(30):
        #parameters  
        BzImp=0.01
        W=0
        BzChain=0
        BxChain=0.3
        Jzcoup=0.2
        Jxcoup=0.4
        Jx=1
        Jz=0
        Nspin1=0
        Nspin2=11
        Nspin3=22
        Nm=4
        Nmm=7
        N=13
        a=0.0001
        #tlist=np.linspace(0, 100000, 251)
        tlist=np.linspace(0,100000,51)
        options1 = {
            "nsteps": 100000}
        Dlist=[]
        for _ in range(N):
            Di=uniform.rvs(loc=-W, scale=2*W, size=1, random_state=None)
            Dlist.append(Di[0])
        #print(Dlist)
        #Setup operators for individual qubits   
        sx_list, sy_list, sz_list = [], [], []
        for i in range(N):
            op_list = [qeye(2)] * N
            op_list[i] = sigmax()
            sx_list.append(tensor(op_list))
            op_list[i] = sigmay()
            sy_list.append(tensor(op_list))
            op_list[i] = sigmaz()
            sz_list.append(tensor(op_list))
        #hamiltonian impurity
        HS = BzImp * (sz_list[Nspin1])
        HS1 = (BzImp+a) * (sz_list[Nspin1])
        HS2 = (BzImp-a) * (sz_list[Nspin1])
        #hamiltonian chain HB - energy splitting terms
        HB = 0
        for i in range(N):
            if i == Nspin1:
                continue
            HB += BzChain * sz_list[i] + BxChain * sx_list[i]
        #hamiltonian chain HB - interaction terms
        for i in range(N-1):
            if i == Nspin1:
                continue
            HB += Jz * sz_list[i]*sz_list[i+1] 
            HB += (Jx/4) * (sx_list[i]+1j*sy_list[i]) * (sx_list[i+1]-1j*sy_list[i+1]) 
            HB += (Jx/4) * (sx_list[i]-1j*sy_list[i]) * (sx_list[i+1]+1j*sy_list[i+1])
        #disorder term Hd
        Hd=0
        for i in range(N):
            Hd += Dlist[i] * sz_list[i]
            #print(Dlist[i])
        #free hamiltonian H0
        H0 = HB + HS + Hd
        H0_OG = HB + HS 
        H01 = HB + HS1 + Hd
        H02 = HB + HS2 + Hd
        #interaction hamiltonian 
        Hi = (Jxcoup/4) * ((sx_list[Nspin1]+1j*sy_list[Nspin1]))*(sx_list[Nm]-1j*sy_list[Nm]) + (Jxcoup/4) * ((sx_list[Nspin1]-1j*sy_list[Nspin1]))*(sx_list[Nm]+1j*sy_list[Nm]) + Jzcoup * ((sz_list[Nspin1])) * sz_list[Nm]
        #full hamiltonian
        H = H0 + Hi 
        H1= H01 + Hi 
        H2 = H02 + Hi 
        #H0 basis
        eigvalsH0, eigvecsH0 = H0_OG.eigenstates()
        #Hi basis
        #eigvalsHi, eigvecsHi = Hi.eigenstates()
        #operators that we want to measure the average value of
        #e_ops = [sz_list[Nspin1]]
        #initialstate
        inket1=tensor(spin_state(1/2,1/2),spin_state(1/2,-1/2))
        inket2=tensor(inket1,spin_state(1/2,1/2))
        inket3=tensor(inket2,spin_state(1/2,-1/2))
        inket4=tensor(inket3,spin_state(1/2,1/2))
        inket5=tensor(inket4,spin_state(1/2,-1/2))
        inket6=tensor(inket5,spin_state(1/2,1/2))
        inket7=tensor(inket6,spin_state(1/2,-1/2))
        inket8=tensor(inket7,spin_state(1/2,1/2))
        inket9=tensor(inket8,spin_state(1/2,-1/2))
        inket10=tensor(inket9,spin_state(1/2,1/2))
        inket11=tensor(inket10,spin_state(1/2,-1/2))
        inket12=tensor(inket11,spin_state(1/2,1/2))
        inket13=tensor(inket12,spin_state(1/2,-1/2))
        inket14=tensor(inket13,spin_state(1/2,1/2))
        inket15=tensor(inket14,spin_state(1/2,-1/2)) 
        inket16=tensor(inket15,spin_state(1/2,1/2))
        #inket=inket10
        inket=eigvecsH0[1000]
        #Schrodinger
        result = sesolve(H, inket, tlist, options=options1)
        result1 = sesolve(H1, inket, tlist, options=options1)
        result2 = sesolve(H2, inket, tlist, options=options1)
        #average sigma z 
        wavefunc=result.states
        wavefuncplus=result1.states
        wavefuncminus=result2.states
        #sigma = result.expect[0]
        #fisherinformation
        F=0
        listF=[]
        for i in range(len(wavefunc)):
            F = 4*((wavefuncplus[i].dag()-wavefuncminus[i].dag())/(2*a))*((wavefuncplus[i]-wavefuncminus[i])/(2*a)) - 4*(wavefunc[i].dag()*((wavefuncplus[i]-wavefuncminus[i])/(2*a)))*(((wavefuncplus[i].dag()-wavefuncminus[i].dag())/(2*a))*wavefunc[i])    
            listF.append(np.real(F))
        #print(listF)
        #newesttimeslist = newtimeslist[:250]
        #newlistforFisher = []
        #for k in range(len(tlist)):
        #    newlistforFisher.append(listF[k][0][0][0].real)
        #print(newlistforFisher[0])
        #print(newlistforFisher[-1])
        #for a in range(len(tlist)):
            #print(listF[a])
        #np.savetxt('coolertext',newlistforFisher)
        #longtimeave
        #print(sigma[-1])
        #print(longtimeave)
        #print(fluctuations)
        #plotting
        #fig1 = plt.figure(1)
        #plt.plot(result.times, sigma, '-')
        #plt.plot(result.times, 0*result.times + longtimeave, 'r')
        #plt.plot(result.times, 0*result.times + fluctuations, 'b--')
        #label_size = 20
        #label_size2 = 15
        #label_size3 = 15
        #plt.rc('text', usetex = True)
        #plt.title(r'$sigma_z1(t)$',
        #        fontsize = label_size2)
        
        #plt.rc('xtick', labelsize=label_size) 
        #plt.rc('ytick', labelsize=label_size)

        #plt.ylim([-1,1])
        #plt.xlim([0,100])

        #plt.xlabel(r'$t$', fontsize = label_size3)
        #plt.ylabel(r'$sigma_z1$', fontsize = label_size3)

        #fname = 'figures/btc_eig_N{}_strong_jmat.pdf'.format(N)
        #savefile = False
        #plt.show()
            
        #if savefile == True:    
        #    plt.savefig(fname, bbox_inches='tight')

        #    plt.show()
        #    plt.close()   
        cumulative_sum = [cumulative_sum[i] + listF[i] for i in range(len(listF))]    
        print(cumulative_sum)  
    final = []
    for k in range(len(cumulative_sum)):
        final.append(cumulative_sum[k]/30)
    for o in range(len(final)):
        print(final[o])    
if __name__ == "__main__":
    main()