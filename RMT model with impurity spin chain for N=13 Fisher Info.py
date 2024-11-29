# from time import clock
import numbers
from operator import inv
from scipy.io import mmwrite
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import derivative

from scipy import misc
from qutip.piqs import dicke
from qutip import *
from qutip.piqs import *
from numpy import *

def main():
    #parameters  
    BzImp=0.01
    BzChain=0
    BxChain=0.3
    Jzcoup=0.2
    Jxcoup=0.4
    Jx=1
    Jz=0
    Nspin1=0
    Nspin2=1
    Nspin3=2
    Nspin4=3
    Nm=4
    N=11
    a=0.0001
    tlist=np.linspace(0, 100000, 51)
    options1 = Options(num_cpus=8,nsteps=1000000)
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
    #free hamiltonian H0
    H0 = HB + HS
    H01 = HB + HS1
    H02 = HB + HS2
    #interaction hamiltonian 
    Hi = (Jxcoup/4) * ((sx_list[Nspin1]+1j*sy_list[Nspin1]))*(sx_list[Nm]-1j*sy_list[Nm]) + (Jxcoup/4) * ((sx_list[Nspin1]-1j*sy_list[Nspin1]))*(sx_list[Nm]+1j*sy_list[Nm]) + Jzcoup * ((sz_list[Nspin1])) * sz_list[Nm]
    #full hamiltonian
    H = H0 + Hi 
    H1= H01 + Hi 
    H2 = H02 + Hi 
    #H0 basis
    eigvalsH0, eigvecsH0 = H0.eigenstates()
    #Hi basis
    #eigvalsHi, eigvecsHi = Hi.eigenstates()
    #operators that we want to measure the average value of
    #e_ops = [sz_list[Ns]]
    #initialstate
    inket2=tensor(spin_state(1/2,1/2),spin_state(1/2,-1/2))
    inket3=tensor(inket2,spin_state(1/2,1/2))
    inket4=tensor(inket3,spin_state(1/2,-1/2))
    inket5=tensor(inket4,spin_state(1/2,1/2))
    inket6=tensor(inket5,spin_state(1/2,-1/2))
    inket7=tensor(inket6,spin_state(1/2,1/2))
    inket8=tensor(inket7,spin_state(1/2,-1/2))
    inket9=tensor(inket8,spin_state(1/2,1/2))
    inket10=tensor(inket9,spin_state(1/2,-1/2))
    inket11=tensor(inket10,spin_state(1/2,1/2))
    inket12=tensor(inket11,spin_state(1/2,-1/2))
    inket13=tensor(inket12,spin_state(1/2,1/2))
    inket14=tensor(inket13,spin_state(1/2,-1/2))
    inket15=tensor(inket14,spin_state(1/2,1/2))
    inket16=tensor(inket15,spin_state(1/2,-1/2))
    inket17=tensor(inket16,spin_state(1/2,1/2))
    inket=eigvecsH0[1000]
    #inket=inket12
    #initial energy
    #init=inket14.dag()*H
    #initialenergy=init*inket14
    #print(initialenergy)
    #value = min(eigvalsH0, key=lambda x:abs(x-initialenergy))
    #print(value)
    #newlist=[i for i in eigvalsH0]
    #index = newlist.index(value)
    #print(index)
    #print(eigvalsH0[index])
    #Schrodinger
    result = sesolve(H, inket, tlist, options=options1)
    result1 = sesolve(H1, inket, tlist, options=options1)
    result2 = sesolve(H2, inket, tlist, options=options1)
    #WaveFunctionStates
    wavefunc=result.states
    wavefuncplus=result1.states
    wavefuncminus=result2.states
    #derivative
    #fisherinformation
    F=0
    listF=[]
    for i in range(len(wavefunc)):
        F = 4*((wavefuncplus[i].dag()-wavefuncminus[i].dag())/(2*a))*((wavefuncplus[i]-wavefuncminus[i])/(2*a)) - 4*(wavefunc[i].dag()*((wavefuncplus[i]-wavefuncminus[i])/(2*a)))*(((wavefuncplus[i].dag()-wavefuncminus[i].dag())/(2*a))*wavefunc[i])
        listF.append(F)
    #newesttimeslist = newtimeslist[:250]
    newlistforFisher = []
    for k in range(len(tlist)):
        newlistforFisher.append(listF[k].real)
    #print(newlistforFisher[0])
    #print(newlistforFisher[-1])
    for a in range(len(tlist)):
        print(newlistforFisher[a])
    np.savetxt('coolertext',newlistforFisher)
    #plotting
    fig1 = plt.figure(1)
    plt.plot(result.times, newlistforFisher, '-')
    label_size = 20
    label_size2 = 15
    label_size3 = 15
    plt.rc('text', usetex = True)
    plt.title(r'$F_\chi (t)$',
            fontsize = label_size2)
    
    plt.rc('xtick', labelsize=label_size) 
    plt.rc('ytick', labelsize=label_size)

    plt.ylim([0,10000000])
    plt.xlim([0,10000])

    plt.xlabel(r'$t$', fontsize = label_size3)
    plt.ylabel(r'$F_\chi$', fontsize = label_size3)

    fname = 'figures/btc_eig_N{}_strong_jmat.pdf'.format(N)
    savefile = False
    plt.show()
        
    if savefile == True:    
        plt.savefig(fname, bbox_inches='tight')

        plt.show()
        plt.close()        
if __name__ == "__main__":
    main()