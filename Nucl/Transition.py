#!/usr/bin/env python3
import numpy as np
from sympy import N
from sympy.physics.wigner import wigner_3j
from scipy.special import factorial2
from scipy.constants import physical_constants
def Rp2_to_Rch2( Rp2, Z, N, fs_corrections=None):
    """
    inputs:
        Rp2: mean squared point proton radius
        Z: proton number
        N: neutron number
    output:
        mean squared charge radii
    """
    if(fs_corrections==None):
#        rcp2 = 0.8783**2 # CODATA
#        rcn2 = -0.115    # CODATA
#        rcp2 = 0.709     # Nature 466, 213 (2010).
#        rcn2 = -0.106    # Phys. Rev. Lett. 124, 082501
        rcp2 = 0.8409**2 # PDG 2024 
        rcn2 = -0.1155 # PDG 2024
    else:
        rcp2, rcn2 = fs_corrections   

    DF = 0.033
    return Rp2 + rcp2 + N/Z * rcn2 + DF
def Rch2_to_Rp2( Rch2, Z, N, fs_corrections=None):
    if(fs_corrections==None):
#        rcp2 = 0.8783**2 # CODATA
#        rcn2 = -0.115    # CODATA
#        rcp2 = 0.709     # Nature 466, 213 (2010).
#        rcn2 = -0.106    # Phys. Rev. Lett. 124, 082501
        rcp2 = 0.8409**2 # PDG 2024 
        rcn2 = -0.1155 # PDG 2024
    else:
        rcp2, rcn2 = fs_corrections   
    DF = 0.033
    return Rch2 - rcp2 - N/Z * rcn2 - DF

def ME_to_moment( ME, J, lam ):
    """
    inputs:
        ME = (J || O^lam || J)
        J: Angular momentum of the state
        lam: rank of operator
    output:
        static moment
    """
    if(lam > 2):
        print("Do not use for lam>2 case")
        return 0
    if( abs(ME) < 1.e-10 ): return 0
    return np.sqrt(4**(lam-1) * 4 * np.pi / (2*lam+1) ) * N(wigner_3j(J,lam,J,-J,0,J)) * ME

def BEM(ME, Jinit):
    """
    inputs:
        ME = (Jfinal || O^lam || Jinit)
        Jinit
    output:
        BE(lam) = ME**2 / (2*Jinit+1)
    """
    return ME**2 / (2*Jinit+1)

def ME_to_inverse_half_life( ME, Jinit, lam, Ediff, EM ):
    """
    inputs:
        ME = (Jfinal || O^lam || Jinit)
        EM: "E" or "M"
    output:
        inverse life time (additive quantity)
        ln2 / T_1/2
    """
    if( abs(ME) < 1.e-10 ): return 0
    hc = physical_constants["Planck constant over 2 pi times c in MeV fm"][0]
    if(EM=="E"): return 5.498e22 * (Ediff / hc)**(2*lam+1) * (lam+1) / (lam * factorial2(2*lam+1)**2) * BEM(ME, Jinit)
    if(EM=="M"): return 6.080e20 * (Ediff / hc)**(2*lam+1) * (lam+1) / (lam * factorial2(2*lam+1)**2) * BEM(ME, Jinit)

def RME_to_ME( RME, Jbra, lam, Jket, Mbra, mu, Mket):
    return RME * (-1)**(Jbra-Mbra) * N(wigner_3j(Jbra,lam,Jket,-Mbra,mu,Mket))
def ME_to_RME( ME, Jbra, lam, Jket, Mbra, mu, Mket):
    return ME / (-1)**(Jbra-Mbra) * N(wigner_3j(Jbra,lam,Jket,-Mbra,mu,Mket))

def mu_sp(l,j,tz,gs=None,gl=None):
    """
    inputs:
        l: orbital angular momentum
        j: total single-particle angular momentum
       tz: z-compoent of isotpin -1:proton 1:neutron
    outputs:
        Single-particle magnetic moment in the unit of mu_N (Schmidt limit)
    """
    if(tz==-1):
        if(gl==None): gl = 1
        if(gs==None): gs = 5.586
    if(tz== 1):
        if(gl==None): gl = 0
        if(gs==None): gs = -3.826
    if(j == l+0.5): return gl*l + 0.5*gs
    elif(j == l-0.5): return (gl*(l+1) - 0.5*gs)*j/(j+1)
    else:
        print("Error: j has to be l-1/2 or l+1/2")
        return None
def Q_sp(j,A):
    """
    inputs:
        j: total single-particle angular momentum
        A: mass number
    outputs:
        Single-particle Q moment in the unit of e^2 fm^2
        (3-4*j*(j+1))/(2*(j+1)*(2*j+3)) * \int dr r^4 rho(r)
        if we take rho as the squre well, it becomes (3-4*j*(j+1))/(2*(j+1)*(2*j+3)) * 3 / 5 * R^2
    """
    R = 1.2 * A **(1/3)
    return (3-4*j*(j+1))/(2*(j+1)*(2*j+3)) * 3 * R**2 / 5

if(__name__=="__main__"):
    print(mu_sp(4,4.5,-1))
    print(mu_sp(1,0.5,-1))
    print(mu_sp(1,1.5,1))
    #print(mu_sp(1,1.5,1))
    #print(mu_sp(3,2.5,1))
    #print(mu_sp(3,3.5,1))
