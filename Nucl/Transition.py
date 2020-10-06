#!/usr/bin/env python3
import numpy as np
from sympy import N
from sympy.physics.wigner import wigner_3j
from scipy.special import factorial2
from scipy.constants import physical_constants
def Rp2_to_Rch2( Rp2, Z, N ):
    """
    inputs:
        Rp2: mean squared point proton radius
        Z: proton number
        N: neutron number
    output:
        mean squared charge radii
    """
    rcp2 = 0.8783**2 # CODATA
    rcn2 = -0.115    # CODATA
    #rcp2 = 0.709     # Nature 466, 213 (2010).
    #rcn2 = -0.106    # Phys. Rev. Lett. 124, 082501

    DF = 0.033
    return Rp2 + rcp2 + N/Z * rcn2 + DF

def ME_to_moment( ME, J, lam ):
    """
    inputs:
        ME = (J || O^lam || J)
        J: Angular momentum of the state
        lam: rank of operator
    output:
        static moment
    """
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
    if(EM=="E"): return 5.498e22 * (ediff / hc)**(2*rank+1) * (rank+1) / (rank * factorial2(2*lam+1)**2) * BEM(ME, Jinit)
    if(EM=="M"): return 6.080e20 * (ediff / hc)**(2*rank+1) * (rank+1) / (rank * factorial2(2*lam+1)**2) * BEM(ME, Jinit)
