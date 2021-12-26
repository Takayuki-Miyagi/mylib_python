#!/usr/bin/env python3
import numpy as np
from scipy.special import gamma, assoc_laguerre, eval_gegenbauer, eval_genlaguerre
from scipy import integrate
from scipy.constants import physical_constants
from sympy import N
from sympy.physics.wigner import wigner_3j, wigner_6j, wigner_9j

def HO_radial(r, n, l, hw):
    hc = physical_constants['reduced Planck constant times c in MeV fm'][0]
    m_n = physical_constants['neutron mass energy equivalent in MeV'][0]
    m_p = physical_constants['proton mass energy equivalent in MeV'][0]
    m_nucl = (m_p + m_n)*0.5
    b = np.sqrt(hc**2 / (m_nucl * hw))
    x = r / b
    return np.sqrt( (2.0/b) * (gamma(n+1) / gamma(n+l+1.5)) ) * (1.0/b) * (x**l) * \
            np.exp(-0.5*x*x) * assoc_laguerre(x*x, n, l+0.5)

def Yl_red(l1, l2, l):
    """
    < l1 || Y_l || l2 >
    Note: all inputs are not not doubled
    """
    return (-1)**l1 * np.sqrt((2*l1+1)*(2*l2+1)*(2*l+1) / (4*np.pi)) * N(wigner_3j(l1, l, l2, 0, 0, 0))

def Ysigma(l1, j1, l2, j2, l, s, k):
    """
    < l1, j1 || [Y_l sigma_s]_k || l2, j2 >
    Note: all inputs are not not doubled
    """
    sfact = np.sqrt(2.0)
    if(s==1): sfact = np.sqrt(6.0)
    return np.sqrt(float((2*j1+1)*(2*j2+1)*(2*k+1))) * N(wigner_9j(l1, 0.5, j1, l2, 0.5, j2, l, s, k, prec=8)) * Yl_red(l1, l2, l) * sfact

if(__name__=="__main__"):
    print(HO_radial(0, 0, 0, 20))
    print(Ysigma(0, 0.5, 0, 0.5, 0, 1, 1))
