#!/usr/bin/env python3
import numpy as np
from scipy.interpolate import UnivariateSpline
import scipy.integrate as integrate
from scipy.special import spherical_jn
from scipy.constants import physical_constants

hc = physical_constants['reduced Planck constant times c in MeV fm'][0]
m_n = physical_constants['neutron mass energy equivalent in MeV'][0]
m_p = physical_constants['proton mass energy equivalent in MeV'][0]
m_nucl = (m_p + m_n)*0.5

def CMCorrection(r, rho, r_out, b2CM):
    q = np.linspace(0, 8, 100)
    F = Rho2F(r, rho, q, b2CM)
    return F2Rho(q, F, r_out)
def Rho2F(r, rho, q_out, b2CM=None, rank=0):
    """
    rank: rank of the density (usually 0)
    b2CM: (hc)**2 / A m_nucl hwCM in unit of fm2
    output form factor is normalized such that F(q=0) = int dr r^2 rho(r)
    """
    CMcorrection = [1] * len(q_out)
    if(b2CM!=None): CMcorrection = np.exp(q_out**2 * b2CM**2 * 0.25)
    spl = UnivariateSpline(r, rho, s=0, k=4)
    F = []
    for idx, Q in enumerate(q_out):
        f = 4*np.pi*integrate.quad(lambda x: x**2 * spherical_jn(rank,x*Q) * spl(x), 0, max(r))[0]
        f *= CMcorrection[idx]
        F.append(f)
    return np.array(F)

def F2Rho(q, F, r_out, rank=0):
    """
    rank: rank of the density (usually 0)
    b2CM: (hc)**2 / A m_nucl hwCM in unit of fm2
    output form factor is normalized such that F(q=0) = int dr r^2 rho(r)
    """
    spl = UnivariateSpline(q, F, s=0, k=4)
    rho = []
    for r in r_out:
        tmp = integrate.quad(lambda q: q**2 * spherical_jn(rank,q*r) * spl(q), 0, max(q))[0]
        tmp *= 4*np.pi / (2*np.pi)**3
        rho.append(tmp)
    return np.array(rho)

def DensityIntegral(r, rho, power=0):
    spl = UnivariateSpline(r, rho, s=0, k=4)
    return 4*np.pi*integrate.quad(lambda x: x**(2+power) * spl(x), 0, max(r))[0]

def GetB2CM(hwCM, A):
    return hc**2 / (A * m_nucl * hwCM)
