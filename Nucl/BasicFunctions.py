#!/usr/bin/env python3
import numpy as np
from scipy.special import gamma, assoc_laguerre, eval_gegenbauer, eval_genlaguerre
from scipy import integrate
from scipy.constants import physical_constants
from sympy import N
from scipy.special import sph_harm
from sympy.physics.wigner import wigner_3j, wigner_6j, wigner_9j, clebsch_gordan

hc = physical_constants['reduced Planck constant times c in MeV fm'][0]
m_n = physical_constants['neutron mass energy equivalent in MeV'][0]
m_p = physical_constants['proton mass energy equivalent in MeV'][0]
m_nucl = (m_p + m_n)*0.5
pauli1 = np.array([[1,0],[0,1]])
paulix = np.array([[0,1],[1,0]])
pauliy = np.array([[0,-1j],[1j,0]])
pauliz = np.array([[1,0],[0,-1]])
def HO_radial(r, n, l, hw):
    hc = physical_constants['reduced Planck constant times c in MeV fm'][0]
    m_n = physical_constants['neutron mass energy equivalent in MeV'][0]
    m_p = physical_constants['proton mass energy equivalent in MeV'][0]
    m_nucl = (m_p + m_n)*0.5
    b = np.sqrt(hc**2 / (m_nucl * hw))
    x = r / b
    return np.sqrt( (2.0/b) * (gamma(n+1) / gamma(n+l+1.5)) ) * (1.0/b) * (x**l) * \
            np.exp(-0.5*x*x) * assoc_laguerre(x*x, n, l+0.5)

def RadialInt(n1, l1, n2, l2, hw, power):
    res = integrate.quad(lambda r: r**(power+2) * HO_radial(r, n1, l1, hw) * HO_radial(r, n2, l2, hw), 0, np.inf)
    return res[0]

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

def QdotQ(hw, n1, l1, n2, l2):
    if(abs(n1 - n2) > 2): return 0
    if(l1 != l2): return 0
    me = 0.0
    for n in range(12):
        for l in range(abs(l1-2), l1+3):
            me += wigner_6j(2, 2, 0, l2, l1, l) * multipole_me(hw, 2, n1, l1, n, l) * multipole_me(hw, 2, n, l, n2, l2)
    return me / np.sqrt(5)


def multipole_me(hw, lam, n1, l1, n2, l2):
    """
    set < p || Q || q >
    """
    me = RadialInt(n1, l1, n2, l2, hw, lam) * \
            (-1.0)**l1 * np.sqrt((2*lam+1)*(2*l1+1)*(2*l2+1)/ (4*np.pi)) * float(wigner_3j(l1, lam, l2, 0, 0, 0))
    return me

def get_spherical(Q):
    Ql = np.sqrt(np.dot(Q,Q))
    theta = np.arccos(Q[2]/Ql)
    if(Q[0]**2 > 1.e-8 and np.dot(Q[:2], Q[:2]) > 1.e-8):
        phi = np.arccos(Q[0]/np.sqrt(np.dot(Q[:2],Q[:2])))
    else:
        phi = 0
    if(Q[1]<0): phi*=-1
    return Ql, theta, phi

def e_sph(m):
    if(m==0): return np.array([0,0,1])
    if(m==-1): return np.array([1/np.sqrt(2),-1j/np.sqrt(2),0])
    if(m== 1): return np.array([-1/np.sqrt(2),-1j/np.sqrt(2),0])

def sigma(m):
    if(m==0): return pauliz
    if(m==-1): return (paulix - pauliy * 1j)/np.sqrt(2)
    if(m== 1): return -(paulix + pauliy * 1j)/np.sqrt(2)

def Ylm(Q, l, m):
    Ql, theta, phi = get_spherical(Q)
    return sph_harm(m, l, phi, theta)

def YJLM(Q,J,L,M):
    Ql, theta_Q, phi_Q = get_spherical(Q)
    Yjlm = np.zeros(3,dtype=np.complex)
    for lam in [-1,0,1]:
        if(abs(M-lam)>L): continue
        Yjlm += np.float(clebsch_gordan(L,1,J,M-lam,lam,M).evalf()) * Ylm(Q,L,M-lam) * e_sph(lam)
    return Yjlm


def VecYLM(Q,L,M):
    """
    \vec{Y}_{LM}(\hat{\vec{Q}})
    """
    Ql, theta_Q, phi_Q = get_spherical(Q)
    if(M > L): return np.zeros(3,dtype=np.complex)
    return Ylm(Q,L,M) * Q / Ql

def VecPsi(Q,L,M):
    """
    \vec{\Psi}_{LM}(\hat{\vec{Q}})
    """
    Ql, theta_Q, phi_Q = get_spherical(Q)
    Psi = np.zeros(3,dtype=np.complex)
    for lam in [-1,0,1]:
        if(abs(M-lam)>L-1): continue
        Psi += np.float(clebsch_gordan(L-1,1,L,M-lam,lam,M).evalf()) * Ylm(Q,L-1,M-lam) * e_sph(lam) * np.sqrt(L+1)
    for lam in [-1,0,1]:
        if(abs(M-lam)>L+1): continue
        Psi += np.float(clebsch_gordan(L+1,1,L,M-lam,lam,M).evalf()) * Ylm(Q,L+1,M-lam) * e_sph(lam) * np.sqrt(L)
    Psi *= np.sqrt(1/(2*L+1))
    return Psi

def VecPhi(Q,L,M):
    """
    \vec{\Phi}_{LM}(\hat{\vec{Q}})
    """
    Ql, theta_Q, phi_Q = get_spherical(Q)
    Phi = np.zeros(3,dtype=np.complex)
    for lam in [-1,0,1]:
        if(abs(M-lam)>L): continue
        Phi += np.float(clebsch_gordan(L,1,L,M-lam,lam,M).evalf()) * Ylm(Q,L,M-lam) * e_sph(lam)
    return Phi

def Sig1(s1, s2):
    return pauli1[(1-s1)//2, (1-s2)//2]
def SigX(s1, s2):
    return paulix[(1-s1)//2, (1-s2)//2]
def SigY(s1, s2):
    return pauliy[(1-s1)//2, (1-s2)//2]
def SigZ(s1, s2):
    return pauliz[(1-s1)//2, (1-s2)//2]
def SigVec(s1, s2):
    return np.array([SigX(s1,s2), SigY(s1,s2), SigZ(s1,s2)])

if(__name__=="__main__"):
    print(HO_radial(0, 0, 0, 20))
    print(Ysigma(0, 0.5, 0, 0.5, 0, 1, 1))
