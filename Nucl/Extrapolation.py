#!/usr/bin/env python3
import numpy as np
from scipy.constants import physical_constants
from sympy.functions.special.bessel import jn_zeros
from scipy.optimize import curve_fit
from scipy.special import gammainc, gamma, erfc
from scipy.interpolate import UnivariateSpline

hc = physical_constants['reduced Planck constant times c in MeV fm'][0]
m_n = physical_constants['neutron mass energy equivalent in MeV'][0]
m_p = physical_constants['proton mass energy equivalent in MeV'][0]
m_nucl = (m_p + m_n)*0.5

"""
Extrapolation for E3max and (emax, hw)
"""
def extrap_e3max(x, y, x_eval, par_init=[0,0,10,5], n_power=2, n_samples=10000, bounds=[(-np.inf,-1.e4,8,1.e-8),(np.inf,1.e4,24,10)], is_err=True):
    """
    f = A \gamma_{\frac{2}{n}} \left[\left(\frac{E_{\rm 3max} - \mu}{\sigma}\right)^{n}\right] + C
    x: E3max (list like)
    y: Energy (list like)
    x_eval: E3max used for the interpolation (list like)
    n_power: power of the fitting function
    par_init: initial parameter for the fitting: (A, C, mu, sigma)
    n_samples: number of samples generated with the fitted covariance
    """
    def _E3form(_, A, C, mu, sigma):
        return A*gammainc(2/float(n_power), abs((_-mu)/sigma)**n_power )+C
    popt, pcov = curve_fit(_E3form, x, y, par_init, bounds=bounds)
    samples = np.random.multivariate_normal(popt, pcov, n_samples)
    if(is_err):
        f, f_err = [], []
        for _ in x_eval:
            energies = []
            for sample in samples:
                energies.append(_E3form(_,*sample))
            energies = np.array(energies)
            f.append(energies.mean())
            f_err.append(energies.std())
        spl_c = UnivariateSpline(x_eval,np.array(f),s=0,k=4)
        spl_l = UnivariateSpline(x_eval,np.array(f)-np.array(f_err),s=0,k=4)
        spl_u = UnivariateSpline(x_eval,np.array(f)+np.array(f_err),s=0,k=4)
        return spl_c, spl_l, spl_u
    else:
        f = _E3form(x_eval, *popt)
        spl = UnivariateSpline(x_eval,np.array(f),s=0,k=4)
        return spl

def L2(N, hw):
    return np.sqrt( 2*(N+3.5)) * np.sqrt(hc**2 / (hw*m_nucl) )
def Leff_approx(N, hw, occs):
    r1, r2 = 0, 0
    for key, occ in occs.items():
        n, l = key
        if((N+l)%2==0): nl = N
        if((N+l)%2==1): nl = N-1
        anl2 = float(jn_zeros(l,n+1)[-1]**2)
        knl2 = anl2 / L2(nl,hw)**2
        r1 += occ * anl2
        r2 += occ * knl2
    Leff = np.sqrt(r1/r2)
    Neff = (Leff**2 / hc**2 * hw * m_nucl) * 0.5 - 3.5
    return L2(Neff,hw)


def extrap_emax(df, x_eval, occs, UVcut=550, n_sample=10000, is_err=True):
    """
    f = En + a0*np.exp(-2*k*L)
    df: DataFrame type; it at least should have entries "emax", "hw", and "En"
    x_eval: emax used for the interpolation (list like)
    occs: dictionary {(n0, l0): occ number, (n1, l1): occ number, ...} 
        ex.) For 16O, it should be {(0,0): 4, (0,1): 12}
    n_samples: number of samples generated with the fitted covariance
    """
    def energy_L(L, En, k, a0):
        return En + a0*np.exp(-2*k*L)
    df["L2"] = L2(df["emax"], df["hw"])
    Leffs = []
    for index, row in df.iterrows():
        Leffs.append(Leff_approx(row['emax'], row['hw'], occs))
    df["Leff"] = Leffs
    df["LUV"] = np.sqrt(2*(df["emax"])) / np.sqrt(hc**2 / (df["hw"] * m_nucl)) * hc
    _ = df[df["LUV"]>UVcut]
    A = sum([x for x in occs.values()])
    Ens, Ls = df["En"].to_numpy(), df["Leff"].to_numpy()
    E_ref, L_ref = min(Ens), max(Ls)
    k_ref = np.sqrt(abs(E_ref *  1.e3)) / hc / 10
    a_ref = (max(Ens)-min(Ens)) / (np.exp(-2*k_ref*min(Ls)) - np.exp(-2*k_ref*max(Ls)))
    p0 = [E_ref, k_ref, 1.e7] 
    popt, pcov = curve_fit(energy_L, _["Leff"], _["En"], p0)
    if(is_err):
        f, f_err = [], []
        for _ in x_eval:
            energies = []
            for sample in samples:
                energies.append(energy_L(_,*sample))
            energies = np.array(energies)
            f.append(energies.mean())
            f_err.append(energies.std())
        spl_c = UnivariateSpline(x_eval,np.array(f),s=0,k=4)
        spl_l = UnivariateSpline(x_eval,np.array(f)-np.array(f_err),s=0,k=4)
        spl_u = UnivariateSpline(x_eval,np.array(f)+np.array(f_err),s=0,k=4)
        return _, spl_c, spl_l, spl_u
    else:
        f = energy_L(x_eval, *popt)
        spl = UnivariateSpline(x_eval,np.array(f),s=0,k=4)
        return _, spl

    


