#
# some functions for nuclear matter parameters
#
def extract_parameters( d_snm, e_snm, d_pnm=None, e_pnm=None, n0=None ):
    #
    # input:
    #   d_snm, e_snm: densities and energies of symmetric nuclear matter
    #   d_pnm, e_pnm: densities and energies of pure neutron matter
    # output:
    #   n0: saturation density
    #   e0: saturation energy
    #    K: incompressibility
    #    S: symmetry energy
    #    L: slope parameter
    # Ksym: parameter
    from scipy.interpolate import UnivariateSpline
    from scipy.optimize import minimize
    snm_spl = UnivariateSpline(d_snm,e_snm,s=0,k=4)
    if(n0==None):
        res = minimize(snm_spl,(min(d_snm)+max(d_snm))/2,bounds=((min(d_snm),max(d_snm)),))
        n0 = res.x[0]
        e0  = res.fun[0]
    else:
        e0 = snm_spl(n0)
    K = snm_spl.derivatives(n0)[2] * 9 * n0**2
    M = snm_spl.derivatives(n0)[3] * 27 * n0**3 # cubic term
    if(d_pnm==None and e_pnm==None): return n0, e0, K, M, 0, 0, 0

    pnm_spl = UnivariateSpline(d_pnm,e_pnm,s=0,k=4)
    S = pnm_spl(n0) - e0
    L = pnm_spl.derivatives(n0)[1] * 3 * n0
    Ksym = pnm_spl.derivatives(n0)[2] * 9 * n0**2 - K
    return n0,e0,K,M,S,L,Ksym

def extract_symmetry_energy_parameters( d_snm, e_snm, d_pnm, e_pnm, n0=None ):
    #
    # input:
    #   d_snm, e_snm: densities and energies of symmetric nuclear matter
    #   d_pnm, e_pnm: densities and energies of pure neutron matter
    # output:
    #   n0: saturation density
    #   e0: saturation energy
    #    K: incompressibility
    #    S: symmetry energy
    #    L: slope parameter
    from scipy.interpolate import UnivariateSpline
    from scipy.optimize import minimize
    snm_spl = UnivariateSpline(d_snm,e_snm,s=0,k=4)
    if(n0==None):
        res = minimize(snm_spl,(min(d_snm)+max(d_snm))/2,bounds=((min(d_snm),max(d_snm)),))
        n0 = res.x[0]
        e0  = res.fun[0]
    else:
        e0 = snm_spl(n0)
    e_sym = []
    for i in range(min(len(e_snm), len(e_pnm))):
        dn_snm = d_snm[i]
        if( dn_snm != d_pnm[i] ): continue
        e_sym.append( e_pnm[i] - e_snm[i] )
    sym_spl = UnivariateSpline(d_pnm,e_sym,s=0,k=4)
    S = sym_spl(n0)
    L = sym_spl.derivatives(n0)[1] * 3 * n0
    K = sym_spl.derivatives(n0)[2] * 9 * n0**2
    return n0, S, L, K
