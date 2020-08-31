#
# some functions for nuclear matter parameters
#
def extract_parameters( d_snm, e_snm, d_pnm, e_pnm, n0=None ):
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
    pnm_spl = UnivariateSpline(d_pnm,e_pnm,s=0,k=4)
    if(n0==None):
        res = minimize(snm_spl,(min(d_snm)+max(d_snm))/2,bounds=((min(d_snm),max(d_snm)),))
        n0 = res.x[0]
        e0  = res.fun[0]
    else:
        e0 = snm_spl(n0)
    K = snm_spl.derivatives(n0)[2] * 9 * n0**2
    S = pnm_spl(n0) - e0
    L = pnm_spl.derivatives(n0)[1] * 3 * n0
    return n0,e0,K,S,L
