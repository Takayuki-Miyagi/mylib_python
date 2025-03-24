#!/usr/bin/env python3
import sys, os, subprocess, itertools, math
import numpy as np
import functools
import copy
import gzip
from scipy.constants import physical_constants
from scipy.special import gamma
from sympy.physics.wigner import wigner_3j, wigner_6j, wigner_9j, clebsch_gordan
import pandas as pd
if(__package__==None or __package__==""):
    from Orbits import Orbits, OrbitsIsospin
    import ModelSpace
    import nushell2snt
    import BasicFunctions
else:
    from . import Orbits, OrbitsIsospin
    from . import ModelSpace
    from . import nushell2snt
    from . import BasicFunctions

_paulix = np.array([[0,1],[1,0]])
_pauliy = np.array([[0,-1j],[1j,0]])
_pauliz = np.array([[1,0],[0,-1]])
@functools.lru_cache(maxsize=None)
def _sixj(j1, j2, j3, j4, j5, j6):
    return float(wigner_6j(j1, j2, j3, j4, j5, j6))
@functools.lru_cache(maxsize=None)
def _ninej(j1, j2, j3, j4, j5, j6, j7, j8, j9):
    return float(wigner_9j(j1, j2, j3, j4, j5, j6, j7, j8, j9))
@functools.lru_cache(maxsize=None)
def _clebsch_gordan(j1, j2, j3, m1, m2, m3):
    return float(clebsch_gordan(j1, j2, j3, m1, m2, m3))
def _ls_coupling(la, ja, lb, jb, Lab, Sab, J):
    return np.sqrt( (2*ja+1)*(2*jb+1)*(2*Lab+1)*(2*Sab+1) ) * \
            np.float( wigner_9j( la, 0.5, ja, lb, 0.5, jb, Lab, Sab, J) )

def _ljidx_to_lj(lj):
    return math.floor((lj+1)/2), math.floor(2*(int(lj/2) + 1/2))

def _lj_to_ljidx(l,j):
    return math.floor(l+j/2-1/2)

class Operator:
    def __init__(self, rankJ=0, rankP=1, rankZ=0, ms=None, reduced=True, filename=None, verbose=False, comment="!", p_core=None, n_core=None, skew=False):
        self.ms = ms
        self.rankJ = rankJ
        self.rankP = rankP
        self.rankZ = rankZ
        self.reduced = reduced
        self.verbose = verbose
        self.zero = 0.0
        self.one = None
        self.two = {}
        self.three = {}
        self.p_core = p_core
        self.n_core = n_core
        self.kshell_options = []
        self.ls_couple_store = {}
        self.sixj_store = {}
        self.skew = skew
        if( self.rankJ == 0 and self.rankP==1 and self.rankZ==0): self.reduced = False
        if( ms != None ): self.allocate_operator( ms )
        if( filename != None ): self.read_operator_file( filename, comment=comment )

    def set_options(self, **kwargs):
        if('ms' in kwargs): self.ms = kwargs['ms']
        if('rankJ' in kwargs): self.rankJ = kwargs['rankJ']
        if('rankP' in kwargs): self.rankP = kwargs['rankP']
        if('rankZ' in kwargs): self.rankZ = kwargs['rankZ']
        if('reduced' in kwargs): self.reduced = kwargs['reduced']
        if('verbose' in kwargs): self.verbose = kwargs['verbose']
        if('zero' in kwargs): self.zero = kwargs['zero']
        if('one' in kwargs): self.one = kwargs['one']
        if('two' in kwargs): self.two = kwargs['two']
        if('three' in kwargs): self.three = kwargs['three']
        if('p_core' in kwargs): self.p_core = kwargs['p_core']
        if('n_core' in kwargs): self.n_core = kwargs['n_core']
        if('kshell_options' in kwargs): self.kshell_options = kwargs['kshell_options']

    def __add__(self, other):
        if(self.rankJ != other.rankJ): raise ValueError
        if(self.rankP != other.rankP): raise ValueError
        if(self.rankZ != other.rankZ): raise ValueError
        if(self.reduced != other.reduced): raise ValueError
        if(self.skew != other.skew): raise ValueError
        target = Operator(ms=self.ms, rankJ=self.rankJ, rankP=self.rankP, rankZ=self.rankZ, reduced=self.reduced, \
                p_core=self.p_core, n_core=self.n_core, skew=self.skew)
        target.zero = self.zero + other.zero
        target.one = self.one + other.one
        for channels in self.two.keys():
            for idxs in self.two[channels].keys():
                me = self.two[channels][idxs]
                target.set_2bme_from_mat_indices(*channels,*idxs,me)
        for channels in other.two.keys():
            for idxs in other.two[channels].keys():
                me1 = other.two[channels][idxs]
                me2 = self.get_2bme_from_mat_indices(*channels,*idxs)
                target.set_2bme_from_mat_indices(*channels,*idxs,me1+me2)
        return target

    def __sub__(self, other):
        if(self.rankJ != other.rankJ): raise ValueError
        if(self.rankP != other.rankP): raise ValueError
        if(self.rankZ != other.rankZ): raise ValueError
        if(self.reduced != other.reduced): raise ValueError
        if(self.skew != other.skew): raise ValueError
        target = Operator(ms=self.ms, rankJ=self.rankJ, rankP=self.rankP, rankZ=self.rankZ, reduced=self.reduced, \
                p_core=self.p_core, n_core=self.n_core, skew=self.skew)
        target.zero = self.zero - other.zero
        target.one = self.one - other.one
        for channels in self.two.keys():
            for idxs in self.two[channels].keys():
                me = self.two[channels][idxs]
                target.set_2bme_from_mat_indices(*channels,*idxs,me)
        for channels in other.two.keys():
            for idxs in other.two[channels].keys():
                me1 = other.two[channels][idxs]
                me2 = self.get_2bme_from_mat_indices(*channels,*idxs)
                target.set_2bme_from_mat_indices(*channels,*idxs,me2-me1)
        return target

    def __mul__(self, coef):
        target = Operator(ms=self.ms, rankJ=self.rankJ, rankP=self.rankP, rankZ=self.rankZ, reduced=self.reduced, \
                p_core=self.p_core, n_core=self.n_core, skew=self.skew)
        target.zero = self.zero * coef
        target.one = self.one * coef
        for channels in self.two.keys():
            for idxs in self.two[channels].keys():
                me1 = self.two[channels][idxs]
                target.set_2bme_from_mat_indices(*channels,*idxs,me1*coef)
        return target

    def __truediv__(self, coef):
        return self.__mul__(1/coef)

    def espe(self, occs, method="no", bare=False):
        """
        calculate effective single-particle energies with occupations
            occs: dictionary {(n1,l1,j1,tz1):occ1, (n2,l2,j2,tz2):occ2, ...}
            j and tz has to be double (integer)
        return
            espes: dictionary {(n1,l1,j1,tz1):espe1, (n2,l2,j2,tz2):espe2, ...}
        """
        if(self.ms.rank==3): raise "ESPEs with three-body is not implemented."
        if(self.rankJ!=0): raise "Operator rank should be 0"
        if(self.rankP!=1): raise "Operator parity should be 1"
        if(self.rankZ!=0): raise "Operator pn rank should be 0"
        A = self.p_core + self.n_core
        for orb in occs.keys():
            A += occs[orb] * float(orb[2]+1)
        mass_dep = 1
        if(self.kshell_options[0]==1): mass_dep =(float(A) / float(self.kshell_options[1]))**float(self.kshell_options[2])
        orbits = self.ms.orbits
        espes = {}
        for a in range(1, orbits.get_num_orbits()+1):
            oa = orbits.get_orbit(a)
            espe = 0.0
            for b in range(1, orbits.get_num_orbits()+1):
                ob = orbits.get_orbit(b)
                norm = 1.0
                if(a==b): norm = 2.0
                try:
                    occ = occs[(ob.n, ob.l, ob.j, ob.z)] # 0 <= occ <= 1
                except:
                    if(self.verbose): print("occupation is not given, n,l,j,z:", ob.n, ob.l, ob.j, ob.z)
                    occ = 0.0
                if(method=="no"):
                    Jmin = abs(oa.j-ob.j)//2
                    Jmax =    (oa.j+ob.j)//2
                    sumV = 0.0
                    for J in range(Jmin,Jmax+1):
                        if(a==b and J%2==1): continue
                        sumV += self.get_2bme_from_indices(a,b,a,b,J,J) * (2*J+1) * mass_dep
                    espe += sumV * occ / (oa.j+1) * norm
                elif(method=="monopole"):
                    v = self.get_2bme_monopole(a,b,a,b) / norm # unnormalized -> normalized
                    espe += v * occ * (ob.j+1) * mass_dep
            if(bare): espes[(oa.n,oa.l,oa.j,oa.z)] = self.get_1bme(a,a)
            else: espes[(oa.n,oa.l,oa.j,oa.z)] = espe + self.get_1bme(a,a)
        return espes

    def allocate_operator(self, ms):
        self.ms = copy.deepcopy(ms)
        self.zero = 0.0
        if(ms.orbits.get_num_orbits()>0): self.one = np.zeros( (ms.orbits.get_num_orbits(), ms.orbits.get_num_orbits() ))
        if(self.ms.rank==1): return
        two = ms.two
        for ichbra in range(ms.two.get_number_channels()):
            chbra = two.get_channel(ichbra)
            for ichket in range(ichbra+1):
                chket = two.get_channel(ichket)
                if( self._triag( chbra.J, chket.J, self.rankJ )): continue
                if( chbra.P * chket.P * self.rankP != 1): continue
                if( abs(chbra.Z-chket.Z) != self.rankZ): continue
                self.two[(ichbra,ichket)] = {}
        if(self.ms.rank==2): return
        three = ms.three
        for ichbra in range(ms.three.get_number_channels()):
            chbra = three.get_channel(ichbra)
            for ichket in range(ichbra+1):
                chket = three.get_channel(ichket)
                if( self._triag( chbra.J, chket.J, 2*self.rankJ )): continue
                if( chbra.P * chket.P * self.rankP != 1): continue
                if( self._triag( chbra.T, chket.T, 2*self.rankZ )): continue
                self.three[(ichbra,ichket)] = {}

    def count_nonzero_1bme(self):
        # count independent entries
        orbits = self.ms.orbits
        counter = 0
        for oa in orbits.orbits:
            for ob in orbits.orbits:
                a = orbits.get_orbit_index_from_orbit(oa)
                b = orbits.get_orbit_index_from_orbit(ob)
                if(b>a): continue
                if(abs(self.one[a-1,b-1]) < 1.e-10): continue
                counter += 1
        return counter

    def count_nonzero_2bme(self):
        # count independent entries
        counter = 0
        for channels in self.two.keys():
            for idxs in self.two[channels].keys():
                if(channels[0]==channels[1] and idxs[1]>idxs[0]): continue
                if(abs(self.two[channels][idxs]) < 1.e-10): continue
                counter += 1
        return counter

    def count_nonzero_3bme(self):
        # count independent entries
        counter = 0
        for channels in self.three.keys():
            for idxs in self.three[channels].keys():
                if(channels[0]==channels[1] and idxs[1]>idxs[0]): continue
                if(abs(self.three[channels][idxs]) < 1.e-10): continue
                counter += 1
        return counter

    def set_0bme( self, me ):
        self.zero = me

    def set_1bme( self, a, b, me):
        orbits = self.ms.orbits
        oa = orbits.get_orbit(a)
        ob = orbits.get_orbit(b)
        if( self._triag(oa.j, ob.j, 2*self.rankJ)): raise ValueError("Operator rank mismatch")
        if( (-1)**(oa.l+ob.l) * self.rankP != 1): raise ValueError("Operator parity mismatch")
        if( abs(oa.z-ob.z) != 2*self.rankZ): raise ValueError("Operator pn mismatch")
        if(self.skew and a==b and abs(me) > 1.e-16): print("Diagonal matrix element has to be 0")
        self.one[a-1,b-1] = me
        self.one[b-1,a-1] = me * (-1)**( (ob.j-oa.j)//2 )
        if(self.skew): self.one[b-1,a-1] *= -1

    def set_2bme_from_mat_indices( self, chbra, chket, bra, ket, me ):
        if(self.skew and chbra==chket and bra==ket and abs(me) > 1.e-16): raise ValueError("Diagonal matrix element has to be 0")
        if( chbra < chket ):
            if(self.verbose): print("Warning:" + sys._getframe().f_code.co_name )
            return
        self.two[(chbra,chket)][(bra,ket)] = me
        if( chbra == chket and self.skew): self.two[(chbra,chket)][(ket,bra)] = -me
        if( chbra == chket and not self.skew): self.two[(chbra,chket)][(ket,bra)] = me

    def set_2bme_from_indices( self, a, b, c, d, Jab, Jcd, me ):
        two = self.ms.two
        orbits = two.orbits
        oa = orbits.get_orbit(a)
        ob = orbits.get_orbit(b)
        oc = orbits.get_orbit(c)
        od = orbits.get_orbit(d)
        Pab = (-1)**(oa.l+ob.l)
        Pcd = (-1)**(oc.l+od.l)
        Zab = (oa.z + ob.z)//2
        Zcd = (oc.z + od.z)//2
        if( self._triag( Jab, Jcd, self.rankJ )): raise ValueError("Operator rank mismatch")
        if( Pab * Pcd * self.rankP != 1): raise ValueError("Operator parity mismatch")
        if( abs(Zab-Zcd) != self.rankZ): raise ValueError("Operator pn mismatch")
        ichbra = two.get_index(Jab,Pab,Zab)
        ichket = two.get_index(Jcd,Pcd,Zcd)
        phase = 1
        if( ichbra >= ichket ):
            aa, bb, cc, dd, = a, b, c, d
        else:
            ichbra, ichket = ichket, ichbra
            phase *=  (-1)**(Jcd-Jab)
            if(self.skew): phase *= -1
            aa, bb, cc, dd, = c, d, a, b
        chbra = two.get_channel(ichbra)
        chket = two.get_channel(ichket)
        bra = chbra.index_from_indices[(aa,bb)]
        ket = chket.index_from_indices[(cc,dd)]
        phase *= chbra.phase_from_indices[(aa,bb)] * chket.phase_from_indices[(cc,dd)]
        self.set_2bme_from_mat_indices(ichbra,ichket,bra,ket,me*phase)

    def set_2bme_from_orbits( self, oa, ob, oc, od, Jab, Jcd, me ):
        orbits = self.ms.orbits
        a = orbits.orbit_index_from_orbit( oa )
        b = orbits.orbit_index_from_orbit( ob )
        c = orbits.orbit_index_from_orbit( oc )
        d = orbits.orbit_index_from_orbit( od )
        self.set_2bme_from_indices( a, b, c, d, Jab, Jcd, me )

    def set_3bme_from_mat_indices( self, chbra, chket, bra, ket, me ):
        if(self.skew and chbra==chket and bra==ket and abs(me) > 1.e-16): raise ValueError("Diagonal matrix element has to be 0")
        if( chbra < chket ):
            if(self.verbose): print("Warning:" + sys._getframe().f_code.co_name )
            return
        self.three[(chbra,chket)][(bra,ket)] = me
        if(chbra == chket and self.skew): self.three[(chbra,chket)][ket,bra] = -me
        if(chbra == chket and not self.skew): self.three[(chbra,chket)][ket,bra] = me

    def set_3bme_from_indices( self, a, b, c, Jab, Tab, d, e, f, Jde, Tde, Jbra, Tbra, Jket, Tket, me ):
        three = self.ms.three
        iorbits = three.orbits
        if( not a>=b>=c ):
            print( "In three body exchange of indices is not supported. " )
            return
        if( not d>=e>=f ):
            print( "In three body exchange of indices is not supported. " )
            return
        oa = iorbits.get_orbit(a)
        ob = iorbits.get_orbit(b)
        oc = iorbits.get_orbit(c)
        od = iorbits.get_orbit(d)
        oe = iorbits.get_orbit(e)
        of = iorbits.get_orbit(f)
        ea = 2*oa.n + oa.l
        eb = 2*ob.n + ob.l
        ec = 2*oc.n + oc.l
        ed = 2*od.n + od.l
        ee = 2*oe.n + oe.l
        ef = 2*of.n + of.l
        if(ea > self.ms.emax ): return
        if(eb > self.ms.emax ): return
        if(ec > self.ms.emax ): return
        if(ed > self.ms.emax ): return
        if(ee > self.ms.emax ): return
        if(ef > self.ms.emax ): return
        if(ea+eb > self.ms.e2max ): return
        if(ea+ec > self.ms.e2max ): return
        if(eb+ec > self.ms.e2max ): return
        if(ed+ee > self.ms.e2max ): return
        if(ed+ef > self.ms.e2max ): return
        if(ee+ef > self.ms.e2max ): return
        if(ea+eb+ec > self.ms.e3max ): return
        if(ed+ee+ef > self.ms.e3max ): return
        Pbra = (-1)**(oa.l+ob.l+oc.l)
        Pket = (-1)**(od.l+oe.l+of.l)
        if( self._triag( Jbra, Jket, 2*self.rankJ )):
            if(self.verbose): print("Warning: J, " + sys._getframe().f_code.co_name )
            return
        if( Pbra * Pket * self.rankP != 1):
            if(self.verbose): print("Warning: Parity, " + sys._getframe().f_code.co_name )
            return
        if( self._triag( Tbra, Tket, 2*self.rankZ) ):
            if(self.verbose): print("Warning: Z, " + sys._getframe().f_code.co_name )
            return
        ichbra = three.get_index(Jbra,Pbra,Tbra)
        ichket = three.get_index(Jket,Pket,Tket)
        phase = 1
        if( ichbra >= ichket ):
            i, j, k, l, m, n = a, b, c, d, e, f
            Jij, Tij, Jlm, Tlm = Jab, Tab, Jde, Tde
        else:
            ichbra, ichket = ichket, ichbra
            phase *=  (-1)**((Jket+Tket-Jbra-Tbra)//2)
            i, j, k, l, m, n = d, e, f, a, b, c
            Jij, Tij, Jlm, Tlm = Jde, Tde, Jab, Tab
        chbra = three.get_channel(ichbra)
        chket = three.get_channel(ichket)
        bra = chbra.index_from_indices[(i,j,k,Jij,Tij)]
        ket = chket.index_from_indices[(l,m,n,Jlm,Tlm)]
        self.set_3bme_from_mat_indices(ichbra,ichket,bra,ket,me*phase)

    def get_0bme(self):
        return self.zero

    def get_1bme(self,a,b):
        return self.one[a-1,b-1]

    def get_1bme_Mscheme(self, p, mdp, q, mdq):
        if(not self.reduced):
            print('Convert matrix elements to reduced one first')
            return None
        orbs = self.ms.orbits
        o_p, o_q = orbs.get_orbit(p), orbs.get_orbit(q)
        me = _clebsch_gordan(o_q.j*0.5, self.rankJ, o_p.j*0.5, mdq*0.5, (mdp-mdq)*0.5, mdp*0.5) / np.sqrt(o_p.j+1) * self.get_1bme(p, q)
        return me

    def get_1bme_Mscheme_nlms(self, nlmstz1, nlmstz2):
        if(not self.reduced):
            print('Convert matrix elements to reduced one first')
            return None
        orbs = self.ms.orbits
        n1, l1, ml1, s1d, tz1d = nlmstz1
        n2, l2, ml2, s2d, tz2d = nlmstz2
        m1d = 2*ml1 + s1d
        m2d = 2*ml2 + s2d
        me = 0.0
        for j1d, j2d in itertools.product(range(abs(2*l1-1),2*l1+3,2), range(abs(2*l2-1),2*l2+3,2)):
            coef =  _clebsch_gordan(l1, 0.5, j1d*0.5, ml1, s1d*0.5, m1d*0.5) * \
                    _clebsch_gordan(l2, 0.5, j2d*0.5, ml2, s2d*0.5, m2d*0.5)
            i1 = orbs.get_orbit_index(n1, l1, j1d, tz1d)
            i2 = orbs.get_orbit_index(n2, l2, j2d, tz2d)
            me += self.get_1bme_Mscheme(i1, m1d, i2, m2d) * coef
        return me

    def get_2bme_from_mat_indices(self,chbra,chket,bra,ket):
        if( chbra < chket ):
            if(self.verbose): print("Warning:" + sys._getframe().f_code.co_name )
            return 0
        try:
            return self.two[(chbra,chket)][(bra,ket)]
        except:
            if(self.verbose): print("Nothing here " + sys._getframe().f_code.co_name )
            return 0

    def get_2bme_from_indices( self, a, b, c, d, Jab, Jcd ):
        if(self.ms.rank <= 1): return 0
        two = self.ms.two
        orbits = two.orbits
        oa = orbits.get_orbit(a)
        ob = orbits.get_orbit(b)
        oc = orbits.get_orbit(c)
        od = orbits.get_orbit(d)
        Pab = (-1)**(oa.l+ob.l)
        Pcd = (-1)**(oc.l+od.l)
        Zab = (oa.z + ob.z)//2
        Zcd = (oc.z + od.z)//2
        if( self._triag( Jab, Jcd, self.rankJ )):
            if(self.verbose): print("Operator rank mismatch: return 0")
            return 0.0
        if( Pab * Pcd * self.rankP != 1):
            if(self.verbose): print("Operator parity mismatch: return 0")
            return 0.0
        if( abs(Zab-Zcd) != self.rankZ):
            if(self.verbose): print("Operator pn mismatch: return 0")
            return 0.0
        try:
            ichbra = two.get_index(Jab,Pab,Zab)
            ichket = two.get_index(Jcd,Pcd,Zcd)
        except:
            if(self.verbose): print("Warning: channel bra & ket index, " + sys._getframe().f_code.co_name )
            return 0.0
        phase = 1
        if( ichbra >= ichket ):
            aa, bb, cc, dd, = a, b, c, d
        else:
            ichbra, ichket = ichket, ichbra
            phase *=  (-1)**(Jcd-Jab)
            if(self.skew): phase *= -1
            aa, bb, cc, dd, = c, d, a, b
        chbra = two.get_channel(ichbra)
        chket = two.get_channel(ichket)
        try:
            bra = chbra.index_from_indices[(aa,bb)]
            ket = chket.index_from_indices[(cc,dd)]
        except:
            if(self.verbose): print("Warning: bra & ket index, " + sys._getframe().f_code.co_name )
            return 0.0
        phase *= chbra.phase_from_indices[(aa,bb)] * chket.phase_from_indices[(cc,dd)]
        return self.get_2bme_from_mat_indices(ichbra,ichket,bra,ket)*phase

    def get_2bme_from_orbits( self, oa, ob, oc, od, Jab, Jcd ):
        if(self.ms.rank <= 1): return 0.0
        orbits = self.ms.orbits
        a = orbits.orbit_index_from_orbit( oa )
        b = orbits.orbit_index_from_orbit( ob )
        c = orbits.orbit_index_from_orbit( oc )
        d = orbits.orbit_index_from_orbit( od )
        return self.get_2bme_from_indices( a, b, c, d, Jab, Jcd )

    def get_2bme_monopole(self, a, b, c, d):
        if(self.ms.rank <= 1): return 0.0
        norm = 1.0
        if(a==b): norm *= np.sqrt(2.0)
        if(c==d): norm *= np.sqrt(2.0)
        orbits = self.ms.orbits
        oa = orbits.get_orbit(a)
        ob = orbits.get_orbit(b)
        oc = orbits.get_orbit(c)
        od = orbits.get_orbit(d)
        Jmin = max(abs(oa.j-ob.j)//2, abs(oc.j-od.j)//2)
        Jmax = min(   (oa.j+ob.j)//2,    (oc.j+od.j)//2)
        sumJ = 0.0
        sumV = 0.0
        for J in range(Jmin,Jmax+1):
            if(a==b and J%2==1): continue
            if(c==d and J%2==1): continue
            sumV += self.get_2bme_from_indices(a,b,c,d,J,J) * (2*J+1)
            sumJ += (2*J+1)
        return sumV / sumJ * norm

    def get_3bme_from_mat_indices( self, chbra, chket, bra, ket ):
        if( chbra < chket ):
            if(self.verbose): print("Warning:" + sys._getframe().f_code.co_name )
            return
        return self.three[(chbra,chket)][(bra,ket)]

    def get_3bme_from_indices( self, a, b, c, Jab, Tab, d, e, f, Jde, Tde, Jbra, Tbra, Jket, Tket ):
        three = self.ms.three
        iorbits = three.orbits
        if( not a>=b>=c ): print( "In three body exchange of indices is not supported. " )
        if( not d>=e>=f ): print( "In three body exchange of indices is not supported. " )
        oa = orbits.get_orbit(a)
        ob = orbits.get_orbit(b)
        oc = orbits.get_orbit(c)
        od = orbits.get_orbit(d)
        oe = orbits.get_orbit(e)
        of = orbits.get_orbit(f)
        Pbra = (-1)**(oa.l+ob.l+oc.l)
        Pket = (-1)**(od.l+oe.l+of.l)
        if( self._triag( Jbra, Jket, 2*self.rankJ )):
            if(self.verbose): print("Warning: J, " + sys._getframe().f_code.co_name )
            return
        if( Pbra * Pket * self.rankP != 1):
            if(self.verbose): print("Warning: Parity, " + sys._getframe().f_code.co_name )
            return
        if( self._triag( Tbra, Tket, 2*self.rankZ) ):
            if(self.verbose): print("Warning: Z, " + sys._getframe().f_code.co_name )
            return
        ichbra = three.get_index(Jbra,Pbra,Tbra)
        ichket = three.get_index(Jket,Pket,Tket)
        phase = 1
        if( ichbra >= ichket ):
            i, j, k, l, m, n = a, b, c, d, e, f
            Jij, Tij, Jlm, Tlm = Jab, Tab, Jde, Tde
        else:
            ichbra, ichket = ichket, ichbra
            phase *=  (-1)**((Jket+Tket-Jbra-Tbra)//2)
            i, j, k, l, m, n = d, e, f, a, b, c
            Jij, Tij, Jlm, Tlm = Jde, Tde, Jab, Tab
        chbra = three.get_channel(ichbra)
        chket = three.get_channel(ichket)
        bra = chbra.index_from_indices[(i,j,k,Jij,Tij)]
        ket = chket.index_from_indices[(l,m,n,Jlm,Tlm)]
        return self.set_3bme_from_mat_indices(ichbra,ichket,bra,ket) * phase

    def read_operator_file(self, filename, spfile=None, opfile2=None, comment="!", istore=None, A=None, MuCapType=None):
        """
        istore: for muon capture operator file. This will be removed. DO NOT use. Use MuCapType instead.
        MuCapType: tuple (n, k, w, u, x). n, k, w, u are 'int', while x is str. k=0,1, u is rank of the operator.
            n is the forbiddeness. x takes 'p', '+', '-', ''
        """
        if(filename.find(".snt") != -1):
            self._read_operator_snt(filename, comment, A)
            if( self.count_nonzero_1bme() + self.count_nonzero_2bme() == 0):
                print("The number of non-zero operator matrix elements is 0 better to check: "+ filename + "!!")
            return
        if(filename.find(".me2j") != -1):
            self._read_general_operator(filename, comment)
            if( self.count_nonzero_1bme() + self.count_nonzero_2bme() == 0):
                print("The number of non-zero operator matrix elements is 0 better to check: "+ filename + "!!")
            return
        if(filename.find(".navratil") != -1):
            self._read_general_operator_navratil(filename, comment)
            if( self.count_nonzero_1bme() + self.count_nonzero_2bme() == 0):
                print("The number of non-zero operator matrix elements is 0 better to check: "+ filename + "!!")
            return
        if(filename.find(".readable.txt") != -1):
            self._read_3b_operator_readabletxt(filename, comment)
            if( self.count_nonzero_1bme() + self.count_nonzero_2bme() + self.count_nonzero_3bme() == 0):
                print("The number of non-zero operator matrix elements is 0 better to check: "+ filename + "!!")
            return
        if(filename.find(".int") != -1):
            if(spfile == None):
                print("No sp file!"); return
            if( self.rankJ==0 and self.rankP==1 and self.rankZ==0 ):
                nushell2snt.scalar( spfile, filename, "tmp.snt" )
            else:
                if(opfile2 == None):
                    print("No op2 file!"); return
                nushell2snt.tensor( spfile, filename, opfile2, "tmp.snt" )
            self._read_operator_snt("tmp.snt", "!")
            subprocess.call("rm tmp.snt", shell=True)
            if( self.count_nonzero_1bme() + self.count_nonzero_2bme() == 0):
                print("The number of non-zero operator matrix elements is 0 better to check: "+ filename + "!!")
            return
        if(filename.find(".lotta") != -1):
            if( istore != None ):
                self._read_lotta_format(filename,istore)
                print("This mode is deprecated")
                return
            if( MuCapType == None):
                print("Specify the operator type!")
                return
            istore = self._idx_from_MuCapType(MuCapType)
            if(istore==None): return
            self._read_lotta_format(filename,istore)
            if( self.count_nonzero_1bme() == 0):
                print("The number of non-zero operator matrix elements is 0 better to check: "+ filename + "!!")
            return
        if(filename.find("jiangming") !=-1):
            self._read_jiangming_format(filename)
            if( self.count_nonzero_1bme() + self.count_nonzero_2bme() == 0):
                print("The number of non-zero operator matrix elements is 0 better to check: "+ filename + "!!")
            return
        print("Unknown file format in " + sys._getframe().f_code.co_name )
        return

    def _idx_from_MuCapType(self,op_type):
        n, k, w, u, x = op_type
        if(  k==0 and w==n   and u==n   and x=='s'): return  0
        elif(k==1 and w==n   and u==n   and x=='s'): return  1
        elif(k==1 and w==n   and u==n+1 and x=='s'): return  2
        elif(k==1 and w==n+2 and u==n+1 and x=='s'): return  3
        elif(k==0 and w==n   and u==n   and x=='+'): return  4
        elif(k==0 and w==n   and u==n   and x=='-'): return  5
        elif(k==1 and w==n   and u==n   and x=='+'): return  6
        elif(k==1 and w==n   and u==n   and x=='-'): return  7
        elif(k==1 and w==n   and u==n+1 and x=='-'): return  8
        elif(k==1 and w==n+2 and u==n+1 and x=='+'): return  9
        elif(k==1 and w==n-1 and u==n   and x=='p'): return 10
        elif(k==1 and w==n+1 and u==n   and x=='p'): return 11
        elif(k==1 and w==n+1 and u==n+1 and x=='p'): return 12
        elif(k==0 and w==n+1 and u==n+1 and x=='p'): return 13
        else:
            print("Unknown muon capture operator type! Check the MuCapType!")
            return None

    def _MuCapType_from_idx(self,i,n):
        if(  i== 0): return (n,0,n  ,n  ,'s')
        elif(i== 1): return (n,1,n  ,n  ,'s')
        elif(i== 2): return (n,1,n  ,n+1,'s')
        elif(i== 3): return (n,1,n+2,n+1,'s')
        elif(i== 4): return (n,0,n  ,n  ,'+')
        elif(i== 5): return (n,0,n  ,n  ,'-')
        elif(i== 6): return (n,1,n  ,n  ,'+')
        elif(i== 7): return (n,1,n  ,n  ,'-')
        elif(i== 8): return (n,1,n  ,n+1,'-')
        elif(i== 9): return (n,1,n+2,n+1,'+')
        elif(i==10): return (n,1,n-1,n  ,'p')
        elif(i==11): return (n,1,n+1,n  ,'p')
        elif(i==12): return (n,1,n+1,n+1,'p')
        elif(i==13): return (n,0,n+1,n+1,'p')
        else:
            print("Unknown muon capture operator type! Check the MuCapType!")
            return None

    def _triag(self,J1,J2,J3):
        b = True
        if(abs(J1-J2) <= J3 <= J1+J2): b = False
        return b

    def _read_operator_snt(self, filename, comment="!", A=None):
        f = open(filename, 'r')
        if(os.path.getsize(filename) == 0):
            print("{:s} is empty. The file might be curshed!".format(filename))
            return
        line = f.readline()
        b = True
        zerobody=0
        while b == True:
            line = f.readline()
            if(line.find("zero body") != -1 or \
                    line.find("Zero body") != -1 or \
                    line.find("Zero Body") != -1):
                data = line.split()
                zerobody = float(data[4])
            if(line.find("zero-body") != -1 or \
                    line.find("Zero-body") != -1 or \
                    line.find("Zero-Body") != -1):
                data = line.split()
                zerobody = float(data[3])
            b = line.startswith(comment) or line.startswith("#")
        data = line.split()
        norbs = int(data[0]) + int(data[1])
        self.p_core = int(data[2])
        self.n_core = int(data[3])

        b = True
        while b == True:
            x = f.tell()
            line = f.readline()
            b = line.startswith(comment) or line.startswith("#")
        f.seek(x)

        orbs = Orbits()
        for i in range(norbs):
            line = f.readline()
            data = line.split()
            idx, n, l, j, z = int(data[0]), int(data[1]), int(data[2]), int(data[3]), int(data[4])
            orbs.add_orbit(n,l,j,z)
        ms = ModelSpace()
        ms.set_modelspace_from_orbits( orbs )
        self.allocate_operator( ms )
        self.set_0bme( zerobody )

        b = True
        while b == True:
            line = f.readline()
            b = line.startswith(comment) or line.startswith("#")
        data = line.split()
        n = int(data[0])
        method = int(data[1])
        fact1 = 1.0
        if(method==10 and A==None):
            print(" Need to set mass number! ")
            sys.exit()
        if(A!=None and method==10):
            hw = float(data[2])
            fact1 = (1-1/float(A))*hw


        b = True
        while b == True:
            x = f.tell()
            line = f.readline()
            b = line.startswith(comment) or line.startswith("#")
        f.seek(x)

        for i in range(n):
            line = f.readline()
            data = line.split()
            a, b, me = int(data[0]), int(data[1]), float(data[2])
            self.set_1bme(a,b,me*fact1)

        b = True
        while b == True:
            line = f.readline()
            b = line.startswith(comment) or line.startswith("#")
        data = line.split()
        n = int(data[0])
        method = int(data[1])
        self.kshell_options.append(method)
        if(len(data)>2): self.kshell_options.append(int(float(data[2])))
        if(len(data)>3): self.kshell_options.append(float(data[3]))
        fact2 = 1.0
        if(method==10 and A==None):
            print(" Need to set mass number! ")
            sys.exit()
        if(A!=None and method==10):
            hw = float(data[2])
            fact2 = hw/float(A)

        b = True
        while b == True:
            x = f.tell()
            line = f.readline()
            b = line.startswith(comment) or line.startswith("#")
        f.seek(x)

        for i in range(int(data[0])):
            line = f.readline()
            data = line.split()
            if(self.rankJ==0 and self.rankZ==0 and self.rankP==1):
                a, b, c, d = int(data[0]), int(data[1]), int(data[2]), int(data[3])
                Jab, me = int(data[4]), float(data[5])
                Jcd = Jab
            else:
                a, b, c, d = int(data[0]), int(data[1]), int(data[2]), int(data[3])
                Jab, Jcd, me = int(data[4]), int(data[5]), float(data[6])
            if(A!=None and method==10):
                Tcm = float(data[6])
                self.set_2bme_from_indices(a,b,c,d,Jab,Jcd,me + Tcm*fact2)
            else:
                self.set_2bme_from_indices(a,b,c,d,Jab,Jcd,me)
        f.close()

    def _read_lotta_format_old(self, filename, ime ):
        orbs = Orbits(verbose=False)
        f = open(filename, "r")
        lines = f.readlines()
        f.close()
        idx = 0
        for line in lines[1:]:
            entry = line.split()
            p_n = int(entry[3])
            p_l = int(entry[4])
            p_j = int(entry[5])
            orbs.add_orbit(p_n, p_l, p_j, -1)
        for line in lines[1:]:
            entry = line.split()
            n_n = int(entry[0])
            n_l = int(entry[1])
            n_j = int(entry[2])
            orbs.add_orbit(n_n, n_l, n_j, 1)
        ms = ModelSpace(rank=1)
        ms.set_modelspace_from_orbits( orbs )
        self.allocate_operator( ms )
        self.set_0bme( 0.0 )

        for line in lines[1:]:
            entry = line.split()
            n_n = int(entry[0])
            n_l = int(entry[1])
            n_j = int(entry[2])
            p_n = int(entry[3])
            p_l = int(entry[4])
            p_j = int(entry[5])
            mes = [ float(entry[i+6]) for i in range(len(entry)-6) ]
            i = orbs.get_orbit_index(n_n,n_l,n_j, 1)
            j = orbs.get_orbit_index(p_n,p_l,p_j,-1)
            me = mes[ime]
            if( abs(me) < 1.e-8): continue
            self.set_1bme( i, j, me )

    def _read_lotta_format(self, filename, ime ):
        orbs = Orbits(verbose=False)
        f = open(filename, "r")
        lines = f.readlines()
        f.close()
        idx = 0
        for line in lines[1:]:
            entry = line.split()
            p_n = int(entry[4])
            p_l = int(entry[5])
            p_j = int(entry[6])
            orbs.add_orbit(p_n, p_l, p_j, -1)
        for line in lines[1:]:
            entry = line.split()
            n_n = int(entry[1])
            n_l = int(entry[2])
            n_j = int(entry[3])
            orbs.add_orbit(n_n, n_l, n_j, 1)
        ms = ModelSpace(rank=1)
        ms.set_modelspace_from_orbits( orbs )
        self.allocate_operator( ms )
        self.set_0bme( 0.0 )

        for line in lines[1:]:
            entry = line.split()
            n_n = int(entry[1])
            n_l = int(entry[2])
            n_j = int(entry[3])
            p_n = int(entry[4])
            p_l = int(entry[5])
            p_j = int(entry[6])
            mes = [ float(entry[i+7]) for i in range(len(entry)-7) ]
            i = orbs.get_orbit_index(n_n,n_l,n_j, 1)
            j = orbs.get_orbit_index(p_n,p_l,p_j,-1)
            me = mes[ime]
            if( abs(me) < 1.e-8): continue
            self.set_1bme( i, j, me )

    def _read_jiangming_format(self, filename):
        f = open(filename, "r")
        lines = f.readlines()
        f.close()
        if(self.ms==None): raise ValueError("Define model-space first!")
        orbits = self.ms.orbits
        for line in lines:
            line_data = line.split(",")
            indx = [int(x) for x in line_data[:-1]]
            ME = float(line_data[-1])
            n1, n2, n3, n4, lj1, lj2, lj3, lj4, J = indx
            l1, j1 = _ljidx_to_lj(lj1)
            l2, j2 = _ljidx_to_lj(lj2)
            l3, j3 = _ljidx_to_lj(lj3)
            l4, j4 = _ljidx_to_lj(lj4)
            i = orbits.get_orbit_index(n1, l1, j1, -1)
            j = orbits.get_orbit_index(n2, l2, j2, -1)
            k = orbits.get_orbit_index(n3, l3, j3,  1)
            l = orbits.get_orbit_index(n4, l4, j4,  1)
            self.set_2bme_from_indices(i,j,k,l,J,J,ME)
        return

    def _read_general_operator(self, filename, comment="!"):
        if(filename.find(".gz") != -1): f = gzip.open(filename, "r")
        else: f = open(filename,"r")
        header = f.readline()
        header = f.readline()
        dat = header.split()
        self.rankJ = int(dat[0])
        self.rankP = int(dat[1])
        self.rankZ = int(dat[2])
        emax = int(dat[3])
        e2max = int(dat[4])

        ms = ModelSpace()
        ms.set_modelspace_from_boundaries( emax=emax )
        self.allocate_operator( ms )
        iorbits = OrbitsIsospin( emax=emax )
        orbits = ms.orbits
        data = f.readline()
        self.set_0bme( float(data) )
        for oi in iorbits.orbits:
            pi = orbits.get_orbit_index( oi.n, oi.l, oi.j, -1)
            ni = orbits.get_orbit_index( oi.n, oi.l, oi.j,  1)
            for oj in iorbits.orbits:
                pj = orbits.get_orbit_index( oj.n, oj.l, oj.j, -1)
                nj = orbits.get_orbit_index( oj.n, oj.l, oj.j,  1)
                if( (-1)**(oi.l+oj.l) * self.rankP != 1): continue
                if( self._triag(oi.j, oj.j, 2*self.rankJ ) ): continue
                data = [ float(x) for x in f.readline().split() ]
                if( abs(data[0]) > 1.e-16 ): self.set_1bme( pi,pj,data[0] )
                if( abs(data[1]) > 1.e-16 ): self.set_1bme( ni,nj,data[1] )
                if( abs(data[2]) > 1.e-16 ): self.set_1bme( ni,pj,data[2] )
                if( abs(data[3]) > 1.e-16 ): self.set_1bme( pi,nj,data[3] )

        norbs = iorbits.get_num_orbits()+1
        for i in range(1,norbs):
            oi = iorbits.get_orbit(i)
            pi = orbits.get_orbit_index(oi.n, oi.l, oi.j, -1)
            ni = orbits.get_orbit_index(oi.n, oi.l, oi.j,  1)
            for j in range(1,i+1):
                oj = iorbits.get_orbit(j)
                pj = orbits.get_orbit_index(oj.n, oj.l, oj.j, -1)
                nj = orbits.get_orbit_index(oj.n, oj.l, oj.j,  1)
                if( 2*oi.n+oi.l+2*oj.n+oj.l > e2max ): continue

                for k in range(1,norbs):
                    ok = iorbits.get_orbit(k)
                    pk = orbits.get_orbit_index(ok.n, ok.l, ok.j, -1)
                    nk = orbits.get_orbit_index(ok.n, ok.l, ok.j,  1)
                    for l in range(1,k+1):
                        ol = iorbits.get_orbit(l)
                        pl = orbits.get_orbit_index(ol.n, ol.l, ol.j, -1)
                        nl = orbits.get_orbit_index(ol.n, ol.l, ol.j,  1)
                        if( 2*ok.n+ok.l+ 2*ol.n+ol.l > e2max ): continue
                        if( (-1)**( oi.l+oj.l+ok.l+ol.l ) * self.rankP != 1): continue
                        for Jij in range( int(abs( oi.j-oj.j ))//2, (oi.j+oj.j)//2+1 ):
                            for Jkl in range( int(abs( ok.j-ol.j ))//2, (ok.j+ol.j)//2+1 ):
                                if( self._triag(Jij, Jkl, self.rankJ ) ): continue
                                data = [ float(x) for x in f.readline().split() ]
                                if( abs(data[0]) > 1.e-16 ): self.set_2bme_from_indices(pi,pj,pk,pl,Jij,Jkl,data[0])
                                if( abs(data[1]) > 1.e-16 ): self.set_2bme_from_indices(pi,pj,pk,nl,Jij,Jkl,data[1])
                                if( abs(data[2]) > 1.e-16 ): self.set_2bme_from_indices(pi,pj,nk,pl,Jij,Jkl,data[2])
                                if( abs(data[3]) > 1.e-16 ): self.set_2bme_from_indices(pi,pj,nk,nl,Jij,Jkl,data[3])
                                if( abs(data[4]) > 1.e-16 ): self.set_2bme_from_indices(pi,nj,pk,nl,Jij,Jkl,data[4])
                                if( abs(data[5]) > 1.e-16 ): self.set_2bme_from_indices(pi,nj,nk,pl,Jij,Jkl,data[5])
                                if( abs(data[6]) > 1.e-16 ): self.set_2bme_from_indices(pi,nj,nk,nl,Jij,Jkl,data[6])
                                if( abs(data[7]) > 1.e-16 ): self.set_2bme_from_indices(ni,pj,nk,pl,Jij,Jkl,data[7])
                                if( abs(data[8]) > 1.e-16 ): self.set_2bme_from_indices(ni,pj,nk,nl,Jij,Jkl,data[8])
                                if( abs(data[9]) > 1.e-16 ): self.set_2bme_from_indices(ni,nj,nk,nl,Jij,Jkl,data[9])
        f.close()
    def _read_3b_operator_readabletxt(self, filename, comment="!"):
        if( self.ms == None ):
            ms = ModelSpace(rank=3)
            ms.set_modelspace_from_boundaries( emax=6, e2max=6, e3max=6 )
            self.allocate_operator( ms )
        iorbits = self.ms.iorbits
        f = open(filename,"r")
        header = f.readline()
        header = f.readline()

        line = f.readline()
        while len(line) != 0:
            data = [ int(x) for x in line.split()[:-1] ]
            data.append(float(line.split()[-1]))
            if(self.rankJ==0 and self.rankP==1 and self.rankZ==0):
                i, j, k, Jij, Tij, l, m, n, Jlm, Tlm, Jbra, Tbra, Jket, Tket, ME = data
            else:
                i, j, k, Jij, Tij, l, m, n, Jlm, Tlm, Jbra, Tbra, Jket, Tket, ME = data
            if(abs(ME) > 1.e-6): self.set_3bme_from_indices(i,j,k,Jij,Tij,l,m,n,Jlm,Tlm,Jbra,Tbra,Jket,Tket,ME)
            line = f.readline()
        f.close()
    def _read_general_operator_navratil(self, filename, comment="!"):
        emax=16
        ms = ModelSpace()
        ms.set_modelspace_from_boundaries( emax=emax, e2max=emax )
        self.allocate_operator( ms )
        iorbits = OrbitsIsospin( emax=emax )
        orbits = ms.orbits

        if(filename.find(".gz") != -1): f = gzip.open(filename, "r")
        else: f = open(filename,"r")
        header = f.readline()
        while header[0] == comment:
            header = f.readline()
        self.rankJ = int(header)
        header = f.readline()
        self.rankZ = int(header)
        header = f.readline()
        self.rankP = int(header)
        header = f.readline()
        while header[0] == comment:
            header = f.readline()
        line = header

        while len(line) != 0:
            data = [ int(x) for x in line.split()[:-1] ]
            data.append(float(line.split()[-1]))
            oi = iorbits.get_orbit( data[0] )
            oj = iorbits.get_orbit( data[1] )
            ok = iorbits.get_orbit( data[2] )
            ol = iorbits.get_orbit( data[3] )
            ni = orbits.get_orbit_index(oi.n, oi.l, oi.j, 1)
            nj = orbits.get_orbit_index(oj.n, oj.l, oj.j, 1)
            pk = orbits.get_orbit_index(ok.n, ok.l, ok.j,-1)
            pl = orbits.get_orbit_index(ol.n, ol.l, ol.j,-1)
            if(abs(data[-1]) > 1.e-10): self.set_2bme_from_indices(ni,nj,pk,pl,data[4],data[5],data[-1])
            line = f.readline()
        f.close()

    def write_operator_file(self, filename, **kwargs):
        if(filename.find(".snt") != -1):
            self._write_operator_snt( filename )
        if(filename.find("FCIDUMP") != -1):
            self._write_operator_fcidump(filename, **kwargs)
        if(filename.find(".op.me2j") != -1):
            self._write_general_operator( filename )
        if(filename.find(".me2j") != -1):
            if(self.rankJ==0 and self.rankP==1 and self.rankZ==0):
                print("Not implemented yet")
            else:
                self._write_general_operator( filename )
        if(filename.find(".lotta") != -1):
            self._write_operator_lotta( filename )

    def write_nme_file(self):
        """
        Create an input file of the transit.exe for double-beta decay nuclear matrix element calcualtions
        """
        out = ""
        for key in self.two.keys():
            a,b,c,d,Jab,Jcd = key
            oa = self.orbs.get_orbit(a)
            ob = self.orbs.get_orbit(b)
            oc = self.orbs.get_orbit(c)
            od = self.orbs.get_orbit(d)
            out += "{0:3d} {1:3d} {2:3d} {3:3d} {4:3d} {5:3d}".format(\
                    oa.n, oa.l, oa.j, ob.n, ob.l, ob.j)
            out += "{0:3d} {1:3d} {2:3d} {3:3d} {4:3d} {5:3d}".format(\
                    oc.n, oc.l, oc.j, od.n, od.l, od.j)
            out += "{0:3d} {1:16.8e}".format(Jab, self.two[key] / np.sqrt(2*Jab+1)) + "\n"
        f=open("nme.dat","w")
        f.write(out)
        f.close()

    def _write_general_operator(self, filename):
        f = open(filename, "w")
        f.write(" Written by python script \n")
        f.write(" {:3d} {:3d} {:3d} {:3d} {:3d}\n".format( self.rankJ, self.rankP, self.rankZ, self.ms.emax, self.ms.e2max ))
        f.write("{:16.8e}\n".format( self.zero ) )

        orbits = self.ms.orbits
        iorbits = OrbitsIsospin( emax=self.ms.emax )
        norbs = iorbits.get_num_orbits() + 1
        for i in range(1,norbs):
            oi = iorbits.get_orbit(i)
            pi = orbits.get_orbit_index(oi.n, oi.l, oi.j, -1)
            ni = orbits.get_orbit_index(oi.n, oi.l, oi.j,  1)
            for j in range(1,norbs):
                oj = iorbits.get_orbit(j)
                pj = orbits.get_orbit_index(oj.n, oj.l, oj.j, -1)
                nj = orbits.get_orbit_index(oj.n, oj.l, oj.j,  1)
                if( (-1)**( oi.l+oj.l ) * self.rankP != 1): continue
                if( self._triag( oi.j, oj.j, 2*self.rankJ ) ): continue
                me_pp = 0.0; me_nn = 0.0; me_np = 0.0; me_pn = 0.0
                if(pi>=0 and pj>=0 ):
                    me_pp = self.get_1bme(pi,pj)
                    me_nn = self.get_1bme(ni,nj)
                    me_np = self.get_1bme(ni,pj)
                    me_pn = self.get_1bme(pi,nj)
                f.write("{:16.8e} {:16.8e} {:16.8e} {:16.8e}\n".format(\
                        me_pp, me_nn, me_np, me_pn ) )

        for i in range(1,norbs):
            oi = iorbits.get_orbit(i)
            pi = orbits.get_orbit_index(oi.n, oi.l, oi.j, -1)
            ni = orbits.get_orbit_index(oi.n, oi.l, oi.j,  1)
            for j in range(1,i+1):
                oj = iorbits.get_orbit(j)
                pj = orbits.get_orbit_index(oj.n, oj.l, oj.j, -1)
                nj = orbits.get_orbit_index(oj.n, oj.l, oj.j,  1)
                if( oi.e + oj.e > self.ms.e2max ): continue

                for k in range(1,norbs):
                    ok = iorbits.get_orbit(k)
                    pk = orbits.get_orbit_index(ok.n, ok.l, ok.j, -1)
                    nk = orbits.get_orbit_index(ok.n, ok.l, ok.j,  1)
                    for l in range(1,k+1):
                        ol = iorbits.get_orbit(k)
                        pl = orbits.get_orbit_index(ol.n, ol.l, ol.j, -1)
                        nl= orbits.get_orbit_index(ol.n, ol.l, ol.j,  1)
                        if( ok.e + ol.e > self.ms.e2max ): continue
                        if( (-1)**( oi.l+oj.l+ok.l+ol.l ) * self.rankP != 1): continue
                        for Jij in range( int(abs( oi.j-oj.j ))//2, ( oi.j+oj.j )//2+1 ):
                            for Jkl in range( int(abs( ok.j-ol.j ))//2, (ok.j+ol.j)//2+1 ):
                                if( self._triag(Jij, Jkl, self.rankJ ) ): continue

                                me_pppp = self.get_2bme_from_indices(pi,pj,pk,pl,Jij,Jkl)
                                me_pppn = self.get_2bme_from_indices(pi,pj,pk,nl,Jij,Jkl)
                                me_ppnp = self.get_2bme_from_indices(pi,pj,nk,pl,Jij,Jkl)
                                me_ppnn = self.get_2bme_from_indices(pi,pj,nk,nl,Jij,Jkl)
                                me_pnpn = self.get_2bme_from_indices(pi,nj,pk,nl,Jij,Jkl)
                                me_pnnp = self.get_2bme_from_indices(pi,nj,nk,pl,Jij,Jkl)
                                me_pnnn = self.get_2bme_from_indices(pi,nj,nk,nl,Jij,Jkl)
                                me_npnp = self.get_2bme_from_indices(ni,pj,nk,pl,Jij,Jkl)
                                me_npnn = self.get_2bme_from_indices(ni,pj,nk,nl,Jij,Jkl)
                                me_nnnn = self.get_2bme_from_indices(ni,nj,nk,nl,Jij,Jkl)
                                f.write("{:16.8e} {:16.8e} {:16.8e} {:16.8e} {:16.8e} {:16.8e} {:16.8e} {:16.8e} {:16.8e} {:16.8e} \n".format(\
                                        me_pppp, me_pppn, me_ppnp, me_ppnn, me_pnpn, me_pnnp, me_pnnn, me_npnp, me_npnn, me_nnnn))
        f.close()

    def _write_operator_snt(self, filename):
        if(self.p_core == None or self.n_core == None): raise ValueError("set p_core and n_core before calling")
        orbits = self.ms.orbits
        p_norbs = 0; n_norbs = 0
        for o in orbits.orbits:
            if( o.z ==-1 ): p_norbs += 1
            if( o.z == 1 ): n_norbs += 1
        prt = ""
        prt += "! J:{:3d} P:{:3d} Tz:{:3d}\n".format( self.rankJ, self.rankP, self.rankZ )
        prt += "! Zero body term: {:16.8e}\n".format(self.zero)
        prt += "! model space \n"
        prt += " {0:3d} {1:3d} {2:3d} {3:3d} \n".format( p_norbs, n_norbs, self.p_core, self.n_core )
        norbs = orbits.get_num_orbits()+1
        for i in range(1,norbs):
            o = orbits.get_orbit(i)
            prt += "{0:5d} {1:3d} {2:3d} {3:3d} {4:3d} \n".format( i, o.n, o.l, o.j, o.z )

        norbs = orbits.get_num_orbits()+1
        prt += "! one-body part\n"
        prt += "{0:5d} {1:3d}\n".format(self.count_nonzero_1bme(), 0)

        for oa in orbits.orbits:
            for ob in orbits.orbits:
                a = orbits.get_orbit_index_from_orbit(oa)
                b = orbits.get_orbit_index_from_orbit(ob)
                if(b>a): continue
                if(abs(self.get_1bme(a,b)) < 1.e-10): continue
                prt += "{0:3d} {1:3d} {2:16.8e}\n".format(a, b, me)
        if( self.ms.rank==1 ):
            prt += "! two-body part\n"
            prt += "{0:10d} {1:3d}\n".format( 0, 0 )
            f = open(filename, "w")
            f.write(prt)
            f.close()
            return
        prt += "! two-body part\n"
        prt += "{:10d} ".format(self.count_nonzero_2bme())
        if(len(self.kshell_options)==0): prt += "{:3d} ".format(0)
        if(len(self.kshell_options)>0): prt += "{:3d} ".format(self.kshell_options[0])
        if(len(self.kshell_options)>1): prt += "{:3d} ".format(self.kshell_options[1])
        if(len(self.kshell_options)>2): prt += "{:10.6f} ".format(self.kshell_options[2])
        prt += "\n"
        scalar = False
        if(self.rankJ == 0 and self.rankZ == 0): scalar = True
        for channels in self.two.keys():
            for idxs in self.two[channels].keys():
                if(channels[0]==channels[1] and idxs[1]>idxs[0]): continue
                if(abs(self.two[channels][idxs]) < 1.e-10): continue
                chbra = self.ms.two.get_channel(channels[0])
                chket = self.ms.two.get_channel(channels[1])
                a, b = chbra.get_indices(idxs[0])
                c, d = chket.get_indices(idxs[1])
                if(scalar):
                    prt += "{0:3d} {1:3d} {2:3d} {3:3d} {4:3d} {5:16.8e}\n".format( a, b, c, d, chket.J, self.two[channels][idxs])
                else:
                    prt += "{0:3d} {1:3d} {2:3d} {3:3d} {4:3d} {5:3d} {6:16.8e}\n".format( a, b, c, d, chbra.J, chket.J, self.two[channels][idxs])
        f = open(filename, "w")
        f.write(prt)
        f.close()

    def _write_operator_fcidump(self, filename, **kwargs):
        header = 'symmetry="U1",\n'
        header+= 'v_sym=4,\n'
        norbs_m = 0
        orbits = self.ms.orbits
        for oi in orbits.orbits:
            norbs_m += oi.j + 1
        header+= f'norb={norbs_m},\n'
        header+= f'E_REF= ,\n'
        header+= f'n_proton_tot= {kwargs["n_proton"]},\n'
        header+= f'n_neutron_tot= {kwargs["n_neutron"]},\n'
        header+= f'n_cutoff= 14,\n'
        orbm_2_idx = {}
        m_orbits = []
        header+= 'twojz_i='
        i = 0
        for oi in orbits.orbits:
            for jz in range(-oi.j, oi.j+2, 2):
                header+= f' {jz},'
                m_orbits.append((oi, jz))
                i += 1
                orbm_2_idx[(oi,jz)] = i

        header+= '\n'
        header+= 'par_i='
        for oi in orbits.orbits:
            for jz in range(-oi.j, oi.j+2, 2):
                header+= f' {(-1)**oi.l},'
        header+= '\n'
        header+= 'l_i='
        for oi in orbits.orbits:
            for jz in range(-oi.j, oi.j+2, 2):
                header+= f' {oi.l},'
        header+= '\n'
        header+= 'n_proton_i='
        for oi in orbits.orbits:
            for jz in range(-oi.j, oi.j+2, 2):
                if(oi.z==-1): header+= f' 1,'
                if(oi.z== 1): header+= f' 0,'
        header+= '\n'
        header+= 'n_neutron_i='
        for oi in orbits.orbits:
            for jz in range(-oi.j, oi.j+2, 2):
                if(oi.z==-1): header+= f' 0,'
                if(oi.z== 1): header+= f' 1,'
        header+= '\n'
        self.to_reduced()
        v1b = {}
        v2b = {}
        for i, bra in enumerate(m_orbits):
            for j, ket in enumerate(m_orbits):
                if(i<j): continue
                o_bra, m_bra = bra
                o_ket, m_ket = ket
                ii = orbm_2_idx[(o_bra,m_bra)]
                jj = orbm_2_idx[(o_ket,m_ket)]
                if((-1)**(o_bra.l+o_ket.l) * self.rankP != 1): continue
                if(abs(o_bra.z - o_ket.z)//2 != self.rankZ): continue
                if(not abs(o_bra.j - o_ket.j)//2 <= self.rankJ <= (o_bra.j + o_ket.j)//2): continue
                if(abs(m_bra-m_ket)//2 > self.rankJ): continue
                me = (-1)**((o_bra.j - m_bra)//2) * \
                        float(wigner_3j(o_bra.j*0.5, self.rankJ, o_ket.j*0.5, -m_bra*0.5, (m_bra-m_ket)*0.5, m_ket*0.5)) * \
                        self.get_1bme(orbits.get_orbit_index_from_orbit(o_bra), orbits.get_orbit_index_from_orbit(o_ket))
                if(abs(me) < 1.e-16): continue
                v1b[(ii,jj)] = me

        for i, a in enumerate(m_orbits):
            for j, b in enumerate(m_orbits):
                if(i<j): continue
                for k, c in enumerate(m_orbits):
                    for l, d in enumerate(m_orbits):
                        if(k>i): continue
                        if(i==k and l>j): continue
                        if(i!=k and l>k): continue
                        if(k<l): continue
                        oa, ma = a
                        ob, mb = b
                        oc, mc = c
                        od, md = d

                        ii = orbm_2_idx[(oa,ma)]
                        jj = orbm_2_idx[(ob,mb)]
                        kk = orbm_2_idx[(oc,mc)]
                        ll = orbm_2_idx[(od,md)]
                        if((-1)**(oa.l+ob.l+oc.l+od.l) * self.rankP != 1): continue
                        if(abs(oa.z+ob.z - oc.z-od.z)//2 != self.rankZ): continue
                        if(abs(ma+mb-mc-md)//2 > self.rankJ): continue
                        aa = orbits.get_orbit_index_from_orbit(oa)
                        bb = orbits.get_orbit_index_from_orbit(ob)
                        cc = orbits.get_orbit_index_from_orbit(oc)
                        dd = orbits.get_orbit_index_from_orbit(od)
                        me = self.get_2bme_Mscheme(aa,ma,bb,mb,cc,mc,dd,md,ma+mb-mc-md)
                        if(abs(me) < 1.e-6): continue
                        v2b[(ii,jj,kk,ll)] = me
        header+= f'size_one_body={len(v1b)},\n'
        header+= f'size_two_body={len(v2b)},\n'
        header+= f'!END\n'
        prtme = 'v_ijkl = \n'
        for key, val in v2b.items():
            prtme += f'{val:14.6e} {key[0]:3d} {key[1]:3d} {key[2]:3d} {key[3]:3d} {0:3d}\n'
        for key, val in v1b.items():
            prtme += f'{val:14.6e} {key[0]:3d} {key[1]:3d} {0:3d} {0:3d} {0:3d}\n'
        prtme += f'{self.get_0bme():14.6e} {0:3d} {0:3d} {0:3d} {0:3d} {0:3d}'
        f = open(filename, 'w')
        f.write(header)
        f.write(prtme)
        f.close()

    def _write_operator_lotta(self, filename):
        orbits = self.ms.orbits
        norbs = orbits.get_num_orbits()
        prt = "{:>4s} {:>4s} {:>4s} {:>4s} {:>4s} {:>4s} {:>18s}\n".format( "NN","LN","JN","NP","LP","JP","ME" )
        for i in range(1,norbs+1):
            for j in range(1,norbs+1):
                oi = orbits.get_orbit(i)
                oj = orbits.get_orbit(j)
                if(oi.z != 1): continue
                if(oj.z !=-1): continue
                me = self.get_1bme(i,j)
                if( oi.j==29 and oj.j==29): print(me)
                if( abs(me) < 1.e-10): continue
                prt += "{:4d} {:4d} {:4d} {:4d} {:4d} {:4d} {:18.8e}\n".format( oi.n, oi.l, oi.j, oj.n, oj.l, oj.j, me )
        f = open(filename, "w")
        f.write(prt)
        f.close()
        return

    def print_operator(self):
        orbits = self.ms.orbits
        print("Print Operator")
        print("zero-body term "+str(self.zero))
        print("one-body term:")
        print("  a   b       MtxElm")
        for a in range(1,orbits.get_num_orbits()+1):
            for b in range(1,orbits.get_num_orbits()+1):
                me = self.get_1bme(a,b)
                if(abs(me) < 1e-8): continue
                print("{0:3d} {1:3d} {2:12.6f}".format(a,b,me))
        if(self.ms.rank==1): return
        print("two-body term:")
        print("  a   b   c   d Jab Jcd       MtxElm")
        for a in range(1, orbits.get_num_orbits()+1):
            oa = orbits.get_orbit(a)
            for b in range(a, orbits.get_num_orbits()+1):
                ob = orbits.get_orbit(b)
                if( oa.e + ob.e > self.ms.e2max ): continue

                for c in range(1, orbits.get_num_orbits()+1):
                    oc = orbits.get_orbit(c)
                    for d in range(a, orbits.get_num_orbits()+1):
                        od = orbits.get_orbit(d)
                        if( oc.e + od.e > self.ms.e2max ): continue
                        if((-1)**(oa.l+ob.l+oc.l+od.l) * self.rankP != 1): continue
                        for Jab in range( int(abs(oa.j-ob.j)/2), int((oa.j+ob.j)/2)+1):
                            if(a == b and Jab%2 == 1): continue
                            for Jcd in range( int(abs(oc.j-od.j)/2), int((oc.j+od.j)/2+1)):
                                if(c == d and Jcd%2 == 1): continue

                                if(self._triag(Jab,Jcd,self.rankJ)): continue
                                if( abs(oa.z+ob.z-oc.z-od.z) != 2*self.rankZ ): continue
                                me = self.get_2bme_from_indices(a,b,c,d,Jab,Jcd)
                                if(abs(me) < 1e-8): continue
                                print("{0:3d} {1:3d} {2:3d} {3:3d} {4:3d} {5:3d} {6:12.6f}".format(a,b,c,d,Jab,Jcd,me))
        if(self.ms.rank==2): return
        three = self.ms.three
        print("three-body term:")
        print("   a   b   c Jab Tab d  e  f Jde Tde Jbra Tbra Jket Tket      MtxElm")
        for ichbra in range(three.get_number_channels()):
            chbra = three.get_channel(ichbra)
            for ichket in range(ichbra+1):
                chket = three.get_channel(ichket)
                if( self._triag( chbra.J, chket.J, self.rankJ )): continue
                if( chbra.P * chket.P * self.rankP != 1): continue
                if( self._triag( chbra.T, chket.T, self.rankZ )): continue
                for key in self.three[(ichbra,ichket)].keys():
                    bra, ket = key
                    a = chbra.orbit1_index[bra]
                    b = chbra.orbit2_index[bra]
                    c = chbra.orbit3_index[bra]
                    Jab = chbra.J12_index[bra]
                    Tab = chbra.T12_index[bra]
                    d = chket.orbit1_index[ket]
                    e = chket.orbit2_index[ket]
                    f = chket.orbit3_index[ket]
                    Jde = chket.J12_index[ket]
                    Tde = chket.T12_index[ket]
                    ME = self.three[(ichbra,ichket)][(bra,ket)]
                    if(abs(ME) < 1e-8): continue
                    line = "{:3d} {:3d} {:3d} {:3d} {:3d}".format(a,b,c,Jab,Tab)
                    line += "{:3d} {:3d} {:3d} {:3d} {:3d}".format(d,e,f,Jde,Tde)
                    line += "{:3d} {:3d} {:3d} {:3d}".format(chbra.J, chbra.T, chket.J, chket.T)
                    line += "{:12.6f}".format(ME)
                    print(line)
    def embed_one_to_two(self,A=2):
        two = self.ms.two
        orbits = self.ms.orbits
        scalar = False
        if(self.rankJ == 0 and self.rankZ == 0 and self.rankP==1): scalar = True
        for ichbra in range(two.get_number_channels()):
            chbra = two.get_channel(ichbra)
            for ichket in range(ichbra+1):
                chket = two.get_channel(ichket)
                if( self._triag( chbra.J, chket.J, self.rankJ )): continue
                if( chbra.P * chket.P * self.rankP != 1): continue
                if( abs(chbra.Z-chket.Z) != self.rankZ): continue
                for bra in range(chbra.get_number_states()):
                    a = chbra.orbit1_index[bra]
                    b = chbra.orbit2_index[bra]
                    ketmax = chket.get_number_states()
                    if( ichbra==ichket ): ketmax=bra+1
                    for ket in range(ketmax):
                        c = chket.orbit1_index[ket]
                        d = chket.orbit2_index[ket]

                        me = self._get_embed_1bme_2(a,b,c,d,ichbra,ichket,scalar) / float(A-1)
                        me_original = self.get_2bme_from_indices(a,b,c,d,chbra.J,chket.J)
                        me += me_original
                        if(abs(me) > 1.e-8): self.set_2bme_from_indices(a,b,c,d,chbra.J,chket.J,me)
        self.one = np.zeros( (orbits.get_num_orbits(), orbits.get_num_orbits() ))

    def truncate(self, ms_new):
        op = Operator(ms=ms_new, rankJ=self.rankJ, rankP=self.rankP, rankZ=self.rankZ, reduced=self.reduced, p_core=self.p_core, n_core=self.n_core, skew=self.skew)
        op.kshell_options = self.kshell_options
        orb = self.ms.orbits
        orb_new = ms_new.orbits
        op.set_0bme(self.get_0bme())
        for i, j in itertools.product(list(range(1,orb_new.get_num_orbits()+1)), repeat=2):
            oi = orb_new.get_orbit(i)
            oj = orb_new.get_orbit(j)
            try:
                ii = orb.get_orbit_index_from_orbit(oi)
                jj = orb.get_orbit_index_from_orbit(oj)
                me = self.get_1bme(ii,jj)
                if(abs(me)>1.e-16): op.set_1bme(i,j,me)
            except:
                continue
        for channel in self.two.keys():
            tbc_bra_old = self.ms.two.get_channel(channel[0])
            tbc_ket_old = self.ms.two.get_channel(channel[1])
            try:
                tbc_bra_new = op.ms.two.get_channel_from_JPZ(tbc_bra_old.J, tbc_bra_old.P, tbc_bra_old.Z)
                tbc_ket_new = op.ms.two.get_channel_from_JPZ(tbc_ket_old.J, tbc_ket_old.P, tbc_ket_old.Z)
            except:
                continue
            Jij = tbc_bra_new.J
            Jkl = tbc_ket_new.J
            for idx in self.two[channel].keys():
                me = self.two[channel][idx]
                ii, jj = tbc_bra_old.get_indices(idx[0])
                kk, ll = tbc_ket_old.get_indices(idx[1])
                oi = orb.get_orbit(ii)
                oj = orb.get_orbit(jj)
                ok = orb.get_orbit(kk)
                ol = orb.get_orbit(ll)
                try:
                    i = orb_new.get_orbit_index_from_orbit(oi)
                    j = orb_new.get_orbit_index_from_orbit(oj)
                    k = orb_new.get_orbit_index_from_orbit(ok)
                    l = orb_new.get_orbit_index_from_orbit(ol)
                except:
                    continue
                op.set_2bme_from_indices(i,j,k,l,Jij,Jkl,me)
        if(len(self.three) != 0): raise ValueError("Not ready to use!")
        return op

    def to_reduced(self):
        if(self.rankJ!=0 or self.rankP!=1 or self.rankZ!=0 or self.reduced): return
        orbits = self.ms.orbits
        for i,j in itertools.product(list(range(orbits.get_num_orbits())), repeat=2):
            oi = orbits.get_orbit(i+1)
            self.one[i,j] *= np.sqrt(oi.j+1)
        for channel in self.two.keys():
            chbra = self.ms.two.get_channel(channel[0])
            chket = self.ms.two.get_channel(channel[1])
            Jbra = chbra.J; Jket = chket.J
            for key in self.two[channel].keys():
                self.two[channel][key] *= np.sqrt(2*Jket+1)
        self.reduced=True
        return

    def to_nonreduced(self):
        if(self.rankJ!=0 or self.rankP!=1 or self.rankZ!=0 or not self.reduced): return
        orbits = self.ms.orbits
        for i,j in itertools.product(list(range(orbits.get_num_orbits())), repeat=2):
            oi = orbits.get_orbit(i+1)
            self.one[i,j] /= np.sqrt(oi.j+1)
        for channel in self.two.keys():
            chbra = self.ms.two.get_channel(channel[0])
            chket = self.ms.two.get_channel(channel[1])
            Jbra = chbra.J; Jket = chket.J
            for key in self.two[channel].keys():
                self.two[channel][key] /= np.sqrt(2*Jket+1)
        self.reduced=False
        return

    def _get_embed_1bme_2(self,a,b,c,d,ichbra,ichket,scalar):
        two = self.ms.two
        chbra = two.get_channel(ichbra)
        chket = two.get_channel(ichket)
        Jab = chbra.J
        Jcd = chket.J
        lam = self.rankJ
        orbits = self.ms.orbits
        oa = orbits.get_orbit(a)
        ob = orbits.get_orbit(b)
        oc = orbits.get_orbit(c)
        od = orbits.get_orbit(d)
        me = 0.0
        if( scalar ):
            if(b==d): me += self.get_1bme(a,c)
            if(a==c): me += self.get_1bme(b,d)
            if(a==d): me -= self.get_1bme(b,c) * (-1.0)**( (oa.j+ob.j)//2 - Jab )
            if(b==c): me -= self.get_1bme(a,d) * (-1.0)**( (oa.j+ob.j)//2 - Jab )
            if(a==b): me /= np.sqrt(2.0)
            if(c==d): me /= np.sqrt(2.0)
            return me
        if(b==d): me += self.get_1bme(a,c) * (-1.0)**( (oa.j+ob.j)//2 + Jcd     ) * _sixj(Jab,Jcd,lam,oc.j*0.5,oa.j*0.5,ob.j*0.5)
        if(a==c): me += self.get_1bme(b,d) * (-1.0)**( (oc.j+od.j)//2 - Jab     ) * _sixj(Jab,Jcd,lam,od.j*0.5,ob.j*0.5,oa.j*0.5)
        if(b==c): me -= self.get_1bme(a,d) * (-1.0)**( (oa.j+ob.j+oc.j+od.j)//2 ) * _sixj(Jab,Jcd,lam,od.j*0.5,oa.j*0.5,ob.j*0.5)
        if(a==d): me -= self.get_1bme(b,c) * (-1.0)**( Jcd - Jab                ) * _sixj(Jab,Jcd,lam,oc.j*0.5,ob.j*0.5,oa.j*0.5)
        me *= np.sqrt( (2*Jab+1)*(2*Jcd+1) ) * (-1.0)**lam
        if(a==b): me /= np.sqrt(2.0)
        if(c==d): me /= np.sqrt(2.0)
        return me
    def spin_tensor_decomposition(self, chan_J=None):
        if(self.rankJ != 0):
            print("Spin-tensor decomposition is not defined for a non-scalar operator")
            return None
        ops = []
        ops.append( Operator( rankJ=self.rankJ, rankP=self.rankP, rankZ=self.rankZ, ms=self.ms ) )
        ops.append( Operator( rankJ=self.rankJ, rankP=self.rankP, rankZ=self.rankZ, ms=self.ms ) )
        ops.append( Operator( rankJ=self.rankJ, rankP=self.rankP, rankZ=self.rankZ, ms=self.ms ) )
        ms = self.ms.two
        orbits = ms.orbits

        if(len(self.ls_couple_store)==0):
            for oa, ob in itertools.product(orbits.orbits, repeat=2):
                for Lab, Sab in itertools.product(range( abs(oa.l-ob.l), oa.l+ob.l+1 ),[0,1]):
                    for J in range(max(abs(Lab-Sab),abs(oa.j-ob.j)//2), min(Lab+Sab,(oa.j+ob.j)//2)+1):
                        self.ls_couple_store[(oa,ob,Lab,Sab,J)] = _ls_coupling(oa.l, oa.j*0.5, ob.l, ob.j*0.5, Lab, Sab, J)
        if(len(self.sixj_store)==0):
            lmax=-9999
            for oa in orbits.orbits:
                lmax = max(lmax, oa.l)
            for Lab, Lcd, Sab, Scd in itertools.product(range(2*lmax+1),range(2*lmax+1),[0,1],[0,1]):
                for J, JJ in itertools.product(range(max(abs(Lab-Sab), abs(Lcd-Scd)), min(Lab+Sab,Lcd+Scd)+1), [0,1,2]):
                    self.sixj_store[(Lab,Sab,J,Scd,Lcd,JJ)] = np.float(wigner_6j(Lab,Sab,J,Scd,Lcd,JJ))

        for ch_key in self.two.keys():
            ichbra = ch_key[0]
            ichket = ch_key[1]
            chbra = ms.get_channel(ichbra)
            chket = ms.get_channel(ichket)
            J = chket.J
            if(chan_J!=None and chan_J!=J): continue
            for key in self.two[ch_key].keys():
                a, b = chbra.get_indices(key[0])
                c, d = chket.get_indices(key[1])
                oa = orbits.get_orbit(a)
                ob = orbits.get_orbit(b)
                oc = orbits.get_orbit(c)
                od = orbits.get_orbit(d)
                norm1 = 1.0
                if(a==b): norm1 *= np.sqrt(2)
                if(c==d): norm1 *= np.sqrt(2)

                for rank in [0,1,2]:
                    sum3 = 0.0
                    for Lab, Lcd, Sab, Scd in itertools.product(range( abs(oa.l-ob.l), oa.l+ob.l+1 ),range( abs(oc.l-od.l), oc.l+od.l+1 ),[0,1],[0,1]):
                        if(self._triag( Lab, Sab, J )): continue
                        if(self._triag( Lcd, Scd, J )): continue
                        if(self._triag( Lab, Lcd, rank )): continue
                        if(self._triag( Sab, Scd, rank )): continue
                        Cab = self.ls_couple_store[(oa,ob,Lab,Sab,J)]
                        Ccd = self.ls_couple_store[(oc,od,Lcd,Scd,J)]
                        SixJ = self.sixj_store[(Lab,Sab,J,Scd,Lcd,rank)]
                        if(abs(Cab*Ccd*SixJ) < 1.e-16): continue
                        sum2 = 0.0
                        for JJ in range( max(abs(Lab-Sab),abs(Lcd-Scd)), min(Lab+Sab, Lcd+Scd)+1):
                            SixJJ = self.sixj_store[(Lab,Sab,JJ,Scd,Lcd,rank)]
                            if(abs(SixJJ) < 1.e-16): continue
                            sum1 = 0.0
                            for jaa, jbb, jcc, jdd in itertools.product(range(abs(2*oa.l-1),2*oa.l+3,2), range(abs(2*ob.l-1),2*ob.l+3,2), range(abs(2*oc.l-1),2*oc.l+3,2), range(abs(2*od.l-1),2*od.l+3,2)):
                                if(self._triag(jaa, jbb, 2*JJ)): continue
                                if(self._triag(jcc, jdd, 2*JJ)): continue
                                try:
                                    aa = orbits.get_orbit_index(oa.n, oa.l, jaa, oa.z)
                                    bb = orbits.get_orbit_index(ob.n, ob.l, jbb, ob.z)
                                    cc = orbits.get_orbit_index(oc.n, oc.l, jcc, oc.z)
                                    dd = orbits.get_orbit_index(od.n, od.l, jdd, od.z)
                                    norm2 = 1.0
                                    if(aa==bb): norm2 *= np.sqrt(2)
                                    if(cc==dd): norm2 *= np.sqrt(2)
                                    oaa = orbits.get_orbit(aa)
                                    obb = orbits.get_orbit(bb)
                                    occ = orbits.get_orbit(cc)
                                    odd = orbits.get_orbit(dd)
                                except:
                                    continue
                                CCab = self.ls_couple_store[(oaa,obb,Lab,Sab,JJ)]
                                CCcd = self.ls_couple_store[(occ,odd,Lcd,Scd,JJ)]
                                sum1 += self.get_2bme_from_indices(aa,bb,cc,dd,JJ,JJ) * CCab * CCcd * norm2
                            sum2 += sum1 * SixJJ * (2*JJ+1)*(-1)**(JJ+J)
                        sum3 += sum2 * SixJ * Cab * Ccd
                    ops[rank].set_2bme_from_indices(a,b,c,d,J,J, (2*rank+1)*sum3/norm1)
        return ops

    def to_DataFrame(self, rank=None):
        if(rank==0 or rank==None): zero = pd.DataFrame([{"0 body":self.zero},])
        if(rank==1 or rank==None):
            orbits = self.ms.orbits
            tmp = []
            for a in range(1,orbits.get_num_orbits()+1):
                for b in range(1, a+1):
                    tmp.append({"a":a,"b":b,"1 body":self.get_1bme(a,b)})
            if(len(tmp)==0):
                one = pd.DataFrame()
            else:
                one = pd.DataFrame(tmp)
                one = one.iloc[list(~one["1 body"].eq(0)),:].reset_index(drop=True)
        if(rank==2 or rank==None):
            tmp = []
            for channels in self.two.keys():
                chbra = self.ms.two.get_channel(channels[0])
                chket = self.ms.two.get_channel(channels[1])
                Jab = chbra.J
                Jcd = chket.J
                for idx in self.two[channels].keys():
                    a, b = chbra.get_indices(idx[0])
                    c, d = chket.get_indices(idx[1])
                    tmp.append({"a":a, "b":b, "c":c, "d":d, "Jab":Jab, "Jcd":Jcd, "2 body":self.two[channels][idx]})
            if(len(tmp)==0):
                two = pd.DataFrame()
            else:
                two = pd.DataFrame(tmp)
                two = two.iloc[list(~two["2 body"].eq(0)),:].reset_index(drop=True)
        if(rank==0): return zero
        if(rank==1): return one
        if(rank==2): return two
        if(rank==None): return zero, one, two

    def get_2bme_Mscheme(self, p, mdp, q, mdq, r, mdr, s, mds, mud):
        if(not self.reduced):
            print('Convert matrix elements to reduced one first')
            return None
        orbs = self.ms.orbits
        o_p, o_q, o_r, o_s = orbs.get_orbit(p), orbs.get_orbit(q), orbs.get_orbit(r), orbs.get_orbit(s)
        norm = 1
        if(p==q): norm *= np.sqrt(2.0)
        if(r==s): norm *= np.sqrt(2.0)
        me = 0.0
        for Jpq in range(abs(o_p.j-o_q.j)//2, (o_p.j+o_q.j)//2+1):
            if(p==q and Jpq%2==1): continue
            Mpq = (mdp + mdq)//2
            if(abs(Mpq) > Jpq): continue
            for Jrs in range(abs(o_r.j-o_s.j)//2, (o_r.j+o_s.j)//2+1):
                if(r==s and Jrs%2==1): continue
                Mrs = (mdr + mds)//2
                if(abs(Mrs) > Jrs): continue
                if(not abs(Jpq-Jrs) <= self.rankJ <= Jpq+Jrs): continue
                me += _clebsch_gordan(o_p.j*0.5, o_q.j*0.5, Jpq, mdp*0.5, mdq*0.5, Mpq) * \
                        _clebsch_gordan(o_r.j*0.5, o_s.j*0.5, Jrs, mdr*0.5, mds*0.5, Mrs) * \
                        _clebsch_gordan(Jrs, self.rankJ, Jpq, Mrs, mud*0.5, Mpq) / np.sqrt(2*Jpq+1) * \
                        self.get_2bme_from_indices(p, q, r, s, Jpq, Jrs)
        me *= norm
        return me

    def get_2bme_Mscheme_nlms(self, nlmstz1, nlmstz2, nlmstz3, nlmstz4, mud):
        if(not self.reduced):
            print('Convert matrix elements to reduced one first')
            return None
        orbs = self.ms.orbits
        n1, l1, ml1, s1d, tz1d = nlmstz1
        n2, l2, ml2, s2d, tz2d = nlmstz2
        n3, l3, ml3, s3d, tz3d = nlmstz3
        n4, l4, ml4, s4d, tz4d = nlmstz4
        m1d = 2*ml1 + s1d
        m2d = 2*ml2 + s2d
        m3d = 2*ml3 + s3d
        m4d = 2*ml4 + s4d
        me = 0.0
        for j1d, j2d, j3d, j4d in itertools.product(range(abs(2*l1-1),2*l1+3,2), range(abs(2*l2-1),2*l2+3,2), range(abs(2*l3-1),2*l3+3,2), range(abs(2*l4-1),2*l4+3,2)):
            coef =  _clebsch_gordan(l1, 0.5, j1d*0.5, ml1, s1d*0.5, m1d*0.5) * \
                    _clebsch_gordan(l2, 0.5, j2d*0.5, ml2, s2d*0.5, m2d*0.5) * \
                    _clebsch_gordan(l3, 0.5, j3d*0.5, ml3, s3d*0.5, m3d*0.5) * \
                    _clebsch_gordan(l4, 0.5, j4d*0.5, ml4, s4d*0.5, m4d*0.5)
            i1 = orbs.get_orbit_index(n1, l1, j1d, tz1d)
            i2 = orbs.get_orbit_index(n2, l2, j2d, tz2d)
            i3 = orbs.get_orbit_index(n3, l3, j3d, tz3d)
            i4 = orbs.get_orbit_index(n4, l4, j4d, tz4d)
#            print(i1,i2,i3,i4,m1d,m2d,m3d,m4d,self.get_2bme_Mscheme(i1, m1d, i2, m2d, i3, m3d, i4, m4d, mud),coef)
            me += self.get_2bme_Mscheme(i1, m1d, i2, m2d, i3, m3d, i4, m4d, mud) * coef
        return me

    def get_2bme_from_Mscheme(self, a, b, c, d, Jab, Jcd):
        orbits = self.ms.orbits
        oa, ob, oc, od = orbits.get_orbit(a), orbits.get_orbit(b), orbits.get_orbit(c), orbits.get_orbit(d)
        me = 0.0
        Mab, Mcd = 0, 0
        if(abs(_clebsch_gordan(Jcd, self.rankJ, Jab, Mcd, Mab-Mcd, Mab)) < 1.e-8):
            Mab, Mcd = Jab, Jcd
        norm = 1
        if(a==b): norm /= np.sqrt(2)
        if(c==d): norm /= np.sqrt(2)
        for mda, mdb, mdc, mdd in itertools.product(range(-oa.j, oa.j+2, 2), range(-ob.j, ob.j+2, 2), range(-oc.j, oc.j+2, 2), range(-od.j, od.j+2, 2)):
            if((mda + mdb)//2 != Mab): continue
            if((mdc + mdd)//2 != Mcd): continue
            for sa, sb, sc, sd in itertools.product([-1,1], repeat=4):
                mla, mlb, mlc, mld = (mda - sa)//2, (mdb - sb)//2, (mdc - sc)//2, (mdd - sd)//2
                coef =  _clebsch_gordan(oa.l, 0.5, oa.j*0.5, mla, sa*0.5, mda*0.5) * \
                        _clebsch_gordan(ob.l, 0.5, ob.j*0.5, mlb, sb*0.5, mdb*0.5) * \
                        _clebsch_gordan(oc.l, 0.5, oc.j*0.5, mlc, sc*0.5, mdc*0.5) * \
                        _clebsch_gordan(od.l, 0.5, od.j*0.5, mld, sd*0.5, mdd*0.5) * \
                        _clebsch_gordan(oa.j*0.5, ob.j*0.5, Jab, mda*0.5, mdb*0.5, Mab) * \
                        _clebsch_gordan(oc.j*0.5, od.j*0.5, Jcd, mdc*0.5, mdd*0.5, Mcd) * \
                        np.sqrt(2*Jab+1) / _clebsch_gordan(Jcd, self.rankJ, Jab, Mcd, Mab-Mcd, Mab)
                aa = [oa.n, oa.l, mla, sa, oa.z]
                bb = [ob.n, ob.l, mlb, sb, ob.z]
                cc = [oc.n, oc.l, mlc, sc, oc.z]
                dd = [od.n, od.l, mld, sd, od.z]
                me += coef * self.get_2bme_Mscheme_nlms(aa, bb, cc, dd, 2*(Mab-Mcd))
        return me*norm

    def compare_operators(self, op, ax):
        orbs = self.ms.orbits
        norbs = orbs.get_num_orbits()
        x, y = [], []
        for i, j, in itertools.product(list(range(1,norbs+1)), repeat=2):
            if(i>j): continue
            me1 = self.get_1bme(i,j)
            me2 = op.get_1bme(i,j)
            if(abs(me1) < 1.e-8): continue
            x.append(me1)
            y.append(me2)
        ax.plot(x,y,ms=4,marker="o",c="r",mfc="orange",ls="",label="one-body")
        if(len(x)>0):
            vmin = min(x+y)
            vmax = max(x+y)
        else:
            vmin = 1e100
            vmax =-1e100

        x, y = [], []
        for channels in self.two.keys():
            for idxs in self.two[channels].keys():
                me1 = self.two[channels][idxs]
                me2 = op.get_2bme_from_mat_indices(*channels,*idxs)
                if(abs(me1) < 1.e-8): continue
                x.append(me1)
                y.append(me2)
        ax.plot(x,y,ms=4,marker="s",c="b",mfc="skyblue",ls="",label="two-body")
        vmin = min(x+y+[vmin,])
        vmax = max(x+y+[vmax,])
        ax.plot([vmin,vmax],[vmin,vmax],ls=":",lw=0.8,label="y=x",c="k")

    def set_number_op(self, normalization=None):
        """
        set < p || Q || q >
        """
        self.allocate_operator(self.ms)
        orbits = self.ms.orbits
        if(self.rankJ != 0 or self.rankP != 1 or self.rankZ !=0): raise ValueError()
        for p in range(1,orbits.get_num_orbits()+1):
            for q in range(p,orbits.get_num_orbits()+1):
                op = orbits.get_orbit(p)
                oq = orbits.get_orbit(q)
                if(op.n != oq.n): continue
                if(op.l != oq.l): continue
                if(op.j != oq.j): continue
                if(op.z != oq.z): continue
                me = 1.0
                if(normalization!=None): me /= normalization
                self.set_1bme(p,q,me)
        self.reduced=False

    def set_electric_op(self, hw, e_p=1.0, e_n=0.0):
        """
        set < p || Q || q >
        """
        if(self.rankP != (-1)**(self.rankJ)): raise ValueError
        if(self.rankZ != 0): raise ValueError
        self.allocate_operator(self.ms)
        hc = physical_constants['reduced Planck constant times c in MeV fm'][0]
        m = (physical_constants['proton mass energy equivalent in MeV'][0] + physical_constants['neutron mass energy equivalent in MeV'][0])/2
        b = np.sqrt(hc**2 / (m*hw))
        lam = self.rankJ
        orbits = self.ms.orbits
        for p in range(1,orbits.get_num_orbits()+1):
            for q in range(p,orbits.get_num_orbits()+1):
                op = orbits.get_orbit(p)
                oq = orbits.get_orbit(q)
                if(op.z != oq.z): continue
                if((op.l+oq.l+lam)%2==1): continue
                if(self._triag(op.j, 2*lam, oq.j)): continue
                if(op.z == 1): e_ch = e_n
                if(op.z ==-1): e_ch = e_p
                I = self._radial_integral(op.n, op.l, oq.n, oq.l, lam) * b**lam
                me = 1/np.sqrt(4*np.pi) * (-1)**((oq.j-1)//2+lam) * np.sqrt((2*lam+1)*(op.j+1)*(oq.j+1)) * \
                        float(wigner_3j(op.j*0.5, oq.j*0.5, lam, 0.5, -0.5, 0)) * I * e_ch
                self.set_1bme(p,q,me)
        self.reduced=True

    def set_magnetic_op(self, hw, gl_p=1.0, gs_p=None, gl_n=0.0, gs_n=None, rankT=None):
        """
        set < p || M || q >
        """
        if(self.rankP != (-1)**(self.rankJ+1)): raise ValueError
        if(self.rankZ != 0): raise ValueError
        self.allocate_operator(self.ms)
        hc = physical_constants['reduced Planck constant times c in MeV fm'][0]
        m = (physical_constants['proton mass energy equivalent in MeV'][0] + physical_constants['neutron mass energy equivalent in MeV'][0])/2
        b = np.sqrt(hc**2 / (m*hw))
        if(gs_p==None): gs_p = physical_constants['proton mag. mom. to nuclear magneton ratio'][0]*2
        if(gs_n==None): gs_n = physical_constants['neutron mag. mom. to nuclear magneton ratio'][0]*2
        lam = self.rankJ
        orbits = self.ms.orbits
        for p in range(1,orbits.get_num_orbits()+1):
            for q in range(p,orbits.get_num_orbits()+1):
                op = orbits.get_orbit(p)
                oq = orbits.get_orbit(q)
                if(op.z != oq.z): continue
                if((op.l+oq.l+lam)%2==0): continue
                if(not abs(op.j-oq.j) <= 2*lam <= op.j+oq.j): continue
                if(op.z==-1):
                    gl = gl_p
                    gs = gs_p
                if(op.z== 1):
                    gl = gl_n
                    gs = gs_n
                if(rankT==0):
                    gl = (gl_p + gl_n)*0.5
                    gs = (gs_p + gs_n)*0.5
                if(rankT==1):
                    gl = (gl_p - gl_n)*0.5*(-op.z)
                    gs = (gs_p - gs_n)*0.5*(-op.z)
                kappa = (-1)**(op.l+(op.j+1)//2)*(op.j+1)*0.5 + (-1)**(oq.l+(oq.j+1)//2)*(oq.j+1)*0.5
                I = self._radial_integral(op.n, op.l, oq.n, oq.l, lam-1) * b**(lam-1)
                me = 1/np.sqrt(4*np.pi) * (-1)**((oq.j-1)//2+lam) * np.sqrt((2*lam+1)*(op.j+1)*(oq.j+1)) * \
                        float(wigner_3j(op.j*0.5, oq.j*0.5, lam, 0.5, -0.5, 0)) * \
                        (lam-kappa) * (gl * (1 + kappa/(lam+1)) - 0.5*gs) * I
                self.set_1bme(p,q,me)
        self.reduced=True

    def _radial_integral(self, na, la, nb, lb, lam):
        res = 0.0
        if((la+lb+lam)%2==1): raise ValueError('invalid')
        tau_a = max((lb-la+lam)//2, 0)
        tau_b = max((la-lb+lam)//2, 0)
        for sigma in range(max(0,na-tau_a,nb-tau_b), min(na, nb)+1):
            res += gamma((la+lb+lam)/2 + sigma + 1.5) / (gamma(sigma+1)*gamma(na-sigma+1)*gamma(nb-sigma+1)*\
                    gamma(sigma+tau_a-na+1)*gamma(sigma+tau_b-nb+1))
        res *= (-1)**(na+nb) * np.sqrt(gamma(na+1)*gamma(nb+1) / (gamma(na+la+1.5)*gamma(nb+lb+1.5))) * gamma(tau_a+1) * gamma(tau_b+1)
        return res

    def set_gamow_teller_op(self):
        """
        set < p || sigma || q > < p or n| tau_+/- | n or p > = \sqrt{6 (2jp+1)(2jq+1)} {1/2 1/2 1} (-1)**(jp+lp+3/2)
                                                                                       {jq  jp  l}
        """
        if(self.rankJ != 1): raise ValueError
        if(self.rankP != 1): raise ValueError
        if(self.rankZ != 1): raise ValueError
        self.allocate_operator(self.ms)
        orbits = self.ms.orbits
        for p in range(1,orbits.get_num_orbits()+1):
            for q in range(p,orbits.get_num_orbits()+1):
                op = orbits.get_orbit(p)
                oq = orbits.get_orbit(q)
                if(op.z == oq.z): continue
                if(op.n != oq.n): continue
                if(op.l != oq.l): continue
                if(not abs(op.j-oq.j) <= 2 <= op.j+oq.j): continue
                me = np.sqrt(6*(op.j+1)*(oq.j+1)) * (-1)**((op.j+3)//2 + op.l) * float(wigner_6j(0.5, 0.5, 1, 0.5*oq.j, 0.5*op.j, op.l))
                self.set_1bme(p,q,me)
        self.reduced=True


    def set_fermi_op(self):
        """
        set < p || 1 || q > < p or n| tau_+/- | n or p > = \sqrt{(2j_p+1)}
        """
        if(self.rankJ != 0): raise ValueError
        if(self.rankP != 1): raise ValueError
        if(self.rankZ != 1): raise ValueError
        self.allocate_operator(self.ms)
        orbits = self.ms.orbits
        for p in range(1,orbits.get_num_orbits()+1):
            for q in range(p,orbits.get_num_orbits()+1):
                op = orbits.get_orbit(p)
                oq = orbits.get_orbit(q)
                if(op.z == oq.z): continue
                if(op.n != oq.n): continue
                if(op.l != oq.l): continue
                if(op.j != oq.j): continue
                self.set_1bme(p,q,np.sqrt((op.j+1)))
        self.reduced=True

    def set_rp4(self, hw):
        if(self.rankJ != 0): raise ValueError
        if(self.rankP != 1): raise ValueError
        if(self.rankZ != 0): raise ValueError
        self.allocate_operator(self.ms)
        orbits = self.ms.orbits
        for p in range(1,orbits.get_num_orbits()+1):
            for q in range(p,orbits.get_num_orbits()+1):
                op = orbits.get_orbit(p)
                oq = orbits.get_orbit(q)
                if(op.z != oq.z): continue
                if(op.l != oq.l): continue
                if(op.j != oq.j): continue
                me = BasicFunctions.RadialInt(op.n, op.l, oq.n, oq.l, hw, 4)
                self.set_1bme(p,q,me)
        self.reduced=False

    def set_double_fermi_op(self):
        """
        set < pq:J || 1 || rs:J > < pp or nn | (tau tau)_+/- | nn or pp > = \sqrt{(2J+1)} 2
        """
        if(self.rankJ != 0): raise ValueError
        if(self.rankP != 1): raise ValueError
        if(self.rankZ != 2): raise ValueError
        self.allocate_operator(self.ms)
        orbits = self.ms.orbits
        for channels in self.two.keys():
            chbra, chket = self.ms.two.get_channel(channels[0]), self.ms.two.get_channel(channels[1])
            if(chbra.J != chket.J): continue
            if(chbra.P != chket.P): continue
            if(abs(chbra.Z - chket.Z) != 2): continue
            J = chket.J
            for ibra in range(chbra.get_number_states()):
                p, q = chbra.get_indices(ibra)
                o_p, o_q = chbra.get_orbits(ibra)
                for iket in range(chket.get_number_states()):
                    r, s = chket.get_indices(iket)
                    o_r, o_s = chket.get_orbits(iket)
                    me = 0.0
                    if(o_p.n == o_r.n and o_p.l == o_r.l and o_p.j == o_r.j and \
                            o_q.n == o_s.n and o_q.l == o_s.l and o_q.j == o_s.j):
                        me += np.sqrt(2*J+1) * 2
                    if(o_p.n == o_s.n and o_p.l == o_s.l and o_p.j == o_s.j and \
                            o_q.n == o_r.n and o_q.l == o_r.l and o_q.j == o_r.j):
                        me += np.sqrt(2*J+1) * 2 * (-1)**((o_r.j+o_s.j)/2 - J+1)
                    if(p==q): me /= np.sqrt(2)
                    if(r==s): me /= np.sqrt(2)
                    if(abs(me) > 1.e-8): self.set_2bme_from_indices(p, q, r, s, J, J, me)
        self.reduced=True

    def reduce(self):
        if(self.reduced): return
        if(self.rankJ != 0): return
        orb = self.ms.orbits
        for i, j in itertools.product(list(range(1,orb.get_num_orbits()+1)), repeat=2):
            if(i>j): continue
            oi = orb.get_orbit(i)
            oj = orb.get_orbit(j)
            me = self.get_1bme(i,j)
            if(abs(me)<1.e-16): continue
            self.set_1bme(i,j,np.sqrt(oj.j+1)*me)

        for channel in self.two.keys():
            tbc_bra = self.ms.two.get_channel(channel[0])
            tbc_ket = self.ms.two.get_channel(channel[1])
            Jij = tbc_bra.J
            Jkl = tbc_ket.J
            for idx in self.two[channel].keys():
                me = self.two[channel][idx]
                self.two[channel][idx] = me * np.sqrt(2*Jij+1)

    def unreduce(self):
        if(not self.reduced): return
        if(self.rankJ != 0): return
        orb = self.ms.orbits
        for i, j in itertools.product(list(range(1,orb.get_num_orbits()+1)), repeat=2):
            if(i>j): continue
            oi = orb.get_orbit(i)
            oj = orb.get_orbit(j)
            me = op.get_1bme(i,j)
            op.set_1bme(i,j,me/np.sqrt(oj.j+1))

        for channel in self.two.keys():
            tbc_bra = self.ms.two.get_channel(channel[0])
            tbc_ket = self.ms.two.get_channel(channel[1])
            Jij = tbc_bra.J
            Jkl = tbc_ket.J
            for idx in self.two[channel].keys():
                me = self.two[channel][idx]
                self.two[channel][idx] = me / np.sqrt(2*Jij+1)

    def surface_delta_interaction(self, R0, hw, g0, g1=0, g2=0, g3=0):
        """
        surface-delta interaction, V * delta( r1 - r2 ) * delta( |r1| - R0 )
        V0 = g0 + g1 (sigma1 . sigma2) + g2 (tau1 . tau2) + g3 (sigma1 . sigma2) (tau1 . tau2)
        g0, g1, g2, g3 are in unit of MeV fm4
        hw is the frequency for the HO basis function
        """
        if(__package__==None or __package__==""):
            from BasicFunctions import HO_radial, Ysigma
        else:
            from .BasicFunctions import HO_radial, Ysigma
        from sympy.physics.wigner import wigner_3j, wigner_6j, wigner_9j
        from sympy import N
        def me_sdi(oa, ob, oc, od, J, verbose=False):
            radial_part = R0**4 * HO_radial(R0, oa.n, oa.l, hw) * HO_radial(R0, ob.n, ob.l, hw) * \
                    HO_radial(R0, oc.n, oc.l, hw) * HO_radial(R0, od.n, od.l, hw)
            t1t2_1, t1t2_d = 0, 0
            if(abs(oa.z + ob.z)==2):
                t1t2_1 = 1
                t1t2_d = 1
            elif(oa.z+ob.z==0 and oa.z==oc.z):
                t1t2_1 = 1
                t1t2_d = -1
            elif(oa.z+ob.z==0 and oa.z!=oc.z):
                t1t2_1 = 0
                t1t2_d = 2
            if(verbose): print(t1t2_1, t1t2_d)
            term_1, term_s = 0, 0
            for l in range(max(abs(oa.l-oc.l), abs(ob.l-od.l)), min(oa.l+oc.l, ob.l+od.l)+1):
                ang = (-1)**((ob.j+oc.j)/2+J) * N(wigner_6j(oa.j*0.5, ob.j*0.5, J, od.j*0.5, oc.j*0.5, l, prec=8))
                print(oa.j, ob.j, oc.j, od.j, J, l)
                if(verbose): print('l, ang', l, ang)
                if(abs(ang)<1.e-8): continue
                term_1 += ang * Ysigma(oa.l, oa.j*0.5, oc.l, oc.j*0.5, l, 0, l) * Ysigma(ob.l, ob.j*0.5, od.l, od.j*0.5, l, 0, l)
                for k in range(abs(l-1), l+2):
                    term_s += ang * Ysigma(oa.l, oa.j*0.5, oc.l, oc.j*0.5, l, 1, k) * Ysigma(ob.l, ob.j*0.5, od.l, od.j*0.5, l, 1, k)
            return g0 * term_1 * t1t2_1 + g1 * term_s * t1t2_1 + g2 * term_1 * t1t2_d + g3 * term_s * t1t2_d
        tbs = self.ms.two
        fac_norm = 1/np.sqrt(2.0)
        for tbc in tbs.channels:
            J = tbc.J
            for ibra in range(tbc.get_number_states()):
                for iket in range(ibra+1):
                    a, b = tbc.get_indices(ibra)
                    c, d = tbc.get_indices(iket)
                    oa, ob = tbc.get_orbits(ibra)
                    oc, od = tbc.get_orbits(iket)
                    norm = 1
                    if(a==b): norm *= fac_norm
                    if(c==d): norm *= fac_norm
                    me = me_sdi(oa, ob, oc, od, J)
                    me-= me_sdi(oa, ob, od, oc, J) * (-1)**((oc.j+od.j)/2 + J)
                    self.set_2bme_from_indices(a, b, c, d, J, J, me*norm)

    def set_pairing(self, G):
        """
        Eq. (12.11) from J. Suhonen, From Nucleons to Nucleus, Vol. 23 (Springer Berlin Heidelberg, Berlin, Heidelberg, 2007).
        <pp:0 | V | qq:0 > = G sqrt([jp][jq]) / 2
        """
        tbs = self.ms.two
        for tbc in tbs.channels:
            J = tbc.J
            if(J!=0): continue
            for ibra in range(tbc.get_number_states()):
                for iket in range(ibra+1):
                    a, b = tbc.get_indices(ibra)
                    c, d = tbc.get_indices(iket)
                    if(a != b): continue
                    if(c != d): continue
                    oa, ob = tbc.get_orbits(ibra)
                    oc, od = tbc.get_orbits(iket)
                    me = G * np.sqrt((oa.j+1)*(oc.j+1)) * 0.5
                    self.set_2bme_from_indices(a, b, c, d, J, J, me)

    def set_QQforce(self, H, hw):
        """
        Eq. (8.55) from J. Suhonen, From Nucleons to Nucleus, Vol. 23 (Springer Berlin Heidelberg, Berlin, Heidelberg, 2007).
        """
        E2 = Operator(rankJ=2, ms=self.ms)
        E2.set_electric_op(hw, e_p=1.0, e_n=1.0)
        orbits = self.ms.orbits
        def me_NA(a,b,c,d,J):
            oa, ob, oc, od = orbits.get_orbit(a), orbits.get_orbit(b), orbits.get_orbit(c), orbits.get_orbit(d)
            me = (-1)**((oa.j + ob.j)/2+J) * float(wigner_6j(oa.j*0.5, ob.j*0.5, J, od.j*0.5, oc.j*0.5, 2)) * E2.get_1bme(c,a) * E2.get_1bme(b,d)
            return me

        tbs = self.ms.two
        for tbc in tbs.channels:
            J = tbc.J
            for ibra in range(tbc.get_number_states()):
                for iket in range(ibra+1):
                    a, b = tbc.get_indices(ibra)
                    c, d = tbc.get_indices(iket)
                    oa, ob = tbc.get_orbits(ibra)
                    oc, od = tbc.get_orbits(iket)
                    norm = 1.0
                    if(a==b): norm /= np.sqrt(2.0)
                    if(c==d): norm /= np.sqrt(2.0)
                    me = me_NA(a,b,c,d,J)
                    me -= me_NA(a,b,d,c,J) * (-1.0)**((oc.j+od.j)/2+J)
                    me *= norm * H
                    self.set_2bme_from_indices(a, b, c, d, J, J, me)

    def set_QdotQ(self, hw, e_p=1, e_n=0, full_space=True):
        """
        Similar to QQ force but it is O = [\sum_i Q_i x \sum_j Q_j]_0
        """
        E2 = Operator(rankJ=2, ms=self.ms)
        E2.set_electric_op(hw, e_p=e_p, e_n=e_n)
        orbits = self.ms.orbits
        for op in orbits.orbits:
            p = orbits.get_orbit_index_from_orbit(op)
            for oq in orbits.orbits:
                q = orbits.get_orbit_index_from_orbit(oq)
                if(op.z != oq.z): continue
                if(op.l != oq.l): continue
                if(op.j != oq.j): continue
                e = 0
                if(op.z == -1): e = e_p
                if(op.z ==  1): e = e_n
                me = 0.0
                if(full_space): me = BasicFunctions.RadialInt(op.n, op.l, oq.n, oq.l, hw, 4) * np.sqrt(5) / (4*np.pi) * e**2 # free-space
                else:
                    for o in orbits.orbits:
                        i = orbits.get_orbit_index_from_orbit(o)
                        me += (-1.0)**((o.j+op.j)//2+1) * E2.get_1bme(p,i) * E2.get_1bme(i,q) / float(op.j+1) / np.sqrt(5) # within the model space
                self.set_1bme(p,q,me)

        def me_NA(a,b,c,d,J):
            oa, ob, oc, od = orbits.get_orbit(a), orbits.get_orbit(b), orbits.get_orbit(c), orbits.get_orbit(d)
            me = (-1.0)**((ob.j + oc.j)//2+J) * float(wigner_6j(oa.j*0.5, ob.j*0.5, J, od.j*0.5, oc.j*0.5, 2)) * E2.get_1bme(a,c) * E2.get_1bme(b,d)
            return me

        tbs = self.ms.two
        for tbc in tbs.channels:
            J = tbc.J
            for ibra in range(tbc.get_number_states()):
                for iket in range(ibra+1):
                    a, b = tbc.get_indices(ibra)
                    c, d = tbc.get_indices(iket)
                    oa, ob = tbc.get_orbits(ibra)
                    oc, od = tbc.get_orbits(iket)
                    norm = 1.0
                    if(a==b): norm /= np.sqrt(2.0)
                    if(c==d): norm /= np.sqrt(2.0)
                    me = me_NA(a,b,c,d,J)
                    me -= me_NA(a,b,d,c,J) * (-1.0)**((oc.j+od.j)//2+J)
                    me *= norm * 2 / np.sqrt(5)
                    self.set_2bme_from_indices(a, b, c, d, J, J, me)

    def set_pairing_QQ(self, hw, g_p=0, g_QQ=0):
        Hp, HQQ = Operator(ms=self.ms), Operator(ms=self.ms)
        Hp.set_pairing(g_p)
        HQQ.set_QQforce(g_QQ, hw)
        tmp = Hp + HQQ
        self.two = tmp.two

    def mass_dependent_tbme(self, A):
        mass_dep = 1
        if(self.kshell_options[0]==1): mass_dep =(float(A) / float(self.kshell_options[1]))**float(self.kshell_options[2])
        for channels in self.two:
            ichbra, ichket = channels
            chbra, chket = self.ms.two.get_channel(ichbra), self.ms.two.get_channel(ichket)
            for ibra in range(chbra.get_number_states()):
                for iket in range(chket.get_number_states()):
                    try:
                        self.two[channels][(ibra,iket)] *= mass_dep
                    except:
                        pass

    def operator_ovlap(op1, op2, normalize=False):
        def full_matrix_2body(op):
            ms2 = op.ms.two
            orbs = ms2.orbits
            idxm_to_idx_m = {}
            idx = 0
            for o in orbs.orbits:
                for m in range(-o.j, o.j+2, 2):
                    idxm_to_idx_m[idx] = (orbs.get_orbit_index_from_orbit(o),m)
                    idx += 1
            ijlist = list(itertools.combinations_with_replacement(list(range(len(idxm_to_idx_m))),2))
            mat = np.zeros((len(ijlist),len(ijlist)))
            for ibra, ij in enumerate(ijlist):
                for iket, kl in enumerate(ijlist):
                    i, j = ij
                    k, l = kl
                    i_i, m_i = idxm_to_idx_m[i]
                    i_j, m_j = idxm_to_idx_m[j]
                    i_k, m_k = idxm_to_idx_m[k]
                    i_l, m_l = idxm_to_idx_m[l]
                    mat[ibra,iket] = op.get_2bme_Mscheme(i_i, m_i, i_j, m_j, i_k, m_k, i_l, m_l, m_i+m_j-m_k-m_l)
            return mat

        if(op1.rankJ != op2.rankJ): raise ValueError
        if(op1.rankP != op2.rankP): raise ValueError
        if(op1.rankZ != op2.rankZ): raise ValueError
        op = op2.truncate(op1.ms)
        if(not op.reduced): op.to_reduced()
        if(not op1.reduced): op1.to_reduced()
        if(normalize):
            ovlp1 = np.trace(np.matmul(op1.one, op.one)) / np.sqrt(np.trace(np.matmul(op1.one, op1.one)) * np.trace(np.matmul(op.one, op.one)))
        else:
            ovlp1 = np.trace(np.matmul(op1.one, op.one))
        mat1 = full_matrix_2body(op1)
        mat2 = full_matrix_2body(op)
        if(normalize):
            ovlp2 = np.trace(np.matmul(mat1, mat2)) / np.sqrt(np.trace(np.matmul(mat1, mat1)) * np.trace(np.matmul(mat2, mat2)))
        else:
            ovlp2 = np.trace(np.matmul(mat1, mat2))
        op.to_nonreduced()
        op1.to_nonreduced()
        return ovlp1, ovlp2

    def set_tau1_x_tau2(self):
        """
        (tau1 x tau2) (sigma1 x sigma2) = - i(tau1 x tau2) i(sigma1 x sigma2)
        """
        tbs = self.ms.two
        op2 = self.two
        orbits = self.ms.orbits
        def me_NA(a,b,c,d,Jab,Jcd):
            oa, ob, oc, od = orbits.get_orbit(a), orbits.get_orbit(b), orbits.get_orbit(c), orbits.get_orbit(d)
            if(oa.n != oc.n or oa.l != oc.l): return 0
            if(ob.n != od.n or ob.l != od.l): return 0
            i_a, i_b, i_c, i_d = (1-oa.z)//2, (1-ob.z)//2, (1-oc.z)//2, (1-od.z)//2
            if(oa.z + ob.z - oc.z - od.z == 0):
                me = _paulix[i_a,i_c] * _pauliy[i_b,i_d] - _pauliy[i_a,i_c] * _paulix[i_b,i_d]
            if(oa.z + ob.z - oc.z - od.z == 2):
                me = (_pauliy[i_a,i_c] * _pauliz[i_b,i_d] - _pauliz[i_a,i_c] * _pauliy[i_b,i_d]) + 1j * (_pauliz[i_a,i_c] * _paulix[i_b,i_d] - _paulix[i_a,i_c] * _pauliz[i_b,i_d])
            if(oa.z + ob.z - oc.z - od.z ==-2):
                me = (_pauliy[i_a,i_c] * _pauliz[i_b,i_d] - _pauliz[i_a,i_c] * _pauliy[i_b,i_d]) - 1j * (_pauliz[i_a,i_c] * _paulix[i_b,i_d] - _paulix[i_a,i_c] * _pauliz[i_b,i_d])
            me *= 1j
            me *= np.sqrt((2*Jab+1)*(2*Jcd+1)*6) * _ninej(0.5*oa.j, 0.5*ob.j, Jab, 0.5*oc.j, 0.5*od.j, Jcd, 1, 1, 1) * \
                    np.sqrt(6*(oa.j+1)*(oc.j+1)) * (-1)**((oa.j+3)//2 + oa.l) * _sixj(0.5, 0.5, 1, 0.5*oc.j, 0.5*oa.j, oa.l) * \
                    np.sqrt(6*(ob.j+1)*(od.j+1)) * (-1)**((ob.j+3)//2 + ob.l) * _sixj(0.5, 0.5, 1, 0.5*od.j, 0.5*ob.j, ob.l)
            me *= -1
            return me.real
        def _me(a,b,c,d,Jab,Jcd):
            norm = 1.0
            if(a==b): norm /= np.sqrt(2.0)
            if(c==d): norm /= np.sqrt(2.0)
            oa, ob, oc, od = orbits.get_orbit(a), orbits.get_orbit(b), orbits.get_orbit(c), orbits.get_orbit(d)
            me = me_NA(a,b,c,d,Jab,Jcd)
            me -= me_NA(a,b,d,c,Jab,Jcd) * (-1.0)**((oc.j+od.j)//2+Jcd)
            me *= norm
            return me

        for channels in op2.keys():
            chbra = tbs.get_channel(channels[0])
            chket = tbs.get_channel(channels[1])
            for ibra in range(chbra.get_number_states()):
                for iket in range(chket.get_number_states()):
                    a, b = chbra.get_indices(ibra)
                    c, d = chket.get_indices(iket)
                    me = _me(a, b, c, d, chbra.J, chket.J)
                    self.set_2bme_from_indices(a, b, c, d, chbra.J, chket.J, me)
    def set_density_at_r(self, r, iso=0):
        pass
    def set_kinetic_density_at_r(self, r, iso=0):
        pass
    def set_spin_orbit_density_at_r(self, r, iso=0):
        pass


def main():
    ms = ModelSpace.ModelSpace()
    ms.set_modelspace_from_boundaries(4)
    op = Operator(verbose=False)
    op.allocate_operator(ms)
    op.print_operator()
    ham = Operator(filename='XXX.snt')
    c, ls, t = ham.spin_tensor_decomposition()
if(__name__=="__main__"):
    main()

