#!/usr/bin/env python3
import os, sys, copy, gzip, subprocess, time, itertools
import functools
import numpy as np
import pandas as pd
from sympy.physics.wigner import wigner_3j, wigner_6j, wigner_9j, clebsch_gordan
from sympy.physics.quantum.cg import CG
import functools
if(__package__==None or __package__==""):
    import ModelSpace
else:
    from . import Orbits
    from . import ModelSpace

@functools.lru_cache(maxsize=None)
def _sixj(j1, j2, j3, j4, j5, j6):
    return float(wigner_6j(j1, j2, j3, j4, j5, j6))
@functools.lru_cache(maxsize=None)
def _ninej(j1, j2, j3, j4, j5, j6, j7, j8, j9):
    return float(wigner_9j(j1, j2, j3, j4, j5, j6, j7, j8, j9))
@functools.lru_cache(maxsize=None)
def _clebsch_gordan(j1, j2, j3, m1, m2, m3):
    return float(clebsch_gordan(j1, j2, j3, m1, m2, m3))


class TransitionDensity:
    def __init__(self, Jbra=None, Jket=None, wflabel_bra=None, wflabel_ket=None, ms=None, filename=None, file_format="kshell", verbose=False):
        self.Jbra = Jbra
        self.Jket = Jket
        self.wflabel_bra = wflabel_bra
        self.wflabel_ket = wflabel_ket
        self.ms = copy.deepcopy(ms)
        self.verbose = verbose
        self.bin_header = None
        self.one = {}
        self.two = {}
        self.three = {}
        if( ms != None ): self.allocate_density( ms )
        if( filename != None ): self.read_density_file( filename, file_format )

    def __add__(self, other):
        if(self.Jbra != other.Jbra): raise ValueError
        if(self.Jket != other.Jket): raise ValueError
        if(self.wflabel_bra != other.wflabel_bra): raise ValueError
        if(self.wflabel_ket != other.wflabel_ket): raise ValueError
        target = TransitionDensity(Jbra=self.Jbra, Jket=self.Jket, wflabel_bra=self.wflabel_bra, wflabel_ket=self.wflabel_ket, ms=self.ms)
        orbs = self.ms.orbits
        norbs = orbs.get_num_orbits()
        for i, j, in itertools.product(list(range(1,norbs+1)), repeat=2):
            oi = orbs.get_orbit(i)
            oj = orbs.get_orbit(j)
            for J in range(abs(oi.j-oj.j)//2, (oi.j+oj.j)//2+1):
                me1 = self.get_1btd(i,j,J)
                me2 = other.get_1btd(i,j,J)
                self.set_1btd(i,j,J,me1+me2)

        for channels in self.two.keys():
            for idxs in self.two[channels].keys():
                me1 = self.two[channels][idxs]
                me2 = other.get_2btd_from_mat_indices(*channels,*idxs)
                target.set_2btd_form_mat_indices(*channels,*idxs,me1+me2)
        return target

    def __sub__(self, other):
        if(self.Jbra != other.Jbra): raise ValueError
        if(self.Jket != other.Jket): raise ValueError
        if(self.wflabel_bra != other.wflabel_bra): raise ValueError
        if(self.wflabel_ket != other.wflabel_ket): raise ValueError
        target = TransitionDensity(Jbra=self.Jbra, Jket=self.Jket, wflabel_bra=self.wflabel_bra, wflabel_ket=self.wflabel_ket, ms=self.ms)
        orbs = self.ms.orbits
        norbs = orbs.get_num_orbits()
        for i, j, in itertools.product(list(range(1,norbs+1)), repeat=2):
            oi = orbs.get_orbit(i)
            oj = orbs.get_orbit(j)
            for J in range(abs(oi.j-oj.j)//2, (oi.j+oj.j)//2+1):
                me1 = self.get_1btd(i,j,J)
                me2 = other.get_1btd(i,j,J)
                self.set_1btd(i,j,J,me1+me2)

        for channels in self.two.keys():
            for idxs in self.two[channels].keys():
                me1 = self.two[channels][idxs]
                me2 = other.get_2btd_from_mat_indices(*channels,*idxs)
                target.set_2btd_form_mat_indices(*channels,*idxs,me1-me2)
        return target

    def __mul__(self, coef):
        target = TransitionDensity(Jbra=self.Jbra, Jket=self.Jket, wflabel_bra=self.wflabel_bra, wflabel_ket=self.wflabel_ket, ms=self.ms)
        orbs = self.ms.orbits
        norbs = orbs.get_num_orbits()
        for i, j, in itertools.product(list(range(1,norbs+1)), repeat=2):
            oi = orbs.get_orbit(i)
            oj = orbs.get_orbit(j)
            for J in range(abs(oi.j-oj.j)//2, (oi.j+oj.j)//2+1):
                me1 = self.get_1btd(i,j,J)
                me2 = other.get_1btd(i,j,J)
                self.set_1btd(i,j,J,me1*coef)

        for channels in self.two.keys():
            for idxs in self.two[channels].keys():
                me1 = self.two[channels][idxs]
                target.set_2btd_form_mat_indices(*channels,*idxs,me1*coef)
        return target

    def __truediv__(self, coef):
        return self.__mul__(1/coef)

    def allocate_density( self, ms ):
        self.ms = copy.deepcopy(ms)
        orbits = ms.orbits
        self.one = {}
        two = ms.two
        for ichbra in range(two.get_number_channels()):
            chbra = two.get_channel(ichbra)
            for ichket in range(two.get_number_channels()):
                chket = two.get_channel(ichket)
                self.two[(ichbra,ichket)] = {}
        if(self.ms.rank==2): return
        three = ms.three
        for ichbra in range(three.get_number_channels()):
            chbra = three.get_channel(ichbra)
            for ichket in range(three.get_number_channels()):
                chket = three.get_channel(ichket)
                self.three[(ichbra,ichket)] = {}
    def count_nonzero_1btd(self):
        return len(self.one)
    def count_nonzero_2btd(self):
        counter = 0
        two = self.ms.two
        nch = two.get_number_channels()
        for i in range(nch):
            chbra = two.get_channel(i)
            for j in range(nch):
                chket = two.get_channel(j)
                counter += len( self.two[(i,j)] )
        return counter
    def set_1btd( self, a, b, jrank, me):
        if(abs(me) < 1.e-16): return
        self.one[(a,b,jrank)] = me
    def set_2btd_from_mat_indices( self, chbra, chket, bra, ket, jrank, me ):
        if(abs(me) < 1.e-16): return
        self.two[(chbra,chket)][(bra,ket,jrank)] = me
    def set_2btd_from_indices( self, a, b, c, d, Jab, Jcd, jrank, me ):
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
        if( self._triag( Jab, Jcd, jrank )): raise ValueError("Transition density rank mismatch")
        ichbra = two.get_index(Jab,Pab,Zab)
        ichket = two.get_index(Jcd,Pcd,Zcd)
        chbra = two.get_channel(ichbra)
        chket = two.get_channel(ichket)
        bra = chbra.index_from_indices[(a,b)]
        ket = chket.index_from_indices[(c,d)]
        phase = chbra.phase_from_indices[(a,b)] * chket.phase_from_indices[(c,d)]
        self.set_2btd_from_mat_indices(ichbra,ichket,bra,ket,jrank,me*phase)
    def set_2btd_from_orbits( self, oa, ob, oc, od, Jab, Jcd, jrank, me ):
        orbits = self.ms.orbits
        a = orbits.orbit_index_from_orbit( oa )
        b = orbits.orbit_index_from_orbit( ob )
        c = orbits.orbit_index_from_orbit( oc )
        d = orbits.orbit_index_from_orbit( od )
        self.set_2btd_from_indices( a, b, c, d, Jab, Jcd, jrank, me )

    def get_1btd(self,*args):
        try:
            return self.one[args]
        except:
            return 0
    def get_1btd_Mscheme(self, p, mp2, q, mq2, Mbra, Mket, scalar=False):
        orbs = self.ms.orbits
        op = orbs.get_orbit(p)
        oq = orbs.get_orbit(q)
        v = 0.0
        for lam in range(max(int(abs(self.Jbra-self.Jket)), abs(op.j-oq.j)//2), min((op.j+oq.j)//2, int(self.Jbra+self.Jket))+1):
            if(scalar==True and lam>0): continue
            v += (-1)**(self.Jket-Mket+(oq.j-mq2)//2) * \
                    _clebsch_gordan(self.Jket, self.Jbra, lam, Mket, -Mbra, Mket-Mbra) * \
                    _clebsch_gordan(op.j*0.5, oq.j*0.5, lam, mp2*0.5, -mq2*0.5, Mket-Mbra) * \
                    self.get_1btd(p, q, lam)
        return v

    def get_2btd_from_mat_indices(self, chbra, chket, bra, ket, jrank):
        try:
            return self.two[(chbra,chket)][(bra,ket,jrank)]
        except:
            #if(self.verbose): print("Nothing here " + sys._getframe().f_code.co_name )
            return 0
    def get_2btd_from_indices( self, a, b, c, d, Jab, Jcd, jrank ):
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
        if( self._triag( Jab, Jcd, jrank )):
            if(self.verbose): print("Warning: J, " + sys._getframe().f_code.co_name )
            return 0
        try:
            ichbra = two.get_index(Jab,Pab,Zab)
            ichket = two.get_index(Jcd,Pcd,Zcd)
        except:
            if(self.verbose): print("Warning: channel bra & ket index, " + sys._getframe().f_code.co_name )
            return 0
        chbra = two.get_channel(ichbra)
        chket = two.get_channel(ichket)
        try:
            bra = chbra.index_from_indices[(a,b)]
            ket = chket.index_from_indices[(c,d)]
        except:
            if(self.verbose): print("Warning: bra & ket index, " + sys._getframe().f_code.co_name )
            return 0
        phase = chbra.phase_from_indices[(a,b)] * chket.phase_from_indices[(c,d)]
        return self.get_2btd_from_mat_indices(ichbra,ichket,bra,ket,jrank)*phase
    def get_2btd_from_orbits( self, oa, ob, oc, od, Jab, Jcd, jrank ):
        if(self.ms.rank <= 1): return 0
        orbits = self.ms.orbits
        a = orbits.get_orbit_index_from_orbit( oa )
        b = orbits.get_orbit_index_from_orbit( ob )
        c = orbits.get_orbit_index_from_orbit( oc )
        d = orbits.get_orbit_index_from_orbit( od )
        return self.get_2btd_from_indices( a, b, c, d, Jab, Jcd, jrank )

    def get_2btd_Mscheme(self, p, mdp, q, mdq, r, mdr, s, mds, Mbra, Mket, add_cg=True):
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
                if(Mpq-Mrs != Mket-Mbra): continue
                for lam in range(max(abs(Jpq-Jrs), int(abs(self.Jbra-self.Jket))), min(Jpq+Jrs, int(self.Jbra+self.Jket))+1):
                    if(add_cg):
                        me += (-1)**(self.Jket+Jrs-Mket-Mrs) * \
                                _clebsch_gordan(self.Jket, self.Jbra, lam, Mket, -Mbra, Mket-Mbra) * \
                                _clebsch_gordan(Jpq, Jrs, lam, Mpq, -Mrs, Mket-Mbra) * \
                                _clebsch_gordan(o_p.j*0.5, o_q.j*0.5, Jpq, mdp*0.5, mdq*0.5, Mpq) * \
                                _clebsch_gordan(o_r.j*0.5, o_s.j*0.5, Jrs, mdr*0.5, mds*0.5, Mrs) * \
                                self.get_2btd_from_indices(p, q, r, s, Jpq, Jrs, lam)
                    else:
                        me += (-1)**(self.Jket+Jrs-Mket-Mrs) * \
                                _clebsch_gordan(Jpq, Jrs, lam, Mpq, -Mrs, Mket-Mbra) * \
                                _clebsch_gordan(o_p.j*0.5, o_q.j*0.5, Jpq, mdp*0.5, mdq*0.5, Mpq) * \
                                _clebsch_gordan(o_r.j*0.5, o_s.j*0.5, Jrs, mdr*0.5, mds*0.5, Mrs) * \
                                self.get_2btd_from_indices(p, q, r, s, Jpq, Jrs, lam)
        me *= norm
        return me

    def get_2btd_from_Mscheme(self, p, q, r, s, Jpq, Jrs, jrank, Mbra, Mket):
        orbs = self.ms.orbits
        o_p, o_q, o_r, o_s = orbs.get_orbit(p), orbs.get_orbit(q), orbs.get_orbit(r), orbs.get_orbit(s)
        norm = 1
        if(p==q): norm /= np.sqrt(2.0)
        if(r==s): norm /= np.sqrt(2.0)
        cg = _clebsch_gordan(self.Jket, self.Jbra, jrank, Mket, -Mbra, Mket-Mbra)
        add_cg = True
        if(abs(cg) < 1.e-16):
            add_cg = False
        else:
            fact = 1 / cg
        me = 0.0
        for mdp in range(-o_p.j, o_p.j+2, 2):
            for mdq in range(-o_q.j, o_q.j+2, 2):
                Mpq = (mdp + mdq)//2
                if(abs(Mpq) > Jpq): continue

                for mdr in range(-o_r.j, o_r.j+2, 2):
                    for mds in range(-o_s.j, o_s.j+2, 2):
                        Mrs = (mdr + mds)//2
                        if(abs(Mrs) > Jrs): continue
                        if(Mpq-Mrs != Mket-Mbra): continue
                        if(add_cg):
                            me += (-1)**(self.Jket+Jrs-Mket-Mrs) * fact * \
                                _clebsch_gordan(Jpq, Jrs, jrank, Mpq, -Mrs, Mket-Mbra) * \
                                _clebsch_gordan(o_p.j*0.5, o_q.j*0.5, Jpq, mdp*0.5, mdq*0.5, Mpq) * \
                                _clebsch_gordan(o_r.j*0.5, o_s.j*0.5, Jrs, mdr*0.5, mds*0.5, Mrs) * \
                                self.get_2btd_Mscheme(p, mdp, q, mdq, r, mdr, s, mds, Mbra, Mket)
                        else:
                            me += (-1)**(self.Jket+Jrs-Mket-Mrs) * \
                                _clebsch_gordan(Jpq, Jrs, jrank, Mpq, -Mrs, Mket-Mbra) * \
                                _clebsch_gordan(o_p.j*0.5, o_q.j*0.5, Jpq, mdp*0.5, mdq*0.5, Mpq) * \
                                _clebsch_gordan(o_r.j*0.5, o_s.j*0.5, Jrs, mdr*0.5, mds*0.5, Mrs) * \
                                self.get_2btd_Mscheme(p, mdp, q, mdq, r, mdr, s, mds, Mbra, Mket, add_cg=False)
        return me * norm

    def _triag(self,J1,J2,J3):
        b = True
        if(abs(J1-J2) <= J3 <= J1+J2): b = False
        return b
    def read_density_file(self, filename=None, file_format="kshell"):
        if(filename == None):
            print(" set file name!")
            return
        if( file_format=="kshell"):
            if( filename.find("bin")!=-1): self._read_td_binary(filename)
            else: self._read_td_kshell_format(filename)
            if(self.ms == None):
                self.one=None
                self.two=None
                return
            if( self.count_nonzero_1btd() + self.count_nonzero_2btd() == 0):
                print("The number of non-zero transition density matrix elements is 0 better to check: "+ filename + "!! " + \
                        "Jbra=" + str(self.Jbra) + " (wf label:"+ str(self.wflabel_bra)+"), Jket="+str(self.Jket)+" (wf label:"+str(self.wflabel_ket)+")")
            return
        if( file_format=="nutbar"):
            self._read_td_nutbar_format(filename)
            if( self.count_nonzero_1btd() + self.count_nonzero_2btd() == 0):
                print("The number of non-zero transition density matrix elements is 0 better to check: "+ filename + "!!")
            return
        if( file_format=="me1j"):
            self._read_me1j_density(filename)
            if( self.count_nonzero_1btd() == 0):
                print("The number of non-zero transition density matrix elements is 0 better to check: "+ filename + "!!")
            return

    def _skip_comment(self,f,comment="#"):
        while True:
            x = f.tell()
            line = f.readline()
            if(line[0] != comment):
                f.seek(x)
                return
    def _read_td_kshell_format(self, filename):
        if(not os.path.exists(filename)):
            print("file is not found {}".format(filename))
            return
        f = open(filename, 'r')
        orbs = Orbits()
        flag=True
        for i in range(200):
            line = f.readline()
            if(line[1:12] == "model space"):
                flag=False
                break
        if(flag):
            print("The file {:s} might be crushed. Please check it!".format(filename))
            return
        while True:
            entry = f.readline().split()
            if( len(entry)==0 ): break
            if( entry[0]=='k,'): continue
            orbs.add_orbit( int(entry[1]), int(entry[2]), int(entry[3]), int(entry[4]) )
        #ms = ModelSpace.ModelSpace()
        ms = ModelSpace()
        ms.set_modelspace_from_orbits( orbs )
        self.allocate_density( ms )

        lines = f.readlines()
        f.close()

        i = 0
        i_obtd = 0
        i_tbtd = 0
        store_obtd=False
        store_tbtd=False
        for line in lines:
            i += 1
            if(not line.startswith("OBTD:")): i_obtd = 0
            if(not line.startswith("TBTD:")): i_tbtd = 0
            if(line.startswith("OBTD:")):
                if(i_obtd == 0):
                    str1 = lines[i-3]
                    if(str1[0:4] != 'w.f.'): print("see file "+filename+" at line "+str(i))
                    d = str1.split()
                    j2_bra = int(d[2][:-3])
                    j2_ket = int(d[5][:-3])
                    i_bra = int(d[3][:-1])
                    i_ket = int(d[6][:-1])
                    if(j2_bra == int(2*self.Jbra) and j2_ket == int(2*self.Jket) and
                            i_bra == self.wflabel_bra and i_ket == self.wflabel_ket): store_obtd=True
                    if(j2_bra != int(2*self.Jbra) or j2_ket != int(2*self.Jket) or
                            i_bra != self.wflabel_bra or i_ket != self.wflabel_ket): store_obtd=False
                    if(not store_obtd): i_obtd = -1
                if(store_obtd):
                    data = line.split()
                    a, b, jr, wf_label_bra, wf_label_ket, me = int(data[1]), int(data[2]), int(data[4]), \
                            int(data[6]), int(data[7]), float(data[9])
                    self.set_1btd(a,b,jr,me)
                    i_obtd += 1
                continue
            if(line.startswith("TBTD")):
                if(i_tbtd == 0):
                    str1 = lines[i-3]
                    if(str1[0:4] != 'w.f.'): print("see file "+filename+" at line "+str(i))
                    d = str1.split()
                    j2_bra = int(d[2][:-3])
                    j2_ket = int(d[5][:-3])
                    i_bra = int(d[3][:-1])
                    i_ket = int(d[6][:-1])
                    if(j2_bra == int(2*self.Jbra) and j2_ket == int(2*self.Jket) and
                            i_bra == self.wflabel_bra and i_ket == self.wflabel_ket): store_tbtd=True
                    if(j2_bra != int(2*self.Jbra) or j2_ket != int(2*self.Jket) or
                            i_bra != self.wflabel_bra or i_ket != self.wflabel_ket): store_tbtd=False
                    if(not store_tbtd): i_tbtd = -1
                if(store_tbtd):
                    data = line.split()
                    a, b, c, d, Jab, Jcd, Jr, wf_label_bra, wf_label_ket, me = \
                            int(data[1]), int(data[2]), int(data[3]), int(data[4]), \
                            int(data[6]), int(data[7]), int(data[8]), \
                            int(data[10]), int(data[11]), float(data[13])
                    self.set_2btd_from_indices(a,b,c,d,Jab,Jcd,Jr,me)
                    i_tbtd += 1
                continue

    def _read_td_nutbar_format(self, filename):
        f = open(filename,"r")
        self._skip_comment(f)
        orbs = Orbits()
        while True:
            x = f.tell()
            entry = f.readline().split()
            if(entry[0] == "#"):
                f.seek(x)
                break
            orbs.add_orbit( int(entry[1]), int(entry[2]), int(entry[3]), -int(entry[4]) )
        ms = ModelSpace.ModelSpace()
        ms.set_modelspace_from_orbits( orbs )
        self.allocate_density( ms )
        two_body_kets = []
        self._skip_comment(f)
        while True:
            x = f.tell()
            entry = f.readline().split()
            if(len(entry) == 0): break
            two_body_kets.append( (int(entry[1]), int(entry[2]), int(entry[3])) )
        entry = f.readline().split()
        self.Jbra = float(entry[6])
        self.wflabel_bra = int(entry[7])
        self.Jket = float(entry[8])
        self.wflabel_ket = int(entry[9])
        Jrank = int(float(  entry[10]))
        line = f.readline()
        while True:
            x = f.tell()
            entry = f.readline().split()
            if(len(entry) == 0): break
            self.set_1btd( int(entry[0]), int(entry[1]), Jrank, float(entry[2]) )
        line = f.readline()
        while True:
            x = f.tell()
            entry = f.readline().split()
            if(len(entry) == 0): break
            a = two_body_kets[int(entry[0])][0]
            b = two_body_kets[int(entry[0])][1]
            Jab= two_body_kets[int(entry[0])][2]
            c = two_body_kets[int(entry[1])][0]
            d = two_body_kets[int(entry[1])][1]
            Jcd= two_body_kets[int(entry[1])][2]
            self.set_2btd_from_indices( a, b, c, d, Jab, Jcd, Jrank, float(entry[2]))
        print("----------------------------------------")
        print(" read nutbar-format density ")
        print(" < Jf_nf | lambda | Ji_ni > ")
        print(" Ji = {0:6.2f}, ni = {1:3d}".format( self.Jket, self.wflabel_ket ))
        print(" lambda = {0:3d}         ".format( Jrank ))
        print(" Jf = {0:6.2f}, nf = {1:3d}".format( self.Jbra, self.wflabel_bra ))
        print("----------------------------------------")
        f.close()

    def _read_me1j_density(self, filename):
        orbs = self.ms.orbits
        emax, lmax = orbs.emax, orbs.lmax
        orbits_remap = []
        for e in range(emax+1):
            lmin = e%2
            for l in range(lmin, min(e,lmax)+1, 2):
                n = (e-l)//2
                for twoj in range(abs(2*l-1), 2*l+2, 2):
                    for tz in [-1,1]:
                        orbits_remap.append(orbs.get_orbit_index(n, l, twoj, tz))
        nljmax = len(orbits_remap)
        f = open(filename,'r')
        line = f.readline()
        line = f.readline()

        icount = 0
        data = f.readline().split()
        for nlj1 in range(nljmax):
            a = orbits_remap[nlj1]
            o1 = orbs.get_orbit(a)
            e1 = 2 * o1.n + o1.l;
            if (e1 > emax): continue
            for nlj2 in range(nlj1+1):
                b = orbits_remap[nlj2]
                o2 = orbs.get_orbit(b)
                e2 = 2 * o2.n + o2.l;
                if (e2 > emax): continue

                me = float(data[icount%10])
                if(icount%10 == 9): data = f.readline().split()
                icount += 1
                self.set_1btd(a,b,0,me)
                self.set_1btd(b,a,0,me)
        f.close()

    def print_occupation(self, filename):
        orbs = self.ms.orbits
        prt = ''
        for op in orbs.orbits:
            p = orbs.get_orbit_index(op.n, op.l, op.j, op.z)
            prt += f"{op.n:3d} {op.l:3d} {op.j:4d} {op.z:4d} {self.get_1btd(p,p,0):16.6e}\n"
        f = open(filename, 'w')
        f.write(prt)
        f.close()
        return

    def print_density(self):
        orbits = self.ms.orbits
        print(" Model spapce ")
        for i in range(1, orbits.get_num_orbits()+1):
            o = orbits.get_orbit(i)
            print("{0:3d}, {1:3d}, {2:3d}, {3:3d}, {4:3d}".format(i, o.n, o.l, o.j, o.z ))
        print(" One body ")
        for key in self.one.keys():
            a = key[0]
            b = key[1]
            j = key[2]
            if(abs(self.one[key])>1.e-8): print("{0:3d}, {1:3d}, {2:3d}, {3:12.6f}".format(a,b,j,self.one[key]))
        print(" Two body ")
        two = self.ms.two
        for ichbra in range(two.get_number_channels()):
            chbra = two.get_channel(ichbra)
            for ichket in range(two.get_number_channels()):
                chket = two.get_channel(ichket)
                for key in self.two[(ichbra,ichket)].keys():
                    bra, ket, Jr = key
                    a, b = chbra.orbit1_index[bra], chbra.orbit2_index[bra]
                    c, d = chket.orbit1_index[ket], chket.orbit2_index[ket]
                    tbtd = self.two[(ichbra,ichket)][key]
                    if(abs(tbtd)>1.e-8): print("{0:3d}, {1:3d}, {2:3d}, {3:3d}, {4:3d}, {5:3d}, {6:3d}, {7:12.6f}".format(a,b,c,d,chbra.J,chket.J,Jr,tbtd))

    def _read_td_binary(self, filename, byte_order='little'):
        with open(filename, "rb") as fp:
            if(self.ms==None):
                n_orbs = fp.read(4); n_orbs = int.from_bytes(n_orbs,byteorder=byte_order, signed=True)
                orbs = Orbits()
                for i in range(n_orbs):
                    tmp = []
                    for j in range(4):
                        idx = int.from_bytes(fp.read(4), byteorder=byte_order, signed=True)
                        tmp.append(idx)
                    orbs.add_orbit(*tuple(tmp))
                ms = ModelSpace()
                ms.set_modelspace_from_orbits(orbs)
                self.allocate_density( ms )
            else: fp.seek(self.ms.orbits.get_num_orbits()*4*4 + 4, 0)

            if(self.bin_header==None):
                n = int.from_bytes(fp.read(4), byteorder=byte_order, signed=True)
                self.bin_header = []
                for i in range(n):
                    J2bra = int.from_bytes(fp.read(4), byteorder=byte_order, signed=True)
                    wflabel_bra = int.from_bytes(fp.read(4), byteorder=byte_order, signed=True)
                    J2ket = int.from_bytes(fp.read(4), byteorder=byte_order, signed=True)
                    wflabel_ket = int.from_bytes(fp.read(4), byteorder=byte_order, signed=True)
                    self.bin_header.append((J2bra,wflabel_bra,J2ket,wflabel_ket))
                    #print(J2bra, wflabel_bra, J2ket, wflabel_ket)
            else: fp.seek( 4+len(self.bin_header)*4*4, 1 )

            try:
                n = self.bin_header.index((int(self.Jbra*2), self.wflabel_bra, int(self.Jket*2), self.wflabel_ket))
            except:
                raise ValueError("Mismatch of J and wf label was detected!")
            for i in range(n):
                n1 = int.from_bytes(fp.read(4), byteorder=byte_order, signed=True)
                n2 = int.from_bytes(fp.read(4), byteorder=byte_order, signed=True)
                skip_block_size = (n1*20) + (n2*36)
                fp.seek(skip_block_size, 1)

            bin_parser=[]
            n1 = int.from_bytes(fp.read(4), byteorder=byte_order, signed=True)
            n2 = int.from_bytes(fp.read(4), byteorder=byte_order, signed=True)
            arrs = []
            for i in range(n1):
                a = int.from_bytes(fp.read(4), byteorder=byte_order, signed=True)
                b = int.from_bytes(fp.read(4), byteorder=byte_order, signed=True)
                rank = int.from_bytes(fp.read(4), byteorder=byte_order, signed=True)
                arrs.append((a,b,rank))
            bin_parser.append(arrs)

            arrs = []
            for i in range(n2):
                a = int.from_bytes(fp.read(4), byteorder=byte_order, signed=True)
                b = int.from_bytes(fp.read(4), byteorder=byte_order, signed=True)
                c = int.from_bytes(fp.read(4), byteorder=byte_order, signed=True)
                d = int.from_bytes(fp.read(4), byteorder=byte_order, signed=True)
                Jab = int.from_bytes(fp.read(4), byteorder=byte_order, signed=True)
                Jcd = int.from_bytes(fp.read(4), byteorder=byte_order, signed=True)
                rank = int.from_bytes(fp.read(4), byteorder=byte_order, signed=True)
                arrs.append((a,b,c,d,Jab,Jcd,rank))
            bin_parser.append(arrs)

            for i in range(n1):
                idxs = bin_parser[0][i]
                v = np.frombuffer(fp.read(8),dtype=np.float64)[0]
                self.set_1btd(*idxs, v)
            for i in range(n2):
                idxs = bin_parser[1][i]
                v = np.frombuffer(fp.read(8),dtype=np.float64)[0]
                self.set_2btd_from_indices(*idxs, v)
        return

    def calc_density(kshl_dir, fn_snt, fn_ptn_bra, fn_ptn_ket, fn_wf_bra, fn_wf_ket, i_wfs=None, fn_density=None, \
            header="", batch_cmd=None, run_cmd=None, fn_input="transit.input", calc_SF=False, binary_output=False):
        if(fn_density==None):
            basename = os.path.basename(fn_snt)
            fn_out = "density_" + os.path.splitext(basename)[0] + ".txt"
        if(fn_density!=None): fn_out = fn_density
        fn_density_out = "none"
        if(binary_output): fn_density_out = os.path.splitext(fn_out)[0]+".bin"
        fn_script = os.path.splitext(fn_out)[0] + ".sh"
        cmd = "cp " + kshl_dir + "/transit.exe ./"
        subprocess.call(cmd,shell=True)
        prt = header + '\n'
        prt += 'echo "start runnning ' + fn_out + ' ..."\n'
        prt += 'cat >' + fn_input + ' <<EOF\n'
        prt += '&input\n'
        prt += '  fn_int   = "' + fn_snt + '"\n'
        prt += '  fn_ptn_l = "' + fn_ptn_bra + '"\n'
        prt += '  fn_ptn_r = "' + fn_ptn_ket + '"\n'
        prt += '  fn_load_wave_l = "' + fn_wf_bra + '"\n'
        prt += '  fn_load_wave_r = "' + fn_wf_ket + '"\n'
        if(fn_density_output!='none'): prt += '  fn_density = "' + fn_density_output + '"\n'
        if(i_wfs!=None):
            prt += '  n_eig_lr_pair = '
            for lr in i_wfs:
                prt += str(lr[0]) + ', ' + str(lr[1]) + ', '
            prt += '\n'
        prt += '  hw_type = 2\n'
        prt += '  eff_charge = 1.5, 0.5\n'
        prt += '  gl = 1.0, 0.0\n'
        prt += '  gs = 3.91, -2.678\n'
        if(not calc_SF): prt += '  is_tbtd = .true.\n'
        prt += '&end\n'
        prt += 'EOF\n'
        if(run_cmd == None):
            prt += './transit.exe ' + fn_input + ' > ' + fn_out + ' 2>&1\n'
        if(run_cmd != None):
            prt += run_cmd + ' ./transit.exe ' + fn_input + ' > ' + fn_out + ' 2>&1\n'
        prt += 'rm ' + fn_input + '\n'
        f = open(fn_script,'w')
        f.write(prt)
        f.close()
        os.chmod(fn_script, 0o755)
        if(batch_cmd == None): cmd = "./" + fn_script
        if(batch_cmd != None): cmd = batch_cmd + " " + fn_script
        subprocess.call(cmd, shell=True)
        time.sleep(1)

    def calc_me_0vbb(kshl_dir, fn_snt, fn_0v_snt, fn_ptn_bra, fn_ptn_ket, fn_wf_bra, fn_wf_ket, fn_density=None, \
            header="", batch_cmd=None, run_cmd=None, fn_input="transit.input"):
        if(fn_density==None):
            basename = os.path.basename(fn_snt)
            fn_out = "calc_me_0vbb_" + os.path.splitext(basename)[0] + ".dat"
        if(fn_density!=None): fn_out = fn_density
        fn_script = os.path.splitext(fn_out)[0] + ".sh"
        cmd = "cp " + kshl_dir + "/transit.exe ./"
        subprocess.call(cmd,shell=True)
        prt = header + '\n'
        prt += 'echo "start runnning ' + fn_out + ' ..."\n'
        prt += 'cat >' + fn_input + ' <<EOF\n'
        prt += '&input\n'
        prt += '  fn_int   = "' + fn_snt + '"\n'
        prt += '  fn_ptn_l = "' + fn_ptn_bra + '"\n'
        prt += '  fn_ptn_r = "' + fn_ptn_ket + '"\n'
        prt += '  fn_load_wave_l = "' + fn_wf_bra + '"\n'
        prt += '  fn_load_wave_r = "' + fn_wf_ket + '"\n'
        prt += '  hw_type = 2\n'
        prt += '  eff_charge = 1.5, 0.5\n'
        prt += '  gl = 1.0, 0.0\n'
        prt += '  gs = 3.91, -2.678\n'
        prt += '&end\n'
        prt += 'EOF\n'
        from . import Op
        nme = Op(fn_0v_snt, rankJ=0, rankP=1, rankZ=2)
        nme.read_operator_file()
        nme.write_nme_file()
        if(run_cmd == None):
            prt += './transit.exe ' + fn_input + ' > ' + fn_out + ' 2>&1\n'
        if(run_cmd != None):
            prt += run_cmd + ' ./transit.exe ' + fn_input + ' > ' + fn_out + ' 2>&1\n'
        prt += 'rm ' + fn_input + '\n'
        f = open(fn_script,'w')
        f.write(prt)
        f.close()
        os.chmod(fn_script, 0o755)
        if(batch_cmd == None): cmd = "./" + fn_script
        if(batch_cmd != None): cmd = batch_cmd + " " + fn_script
        subprocess.call(cmd, shell=True)
        time.sleep(1)

    def eval_expectation_value( self, op, J1=None, J2=None, pn=None ):
        return self.calc_expectation_value( op, J1, J2, pn )
    def eval( self, op, J1=None, J2=None, pn=None ):
        return self.calc_expectation_value( op, J1, J2, pn )
    def calc_expectation_value( self, op, J1=None, J2=None, pn=None ):
        orbits_de = self.ms.orbits
        orbits_op = op.ms.orbits
        norbs = orbits_op.get_num_orbits()
        if(pn!=None):
            if(pn=="pp"): tz=-1
            if(pn=="nn"): tz=1
            if(pn=="pn"): tz=0
        else: tz=None

        zero = op.get_0bme()
        one = 0.0
        for i, j, in itertools.product(list(range(1,norbs+1)), repeat=2):
            oi = orbits_op.get_orbit(i)
            i_d = orbits_de.get_orbit_index(oi.n, oi.l, oi.j, oi.z)
            oj = orbits_op.get_orbit(j)
            j_d = orbits_de.get_orbit_index(oj.n, oj.l, oj.j, oj.z)
            if( tz!=None and oi.z!=tz and oj.z!=tz ): continue
            if( J1!=None and oi.j!=J1 and oj.j!=J1 ): continue
            if( abs(oi.z-oj.z) != 2*op.rankZ): continue
            if((-1)**(oi.l+oj.l) * op.rankP != 1): continue
            if( self._triag( oi.j, oj.j, 2*op.rankJ )): continue
            if( op.rankJ==0 and op.rankP==1 and op.rankZ==0 and (not op.reduced) ):
                one += op.get_1bme(i,j) * self.get_1btd(i_d,j_d,op.rankJ) * np.sqrt(oj.j+1) / np.sqrt(2*self.Jbra+1)
            else:
                #print( "{:3d}{:3d}{:10.6f}{:10.6f}{:10.6f}".format(i_d,j_d,\
                #        op.get_1bme(i,j), self.get_1btd(i_d,j_d,op.rankJ), op.get_1bme(i,j) * self.get_1btd(i_d,j_d,op.rankJ) ))
                one += op.get_1bme(i,j) * self.get_1btd(i_d,j_d,op.rankJ)

        two = 0.0
        ijlist = list(itertools.combinations_with_replacement(list(range(1,norbs+1)),2))
        for ij, kl in itertools.product(ijlist, repeat=2):
            i, j = ij
            k, l = kl
            oi = orbits_op.get_orbit(i)
            oj = orbits_op.get_orbit(j)
            ok = orbits_op.get_orbit(k)
            ol = orbits_op.get_orbit(l)

            i_d = orbits_de.get_orbit_index(oi.n, oi.l, oi.j, oi.z)
            j_d = orbits_de.get_orbit_index(oj.n, oj.l, oj.j, oj.z)
            k_d = orbits_de.get_orbit_index(ok.n, ok.l, ok.j, ok.z)
            l_d = orbits_de.get_orbit_index(ol.n, ol.l, ol.j, ol.z)
            if((-1)**(oi.l+oj.l+ok.l+ol.l) * op.rankP != 1): continue
            if( abs(oi.z+oj.z-ok.z-ol.z) != 2*op.rankZ): continue
            if( tz!=None and (oi.z+oj.z)//2!=tz and (ok.z+ol.z)//2!=tz ): continue

            Jijlist = list(range( int(abs(oi.j-oj.j)/2), int((oi.j+oj.j)/2)+1))
            Jkllist = list(range( int(abs(ok.j-ol.j)/2), int((ok.j+ol.j)/2)+1))
            for Jij, Jkl in itertools.product(Jijlist, Jkllist):
                if(i == j and Jij%2 == 1): continue
                if(k == l and Jkl%2 == 1): continue
                if( self._triag( Jij, Jkl, op.rankJ )): continue
                if( J2!=None and Jij!=J2 and Jkl!=J2 ): continue
                if(op.rankJ==0 and op.rankP==1 and op.rankZ==0 and (not op.reduced)):
                    two += op.get_2bme_from_indices(i,j,k,l,Jij,Jkl) * self.get_2btd_from_indices(i_d,j_d,k_d,l_d,Jij,Jkl,op.rankJ) * \
                            np.sqrt(2*Jij+1)/np.sqrt(2*self.Jbra+1)
                    #print("{:3d},{:3d},{:3d},{:3d},{:3d},{:3d},{:3d},{:12.6f}".format(i_d,j_d,k_d,l_d,Jij,Jkl,op.rankJ,self.get_2btd_from_indices(i_d,j_d,k_d,l_d,Jij,Jkl,op.rankJ)))
                    #print("{:3d},{:3d},{:3d},{:3d},{:3d},{:3d},{:12.6f}".format(i,j,k,l,Jij,Jkl,op.get_2bme_from_indices(i,j,k,l,Jij,Jkl)))
                else:
                    two += op.get_2bme_from_indices(i,j,k,l,Jij,Jkl) * self.get_2btd_from_indices(i_d,j_d,k_d,l_d,Jij,Jkl,op.rankJ)
                    #print("{:3d},{:3d},{:3d},{:3d},{:3d},{:3d},{:3d},{:12.6f}".format(i_d,j_d,k_d,l_d,Jij,Jkl,op.rankJ,self.get_2btd_from_indices(i_d,j_d,k_d,l_d,Jij,Jkl,op.rankJ)))
                    #print("{:3d},{:3d},{:3d},{:3d},{:3d},{:3d},{:12.6f}".format(i,j,k,l,Jij,Jkl,op.get_2bme_from_indices(i,j,k,l,Jij,Jkl)))
                    #if(abs(op.get_2bme_from_indices(i,j,k,l,Jij,Jkl) * self.get_2btd_from_indices(i_d,j_d,k_d,l_d,Jij,Jkl,op.rankJ))>1.e-16):
                    #    print("{:3d},{:3d},{:3d},{:3d},{:3d},{:3d},{:16.10f},{:16.10f}".format(i,j,k,l,Jij,Jkl,op.get_2bme_from_indices(i,j,k,l,Jij,Jkl),\
                    #        self.get_2btd_from_indices(i_d,j_d,k_d,l_d,Jij,Jkl,op.rankJ)))
        return zero,one,two
#    def eval_NME_non_closure(self, that, op):
#        """
#        NME(K) = sum_{i<=j:proton} sum_{k<=l:neutron} (ij|Op|kl) (Jf|ci^t ck|K) (K|cj^t cl|Ji)
#        self => (Jf|c^t c|K)
#        that => (K|c^t c|Ji)
#        """
#        orbits_de = self.ms.orbits
#        orbits_op = op.ms.orbits
#        norbs = orbits_op.get_num_orbits()
#        res = 0
#        if(self.Jket != that.Jbra): raise ValueError()
#        Jf = self.Jbra
#        K = self.Jket
#        Ji = that.Jket
#        #ijlist = list(itertools.combinations_with_replacement(list(range(1,norbs+1)),2))
#        ijlist = list(itertools.product(list(range(1,norbs+1)),repeat=2))
#        for ij, kl in itertools.product(ijlist, repeat=2):
#            i, j = ij
#            k, l = kl
#            norm = 1
#            if(i==j): norm *= np.sqrt(2)
#            if(k==l): norm *= np.sqrt(2)
#            oi = orbits_op.get_orbit(i)
#            oj = orbits_op.get_orbit(j)
#            ok = orbits_op.get_orbit(k)
#            ol = orbits_op.get_orbit(l)
#
#            i_d = orbits_de.get_orbit_index(oi.n, oi.l, oi.j, oi.z)
#            j_d = orbits_de.get_orbit_index(oj.n, oj.l, oj.j, oj.z)
#            k_d = orbits_de.get_orbit_index(ok.n, ok.l, ok.j, ok.z)
#            l_d = orbits_de.get_orbit_index(ol.n, ol.l, ol.j, ol.z)
#            if((-1)**(oi.l+oj.l+ok.l+ol.l) * op.rankP != 1): continue
#            if( abs(oi.z+oj.z-ok.z-ol.z) != 4): continue
#            Jijlist = list(range( int(abs(oi.j-oj.j)/2), int((oi.j+oj.j)/2)+1))
#            Jkllist = list(range( int(abs(ok.j-ol.j)/2), int((ok.j+ol.j)/2)+1))
#            Jiklist = list(range( int(abs(oi.j-ok.j)/2), int((oi.j+ok.j)/2)+1))
#            Jjllist = list(range( int(abs(oj.j-ol.j)/2), int((oj.j+ol.j)/2)+1))
#            for Jij, Jkl in itertools.product(Jijlist, Jkllist):
#                if(i == j and Jij%2 == 1): continue
#                if(k == l and Jkl%2 == 1): continue
#                if( self._triag( Jij, Jkl, op.rankJ )): continue
#                for Jik, Jjl in itertools.product(Jiklist, Jjllist):
#                    if( self._triag( Jik, Jjl, op.rankJ )): continue
#                    #res += float(wigner_9j(0.5*oi.j, 0.5*oj.j, Jij, 0.5*ok.j, 0.5*ol.j, Jkl, Jik, Jjl, op.rankJ)) * \
#                    #        float(wigner_6j(Jik, Jjl, op.rankJ, Ji, Jf, K)) * \
#                    #        np.sqrt((2*Jij+1)*(2*Jkl+1)/(2*op.rankJ+1))*(2*Jik+1)*(2*Jjl+1) * (-1)**(Ji+Jf+op.rankJ) * \
#                    #        op.get_2bme_from_indices(i,j,k,l,Jij,Jkl) * \
#                    #        self.get_1btd(i_d,k_d,Jik) * that.get_1btd(j_d,l_d,Jjl) * norm * 0.25
#                    res += float(_ninej(0.5*oi.j, 0.5*oj.j, Jij, 0.5*ok.j, 0.5*ol.j, Jkl, Jik, Jjl, op.rankJ)) * \
#                            float(_sixj(Jik, Jjl, op.rankJ, Ji, Jf, K)) * \
#                            np.sqrt((2*Jij+1)*(2*Jkl+1)/(2*op.rankJ+1))*(2*Jik+1)*(2*Jjl+1) * (-1)**(Ji+Jf+op.rankJ) * \
#                            op.get_2bme_from_indices(i,j,k,l,Jij,Jkl) * \
#                            self.get_1btd(i_d,k_d,Jik) * that.get_1btd(j_d,l_d,Jjl) * norm * 0.25
#        return res

    def to_DataFrame(self, rank=None):
        if(rank==1 or rank==None):
            orbits = self.ms.orbits
            tmp = []
            for idx in self.one.keys():
                tmp.append({"a":idx[0],"b":idx[1],"rank":idx[2],"1 body":self.one[idx]})
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
                    tmp.append({"a":a, "b":b, "c":c, "d":d, "Jab":Jab, "Jcd":Jcd, "rank":idx[2],"2 body":self.two[channels][idx]})
            if(len(tmp)==0):
                two = pd.DataFrame()
            else:
                two = pd.DataFrame(tmp)
                two = two.iloc[list(~two["2 body"].eq(0)),:].reset_index(drop=True)
        if(rank==1): return one
        if(rank==2): return two
        if(rank==None): return one, two

    def compare_transition_densities(self, op, ax):
        orbs = self.ms.orbits
        norbs = orbs.get_num_orbits()
        x, y = [], []
        for i, j, in itertools.product(list(range(1,norbs+1)), repeat=2):
            me1 = self.get_1btd(i,j)
            me2 = op.get_1btd(i,j)
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
                me2 = op.get_2btd_from_mat_indices(*channels,*idxs)
                if(abs(me1) < 1.e-8): continue
                x.append(me1)
                y.append(me2)
        ax.plot(x,y,ms=4,marker="s",c="b",mfc="skyblue",ls="",label="two-body")
        vmin = min(x+y+[vmin,])
        vmax = max(x+y+[vmax,])
        ax.plot([vmin,vmax],[vmin,vmax],ls=":",lw=0.8,label="y=x",c="k")

    def get_sp_entropy(self, i, m2, Mbra, Mket):
        if(self.Jbra != self.Jket): raise "bra and ket wave functions need to be the same"
        if(self.wflabel_bra != self.wflabel_ket): raise "bra and ket wave functions need to be the same"
        oi = self.ms.orbits.get_orbit(i)
        nprob = self.get_1btd_Mscheme(i, m2, i, m2, Mbra, Mket)
        if(nprob<1.e-16): nprob=1.e-16
        if(nprob>1.0): nprob=1-1.e-16
        s = - nprob * np.log(nprob) - (1-nprob) * np.log((1-nprob))
        return s

    def get_sp_entropy_from_orbit(self, oi, m2, Mbra, Mket):
        return self.get_sp_entropy(self.ms.orbits.get_orbit_index_from_orbit(oi), m2, Mbra, Mket)

    def get_sp_entropy_from_qns(self, n, l, j, z, m2, Mbra, Mket):
        return self.get_sp_entropy(self.ms.orbits.get_orbit_index_from_tuple((n, l, j, z)), m2, Mbra, Mket)

    def get_2b_Mscheme_entropy(self, i1, m1, i2, m2, Mbra, Mket):
        orbits = self.ms.orbits
        o1 = orbits.get_orbit(i1)
        o2 = orbits.get_orbit(i2)
        return self.get_2b_Mscheme_entropy_from_qns(o1.n, o1.l, o1.j, o1.z, m1, o2.n, o2.l, o2.j, o2.z, m2, Mbra, Mket)

    def get_2b_Mscheme_entropy_from_qns(self, n1, l1, j1, z1, m1, n2, l2, j2, z2, m2, Mbra, Mket):
        if(self.Jbra != self.Jket): raise "bra and ket wave functions need to be the same"
        if(self.wflabel_bra != self.wflabel_ket): raise "bra and ket wave functions need to be the same"
        orbits = self.ms.orbits
        i1 = orbits.get_orbit_index(n1, l1, j1, z1)
        i2 = orbits.get_orbit_index(n2, l2, j2, z2)
        o1 = orbits.get_orbit(i1)
        o2 = orbits.get_orbit(i2)
        rho = np.zeros((4,4))
        tbtd = self.get_2btd_Mscheme(i1, m1, i2, m2, i1, m1, i2, m2, Mbra, Mket)
        rho[0,0] = 1 - self.get_1btd_Mscheme(i1,m1,i1,m1,Mbra,Mket) - self.get_1btd_Mscheme(i2,m2,i2,m2,Mbra,Mket) + tbtd
        rho[1,1] = self.get_1btd_Mscheme(i2,m2,i2,m2,Mbra,Mket) - tbtd
        rho[2,2] = self.get_1btd_Mscheme(i1,m1,i1,m1,Mbra,Mket) - tbtd
        rho[3,3] = tbtd
        rho[1,2] = self.get_1btd_Mscheme(i1,m1,i2,m2,Mbra,Mket)
        rho[2,1] = self.get_1btd_Mscheme(i2,m2,i1,m1,Mbra,Mket)
        e_val, e_vec = np.linalg.eigh(rho)
        s = 0
        for e in e_val:
            if(e<1.e-16): e=1.e-16
            if(e>1.0): e=1-1.e-16
            s -= e * np.log(e)
        return s

    def get_2b_Mscheme_mutual_info_from_qns(self, n1, l1, j1, z1, m1, n2, l2, j2, z2, m2, Mbra, Mket):
        if(self.Jbra != self.Jket): raise "bra and ket wave functions need to be the same"
        if(self.wflabel_bra != self.wflabel_ket): raise "bra and ket wave functions need to be the same"
        if(n1==n2 and l1==l2 and j1==j2 and z1==z2 and m1==m2): return 0
        return self.get_sp_entropy_from_qns(n1, l1, j1, z1, m1, Mbra, Mket) + \
                self.get_sp_entropy_from_qns(n2, l2, j2, z2, m2, Mbra, Mket) - \
                self.get_2b_Mscheme_entropy_from_qns(n1, l1, j1, z1, m1, n2, l2, j2, z2, m2, Mbra, Mket)

    def get_2b_Mscheme_mutual_info(self, i1, m1, i2, m2, Mbra, Mket):
        if(self.Jbra != self.Jket): raise "bra and ket wave functions need to be the same"
        if(self.wflabel_bra != self.wflabel_ket): raise "bra and ket wave functions need to be the same"
        if(i1==i2 and m1==m2): return 0
        return self.get_sp_entropy(i1, m1, Mbra, Mket) + \
                self.get_sp_entropy(i2, m2, Mbra, Mket) - \
                self.get_2b_Mscheme_entropy(i1, m1, i2, m2, Mbra, Mket)


def main():
    file_td="transition-density-file-name"
    TD = TransitionDensity()
if(__name__=="__main__"):
    main()
