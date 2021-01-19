#!/usr/bin/env python3
import sys, subprocess
import numpy as np
import copy
import gzip
from sympy import N
from sympy.physics.wigner import wigner_6j, wigner_9j
if(__package__==None or __package__==""):
    import ModelSpace
    import nushell2snt
else:
    from . import Orbits, OrbitsIsospin
    from . import ModelSpace
    from . import nushell2snt

def _ls_coupling(la, ja, lb, jb, Lab, Sab, J):
    return np.sqrt( (2*ja+1)*(2*jb+1)*(2*Lab+1)*(2*Sab+1) ) * \
            N( wigner_9j( la, 0.5, ja, lb, 0.5, jb, Lab, Sab, J) )


class Operator:
    def __init__(self, rankJ=0, rankP=1, rankZ=0, ms=None, reduced=False, filename=None, verbose=False):
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
        if( rankJ != 0 ): self.reduced = True
        if( ms != None ): self.allocate_operator( ms )
        if( filename != None ): self.read_operator_file( filename )

    def allocate_operator(self, ms):
        self.ms = copy.deepcopy(ms)
        self.zero = 0.0
        self.one = np.zeros( (ms.orbits.get_num_orbits(), ms.orbits.get_num_orbits() ))
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
        counter = 0
        norbs = self.ms.orbits.get_num_orbits()
        for i in range(norbs):
            for j in range(norbs):
                if( abs( self.one[i,j] ) > 1.e-10 ): counter += 1
        return counter
    def count_nonzero_2bme(self):
        counter = 0
        two = self.ms.two
        nch = two.get_number_channels()
        for i in range(nch):
            chbra = two.get_channel(i)
            for j in range(i+1):
                chket = two.get_channel(j)
                if( self._triag( chbra.J, chket.J, self.rankJ )): continue
                if( chbra.P * chket.P * self.rankP != 1): continue
                if( abs(chbra.Z-chket.Z) != self.rankZ): continue
                counter += len( self.two[(i,j)] )
        return counter
    def count_nonzero_3bme(self):
        counter = 0
        three = self.ms.three
        nch = three.get_number_channels()
        for i in range(nch):
            chbra = three.get_channel(i)
            for j in range(i+1):
                chket = three.get_channel(j)
                if( self._triag( chbra.J, chket.J, 2*self.rankJ )): continue
                if( self._triag( chbra.T, chket.T, 2*self.rankZ )): continue
                if( chbra.P * chket.P * self.rankP != 1): continue
                counter += len( self.three[(i,j)] )
        return counter
    def set_0bme( self, me ):
        self.zero = me
    def set_1bme( self, a, b, me):
        orbits = self.ms.orbits
        oa = orbits.get_orbit(a)
        ob = orbits.get_orbit(b)
        if( self._triag(oa.j, ob.j, 2*self.rankJ)):
            if(self.verbose): print("Warning: J, " + sys._getframe().f_code.co_name )
            return
        if( (-1)**(oa.l+ob.l) * self.rankP != 1):
            if(self.verbose): print("Warning: Parity, " + sys._getframe().f_code.co_name )
            return
        if( abs(oa.z-ob.z) != 2*self.rankZ):
            if(self.verbose): print("Warning: Z, " + sys._getframe().f_code.co_name )
            return
        self.one[a-1,b-1] = me
        self.one[b-1,a-1] = me * (-1)**( (ob.j-oa.j)//2 )
    def set_2bme_from_mat_indices( self, chbra, chket, bra, ket, me ):
        if( chbra < chket ):
            if(self.verbose): print("Warning:" + sys._getframe().f_code.co_name )
            return
        self.two[(chbra,chket)][(bra,ket)] = me
        if( chbra == chket ): self.two[(chbra,chket)][ket,bra] = me
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
        if( self._triag( Jab, Jcd, self.rankJ )):
            if(self.verbose): print("Warning: J, " + sys._getframe().f_code.co_name )
            return
        if( Pab * Pcd * self.rankP != 1):
            if(self.verbose): print("Warning: Parity, " + sys._getframe().f_code.co_name )
            return
        if( abs(Zab-Zcd) != self.rankZ):
            if(self.verbose): print("Warning: Z, " + sys._getframe().f_code.co_name )
            return
        ichbra_tmp = two.get_index(Jab,Pab,Zab)
        ichket_tmp = two.get_index(Jcd,Pcd,Zcd)
        phase = 1
        if( ichbra_tmp >= ichket_tmp ):
            ichbra = ichbra_tmp
            ichket = ichket_tmp
            aa, bb, cc, dd, = a, b, c, d
        else:
            ichbra = ichket_tmp
            ichket = ichbra_tmp
            phase *=  (-1)**(Jcd-Jab)
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
        if( chbra < chket ):
            if(self.verbose): print("Warning:" + sys._getframe().f_code.co_name )
            return
        self.three[(chbra,chket)][(bra,ket)] = me
        if( chbra == chket ): self.three[(chbra,chket)][ket,bra] = me
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
        ichbra_tmp = three.get_index(Jbra,Pbra,Tbra)
        ichket_tmp = three.get_index(Jket,Pket,Tket)
        phase = 1
        if( ichbra_tmp >= ichket_tmp ):
            ichbra = ichbra_tmp
            ichket = ichket_tmp
            i, j, k, l, m, n = a, b, c, d, e, f
            Jij, Tij, Jlm, Tlm = Jab, Tab, Jde, Tde
        else:
            ichbra = ichket_tmp
            ichket = ichbra_tmp
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
            if(self.verbose): print("Warning: J, " + sys._getframe().f_code.co_name )
            return 0
        if( Pab * Pcd * self.rankP != 1):
            if(self.verbose): print("Warning: Parity, " + sys._getframe().f_code.co_name )
            return 0
        if( abs(Zab-Zcd) != self.rankZ):
            if(self.verbose): print("Warning: Z, " + sys._getframe().f_code.co_name )
            return 0
        try:
            ichbra_tmp = two.get_index(Jab,Pab,Zab)
            ichket_tmp = two.get_index(Jcd,Pcd,Zcd)
        except:
            if(self.verbose): print("Warning: channel bra & ket index, " + sys._getframe().f_code.co_name )
            return 0
        phase = 1
        if( ichbra_tmp >= ichket_tmp ):
            ichbra = ichbra_tmp
            ichket = ichket_tmp
            aa, bb, cc, dd, = a, b, c, d
        else:
            ichbra = ichket_tmp
            ichket = ichbra_tmp
            phase *=  (-1)**(Jcd-Jab)
            aa, bb, cc, dd, = c, d, a, b
        chbra = two.get_channel(ichbra)
        chket = two.get_channel(ichket)
        try:
            bra = chbra.index_from_indices[(aa,bb)]
            ket = chket.index_from_indices[(cc,dd)]
        except:
            if(self.verbose): print("Warning: bra & ket index, " + sys._getframe().f_code.co_name )
            return 0
        phase *= chbra.phase_from_indices[(aa,bb)] * chket.phase_from_indices[(cc,dd)]
        return self.get_2bme_from_mat_indices(ichbra,ichket,bra,ket)*phase
    def get_2bme_from_orbits( self, oa, ob, oc, od, Jab, Jcd ):
        if(self.ms.rank <= 1): return 0
        orbits = self.ms.orbits
        a = orbits.orbit_index_from_orbit( oa )
        b = orbits.orbit_index_from_orbit( ob )
        c = orbits.orbit_index_from_orbit( oc )
        d = orbits.orbit_index_from_orbit( od )
        return self.get_2bme_from_indices( a, b, c, d, Jab, Jcd )
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
        ichbra_tmp = three.get_index(Jbra,Pbra,Tbra)
        ichket_tmp = three.get_index(Jket,Pket,Tket)
        phase = 1
        if( ichbra_tmp >= ichket_tmp ):
            ichbra = ichbra_tmp
            ichket = ichket_tmp
            i, j, k, l, m, n = a, b, c, d, e, f
            Jij, Tij, Jlm, Tlm = Jab, Tab, Jde, Tde
        else:
            print("flip")
            ichbra = ichket_tmp
            ichket = ichbra_tmp
            phase *=  (-1)**((Jket+Tket-Jbra-Tbra)//2)
            i, j, k, l, m, n = d, e, f, a, b, c
            Jij, Tij, Jlm, Tlm = Jde, Tde, Jab, Tab
        chbra = three.get_channel(ichbra)
        chket = three.get_channel(ichket)
        bra = chbra.index_from_indices[(i,j,k,Jij,Tij)]
        ket = chket.index_from_indices[(l,m,n,Jlm,Tlm)]
        return self.set_3bme_from_mat_indices(ichbra,ichket,bra,ket) * phase

    def read_operator_file(self, filename, spfile=None, opfile2=None, comment="!", istore=None, A=None):
        if(filename.find(".snt") != -1):
            self._read_operator_snt(filename, comment, A)
            if( self.count_nonzero_1bme() + self.count_nonzero_2bme() == 0):
                print("The number of non-zero operator matrix elements is 0 better to check: "+ filename + "!!")
            return
        if(filename.find(".op.me2j") != -1):
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
            if( istore == None ): self._read_lotta_format(filename,0)
            if( istore != None ): self._read_lotta_format(filename,istore)
            if( self.count_nonzero_1bme() == 0):
                print("The number of non-zero operator matrix elements is 0 better to check: "+ filename + "!!")
            return
        print("Unknown file format in " + sys._getframe().f_code.co_name )
        return

    def _triag(self,J1,J2,J3):
        b = True
        if(abs(J1-J2) <= J3 <= J1+J2): b = False
        return b

    def _read_operator_snt(self, filename, comment="!", A=None):
        f = open(filename, 'r')
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
            b = line.startswith(comment)
        data = line.split()
        norbs = int(data[0]) + int(data[1])

        b = True
        while b == True:
            x = f.tell()
            line = f.readline()
            b = line.startswith(comment)
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
            b = line.startswith(comment)
        data = line.split()
        n = int(data[0])
        method = int(data[1])
        if(A!=None): hw = float(data[2])
        fact1 = 1.0
        if(method==10 and A==None):
            print(" Need to set mass number! ")
            sys.exit()
        if(A!=None and method==10): fact1 = (1-1/float(A))*hw


        b = True
        while b == True:
            x = f.tell()
            line = f.readline()
            b = line.startswith(comment)
        f.seek(x)

        for i in range(n):
            line = f.readline()
            data = line.split()
            a, b, me = int(data[0]), int(data[1]), float(data[2])
            self.set_1bme(a,b,me*fact1)

        b = True
        while b == True:
            line = f.readline()
            b = line.startswith(comment)
        data = line.split()
        n = int(data[0])
        method = int(data[1])
        if(A!=None): hw = float(data[2])
        fact2 = 1.0
        if(method==10 and A==None):
            print(" Need to set mass number! ")
            sys.exit()
        if(A!=None and method==10): fact2 = hw/float(A)

        b = True
        while b == True:
            x = f.tell()
            line = f.readline()
            b = line.startswith(comment)
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
            #if(2*p_n + p_l != 2): continue
            orbs.add_orbit(p_n, p_l, p_j, -1)
        for line in lines[1:]:
            entry = line.split()
            n_n = int(entry[0])
            n_l = int(entry[1])
            n_j = int(entry[2])
            #if(2*n_n + n_l != 2): continue
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
            #if(2*p_n + p_l != 2): continue
            #if(2*n_n + n_l != 2): continue
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
                if( abs(data[0]) > 1.e-10 ): self.set_1bme( pi,pj,data[0] )
                if( abs(data[1]) > 1.e-10 ): self.set_1bme( ni,nj,data[1] )
                if( abs(data[2]) > 1.e-10 ): self.set_1bme( ni,pj,data[2] )
                if( abs(data[3]) > 1.e-10 ): self.set_1bme( pi,nj,data[3] )

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
                                if( abs(data[0]) > 1.e-10 ): self.set_2bme_from_indices(pi,pj,pk,pl,Jij,Jkl,data[0])
                                if( abs(data[1]) > 1.e-10 ): self.set_2bme_from_indices(pi,pj,pk,nl,Jij,Jkl,data[1])
                                if( abs(data[2]) > 1.e-10 ): self.set_2bme_from_indices(pi,pj,nk,pl,Jij,Jkl,data[2])
                                if( abs(data[3]) > 1.e-10 ): self.set_2bme_from_indices(pi,pj,nk,nl,Jij,Jkl,data[3])
                                if( abs(data[4]) > 1.e-10 ): self.set_2bme_from_indices(pi,nj,pk,nl,Jij,Jkl,data[4])
                                if( abs(data[5]) > 1.e-10 ): self.set_2bme_from_indices(pi,nj,nk,pl,Jij,Jkl,data[5])
                                if( abs(data[6]) > 1.e-10 ): self.set_2bme_from_indices(pi,nj,nk,nl,Jij,Jkl,data[6])
                                if( abs(data[7]) > 1.e-10 ): self.set_2bme_from_indices(ni,pj,nk,pl,Jij,Jkl,data[7])
                                if( abs(data[8]) > 1.e-10 ): self.set_2bme_from_indices(ni,pj,nk,nl,Jij,Jkl,data[8])
                                if( abs(data[9]) > 1.e-10 ): self.set_2bme_from_indices(ni,nj,nk,nl,Jij,Jkl,data[9])
        f.close()
    def _read_3b_operator_readabletxt(self, filename, comment="!"):
        if( len( self.ms.three.channels ) == 0 ):
            ms = ModelSpace()
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

    def write_operator_file(self, filename):
        if(filename.find(".snt") != -1):
            self._write_operator_snt( filename )
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
            out += "{0:3d} {1:16.10f}".format(Jab, self.two[key] / np.sqrt(2*Jab+1)) + "\n"
        f=open("nme.dat","w")
        f.write(out)
        f.close()

    def _write_general_operator(self, filename):
        f = open(filename, "w")
        f.write(" Written by python script \n")
        f.write(" {:3d} {:3d} {:3d} {:3d} {:3d}\n".format( self.rankJ, self.rankP, self.rankZ, self.ms.emax, self.ms.e2max ))
        f.write("{:14.8f}\n".format( self.zero ) )

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
                f.write("{:14.8f} {:14.8f} {:14.8f} {:14.8f}\n".format(\
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
                                f.write("{:14.8f} {:14.8f} {:14.8f} {:14.8f} {:14.8f} {:14.8f} {:14.8f} {:14.8f} {:14.8f} {:14.8f} \n".format(\
                                        me_pppp, me_pppn, me_ppnp, me_ppnn, me_pnpn, me_pnnp, me_pnnn, me_npnp, me_npnn, me_nnnn))
        f.close()

    def _write_operator_snt(self, filename):
        orbits = self.ms.orbits
        p_norbs = 0; n_norbs = 0
        for o in orbits.orbits:
            if( o.z ==-1 ): p_norbs += 1
            if( o.z == 1 ): n_norbs += 1
        prt = ""
        prt  += " {:3d} {:3d} {:3d}\n".format( self.rankJ, self.rankP, self.rankZ )
        prt += "! model space \n"
        prt += " {0:3d} {1:3d} {2:3d} {3:3d} \n".format( p_norbs, n_norbs, 0, 0 )
        norbs = orbits.get_num_orbits()+1
        for i in range(1,norbs):
            o = orbits.get_orbit(i)
            prt += "{0:5d} {1:3d} {2:3d} {3:3d} {4:3d} \n".format( i, o.n, o.l, o.j, o.z )

        norbs = orbits.get_num_orbits()+1
        prt += "! one-body part\n"
        prt += "{0:5d} {1:3d}\n".format( self.count_nonzero_1bme(), 0 )
        for i in range(1,norbs):
            for j in range(1,norbs):
                me = self.get_1bme(i,j)
                if( abs(me) < 1.e-10): continue
                prt += "{0:3d} {1:3d} {2:15.8f}\n".format( i, j, me )
        if( self.ms.rank==1 ):
            prt += "! two-body part\n"
            prt += "{0:10d} {1:3d}\n".format( 0, 0 )
            f = open(filename, "w")
            f.write(prt)
            f.close()
            return
        prt += "! two-body part\n"
        prt += "{0:10d} {1:3d}\n".format( self.count_nonzero_2bme(), 0 )
        scalar = False
        if(self.rankJ == 0 and self.rankZ == 0): scalar = True
        two = self.ms.two
        for ichbra in range(two.get_number_channels()):
            chbra = two.get_channel(ichbra)
            for ichket in range(ichbra+1):
                chket = two.get_channel(ichket)
                if( self._triag( chbra.J, chket.J, self.rankJ )): continue
                if( chbra.P * chket.P * self.rankP != 1): continue
                if( abs(chbra.Z-chket.Z) != self.rankZ): continue
                for bra, ket in self.two[(ichbra,ichket)].keys():
                    a = chbra.orbit1_index[bra]
                    b = chbra.orbit2_index[bra]
                    c = chket.orbit1_index[ket]
                    d = chket.orbit2_index[ket]
                    if(scalar):
                        prt += "{0:3d} {1:3d} {2:3d} {3:3d} {4:3d} {5:15.8f}\n".format( a, b, c, d, chket.J, self.two[(ichbra,ichket)][(bra,ket)])
                    else:
                        prt += "{0:3d} {1:3d} {2:3d} {3:3d} {4:3d} {5:3d} {6:15.8f}\n".format( a, b, c, d, chbra.J, chket.J, self.two[(ichbra,ichket)][(bra,ket)])
        f = open(filename, "w")
        f.write(prt)
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
        if(b==d): me += self.get_1bme(a,c) * (-1.0)**( (oa.j+ob.j)//2 + Jcd     ) * N( wigner_6j(Jab,Jcd,lam,oc.j*0.5,oa.j*0.5,ob.j*0.5) )
        if(a==c): me += self.get_1bme(b,d) * (-1.0)**( (oc.j+od.j)//2 - Jab     ) * N( wigner_6j(Jab,Jcd,lam,od.j*0.5,ob.j*0.5,oa.j*0.5) )
        if(b==c): me -= self.get_1bme(a,d) * (-1.0)**( (oa.j+ob.j+oc.j+od.j)//2 ) * N( wigner_6j(Jab,Jcd,lam,od.j*0.5,oa.j*0.5,ob.j*0.5) )
        if(a==d): me -= self.get_1bme(b,c) * (-1.0)**( Jcd - Jab                ) * N( wigner_6j(Jab,Jcd,lam,oc.j*0.5,ob.j*0.5,oa.j*0.5) )
        me *= np.sqrt( (2*Jab+1)*(2*Jcd+1) ) * (-1.0)**lam
        if(a==b): me /= np.sqrt(2.0)
        if(c==d): me /= np.sqrt(2.0)
        return me
    def spin_tensor_decomposition(self):
        if(self.rankJ != 0):
            print("Spin-tensor decomposition is not defined for a non-scalar operator")
            return None
        ops = []
        ops.append( Operator( rankJ=self.rankJ, rankP=self.rankP, rankZ=self.rankZ, ms=self.ms ) )
        ops.append( Operator( rankJ=self.rankJ, rankP=self.rankP, rankZ=self.rankZ, ms=self.ms ) )
        ops.append( Operator( rankJ=self.rankJ, rankP=self.rankP, rankZ=self.rankZ, ms=self.ms ) )
        ms = self.ms.two
        orbits = ms.orbits
        for ch_key in self.two.keys():
            ichbra = ch_key[0]
            ichket = ch_key[1]
            chbra = ms.get_channel(ichbra)
            chket = ms.get_channel(ichket)
            J = chket.J
            for key in self.two[ch_key].keys():
                a = chbra.orbit1_index[key[0]]
                b = chbra.orbit2_index[key[0]]
                c = chket.orbit1_index[key[1]]
                d = chket.orbit2_index[key[1]]
                oa = orbits.get_orbit(a)
                ob = orbits.get_orbit(b)
                oc = orbits.get_orbit(c)
                od = orbits.get_orbit(d)

                for rank in [0,1,2]:
                    sum3 = 0.0
                    for Lab in range( abs(oa.l-ob.l), oa.l+ob.l+1 ):
                        for Sab in [0,1]:
                            if(self._triag( Lab, Sab, J )): continue
                            Cab = _ls_coupling(oa.l, oa.j*0.5, ob.l, ob.j*0.5, Lab, Sab, J)
                            if(abs(Cab) < 1.e-10): continue
                            for Lcd in range( abs(oc.l-od.l), oc.l+od.l+1 ):
                                for Scd in [0,1]:
                                    if(self._triag( Lcd, Scd, J )): continue
                                    Ccd = _ls_coupling(oc.l, oc.j*0.5, od.l, od.j*0.5, Lcd, Scd, J)
                                    if(abs(Ccd) < 1.e-10): continue
                                    SixJ = N(wigner_6j(Lab,Sab,J,Scd,Lcd,rank))
                                    if(abs(SixJ) < 1.e-10): continue

                                    sum2 = 0.0
                                    for JJ in range( max(abs(Lab-Sab),abs(Lcd-Scd)), min(Lab+Sab, Lcd+Scd)+1):
                                        SixJJ = N(wigner_6j(Lab,Sab,JJ,Scd,Lcd,rank))
                                        if(abs(SixJJ) < 1.e-10): continue
                                        sum1 = 0.0
                                        for jaa in [abs(2*oa.l-1), 2*oa.l+1]:
                                            try:
                                                aa = orbits.get_orbit_index(oa.n, oa.l, jaa, oa.z)
                                            except:
                                                continue
                                            for jbb in [abs(2*ob.l-1), 2*ob.l+1]:
                                                try:
                                                    bb = orbits.get_orbit_index(ob.n, ob.l, jbb, ob.z)
                                                except:
                                                    continue
                                                CCab = _ls_coupling(oa.l, jaa*0.5, ob.l, jbb*0.5, Lab, Sab, JJ)
                                                for jcc in [abs(2*oc.l-1), 2*oc.l+1]:
                                                    try:
                                                        cc = orbits.get_orbit_index(oc.n, oc.l, jcc, oc.z)
                                                    except:
                                                        continue
                                                    for jdd in [abs(2*od.l-1), 2*od.l+1]:
                                                        try:
                                                            dd = orbits.get_orbit_index(od.n, od.l, jdd, od.z)
                                                        except:
                                                            continue
                                                        CCcd = _ls_coupling(oc.l, jcc*0.5, od.l, jdd*0.5, Lcd, Scd, JJ)
                                                        sum1 += self.get_2bme_from_indices(aa,bb,cc,dd,JJ,JJ) * CCab * CCcd
                                        sum2 += sum1 * SixJJ * (2*JJ+1)*(-1)**JJ
                                    sum3 += sum2 * SixJ * Cab * Ccd
                    ops[rank].set_2bme_from_indices(a,b,c,d,J,J, (-1)**J*(2*rank+1)*sum3)
                sys.exit()
        return ops



def main():
    ms = ModelSpace.ModelSpace()
    ms.set_modelspace_from_boundaries(4)
    op = Operator(verbose=False)
    op.allocate_operator(ms)
    op.print_operator()
if(__name__=="__main__"):
    main()

