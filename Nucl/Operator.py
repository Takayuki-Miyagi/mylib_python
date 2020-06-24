#!/usr/bin/env python3
import sys
import numpy as np
import copy
import gzip
if(__package__==None or __package__==""):
    import ModelSpace
else:
    from . import Orbits, OrbitsIsospin
    from . import ModelSpace

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
                if( self._triag( chbra.J, chket.J, self.rankJ )): continue
                if( chbra.P * chket.P * self.rankP != 1): continue
                if( self._triag( chbra.T, chket.T, self.rankT )): continue
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
        Pcd = (-1)**(oa.l+ob.l)
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

    def read_operator_file(self, filename, spfile=None, opfile2=None, comment="!", istore=None):
        if(filename.find(".snt") != -1):
            self._read_operator_snt(filename, comment)
            return
        if(filename.find(".op.me2j") != -1):
            self._read_general_operator(filename, comment)
            return
        if(filename.find(".navratil") != -1):
            self._read_general_operator_navratil(filename, comment)
            return
        if(filename.find(".int") != -1):
            if(spfile == None):
                print("No sp file!"); return
            import nushell2snt
            if( rankJ==0 and rankP==1 and rankZ==0 ):
                nushell2snt.scalar( spfile, filename, "tmp.snt" )
            else:
                if(opfile2 == None):
                    print("No op2 file!"); return
                nushell2snt.tensor( spfile, filename, opfile2, "tmp.snt" )
            self._read_operator_snt("tmp.snt", "!")
            subprocess.call("rm tmp.snt", shell=True)
            return
        if(filename.find(".lotta") != -1):
            if( istore == None ): self._read_lotta_format(filename,0)
            if( istore != None ): self._read_lotta_format(filename,istore)
            return
        print("Unknown file format in " + sys._getframe().f_code.co_name )
        return

    def _triag(self,J1,J2,J3):
        b = True
        if(abs(J1-J2) <= J3 <= J1+J2): b = False
        return b

    def _read_operator_snt(self, filename, comment="!"):
        f = open(filename, 'r')
        line = f.readline()
        b = True
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
            self.set_1bme(a,b,me)

        b = True
        while b == True:
            line = f.readline()
            b = line.startswith(comment)
        data = line.split()
        n = int(data[0])

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
            self.set_2bme_from_indices(a,b,c,d,Jab,Jcd,me)
        f.close()

    def _read_lotta_format(self, filename, ime ):
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
            exist = False
            orbs.add_orbit(p_n, p_l, p_j, -1)
        for line in lines[1:]:
            entry = line.split()
            n_n = int(entry[0])
            n_l = int(entry[1])
            n_j = int(entry[2])
            exist = False
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
        E = 0
        orbits = self.ms.orbits
        norbs = orbits.get_num_orbits() + 1
        for i in range(1,norbs):
            o = orbits.get_orbit(i)
            E = max(E, 2*o.n+o.l)
        f.write(" {:3d} {:3d} {:3d} {:3d} {:3d}\n".format( self.rankJ, self.rankP, self.rankZ, E, 2*E ))
        f.write("{:14.8f}\n".format( self.zero ) )

        iorbits = OrbitsIsospin( emax=E )
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

                for k in range(1,norbs):
                    ok = iorbits.get_orbit(k)
                    pk = orbits.get_orbit_index(ok.n, ok.l, ok.j, -1)
                    nk = orbits.get_orbit_index(ok.n, ok.l, ok.j,  1)
                    for l in range(1,k+1):
                        ol = iorbits.get_orbit(k)
                        pl = orbits.get_orbit_index(ol.n, ol.l, ol.j, -1)
                        nl= orbits.get_orbit_index(ol.n, ol.l, ol.j,  1)
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

        norbs = orbits.get_num_orbits()
        prt += "! one-body part\n"
        prt += "{0:5d} {1:3d}\n".format( self.count_nonzero_1bme(), 0 )
        for i in range(1,norbs):
            for j in range(1,norbs):
                me = self.get_1bme(i,j)
                if( abs(me) < 1.e-10): continue
                prt += "{0:3d} {1:3d} {2:15.8f}\n".format( i, j, me )

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

def main():
    ms = ModelSpace.ModelSpace()
    ms.set_modelspace_from_boundaries(4)
    op = Operator(verbose=False)
    op.allocate_operator(ms)
    op.print_operator()
if(__name__=="__main__"):
    main()

