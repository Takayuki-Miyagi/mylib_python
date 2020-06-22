#!/usr/bin/env python3
import sys
import numpy as np
import copy
if(__package__==None):
    import ModelSpace
else:
    from . import Orbits
    from . import ModelSpace

class Operator:
    def __init__(self, rankJ=0, rankP=1, rankZ=0, ms=None, reduced=False):
        self.ms = ms
        self.rankJ = rankJ
        self.rankP = rankP
        self.rankZ = rankZ
        self.reduced = reduced
        if( rankJ != 0 ): self.reduced = True
        if(ms == None):
            self.zero = 0.0
            self.one = None
            self.two = {}
            self.three = {}
            return

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
                #self.two[(ichbra,ichket)] = np.zeros( (chbra.get_number_states(), chket.get_number_states()) )
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
                #self.three[(ichbra,ichket)] = np.zeros( chbra.get_number_states(), chket.get_number_states() )
                self.three[(ichbra,ichket)] = {}
    def set_0bme( self, me ):
        self.zero = me
    def set_1bme( self, a, b, me):
        orbits = self.ms.orbits
        oa = orbits.get_orbit(a)
        ob = orbits.get_orbit(b)
        if( self._triag(oa.j, ob.j, 2*self.rankJ)):
            print("Warning: J, " + sys._getframe().f_code.co_name )
            return
        if( (-1)**(oa.l+ob.l) * self.rankP != 1):
            print("Warning: Parity, " + sys._getframe().f_code.co_name )
            return
        if( abs(oa.z-ob.z) != 2*self.rankZ):
            print("Warning: Z, " + sys._getframe().f_code.co_name )
            return
        self.one[a-1,b-1] = me
        self.one[b-1,a-1] = me * (-1)**( (ob.j-oa.j)//2 )
    def set_2bme_from_mat_indices( self, chbra, chket, bra, ket, me ):
        if( chbra < chket ):
            print("Warning:" + sys._getframe().f_code.co_name )
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
        Pcd = (-1)**(oa.l+ob.l)
        Zab = (oa.z + ob.z)//2
        Zcd = (oc.z + od.z)//2
        if( self._triag( Jab, Jcd, self.rankJ )):
            print("Warning: J, " + sys._getframe().f_code.co_name )
            return
        if( Pab * Pcd * self.rankP != 1):
            print("Warning: Parity, " + sys._getframe().f_code.co_name )
            return
        if( abs(Zab-Zcd) != self.rankZ):
            print("Warning: Z, " + sys._getframe().f_code.co_name )
            return
        ichbra_tmp = two.get_index(Jab,Pab,Zab)
        ichket_tmp = two.get_index(Jcd,Pcd,Zcd)
        phase = 1
        if( ichbra_tmp >= ichket_tmp ):
            ichbra = ichbra_tmp
            ichket = ichket_tmp
        else:
            ichbra = ichket_tmp
            ichket = ichbra_tmp
            phase *=  (-1)**(Jcd-Jab)
        chbra = two.get_channel(ichbra)
        chket = two.get_channel(ichket)
        bra = chbra.index_from_indices[(a,b)]
        ket = chbra.index_from_indices[(c,d)]
        phase *= chbra.phase_from_indices[(a,b)] * chket.phase_from_indices[(c,d)]
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
            print("Warning:" + sys._getframe().f_code.co_name )
            return 0
        try:
            return self.two[(chbra,chket)][(bra,ket)]
        except:
            print("Nothing here" + sys._getframe().f_code.co_name )
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
            print("Warning: J, " + sys._getframe().f_code.co_name )
            return 0
        if( Pab * Pcd * self.rankP != 1):
            print("Warning: Parity, " + sys._getframe().f_code.co_name )
            return 0
        if( abs(Zab-Zcd) != self.rankZ):
            print("Warning: Z, " + sys._getframe().f_code.co_name )
            return 0
        ichbra_tmp = two.get_index(Jab,Pab,Zab)
        ichket_tmp = two.get_index(Jcd,Pcd,Zcd)
        phase = 1
        if( ichbra_tmp >= ichket_tmp ):
            ichbra = ichbra_tmp
            ichket = ichket_tmp
        else:
            ichbra = ichket_tmp
            ichket = ichbra_tmp
            phase *=  (-1)**(Jcd-Jab)
        chbra = two.get_channel(ichbra)
        chket = two.get_channel(ichket)
        bra = chbra.index_from_indices[(a,b)]
        ket = chbra.index_from_indices[(c,d)]
        phase *= chbra.phase_from_indices[(a,b)] * chket.phase_from_indices[(c,d)]
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
        print("In Op.py: Unknown file format")
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

        orbs = Orbits.Orbits()
        for i in range(norbs):
            line = f.readline()
            data = line.split()
            idx, n, l, j, z = int(data[0]), int(data[1]), int(data[2]), int(data[3]), int(data[4])
            orbs.add_orbit(n,l,j,z)
        ms = ModelSpace.ModelSpace()
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
        orbs = Orbits.Orbits(verbose=False)
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
        ms = ModelSpace.ModelSpace(rank=1)
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
            for b in range(a, orbits.get_num_orbits()+1):

                for c in range(1, orbits.get_num_orbits()+1):
                    for d in range(a, orbits.get_num_orbits()+1):
                        oa = orbits.get_orbit(a)
                        ob = orbits.get_orbit(b)
                        oc = orbits.get_orbit(c)
                        od = orbits.get_orbit(d)
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
    ms.set_modelspace_from_boundaries(2)
    ms.print_modelspace_summary()
if(__name__=="__main__"):
    main()

