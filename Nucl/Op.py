#!/usr/bin/env python3
import numpy as np
from . import Orbit
class Op:
    def __init__(self, file_op=None, file_sp=None, file_op2=None, rankJ=0, rankP=1, rankZ=0, file_format="snt"):
        self.n_porb = 0
        self.n_norb = 0
        self.zcore = 0
        self.ncore = 0
        self.zero = 0.0
        self.one = {}
        self.two = {}
        self.orbs = []
        self.rankJ = rankJ
        self.rankP = rankP
        self.rankZ = rankZ
        self.file_op = file_op
        self.file_op2 = file_op2
        self.file_sp = file_sp
        self.file_format = file_format

    def set_orbits(self, orbs):
        self.orbs = orbs

    def set_zero(self,me):
        self.zero = me

    def set_obme(self,a,b,me):
        ab = (a,b)
        self.one[ab] = me

    def set_tbme(self,a,b,c,d,Jab,Jcd,me):
        abcd = (a,b,c,d,Jab,Jcd)
        self.two[abcd] = me

    def get_zero(self):
        return self.zero

    def get_obme(self,a,b):
        oa = self.orbs.get_orbit(a)
        ob = self.orbs.get_orbit(b)
        if(self._triag(oa.j,ob.j,2*self.rankJ)): return 0.0
        if(abs(oa.z-ob.z) != 2*self.rankZ): return 0.0
        if((-1)**(oa.l+ob.l) * self.rankP != 1): return 0.0
        ex_bk = False
        if((a,b) in self.one): v = self.one[(a,b)]
        if((b,a) in self.one): v = self.one[(b,a)]; ex_bk = True
        fact = 1.0
        if(ex_bk):
            fact *= (-1.0) ** ((ob.j - oa.j)/2)
        try:
            return v * fact
        except:
            #print("Warning: not found one-body matrix element: (a,b) = ", a, b)
            return 0.0

    def _get_phase(self,a,b,Jab):
        oa = self.orbs.get_orbit(a)
        ob = self.orbs.get_orbit(b)
        return -(-1.0) ** ((oa.j + ob.j)/2 - Jab)

    def _triag(self,J1,J2,J3):
        b = True
        if(abs(J1-J2) <= J3 <= J1+J2): b = False
        return b

    def get_tbme(self,a,b,c,d,Jab,Jcd):
        oa = self.orbs.get_orbit(a)
        ob = self.orbs.get_orbit(b)
        oc = self.orbs.get_orbit(c)
        od = self.orbs.get_orbit(d)
        if(self._triag(Jab,Jcd,self.rankJ)):
            #print("warining: J")
            return 0.0
        if(abs(oa.z+ob.z-oc.z-od.z) != 2*self.rankZ):
            #print("warining: Z")
            return 0.0
        if((-1)**(oa.l + ob.l + oc.l + od.l) * self.rankP != 1):
            #print("warining: P")
            return 0.0
        ex_ab, ex_cd, ex_bk = False, False, False
        if((a,b,c,d,Jab,Jcd) in self.two): v = self.two[(a,b,c,d,Jab,Jcd)]
        if((c,d,a,b,Jcd,Jab) in self.two): v = self.two[(c,d,a,b,Jcd,Jab)]; ex_bk = True

        if((b,a,c,d,Jab,Jcd) in self.two): v = self.two[(b,a,c,d,Jab,Jcd)]; ex_ab = True
        if((c,d,b,a,Jcd,Jab) in self.two): v = self.two[(c,d,b,a,Jcd,Jab)]; ex_ab = True; ex_bk = True

        if((a,b,d,c,Jab,Jcd) in self.two): v = self.two[(a,b,d,c,Jab,Jcd)]; ex_cd = True
        if((d,c,a,b,Jcd,Jab) in self.two): v = self.two[(d,c,a,b,Jcd,Jab)]; ex_cd = True; ex_bk = True

        if((b,a,d,c,Jab,Jcd) in self.two): v = self.two[(b,a,d,c,Jab,Jcd)]; ex_ab = True; ex_cd = True
        if((d,c,b,a,Jcd,Jab) in self.two): v = self.two[(d,c,b,a,Jcd,Jab)]; ex_ab = True; ex_cd = True; ex_bk = True

        fact = 1.0
        if(ex_bk): fact *= (-1.0) ** (Jcd-Jab)
        if(ex_ab): fact *= self._get_phase(a,b,Jab)
        if(ex_cd): fact *= self._get_phase(c,d,Jcd)
        try:
            return v * fact
        except:
            #print("Warning: not found two-body matrix element, (a,b,c,d,Jab,Jcd)=",a,b,c,d,Jab,Jcd)
            return 0.0

    def read_operator_file(self, comment="!", istore=None):
        if(self.file_format == "snt"):
            self._read_operator_snt(self.file_op, comment)
            return
        if(self.file_format == "int"):
            if(self.file_sp == None):
                print("No sp file!"); return
            if(self.file_op == None):
                print("No op file!"); return
            import nushell2snt
            if( rankJ==0 and rankP==1 and rankZ==0 ):
                nushell2snt.scalar( self.file_sp, self.file_op, "tmp.snt" )
            else:
                if(self.file_op == None):
                    print("No op2 file!"); return
                nushell2snt.tensor( self.file_sp, self.file_op, self.file_op2, "tmp.snt" )
            self._read_operator_snt("tmp.snt", "!")
            subprocess.call("rm tmp.snt", shell=True)
            return
        if(self.file_format == "lotta"):
            if( istore == None ): self._read_lotta_format(self.file_op,0)
            if( istore != None ): self._read_lotta_format(self.file_op,istore)
            return
        print("In Op.py: Unknown file format")
        return

    def _read_operator_snt(self, filename, comment="!"):
        orbs = Orbit.Orbits()
        f = open(filename, 'r')
        line = f.readline()
        b = True
        while b == True:
            line = f.readline()
            if(line.find("zero body") != -1 or \
                    line.find("Zero body") != -1 or \
                    line.find("Zero Body") != -1):
                data = line.split()
                self.zero = float(data[4])
            if(line.find("zero-body") != -1 or \
                    line.find("Zero-body") != -1 or \
                    line.find("Zero-Body") != -1):
                data = line.split()
                self.zero = float(data[3])
            b = line.startswith(comment)
        data = line.split()
        norbs = int(data[0]) + int(data[1])
        self.n_porb = int(data[0])
        self.n_norb = int(data[0])
        self.zcore = int(data[2])
        self.ncore = int(data[3])

        b = True
        while b == True:
            x = f.tell()
            line = f.readline()
            b = line.startswith(comment)
        f.seek(x)

        for i in range(norbs):
            line = f.readline()
            data = line.split()
            idx, n, l, j, z = int(data[0]), int(data[1]), int(data[2]), int(data[3]), int(data[4])
            orbs.add_orbit(n,l,j,z,idx)
        self.set_orbits(orbs)

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
            self.set_obme(a,b,me)

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
            if(self.rankJ == 0 and self.rankZ == 0):
                a, b, c, d = int(data[0]), int(data[1]), int(data[2]), int(data[3])
                Jab, me = int(data[4]), float(data[5])
                Jcd = Jab
            else:
                a, b, c, d = int(data[0]), int(data[1]), int(data[2]), int(data[3])
                Jab, Jcd, me = int(data[4]), int(data[5]), float(data[6])
            self.set_tbme(a,b,c,d,Jab,Jcd,me)
        f.close()

    def _read_lotta_format(self, filename, ime ):
        orbs = Orbit.Orbits()
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
            for key in orbs.idx_orb.keys():
                o = orbs.get_orbit(key)
                if( o.n == p_n and o.l == p_l and o.j == p_j and o.z == -1 ):
                    exist=True; break
            if( not exist ):
                idx += 1
                orbs.add_orbit(p_n, p_l, p_j, -1, idx)
        self.n_porb = idx
        for line in lines[1:]:
            entry = line.split()
            n_n = int(entry[0])
            n_l = int(entry[1])
            n_j = int(entry[2])
            exist = False
            for key in orbs.idx_orb.keys():
                o = orbs.get_orbit(key)
                if( o.n == n_n and o.l == n_l and o.j == n_j and o.z == 1 ):
                    exist=True; break
            if( not exist ):
                idx += 1
                orbs.add_orbit(n_n, n_l, n_j, 1, idx)
        norbs = idx
        self.n_norb = norbs - self.n_porb
        self.set_orbits(orbs)
        for line in lines[1:]:
            entry = line.split()
            n_n = int(entry[0])
            n_l = int(entry[1])
            n_j = int(entry[2])
            p_n = int(entry[3])
            p_l = int(entry[4])
            p_j = int(entry[5])
            mes = [ float(entry[i+6]) for i in range(len(entry)-6) ]
            i = self.orbs.nljz_idx[(n_n,n_l,n_j, 1)]
            j = self.orbs.nljz_idx[(p_n,p_l,p_j,-1)]
            me = mes[ime]
            if( abs(me) < 1.e-8): continue
            self.set_obme( i, j, me )
            self.set_obme( j, i, me*(-1.0)**( ( p_j-n_j )/2 ) )

    def write_nme_file(self):
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

    def write_operator_file(self, filename):
        if(filename.find(".snt") != -1):
            self._write_operator_snt( filename )
        if(filename.find(".op.me2j") != -1):
            self._write_general_operator( filename )

    def _write_general_operator(self, filename):
        prt = ""
        prt += " Written by python script \n"
        E = 0
        for orb in self.orbs.idx_orb:
            orbit = self.orbs.idx_orb[orb]
            E = max(E, 2*orbit.n+orbit.l)
        prt += " {:3d} {:3d} {:3d} {:3d} {:3d}\n".format( self.rankJ, self.rankP, self.rankZ, E, 2*E )
        prt += "{:14.8f}\n".format( self.zero )

        nlj = []
        for N in range(E+1):
            for l in range(N+1):
                if( (N-l)%2==1 ): continue
                n = (N-l)//2
                for j in [2*l-1, 2*l+1]:
                    if(j < 0): continue
                    nlj.append( (n, l, j) )
        for i in range(len(nlj)):
            nlj1 = nlj[i]
            try:
                pi = self.orbs.nljz_idx[ (nlj1[0], nlj1[1], nlj1[2], -1) ]
                ni = self.orbs.nljz_idx[ (nlj1[0], nlj1[1], nlj1[2],  1) ]
            except:
                pi = -1
                ni = -1
            for j in range(len(nlj)):
                nlj2 = nlj[j]
                try:
                    pj = self.orbs.nljz_idx[ (nlj2[0], nlj2[1], nlj2[2], -1) ]
                    nj = self.orbs.nljz_idx[ (nlj2[0], nlj2[1], nlj2[2],  1) ]
                except:
                    pj = -1
                    nj = -1
                if( (-1)**(nlj1[1]+nlj2[1]) * self.rankP != 1): continue
                if( self._triag(nlj1[2], nlj2[2], 2*self.rankJ ) ): continue
                me_pp = 0.0; me_nn = 0.0; me_np = 0.0; me_pn = 0.0
                if(pi>=0 and pj>=0 ):
                    me_pp = self.get_obme(pi,pj)
                    me_nn = self.get_obme(ni,nj)
                    me_np = self.get_obme(ni,pj)
                    me_pn = self.get_obme(pi,nj)
                prt += "{:14.8f} {:14.8f} {:14.8f} {:14.8f}\n".format(\
                        me_pp, me_nn, me_np, me_pn )

        for i in range(len(nlj)):
            nlj1 = nlj[i]
            try:
                pi = self.orbs.nljz_idx[ (nlj1[0], nlj1[1], nlj1[2], -1) ]
                ni = self.orbs.nljz_idx[ (nlj1[0], nlj1[1], nlj1[2],  1) ]
            except:
                pi = -1
                ni = -1
            for j in range(i+1):
                nlj2 = nlj[j]
                try:
                    pj = self.orbs.nljz_idx[ (nlj2[0], nlj2[1], nlj2[2], -1) ]
                    nj = self.orbs.nljz_idx[ (nlj2[0], nlj2[1], nlj2[2],  1) ]
                except:
                    pj = -1
                    nj = -1

                for k in range(len(nlj)):
                    nlj3 = nlj[k]
                    try:
                        pk = self.orbs.nljz_idx[ (nlj3[0], nlj3[1], nlj3[2], -1) ]
                        nk = self.orbs.nljz_idx[ (nlj3[0], nlj3[1], nlj3[2],  1) ]
                    except:
                        pk = -1
                        nk = -1
                    for l in range(k+1):
                        nlj4 = nlj[l]
                        try:
                            pl = self.orbs.nljz_idx[ (nlj4[0], nlj4[1], nlj4[2], -1) ]
                            nl = self.orbs.nljz_idx[ (nlj4[0], nlj4[1], nlj4[2],  1) ]
                        except:
                            pl = -1
                            nl = -1
                        if( (-1)**( nlj1[1]+nlj2[1]+nlj3[1]+nlj4[1] ) * self.rankP != 1): continue
                        for Jij in range( int(abs( nlj1[2]-nlj2[2] ))//2, (nlj1[2]+nlj2[2])//2 ):
                            for Jkl in range( int(abs( nlj3[2]-nlj4[2] ))//2, (nlj3[2]+nlj4[2])//2 ):
                                if( self._triag(Jij, Jkl, self.rankJ ) ): continue

                                me_pppp = 0.0; me_pppn = 0.0; me_ppnp = 0.0; me_ppnn = 0.0; me_pnpn = 0.0
                                me_pnnp = 0.0; me_pnnn = 0.0; me_npnp = 0.0; me_npnn = 0.0; me_nnnn = 0.0
                                if(pi>=0 and pj>=0 and pk>=0 and pl>= 0):
                                    me_pppp = self.get_tbme(pi,pj,pk,pl,Jij,Jkl)
                                    me_pppn = self.get_tbme(pi,pj,pk,nl,Jij,Jkl)
                                    me_ppnp = self.get_tbme(pi,pj,nk,pl,Jij,Jkl)
                                    me_ppnn = self.get_tbme(pi,pj,nk,nl,Jij,Jkl)
                                    me_pnpn = self.get_tbme(pi,nj,pk,nl,Jij,Jkl)
                                    me_pnnp = self.get_tbme(pi,nj,nk,pl,Jij,Jkl)
                                    me_pnnn = self.get_tbme(pi,nj,nk,nl,Jij,Jkl)
                                    me_npnp = self.get_tbme(ni,pj,nk,pl,Jij,Jkl)
                                    me_npnn = self.get_tbme(ni,pj,nk,nl,Jij,Jkl)
                                    me_nnnn = self.get_tbme(ni,nj,nk,nl,Jij,Jkl)
                                prt += "{:14.8f} {:14.8f} {:14.8f} {:14.8f} {:14.8f} {:14.8f} {:14.8f} {:14.8f} {:14.8f} {:14.8f} \n".format(\
                                        me_pppp, me_pppn, me_ppnp, me_ppnn, me_pnpn, me_pnnp, me_pnnn, me_npnp, me_npnn, me_nnnn)
        f = open(filename, "w")
        f.write(prt)
        f.close()


    def _write_operator_snt(self, filename):
        prt = ""
        pr  += " {:3d} {:3d} {:3d}".format( self.rankJ, self.rankP, self.rankZ )
        prt += "! model space \n"
        prt += " {0:3d} {1:3d} {2:3d} {3:3d} \n".format( self.n_porb, self.n_norb, self.zcore, self.ncore )
        for i in range(1, 1+self.n_porb+self.n_norb):
            o = self.orbs.get_orbit(i)
            prt += "{0:5d} {1:3d} {2:3d} {3:3d} {4:3d}\n".format( i, o.n, o.l, o.j, o.z )
        prt += "! one-body part\n"
        prt += "{0:5d} {1:3d}\n".format( len(self.one), 0 )
        for key in self.one.keys():
            prt += "{0:3d} {1:3d} {2:15.8f}\n".format( key[0], key[1], self.one[key] )
        prt += "! two-body part\n"
        prt += "{0:10d} {1:3d}\n".format( len(self.two), 0 )
        scalar = False
        if(self.rankJ == 0 and self.rankZ == 0): scalar = True
        for key in self.two.keys():
            if(scalar):
                prt += "{0:3d} {1:3d} {2:3d} {3:3d} {4:3d} {5:15.8f}\n".format( key[0], key[1], key[2], key[3], key[4], self.two[key])
            else:
                prt += "{0:3d} {1:3d} {2:3d} {3:3d} {4:3d} {5:3d} {6:15.8f}\n".format( key[0], key[1], key[2], key[3], key[4], key[5], self.two[key])
        f = open(filename, "w")
        f.write(prt)
        f.close()

    def PrintOperator(self):
        print("Print Operator")
        print("zero-body term "+str(self.zero))
        print("one-body term:")
        print("  a   b       MtxElm")
        for a in range(1, self.orbs.norbs+1):
            for b in range(a, self.orbs.norbs+1):
                me = self.get_obme(a,b)
                if(abs(me) < 1e-8): continue
                print("{0:3d} {1:3d} {2:12.6f}".format(a,b,me))
        print("two-body term:")
        print("  a   b   c   d Jab Jcd       MtxElm")
        for a in range(1, self.orbs.norbs+1):
            for b in range(a, self.orbs.norbs+1):

                for c in range(1, self.orbs.norbs+1):
                    for d in range(a, self.orbs.norbs+1):
                        oa = self.orbs.get_orbit(a)
                        ob = self.orbs.get_orbit(b)
                        oc = self.orbs.get_orbit(c)
                        od = self.orbs.get_orbit(d)

                        if((-1)**(oa.l+ob.l+oc.l+od.l) * self.rankP != 1): continue

                        for Jab in range( int(abs(oa.j-ob.j)/2), int((oa.j+ob.j)/2)+1):
                            if(a == b and Jab%2 == 1): continue
                            for Jcd in range( int(abs(oc.j-od.j)/2), int((oc.j+od.j)/2+1)):
                                if(c == d and Jcd%2 == 1): continue

                                if(self._triag(Jab,Jcd,self.rankJ)): continue

                                me = self.get_tbme(a,b,c,d,Jab,Jcd)
                                if(abs(me) < 1e-8): continue
                                print("{0:3d} {1:3d} {2:3d} {3:3d} {4:3d} {5:3d} {6:12.6f}".format(a,b,c,d,Jab,Jcd,me))

def main():
    file_op="snt-file"
    rankJ = 0
    rankP = 1
    rankZ = 0
    Op = NuOps(file_op, rankJ, rankP, rankZ)
    Op.read_operator_file(comment="#")
    Op.PrintOperator()

if(__name__=='__main__'):
    main()
