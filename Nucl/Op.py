#!/usr/bin/env python3
from . import Orbit
class Op:
    def __init__(self, file_op, rankJ=0, rankP=1, rankZ=0, file_format="snt"):
        self.zero = 0.0
        self.one = {}
        self.two = {}
        self.orbs = []
        self.rankJ = rankJ
        self.rankP = rankP
        self.rankZ = rankZ
        self.file_op = file_op
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
        if(oa.z - ob.z - 2*self.rankZ != 0): return 0.0
        if((-1)**(oa.l+ob.l) * self.rankP != 1): return 0.0
        ex_bk = False
        if((a,b) in self.one): v = self.one[(a,b)]
        if((b,a) in self.one): v = self.one[(b,a)]; ex_bk = True
        fact = 1.0
        if(ex_bk):
            fact *= (-1.0) ** ((oa.j - ob.j)/2)
        try:
            return v * fact
        except:
            print("Warning: not found one-body matrix element: (a,b) = ", a, b)
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
            print("warining: J")
            return 0.0
        if(oa.z+ob.z-oc.z-od.z - 2*self.rankZ != 0):
            print("warining: Z")
            return 0.0
        if((-1)**(oa.l + ob.l + oc.l + od.l) * self.rankP != 1):
            print("warining: P")
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
        if(ex_bk): fact *= (-1.0) ** (Jab-Jcd)
        if(ex_ab): fact *= self._get_phase(a,b,Jab)
        if(ex_cd): fact *= self._get_phase(c,d,Jcd)
        try:
            return v * fact
        except:
            print("Warning: not found two-body matrix element, (a,b,c,d,Jab,Jcd)=",a,b,c,d,Jab,Jcd)
            return 0.0

    def read_operator_file(self, comment="!"):
        if(self.file_format == "snt"):
            self._read_operator_snt(comment)
            return
        print("In NuHamil.py: Unknown file format")
        return

    def _read_operator_snt(self, comment="!"):
        orbs = Orbit.Orbits()
        f = open(self.file_op, 'r')
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
