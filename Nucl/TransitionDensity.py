#!/usr/bin/env python3
from . import Orbit
class TransitionDensity:
    def __init__(self, file_td, Jbra, Jket, wfbra, wfket):
        self.one = {}
        self.two = {}
        self.orbs = Orbit.Orbits()
        self.Jbra = Jbra
        self.Jket = Jket
        self.wfbra = wfbra
        self.wfket = wfket
        self.file_td = file_td

    def set_orbits(self, orbs):
        self.orbs = orbs

    def set_obtd(self,a,b,jr,me):
        abj = (a,b,jr)
        self.one[abj] = me

    def set_tbtd(self,a,b,c,d,Jab,Jcd,Jr,me):
        abcd = (a,b,c,d,Jab,Jcd,Jr)
        self.two[abcd] = me

    def get_obtd(self,a,b,jr,Zr):
        oa = self.orbs.get_orbit(a)
        ob = self.orbs.get_orbit(b)
        if(self._triag(oa.j,ob.j,2*jr)): return 0.0
        if(oa.z - ob.z - 2*Zr != 0): return 0.0
        abj = (a,b,jr)
        try:
            return self.one[abj]
        except:
            #print("Warning: not found one-body transition density: (a,b)=", a, b)
            return 0.0

    def _get_phase(self,a,b,Jab):
        oa = self.orbs.get_orbit(a)
        ob = self.orbs.get_orbit(b)
        return -(-1.0) ** ((oa.j + ob.j)/2 - Jab)

    def _triag(self,J1,J2,J3):
        b = True
        if(abs(J1-J2) <= J3 <= J1+J2): b = False
        return b

    def get_tbtd(self,a,b,c,d,Jab,Jcd,Jr,Zr):

        oa = self.orbs.get_orbit(a)
        ob = self.orbs.get_orbit(b)
        oc = self.orbs.get_orbit(c)
        od = self.orbs.get_orbit(d)
        if(self._triag(Jab,Jcd,Jr)): return 0.0
        if(oa.z + ob.z - oc.z - od.z - 2*Zr != 0): return 0.0
        ex_ab, ex_cd = False, False

        if((a,b,c,d,Jab,Jcd,Jr) in self.two): v = self.two[(a,b,c,d,Jab,Jcd,Jr)]
        if((b,a,c,d,Jab,Jcd,Jr) in self.two): v = self.two[(b,a,c,d,Jab,Jcd,Jr)]; ex_ab = True
        if((a,b,d,c,Jab,Jcd,Jr) in self.two): v = self.two[(a,b,d,c,Jab,Jcd,Jr)]; ex_cd = True
        if((b,a,d,c,Jab,Jcd,Jr) in self.two): v = self.two[(b,a,d,c,Jab,Jcd,Jr)]; ex_ab = True; ex_cd = True

        fact = 1.0
        if(ex_ab): fact *= self._get_phase(a,b,Jab)
        if(ex_cd): fact *= self._get_phase(c,d,Jcd)
        try:
            return v * fact
        except:
            #print("Warning: not found two-body transition density: (a,b,c,d,Jab,Jcd)=",a,b,c,d,Jab,Jcd)
            return 0.0

    def read_td_file(self):
        f = open(self.file_td, 'r')

        self._find_label(f)

        tf = False
        while tf == False:
            line = f.readline()
            tf = line.startswith("OBTD")
            if(tf == True):
                data = line.split()
                a, b, jr, wf_label_bra, wf_label_ket, me = int(data[1]), int(data[2]), int(data[4]), \
                        int(data[6]), int(data[7]), float(data[9])
                self.set_obtd(a,b,jr,me)

        tf = True
        while tf == True:
            line = f.readline()
            tf = line.startswith("OBTD")
            if(tf == True):
                data = line.split()
                a, b, jr, wf_label_bra, wf_label_ket, me = int(data[1]), int(data[2]), int(data[4]), \
                        int(data[6]), int(data[7]), float(data[9])
                self.set_obtd(a,b,jr,me)

        tf = False
        while tf == False:
            line = f.readline()
            tf = line.startswith("TBTD")
            if(tf == True):
                data = line.split()
                a, b, c, d, Jab, Jcd, Jr, wf_label_bra, wf_label_ket, me = \
                        int(data[1]), int(data[2]), int(data[3]), int(data[4]), \
                        int(data[6]), int(data[7]), int(data[8]), \
                        int(data[10]), int(data[11]), float(data[13])
                self.set_tbtd(a,b,c,d,Jab,Jcd,Jr,me)

        tf = True
        while tf == True:
            line = f.readline()
            tf = line.startswith("TBTD")
            if(tf == True):
                data = line.split()
                a, b, c, d, Jab, Jcd, Jr, wf_label_bra, wf_label_ket, me = \
                        int(data[1]), int(data[2]), int(data[3]), int(data[4]), \
                        int(data[6]), int(data[7]), int(data[8]), \
                        int(data[10]), int(data[11]), float(data[13])
                self.set_tbtd(a,b,c,d,Jab,Jcd,Jr,me)
        f.close()
    def _find_label(self,f):
        tf = False
        while tf == False:
            line = f.readline()
            if(line[0:4] == 'w.f.'):
                data = line.split()
                i_bra = int(data[3][:-1])
                i_ket = int(data[6][:-1])
                if(i_bra == self.wfbra and i_ket == self.wfket): return


def main():
    file_td="transition-density-file-name"
    Jbra = 0
    Jket = 0
    wfbra = 1
    wfket = 1
    TD = TransitionDensity(file_td, Jbra, Jket, wfbra, wfket)
    TD.read_td_file()

if(__name__=="__main__"):
    main()
