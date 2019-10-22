#!/usr/bin/env python3
import os
import subprocess
import time
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
                    if(str1[0:4] != 'w.f.'): print("see file "+self.file_td+" at line "+str(i))
                    d = str1.split()
                    j2_bra = int(d[2][:-3])
                    j2_ket = int(d[5][:-3])
                    if(j2_bra == int(2*self.Jbra) and j2_ket == int(2*self.Jket)): store_obtd=True
                    if(j2_bra != int(2*self.Jbra) or j2_ket != int(2*self.Jket)): store_obtd=False
                if(store_obtd):
                    data = line.split()
                    a, b, jr, wf_label_bra, wf_label_ket, me = int(data[1]), int(data[2]), int(data[4]), \
                            int(data[6]), int(data[7]), float(data[9])
                    self.set_obtd(a,b,jr,me)
                    i_obtd += 1
                continue
            if(line.startswith("TBTD")):
                if(i_tbtd == 0):
                    str1 = lines[i-3]
                    if(str1[0:4] != 'w.f.'): print("see file "+self.file_td+" at line "+str(i))
                    d = str1.split()
                    j2_bra = int(d[2][:-3])
                    j2_ket = int(d[5][:-3])
                    if(j2_bra == int(2*self.Jbra) and j2_ket == int(2*self.Jket)): store_tbtd=True
                    if(j2_bra != int(2*self.Jbra) or j2_ket != int(2*self.Jket)): store_tbtd=False
                if(store_tbtd):
                    data = line.split()
                    a, b, c, d, Jab, Jcd, Jr, wf_label_bra, wf_label_ket, me = \
                            int(data[1]), int(data[2]), int(data[3]), int(data[4]), \
                            int(data[6]), int(data[7]), int(data[8]), \
                            int(data[10]), int(data[11]), float(data[13])
                    self.set_tbtd(a,b,c,d,Jab,Jcd,Jr,me)
                    i_tbtd += 1
                continue

    def read_td_file_old(self):
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

    def calc_density(kshl_dir, fn_snt, fn_ptn_bra, fn_ptn_ket, fn_wf_bra, fn_wf_ket, fn_density=None, \
            header="", batch_cmd=None, run_cmd=None, fn_input="transit.input"):
        if(fn_density==None):
            basename = os.path.basename(fn_snt)
            fn_out = "density_" + os.path.splitext(basename)[0] + ".dat"
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
        prt += '  is_tbtd = .true.\n'
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
