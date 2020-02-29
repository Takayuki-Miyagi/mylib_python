#!/usr/bin/env python3
import os
import subprocess
import time
from . import Orbit
class TransitionDensity:
    def __init__(self, file_td=None, Jbra=0, Jket=0, wfbra=1, wfket=1, file_format="kshell"):
        self.one = {}
        self.two = {}
        self.orbs = Orbit.Orbits()
        self.Jbra = Jbra
        self.Jket = Jket
        self.wfbra = wfbra
        self.wfket = wfket
        self.file_td = file_td
        self.file_format = file_format

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
        if((a,b,jr) in self.one):
            abj = (a,b,jr)
            fact = 1.0
        if((b,a,jr) in self.one):
            abj = (b,a,jr)
            fact = (-1.0)**((ob.j-oa.j)/2)
        try:
            return self.one[abj]*fact
        except:
            print("Warning: not found one-body transition density: (a,b)=", a, b)
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
        if((c,d,a,b,Jcd,Jab,Jr) in self.two): v = self.two[(c,d,a,b,Jcd,Jab,Jr)]; ex_bk = True
        if((b,a,c,d,Jab,Jcd,Jr) in self.two): v = self.two[(b,a,c,d,Jab,Jcd,Jr)]; ex_ab = True
        if((c,d,b,a,Jcd,Jab,Jr) in self.two): v = self.two[(c,d,b,a,Jcd,Jab,Jr)]; ex_ab = True; ex_bk = True
        if((a,b,d,c,Jab,Jcd,Jr) in self.two): v = self.two[(a,b,d,c,Jab,Jcd,Jr)]; ex_cd = True
        if((d,c,a,b,Jcd,Jab,Jr) in self.two): v = self.two[(d,c,a,b,Jcd,Jab,Jr)]; ex_cd = True; ex_bk = True
        if((b,a,d,c,Jab,Jcd,Jr) in self.two): v = self.two[(b,a,d,c,Jab,Jcd,Jr)]; ex_ab = True; ex_cd = True
        if((d,c,b,a,Jcd,Jab,Jr) in self.two): v = self.two[(d,c,b,a,Jcd,Jab,Jr)]; ex_ab = True; ex_cd = True; ex_bk = True

        fact = 1.0
        if(ex_ab): fact *= self._get_phase(a,b,Jab)
        if(ex_cd): fact *= self._get_phase(c,d,Jcd)
        if(ex_bk): fact *= (-1.0)**(Jcd-Jab)
        try:
            return v * fact
        except:
            #print("Warning: not found two-body transition density: (a,b,c,d,Jab,Jcd)=",a,b,c,d,Jab,Jcd)
            return 0.0

    def read_td_file(self):
        if(self.file_td == None):
            print(" set file name!")
            return
        if( self.file_format=="kshell"):
            self._read_td_kshell_format()
            return
        if( self.file_format=="nutbar"):
            self._read_td_nutbar_format()
            return

    def _skip_comment(self,f,comment="#"):
        while True:
            x = f.tell()
            line = f.readline()
            if(line[0] != comment):
                f.seek(x)
                return

    def _read_td_kshell_format(self):
        f = open(self.file_td, 'r')
        orbs = Orbit.Orbits()
        while True:
            line = f.readline()
            if(line[1:12] == "model space"):
                break
        while True:
            entry = f.readline().split()
            if( len(entry)==0 ): break
            if( entry[0]=='k,'): continue
            orbs.add_orbit( int(entry[1]), int(entry[2]), int(entry[3]), int(entry[4]), int(entry[0]))
        self.set_orbits(orbs)
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
                    i_bra = int(d[3][:-1])
                    i_ket = int(d[6][:-1])
                    if(j2_bra == int(2*self.Jbra) and j2_ket == int(2*self.Jket) and
                            i_bra == self.wfbra and i_ket == self.wfket): store_obtd=True
                    if(j2_bra != int(2*self.Jbra) or j2_ket != int(2*self.Jket) or
                            i_bra != self.wfbra or i_ket != self.wfket): store_obtd=False
                    if(not store_obtd): i_obtd = -1
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
                    i_bra = int(d[3][:-1])
                    i_ket = int(d[6][:-1])
                    if(j2_bra == int(2*self.Jbra) and j2_ket == int(2*self.Jket) and
                            i_bra == self.wfbra and i_ket == self.wfket): store_tbtd=True
                    if(j2_bra != int(2*self.Jbra) or j2_ket != int(2*self.Jket) or
                            i_bra != self.wfbra or i_ket != self.wfket): store_tbtd=False
                    if(not store_tbtd): i_tbtd = -1
                if(store_tbtd):
                    data = line.split()
                    a, b, c, d, Jab, Jcd, Jr, wf_label_bra, wf_label_ket, me = \
                            int(data[1]), int(data[2]), int(data[3]), int(data[4]), \
                            int(data[6]), int(data[7]), int(data[8]), \
                            int(data[10]), int(data[11]), float(data[13])
                    self.set_tbtd(a,b,c,d,Jab,Jcd,Jr,me)
                    i_tbtd += 1
                continue

    def _read_td_nutbar_format(self):
        f = open(self.file_td,"r")
        self._skip_comment(f)
        orbs = Orbit.Orbits()
        while True:
            x = f.tell()
            entry = f.readline().split()
            if(entry[0] == "#"):
                f.seek(x)
                break
            orbs.add_orbit( int(entry[1]), int(entry[2]), int(entry[3]), -int(entry[4]), int(entry[0]))
        self.set_orbits(orbs)
        two_body_kets = []
        self._skip_comment(f)
        while True:
            x = f.tell()
            entry = f.readline().split()
            if(len(entry) == 0): break
            two_body_kets.append( (int(entry[1]), int(entry[2]), int(entry[3])) )
        entry = f.readline().split()
        self.Jbra = float(entry[6])
        self.wfbra = int(entry[7])
        self.Jket = float(entry[8])
        self.wfket = int(entry[9])
        Jrank = int(float(  entry[10]))
        line = f.readline()
        while True:
            x = f.tell()
            entry = f.readline().split()
            if(len(entry) == 0): break
            self.set_obtd( int(entry[0]), int(entry[1]), Jrank, float(entry[2]) )
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
            self.set_tbtd( a, b, c, d, Jab, Jcd, Jrank, float(entry[2]))
        print("----------------------------------------")
        print(" read nutbar-format density ")
        print(" < Jf_nf | lambda | Ji_ni > ")
        print(" Ji = {0:6.2f}, ni = {1:3d}".format( self.Jket, self.wfket ))
        print(" lambda = {0:3d}         ".format( Jrank ))
        print(" Jf = {0:6.2f}, nf = {1:3d}".format( self.Jbra, self.wfbra ))
        print("----------------------------------------")
        f.close()

    def print_density(self):
        print(" Model spapce ")
        for key in self.orbs.idx_orb.keys():
            o = self.orbs.get_orbit(key)
            print("{0:3d}, {1:3d}, {2:3d}, {3:3d}, {4:3d}".format(key, o.n, o.l, o.j, o.z ))
        print(" One body ")
        for key in self.one.keys():
            a = key[0]
            b = key[1]
            j = key[2]
            print("{0:3d}, {1:3d}, {2:3d}, {3:12.6f}".format(a,b,j,self.one[key]))
        print(" Two body ")
        for key in self.two.keys():
            a = key[0]
            b = key[1]
            c = key[2]
            d = key[3]
            Jab = key[4]
            Jcd = key[5]
            Jr  = key[6]
            print("{0:3d}, {1:3d}, {2:3d}, {3:3d}, {4:3d}, {5:3d}, {6:3d}, {7:12.6f}".format(a,b,c,d,Jab,Jcd,Jr,self.two[key]))


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
        nme = Op(fn_0v_snt, rankJ=0, rankP=1, rankZ=-2)
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
