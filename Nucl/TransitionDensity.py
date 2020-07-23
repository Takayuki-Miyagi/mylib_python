#!/usr/bin/env python3
import os, sys, copy, gzip, subprocess, time
import numpy as np
if(__package__==None or __package__==""):
    import ModelSpace
else:
    from . import Orbits
    from . import ModelSpace

class TransitionDensity:
    def __init__(self, Jbra=None, Jket=None, wflabel_bra=None, wflabel_ket=None, ms=None, filename=None, file_format="kshell", verbose=False):
        self.Jbra = Jbra
        self.Jket = Jket
        self.wflabel_bra = wflabel_bra
        self.wflabel_ket = wflabel_ket
        self.ms = copy.deepcopy(ms)
        self.verbose = verbose
        self.one = {}
        self.two = {}
        self.three = {}
        if( ms != None ): self.allocate_density( ms )
        if( filename != None ): self.read_density_file( filename, file_format )
    def allocate_density( self, ms ):
        self.ms = copy.deepcopy(ms)
        orbits = ms.orbits
        self.one = {}
        two = ms.two
        for ichbra in range(two.get_number_channels()):
            chbra = two.get_channel(ichbra)
            for ichket in range(ichbra+1):
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
            for j in range(i+1):
                chket = two.get_channel(j)
                counter += len( self.two[(i,j)] )
        return counter
    def set_1btd( self, a, b, jrank, me):
        orbits = self.ms.orbits
        oa = orbits.get_orbit(a)
        ob = orbits.get_orbit(b)
        me_rank = {jrank: me}
        self.one[(a,b,jrank)] = me
        self.one[(b,a,jrank)] = me * (-1)**( (ob.j-oa.j)//2 )
    def set_2btd_from_mat_indices( self, chbra, chket, bra, ket, jrank, me ):
        if( chbra < chket ):
            if(self.verbose): print("Warning:" + sys._getframe().f_code.co_name )
            return
        self.two[(chbra,chket)][(bra,ket,jrank)] = me
    def set_2btd_from_indices( self, a, b, c, d, Jab, Jcd, jrank, me ):
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
        if( self._triag( Jab, Jcd, jrank )):
            if(self.verbose): print("Warning: J, " + sys._getframe().f_code.co_name )
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
    def get_2btd_from_mat_indices(self, chbra, chket, bra, ket, jrank):
        if( chbra < chket ):
            if(self.verbose): print("Warning:" + sys._getframe().f_code.co_name )
            return 0
        try:
            return self.two[(chbra,chket)][(bra,ket,jrank)]
        except:
            if(self.verbose): print("Nothing here " + sys._getframe().f_code.co_name )
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
        Pcd = (-1)**(oa.l+ob.l)
        Zab = (oa.z + ob.z)//2
        Zcd = (oc.z + od.z)//2
        if( self._triag( Jab, Jcd, jrank )):
            if(self.verbose): print("Warning: J, " + sys._getframe().f_code.co_name )
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
        return self.get_2btd_from_mat_indices(ichbra,ichket,bra,ket,jrank)*phase
    def get_2btd_from_orbits( self, oa, ob, oc, od, Jab, Jcd, jrank ):
        if(self.ms.rank <= 1): return 0
        orbits = self.ms.orbits
        a = orbits.orbit_index_from_orbit( oa )
        b = orbits.orbit_index_from_orbit( ob )
        c = orbits.orbit_index_from_orbit( oc )
        d = orbits.orbit_index_from_orbit( od )
        return self.get_2btd_from_indices( a, b, c, d, Jab, Jcd, jrank )

    def _triag(self,J1,J2,J3):
        b = True
        if(abs(J1-J2) <= J3 <= J1+J2): b = False
        return b
    def read_density_file(self, filename=None, file_format="kshell"):
        if(filename == None):
            print(" set file name!")
            return
        if( file_format=="kshell"):
            self._read_td_kshell_format(filename)
            if( self.count_nonzero_1btd() + self.count_nonzero_2btd() == 0):
                print("The number of non-zero transition density matrix elements is 0 better to check: "+ filename + "!! " + \
                        "Jbra=" + str(self.Jbra) + " (wf label:"+ str(self.wflabel_bra)+"), Jket="+str(self.Jket)+" (wf label:"+str(self.wflabel_ket)+")")
            return
        if( file_format=="nutbar"):
            self._read_td_nutbar_format(filename)
            if( self.count_nonzero_1btd() + self.count_nonzero_2btd() == 0):
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
        f = open(filename, 'r')
        orbs = Orbits()
        while True:
            line = f.readline()
            if(line[1:12] == "model space"):
                break
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
            print("{0:3d}, {1:3d}, {2:3d}, {3:12.6f}".format(a,b,j,self.one[key]))
        print(" Two body ")
        two = self.ms.two
        for ichbra in range(two.get_number_channels()):
            chbra = two.get_channel(ichbra)
            for ichket in range(ichbra+1):
                chket = two.get_channel(ichket)
                for key in self.two[(ichbra,ichket)].keys():
                    bra, ket, Jr = key
                    a, b = chbra.orbit1_index[bra], chbra.orbit2_index[bra]
                    c, d = chket.orbit1_index[ket], chket.orbit2_index[ket]
                    tbtd = self.two[(ichbra,ichket)][key]
                    print("{0:3d}, {1:3d}, {2:3d}, {3:3d}, {4:3d}, {5:3d}, {6:3d}, {7:12.6f}".format(a,b,c,d,chbra.J,chket.J,Jr,tbtd))


    def calc_density(kshl_dir, fn_snt, fn_ptn_bra, fn_ptn_ket, fn_wf_bra, fn_wf_ket, i_wfs=None, fn_density=None, \
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
        if(i_wfs!=None):
            prt += '  n_eig_lr_pair = '
            for lr in i_wfs:
                prt += str(lr[0]) + ', ' + str(lr[1]) + ', '
            prt += '\n'
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

    def calc_expectation_value( self, op ):
        orbits_de = self.ms.orbits
        orbits_op = op.ms.orbits
        norbs = orbits_op.get_num_orbits()

        zero = op.get_0bme()
        one = 0
        for i in range(1, norbs+1):
            oi = orbits_op.get_orbit(i)
            i_d = orbits_de.get_orbit_index(oi.n, oi.l, oi.j, oi.z)
            for j in range(1, norbs+1):
                oj = orbits_op.get_orbit(j)
                j_d = orbits_de.get_orbit_index(oj.n, oj.l, oj.j, oj.z)
                if( oi.z-oj.z != 2*op.rankZ): continue # only <n|O|p> if rankZ != 0
                if( op.rankJ==0 and op.rankP==1 and op.rankZ==0 ):
                    one += op.get_1bme(i,j) * self.get_1btd(i_d,j_d,op.rankJ) * np.sqrt(oj.j+1) / np.sqrt(2*self.Jbra+1)
                else:
                    #print(op.get_1bme(i,j), self.get_1btd(i_d,j_d,op.rankJ), op.get_1bme(i,j) * self.get_1btd(i_d,j_d,op.rankJ) / np.sqrt(2*op.rankJ+1))
                    one += op.get_1bme(i,j) * self.get_1btd(i_d,j_d,op.rankJ)

        two = 0
        for i in range(1, norbs+1):
            oi = orbits_op.get_orbit(i)
            for j in range(i, norbs+1):
                oj = orbits_op.get_orbit(j)
                for k in range(1, norbs+1):
                    ok = orbits_op.get_orbit(k)
                    for l in range(k, norbs+1):
                        ol = orbits_op.get_orbit(l)

                        i_d = orbits_de.get_orbit_index(oi.n, oi.l, oi.j, oi.z)
                        j_d = orbits_de.get_orbit_index(oj.n, oj.l, oj.j, oj.z)
                        k_d = orbits_de.get_orbit_index(ok.n, ok.l, ok.j, ok.z)
                        l_d = orbits_de.get_orbit_index(ol.n, ol.l, ol.j, ol.z)
                        if((-1)**(oi.l+oj.l+ok.l+ol.l) * op.rankP != 1): continue
                        if( oi.z+oj.z-ok.z-ol.z != 2*op.rankZ): continue # only <nn|O|pp>, <nn|O|np>, <np|O|pp> if rankZ !=0
                        for Jij in range( int(abs(oi.j-oj.j)/2), int((oi.j+oj.j)/2)+1):
                            if(i == j and Jij%2 == 1): continue
                            for Jkl in range( int(abs(ok.j-ol.j)/2), int((ok.j+ol.j)/2+1)):
                                if(k == l and Jkl%2 == 1): continue
                                if( self._triag( Jij, Jkl, op.rankJ )): continue
                                if(op.rankJ==0 and op.rankP==1 and op.rankZ==0):
                                    two += op.get_2bme_from_indices(i,j,k,l,Jij,Jkl) * self.get_2btd_from_indices(i_d,j_d,k_d,l_d,Jij,Jkl,op.rankJ) * \
                                            np.sqrt(2*Jij+1)/np.sqrt(2*self.Jbra+1)
                                    #print("{:3d},{:3d},{:3d},{:3d},{:3d},{:3d},{:3d},{:12.6f}".format(i_d,j_d,k_d,l_d,Jij,Jkl,op.rankJ,self.get_2btd_from_indices(i_d,j_d,k_d,l_d,Jij,Jkl,op.rankJ)))
                                    #print("{:3d},{:3d},{:3d},{:3d},{:3d},{:3d},{:12.6f}".format(i,j,k,l,Jij,Jkl,op.get_2bme_from_indices(i,j,k,l,Jij,Jkl)))
                                else:
                                    two += op.get_2bme_from_indices(i,j,k,l,Jij,Jkl) * self.get_2btd_from_indices(i_d,j_d,k_d,l_d,Jij,Jkl,op.rankJ)
        return zero,one,two

def main():
    file_td="transition-density-file-name"
    TD = TransitionDensity()
if(__name__=="__main__"):
    main()
