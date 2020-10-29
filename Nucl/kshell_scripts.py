#!/usr/bin/env python3
import os, sys, time, subprocess, re, itertools
if(__package__==None or __package__==""):
    import PeriodicTable
    import Operator
else:
    from . import PeriodicTable
    from . import Operator
class kshell_scripts:
    def __init__(self, kshl_dir=None, fn_snt=None, Nucl=None, states="+10,-10", hw_truncation=None):
        self.kshl_dir = kshl_dir
        self.fn_snt = fn_snt
        self.Nucl = Nucl
        self.states = states
        self.hw_truncation=hw_truncation
        isdigit = re.search(r'\d+', self.Nucl)
        self.A = int( isdigit.group() )
        asc = self.Nucl[:isdigit.start()] + self.Nucl[isdigit.end():]
        asc = asc.lower()
        asc = asc[0].upper() + asc[1:]
        self.Z = PeriodicTable.periodic_table.index(asc)
        self.N = self.A-self.Z
        self.fn_ptns = {}
        self.fn_wfs = {}
        for state in self.states.split(","):
            state_str = self._state_string(state)
            self.fn_ptns[state] = "{:s}_{:s}_{:s}.ptn".format(self.Nucl, os.path.splitext(os.path.basename(self.fn_snt))[0], state_str[-1] )
            self.fn_wfs[state] = "{:s}_{:s}_{:s}.wav".format(self.Nucl, os.path.splitext(os.path.basename(self.fn_snt))[0], state_str )

    def _get_wf_index( self, fn_summary ):
        jpn_to_idx = {}
        f = open( fn_summary, "r" )
        lines = f.readlines()
        f.close()
        logs = set()
        idxs = {}
        for line in lines[5:]:
            dat = line.split()
            if( len(dat) == 0 ): continue
            if( dat[-1] in logs ):
                idxs[ dat[-1] ] += 1
            else:
                idxs[ dat[-1] ] = 1
                logs.add( dat[-1] )
            jpn_to_idx[(dat[1],dat[2],int(dat[3]))] = (dat[-1], idxs[ dat[-1] ])
        return jpn_to_idx
    def _state_string(self, state):
        """
        example:
        0+1 -> j0p
        0+2 -> j0p
        2+2 -> j4p
        0.5-2 -> j1n
        1.5-2 -> j3n
        +1 -> m0p or m1p
        -1 -> m0n or m1n
        """
        isdigit = re.findall(r'\d+', state)
        if( len(isdigit)==3 ):
            if( state.find("+")!=-1): state_str = "j{:d}p".format(int(2*int(isdigit[0])+1))
            if( state.find("-")!=-1): state_str = "j{:d}n".format(int(2*int(isdigit[0])+1))
        elif( len(isdigit)==2 ):
            if( state.find("+")!=-1): state_str = "j{:d}p".format(int(2*int(isdigit[0])))
            if( state.find("-")!=-1): state_str = "j{:d}n".format(int(2*int(isdigit[0])))
        elif( len(isdigit)==1 ):
            if( state.find("+")!=-1): state_str = "m0p"
            if( state.find("-")!=-1): state_str = "m0n"
            if(self.A%2==1):
                if( state.find("+")!=-1): state_str = "m1p"
                if( state.find("-")!=-1): state_str = "m1n"
        return state_str

    def _i2prty(self, i):
        if(i == 1): return '+'
        else: return '-'
    def get_occupation(self, logs=None):
        if(logs==None):
            logs = []
            states = self.states.split(",")
            for state in states:
                state_str = self._state_string(state)
                log = "log_{:s}_{:s}_{:s}.txt".format(self.Nucl, os.path.splitext( os.path.basename(self.fn_snt))[0], state_str)
                logs.append(log)
        e_data = {}
        for log in logs:
            f = open(log,"r")
            while True:
                line = f.readline()
                if(not line): break
                if(line[6:10] == "<H>:"):
                    dat = line.split()
                    n_eig= int(dat[0])
                    ene  = float(dat[2])
                    mtot = int(dat[6][:-2])
                    prty = int(dat[8])
                    prty = self._i2prty(self,prty)
                    while ene in e_data: ene += 0.000001
                    line = f.readline()
                    if line[42:45] != ' T:': continue
                    tt = int(line[45:48])
                    line = f.readline()
                    data = line.split()
                    if(line[0:7] ==" <p Nj>"):
                        plist = []
                        for i in range(len(data)-2):
                            plist.append(float(data[i+2]))
                    line = f.readline()
                    data = line.split()
                    if(line[0:7] ==" <n Nj>"):
                        nlist = []
                        for i in range(len(data)-2):
                            nlist.append(float(data[i+2]))
                    while len(line)!=0:
                        line = f.readline()
                        data = line.split()
                        if(line[0:4] ==" hw:"):
                            hws = {}
                            for i in range(len(data)-1):
                                hw, prob = data[i+1].split(":")
                                hws[int(hw)] = float(prob)
                            break
                    e_data[ round(ene,3) ] = (log, mtot, prty, n_eig, tt, plist, nlist, hws)
            f.close()
        return e_data

    def run_kshell(self, header="", batch_cmd=None, run_cmd=None, dim_cnt=False, args=None):
        fn_script = "{:s}_{:s}".format(self.Nucl, os.path.splitext(os.path.basename(self.fn_snt))[0])
        if(args != None):
            if( 'beta_cm' in args): fn_script += "_betacm{:d}".format(args['beta_cm'])
        if(self.hw_truncation != None): fn_script += "_hw" + str(self.hw_truncation)
        if(not os.path.isfile(self.fn_snt)):
            print(fn_snt, "not found")
            return
        unnatural=False
        if( self.states.find("-") != -1 and self.states.find("+")!=-1 ): unnatural=True
        f = open('ui.in','w')
        f.write('\n')
        f.write(self.fn_snt+'\n')
        f.write(self.Nucl+'\n')
        f.write(fn_script+'\n')
        f.write(self.states+'\n')
        if(self.hw_truncation==None): f.write('\n')
        else:
            f.write('2\n')
            f.write(str(self.hw_truncation)+'\n')
        if(unnatural):
            if(self.hw_truncation==None): f.write('\n')
            else:
                f.write('2\n')
                f.write(str(self.hw_truncation)+'\n')
        if(args!=None):
            for key in args.keys():
                f.write('{:s}={:s}'.format(key, str(args[key])))
        f.write('\n')
        f.write('\n')
        f.write('\n')
        f.close()
        cmd = 'python2 '+self.kshl_dir+'/kshell_ui.py < ui.in'
        subprocess.call(cmd, shell=True)
        f = open(fn_script+".sh", "r")
        lines = f.readlines()
        f.close()
        prt = ""
        for line in lines[:3]:
            prt += line
        if( header != "" ): prt = header
        for line in lines[3:]:
            if(line.find("./kshell.exe") != -1):
                if( run_cmd == None ): prt += "./kshell.exe " + line[18:]
                if( run_cmd != None ): prt += run_cmd + " ./kshell.exe " + line[18:]
            else:
                prt += line
        f = open(fn_script+".sh", "w")
        f.write(prt)
        f.close()

        subprocess.call("rm ui.in", shell=True)
        if( dim_cnt ):
            if( os.path.exists( fn_script+'_p.ptn' ) ):
                cmd = 'python2 ' + self.kshl_dir+'/count_dim.py ' + fn_snt + ' ' + fn_script + '_p.ptn'
                print(cmd)
                subprocess.call(cmd, shell=True)
            if( os.path.exists( fn_script+'_n.ptn' ) ):
                cmd = 'python2 ' + self.kshl_dir+'/count_dim.py ' + fn_snt + ' ' + fn_script + '_n.ptn'
                subprocess.call(cmd, shell=True)
        else:
            fn_script += ".sh"
            os.chmod(fn_script, 0o755)
            if(batch_cmd == None): cmd = "./" + fn_script
            if(batch_cmd != None): cmd = batch_cmd + " " + fn_script
            subprocess.call(cmd, shell=True)

        time.sleep(1)

    def run_kshell_lsf(self, fn_ptn_init, fn_ptn, fn_wf, fn_wf_out, J2, \
            op=None, fn_input=None, n_vec=100, header="", batch_cmd=None, run_cmd=None, \
            fn_operator=None, operator_irank=0, operator_nbody=1, operator_iprty=1):
        fn_script = os.path.basename(os.path.splitext(fn_wf_out)[0]) + ".sh"
        fn_out = "log_" + os.path.basename(os.path.splitext(fn_wf_out)[0]) + ".txt"
        if(fn_input==None): fn_input = os.path.basename(os.path.splitext(fn_wf_out)[0]) + ".input"
        if(op==None and fn_operator==None):
            print("Put either op or fn_operator")
            return
        if(op!=None and fn_operator!=None):
            print("You cannot put both op and fn_operator")
            return
        if(not os.path.isfile(self.fn_snt)):
            print(self.fn_snt, "not found")
            return
        cmd = "cp " + self.kshl_dir + "/kshell.exe ./"
        subprocess.call(cmd,shell=True)
        prt = header + '\n'
        prt += 'echo "start runnning ' + fn_out + ' ..."\n'
        prt += 'cat >' + fn_input + ' <<EOF\n'
        prt += '&input\n'
        prt += '  fn_int   = "' + self.fn_snt + '"\n'
        prt += '  fn_ptn = "' + fn_ptn + '"\n'
        prt += '  fn_ptn_init = "' + fn_ptn_init + '"\n'
        prt += '  fn_load_wave = "' + fn_wf + '"\n'
        prt += '  fn_save_wave = "' + fn_wf_out + '"\n'
        prt += '  max_lanc_vec = '+str(n_vec)+'\n'
        prt += '  n_eigen = '+str(n_vec)+'\n'
        prt += '  n_restart_vec = '+str(min(n_vec,200))+'\n'
        prt += '  mtot = '+str(J2)+'\n'
        prt += '  maxiter = 1\n'
        prt += '  is_double_j = .true.\n'
        if(op!=None): prt += '  op_type_init = "'+str(op)+'"\n'
        if(fn_operator!=None):
            prt += '  fn_operator = "'+str(fn_operator)+'"\n'
            prt += '  operator_irank = '+str(operator_irank)+'\n'
            prt += '  operator_nbody = '+str(operator_nbody)+'\n'
            prt += '  operator_iprty = '+str(operator_iprty)+'\n'
        prt += '  eff_charge = 1.0, 0.0\n'
        prt += '  e1_charge = 1.0, 0.0\n'
        prt += '&end\n'
        prt += 'EOF\n'
        if(run_cmd == None):
            prt += './kshell.exe ' + fn_input + ' > ' + fn_out + ' 2>&1\n'
        if(run_cmd != None):
            prt += run_cmd + ' ./kshell.exe ' + fn_input + ' > ' + fn_out + ' 2>&1\n'
        prt += 'rm ' + fn_input + '\n'
        f = open(fn_script,'w')
        f.write(prt)
        f.close()
        os.chmod(fn_script, 0o755)
        if(batch_cmd == None): cmd = "./" + fn_script
        if(batch_cmd != None): cmd = batch_cmd + " " + fn_script
        subprocess.call(cmd, shell=True)
        time.sleep(1)

#    def calc_density(self, fn_ptn_bra, fn_ptn_ket, fn_wf_bra, fn_wf_ket, i_wfs=None, fn_density=None, \
#            header="", batch_cmd=None, run_cmd=None, fn_input="transit.input", calc_SF=False):
#        if(fn_density==None):
#            basename = os.path.basename(self.fn_snt)
#            fn_out = "density_{:s}.dat".format(os.path.splitext(basename)[0])
#        if(fn_density!=None): fn_out = fn_density
#        fn_script = os.path.splitext(fn_out)[0] + ".sh"
#        cmd = "cp " + self.kshl_dir + "/transit.exe ./"
#        subprocess.call(cmd,shell=True)
#        prt = header + '\n'
#        prt += 'echo "start runnning ' + fn_out + ' ..."\n'
#        prt += 'cat >' + fn_input + ' <<EOF\n'
#        prt += '&input\n'
#        prt += '  fn_int   = "' + self.fn_snt + '"\n'
#        prt += '  fn_ptn_l = "' + fn_ptn_bra + '"\n'
#        prt += '  fn_ptn_r = "' + fn_ptn_ket + '"\n'
#        prt += '  fn_load_wave_l = "' + fn_wf_bra + '"\n'
#        prt += '  fn_load_wave_r = "' + fn_wf_ket + '"\n'
#        if(i_wfs!=None):
#            prt += '  n_eig_lr_pair = '
#            for lr in i_wfs:
#                prt += str(lr[0]) + ', ' + str(lr[1]) + ', '
#            prt += '\n'
#        prt += '  hw_type = 2\n'
#        prt += '  eff_charge = 1.5, 0.5\n'
#        prt += '  gl = 1.0, 0.0\n'
#        prt += '  gs = 3.91, -2.678\n'
#        if(not calc_SF): prt += '  is_tbtd = .true.\n'
#        prt += '&end\n'
#        prt += 'EOF\n'
#        if(run_cmd == None):
#            prt += './transit.exe ' + fn_input + ' > ' + fn_out + ' 2>&1\n'
#        if(run_cmd != None):
#            prt += run_cmd + ' ./transit.exe ' + fn_input + ' > ' + fn_out + ' 2>&1\n'
#        prt += 'rm ' + fn_input + '\n'
#        f = open(fn_script,'w')
#        f.write(prt)
#        f.close()
#        os.chmod(fn_script, 0o755)
#        if(batch_cmd == None): cmd = "./" + fn_script
#        if(batch_cmd != None): cmd = batch_cmd + " " + fn_script
#        subprocess.call(cmd, shell=True)
#        time.sleep(1)

class transit_scripts:
    def __init__(self, kshl_dir=None, i_wfs=None):
        self.kshl_dir = kshl_dir
        self.i_wfs = i_wfs
        self.filenames = {}

    def set_filenames(self, ksh_l, ksh_r, states_list=None, calc_SF=False):
        if(states_list==None):
            states_list = [(x,y) for x,y in itertools.product( ksh_l.states.split(","), ksh_r.states.split(",") )]
        bra_side = ksh_l
        ket_side = ksh_r
        flip=False

        if( ksh_l.A < ksh_r.A ):
            bra_side = ksh_r
            ket_side = ksh_l
            flip=True

        for states in states_list:
            state_l = states[0]
            state_r = states[1]
            if(flip):
                state_l = states[1]
                state_r = states[0]
            str_l = bra_side._state_string(state_l)
            str_r = ket_side._state_string(state_r)
            fn_density = "density_{:s}_{:s}{:s}_{:s}{:s}.txt".format(os.path.splitext( os.path.basename( ket_side.fn_snt ) )[0], bra_side.Nucl,str_l, ket_side.Nucl,str_r )
            if(calc_SF): fn_density = "SF_{:s}_{:s}{:s}_{:s}{:s}.txt".format(os.path.splitext( os.path.basename( ket_side.fn_snt ) )[0], bra_side.Nucl,str_l, ket_side.Nucl,str_r )
            self.filenames[(state_l,state_r)] = fn_density
        return flip

    def calc_density(self, ksh_l, ksh_r, states_list=None, header="", batch_cmd=None, run_cmd=None, calc_SF=False):
        if(states_list==None):
            states_list = [(x,y) for x,y in itertools.product( ksh_l.states.split(","), ksh_r.states.split(",") )]
        bra_side = ksh_l
        ket_side = ksh_r
        flip=False

        if( ksh_l.A < ksh_r.A ):
            bra_side = ksh_r
            ket_side = ksh_l
            flip=True

        for states in states_list:
            state_l = states[0]
            state_r = states[1]
            if(flip):
                state_l = states[1]
                state_r = states[0]
            str_l = bra_side._state_string(state_l)
            str_r = ket_side._state_string(state_r)
            fn_density = "density_{:s}_{:s}{:s}_{:s}{:s}.txt".format(os.path.splitext( os.path.basename( ket_side.fn_snt ) )[0], bra_side.Nucl,str_l, ket_side.Nucl,str_r )
            if(calc_SF): fn_density = "SF_{:s}_{:s}{:s}_{:s}{:s}.txt".format(os.path.splitext( os.path.basename( ket_side.fn_snt ) )[0], bra_side.Nucl,str_l, ket_side.Nucl,str_r )

            fn_script = os.path.splitext(fn_density)[0] + ".sh"
            fn_input = os.path.splitext(fn_density)[0] + ".input"
            cmd = "cp " + self.kshl_dir + "/transit.exe ./"
            subprocess.call(cmd,shell=True)
            prt = header + '\n'
            prt += 'echo "start runnning ' + fn_density + ' ..."\n'
            prt += 'cat >' + fn_input + ' <<EOF\n'
            prt += '&input\n'
            prt += '  fn_int   = "' + ket_side.fn_snt + '"\n'
            prt += '  fn_ptn_l = "' + bra_side.fn_ptns[state_l]+ '"\n'
            prt += '  fn_ptn_r = "' + ket_side.fn_ptns[state_r]+ '"\n'
            prt += '  fn_load_wave_l = "' + bra_side.fn_wfs[state_l] + '"\n'
            prt += '  fn_load_wave_r = "' + ket_side.fn_wfs[state_r] + '"\n'
            if(self.i_wfs!=None):
                prt += '  n_eig_lr_pair = '
                for lr in i_wfs:
                    if(flip):
                        prt += str(lr[1]) + ', ' + str(lr[0]) + ', '
                    else:
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
                prt += './transit.exe ' + fn_input + ' > ' + fn_density + ' 2>&1\n'
            if(run_cmd != None):
                prt += run_cmd + ' ./transit.exe ' + fn_input + ' > ' + fn_density + ' 2>&1\n'
            prt += 'rm ' + fn_input + '\n'
            f = open(fn_script,'w')
            f.write(prt)
            f.close()
            os.chmod(fn_script, 0o755)
            if(batch_cmd == None): cmd = "./" + fn_script
            if(batch_cmd != None): cmd = batch_cmd + " " + fn_script
            subprocess.call(cmd, shell=True)
            time.sleep(1)

    def calc_espe(self, kshl, snts=None, states_dest="+20,-20", header="", batch_cmd=None, run_cmd=None, step="full", mode="hole", N_states=None, kshell_args=None):
        """
        snts = [ snt_file_for_Z-1_N, snt_file_for_Z_N-1, snt_file_for_Z+1_N, snt_file_for_Z_N+1 ]
        """
        if(mode=="hole"):
            min_idx = 0
            max_idx = 2
        elif(mode=="particle"):
            min_idx = 2
            max_idx = 4
        else:
            min_idx = 0
            max_idx = 4
        if(snts==None):
            snts = [kshl.fn_snt] * 4
        if(step=="diagonalize" or step=="full"):
            kshl.run_kshell(header=header, batch_cmd=batch_cmd, run_cmd=run_cmd, args=kshell_args)
            for idx in range(min_idx,max_idx):
                fn_snt = snts[idx]
                if(idx==0): Z, N = kshl.Z-1, kshl.N
                if(idx==1): Z, N = kshl.Z, kshl.N-1
                if(idx==2): Z, N = kshl.Z+1, kshl.N
                if(idx==3): Z, N = kshl.Z, kshl.N+1
                Nucl = "{:s}{:d}".format(PeriodicTable.periodic_table[Z],Z+N)
                kshl_tr = kshell_scripts(kshl_dir=kshl.kshl_dir, fn_snt=fn_snt, Nucl=Nucl, states=states_dest)
                kshl_tr.run_kshell(header=header, batch_cmd=batch_cmd, run_cmd=run_cmd, args=kshell_args)
        if(step=="density" or step=="full"):
            for idx in range(min_idx,max_idx):
                fn_snt = snts[idx]
                if(idx==0): Z, N = kshl.Z-1, kshl.N
                if(idx==1): Z, N = kshl.Z, kshl.N-1
                if(idx==2): Z, N = kshl.Z+1, kshl.N
                if(idx==3): Z, N = kshl.Z, kshl.N+1
                Nucl = "{:s}{:d}".format(PeriodicTable.periodic_table[Z],Z+N)
                kshl_tr = kshell_scripts(kshl_dir=kshl.kshl_dir, fn_snt=fn_snt, Nucl=Nucl, states=states_dest)
                trs = transit_scripts(kshl_dir=kshl.kshl_dir)
                trs.calc_density(kshl,kshl_tr,calc_SF=True)
        # final step
        espe = {}
        sum_sf = {}
        for idx in range(min_idx,max_idx):
            fn_snt = snts[idx]
            if(idx==0): Z, N = kshl.Z-1, kshl.N
            if(idx==1): Z, N = kshl.Z, kshl.N-1
            if(idx==2): Z, N = kshl.Z+1, kshl.N
            if(idx==3): Z, N = kshl.Z, kshl.N+1
            Nucl = "{:s}{:d}".format(PeriodicTable.periodic_table[Z],Z+N)
            kshl_tr = kshell_scripts(kshl_dir=kshl.kshl_dir, fn_snt=fn_snt, Nucl=Nucl, states=states_dest)
            trs = transit_scripts(kshl_dir=kshl.kshl_dir)
            flip = trs.set_filenames(kshl, kshl_tr, calc_SF=True)
            if(flip):
                Hm_bra = Operator(filename = kshl_tr.fn_snt)
                Hm_ket = Operator(filename = kshl.fn_snt)
            else:
                Hm_bra = Operator(filename = kshl.fn_snt)
                Hm_ket = Operator(filename = kshl_tr.fn_snt)
            for key in trs.filenames.keys():
                fn = trs.filenames[key]
                espe_each, sum_sf_each = trs.read_sf_file(fn, Hm_bra, Hm_ket, N_states=N_states)
                for key in espe_each:
                    if( key in espe ):
                        espe[key] += espe_each[key]
                        sum_sf[key] += sum_sf_each[key]
                    else:
                        espe[key] = espe_each[key]
                        sum_sf[key] = sum_sf_each[key]
        return espe, sum_sf
    def read_sf_file(self,fn, Hm_bra, Hm_ket, N_states=None):
        f = open(fn,'r')
        lines = f.readlines()
        f.close()
        espe = {}
        sum_sfs = {}
        read=False
        energy = 0.0
        sum_sf=0.0
        for line in lines:
            if( line[:7] == "orbit :" ):
                data = line.split()
                n, l, j, pn = int(data[2]), int(data[3]), int(data[4]), int(data[5])
                label = (n,l,j,pn)
            if( line[:51]==" 2xJf      Ef      2xJi     Ei       Ex       C^2*S" ):
                read=True
            else:
                if(read):
                    data = line.split()
                    if(len(data)==0):
                        read=False
                        espe[label] = energy
                        sum_sfs[label] = sum_sf
                        print("{:s}{:4d}{:4d}{:4d}{:4d}{:12.6f}".format(fn,*label,sum_sf))
                        energy = 0.0
                        sum_sf = 0.0
                        continue
                    i_bra = int(data[1][:-1])
                    i_ket = int(data[4][:-1])
                    en_bra = float(data[2]) + Hm_bra.get_0bme()
                    en_ket = float(data[5]) + Hm_ket.get_0bme()
                    if(N_states != None):
                        if(i_bra > N_states): continue
                        if(i_ket > N_states): continue
                    CS = float(data[7]) / (label[2]+1)
                    sum_sf += CS * (label[2]+1)
                    energy += CS * (en_bra - en_ket)
                else:
                    continue
        return espe, sum_sfs
