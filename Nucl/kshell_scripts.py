#!/usr/bin/env python3
def i2prty(i):
    if(i == 1): return '+'
    else: return '-'

def get_occupation(logs):
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
                prty = i2prty(prty)
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

def get_wf_index( fn_summary ):
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

def calc_density(kshl_dir, fn_snt, fn_ptn_bra, fn_ptn_ket, fn_wf_bra, fn_wf_ket, i_wfs=None, fn_density=None, \
        header="", batch_cmd=None, run_cmd=None, fn_input="transit.input", calc_SF=False):
    import os, sys, time, subprocess
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
    if(not calc_SF): prt += '  is_tbtd = .true.\n'
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

def run_kshell(kshl_dir, Nucl, fn_snt, valence_Z, valence_N, states, header="", batch_cmd=None, run_cmd=None, dim_cnt=False,
        hw_max_p=None, hw_max_n=None):
    import os, sys, time, subprocess
    fn_script = Nucl+ "_" + os.path.basename(os.path.splitext(fn_snt)[0])
    if(not os.path.isfile(fn_snt)):
        print(fn_snt, "not found")
        return
    unnatural=False
    if( states.find("-") != -1 and states.find("+")!=-1 ): unnatural=True
    f = open('ui.in','w')
    f.write('\n')
    f.write(fn_snt+'\n')
    f.write(str(valence_Z)+","+str(valence_N)+'\n')
    f.write(fn_script+'\n')
    f.write(states+'\n')
    if(hw_max_p==None): f.write('\n')
    else:
        f.write('2\n')
        f.write(str(hw_max_p)+'\n')
    if(unnatural):
        if(hw_max_n==None): f.write('\n')
        else:
            f.write('2\n')
            f.write(str(hw_max_n)+'\n')
    f.write('beta_cm=0\n')
    f.write('mode_lv_hdd=0\n')
    f.write('\n')
    f.write('\n')
    f.write('\n')
    f.close()
    cmd = 'python '+kshl_dir+'/kshell_ui.py < ui.in'
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
            cmd = 'python ' + kshl_dir+'/count_dim.py ' + fn_snt + ' ' + fn_script + '_p.ptn'
            print(cmd)
            subprocess.call(cmd, shell=True)
        if( os.path.exists( fn_script+'_n.ptn' ) ):
            cmd = 'python ' + kshl_dir+'/count_dim.py ' + fn_snt + ' ' + fn_script + '_n.ptn'
            subprocess.call(cmd, shell=True)
    else:
        fn_script += ".sh"
        os.chmod(fn_script, 0o755)
        if(batch_cmd == None): cmd = "./" + fn_script
        if(batch_cmd != None): cmd = batch_cmd + " " + fn_script
        subprocess.call(cmd, shell=True)

    time.sleep(1)

def run_kshell_lsf(kshl_dir, fn_snt, fn_ptn_init, fn_ptn, fn_wf, fn_wf_out, J2, \
        op=None, fn_input=None, n_vec=100, header="", batch_cmd=None, run_cmd=None, \
        fn_operator=None, operator_irank=0, operator_nbody=1, operator_iprty=1):
    import os, sys, time, subprocess
    fn_script = os.path.basename(os.path.splitext(fn_wf_out)[0]) + ".sh"
    fn_out = "log_" + os.path.basename(os.path.splitext(fn_wf_out)[0]) + ".txt"
    if(fn_input==None): fn_input = os.path.basename(os.path.splitext(fn_wf_out)[0]) + ".input"
    if(op==None and fn_operator==None):
        print("Put either op or fn_operator")
        return
    if(op!=None and fn_operator!=None):
        print("You cannot put both op and fn_operator")
        return
    if(not os.path.isfile(fn_snt)):
        print(fn_snt, "not found")
        return
    cmd = "cp " + kshl_dir + "/kshell.exe ./"
    subprocess.call(cmd,shell=True)
    prt = header + '\n'
    prt += 'echo "start runnning ' + fn_out + ' ..."\n'
    prt += 'cat >' + fn_input + ' <<EOF\n'
    prt += '&input\n'
    prt += '  fn_int   = "' + fn_snt + '"\n'
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
