#!/usr/bin/env python3
import os, sys, time, subprocess, re, itertools
import numpy as np
import pandas as pd
if(__package__==None or __package__==""):
    import PeriodicTable
    import Operator
    import TransitionDensity
else:
    from . import PeriodicTable
    from . import Operator
    from . import TransitionDensity

def _i2prty(i):
    if(i == 1): return '+'
    elif(i ==-1): return '-'
    else:
        print("Error in _i2prty()")
        return None

def _prty2i(prty):
    if(prty == "+"): return 1
    elif(prty=="-"): return -1
    else:
        print("Error in _prty2i()")
        return None

def _PosNeg2i(prty):
    if(prty == "p"): return 1
    elif(prty=="n"): return -1
    else:
        print("Error in _PosNeg2i()")
        return None

def _none_check(var, var_name):
    """
    Just check if the variable is None or not.
    """
    if(var==None):
        print("{:s} can't be None.".format(var_name))
        return True
    else:
        return False

def _file_exists(fn):
    if(os.path.exists(fn)):
        return False
    else:
        print("File not found, {:s}".format(fn))
        return True

def _ZNA_from_str(Nucl):
    """
    ex.) Nucl="O16" -> Z=8, N=8, A=16
    """
    isdigit = re.search(r'\d+', Nucl)
    A = int( isdigit.group() )
    asc = Nucl[:isdigit.start()] + Nucl[isdigit.end():]
    asc = asc.lower()
    asc = asc[0].upper() + asc[1:]
    Z = PeriodicTable.periodic_table.index(asc)
    N = A-Z
    return Z, N, A

def _str_to_state(string):
    """
    0+1 -> ('0', '+', 1)
    1+1 -> ('1', '+', 1)
    0.5+1 -> ('1/2', '+', 1)
    0.5-1 -> ('1/2', '-', 1)
    ...
    """
    if( string.find("+") != -1 ):
        J2 = int( 2*float( string.split("+")[0] ))
        nth = int( string.split("+")[1] )
        prty = "+"
    if( string.find("-") != -1 ):
        J2 = int( 2*float( string.split("-")[0] ))
        nth = int( string.split("-")[1] )
        prty = "-"
    if( J2%2==0 ): return (str(J2//2), prty, nth)
    if( J2%2==1 ): return (str(J2)+"/2", prty, nth)

def _str_J_to_Jfloat(string):
    """
    '0' -> 0
    '1' -> 1
    '1/2' -> 0.5
    '3/2' -> 1.5
    """
    if(string.find("/")!=-1): return float(string[:-2])*0.5
    return float(string)

def _Jfloat_to_str(J):
    """
    0 -> '0'
    1 -> '1'
    0.5 -> '1/2'
    1.5 -> '3/2'
    """
    if( int(2*J+0.01)%2 == 0): return str(J)
    if( int(2*J+0.01)%2 == 1): return str(int(2*J))+"/2"

def _str_to_state_Jfloat(string):
    """
    0+1 -> (0, '+', 1)
    1+1 -> (1, '+', 1)
    0.5+1 -> (0.5, '+', 1)
    0.5-1 -> (0.5, '-', 1)
    ...
    """
    if( string.find("+") != -1 ):
        J = float( string.split("+")[0] )
        nth = int( string.split("+")[1] )
        prty = "+"
    if( string.find("-") != -1 ):
        J = float( string.split("-")[0] )
        nth = int( string.split("-")[1] )
        prty = "-"
    return (J, prty, nth)

class kshell_scripts:
    def __init__(self, kshl_dir=None, fn_snt=None, Nucl=None, states=None, hw_truncation=None, ph_truncation=None, \
            run_args=None, verbose=False):
        """
        kshl_dir: path to KSHELL exe file directory
        fn_snt: file name of the interaction file, snt file
        Nucl: target nucleid you want to calculate. ex) "O18"
        states: string specifying the states you want to calculate.
            ex) "+10,-10" means 10 positive parity states and 10 negative parity states
                "0.5+2,1.5-2,2.5+6" means two 1/2+ states, two 3/2- states, and 6 5/2+ states
        hw_truncation: int
        ph_truncation: "(oribit index)_(min occ)_(max occ)-(orbit index)_(min)_(max)-..."
            note: it is also possible to do like "(orbit index1)_(orbit index2)_(orbit index3)_..._(min occ)_(max occ)".
            Here, (min occ) and (max occ) are truncation of sum of the occupations of the orbits.
        run_args: additional arguments for kshell run
        """
        self.kshl_dir = kshl_dir
        self.Nucl = Nucl
        self.verbose=verbose
        if(Nucl != None): self.Z, self.N, self.A = _ZNA_from_str(self.Nucl)
        self.states = states
        self.hw_truncation=hw_truncation
        self.ph_truncation=ph_truncation
        self.plot_position=0
        self.run_args = {"beta_cm":0, "mode_lv_hdd":0}
        if(run_args!=None): self.run_args=run_args
        self.edict_previous={}
        self.fn_snt = fn_snt
        if(fn_snt != None and states != None): self.set_filenames()

    def get_x_position(self): return self.plot_position

    def set_filenames(self):
        if(self.Nucl == None): raise ValueError("Target nucleid is not defined!")
        if(self.fn_snt == None): raise ValueError("snt file is not defined!")
        if(self.states == None): raise ValueError("Target states are not defined!")
        self.fn_ptns = {}
        self.fn_wfs = {}
        for state in self.states.split(","):
            state_str = self._state_string(state)
            self.fn_ptns[state] = "{:s}_{:s}".format(self.Nucl, os.path.splitext(os.path.basename(self.fn_snt))[0])
            self.fn_wfs[state] = "{:s}_{:s}".format(self.Nucl, os.path.splitext(os.path.basename(self.fn_snt))[0])
            if(self.run_args != None):
                if( 'beta_cm' in self.run_args and self.run_args['beta_cm'] != 0):
                    self.fn_ptns[state] += "_betacm{:d}".format(self.run_args['beta_cm'])
                    self.fn_wfs[state] += "_betacm{:d}".format(self.run_args['beta_cm'])
            if(self.hw_truncation!=None):
                self.fn_ptns[state] += "_hw{:d}".format(self.hw_truncation)
                self.fn_wfs[state] += "_hw{:d}".format(self.hw_truncation)
            if(self.ph_truncation!=None):
                self.fn_ptns[state] += "_ph{:s}".format(self.ph_truncation)
                self.fn_wfs[state] += "_ph{:s}".format(self.ph_truncation)
            self.fn_ptns[state] += "_{:s}.ptn".format(state_str[-1])
            self.fn_wfs[state] += "_{:s}.wav".format(state_str)

    def set_options(self, set_filenames=True, **kwargs):
        if('fn_snt' in kwargs): self.fn_snt = kwargs['fn_snt']
        if('Nucl' in kwargs):
            self.Nucl = kwargs['Nucl']
            self.Z, self.N, self.A = _ZNA_from_str(self.Nucl)
        if('run_args' in kwargs): self.run_args = kwargs['run_args']
        if('hw_truncation' in kwargs): self.hw_truncation = kwargs['hw_truncation']
        if('ph_truncation' in kwargs): self.ph_truncation = kwargs['ph_truncation']
        if(set_filenames): self.set_filenames()

    def set_snt_file(self, fn_snt, set_other_files=True):
        self.fn_snt = fn_snt
        if(set_other_files): self.set_filenames()

    def set_truncations(self, hw_truncation=None, ph_truncation=None, set_other_files=True):
        self.hw_truncation = hw_truncation
        self.ph_truncation = ph_truncation
        if(set_other_files): self.set_filenames()

    def set_nucl(self, nucl, set_other_files=True):
        self.Nucl = nucl
        self.Z, self.N, self.A = _ZNA_from_str(self.Nucl)
        if(set_other_files): self.set_filenames()

    def set_run_args(self, run_args):
        self.run_args = run_args

    def get_wf_index( self, fn_summary="", use_logs=False ):
        jpn_to_idx = {}
        logs = set()
        idxs = {}
        if(use_logs):
            data = self.get_occupation()
            for key in data.keys():
                fn_log = data[key][1]
                if( fn_log in logs ):
                    idxs[ fn_log ] += 1
                else:
                    idxs[ fn_log ] = 1
                    logs.add( fn_log )
                jpn_to_idx[key] = (fn_log, idxs[ fn_log ])
        elif(fn_summary!=""):
            f = open( fn_summary, "r" )
            lines = f.readlines()
            f.close()
            for line in lines[5:]:
                dat = line.split()
                if( len(dat) == 0 ): continue
                if( dat[-1] in logs ):
                    idxs[ dat[-1] ] += 1
                else:
                    idxs[ dat[-1] ] = 1
                    logs.add( dat[-1] )
                jpn_to_idx[(dat[1],dat[2],int(dat[3]))] = (dat[-1], idxs[ dat[-1] ])
        else:
            raise ValueError()
        return jpn_to_idx

    def sort_levels(self, levels, thresh=1.e-4, **kwargs):
        """
        Remove the identical states
        """
        levels = sorted(levels.items(), key=lambda x:x[1])
        en = []
        remap_levels = {}
        current_i = {}
        for i, level in enumerate(levels):
            if(i==0):
                en.append(level)
                current_i[(level[0][0], level[0][1])] = 1
            else:
                level_prev = en[-1]
                if(level[0][0]==level_prev[0][0] and level[0][1]==level_prev[0][1] and abs(level[1]-level_prev[1]) < thresh): continue
                if(not (level[0][0], level[0][1]) in current_i): current_i[(level[0][0], level[0][1])] = 1
                else: current_i[(level[0][0], level[0][1])] += 1
                en.append(((level[0][0],level[0][1],current_i[(level[0][0],level[0][1])]), level[1]))
            remap_levels[level[0]] = (level[0][0],level[0][1],current_i[(level[0][0],level[0][1])])
        levels = {}
        for level in en:
            levels[level[0]] = level[1]
        if(len(kwargs)==0): return levels
        res = []
        res.append(levels)
        if('jpn_to_idx' in kwargs):
            tmp = {}
            for state in kwargs['jpn_to_idx'].keys():
                if(not state in remap_levels): continue
                tmp[remap_levels[state]] = kwargs['jpn_to_idx'][state]
            res.append(tmp)
        if('occ' in kwargs):
            tmp = {}
            for state in kwargs['occ'].keys():
                if(not state in remap_levels): continue
                tmp[remap_levels[state]] = kwargs['occ'][state]
            res.append(tmp)
        return res

    def wfname_from_state(self, state):
        """
        return the wave function name of the specified state.
        state: ex: ('0','+',1), ('1/2','+',1), so J is string not doubled
        """
        wf_labels = self.get_wf_index(use_logs=True)
        fn_log = wf_labels[state][0]
        fn_wav = fn_log.split("log_")[1].split(".txt")[0]+".wav"
        return fn_wav

    def _number_of_states(self, state):
        """
        example:
        0+1 -> 1
        0+2 -> 2
        2+2 -> 2
        0.5-4 -> 4
        1.5-10 -> 10
        +10 -> 10
        -5 -> 5
        """
        if( state.find("+")!=-1): return int(state.split("+")[-1])
        if( state.find("-")!=-1): return int(state.split("-")[-1])

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
        if(state.find("+")!=-1): _str = state.split("+")
        if(state.find("-")!=-1): _str = state.split("-")
        if(len(_str)!=2): raise ValueError("Input value is not correct: "+state)
        try:
            n = int(_str[1])
        except:
            raise ValueError("Input value is not correct: "+state)
        if(_str[0] == ""):
            if( state.find("+")!=-1): state_str = "m0p"
            if( state.find("-")!=-1): state_str = "m0n"
            if(self.A%2==1):
                if( state.find("+")!=-1): state_str = "m1p"
                if( state.find("-")!=-1): state_str = "m1n"
        elif(_str != ""):
            try:
                j_double = int(2*float(_str[0]))
            except:
                raise ValueError("Input value is not correct: "+state)
            if( state.find("+")!=-1): state_str = "j{:d}p".format(j_double)
            if( state.find("-")!=-1): state_str = "j{:d}n".format(j_double)
        return state_str

    def logs_to_dictionary(self, logs=None, isospin=False):
        H = Operator()
        H.read_operator_file(self.fn_snt,A=self.A)
        if(logs==None):
            logs = []
            states = self.states.split(",")
            for state in states:
                state_str = self._state_string(state)
                log = "log_{:s}.txt".format(os.path.splitext( os.path.basename(self.fn_wfs[state]))[0])
                logs.append(log)
        e_data = {}
        Njpi = {}
        for log in logs:
            if(not os.path.exists(log)):
                print(f"{log} is not found")
                continue
            f = open(log,"r")
            while True:
                line = f.readline()
                if(not line): break
                dat = line.split()
                if(len(dat) < 2): continue
                if(dat[1] == "<H>:"):
                    dat = line.split()
                    n_eig= int(dat[0])
                    ene  = float(dat[2]) + H.get_0bme()
                    J = dat[6]
                    if(self.A%2==0): J = str(int(dat[6][:-2])//2)
                    prty = int(dat[8])
                    prty = _i2prty(prty)
                    if(not (J,prty) in Njpi): Njpi[(J,prty)]=1
                    else: Njpi[(J,prty)]+=1
                    line = f.readline()
                    TT = float(line.split()[1])
                    if(not isospin): e_data[(J,prty,Njpi[(J,prty)])] = ene
                    if(isospin): e_data[(J,prty,Njpi[(J,prty)])] = (ene, -0.5 + 0.5*np.sqrt(1+4*TT))
            f.close()
        return e_data

    def get_occupation(self, logs=None, hw_ex=False):
        #fn_summary = self.summary_filename()
        H = Operator()
        H.read_operator_file(self.fn_snt,A=self.A)
        if(logs==None):
            logs = []
            states = self.states.split(",")
            for state in states:
                state_str = self._state_string(state)
                log = "log_{:s}.txt".format(os.path.splitext( os.path.basename(self.fn_wfs[state]))[0])
                logs.append(log)
        e_data = {}
        Njpi = {}
        for log in logs:
            if(not os.path.exists(log)):
                print(f"{log} is not found")
                continue
            f = open(log,"r")
            while True:
                line = f.readline()
                if(not line): break
                dat = line.split()
                if(len(dat) < 2): continue
                if(dat[1] == "<H>:"):
                    dat = line.split()
                    n_eig= int(dat[0])
                    ene  = float(dat[2]) + H.get_0bme()
                    mtot = int(dat[6][:-2])
                    J = dat[6]
                    if(self.A%2==0): J = str(int(dat[6][:-2])//2)
                    prty = int(dat[8])
                    prty = _i2prty(prty)
                    if(not (J,prty) in Njpi): Njpi[(J,prty)]=1
                    else: Njpi[(J,prty)]+=1
                    hws = None
                    while ene in e_data: ene += 0.000001
                    line = f.readline()
                    dat = line.split()
                    if(dat[0]=="<Hcm>:"): tt = int(dat[5][:-2])
                    if(dat[0]=="<TT>:"): tt = int(dat[3][:-2])
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
                    if(hw_ex):
                        while len(line)!=0:
                            line = f.readline()
                            data = line.split()
                            if(line[0:4] ==" hw:"):
                                hws = {}
                                for i in range(len(data)-1):
                                    hw, prob = data[i+1].split(":")
                                    hws[int(hw)] = float(prob)
                                break
                    if(hws!=None): e_data[ (J,prty,Njpi[(J,prty)]) ] = (ene, log, tt, plist, nlist, hws)
                    if(hws==None): e_data[ (J,prty,Njpi[(J,prty)]) ] = (ene, log, tt, plist, nlist)
            f.close()
        return e_data

    def run_kshell(self, header="", batch_cmd=None, run_cmd=None, dim_cnt=False, gen_partition=False, fn_script=None, dim_thr=None, python_version='python3', run_script=True):
        """
        header: string, specifying the resource allocation.
        batch_cmd: string, command submitting jobs (this can be None) ex.) "qsub"
        run_cmd: string, command to run a job (this can be None) ex.) "srun"
        dim_cnt: switch for dimension count mode
        gen_partition: switch for only generating the partition file
        fn_script: string, file name of the script (this is optional)
        """
        if(fn_script==None):
            fn_script = "{:s}_{:s}".format(self.Nucl, os.path.splitext(os.path.basename(self.fn_snt))[0])
            if(self.run_args != None):
                if( 'beta_cm' in self.run_args and self.run_args['beta_cm'] != 0): fn_script += "_betacm{:d}".format(self.run_args['beta_cm'])
            if(self.hw_truncation != None): fn_script += "_hw" + str(self.hw_truncation)
            if(self.ph_truncation != None): fn_script += "_ph" + str(self.ph_truncation)
        if(not os.path.isfile(self.fn_snt)):
            print(self.fn_snt, "not found")
            return
        if(python_version=='python2'):
            print('The current default python version is python2. Although the calculations will be done properly, this is not cool. In the future, the defalut version will be switched to python3.')
            print('If you do not want to see this warning, change python_version="python3" at line 453 in kshell_scripts.py')
        unnatural=False
        if( self.states.find("-") != -1 and self.states.find("+")!=-1 ): unnatural=True
        f = open('ui.in','w')
        f.write('\n')
        f.write(self.fn_snt+'\n')
        f.write(self.Nucl+'\n')
        f.write(fn_script+'\n')
        f.write(self.states+'\n')
        if(self.hw_truncation==None and self.ph_truncation==None): f.write('\n')
        if(self.hw_truncation==None and self.ph_truncation!=None):
            f.write('1\n')
            for tr in self.ph_truncation.split("-"):
                strs = tr.split("_")
                f.write(" ".join(strs[:-2])+'\n')
                f.write(strs[-2]+" "+strs[-1]+'\n')
            f.write('\n')
        if(self.hw_truncation!=None and self.ph_truncation==None):
            f.write('2\n')
            f.write(str(self.hw_truncation)+'\n')
        if(self.hw_truncation!=None and self.ph_truncation!=None):
            f.write('3\n')
            f.write(str(self.hw_truncation)+'\n')
            for tr in self.ph_truncation.split("-"):
                strs = tr.split("_")
                f.write(strs[0]+'\n')
                f.write(strs[1]+" "+strs[2]+'\n')
                f.write(" ".join(strs[:-2])+'\n')
                f.write(strs[-2]+" "+strs[-1]+'\n')
            f.write('\n')
        if(unnatural):
            if(self.hw_truncation==None and self.ph_truncation==None): f.write('\n')
            if(self.hw_truncation==None and self.ph_truncation!=None):
                f.write('1\n')
                for tr in self.ph_truncation.split("-"):
                    strs = tr.split("_")
                    f.write(" ".join(strs[:-2])+'\n')
                    f.write(strs[-2]+" "+strs[-1]+'\n')
                f.write('\n')
            if(self.hw_truncation!=None and self.ph_truncation==None):
                f.write('2\n')
                f.write(str(self.hw_truncation)+'\n')
            if(self.hw_truncation!=None and self.ph_truncation!=None):
                f.write('3\n')
                f.write(str(self.hw_truncation)+'\n')
                for tr in self.ph_truncation.split("-"):
                    strs = tr.split("_")
                    f.write(" ".join(strs[:-2])+'\n')
                    f.write(strs[-2]+" "+strs[-1]+'\n')
                f.write('\n')
        if(self.run_args!=None):
            for key in self.run_args.keys():
                f.write('{:s}={:s}\n'.format(key, str(self.run_args[key])))
        f.write('\n')
        f.write('\n')
        f.write('\n')
        f.close()
        if(self.verbose): cmd = python_version + ' '+self.kshl_dir+'/kshell_ui.py < ui.in'
        if(not self.verbose): cmd = python_version + ' '+self.kshl_dir+'/kshell_ui.py < ui.in silent'
        subprocess.call(cmd, shell=True)
        subprocess.call("rm ui.in",shell=True)
        if(not self.verbose): subprocess.call("rm save_input_ui.txt",shell=True)
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

        if(gen_partition): return
        if( dim_cnt or dim_thr!=None):
            res = []
            for fn_ptn in self.fn_ptns.values():
                cmd = python_version +' ' + self.kshl_dir+'/count_dim.py ' + self.fn_snt + ' ' + fn_ptn
                #subprocess.call(cmd, shell=True)
                process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                res.append(process.communicate()[0])
            if(dim_cnt): return res
            if(dim_thr!=None and res!=None):
                if(res[0].decode('utf-8') != ''):
                    line = res[0].decode('utf-8').split('\n')[-2]
                    dimp = float(line.split()[2])
                    if(dimp > dim_thr):
                        print(f'I stop because dimension might be too large: {dimp:.2e}')
                        print(f'Make sure that you are submitting the job with multi nodes and increase dim_thr.')
                        return
        fn_script += ".sh"
        os.chmod(fn_script, 0o755)
        if run_script:
            if(batch_cmd == None): cmd = "./" + fn_script
            if(batch_cmd != None): cmd = batch_cmd + " " + fn_script
            subprocess.call(cmd, shell=True)
            if(batch_cmd != None): time.sleep(1)
        return fn_script

    def run_kshell_lsf(self, fn_ptn_init, fn_ptn, fn_wf, fn_wf_out, J2, \
            op=None, fn_input=None, n_vec=100, header="", batch_cmd=None, run_cmd=None, \
            fn_operator=None, operator_irank=0, operator_nbody=1, operator_iprty=1, neig_load_wave=1, \
            need_converged_vec=False, T2_projection=None, maxiter=None):
        """
        This is for Lanczos strength function method. |v1> = Op |v0> and do Lanczos starting from |v1>

        fn_ptn_init: string, partition file for the initial state |v0>
        fn_ptn: string, partition file for the state |v1>
        fn_wf: string, input wave function |v0> file name
        fn_wf_out: string, output wave function file name
        J2: int, twice of the angular momentum of the |v1> state.
        op: string, operator name defined in KSHELL
        fn_input: string, file name for the fortran namelist
        n_vec: int, the number of states you want to calculate.
        header: string, specifying the resource allocation.
        batch_cmd: string, command submitting jobs (this can be None) ex.) "qsub"
        run_cmd: string, command to run a job (this can be None) ex.) "srun"
        fn_operator: string, operator file name, instead of op, you can use your own operator to generate |v1>
            CAUTION, at the moment KSHELL will use only the one-body part of the operator
        operator_irank: int, angular momentum rank of Op
        operator_iprty: int, parity of Op
        operator_nbody: int, a KSHELL intrinsic number.
        maxiter: number of iterations
            from KSHELL operator_jscheme.f90
            !  nbody =  0   copy
            !           1   one-body int. cp+ cp,   cn+ cn
            !           2   two-body int. c+c+cc
            !           5   two-body transition density (init_tbtd_op, container)
            !          10   one-body transition density (init_obtd_beta, container)
            !          11   two-body transition density for cp+ cn type
            !          12   two-body transition density for cn+ cp type
            !          13   two-body transition density for cp+ cp+ cn cn (init_tbtd_ppnn)
            !          -1   cp+     for s-factor
            !          -2   cn+     for s-factor
            !          -3   cp+ cp+ for 2p s-factor
            !          -4   cn+ cn+ for 2n s-factor
            !          -5   cp+ cn+ reserved, NOT yet available
            !          -6   cp      not available
            !          -7   cn      not available
            !          -10  cp+ cn  for beta decay
            !          -11  cn+ cp  for beta decay (only in set_ob_channel)
            !          -12  cp+ cp+ cn cn  for 0v-bb decay
            !          -13  cn+ cn+ cp cp  for 0v-bb decay (not yet used)
            !          -14  cp+ cn  for beta decay 1+2
            !          -15  cn+ cp  for beta decay 1+2
        """
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
        if(header==""):
            header = "#!/bin/sh\n"
            header+= "export OMP_STACKSIZE=1g\n"
            header+= "export GFORTRAN_UNBUFFERED_PRECONNECTED=y\n"
            header+= "# ulimit -s unlimited\n"

        cmd = "cp " + self.kshl_dir + "/kshell.exe ./"
        subprocess.call(cmd,shell=True)
        prt = header + '\n'
        #prt += 'echo "start runnning ' + fn_out + ' ..."\n'
        prt += 'cat >' + fn_input + ' <<EOF\n'
        prt += '&input\n'
        prt += '  fn_int   = "' + self.fn_snt + '"\n'
        prt += '  fn_ptn = "' + fn_ptn + '"\n'
        prt += '  fn_ptn_init = "' + fn_ptn_init + '"\n'
        prt += '  fn_load_wave = "' + fn_wf + '"\n'
        prt += '  fn_save_wave = "' + fn_wf_out + '"\n'
        prt += '  n_eigen = '+str(n_vec)+'\n'
        prt += '  mtot = '+str(J2)+'\n'
        prt += '  neig_load_wave = '+str(neig_load_wave)+'\n'
        prt += '  is_double_j = .true.\n'
        if(not need_converged_vec):
            prt += '  max_lanc_vec = '+str(n_vec)+'\n'
            prt += '  n_restart_vec = '+str(min(n_vec,200))+'\n'
            if(maxiter == None): prt += '  maxiter = 1\n'
            if(maxiter != None): prt += '  maxiter = '+str(maxiter)+'\n' # put 0 if you want to calculate only |v> = Op|vn>
        if(T2_projection!=None): prt += '  tt_proj = '+str(T2_projection)+'\n'
        if(op!=None): prt += '  op_type_init = "'+str(op)+'"\n'
        if(fn_operator!=None):
            prt += '  fn_op_init_wf = "'+str(fn_operator)+'"\n'
            prt += '  irank_op_init_wf = '+str(operator_irank)+'\n'
            prt += '  nbody_op_init_wf = '+str(operator_nbody)+'\n'
            prt += '  iprty_op_init_wf = '+str(1+(1-operator_iprty)//2)+'\n'
        prt += '  eff_charge = 1.0, 0.0\n'
        prt += '  e1_charge =' +str(self.N/self.A)+ ', '+str(-self.Z/self.A) +'\n'
        if(self.run_args!=None):
            for key in self.run_args.keys():
                prt += '{:s}={:s}\n'.format(key, str(self.run_args[key]))
        prt += '&end\n'
        prt += 'EOF\n'
        if(run_cmd == None): prt += './kshell.exe ' + fn_input + ' > ' + fn_out + ' 2>&1\n'
        if(run_cmd != None): prt += run_cmd + ' ./kshell.exe ' + fn_input + ' > ' + fn_out + ' 2>&1\n\n\n'
        prt += 'rm -f tmp_snapshot_' + fn_ptn + "_" + str(J2) + "_* " + \
                'tmp_lv_' + fn_ptn + '_' + str(J2) + "_* " + \
                fn_input + '\n\n\n'
        prt += './collect_logs.py log_*' + self.basename() + '* > ' + \
                self.summary_filename() + '\n\n'
        f = open(fn_script,'w')
        f.write(prt)
        f.close()
        os.chmod(fn_script, 0o755)
        if(batch_cmd == None): cmd = "./" + fn_script
        if(batch_cmd != None): cmd = batch_cmd + " " + fn_script
        subprocess.call(cmd, shell=True)
        if(batch_cmd != None): time.sleep(1)

    def run_kshell_ias_lsf(self, initial_state=(0,"+",1), target_J2=0, target_prty="+", n_vec=1, \
            fn_operator=None, mode = "", T2_projection=None, \
            batch_cmd=None, run_cmd=None, header="", need_converged_vec=False, set_filename=False):
        H = Operator(filename=self.fn_snt)
        if(mode==""):
            Op = Operator(ms=H.ms, p_core=H.p_core, n_core=H.n_core)
            if(fn_operator==None):
                fn_operator = "Op_Num_"+os.path.basename(self.fn_snt)
                Op.set_number_op(normalization=(self.A-H.p_core-H.n_core))
                Op.reduce()
                Op.write_operator_file(fn_operator)
        elif(mode=="p<-n" or mode=="n<-p"):
            Op = Operator(ms=H.ms, rankZ=1, p_core=H.p_core, n_core=H.n_core)
            if(fn_operator==None):
                fn_operator = "Op_IAS_"+os.path.basename(self.fn_snt)
                Op.set_fermi_op()
                Op.write_operator_file(fn_operator)
        elif(mode=="pp<-nn" or mode=="nn<-pp"):
            Op = Operator(ms=H.ms, rankZ=2, p_core=H.p_core, n_core=H.n_core)
            if(fn_operator==None):
                fn_operator = "Op_DIAS_"+os.path.basename(self.fn_snt)
                Op.set_double_fermi_op()
                Op.write_operator_file(fn_operator)
        else:
            raise ValueError()
        if(mode==""):
            op_nbody = 1
            Nucl_target = PeriodicTable.periodic_table[self.Z]+str(self.A)
        elif(mode=="p<-n"):
            op_nbody = -14
            Nucl_target = PeriodicTable.periodic_table[self.Z+1]+str(self.A)
        elif(mode=="n<-p"):
            op_nbody = -15
            Nucl_target = PeriodicTable.periodic_table[self.Z-1]+str(self.A)
        elif(mode=="pp<-nn"):
            op_nbody = -12
            Nucl_target = PeriodicTable.periodic_table[self.Z+2]+str(self.A)
        elif(mode=="nn<-pp"):
            op_nbody = -13
            Nucl_target = PeriodicTable.periodic_table[self.Z-2]+str(self.A)
        else:
            raise ValueError()
        target_state_str = str(target_J2//2) + target_prty + str(n_vec)
        if(self.A%2==1): target_state_str = str(0.5*target_J2) + target_prty + str(n_vec)
        ksh_target = kshell_scripts(self.kshl_dir, self.fn_snt, Nucl_target, target_state_str,\
                hw_truncation=self.hw_truncation, ph_truncation=self.ph_truncation, \
                run_args=self.run_args)
        ksh_target.run_kshell(gen_partition=True)
        key = ""
        for _ in self.fn_ptns.keys():
            if(_[0] == "+" or _[0] == "-"): raise ValueError("Please run initital state calculation with J-scheme (double-lanczos)")
            _st = _str_to_state_Jfloat(_)
            if(_st[0] == initial_state[0] and _st[1] == initial_state[1]): key = _
        ksh_target.fn_wfs[target_state_str] = "LSF" + "_" + os.path.basename(ksh_target.fn_wfs[target_state_str])
        if(not set_filename): ksh_target.run_kshell_lsf(self.fn_ptns[key], ksh_target.fn_ptns[target_state_str], \
                self.fn_wfs[key], ksh_target.fn_wfs[target_state_str],\
                J2=target_J2, n_vec=n_vec, header=header, batch_cmd=batch_cmd, run_cmd=run_cmd, fn_operator=fn_operator, \
                operator_irank=0, operator_nbody=op_nbody, operator_iprty=1, neig_load_wave=initial_state[2],\
                T2_projection=T2_projection, need_converged_vec=need_converged_vec)
        return ksh_target

    def basename(self):
        return "{:s}_{:s}".format(self.Nucl, os.path.splitext(os.path.basename(self.fn_snt))[0])

    def summary_filename(self):
        fn_summary = "summary_{:s}_{:s}".format(self.Nucl, os.path.splitext(os.path.basename(self.fn_snt))[0])
        if(self.run_args != None):
            if( 'beta_cm' in self.run_args and self.run_args['beta_cm'] != 0): fn_summary += "_betacm{:d}".format(self.run_args['beta_cm'])
        if(self.hw_truncation != None): fn_summary += "_hw" + str(self.hw_truncation)
        if(self.ph_truncation != None): fn_summary += "_ph" + str(self.ph_truncation)
        fn_summary += ".txt"
        return fn_summary

    def lowest_from_logs(self):
        """
        output: J, prty, Energy
            J: string, angular momentum, should be like '0', '1/2', 3/2'
            prty: string, parity, should be '+' or '-'
            Energy: lowest energy in the summary file (no need to be the ground-state energy)
        """
        edict = self.logs_to_dictionary()
        if(edict == {}): return None, None, None
        levels = sorted(edict.items(), key=lambda x:x[1])
        return levels[0][0][0], levels[0][0][1], levels[0][1]


    def lowest_from_summary(self):
        """
        output: J, prty, Energy
            J: string, angular momentum, should be like '0', '1/2', 3/2'
            prty: string, parity, should be '+' or '-'
            Energy: lowest energy in the summary file (no need to be the ground-state energy)
        """
        edict = self.summary_to_dictionary()
        if(edict == {}): return None, None, None
        levels = sorted(edict.items(), key=lambda x:x[1])
        return levels[0][0][0], levels[0][0][1], levels[0][1]

    def energy_from_summary(self, state):
        """
        state: tuple of (J, prty, nth)
            J: string, angular momentum, should be like '0', '1/2', 3/2'
            prty: string, parity, should be '+' or '-'
            nth: int
        """
        edict = self.summary_to_dictionary()
        if(edict == {}): return None
        try:
            return edict[state]
        except:
            return None

    def summary_to_dictionary(self, comment_snt="!"):
        fn_summary = self.summary_filename()
        H = Operator()
        H.read_operator_file(self.fn_snt,comment=comment_snt,A=self.A)
        if(not os.path.exists(fn_summary)): return {}
        f = open(fn_summary,'r')
        lines = f.readlines()
        f.close()
        edict={}
        for line in lines:
            data = line.split()
            try:
                N = int(data[0])
                J = data[1]
                P = data[2]
                i = int(data[3])
                e = float(data[5])
                eex = float(data[6])
                edict[(J,P,i)] = e + H.get_0bme()
            except:
                continue
        return edict

    def plot_levels(self, ax, edict=None, \
            absolute=False, show_Jpi=False, connect=True, \
            bar_width=0.3, lw=1, window_size=4, color_mode="parity", \
            states=None):
        """
        Draw energy levels
        ax: matplotlib.axes, the axis you want to draw
        """
        if(edict==None): edict = self.summary_to_dictionary()
        self._plot_levels(ax, edict, \
                absolute=absolute, show_Jpi=show_Jpi, connect=connect, \
                bar_width=bar_width, lw=lw, window_size=window_size, \
                color_mode=color_mode, states=states)

    def set_Jpi_labels(self, ax, edict=None, absolute=False, lw=1, bar_width=0.3, window_size=4, color_mode="parity", states=None):
        if(edict==None): edict = self.summary_to_dictionary()
        if(edict=={}): return
        if(not absolute):
            tmp = edict
            Emin = np.inf
            for E in tmp.values():
                Emin = min(Emin, E)
            for key in tmp.keys():
                edict[key] = tmp[key]-Emin
        if(states != None):
            states_list = []
            for _ in states.split(","):
                J, prty, n = _str_to_state(_)
                for i in range(1,n+1):
                    states_list.append((J,prty,i))
        x = self.plot_position-1
        fs = 2 # fontsize is assumed to be 2 mm
        bbox = ax.get_window_extent()
        width, height = bbox.width, bbox.height # in pixel
        width *= 2.54/100 # in cm
        height *= 2.54/100 # in cm
        h = 10 * height / window_size # mm / MeV
        levels = sorted(edict.items(), key=lambda x:x[1])
        first = levels[0]
        key = first[0]
        y = first[1]
        label = "$"+key[0]+"^{"+key[1]+"}_{"+str(key[2])+"}$"
        ax.plot([x+bar_width,x+bar_width+0.2],[y,y],lw=0.8*lw,c=self._get_color(key,color_mode),ls=":")
        ax.annotate(label, xy=(x+bar_width+0.2,y), color=self._get_color(key,color_mode))
        if(absolute): y_back = y
        else: y_back = 0
        for i in range(1,len(levels)):
            level = levels[i]
            key = level[0]
            if(states!=None and (not key in states_list)): continue
            e = level[1]
            label = "$"+key[0]+"^{"+key[1]+"}_{"+str(key[2])+"}$"
            if((e-y)*h < fs): y+= (fs+0.2)/h
            else: y=e
            #print(key,f'{e:12.6f} {y_back:12.6f} {y:12.6f}')
            ax.plot([x+bar_width,x+bar_width+0.2],[e,y],lw=0.8*lw,c=self._get_color(key,color_mode),ls=":")
            ax.annotate(label, xy=(x+bar_width+0.2,y),color=self._get_color(key,color_mode))
            y_back=y

    def _plot_levels(self, ax, edict, \
            absolute=False, show_Jpi=False, connect=True, \
            bar_width=0.3, lw=1, window_size=4, color_mode="parity", states=None):
        if(states != None):
            states_list = []
            for _ in states.split(","):
                J, prty, n = _str_to_state(_)
                for i in range(1,n+1):
                    states_list.append((J,prty,i))
        if(not absolute):
            tmp = edict
            Emin = np.inf
            for E in tmp.values():
                Emin = min(Emin, E)
            for key in tmp.keys():
                edict[key] = tmp[key]-Emin
        x = self.plot_position
        for key in edict.keys():
            if(states!=None and (not key in states_list)): continue
            y = edict[key]
            ax.plot([x-bar_width,x+bar_width],[y,y],lw=lw,c=self._get_color(key,color_mode))
        if(connect and len(self.edict_previous)!=0):
            for key in self.edict_previous.keys():
                if(states!=None and (not key in states_list)): continue
                if(key in edict):
                    yl = self.edict_previous[key]
                    yr = edict[key]
                    ax.plot([x-1+bar_width,x-bar_width],[yl,yr],lw=0.8*lw,ls=":",c=self._get_color(key,color_mode))
        self.plot_position+=1
        if(show_Jpi): self.set_Jpi_labels(ax, edict, absolute=absolute, lw=lw, \
                bar_width=bar_width, window_size=window_size, color_mode=color_mode, \
                states=states)
        self.edict_previous=edict

    def _get_color(self, key, color_mode):
        color_list_p = ['red','salmon','orange','darkgoldenrod','gold','olive', 'lime','forestgreen','turquoise','teal','skyblue']
        color_list_n = ['navy','blue','mediumpurple','blueviolet','mediumorchid','purple','magenta','pink','crimson']
        if(key[0]=="-1" or key[0]=='?'): return "k"
        if(self.A%2==0): Jdouble = int(key[0])*2
        if(self.A%2==1): Jdouble = int(key[0][:-2])
        P = key[1]
        if(color_mode=="parity"):
            if(P=="+"): return "red"
            if(P=="-"): return "blue"
        if(color_mode=="grey"):
            return "grey"
        elif(color_mode=="spin_parity"):
            idx = int(Jdouble/2)
            if(P=="+"): return color_list_p[idx%len(color_list_p)]
            if(P=="-"): return color_list_n[idx%len(color_list_n)]

    def espe(self, states=None, bare=False):
        """
        state: list of tuple ex.) [('1/2', '+', 1), ('3/2', '+', 1), ...]
        """
        e_data = self.get_occupation()
        H = Operator()
        H.read_operator_file(self.fn_snt)
        if(states!=None):
            #wf_index = self.get_wf_index(self.summary_filename())
            wf_index = self.get_wf_index(use_logs=True)
            sts = [ (_[0],_[1],wf_index[_][-1]) for _ in states ]
        espes = {}
        for key, vals in e_data.items():
            if(states!=None):
                if(not key in sts): continue
            occs = {}
            for i in range(1,len(vals[3])+1):
                oi = H.ms.orbits.get_orbit(i)
                occs[oi.get_nljz()] = vals[3][i-1] / (oi.j+1)
            for i in range(1,len(vals[4])+1):
                oi = H.ms.orbits.get_orbit(i+len(vals[3]))
                occs[oi.get_nljz()] = vals[4][i-1] / (oi.j+1)
            espes[key] = H.espe(occs, bare=bare)
        return espes

    def espe_lowest_filling(self):
        """
        ESPE assuming the lowest filling wrt SPEs.
        """
        H = Operator()
        H.read_operator_file(self.fn_snt)
        spes_p = {}
        spes_n = {}
        for i in range(H.ms.orbits.get_num_orbits()):
            oi = H.ms.orbits.get_orbit(i)
            if(oi.z == -1): spes_p[(oi.n, oi.l, oi.j)] = H.get_1bme(i,i)
            if(oi.z ==  1): spes_n[(oi.n, oi.l, oi.j)] = H.get_1bme(i,i)
        spes_p = sorted(spes_p.items(), key=lambda x: x[1])
        spes_n = sorted(spes_n.items(), key=lambda x: x[1])
        occs = {}
        Z = self.Z - H.p_core
        for spe in spes_p:
            n, l, j = spe[0]
            if(Z<0):
                occs[(n, l, j, -1)] = 0
            else:
                occs[(n, l, j, -1)] = float(min(Z,j+1)) / float(j+1)
            Z -= (j + 1)
        N = self.N - H.n_core
        for spe in spes_n:
            n, l, j = spe[0]
            if(N<0):
                occs[(n, l, j, 1)] = 0
            else:
                occs[(n, l, j, 1)] = float(min(N,j+1)) / float(j+1)
            N -= (j + 1)
        espe = H.espe(occs)
        return espe

class transit_scripts:
    def __init__(self, kshl_dir=None, verbose=False, bin_output=False):
        self.kshl_dir = kshl_dir
        self.verbose = verbose
        self.filenames = {}
        self.bin_output = bin_output

    def set_filenames(self, ksh_l, ksh_r, states_list=None, calc_SF=False):
        self.filenames = {}
        if(states_list==None):
            states_list = [(x,y) for x,y in itertools.product( ksh_l.states.split(","), ksh_r.states.split(",") )]
        bra_side = ksh_l
        ket_side = ksh_r
        flip=False

        if( ksh_l.Z < ksh_r.Z ):
            bra_side = ksh_r
            ket_side = ksh_l
            flip=True
        if( ksh_l.A < ksh_r.A ):
            bra_side = ksh_r
            ket_side = ksh_l
            flip=True

        not_calculate = {}
        for states in states_list:
            state_l = states[0]
            state_r = states[1]
            if(flip):
                state_l = states[1]
                state_r = states[0]
            if(bra_side.Nucl == ket_side.Nucl and (state_r,state_l) in states_list):
                if((state_r,state_l) in not_calculate): continue
                not_calculate[(state_l,state_r)] = 0
            str_l = bra_side._state_string(state_l)
            str_r = ket_side._state_string(state_r)
            fn_density = "density"
            if(calc_SF): fn_density = "SF"
            fn_density += "_{:s}".format(os.path.splitext( os.path.basename( ket_side.fn_snt ) )[0])
            if(ket_side.hw_truncation!=None): fn_density += "_hw{:d}".format(ket_side.hw_truncation)
            if(ket_side.ph_truncation!=None): fn_density += "_ph{:s}".format(ket_side.ph_truncation)
            fn_density += "_{:s}{:s}_{:s}{:s}".format(bra_side.Nucl,str_l,ket_side.Nucl,str_r)
            if(not calc_SF and self.bin_output): fn_density += ".bin"
            else: fn_density += ".txt"
            self.filenames[(state_l,state_r)] = fn_density
        return flip

    def density_file_from_state(self, ksh_l, ksh_r, state_l, state_r, calc_SF=False):
        """
        return the density file file name using the left and right states
        state: ex: ('0','+',1), ('1/2','+',1), so J is string not doubled
        """
        bra_side = ksh_l
        ket_side = ksh_r
        flip=False
        if( ksh_l.Z < ksh_r.Z ):
            bra_side = ksh_r
            ket_side = ksh_l
            flip = True
        elif( ksh_l.A < ksh_r.A ):
            bra_side = ksh_r
            ket_side = ksh_l
            flip = True
        elif( state_l[1]=="-" and state_r[1]=="+"):
            ksh_l, ksh_r = ksh_r, ksh_l
            flip = True

        if(flip):
            wf_bra = bra_side.wfname_from_state(state_r)
            wf_ket = ket_side.wfname_from_state(state_l)
        else:
            wf_bra = bra_side.wfname_from_state(state_l)
            wf_ket = ket_side.wfname_from_state(state_r)
        str_l = wf_bra.split("_")[-1].split(".wav")[0]
        str_r = wf_ket.split("_")[-1].split(".wav")[0]
        fn_density = "density"
        if(calc_SF): fn_density = "SF"
        fn_density += "_{:s}".format(os.path.splitext( os.path.basename( ket_side.fn_snt ) )[0])
        if(ket_side.hw_truncation!=None): fn_density += "_hw{:d}".format(ket_side.hw_truncation)
        if(ket_side.ph_truncation!=None): fn_density += "_ph{:s}".format(ket_side.ph_truncation)
        fn_density += "_{:s}{:s}_{:s}{:s}".format(bra_side.Nucl,str_l,ket_side.Nucl,str_r)
        if(not calc_SF and self.bin_output): fn_density += ".bin"
        else: fn_density += ".txt"
        return fn_density, flip

    def calc_op_expvals(self, ksh_l, ksh_r, fn_op, states_list=None, header="", batch_cmd=None, run_cmd=None, \
            i_wfs=None, rankJ=0, rankP=1, rankZ=0, op_nbody=2, OpStr="Op"):
        """
        calculate < ksh_l | [a^t a] | ksh_r > and < ksh_l | [a^t a^t a a] | ksh_r >
        input:
            kshl_l, ksh_r (kshell_scripts class)
            state_list (list):
                ex.) [(0+2, 0+2), (0-1, 2-2)] -> <0+1|0+1>, <0+1|0+2>, <0+2|0+1>, <0+2|0+2>, <0-1|2-1>, <0-1|2-2>
                     [(0.5+1,1.5+1),] -> <0.5+1|1.5+1>
                     [(+2,+2),] -> <J+1|J+1>, <J+1|J+2>, <J+2|J+1>, <J+2|J+2>
            header (str): header specifying resource
            batch_cmd (str): 'sbatch', 'qsub', ...
            run_cmd (str): 'srun', ...
            i_fws (list): ex.) [(1,1),(2,2),(3,3),(4,4)]
        """
        if(states_list==None):
            states_list = [(x,y) for x,y in itertools.product( ksh_l.states.split(","), ksh_r.states.split(",") )]
        bra_side = ksh_l
        ket_side = ksh_r
        flip=False

        if( ksh_l.Z < ksh_r.Z ):
            bra_side = ksh_r
            ket_side = ksh_l
            flip=True
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
            if(_PosNeg2i(str_l[-1]) * _PosNeg2i(str_r[-1]) * rankP != 1): continue
            if( _file_exists(bra_side.fn_ptns[state_l]) or  _file_exists(ket_side.fn_ptns[state_r]) or \
                    _file_exists(bra_side.fn_wfs[state_l]) or  _file_exists(ket_side.fn_wfs[state_r])):
                continue
            fn_out = OpStr
            fn_out += "_{:s}".format(os.path.splitext( os.path.basename( ket_side.fn_snt ) )[0])
            if(ket_side.hw_truncation!=None): fn_out += "_hw{:d}".format(ket_side.hw_truncation)
            if(ket_side.ph_truncation!=None): fn_out += "_ph{:s}".format(ket_side.ph_truncation)
            fn_out += "_{:s}{:s}_{:s}{:s}".format(bra_side.Nucl,str_l,ket_side.Nucl,str_r)
            fn_out += ".txt"

            fn_script = os.path.splitext(fn_out)[0] + ".sh"
            fn_input = os.path.splitext(fn_out)[0] + ".input"
            cmd = "cp " + self.kshl_dir + "/transit.exe ./"
            subprocess.call(cmd,shell=True)
            prt = header + '\n'
            prt += 'cat >' + fn_input + ' <<EOF\n'
            prt += '&input\n'
            prt += '  fn_int   = "' + ket_side.fn_snt + '"\n'
            prt += '  fn_ptn_l = "' + bra_side.fn_ptns[state_l]+ '"\n'
            prt += '  fn_ptn_r = "' + ket_side.fn_ptns[state_r]+ '"\n'
            prt += '  fn_load_wave_l = "' + bra_side.fn_wfs[state_l] + '"\n'
            prt += '  fn_load_wave_r = "' + ket_side.fn_wfs[state_r] + '"\n'
            prt += '  fn_op = "' + fn_op + '"\n'
            prt += '  irank_op = ' + str(rankJ) + '\n'
            prt += '  nbody_op = ' + str(op_nbody) + '\n'
            if(i_wfs!=None):
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
            if(batch_cmd != None): time.sleep(1)

    def calc_density(self, ksh_l, ksh_r, states_list=None, header="", batch_cmd=None, run_cmd=None, \
            i_wfs=None, calc_SF=False, parity_mix=True, run_script=True):
        """
        calculate < ksh_l | [a^t a] | ksh_r > and < ksh_l | [a^t a^t a a] | ksh_r >
        input:
            kshl_l, ksh_r (kshell_scripts class)
            state_list (list):
                ex.) [(0+2, 0+2), (0-1, 2-2)] -> <0+1|0+1>, <0+1|0+2>, <0+2|0+1>, <0+2|0+2>, <0-1|2-1>, <0-1|2-2>
                     [(0.5+1,1.5+1),] -> <0.5+1|1.5+1>
                     [(+2,+2),] -> <J+1|J+1>, <J+1|J+2>, <J+2|J+1>, <J+2|J+2>
            header (str): header specifying resource
            batch_cmd (str): 'sbatch', 'qsub', ...
            run_cmd (str): 'srun', ...
            i_fws (list): ex.) [(1,1),(2,2),(3,3),(4,4)]
            calc_SF (bool): switch for calculation of spectroscopic factor
            parity_mix (bool)
        """
        if(header==""):
            header = "#!/bin/sh\n"
            header+= "export OMP_STACKSIZE=1g\n"
            header+= "export GFORTRAN_UNBUFFERED_PRECONNECTED=y\n"
            header+= "# ulimit -s unlimited\n"
        if(states_list==None):
            states_list = [(x,y) for x,y in itertools.product( ksh_l.states.split(","), ksh_r.states.split(",") )]
        bra_side = ksh_l
        ket_side = ksh_r
        flip=False

        if( ksh_l.Z < ksh_r.Z ):
            bra_side = ksh_r
            ket_side = ksh_l
            flip=True
        if( ksh_l.A < ksh_r.A ):
            bra_side = ksh_r
            ket_side = ksh_l
            flip=True

        density_files = []
        not_calculate = {}
        for states in states_list:
            state_l = states[0]
            state_r = states[1]
            if(flip):
                state_l = states[1]
                state_r = states[0]
            if(bra_side.Nucl == ket_side.Nucl and (state_r,state_l) in states_list):
                if((state_r,state_l) in not_calculate): continue
                not_calculate[(state_l,state_r)] = 0
            str_l = bra_side._state_string(state_l)
            str_r = ket_side._state_string(state_r)
            if(not parity_mix and str_l[-1] != str_r[-1]): continue
            if run_script:
                if((_file_exists(bra_side.fn_ptns[state_l]) or  _file_exists(ket_side.fn_ptns[state_r]) or \
                        _file_exists(bra_side.fn_wfs[state_l]) or  _file_exists(ket_side.fn_wfs[state_r]))):
                    density_files.append(None)
                    continue
            fn_density_output = "none"
            fn_density = "density"
            if(calc_SF): fn_density = "SF"
            fn_density += "_{:s}".format(os.path.splitext( os.path.basename( ket_side.fn_snt ) )[0])
            if(ket_side.hw_truncation!=None): fn_density += "_hw{:d}".format(ket_side.hw_truncation)
            if(ket_side.ph_truncation!=None): fn_density += "_ph{:s}".format(ket_side.ph_truncation)
            fn_density += "_{:s}{:s}_{:s}{:s}".format(bra_side.Nucl,str_l,ket_side.Nucl,str_r)
            if(not calc_SF and self.bin_output): fn_density_output = fn_density + ".bin"
            fn_density += ".txt"

            if(not calc_SF and self.bin_output): density_files.append(fn_density_output)
            else: density_files.append(fn_density)
            fn_script = os.path.splitext(fn_density)[0] + ".sh"
            fn_input = os.path.splitext(fn_density)[0] + ".input"
            cmd = "cp " + self.kshl_dir + "/transit.exe ./"
            subprocess.call(cmd,shell=True)
            prt = header + '\n'
            #prt += 'echo "start runnning ' + fn_density + ' ..."\n'
            prt += 'cat >' + fn_input + ' <<EOF\n'
            prt += '&input\n'
            prt += '  fn_int   = "' + ket_side.fn_snt + '"\n'
            prt += '  fn_ptn_l = "' + bra_side.fn_ptns[state_l]+ '"\n'
            prt += '  fn_ptn_r = "' + ket_side.fn_ptns[state_r]+ '"\n'
            prt += '  fn_load_wave_l = "' + bra_side.fn_wfs[state_l] + '"\n'
            prt += '  fn_load_wave_r = "' + ket_side.fn_wfs[state_r] + '"\n'
            if(fn_density_output!='none'): prt += '  fn_density = "' + fn_density_output + '"\n'
            if(i_wfs!=None):
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
            if(run_script):
                if(batch_cmd == None): cmd = "./" + fn_script
                if(batch_cmd != None): cmd = batch_cmd + " " + fn_script
                subprocess.call(cmd, shell=True)
                if(batch_cmd != None): time.sleep(1)
            else:
                return fn_script
        if(run_script): return density_files, flip

    def calc_espe(self, kshl, snts=None, states_dest="+20,-20", header="", batch_cmd=None, run_cmd=None, step="full", mode="hole", N_states=None):
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
            kshl.run_kshell(header=header, batch_cmd=batch_cmd, run_cmd=run_cmd)
            for idx in range(min_idx,max_idx):
                fn_snt = snts[idx]
                if(idx==0): Z, N = kshl.Z-1, kshl.N
                if(idx==1): Z, N = kshl.Z, kshl.N-1
                if(idx==2): Z, N = kshl.Z+1, kshl.N
                if(idx==3): Z, N = kshl.Z, kshl.N+1
                Nucl = "{:s}{:d}".format(PeriodicTable.periodic_table[Z],Z+N)
                kshl_tr = kshell_scripts(kshl_dir=kshl.kshl_dir, fn_snt=fn_snt, Nucl=Nucl, states=states_dest)
                kshl_tr.run_kshell(header=header, batch_cmd=batch_cmd, run_cmd=run_cmd)
        if(step=="density" or step=="full"):
            for idx in range(min_idx,max_idx):
                fn_snt = snts[idx]
                if(idx==0): Z, N = kshl.Z-1, kshl.N
                if(idx==1): Z, N = kshl.Z, kshl.N-1
                if(idx==2): Z, N = kshl.Z+1, kshl.N
                if(idx==3): Z, N = kshl.Z, kshl.N+1
                Nucl = "{:s}{:d}".format(PeriodicTable.periodic_table[Z],Z+N)
                kshl_tr = kshell_scripts(kshl_dir=kshl.kshl_dir, fn_snt=fn_snt, Nucl=Nucl, states=states_dest)
                trs = transit_scripts(kshl_dir=kshl.kshl_dir,verbose=self.verbose)
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
            trs = transit_scripts(kshl_dir=kshl.kshl_dir,verbose=self.verbose)
            flip = trs.set_filenames(kshl, kshl_tr, calc_SF=True)
            if(flip):
                Hm_bra = Operator(filename = kshl_tr.fn_snt)
                Hm_ket = Operator(filename = kshl.fn_snt)
            else:
                Hm_bra = Operator(filename = kshl.fn_snt)
                Hm_ket = Operator(filename = kshl_tr.fn_snt)
            for key in trs.filenames.keys():
                fn = trs.filenames[key]
                if(not os.path.exists(fn)):
                    print("{:s} is not found!".format(fn))
                    continue
                if(idx==0 or idx==1): espe_each, sum_sf_each = trs.espe_sf_file(fn, Hm_bra, Hm_ket, "a^t a", N_states=N_states)
                if(idx==2 or idx==3): espe_each, sum_sf_each = trs.espe_sf_file(fn, Hm_bra, Hm_ket, "a a^t", N_states=N_states)
                for key in espe_each:
                    if( key in espe ):
                        espe[key] += espe_each[key]
                        sum_sf[key] += sum_sf_each[key]
                    else:
                        espe[key] = espe_each[key]
                        sum_sf[key] = sum_sf_each[key]
        return espe, sum_sf

    def read_sf_file(self, fn, mode="a^t a", N_states=None, Hm_bra=None, Hm_ket=None, ksh_bra=None, ksh_ket=None, type_output="DataFrame"):
        if(not os.path.exists(fn)):
            print("{:s} is not found!".format(fn))
            return None
        e0_bra, e0_ket = 0.0, 0.0
        if(Hm_bra != None): e0_bra = Hm_bra.get_0bme()
        if(Hm_ket != None): e0_ket = Hm_ket.get_0bme()
        idx_to_jpn_bra, idx_to_jpn_ket = {}, {}
        if(ksh_bra != None and ksh_ket != None):
            jpn_to_idx_bra = ksh_bra.get_wf_index(use_logs=True)
            jpn_to_idx_ket = ksh_ket.get_wf_index(use_logs=True)
            idx_to_jpn_bra = {v: k for k, v in jpn_to_idx_bra.items()}
            idx_to_jpn_ket = {v: k for k, v in jpn_to_idx_ket.items()}
            if(ksh_bra.A < ksh_ket.A):
                ksh_bra, ksh_ket = ksh_ket, ksh_bra
                idx_to_jpn_bra, idx_to_jpn_ket = idx_to_jpn_ket, idx_to_jpn_bra
        f = open(fn,'r')
        lines = f.readlines()
        f.close()
        if(type_output=="dict"): sfs = {}
        if(type_output=="DataFrame"): sfs = pd.DataFrame()
        read=False
        for line in lines:
            if( line.find('fn_load_wave_l')!=-1):
                fn_wfbra = line.split()[2]
                fn_logbra = f"log_{os.path.splitext(fn_wfbra)[0]}.txt"
            if( line.find('fn_load_wave_r')!=-1):
                fn_wfket = line.split()[2]
                fn_logket = f"log_{os.path.splitext(fn_wfket)[0]}.txt"
            if( line[:7] == "orbit :" ):
                data = line.split()
                n_sp, l_sp, j_sp, pn = int(data[2]), int(data[3]), int(data[4]), int(data[5])
                if(pn==-1): label = (n_sp,l_sp,f'{j_sp}/2','proton')
                if(pn== 1): label = (n_sp,l_sp,f'{j_sp}/2','neutron')
            if( line[:51]==" 2xJf      Ef      2xJi     Ei       Ex       C^2*S" ):
                read=True
            else:
                if(read):
                    data = line.split()
                    if(len(data)==0):
                        read=False
                        continue
                    i_bra = int(data[1][:-1])
                    i_ket = int(data[4][:-1])
                    J2_bra = int(data[0][:-1])
                    J2_ket = int(data[3][:-1])
                    en_bra = float(data[2])+e0_bra
                    en_ket = float(data[5])+e0_ket
                    if(N_states != None):
                        if(i_bra > N_states): continue
                        if(i_ket > N_states): continue
                    if(mode=="a^t a"):
                        CS = float(data[7]) / (j_sp+1)
                    elif(mode=="a a^t"):
                        CS = float(data[7]) / (j_sp+1) * (J2_bra+1)/(J2_ket+1)
                    if(len(idx_to_jpn_ket)==0):
                        if(type_output=="dict"): sfs[(*label,J2_bra,i_bra,J2_ket,i_ket)] = (CS*(j_sp+1), en_bra, en_ket)
                        if(type_output=="DataFrame"): sfs = pd.concat([sfs,pd.DataFrame([[*label,J2_bra,i_bra,J2_ket,i_ket,CS*(j_sp+1), en_bra, en_ket]])], ignore_index=True)
                    else:
                        label_bra = idx_to_jpn_bra[(fn_logbra,i_bra)]
                        label_ket = idx_to_jpn_ket[(fn_logket,i_ket)]
                        if(type_output=="dict"): sfs[(*label,ksh_bra.Nucl,*label_bra,ksh_ket.Nucl,*label_ket)] = (CS*(j_sp+1), en_bra, en_ket)
                        if(type_output=="DataFrame"): sfs = pd.concat([sfs,pd.DataFrame([[*label,ksh_bra.Nucl,*label_bra,ksh_ket.Nucl,*label_ket,CS*(j_sp+1), en_bra, en_ket]])], ignore_index=True)

                else:
                    continue
        if(len(idx_to_jpn_ket)!=0 and type_output=="DataFrame"): sfs.columns = ["n","l","j","p/n","Nucl. bra", "J bra","Parity bra","n bra","Nucl. ket", "J ket","Parity ket","n ket","CS^2","En bra","En ket"]
        return sfs

    def read_tsf_file(self, fn, mode="a^ta^t aa", N_states=None, Hm_bra=None, Hm_ket=None):
        if(not os.path.exists(fn)):
            print("{:s} is not found!".format(fn))
            return None
        e0_bra, e0_ket = 0.0, 0.0
        if(Hm_bra != None): e0_bra = Hm_bra.get_0bme()
        if(Hm_ket != None): e0_ket = Hm_ket.get_0bme()
        f = open(fn,'r')
        lines = f.readlines()
        f.close()
        if(type_output=="dict"): sfs = {}
        if(type_output=="DataFrame"): sfs = pd.DataFrame()
        read=False
        for line in lines:
            data = line.split()
            if( len(data)==6 and data[0].strip()=="2xJf"):
                read=True
            else:
                if(read):
                    data = line.split()
                    if(len(data)==0):
                        continue
                    if(data[0]=="TNA"):
                        a, b, J = int(data[3]), int(data[7]), int(data[11][:-1])
                        label = (a,b,J)
                        continue
                    if(data[0]=="total" and data[1]=="elapsed"): break
                    i_bra = int(data[1][:-1])
                    i_ket = int(data[4][:-1])
                    J2_bra = int(data[0][:-1])
                    J2_ket = int(data[3][:-1])
                    en_bra = float(data[2])+e0_bra
                    en_ket = float(data[5])+e0_ket
                    if(N_states != None):
                        if(i_bra > N_states): continue
                        if(i_ket > N_states): continue
                    if(mode=="a^ta^t aa"):
                        CS = float(data[7])**2 / (2*label[2]+1)
                    elif(mode=="aa a^ta^t"):
                        CS = float(data[7])**2 / (2*label[2]+1) * (J2_bra+1)/(J2_ket+1)
                    sfs[(*label,J2_bra,i_bra,J2_ket,i_ket)] = (CS * (2*label[2]+1), en_bra, en_ket)
                    if(type_output=="dict"): sfs[(*label,J2_bra,i_bra,J2_ket,i_ket)] = (CS * (2*label[2]+1), en_bra, en_ket)
                    if(type_output=="DataFrame"): sfs = pd.concat([sfs,pd.DataFrame([[*label,J2_bra,i_bra,J2_ket,i_ket,CS * (2*label[2]+1), en_bra, en_ket]])], ignore_index=True)
                else:
                    continue
        if(type_output=="DataFrame"): sfs.columns = ["p","q","Jpq", "J2 bra", "wflabel bra","J2 ket","wflabel ket","TNA","En bra","En ket"]
        return sfs

    def check_norm_tsf(self, fn, mode, target_spin, N_states=None, sfs=None):
        if(sfs==None): sfs = self.read_tsf_file(fn, mode, N_states)
        if(sfs==None): return
        n = 0
        for _ in sfs.keys():
            if(mode=="a^ta^t aa" and target_spin!=_[4]): n += sfs[_][0]
            if(mode=="aa a^ta^t" and target_spin!=_[6]): n += sfs[_][0]
        if(mode=="a^ta^t aa"): print("Number of occupied pairs in the model-space:   {:.2f}".format(n))
        if(mode=="aa a^ta^t"): print("Number of unoccupied pairs in the model-space: {:.2f}".format(n))

    def espe_sf_file(self, fn, Hm_bra, Hm_ket, mode, N_states=None):
        if(not os.path.exists(fn)):
            print("{:s} is not found!".format(fn))
            return None, None
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
                        if(self.verbose): print("{:s}{:4d}{:4d}{:4d}{:4d}{:12.6f}".format(fn,*label,sum_sf))
                        energy = 0.0
                        sum_sf = 0.0
                        continue
                    i_bra = int(data[1][:-1])
                    i_ket = int(data[4][:-1])
                    J2_bra = int(data[0][:-1])
                    J2_ket = int(data[3][:-1])
                    en_bra = float(data[2]) + Hm_bra.get_0bme()
                    en_ket = float(data[5]) + Hm_ket.get_0bme()
                    if(N_states != None):
                        if(i_bra > N_states): continue
                        if(i_ket > N_states): continue
                    if(mode=="a^t a"):
                        CS = float(data[7]) / (label[2]+1)
                    elif(mode=="a a^t"):
                        CS = float(data[7]) / (label[2]+1) * (J2_bra+1)/(J2_ket+1)
                    sum_sf += CS * (label[2]+1)
                    energy += CS * (en_bra - en_ket)
                else:
                    continue
        return espe, sum_sfs

    def exp_dag(self, fn_sfs, fn_dag, ksh_l, ksh_r, states):
        """  
        fn_sfs: S-factor file
        fn_dag: dagger operator file
        ksh_l, ksh_r: kshell_script class
        states: bra and ket states, [('0','+',1),('1/2','+',1)]
        """
        def read_dag(fn):
            fp = open(fn, 'r') 
            lines = fp.readlines()
            fp.close()

            dag = {} 
            for line in lines:
                if(line[0]=='!'): continue
                data = line.split()
                if(len(data) != 2): raise ValueError("a^t a^t a type operatror is not implemented yet...")
                dag[int(data[0])] = float(data[1])
            return dag

        state_l, state_r = states
        sfs = self.read_sf_file(fn_sfs, ksh_bra=ksh_l, ksh_ket=ksh_r)
        Ham = Operator(filename=ksh_l.fn_snt)
        orbits = Ham.ms.orbits
        dag = read_dag(fn_dag)
        val = 0
        for i in dag.keys():
            oi = orbits.get_orbit(i)
            if(oi.z==-1): pn = 'proton'
            if(oi.z== 1): pn = 'neutron'
            _ = sfs[(sfs['n']==oi.n) & (sfs['l']==oi.l) & (sfs['j']==f'{oi.j}/2') & (sfs['p/n']==pn) & 
                    (sfs['J bra']==state_l[0]) & (sfs['Parity bra']==state_l[1]) & (sfs['n bra']==state_l[2]) &
                    (sfs['J ket']==state_r[0]) & (sfs['Parity ket']==state_r[1]) & (sfs['n ket']==state_r[2])]
            if(_.empty): continue
            amp = np.sqrt(float(_['CS^2']))
            val += amp * dag[i] / np.sqrt(oi.j+1)
        return val


    def wf_overlap(self, ksh_l, ksh_r, header="", batch_cmd=None, run_cmd=None):
        Opl = Operator(filename=ksh_l.fn_snt)
        Opr = Operator(filename=ksh_r.fn_snt)
        if(Opl.ms.orbits.get_num_orbits() != Opr.ms.orbits.get_num_orbits()):
            print("The single-particle orbits definition has to be the same in left and right snt files.")
            return None
        for i in range(1,Opl.ms.orbits.get_num_orbits()+1):
            o1 = Opl.ms.orbits.get_orbit(i)
            o2 = Opr.ms.orbits.get_orbit(i)
            if(not Opl.ms.orbits.is_same_orbit(o1,o2)):
                print("The single-particle orbits definition has to be the same in left and right snt files: "+str(i))
                return None
        cmd = "cp " + self.kshl_dir + "/calc_overlap.exe ./"
        subprocess.call(cmd,shell=True)
        fn_overlaps = []
        for state_l, state_r in itertools.product(ksh_l.fn_ptns.keys(), ksh_r.fn_ptns.keys()):
            fn_base = "Overlap_" + os.path.splitext(ksh_l.fn_wfs[state_l])[0] + "_" + os.path.splitext(ksh_r.fn_wfs[state_r])[0]
            fn_input = fn_base + ".input"
            fn_script = fn_base + ".sh"
            fn_overlap = fn_base + ".txt"
            fn_overlaps.append(fn_overlap)
            prt = header + '\n'
            prt += 'cat >' + fn_input + ' <<EOF\n'
            prt += '&input\n'
            prt += '  fn_snt   = "' + ksh_l.fn_snt + '"\n'
            prt += '  fn_ptn_l = "' + ksh_l.fn_ptns[state_l]+ '"\n'
            prt += '  fn_ptn_r = "' + ksh_r.fn_ptns[state_r]+ '"\n'
            prt += '  fn_load_wave_l = "' + ksh_l.fn_wfs[state_l] + '"\n'
            prt += '  fn_load_wave_r = "' + ksh_r.fn_wfs[state_r] + '"\n'
            prt += '&end\n'
            prt += 'EOF\n'
            if(run_cmd == None):
                prt += './calc_overlap.exe ' + fn_input + ' > ' + fn_overlap + ' 2>&1\n'
            if(run_cmd != None):
                prt += run_cmd + ' ./calc_overlap.exe ' + fn_input + ' > ' + fn_overlap + ' 2>&1\n'
            prt += 'rm ' + fn_input + '\n'
            f = open(fn_script,'w')
            f.write(prt)
            f.close()
            os.chmod(fn_script, 0o755)
            if(batch_cmd == None): cmd = "./" + fn_script
            if(batch_cmd != None): cmd = batch_cmd + " " + fn_script
            subprocess.call(cmd, shell=True)
            if(batch_cmd != None): time.sleep(1)
        return fn_overlaps


class kshell_toolkit:
    def calc_exp_vals(kshl_dir, fn_snt, fn_op, Nucl, states_list, hw_truncation=None, ph_truncation=None,
            run_args=None, Nucl_daughter=None, fn_snt_daughter=None,
            op_rankJ=0, op_rankP=1, op_rankZ=0, verbose=False, mode="all",
            header="", batch_cmd=None, run_cmd=None, type_output="list", comment_sntfile="!",
            bin_density=False):
        """
        inputs:
            kshel_dir (str)    : path to kshell exe files
            fn_snt (str)       : file name of snt
            Nucl (str)         : target nuclide
            states_list (list) : combinations of < bra | and | ket >
                ex.) even-mass case states_list should be like [(0+2, 0+2), (0+1, 2+2)]: returns <0+1|Op|0+1>, <0+1|Op|0+2>, <0+2|Op|0+2>, <0+1|Op|2+1>, <0+1|Op|2+2>
                     odd-mass case state_list should be like [(0.5+1, 0.5+1), ]: <1/2+1| Op |1/2+1>
                     [(+2,+2),(-2,-2)]:
                     <J+1|Op|J+1>, <J+1|Op|J+2>, <J+2|Op|J+2>, <J-1|Op|J-1>, <J-1|Op|J-2>, <J-2|Op|J-2>
            hw_truncation (int): you can introduce hw truncation
            ph_truncation (str): you can introduce particle-hole truncation by "(oribit index)_(min occ)_(max occ)-(orbit index)_(min)_(max)-..."
            run_args (dict)    : Additional arguments for kshell_ui.py
            Nucl_daughter (str): Use this for b or bb decay
            fn_snt_daughter (str): put snt file name if you want to use one different from parent nucleus
            mode               : If you can run the calculations interactively, you can use mode='all'.
                Otherwise, submit jobs step-by-step, mode='diag'->mode='density'->mode='eval'
            header (str)       : for reasorce allocation
            batch_cmd (str)    : job submit command e.g., 'qsub', 'sbatch'
            run_cmd (str)      : execute command e.g., 'srun', default is './'
            verbose (bool)     : for debug
            return value is the normal matrix element for scalar operator (op_rankJ=0, op_rankP=1, op_rankZ=0), otherwise, it is reduced matrix element.
        """

        print('This method is deprecated and will be removed in near future. Use calc_op_exp_vals instead!')
        if(mode=="all" and batch_cmd!=None):
            print("mode='all' is only for a run on local machine. Please select mode from the following:")
            print("mode='diag':    Diagonalization with KSHELL")
            print("mode='density': Calculation of transition density")
            print("mode='eval':    Calculation of expectation value with given operator and states")
            return None
        parity_mixing=False
        if(op_rankP==-1): parity_mixing=True
        if(Nucl_daughter==None): Nucl_daughter=Nucl
        if(fn_snt_daughter==None): fn_snt_daughter=fn_snt

        if(fn_op!="" and fn_op!=None): op = Operator(filename=fn_op, rankJ=op_rankJ, rankP=op_rankP, rankZ=op_rankZ, verbose=verbose, comment=comment_sntfile)
        if(op==None): raise ValueError()

        if(type_output=="list"): exp_vals = []
        if(type_output=="dict"): exp_vals = {}
        if(type_output=="DataFrame"): exp_vals = pd.DataFrame()
        for lr in states_list:
            bra = lr[0]
            ket = lr[1]
            if(bra == ket and Nucl==Nucl_daughter and fn_snt==fn_snt_daughter):
                kshl = kshell_scripts(kshl_dir=kshl_dir, fn_snt=fn_snt, Nucl=Nucl, states=bra,
                        hw_truncation=hw_truncation, ph_truncation=ph_truncation, run_args=run_args,
                        verbose=verbose)
                if(mode=="diag" or mode=="all"):
                    kshl.run_kshell(header=header, batch_cmd=batch_cmd, run_cmd=run_cmd)
                    if(mode=="diag"): continue
                trs = transit_scripts(kshl_dir=kshl_dir,bin_output=bin_density)
                if(mode=="density" or mode=="all"):
                    fn_den, flip = trs.calc_density(kshl, kshl, states_list=[lr,],
                            header=header, batch_cmd=batch_cmd, run_cmd=run_cmd, parity_mix=parity_mixing)
                    if(mode=="density"): continue
                if(mode=="eval" or mode=="all"):
                    flip = trs.set_filenames(kshl, kshl, states_list=[lr,])
                    fn_density = trs.filenames[lr]
                    n = kshl._number_of_states(ket)
                    wf_index = kshl.get_wf_index(use_logs=True)
                    for state_bra, state_ket in itertools.product(list(wf_index.keys()), repeat=2):
                        Jbra, Pbra, nn_bra = state_bra
                        Jket, Pket, nn_ket = state_ket
                        Jfbra = _str_J_to_Jfloat(Jbra)
                        Jfket = _str_J_to_Jfloat(Jket)
                        if( _prty2i(Pbra) * _prty2i(Pket) * op_rankP == -1): continue
                        if( not int(abs(Jfbra-Jfket)) <= op_rankJ <= int(Jfbra+Jfket) ): continue
                        en_bra = kshl.energy_from_summary((Jbra,Pbra,nn_bra))
                        en_ket = kshl.energy_from_summary((Jket,Pket,nn_ket))
                        if(flip): Density = TransitionDensity(filename=fn_density, Jbra=Jfket, wflabel_bra=wf_index[state_ket][-1], \
                                Jket=Jfbra, wflabel_ket=wf_index[state_bra][-1])
                        if(not flip): Density = TransitionDensity(filename=fn_density, Jbra=Jfbra, wflabel_bra=wf_index[state_bra][-1], \
                                Jket=Jfket, wflabel_ket=wf_index[state_ket][-1], verbose=verbose)
                        if(type_output=="list"): exp_vals.append((Jbra,Pbra,nn_bra,en_bra,Jket,Pket,nn_ket,en_ket,*Density.eval(op)))
                        if(type_output=="dict"): exp_vals[(Jbra,Pbra,nn_bra,en_bra,Jket,Pket,nn_ket,en_ket)] = Density.eval(op)
                        if(type_output=="DataFrame"):
                            _ = Density.eval(op)
                            if(flip): _ = [Nucl,Jket,Pket,nn_ket,en_ket,Nucl,Jbra,Pbra,nn_bra,en_bra,*_]
                            if(not flip): _ = [Nucl,Jbra,Pbra,nn_bra,en_bra,Nucl,Jket,Pket,nn_ket,en_ket,*_]
                            exp_vals = pd.concat([exp_vals,pd.DataFrame([_])],ignore_index=True)

            else:
                kshl_l = kshell_scripts(kshl_dir=kshl_dir, fn_snt=fn_snt_daughter, Nucl=Nucl_daughter, states=bra,
                        hw_truncation=hw_truncation, run_args=run_args, ph_truncation=ph_truncation, verbose=verbose)
                kshl_r = kshell_scripts(kshl_dir=kshl_dir, fn_snt=fn_snt, Nucl=Nucl, states=ket,
                        hw_truncation=hw_truncation, run_args=run_args, ph_truncation=ph_truncation, verbose=verbose)
                if(mode=="diag" or mode=="all"):
                    kshl_l.run_kshell(header=header, batch_cmd=batch_cmd, run_cmd=run_cmd)
                    kshl_r.run_kshell(header=header, batch_cmd=batch_cmd, run_cmd=run_cmd)
                    if(mode=="diag"): continue
                trs = transit_scripts(kshl_dir=kshl_dir,bin_output=bin_density)
                if(mode=="density" or mode=="all"):
                    fn_den, flip = trs.calc_density(kshl_l, kshl_r, states_list=[lr,],
                            header=header, batch_cmd=batch_cmd, run_cmd=run_cmd, parity_mix=parity_mixing)
                    if(mode=="density"): continue
                if(mode=="eval" or mode=="all"):
                    flip = trs.set_filenames(kshl_l, kshl_r, states_list=[lr,])
                    if(flip): fn_density = trs.filenames[lr[::-1]]
                    else:     fn_density = trs.filenames[lr]
                    n_bra = kshl_l._number_of_states(bra)
                    n_ket = kshl_r._number_of_states(ket)
                    wf_index_bra = kshl_l.get_wf_index(use_logs=True)
                    wf_index_ket = kshl_r.get_wf_index(use_logs=True)
                    for state_bra, state_ket in itertools.product(list(wf_index_bra.keys()), list(wf_index_ket.keys())):
                        Jbra, Pbra, nn_bra = state_bra
                        Jket, Pket, nn_ket = state_ket
                        Jfbra = _str_J_to_Jfloat(Jbra)
                        Jfket = _str_J_to_Jfloat(Jket)
                        if( _prty2i(Pbra) * _prty2i(Pket) * op_rankP == -1): continue
                        if( not int(abs(Jfbra-Jfket)) <= op_rankJ <= int(Jfbra+Jfket) ): continue
                        en_bra = kshl_l.energy_from_summary((Jbra,Pbra,nn_bra))
                        en_ket = kshl_r.energy_from_summary((Jket,Pket,nn_ket))
                        if(flip): Density = TransitionDensity(filename=fn_density, Jbra=Jfket, wflabel_bra=wf_index_ket[state_ket][-1], \
                                Jket=Jfbra, wflabel_ket=wf_index_bra[state_bra][-1], verbose=verbose)
                        if(not flip): Density = TransitionDensity(filename=fn_density, Jbra=Jfbra, wflabel_bra=wf_index_bra[state_bra][-1], \
                                Jket=Jfket, wflabel_ket=wf_index_ket[state_ket][-1])
                        _ = Density.eval(op)
                        if(type_output=="list"): exp_vals.append((Jbra,Pbra,nn_bra,en_bra,Jket,Pket,nn_ket,en_ket,*_))
                        if(type_output=="dict"): exp_vals[(Jbra,Pbra,nn_bra,en_bra,Jket,Pket,nn_ket,en_ket)] = _
                        if(type_output=="DataFrame"):
                            if(flip): _ = [Nucl,Jket,Pket,nn_ket,en_ket,Nucl_daughter,Jbra,Pbra,nn_bra,en_bra,*_]
                            if(not flip): _ = [Nucl_daughter,Jbra,Pbra,nn_bra,en_bra,Nucl,Jket,Pket,nn_ket,en_ket,*_]
                            exp_vals = pd.concat([exp_vals,pd.DataFrame([_])],ignore_index=True)
        if(mode=="diag" or mode=="density"): return None
        if(type_output=="DataFrame"): exp_vals.columns = ["Nucl bra","J bra","P bra","n bra","Energy bra","Nucl ket","J ket","P ket","n ket","Energy ket","Zero","One","Two"]
        return exp_vals

    def calc_op_exp_vals(kshl_dir, fn_snt, Nucl, states_list, hw_truncation=None, ph_truncation=None,
            run_args=None, Nucl_daughter=None, fn_snt_daughter=None,
            op_rankJ=0, op_rankP=1, op_rankZ=0, verbose=False, mode="all",
            header="", batch_cmd=None, run_cmd=None, type_output="list", comment_sntfile="!",
            bin_density=False, op=None, fn_op='', output_file=None):
        """
        inputs:
            kshel_dir (str)    : path to kshell exe files
            fn_snt (str)       : file name of snt
            Nucl (str)         : target nuclide
            states_list (list) : combinations of < bra | and | ket >
                ex.) even-mass case states_list should be like [(0+2, 0+2), (0+1, 2+2)]: returns <0+1|Op|0+1>, <0+1|Op|0+2>, <0+2|Op|0+2>, <0+1|Op|2+1>, <0+1|Op|2+2>
                     odd-mass case state_list should be like [(0.5+1, 0.5+1), ]: <1/2+1| Op |1/2+1>
                     [(+2,+2),(-2,-2)]:
                     <J+1|Op|J+1>, <J+1|Op|J+2>, <J+2|Op|J+2>, <J-1|Op|J-1>, <J-1|Op|J-2>, <J-2|Op|J-2>
            hw_truncation (int): you can introduce hw truncation
            ph_truncation (str): you can introduce particle-hole truncation by "(oribit index)_(min occ)_(max occ)-(orbit index)_(min)_(max)-..."
            run_args (dict)    : Additional arguments for kshell_ui.py
            Nucl_daughter (str): Use this for b or bb decay
            fn_snt_daughter (str): put snt file name if you want to use one different from parent nucleus
            mode               : If you can run the calculations interactively, you can use mode='all'.
                Otherwise, submit jobs step-by-step, mode='diag'->mode='density'->mode='eval'
            header (str)       : for reasorce allocation
            batch_cmd (str)    : job submit command e.g., 'qsub', 'sbatch'
            run_cmd (str)      : execute command e.g., 'srun', default is './'
            verbose (bool)     : for debug
            return value is the normal matrix element for scalar operator (op_rankJ=0, op_rankP=1, op_rankZ=0), otherwise, it is reduced matrix element.
        """
        if(mode=="all" and batch_cmd=="qsub"):
            print("mode='all' is only for a run on local machine. Please select mode from the following:")
            print("mode='diag':    Diagonalization with KSHELL")
            print("mode='density': Calculation of transition density")
            print("mode='eval':    Calculation of expectation value with given operator and states")
            return None
        parity_mixing=False
        if(op_rankP==-1): parity_mixing=True
        if(Nucl_daughter==None): Nucl_daughter=Nucl
        if(fn_snt_daughter==None): fn_snt_daughter=fn_snt


        if(fn_op!="" and fn_op!=None and op==None): op = Operator(filename=fn_op, rankJ=op_rankJ, rankP=op_rankP, rankZ=op_rankZ, verbose=verbose, comment=comment_sntfile)
        if(op==None): raise ValueError()
        op_rankJ = op.rankJ
        op_rankP = op.rankP
        op_rankZ = op.rankZ

        if(type_output=="list"): exp_vals = []
        elif(type_output=="dict"): exp_vals = {}
        elif(type_output=="DataFrame"): exp_vals = pd.DataFrame()

        run_script = True
        if (mode=="all" and batch_cmd=="sbatch"):
            run_script=False
            if(not output_file):
                print("output_file is required to run with mode all on slurm system")
                return None
        for lr in states_list:
            bra = lr[0]
            ket = lr[1]
            final_eq_initial = (bra == ket and Nucl==Nucl_daughter and fn_snt==fn_snt_daughter)
            kshl_r = kshell_scripts(kshl_dir=kshl_dir, fn_snt=fn_snt, Nucl=Nucl, states=ket,
                        hw_truncation=hw_truncation, run_args=run_args, ph_truncation=ph_truncation, verbose=verbose)
            if final_eq_initial:
                kshl_l = kshl_r
            else:
                kshl_l = kshell_scripts(kshl_dir=kshl_dir, fn_snt=fn_snt_daughter, Nucl=Nucl_daughter, states=bra,
                        hw_truncation=hw_truncation, run_args=run_args, ph_truncation=ph_truncation, verbose=verbose)
            if(mode=="diag" or mode=="all"):
                fn_diag_l = kshl_l.run_kshell(header=header, batch_cmd=batch_cmd, run_cmd=run_cmd, run_script=run_script)
                if final_eq_initial:
                    fn_diag_r = fn_diag_l
                else:
                    fn_diag_r = kshl_r.run_kshell(header=header, batch_cmd=batch_cmd, run_cmd=run_cmd, run_script=run_script)
                if(mode=="diag"): continue
            trs = transit_scripts(kshl_dir=kshl_dir,bin_output=bin_density)
            if(mode=="density" or mode=="all"):
                fn_job_den = trs.calc_density(kshl_l, kshl_r, states_list=[lr,],
                        header=header, batch_cmd=batch_cmd, run_cmd=run_cmd, parity_mix=parity_mixing, run_script=run_script)
                if(mode=="density"): continue
            if((mode=="eval" or mode=="all") and run_script):
                flip = trs.set_filenames(kshl_l, kshl_r, states_list=[lr,])
                if(flip): fn_density = trs.filenames[lr[::-1]]
                else:     fn_density = trs.filenames[lr]
                n_bra = kshl_l._number_of_states(bra)
                n_ket = kshl_r._number_of_states(ket)
                wf_index_bra = kshl_l.get_wf_index(use_logs=True)
                wf_index_ket = kshl_r.get_wf_index(use_logs=True)
                for state_bra, state_ket in itertools.product(list(wf_index_bra.keys()), list(wf_index_ket.keys())):
                    Jbra, Pbra, nn_bra = state_bra
                    Jket, Pket, nn_ket = state_ket
                    Jfbra = _str_J_to_Jfloat(Jbra)
                    Jfket = _str_J_to_Jfloat(Jket)
                    if( _prty2i(Pbra) * _prty2i(Pket) * op_rankP == -1): continue
                    if( not int(abs(Jfbra-Jfket)) <= op_rankJ <= int(Jfbra+Jfket) ): continue
                    en_bra = kshl_l.energy_from_summary((Jbra,Pbra,nn_bra))
                    en_ket = kshl_r.energy_from_summary((Jket,Pket,nn_ket))
                    if(flip): Density = TransitionDensity(filename=fn_density, Jbra=Jfket, wflabel_bra=wf_index_ket[state_ket][-1], \
                            Jket=Jfbra, wflabel_ket=wf_index_bra[state_bra][-1], verbose=verbose)
                    if(not flip): Density = TransitionDensity(filename=fn_density, Jbra=Jfbra, wflabel_bra=wf_index_bra[state_bra][-1], \
                            Jket=Jfket, wflabel_ket=wf_index_ket[state_ket][-1])
                    _ = Density.eval(op)
                    if(type_output=="list"): exp_vals.append((Jbra,Pbra,nn_bra,en_bra,Jket,Pket,nn_ket,en_ket,*_))
                    if(type_output=="dict"): exp_vals[(Jbra,Pbra,nn_bra,en_bra,Jket,Pket,nn_ket,en_ket)] = _
                    if(type_output=="DataFrame"):
                        if(flip): vals = [Nucl,Jket,Pket,nn_ket,en_ket,Nucl_daughter,Jbra,Pbra,nn_bra,en_bra,*_]
                        if(not flip): vals = [Nucl_daughter,Jbra,Pbra,nn_bra,en_bra,Nucl,Jket,Pket,nn_ket,en_ket,*_]
                        exp_vals = pd.concat([exp_vals,pd.DataFrame([vals])],ignore_index=True)
                    if output_file:
                        exp_values = pd.DataFrame()
                        if(flip): vals = [fn_op,Nucl,Jket,Pket,nn_ket,en_ket,Nucl_daughter,Jbra,Pbra,nn_bra,en_bra,*_]
                        if(not flip): vals = [fn_op,Nucl_daughter,Jbra,Pbra,nn_bra,en_bra,Nucl,Jket,Pket,nn_ket,en_ket,*_]
                        exp_values = pd.DataFrame([vals], columns=["op_file","Nucl bra","J bra","P bra","n bra","Energy bra","Nucl ket","J ket","P ket","n ket","Energy ket","Zero","One","Two"]).set_index('op_file')
                        if(not os.path.isfile(output_file)):
                            exp_values.to_csv(output_file)
                        else:
                            exp_values.to_csv(output_file, mode='a', header=False)
            if(not run_script):
                #Write the python script to do the job for eval step:
                fn_eval = f"{os.path.basename(fn_op)}_eval.py"
                eval_script = "#!/usr/bin/env python3\n"
                eval_script += "import os\n"
                eval_script += "import sys\n"
                eval_script += "HOME = os.path.expanduser(\"~\")\n"
                eval_script += "sys.path.append(HOME)\n"
                eval_script += "from mylib_python.Nucl import kshell_toolkit\n"
                eval_script += f"kshell_toolkit.calc_op_exp_vals('{kshl_dir}', '{fn_snt}', '{Nucl}', {states_list}, hw_truncation={hw_truncation}, ph_truncation={ph_truncation},"
                eval_script += f"Nucl_daughter='{Nucl_daughter}', fn_snt_daughter='{fn_snt_daughter}',"
                eval_script += f"op_rankJ={op_rankJ}, op_rankP={op_rankP}, op_rankZ={op_rankZ}, mode='eval',"
                eval_script += f"fn_op='{fn_op}', output_file='{output_file}')"
                f = open(fn_eval, "w")
                f.write(eval_script)
                f.close()
                os.chmod(fn_eval, 0o755)
                #Bash script to call the job for this step
                eval_script = "#!/usr/bin/bash\n"
                eval_script += f"{run_cmd} python {fn_eval}"
                fn_eval = f"{os.path.basename(fn_op)}_eval.sh"
                f = open(fn_eval, "w")
                f.write(eval_script)
                f.close()
                os.chmod(fn_eval, 0o755)
                os.chmod(fn_job_den, 0o755)
                #Script the submit all the jobs sequentially
                fn_script = f"calc_expval_{os.path.basename(fn_op)}.sh"
                submit_script ="#!/usr/bin/bash\n"
                submit_script += f"diagNucl_ID=$(sbatch --parsable {fn_diag_r})\n"
                if(bra == ket and Nucl==Nucl_daughter and fn_snt==fn_snt_daughter):
                    submit_script += f"density_ID=$(sbatch --parsable --dependency=afterok:${{diagNucl_ID}} {fn_job_den})\n"
                else:
                    submit_script += f"diagNuclDaughter_ID=$(sbatch --parsable {fn_diag_l})\n"
                    submit_script += f"density_ID=$(sbatch --parsable --dependency=afterok:${{diagNucl_ID}},${{diagNuclDaughter_ID}} {fn_job_den})\n"
                submit_script += f"sbatch --dependency=afterok:${{density_ID}} {fn_eval}"
                f = open(fn_script, "w")
                f.write(submit_script)
                f.close()
                os.chmod(fn_script, 0o755)
                cmd = f"./{fn_script}"
                subprocess.call(cmd, shell=True)
                return None

            if(mode=="diag" or mode=="density"): return None
            if(type_output=="DataFrame"): exp_vals.columns = ["Nucl bra","J bra","P bra","n bra","Energy bra","Nucl ket","J ket","P ket","n ket","Energy ket","Zero","One","Two"]
        return exp_vals


    def calc_sum_rule_Tz00(kshl_dir, Nucl, fn_snt, fn_op_l, fn_op_r, initial_state, inter_states,\
            hw_truncation=None, ph_truncation=None, run_args=None, op_type=2, op_rankJ_l=0, op_rankP_l=1, op_rankZ_l=0, \
            op_rankJ_r=0, op_rankP_r=1, op_rankZ_r=0, mode="all",\
            batch_cmd=None, run_cmd=None, header="", inter_prty=[-1,1], method="lsf", en_power=0, final_state=None, \
            verbose=False):
        """
        Do not use with the operator that changes the Z and N.
        return value: sum_i ( final | Op_left | i ) ( i | Op_right | init ) * (E_i - E_0)^(en_power)
        Note that the pivot vector for the state | i ) is Op_right | init )
        kshl_dir: path to kshell bin
        Nucl: str ex.) O16
        fn_snt: Hamiltonian file
        fn_op_l: file name of Op_left
        fn_op_r: file name of Op_right
        initial_state: initial state (J, parity, n), J is un-doubled angular momentum
        inter_states: intermediate states ex.) "0+20,2+20,4+20" or so
        """
        Z, N, A = _ZNA_from_str(Nucl)
        Nucl_middle = Nucl
        if(op_type==-10 or op_type==-14): Nucl_middle = PeriodicTable.periodic_table[Z+1] + str(A)
        if(op_type==-11 or op_type==-15): Nucl_middle = PeriodicTable.periodic_table[Z-1] + str(A)
        if(op_type==-12): Nucl_middle = PeriodicTable.periodic_table[Z+2] + str(A)
        if(op_type==-13): Nucl_middle = PeriodicTable.periodic_table[Z-2] + str(A)
        JInit, ParityInit, NInit = initial_state
        JFinal, ParityFinal, NFinal = initial_state
        state_ini = str(JInit) + ParityInit + str(NInit)
        if(final_state != None): JFinal, ParityFinal, NFinal = final_state
        state_fin = str(JFinal) + ParityFinal + str(NFinal)
        states_init = str(JInit) + ParityInit + str(NInit) + "," +\
                str(JFinal) + ParityFinal + str(NFinal)
        if(JInit==JFinal and ParityInit==ParityFinal):
            states_init = str(JInit) + ParityInit + str(max(NInit,NFinal))
            state_ini = states_init
            state_fin = states_init

        Ham = Operator(filename=fn_snt)
        Opl = Operator(filename=fn_op_l, rankJ=op_rankJ_l, rankP=op_rankP_l, rankZ=op_rankZ_l)
        Opr = Operator(filename=fn_op_r, rankJ=op_rankJ_r, rankP=op_rankP_r, rankZ=op_rankZ_r)
        ksh_init = kshell_scripts(kshl_dir=kshl_dir, fn_snt=fn_snt, Nucl=Nucl, states=states_init,\
                hw_truncation=hw_truncation, ph_truncation=ph_truncation, run_args=run_args)
        ksh_ex = kshell_scripts(kshl_dir=kshl_dir, fn_snt=fn_snt, Nucl=Nucl_middle, states=inter_states,\
                hw_truncation=hw_truncation, ph_truncation=ph_truncation, run_args=run_args)
        df = pd.DataFrame()
        sum_rule = 0
        if(method=="lsf"):
            ksh_ex.run_kshell(gen_partition=True)
            for ex_state in inter_states.split(","):
                J, prty, ninter = _str_to_state_Jfloat(ex_state)
                J2 = int(2*J)
                fn_out = "LSF" + str(ninter) + "_" + os.path.basename(ksh_ex.fn_wfs[ex_state])
                ksh_ex.fn_wfs[ex_state] = fn_out
        trs = transit_scripts(kshl_dir=kshl_dir, bin_output=True)
        if(mode=="kshell" or mode == "all"):
            ksh_init.run_kshell(header=header, batch_cmd=batch_cmd, run_cmd=run_cmd)
        if(mode=="inter" or mode=="all"):
            if(method=="lsf"):
                ksh_ex.run_kshell(gen_partition=True)
                for ex_state in inter_states.split(","):
                    J, prty, ninter = _str_to_state_Jfloat(ex_state)
                    if(not abs(J-JInit) <= op_rankJ_r <= J+JInit): continue
                    if(_prty2i(prty) * _prty2i(ParityInit) * op_rankP_r != 1): continue
                    J2 = int(2*J)
                    ksh_ex.run_kshell_lsf(ksh_init.fn_ptns[state_ini], ksh_ex.fn_ptns[ex_state], \
                            ksh_init.fn_wfs[state_ini], ksh_ex.fn_wfs[ex_state], J2, n_vec=ninter, header=header, \
                            batch_cmd=batch_cmd, run_cmd=run_cmd, fn_operator=fn_op_r, \
                            operator_irank=op_rankJ_r, operator_nbody=op_type, operator_iprty=op_rankP_r,\
                            neig_load_wave=NInit)
            if(method=="direct"):
                ksh_ex.run_kshell(header=header, batch_cmd=batch_cmd, run_cmd=run_cmd)
        if(mode=="density" or mode=="all"):
            i_wfs = {}
            for in_state in ksh_init.states.split(","):
                Jin, Pin, Nin = _str_to_state_Jfloat(in_state)
                pairs = [(i,i) for i in range(1,Nin+1)]
                i_wfs[(Jin,Pin,Jin,Pin)] = [pairs,"ini","ini",(in_state,in_state)]
            for in_state in ksh_init.states.split(","):
                Jin, Pin, Nin = _str_to_state_Jfloat(in_state)
                for ex_state in inter_states.split(","):
                    Jex, Pex, Nex = _str_to_state_Jfloat(ex_state)
                    pairs = [(i,j) for i in range(1,Nin+1) for j in range(1,Nex+1)]
                    if((Jin,Pin,Jex,Pex) in i_wfs and Nucl==Nucl_middle):
                        tmp = i_wfs[(Jin,Pin,Jex,Pex)]
                        pairs += tmp[0]
                        i_wfs[(Jin,Pin,Jex,Pex)] = [pairs,"ini","ex",(in_state,ex_state)]
                    else:
                        i_wfs[(Jin,Pin,Jex,Pex)] = [pairs,"ini","ex",(in_state,ex_state)]
            for ex_state in inter_states.split(","):
                Jin, Pin, Nin = _str_to_state_Jfloat(ex_state)
                pairs = [(i,i) for i in range(1,Nin+1)]
                if((Jin,Pin,Jin,Pin) in i_wfs and Nucl==Nucl_middle):
                    tmp = i_wfs[(Jin,Pin,Jin,Pin)]
                    pairs += tmp[0]
                    i_wfs[(Jin,Pin,Jin,Pin)] = [pairs,"ex","ex",(ex_state,ex_state)]
                else:
                    i_wfs[(Jin,Pin,Jin,Pin)] = [pairs,"ex","ex",(ex_state,ex_state)]
            for key in i_wfs.keys():
                pairs, bra_, ket_, state_pair = i_wfs[key]
                pairs = list(set(pairs))
                if(bra_=="ini" and ket_=="ini"):
                    density_files, flip = trs.calc_density(ksh_init, ksh_init, states_list=[state_pair,], \
                            header=header, batch_cmd=batch_cmd, run_cmd=run_cmd, i_wfs=pairs)
                elif(bra_=="ini" and ket_=="ex"):
                    density_files, flip = trs.calc_density(ksh_init, ksh_ex, states_list=[state_pair,], \
                            header=header, batch_cmd=batch_cmd, run_cmd=run_cmd, i_wfs=pairs)
                elif(bra_=="ex" and ket_=="ex"):
                    density_files, flip = trs.calc_density(ksh_ex, ksh_ex, states_list=[state_pair,], \
                            header=header, batch_cmd=batch_cmd, run_cmd=run_cmd, i_wfs=pairs)
        if(mode=="eval" or mode=="all"):
            columns = ['Jf','Pf','nf','Jmid','Pmid','nmid','Ji','Pi','ni','Ex energy','<Op_l>','<Op_r>']
            if(verbose):
                line = f"{'Jf':>4s},{'Pf':>3s},{'nf':>3s},"
                line+= f"{'Jmid':>5s},{'Pmid':>5s},{'nmid':>5s},"
                line+= f"{'Ji':>4s},{'Pi':>3s},{'ni':>3s},"
                line+= f"{'Ex energy':>12s},{'<Op_l>':>12s},{'<Op_r>':>12s},{'Cntr':>12s},{'Sum':>12s}"
                print(line)
            flip = trs.set_filenames(ksh_init, ksh_init)
            d_ini_ini = TransitionDensity(filename=trs.filenames[(state_ini,state_ini)], \
                    Jbra=JInit, Jket=JInit, wflabel_bra=NInit, wflabel_ket=NInit)
            d_fin_fin = TransitionDensity(filename=trs.filenames[(state_fin,state_fin)], \
                    Jbra=JFinal, Jket=JFinal, wflabel_bra=NFinal, wflabel_ket=NFinal)
            E_ini = sum(d_ini_ini.eval(Ham))
            E_fin = sum(d_fin_fin.eval(Ham))
            for ex_state in inter_states.split(","):
                J, prty, ninter = _str_to_state_Jfloat(ex_state)
                if(not abs(J-JInit) <= op_rankJ_r <= J+JInit): continue
                if(_prty2i(prty) * _prty2i(ParityInit) * op_rankP_r != 1): continue
                if(not abs(J-JFinal) <= op_rankJ_l <= J+JFinal): continue
                if(_prty2i(prty) * _prty2i(ParityFinal) * op_rankP_l != 1): continue
                for i in range(1,ninter+1):
                    flip = trs.set_filenames(ksh_ex, ksh_ex)
                    d_ex_ex = TransitionDensity(filename=trs.filenames[(ex_state,ex_state)], \
                            Jbra=J, Jket=J, wflabel_bra=i, wflabel_ket=i)
                    Ex = sum(d_ex_ex.eval(Ham)) - (E_ini + E_fin)*0.5
                    flip = trs.set_filenames(ksh_init, ksh_ex)
                    if(flip):
                        d_ini_ex = TransitionDensity(filename=trs.filenames[(ex_state,state_ini)], \
                                Jbra=J, Jket=JInit, wflabel_bra=i, wflabel_ket=NInit)
                        d_fin_ex = TransitionDensity(filename=trs.filenames[(ex_state,state_fin)], \
                                Jbra=J, Jket=JFinal, wflabel_bra=i, wflabel_ket=NFinal)
                    else:
                        d_ini_ex = TransitionDensity(filename=trs.filenames[(state_ini,ex_state)], \
                                Jbra=JInit, Jket=J, wflabel_bra=NInit, wflabel_ket=i)
                        d_fin_ex = TransitionDensity(filename=trs.filenames[(state_fin,ex_state)], \
                                Jbra=JFinal, Jket=J, wflabel_bra=NFinal, wflabel_ket=i)
                    op_l = sum(d_fin_ex.eval(Opl))
                    op_r = sum(d_ini_ex.eval(Opr))
                    cntr = op_l * Ex**en_power * op_r / (2*JInit+1)
                    sum_rule += cntr
                    _ = [JFinal,ParityFinal,NFinal,J,prty,i,JInit,ParityInit,NInit,Ex,op_l,op_r]
                    df = pd.concat([df,pd.DataFrame([_])], ignore_index=True)
                    if(verbose):
                        line = f"{JFinal:4.1f},{ParityFinal:>3s},{NFinal:3d},"
                        line+= f"{J:5.1f},{prty:>5s},{i:5d},"
                        line+= f"{JInit:4.1f},{ParityInit:>3s},{NInit:3d},"
                        line+= f"{Ex:12.6f},{op_l:12.6f},{op_r:12.6f},{cntr:12.6f},{sum_rule:12.6f}"
                        print(line)
            df.columns = columns
        return sum_rule, df

    def calc_sum_rule_Tz11(kshl_dir, Nucl, fn_snt, fn_op_l, fn_op_r, initial_state, inter_states,\
            hw_truncation=None, ph_truncation=None, run_args=None, op_type=-14, op_rankJ_l=0, op_rankP_l=1, op_rankJ_r=0, op_rankP_r=1,\
            mode="all", batch_cmd=None, run_cmd=None, header="", inter_prty=[-1,1], method="lsf", en_power=0, final_state=None, \
            verbose=False):
        Z, N, A = _ZNA_from_str(Nucl)
        if(op_type==-14 or op_type==-10):
            Nucl_middle = PeriodicTable.periodic_table[Z+1] + str(A)
            Nucl_final = PeriodicTable.periodic_table[Z+2] + str(A)
        elif(op_type==-15 or op_type==-11):
            Nucl_middle = PeriodicTable.periodic_table[Z-1] + str(A)
            Nucl_final = PeriodicTable.periodic_table[Z-2] + str(A)
        else:
            print(f"It seems the value of 'op_type' is wrong: {op_type:3d}")
        JInit, ParityInit, NInit = initial_state
        JFinal, ParityFinal, NFinal = initial_state
        state_ini = str(JInit) + ParityInit + str(NInit)
        if(final_state != None): JFinal, ParityFinal, NFinal = final_state
        state_fin = str(JFinal) + ParityFinal + str(NFinal)
        Ham = Operator(filename=fn_snt)
        Opl = Operator(filename=fn_op_l, rankJ=op_rankJ_l, rankP=op_rankP_l, rankZ=1)
        Opr = Operator(filename=fn_op_r, rankJ=op_rankJ_r, rankP=op_rankP_r, rankZ=1)
        ksh_r = kshell_scripts(kshl_dir=kshl_dir, fn_snt=fn_snt, Nucl=Nucl, states=state_ini,\
                hw_truncation=hw_truncation, ph_truncation=ph_truncation, run_args=run_args)
        ksh_l = kshell_scripts(kshl_dir=kshl_dir, fn_snt=fn_snt, Nucl=Nucl_final, states=state_fin,\
                hw_truncation=hw_truncation, ph_truncation=ph_truncation, run_args=run_args)
        ksh_ex = kshell_scripts(kshl_dir=kshl_dir, fn_snt=fn_snt, Nucl=Nucl_middle, states=inter_states,\
                hw_truncation=hw_truncation, ph_truncation=ph_truncation, run_args=run_args)
        df = pd.DataFrame()
        sum_rule = 0
        if(method=="lsf"):
            ksh_ex.run_kshell(gen_partition=True)
            for ex_state in inter_states.split(","):
                J, prty, ninter = _str_to_state_Jfloat(ex_state)
                J2 = int(2*J)
                fn_out = "LSF" + str(ninter) + "_" + os.path.basename(ksh_ex.fn_wfs[ex_state])
                ksh_ex.fn_wfs[ex_state] = fn_out
        trs_r = transit_scripts(kshl_dir=kshl_dir, bin_output=True)
        trs_l = transit_scripts(kshl_dir=kshl_dir, bin_output=True)
        if(mode=="kshell" or mode == "all"):
            ksh_r.run_kshell(header=header, batch_cmd=batch_cmd, run_cmd=run_cmd)
            ksh_l.run_kshell(header=header, batch_cmd=batch_cmd, run_cmd=run_cmd)
        if(mode=="inter" or mode=="all"):
            if(method=="lsf"):
                ksh_ex.run_kshell(gen_partition=True)
                for ex_state in inter_states.split(","):
                    J, prty, ninter = _str_to_state_Jfloat(ex_state)
                    if(not abs(J-JInit) <= op_rankJ_r <= J+JInit): continue
                    if(_prty2i(prty) * _prty2i(ParityInit) * op_rankP_r != 1): continue
                    J2 = int(2*J)
                    ksh_ex.run_kshell_lsf(ksh_r.fn_ptns[state_ini], ksh_ex.fn_ptns[ex_state], \
                            ksh_r.fn_wfs[state_ini], ksh_ex.fn_wfs[ex_state], J2, n_vec=ninter, header=header, \
                            batch_cmd=batch_cmd, run_cmd=run_cmd, fn_operator=fn_op_r, \
                            operator_irank=op_rankJ_r, operator_nbody=op_type, operator_iprty=op_rankP_r,\
                            neig_load_wave=NInit)
            if(method=="direct"):
                ksh_ex.run_kshell(header=header, batch_cmd=batch_cmd, run_cmd=run_cmd)
        if(mode=="density" or mode=="all"):
            density_files, flip = trs_r.calc_density(ksh_r, ksh_r, states_list=[(state_ini,state_ini),], \
                    header=header, batch_cmd=batch_cmd, run_cmd=run_cmd)
            for ex_state in inter_states.split(","):
                density_files, flip = trs_r.calc_density(ksh_r, ksh_ex, states_list=[(state_ini,ex_state),], \
                        header=header, batch_cmd=batch_cmd, run_cmd=run_cmd)
                Jex, Pex, Nex = _str_to_state_Jfloat(ex_state)
                i_wfs = [(i,i) for i in range(1,Nex+1)]
                density_files, flip = trs_r.calc_density(ksh_ex, ksh_ex, states_list=[(ex_state,ex_state),], \
                        header=header, batch_cmd=batch_cmd, run_cmd=run_cmd, i_wfs=i_wfs)
            density_files, flip = trs_l.calc_density(ksh_l, ksh_l, states_list=[(state_ini,state_ini),], \
                    header=header, batch_cmd=batch_cmd, run_cmd=run_cmd)
            for ex_state in inter_states.split(","):
                density_files, flip = trs_l.calc_density(ksh_l, ksh_ex, states_list=[(state_ini,ex_state),], \
                        header=header, batch_cmd=batch_cmd, run_cmd=run_cmd)
        if(mode=="eval" or mode=="all"):
            columns = ['Jf','Pf','nf','Jmid','Pmid','nmid','Ji','Pi','ni','Ex energy','<Op_l>','<Op_r>']
            if(verbose):
                line = f"{'Jf':>4s},{'Pf':>3s},{'nf':>3s},"
                line+= f"{'Jmid':>5s},{'Pmid':>5s},{'nmid':>5s},"
                line+= f"{'Ji':>4s},{'Pi':>3s},{'ni':>3s},"
                line+= f"{'Ex energy':>12s},{'<Op_l>':>12s},{'<Op_r>':>12s},{'Cntr':>12s},{'Sum':>12s}"
                print(line)
            flip_l = trs_l.set_filenames(ksh_l, ksh_l)
            flip_r = trs_r.set_filenames(ksh_r, ksh_r)
            d_r_r = TransitionDensity(filename=trs_r.filenames[(state_ini,state_ini)], \
                    Jbra=JInit, Jket=JInit, wflabel_bra=NInit, wflabel_ket=NInit)
            d_l_l = TransitionDensity(filename=trs_l.filenames[(state_fin,state_fin)], \
                    Jbra=JFinal, Jket=JFinal, wflabel_bra=NFinal, wflabel_ket=NFinal)
            E_ini = sum(d_r_r.eval(Ham))
            E_fin = sum(d_l_l.eval(Ham))
            for ex_state in inter_states.split(","):
                J, prty, ninter = _str_to_state_Jfloat(ex_state)
                if(not abs(J-JInit) <= op_rankJ_r <= J+JInit): continue
                if(_prty2i(prty) * _prty2i(ParityInit) * op_rankP_r != 1): continue
                if(not abs(J-JFinal) <= op_rankJ_l <= J+JFinal): continue
                if(_prty2i(prty) * _prty2i(ParityFinal) * op_rankP_l != 1): continue
                for i in range(1,ninter+1):
                    flip = trs_r.set_filenames(ksh_ex, ksh_ex)
                    d_ex_ex = TransitionDensity(filename=trs_r.filenames[(ex_state,ex_state)], \
                            Jbra=J, Jket=J, wflabel_bra=i, wflabel_ket=i)
                    Ex = sum(d_ex_ex.eval(Ham)) - (E_ini + E_fin)*0.5
                    flip_r = trs_r.set_filenames(ksh_r, ksh_ex)
                    flip_l = trs_l.set_filenames(ksh_l, ksh_ex)
                    if(flip_r):
                        d_r_ex = TransitionDensity(filename=trs_r.filenames[(ex_state,state_ini)], \
                                Jbra=J, Jket=JInit, wflabel_bra=i, wflabel_ket=NInit)
                    else:
                        d_r_ex = TransitionDensity(filename=trs_r.filenames[(state_ini,ex_state)], \
                                Jbra=JInit, Jket=J, wflabel_bra=NInit, wflabel_ket=i)

                    if(flip_l):
                        d_l_ex = TransitionDensity(filename=trs_l.filenames[(ex_state,state_fin)], \
                                Jbra=J, Jket=JFinal, wflabel_bra=i, wflabel_ket=NFinal)
                    else:
                        d_l_ex = TransitionDensity(filename=trs_l.filenames[(state_fin,ex_state)], \
                                Jbra=JFinal, Jket=J, wflabel_bra=NFinal, wflabel_ket=i)
                    op_l = sum(d_l_ex.eval(Opl))
                    op_r = sum(d_r_ex.eval(Opr))
                    cntr = op_l * Ex**en_power * op_r / (2*JInit+1)
                    sum_rule += cntr
                    _ = [JFinal,ParityFinal,NFinal,J,prty,i,JInit,ParityInit,NInit,Ex,op_l,op_r]
                    df = pd.concat([df,pd.DataFrame([_])], ignore_index=True)
                    if(verbose):
                        line = f"{JFinal:4.1f},{ParityFinal:>3s},{NFinal:3d},"
                        line+= f"{J:5.1f},{prty:>5s},{i:5d},"
                        line+= f"{JInit:4.1f},{ParityInit:>3s},{NInit:3d},"
                        line+= f"{Ex:12.6f},{op_l:12.6f},{op_r:12.6f},{cntr:12.6f},{sum_rule:12.6f}"
                        print(line)
            df.columns = columns
        return sum_rule, df

    def calc_sum_rule(kshl_dir, Nucl, fn_snt, fn_op_l, fn_op_r, initial_state, inter_states,\
            hw_truncation=None, ph_truncation=None, run_args=None, op_type=2, op_rankJ_l=0, op_rankP_l=1, op_rankZ_l=0,
            op_rankJ_r=0, op_rankP_r=1, op_rankZ_r=0, \
            mode="all", batch_cmd=None, run_cmd=None, header="", inter_prty=[-1,1], method="lsf", en_power=0, final_state=None, \
            verbose=False, Nucl_final=None):
        if(Nucl_final==None): Nucl_final=Nucl
        if(op_rankZ_l == 0 and op_rankZ_r==0):
            return kshell_toolkit.calc_sum_rule_Tz00(kshl_dir, Nucl, fn_snt, fn_op_l, fn_op_r, initial_state, inter_states, \
                    hw_truncation=hw_truncation, ph_truncation=ph_truncation, run_args=run_args, op_type=op_type, \
                    op_rankJ_l=op_rankJ_l, op_rankP_l=op_rankP_l, op_rankJ_r=op_rankJ_r, op_rankP_r=op_rankP_r, \
                    mode=mode, batch_cmd=batch_cmd, run_cmd=run_cmd, header=header, inter_prty=inter_prty, method=method,\
                    en_power=en_power, final_state=final_state, verbose=verbose)
        if(Nucl_final==Nucl):
            return kshell_toolkit.calc_sum_rule_Tz00(kshl_dir, Nucl, fn_snt, fn_op_l, fn_op_r, initial_state, inter_states, \
                    hw_truncation=hw_truncation, ph_truncation=ph_truncation, run_args=run_args, op_type=op_type, \
                    op_rankJ_l=op_rankJ_l, op_rankP_l=op_rankP_l, op_rankJ_r=op_rankJ_r, op_rankP_r=op_rankP_r,
                    op_rankZ_l=op_rankZ_l, op_rankZ_r=op_rankZ_r, \
                    mode=mode, batch_cmd=batch_cmd, run_cmd=run_cmd, header=header, inter_prty=inter_prty, method=method,\
                    en_power=en_power, final_state=final_state, verbose=verbose)
        if(op_rankZ_l == 1 and op_rankZ_r==1 and Nucl_final!=Nucl):
            return kshell_toolkit.calc_sum_rule_Tz11(kshl_dir, Nucl, fn_snt, fn_op_l, fn_op_r, initial_state, inter_states, \
                    hw_truncation=hw_truncation, ph_truncation=ph_truncation, run_args=run_args, op_type=op_type, \
                    op_rankJ_l=op_rankJ_l, op_rankP_l=op_rankP_l, op_rankJ_r=op_rankJ_r, op_rankP_r=op_rankP_r, \
                    mode=mode, batch_cmd=batch_cmd, run_cmd=run_cmd, header=header, inter_prty=inter_prty, method=method,\
                    en_power=en_power, final_state=final_state, verbose=verbose)
        else:
            print("Not implemented: ")
            return

