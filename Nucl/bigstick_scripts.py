#!/usr/bin/env python3
import os, sys, time, subprocess, re, itertools
import numpy as np
if(__package__==None or __package__==""):
    import PeriodicTable
else:
    from . import PeriodicTable
def _str_to_states(string):
    if(string.find("+") !=-1):
        m2 = int(2*float(string.split("+")[0] ))
        num = int(string.split("+")[1])
        prty = "+"
    if(string.find("-") !=-1):
        m2 = int(2*float(string.split("-")[0] ))
        num = int(string.split("-")[1])
        prty = "-"
    return (m2, prty, num)

def _ZNA_from_str(Nucl):
    isdigit = re.search(r'\d+', Nucl)
    A = int(isdigit.group())
    asc = Nucl[:isdigit.start()] + Nucl[isdigit.end():]
    asc = asc[0].upper() + asc[1:]
    Z = PeriodicTable.periodic_table.index(asc)
    N = A-Z
    return Z, N, A

class bigstick_scripts:
    def __init__(self, exe=None, fn="", Nucl=None, Nspsmax=None, Nmax=None, hw=20, beta=0, states=None, Nlan=100):
        self.exe = exe
        self.fn = fn
        self.Nucl = Nucl
        self.Nspsmax = Nspsmax
        self.Nmax = Nmax
        self.hw = hw
        self.beta = beta
        if(states!=None):
            m2, prty, num = _str_to_states(states)
            self.m2, self.prty, self.num = m2, prty, num
        if(Nucl!=None):
            Z, N, A = _ZNA_from_str(Nucl)
            self.Z, self.N, self.A = Z, N, A
        self.Nlan = Nlan

    def run(self, header="#!/bin/sh", batch_cmd="", run_cmd=""):
        fn_out = f"{self.Nucl}_Nmax{self.Nmax}_hw{self.hw}_{os.path.splitext(self.fn)[0]}"
        fn_input = f"Input_{fn_out}"
        fn_script = f"{fn_out}.sh"
        ipt =  f"n ! menu choice\n"
        ipt += f"{fn_out} ! filename \n"
        ipt += f"auto ! name of .sps file\n"
        ipt += f"{self.Nspsmax} ! max principal number for autofill of orbits \n"
        ipt += f"{self.Z} {self.N}  ! # of valence protons, neutrons\n"
        ipt += f"{self.m2}     ! 2 x Jz of system\n"
        ipt += f"{self.prty}   ! parity of sytem\n"
        ipt += f"y ! truncate?\n"
        ipt += f"{self.Nmax} ! truncation \n"
        ipt += f"0 ! lanczos fragment size (0 = use default) \n"
        ipt += f"{self.fn} ! TBME filename \n"
        ipt += f"P \n"
        ipt += f"{self.hw} {self.beta} \n"
        ipt += f"end \n"
        ipt += f"ld ! Lanczos menu option \n"
        ipt += f"{self.num} {self.Nlan} ! # states to keep, max # iterations \n"
        ipt += f" ! Not optimizing initial pivot vector \n"
        f = open(fn_input, 'w')
        f.write(ipt)
        f.close()

        header += "\n"
        if(run_cmd==""): header += f"{self.exe} < {fn_input}"
        if(run_cmd!=""): header += f"srun {self.exe} < {fn_ipt}"
        f = open(fn_script, 'w')
        f.write(header)
        f.close()
        os.chmod(fn_script, 0o755)
        if(batch_cmd==""): cmd = f"./{fn_script}"
        if(batch_cmd!=""): cmd = f"{batch_cmd} {fn_script}"
        subprocess.call(cmd, shell=True)
        time.sleep(1)
