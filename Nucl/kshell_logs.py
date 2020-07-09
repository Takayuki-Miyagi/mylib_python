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

