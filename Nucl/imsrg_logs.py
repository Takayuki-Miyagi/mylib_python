#!/usr/bin/env python3
def get_single_particle_energies(log):
    f = open(log,"r")
    e_HF = {}
    e_imsrg = {}
    nlist = []
    llist = []
    jlist = []
    zlist = []
    while True:
        line = f.readline()
        if( not line): break
        if( line[0:47] == "HF Single particle energies and wave functions:" ):
            line = f.readline()
            while True:
                line = f.readline()
                if( len(line) == 1 ): break
                dat = line.split()
                n = int(dat[1])
                l = int(dat[2])
                j = int(dat[3])
                z = int(dat[4])
                spe = float(dat[5])
                occ = float(dat[6])
                wf = [ float(x) for x in dat[9:] ]
                e_HF[(n,l,j,z)] = (spe, occ, wf)
                nlist.append(n)
                llist.append(l)
                jlist.append(j)
                zlist.append(z)
        if( line[0:29] == "Before doing so, the spes are"):
        #if( line[0:12] == "The spes are"):
            while True:
                line = f.readline()
                if(line[0:12]=="SetReference"): break
                if(line[0:4]=="Core"): break
                if(line[0:4]=="Doin"): break
                dat = line.split()
                idx = int(dat[0])
                spe = float(dat[2])
                occ = e_HF[(nlist[idx],llist[idx],jlist[idx],zlist[idx])][1]
                e_imsrg[(nlist[idx],llist[idx],jlist[idx],zlist[idx])] = (spe,occ)
            break
    f.close()
    return e_HF, e_imsrg

