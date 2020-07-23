#!/usr/bin/env python3
def get_single_particle_energies(log):
    f = open(log,"r")
    e_data = {}
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
                e_data[(n,l,j,z)] = (spe, occ, wf)
            break
    f.close()
    return e_data

