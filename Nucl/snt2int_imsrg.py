#!/usr/bin/env python2
#
# by T. Ichikawa 2017/01/16
# Modified by T. Miyagi 2020/11/19
#

import sys, os
import numpy as np
from math import sqrt

def orb2char(n,l,j,z):
    lorb2c = ['s', 'p', 'd', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
              'm', 'n', 'o']
    if(z==-1): return "p%d%c%02d/2" % (n, lorb2c[l], j)
    if(z== 1): return "n%d%c%02d/2" % (n, lorb2c[l], j)

def read_comment_skip(fp):
    while True:
        arr = fp.readline().split()
        if not arr: return None
        for i in range(len(arr)):
            if arr[i][0]=="!" or arr[i][0]=="#":
                arr = arr[:i]
                break
        if not arr: continue
        try:
            return [int(i) for i in arr]
        except ValueError:
            try:
                return [int(i) for i in arr[:-1]]+[float(arr[-1])]
            except ValueError:
                return arr

if __name__ == "__main__":

    if len(sys.argv)==1:
        print 'usage: snt2int.py hoge.snt out.int'
        sys.exit(1)

    if len(sys.argv) == 3:
        fn_out = sys.argv[2]
        sp_file = os.path.splitext(sys.argv[2])[0]+".sp"
    else:
        # fn_out = sys.argv[1][:-3] + "_" + sys.argv[2][:-4] +".snt"
        fn_out = sys.argv[1][:-4] +".int"
        sp_file = sys.argv[1][:-4]+".sp"
    print " output file : ", fn_out
    print " output file : ", sp_file
    out = ""

    fn_snt = sys.argv[1]

    fp = open(fn_snt, 'r')
    n_jorb, n_core = [0,0], [0,0]
    n_jorb[0], n_jorb[1], n_core[0], n_core[1] \
        = read_comment_skip(fp)
    n_jorb_pn = sum(n_jorb)
    norb, lorb, jorb, itorb, i2nljtz = [], [], [], [], {}
    for i in range(sum(n_jorb)):
        arr = read_comment_skip(fp)
        if i+1!=arr[0]: raise "read error"
        norb.append(arr[1])
        lorb.append(arr[2])
        jorb.append(arr[3])
        itorb.append(arr[4])
        i2nljtz[i+1] = tuple(arr[1:5])
        if (i< n_jorb[0] and arr[4]!=-1) or \
           (i>=n_jorb[0] and arr[4]!= 1):
            raise "read error"

    vob = {}
    spe = []
    nline = read_comment_skip(fp)
    for _ in range(nline[0]):
        arr = read_comment_skip(fp)
        i, j = int(arr[0]), int(arr[1])
        vob[(i,j)] = float(arr[2])
        spe.append(float(arr[2]))
        if i!=j: vob[(j,i)] = vob[(i,j)]

    vtb = {}
    nline = read_comment_skip(fp)
    if nline[1] == 0:
        out += "! no mass dependece\n"
        fmd_mass = 0
    elif nline[1] == 1:
        fmd_mass, fmd_power = float(nline[2]), float(nline[3])
        out += '! mass dependence (A/%s)^%s\n'%(fmd_mass,fmd_power)
    else:
        raise "mass dependence not supported"

    for i in range(nline[0]):
        arr = read_comment_skip(fp)
        i, j, k, l, J  = [ int(arr[i]) for i in range(5) ]
        v = float(arr[5])
        vtb[(i,j,k,l,J)] = v
        sij = (-1) **( (jorb[i-1]+jorb[j-1])/2-J + 1)
        skl = (-1) **( (jorb[k-1]+jorb[l-1])/2-J + 1)
        if i!=j:          vtb[(j,i,k,l,J)] = v*sij
        if k!=l:          vtb[(i,j,l,k,J)] = v*skl
        if i!=j and k!=l: vtb[(j,i,l,k,J)] = v*sij*skl
        if (i,j)!=(k,l):
            vtb[(k,l,i,j,J)] = v
            if i!=j:          vtb[(k,l,j,i,J)] = v*sij
            if k!=l:          vtb[(l,k,i,j,J)] = v*skl
            if i!=j and k!=l: vtb[(l,k,j,i,J)] = v*sij*skl
    fp.close()

    tbij = []
    for i in range(1, sum(n_jorb) + 1):
        for j in range(i, sum(n_jorb) + 1):
            tbij.append((i,j))

    vtb_is = {}
    # T = 1 part
    T = 1
    for ij, (i, j) in enumerate(tbij):
        n1,l1,j1,z1 = i2nljtz[i]
        n2,l2,j2,z2 = i2nljtz[j]
        for k,l in tbij[ij:]:
            n3,l3,j3,z3 = i2nljtz[k]
            n4,l4,j4,z4 = i2nljtz[l]
            if (l1+l2)%2 != (l3+l4)%2: continue # parity
            if max(abs(j1-j2),abs(j3-j4)) > min(j1+j2,j3+j4):
                continue # triangular condition
            if z1+z2 != z3+z4: continue
            T = int(abs(z1+z2)/2)
            Jmin = max(abs(j1-j2), abs(j3-j4)) / 2
            Jmax = min(j1+j2, j3+j4) / 2
            for J in range(Jmin, Jmax + 1):
                if i == j or k == l:
                    if J % 2 != 0: continue
                if i==j and ((j1+j2)//2-J+1-T)%2==0: continue
                if k==l and ((j3+j4)//2-J+1-T)%2==0: continue
                if not (i,j,k,l,J) in vtb:
                    print '! WARNING missing TBME in snt  %3d %3d %3d %3d %3d' % (i,j,k,l,J)
                    vtb_is[(i,j,k,l,J,T)] = 0.
                else:
                    tbme = vtb[(i,j,k,l,J)]
                    TT = T
                    if(T==0):
                        if(n1!=n2 or l1!=l2 or j1!=j2): tbme *= sqrt(2.0)
                        if(n4!=n4 or l3!=l4 or j3!=j4): tbme *= sqrt(2.0)
                        TT = (J+1)%2
                    vtb_is[(i,j,k,l,J,TT)] = tbme


    out += '!'
    for i in range(sum(n_jorb)):
        out += "%2d=%s"%(i+1, orb2char(norb[i], lorb[i], jorb[i], itorb[i]))
    out += '\n'

    out += "-999 "
    for e in spe:
        out += "%15.7f"%e
    out += '%4d %4d %10.6f\n'%(sum(n_core), sum(n_core)+2, 0.0)

    for (i,j,k,l,J,T), v in sorted(vtb_is.items()):
        out += "%3d %3d %3d %3d  %3d %3d %15.7f\n"%(i,j,k,l,J,T,v)

    with open(fn_out, 'w') as f:
        f.write(out)

    out = "!\n"
    out += "pn\n"
    out += "%d %d\n"%( sum(n_core), n_core[0] )
    out += "%d\n"%( sum(n_jorb) )
    out += "2 %d %d\n"%( n_jorb[0], n_jorb[1] )
    for _ in range(1, sum(n_jorb)+1):
        n,l,j,z = i2nljtz[_]
        out += "%d %d %d %d\n"%( _, n, l, j )
    with open(sp_file, "w") as f:
        f.write(out)




