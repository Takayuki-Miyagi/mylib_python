#!/usr/bin/env python3
# converter: NuShellX input format (int) form imsrg++ code to KSHELL input format (snt)
# originally written by N. Shimizu, modified by T. Miyagi
#
#  ./nushell2snt.py foo.sp bar.int foobar.snt
# input:
#    foo.sp   Nushell single particle space file
#    bar.int  Nushell/OXBASH interaction file
# output:
#    foobar.snt  space and interaction file for KSHELL
#
#  ./nushell2snt.py foo.sp bar_1b.op foobar_2b.op hogehoge.snt
import sys
from math import *


lorb2c = ['s', 'p', 'd', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
         'm', 'n', 'o']
tz2c = { -1:'p', 1:'n' }


def read_comment_skip(fp):
    while 1:
        arr = fp.readline().replace(',',' ').split()
        if not arr: return None
        if arr[0][0]=="!" or arr[0][0]=="#": continue
        for n,i in enumerate(arr):
            if i[0]=="!" or i[0]=="#":
                arr = arr[:n]
                break
        try:
            return [int(i) for i in arr]
        except ValueError:
            try:
                return [int(i) for i in arr[:-1]]+[float(arr[-1])]
            except ValueError:
                return arr

def scalar(spfile, intfile, sntfile):
    fp = open(spfile)
    t_pn_isospin = read_comment_skip(fp)
    if(t_pn_isospin[0] == "t"):
        t_pn = False
        print(" Nushell interaction in isospin formalism ")
    elif(t_pn_isospin[0] == "pn"):
        print(" Nushell interaction in proton-neutron formalism ")
        t_pn = True
    else:
        raise "sp file type error"

    out = "! " + spfile + " " + intfile + " " + sntfile
    if(t_pn): out += "  in proton-neutron formalism\n"
    else:     out += "  in isospin formalism\n"

    acore, zcore = read_comment_skip(fp)
    ncore = acore - zcore
    nnorb = read_comment_skip(fp)
    nnorb = nnorb[0]
    nspe = nnorb
    n_major_list = read_comment_skip(fp)
    if(t_pn):
        num, norb_p, norb_n = n_major_list
    else:
        num = n_major_list.pop(0)
        norb_p = sum(n_major_list)
        norb_n = norb_p
    nljtz2i, i2nljtz = {}, {}
    for i,a in enumerate(range(nnorb)):
        ii,n,l,j = read_comment_skip(fp)
        n = n-1  # Nushell n=1,2,3,...
        if(a < norb_p): tz=-1
        else: tz = 1
        nljtz2i[(n,l,j,tz)] = i+1
        i2nljtz[i+1] = (n,l,j,tz)
#        out += "! i,n-1,l,j,tz %5d %5d %5d %5d %5d\n" \
#            % (nljtz2i[(n,l,j,tz)], n,l,j,tz)
    fp.close()

    out+="! model space \n"

    if(not t_pn):
        for i in range(1, nnorb+1):
            n,l,j,tz = i2nljtz[i]
            nljtz2i[(n,l,j,-tz)] = i+nnorb
            i2nljtz[i+nnorb] = (n,l,j,-tz)
        nnorb *= 2

    # write model space
    num_p, num_n = 0, 0
    for i in range(1, nnorb+1):
        n,l,j,tz = i2nljtz[i]
        if(tz==-1): num_p += 1
        if(tz== 1): num_n += 1
    out += " %3d %3d   %3d %3d\n" % (num_p, num_n, zcore, ncore)
    for i in range(1,nnorb+1):
        n,l,j,tz = i2nljtz[i]
        out += "%5d   %3d %3d %3d %3d  !  %2d = %c%2d%c_%2d/2\n" \
            % (i, n, l, j, tz, i, tz2c[tz], n, lorb2c[l], j)

    ### read header of interaction file
    fp = open(intfile)
    v_tbme = {}
#    comment = fp.readline()
#    arr = fp.readline().split()
    arr = read_comment_skip(fp)
    if("." in arr[0]): nline = 10000000 # No line number in .int
    else: nline = int(arr.pop(0))
    if(nline == 0): nline = 10000000
    spe = [float(i) for i in arr]
    massdep = False
    if(nline < 0):
        nline = - nline
        if(abs(ncore + zcore - spe[nspe]) > 1.e-8): raise "not implemented"
        if(abs(int(spe[nspe+1]) - spe[nspe+1]) > 1.e-8): raise "not implemented"
        massdep  = int(spe[nspe+1]), -spe[nspe+2]
        spe  = spe[:nspe]
    if not t_pn: spe *= 2


    # print  one-body part
    out += "! interaction\n"
    out += "! num, method=,  hbar_omega\n"
    out += "!  i  j     <i|H(1b)|j>\n"
    out += " %3d %3d\n" % (nnorb, 0)
    for i in range(1,nnorb+1):
        out += "%3d %3d % 15.8f\n" % (i,i,spe[i-1])

    def add_v_tbme(ijkl, JT, v_tbme):
        if(not ijkl in v_tbme): v_tbme[ijkl] = {}
        v_tbme[ijkl][(JT)] = v

    ns = nnorb/2
    ### read TBME
    for i in range(nline):
        arr = fp.readline().split()
        if not arr: break
        ijkl = tuple( int(i) for i in arr[:4] )
        JT = tuple( int(i) for i in arr[4:6] )
        v = float(arr[6])
        add_v_tbme(ijkl, JT, v_tbme)
        if not t_pn:  # isospin form. to pn form.
            i, j, k, l = ijkl
            add_v_tbme( (i+ns, j+ns, k+ns, l+ns), JT, v_tbme)
            if(i==j and k==l):
                add_v_tbme( (i, j+ns, k, l+ns), JT, v_tbme)
            elif(i==j):
                add_v_tbme( (i, j+ns, k, l+ns), JT, v_tbme)
                add_v_tbme( (i, j+ns, k+ns, l), JT, v_tbme)
            elif(k==l):
                add_v_tbme( (i, j+ns, k, l+ns), JT, v_tbme)
                add_v_tbme( (i+ns, j, k, l+ns), JT, v_tbme)
            else:
                add_v_tbme( (i, j+ns, k, l+ns), JT, v_tbme)
                add_v_tbme( (i, j+ns, k+ns, l), JT, v_tbme)
                add_v_tbme( (i+ns, j, k, l+ns), JT, v_tbme)
                add_v_tbme( (i+ns, j, k+ns, l), JT, v_tbme)

    tbij_tz = {}
    for i in range(1,nnorb+1):
        n1,l1,j1,t1 = i2nljtz[i]
        for j in range(i,nnorb+1):
            n2,l2,j2,t2 = i2nljtz[j]
            tz = t1 + t2
            if(not tz in tbij_tz): tbij_tz[tz] = []
            tbij_tz[tz].append((i,j))

    out += "! TBME\n"
    nline = 0
    out_tbme = ""
    for tz in (-2,0,2):
        for ij,(i,j) in enumerate(tbij_tz[tz]):
            n1,l1,j1,t1 = i2nljtz[i]
            n2,l2,j2,t2 = i2nljtz[j]
            for k,l in tbij_tz[tz][ij:]:
                n3,l3,j3,t3 = i2nljtz[k]
                n4,l4,j4,t4 = i2nljtz[l]
                if((l1+l2)%2 != (l3+l4)%2): continue # parity
                if(max(abs(j1-j2),abs(j3-j4)) > min(j1+j2,j3+j4)): continue # triangular condition
                ex_ij, ex_kl = False, False
                if((i,j,k,l) in v_tbme):
                    vjt = v_tbme[(i,j,k,l)]
                elif((k,l,i,j) in v_tbme):
                    vjt = v_tbme[(k,l,i,j)]
                elif((i,j,l,k) in v_tbme):
                    vjt = v_tbme[(i,j,l,k)]
                    ex_kl = True
                elif((l,k,i,j) in v_tbme):
                    vjt = v_tbme[(l,k,i,j)]
                    ex_kl = True
                elif((j,i,k,l) in v_tbme):
                    vjt = v_tbme[(j,i,k,l)]
                    ex_ij = True
                elif((k,l,j,i) in v_tbme):
                    vjt = v_tbme[(k,l,j,i)]
                    ex_ij = True
                elif((j,i,l,k) in v_tbme):
                    vjt = v_tbme[(j,i,l,k)]
                    ex_ij, ex_kl = True, True
                elif((l,k,j,i) in v_tbme):
                    vjt = v_tbme[(l,k,j,i)]
                    ex_ij, ex_kl = True, True
                else:
                    vjt = {}
                for J in range(max(abs(j1-j2),abs(j3-j4))//2,
                               min(j1+j2,j3+j4)//2+1):
                    vvv = 0.0
                    for T in (1,0):
                        if(abs(tz)>T*2): continue
                        if(not (J,T) in vjt): continue
                        if(i==j and ((j1+j2)//2-J+1-T)%2==0): continue
                        if(k==l and ((j3+j4)//2-J+1-T)%2==0): continue
                        v = vjt[(J,T)]
                        if(ex_ij): v *= -(-1)**((j1+j2)//2-J+1-T)
                        if(ex_kl): v *= -(-1)**((j3+j4)//2-J+1-T)
                        vvv += v
                        pass
##                    if abs(vvv)<1.e-8: continue
                    if(tz==0 and (n1!=n2 or l1!=l2 or j1!=j2)): vvv /= sqrt(2.0)
                    if(tz==0 and (n3!=n4 or l3!=l4 or j3!=j4)): vvv /= sqrt(2.0)

                    if(i==j and ((j1+j2)//2-J+1-(t1+t2)/2)%2==0): continue
                    if(k==l and ((j3+j4)//2-J+1-(t1+t2)/2)%2==0): continue

                    out_tbme += "%3d %3d %3d %3d  %3d   % 15.8f\n" % (i, j, k, l, J, vvv)
                    nline += 1

    if(massdep):
        out += " %10d %3d %3d\n" % (nline, 0, 0)

    f = open(intfile,'r')
    lines = f.readlines()
    comment = ""
    for line in lines:
        if(line[0] == "!"): comment += line
        if(line[0] != "!"): break
    out += out_tbme
    out = comment + out

    fp_out  = open(sntfile, 'w')
    fp_out.write(out)
    fp_out.close()

def tensor(spfile, op1_file, op2_file, sntfile):
    fp = open(spfile)
    t_pn_isospin = read_comment_skip(fp)
    if(t_pn_isospin[0] == "t"):
        t_pn = False
        print(" Nushell interaction in isospin formalism ")
    elif(t_pn_isospin[0] == "pn"):
        print(" Nushell interaction in proton-neutron formalism ")
        t_pn = True
    else:
        raise "sp file type error"

    out=""

    acore, zcore = read_comment_skip(fp)
    ncore = acore - zcore
    nnorb = read_comment_skip(fp)
    nnorb = nnorb[0]
    nspe = nnorb
    n_major_list = read_comment_skip(fp)
    if(t_pn):
        num, norb_p, norb_n = n_major_list
    else:
        num = n_major_list.pop(0)
        norb_p = sum(n_major_list)
        norb_n = norb_p
    nljtz2i, i2nljtz = {}, {}
    for i,a in enumerate(range(nnorb)):
        ii,n,l,j = read_comment_skip(fp)
        n = n-1  # Nushell n=1,2,3,...
        if(a < norb_p): tz=-1
        else: tz = 1
        nljtz2i[(n,l,j,tz)] = i+1
        i2nljtz[i+1] = (n,l,j,tz)
    fp.close()

    out+="! model space \n"

    if(not t_pn):
        for i in range(1, nnorb+1):
            n,l,j,tz = i2nljtz[i]
            nljtz2i[(n,l,j,-tz)] = i+nnorb
            i2nljtz[i+nnorb] = (n,l,j,-tz)
        nnorb *= 2

    # write model space
    num_p, num_n = 0, 0
    for i in range(1, nnorb+1):
        n,l,j,tz = i2nljtz[i]
        if(tz==-1): num_p += 1
        if(tz== 1): num_n += 1
    out += " %3d %3d   %3d %3d\n" % (num_p, num_n, zcore, ncore)
    for i in range(1,nnorb+1):
        n,l,j,tz = i2nljtz[i]
        out += "%5d   %3d %3d %3d %3d  !  %2d = %c%2d%c_%2d/2\n" \
            % (i, n, l, j, tz, i, tz2c[tz], n, lorb2c[l], j)

    ### read header of op1_file
    fp = open(op1_file)
    v_obme = {}
    arr = read_comment_skip(fp)
    if arr:
        v_obme[(int(arr[0]),int(arr[1]))] = float(arr[2])
        nline = 10000000
        for i in range(nline):
            arr = fp.readline().split()
            if not arr: break
            ij = ( int(arr[0]), int(arr[1]) )
            v = float(arr[2])
            v_obme[ij] = v
    # print  one-body part
    out += "! one-body part\n"
    out += " %3d %3d %3d\n" % (len(v_obme), 0, 0)
    for ij in v_obme.keys():
        out += "%3d %3d % 15.8f\n" % (ij[0],ij[1],v_obme[ij])

    ### read header of op2_file
    fp = open(op2_file)
    v_tbme = {}
    arr = read_comment_skip(fp)
    if arr:
        ijklJ = tuple( int(i) for i in arr[:6])
        v = float(arr[6])
        v_tbme[ijklJ] = v
        nline = 10000000
        for i in range(nline):
            arr = fp.readline().split()
            if not arr: break
            ijklJ = tuple( int(i) for i in arr[:6])
            v = float(arr[6])
            v_tbme[ijklJ] = v
    # print  two-body part
    out += "! two-body part\n"
    out += " %8d %3d %3d\n" % (len(v_tbme), 0, 0)
    for ijklJ in v_tbme.keys():
        out += "%3d %3d %3d %3d %3d %3d % 15.8f\n" % (ijklJ[0],\
                ijklJ[1],ijklJ[2],ijklJ[3],ijklJ[4],ijklJ[5],v_tbme[ijklJ])
    f = open(op2_file,'r')
    lines = f.readlines()
    comment = ""
    for line in lines:
        if(line[0] == "!"): comment += line
        if(line[0] != "!"): break
    out = comment + out
    fp_out  = open(sntfile, 'w')
    fp_out.write(out)
    fp_out.close()

if __name__ == "__main__":
    ### read hoge.sp
    if len(sys.argv) < 3:
        print("\ntransform NUSHELL sp file and int file to KSHELL snt file")
        print("\n  usage: KSHELLDIR/bin/nushell2snt.py foo.sp bar.int output.snt\n")
        sys.exit()

    if len(sys.argv) == 4:
        fn_out = sys.argv[3]
        scalar(sys.argv[1], sys.argv[2], fn_out)
    else:
        # fn_out = sys.argv[1][:-3] + "_" + sys.argv[2][:-4] +".snt"
        fn_out = sys.argv[2][:-4] +".snt"
    print(" output file : ", fn_out)

