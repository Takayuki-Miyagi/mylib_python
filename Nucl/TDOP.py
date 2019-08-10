#!/usr/bin/env python3
import sys
import readline
import numpy as np
#
from . import Orbit
from . import Op
from . import TransitionDensity

def main():
    # interactive
    if(len(sys.argv) == 1):
        readline.parse_and_bind("tab: complete")
        print("Insert a file name of operator: ")
        file_op = input()
        print("Insert operator rank J: ")
        op_rankJ = int(input())
        print("Insert operator rank P (1 or -1): ")
        op_rankP = int(input())
        if(op_rankP != -1 and op_rankP != 1):
            print("Parity of operator has to be 1 or -1")
            exit()
        print("Insert operator rank Z: ")
        op_rankZ = int(input())

        print("Insert a file name of transition densities: ")
        file_td = input()
        print("Insert bra state J: ")
        Jbra = int(input())
        print("Insert ket state J: ")
        Jket = int(input())
        print("Insert label of wave function for bra state: ")
        wf_label_bra = int(input())
        print("Insert label of wave function for ket state: ")
        wf_label_ket = int(input())
        f = open('input_calc_observable','w')
        f.write(file_op+'\n')
        f.write(str(op_rankJ)+'\n')
        f.write(str(op_rankP)+'\n')
        f.write(str(op_rankZ)+'\n')
        f.write(file_td+'\n')
        f.write(str(Jbra)+'\n')
        f.write(str(Jket)+'\n')
        f.write(str(wf_label_bra)+'\n')
        f.write(str(wf_label_ket)+'\n')
        f.close()
    if(len(sys.argv) == 2):
        f = open(sys.argv[1],'r')
        lines = f.readlines()
        f.close()
        file_op = lines[0].split('\n')[0]
        op_rankJ = int(lines[1])
        op_rankP = int(lines[2])
        op_rankZ = int(lines[3])
        file_td = lines[4].split('\n')[0]
        Jbra = int(lines[5])
        Jket = int(lines[6])
        wf_label_bra = int(lines[7])
        wf_label_ket = int(lines[8])

    if(len(sys.argv) > 3):
        print("Too many argumetns: stop!")
        sys.exit()

    print('')
    print(' main (calculation for the observable using transition density)')
    print('')

    Op = Operator.Operator(file_op, op_rankJ, op_rankP, op_rankZ)
    TD = TransitionDensity.TransitionDensity(file_td, Jbra, Jket, wf_label_bra, wf_label_ket)

    Op.read_operator_file()
    TD.read_td_file()
    TD.set_orbits(Op.orbs)
    zero,one,two = calc_observable(Op,TD)

    prt = ''
    prt += '# \n'
    prt += '# Calculation using: \n'
    prt += '# ' + file_op + '\n'
    prt += '# ' + file_td + '\n'
    prt += '# n-bra = {0:3d}, n-ket = {1:3d}\n'.format(TD.wfbra,TD.wfket)
    prt += '# \n'
    prt += '# zero-body one-body two-body Total \n'
    prt += '  {0:.4f}   {1:.4f}  {2:.4f}  {3:.4f}'.format(zero,one,two,zero+one+two)
    print(prt)
    f = open("TD2O.output",'w')
    f.write(prt)
    f.close()

def calc_observable(Op,TD):
    orbs = Op.orbs
    TD.set_orbits(Op.orbs)
    zero = Op.zero
    one = 0.0
    for a in range(1,orbs.norbs+1):
        oa = orbs.get_orbit(a)
        for b in range(1,orbs.norbs+1):
            ob = orbs.get_orbit(b)
            if(Op.rankJ == 0 and Op.rankZ ==0):
                one += Op.get_obme(a,b) * TD.get_obtd(a,b,Op.rankJ,Op.rankZ) * \
                        np.sqrt(oa.j+1) / np.sqrt(2*TD.Jbra+1)
            else:
                one += Op.get_obme(a,b) * TD.get_obtd(a,b,Op.rankJ,Op.rankZ)

    two = 0.0
    for a in range(1,orbs.norbs+1):
        for b in range(a,orbs.norbs+1):

            for c in range(1,orbs.norbs+1):
                for d in range(c,orbs.norbs+1):
                    oa = orbs.get_orbit(a)
                    ob = orbs.get_orbit(b)
                    oc = orbs.get_orbit(c)
                    od = orbs.get_orbit(d)

                    if((-1)**(oa.l+ob.l+oc.l+od.l) * Op.rankP != 1): continue
                    if(oa.z + ob.z - oc.z - od.z - 2*Op.rankZ != 0): continue

                    for Jab in range( int(abs(oa.j-ob.j)/2), int((oa.j+ob.j)/2)+1):
                        if(a == b and Jab%2 == 1): continue
                        for Jcd in range( int(abs(oc.j-od.j)/2), int((oc.j+od.j)/2+1)):
                            if(c == d and Jcd%2 == 1): continue
                            if(not abs(Jab-Jcd) <= Op.rankJ <= (Jab+Jcd)): continue
                            if(Op.rankJ == 0 and Op.rankZ ==0):
                                two += Op.get_tbme(a,b,c,d,Jab,Jcd) * TD.get_tbtd(a,b,c,d,Jab,Jcd,Op.rankJ,Op.rankZ) * \
                                        np.sqrt(2*Jab+1)/np.sqrt(2*TD.Jbra+1)

                            else:
                                two += Op.get_tbme(a,b,c,d,Jab,Jcd) * TD.get_tbtd(a,b,c,d,Jab,Jcd,Op.rankJ,Op.rankZ)
    return zero,one,two

if(__name__ == "__main__"):
    main()
