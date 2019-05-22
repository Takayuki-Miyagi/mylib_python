import sys
import matplotlib.cm as cm
from . import Nucl

def set_frame(ax, xrng=None, xlab=None):
    ax.set_xticks(xrng)
    ax.set_xticklabels(xlab, rotation=60)
    ax.set_ylabel('Energy (MeV)')

def get_state_color(Jd,P):
    color_list_p = ['red','salmon','orange','darkgoldenrod','yellow','olive',\
            'lime','forestgreen','turquoise','teal','skyblue']
    color_list_n = ['navy','blue','mediumpurple','blueviolet',\
            'mediumorchid','purple','magenta','pink','crimson']
    idx = int(Jd / 2)
    try:
        if(P=="+"): return color_list_p[idx]
        if(P=="-"): return color_list_n[idx]
    except:
        if(P!="+" and P!="-"): print("something wrong (parity)")
        return "k"

def get_energies_dct(summary, absolute = True, snt=None, comment_snt="!"):
    zero_body = 0.0
    edict = {}
    if(snt != None):
        h=Nucl.Op(snt)
        h.read_operator_file(comment_snt)
        zero_body = h.zero
    f = open(summary,'r')
    lines = f.readlines()
    f.close()
    for line in lines:
        data = line.split()
        try:
            N = int(data[0])
            J = int(data[1])
            P = data[2]
            i = int(data[3])
            e = float(data[5])
            eex = float(data[6])
            if(absolute):
                edict[(J,P,i)] = e + zero_body
            else:
                edict[(J,P,i)] = eex
        except:
            continue
    return edict

def extract_levels(edict, level_list):
    edict2 = {}
    for key in level_list:
        try:
            edict2[key] = edict[key]
        except:
            pass
    return edict2

def draw_energies(axs, edict, xcenter, width):
    for key in edict.keys():
        J = key[0]
        P = key[1]
        i = key[2]
        c = get_state_color(2*J,P)
        axs.plot([xcenter-width,xcenter+width],[edict[key],edict[key]],c=c,lw=2)

def draw_connections(axs, ldict, rdict, xleft, xright):
    dct = ldict
    if(len(ldict)>len(rdict)): dct = rdict
    for key in dct.keys():
        if(key in ldict and key in rdict):
            eleft = ldict[key]
            eright = rdict[key]
            c = get_state_color(2*key[0],key[1])
            axs.plot([xleft,xright],[eleft,eright],ls=':',c=c,lw=1)

def put_JP_auto(axs, dct, x_base, y_thr, xshift):
    eold = 1e20
    x = x_base
    for key in dct.keys():
        J = key[0]
        P = key[1]
        i = key[2]
        if(i != 1): continue
        if(abs(eold - dct[key]) < y_thr): x += xshift
        else: x = x_base
        c = get_state_color(2*key[0],key[1])
        axs.annotate(str(J)+"$^"+P+"$", xy = (x,dct[key]), color =c)
        eold = dct[key]



