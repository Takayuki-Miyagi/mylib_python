import sys
import matplotlib.cm as cm
from . import Nucl

def set_frame(ax, xrng=None, xlab=None):
    ax.set_xticks(xrng)
    ax.set_xticklabels(xlab, rotation=60)
    ax.set_ylabel('Energy (MeV)')

def get_state_color(Jd,P,color_index=None):
    if(color_index == None):
        color_list_p = ['red','salmon','orange','darkgoldenrod','gold','olive',\
                'lime','forestgreen','turquoise','teal','skyblue']
        color_list_n = ['navy','blue','mediumpurple','blueviolet',\
                'mediumorchid','purple','magenta','pink','crimson']
    if(color_index == 1):
        color_list_p = ['red','orange','olive',\
                'lime','forestgreen','turquoise','teal','skyblue']
        color_list_n = ['blue','blueviolet',\
                'mediumorchid','magenta','crimson']
    if(color_index == 2):
        color_list_p = ['red',"k",'orange',\
                'lime','forestgreen','turquoise','teal','skyblue']
        color_list_n = ['red','blue',\
                'lime','magenta','crimson']
    idx = int(Jd / 2)
    try:
        if(Jd<0): return "k"
        if(P=="+"): return color_list_p[idx]
        if(P=="-"): return color_list_n[idx]
    except:
        if(P!="+" and P!="-"): print("something wrong (parity)")
        return "k"

def get_state_symbol(Jd):
    if( Jd<0 ): return "x"
    symbol_list = ["o","^","v","<",">","s","D","p","H","P","X"]
    idx = int(Jd / 2) % len(symbol_list)
    return symbol_list[ idx ]

def get_energies_dct(summary, absolute = True, snt=None, comment_snt="!"):
    zero_body = 0.0
    edict = {}
    if(snt != None):
        h=Nucl.Operator()
        h.read_operator_file(snt,comment=comment_snt)
        zero_body = h.zero
    f = open(summary,'r')
    lines = f.readlines()
    f.close()
    for line in lines:
        data = line.split()
        try:
            N = int(data[0])
            J = data[1]
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

def ground_state_energy(edict):
    egs = 100.0
    for key in edict.keys():
        egs = min(egs, edict[key])
    return egs

def ground_state(edict):
    egs = 1e10
    gs = ("0","+",1)
    for key in edict.keys():
        if(egs > edict[key]):
            gs = key
            egs = edict[key]
    return gs

def energies_wrt_ground(edict):
    egs = ground_state_energy(edict)
    edict2 = {}
    for key in edict.keys():
        edict2[key] = edict[key] - egs
    return edict2

def draw_energies(axs, edict, xcenter, width, color=None, color_index=None, lw=4):
    for key in edict.keys():
        try:
            J = int(key[0])*2
        except:
            J = int(key[0][:-2])
        P = key[1]
        i = key[2]
        c = color
        if(c == None): c = get_state_color(J,P, color_index)
        axs.plot([xcenter-width,xcenter+width],[edict[key],edict[key]],c=c,lw=lw)

def draw_spe_from_imsrg_log(ax, filename, xcenter, width=0.3, lw=1, jmax=None, proton=True, neutron=True):
    fp = open(filename, "r")
    lines = fp.readlines()
    fp.close()
    spe = {}
    flag = False
    for line in lines:
        data = line.split()
        if(flag and len(data)==0): flag=False
        if(len(data)==0): continue
        if(flag): spe[(int(data[1]), int(data[2]), int(data[3]), int(data[4]))] = (float(data[5]), float(data[6]))
        if(data[0]=='i:'): flag=True
    draw_single_particle_energies(ax, spe, xcenter, width=width, lw=lw, jmax=jmax, proton=proton, neutron=neutron)

def draw_single_particle_energies(axs, spe, xcenter, width=0.3, lw=1, jmax=None, proton=True, neutron=True):
    if(jmax == None):
        jmax = 0
        for key in spe.keys():
            jmax = max( key[2], jmax )
    for key in spe.keys():
        n = key[0]
        l = key[1]
        j = key[2]
        z = key[3]
        ls = "-"
        if( not proton and z==-1): continue
        if( not neutron and z==1): continue
        #if( spe[key][1] == 0.0 ): ls = ":"
        if(z ==1):
            c = "b"
            xmax = xcenter + float(j) / float(jmax) * width
            xmin = xcenter
        if(z ==-1):
            c = "r"
            xmax = xcenter
            xmin = xcenter - float(j) / float(jmax) * width
        axs.plot( [xmin, xmax], [spe[key][0], spe[key][0]], ls=ls, c=c, lw=lw )


def plot_energies(axs, edict, xcenter, ms=10, color=None, mfc=None, color_index=None):
    if(mfc == None): mfc = color
    for key in edict.keys():
        try:
            J = int(key[0])*2
        except:
            J = int(key[0][:-2])
        P = key[1]
        i = key[2]
        c = color
        if(c == None): c = get_state_color(J,P, color_index)
        m = get_state_symbol(J)
        axs.plot([xcenter],[edict[key]],c=c,marker=m, ms=ms, mfc=mfc)

def draw_connections(axs, ldict, rdict, xleft, xright, color=None, color_index=None, lw=1):
    dct = ldict
    if(len(ldict)>len(rdict)): dct = rdict
    for key in dct.keys():
        if(key in ldict and key in rdict):
            eleft = ldict[key]
            eright = rdict[key]
            c = color
            try:
                J = int(key[0])*2
            except:
                J = int(key[0][:-2])
            if(c == None): c = get_state_color(J,key[1], color_index)
            axs.plot([xleft,xright],[eleft,eright],ls=':',c=c,lw=lw)

def put_JP_auto(axs, dct, x_base, y_thr, xshift):
    eold = 1e20
    x = x_base
    for key in dct.keys():
        try:
            J = int(key[0])*2
        except:
            J = int(key[0][:-2])
        P = key[1]
        i = key[2]
        if(i != 1): continue
        if(abs(eold - dct[key]) < y_thr): x += xshift
        else: x = x_base
        c = get_state_color(J,key[1])
        axs.annotate(str(key[0])+"$^"+P+"$", xy = (x,dct[key]), color =c)
        eold = dct[key]

def draw_spe(axs, spe, xcenter, width=0.3, pn = "proton",lw=4):
    if(pn == "proton"): c = "red"
    if(pn == "neutron"): c = "blue"
    for key in spe.keys():
        if(key[1] != pn): continue
        if(key[2] == "hole"): ls = "-"
        if(key[2] == "particle"): ls = "--"
        axs.plot([xcenter-width,xcenter+width],[spe[key],spe[key]],c=c,lw=lw,ls=ls)

def put_spe_label(axs, spe, xcenter, pn="proton", fontsize=0.2):
    if(pn == "proton"): c = "red"
    if(pn == "neutron"): c = "blue"
    for key in spe.keys():
        if(key[1] != pn): continue
        axs.annotate(key[0], xy=(xcenter,spe[key]),color=c,fontsize=fontsize)


def draw_connection_spe(axs, spel, sper, xleft=0, xright=1, pn="proton",lw=1):
    if(len(spel) != len(sper)): return
    if(pn == "proton"): c = "red"
    if(pn == "neutron"): c = "blue"
    for key in spel.keys():
        if(key[1] != pn): continue
        axs.plot([xleft,xright],[spel[key],sper[key]],ls=":",c=c,lw=lw)




