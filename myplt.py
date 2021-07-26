#!/usr/bin/env python3
import sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.font_manager
import matplotlib.cm as cm

def set_canvas_grid(spans, height=8, width=8, shx=True, shy=True,
        wspace=0.05, hspace=0.05, unit='cm'):
    """
    spans is list of tuple [(),(),(),...]. Here, tuple determines the size of figure
    (xlength,ylength)
    """
    fig = plt.figure(figsize=(height, width))
    if(unit=="cm"):
        fig.set_figheight(height/2.54)
        fig.set_figwidth(width/2.54)
    elif(unit=="in"):
        fig.set_figheight(height)
        fig.set_figwidth(width)
    grid = plt.GridSpec(height,width)
    grid.update(hspace=hspace, wspace=wspace)
    axs = []
    x = 0
    y = 0
    for space in spans:
        w = space[0] + x
        h = space[1] + y
        try:
            axs.append(plt.subplot(grid[y:h,x:w]))
        except:
            print('Error in set_canvas, maybe first argument is not correct')
            print(str(x) + ', '+ str(w) + ', ' + str(y) + ', ' + str(h))
        x += space[0]
        if(x == width):
            x = 0
            y += space[1]
    return grid, fig, axs

def set_canvas(c=1, r=1, height=6, width=8, shx=True, shy=True, unit="cm"):
    fig, axs = plt.subplots(ncols=c, nrows=r, sharex=shx, sharey=shy )
    if(unit=="cm"):
        fig.set_figheight(height/2.54)
        fig.set_figwidth(width/2.54)
    elif(unit=="in"):
        fig.set_figheight(height)
        fig.set_figwidth(width)
    return fig, axs

def set_style(roman=False,fontsize=0.4,unit="cm"):
    if( unit=="cm" ):
        fs = fontsize * 72 / 2.54
        plt.rcParams['font.size'] = fs
        plt.rcParams['legend.fontsize'] = fs
        plt.rcParams['xtick.labelsize'] = fs
        plt.rcParams['ytick.labelsize'] = fs
        plt.rcParams['axes.labelsize'] = fs
        plt.rcParams['legend.numpoints'] = 1
        plt.rcParams['xtick.top'] = True
        plt.rcParams['xtick.bottom'] = True
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['xtick.major.size'] = 3
        plt.rcParams['xtick.major.width'] = 1
        plt.rcParams['ytick.left'] = True
        plt.rcParams['ytick.right'] = True
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['ytick.major.size'] = 3
        plt.rcParams['ytick.major.width'] = 1
    elif(unit=="in"):
        fs = fontsize * 72
        plt.rcParams['font.size'] = fs
        plt.rcParams['legend.fontsize'] = fs
        plt.rcParams['xtick.labelsize'] = fs
        plt.rcParams['ytick.labelsize'] = fs
        plt.rcParams['axes.labelsize'] = fs
        plt.rcParams['legend.numpoints'] = 1
        plt.rcParams['xtick.top'] = True
        plt.rcParams['xtick.bottom'] = True
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['xtick.major.size'] = 5
        plt.rcParams['xtick.major.width'] = 1
        plt.rcParams['ytick.left'] = True
        plt.rcParams['ytick.right'] = True
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['ytick.major.size'] = 5
        plt.rcParams['ytick.major.width'] = 1
    if(roman):
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.serif'] = 'Times New Roman'
        del matplotlib.font_manager.weight_dict['roman']
        matplotlib.font_manager._rebuild()
    return plt.rcParams

def colors(number=None):
    if(number == None):
        return ["r", "coral", "gold", "limegreen", "seagreen", "aqua", \
               "royalblue", "b", "navy"]
    if(number != None):
        return [cm.jet_r((1.0*i) / (1.0*number)) for i in range(number)]

def markers(pattern=1):
    if(pattern == 1): return ["v", "^", "<", ">", "s", "p", "D", "*", "o"]
    if(pattern == 2): return ["o","s","D","*","p"]
    return ["v", "^", "<", ">", "s", "p", "D", "*", "o"]

def data2d(f,x=0,y=1,nskip=0,comment='#'):
    ff = open(f, 'r')
    lines = ff.readlines()
    ff.close()
    cnt = 0
    X = []; Y = []
    for line in lines:
        cnt += 1
        if(cnt < nskip): continue
        data = line.split()
        if(data[0][0] == comment): continue
        try:
            X.append(float(data[x]))
            Y.append(float(data[y]))
        except:
            print('warning @ myplt.data2d: ' + line[1:10] + "...")
    return X, Y
    data=np.loadtxt(f,comments=comment)
    X=data[:,x]
    Y=data[:,y]
    return X,Y

def data3d(f,x=0,y=1,z=2,comment='#'):
    data=np.loadtxt(f,comments=comment)
    X=data[:,x]
    Y=data[:,y]
    Z=data[:,z]
    xmin=X.min()
    xmax=X.max()
    ymin=Y.min()
    ymax=Y.max()
    n2d = int(np.sqrt(Z.size))
    Z=Z.reshape(n2d,n2d)
    return Z,xmin,xmax,ymin,ymax

def data_point(f, r=0, c=0, comment='#'):
    ff = open(f,'r')
    lines = ff.readlines()
    ff.close()
    data = []
    for line in lines:
        if(line[0] == comment): continue
        data.append(line.split())
    try:
        r = float(data[r][c])
    except:
        r = None
    return r

def data_point_string(f, r=0, c=0, comment='#'):
    ff = open(f,'r')
    lines = ff.readlines()
    ff.close()
    data = []
    for line in lines:
        if(line[0] == comment): continue
        data.append(line.split())
    r = data[r][c]
    return r

if(__name__=='__main__'):
    #plt.rcParams = set_style()
    #fig, axs = set_canvas()
    #f = sys.argv[1]
    #data = np.loadtxt(f,delimiter=',',comments="#")
    #print(data)
    x = np.arange(-10,10,0.1)
    plt.plot(x,x)
    plt.show()

