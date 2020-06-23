#!/usr/bin/env python3
import sys
import numpy as np
import copy
import gzip
if(__package__==None or __package==""):
    import ModelSpace
else:
    from . import Orbits
    from . import ModelSpace

class TransitionDensity:
    def __init__(self, Jbra=None, Jket=None, wflabel_bra=None, wflabel_ket=None, ms=None, filename=None, file_format="kshell", verbose=True):
        self.Jbra = Jbra
        self.Jket = Jket
        self.wflabel_bra = wflabel_bra
        self.wflabel_ket = wflabel_ket
        self.ms = copy.deepcopy(ms)
        self.filename = filename
        self.file_format = file_format
        self.verbose = verbose
        self.one = None
        self.two = {}
        self.three = {}
    def allocate_density( self, ms, Jbra=0, Jket=0 ):
        self.ms = copy.deepcopy(ms)
        orbits = ms.orbits
        self.one = {}
        two = ms.two
        for ichbra in range(two.get_number_channels()):
            chbra = two.get_channel(ichbra)
            for ichket in range(two.get_number_channels()):
                chket = two.get_channel(ichket)
                for Jrank in range( abs(chbra.J-chket.J), chbra.J+chket.J+1):
                    self.two[(ichbra,ichket)] = {}
        if(self.ms.rank==2): return
        three = ms.three
        for ichbra in range(three.get_number_channels()):
            chbra = three.get_channel(ichbra)
            for ichket in range(three.get_number_channels()):
                chket = three.get_channel(ichket)
                self.three[(ichbra,ichket)] = {}
    def count_nonzero_1bme(self):
        counter = 0
        norbs = self.ms.orbits.get_num_orbits()
        for i in range(norbs):
            for j in range(norbs):
                if( abs( self.one[i,j] ) > 1.e-10 ): counter += 1
        return counter
    def count_nonzero_2bme(self):
        counter = 0
        two = self.ms.two
        nch = two.get_number_channels()
        for i in range(nch):
            chbra = two.get_channel(i)
            for j in range(i+1):
                chket = two.get_channel(j)
                counter += len( self.two[(i,j)] )
        return counter
    def set_1btd( self, a, b, me):
        orbits = self.ms.orbits
        oa = orbits.get_orbit(a)
        ob = orbits.get_orbit(b)
        self.one[a-1,b-1] = me
        self.one[b-1,a-1] = me * (-1)**( (ob.j-oa.j)//2 )


