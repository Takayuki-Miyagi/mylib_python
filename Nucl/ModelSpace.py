#!/usr/bin/env python3
import copy
if(__package__==None or __package__==""):
    from Orbits import Orbits, OrbitsIsospin
    import TwoBodySpace
    import ThreeBodySpace
else:
    from . import Orbits, OrbitsIsospin
    from . import TwoBodySpace
    from . import ThreeBodySpace

class ModelSpace:
    def __init__(self, rank=2):
        self.orbits = None
        self.iorbits = None
        self.two = None
        self.three = None
        self.rank = rank
        self.emax = -1
        self.e2max = -1
        self.e3max = -1
    def set_modelspace_from_boundaries( self, emax, e2max=None, e3max=None ):
        self.emax = emax
        self.e2max = e2max
        self.e3max = e3max
        if(e2max == None): self.e2max=2*self.emax
        if(e3max == None): self.e3max=3*self.emax
        if( self.rank==1 ): self.e2max=-1
        if( self.rank==1 ): self.e3max=-1
        self.orbits = Orbits( emax=emax )
        if(self.rank>=2): self.two = TwoBodySpace.TwoBodySpace( orbits=self.orbits, e2max=e2max )
        if(self.rank>=3): self.iorbits = OrbitsIsospin(emax=emax)
        if(self.rank>=3): self.three = ThreeBodySpace.ThreeBodySpace( orbits=self.iorbits, e2max=e2max, e3max=e3max )
    def set_modelspace_from_orbits(self, orbits, e2max=None, e3max=None, iorbits=None):
        self.orbits = copy.deepcopy(orbits)
        self.emax = self.orbits.emax
        self.e2max = e2max
        self.e3max = e3max
        if( self.e2max == None ): self.e2max=2*orbits.emax
        if( self.e3max == None ): self.e3max=3*orbits.emax
        if( self.rank==1 ): self.e2max=-1
        if( self.rank==1 ): self.e3max=-1
        if(self.rank>=2): self.two = TwoBodySpace.TwoBodySpace( orbits=self.orbits, e2max=e2max )
        if(self.rank>=3 and iorbits!=None): self.iorbits = copy.deepcopy(iorbits)
        if(self.rank>=3 and iorbits!=None): self.three = ThreeBodySpace.ThreeBodySpace( orbits=self.iorbits, e2max=e2max, e3max=e3max )
    def print_modelspace_summary(self):
        self.orbits.print_orbits()
        if(self.rank>=2): self.two.print_channels()
        if(self.rank>=3): self.iorbits.print_orbits()
        if(self.rank>=3): self.three.print_channels()


def main():
    ms = ModelSpace()
    ms.set_modelspace_from_boundaries(0)
    ms.print_modelspace_summary()
if(__name__=="__main__"):
    main()
