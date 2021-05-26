#!/usr/bin/env python3
if(__package__==None or __package__==""):
    from Orbits import OrbitsIsospin
else:
    from . import OrbitsIsospin
class ThreeBodyChannel:
    def __init__(self,J=None,P=None,T=None,orbits=None,e2max=None,e3max=None):
        self.J = J
        self.P = P
        self.T = T
        self.orbits = orbits
        self.e2max = e2max
        self.e3max = e3max
        self.orbit1_index = []
        self.orbit2_index = []
        self.orbit3_index = []
        self.J12_index = []
        self.T12_index = []
        self.index_from_indices = {}
        self.number_states = 0
        if( self.J != None and self.P != None and self.T != None and orbits != None ):
            self._set_three_body_channel()
            return
    def _set_three_body_channel(self):
        orbs = self.orbits
        if(self.e2max==None): self.e2max = 2*orbs.emax
        if(self.e3max==None): self.e3max = 3*orbs.emax
        for oa in orbs.orbits:
            ia = orbs.get_orbit_index_from_orbit( oa )
            for ob in orbs.orbits:
                ib = orbs.get_orbit_index_from_orbit( ob )
                if( ia < ib ): continue
                if( oa.e + ob.e > self.e2max ): continue
                for oc in orbs.orbits:
                    ic = orbs.get_orbit_index_from_orbit( oc )
                    if( ib < ic ): continue
                    if( oa.e + oc.e > self.e2max ): continue
                    if( ob.e + oc.e > self.e2max ): continue
                    if( oa.e + ob.e + oc.e > self.e3max ): continue
                    for Jab in range( abs(oa.j-ob.j)//2, (oa.j+ob.j)//2+1):
                        for Tab in [0,1]:
                            if( ia==ib and (Jab+Tab)%2==0 ): continue
                            if( (-1)**(oa.l + ob.l + oc.l) != self.P ): continue
                            if( self._triag( 2*Jab, oc.j, self.J ) ): continue
                            if( self._triag( 2*Tab,    1, self.T ) ): continue
                            self.orbit1_index.append( ia )
                            self.orbit2_index.append( ib )
                            self.orbit3_index.append( ic )
                            self.J12_index.append( Jab )
                            self.T12_index.append( Tab )
                            idx = len( self.orbit1_index )-1
                            self.index_from_indices[(ia,ib,ic,Jab,Tab)] = idx
                            self.number_states = len( self.orbit1_index )
    def get_number_states(self):
        return self.number_states
    def get_indices(self,idx):
        return self.orbit1_index[idx], self.orbit2_index[idx], self.orbit3_index[idx], self.J12_index[idx], self.T12_index[idx]
    def get_orbits(self,idx):
        ia, ib, ic, Jab, Tab = self.get_indices(idx)
        return self.orbits.get_orbit[ia], self.orbits.get_orbit[ib], self.orbits.get_orbit[ic], Jab, Tab
    def get_JPT(self):
        return self.J, self.P, self.T
    def _triag(self,J1,J2,J3):
        b = True
        if(abs(J1-J2) <= J3 <= J1+J2): b = False
        return b

class ThreeBodySpace:
    def __init__(self,orbits=None,e2max=None,e3max=None):
        self.orbits = orbits
        self.e2max = e2max
        self.e3max = e3max
        self.index_from_JPT = {}
        self.number_channels = 0
        self.channels = []
        if( self.orbits != None ):
            if( self.e2max == None ): self.e2max = 2*self.orbits.emax
            if( self.e3max == None ): self.e3max = 3*self.orbits.emax
            for J in range(1,2*self.e3max+5,2):
                for P in [1,-1]:
                    for T in [1,3]:
                        channel = ThreeBodyChannel(J=J,P=P,T=T,orbits=self.orbits,e2max=e2max,e3max=e3max)
                        if( channel.get_number_states() == 0): continue
                        self.channels.append( channel )
                        idx = len(self.channels) - 1
                        self.index_from_JPT[(J,P,T)] = idx
            self.number_channels = len(self.channels)
    def get_number_channels(self):
        return self.number_channels
    def get_index(self,*JPT):
        return self.index_from_JPT[JPT]
    def get_channel(self,idx):
        return self.channels[idx]
    def get_channel_from_JPT(self,*JPT):
        return self.get_channel( self.get_index(*JPT) )
    def print_channels(self):
        print("  Three-body channels list")
        print("  J,par,  T, # of states")
        for channel in self.channels:
            J,P,T = channel.get_JPT()
            print("{:3d},{:3d},{:3d},{:12d}".format(J,P,T,channel.get_number_states()))

def main():
    orbs = Orbits.OrbitsIsospin()
    orbs.set_orbits(emax=6)
    three = ThreeBodySpace(orbits=orbs,e3max=6)
    three.print_channels()
if(__name__=="__main__"):
    main()
