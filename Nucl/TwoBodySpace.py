#!/usr/bin/env python3
if(__package__==None or __package__==""):
    import Orbits
else:
    from . import Orbits
class TwoBodyChannel:
    def __init__(self,J=None,P=None,Z=None,orbits=None,e2max=None):
        self.J = J
        self.P = P
        self.Z = Z
        self.orbits = orbits
        self.e2max = e2max
        self.orbit1_index = []
        self.orbit2_index = []
        self.phase_from_indices = {}
        self.index_from_indices = {}
        self.number_states = 0
        if( self.J != None and self.P != None and self.Z != None and orbits != None ):
            self._set_two_body_channel()
            return
    def _set_two_body_channel(self):
        orbs = self.orbits
        if(self.e2max==None): self.e2max = 2*orbs.emax
        import itertools
        for oa, ob in itertools.combinations_with_replacement( orbs.orbits, 2 ):
            ia = orbs.get_orbit_index_from_orbit( oa )
            ib = orbs.get_orbit_index_from_orbit( ob )
            if( ia == ib and self.J%2==1 ): continue
            if( oa.e + ob.e > self.e2max ): continue
            if( (oa.z + ob.z) != 2*self.Z ): continue
            if( (-1)**(oa.l + ob.l) != self.P ): continue
            if( self._triag( oa.j, ob.j, 2*self.J ) ): continue
            self.orbit1_index.append( ia )
            self.orbit2_index.append( ib )
            idx = len( self.orbit1_index )-1
            self.index_from_indices[(ia,ib)] = idx
            self.index_from_indices[(ib,ia)] = idx
            self.phase_from_indices[(ia,ib)] = 1
            self.phase_from_indices[(ib,ia)] = -(-1)**( (oa.j+ob.j)//2 - self.J )
        self.number_states = len( self.orbit1_index )
    def get_number_states(self):
        return self.number_states
    def get_indices(self,idx):
        return self.orbit1_index[idx], self.orbit2_index[idx]
    def get_orbits(self,idx):
        ia, ib = self.get_indices(idx)
        return self.orbits.get_orbit(ia), self.orbits.get_orbit(ib)
    def get_JPZ(self):
        return self.J, self.P, self.Z
    def _triag(self,J1,J2,J3):
        b = True
        if(abs(J1-J2) <= J3 <= J1+J2): b = False
        return b

class TwoBodySpace:
    def __init__(self,orbits=None,e2max=None):
        self.orbits = orbits
        self.e2max = e2max
        self.index_from_JPZ = {}
        self.channels = []
        self.number_channels = 0
        if( self.orbits != None ):
            if( self.e2max == None ): self.e2max = 2*self.orbits.emax
            for J in range(self.e2max+2):
                for P in [1,-1]:
                    for Z in [-1,0,1]:
                        channel = TwoBodyChannel(J=J,P=P,Z=Z,orbits=self.orbits,e2max=e2max)
                        if( channel.get_number_states() == 0): continue
                        self.channels.append( channel )
                        idx = len(self.channels) - 1
                        self.index_from_JPZ[(J,P,Z)] = idx
            self.number_channels = len(self.channels)
    def get_number_channels(self):
        return self.number_channels
    def get_index(self,*JPZ):
        return self.index_from_JPZ[JPZ]
    def get_channel(self,idx):
        return self.channels[idx]
    def get_channel_from_JPZ(self,*JPZ):
        return self.get_channel( self.get_index(*JPZ) )
    def print_channels(self):
        print("  Two-body channels list ")
        print("  J,par,  Z, # of states")
        for channel in self.channels:
            J,P,Z = channel.get_JPZ()
            print("{:3d},{:3d},{:3d},{:12d}".format(J,P,Z,channel.get_number_states()))

def main():
    orbs = Orbits.Orbits()
    orbs.set_orbits(emax=6)
    two = TwoBodySpace(orbits=orbs)
    two.print_channels()
if(__name__=="__main__"):
    main()
